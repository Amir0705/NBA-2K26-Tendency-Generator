"""Read-only interface to the Postgres NBA warehouse.

Usage pattern (fall-through to live API when warehouse is empty):

    reader = WarehouseReader()
    stats = reader.get_player_stats(pid, season)
    if stats is None:
        stats = live_api.get_player_stats(pid, season)
"""
from __future__ import annotations

import os
from typing import Any


class WarehouseReader:
    """Read-only façade over the Postgres warehouse.

    All methods return ``None`` (or ``[]``) when the requested row is not
    present — callers should fall back to the live API.
    """

    def __init__(self, path: str | None = None) -> None:  # path ignored; kept for API compat
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> Any | None:
        """Return a psycopg connection, or None if DATABASE_URL is not set."""
        if self._conn is not None:
            return self._conn
        url = os.environ.get("DATABASE_URL")
        if not url:
            return None
        try:
            import psycopg  # noqa: PLC0415
            self._conn = psycopg.connect(
                url,
                autocommit=True,
                prepare_threshold=None,
            )
            return self._conn
        except Exception:  # noqa: BLE001
            return None

    def available(self) -> bool:
        """Return True if the Postgres warehouse is reachable."""
        return self._get_conn() is not None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # player_info
    # ------------------------------------------------------------------

    def get_player_info(self, player_id: int) -> dict[str, Any] | None:
        """Return bio row for a player, or None if not found."""
        conn = self._get_conn()
        if conn is None:
            return None
        row = conn.execute(
            """
            SELECT player_id, full_name, position, height_in, weight_lbs, birthdate
            FROM player_info
            WHERE player_id = %s
            """,
            [int(player_id)],
        ).fetchone()
        if row is None:
            return None
        return {
            "player_id": row[0],
            "full_name": row[1],
            "position": row[2],
            "height_in": row[3],
            "weight_lbs": row[4],
            "birthdate": row[5],
        }

    # ------------------------------------------------------------------
    # player_seasons
    # ------------------------------------------------------------------

    def get_player_stats(self, player_id: int, season: str) -> dict[str, Any] | None:
        """Return per-game stats for a player × season, or None if not found."""
        conn = self._get_conn()
        if conn is None:
            return None
        row = conn.execute(
            """
            SELECT
                player_id, season, gp, min_pg, pts_pg, fga_pg, fgm_pg, fg_pct,
                fg3a_pg, fg3m_pg, fg3_pct, fg3a_rate,
                fta_pg, ftm_pg, ft_pct, oreb_pg, dreb_pg, reb_pg,
                ast_pg, stl_pg, blk_pg, tov_pg, pf_pg,
                usg_pct, ts_pct, efg_pct, ortg, drtg
            FROM player_seasons
            WHERE player_id = %s AND season = %s
            """,
            [int(player_id), str(season)],
        ).fetchone()
        if row is None:
            return None
        keys = [
            "player_id", "season", "gp", "min_pg", "pts_pg", "fga_pg", "fgm_pg", "fg_pct",
            "fg3a_pg", "fg3m_pg", "fg3_pct", "fg3a_rate",
            "fta_pg", "ftm_pg", "ft_pct", "oreb_pg", "dreb_pg", "reb_pg",
            "ast_pg", "stl_pg", "blk_pg", "tov_pg", "pf_pg",
            "usg_pct", "ts_pct", "efg_pct", "ortg", "drtg",
        ]
        return dict(zip(keys, row))

    def get_all_player_stats_for_season(
        self, season: str
    ) -> list[dict[str, Any]]:
        """Return all player rows for a season (empty list if not found)."""
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute(
            """
            SELECT
                player_id, season, gp, min_pg, pts_pg, fga_pg, fgm_pg, fg_pct,
                fg3a_pg, fg3m_pg, fg3_pct, fg3a_rate,
                fta_pg, ftm_pg, ft_pct, oreb_pg, dreb_pg, reb_pg,
                ast_pg, stl_pg, blk_pg, tov_pg, pf_pg,
                usg_pct, ts_pct, efg_pct, ortg, drtg
            FROM player_seasons
            WHERE season = %s
            ORDER BY player_id
            """,
            [str(season)],
        ).fetchall()
        keys = [
            "player_id", "season", "gp", "min_pg", "pts_pg", "fga_pg", "fgm_pg", "fg_pct",
            "fg3a_pg", "fg3m_pg", "fg3_pct", "fg3a_rate",
            "fta_pg", "ftm_pg", "ft_pct", "oreb_pg", "dreb_pg", "reb_pg",
            "ast_pg", "stl_pg", "blk_pg", "tov_pg", "pf_pg",
            "usg_pct", "ts_pct", "efg_pct", "ortg", "drtg",
        ]
        return [dict(zip(keys, r)) for r in rows]

    # ------------------------------------------------------------------
    # pbp_profiles
    # ------------------------------------------------------------------

    def get_pbp_profile(self, player_id: int, season: str) -> dict[str, Any] | None:
        """Return PBP contextual profile for a player × season, or None."""
        conn = self._get_conn()
        if conn is None:
            return None
        row = conn.execute(
            """
            SELECT
                player_id, season,
                gp, min_pg, pts_pg, fga_pg, fgm_pg, fg_pct,
                fg3a_pg, fg3m_pg, fg3_pct,
                fta_pg, ftm_pg, ft_pct,
                ast_pg, reb_pg, oreb_pg, dreb_pg,
                stl_pg, blk_pg, tov_pg, pf_pg,
                catch_and_shoot_three_rate, pull_up_three_rate, unassisted_two_rate,
                assisted_2pt_pct, assisted_3pt_pct, putback_rate,
                rim_fga_share_pbp, mid_fga_share_pbp,
                at_rim_frequency, short_mid_frequency, long_mid_frequency,
                corner3_frequency, arc3_frequency,
                at_rim_accuracy, short_mid_accuracy, long_mid_accuracy,
                corner3_accuracy, arc3_accuracy,
                pbp_usage_rate, total_poss, off_poss,
                seconds_per_poss_off, second_chance_off_poss_rate,
                live_ball_turnover_pct, shooting_fouls_drawn_pct,
                three_pt_fouls_drawn_pct,
                bad_pass_turnovers, lost_ball_turnovers, blocks_recovered_pct,
                offensive_fouls, loose_ball_fouls, shooting_fouls,
                offensive_fouls_drawn, loose_ball_fouls_drawn,
                def_poss, name, team_id, team_abbreviation
            FROM pbp_profiles
            WHERE player_id = %s AND season = %s
            """,
            [int(player_id), str(season)],
        ).fetchone()
        if row is None:
            return None
        keys = [
            "player_id", "season",
            "gp", "min", "pts", "fga", "fgm", "fg_pct",
            "fg3a", "fg3m", "fg3_pct",
            "fta", "ftm", "ft_pct",
            "ast", "reb", "oreb", "dreb",
            "stl", "blk", "tov", "pf",
            "catch_and_shoot_three_rate", "pull_up_three_rate", "unassisted_two_rate",
            "assisted_2pt_pct", "assisted_3pt_pct", "putback_rate",
            "rim_fga_share_pbp", "mid_fga_share_pbp",
            "at_rim_frequency", "short_mid_frequency", "long_mid_frequency",
            "corner3_frequency", "arc3_frequency",
            "at_rim_accuracy", "short_mid_accuracy", "long_mid_accuracy",
            "corner3_accuracy", "arc3_accuracy",
            "pbp_usage_rate", "total_poss", "off_poss",
            "seconds_per_poss_off", "second_chance_off_poss_rate",
            "live_ball_turnover_pct", "shooting_fouls_drawn_pct",
            "three_pt_fouls_drawn_pct",
            "bad_pass_turnovers", "lost_ball_turnovers", "blocks_recovered_pct",
            "offensive_fouls", "loose_ball_fouls", "shooting_fouls",
            "offensive_fouls_drawn", "loose_ball_fouls_drawn",
            "def_poss", "name", "team_id", "team_abbreviation",
        ]
        return dict(zip(keys, row))

    def get_player_shots(self, player_id: int, season: str) -> list[dict[str, Any]]:
        """Return raw shot-event payload rows from player_shots table."""
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute(
            """
            SELECT game_id, period, time_remaining, team_id, opponent_team_id,
                   x, y, shot_value, shot_distance, made
            FROM player_shots
            WHERE player_id = %s AND season = %s
            ORDER BY game_id, period, time_remaining
            """,
            [int(player_id), str(season)],
        ).fetchall()
        if not rows:
            return []
        return [
            {
                "game_id": r[0],
                "period": r[1],
                "time_remaining": r[2],
                "team_id": r[3],
                "opponent_team_id": r[4],
                "x": r[5],
                "y": r[6],
                "shot_value": r[7],
                "shot_distance": r[8],
                "made": r[9],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # team_rosters
    # ------------------------------------------------------------------

    def get_team_roster(
        self, team_abbr: str, season: str
    ) -> list[dict[str, Any]]:
        """Return roster rows for a team × season (empty list if not found)."""
        conn = self._get_conn()
        if conn is None:
            return []
        rows = conn.execute(
            """
            SELECT team_abbr, season, player_id, full_name, position
            FROM team_rosters
            WHERE team_abbr = %s AND season = %s
            ORDER BY player_id
            """,
            [str(team_abbr).upper(), str(season)],
        ).fetchall()
        return [
            {
                "team_abbr": r[0],
                "season": r[1],
                "player_id": r[2],
                "full_name": r[3],
                "position": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Convenience: merged player data (stats + bio + optional PBP profile)
    # ------------------------------------------------------------------

    def get_full_player_data(
        self,
        player_id: int,
        season: str,
        *,
        include_pbp: bool = True,
    ) -> dict[str, Any] | None:
        """Return merged dict of stats + bio + (optionally) PBP profile.

        Returns None only when there is no stats row for this player × season.
        Bio and PBP fields are included when available, otherwise absent.
        """
        stats = self.get_player_stats(player_id, season)
        if stats is None:
            return None

        result: dict[str, Any] = dict(stats)

        bio = self.get_player_info(player_id)
        if bio:
            result.update({k: v for k, v in bio.items() if k not in result})

        if include_pbp:
            pbp = self.get_pbp_profile(player_id, season)
            if pbp:
                result.update({k: v for k, v in pbp.items() if k not in result})

        return result
