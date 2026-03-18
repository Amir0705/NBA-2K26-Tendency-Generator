"""PBP Stats API client for player-level contextual features."""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from src.ingest.cache import Cache


class PBPStatsClient:
    """Fetches contextual player data from PBP Stats endpoints."""

    _RATE_LIMIT_SECONDS = 0.35

    def __init__(self, cache: Cache | None = None) -> None:
        self._cache = cache
        self._base_url = (
            os.environ.get("PBPSTATS_BASE_URL", "https://api.pbpstats.com")
            .strip()
            .rstrip("/")
        )
        self._api_key = (os.environ.get("PBPSTATS_API_KEY") or "").strip()
        self._enabled = (os.environ.get("N2K_USE_PBPSTATS", "0").strip() == "1")
        self._pbp_only = (os.environ.get("N2K_PBP_ONLY", "1").strip() == "1")
        self._last_request_time: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether PBP Stats integration is explicitly enabled."""
        return self._enabled or self._pbp_only

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search players using PBP's all-players mapping."""
        if not self.enabled:
            return []

        cache_key = "pbp:all_players:nba"
        mapping = self._cache_get(cache_key)
        if mapping is None:
            payload = self._request_json("/get-all-players-for-league/nba", {})
            players_obj = payload.get("players")
            mapping = players_obj if isinstance(players_obj, dict) else {}
            self._cache_set(cache_key, mapping, ttl_seconds=86400)

        needle = (name or "").strip().lower()
        if not needle:
            return []

        out: list[dict[str, Any]] = []
        for pid_raw, full_name in mapping.items():
            full = str(full_name or "").strip()
            if not full:
                continue
            if needle in full.lower():
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    continue
                out.append(
                    {
                        "player_id": pid,
                        "full_name": full,
                        "team": "",
                        "is_active": True,
                    }
                )
        return out

    def get_player_profile(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> dict[str, float]:
        """Return compact contextual player features from PBP Stats."""
        if not self.enabled:
            return {}

        totals_row = self._get_player_totals_row(player_id, season, season_type)
        summary_row = self._get_player_shot_summary_row(player_id, season, season_type)
        if not totals_row and not summary_row:
            return {}

        def _num(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
            for key in keys:
                if key in row:
                    try:
                        return float(row.get(key, default) or default)
                    except (TypeError, ValueError):
                        continue
            return float(default)

        fga = _num(summary_row, "fga", default=0.0)
        fg2a = _num(summary_row, "fg2a", default=0.0)
        fg3a = _num(summary_row, "fg3a", default=0.0)
        rim_fga = _num(summary_row, "rim_fga", default=0.0)
        short_mid_fga = _num(summary_row, "smr_fga", default=0.0)
        long_mid_fga = _num(summary_row, "lmr_fga", default=0.0)
        putback_fga = _num(summary_row, "putback_fga", default=0.0)

        if fga <= 0.0:
            fga = _num(
                totals_row,
                "AtRimFGA",
                default=0.0,
            ) + _num(totals_row, "ShortMidRangeFGA", default=0.0) + _num(
                totals_row,
                "LongMidRangeFGA",
                default=0.0,
            ) + _num(totals_row, "Corner3FGA", default=0.0) + _num(
                totals_row,
                "Arc3FGA",
                default=0.0,
            )

        assisted_3pt_pct = _num(
            summary_row,
            "afg3pct",
            "assisted3sPct",
            default=_num(totals_row, "Assisted3sPct", default=0.0),
        )
        assisted_2pt_pct = _num(
            summary_row,
            "afg2pct",
            "nonPutbacksAssisted2sPct",
            default=_num(totals_row, "Assisted2sPct", default=0.0),
        )

        gp = _num(totals_row, "GamesPlayed", default=0.0)
        gp_div = max(gp, 1.0)

        # Shot-summary values are season totals for the query scope; convert
        # to per-game because downstream feature formulas assume per-game inputs.
        fga_pg = fga / gp_div
        fg2a_pg = fg2a / gp_div
        fg3a_pg = fg3a / gp_div
        fg2m_pg = _num(summary_row, "fg2m", default=0.0) / gp_div
        fg3m_pg = _num(summary_row, "fg3m", default=0.0) / gp_div

        if assisted_3pt_pct <= 1.0:
            assisted_3pt_pct *= 100.0
        if assisted_2pt_pct <= 1.0:
            assisted_2pt_pct *= 100.0

        fg3a_rate = fg3a_pg / max(fga_pg, 1.0)
        fg2a_rate = fg2a_pg / max(fga_pg, 1.0)
        a3 = max(0.0, min(1.0, assisted_3pt_pct / 100.0))
        a2 = max(0.0, min(1.0, assisted_2pt_pct / 100.0))

        total_poss = _num(totals_row, "TotalPoss", default=0.0)
        second_chance_poss = _num(totals_row, "SecondChanceOffPoss", default=0.0)

        return {
            "gp": gp,
            "min": _num(totals_row, "Minutes", default=0.0) / gp_div,
            "pts": _num(totals_row, "Points", default=0.0) / gp_div,
            "fga": fga_pg,
            "fgm": fg2m_pg + fg3m_pg,
            "fg_pct": _num(summary_row, "fg2pct", default=0.0) * (fg2a_pg / max(fga_pg, 1.0))
            + _num(summary_row, "fg3pct", default=0.0) * (fg3a_pg / max(fga_pg, 1.0)),
            "fg3a": fg3a_pg,
            "fg3m": fg3m_pg,
            "fg3_pct": _num(summary_row, "fg3pct", default=0.0),
            "fta": _num(totals_row, "FTA", default=0.0) / gp_div,
            "ftm": _num(totals_row, "FtPoints", default=0.0) / gp_div,
            "ft_pct": _num(totals_row, "FtPoints", default=0.0) / max(_num(totals_row, "FTA", default=1.0), 1.0),
            "ast": _num(totals_row, "Assists", default=0.0) / gp_div,
            "reb": _num(totals_row, "Rebounds", default=0.0) / gp_div,
            "oreb": _num(totals_row, "OffRebounds", default=0.0) / gp_div,
            "dreb": _num(totals_row, "DefRebounds", default=0.0) / gp_div,
            "stl": _num(totals_row, "Steals", default=0.0) / gp_div,
            "blk": _num(totals_row, "Blocks", default=0.0) / gp_div,
            "tov": (
                _num(totals_row, "LiveBallTurnovers", default=0.0)
                + _num(totals_row, "DeadBallTurnovers", default=0.0)
            ) / gp_div,
            "pf": _num(totals_row, "Fouls", default=0.0) / gp_div,
            "pbp_usage_rate": _num(totals_row, "Usage", default=0.0),
            "assisted_3pt_pct": assisted_3pt_pct,
            "assisted_2pt_pct": assisted_2pt_pct,
            "catch_and_shoot_three_rate": fg3a_rate * a3,
            "pull_up_three_rate": fg3a_rate * (1.0 - a3),
            "unassisted_two_rate": fg2a_rate * (1.0 - a2),
            "rim_fga_share_pbp": rim_fga / max(fga, 1.0),
            "mid_fga_share_pbp": (short_mid_fga + long_mid_fga) / max(fga, 1.0),
            "putback_rate": putback_fga / max(fga, 1.0),
            "live_ball_turnover_pct": _num(totals_row, "LiveBallTurnoverPct", default=0.0),
            "shooting_fouls_drawn_pct": _num(totals_row, "ShootingFoulsDrawnPct", default=0.0),
            "three_pt_fouls_drawn_pct": _num(totals_row, "ThreePtShootingFoulsDrawnPct", default=0.0),
            "seconds_per_poss_off": _num(totals_row, "SecondsPerPossOff", default=0.0),
            "second_chance_off_poss_rate": second_chance_poss / max(total_poss, 1.0),
            "bad_pass_turnovers": _num(totals_row, "BadPassTurnovers", default=0.0),
            "lost_ball_turnovers": _num(totals_row, "LostBallTurnovers", default=0.0),
            "blocks_recovered_pct": _num(totals_row, "BlocksRecoveredPct", default=0.0),
            "offensive_fouls": _num(totals_row, "Offensive Fouls", default=0.0),
            "loose_ball_fouls": _num(totals_row, "Loose Ball Fouls", default=0.0),
            "shooting_fouls": _num(totals_row, "ShootingFouls", default=0.0),
            "offensive_fouls_drawn": _num(totals_row, "Offensive Fouls Drawn", default=0.0),
            "loose_ball_fouls_drawn": _num(totals_row, "Loose Ball Fouls Drawn", default=0.0),
            "at_rim_frequency": _num(totals_row, "AtRimFrequency", default=0.0),
            "short_mid_frequency": _num(totals_row, "ShortMidRangeFrequency", default=0.0),
            "long_mid_frequency": _num(totals_row, "LongMidRangeFrequency", default=0.0),
            "corner3_frequency": _num(totals_row, "Corner3Frequency", default=0.0),
            "arc3_frequency": _num(totals_row, "Arc3Frequency", default=0.0),
            "at_rim_accuracy": _num(totals_row, "AtRimAccuracy", default=0.0),
            "short_mid_accuracy": _num(totals_row, "ShortMidRangeAccuracy", default=0.0),
            "long_mid_accuracy": _num(totals_row, "LongMidRangeAccuracy", default=0.0),
            "corner3_accuracy": _num(totals_row, "Corner3Accuracy", default=0.0),
            "arc3_accuracy": _num(totals_row, "Arc3Accuracy", default=0.0),
            "total_poss": total_poss,
            "off_poss": _num(totals_row, "OffPoss", default=0.0),
            "def_poss": _num(totals_row, "DefPoss", default=0.0),
            "name": str(totals_row.get("Name", "") or ""),
            "team_id": int(_num(totals_row, "TeamId", default=0.0)),
            "team_abbreviation": str(totals_row.get("TeamAbbreviation", "") or ""),
        }

    def get_player_shots(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Return shot-event rows for a player season from PBP get-shots."""
        cache_key = f"pbp:shots:player:{player_id}:{season}:{season_type}"
        rows = self._cache_get(cache_key)
        if rows is None:
            payload = self._request_json(
                "/get-shots/nba",
                {
                    "Season": season,
                    "SeasonType": season_type,
                    "EntityType": "Player",
                    "EntityId": int(player_id),
                },
            )
            raw_rows = payload.get("results")
            rows = raw_rows if isinstance(raw_rows, list) else []
            self._cache_set(cache_key, rows, ttl_seconds=21600)
        return rows

    def get_player_totals_table(
        self,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Return full PBP player totals table for a season."""
        cache_key = f"pbp:totals:player:{season}:{season_type}"
        table = self._cache_get(cache_key)
        if table is None:
            payload = self._request_json(
                "/get-totals/nba",
                {
                    "Season": season,
                    "SeasonType": season_type,
                    "Type": "Player",
                },
            )
            raw_rows = payload.get("multi_row_table_data")
            table = raw_rows if isinstance(raw_rows, list) else []
            self._cache_set(cache_key, table, ttl_seconds=21600)
        return table

    def get_teams(self, league: str = "nba") -> list[dict[str, Any]]:
        """Return list of teams as {'id': str, 'text': str} objects."""
        cache_key = f"pbp:teams:{league}"
        teams = self._cache_get(cache_key)
        if teams is None:
            payload = self._request_json(f"/get-teams/{league}", {})
            raw = payload.get("teams")
            teams = raw if isinstance(raw, list) else []
            self._cache_set(cache_key, teams, ttl_seconds=86400)
        return teams

    def get_team_players_for_season(
        self,
        team_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> dict[str, str]:
        """Return TeamId-season roster mapping of player_id string to full name."""
        cache_key = f"pbp:team_players:{team_id}:{season}:{season_type}"
        players = self._cache_get(cache_key)
        if players is None:
            payload = self._request_json(
                "/get-team-players-for-season",
                {
                    "Season": season,
                    "SeasonType": season_type,
                    "TeamId": int(team_id),
                },
            )
            raw = payload.get("players")
            players = raw if isinstance(raw, dict) else {}
            self._cache_set(cache_key, players, ttl_seconds=43200)
        return players

    def _get_player_totals_row(
        self,
        player_id: int,
        season: str,
        season_type: str,
    ) -> dict[str, Any]:
        cache_key = f"pbp:totals:player:{season}:{season_type}"
        table = self._cache_get(cache_key)
        if table is None:
            payload = self._request_json(
                "/get-totals/nba",
                {
                    "Season": season,
                    "SeasonType": season_type,
                    "Type": "Player",
                },
            )
            raw_rows = payload.get("multi_row_table_data")
            table = raw_rows if isinstance(raw_rows, list) else []
            self._cache_set(cache_key, table, ttl_seconds=21600)

        return self._find_player_row(table, player_id, id_keys=("EntityId",))

    def _get_player_shot_summary_row(
        self,
        player_id: int,
        season: str,
        season_type: str,
    ) -> dict[str, Any]:
        cache_key = f"pbp:shot_summary:player:{season}:{season_type}"
        table = self._cache_get(cache_key)
        if table is None:
            payload = self._request_json(
                "/get-shot-query-summary/nba",
                {
                    "Season": season,
                    "SeasonType": season_type,
                    "Type": "Player",
                },
            )
            raw_rows = payload.get("results")
            table = raw_rows if isinstance(raw_rows, list) else []
            self._cache_set(cache_key, table, ttl_seconds=21600)

        return self._find_player_row(table, player_id, id_keys=("player_id", "PlayerId"))

    def _find_player_row(
        self,
        rows: list[dict[str, Any]],
        player_id: int,
        id_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        pid = int(player_id)
        for row in rows:
            for key in id_keys:
                if key not in row:
                    continue
                try:
                    if int(row.get(key) or 0) == pid:
                        return row
                except (TypeError, ValueError):
                    continue
        return {}

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}

        url = f"{self._base_url}{path}"
        query = dict(params)
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            query["ApiKey"] = self._api_key

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                self._rate_limit()
                response = requests.get(url, params=query, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)

        if last_exc is not None:
            raise RuntimeError(f"PBP request failed for {path}: {last_exc}") from last_exc
        return {}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._RATE_LIMIT_SECONDS:
            time.sleep(self._RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        if self._cache is not None:
            self._cache.set(key, value, ttl_seconds=ttl_seconds)
