"""PBP-backed ingest client keeping legacy method names for compatibility."""
from __future__ import annotations

import os
import unicodedata
from typing import Any

from src.ingest.cache import Cache
from src.ingest.pbpstats_client import PBPStatsClient
from src.seasons import data_mode_for_season
from src.warehouse.reader import WarehouseReader


_TEAM_ID_BY_ABBR: dict[str, int] = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "CHA": 1610612766,
    "CHH": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "VAN": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NOH": 1610612740,
    "NOK": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "SEA": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
    "NJN": 1610612751,
}


def _strip_accents(s: str) -> str:
    """Remove diacritical marks (ć→c, č→c, ñ→n, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


class NBAApiClient:
    """PBP-only ingest adapter preserving existing interface."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache: Cache | None = Cache(cache_dir) if cache_dir else None
        self._pbp = PBPStatsClient(cache=self._cache)
        self._use_nba_bio = (os.environ.get("N2K_USE_NBA_BIO", "1").strip() == "1")
        self._use_warehouse = (os.environ.get("N2K_USE_WAREHOUSE", "1").strip() == "1")
        self._warehouse = WarehouseReader() if self._use_warehouse else None

    @staticmethod
    def _inches_to_height_str(height_in: Any) -> str:
        """Convert numeric inches to nba_api-like '6-7' string."""
        try:
            total_inches = int(round(float(height_in)))
        except (TypeError, ValueError):
            return ""
        if total_inches <= 0:
            return ""
        feet, inches = divmod(total_inches, 12)
        return f"{feet}-{inches}"

    def _warehouse_available(self) -> bool:
        return self._warehouse is not None and self._warehouse.available()

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search players by name through PBP all-player mapping."""
        return self._pbp.search_player(name)

    def get_player_info(self, player_id: int) -> dict[str, Any]:
        """Return player bio info, preferring live NBA profile when available."""
        if self._warehouse_available():
            w_info = self._warehouse.get_player_info(int(player_id))
            if w_info:
                team_id = 0
                team_abbr = ""
                if data_mode_for_season("2024-25") == "pbp":
                    profile = self.get_pbp_player_profile(player_id)
                    team_id = int(profile.get("team_id", 0) or 0)
                    team_abbr = str(profile.get("team_abbreviation", "") or "")
                return {
                    "full_name": str(w_info.get("full_name", "") or ""),
                    "position": str(w_info.get("position", "") or ""),
                    "height": self._inches_to_height_str(w_info.get("height_in")),
                    "weight": str(w_info.get("weight_lbs", "") or ""),
                    "team_id": team_id,
                    "team_abbreviation": team_abbr,
                    "birthdate": str(w_info.get("birthdate", "") or ""),
                }

        profile = self.get_pbp_player_profile(player_id)
        bio = self._get_nba_bio(player_id)

        position = str(bio.get("position", "") or "")
        if "/" in position:
            position = position.split("/")[0].strip()

        height = str(bio.get("height", "") or "")
        weight = str(bio.get("weight", "") or "")
        birthdate = str(bio.get("birthdate", "") or "")

        return {
            "full_name": "",
            "position": position,
            "height": height,
            "weight": weight,
            "team_id": profile.get("team_id", 0),
            "team_abbreviation": profile.get("team_abbreviation", ""),
            "birthdate": birthdate,
        }

    def _get_nba_bio(self, player_id: int) -> dict[str, Any]:
        """Fetch basic player bio from nba_api CommonPlayerInfo with cache/fallback."""
        if not self._use_nba_bio:
            return {}

        cache_key = f"nba:bio:{int(player_id)}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                return cached

        try:
            from nba_api.stats.endpoints import commonplayerinfo

            endpoint = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id), timeout=15)
            rows = endpoint.get_normalized_dict().get("CommonPlayerInfo", [])
            row = rows[0] if rows else {}
            if not isinstance(row, dict):
                row = {}

            bio = {
                "position": str(row.get("POSITION", "") or ""),
                "height": str(row.get("HEIGHT", "") or ""),
                "weight": str(row.get("WEIGHT", "") or ""),
                "birthdate": str(row.get("BIRTHDATE", "") or ""),
            }
            if self._cache is not None:
                self._cache.set(cache_key, bio, ttl_seconds=1209600)
            return bio
        except Exception:  # noqa: BLE001
            return {}

    def get_player_stats(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """Return per-game style player stats from season-appropriate source."""
        if self._warehouse_available():
            w_stats = self._warehouse.get_player_stats(int(player_id), str(season))
            if w_stats:
                usg_pct = float(w_stats.get("usg_pct") or 0.0)
                return {
                    "gp": int(w_stats.get("gp") or 0),
                    "min": float(w_stats.get("min_pg") or 0.0),
                    "pts": float(w_stats.get("pts_pg") or 0.0),
                    "fga": float(w_stats.get("fga_pg") or 0.0),
                    "fgm": float(w_stats.get("fgm_pg") or 0.0),
                    "fg_pct": float(w_stats.get("fg_pct") or 0.0),
                    "fg3a": float(w_stats.get("fg3a_pg") or 0.0),
                    "fg3m": float(w_stats.get("fg3m_pg") or 0.0),
                    "fg3_pct": float(w_stats.get("fg3_pct") or 0.0),
                    "fta": float(w_stats.get("fta_pg") or 0.0),
                    "ftm": float(w_stats.get("ftm_pg") or 0.0),
                    "ft_pct": float(w_stats.get("ft_pct") or 0.0),
                    "oreb": float(w_stats.get("oreb_pg") or 0.0),
                    "dreb": float(w_stats.get("dreb_pg") or 0.0),
                    "reb": float(w_stats.get("reb_pg") or 0.0),
                    "ast": float(w_stats.get("ast_pg") or 0.0),
                    "stl": float(w_stats.get("stl_pg") or 0.0),
                    "blk": float(w_stats.get("blk_pg") or 0.0),
                    "tov": float(w_stats.get("tov_pg") or 0.0),
                    "pf": float(w_stats.get("pf_pg") or 0.0),
                    "plus_minus": 0.0,
                    "usage_rate": (usg_pct * 100.0 if usg_pct <= 1.0 else usg_pct),
                }

        if data_mode_for_season(season) == "legacy":
            return self._get_legacy_player_stats(player_id=player_id, season=season)

        profile = self.get_pbp_player_profile(player_id, season=season)
        return {
            "gp": profile.get("gp", 0),
            "min": profile.get("min", 0.0),
            "pts": profile.get("pts", 0.0),
            "fga": profile.get("fga", 0.0),
            "fgm": profile.get("fgm", 0.0),
            "fg_pct": profile.get("fg_pct", 0.0),
            "fg3a": profile.get("fg3a", 0.0),
            "fg3m": profile.get("fg3m", 0.0),
            "fg3_pct": profile.get("fg3_pct", 0.0),
            "fta": profile.get("fta", 0.0),
            "ftm": profile.get("ftm", 0.0),
            "ft_pct": profile.get("ft_pct", 0.0),
            "oreb": profile.get("oreb", 0.0),
            "dreb": profile.get("dreb", 0.0),
            "reb": profile.get("reb", 0.0),
            "ast": profile.get("ast", 0.0),
            "stl": profile.get("stl", 0.0),
            "blk": profile.get("blk", 0.0),
            "tov": profile.get("tov", 0.0),
            "pf": profile.get("pf", 0.0),
            "plus_minus": 0.0,
            "usage_rate": profile.get("pbp_usage_rate", 0.0),
        }

    def get_shot_chart(
        self, player_id: int, season: str = "2024-25"
    ) -> list[dict[str, Any]]:
        """Compatibility stub: shot-chart is not required in PBP-only mode."""
        _ = player_id
        _ = season
        return []

    def get_player_shots(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> list[dict[str, Any]]:
        """Return PBP shot-event rows for a player season."""
        if data_mode_for_season(season) == "legacy":
            return []

        if self._warehouse_available():
            w_rows = self._warehouse.get_player_shots(int(player_id), str(season))
            if w_rows:
                return w_rows

        return self._pbp.get_player_shots(
            player_id=player_id,
            season=season,
            season_type=season_type,
        )

    def get_team_roster(
        self, team_abbreviation: str, season: str = "2024-25"
    ) -> list[dict[str, Any]]:
        """Return team roster using source appropriate for requested season."""
        if self._warehouse_available():
            w_rows = self._warehouse.get_team_roster(str(team_abbreviation), str(season))
            if w_rows:
                return [
                    {
                        "player_id": int(r.get("player_id") or 0),
                        "full_name": str(r.get("full_name") or ""),
                        "position": str(r.get("position") or ""),
                    }
                    for r in w_rows
                    if int(r.get("player_id") or 0) > 0
                ]

        if data_mode_for_season(season) == "legacy":
            return self._get_legacy_team_roster(team_abbreviation=team_abbreviation, season=season)

        teams = self._pbp.get_teams("nba")
        abbr = (team_abbreviation or "").strip().upper()
        team_id = 0
        for row in teams:
            text = str(row.get("text", "") or "").strip().upper()
            if text == abbr:
                try:
                    team_id = int(row.get("id") or 0)
                except (TypeError, ValueError):
                    team_id = 0
                break
        if team_id <= 0:
            return []

        mapping = self._pbp.get_team_players_for_season(
            team_id=team_id,
            season=season,
            season_type="Regular Season",
        )
        out: list[dict[str, Any]] = []
        for pid_raw, full_name in mapping.items():
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "player_id": pid,
                    "full_name": str(full_name or ""),
                    "position": "",
                }
            )
        return out

    def get_league_averages(self, season: str = "2024-25") -> list[dict[str, Any]]:
        """Return pseudo league rows for percentile computations."""
        if self._warehouse_available():
            rows = self._warehouse.get_all_player_stats_for_season(str(season))
            if rows:
                return [
                    {
                        "PTS": float(r.get("pts_pg") or 0.0),
                        "AST": float(r.get("ast_pg") or 0.0),
                        "REB": float(r.get("reb_pg") or 0.0),
                        "STL": float(r.get("stl_pg") or 0.0),
                        "BLK": float(r.get("blk_pg") or 0.0),
                        "TOV": float(r.get("tov_pg") or 0.0),
                        "FGA": float(r.get("fga_pg") or 0.0),
                        "FG3A": float(r.get("fg3a_pg") or 0.0),
                        "FTA": float(r.get("fta_pg") or 0.0),
                    }
                    for r in rows
                ]

        if data_mode_for_season(season) == "legacy":
            return self._get_legacy_league_rows(season=season)

        rows = self._pbp.get_player_totals_table(season=season, season_type="Regular Season")
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                gp = float(row.get("GamesPlayed") or 0.0)
            except (TypeError, ValueError):
                gp = 0.0
            gp = max(gp, 1.0)

            def _num(key: str) -> float:
                try:
                    return float(row.get(key) or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            fga_pg = _num("FG2A") + _num("FG3A")
            fg3a_pg = _num("FG3A") / gp
            fta_pg = _num("FTA") / gp
            out.append(
                {
                    "PTS": _num("Points") / gp,
                    "AST": _num("Assists") / gp,
                    "REB": _num("Rebounds") / gp,
                    "STL": _num("Steals") / gp,
                    "BLK": _num("Blocks") / gp,
                    "TOV": (_num("LiveBallTurnovers") + _num("DeadBallTurnovers")) / gp,
                    "FGA": fga_pg,
                    "FG3A": fg3a_pg,
                    "FTA": fta_pg,
                }
            )
        return out

    def get_player_playtypes(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> dict[str, float]:
        """Derive playtype-like signals from PBP profile for compatibility."""
        _ = season_type
        if data_mode_for_season(season) == "legacy":
            return {}

        profile = self.get_pbp_player_profile(player_id, season=season)
        gp = max(1.0, float(profile.get("gp", 0.0) or 0.0))
        off_poss = float(profile.get("off_poss", 0.0) or 0.0)
        total_poss = float(profile.get("total_poss", 0.0) or 0.0)
        poss_pg = (off_poss / gp) if off_poss > 0.0 else ((total_poss / gp) * 0.5)
        usg_pct = float(profile.get("pbp_usage_rate", 0.0) or 0.0)
        usg = max(0.0, min(1.0, usg_pct / 100.0))
        rim = max(0.0, min(1.0, float(profile.get("at_rim_frequency", 0.0) or 0.0)))
        mid = max(
            0.0,
            min(
                1.0,
                float(profile.get("short_mid_frequency", 0.0) or 0.0)
                + float(profile.get("long_mid_frequency", 0.0) or 0.0),
            ),
        )
        pull_up = max(0.0, min(1.0, float(profile.get("unassisted_two_rate", 0.0) or 0.0)))

        reb_p36 = float(profile.get("reb", 0.0) or 0.0) * 36.0 / max(float(profile.get("min", 0.0) or 0.0), 1.0)
        blk_p36 = float(profile.get("blk", 0.0) or 0.0) * 36.0 / max(float(profile.get("min", 0.0) or 0.0), 1.0)
        fg3a_rate = float(profile.get("fg3a", 0.0) or 0.0) / max(float(profile.get("fga", 1.0) or 1.0), 1.0)
        size_big_proxy = max(
            0.0,
            min(1.0, 0.55 * ((reb_p36 - 5.0) / 8.0) + 0.25 * ((blk_p36 - 0.2) / 2.1) + 0.20 * ((1.0 - fg3a_rate - 0.35) / 0.55)),
        )
        post_fallback = max(
            0.0,
            0.2
            + 2.8 * size_big_proxy
            + 0.05 * usg_pct
            - 1.2 * (1.0 - size_big_proxy) * max(0.0, min(1.0, (fg3a_rate - 0.35) / 0.30)),
        )
        post_context = poss_pg * max(
            0.0,
            min(
                1.0,
                0.12 * rim + 0.18 * mid + 0.16 * pull_up + 0.06 * (1.0 - fg3a_rate),
            ),
        )
        post_up_possessions = min(8.0, max(0.0, 0.65 * post_fallback + 0.35 * post_context))

        return {
            "isolation_possessions": poss_pg * max(0.0, min(1.0, pull_up * 1.6 + usg * 0.20)),
            "isolation_ppp": float(profile.get("fg_pct", 0.0) or 0.0) * 2.0,
            "pick_and_roll_ball_handler_possessions": poss_pg * max(0.0, min(1.0, usg * 0.18 + pull_up * 0.20)),
            "pick_and_roll_ball_handler_ppp": float(profile.get("fg_pct", 0.0) or 0.0) * 2.0,
            "pick_and_roll_rollman_possessions": poss_pg * max(0.0, min(1.0, rim * 0.35)),
            "pick_and_roll_rollman_ppp": float(profile.get("at_rim_accuracy", profile.get("fg_pct", 0.0)) or 0.0) * 2.0,
            "post_up_possessions": post_up_possessions,
            "post_up_ppp": float(profile.get("fg_pct", 0.0) or 0.0) * 2.0,
            "cuts": poss_pg * max(0.0, min(1.0, rim * 0.22)),
            "cut_ppp": float(profile.get("at_rim_accuracy", profile.get("fg_pct", 0.0)) or 0.0) * 2.0,
            "handoff_possessions": poss_pg * max(0.0, min(1.0, float(profile.get("fg3a", 0.0) or 0.0) / max(float(profile.get("fga", 1.0) or 1.0), 1.0) * 0.18)),
            "handoff_ppp": float(profile.get("fg3_pct", 0.0) or 0.0) * 3.0,
            "spot_up_possessions": poss_pg * max(0.0, min(1.0, float(profile.get("catch_and_shoot_three_rate", 0.0) or 0.0) * 1.3)),
            "spot_up_ppp": float(profile.get("fg3_pct", 0.0) or 0.0) * 3.0,
            "off_screen_possessions": poss_pg * max(0.0, min(1.0, float(profile.get("fg3a", 0.0) or 0.0) / max(float(profile.get("fga", 1.0) or 1.0), 1.0) * 0.12)),
            "off_screen_ppp": float(profile.get("fg3_pct", 0.0) or 0.0) * 3.0,
            "transition_possessions": poss_pg * max(0.0, min(1.0, float(profile.get("second_chance_off_poss_rate", 0.0) or 0.0) + 0.06)),
            "transition_ppp": float(profile.get("ts_pct", 0.0) or 0.0) * 2.0,
        }

    def get_pbp_player_profile(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> dict[str, float]:
        """Return contextual PBP Stats profile for a player."""
        if data_mode_for_season(season) == "legacy":
            return {}

        if self._warehouse_available():
            w_profile = self._warehouse.get_pbp_profile(int(player_id), str(season))
            if w_profile:
                return {k: v for k, v in w_profile.items() if k not in {"player_id", "season"}}

        return self._pbp.get_player_profile(
            player_id=player_id,
            season=season,
            season_type=season_type,
        )

    def _get_legacy_player_stats(self, player_id: int, season: str) -> dict[str, Any]:
        """Return per-game historical stats using nba_api player profile endpoints."""
        cache_key = f"nba:legacy:player_stats:{int(player_id)}:{season}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                return cached

        out: dict[str, Any] = {
            "gp": 0,
            "min": 0.0,
            "pts": 0.0,
            "fga": 0.0,
            "fgm": 0.0,
            "fg_pct": 0.0,
            "fg3a": 0.0,
            "fg3m": 0.0,
            "fg3_pct": 0.0,
            "fta": 0.0,
            "ftm": 0.0,
            "ft_pct": 0.0,
            "oreb": 0.0,
            "dreb": 0.0,
            "reb": 0.0,
            "ast": 0.0,
            "stl": 0.0,
            "blk": 0.0,
            "tov": 0.0,
            "pf": 0.0,
            "plus_minus": 0.0,
            "usage_rate": 0.0,
        }

        try:
            from nba_api.stats.endpoints import playercareerstats

            endpoint = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36="PerGame", timeout=20)
            rows = endpoint.get_normalized_dict().get("SeasonTotalsRegularSeason", [])
            row = next((r for r in rows if str(r.get("SEASON_ID", "")).endswith(season)), None)
            if isinstance(row, dict):
                out.update(
                    {
                        "gp": int(row.get("GP") or 0),
                        "min": float(row.get("MIN") or 0.0),
                        "pts": float(row.get("PTS") or 0.0),
                        "fga": float(row.get("FGA") or 0.0),
                        "fgm": float(row.get("FGM") or 0.0),
                        "fg_pct": float(row.get("FG_PCT") or 0.0),
                        "fg3a": float(row.get("FG3A") or 0.0),
                        "fg3m": float(row.get("FG3M") or 0.0),
                        "fg3_pct": float(row.get("FG3_PCT") or 0.0),
                        "fta": float(row.get("FTA") or 0.0),
                        "ftm": float(row.get("FTM") or 0.0),
                        "ft_pct": float(row.get("FT_PCT") or 0.0),
                        "oreb": float(row.get("OREB") or 0.0),
                        "dreb": float(row.get("DREB") or 0.0),
                        "reb": float(row.get("REB") or 0.0),
                        "ast": float(row.get("AST") or 0.0),
                        "stl": float(row.get("STL") or 0.0),
                        "blk": float(row.get("BLK") or 0.0),
                        "tov": float(row.get("TOV") or 0.0),
                        "pf": float(row.get("PF") or 0.0),
                    }
                )
        except Exception:  # noqa: BLE001
            pass

        if self._cache is not None:
            self._cache.set(cache_key, out, ttl_seconds=2592000)
        return out

    def _get_legacy_team_roster(self, team_abbreviation: str, season: str) -> list[dict[str, Any]]:
        """Return historical team roster from nba_api CommonTeamRoster."""
        abbr = (team_abbreviation or "").strip().upper()
        team_id = int(_TEAM_ID_BY_ABBR.get(abbr) or 0)
        if team_id <= 0:
            return []

        cache_key = f"nba:legacy:team_roster:{abbr}:{season}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, list):
                return cached

        out: list[dict[str, Any]] = []
        try:
            from nba_api.stats.endpoints import commonteamroster

            endpoint = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=20)
            rows = endpoint.get_normalized_dict().get("CommonTeamRoster", [])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    pid = int(row.get("PLAYER_ID") or 0)
                except (TypeError, ValueError):
                    pid = 0
                if pid <= 0:
                    continue
                out.append(
                    {
                        "player_id": pid,
                        "full_name": str(row.get("PLAYER") or ""),
                        "position": str(row.get("POSITION") or ""),
                    }
                )
        except Exception:  # noqa: BLE001
            out = []

        if self._cache is not None:
            self._cache.set(cache_key, out, ttl_seconds=2592000)
        return out

    def _get_legacy_league_rows(self, season: str) -> list[dict[str, Any]]:
        """Return league rows from nba_api for percentile calculations in legacy seasons."""
        cache_key = f"nba:legacy:league_rows:{season}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if isinstance(cached, list):
                return cached

        out: list[dict[str, Any]] = []
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats

            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                timeout=25,
            )
            rows = endpoint.get_normalized_dict().get("LeagueDashPlayerStats", [])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                out.append(
                    {
                        "PTS": float(row.get("PTS") or 0.0),
                        "AST": float(row.get("AST") or 0.0),
                        "REB": float(row.get("REB") or 0.0),
                        "STL": float(row.get("STL") or 0.0),
                        "BLK": float(row.get("BLK") or 0.0),
                        "TOV": float(row.get("TOV") or 0.0),
                        "FGA": float(row.get("FGA") or 0.0),
                        "FG3A": float(row.get("FG3A") or 0.0),
                        "FTA": float(row.get("FTA") or 0.0),
                    }
                )
        except Exception:  # noqa: BLE001
            out = []

        if self._cache is not None:
            self._cache.set(cache_key, out, ttl_seconds=2592000)
        return out
