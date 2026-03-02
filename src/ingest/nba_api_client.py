"""NBA Stats API client using the nba_api library."""
from __future__ import annotations

import time
from typing import Any

from src.ingest.cache import Cache


def _parse_response(endpoint_obj: Any, result_set_index: int = 0) -> list[dict]:
    """Convert an nba_api endpoint result set into a list of dicts."""
    result_set = endpoint_obj.get_dict()["resultSets"][result_set_index]
    headers = result_set["headers"]
    rows = result_set["rowSet"]
    return [dict(zip(headers, row)) for row in rows]


class NBAApiClient:
    """Fetches live player stats from the nba_api library."""

    _RATE_LIMIT_SECONDS = 0.6

    def __init__(self, cache_dir: str | None = None) -> None:
        """
        Initialise the client.

        Parameters
        ----------
        cache_dir: Optional directory path for response caching.
        """
        self._cache: Cache | None = Cache(cache_dir) if cache_dir else None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search for players by name; returns list of matching records."""
        all_players = self._get_all_players()
        name_lower = name.lower()
        results = []
        for p in all_players:
            full_name = (p.get("DISPLAY_FIRST_LAST") or "").lower()
            if name_lower in full_name:
                results.append(
                    {
                        "player_id": p["PERSON_ID"],
                        "full_name": p.get("DISPLAY_FIRST_LAST", ""),
                        "team": p.get("TEAM_ABBREVIATION", ""),
                        "is_active": bool(p.get("ROSTERSTATUS", 0)),
                    }
                )
        return results

    def get_player_info(self, player_id: int) -> dict[str, Any]:
        """Return basic info for *player_id* (position, height, weight, team)."""
        cache_key = f"player_info:{player_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.endpoints import CommonPlayerInfo  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return CommonPlayerInfo(player_id=player_id)

        endpoint = self._with_retry(_call)
        rows = _parse_response(endpoint, 0)
        if not rows:
            return {}
        row = rows[0]
        result = {
            "position": row.get("POSITION", ""),
            "height": row.get("HEIGHT", ""),
            "weight": row.get("WEIGHT", ""),
            "team_id": row.get("TEAM_ID"),
            "team_abbreviation": row.get("TEAM_ABBREVIATION", ""),
        }
        self._cache_set(cache_key, result)
        return result

    def get_player_stats(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Retrieve per-game stats for *player_id*.

        Returns a flat dict with stat keys.
        """
        cache_key = f"player_stats:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.endpoints import PlayerDashboardByGeneralSplits  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return PlayerDashboardByGeneralSplits(
                player_id=player_id,
                season=season,
                per_mode_simple="PerGame",
            )

        endpoint = self._with_retry(_call)
        rows = _parse_response(endpoint, 0)
        if not rows:
            return {}
        row = rows[0]
        result = {
            "gp": row.get("GP", 0),
            "min": row.get("MIN", 0.0),
            "pts": row.get("PTS", 0.0),
            "fga": row.get("FGA", 0.0),
            "fgm": row.get("FGM", 0.0),
            "fg_pct": row.get("FG_PCT", 0.0),
            "fg3a": row.get("FG3A", 0.0),
            "fg3m": row.get("FG3M", 0.0),
            "fg3_pct": row.get("FG3_PCT", 0.0),
            "fta": row.get("FTA", 0.0),
            "ftm": row.get("FTM", 0.0),
            "ft_pct": row.get("FT_PCT", 0.0),
            "oreb": row.get("OREB", 0.0),
            "dreb": row.get("DREB", 0.0),
            "reb": row.get("REB", 0.0),
            "ast": row.get("AST", 0.0),
            "stl": row.get("STL", 0.0),
            "blk": row.get("BLK", 0.0),
            "tov": row.get("TOV", 0.0),
            "pf": row.get("PF", 0.0),
            "plus_minus": row.get("PLUS_MINUS", 0.0),
        }
        self._cache_set(cache_key, result)
        return result

    def get_shot_chart(
        self, player_id: int, season: str = "2024-25"
    ) -> list[dict[str, Any]]:
        """Return raw shot-chart rows for the given player and season."""
        cache_key = f"shot_chart:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.endpoints import ShotChartDetail  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return ShotChartDetail(
                player_id=player_id,
                team_id=0,
                season_nullable=season,
                context_measure_simple="FGA",
            )

        endpoint = self._with_retry(_call)
        rows = _parse_response(endpoint, 0)
        result = [
            {
                "shot_zone_basic": r.get("SHOT_ZONE_BASIC", ""),
                "shot_zone_area": r.get("SHOT_ZONE_AREA", ""),
                "shot_zone_range": r.get("SHOT_ZONE_RANGE", ""),
                "shot_made_flag": r.get("SHOT_MADE_FLAG", 0),
                "loc_x": r.get("LOC_X", 0),
                "loc_y": r.get("LOC_Y", 0),
                "shot_type": r.get("SHOT_TYPE", ""),
                "action_type": r.get("ACTION_TYPE", ""),
            }
            for r in rows
        ]
        self._cache_set(cache_key, result)
        return result

    def get_league_averages(self, season: str = "2024-25") -> dict[str, Any]:
        """Return league-wide per-game averages for percentile calculations."""
        cache_key = f"league_averages:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.endpoints import LeagueDashPlayerStats  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return LeagueDashPlayerStats(
                season=season,
                per_mode_simple="PerGame",
            )

        endpoint = self._with_retry(_call)
        rows = _parse_response(endpoint, 0)
        self._cache_set(cache_key, rows, ttl_seconds=604800)  # 1 week
        return rows  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_all_players(self) -> list[dict]:
        """Fetch all current-season players, using cache if available."""
        cache_key = "all_players"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.endpoints import CommonAllPlayers  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return CommonAllPlayers(is_only_current_season=1)

        endpoint = self._with_retry(_call)
        rows = _parse_response(endpoint, 0)
        self._cache_set(cache_key, rows, ttl_seconds=86400)
        return rows

    def _rate_limit(self) -> None:
        """Enforce minimum gap between consecutive API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._RATE_LIMIT_SECONDS:
            time.sleep(self._RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _with_retry(self, func: Any, max_retries: int = 3) -> Any:
        """Call *func* with exponential-backoff retry on failure."""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError(
            f"API call failed after {max_retries} retries"
        ) from last_exc

    def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        if self._cache is not None:
            self._cache.set(key, value, ttl_seconds=ttl_seconds)
