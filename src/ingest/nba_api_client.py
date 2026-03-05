"""NBA Stats API client using the nba_api library."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.ingest.cache import Cache

logger = logging.getLogger(__name__)


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

        endpoint = self._with_retry(_call, endpoint_name="CommonPlayerInfo")
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
                per_mode_detailed="PerGame",
                timeout=30,
            )

        endpoint = self._with_retry(_call, endpoint_name="PlayerDashboardByGeneralSplits")
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
                timeout=30,
            )

        endpoint = self._with_retry(_call, endpoint_name="ShotChartDetail")
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

    def get_team_roster(
        self, team_abbreviation: str, season: str = "2024-25"
    ) -> list[dict[str, Any]]:
        """Get all players on a team roster.

        Returns list of {player_id, full_name, position}.
        """
        cache_key = f"team_roster:{team_abbreviation}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        from nba_api.stats.static import teams as nba_teams  # noqa: PLC0415

        team_list = nba_teams.get_teams()
        team_info = next(
            (t for t in team_list if t["abbreviation"].upper() == team_abbreviation.upper()),
            None,
        )
        if team_info is None:
            return []

        from nba_api.stats.endpoints import CommonTeamRoster  # noqa: PLC0415

        def _call() -> Any:
            self._rate_limit()
            return CommonTeamRoster(team_id=team_info["id"], season=season)

        endpoint = self._with_retry(_call, endpoint_name="CommonTeamRoster")
        rows = _parse_response(endpoint, 0)
        result = [
            {
                "player_id": int(r.get("PlayerID") or r.get("PLAYER_ID", 0)),
                "full_name": r.get("PLAYER", r.get("DISPLAY_FIRST_LAST", "")),
                "position": r.get("POSITION", ""),
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
                per_mode_detailed="PerGame",
                timeout=30,
            )

        endpoint = self._with_retry(_call, endpoint_name="LeagueDashPlayerStats")
        rows = _parse_response(endpoint, 0)
        self._cache_set(cache_key, rows, ttl_seconds=604800)  # 1 week
        return rows  # type: ignore[return-value]

    def get_play_types(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Return play-type frequency and PPP data for *player_id* using SynergyPlayTypes.

        Keys returned (all float, 0.0 on failure):
            iso_freq, pnr_ball_freq, pnr_roll_freq, post_up_freq,
            spot_up_freq, handoff_freq, cut_freq, off_screen_freq,
            transition_freq, putback_freq
        """
        cache_key = f"play_types:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            from nba_api.stats.endpoints import SynergyPlayTypes  # noqa: PLC0415

            play_type_map = {
                "Isolation": "iso_freq",
                "PRBallHandler": "pnr_ball_freq",
                "PRRollman": "pnr_roll_freq",
                "Postup": "post_up_freq",
                "Spotup": "spot_up_freq",
                "Handoff": "handoff_freq",
                "Cut": "cut_freq",
                "OffScreen": "off_screen_freq",
                "Transition": "transition_freq",
                "OffRebound": "putback_freq",
            }
            result: dict[str, float] = {v: 0.0 for v in play_type_map.values()}

            for play_type in play_type_map:
                def _call(pt: str = play_type) -> Any:
                    self._rate_limit()
                    return SynergyPlayTypes(
                        player_id=player_id,
                        play_type_nullable=pt,
                        type_grouping_nullable="offensive",
                        per_mode_simple="PerGame",
                        season_year=season,
                        timeout=30,
                    )

                endpoint = self._with_retry(_call, endpoint_name=f"SynergyPlayTypes({play_type})")
                rows = _parse_response(endpoint, 0)
                if rows:
                    freq_pct = float(rows[0].get("POSS_PCT", 0.0) or 0.0)
                    result[play_type_map[play_type]] = freq_pct

            self._cache_set(cache_key, result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("SynergyPlayTypes failed for player_id=%s season=%s: %s: %s",
                           player_id, season, type(exc).__name__, exc)
            return {}

    def get_tracking_shots(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Return shot-tracking breakdown for *player_id* using PlayerDashPtShots.

        Keys returned (all float, 0.0 on failure):
            catch_shoot_fga, pull_up_fga, total_tracked_fga,
            avg_dribbles_before_shot
        """
        cache_key = f"tracking_shots:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            from nba_api.stats.endpoints import PlayerDashPtShots  # noqa: PLC0415

            def _call() -> Any:
                self._rate_limit()
                return PlayerDashPtShots(
                    player_id=player_id,
                    season=season,
                    per_mode_simple="Totals",
                    timeout=30,
                )

            endpoint = self._with_retry(_call, endpoint_name="PlayerDashPtShots")
            # result set 0 = GeneralShooting (dribble breakdown)
            dribble_rows = _parse_response(endpoint, 0)

            catch_shoot_fga = 0.0
            pull_up_fga = 0.0
            total_drib_fga = 0.0
            weighted_drib_sum = 0.0

            dribble_count_map = {
                "0 Dribbles": 0.0,
                "1 Dribble": 1.0,
                "2 Dribbles": 2.0,
                "3-6 Dribbles": 4.5,
                "7+ Dribbles": 8.0,
            }

            for row in dribble_rows:
                drib_label = str(row.get("DRIBBLE_RANGE", "") or "")
                fga = float(row.get("FGA", 0) or 0)
                drib_value = dribble_count_map.get(drib_label, -1.0)
                if drib_value < 0:
                    continue
                total_drib_fga += fga
                weighted_drib_sum += fga * drib_value
                if drib_value == 0.0:
                    catch_shoot_fga += fga
                elif drib_value >= 1.0:
                    pull_up_fga += fga

            avg_dribbles = (
                weighted_drib_sum / total_drib_fga if total_drib_fga > 0 else 0.0
            )

            result = {
                "catch_shoot_fga": catch_shoot_fga,
                "pull_up_fga": pull_up_fga,
                "total_tracked_fga": total_drib_fga,
                "avg_dribbles_before_shot": avg_dribbles,
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("PlayerDashPtShots failed for player_id=%s season=%s: %s: %s",
                           player_id, season, type(exc).__name__, exc)
            return {}

    def get_hustle_stats(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Return hustle stats for *player_id* using LeagueHustleStatsPlayer.

        Keys returned (all float, 0.0 on failure):
            deflections, contested_shots_2pt, contested_shots_3pt,
            charges_drawn, loose_balls_recovered, screen_assists, gp
        """
        cache_key = f"hustle_stats:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            from nba_api.stats.endpoints import LeagueHustleStatsPlayer  # noqa: PLC0415

            def _call() -> Any:
                self._rate_limit()
                return LeagueHustleStatsPlayer(
                    season=season,
                    per_mode_time="PerGame",
                    timeout=30,
                )

            endpoint = self._with_retry(_call, endpoint_name="LeagueHustleStatsPlayer")
            rows = _parse_response(endpoint, 0)
            player_row = next(
                (r for r in rows if r.get("PLAYER_ID") == player_id), None
            )
            if not player_row:
                return {}

            result = {
                "deflections": float(player_row.get("DEFLECTIONS", 0.0) or 0.0),
                "contested_shots_2pt": float(
                    player_row.get("CONTESTED_SHOTS_2PT", 0.0) or 0.0
                ),
                "contested_shots_3pt": float(
                    player_row.get("CONTESTED_SHOTS_3PT", 0.0) or 0.0
                ),
                "charges_drawn": float(player_row.get("CHARGES_DRAWN", 0.0) or 0.0),
                "loose_balls_recovered": float(
                    player_row.get("LOOSE_BALLS_RECOVERED", 0.0) or 0.0
                ),
                "screen_assists": float(player_row.get("SCREEN_ASSISTS", 0.0) or 0.0),
                "gp": float(player_row.get("G", 0.0) or 0.0),
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("LeagueHustleStatsPlayer failed for player_id=%s season=%s: %s: %s",
                           player_id, season, type(exc).__name__, exc)
            return {}

    def get_passing_tracking(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Return pass-tracking data for *player_id* using PlayerDashPtPass.

        Keys returned (all float, 0.0 on failure):
            passes_made, potential_assists, ast_adjust
        """
        cache_key = f"passing_tracking:{player_id}:{season}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            from nba_api.stats.endpoints import PlayerDashPtPass  # noqa: PLC0415

            def _call() -> Any:
                self._rate_limit()
                return PlayerDashPtPass(
                    player_id=player_id,
                    season=season,
                    per_mode_simple="PerGame",
                    timeout=30,
                )

            endpoint = self._with_retry(_call, endpoint_name="PlayerDashPtPass")
            # result set 0 = passes made
            rows = _parse_response(endpoint, 0)
            if not rows:
                return {}

            # Aggregate across all pass-to targets
            passes_made = sum(float(r.get("PASSES", 0.0) or 0.0) for r in rows)
            potential_ast = sum(
                float(r.get("POTENTIAL_AST", 0.0) or 0.0) for r in rows
            )
            ast_adjust = sum(
                float(r.get("AST_ADJ", 0.0) or 0.0) for r in rows
            )

            result = {
                "passes_made": passes_made,
                "potential_assists": potential_ast,
                "ast_adjust": ast_adjust,
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("PlayerDashPtPass failed for player_id=%s season=%s: %s: %s",
                           player_id, season, type(exc).__name__, exc)
            return {}

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

        endpoint = self._with_retry(_call, endpoint_name="CommonAllPlayers")
        rows = _parse_response(endpoint, 0)
        self._cache_set(cache_key, rows, ttl_seconds=86400)
        return rows

    def _rate_limit(self) -> None:
        """Enforce minimum gap between consecutive API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._RATE_LIMIT_SECONDS:
            time.sleep(self._RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _with_retry(
        self,
        func: Any,
        max_retries: int = 3,
        endpoint_name: str = "NBA API endpoint",
    ) -> Any:
        """Call *func* with exponential-backoff retry on failure."""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return func()
            except (TypeError, ValueError):
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                time.sleep(wait)
        detail = ""
        if last_exc is not None:
            detail = f": {type(last_exc).__name__}: {last_exc}"
        raise RuntimeError(
            f"{endpoint_name} failed after {max_retries} retries{detail}"
        ) from last_exc

    def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        if self._cache is not None:
            self._cache.set(key, value, ttl_seconds=ttl_seconds)
