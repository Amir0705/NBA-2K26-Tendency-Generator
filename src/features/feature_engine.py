"""Feature engineering: transforms raw NBA stats into a feature vector."""
from __future__ import annotations

import os
from typing import Any

from src.features.shot_zones import ShotZoneAnalyzer, ZONES
from src.seasons import prior_seasons, season_start_year

# Position one-hot keys
_POSITIONS = ("PG", "SG", "SF", "PF", "C")

# Map from nba_api position strings to canonical 2-letter codes
_POSITION_MAP = {
    "guard": "PG",
    "guard-forward": "SG",
    "forward-guard": "SF",
    "forward": "SF",
    "forward-center": "PF",
    "center-forward": "PF",
    "center": "C",
}


def _map_position(raw: str) -> str:
    """Map nba_api position string to canonical position code."""
    key = raw.strip().lower()
    if not key:
        return "SF"

    if key in _POSITION_MAP:
        return _POSITION_MAP[key]

    cleaned = key.replace("/", "-").replace(" ", "")
    abbr_map = {
        "pg": "PG",
        "sg": "SG",
        "sf": "SF",
        "pf": "PF",
        "c": "C",
        "g": "PG",
        "f": "SF",
    }

    if cleaned in abbr_map:
        return abbr_map[cleaned]

    if "-" in cleaned:
        parts = [p for p in cleaned.split("-") if p]
        for part in parts:
            if part in abbr_map:
                return abbr_map[part]
            if part in _POSITION_MAP:
                return _POSITION_MAP[part]

    if "guard" in key:
        return "PG"
    if "center" in key:
        return "C"
    if "forward" in key:
        return "SF"

    return "SF"


def _height_to_inches(height_str: str) -> int:
    """Convert 'ft-in' string (e.g. '6-6') to total inches."""
    try:
        parts = height_str.replace('"', "").split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except Exception:  # noqa: BLE001
        return 78  # league-average fallback (~6-6)


def _weight_to_lbs(weight_str: str) -> int:
    """Convert weight string (e.g. '240') to pounds."""
    try:
        return int(float(str(weight_str).strip()))
    except Exception:  # noqa: BLE001
        return 220


def _per36(stat_per_game: float, minutes_per_game: float) -> float:
    """Compute per-36-minute rate from per-game values; avoids division by zero."""
    return stat_per_game * 36.0 / max(minutes_per_game, 1.0)


def _compute_age(birthdate_str: str, season: str | None = None) -> int:
    """Return player age using season end date when season is provided."""
    try:
        from datetime import date
        bd = birthdate_str[:10]  # 'YYYY-MM-DD...'
        parts = bd.split("-")
        born = date(int(parts[0]), int(parts[1]), int(parts[2]))
        if season:
            start_year = season_start_year(season)
            as_of = date(start_year + 1, 6, 30)
        else:
            as_of = date.today()
        return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))
    except Exception:  # noqa: BLE001
        return 25  # league-average fallback


def _percentile(value: float, all_values: list[float]) -> float:
    """Return empirical percentile (0–1) of *value* in *all_values*."""
    if not all_values:
        return 0.5
    below = sum(1 for v in all_values if v < value)
    return below / len(all_values)


def _first_numeric(stats: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    """Return first numeric value found under the provided keys."""
    for key in keys:
        if key in stats:
            try:
                return float(stats.get(key, default))
            except (TypeError, ValueError):
                continue
    return float(default)


class FeatureEngine:
    """Transforms raw NBA stats into model-ready tendency features."""

    def __init__(self, nba_client: Any) -> None:
        """
        Initialise engine.

        Parameters
        ----------
        nba_client: NBAApiClient instance (or any duck-typed equivalent).
        """
        self._client = nba_client
        self._league_averages: dict[str, list[dict[str, Any]]] = {}
        self._zone_analyzer = ShotZoneAnalyzer()

    def build_features(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Build complete feature vector for a player.

        Parameters
        ----------
        player_id: NBA player ID.
        season:    Season string (e.g. "2024-25").

        Returns
        -------
        Feature dict ready for FormulaLayer.generate().
        """
        if os.environ.get("N2K_PBP_ONLY", "1").strip() == "1":
            pbp_profile = self._client.get_pbp_player_profile(player_id, season=season)
            if pbp_profile:
                return self._build_pbp_only_features(player_id, season, pbp_profile)

        # Fetch raw data
        info = self._client.get_player_info(player_id)
        stats = self._client.get_player_stats(player_id, season=season)
        shot_chart = self._client.get_shot_chart(player_id, season=season)
        try:
            playtypes = self._client.get_player_playtypes(player_id, season=season)
        except Exception:  # noqa: BLE001
            playtypes = {}
        try:
            pbp_profile = self._client.get_pbp_player_profile(player_id, season=season)
        except Exception:  # noqa: BLE001
            pbp_profile = {}
        has_pbp_context = bool(pbp_profile)

        def _clip01(v: float) -> float:
            return max(0.0, min(1.0, v))

        def _scale01(v: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.5
            return _clip01((v - lo) / (hi - lo))

        # --- Player info ---
        position = _map_position(info.get("position", ""))
        height_inches = _height_to_inches(info.get("height", ""))
        weight_lbs = int(info.get("weight") or 0)

        # Age
        age = _compute_age(info.get("birthdate", ""), season=season)

        # --- Volume stats (per game) ---
        gp = int(stats.get("gp", 0))
        min_pg = float(stats.get("min", 0.0))
        pts_pg = float(stats.get("pts", 0.0))
        fga_pg = float(stats.get("fga", 0.0))
        fg3a_pg = float(stats.get("fg3a", 0.0))
        fta_pg = float(stats.get("fta", 0.0))
        ast_pg = float(stats.get("ast", 0.0))
        reb_pg = float(stats.get("reb", 0.0))
        oreb_pg = float(stats.get("oreb", 0.0))
        dreb_pg = float(stats.get("dreb", 0.0))
        stl_pg = float(stats.get("stl", 0.0))
        blk_pg = float(stats.get("blk", 0.0))
        tov_pg = float(stats.get("tov", 0.0))
        pf_pg = float(stats.get("pf", 0.0))

        # --- Shooting efficiency ---
        fg_pct = float(stats.get("fg_pct", 0.0))
        fg3_pct = float(stats.get("fg3_pct", 0.0))
        ft_pct = float(stats.get("ft_pct", 0.0))
        efg_pct = (
            (float(stats.get("fgm", 0.0)) + 0.5 * float(stats.get("fg3m", 0.0)))
            / max(fga_pg, 0.001)
        )
        ts_pct = pts_pg / max(2 * (fga_pg + 0.44 * fta_pg), 0.001)

        # --- Per-36 rates ---
        per36 = lambda s: _per36(s, min_pg)  # noqa: E731
        pts_p36 = per36(pts_pg)
        fga_p36 = per36(fga_pg)
        fg3a_p36 = per36(fg3a_pg)
        fta_p36 = per36(fta_pg)
        ast_p36 = per36(ast_pg)
        reb_p36 = per36(reb_pg)
        oreb_p36 = per36(oreb_pg)
        dreb_p36 = per36(dreb_pg)
        stl_p36 = per36(stl_pg)
        blk_p36 = per36(blk_pg)
        tov_p36 = per36(tov_pg)
        pf_p36 = per36(pf_pg)

        # --- Derived ratios ---
        # USG% proxy: fraction of team possessions used while on court
        # ~2.2 possessions per minute is the empirical team rate
        possessions_used = fga_pg + 0.44 * fta_pg + tov_pg
        team_possessions_while_on = min_pg * 2.2  # ~2.2 poss/min for a team
        usg_pct_proxy = possessions_used / max(team_possessions_while_on, 1.0)

        league_avg_ast_p36 = 5.0  # approximate
        ast_pct_proxy = ast_p36 / max(league_avg_ast_p36, 0.001)
        tov_pct_proxy = tov_pg / max(fga_pg + 0.44 * fta_pg + tov_pg, 0.001)
        ast_to_tov = ast_pg / max(tov_pg, 0.1)
        fg3a_rate = fg3a_pg / max(fga_pg, 1)
        fta_rate = fta_pg / max(fga_pg, 1)
        oreb_pct_proxy = oreb_pg / max(reb_pg, 1)

        # Optional possession-style signals (if available in upstream stats feed)
        usage_rate_pct = _first_numeric(
            stats,
            ("usage_rate", "usage_rate_pct", "usg_pct", "USG_PCT"),
            default=usg_pct_proxy * 100.0,
        )
        # Legacy stat feeds sometimes include usage columns with placeholder 0.
        # Treat non-positive values as missing and fall back to the possessions proxy.
        if usage_rate_pct <= 0.0:
            usage_rate_pct = usg_pct_proxy * 100.0
        pbp_usage_rate = float(pbp_profile.get("pbp_usage_rate", 0.0) or 0.0)
        if pbp_usage_rate > 0.0:
            usage_rate_pct = pbp_usage_rate
            usg_pct_proxy = (0.75 * usg_pct_proxy) + (0.25 * (pbp_usage_rate / 100.0))
        isolation_possessions = float(playtypes.get("isolation_possessions", 0.0) or 0.0)
        pnr_ball_handler_possessions = float(
            playtypes.get("pick_and_roll_ball_handler_possessions", 0.0) or 0.0
        )
        pnr_rollman_possessions = float(
            playtypes.get("pick_and_roll_rollman_possessions", 0.0) or 0.0
        )
        post_up_possessions = float(playtypes.get("post_up_possessions", 0.0) or 0.0)
        cuts_possessions = float(playtypes.get("cuts", 0.0) or 0.0)
        handoff_possessions = float(playtypes.get("handoff_possessions", 0.0) or 0.0)
        spot_up_possessions = float(playtypes.get("spot_up_possessions", 0.0) or 0.0)
        off_screen_possessions = float(playtypes.get("off_screen_possessions", 0.0) or 0.0)
        transition_possessions = float(playtypes.get("transition_possessions", 0.0) or 0.0)

        # PPP efficiency data
        isolation_ppp = float(playtypes.get("isolation_ppp", 0.0) or 0.0)
        pnr_ball_handler_ppp = float(playtypes.get("pick_and_roll_ball_handler_ppp", 0.0) or 0.0)
        pnr_rollman_ppp = float(playtypes.get("pick_and_roll_rollman_ppp", 0.0) or 0.0)
        post_up_ppp = float(playtypes.get("post_up_ppp", 0.0) or 0.0)
        spot_up_ppp = float(playtypes.get("spot_up_ppp", 0.0) or 0.0)

        # Fall back to optional upstream stats fields if Synergy data is unavailable.
        if isolation_possessions <= 0.0:
            isolation_possessions = _first_numeric(
                stats,
                ("isolation_possessions", "iso_possessions", "isolation_poss"),
            )
        if pnr_ball_handler_possessions <= 0.0:
            pnr_ball_handler_possessions = _first_numeric(
                stats,
                (
                    "pick_and_roll_ball_handler_possessions",
                    "pnr_ball_handler_possessions",
                    "pnr_bh_possessions",
                ),
            )
        if pnr_rollman_possessions <= 0.0:
            pnr_rollman_possessions = _first_numeric(
                stats,
                (
                    "pick_and_roll_rollman_possessions",
                    "pnr_rollman_possessions",
                ),
            )
        if post_up_possessions <= 0.0:
            post_up_possessions = _first_numeric(
                stats,
                ("post_up_possessions", "post_ups", "post_possessions"),
            )
        if cuts_possessions <= 0.0:
            cuts_possessions = _first_numeric(
                stats,
                ("cuts", "cut_possessions", "cut_poss"),
            )
        if handoff_possessions <= 0.0:
            handoff_possessions = _first_numeric(
                stats,
                ("handoff_possessions", "handoff_poss", "dhos"),
            )
        midrange_attempts = _first_numeric(
            stats,
            ("midrange_attempts", "mid_range_attempts", "midrange_fga"),
        )

        assisted_2pt_pct = 0.0
        assisted_3pt_pct = 0.0
        catch_and_shoot_three_rate = 0.0
        pull_up_three_rate = 0.0
        unassisted_two_rate = 0.0
        putback_rate = 0.0
        live_ball_turnover_pct = 0.0
        shooting_fouls_drawn_pct = 0.0
        three_pt_fouls_drawn_pct = 0.0
        seconds_per_poss_off = 0.0
        second_chance_off_poss_rate = 0.0

        if not has_pbp_context:
            # Legacy seasons do not include rich shot-context endpoints in this
            # pipeline path, so estimate context signals from box-score shape.
            two_pt_rate = max(0.0, 1.0 - fg3a_rate)
            creator_load = _clip01(
                0.50 * _scale01(usg_pct_proxy, 0.15, 0.35)
                + 0.30 * _scale01(ast_p36, 2.0, 9.0)
                + 0.20 * _scale01(pts_p36, 12.0, 32.0)
            )
            size_big_proxy = _clip01(
                0.55 * _scale01(reb_p36, 5.0, 13.0)
                + 0.25 * _scale01(blk_p36, 0.2, 2.3)
                + 0.20 * _scale01(1.0 - fg3a_rate, 0.35, 0.90)
            )
            rim_proxy = _clip01(
                0.42 * _scale01(fta_rate, 0.12, 0.45)
                + 0.30 * _scale01(oreb_p36, 0.4, 4.5)
                + 0.18 * _scale01(blk_p36, 0.1, 2.2)
                + 0.10 * (1.0 - _scale01(fg3a_rate, 0.15, 0.55))
            )
            mid_proxy = _clip01(0.58 * two_pt_rate + 0.27 * creator_load - 0.25 * rim_proxy)
            # Pure paint-centers (Shaq/Dwight archetype): high rim dominance combined
            # with poor FT% is a reliable signal that mid-range volume is near-zero.
            # Without this cap the high two_pt_rate + usage inflates mid estimates badly.
            if rim_proxy > 0.65 and ft_pct < 0.62:
                paint_factor = min(
                    1.0,
                    (rim_proxy - 0.65) / 0.35 + max(0.0, 0.62 - ft_pct) / 0.22,
                )
                mid_proxy = min(mid_proxy, max(0.06, 0.22 - 0.16 * paint_factor))
            midrange_attempts = max(
                float(midrange_attempts),
                fga_pg * max(0.05, min(0.62, mid_proxy)),
            )

            assisted_2pt_pct = max(30.0, min(78.0, 72.0 - 42.0 * creator_load))
            assisted_3pt_pct = max(55.0, min(92.0, 88.0 - 30.0 * creator_load))
            assisted_2pt_rate = assisted_2pt_pct / 100.0
            assisted_3pt_rate = assisted_3pt_pct / 100.0

            catch_and_shoot_three_rate = fg3a_rate * assisted_3pt_rate
            pull_up_three_rate = fg3a_rate * (1.0 - assisted_3pt_rate)
            unassisted_two_rate = max(0.04, min(0.38, two_pt_rate * (1.0 - assisted_2pt_rate)))
            putback_rate = _clip01(0.06 * (0.65 * _scale01(oreb_p36, 0.4, 4.5) + 0.35 * rim_proxy))
            live_ball_turnover_pct = _clip01(0.24 + 0.42 * _scale01(tov_pct_proxy, 0.08, 0.22))
            shooting_fouls_drawn_pct = _clip01(0.03 + 0.18 * _scale01(fta_rate, 0.08, 0.45))
            three_pt_fouls_drawn_pct = _clip01(0.002 + 0.018 * _scale01(pull_up_three_rate, 0.0, 0.20))
            seconds_per_poss_off = 12.0 + 4.8 * (1.0 - creator_load)
            second_chance_off_poss_rate = _clip01(0.03 + 0.17 * _scale01(oreb_p36, 0.4, 4.5))

            poss_used_pg = fga_pg + 0.44 * fta_pg + tov_pg
            iso_proxy = poss_used_pg * (0.06 + 0.22 * creator_load)
            pnr_bh_proxy = poss_used_pg * (0.10 + 0.25 * creator_load)
            pnr_roll_proxy = poss_used_pg * (0.04 + 0.20 * rim_proxy * (1.0 - creator_load))
            post_proxy = poss_used_pg * (0.03 + 0.17 * size_big_proxy)
            cuts_proxy = poss_used_pg * (0.05 + 0.11 * rim_proxy * (1.0 - creator_load))
            handoff_proxy = poss_used_pg * (0.02 + 0.09 * _scale01(fg3a_rate, 0.10, 0.55))
            spot_up_proxy = poss_used_pg * (0.04 + 0.24 * catch_and_shoot_three_rate)
            off_screen_proxy = poss_used_pg * (0.02 + 0.14 * catch_and_shoot_three_rate)
            transition_proxy = poss_used_pg * (0.06 + 0.12 * _scale01(stl_p36, 0.5, 2.5))

            isolation_possessions = max(isolation_possessions, iso_proxy)
            pnr_ball_handler_possessions = max(pnr_ball_handler_possessions, pnr_bh_proxy)
            pnr_rollman_possessions = max(pnr_rollman_possessions, pnr_roll_proxy)
            post_up_possessions = max(post_up_possessions, post_proxy)
            cuts_possessions = max(cuts_possessions, cuts_proxy)
            handoff_possessions = max(handoff_possessions, handoff_proxy)
            spot_up_possessions = max(spot_up_possessions, spot_up_proxy)
            off_screen_possessions = max(off_screen_possessions, off_screen_proxy)
            transition_possessions = max(transition_possessions, transition_proxy)

            isolation_ppp = max(isolation_ppp, ts_pct * 1.75)
            pnr_ball_handler_ppp = max(pnr_ball_handler_ppp, ts_pct * 1.80)
            pnr_rollman_ppp = max(pnr_rollman_ppp, fg_pct * 1.95)
            post_up_ppp = max(post_up_ppp, fg_pct * 1.85)
            spot_up_ppp = max(spot_up_ppp, fg3_pct * 2.85)

        if has_pbp_context:
            assisted_2pt_pct = float(pbp_profile.get("assisted_2pt_pct", 0.0) or 0.0)
            assisted_3pt_pct = float(pbp_profile.get("assisted_3pt_pct", 0.0) or 0.0)
            catch_and_shoot_three_rate = float(
                pbp_profile.get("catch_and_shoot_three_rate", 0.0) or 0.0
            )
            pull_up_three_rate = float(pbp_profile.get("pull_up_three_rate", 0.0) or 0.0)
            unassisted_two_rate = float(pbp_profile.get("unassisted_two_rate", 0.0) or 0.0)
            putback_rate = float(pbp_profile.get("putback_rate", 0.0) or 0.0)
            live_ball_turnover_pct = float(
                pbp_profile.get("live_ball_turnover_pct", 0.0) or 0.0
            )
            shooting_fouls_drawn_pct = float(
                pbp_profile.get("shooting_fouls_drawn_pct", 0.0) or 0.0
            )
            three_pt_fouls_drawn_pct = float(
                pbp_profile.get("three_pt_fouls_drawn_pct", 0.0) or 0.0
            )
            seconds_per_poss_off = float(pbp_profile.get("seconds_per_poss_off", 0.0) or 0.0)
            second_chance_off_poss_rate = float(
                pbp_profile.get("second_chance_off_poss_rate", 0.0) or 0.0
            )

        # --- Shot zone features ---
        total_minutes = min_pg * gp
        has_shot_chart = len(shot_chart) > 0
        zone_data = self._zone_analyzer.analyze(shot_chart, total_minutes)

        if (not has_shot_chart) and (not has_pbp_context):
            # Legacy seasons often have no shot-chart rows in this path.
            # Build a stable zone profile from available box-score proxies.
            # Allow rim-dominant poor-FT% bigs to reach a very low mid share.
            _mid_share_floor = 0.04 if (rim_proxy > 0.65 and ft_pct < 0.62) else 0.12
            mid_share = max(_mid_share_floor, min(0.48, float(midrange_attempts) / max(fga_pg, 1.0)))
            three_share = max(0.05, min(0.50, fg3a_rate))
            rim_share = max(0.08, min(0.52, 0.30 + 0.22 * fta_rate - 0.16 * mid_share))
            paint_share = max(0.04, 1.0 - rim_share - mid_share - three_share)
            total_share = max(rim_share + paint_share + mid_share + three_share, 1e-6)
            rim_share /= total_share
            paint_share /= total_share
            mid_share /= total_share
            three_share /= total_share

            corner_share = max(0.15, min(0.45, 0.30 + 0.35 * (1.0 - (assisted_3pt_pct / 100.0))))
            c3 = three_share * corner_share
            ab3 = three_share - c3

            rates = {
                "ra": rim_share,
                "paint": paint_share,
                "mid_left": mid_share * 0.32,
                "mid_center": mid_share * 0.36,
                "mid_right": mid_share * 0.32,
                "corner3_left": c3 * 0.50,
                "corner3_right": c3 * 0.50,
                "above_break3": ab3,
            }

            zone_data["zone_fga_rate"].update(rates)
            zone_data["zone_fg_pct"].update(
                {
                    "ra": max(0.48, min(0.78, fg_pct + 0.11)),
                    "paint": max(0.40, min(0.70, fg_pct + 0.05)),
                    "mid_left": max(0.32, min(0.56, fg_pct - 0.06)),
                    "mid_center": max(0.32, min(0.56, fg_pct - 0.05)),
                    "mid_right": max(0.32, min(0.56, fg_pct - 0.06)),
                    "corner3_left": max(0.26, min(0.50, fg3_pct)),
                    "corner3_right": max(0.26, min(0.50, fg3_pct)),
                    "above_break3": max(0.24, min(0.48, fg3_pct - 0.01)),
                }
            )

            for zone_key in ZONES:
                rate = float(zone_data["zone_fga_rate"].get(zone_key, 0.0) or 0.0)
                zone_data["zone_fga_per36"][zone_key] = rate * fga_p36
                zone_data["zone_pref_vs_league"][zone_key] = rate / (1.0 / max(len(ZONES), 1))

            zone_data["sub_zone_distribution_close"] = {"left": 32.0, "middle": 36.0, "right": 32.0}
            zone_data["sub_zone_distribution_mid"] = {
                "left": 20.0,
                "left_center": 20.0,
                "center": 20.0,
                "right_center": 20.0,
                "right": 20.0,
            }
            zone_data["sub_zone_distribution_three"] = {
                "left": 18.0,
                "left_center": 19.0,
                "center": 26.0,
                "right_center": 19.0,
                "right": 18.0,
            }

        # --- Drive right bias: % of close-range shots from right side (LOC_X > 0) ---
        close_shots = [
            s for s in shot_chart
            if s.get("shot_zone_basic", "") in ("Restricted Area", "In The Paint (Non-RA)")
        ]
        if close_shots:
            right_shots = sum(
                1 for s in close_shots if (s.get("loc_x", 0) or 0) > 0
            )
            drive_right_bias = 25.0 + (right_shots / len(close_shots)) * 50.0
        else:
            drive_right_bias = 50.0  # neutral fallback

        # --- League percentiles ---
        league_rows = self._get_league_averages(season)
        pctile_pts = _percentile(pts_pg, [r.get("PTS", 0) for r in league_rows])
        pctile_ast = _percentile(ast_pg, [r.get("AST", 0) for r in league_rows])
        pctile_reb = _percentile(reb_pg, [r.get("REB", 0) for r in league_rows])
        pctile_stl = _percentile(stl_pg, [r.get("STL", 0) for r in league_rows])
        pctile_blk = _percentile(blk_pg, [r.get("BLK", 0) for r in league_rows])
        league_fg3a_rates = [
            r.get("FG3A", 0) / max(r.get("FGA", 1), 1) for r in league_rows
        ]
        league_fta_rates = [
            r.get("FTA", 0) / max(r.get("FGA", 1), 1) for r in league_rows
        ]
        pctile_fg3a_rate = _percentile(fg3a_rate, league_fg3a_rates)
        pctile_fta_rate = _percentile(fta_rate, league_fta_rates)
        pctile_tov = _percentile(tov_pg, [r.get("TOV", 0) for r in league_rows])

        # --- Position one-hot ---
        pos_flags = {f"is_{p.lower()}": (position == p) for p in _POSITIONS}

        features: dict[str, Any] = {
            # Player info
            "position": position,
            "height_inches": height_inches,
            "weight_lbs": weight_lbs,
            "age": age,
            # Volume stats
            "pts_per_game": pts_pg,
            "fga_per_game": fga_pg,
            "fg3a_per_game": fg3a_pg,
            "fta_per_game": fta_pg,
            "ast_per_game": ast_pg,
            "reb_per_game": reb_pg,
            "stl_per_game": stl_pg,
            "blk_per_game": blk_pg,
            "tov_per_game": tov_pg,
            "min_per_game": min_pg,
            "gp": gp,
            # Per-36 rates
            "pts_per36": pts_p36,
            "fga_per36": fga_p36,
            "fg3a_per36": fg3a_p36,
            "fta_per36": fta_p36,
            "ast_per36": ast_p36,
            "reb_per36": reb_p36,
            "oreb_per36": oreb_p36,
            "dreb_per36": dreb_p36,
            "stl_per36": stl_p36,
            "blk_per36": blk_p36,
            "tov_per36": tov_p36,
            "pf_per36": pf_p36,
            # Shooting efficiency
            "fg_pct": fg_pct,
            "fg3_pct": fg3_pct,
            "ft_pct": ft_pct,
            "efg_pct": efg_pct,
            "ts_pct": ts_pct,
            # Derived ratios
            "usg_pct_proxy": usg_pct_proxy,
            "ast_pct_proxy": ast_pct_proxy,
            "tov_pct_proxy": tov_pct_proxy,
            "ast_to_tov": ast_to_tov,
            "fg3a_rate": fg3a_rate,
            "fta_rate": fta_rate,
            "oreb_pct_proxy": oreb_pct_proxy,
            "usage_rate": usage_rate_pct,
            "isolation_possessions": isolation_possessions,
            "pick_and_roll_ball_handler_possessions": pnr_ball_handler_possessions,
            "pick_and_roll_rollman_possessions": pnr_rollman_possessions,
            "post_up_possessions": post_up_possessions,
            "cuts": cuts_possessions,
            "handoff_possessions": handoff_possessions,
            "spot_up_possessions": spot_up_possessions,
            "off_screen_possessions": off_screen_possessions,
            "transition_possessions": transition_possessions,
            "isolation_ppp": isolation_ppp,
            "pick_and_roll_ball_handler_ppp": pnr_ball_handler_ppp,
            "pick_and_roll_rollman_ppp": pnr_rollman_ppp,
            "post_up_ppp": post_up_ppp,
            "spot_up_ppp": spot_up_ppp,
            "midrange_attempts": midrange_attempts,
            "assisted_2pt_pct": assisted_2pt_pct,
            "assisted_3pt_pct": assisted_3pt_pct,
            "catch_and_shoot_three_rate": catch_and_shoot_three_rate,
            "pull_up_three_rate": pull_up_three_rate,
            "unassisted_two_rate": unassisted_two_rate,
            "putback_rate": putback_rate,
            "live_ball_turnover_pct": live_ball_turnover_pct,
            "shooting_fouls_drawn_pct": shooting_fouls_drawn_pct,
            "three_pt_fouls_drawn_pct": three_pt_fouls_drawn_pct,
            "seconds_per_poss_off": seconds_per_poss_off,
            "second_chance_off_poss_rate": second_chance_off_poss_rate,
            "pbp_data_available": (1.0 if pbp_profile else 0.0),
            # Shot zone features
            **{f"zone_fga_rate_{z}": zone_data["zone_fga_rate"][z] for z in ZONES},
            **{f"zone_fg_pct_{z}": zone_data["zone_fg_pct"][z] for z in ZONES},
            **{f"zone_fga_per36_{z}": zone_data["zone_fga_per36"][z] for z in ZONES},
            **{f"zone_pref_vs_league_{z}": zone_data["zone_pref_vs_league"][z] for z in ZONES},
            "sub_zone_distribution_close": zone_data["sub_zone_distribution_close"],
            "sub_zone_distribution_mid": zone_data["sub_zone_distribution_mid"],
            "sub_zone_distribution_three": zone_data["sub_zone_distribution_three"],
            "drive_right_bias": drive_right_bias,
            # Percentiles
            "pctile_pts": pctile_pts,
            "pctile_ast": pctile_ast,
            "pctile_reb": pctile_reb,
            "pctile_stl": pctile_stl,
            "pctile_blk": pctile_blk,
            "pctile_fg3a_rate": pctile_fg3a_rate,
            "pctile_fta_rate": pctile_fta_rate,
            "pctile_tov": pctile_tov,
            # Data quality flags
            "has_shot_chart": has_shot_chart,
            "low_minutes": (gp < 5 or min_pg < 5),
            "games_played": gp,
        }
        features.update(pos_flags)
        return features

    def _build_pbp_only_features(
        self,
        player_id: int,
        season: str,
        pbp: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct full feature vector from PBP-only profile fields."""
        info = self._client.get_player_info(player_id)

        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        def _pctile(v: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.5
            return _clamp01((v - lo) / (hi - lo))

        gp = int(float(pbp.get("gp", 0.0) or 0.0))
        min_pg = float(pbp.get("min", 0.0) or 0.0)
        pts_pg = float(pbp.get("pts", 0.0) or 0.0)
        fga_pg = float(pbp.get("fga", 0.0) or 0.0)
        fg3a_pg = float(pbp.get("fg3a", 0.0) or 0.0)
        fta_pg = float(pbp.get("fta", 0.0) or 0.0)
        ast_pg = float(pbp.get("ast", 0.0) or 0.0)
        reb_pg = float(pbp.get("reb", 0.0) or 0.0)
        oreb_pg = float(pbp.get("oreb", 0.0) or 0.0)
        dreb_pg = float(pbp.get("dreb", 0.0) or 0.0)
        stl_pg = float(pbp.get("stl", 0.0) or 0.0)
        blk_pg = float(pbp.get("blk", 0.0) or 0.0)
        tov_pg = float(pbp.get("tov", 0.0) or 0.0)
        pf_pg = float(pbp.get("pf", 0.0) or 0.0)

        fg_pct = float(pbp.get("fg_pct", 0.0) or 0.0)
        fg3_pct = float(pbp.get("fg3_pct", 0.0) or 0.0)
        ft_pct = float(pbp.get("ft_pct", 0.0) or 0.0)
        fgm_pg = float(pbp.get("fgm", fg_pct * fga_pg) or 0.0)
        fg3m_pg = float(pbp.get("fg3m", fg3_pct * fg3a_pg) or 0.0)
        efg_pct = (fgm_pg + 0.5 * fg3m_pg) / max(fga_pg, 0.001)
        ts_pct = pts_pg / max(2 * (fga_pg + 0.44 * fta_pg), 0.001)

        per36 = lambda s: _per36(float(s), min_pg)  # noqa: E731
        pts_p36 = per36(pts_pg)
        fga_p36 = per36(fga_pg)
        fg3a_p36 = per36(fg3a_pg)
        fta_p36 = per36(fta_pg)
        ast_p36 = per36(ast_pg)
        reb_p36 = per36(reb_pg)
        oreb_p36 = per36(oreb_pg)
        dreb_p36 = per36(dreb_pg)
        stl_p36 = per36(stl_pg)
        blk_p36 = per36(blk_pg)
        tov_p36 = per36(tov_pg)
        pf_p36 = per36(pf_pg)

        fg3a_rate = fg3a_pg / max(fga_pg, 1.0)
        fta_rate = fta_pg / max(fga_pg, 1.0)
        usage_rate_pct = float(pbp.get("pbp_usage_rate", 0.0) or 0.0)
        if usage_rate_pct <= 0.0:
            possessions_used = fga_pg + 0.44 * fta_pg + tov_pg
            team_possessions_while_on = min_pg * 2.2
            usage_rate_pct = (possessions_used / max(team_possessions_while_on, 1.0)) * 100.0
        usg_pct_proxy = max(0.0, usage_rate_pct / 100.0)
        ast_pct_proxy = ast_p36 / 5.0
        tov_pct_proxy = tov_pg / max(fga_pg + 0.44 * fta_pg + tov_pg, 0.001)
        ast_to_tov = ast_pg / max(tov_pg, 0.1)
        oreb_pct_proxy = oreb_pg / max(reb_pg, 1.0)

        at_rim_freq = float(pbp.get("at_rim_frequency", 0.0) or 0.0)
        short_mid_freq = float(pbp.get("short_mid_frequency", 0.0) or 0.0)
        long_mid_freq = float(pbp.get("long_mid_frequency", 0.0) or 0.0)
        corner3_freq = float(pbp.get("corner3_frequency", 0.0) or 0.0)
        arc3_freq = float(pbp.get("arc3_frequency", 0.0) or 0.0)

        shot_freq_total = max(at_rim_freq + short_mid_freq + long_mid_freq + corner3_freq + arc3_freq, 1e-6)
        zra = _clamp01(at_rim_freq / shot_freq_total)
        zmid_total = _clamp01((short_mid_freq + long_mid_freq) / shot_freq_total)
        zc3 = _clamp01(corner3_freq / shot_freq_total)
        zab3 = _clamp01(arc3_freq / shot_freq_total)
        zpaint = _clamp01(1.0 - (zra + zmid_total + zc3 + zab3))

        zmid_l = zmid_total * 0.30
        zmid_c = zmid_total * 0.40
        zmid_r = zmid_total * 0.30

        corner3_acc = float(pbp.get("corner3_accuracy", fg3_pct) or fg3_pct)
        arc3_acc = float(pbp.get("arc3_accuracy", fg3_pct) or fg3_pct)
        rim_acc = float(pbp.get("at_rim_accuracy", fg_pct) or fg_pct)
        short_mid_acc = float(pbp.get("short_mid_accuracy", fg_pct) or fg_pct)
        long_mid_acc = float(pbp.get("long_mid_accuracy", fg_pct) or fg_pct)
        mid_acc = (short_mid_acc + long_mid_acc) / 2.0

        zone_rates = {
            "ra": zra,
            "paint": zpaint,
            "mid_left": zmid_l,
            "mid_center": zmid_c,
            "mid_right": zmid_r,
            "corner3_left": zc3 * 0.5,
            "corner3_right": zc3 * 0.5,
            "above_break3": zab3,
        }
        zone_fg_pct = {
            "ra": rim_acc,
            "paint": max(0.30, rim_acc * 0.78),
            "mid_left": mid_acc,
            "mid_center": mid_acc,
            "mid_right": mid_acc,
            "corner3_left": corner3_acc,
            "corner3_right": corner3_acc,
            "above_break3": arc3_acc,
        }

        total_poss = float(pbp.get("total_poss", 0.0) or 0.0)
        off_poss = float(pbp.get("off_poss", 0.0) or 0.0)
        poss_pg = (off_poss / max(gp, 1)) if off_poss > 0.0 else ((total_poss / max(gp, 1)) * 0.5)
        unassisted_two_rate = float(pbp.get("unassisted_two_rate", 0.0) or 0.0)

        try:
            shots = self._client.get_player_shots(player_id, season=season)
        except Exception:  # noqa: BLE001
            shots = []

        close_counts = {"left": 0.0, "middle": 0.0, "right": 0.0}
        mid_counts = {
            "left": 0.0,
            "left_center": 0.0,
            "center": 0.0,
            "right_center": 0.0,
            "right": 0.0,
        }
        three_counts = {
            "left": 0.0,
            "left_center": 0.0,
            "center": 0.0,
            "right_center": 0.0,
            "right": 0.0,
        }

        def _to_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _lane_key(x: float) -> str:
            if x <= -170.0:
                return "left"
            if x <= -60.0:
                return "left_center"
            if x < 60.0:
                return "center"
            if x < 170.0:
                return "right_center"
            return "right"

        for shot in shots:
            x = _to_float(shot.get("x"), 0.0)
            shot_value = int(_to_float(shot.get("shot_value"), 2.0))
            shot_distance = _to_float(shot.get("shot_distance"), 0.0)

            if shot_value == 3:
                three_counts[_lane_key(x)] += 1.0
            elif shot_distance <= 10.0:
                if x < -40.0:
                    close_counts["left"] += 1.0
                elif x > 40.0:
                    close_counts["right"] += 1.0
                else:
                    close_counts["middle"] += 1.0
            else:
                mid_counts[_lane_key(x)] += 1.0

        def _normalize(raw: dict[str, float], fallback_keys: list[str]) -> dict[str, float]:
            total = sum(raw.values())
            if total <= 0.0:
                even = 100.0 / max(len(fallback_keys), 1)
                return {k: even for k in fallback_keys}
            return {k: (raw.get(k, 0.0) / total) * 100.0 for k in fallback_keys}

        close_dist = _normalize(close_counts, ["left", "middle", "right"])
        mid_dist = _normalize(mid_counts, ["left", "left_center", "center", "right_center", "right"])
        three_dist = _normalize(three_counts, ["left", "left_center", "center", "right_center", "right"])

        isolation_possessions = poss_pg * _clamp01(unassisted_two_rate * 1.6 + usg_pct_proxy * 0.20)
        pnr_ball_handler_possessions = poss_pg * _clamp01(usg_pct_proxy * 0.18 + ast_pct_proxy * 0.03)
        pnr_rollman_possessions = poss_pg * _clamp01(max(0.0, zra * 0.35 - ast_pct_proxy * 0.02))

        # Post-up proxy: use size/usage/self-creation priors plus offensive context.
        # PBP totals do not provide direct post-up possessions, so this must be inferred.
        size_big_proxy = _clamp01(
            0.55 * _pctile(reb_p36, 5.0, 13.0)
            + 0.25 * _pctile(blk_p36, 0.2, 2.3)
            + 0.20 * _pctile(1.0 - fg3a_rate, 0.35, 0.90)
        )
        post_fallback = max(
            0.0,
            0.2
            + 2.8 * size_big_proxy
            + 0.05 * usage_rate_pct
            - 1.2 * (1.0 - size_big_proxy) * _pctile(fg3a_rate, 0.35, 0.65),
        )
        post_context = poss_pg * _clamp01(
            0.12 * zra
            + 0.18 * zmid_total
            + 0.16 * unassisted_two_rate
            + 0.06 * (1.0 - fg3a_rate)
        )
        post_up_possessions = min(8.0, max(0.0, 0.65 * post_fallback + 0.35 * post_context))

        cuts_possessions = poss_pg * _clamp01(zra * 0.22)
        handoff_possessions = poss_pg * _clamp01(fg3a_rate * 0.18)
        spot_up_possessions = poss_pg * _clamp01(float(pbp.get("catch_and_shoot_three_rate", 0.0) or 0.0) * 1.3)
        off_screen_possessions = poss_pg * _clamp01(fg3a_rate * 0.10 + corner3_freq * 0.08)
        transition_possessions = poss_pg * _clamp01(float(pbp.get("second_chance_off_poss_rate", 0.0) or 0.0) * 1.2 + 0.06)

        isolation_ppp = 2.0 * _clamp01(mid_acc * 0.55 + rim_acc * 0.45)
        pnr_ball_handler_ppp = 2.0 * _clamp01(ts_pct * 0.95)
        pnr_rollman_ppp = 2.0 * _clamp01(rim_acc)
        post_up_ppp = 2.0 * _clamp01(mid_acc * 0.50 + rim_acc * 0.50)
        spot_up_ppp = 2.0 * _clamp01(float(pbp.get("catch_and_shoot_three_rate", 0.0) or 0.0) * 0.6 + fg3_pct * 0.4)

        # Position/size inference from PBP profile when no official metadata exists.
        jumbo_creator_proxy = _clamp01(
            0.50 * _pctile(ast_p36, 5.5, 10.5)
            + 0.25 * _pctile(reb_p36, 6.0, 10.0)
            + 0.25 * _pctile(usage_rate_pct, 24.0, 34.0)
        )

        if (blk_p36 >= 1.35 and reb_p36 >= 10.0 and fg3a_rate < 0.14) or (size_big_proxy >= 0.78 and reb_p36 >= 9.5):
            position = "C"
        elif reb_p36 >= 8.0 and fg3a_rate < 0.28:
            position = "PF"
        elif ast_p36 >= 7.0 and usg_pct_proxy >= 0.22:
            position = "PG"
        elif ast_p36 >= 5.5 and fg3a_rate >= 0.35 and usg_pct_proxy >= 0.20:
            position = "PG" if usg_pct_proxy >= 0.25 else "SG"
        elif fg3a_rate >= 0.36 and ast_p36 < 6.0:
            position = "SG"
        else:
            position = "SF"

        height_inches = round(
            75.0
            + 5.0 * size_big_proxy
            + 2.0 * _pctile(oreb_p36, 0.8, 4.2)
            + 1.5 * _pctile(blk_p36, 0.2, 2.3)
        )
        weight_lbs = round(
            190.0
            + 55.0 * size_big_proxy
            + 25.0 * _pctile(oreb_p36, 0.8, 4.2)
            + 12.0 * _pctile(post_up_possessions, 1.0, 6.0)
        )

        if position == "PG":
            height_inches = max(74, min(80, height_inches - 1))
            weight_lbs = max(185, min(225, weight_lbs - 10))
            if reb_p36 >= 7.0 and usage_rate_pct >= 28.0:
                height_inches = max(height_inches, 78)
                weight_lbs = max(weight_lbs, 220)
        elif position == "SG":
            height_inches = max(75, min(81, height_inches))
            weight_lbs = max(195, min(235, weight_lbs - 2))
        elif position == "SF":
            height_inches = max(77, min(83, height_inches + 1))
            weight_lbs = max(210, min(245, weight_lbs + 2))
            if jumbo_creator_proxy >= 0.60:
                height_inches = max(height_inches, 79)
                weight_lbs = max(weight_lbs, 225)
        elif position == "PF":
            height_inches = max(79, min(84, height_inches + 1))
            weight_lbs = max(225, min(260, weight_lbs + 8))
        else:
            height_inches = max(82, min(85, height_inches + 1))
            weight_lbs = max(245, min(275, weight_lbs + 12))

        if ast_p36 >= 8.0 and reb_p36 >= 9.0:
            height_inches = max(height_inches, 81)
            weight_lbs = max(weight_lbs, 240)

        if blk_p36 >= 1.5 and reb_p36 >= 11.0 and fg3a_rate < 0.10:
            position = "C"
            height_inches = max(height_inches, 83)
            weight_lbs = max(weight_lbs, 250)

        raw_pos = str(info.get("position", "") or "").strip()
        mapped_pos = _map_position(raw_pos) if raw_pos else ""
        if mapped_pos:
            position = mapped_pos
            _guard_labeled = ("guard" in raw_pos.lower()) or (mapped_pos == "SG")
            if _guard_labeled and ast_p36 >= 7.0 and usg_pct_proxy >= 0.22:
                position = "PG"

        info_height = _height_to_inches(str(info.get("height", "") or ""))
        info_weight = _weight_to_lbs(str(info.get("weight", "") or ""))
        if info_height > 0:
            height_inches = max(70, min(88, info_height))
        if info_weight > 0:
            weight_lbs = max(160, min(330, info_weight))

        age = _compute_age(str(info.get("birthdate", "") or ""), season=season)

        pctile_pts = _pctile(pts_pg, 5.0, 30.0)
        pctile_ast = _pctile(ast_pg, 1.0, 10.0)
        pctile_reb = _pctile(reb_pg, 2.0, 14.0)
        pctile_stl = _pctile(stl_pg, 0.3, 2.0)
        pctile_blk = _pctile(blk_pg, 0.1, 2.2)
        pctile_fg3a_rate = _pctile(fg3a_rate, 0.05, 0.55)
        pctile_fta_rate = _pctile(fta_rate, 0.08, 0.45)
        pctile_tov = _pctile(tov_pg, 0.8, 4.5)

        zone_fga_per36 = {k: zone_rates[k] * fga_p36 for k in zone_rates}
        zone_pref_vs_league = {k: zone_rates[k] / (1.0 / len(zone_rates)) for k in zone_rates}

        close_left = float(close_dist.get("left", 33.3))
        close_middle = float(close_dist.get("middle", 33.4))
        close_right = float(close_dist.get("right", 33.3))
        side_total = max(close_left + close_right, 1.0)
        drive_right_bias = 25.0 + (close_right / side_total) * 50.0

        bad_pass_turnovers_pg = float(pbp.get("bad_pass_turnovers", 0.0) or 0.0) / max(gp, 1)
        lost_ball_turnovers_pg = float(pbp.get("lost_ball_turnovers", 0.0) or 0.0) / max(gp, 1)
        offensive_fouls_pg = float(pbp.get("offensive_fouls", 0.0) or 0.0) / max(gp, 1)
        loose_ball_fouls_pg = float(pbp.get("loose_ball_fouls", 0.0) or 0.0) / max(gp, 1)
        shooting_fouls_pg = float(pbp.get("shooting_fouls", 0.0) or 0.0) / max(gp, 1)
        offensive_fouls_drawn_pg = float(pbp.get("offensive_fouls_drawn", 0.0) or 0.0) / max(gp, 1)
        loose_ball_fouls_drawn_pg = float(pbp.get("loose_ball_fouls_drawn", 0.0) or 0.0) / max(gp, 1)

        pos_flags = {f"is_{p.lower()}": (position == p) for p in _POSITIONS}
        features: dict[str, Any] = {
            "position": position,
            "height_inches": height_inches,
            "weight_lbs": weight_lbs,
            "age": age,
            "pts_per_game": pts_pg,
            "fga_per_game": fga_pg,
            "fg3a_per_game": fg3a_pg,
            "fta_per_game": fta_pg,
            "ast_per_game": ast_pg,
            "reb_per_game": reb_pg,
            "stl_per_game": stl_pg,
            "blk_per_game": blk_pg,
            "tov_per_game": tov_pg,
            "min_per_game": min_pg,
            "gp": gp,
            "pts_per36": pts_p36,
            "fga_per36": fga_p36,
            "fg3a_per36": fg3a_p36,
            "fta_per36": fta_p36,
            "ast_per36": ast_p36,
            "reb_per36": reb_p36,
            "oreb_per36": oreb_p36,
            "dreb_per36": dreb_p36,
            "stl_per36": stl_p36,
            "blk_per36": blk_p36,
            "tov_per36": tov_p36,
            "pf_per36": pf_p36,
            "fg_pct": fg_pct,
            "fg3_pct": fg3_pct,
            "ft_pct": ft_pct,
            "efg_pct": efg_pct,
            "ts_pct": ts_pct,
            "usg_pct_proxy": usg_pct_proxy,
            "ast_pct_proxy": ast_pct_proxy,
            "tov_pct_proxy": tov_pct_proxy,
            "ast_to_tov": ast_to_tov,
            "fg3a_rate": fg3a_rate,
            "fta_rate": fta_rate,
            "oreb_pct_proxy": oreb_pct_proxy,
            "usage_rate": usage_rate_pct,
            "isolation_possessions": isolation_possessions,
            "pick_and_roll_ball_handler_possessions": pnr_ball_handler_possessions,
            "pick_and_roll_rollman_possessions": pnr_rollman_possessions,
            "post_up_possessions": post_up_possessions,
            "cuts": cuts_possessions,
            "handoff_possessions": handoff_possessions,
            "spot_up_possessions": spot_up_possessions,
            "off_screen_possessions": off_screen_possessions,
            "transition_possessions": transition_possessions,
            "isolation_ppp": isolation_ppp,
            "pick_and_roll_ball_handler_ppp": pnr_ball_handler_ppp,
            "pick_and_roll_rollman_ppp": pnr_rollman_ppp,
            "post_up_ppp": post_up_ppp,
            "spot_up_ppp": spot_up_ppp,
            "midrange_attempts": (short_mid_freq + long_mid_freq) * fga_pg,
            "assisted_2pt_pct": float(pbp.get("assisted_2pt_pct", 0.0) or 0.0),
            "assisted_3pt_pct": float(pbp.get("assisted_3pt_pct", 0.0) or 0.0),
            "catch_and_shoot_three_rate": float(pbp.get("catch_and_shoot_three_rate", 0.0) or 0.0),
            "pull_up_three_rate": float(pbp.get("pull_up_three_rate", 0.0) or 0.0),
            "unassisted_two_rate": float(pbp.get("unassisted_two_rate", 0.0) or 0.0),
            "putback_rate": float(pbp.get("putback_rate", 0.0) or 0.0),
            "live_ball_turnover_pct": float(pbp.get("live_ball_turnover_pct", 0.0) or 0.0),
            "shooting_fouls_drawn_pct": float(pbp.get("shooting_fouls_drawn_pct", 0.0) or 0.0),
            "three_pt_fouls_drawn_pct": float(pbp.get("three_pt_fouls_drawn_pct", 0.0) or 0.0),
            "seconds_per_poss_off": float(pbp.get("seconds_per_poss_off", 0.0) or 0.0),
            "second_chance_off_poss_rate": float(pbp.get("second_chance_off_poss_rate", 0.0) or 0.0),
            "bad_pass_turnovers_per36": _per36(bad_pass_turnovers_pg, min_pg),
            "lost_ball_turnovers_per36": _per36(lost_ball_turnovers_pg, min_pg),
            "offensive_fouls_per36": _per36(offensive_fouls_pg, min_pg),
            "loose_ball_fouls_per36": _per36(loose_ball_fouls_pg, min_pg),
            "shooting_fouls_per36": _per36(shooting_fouls_pg, min_pg),
            "offensive_fouls_drawn_per36": _per36(offensive_fouls_drawn_pg, min_pg),
            "loose_ball_fouls_drawn_per36": _per36(loose_ball_fouls_drawn_pg, min_pg),
            "blocks_recovered_pct": float(pbp.get("blocks_recovered_pct", 0.0) or 0.0),
            "zone_fga_rate_ra": zone_rates["ra"],
            "zone_fga_rate_paint": zone_rates["paint"],
            "zone_fga_rate_mid_left": zone_rates["mid_left"],
            "zone_fga_rate_mid_center": zone_rates["mid_center"],
            "zone_fga_rate_mid_right": zone_rates["mid_right"],
            "zone_fga_rate_corner3_left": zone_rates["corner3_left"],
            "zone_fga_rate_corner3_right": zone_rates["corner3_right"],
            "zone_fga_rate_above_break3": zone_rates["above_break3"],
            "zone_fg_pct_ra": zone_fg_pct["ra"],
            "zone_fg_pct_paint": zone_fg_pct["paint"],
            "zone_fg_pct_mid_left": zone_fg_pct["mid_left"],
            "zone_fg_pct_mid_center": zone_fg_pct["mid_center"],
            "zone_fg_pct_mid_right": zone_fg_pct["mid_right"],
            "zone_fg_pct_corner3_left": zone_fg_pct["corner3_left"],
            "zone_fg_pct_corner3_right": zone_fg_pct["corner3_right"],
            "zone_fg_pct_above_break3": zone_fg_pct["above_break3"],
            "zone_fga_per36_ra": zone_fga_per36["ra"],
            "zone_fga_per36_paint": zone_fga_per36["paint"],
            "zone_fga_per36_mid_left": zone_fga_per36["mid_left"],
            "zone_fga_per36_mid_center": zone_fga_per36["mid_center"],
            "zone_fga_per36_mid_right": zone_fga_per36["mid_right"],
            "zone_fga_per36_corner3_left": zone_fga_per36["corner3_left"],
            "zone_fga_per36_corner3_right": zone_fga_per36["corner3_right"],
            "zone_fga_per36_above_break3": zone_fga_per36["above_break3"],
            "zone_pref_vs_league_ra": zone_pref_vs_league["ra"],
            "zone_pref_vs_league_paint": zone_pref_vs_league["paint"],
            "zone_pref_vs_league_mid_left": zone_pref_vs_league["mid_left"],
            "zone_pref_vs_league_mid_center": zone_pref_vs_league["mid_center"],
            "zone_pref_vs_league_mid_right": zone_pref_vs_league["mid_right"],
            "zone_pref_vs_league_corner3_left": zone_pref_vs_league["corner3_left"],
            "zone_pref_vs_league_corner3_right": zone_pref_vs_league["corner3_right"],
            "zone_pref_vs_league_above_break3": zone_pref_vs_league["above_break3"],
            "sub_zone_distribution_close": {
                "left": close_left,
                "middle": close_middle,
                "right": close_right,
            },
            "sub_zone_distribution_mid": mid_dist,
            "sub_zone_distribution_three": three_dist,
            "drive_right_bias": drive_right_bias,
            "pctile_pts": pctile_pts,
            "pctile_ast": pctile_ast,
            "pctile_reb": pctile_reb,
            "pctile_stl": pctile_stl,
            "pctile_blk": pctile_blk,
            "pctile_fg3a_rate": pctile_fg3a_rate,
            "pctile_fta_rate": pctile_fta_rate,
            "pctile_tov": pctile_tov,
            "has_shot_chart": (len(shots) > 0),
            "low_minutes": (gp < 5 or min_pg < 5),
            "games_played": gp,
            "pbp_data_available": 1.0,
        }
        features.update(pos_flags)
        return features

    def _get_league_averages(self, season: str) -> list[dict]:
        """Return cached league averages, fetching if necessary."""
        if season not in self._league_averages:
            try:
                rows = self._client.get_league_averages(season=season)
                self._league_averages[season] = rows if isinstance(rows, list) else []
            except Exception:  # noqa: BLE001
                self._league_averages[season] = []
        return self._league_averages.get(season, [])

    # Keep old interface for backward compatibility
    def normalise(self, features: dict[str, float]) -> dict[str, float]:
        """Identity pass-through (values already normalised in build_features)."""
        return features

    # ------------------------------------------------------------------
    # Multi-season blending
    # ------------------------------------------------------------------

    @staticmethod
    def _prior_seasons(s0: str) -> list[str]:
        """Return [s1, s2] given s0 season string (e.g. '2025-26' → ['2024-25', '2023-24'])."""
        try:
            return prior_seasons(s0, count=2)
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _dynamic_weights(s0_gp: int) -> tuple[float, float, float]:
        """Dynamic season weights based on s0 games played."""
        if s0_gp < 20:
            return 0.35, 0.45, 0.20
        if s0_gp < 40:
            return 0.45, 0.35, 0.20
        return 0.55, 0.30, 0.15

    # Keys that are never numerically blended (always taken from s0)
    _NON_BLEND_KEYS: frozenset[str] = frozenset({
        "position", "height_inches", "weight_lbs", "age",
        "has_shot_chart", "low_minutes", "games_played", "gp",
        "sub_zone_distribution_close", "sub_zone_distribution_mid",
        "sub_zone_distribution_three", "drive_right_bias",
        "is_pg", "is_sg", "is_sf", "is_pf", "is_c",
    })

    def build_multiseasonal_features(
        self, player_id: int, s0_season: str = "2025-26"
    ) -> dict[str, Any]:
        """
        Build a weighted multi-season feature vector for *player_id*.

        Fetches s0, s1, s2 independently and blends numeric features
        using dynamic weights driven by s0 games played.  Non-numeric
        identity fields (position, height, age, etc.) are always taken
        from s0.  Seasons absent from cache/API are silently dropped and
        the remaining weights are renormalised.

        Parameters
        ----------
        player_id:  NBA player ID.
        s0_season:  Current / most-recent season string (e.g. "2025-26").

        Returns
        -------
        Blended feature dict compatible with FormulaLayer.generate() and
        AttributeCalculator.calculate().
        """
        prior = self._prior_seasons(s0_season)
        seasons = [s0_season] + prior  # [s0, s1, s2]

        # --- collect per-season feature dicts ---
        season_features: list[tuple[float, dict[str, Any]]] = []
        raw_weights = [1.0, 1.0, 1.0]  # placeholders; resolved after s0 GP

        raw_f0: dict[str, Any] = {}
        for i, season in enumerate(seasons):
            try:
                f = self.build_features(player_id, season=season)
                season_features.append((raw_weights[i], f))
                if i == 0:
                    raw_f0 = f
            except Exception:  # noqa: BLE001
                pass  # season unavailable — skip silently

        if not season_features:
            # Absolute fallback: nothing fetched at all
            return self.build_features(player_id, season=s0_season)

        # Resolve weights now that we know s0 GP
        s0_gp = int(raw_f0.get("gp", raw_f0.get("games_played", 0)))
        w0, w1, w2 = self._dynamic_weights(s0_gp)
        all_weights = [w0, w1, w2]

        # Rebuild with correct weights, only for seasons that succeeded
        weighted: list[tuple[float, dict[str, Any]]] = []
        for i, (_, f) in enumerate(season_features):
            weighted.append((all_weights[i], f))

        total_w = sum(w for w, _ in weighted)

        # --- blend numeric features ---
        all_keys: set[str] = set()
        for _, f in weighted:
            all_keys.update(f.keys())

        blended: dict[str, Any] = {}

        # Non-blend keys: from s0 (raw_f0) first, fallback to first available
        base = raw_f0 if raw_f0 else weighted[0][1]
        for k in self._NON_BLEND_KEYS:
            if k in base:
                blended[k] = base[k]

        for k in all_keys:
            if k in self._NON_BLEND_KEYS:
                continue
            vals: list[float] = []
            ws: list[float] = []
            for w, f in weighted:
                v = f.get(k)
                if v is not None:
                    try:
                        vals.append(float(v))
                        ws.append(w)
                    except (TypeError, ValueError):
                        pass
            if vals:
                norm = sum(ws)
                blended[k] = sum(v * w for v, w in zip(vals, ws)) / norm

        # Carry across any non-blend keys missed above (e.g. position flags)
        for k, v in base.items():
            if k not in blended:
                blended[k] = v

        return blended
