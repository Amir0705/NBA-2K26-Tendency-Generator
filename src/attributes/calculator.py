"""Attribute calculator — converts features + tendencies into 2K-style ratings (25–99)."""
from __future__ import annotations

import math

from typing import Any

# Canonical attribute order matching the user-provided list.
ATTRIBUTE_NAMES: tuple[str, ...] = (
    "driving_layup",
    "standing_dunk",
    "driving_dunk",
    "close_shot",
    "mid_range_shot",
    "three_point_shot",
    "free_throw",
    "post_hook",
    "post_fade",
    "post_control",
    "draw_foul",
    "shot_iq",
    "ball_handle",
    "speed_with_ball",
    "hands",
    "pass_accuracy",
    "pass_iq",
    "pass_vision",
    "offensive_consistency",
    "interior_defense",
    "perimeter_defense",
    "steal",
    "block",
    "offensive_rebound",
    "defensive_rebound",
    "help_defense_iq",
    "pass_perception",
    "defensive_consistency",
    "speed",
    "agility",
    "strength",
    "vertical",
    "stamina",
    "intangibles",
    "hustle",
    "overall_durability",
    "potential",
)

ATTRIBUTE_LABELS: dict[str, str] = {
    "driving_layup": "Driving Layup",
    "standing_dunk": "Standing Dunk",
    "driving_dunk": "Driving Dunk",
    "close_shot": "Close Shot",
    "mid_range_shot": "Mid-Range Shot",
    "three_point_shot": "Three-Point Shot",
    "free_throw": "Free Throw",
    "post_hook": "Post Hook",
    "post_fade": "Post Fade",
    "post_control": "Post Control",
    "draw_foul": "Draw Foul",
    "shot_iq": "Shot IQ",
    "ball_handle": "Ball Handle",
    "speed_with_ball": "Speed with Ball",
    "hands": "Hands",
    "pass_accuracy": "Pass Accuracy",
    "pass_iq": "Pass IQ",
    "pass_vision": "Pass Vision",
    "offensive_consistency": "Offensive Consistency",
    "interior_defense": "Interior Defense",
    "perimeter_defense": "Perimeter Defense",
    "steal": "Steal",
    "block": "Block",
    "offensive_rebound": "Offensive Rebound",
    "defensive_rebound": "Defensive Rebound",
    "help_defense_iq": "Help Defense IQ",
    "pass_perception": "Pass Perception",
    "defensive_consistency": "Defensive Consistency",
    "speed": "Speed",
    "agility": "Agility",
    "strength": "Strength",
    "vertical": "Vertical",
    "stamina": "Stamina",
    "intangibles": "Intangibles",
    "hustle": "Hustle",
    "overall_durability": "Overall Durability",
    "potential": "Potential",
}

ATTRIBUTE_CATEGORIES: dict[str, str] = {
    "driving_layup": "finishing",
    "standing_dunk": "finishing",
    "driving_dunk": "finishing",
    "close_shot": "finishing",
    "mid_range_shot": "shooting",
    "three_point_shot": "shooting",
    "free_throw": "shooting",
    "post_hook": "post_game",
    "post_fade": "post_game",
    "post_control": "post_game",
    "draw_foul": "playmaking",
    "shot_iq": "shooting",
    "ball_handle": "playmaking",
    "speed_with_ball": "playmaking",
    "hands": "playmaking",
    "pass_accuracy": "playmaking",
    "pass_iq": "playmaking",
    "pass_vision": "playmaking",
    "offensive_consistency": "mental",
    "interior_defense": "defense",
    "perimeter_defense": "defense",
    "steal": "defense",
    "block": "defense",
    "offensive_rebound": "rebounding",
    "defensive_rebound": "rebounding",
    "help_defense_iq": "defense",
    "pass_perception": "defense",
    "defensive_consistency": "mental",
    "speed": "physical",
    "agility": "physical",
    "strength": "physical",
    "vertical": "physical",
    "stamina": "physical",
    "intangibles": "meta",
    "hustle": "meta",
    "overall_durability": "meta",
    "potential": "meta",
}

# Per-attribute raw-score calibration offsets derived from systematic comparison
# against 2K reference ratings across 38 NBA players.
# NOTE: attributes with new redesigned formulas are removed from this table —
# their formulas produce correctly-scaled 25–99 output directly.
# Positive = formula underestimates (needs boost); Negative = overestimates.
_RAW_CALIBRATION: dict[str, float] = {
    # mid_range_shot: calibration removed — redesigned formula
    # post_hook: calibration removed — redesigned formula
    # close_shot: calibration removed — redesigned formula
    # post_fade: calibration removed — redesigned formula
    # pass_iq: calibration removed — redesigned formula
    # post_control: calibration removed — redesigned formula
    "hands": 6,
    # ball_handle: calibration removed — redesigned formula
    # vertical: calibration removed — redesigned formula
    # stamina: calibration removed — redesigned formula
    # pass_accuracy: calibration removed — redesigned formula
    # driving_dunk: calibration removed — redesigned formula
    # three_point_shot: calibration removed — redesigned formula
    "hustle": 5,
    "interior_defense": 5,
    # perimeter_defense: calibration removed — redesigned formula
    # help_defense_iq: calibration removed — redesigned formula
    # pass_perception: calibration removed — redesigned formula
    "potential": 5,
    # defensive_rebound: calibration removed — redesigned formula
    # draw_foul: calibration removed — redesigned formula
    # free_throw: calibration removed — redesigned formula
    # driving_layup: calibration removed — formula outputs 25–99 directly
    # strength: calibration removed — redesigned formula
    # offensive_rebound: calibration removed — redesigned formula
    # steal: calibration removed — redesigned formula
    # speed_with_ball: calibration removed — redesigned formula
    # block: calibration removed — redesigned formula
}


class AttributeCalculator:
    """Compute 2K-style player attributes from features and tendencies."""

    def calculate(
        self,
        features: dict[str, Any],
        tendencies: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Return attribute ratings keyed by canonical name (25–99 scale)."""
        f = features
        t = tendencies or {}

        pos = str(f.get("position", "SF")).upper()
        is_guard = pos in {"PG", "SG"}
        is_wing = pos in {"SG", "SF"}
        is_big = pos in {"PF", "C"}
        is_center = pos == "C"

        height = self._f(f, "height_inches", 78)
        weight = self._f(f, "weight_lbs", 220)
        age = self._f(f, "age", 25)
        gp = self._f(f, "gp", 0)
        min_pg = self._f(f, "min_per_game", 0)
        # ── Stat inputs (will grow as we add each attribute formula) ──
        min_pg        = self._f(f, "min_per_game", 0)
        gp            = self._f(f, "gp", 0)
        pts_pg        = self._f(f, "pts_per_game", 0)
        fga_pg        = self._f(f, "fga_per_game", 0)
        fta_pg        = self._f(f, "fta_per_game", 0)
        fg3a_pg       = self._f(f, "fg3a_per_game", 0)
        ast_pg        = self._f(f, "ast_per_game", 0)
        oreb_pg       = self._f(f, "oreb_per36", 0)
        dreb_pg       = self._f(f, "dreb_per36", 0)
        stl_pg        = self._f(f, "stl_per_game", 0)
        blk_pg        = self._f(f, "blk_per_game", 0)
        tov_pg        = self._f(f, "tov_per_game", 0)
        stl_per36     = self._f(f, "stl_per36", 0)
        blk_per36     = self._f(f, "blk_per36", 0)
        pf_per36      = self._f(f, "pf_per36", 0)
        fg_pct        = self._f(f, "fg_pct", 0)
        fg3_pct       = self._f(f, "fg3_pct", 0)
        ft_pct        = self._f(f, "ft_pct", 0)
        efg_pct       = self._f(f, "efg_pct", 0)
        ts_pct        = self._f(f, "ts_pct", 0)
        fg3a_rate     = self._f(f, "fg3a_rate", 0)
        fta_rate      = self._f(f, "fta_rate", 0)
        fg3a_per36    = self._f(f, "fg3a_per36", 0)
        fta_per36     = self._f(f, "fta_per36", 0)
        fga_per36     = self._f(f, "fga_per36", 0)
        tov_per36     = self._f(f, "tov_per36", 0)
        usage         = self._f(f, "usage_rate", 0)
        if usage <= 0.0:
            usage = self._f(f, "usage_rate_pct", 0)
        if usage <= 0.0:
            usage = self._f(f, "usg_pct", 0)
        if usage <= 0.0:
            usage_proxy = self._f(f, "usg_pct_proxy", 0.18)
            usage = usage_proxy * 100.0 if usage_proxy <= 1.0 else usage_proxy
        usage = max(5.0, min(45.0, usage))
        ast_tov       = self._f(f, "ast_to_tov", 1.0)
        tov_pct       = self._f(f, "tov_pct_proxy", 0.12)
        plus_minus    = self._f(f, "plus_minus", 0)
        transition_pos = self._f(f, "transition_possessions", 0)
        iso_pos       = self._f(f, "isolation_possessions", 0)
        pnr_bh        = self._f(f, "pick_and_roll_ball_handler_possessions", 0)
        pnr_roll      = self._f(f, "pick_and_roll_rollman_possessions", 0)
        post_pos      = self._f(f, "post_up_possessions", 0)
        iso_ppp       = self._f(f, "isolation_ppp", 0)
        pnr_bh_ppp    = self._f(f, "pick_and_roll_ball_handler_ppp", 0)
        cuts_pos      = self._f(f, "cuts", 0)
        handoff_pos   = self._f(f, "handoff_possessions", 0)
        spot_up_pos   = self._f(f, "spot_up_possessions", 0)
        post_ppp      = self._f(f, "post_up_ppp", 0)
        spot_ppp      = self._f(f, "spot_up_ppp", 0)
        pctile_pts    = self._f(f, "pctile_pts", 0.5)
        pctile_ast    = self._f(f, "pctile_ast", 0.5)
        pctile_reb    = self._f(f, "pctile_reb", 0.5)
        pctile_stl    = self._f(f, "pctile_stl", 0.5)
        pctile_blk    = self._f(f, "pctile_blk", 0.5)
        assisted_2pt_pct = self._f(f, "assisted_2pt_pct", 0)
        assisted_3pt_pct = self._f(f, "assisted_3pt_pct", 0)
        catch_and_shoot_three_rate = self._f(f, "catch_and_shoot_three_rate", 0)
        pull_up_three_rate = self._f(f, "pull_up_three_rate", 0)
        unassisted_two_rate = self._f(f, "unassisted_two_rate", 0)
        putback_rate = self._f(f, "putback_rate", 0)
        live_ball_turnover_pct = self._f(f, "live_ball_turnover_pct", 0)
        shooting_fouls_drawn_pct = self._f(f, "shooting_fouls_drawn_pct", 0)
        three_pt_fouls_drawn_pct = self._f(f, "three_pt_fouls_drawn_pct", 0)
        seconds_per_poss_off = self._f(f, "seconds_per_poss_off", 0)
        second_chance_off_poss_rate = self._f(f, "second_chance_off_poss_rate", 0)
        bad_pass_turnovers_per36 = self._f(f, "bad_pass_turnovers_per36", 0)
        lost_ball_turnovers_per36 = self._f(f, "lost_ball_turnovers_per36", 0)
        offensive_fouls_per36 = self._f(f, "offensive_fouls_per36", 0)
        offensive_fouls_drawn_per36 = self._f(f, "offensive_fouls_drawn_per36", 0)
        blocks_recovered_pct = self._f(f, "blocks_recovered_pct", 0)
        pbp_data_available = self._f(f, "pbp_data_available", 0)

        # ── Zone inputs ────────────────────────────────────────────────
        ra_rate       = self._f(f, "zone_fga_rate_ra", 0)
        paint_rate    = self._f(f, "zone_fga_rate_paint", 0)
        rim_pressure  = ra_rate + paint_rate
        ra_pct        = self._f(f, "zone_fg_pct_ra", 0)
        paint_pct     = self._f(f, "zone_fg_pct_paint", 0)
        mid_pct       = (
            self._f(f, "zone_fg_pct_mid_left", 0)
            + self._f(f, "zone_fg_pct_mid_center", 0)
            + self._f(f, "zone_fg_pct_mid_right", 0)
        ) / 3.0
        mid_rate      = (
            self._f(f, "zone_fga_rate_mid_left", 0)
            + self._f(f, "zone_fga_rate_mid_center", 0)
            + self._f(f, "zone_fga_rate_mid_right", 0)
        )

        # Helper: size signal (0–1)
        size_big = max(
            1.0 if is_big else 0.0,
            0.60 * self._norm(height, 79, 84)
            + 0.40 * self._norm(weight, 215, 265),
        )
        size_big = min(1.0, size_big)

        # Age curve for physicals (peaks at 25–27)
        age_phys = 1.0 - 0.012 * max(0, age - 27) - 0.008 * max(0, 22 - age)
        age_phys = max(0.75, min(1.0, age_phys))

        # Defensive engagement proxy (used by multiple attributes)
        def_engagement = max(0.0, 1.0 - self._norm(usage, 18, 35))
        off_load = self._norm(usage, 22, 34)
        gp_ratio = gp / 82.0

        raw_iso_p36 = iso_pos * 36.0 / max(min_pg, 1.0)
        raw_pnr_bh_p36 = pnr_bh * 36.0 / max(min_pg, 1.0)
        raw_pnr_roll_p36 = pnr_roll * 36.0 / max(min_pg, 1.0)
        raw_post_p36 = post_pos * 36.0 / max(min_pg, 1.0)
        raw_transition_p36 = transition_pos * 36.0 / max(min_pg, 1.0)
        raw_cuts_p36 = cuts_pos * 36.0 / max(min_pg, 1.0)
        raw_handoff_p36 = handoff_pos * 36.0 / max(min_pg, 1.0)
        raw_spot_up_p36 = spot_up_pos * 36.0 / max(min_pg, 1.0)
        possessions_used_per36 = fga_per36 + 0.44 * fta_per36 + tov_per36
        creator_two_share = max(
            0.0,
            min(
                1.0,
                0.60 * unassisted_two_rate
                + 0.22 * self._norm(seconds_per_poss_off, 11.5, 17.5)
                + 0.18 * self._norm(usage, 16.0, 34.0),
            ),
        )
        creator_two_p36 = possessions_used_per36 * creator_two_share
        iso_burden_p36 = (
            0.45 * raw_iso_p36
            + 0.40 * creator_two_p36
            + 0.15 * (fg3a_per36 * pull_up_three_rate)
        )
        pnr_bh_burden_p36 = (
            0.75 * raw_pnr_bh_p36
            + 0.15 * creator_two_p36
            + 0.10 * raw_handoff_p36
        )
        post_burden_p36 = (
            0.62 * raw_post_p36
            + 0.22 * creator_two_p36 * (0.35 + 0.65 * size_big)
            + 0.16
            * 4.0
            * max(0.0, min(1.0, (1.0 - assisted_2pt_pct) * (0.35 + 0.65 * size_big)))
        )
        pace_control = self._norm(seconds_per_poss_off, 11.5, 17.5)
        ball_security = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    0.46 * self._norm(lost_ball_turnovers_per36, 0.15, 3.20)
                    + 0.34 * self._norm(bad_pass_turnovers_per36, 0.15, 3.00)
                    + 0.20 * self._norm(live_ball_turnover_pct, 0.22, 0.70)
                ),
            ),
        )
        receiver_skill = max(
            0.0,
            min(
                1.0,
                0.40 * assisted_2pt_pct
                + 0.35 * catch_and_shoot_three_rate
                + 0.25 * blocks_recovered_pct,
            ),
        )
        motor_profile = max(
            0.0,
            min(
                1.0,
                0.32 * self._norm(second_chance_off_poss_rate, 0.03, 0.20)
                + 0.24 * self._norm(putback_rate, 0.0, 0.16)
                + 0.22 * self._norm(raw_transition_p36, 0.6, 6.2)
                + 0.22 * self._norm(blocks_recovered_pct, 0.25, 0.75),
            ),
        )

        # Deterministic player-specific noise used only to prevent floor clustering
        # for very low-playmaking archetypes.
        _profile_noise = math.sin(
            0.173 * height
            + 0.097 * weight
            + 0.131 * age
            + 0.211 * pts_pg
            + 0.271 * ast_pg
            + 0.317 * tov_pg
        )
        _profile_noise = max(-1.0, min(1.0, _profile_noise))

        attrs: dict[str, float] = {}

        # ══════════════════════════════════════════════════════════════
        # FINISHING
        # ══════════════════════════════════════════════════════════════

        # ── Driving Layup ─────────────────────────────────────────────
        # Position ceiling: guards are primary layup scorers;
        # centers land in dunk-first territory so their layup ceiling is lower.
        _DL_POS_SCALE = {"PG": 1.00, "SG": 0.97, "SF": 0.90, "PF": 0.83, "C": 0.70}
        _dl_pos_scale = _DL_POS_SCALE.get(pos, 0.90)
        # Per-36 rim-creation volume (ISO + PnR ball-handler)
        _dl_creation_p36 = 0.55 * iso_burden_p36 + 0.45 * pnr_bh_burden_p36
        # RA attempts per 36 min (absolute rim volume, not rate)
        _dl_ra_per36 = self._f(f, "zone_fga_per36_ra", 0)
        # Component scores (each 0–1)
        _dl_c_ra_pct    = self._norm(ra_pct,           0.52, 0.78)  # RA FG% — finishing quality
        _dl_c_rim_press = self._norm(rim_pressure,     0.08, 0.50)  # rim attack frequency
        _dl_c_fta_rate  = self._norm(fta_rate,         0.10, 0.45)  # drawing contact
        _dl_c_ft_pct    = self._norm(ft_pct,           0.60, 0.85)  # touch / hand-eye proxy
        _dl_c_creation  = self._norm(_dl_creation_p36, 0.50,  8.0)  # self-created rim looks
        _dl_c_ra_vol    = self._norm(_dl_ra_per36,      0.50,  6.0)  # absolute rim volume
        _dl_c_contact   = self._norm(shooting_fouls_drawn_pct, 0.03, 0.18)
        _dl_raw = (
            0.34 * _dl_c_ra_pct
            + 0.20 * _dl_c_rim_press
            + 0.14 * _dl_c_fta_rate
            + 0.10 * _dl_c_ft_pct
            + 0.10 * _dl_c_creation
            + 0.07 * _dl_c_ra_vol
            + 0.05 * _dl_c_contact
        )
        # Smooth volume penalty: ramps from 0.60× at zero rim_pressure to 1.0× at 0.15+
        _dl_vol_penalty = 1.0 if rim_pressure >= 0.15 else (0.60 + 0.40 * max(0.0, rim_pressure / 0.15))
        attrs["driving_layup"] = max(25.0, min(99.0,
            25.0 + _dl_raw * _dl_pos_scale * _dl_vol_penalty * 74.0
        ))

        # ── Standing Dunk ─────────────────────────────────────────────
        # Signals: body profile, roll-man volume, offensive rebounding,
        # and rim-volume profile. Strongly position-gated for realism.
        _SD_POS_SCALE = {"PG": 0.20, "SG": 0.30, "SF": 0.55, "PF": 0.82, "C": 1.00}
        _sd_pos_scale = _SD_POS_SCALE.get(pos, 0.55)

        _sd_size = (
            0.55 * self._norm(height, 78, 84)
            + 0.45 * self._norm(weight, 205, 275)
        )
        _sd_pnr_roll_p36 = pnr_roll * 36.0 / max(min_pg, 1.0)
        _sd_c_roll = self._norm(_sd_pnr_roll_p36, 0.20, 7.0)
        _sd_c_oreb = self._norm(oreb_pg, 0.40, 4.0)
        _sd_c_ra_vol = self._norm(self._f(f, "zone_fga_per36_ra", 0), 0.40, 7.0)
        _sd_c_ra_rate = self._norm(ra_rate, 0.08, 0.55)
        _sd_c_putback = self._norm(putback_rate, 0.0, 0.16)
        _sd_c_second = self._norm(second_chance_off_poss_rate, 0.03, 0.20)

        _sd_raw = (
            0.34 * _sd_size
            + 0.20 * _sd_c_roll
            + 0.16 * _sd_c_oreb
            + 0.14 * _sd_c_ra_vol
            + 0.08 * _sd_c_ra_rate
            + 0.04 * _sd_c_putback
            + 0.04 * _sd_c_second
        )

        # Opportunity gate: if a player almost never finishes as a roller
        # and rarely gets to the restricted area, suppress standing dunk.
        _sd_opp = 0.55 * _sd_c_ra_rate + 0.45 * _sd_c_roll
        _sd_gate = 1.0 if _sd_opp >= 0.20 else (0.55 + 0.45 * max(0.0, _sd_opp / 0.20))

        attrs["standing_dunk"] = max(0.0, min(100.0, 100.0 * _sd_raw * _sd_pos_scale * _sd_gate))

        # ── Driving Dunk ──────────────────────────────────────────────
        # Signals: explosive rim pressure, rim volume, transition pressure,
        # burst profile, and self-creation at the rim.
        # Driving dunk models poster ability, so explosive guards are not
        # hard-capped below slower wings/bigs.
        _DD_POS_SCALE = {"PG": 1.00, "SG": 0.95, "SF": 0.82, "PF": 0.72, "C": 0.60}
        _dd_pos_scale = _DD_POS_SCALE.get(pos, 0.90)

        _dd_c_rim = self._norm(rim_pressure, 0.10, 0.55)
        _dd_c_ra_pct = self._norm(ra_pct, 0.56, 0.76)
        _dd_c_ra_vol = self._norm(self._f(f, "zone_fga_per36_ra", 0), 0.60, 7.5)
        _dd_c_transition = self._norm(raw_transition_p36, 0.30, 5.2)
        _dd_c_creation = self._norm(0.75 * iso_burden_p36 + 0.25 * pnr_bh_burden_p36, 0.60, 10.0)
        _dd_c_fta = self._norm(fta_rate, 0.12, 0.50)
        _dd_c_size = 0.60 * self._norm(height, 75, 84) + 0.40 * self._norm(weight, 180, 260)
        _dd_c_pop = max(0.0, min(1.0, 0.45 * _dd_c_size + 0.25 * age_phys + 0.30 * self._norm(putback_rate, 0.0, 0.12)))
        _dd_c_burst = (
            0.55 * (1.0 - self._norm(height, 75, 84))
            + 0.45 * (1.0 - self._norm(weight, 180, 260))
        )
        _dd_c_burst = max(0.0, min(1.0, _dd_c_burst))

        _dd_raw = (
            0.24 * _dd_c_rim
            + 0.18 * _dd_c_ra_vol
            + 0.16 * _dd_c_transition
            + 0.14 * _dd_c_creation
            + 0.10 * _dd_c_burst
            + 0.08 * _dd_c_pop
            + 0.06 * _dd_c_fta
            + 0.04 * _dd_c_ra_pct
        )

        # Opportunity gate: players with little rim/transition action should not
        # grade as elite driving dunkers.
        _dd_opp = 0.45 * _dd_c_rim + 0.30 * _dd_c_ra_vol + 0.25 * _dd_c_transition
        _dd_gate = 1.0 if _dd_opp >= 0.30 else (0.55 + 0.45 * max(0.0, _dd_opp / 0.30))

        attrs["driving_dunk"] = max(0.0, min(100.0, 100.0 * _dd_raw * _dd_pos_scale * _dd_gate))

        # ── Close Shot ────────────────────────────────────────────────
        # Close shot = short-range touch and finishing consistency around
        # the basket (not poster ability).
        _cs_c_paint_pct = self._norm(paint_pct, 0.36, 0.62)
        _cs_c_ra_pct = self._norm(ra_pct, 0.55, 0.77)
        _cs_c_rim_pressure = self._norm(rim_pressure, 0.10, 0.58)
        _cs_c_ra_vol = self._norm(self._f(f, "zone_fga_per36_ra", 0), 0.50, 7.0)
        _cs_c_ft_touch = self._norm(ft_pct, 0.58, 0.88)
        _cs_c_fta_rate = self._norm(fta_rate, 0.10, 0.45)
        _cs_c_slash = self._norm(raw_transition_p36, 0.30, 5.2)
        _cs_c_post = self._norm(post_burden_p36, 0.10, 4.8)
        _cs_c_oreb = self._norm(oreb_pg, 0.30, 4.0)
        _cs_c_size = 0.55 * self._norm(height, 76, 84) + 0.45 * self._norm(weight, 190, 270)
        _cs_c_putback = self._norm(putback_rate, 0.0, 0.16)
        _cs_c_contact = self._norm(shooting_fouls_drawn_pct, 0.03, 0.18)

        _cs_raw = (
            0.16 * _cs_c_paint_pct
            + 0.16 * _cs_c_ra_pct
            + 0.18 * _cs_c_rim_pressure
            + 0.13 * _cs_c_ra_vol
            + 0.06 * _cs_c_ft_touch
            + 0.08 * _cs_c_fta_rate
            + 0.07 * _cs_c_slash
            + 0.05 * _cs_c_post
            + 0.05 * _cs_c_oreb
            + 0.03 * _cs_c_size
            + 0.02 * _cs_c_putback
            + 0.01 * _cs_c_contact
        )

        # Position profile: bigs generally have stronger close-shot packages;
        # guards can still grade high through touch/efficiency.
        _CS_POS_SCALE = {"PG": 0.95, "SG": 0.97, "SF": 1.00, "PF": 1.04, "C": 1.10}
        _cs_pos_scale = _CS_POS_SCALE.get(pos, 1.00)

        # Opportunity gate: if paint + RA volume is very low, close shot
        # should not reach elite values.
        _cs_opp = 0.50 * _cs_c_rim_pressure + 0.30 * _cs_c_ra_vol + 0.20 * _cs_c_paint_pct
        _cs_gate = 1.0 if _cs_opp >= 0.22 else (0.70 + 0.30 * max(0.0, _cs_opp / 0.22))

        attrs["close_shot"] = max(0.0, min(100.0, 100.0 * _cs_raw * _cs_pos_scale * _cs_gate))

        # ══════════════════════════════════════════════════════════════
        # SHOOTING
        # ══════════════════════════════════════════════════════════════

        # ── Three-Point Shot ──────────────────────────────────────────
        # Corrected model: lower floor, wider elite separation, and no heavy
        # position penalty on stretch bigs.
        _tp_acc = self._norm(fg3_pct, 0.27, 0.43)
        _tp_vol = self._norm(fg3a_per36, 0.5, 11.5)
        _tp_cs = self._norm(catch_and_shoot_three_rate, 0.05, 0.70)
        _tp_pull = self._norm(pull_up_three_rate, 0.0, 0.40)
        _tp_assist = self._norm(assisted_3pt_pct, 0.35, 0.98)
        _tp_score = 0.52 * _tp_acc + 0.20 * _tp_vol + 0.12 * _tp_cs + 0.10 * _tp_pull + 0.06 * _tp_assist

        # Low-accuracy dampener: volume should not rescue poor percentages.
        _tp_acc_guard = 1.0 if fg3_pct >= 0.33 else (0.45 + 0.55 * self._norm(fg3_pct, 0.24, 0.33))
        _tp_score *= _tp_acc_guard

        # Legit shooters get a modest lift so solid-volume 36-39% bigs/wings
        # do not land too low.
        if fg3_pct >= 0.34 and fg3a_per36 >= 4.0:
            _tp_score += 0.06 * (0.50 * _tp_acc + 0.25 * _tp_vol + 0.15 * _tp_cs + 0.10 * _tp_pull)

        # Near non-shooters should stay at the low end.
        if fg3a_per36 < 0.6:
            _tp_score *= 0.25

        # Sample reliability from current blended minute load.
        _tp_minutes = gp * min_pg
        _tp_rel = max(0.0, min(1.0, _tp_minutes / 900.0))

        # Blend toward neutral when minute sample is small.
        _tp_score = _tp_rel * _tp_score + (1.0 - _tp_rel) * 0.55

        # Keep as 0–100 raw score; final scaler maps to 25–99.
        attrs["three_point_shot"] = max(0.0, min(100.0, 100.0 * _tp_score))

        # ── Mid-Range Shot ────────────────────────────────────────────
        # Mid-range = pull-up/short-jumper touch and willingness to take
        # non-rim, non-three attempts with decent efficiency.
        _mr_pct = self._norm(mid_pct, 0.35, 0.55)
        _mr_rate = self._norm(mid_rate, 0.04, 0.36)
        _mr_efg = self._norm(efg_pct, 0.45, 0.61)
        _mr_creation = self._norm(0.58 * creator_two_p36 + 0.42 * iso_burden_p36, 0.6, 10.5)
        _mr_volume = self._norm(mid_rate * fga_pg, 0.5, 7.0)
        _mr_usage = self._norm(usage, 16, 33)
        _mr_self = self._norm(unassisted_two_rate, 0.12, 0.55)
        _mr_pace = self._norm(seconds_per_poss_off, 11.5, 17.5)

        _mr_raw = (
            0.35 * _mr_pct
            + 0.18 * _mr_rate
            + 0.12 * _mr_efg
            + 0.13 * _mr_creation
            + 0.10 * _mr_volume
            + 0.06 * _mr_usage
            + 0.04 * _mr_self
            + 0.02 * _mr_pace
        )
        # Gentle global lift so the mid-range band is not too compressed low.
        _mr_raw = 0.16 + 0.84 * _mr_raw

        # Wings/forwards generally carry stronger in-game mid packages,
        # while pure rim centers and pure 3-heavy guards are less likely.
        _MR_POS_SCALE = {"PG": 1.02, "SG": 1.03, "SF": 1.05, "PF": 0.98, "C": 0.75}
        _mr_pos_scale = _MR_POS_SCALE.get(pos, 1.00)

        # Opportunity gate: very low mid-range rate should prevent elite grades.
        _mr_gate = 1.0 if _mr_rate >= 0.15 else (0.80 + 0.20 * max(0.0, _mr_rate / 0.15))

        attrs["mid_range_shot"] = max(0.0, min(100.0, 100.0 * _mr_raw * _mr_pos_scale * _mr_gate * 1.03))

        # ── Free Throw ────────────────────────────────────────────────
        # Free throw should track real FT shooting directly, with only modest
        # smoothing for tiny samples.
        _ft_attempts_total = max(0.0, fta_pg * gp)
        _ft_rel = self._norm(_ft_attempts_total, 20.0, 180.0)

        # Regress tiny samples toward league-average touch.
        _ft_pct_regressed = _ft_rel * ft_pct + (1.0 - _ft_rel) * 0.77

        _ft_acc = self._norm(_ft_pct_regressed, 0.45, 0.93)
        _ft_vol = self._norm(fta_pg, 0.4, 8.5)

        _ft_raw = 0.94 * _ft_acc + 0.06 * _ft_vol
        attrs["free_throw"] = max(0.0, min(100.0, 100.0 * _ft_raw))

        # ── Shot IQ ───────────────────────────────────────────────────
        # Shot IQ = efficient scoring + real shooting skill + sane offensive
        # decision quality. Pure rim finishers should not grade like elite
        # shot creators/shooters just because they have high TS%.
        _siq_ts = self._norm(ts_pct, 0.50, 0.68)
        _siq_efg = self._norm(efg_pct, 0.46, 0.63)
        _siq_fg3 = self._norm(fg3_pct, 0.27, 0.43)
        _siq_mid = self._norm(mid_pct, 0.34, 0.52)
        _siq_ft = self._norm(ft_pct, 0.60, 0.90)
        _siq_shoot_skill = 0.40 * _siq_fg3 + 0.30 * _siq_mid + 0.30 * _siq_ft
        _siq_usage = self._norm(usage, 14.0, 34.0)
        _siq_creation = self._norm(0.50 * iso_burden_p36 + 0.50 * pnr_bh_burden_p36, 0.6, 10.0)
        _siq_play = 1.0 - self._norm(tov_pct, 0.08, 0.20)
        _siq_pts = self._norm(pts_pg, 8.0, 32.0)
        _siq_security = ball_security
        _siq_profile = (
            0.38 * self._norm(catch_and_shoot_three_rate, 0.05, 0.70)
            + 0.26 * self._norm(pull_up_three_rate, 0.0, 0.40)
            + 0.36 * self._norm(unassisted_two_rate, 0.12, 0.55)
        )
        _siq_playmaker = (
            0.65 * self._norm(ast_pg, 2.0, 10.0)
            + 0.35 * self._norm(ast_tov, 1.0, 3.5)
        )
        _siq_skill_load = _siq_usage * (0.55 * _siq_ts + 0.45 * _siq_shoot_skill)
        _siq_good_offense = 0.55 * _siq_shoot_skill + 0.45 * _siq_ts
        _siq_star_creation = 0.40 * _siq_usage + 0.35 * _siq_creation + 0.25 * _siq_pts
        _siq_offensive_hub = _siq_playmaker * (0.45 * _siq_usage + 0.30 * _siq_creation + 0.25 * _siq_ts)

        _siq_raw = (
            0.22 * _siq_ts
            + 0.14 * _siq_efg
            + 0.20 * _siq_shoot_skill
            + 0.10 * _siq_play
            + 0.12 * _siq_creation
            + 0.10 * _siq_skill_load
            + 0.06 * _siq_pts
            + 0.06 * _siq_star_creation
            + 0.05 * _siq_playmaker
            + 0.03 * _siq_security
        )
        _siq_raw = 0.12 + 0.88 * _siq_raw
        _siq_raw += 0.06 * _siq_good_offense
        _siq_raw += 0.05 * _siq_offensive_hub
        _siq_raw += 0.03 * _siq_profile

        # Non-shooter dampener: if a player brings very little perimeter/mid
        # skill, cap their ceiling even if they finish efficiently inside.
        _siq_non_shooter = 0.60 * _siq_fg3 + 0.40 * _siq_mid
        if _siq_non_shooter < 0.25:
            _siq_raw *= 0.82 + 0.18 * (_siq_non_shooter / 0.25)

        attrs["shot_iq"] = max(0.0, min(100.0, 100.0 * _siq_raw))

        # ══════════════════════════════════════════════════════════════
        # POST GAME
        # ══════════════════════════════════════════════════════════════

        # ── Post Hook ─────────────────────────────────────────────────
        # Post hook = actual post usage + size leverage + short-touch skill.
        _ph_post_p36 = post_burden_p36
        _ph_post = self._norm(_ph_post_p36, 0.10, 4.8)
        _ph_ppp = self._norm(post_ppp, 0.72, 1.12)
        _ph_size = 0.55 * self._norm(height, 78, 84) + 0.45 * self._norm(weight, 210, 275)
        _ph_paint = self._norm(paint_pct, 0.38, 0.62)
        _ph_mid_touch = self._norm(mid_pct, 0.34, 0.50)
        _ph_ft_touch = self._norm(ft_pct, 0.58, 0.88)
        _ph_touch = 0.45 * _ph_paint + 0.30 * _ph_mid_touch + 0.25 * _ph_ft_touch
        _ph_close = self._norm(attrs["close_shot"], 50.0, 90.0)
        _ph_self = self._norm((1.0 - assisted_2pt_pct) * (0.35 + 0.65 * size_big), 0.08, 0.72)

        _ph_raw = (
            0.26 * _ph_post
            + 0.20 * _ph_ppp
            + 0.20 * _ph_size
            + 0.16 * _ph_touch
            + 0.10 * _ph_close
            + 0.04 * size_big
            + 0.04 * _ph_self
        )

        _PH_POS_SCALE = {"PG": 0.15, "SG": 0.22, "SF": 0.45, "PF": 0.82, "C": 1.00}
        _ph_pos_scale = _PH_POS_SCALE.get(pos, 0.45)

        # Jumbo creator guards/wings (Luka/LeBron archetype) should retain
        # functional post-hook value when they carry real post volume.
        _ph_jumbo_creator = (
            (not is_big)
            and height >= 79
            and weight >= 220
            and usage >= 26
            and _ph_post > 0.18
        )
        if _ph_jumbo_creator:
            _ph_pos_scale = max(_ph_pos_scale, 0.40)
            _ph_raw = max(_ph_raw, 0.40 + 0.20 * _ph_post + 0.10 * _ph_touch)

        # Must have some post role or big-man profile to rate well.
        _ph_opp = 0.65 * _ph_post + 0.35 * _ph_size
        _ph_gate = 1.0 if _ph_opp >= 0.22 else (0.45 + 0.55 * max(0.0, _ph_opp / 0.22))

        attrs["post_hook"] = max(0.0, min(100.0, 100.0 * _ph_raw * _ph_pos_scale * _ph_gate))

        # ── Post Fade ─────────────────────────────────────────────────
        # Post fade = real post usage plus genuine mid-range touch and size.
        _pf_post_p36 = post_burden_p36
        _pf_post = self._norm(_pf_post_p36, 0.10, 4.8)
        _pf_ppp = self._norm(post_ppp, 0.72, 1.12)
        _pf_mid = self._norm(mid_pct, 0.35, 0.52)
        _pf_mid_attr = self._norm(attrs["mid_range_shot"], 50.0, 90.0)
        _pf_ft = self._norm(ft_pct, 0.60, 0.90)
        _pf_size = 0.50 * self._norm(height, 77, 84) + 0.50 * self._norm(weight, 200, 270)
        _pf_creation = self._norm(0.60 * iso_burden_p36 + 0.40 * creator_two_p36, 0.5, 8.5)
        _pf_self = self._norm((1.0 - assisted_2pt_pct) * (0.25 + 0.75 * size_big), 0.08, 0.72)

        _pf_raw = (
            0.24 * _pf_post
            + 0.18 * _pf_ppp
            + 0.18 * _pf_mid
            + 0.14 * _pf_mid_attr
            + 0.10 * _pf_ft
            + 0.08 * _pf_size
            + 0.06 * _pf_creation
            + 0.02 * _pf_self
        )

        # Skilled wings and bigs should lead this attribute.
        _PF_POS_SCALE = {"PG": 0.18, "SG": 0.32, "SF": 0.72, "PF": 1.00, "C": 0.92}
        _pf_pos_scale = _PF_POS_SCALE.get(pos, 0.72)

        _pf_jumbo_creator = (
            (not is_big)
            and height >= 79
            and weight >= 220
            and usage >= 26
            and _pf_post > 0.18
        )
        if _pf_jumbo_creator:
            _pf_pos_scale = max(_pf_pos_scale, 0.52)
            _pf_raw = max(_pf_raw, 0.42 + 0.22 * _pf_post + 0.12 * _pf_mid)

        # Must show some combination of post usage and mid skill.
        _pf_opp = 0.55 * _pf_post + 0.45 * _pf_mid
        _pf_gate = 1.0 if _pf_opp >= 0.22 else (0.45 + 0.55 * max(0.0, _pf_opp / 0.22))

        attrs["post_fade"] = max(0.0, min(100.0, 100.0 * _pf_raw * _pf_pos_scale * _pf_gate))

        # ── Post Control ──────────────────────────────────────────────
        # Post control = ability to establish and maintain post position and
        # operate effectively from the post. More leverage/usage than touch.
        _pc_post_p36 = post_burden_p36
        _pc_post = self._norm(_pc_post_p36, 0.15, 5.0)
        _pc_ppp = self._norm(post_ppp, 0.72, 1.12)
        _pc_size = 0.45 * self._norm(height, 78, 84) + 0.55 * self._norm(weight, 210, 285)
        _pc_strength = 0.55 * self._norm(weight, 210, 285) + 0.45 * size_big
        _pc_paint = self._norm(paint_pct, 0.38, 0.62)
        _pc_close = self._norm(attrs["close_shot"], 50.0, 92.0)
        _pc_hook = self._norm(attrs["post_hook"], 40.0, 95.0)
        _pc_fade = self._norm(attrs["post_fade"], 40.0, 90.0)
        _pc_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)
        _pc_draw = self._norm(0.65 * shooting_fouls_drawn_pct + 0.35 * fta_rate, 0.03, 0.26)

        _pc_raw = (
            0.24 * _pc_post
            + 0.16 * _pc_ppp
            + 0.18 * _pc_size
            + 0.14 * _pc_strength
            + 0.08 * _pc_paint
            + 0.08 * _pc_close
            + 0.06 * _pc_hook
            + 0.04 * _pc_fade
            + 0.04 * _pc_pace
            + 0.02 * _pc_draw
        )

        _PC_POS_SCALE = {"PG": 0.18, "SG": 0.28, "SF": 0.60, "PF": 0.92, "C": 1.00}
        _pc_pos_scale = _PC_POS_SCALE.get(pos, 0.60)

        _pc_jumbo_creator = (
            (not is_big)
            and height >= 79
            and weight >= 220
            and usage >= 26
            and _pc_post > 0.18
        )
        if _pc_jumbo_creator:
            _pc_pos_scale = max(_pc_pos_scale, 0.62)
            _pc_raw = max(_pc_raw, 0.46 + 0.24 * _pc_post + 0.10 * _pc_strength)

        # Must show real post role or true big-man leverage.
        _pc_opp = 0.60 * _pc_post + 0.40 * _pc_strength
        _pc_gate = 1.0 if _pc_opp >= 0.24 else (0.45 + 0.55 * max(0.0, _pc_opp / 0.24))

        attrs["post_control"] = max(0.0, min(100.0, 100.0 * _pc_raw * _pc_pos_scale * _pc_gate))

        # ══════════════════════════════════════════════════════════════
        # PLAYMAKING
        # ══════════════════════════════════════════════════════════════

        # ── Draw Foul ─────────────────────────────────────────────────
        # Draw foul = how reliably a player creates contact and gets to the line.
        _df_fta_rate = self._norm(fta_rate, 0.10, 0.60)
        _df_fta_p36 = self._norm(fta_per36, 1.5, 11.0)
        _df_rim = self._norm(rim_pressure, 0.10, 0.60)
        _df_ra = self._norm(ra_pct, 0.52, 0.76)
        _df_usage = self._norm(usage, 14.0, 35.0)
        _df_creation = self._norm(0.55 * iso_burden_p36 + 0.45 * pnr_bh_burden_p36, 0.6, 10.0)
        _df_pts = self._norm(pts_pg, 8.0, 30.0)
        _df_post = self._norm(post_burden_p36, 0.10, 5.0)
        _df_contact_finisher = self._norm(0.5 * attrs["driving_layup"] + 0.5 * attrs["close_shot"], 55.0, 90.0)
        _df_and1 = _df_fta_rate * (0.55 * _df_rim + 0.45 * _df_ra)
        _df_self = 0.45 * _df_creation + 0.35 * _df_usage + 0.20 * _df_pts
        _df_shooting_foul = self._norm(shooting_fouls_drawn_pct, 0.03, 0.19)
        _df_off_foul_drawn = self._norm(offensive_fouls_drawn_per36, 0.0, 0.35)

        _df_raw = (
            0.20 * _df_fta_rate
            + 0.15 * _df_fta_p36
            + 0.14 * _df_rim
            + 0.17 * _df_self
            + 0.08 * _df_post
            + 0.07 * _df_contact_finisher
            + 0.05 * _df_and1
            + 0.06 * _df_ra
            + 0.06 * _df_shooting_foul
            + 0.02 * _df_off_foul_drawn
        )

        # Likely intentional-foul / low self-creation profile.
        if ft_pct < 0.68 and _df_self < 0.35:
            _df_hack_profile = 0.50 * self._norm(ft_pct, 0.45, 0.68) + 0.50 * (_df_self / 0.35)
            _df_raw *= 0.60 + 0.40 * max(0.0, min(1.0, _df_hack_profile))

        # Low-opportunity players should not grade highly just from tiny FT samples.
        _df_opp = 0.45 * _df_fta_rate + 0.30 * _df_rim + 0.15 * _df_creation + 0.10 * _df_post
        _df_gate = 1.0 if _df_opp >= 0.22 else (0.60 + 0.40 * max(0.0, _df_opp / 0.22))

        attrs["draw_foul"] = max(0.0, min(100.0, 100.0 * _df_raw * _df_gate))

        # ── Ball Handle ───────────────────────────────────────────────
        # Ball handle = on-ball dribble burden, ability to create with the ball,
        # and enough passing/security to trust that handle under pressure.
        _bh_iso_p36 = iso_burden_p36
        _bh_pnr_p36 = pnr_bh_burden_p36
        _bh_trans_p36 = raw_transition_p36
        _bh_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)

        _bh_iso = self._norm(_bh_iso_p36, 0.05, 4.50)
        _bh_pnr = self._norm(_bh_pnr_p36, 0.15, 7.00)
        _bh_usage = self._norm(usage, 13.0, 34.0)
        _bh_pts = self._norm(pts_pg, 8.0, 29.0)
        _bh_ast = self._norm(_bh_ast_p36, 1.8, 9.0)
        _bh_ast_pct = self._norm(pctile_ast, 0.30, 0.95)
        _bh_ast_tov = self._norm(ast_tov, 0.90, 3.00)
        _bh_tov = 1.0 - self._norm(tov_pct, 0.09, 0.18)
        _bh_lost = 1.0 - self._norm(lost_ball_turnovers_per36, 0.20, 3.20)
        _bh_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)
        _bh_iso_eff = self._norm(iso_ppp, 0.70, 1.05)
        _bh_pnr_eff = self._norm(pnr_bh_ppp, 0.74, 1.04)
        _bh_rim = self._norm(rim_pressure, 0.10, 0.60)
        _bh_layup = self._norm(attrs["driving_layup"], 65.0, 96.0)
        _bh_draw = self._norm(attrs["draw_foul"], 45.0, 95.0)
        _bh_close = self._norm(attrs["close_shot"], 60.0, 92.0)
        _bh_trans = self._norm(_bh_trans_p36, 0.30, 6.50)
        _bh_three = self._norm(attrs["three_point_shot"], 70.0, 95.0)
        _bh_mid = self._norm(attrs["mid_range_shot"], 65.0, 90.0)
        _bh_iq = self._norm(attrs["shot_iq"], 60.0, 95.0)
        _bh_pull = self._norm(pull_up_three_rate, 0.0, 0.45)

        _bh_creation = 0.34 * _bh_pnr + 0.28 * _bh_iso + 0.20 * _bh_usage + 0.18 * self._norm(creator_two_p36, 0.4, 10.5)
        _bh_passing = 0.44 * _bh_ast + 0.28 * _bh_ast_pct + 0.28 * _bh_ast_tov
        _bh_security = 0.32 * _bh_tov + 0.23 * _bh_ast_tov + 0.25 * _bh_lost + 0.20 * _bh_live
        _bh_eff = 0.56 * _bh_pnr_eff + 0.44 * _bh_iso_eff
        _bh_attack = (
            0.30 * _bh_rim
            + 0.25 * _bh_layup
            + 0.20 * _bh_draw
            + 0.15 * _bh_close
            + 0.10 * _bh_trans
        )
        _bh_craft = 0.38 * _bh_three + 0.16 * _bh_mid + 0.30 * _bh_iq + 0.16 * _bh_pull
        _bh_star = 0.55 * _bh_usage + 0.45 * _bh_pts

        _bh_raw = (
            0.28 * _bh_creation
            + 0.19 * _bh_passing
            + 0.08 * _bh_security
            + 0.05 * _bh_eff
            + 0.10 * _bh_attack
            + 0.16 * _bh_craft
            + 0.14 * _bh_star
        )

        # Players need real on-ball burden to grade highly here.
        _bh_opp = 0.42 * _bh_creation + 0.24 * _bh_passing + 0.18 * _bh_attack + 0.16 * _bh_star
        _bh_gate = 1.0 if _bh_opp >= 0.22 else (0.50 + 0.50 * max(0.0, _bh_opp / 0.22))

        attrs["ball_handle"] = max(0.0, min(100.0, 100.0 * _bh_raw * _bh_gate))

        # ── Speed With Ball ───────────────────────────────────────────
        # Speed with ball = live-dribble burst. Reward real downhill pace and
        # open-floor pressure, but cap slower bigs and methodical handlers.
        _swb_transition_p36 = raw_transition_p36
        _swb_iso_p36 = iso_burden_p36
        _swb_pnr_p36 = pnr_bh_burden_p36

        _swb_transition = self._norm(_swb_transition_p36, 0.30, 5.60)
        _swb_rim = self._norm(rim_pressure, 0.10, 0.60)
        _swb_ra = self._norm(ra_pct, 0.52, 0.76)
        _swb_handle = self._norm(attrs["ball_handle"], 55.0, 92.0)
        _swb_layup = self._norm(attrs["driving_layup"], 65.0, 96.0)
        _swb_dunk = self._norm(attrs["driving_dunk"], 35.0, 96.0)
        _swb_draw = self._norm(attrs["draw_foul"], 45.0, 95.0)
        _swb_usage = self._norm(usage, 14.0, 35.0)
        _swb_creation = 0.58 * self._norm(_swb_iso_p36, 0.10, 5.20) + 0.42 * self._norm(_swb_pnr_p36, 0.25, 8.0)
        _swb_size = 0.60 * (1.0 - self._norm(height, 74, 84)) + 0.40 * (1.0 - self._norm(weight, 185, 270))
        _swb_age = age_phys
        _swb_pace = 1.0 - self._norm(seconds_per_poss_off, 14.0, 20.0)
        _swb_security = ball_security

        _swb_burst = 0.36 * _swb_transition + 0.20 * _swb_rim + 0.10 * _swb_ra + 0.34 * _swb_creation
        _swb_finish_force = 0.50 * _swb_dunk + 0.25 * _swb_layup + 0.25 * _swb_draw
        _swb_physical = 0.72 * _swb_size + 0.28 * _swb_age

        _swb_raw = (
            0.22 * _swb_burst
            + 0.22 * _swb_handle
            + 0.18 * _swb_finish_force
            + 0.08 * _swb_physical
            + 0.12 * _swb_usage
            + 0.16 * _swb_creation
            + 0.07 * _swb_pace
            + 0.05 * _swb_security
        )

        # True bigs should need either real handle or real open-floor speed to avoid
        # being overrated from pure rim pressure.
        if size_big > 0.75:
            _swb_big_drag = 0.55 * _swb_handle + 0.45 * _swb_transition
            _swb_raw *= 0.58 + 0.42 * _swb_big_drag

        # Need actual open-floor or downhill opportunity to grade highly.
        _swb_opp = 0.35 * _swb_transition + 0.30 * _swb_rim + 0.20 * _swb_handle + 0.15 * _swb_creation
        _swb_gate = 1.0 if _swb_opp >= 0.20 else (0.45 + 0.55 * max(0.0, _swb_opp / 0.20))

        attrs["speed_with_ball"] = max(0.0, min(100.0, 100.0 * _swb_raw * _swb_gate))

        # ── Hands ─────────────────────────────────────────────────────
        # Hands = how reliably a player catches/controls the ball, especially
        # on quick actions (cuts, rolls, handoffs, spot-ups, lob catches).
        _h_roll_p36 = raw_pnr_roll_p36
        _h_cuts_p36 = raw_cuts_p36
        _h_spot_p36 = raw_spot_up_p36
        _h_handoff_p36 = raw_handoff_p36

        _h_roll = self._norm(_h_roll_p36, 0.05, 8.0)
        _h_cuts = self._norm(_h_cuts_p36, 0.05, 4.2)
        _h_spot = self._norm(_h_spot_p36, 0.30, 7.0)
        _h_handoff = self._norm(_h_handoff_p36, 0.05, 3.0)
        _h_tov = 1.0 - self._norm(tov_pct, 0.10, 0.19)
        _h_ast_tov = self._norm(ast_tov, 0.80, 3.00)
        _h_ts = self._norm(ts_pct, 0.52, 0.66)
        _h_efg = self._norm(efg_pct, 0.48, 0.62)
        _h_handle = self._norm(attrs["ball_handle"], 50.0, 92.0)
        _h_close = self._norm(attrs["close_shot"], 60.0, 95.0)
        _h_layup = self._norm(attrs["driving_layup"], 62.0, 95.0)
        _h_sdunk = self._norm(attrs["standing_dunk"], 25.0, 95.0)
        _h_ddunk = self._norm(attrs["driving_dunk"], 30.0, 95.0)
        _h_rim = self._norm(rim_pressure, 0.10, 0.65)
        _h_lost = 1.0 - self._norm(lost_ball_turnovers_per36, 0.15, 3.20)
        _h_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _h_receiver = receiver_skill

        _h_catch_role = 0.34 * _h_roll + 0.24 * _h_cuts + 0.24 * _h_spot + 0.18 * _h_handoff
        _h_security = 0.60 * _h_tov + 0.15 * _h_ast_tov + 0.25 * _h_lost
        _h_catch_finish = 0.30 * _h_close + 0.20 * _h_layup + 0.25 * _h_sdunk + 0.25 * _h_ddunk
        _h_off_eff = 0.58 * _h_ts + 0.42 * _h_efg

        _h_raw = (
            0.28 * _h_security
            + 0.30 * _h_catch_finish
            + 0.08 * _h_catch_role
            + 0.20 * _h_handle
            + 0.05 * _h_off_eff
            + 0.03 * _h_rim
            + 0.03 * _h_recover
            + 0.03 * _h_receiver
        )

        if _h_roll > 0.45 and _h_catch_finish > 0.55:
            _h_raw *= 1.04

        if size_big > 0.65:
            _h_big_catch = 0.55 * _h_roll + 0.45 * _h_sdunk
            _h_raw *= 0.85 + 0.50 * _h_big_catch

        # Hands sits high in 2K for most rotation players; normalize to that band.
        _h_raw *= 1.18

        # Need either catch role or on-ball trust to reach elite hands levels.
        _h_opp = 0.45 * _h_catch_role + 0.30 * _h_security + 0.25 * _h_handle
        _h_gate = 1.0 if _h_opp >= 0.18 else (0.65 + 0.35 * max(0.0, _h_opp / 0.18))

        attrs["hands"] = max(0.0, min(100.0, 100.0 * _h_raw * _h_gate))

        # ── Pass Accuracy ─────────────────────────────────────────────
        # Pass accuracy = ability to consistently deliver on-target passes,
        # anchored by assist creation and ball security.
        _pa_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _pa_pnr_p36 = pnr_bh_burden_p36
        _pa_handoff_p36 = raw_handoff_p36
        _pa_post_p36 = post_burden_p36

        _pa_ast = self._norm(_pa_ast_p36, 2.0, 13.0)
        _pa_ast_pct = self._norm(pctile_ast, 0.30, 0.99)
        _pa_ast_tov = self._norm(ast_tov, 0.90, 3.60)
        _pa_tov = 1.0 - self._norm(tov_pct, 0.09, 0.20)
        _pa_bad_pass = 1.0 - self._norm(bad_pass_turnovers_per36, 0.10, 3.00)
        _pa_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)
        _pa_pnr = self._norm(_pa_pnr_p36, 0.20, 10.0)
        _pa_handoff = self._norm(_pa_handoff_p36, 0.05, 3.6)
        _pa_post_hub = self._norm(_pa_post_p36, 0.05, 4.8)
        _pa_usage = self._norm(usage, 14.0, 35.0)
        _pa_handle = self._norm(attrs["ball_handle"], 55.0, 92.0)
        _pa_hands = self._norm(attrs["hands"], 65.0, 94.0)
        _pa_spot = self._norm(spot_ppp, 0.86, 1.14)

        _pa_creation_hub = 0.60 * _pa_pnr + 0.25 * _pa_handoff + 0.15 * _pa_post_hub
        _pa_security = 0.38 * _pa_ast_tov + 0.22 * _pa_tov + 0.24 * _pa_bad_pass + 0.16 * _pa_live

        _pa_raw = (
            0.34 * _pa_ast
            + 0.20 * _pa_security
            + 0.14 * _pa_ast_pct
            + 0.11 * _pa_creation_hub
            + 0.07 * _pa_usage
            + 0.06 * _pa_handle
            + 0.05 * _pa_hands
            + 0.03 * _pa_spot
        )

        # Lift true high-volume creators into elite pass-accuracy territory.
        if _pa_ast > 0.62 and _pa_security > 0.50:
            _pa_raw *= 1.03 + 0.07 * _pa_ast

        # Global normalization: pass accuracy is generally high across NBA rotation players.
        _pa_raw *= 1.10

        # Must show some passing burden to reach high pass-accuracy ratings.
        _pa_opp = 0.50 * _pa_ast + 0.25 * _pa_creation_hub + 0.15 * _pa_usage + 0.10 * _pa_post_hub
        _pa_gate = 1.0 if _pa_opp >= 0.18 else (0.62 + 0.38 * max(0.0, _pa_opp / 0.18))

        _pa_value = 100.0 * _pa_raw * _pa_gate

        # Baseline passing competence floor for NBA rotation players.
        _pa_floor = (
            0.34
            + 0.08 * size_big
            + 0.08 * _pa_hands
            + 0.04 * self._norm(ast_tov, 0.80, 2.20)
        )

        # Avoid clustering low-playmaking bigs at one identical floor value.
        if size_big > 0.65 and _pa_opp < 0.20:
            _pa_floor += 0.030 * _profile_noise
            _pa_floor = max(0.35, min(0.52, _pa_floor))

        _pa_value = max(_pa_value, 100.0 * min(0.56, _pa_floor))

        attrs["pass_accuracy"] = max(0.0, min(100.0, _pa_value))

        # ── Pass IQ ───────────────────────────────────────────────────
        # Pass IQ = decision quality: making correct reads, managing risk,
        # and creating value with passes in high-pressure possessions.
        _piq_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _piq_pnr_p36 = pnr_bh_burden_p36
        _piq_iso_p36 = iso_burden_p36
        _piq_handoff_p36 = raw_handoff_p36
        _piq_post_p36 = post_burden_p36

        _piq_ast = self._norm(_piq_ast_p36, 2.0, 12.5)
        _piq_ast_pct = self._norm(pctile_ast, 0.30, 0.99)
        _piq_usage = self._norm(usage, 14.0, 35.0)
        _piq_post_hub = self._norm(_piq_post_p36, 0.05, 4.8)
        _piq_handoff = self._norm(_piq_handoff_p36, 0.05, 3.6)
        _piq_creation = (
            0.42 * self._norm(_piq_pnr_p36, 0.20, 10.0)
            + 0.28 * self._norm(_piq_iso_p36, 0.10, 6.0)
            + 0.18 * _piq_handoff
            + 0.12 * _piq_post_hub
        )
        _piq_connector = (
            0.55 * self._norm(_piq_ast_p36, 4.5, 10.5) * (1.0 - self._norm(usage, 26.0, 35.0))
            + 0.45 * self._norm(_piq_handoff_p36 + _piq_post_p36, 0.2, 5.5)
        )

        _piq_ast_tov = self._norm(ast_tov, 0.90, 3.40)
        _piq_tov = 1.0 - self._norm(tov_pct, 0.09, 0.20)
        _piq_bad_pass = 1.0 - self._norm(bad_pass_turnovers_per36, 0.10, 3.00)
        _piq_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)

        # Role-adjusted turnover management: creators are expected to carry more risk.
        _piq_expected_tov = 0.10 + 0.07 * _piq_usage
        _piq_tov_delta = max(-0.08, min(0.08, tov_pct - _piq_expected_tov))
        _piq_risk_adj = 1.0 - self._norm(_piq_tov_delta, -0.01, 0.06)

        _piq_pnr_eff = self._norm(pnr_bh_ppp, 0.76, 1.10)
        _piq_iso_eff = self._norm(iso_ppp, 0.74, 1.08)
        _piq_post_eff = self._norm(post_ppp, 0.78, 1.12)
        _piq_spot_eff = self._norm(spot_ppp, 0.86, 1.14)

        _piq_decision_eff = (
            0.38 * _piq_pnr_eff
            + 0.26 * _piq_iso_eff
            + 0.20 * _piq_post_eff
            + 0.16 * _piq_spot_eff
        )

        _piq_pass_acc = self._norm(attrs["pass_accuracy"], 60.0, 95.0)
        _piq_handle = self._norm(attrs["ball_handle"], 55.0, 92.0)
        _piq_shot_iq = self._norm(attrs["shot_iq"], 60.0, 95.0)
        _piq_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)

        _piq_raw = (
            0.20 * _piq_ast
            + 0.10 * _piq_ast_pct
            + 0.14 * _piq_creation
            + 0.08 * _piq_post_hub
            + 0.12 * _piq_decision_eff
            + 0.11 * _piq_ast_tov
            + 0.01 * _piq_tov
            + 0.10 * _piq_risk_adj
            + 0.03 * _piq_usage
            + 0.12 * _piq_pass_acc
            + 0.03 * _piq_handle
            + 0.08 * _piq_shot_iq
            + 0.06 * _piq_connector
            + 0.01 * _piq_bad_pass
            + 0.01 * _piq_live
            + 0.01 * _piq_pace
        )

        if _piq_ast > 0.55 and (_piq_handoff > 0.35 or _piq_post_hub > 0.35):
            _piq_raw *= 1.05

        # Decision-maker boost: high-level pass engines and connector hubs.
        _piq_playmaker_bonus = 0.0
        if _piq_ast > 0.50:
            _piq_playmaker_bonus += 0.030 * _piq_pass_acc + 0.025 * _piq_ast_tov
        if _piq_connector > 0.40:
            _piq_playmaker_bonus += 0.045 * _piq_connector
        _piq_raw += _piq_playmaker_bonus

        # Good decision-makers should retain baseline value even in smaller playmaking roles.
        _piq_floor = (
            0.42
            + 0.10 * _piq_shot_iq
            + 0.10 * _piq_pass_acc
            + 0.08 * _piq_ast_tov
        )

        # Must show some playmaking decision load to reach elite levels.
        _piq_opp = 0.42 * _piq_ast + 0.28 * _piq_creation + 0.18 * _piq_usage + 0.12 * _piq_post_hub
        _piq_gate = 1.0 if _piq_opp >= 0.16 else (0.66 + 0.34 * max(0.0, _piq_opp / 0.16))

        _piq_floor_cap = 0.68
        if size_big > 0.65 and _piq_opp < 0.18:
            _piq_floor += 0.028 * _profile_noise
            _piq_floor = max(0.36, min(0.54, _piq_floor))
            _piq_floor_cap = 0.56

        _piq_value = 100.0 * _piq_raw * _piq_gate
        _piq_value = max(_piq_value, 100.0 * min(_piq_floor_cap, _piq_floor))

        attrs["pass_iq"] = max(0.0, min(100.0, _piq_value))

        # ── Pass Vision ───────────────────────────────────────────────
        # Pass vision = seeing windows and opportunities early enough to create
        # quality passing chances (distinct from pure pass execution quality).
        _pv_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _pv_pnr_p36 = pnr_bh_burden_p36
        _pv_iso_p36 = iso_burden_p36
        _pv_handoff_p36 = raw_handoff_p36
        _pv_post_p36 = post_burden_p36

        _pv_ast = self._norm(_pv_ast_p36, 1.8, 12.5)
        _pv_ast_pct = self._norm(pctile_ast, 0.30, 0.99)
        _pv_usage = self._norm(usage, 14.0, 35.0)

        _pv_pnr = self._norm(_pv_pnr_p36, 0.20, 10.0)
        _pv_iso = self._norm(_pv_iso_p36, 0.10, 6.0)
        _pv_handoff = self._norm(_pv_handoff_p36, 0.05, 3.6)
        _pv_post_hub = self._norm(_pv_post_p36, 0.05, 4.8)
        _pv_creation = 0.44 * _pv_pnr + 0.20 * _pv_iso + 0.20 * _pv_handoff + 0.16 * _pv_post_hub

        _pv_pass_iq = self._norm(attrs["pass_iq"], 60.0, 96.0)
        _pv_pass_acc = self._norm(attrs["pass_accuracy"], 60.0, 95.0)
        _pv_shot_iq = self._norm(attrs["shot_iq"], 60.0, 95.0)
        _pv_size_view = 0.65 * self._norm(height, 74, 84) + 0.35 * self._norm(weight, 185, 265)
        _pv_orchestrator = self._norm(_pv_ast_p36, 4.8, 10.8) * (1.0 - self._norm(usage, 24.0, 35.0))
        _pv_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)

        # Visionary passers can carry slightly elevated risk while still creating value.
        _pv_expected_tov = 0.10 + 0.07 * _pv_usage
        _pv_tov_delta = max(-0.08, min(0.08, tov_pct - _pv_expected_tov))
        _pv_risk_span = self._norm(_pv_tov_delta, -0.01, 0.05)

        _pv_raw = (
            0.24 * _pv_ast
            + 0.10 * _pv_ast_pct
            + 0.25 * _pv_creation
            + 0.14 * _pv_pass_iq
            + 0.09 * _pv_pass_acc
            + 0.03 * _pv_shot_iq
            + 0.04 * _pv_size_view
            + 0.03 * _pv_usage
            + 0.02 * _pv_risk_span
            + 0.06 * _pv_orchestrator
            + 0.04 * _pv_pace
        )

        if _pv_ast > 0.55 and (_pv_handoff > 0.30 or _pv_post_hub > 0.30):
            _pv_raw *= 1.05

        if _pv_ast > 0.55 and _pv_creation > 0.50:
            _pv_raw *= 1.08 + 0.04 * _pv_ast

        if _pv_orchestrator > 0.50 and _pv_pass_iq > 0.45:
            _pv_raw *= 1.06

        _pv_raw *= 1.12

        # Need real playmaking read load for elite vision grades.
        _pv_opp = 0.40 * _pv_ast + 0.36 * _pv_creation + 0.16 * _pv_usage + 0.08 * _pv_post_hub
        _pv_gate = 1.0 if _pv_opp >= 0.14 else (0.64 + 0.36 * max(0.0, _pv_opp / 0.14))

        attrs["pass_vision"] = max(0.0, min(100.0, 100.0 * _pv_raw * _pv_gate))

        # ── Offensive Consistency ─────────────────────────────────────
        # Offensive consistency = how reliably a player provides sound offense
        # over time, adjusted for role burden and decision quality.
        _oc_creation_p36 = iso_burden_p36 + pnr_bh_burden_p36

        _oc_eff = 0.55 * self._norm(ts_pct, 0.52, 0.67) + 0.45 * self._norm(efg_pct, 0.48, 0.62)
        _oc_usage = self._norm(usage, 12.0, 35.0)
        _oc_prod = self._norm(pts_pg, 6.0, 32.0)
        _oc_load = self._norm((pts_pg * usage) / 100.0, 1.6, 10.5)
        _oc_iq = 0.65 * self._norm(attrs["shot_iq"], 55.0, 95.0) + 0.35 * self._norm(attrs["pass_iq"], 55.0, 96.0)
        _oc_creation = 0.55 * self._norm(_oc_creation_p36, 0.4, 14.0) + 0.45 * self._norm(attrs["ball_handle"], 50.0, 92.0)
        _oc_foul = self._norm(attrs["draw_foul"], 45.0, 95.0)
        _oc_ft = self._norm(ft_pct, 0.60, 0.92)
        _oc_security = ball_security
        _oc_reliability = 0.55 * self._norm(gp, 20.0, 82.0) + 0.45 * self._norm(min_pg, 16.0, 36.0)
        _oc_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)

        _oc_raw = (
            0.23 * _oc_eff
            + 0.14 * _oc_iq
            + 0.19 * _oc_prod
            + 0.15 * _oc_usage
            + 0.09 * _oc_load
            + 0.10 * _oc_creation
            + 0.08 * _oc_foul
            + 0.02 * _oc_ft
            + 0.00 * _oc_security
            + 0.00 * _oc_pace
        )
        _oc_raw += 0.03 * _oc_security + 0.02 * _oc_pace

        # Star scorers who carry heavy load efficiently tend to sit near the top.
        if _oc_prod > 0.60 and _oc_eff > 0.55 and _oc_iq > 0.60:
            _oc_raw *= 1.06 + 0.06 * _oc_prod

        if _oc_load > 0.62 and _oc_eff > 0.50:
            _oc_raw *= 1.06 + 0.05 * _oc_load

        # Poor shot quality + poor efficiency should sink consistency quickly.
        if attrs["shot_iq"] < 60 and _oc_eff < 0.45:
            _oc_raw *= 0.86 + 0.14 * _oc_iq

        _oc_raw *= 1.10

        _oc_opp = 0.42 * _oc_prod + 0.30 * _oc_usage + 0.18 * _oc_iq + 0.10 * _oc_creation
        _oc_gate = 1.0 if _oc_opp >= 0.18 else (0.55 + 0.45 * max(0.0, _oc_opp / 0.18))

        _oc_value = 100.0 * _oc_raw * _oc_gate * (0.82 + 0.18 * _oc_reliability)

        attrs["offensive_consistency"] = max(0.0, min(90.0, _oc_value))

        # ══════════════════════════════════════════════════════════════
        # DEFENSE
        # ══════════════════════════════════════════════════════════════

        # ── Interior Defense ──────────────────────────────────────────
        _id_size = (
            0.55 * self._norm(height, 76, 84)
            + 0.45 * self._norm(weight, 200, 275)
        )
        _id_strength = (
            0.65 * self._norm(weight, 205, 280)
            + 0.35 * self._norm(height, 76, 84)
        )
        _id_blk = self._norm(blk_per36, 0.12, 2.70)
        _id_dreb = self._norm(dreb_pg, 2.8, 10.8)
        _id_disc = 1.0 - self._norm(pf_per36, 1.8, 4.9)
        _id_engage = def_engagement
        _id_rim = self._norm(blk_per36 + 0.40 * dreb_pg, 1.3, 6.7)
        _id_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _id_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _id_anchor = (
            0.34 * _id_rim
            + 0.22 * _id_blk
            + 0.18 * _id_size
            + 0.14 * _id_dreb
            + 0.06 * _id_strength
            + 0.04 * _id_disc
            + 0.06 * _id_recover
        )

        if is_big:
            _id_raw = (
                0.56 * _id_anchor
                + 0.14 * _id_strength
                + 0.12 * _id_disc
                + 0.10 * _id_engage
                + 0.08 * _id_dreb
            )
        elif is_wing:
            _id_raw = (
                0.45 * _id_anchor
                + 0.18 * _id_strength
                + 0.14 * _id_disc
                + 0.13 * _id_engage
                + 0.10 * _id_dreb
            )
        else:
            _id_raw = (
                0.33 * _id_anchor
                + 0.16 * _id_strength
                + 0.18 * _id_disc
                + 0.20 * _id_engage
                + 0.13 * _id_dreb
            )

        # Interior defense opportunity gate: smaller low-rim profiles should not
        # grade like true paint anchors.
        _id_opp = 0.58 * _id_size + 0.42 * _id_rim
        _id_gate = 1.0 if _id_opp >= 0.34 else (0.58 + 0.42 * max(0.0, _id_opp / 0.34))

        # Elite anchor recognition for dominant rim-protecting bigs.
        if is_big and _id_rim > 0.72 and _id_blk > 0.62 and _id_dreb > 0.56:
            _id_raw *= 1.08 + 0.06 * _id_rim

        # Mobile elite-big recognition (switchable rim protectors).
        if is_big and _id_blk > 0.66 and _id_dreb > 0.54 and _id_strength < 0.62:
            _id_raw *= 1.08 + 0.05 * _id_blk

        # Foul-prone elite shot blockers should not collapse too far when
        # rim-protection volume is truly high.
        if is_big and _id_blk > 0.58 and _id_disc < 0.35 and blk_per36 >= 1.60:
            _id_raw *= 1.08

        # Raw floor: elite shot-blocking bigs (JJJ/Myles tier).
        # Blk per 36 is the hardest rim-deterrence signal; a volume of 1.60+
        # is already elite regardless of foul profile or rebounding role.
        if is_big and blk_per36 >= 1.60:
            _id_raw = max(_id_raw, 0.72)

        # Raw floor: dominant mobile rim-protectors with rebounding (Mobley tier).
        if is_big and blk_per36 >= 1.75 and _id_dreb > 0.50:
            _id_raw = max(_id_raw, 0.76)

        # High-offense guards/creators with only event steals should not grade high inside.
        if (not is_big) and usage > 30 and _id_size < 0.45 and _id_blk < 0.28:
            _id_raw *= 0.88 + 0.12 * _id_disc

        _id_value = 100.0 * _id_raw * _id_gate * (0.84 + 0.16 * _id_reliability)
        attrs["interior_defense"] = max(0.0, min(100.0, _id_value))

        # ── Perimeter Defense ─────────────────────────────────────────
        # Ability to stay in front of ball-handlers, contest perimeter shots,
        # and disrupt 1-on-1 assignments.
        #
        # Signals:
        #   stl_per36   → lateral pressure / on-ball disruption
        #   pf_per36    → discipline (getting beat = reaching/fouling)
        #   height      → length for contesting perimeter shots
        #   weight      → physical tools to body up
        #   usage       → low usage ≡ dedicated defender
        #   blk_per36   → wing shot-contest contribution

        _pd_stl      = self._norm(stl_per36, 0.50, 2.50)
        _pd_disc     = 1.0 - self._norm(pf_per36, 1.5, 4.6)
        _pd_length   = self._norm(height, 73, 81)
        _pd_body     = self._norm(weight, 170, 240)
        _pd_engage   = def_engagement
        _pd_contest  = self._norm(blk_per36, 0.05, 1.60)
        _pd_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        # Lateral quickness proxy: lighter body at appropriate height moves better.
        _pd_quickness = (
            0.55 * (1.0 - self._norm(weight, 165, 255))
            + 0.45 * self._norm(height, 73, 80)
        )

        # On-ball pressure composite.
        _pd_pressure = (
            0.38 * _pd_stl
            + 0.25 * _pd_disc
            + 0.20 * _pd_quickness
            + 0.17 * _pd_length
        )

        if is_guard:
            _pd_raw = (
                0.52 * _pd_pressure
                + 0.18 * _pd_engage
                + 0.15 * _pd_contest
                + 0.15 * _pd_body
            )
        elif is_big:
            _pd_raw = (
                0.40 * _pd_pressure
                + 0.22 * _pd_engage
                + 0.20 * _pd_contest
                + 0.18 * _pd_body
            )
        else:  # wing
            _pd_raw = (
                0.48 * _pd_pressure
                + 0.20 * _pd_engage
                + 0.18 * _pd_contest
                + 0.14 * _pd_body
            )

        _pd_raw *= 1.10

        # High-usage creators: big offensive load correlates with reduced
        # defensive investment on the ball.
        if usage > 32 and _pd_engage < 0.22:
            _pd_raw *= 0.88 + 0.12 * _pd_disc

        # Steals-only gambler penalty: high steals + tiny frame + heavy usage
        # = reaching and gambling, not disciplined perimeter defense.
        _pd_gambler = (
            self._norm(stl_per36, 1.20, 2.40)
            * (1.0 - self._norm(weight, 180, 250))
        )
        if _pd_gambler > 0.45 and usage > 30:
            _pd_raw *= 0.90 + 0.10 * (1.0 - _pd_gambler)

        # High-usage low-event defenders (small guards with weak stocks) should
        # not grade near average perimeter defense by default.
        if usage > 27 and _pd_engage < 0.40 and _pd_stl < 0.45 and _pd_contest < 0.25:
            _pd_raw *= 0.82 + 0.18 * _pd_disc

        if height <= 76 and weight <= 200 and usage > 24 and stl_per36 < 1.00 and blk_per36 < 0.45:
            _pd_raw *= 0.84 + 0.16 * _pd_disc

        # Tiny high-usage guards are often hunted defensively even with okay
        # steal counts; keep perimeter grades out of stopper ranges.
        if height <= 75 and weight <= 180 and usage >= 26 and blk_per36 < 0.40:
            _pd_raw *= 0.78 + 0.22 * _pd_disc

        # True bigs over ~6'10 are limited on the perimeter regardless of
        # other traits.
        if is_big and height > 82:
            _pd_raw *= 0.88

        _pd_opp = 0.48 * _pd_pressure + 0.32 * _pd_quickness + 0.20 * _pd_length
        _pd_gate = 1.0 if _pd_opp >= 0.32 else (0.58 + 0.42 * max(0.0, _pd_opp / 0.32))

        if is_center and height >= 83:
            _pd_raw *= 0.78 + 0.22 * _pd_quickness
            _pd_raw = min(_pd_raw, 0.40 + 0.24 * _pd_quickness + 0.16 * _pd_contest)

        # ── Stopper-recognition floors ───────────────────────────────
        # Proxy stats (stl, pf) don't fully capture coverage quality; floors
        # anchor known archetypes so the formula doesn't under-rate them.

        # Tier 1 — Elite two-way wing/guard (Kawhi tier):
        # Dominant stl volume + elite discipline = true elite regardless of usage.
        if (not is_big) and stl_per36 >= 1.70 and _pd_disc >= 0.80:
            _pd_raw = max(_pd_raw, 0.87)

        # Tier 2 — Elite on-ball disruptor at low usage (Caruso tier):
        # Extreme stl rate + full defensive commitment + adequate discipline.
        if (not is_big) and stl_per36 >= 1.80 and _pd_disc >= 0.40 and usage < 20:
            _pd_raw = max(_pd_raw, 0.88)

        # Tier 3 — Veteran all-around perimeter defender (Jrue/White tier):
        # Consistent stl + solid discipline + moderate usage.
        if (not is_big) and stl_per36 >= 1.00 and _pd_disc >= 0.75 and usage < 25:
            _pd_raw = max(_pd_raw, 0.74)

        # Tier 4 — Physical dedicated stopper guard (Dort tier):
        # Guard + very low usage + heavy build = fully committed physical specialist.
        if is_guard and usage < 18 and weight >= 210 and stl_per36 >= 0.90:
            _pd_raw = max(_pd_raw, 0.82)

        # Tier 5 — Heavy-wing switchable defender (Draymond tier):
        # Low-usage, physically built wing who covers guard assignments.
        if is_wing and (not is_big) and usage < 20 and weight >= 215 and stl_per36 >= 1.20:
            _pd_raw = max(_pd_raw, 0.74)

        _pd_value = 100.0 * _pd_raw * _pd_gate * (0.82 + 0.18 * _pd_reliability)
        if is_center and height >= 83:
            _pd_value = min(_pd_value, 42.0 + 12.0 * _pd_quickness + 8.0 * _pd_contest)
        attrs["perimeter_defense"] = max(0.0, min(100.0, _pd_value))

        # ── Steal ─────────────────────────────────────────────────────
        # Ability to generate clean deflections/strips without constant gambling.
        _sl_activity = self._norm(stl_per36, 0.45, 2.80)
        _sl_disc = 1.0 - self._norm(pf_per36, 1.6, 4.8)
        _sl_length = self._norm(height, 73, 81)
        _sl_quickness = 1.0 - self._norm(weight, 170, 255)
        _sl_tools = 0.58 * _sl_length + 0.42 * _sl_quickness
        _sl_engage = def_engagement
        _sl_contest = self._norm(blk_per36, 0.05, 1.60)
        _sl_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 90.0)
        _sl_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_guard:
            _sl_raw = (
                0.60 * _sl_activity
                + 0.14 * _sl_poa
                + 0.11 * _sl_disc
                + 0.09 * _sl_tools
                + 0.06 * _sl_engage
            )
        elif is_big:
            _sl_raw = (
                0.49 * _sl_activity
                + 0.14 * _sl_poa
                + 0.14 * _sl_disc
                + 0.12 * _sl_tools
                + 0.11 * _sl_contest
            )
        else:  # wing
            _sl_raw = (
                0.56 * _sl_activity
                + 0.16 * _sl_poa
                + 0.12 * _sl_disc
                + 0.10 * _sl_tools
                + 0.06 * _sl_engage
            )

        _sl_raw *= 1.08

        # High-offense creators with weak engagement usually rack up opportunistic
        # steals but don't sustain elite steal pressure.
        if usage > 32 and _sl_engage < 0.22:
            _sl_raw *= 0.90 + 0.10 * _sl_disc

        # Reaching/gambling profile: high steals + poor discipline + tiny frame.
        _sl_gambler = _sl_activity * (1.0 - _sl_disc) * (1.0 - self._norm(weight, 180, 250))
        if _sl_gambler > 0.42 and usage > 30:
            _sl_raw *= 0.90 + 0.10 * (1.0 - _sl_gambler)

        # Very tall true bigs are naturally capped for on-ball steals.
        if is_big and height > 82 and stl_per36 < 1.10:
            _sl_raw *= 0.90

        # Archetype floors
        if (not is_big) and stl_per36 >= 1.90 and usage < 22:
            _sl_raw = max(_sl_raw, 0.90)

        if (not is_big) and stl_per36 >= 1.65 and _sl_disc >= 0.72:
            _sl_raw = max(_sl_raw, 0.83)

        if (not is_big) and stl_per36 >= 1.05 and _sl_disc >= 0.74 and usage < 26:
            _sl_raw = max(_sl_raw, 0.71)

        # Physical low-usage stopper guards still generate meaningful strips
        # even when aggressive foul profiles reduce discipline signals.
        if is_guard and usage < 18 and weight >= 205 and stl_per36 >= 1.05:
            _sl_raw = max(_sl_raw, 0.67)

        # Heavy wings with real event creation (Draymond archetype) should stay
        # above average steal ratings despite high foul counts.
        if is_wing and (not is_big) and usage < 20 and weight >= 215 and stl_per36 >= 1.25:
            _sl_raw = max(_sl_raw, 0.70)

        if is_big and stl_per36 >= 1.20 and _sl_disc >= 0.55:
            _sl_raw = max(_sl_raw, 0.63)

        _sl_value = 100.0 * _sl_raw * (0.80 + 0.20 * _sl_reliability)
        attrs["steal"] = max(0.0, min(100.0, _sl_value))

        # ── Block ─────────────────────────────────────────────────────
        # Ability to protect the rim and contest shots vertically.
        _bl_activity = self._norm(blk_per36, 0.15, 3.60)
        _bl_size = (
            0.55 * self._norm(height, 76, 85)
            + 0.45 * self._norm(weight, 190, 280)
        )
        _bl_rim = self._norm(blk_per36 + 0.35 * dreb_pg, 1.2, 6.9)
        _bl_disc = 1.0 - self._norm(pf_per36, 1.7, 4.9)
        _bl_engage = def_engagement
        _bl_interior = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _bl_recovery = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _bl_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_big:
            _bl_raw = (
                0.49 * _bl_activity
                + 0.18 * _bl_rim
                + 0.15 * _bl_size
                + 0.10 * _bl_interior
                + 0.06 * _bl_disc
                + 0.02 * _bl_recovery
            )
        elif is_wing:
            _bl_raw = (
                0.44 * _bl_activity
                + 0.16 * _bl_rim
                + 0.15 * _bl_size
                + 0.13 * _bl_interior
                + 0.10 * _bl_engage
                + 0.02 * _bl_recovery
            )
        else:
            _bl_raw = (
                0.46 * _bl_activity
                + 0.13 * _bl_rim
                + 0.14 * _bl_size
                + 0.12 * _bl_interior
                + 0.13 * _bl_engage
                + 0.02 * _bl_recovery
            )

        _bl_raw *= 1.08

        # Usage-heavy creators with poor engagement usually don't sustain
        # real shot-block impact outside occasional highlights.
        if (not is_big) and usage > 32 and blk_per36 < 0.80:
            _bl_raw *= 0.86 + 0.14 * _bl_disc

        # Very small guards are naturally capped as shot blockers.
        if is_guard and height < 75 and blk_per36 < 0.60:
            _bl_raw *= 0.90

        # Very tall traditional bigs can protect the rim even with lower steal events.
        if is_big and height >= 84 and blk_per36 >= 1.70:
            _bl_raw *= 1.05

        # Archetype floors.
        if is_big and blk_per36 >= 3.20:
            _bl_raw = max(_bl_raw, 0.92)

        if is_big and blk_per36 >= 2.10:
            _bl_raw = max(_bl_raw, 0.80)

        if is_big and blk_per36 >= 1.80 and height >= 84:
            _bl_raw = max(_bl_raw, 0.78)

        if is_big and blk_per36 >= 1.65 and dreb_pg >= 5.2:
            _bl_raw = max(_bl_raw, 0.75)

        # Mobile rebounding bigs can still be quality weakside shot blockers
        # even below classic anchor block volume.
        if is_big and blk_per36 >= 0.75 and dreb_pg >= 7.2 and height >= 81 and weight <= 265 and usage < 30:
            _bl_raw = max(_bl_raw, 0.58)

        if is_guard and blk_per36 >= 1.20 and height >= 76:
            _bl_raw = max(_bl_raw, 0.66)

        if is_guard and blk_per36 >= 0.75 and height >= 76:
            _bl_raw = max(_bl_raw, 0.57)

        if is_wing and (not is_big) and blk_per36 >= 0.95 and height >= 77 and usage < 22:
            _bl_raw = max(_bl_raw, 0.60)

        # Jumbo wings with strong weakside activity (Giannis archetype).
        if is_wing and height >= 82 and blk_per36 >= 0.95 and dreb_pg >= 8.5 and stl_per36 >= 1.0:
            _bl_raw = max(_bl_raw, 0.56)

        _bl_value = 100.0 * _bl_raw * (0.82 + 0.18 * _bl_reliability)
        attrs["block"] = max(0.0, min(100.0, _bl_value))

        # ── Offensive Rebound ─────────────────────────────────────────
        # Offensive Rebound = second-chance creation via timing, positioning,
        # and physical presence on the offensive glass.
        _or_rate = self._norm(oreb_pg, 0.30, 4.40)
        _or_size = (
            0.55 * self._norm(height, 76, 85)
            + 0.45 * self._norm(weight, 190, 280)
        )
        _or_strength = self._norm(weight, 190, 285)
        _or_disc = 1.0 - self._norm(pf_per36, 1.8, 4.9)
        _or_blk = self._norm(blk_per36, 0.10, 2.60)
        _or_crash = 1.0 - self._norm(usage, 18, 35)
        _or_second = self._norm(second_chance_off_poss_rate, 0.03, 0.20)
        _or_putback = self._norm(putback_rate, 0.0, 0.16)
        _or_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_big:
            _or_raw = (
                0.55 * _or_rate
                + 0.16 * _or_size
                + 0.12 * _or_strength
                + 0.07 * _or_blk
                + 0.06 * _or_disc
                + 0.05 * _or_second
                + 0.02 * _or_putback
            )
        elif is_wing:
            _or_raw = (
                0.50 * _or_rate
                + 0.18 * _or_size
                + 0.12 * _or_strength
                + 0.08 * _or_crash
                + 0.08 * _or_disc
                + 0.02 * _or_second
                + 0.02 * _or_putback
            )
        else:  # guard
            _or_raw = (
                0.42 * _or_rate
                + 0.20 * _or_crash
                + 0.14 * _or_size
                + 0.12 * _or_strength
                + 0.08 * _or_disc
                + 0.02 * _or_second
                + 0.02 * _or_putback
            )

        _or_raw *= 1.08

        # High-usage non-bigs usually leak out for transition creation instead
        # of crashing every possession.
        if (not is_big) and usage > 31 and oreb_pg < 1.20:
            _or_raw *= 0.90 + 0.10 * _or_disc

        # Archetype floors.
        if is_big and oreb_pg >= 3.80:
            _or_raw = max(_or_raw, 0.86)

        if is_big and oreb_pg >= 3.20:
            _or_raw = max(_or_raw, 0.78)

        if is_big and oreb_pg >= 2.60:
            _or_raw = max(_or_raw, 0.70)

        # Elite shot-blocking bigs with even moderate crash volume (JJJ archetype)
        # should grade in the 80s band on offensive rebounding.
        if is_big and blk_per36 >= 1.60 and oreb_pg >= 1.10:
            _or_raw = max(_or_raw, 0.72)

        # Rebounding wings/guards (Hart/Luka lane).
        if (not is_big) and oreb_pg >= 1.60 and dreb_pg >= 6.5 and height >= 77:
            _or_raw = max(_or_raw, 0.58)

        # Jumbo creator wing who still attacks weakside boards (Giannis lane).
        if is_wing and usage > 34 and oreb_pg >= 2.60 and dreb_pg >= 8.5:
            _or_raw = max(_or_raw, 0.66)

        _or_value = 100.0 * _or_raw * (0.84 + 0.16 * _or_reliability)
        attrs["offensive_rebound"] = max(0.0, min(100.0, _or_value))

        # ── Defensive Rebound ─────────────────────────────────────────
        # Defensive Rebound = ending possessions through positioning,
        # box-outs, and securing contested misses.
        _dr_rate = self._norm(dreb_pg, 2.5, 10.8)
        _dr_size = (
            0.54 * self._norm(height, 76, 85)
            + 0.46 * self._norm(weight, 190, 285)
        )
        _dr_strength = self._norm(weight, 190, 290)
        _dr_disc = 1.0 - self._norm(pf_per36, 1.8, 4.9)
        _dr_blk = self._norm(blk_per36, 0.10, 2.60)
        _dr_stl = self._norm(stl_per36, 0.50, 2.20)
        _dr_engage = def_engagement
        _dr_interior = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _dr_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_big:
            _dr_raw = (
                0.46 * _dr_rate
                + 0.18 * _dr_size
                + 0.12 * _dr_strength
                + 0.10 * _dr_interior
                + 0.08 * _dr_disc
                + 0.06 * _dr_blk
            )
        elif is_wing:
            _dr_raw = (
                0.43 * _dr_rate
                + 0.17 * _dr_size
                + 0.14 * _dr_engage
                + 0.10 * _dr_disc
                + 0.08 * _dr_blk
                + 0.08 * _dr_interior
            )
        else:  # guard
            _dr_raw = (
                0.45 * _dr_rate
                + 0.17 * _dr_engage
                + 0.12 * _dr_size
                + 0.10 * _dr_disc
                + 0.08 * _dr_stl
                + 0.08 * _dr_interior
            )

        _dr_raw *= 1.08

        # High-usage non-bigs with low rebound volume should not overgrade.
        if (not is_big) and usage > 34 and dreb_pg < 5.5:
            _dr_raw *= 0.90 + 0.10 * _dr_disc

        # Archetype floors.
        if is_big and dreb_pg >= 9.50:
            _dr_raw = max(_dr_raw, 0.88)

        if is_big and dreb_pg >= 8.80:
            _dr_raw = max(_dr_raw, 0.82)

        if is_big and dreb_pg >= 7.50:
            _dr_raw = max(_dr_raw, 0.74)

        # Elite shot-blocking bigs who finish possessions at a moderate level
        # should still sit in the 80s on defensive rebounding.
        if is_big and blk_per36 >= 1.60 and dreb_pg >= 5.0:
            _dr_raw = max(_dr_raw, 0.72)

        # Rebounding guards/wings.
        if (not is_big) and dreb_pg >= 7.0 and height >= 77:
            _dr_raw = max(_dr_raw, 0.67)

        if is_guard and dreb_pg >= 6.8 and usage < 20:
            _dr_raw = max(_dr_raw, 0.64)

        # Jumbo wing glass cleaner (Giannis archetype).
        if is_wing and height >= 82 and dreb_pg >= 8.8:
            _dr_raw = max(_dr_raw, 0.78)

        _dr_value = 100.0 * _dr_raw * (0.84 + 0.16 * _dr_reliability)
        attrs["defensive_rebound"] = max(0.0, min(100.0, _dr_value))

        # ── Help Defense IQ ───────────────────────────────────────────
        # Help Defense IQ = rotation timing, tag/help decisions, and
        # team-defense positioning.
        _hd_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 92.0)
        _hd_rim = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _hd_stl = self._norm(stl_per36, 0.45, 2.40)
        _hd_blk = self._norm(blk_per36, 0.15, 3.20)
        _hd_dreb = self._norm(dreb_pg, 2.8, 10.8)
        _hd_disc = 1.0 - self._norm(pf_per36, 1.7, 4.9)
        _hd_engage = def_engagement
        _hd_balance = 1.0 - abs(_hd_poa - _hd_rim)
        _hd_tools = 0.58 * self._norm(height, 74, 84) + 0.42 * self._norm(weight, 185, 270)
        _hd_events = self._norm(stl_per36 + 0.90 * blk_per36, 1.0, 5.0)
        _hd_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _hd_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_guard:
            _hd_raw = (
                0.36 * _hd_poa
                + 0.18 * _hd_stl
                + 0.14 * _hd_disc
                + 0.12 * _hd_engage
                + 0.10 * _hd_balance
                + 0.08 * _hd_rim
                + 0.02 * _hd_recover
            )
        elif is_big:
            _hd_raw = (
                0.32 * _hd_rim
                + 0.18 * _hd_blk
                + 0.16 * _hd_dreb
                + 0.12 * _hd_disc
                + 0.12 * _hd_engage
                + 0.08 * _hd_poa
                + 0.02 * _hd_recover
            )
        else:  # wing
            _hd_raw = (
                0.26 * _hd_poa
                + 0.20 * _hd_rim
                + 0.13 * _hd_stl
                + 0.11 * _hd_blk
                + 0.12 * _hd_disc
                + 0.10 * _hd_engage
                + 0.06 * _hd_balance
                + 0.02 * _hd_recover
            )

        _hd_raw *= 1.10

        # High-usage offensive engines with low defensive engagement generally
        # miss rotations more often.
        if usage > 32 and _hd_engage < 0.24 and (_hd_poa < 0.72 or _hd_rim < 0.72):
            _hd_raw *= 0.89 + 0.11 * _hd_disc

        # Steals-only gambler profile: event steals without team-defense signals.
        _hd_gambler = _hd_stl * (1.0 - _hd_disc) * (1.0 - _hd_blk)
        if _hd_gambler > 0.40 and usage > 30:
            _hd_raw *= 0.90 + 0.10 * (1.0 - _hd_gambler)

        # High-usage players with weak activity + weak discipline should grade
        # clearly lower as help defenders.
        if usage > 28 and _hd_engage < 0.40 and _hd_events < 0.28 and _hd_disc < 0.55:
            _hd_raw *= 0.80 + 0.20 * _hd_disc

        if height <= 75 and weight <= 180 and usage >= 26 and blk_per36 < 0.40:
            _hd_raw *= 0.80 + 0.20 * _hd_disc

        # Smart low-usage connective defenders should stand out.
        if (not is_big) and usage < 23 and _hd_poa > 0.62 and _hd_disc > 0.60 and _hd_events > 0.35:
            _hd_raw *= 1.06 + 0.05 * _hd_balance

        # Archetype floors.
        if is_big and blk_per36 >= 1.60 and dreb_pg >= 7.2:
            _hd_raw = max(_hd_raw, 0.86)

        # Active weakside bigs with strong interior command (Mobley/JJJ/AD lane).
        if is_big and blk_per36 >= 1.55:
            _hd_raw = max(_hd_raw, 0.78)

        if (not is_big) and stl_per36 >= 1.10 and usage < 25:
            _hd_raw = max(_hd_raw, 0.80)

        # High-usage elite wing stoppers still provide strong team help reads.
        if is_wing and stl_per36 >= 1.70 and pf_per36 <= 2.1:
            _hd_raw = max(_hd_raw, 0.74)

        # High-IQ read/react bigs with elite glass + hands but moderate block volume.
        if is_big and dreb_pg >= 8.8 and stl_per36 >= 1.25 and _hd_disc > 0.45:
            _hd_raw = max(_hd_raw, 0.74)

        # Jumbo wing help lane (Giannis archetype).
        if is_wing and height >= 82 and blk_per36 >= 0.95 and dreb_pg >= 8.5 and stl_per36 >= 1.0:
            _hd_raw = max(_hd_raw, 0.74)

        # Versatile low-usage wing helper lane (Draymond archetype).
        if is_wing and usage < 22 and stl_per36 >= 1.20 and blk_per36 >= 0.90:
            _hd_raw = max(_hd_raw, 0.76)

        _hd_value = 100.0 * _hd_raw * (0.82 + 0.18 * _hd_reliability)
        attrs["help_defense_iq"] = max(0.0, min(100.0, _hd_value))

        # ── Pass Perception ───────────────────────────────────────────
        # Pass Perception = anticipation of passing lanes and timely reads.
        _pp_stl = self._norm(stl_per36, 0.45, 2.50)
        _pp_disc = 1.0 - self._norm(pf_per36, 1.7, 4.9)
        _pp_engage = def_engagement
        _pp_len = self._norm(height, 73, 82)
        _pp_tools = (
            0.66 * _pp_len
            + 0.34 * (1.0 - self._norm(weight, 175, 255))
        )
        _pp_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 92.0)
        _pp_help = self._norm(_hd_raw, 0.45, 0.90)
        _pp_stock = self._norm(stl_per36 + 0.45 * blk_per36, 0.9, 4.0)
        _pp_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _pp_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )
        _pp_jumbo_reader = (
            (not is_big)
            and height >= 79
            and stl_per36 >= 1.45
            and dreb_pg >= 6.5
            and usage >= 33
        )

        if is_guard:
            _pp_raw = (
                0.36 * _pp_stl
                + 0.18 * _pp_poa
                + 0.19 * _pp_help
                + 0.11 * _pp_disc
                + 0.08 * _pp_tools
                + 0.06 * _pp_engage
                + 0.02 * _pp_recover
            )
        elif is_big:
            _pp_raw = (
                0.27 * _pp_stl
                + 0.15 * _pp_poa
                + 0.27 * _pp_help
                + 0.13 * _pp_disc
                + 0.10 * _pp_tools
                + 0.06 * _pp_stock
                + 0.02 * _pp_recover
            )
        else:  # wing
            _pp_raw = (
                0.32 * _pp_stl
                + 0.18 * _pp_poa
                + 0.23 * _pp_help
                + 0.11 * _pp_disc
                + 0.08 * _pp_tools
                + 0.06 * _pp_engage
                + 0.02 * _pp_recover
            )

        _pp_raw *= 1.10

        # High-usage creators with weak engagement usually jump passing lanes
        # less consistently across possessions.
        if usage > 32 and _pp_engage < 0.24 and (not _pp_jumbo_reader):
            _pp_raw *= 0.90 + 0.10 * _pp_disc

        # Event-steal gamblers: steals without discipline/help context should
        # not map to elite pass perception.
        _pp_gambler = _pp_stl * (1.0 - _pp_disc) * (1.0 - _pp_help)
        if _pp_gambler > 0.40 and usage > 30:
            _pp_raw *= 0.90 + 0.10 * (1.0 - _pp_gambler)

        # Archetype floors.
        if (not is_big) and stl_per36 >= 1.70 and _pp_help > 0.55:
            _pp_raw = max(_pp_raw, 0.84)

        if (not is_big) and stl_per36 >= 1.10 and _pp_help > 0.55 and usage < 25:
            _pp_raw = max(_pp_raw, 0.74)

        if is_big and _pp_help > 0.60 and dreb_pg >= 7.0:
            _pp_raw = max(_pp_raw, 0.70)

        # Smart read/react bigs with elite hands and glass (Jokic lane).
        if is_big and stl_per36 >= 1.35 and dreb_pg >= 8.8 and _pp_disc > 0.45:
            _pp_raw = max(_pp_raw, 0.74)

        # Low-usage wing quarterback helpers (Draymond lane).
        if is_wing and usage < 22 and stl_per36 >= 1.20 and blk_per36 >= 0.85:
            _pp_raw = max(_pp_raw, 0.80)

        # High-usage jumbo lane-readers (Luka archetype): big creators who
        # consistently intercept cross-court and skip passes from anticipation.
        if _pp_jumbo_reader:
            _pp_raw = max(_pp_raw, 0.52)

        _pp_raw *= 0.82 + 0.18 * _pp_reliability
        _pp_raw *= 0.90 + 0.10 * _pp_help

        # Final-stage floors after reliability/context scaling.
        if (not is_big) and stl_per36 >= 1.70 and _pp_help > 0.55:
            _pp_raw = max(_pp_raw, 0.83)

        if (not is_big) and stl_per36 >= 1.10 and _pp_help > 0.55 and usage < 25:
            _pp_raw = max(_pp_raw, 0.75)

        if is_big and _pp_help > 0.60 and dreb_pg >= 7.0:
            _pp_raw = max(_pp_raw, 0.71)

        if is_big and stl_per36 >= 1.35 and dreb_pg >= 8.8 and _pp_disc > 0.45:
            _pp_raw = max(_pp_raw, 0.75)

        if is_wing and usage < 22 and stl_per36 >= 1.20 and blk_per36 >= 0.85:
            _pp_raw = max(_pp_raw, 0.79)

        if _pp_jumbo_reader:
            _pp_raw = max(_pp_raw, 0.52)

        attrs["pass_perception"] = max(0.0, min(100.0, 100.0 * _pp_raw))

        # ── Defensive Consistency ─────────────────────────────────────
        _dc_stl = self._norm(stl_per36, 0.55, 2.30)
        _dc_blk = self._norm(blk_per36, 0.10, 2.60)
        _dc_dreb = self._norm(dreb_pg, 2.8, 10.8)
        _dc_disc = 1.0 - self._norm(self._f(f, "pf_per36", 3.0), 1.6, 4.8)
        _dc_size = min(1.0, 0.70 * size_big + 0.30 * self._norm(height, 74, 83))
        _dc_engage = def_engagement
        _dc_stock = self._norm(stl_per36 + 0.85 * blk_per36, 1.0, 4.7)
        _dc_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _dc_reliability = (
            0.58 * self._norm(min_pg, 12, 34)
            + 0.42 * self._norm(gp, 25, 82)
        )

        _dc_poa = (
            0.33 * _dc_stl
            + 0.28 * _dc_disc
            + 0.21 * _dc_engage
            + 0.18 * self._norm(height, 73, 80)
        )
        _dc_anchor = (
            0.42 * _dc_blk
            + 0.27 * _dc_dreb
            + 0.21 * _dc_size
            + 0.10 * _dc_disc
        )

        if is_guard:
            _dc_raw = 0.56 * _dc_poa + 0.18 * _dc_anchor + 0.22 * _dc_stock + 0.04 * _dc_recover
        elif is_big:
            _dc_raw = 0.24 * _dc_poa + 0.56 * _dc_anchor + 0.16 * _dc_stock + 0.04 * _dc_recover
        else:
            _dc_raw = 0.44 * _dc_poa + 0.32 * _dc_anchor + 0.20 * _dc_stock + 0.04 * _dc_recover

        # Base lift keeps average defenders from collapsing too low.
        _dc_raw *= 1.13

        # Elite low-usage POA defenders (Caruso/Dort style) should stand out.
        if is_guard and _dc_poa > 0.70 and _dc_disc > 0.55:
            _dc_raw *= 1.08 + 0.10 * _dc_stl

        # Usage-heavy creators with weak engagement should grade lower even
        # when they generate some steals.
        if usage > 31 and _dc_engage < 0.30:
            _dc_raw *= 0.96 + 0.04 * _dc_disc

        # Bigger rebounding guards/wings can still be passable team defenders
        # even with heavy on-ball creation load.
        if (not is_big) and height >= 78 and _dc_dreb > 0.50 and _dc_disc > 0.35:
            _dc_raw *= 1.05 + 0.05 * self._norm(dreb_pg, 5.5, 9.0)

        # "Steals-only" profiles: event steals without anchor signals.
        _dc_steals_only = (
            self._norm(stl_per36, 1.35, 2.40)
            * (1.0 - self._norm(blk_per36, 0.35, 1.25))
            * (1.0 - _dc_size)
        )
        if _dc_steals_only > 0.40 and usage > 28:
            _dc_raw *= 0.88 + 0.12 * (1.0 - _dc_steals_only)

        # Poor-stock high-usage defenders should separate clearly from true
        # two-way players in defensive consistency.
        if usage > 28 and _dc_engage < 0.40 and _dc_stock < 0.22 and _dc_disc < 0.55:
            _dc_raw *= 0.78 + 0.22 * _dc_disc

        if height <= 75 and weight <= 180 and usage >= 26 and blk_per36 < 0.40:
            _dc_raw *= 0.76 + 0.24 * _dc_disc

        # POA stopper boost: disciplined perimeter defenders with real activity
        # should grade clearly above average even without huge block numbers.
        if (
            (not is_big)
            and _dc_disc > 0.68
            and _dc_poa > 0.52
            and _dc_stock > 0.34
            and usage < 26
        ):
            _dc_raw *= 1.08 + 0.06 * _dc_disc

        # Veteran guard/wing defenders with strong discipline + moderate steal
        # activity should avoid getting stuck in mid-tier buckets.
        if (
            (not is_big)
            and _dc_disc > 0.78
            and _dc_poa > 0.60
            and usage < 24
            and 1.10 <= stl_per36 <= 2.00
        ):
            _dc_raw *= 1.10 + 0.03 * _dc_disc

        if (
            (not is_big)
            and _dc_disc > 0.85
            and stl_per36 >= 1.25
            and usage < 24
        ):
            _dc_raw *= 1.10

        # Elite perimeter stopper lane (Dort-style): low-usage defenders with
        # strong POA pressure and real event creation should grade near elite.
        if (
            (not is_big)
            and usage < 19
            and _dc_poa > 0.56
            and stl_per36 >= 1.05
            and _dc_stock > 0.20
        ):
            _dc_raw *= 1.12 + 0.05 * _dc_poa

        # Some elite assignment stoppers carry very low offensive role but still
        # generate both steals and weak-side blocks; avoid over-penalizing them
        # for aggressive foul profiles.
        if (
            (not is_big)
            and usage < 18
            and min_pg >= 26
            and stl_per36 >= 1.00
            and blk_per36 >= 0.45
        ):
            _dc_raw *= 1.24

        # High-foul stopper correction (very narrow): certain elite wing stoppers
        # play hyper-aggressive defense that inflates fouls; avoid burying them.
        if (
            (not is_big)
            and usage < 18
            and min_pg >= 26
            and pf_per36 >= 3.2
            and stl_per36 >= 1.00
            and blk_per36 >= 0.50
        ):
            _dc_raw = max(_dc_raw, 0.80)

        # Elite anchor lane (Gobert/Mobley-style): rim protection + glass + size.
        if (
            is_big
            and _dc_anchor > 0.56
            and _dc_blk > 0.45
            and _dc_dreb > 0.52
        ):
            _dc_raw *= 1.07 + 0.08 * _dc_anchor

        # Prevent over-penalizing jumbo creators who carry huge offense but still
        # provide acceptable team defense activity.
        if (
            (not is_big)
            and usage > 34
            and height >= 78
            and _dc_dreb > 0.55
            and _dc_poa > 0.50
            and _dc_disc > 0.40
        ):
            _dc_raw = max(_dc_raw, 0.56)

        _dc_raw *= 0.80 + 0.20 * _dc_reliability
        attrs["defensive_consistency"] = max(0.0, min(100.0, 100.0 * _dc_raw))

        # ══════════════════════════════════════════════════════════════
        # PHYSICAL
        # ══════════════════════════════════════════════════════════════

        # ── Speed ─────────────────────────────────────────────────────
        # Speed = off-ball + open-floor movement independent of handle skill.
        _sp_transition = self._norm(transition_pos, 0.8, 6.2)
        _sp_swb = self._norm(attrs.get("speed_with_ball", 50.0), 18.0, 75.0)
        _sp_size = (
            0.58 * (1.0 - self._norm(height, 74, 85))
            + 0.42 * (1.0 - self._norm(weight, 180, 285))
        )
        _sp_rim = self._norm(rim_pressure, 0.18, 0.75)
        _sp_age = age_phys
        _sp_def_activity = self._norm(stl_per36 + 0.35 * blk_per36, 0.8, 3.0)
        _sp_pace = 1.0 - self._norm(seconds_per_poss_off, 14.0, 20.0)
        _sp_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_guard:
            _sp_raw = (
                0.28 * _sp_transition
                + 0.34 * _sp_swb
                + 0.20 * _sp_size
                + 0.08 * _sp_age
                + 0.07 * _sp_def_activity
                + 0.05 * _sp_pace
            )
        elif is_big:
            _sp_raw = (
                0.30 * _sp_transition
                + 0.18 * _sp_swb
                + 0.30 * _sp_size
                + 0.12 * _sp_age
                + 0.05 * _sp_rim
                + 0.05 * _sp_pace
            )
        else:  # wing
            _sp_raw = (
                0.24 * _sp_transition
                + 0.32 * _sp_swb
                + 0.22 * _sp_size
                + 0.10 * _sp_age
                + 0.07 * _sp_rim
                + 0.05 * _sp_pace
            )

        _sp_raw *= 1.08

        # Heavy true big drag.
        if is_big and height >= 82:
            _sp_big_mobility = 0.60 * _sp_transition + 0.40 * _sp_size
            _sp_raw *= 0.52 + 0.48 * _sp_big_mobility

        if is_big and height >= 83 and weight >= 250:
            _sp_raw *= 0.70 + 0.30 * _sp_transition
            _sp_raw = min(_sp_raw, 0.34 + 0.14 * _sp_transition + 0.10 * _sp_swb)

        # Very large centers with low open-floor role should stay clearly slower.
        if is_center and height >= 84 and weight >= 270 and transition_pos < 1.2:
            _sp_raw = min(_sp_raw, 0.22)

        # High-usage creators with low transition profile tend to pace/manage speed.
        if (not is_big) and usage > 34 and transition_pos < 3.0 and _sp_swb < 0.70:
            _sp_raw *= 0.93 + 0.07 * _sp_swb

        if is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _sp_raw *= 0.84 + 0.16 * _sp_swb
            _sp_raw = min(_sp_raw, 0.50 + 0.18 * _sp_swb + 0.12 * _sp_transition)

        # Aging jumbo stars should lose top-end speed while retaining functional pace.
        if age >= 35 and height >= 78 and weight >= 220:
            _sp_vet_drag = self._norm(age, 34, 41)
            _sp_raw *= 0.86 + 0.14 * (1.0 - _sp_vet_drag)

        if age >= 37 and height >= 78 and weight >= 220:
            _sp_raw = min(_sp_raw, 0.54 + 0.18 * _sp_transition + 0.12 * _sp_swb)

        # Jumbo primary wings can be fast in space without having true small-guard
        # end-to-end speed. Add a size-based drag for that archetype.
        if is_wing and height >= 79 and weight >= 225 and usage >= 32:
            _sp_raw *= 0.92 + 0.08 * _sp_transition

        # Some jumbo heliocentric creators build transition speed through pace
        # control and handling rather than true burst; keep them below explosive
        # athletic wings/guards unless they generate extreme rim pressure.
        if is_wing and height >= 79 and weight >= 225 and usage >= 35 and rim_pressure < 0.45:
            _sp_raw *= 0.86 + 0.14 * _sp_swb
            _sp_raw = min(_sp_raw, 0.67)

        # Opportunity gate: need real open-floor role to grade elite speed.
        _sp_opp = 0.35 * _sp_transition + 0.35 * _sp_swb + 0.20 * _sp_size + 0.10 * _sp_rim
        _sp_gate = 1.0 if _sp_opp >= 0.20 else (0.56 + 0.44 * max(0.0, _sp_opp / 0.20))

        # Archetype floors.
        if is_guard and transition_pos >= 4.2 and _sp_swb > 0.70:
            _sp_raw = max(_sp_raw, 0.84)

        # Skilled guards with strong functional speed even in lower transition volume.
        if is_guard and _sp_swb > 0.60 and transition_pos >= 3.0 and height <= 76:
            _sp_raw = max(_sp_raw, 0.66)

        if is_wing and height >= 80 and transition_pos >= 4.8 and _sp_rim > 0.55:
            _sp_raw = max(_sp_raw, 0.76)

        # Primary scoring wings with real movement tools should not collapse
        # from low transition play-type volume alone.
        if is_wing and height >= 79 and _sp_swb > 0.60 and usage >= 28:
            _sp_raw = max(_sp_raw, 0.60)

        _sp_value = 100.0 * _sp_raw * _sp_gate * (0.84 + 0.16 * _sp_reliability)
        if is_guard and height >= 79 and weight >= 220 and usage >= 28:
            _sp_value = min(_sp_value, 72.0 + 8.0 * _sp_swb + 6.0 * _sp_transition)
        if is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _sp_value = min(_sp_value, 62.0 + 10.0 * _sp_swb + 6.0 * _sp_transition)
        if is_big and height >= 83 and weight >= 250:
            _sp_value = min(_sp_value, 52.0 + 12.0 * _sp_transition + 8.0 * _sp_swb)
        if age >= 35 and height >= 78 and weight >= 220:
            _sp_value = min(_sp_value, 66.0 + 8.0 * _sp_swb + 6.0 * _sp_transition)
        attrs["speed"] = max(0.0, min(100.0, _sp_value))

        # ── Agility ───────────────────────────────────────────────────
        # Agility = lateral quickness, balance, and change-of-direction.
        _ag_spd = self._norm(attrs.get("speed", 50.0), 18.0, 78.0)
        _ag_swb = self._norm(attrs.get("speed_with_ball", 50.0), 18.0, 75.0)
        _ag_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 92.0)
        _ag_size = (
            0.62 * (1.0 - self._norm(height, 74, 85))
            + 0.38 * (1.0 - self._norm(weight, 180, 285))
        )
        _ag_age = age_phys
        _ag_stl = self._norm(stl_per36, 0.50, 2.20)
        _ag_pace = 1.0 - self._norm(seconds_per_poss_off, 14.0, 20.0)
        _ag_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if is_guard:
            _ag_raw = (
                0.28 * _ag_spd
                + 0.26 * _ag_swb
                + 0.22 * _ag_size
                + 0.12 * _ag_poa
                + 0.08 * _ag_age
                + 0.04 * _ag_pace
            )
        elif is_big:
            _ag_raw = (
                0.26 * _ag_spd
                + 0.16 * _ag_swb
                + 0.30 * _ag_size
                + 0.18 * _ag_poa
                + 0.06 * _ag_age
                + 0.04 * _ag_pace
            )
        else:  # wing
            _ag_raw = (
                0.28 * _ag_spd
                + 0.22 * _ag_swb
                + 0.22 * _ag_size
                + 0.14 * _ag_poa
                + 0.10 * _ag_age
                + 0.04 * _ag_pace
            )

        _ag_raw *= 1.08

        # Heavy big drag for lateral movement.
        if is_big and height >= 82:
            _ag_big_mobility = 0.55 * _ag_size + 0.45 * _ag_poa
            _ag_raw *= 0.48 + 0.52 * _ag_big_mobility

        if is_big and height >= 83 and weight >= 250:
            _ag_raw *= 0.68 + 0.32 * _ag_poa
            _ag_raw = min(_ag_raw, 0.30 + 0.18 * _ag_poa + 0.10 * _ag_spd)

        # Very large centers should remain clearly limited laterally.
        if is_center and height >= 84 and weight >= 270:
            _ag_raw = min(_ag_raw, 0.22)

        # High-usage jumbo creators can be functional movers without elite
        # twitch agility.
        if is_wing and height >= 79 and weight >= 225 and usage >= 35:
            _ag_raw *= 0.88 + 0.12 * _ag_swb

        if is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _ag_raw *= 0.84 + 0.16 * _ag_swb
            _ag_raw = min(_ag_raw, 0.50 + 0.18 * _ag_swb + 0.12 * _ag_poa)

        if age >= 35 and height >= 78 and weight >= 220:
            _ag_vet_drag = self._norm(age, 34, 41)
            _ag_raw *= 0.86 + 0.14 * (1.0 - _ag_vet_drag)

        if age >= 37 and height >= 78 and weight >= 220:
            _ag_raw = min(_ag_raw, 0.52 + 0.16 * _ag_swb + 0.14 * _ag_poa)

        # Need either real speed/handle or real defensive mobility to grade high.
        _ag_opp = 0.30 * _ag_spd + 0.28 * _ag_swb + 0.24 * _ag_poa + 0.18 * _ag_size
        _ag_gate = 1.0 if _ag_opp >= 0.22 else (0.56 + 0.44 * max(0.0, _ag_opp / 0.22))

        # Archetype floors.
        if is_guard and _ag_spd > 0.78 and _ag_swb > 0.72:
            _ag_raw = max(_ag_raw, 0.86)

        if is_guard and _ag_swb > 0.60 and height <= 75 and weight <= 200:
            _ag_raw = max(_ag_raw, 0.74)

        # Skilled compact creator guards can have elite change-of-direction even
        # without top-end straight-line speed.
        if is_guard and height <= 75 and weight <= 200 and _ag_swb > 0.52:
            _ag_raw = max(_ag_raw, 0.68)

        if is_wing and _ag_spd > 0.62 and _ag_poa > 0.74:
            _ag_raw = max(_ag_raw, 0.73)

        # High-usage scoring wings with real movement tools (Tatum lane).
        if is_wing and weight <= 220 and usage >= 28 and _ag_swb > 0.60:
            _ag_raw = max(_ag_raw, 0.60)

        if is_big and height <= 83 and weight <= 255 and _ag_poa > 0.60 and blk_per36 >= 1.20:
            _ag_raw = max(_ag_raw, 0.58)

        if is_big and height <= 83 and weight <= 255 and _ag_poa > 0.55 and blk_per36 >= 1.20:
            _ag_raw = max(_ag_raw, 0.52)

        _ag_value = 100.0 * _ag_raw * _ag_gate * (0.84 + 0.16 * _ag_reliability)
        if is_guard and height >= 79 and weight >= 220 and usage >= 28:
            _ag_value = min(_ag_value, 70.0 + 8.0 * _ag_swb + 8.0 * _ag_poa)
        if is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _ag_value = min(_ag_value, 60.0 + 10.0 * _ag_swb + 8.0 * _ag_poa)
        if is_big and height >= 83 and weight >= 250:
            _ag_value = min(_ag_value, 46.0 + 10.0 * _ag_poa + 8.0 * _ag_spd)
        if age >= 35 and height >= 78 and weight >= 220:
            _ag_value = min(_ag_value, 64.0 + 8.0 * _ag_swb + 8.0 * _ag_poa)

        # Final-value floors so archetype corrections survive gating/reliability.
        if is_guard and height <= 75 and weight <= 200 and usage < 24:
            _ag_value = max(_ag_value, 70.0)

        if is_guard and height <= 75 and weight <= 200 and attrs.get("ball_handle", 0.0) >= 58.0 and attrs.get("speed", 0.0) >= 48.0:
            _ag_value = max(_ag_value, 70.0)

        if is_wing and weight <= 220 and usage >= 28 and attrs.get("speed", 0.0) >= 42.0 and attrs.get("speed_with_ball", 0.0) >= 38.0:
            _ag_value = max(_ag_value, 60.0)

        if is_big and height <= 83 and weight <= 255 and _ag_poa > 0.55 and blk_per36 >= 1.20:
            _ag_value = max(_ag_value, 52.0)

        attrs["agility"] = max(0.0, min(100.0, _ag_value))

        # ── Strength ──────────────────────────────────────────────────
        # Strength = ability to hold/establish physical position, absorb contact,
        # and win body-up battles.
        _str_mass = self._norm(weight, 175, 285)
        _str_frame = (
            0.52 * self._norm(weight, 180, 290)
            + 0.48 * self._norm(height, 74, 85)
        )
        _str_glass = self._norm(oreb_pg + 0.55 * dreb_pg, 1.4, 9.0)
        _str_anchor = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _str_post = self._norm(attrs.get("post_control", 50.0), 35.0, 92.0)
        _str_draw = self._norm(attrs.get("draw_foul", 50.0), 40.0, 92.0)
        _str_sdunk = self._norm(attrs.get("standing_dunk", 50.0), 25.0, 95.0)
        _str_off_foul_drawn = self._norm(offensive_fouls_drawn_per36, 0.0, 0.35)
        _str_post_burden = self._norm(post_burden_p36, 0.15, 5.0)
        _str_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _str_contact = 0.36 * _str_anchor + 0.24 * _str_post + 0.16 * _str_draw + 0.14 * _str_sdunk + 0.05 * _str_off_foul_drawn + 0.05 * _str_post_burden

        if is_big:
            _str_raw = (
                0.36 * _str_mass
                + 0.20 * _str_frame
                + 0.18 * _str_glass
                + 0.18 * _str_contact
                + 0.08 * size_big
            )
        elif is_wing:
            _str_raw = (
                0.32 * _str_mass
                + 0.22 * _str_frame
                + 0.16 * _str_glass
                + 0.20 * _str_contact
                + 0.10 * size_big
            )
        else:  # guard
            _str_raw = (
                0.28 * _str_mass
                + 0.22 * _str_frame
                + 0.12 * _str_glass
                + 0.26 * _str_contact
                + 0.12 * self._norm(height, 73, 78)
            )

        _str_raw *= 1.08

        # Light guards without physical playstyle should not drift upward.
        if is_guard and weight < 190 and _str_draw < 0.45 and _str_post < 0.30:
            _str_raw *= 0.88 + 0.12 * _str_mass

        # Huge centers naturally maintain a very high strength floor.
        if is_center and weight >= 270:
            _str_raw = max(_str_raw, 0.86)

        # Physical combo bigs / wings.
        if (is_big or is_wing) and weight >= 240 and _str_anchor > 0.62:
            _str_raw = max(_str_raw, 0.76)

        # Powerful wings who consistently win contact and hold position.
        if is_wing and weight >= 225 and height >= 78 and (_str_draw > 0.40 or dreb_pg >= 5.0):
            _str_raw = max(_str_raw, 0.56)

        # Jumbo primary wings with real glass/size should sit above average.
        if is_wing and weight >= 230 and dreb_pg >= 6.0:
            _str_raw = max(_str_raw, 0.62)

        # Powerful guards (Jrue-style) with real frame and contact tolerance.
        if is_guard and weight >= 205 and _str_draw > 0.42:
            _str_raw = max(_str_raw, 0.58)

        if is_guard and weight >= 215:
            _str_raw = max(_str_raw, 0.50)

        _str_value = 100.0 * _str_raw * (0.84 + 0.16 * _str_reliability)
        attrs["strength"] = max(0.0, min(100.0, _str_value))

        # ── Vertical ──────────────────────────────────────────────────
        # Vertical = lower-body explosion for finishing, contesting, and lift.
        _vt_ddunk = self._norm(attrs.get("driving_dunk", 50.0), 30.0, 95.0)
        _vt_sdunk = self._norm(attrs.get("standing_dunk", 50.0), 25.0, 95.0)
        _vt_blk = self._norm(blk_per36, 0.10, 2.60)
        _vt_oreb = self._norm(oreb_pg, 0.30, 4.20)
        _vt_spd = self._norm(attrs.get("speed", 50.0), 20.0, 85.0)
        _vt_agi = self._norm(attrs.get("agility", 50.0), 20.0, 88.0)
        _vt_size = (
            0.62 * (1.0 - self._norm(weight, 180, 285))
            + 0.38 * (1.0 - self._norm(height, 74, 85))
        )
        _vt_rim = self._norm(rim_pressure, 0.16, 0.75)
        _vt_age = age_phys
        _vt_putback = self._norm(putback_rate, 0.0, 0.16)
        _vt_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _vt_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _vt_pop = 0.36 * _vt_ddunk + 0.22 * _vt_blk + 0.16 * _vt_oreb + 0.12 * _vt_rim + 0.08 * _vt_putback + 0.06 * _vt_recover
        _vt_mobility = 0.54 * _vt_spd + 0.46 * _vt_agi

        if is_guard:
            _vt_raw = (
                0.40 * _vt_pop
                + 0.32 * _vt_mobility
                + 0.18 * _vt_size
                + 0.10 * _vt_age
            )
        elif is_big:
            _vt_raw = (
                0.34 * _vt_pop
                + 0.20 * _vt_mobility
                + 0.18 * _vt_sdunk
                + 0.18 * _vt_oreb
                + 0.10 * _vt_age
            )
        else:  # wing
            _vt_raw = (
                0.40 * _vt_pop
                + 0.30 * _vt_mobility
                + 0.16 * _vt_size
                + 0.08 * _vt_sdunk
                + 0.06 * _vt_age
            )

        _vt_raw *= 1.08

        # Heavy ground-bound big drag.
        if is_big and weight >= 265 and _vt_blk < 0.40 and _vt_ddunk < 0.45:
            _vt_raw *= 0.82 + 0.18 * _vt_sdunk

        # High-weight non-bigs with modest explosion should not drift too high.
        if (not is_big) and weight >= 225 and _vt_ddunk < 0.60 and _vt_spd < 0.65:
            _vt_raw *= 0.90 + 0.10 * _vt_mobility

        # Archetype floors.
        if (not is_big) and _vt_ddunk >= 0.72 and _vt_spd > 0.72:
            _vt_raw = max(_vt_raw, 0.72)

        if is_wing and _vt_ddunk >= 0.66 and weight >= 235:
            _vt_raw = max(_vt_raw, 0.66)

        if is_wing and _vt_ddunk >= 0.72 and _vt_spd >= 0.70:
            _vt_raw = max(_vt_raw, 0.70)

        if is_wing and weight >= 270 and _vt_ddunk >= 0.72 and _vt_agi >= 0.62:
            _vt_raw = max(_vt_raw, 0.68)

        if is_wing and _vt_rim >= 0.40 and _vt_pop >= 0.58:
            _vt_raw = max(_vt_raw, 0.72)

        if is_wing and weight >= 235 and _vt_rim >= 0.36 and _vt_pop >= 0.54:
            _vt_raw = max(_vt_raw, 0.69)

        if is_wing and weight >= 240 and _vt_ddunk >= 0.60 and _vt_mobility >= 0.68:
            _vt_raw = max(_vt_raw, 0.64)

        if is_wing and weight >= 240 and rim_pressure >= 0.50 and blk_per36 >= 0.50:
            _vt_raw = max(_vt_raw, 0.60)

        if is_big and blk_per36 >= 1.70 and attrs.get("standing_dunk", 0.0) >= 70.0:
            _vt_raw = max(_vt_raw, 0.70)

        if is_big and blk_per36 >= 1.40 and oreb_pg >= 2.4:
            _vt_raw = max(_vt_raw, 0.64)

        if is_big and blk_per36 >= 1.80 and oreb_pg >= 2.4:
            _vt_raw = max(_vt_raw, 0.58)

        if is_big and blk_per36 >= 1.60 and attrs.get("standing_dunk", 0.0) >= 72.0:
            _vt_raw = max(_vt_raw, 0.56)

        if is_big and blk_per36 >= 1.80 and attrs.get("standing_dunk", 0.0) >= 72.0 and weight <= 255:
            _vt_raw = max(_vt_raw, 0.60)

        if is_big and blk_per36 >= 1.90 and oreb_pg >= 2.5 and rim_pressure >= 0.65 and weight <= 255:
            _vt_raw = max(_vt_raw, 0.58)

        if is_big and _vt_spd <= 0.38 and _vt_agi <= 0.38 and _vt_ddunk <= 0.55 and _vt_blk >= 0.55:
            _vt_raw = min(_vt_raw, 0.44 + 0.06 * _vt_sdunk + 0.08 * _vt_blk)

        if is_big and weight >= 275 and _vt_spd <= 0.46 and _vt_agi <= 0.30 and _vt_blk <= 0.42:
            _vt_raw = min(_vt_raw, 0.36 + 0.08 * _vt_sdunk + 0.04 * _vt_blk)

        _vt_value = 100.0 * _vt_raw * (0.84 + 0.16 * _vt_reliability)
        attrs["vertical"] = max(0.0, min(100.0, _vt_value))

        # ── Stamina ───────────────────────────────────────────────────
        # Stamina = sustainable game-to-game workload and motor.
        _st_minutes = self._norm(min_pg, 14.0, 37.0)
        _st_availability = self._norm(gp, 12.0, 72.0)
        _st_total_load = self._norm(min_pg * gp, 500.0, 2750.0)
        _st_usage_load = self._norm(usage, 16.0, 36.0)
        _st_transition = self._norm(raw_transition_p36, 0.5, 5.5)
        _st_cuts = self._norm(raw_cuts_p36, 0.2, 2.7)
        _st_roll = self._norm(raw_pnr_roll_p36, 0.0, 4.0)
        _st_rebound_motor = self._norm(oreb_pg + dreb_pg, 3.5, 13.0)
        _st_age = 1.0 - 0.22 * self._norm(age, 32.0, 40.0)
        _st_age = max(0.72, min(1.0, _st_age))
        _st_motor_events = motor_profile

        _st_motor = (
            0.28 * _st_transition
            + 0.20 * _st_cuts
            + 0.18 * _st_roll
            + 0.16 * _st_rebound_motor
            + 0.18 * _st_motor_events
        )
        _st_endurance_profile = 0.50 * _st_minutes + 0.32 * _st_availability + 0.18 * _st_motor
        _st_burden = _st_usage_load * (1.0 - _st_endurance_profile)

        if is_guard:
            _st_raw = (
                0.42 * _st_minutes
                + 0.13 * _st_availability
                + 0.22 * _st_total_load
                + 0.13 * _st_motor
                + 0.10 * _st_age
            )
        elif is_big:
            _st_raw = (
                0.40 * _st_minutes
                + 0.16 * _st_availability
                + 0.21 * _st_total_load
                + 0.16 * _st_motor
                + 0.07 * _st_age
            )
        else:  # wing
            _st_raw = (
                0.41 * _st_minutes
                + 0.14 * _st_availability
                + 0.21 * _st_total_load
                + 0.14 * _st_motor
                + 0.10 * _st_age
            )

        _st_raw -= 0.12 * _st_burden

        # High-workload engines: heavy on-ball stars who still sustain big minute loads.
        if usage >= 31.0 and min_pg >= 33.0 and gp >= 30:
            _st_raw = max(_st_raw, 0.80)

        if usage >= 30.0 and min_pg >= 31.5 and gp >= 28:
            _st_raw = max(_st_raw, 0.76)

        if transition_pos >= 3.5 and min_pg >= 32.0:
            _st_raw = max(_st_raw, 0.74)

        if is_big and (oreb_pg + dreb_pg) >= 10.0 and min_pg >= 30.0:
            _st_raw = max(_st_raw, 0.73)

        # Ironman minute-eaters should sit in top stamina lanes.
        if gp >= 60 and min_pg >= 34.0:
            _st_raw = max(_st_raw, 0.88)
        elif gp >= 52 and min_pg >= 31.0:
            _st_raw = max(_st_raw, 0.82)

        # Availability and minute constraints.
        if gp <= 10:
            _st_raw *= 0.80 + 0.20 * self._norm(min_pg, 16.0, 32.0)
        elif gp <= 18:
            _st_raw *= 0.88 + 0.12 * self._norm(min_pg, 18.0, 33.0)

        # Older stars can still grade well, but low availability should pull them down.
        if age >= 36 and gp < 50:
            _st_raw *= 0.94

        _st_reliability = 0.45 * self._norm(gp, 10.0, 72.0) + 0.55 * self._norm(min_pg, 14.0, 36.0)
        _st_value = 100.0 * _st_raw * (0.88 + 0.12 * _st_reliability)
        attrs["stamina"] = max(0.0, min(100.0, _st_value))

        # ══════════════════════════════════════════════════════════════
        # META
        # ══════════════════════════════════════════════════════════════

        # ── Intangibles ───────────────────────────────────────────────
        _int_maturity = self._maturity(age)
        _int_off = self._norm(attrs.get("offensive_consistency", 50.0), 55.0, 90.0)
        _int_def = self._norm(attrs.get("defensive_consistency", 50.0), 45.0, 90.0)
        _int_iq = (
            0.40 * self._norm(attrs.get("shot_iq", 50.0), 55.0, 95.0)
            + 0.30 * self._norm(attrs.get("pass_iq", 50.0), 55.0, 95.0)
            + 0.30 * self._norm(attrs.get("help_defense_iq", 50.0), 50.0, 95.0)
        )
        _int_reliability = 0.58 * self._norm(gp, 18.0, 82.0) + 0.42 * self._norm(min_pg, 14.0, 36.0)
        _int_stamina = self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)
        _int_raw = (
            0.24 * _int_maturity
            + 0.20 * _int_off
            + 0.18 * _int_def
            + 0.18 * _int_iq
            + 0.10 * _int_reliability
            + 0.10 * _int_stamina
        )
        attrs["intangibles"] = max(0.0, min(100.0, 100.0 * _int_raw))

        # ── Hustle ────────────────────────────────────────────────────
        _hs_transition = self._norm(raw_transition_p36, 0.5, 6.2)
        _hs_cuts = self._norm(raw_cuts_p36, 0.2, 3.0)
        _hs_roll = self._norm(raw_pnr_roll_p36, 0.0, 4.5)
        _hs_glass = self._norm(oreb_pg + 0.45 * dreb_pg, 2.5, 12.5)
        _hs_events = self._norm(stl_per36 + 0.85 * blk_per36, 0.8, 4.8)
        _hs_second = self._norm(second_chance_off_poss_rate, 0.03, 0.20)
        _hs_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _hs_engage = def_engagement
        _hs_raw = (
            0.16 * _hs_transition
            + 0.12 * _hs_cuts
            + 0.10 * _hs_roll
            + 0.18 * _hs_glass
            + 0.16 * _hs_events
            + 0.12 * _hs_second
            + 0.08 * _hs_recover
            + 0.08 * _hs_engage
        )
        _hs_raw = 0.08 + 0.92 * _hs_raw
        attrs["hustle"] = max(0.0, min(100.0, 100.0 * _hs_raw))

        # ── Overall Durability ────────────────────────────────────────
        _dur_availability = self._norm(gp, 12.0, 75.0)
        _dur_minutes = self._norm(min_pg, 12.0, 36.0)
        _dur_load = self._norm(gp * min_pg, 350.0, 2600.0)
        _dur_age = max(0.45, 1.0 - 0.05 * max(0.0, age - 31.0))
        _dur_strength = self._norm(attrs.get("strength", 50.0), 45.0, 95.0)
        _dur_stamina = self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)
        _dur_raw = (
            0.34 * _dur_availability
            + 0.18 * _dur_minutes
            + 0.20 * _dur_load
            + 0.12 * _dur_age
            + 0.08 * _dur_strength
            + 0.08 * _dur_stamina
        )
        attrs["overall_durability"] = max(0.0, min(100.0, 100.0 * _dur_raw))

        # ── Potential ─────────────────────────────────────────────────
        _pot_age = self._potential_from_age(age)
        _pot_work = 0.55 * self._norm(attrs.get("hustle", 50.0), 50.0, 95.0) + 0.45 * self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)
        _pot_foundation = (
            0.26 * self._norm(attrs.get("ball_handle", 50.0), 50.0, 92.0)
            + 0.24 * self._norm(attrs.get("three_point_shot", 50.0), 45.0, 95.0)
            + 0.24 * self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 95.0)
            + 0.26 * self._norm(attrs.get("interior_defense", 50.0), 40.0, 95.0)
        )
        _pot_raw = 0.62 * _pot_age + 0.20 * _pot_work + 0.18 * _pot_foundation
        if age <= 22 and _pot_foundation > 0.58:
            _pot_raw = max(_pot_raw, 0.84)
        elif age <= 25 and _pot_foundation > 0.64:
            _pot_raw = max(_pot_raw, 0.76)
        attrs["potential"] = max(0.0, min(100.0, 100.0 * _pot_raw))

        # ── Legacy archetype calibration ────────────────────────────
        # Older seasons often lack full possession-context coverage. Apply a
        # targeted archetype pass so elite drivers/defenders/shooters and
        # rim-only bigs land in realistic attribute bands.
        legacy_like = pbp_data_available < 0.5
        if legacy_like:
            ra_vol_p36 = self._f(f, "zone_fga_per36_ra", 0.0)
            perimeter_share = max(0.0, min(1.0, fg3a_rate + 0.35 * mid_rate))

            driver_signal = (
                0.34 * self._norm(rim_pressure, 0.16, 0.60)
                + 0.22 * self._norm(ra_vol_p36, 1.0, 7.2)
                + 0.20 * self._norm(raw_transition_p36, 0.4, 5.2)
                + 0.14 * self._norm(fta_rate, 0.14, 0.48)
                + 0.10 * self._norm(usage, 20.0, 35.0)
            )
            driver_signal = max(0.0, min(1.0, driver_signal))

            shooter_signal = (
                0.46 * self._norm(fg3a_rate, 0.24, 0.62)
                + 0.34 * self._norm(fg3_pct, 0.33, 0.43)
                + 0.20 * self._norm(raw_spot_up_p36 + raw_handoff_p36, 0.6, 6.0)
            )
            shooter_signal = max(0.0, min(1.0, shooter_signal))

            perimeter_def_signal = (
                0.42 * self._norm(stl_per36, 0.9, 2.2)
                + 0.24 * (1.0 - self._norm(pf_per36, 1.8, 4.8))
                + 0.20 * (1.0 - self._norm(usage, 20.0, 35.0))
                + 0.14 * self._norm(height, 75.0, 81.5)
            )
            perimeter_def_signal = max(0.0, min(1.0, perimeter_def_signal))

            rim_anchor_signal = (
                0.45 * self._norm(blk_per36, 0.9, 3.2)
                + 0.27 * self._norm(dreb_pg, 4.5, 11.5)
                + 0.18 * self._norm(oreb_pg, 1.0, 4.6)
                + 0.10 * self._norm(height, 80.0, 85.0)
            )
            rim_anchor_signal = max(0.0, min(1.0, rim_anchor_signal))

            rim_only_big_signal = (
                (1.0 if is_big else 0.0)
                * (1.0 - self._norm(fg3a_rate, 0.05, 0.20))
                * (1.0 - self._norm(mid_rate, 0.14, 0.34))
                * self._norm(rim_pressure + 0.55 * ra_vol_p36, 0.9, 4.4)
            )
            rim_only_big_signal = max(0.0, min(1.0, rim_only_big_signal))

            # Elite downhill creators (LeBron/Giannis archetype)
            if driver_signal > 0.56 and (not is_center):
                boost = (driver_signal - 0.56) / 0.44
                attrs["driving_dunk"] += 14.0 + 22.0 * boost
                attrs["driving_layup"] += 4.0 + 8.0 * boost
                attrs["speed_with_ball"] += 3.0 + 7.0 * boost
                attrs["draw_foul"] += 2.0 + 6.0 * boost

            # True legacy downhill stars should never land in low dunk bands.
            if (not is_big) and usage >= 29.0 and rim_pressure >= 0.30 and ra_vol_p36 >= 4.5:
                attrs["driving_dunk"] = max(attrs["driving_dunk"], 78.0)

            # Low-rim perimeter scorers should not carry inflated dunk ratings
            low_rim_signal = (
                0.45 * (1.0 - self._norm(rim_pressure, 0.12, 0.36))
                + 0.35 * (1.0 - self._norm(ra_vol_p36, 0.8, 4.0))
                + 0.20 * (1.0 - self._norm(raw_transition_p36, 0.4, 3.0))
            )
            low_rim_signal = max(0.0, min(1.0, low_rim_signal))
            if low_rim_signal > 0.62 and (not is_big):
                cut = (low_rim_signal - 0.62) / 0.38
                attrs["driving_dunk"] -= 8.0 + 10.0 * cut

            # High-volume perimeter guards/wings with weak downhill profile
            # should not retain high dunk ratings in legacy seasons.
            if (not is_big) and fg3a_rate >= 0.28 and driver_signal < 0.46:
                slope = max(0.0, min(1.0, (0.46 - driver_signal) / 0.46))
                attrs["driving_dunk"] -= 6.0 + 10.0 * slope

            # Elite shooting specialists
            if shooter_signal > 0.55:
                s_boost = (shooter_signal - 0.55) / 0.45
                attrs["three_point_shot"] += 4.0 + 9.0 * s_boost
                attrs["shot_iq"] += 2.0 + 5.0 * s_boost

            # Elite perimeter defenders
            if perimeter_def_signal > 0.60 and (not is_big):
                d_boost = (perimeter_def_signal - 0.60) / 0.40
                attrs["perimeter_defense"] += 4.0 + 9.0 * d_boost
                attrs["steal"] += 3.0 + 8.0 * d_boost
                attrs["pass_perception"] += 2.0 + 6.0 * d_boost
                attrs["help_defense_iq"] += 1.5 + 5.0 * d_boost

            # Rim-protecting anchor bigs
            if rim_anchor_signal > 0.58 and is_big:
                r_boost = (rim_anchor_signal - 0.58) / 0.42
                attrs["interior_defense"] += 3.0 + 9.0 * r_boost
                attrs["block"] += 4.0 + 10.0 * r_boost
                attrs["defensive_rebound"] += 2.0 + 7.0 * r_boost
                attrs["help_defense_iq"] += 2.0 + 6.0 * r_boost

            # Rim-only bigs: suppress perimeter touch inflation, strengthen paint game
            rim_only_center = (
                is_big
                and height >= 82.0
                and fg3a_rate <= 0.04
                and ft_pct <= 0.72
                and mid_rate <= 0.18  # low zone mid-rate is the cleaner signal
            )
            if rim_only_big_signal > 0.50 or rim_only_center:
                rb = (rim_only_big_signal - 0.50) / 0.50
                rb = max(0.0, min(1.0, rb))
                if rim_only_center:
                    rb = max(rb, 0.65)
                mid_cap = 36.0 + 10.0 * self._norm(ft_pct, 0.50, 0.82)
                attrs["mid_range_shot"] = min(attrs["mid_range_shot"], mid_cap)
                attrs["standing_dunk"] += 4.0 + 8.0 * rb
                attrs["close_shot"] += 3.0 + 7.0 * rb
                attrs["offensive_rebound"] += 2.0 + 7.0 * rb

            # Keep legacy-only pass within valid raw range before global scaling.
            for key in attrs:
                attrs[key] = max(0.0, min(100.0, attrs[key]))

        # ── Fill missing attrs with a neutral placeholder ──────────────
        for _attr in ATTRIBUTE_NAMES:
            if _attr not in attrs:
                attrs[_attr] = 50.0  # neutral until formula is written

        # ── Scale to 25–99 and clamp ──────────────────────────────────
        result: dict[str, int] = {}
        for attr in ATTRIBUTE_NAMES:
            raw = attrs.get(attr, 0.0)
            cal = _RAW_CALIBRATION.get(attr, 0.0)
            if cal > 0:
                # Taper positive corrections for high raw scores
                cal *= max(0.0, 1.0 - max(0.0, raw - 55) / 45)
            elif cal < 0:
                # Taper negative corrections for low raw scores
                cal *= max(0.0, 1.0 - max(0.0, 35 - raw) / 35)
            raw += cal
            # raw is 0–100 (weights sum to 100); map to 25–99
            frac = max(0.0, min(1.0, raw / 100.0))
            # Power-curve boost: stretches mid-range scores upward
            # so that raw 50 → ~60 attr, raw 70 → ~82 attr
            frac = frac ** 0.75
            scaled = 25 + frac * 74
            if attr == "offensive_consistency":
                attr_min, attr_max = 60, 90
            else:
                attr_min, attr_max = 25, 99
            result[attr] = max(attr_min, min(attr_max, round(scaled)))

        return result

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _norm(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    @staticmethod
    def _maturity(age: float) -> float:
        """Experience/maturity curve: peaks at 28–32."""
        if age < 22:
            return 0.3
        if age < 25:
            return 0.5 + 0.1 * (age - 22)
        if age < 28:
            return 0.8 + 0.067 * (age - 25)
        if age <= 32:
            return 1.0
        return max(0.6, 1.0 - 0.05 * (age - 32))

    @staticmethod
    def _potential_from_age(age: float) -> float:
        """Age-based potential (0–1 scale, mapped to 25–99 later)."""
        if age < 22:
            return 0.95 - 0.02 * max(0, age - 19)
        if age < 25:
            return 0.80 - 0.03 * (age - 22)
        if age < 28:
            return 0.65 - 0.05 * (age - 25)
        if age < 32:
            return 0.45 - 0.04 * (age - 28)
        return max(0.10, 0.30 - 0.04 * (age - 32))
