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
    # hands: calibration removed — archetype-aware formula + final bands
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
        if min_pg <= 0:
            min_pg = self._f(f, "minutes_per_game", 0)
        if min_pg <= 0 and gp > 0:
            total_min = self._f(f, "minutes", 0)
            if total_min <= 0:
                total_min = self._f(f, "min", 0)
            if total_min > 0:
                min_pg = total_min / max(gp, 1.0)

        # ── Stat inputs (will grow as we add each attribute formula) ──
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
        if assisted_3pt_pct > 1.0:
            assisted_3pt_pct /= 100.0
        catch_and_shoot_three_rate = self._f(f, "catch_and_shoot_three_rate", 0)
        pull_up_three_rate = self._f(f, "pull_up_three_rate", 0)
        unassisted_two_rate = self._f(f, "unassisted_two_rate", 0)
        putback_rate = self._f(f, "putback_rate", 0)
        live_ball_turnover_pct = self._f(f, "live_ball_turnover_pct", 0)
        shooting_fouls_drawn_pct = self._f(f, "shooting_fouls_drawn_pct", 0)
        three_pt_fouls_drawn_pct = self._f(f, "three_pt_fouls_drawn_pct", 0)
        seconds_per_poss_off = self._f(f, "seconds_per_poss_off", 0)
        second_chance_off_poss_rate = self._f(f, "second_chance_off_poss_rate", 0)
        off_poss = self._f(f, "off_poss", 0)
        bad_pass_turnovers_per36 = self._f(f, "bad_pass_turnovers_per36", 0)
        lost_ball_turnovers_per36 = self._f(f, "lost_ball_turnovers_per36", 0)
        offensive_fouls_per36 = self._f(f, "offensive_fouls_per36", 0)
        offensive_fouls_drawn_per36 = self._f(f, "offensive_fouls_drawn_per36", 0)
        loose_ball_fouls_per36 = self._f(f, "loose_ball_fouls_per36", 0)
        loose_ball_fouls_drawn_per36 = self._f(f, "loose_ball_fouls_drawn_per36", 0)
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
        # Driving layup = quality finishing on moving rim attacks.
        # Built from: rim conversion, downhill volume, self-creation burden,
        # contact control, touch, and sample reliability.
        _dl_ra_per36 = self._f(f, "zone_fga_per36_ra", 0.0)
        _dl_creation_p36 = 0.62 * iso_burden_p36 + 0.38 * pnr_bh_burden_p36
        _dl_transition = self._norm(raw_transition_p36, 0.35, 5.8)
        _dl_volume = 0.58 * self._norm(rim_pressure, 0.10, 0.52) + 0.42 * self._norm(_dl_ra_per36, 0.7, 7.2)
        _dl_efficiency = 0.72 * self._norm(ra_pct, 0.52, 0.79) + 0.28 * self._norm(ts_pct, 0.50, 0.66)
        _dl_creation = 0.66 * self._norm(_dl_creation_p36, 0.35, 8.8) + 0.34 * self._norm(unassisted_two_rate, 0.10, 0.62)
        _dl_contact = (
            0.50 * self._norm(fta_rate, 0.11, 0.48)
            + 0.30 * self._norm(shooting_fouls_drawn_pct, 0.03, 0.20)
            + 0.20 * (1.0 - self._norm(offensive_fouls_per36, 0.10, 1.40))
        )
        _dl_touch = 0.65 * self._norm(ft_pct, 0.58, 0.88) + 0.35 * self._norm(fg_pct, 0.42, 0.60)
        _dl_ball_security = 1.0 - self._norm(live_ball_turnover_pct, 0.24, 0.78)
        _dl_athletic = (
            0.46 * self._norm(raw_transition_p36, 0.5, 6.2)
            + 0.34 * self._norm(rim_pressure, 0.12, 0.52)
            + 0.20 * self._norm(shooting_fouls_drawn_pct, 0.03, 0.20)
        )
        _dl_craft = (
            0.44 * self._norm(unassisted_two_rate, 0.12, 0.62)
            + 0.32 * self._norm(ft_pct, 0.60, 0.90)
            + 0.24 * self._norm(ast_tov, 0.90, 3.20)
        )

        _dl_raw = (
            0.23 * _dl_efficiency
            + 0.20 * _dl_volume
            + 0.13 * _dl_creation
            + 0.13 * _dl_contact
            + 0.11 * _dl_touch
            + 0.10 * _dl_athletic
            + 0.06 * _dl_craft
            + 0.04 * _dl_ball_security
        )

        # Light role prior (not a hard position gate): guards/wings generally
        # create more moving layup reps than back-to-basket bigs.
        _dl_role_scale = {"PG": 1.02, "SG": 1.01, "SF": 1.00, "PF": 0.96, "C": 0.92}.get(pos, 1.00)
        _dl_raw *= _dl_role_scale

        # Opportunity gate: players with very low downhill/rim profile should
        # not land in elite layup bands.
        _dl_opp = 0.50 * self._norm(rim_pressure, 0.08, 0.34) + 0.30 * self._norm(_dl_ra_per36, 0.5, 4.8) + 0.20 * _dl_creation
        _dl_gate = 1.0 if _dl_opp >= 0.22 else (0.62 + 0.38 * max(0.0, _dl_opp / 0.22))

        # Sample reliability from minute load.
        _dl_reliability = 0.55 * self._norm(min_pg, 12.0, 35.0) + 0.45 * self._norm(gp, 20.0, 82.0)
        _dl_value = 100.0 * _dl_raw * _dl_gate * (0.84 + 0.16 * _dl_reliability)

        # Aging taper for non-explosive veteran profiles.
        if age >= 35 and _dl_athletic < 0.62:
            _dl_value *= 0.88 + 0.12 * _dl_athletic

        # Archetype floors/caps for cleaner separation.
        if (not is_big) and age <= 33 and rim_pressure >= 0.28 and ra_pct >= 0.61 and _dl_creation_p36 >= 2.4:
            _dl_value = max(_dl_value, 80.0)

        if (not is_big) and _dl_athletic >= 0.56 and _dl_contact >= 0.52 and _dl_volume >= 0.56:
            _dl_value = max(_dl_value, 77.0)

        # Modern high-usage slashers: keep strong downhill guards/wings from
        # dropping too low when efficiency is noisy year-to-year.
        if (not is_big) and age <= 30 and pts_pg >= 24.0 and rim_pressure >= 0.22 and _dl_creation >= 0.48:
            _dl_value = max(_dl_value, 76.0)

        if (not is_big) and _dl_craft >= 0.64 and _dl_efficiency >= 0.58 and _dl_creation >= 0.52 and rim_pressure >= 0.18:
            _dl_value = max(_dl_value, 74.0)

        # High-usage perimeter creators (Luka lane): strong craft + decision
        # load should not collapse to low layup bands even if vertical pop is
        # below elite slasher tiers.
        if (
            (not is_big)
            and age <= 31
            and usage >= 30.0
            and _dl_creation >= 0.60
            and _dl_craft >= 0.62
            and _dl_contact >= 0.50
            and rim_pressure >= 0.14
        ):
            _dl_value = max(_dl_value, 80.0)

        # Jumbo creator variant: bigger initiators with high touch/creation keep
        # a solid layup baseline even without top-end sprint profile.
        if (
            (not is_big)
            and age <= 32
            and height >= 78.0
            and usage >= 28.0
            and _dl_creation >= 0.62
            and _dl_touch >= 0.60
            and rim_pressure >= 0.13
        ):
            _dl_value = max(_dl_value, 81.0)

        # Heliocentric creator lane (Luka archetype): very high usage + passing
        # burden creators should not sit in low layup bands even when their rim
        # pressure is more craft/timing based than pure burst.
        if (
            (not is_big)
            and age <= 31
            and usage >= 31.0
            and ast_pg >= 7.0
            and _dl_creation >= 0.55
            and _dl_craft >= 0.58
        ):
            _dl_value = max(_dl_value, 69.0)

        # Hard ceiling for very late-prime veterans unless still clearly explosive.
        if age >= 38 and (not is_big) and _dl_athletic < 0.62:
            _dl_value = min(_dl_value, 79.0)

        if is_center and rim_pressure < 0.18 and _dl_creation_p36 < 1.6:
            _dl_value = min(_dl_value, 78.0)

        attrs["driving_layup"] = max(0.0, min(100.0, _dl_value))

        # ── Standing Dunk ─────────────────────────────────────────────
        # Standing dunk = finishing from a standstill (off seals, rolls, putbacks).
        # Guards should remain very low; PF/C should carry most of this skill.
        _sd_roll_p36 = pnr_roll * 36.0 / max(min_pg, 1.0)
        _sd_anchor_tools = (
            0.36 * self._norm(height, 78.0, 84.0)
            + 0.32 * self._norm(weight, 210.0, 280.0)
            + 0.20 * self._norm(oreb_pg, 0.6, 4.8)
            + 0.12 * self._norm(putback_rate, 0.0, 0.18)
        )
        _sd_role_usage = (
            0.34 * self._norm(_sd_roll_p36, 0.2, 8.0)
            + 0.26 * self._norm(ra_rate, 0.08, 0.62)
            + 0.22 * self._norm(second_chance_off_poss_rate, 0.03, 0.22)
            + 0.18 * self._norm(self._f(f, "zone_fga_per36_ra", 0.0), 0.4, 7.5)
        )
        _sd_touch = 0.62 * self._norm(ra_pct, 0.54, 0.78) + 0.38 * self._norm(ft_pct, 0.52, 0.82)
        _sd_perimeter_penalty = 1.0 - 0.34 * self._norm(fg3a_rate, 0.18, 0.65)

        _sd_core = (
            0.47 * _sd_anchor_tools
            + 0.39 * _sd_role_usage
            + 0.14 * _sd_touch
        )
        _sd_core = max(0.0, min(1.0, _sd_core * _sd_perimeter_penalty))

        # Position bands are on raw 0-100 scale before global 25-99 mapping.
        # This keeps guards in the 25-40 final range and centers/power bigs high.
        _sd_bands: dict[str, tuple[float, float]] = {
            "PG": (0.0, 8.0),
            "SG": (1.0, 14.0),
            "SF": (6.0, 40.0),
            "PF": (22.0, 82.0),
            "C": (30.0, 99.0),
        }
        _sd_min, _sd_max = _sd_bands.get(pos, (6.0, 42.0))
        _sd_value = _sd_min + (_sd_max - _sd_min) * _sd_core

        # Guards almost never use standstill dunks in real game context.
        if pos == "PG":
            _sd_value = min(_sd_value, 8.0)
        elif pos == "SG":
            _sd_value = min(_sd_value, 14.0)

        # Non-bigs with low paint role stay clearly low.
        if (not is_big) and _sd_role_usage < 0.34:
            _sd_value = min(_sd_value, 12.0)

        # True roll/putback bigs should not land in weak standing-dunk bands.
        if is_big and _sd_roll_p36 >= 2.0 and oreb_pg >= 1.8 and ra_rate >= 0.24:
            _sd_value = max(_sd_value, 55.0)

        if is_center and height >= 82.0 and weight >= 240.0 and ra_rate >= 0.30:
            _sd_value = max(_sd_value, 68.0)

        # Tall SF exception (Giannis lane): very tall wings can carry stronger
        # standstill dunk profiles, but keep the ceiling below true big bands.
        if pos == "SF" and height >= 82.0 and ra_rate >= 0.24:
            _sf_tall_floor = 54.0 + 6.0 * self._norm(oreb_pg, 0.8, 3.2)
            _sd_value = max(_sd_value, _sf_tall_floor)
            _sd_value = min(_sd_value, 62.0)

        attrs["standing_dunk"] = max(0.0, min(100.0, _sd_value))

        # ── Driving Dunk ──────────────────────────────────────────────
        # Driving dunk = moving dunk finishing with speed/momentum and
        # posterization threat vs contact at the rim.
        _dd_ra_per36 = self._f(f, "zone_fga_per36_ra", 0.0)
        _dd_creation_p36 = 0.62 * iso_burden_p36 + 0.38 * pnr_bh_burden_p36

        _dd_momentum = (
            0.46 * self._norm(raw_transition_p36, 0.35, 5.8)
            + 0.34 * self._norm(rim_pressure, 0.12, 0.56)
            + 0.20 * self._norm(_dd_ra_per36, 0.6, 7.8)
        )
        _dd_attack_load = (
            0.44 * self._norm(_dd_creation_p36, 0.6, 10.5)
            + 0.34 * self._norm(unassisted_two_rate, 0.10, 0.60)
            + 0.22 * self._norm(usage, 16.0, 36.0)
        )
        _dd_contact = (
            0.46 * self._norm(fta_rate, 0.12, 0.50)
            + 0.34 * self._norm(shooting_fouls_drawn_pct, 0.03, 0.20)
            + 0.20 * self._norm(ra_pct, 0.56, 0.78)
        )

        _dd_size = 0.55 * self._norm(height, 75.0, 84.0) + 0.45 * self._norm(weight, 180.0, 265.0)
        _dd_elastic_burst = (
            0.52 * age_phys
            + 0.30 * (1.0 - self._norm(weight, 185.0, 270.0))
            + 0.18 * (1.0 - self._norm(height, 76.0, 84.0))
        )
        _dd_elastic_burst = max(0.0, min(1.0, _dd_elastic_burst))
        _dd_pop = max(0.0, min(1.0, 0.58 * _dd_size + 0.42 * self._norm(putback_rate, 0.0, 0.14)))

        # Poster proxy: attack load + contact appetite + pop through traffic.
        _dd_poster = max(0.0, min(1.0, 0.40 * _dd_contact + 0.34 * _dd_pop + 0.26 * _dd_attack_load))

        _dd_raw = (
            0.30 * _dd_momentum
            + 0.19 * _dd_attack_load
            + 0.17 * _dd_contact
            + 0.14 * _dd_poster
            + 0.11 * _dd_elastic_burst
            + 0.09 * self._norm(ra_pct, 0.56, 0.78)
        )

        # Light role prior (not a hard gate): guards can still be elite,
        # but PF/C generally convert momentum dunks less frequently.
        _dd_role_scale = {"PG": 1.02, "SG": 1.00, "SF": 0.99, "PF": 0.95, "C": 0.90}.get(pos, 1.00)
        _dd_raw *= _dd_role_scale

        # Opportunity gate: no real downhill/rim role means no elite driving dunk.
        _dd_opp = 0.42 * self._norm(rim_pressure, 0.10, 0.38) + 0.30 * self._norm(_dd_ra_per36, 0.6, 5.2) + 0.28 * self._norm(raw_transition_p36, 0.30, 4.2)
        _dd_gate = 1.0 if _dd_opp >= 0.26 else (0.56 + 0.44 * max(0.0, _dd_opp / 0.26))

        _dd_reliability = 0.55 * self._norm(min_pg, 12.0, 35.0) + 0.45 * self._norm(gp, 20.0, 82.0)
        _dd_value = 100.0 * _dd_raw * _dd_gate * (0.84 + 0.16 * _dd_reliability)

        _dd_explosive_signal = max(0.0, min(1.0, 0.42 * _dd_momentum + 0.34 * _dd_pop + 0.24 * _dd_poster))

        # Archetype lanes.
        if (not is_big) and _dd_momentum >= 0.64 and _dd_poster >= 0.60 and _dd_contact >= 0.56:
            _dd_value = max(_dd_value, 76.0)

        if (not is_big) and _dd_momentum >= 0.72 and _dd_poster >= 0.68 and _dd_attack_load >= 0.60 and _dd_pop >= 0.62:
            _dd_value = max(_dd_value, 82.0)

        # True elite poster guards/wings (Ja/Ant lane).
        if (not is_big) and age <= 29 and _dd_momentum >= 0.76 and _dd_poster >= 0.70 and _dd_pop >= 0.66:
            _dd_value = max(_dd_value, 86.0)

        if is_guard and age <= 28 and _dd_momentum >= 0.80 and _dd_poster >= 0.72 and _dd_contact >= 0.60:
            _dd_value = max(_dd_value, 88.0)

        # Young explosive guards/wings (Ja/Ant lane).
        if is_guard and age <= 26 and _dd_momentum >= 0.74 and _dd_poster >= 0.66 and _dd_pop >= 0.64:
            _dd_value = max(_dd_value, 85.0)

        if (not is_big) and age <= 25 and rim_pressure >= 0.24 and _dd_contact >= 0.56 and _dd_momentum >= 0.70:
            _dd_value = max(_dd_value, 83.0)

        # Very low downhill role should suppress guard inflation.
        if is_guard and _dd_opp < 0.16:
            _dd_value = min(_dd_value, 46.0)

        # Craft-heavy / low-pop guards should not grade as elite dunkers.
        if is_guard and (_dd_pop < 0.56 or _dd_explosive_signal < 0.56):
            _dd_value = min(_dd_value, 72.0)

        # Guard non-dunker suppression: low rim pressure + low RA volume +
        # limited transition burst should sit in clearly low bands.
        if is_guard and rim_pressure < 0.16 and _dd_ra_per36 < 2.2 and raw_transition_p36 < 1.8:
            _dd_value = min(_dd_value, 46.0)

        if is_guard and rim_pressure < 0.14 and _dd_ra_per36 < 1.8:
            _dd_value = min(_dd_value, 40.0)

        if is_guard and age >= 33 and (_dd_pop < 0.60 or raw_transition_p36 < 1.8):
            _dd_value = min(_dd_value, 64.0)

        # High-usage creators with weaker poster profile (Luka lane) should sit
        # in solid but non-elite bands.
        if is_guard and usage >= 30.0 and _dd_poster < 0.58 and _dd_pop < 0.60:
            _dd_value = min(_dd_value, 70.0)

        if is_guard and usage >= 30.0 and _dd_poster < 0.56 and _dd_momentum < 0.66:
            _dd_value = min(_dd_value, 66.0)

        # Hard creator cap: heliocentric guards with modest downhill pressure
        # should not sit in poster-dunker bands.
        if is_guard and usage >= 30.0 and rim_pressure < 0.23 and raw_transition_p36 < 2.8:
            _dd_value = min(_dd_value, 68.0)

        # Clear guard non-dunker cap (Curry lane).
        if is_guard and rim_pressure < 0.18 and _dd_ra_per36 < 2.4 and raw_transition_p36 < 2.2:
            _dd_value = min(_dd_value, 52.0)

        if is_guard and rim_pressure < 0.14 and _dd_ra_per36 < 1.8:
            _dd_value = min(_dd_value, 45.0)

        # Older non-big players should need clear explosive profile to retain
        # high driving dunk ratings.
        if (not is_big) and age >= 36 and (_dd_explosive_signal < 0.66 or _dd_momentum < 0.70):
            _dd_value = min(_dd_value, 74.0)

        if (not is_big) and age >= 38 and (_dd_explosive_signal < 0.72 or _dd_momentum < 0.72):
            _dd_value = min(_dd_value, 70.0)

        # Traditional centers with low open-floor role should not become elite
        # driving dunkers from size alone.
        if is_center and raw_transition_p36 < 0.9 and _dd_creation_p36 < 1.6:
            _dd_value = min(_dd_value, 62.0)

        # Bigs without clear momentum/poster profile should not sit in high bands.
        if is_big and _dd_explosive_signal < 0.56:
            _dd_value = min(_dd_value, 74.0)

        if is_center and _dd_explosive_signal < 0.54:
            _dd_value = min(_dd_value, 58.0)

        if is_center and rim_pressure < 0.22 and raw_transition_p36 < 1.4:
            _dd_value = min(_dd_value, 56.0)

        # Benchmark realism gates (explicit profile separation).
        # Young explosive high-pressure guards should sit in clear elite bands.
        if is_guard and age <= 25 and usage >= 30.0 and ra_rate >= 0.28 and fta_rate >= 0.32:
            _dd_value = max(_dd_value, 82.0)

        if is_guard and age <= 24 and usage >= 30.0 and ra_rate >= 0.20 and fta_rate >= 0.30:
            _dd_value = max(_dd_value, 80.0)

        # High-usage low-rim creators (Luka lane) should not sit in elite bands.
        if is_guard and usage >= 30.0 and ra_rate < 0.17:
            _dd_value = min(_dd_value, 55.0)

        # Older non-explosive guards should remain in low-to-mid bands.
        if is_guard and age >= 35:
            _dd_value = min(_dd_value, 45.0)

        # Very old wings without extreme rim pressure should not remain elite.
        if (not is_guard) and (not is_big) and age >= 38 and ra_rate < 0.34:
            _dd_value = min(_dd_value, 58.0)

        # Centers need dominant rim pressure to enter high driving-dunk tiers.
        if is_center and ra_rate < 0.36:
            _dd_value = min(_dd_value, 55.0)

        attrs["driving_dunk"] = max(0.0, min(100.0, _dd_value))

        # ── Close Shot ────────────────────────────────────────────────
        # Close shot = short-range touch around the basket without requiring
        # downhill drive speed. Prioritize post/touch craft, especially for bigs.
        _cs_touch = (
            0.40 * self._norm(ft_pct, 0.58, 0.90)
            + 0.22 * self._norm(ts_pct, 0.49, 0.66)
            + 0.20 * self._norm(paint_pct, 0.38, 0.64)
            + 0.18 * self._norm(ra_pct, 0.56, 0.79)
        )
        _cs_post_skill = (
            0.46 * self._norm(post_burden_p36, 0.30, 6.2)
            + 0.34 * self._norm(post_ppp, 0.72, 1.16)
            + 0.20 * self._norm(height, 77.0, 84.0)
        )
        _cs_inside_presence = (
            0.38 * self._norm(paint_rate, 0.10, 0.34)
            + 0.30 * self._norm(ra_rate, 0.08, 0.34)
            + 0.22 * self._norm(self._f(f, "zone_fga_per36_ra", 0), 0.5, 6.5)
            + 0.10 * self._norm(oreb_pg, 0.30, 4.0)
        )

        # Penalize profiles that rely heavily on drive-first creation rather
        # than close-touch/post skill.
        _cs_drive_dependency = (
            0.56 * self._norm(raw_transition_p36, 0.8, 6.0)
            + 0.44 * self._norm(unassisted_two_rate, 0.20, 0.70)
        )

        _cs_raw = (
            0.42 * _cs_touch
            + 0.32 * _cs_post_skill
            + 0.20 * _cs_inside_presence
            + 0.06 * self._norm(fta_rate, 0.10, 0.45)
        )

        if _cs_drive_dependency > 0.62 and _cs_post_skill < 0.46:
            _cut = max(0.0, min(1.0, (_cs_drive_dependency - 0.62) / 0.38))
            _cs_raw -= 0.07 + 0.10 * _cut

        # Position profile: bigs and post scorers should grade higher, but
        # guards with elite touch can still be solid.
        _CS_POS_SCALE = {"PG": 0.92, "SG": 0.95, "SF": 0.99, "PF": 1.06, "C": 1.13}
        _cs_pos_scale = _CS_POS_SCALE.get(pos, 1.00)

        _cs_opp = 0.44 * _cs_post_skill + 0.34 * _cs_inside_presence + 0.22 * self._norm(paint_pct, 0.38, 0.64)
        _cs_gate = 1.0 if _cs_opp >= 0.22 else (0.72 + 0.28 * max(0.0, _cs_opp / 0.22))

        _cs_value = 100.0 * _cs_raw * _cs_pos_scale * _cs_gate

        # True post bigs should have reliably strong close-touch bands.
        if is_big and post_burden_p36 >= 2.4 and post_ppp >= 0.90:
            _cs_value = max(_cs_value, 70.0)
        if is_center and post_burden_p36 >= 3.4 and post_ppp >= 0.96:
            _cs_value = max(_cs_value, 74.0)

        # Rim-runner centers with limited touch/post craft should not sit in
        # elite close-shot tiers from size/volume alone.
        if is_center and ft_pct < 0.72 and post_ppp < 0.92 and post_burden_p36 < 2.2:
            _cs_value = min(_cs_value, 73.0)

        # Non-big interior scorers with elite short-area efficiency can still
        # carry strong close-touch ratings even with modest post volume.
        if (not is_big) and age <= 27 and paint_rate >= 0.22 and ra_rate >= 0.16 and paint_pct >= 0.56:
            _cs_value = max(_cs_value, 69.0)

        # Elite non-big interior finishers (Zion lane): very high rim share +
        # credible post efficiency should push above generic guard/wing bands.
        if (not is_big) and age <= 27 and ra_rate >= 0.40 and post_burden_p36 >= 2.0 and post_ppp >= 0.95:
            _cs_value = max(_cs_value, 66.0)

        if (not is_big) and age <= 27 and ra_rate >= 0.48 and post_burden_p36 >= 3.0 and post_ppp >= 0.98:
            _cs_value = max(_cs_value, 68.0)

        # Guards can be solid, but non-interior guards should not grade like post bigs.
        if is_guard and paint_rate < 0.18 and ra_rate < 0.10:
            _cs_value = min(_cs_value, 57.0)
        if is_guard and ft_pct >= 0.84 and paint_pct >= 0.50:
            _cs_value = max(_cs_value, 59.0)

        attrs["close_shot"] = max(0.0, min(100.0, _cs_value))

        # ══════════════════════════════════════════════════════════════
        # SHOOTING
        # ══════════════════════════════════════════════════════════════

        # ── Three-Point Shot ──────────────────────────────────────────
        # Corrected model: lower floor, wider elite separation, and no heavy
        # Three-point shot = volume-backed accuracy across catch-and-shoot,
        # off-the-dribble, and corner contexts.
        _tp_attempts_pg = max(0.0, fg3a_pg)
        _tp_attempts_total = max(0.0, _tp_attempts_pg * gp)
        _tp_rel = self._norm(_tp_attempts_total, 45.0, 520.0)

        # Regress tiny samples so 50% on ~1 attempt does not grade elite.
        _tp_baseline = 0.335 if is_big else 0.350
        _tp_pct_reg = _tp_rel * fg3_pct + (1.0 - _tp_rel) * _tp_baseline

        _tp_acc = self._norm(_tp_pct_reg, 0.29, 0.44)
        _tp_vol_pg = self._norm(_tp_attempts_pg, 1.0, 11.0)
        _tp_vol_p36 = self._norm(fg3a_per36, 1.0, 13.0)

        _tp_cs = self._norm(catch_and_shoot_three_rate, 0.04, 0.68)
        _tp_pull = self._norm(pull_up_three_rate, 0.0, 0.36)

        _tp_corner_rate = self._f(f, "zone_fga_rate_corner3_left", 0.0) + self._f(f, "zone_fga_rate_corner3_right", 0.0)
        _tp_corner_pct = 0.5 * (
            self._f(f, "zone_fg_pct_corner3_left", fg3_pct)
            + self._f(f, "zone_fg_pct_corner3_right", fg3_pct)
        )
        _tp_corner = (
            0.55 * self._norm(_tp_corner_rate, 0.03, 0.20)
            + 0.45 * self._norm(_tp_corner_pct, 0.31, 0.46)
        )

        _tp_pressure = self._norm(
            0.44 * pull_up_three_rate
            + 0.34 * (usage / 100.0)
            + 0.22 * ft_pct,
            0.20,
            0.60,
        )

        _tp_raw = (
            0.42 * _tp_acc
            + 0.28 * _tp_vol_pg
            + 0.10 * _tp_vol_p36
            + 0.08 * _tp_cs
            + 0.06 * _tp_pull
            + 0.03 * _tp_corner
            + 0.03 * _tp_pressure
        )

        # Creation bonus for high-volume pull-up shooters.
        if _tp_attempts_pg >= 5.5 and _tp_pull >= 0.45 and _tp_pct_reg >= 0.36:
            _tp_raw += 0.03 + 0.03 * self._norm(_tp_attempts_pg, 5.5, 10.5)

        if _tp_pct_reg >= 0.39 and _tp_attempts_pg >= 5.0:
            _tp_raw += 0.04

        if is_big and _tp_pct_reg >= 0.39 and _tp_attempts_pg >= 3.8:
            _tp_raw += 0.03

        # Spot-up only shooters need real volume to stay elite.
        if _tp_pull < 0.10 and _tp_cs >= 0.60 and _tp_attempts_pg < 3.2:
            _tp_raw -= 0.02

        _TP_POS_SCALE = {"PG": 1.03, "SG": 1.04, "SF": 1.02, "PF": 0.96, "C": 0.90}
        _tp_pos_scale = _TP_POS_SCALE.get(pos, 1.00)

        # Strong volume gate: low-attempt profiles cannot enter elite tiers.
        _tp_vol_gate = 1.0 if _tp_attempts_pg >= 2.5 else (0.62 + 0.38 * (_tp_attempts_pg / 2.5))

        _tp_touch_gate = 1.0 if _tp_pct_reg >= 0.335 else (0.68 + 0.32 * self._norm(_tp_pct_reg, 0.28, 0.335))

        _tp_value = 100.0 * max(0.0, min(1.0, _tp_raw)) * _tp_pos_scale * _tp_vol_gate * _tp_touch_gate

        # Big-man context: 35% is meaningful for bigs only with real volume.
        if is_big and _tp_attempts_pg >= 2.3 and _tp_pct_reg >= 0.35:
            _tp_value = max(_tp_value, 62.0)

        # Volume-backed efficiency floors.
        if (not is_big) and _tp_attempts_pg >= 5.5 and _tp_pct_reg >= 0.40:
            _tp_value = max(_tp_value, 76.0)
        elif (not is_big) and _tp_attempts_pg >= 5.0 and _tp_pct_reg >= 0.38:
            _tp_value = max(_tp_value, 72.0)
        elif (not is_big) and _tp_attempts_pg >= 4.5 and _tp_pct_reg >= 0.36:
            _tp_value = max(_tp_value, 67.0)

        if is_big and _tp_attempts_pg >= 3.8 and _tp_pct_reg >= 0.38:
            _tp_value = max(_tp_value, 61.0)
        elif is_big and _tp_attempts_pg >= 3.0 and _tp_pct_reg >= 0.35:
            _tp_value = max(_tp_value, 56.0)

        # Decent stretch-big floor (Bam lane): usable 3PT volume with
        # acceptable percentage and touch should land in low-70s final bands.
        if is_big and _tp_attempts_pg >= 1.4 and _tp_pct_reg >= 0.30 and ft_pct >= 0.74:
            _tp_value = max(_tp_value, 55.0)

        # Decent stretch-big lanes: reward PF/C shooters with usable volume
        # and acceptable percentage, without treating them like guard snipers.
        if is_big and _tp_attempts_pg >= 1.0 and _tp_pct_reg >= 0.32 and ft_pct >= 0.70:
            _tp_value = max(_tp_value, 30.0)
        if is_big and _tp_attempts_pg >= 1.6 and _tp_pct_reg >= 0.34 and ft_pct >= 0.72:
            _tp_value = max(_tp_value, 34.0)

        # Upper-middle stability: meaningful-volume non-big shooters with
        # solid efficiency should not collapse into low tiers.
        if (not is_big) and _tp_attempts_pg >= 4.2 and _tp_pct_reg >= 0.355:
            _tp_value = max(_tp_value, 66.0)
        if (not is_big) and _tp_attempts_pg >= 5.0 and _tp_pct_reg >= 0.375:
            _tp_value = max(_tp_value, 70.0)

        # Specialist marksmen (Kennard lane): elite efficiency on real volume
        # should grade higher even when most attempts are catch-and-shoot.
        if (not is_big) and _tp_attempts_pg >= 4.8 and _tp_pct_reg >= 0.405 and pull_up_three_rate <= 0.18:
            _tp_value = max(_tp_value, 74.0)

        # High-volume role shooters (Smart lane): solid efficiency on real
        # volume with mostly catch-and-shoot usage should sit around low 70s.
        if (
            (not is_big)
            and _tp_attempts_pg >= 4.4
            and _tp_pct_reg >= 0.335
            and ft_pct >= 0.74
            and pull_up_three_rate <= 0.22
            and catch_and_shoot_three_rate >= 0.18
        ):
            _tp_value = max(_tp_value, 55.0)

        # Ultra-elite high-volume shooters can sit at the very top.
        if (not is_big) and _tp_attempts_pg >= 10.0 and _tp_pct_reg >= 0.39:
            _tp_value = max(_tp_value, 93.0)

        # Tiny-volume hard cap.
        if _tp_attempts_pg < 1.5:
            _tp_value = min(_tp_value, 62.0)

        # Bigs rarely shoot high-volume 3s; non-stretch bigs should remain low.
        if is_center and _tp_attempts_pg < 1.8:
            _tp_value = min(_tp_value, 52.0)

        # Small global loosen pass so the full pool lands slightly higher.
        _tp_value += 2.5

        attrs["three_point_shot"] = max(0.0, min(100.0, _tp_value))

        # ── Mid-Range Shot ────────────────────────────────────────────
        # Mid-range = inside-the-arc, outside-the-paint jumpers with strong
        # volume + shot-creation value (contested/off-dribble profile).
        _mr_attempts_pg = mid_rate * fga_pg
        _mr_attempts_total = max(0.0, _mr_attempts_pg * gp)
        _mr_rel = self._norm(_mr_attempts_total, 28.0, 210.0)

        # Regress tiny samples toward neutral mid efficiency so 1-shot players
        # cannot grade as elite just from percentage noise.
        _mr_pct_reg = _mr_rel * mid_pct + (1.0 - _mr_rel) * 0.41

        _mr_pct = self._norm(_mr_pct_reg, 0.36, 0.53)
        _mr_volume = self._norm(_mr_attempts_pg, 0.7, 5.6)
        _mr_rate = self._norm(mid_rate, 0.06, 0.34)
        _mr_touch = self._norm(0.58 * _mr_pct_reg + 0.30 * ft_pct + 0.12 * fg3_pct, 0.40, 0.60)

        _mr_creation_load = self._norm(0.56 * creator_two_p36 + 0.30 * iso_burden_p36 + 0.14 * pnr_bh_burden_p36, 0.8, 10.8)
        _mr_self = self._norm(unassisted_two_rate, 0.14, 0.60)
        _mr_contested = self._norm(0.65 * shooting_fouls_drawn_pct + 0.35 * usage / 100.0, 0.09, 0.27)
        _mr_off_dribble = max(0.0, min(1.0, 0.52 * _mr_creation_load + 0.33 * _mr_self + 0.15 * _mr_contested))

        _mr_raw = (
            0.30 * _mr_pct
            + 0.26 * _mr_volume
            + 0.14 * _mr_rate
            + 0.16 * _mr_off_dribble
            + 0.14 * _mr_touch
        )

        # Creation reward: contested/self-created mid profiles should outrank
        # spot-up-only wings with similar raw percentages.
        if _mr_off_dribble >= 0.62 and _mr_attempts_pg >= 2.3:
            _mr_raw += 0.02 + 0.03 * self._norm(_mr_pct_reg, 0.40, 0.50)

        # Spot-up heavy dampener for low-creation profiles.
        _mr_spot = self._norm(raw_spot_up_p36 + raw_handoff_p36, 0.8, 6.0)
        if _mr_spot >= 0.62 and _mr_off_dribble < 0.48:
            _cut = self._norm(_mr_spot - _mr_off_dribble, 0.10, 0.55)
            _mr_raw -= 0.05 + 0.08 * _cut

        # Position profile: wings/guards often carry stronger jumper packages;
        # true centers get a smaller baseline unless volume/skill supports it.
        _MR_POS_SCALE = {"PG": 1.03, "SG": 1.05, "SF": 1.05, "PF": 0.98, "C": 0.78}
        _mr_pos_scale = _MR_POS_SCALE.get(pos, 1.00)

        # Strong volume gate: low-frequency mid shooters should not reach elite bands.
        _mr_vol_gate = 1.0 if _mr_attempts_pg >= 2.0 else (0.60 + 0.40 * max(0.0, _mr_attempts_pg / 2.0))

        # Touch gate: keep very poor shooting touch from inflating mid rating.
        _mr_shoot = 0.48 * self._norm(_mr_pct_reg, 0.37, 0.51) + 0.34 * self._norm(ft_pct, 0.62, 0.88) + 0.18 * self._norm(fg3_pct, 0.28, 0.40)
        _mr_touch_gate = 1.0 if _mr_shoot >= 0.46 else (0.68 + 0.32 * max(0.0, _mr_shoot / 0.46))

        _mr_value = 100.0 * max(0.0, min(1.0, _mr_raw)) * _mr_pos_scale * _mr_vol_gate * _mr_touch_gate

        # Compress extreme top-end clustering so elite creators separate
        # naturally instead of collapsing to many 95-99 outcomes.
        if _mr_value > 82.0:
            _mr_value = 82.0 + 0.55 * (_mr_value - 82.0)

        # Elite creator floor (booker/kd/kawhi lane): high volume + strong
        # self-creation + strong regressed efficiency.
        if _mr_attempts_pg >= 3.0 and _mr_off_dribble >= 0.62 and _mr_pct_reg >= 0.44:
            _mr_value = max(_mr_value, 74.0)

        # Tiny-volume cap: strong % on very low volume should remain capped.
        if _mr_attempts_pg < 1.2:
            _mr_value = min(_mr_value, 63.0)

        if _mr_attempts_pg < 2.0:
            _mr_value = min(_mr_value, 72.0)

        # Non-shooter cap: rim-only/limited-touch players should not sit high.
        if fg3a_rate < 0.14 and fg3_pct < 0.34 and _mr_off_dribble < 0.45:
            non_shooter_mid_cap = 50.0 + 16.0 * self._norm(ft_pct, 0.55, 0.82) + 10.0 * self._norm(_mr_pct_reg, 0.37, 0.49)
            _mr_value = min(_mr_value, non_shooter_mid_cap)

        # Guard moderation: avoid over-inflated top-end bands for guards whose
        # mid profile is more efficiency-driven than true shot-creation heavy.
        if is_guard:
            if _mr_off_dribble < 0.22 and unassisted_two_rate < 0.35 and usage < 24.0:
                _mr_value = min(_mr_value, 76.0)

            if _mr_off_dribble < 0.32 and unassisted_two_rate < 0.46 and shooting_fouls_drawn_pct < 0.12:
                _mr_value = min(_mr_value, 80.0)

            if usage < 24.0 and unassisted_two_rate < 0.33 and shooting_fouls_drawn_pct < 0.10:
                _mr_value = min(_mr_value, 77.0)

            if fg3_pct < 0.34 and mid_rate > 0.40:
                _mr_value = min(_mr_value, 79.0)

            if _mr_value > 91.0 and fg3_pct < 0.38 and _mr_pct_reg < 0.505:
                _mr_value = min(_mr_value, 83.0)

        attrs["mid_range_shot"] = max(0.0, min(100.0, _mr_value))

        # ── Free Throw ────────────────────────────────────────────────
        # Free throw is driven primarily by FT% bands, with only light
        # sample/volume adjustments so lower-attempt elite shooters can still
        # grade in elite tiers.
        _ft_attempts_total = max(0.0, fta_pg * gp)
        _ft_rel = self._norm(_ft_attempts_total, 25.0, 260.0)

        # Regress toward neutral league-average FT touch on low samples.
        _ft_pct_regressed = _ft_rel * ft_pct + (1.0 - _ft_rel) * 0.765

        # Keep the anchor mostly tied to observed FT%, with regression as a
        # stabilizer (not a hard override).
        _ft_anchor = 0.72 * _ft_pct_regressed + 0.28 * ft_pct

        if _ft_anchor >= 0.90:
            _ft_value = 90.0 + 9.0 * self._norm(_ft_anchor, 0.90, 0.95)
        elif _ft_anchor >= 0.80:
            _ft_value = 85.0 + 5.0 * self._norm(_ft_anchor, 0.80, 0.90)
        elif _ft_anchor >= 0.70:
            _ft_value = 75.0 + 10.0 * self._norm(_ft_anchor, 0.70, 0.80)
        elif _ft_anchor >= 0.60:
            _ft_value = 62.0 + 8.0 * self._norm(_ft_anchor, 0.60, 0.70)
        else:
            _ft_value = 40.0 + 30.0 * self._norm(_ft_anchor, 0.40, 0.60)

        # Light volume/context adjustment only.
        _ft_value += 1.6 * self._norm(fta_pg, 0.2, 9.5)

        # Missing-data fallback: avoid giving a strong neutral FT grade when
        # there is effectively no usable free-throw sample.
        if _ft_attempts_total < 1.0 and ft_pct <= 0.0:
            _ft_value = 60.0

        # Tiny-sample guardrails: do not destroy elite banding, only prevent
        # extreme top-end from very sparse samples.
        if _ft_attempts_total < 15.0:
            _ft_value = min(_ft_value, 94.0)
        elif _ft_attempts_total < 35.0:
            _ft_value = min(_ft_value, 96.0)

        # Reliable elite FT shooters should not be dragged down if accuracy
        # and sample size both clearly support it.
        if _ft_attempts_total >= 120.0 and _ft_anchor >= 0.90:
            _ft_value = max(_ft_value, 90.0)
        elif _ft_attempts_total >= 80.0 and _ft_anchor >= 0.86:
            _ft_value = max(_ft_value, 86.0)

        attrs["free_throw"] = max(0.0, min(100.0, _ft_value))

        # ── Shot IQ ───────────────────────────────────────────────────
        # Shot IQ (simple model):
        # - Low-usage players should sit in 90-93 when they take clean shots.
        # - High-usage stars should generally sit in low/mid-80s.
        _siq_ts = self._norm(ts_pct, 0.50, 0.68)
        _siq_efg = self._norm(efg_pct, 0.46, 0.63)
        _siq_fg3 = self._norm(fg3_pct, 0.27, 0.43)
        _siq_mid = self._norm(mid_pct, 0.34, 0.52)
        _siq_ft = self._norm(ft_pct, 0.60, 0.90)
        _siq_shoot_skill = 0.42 * _siq_fg3 + 0.28 * _siq_mid + 0.30 * _siq_ft
        _siq_play = 1.0 - self._norm(tov_pct, 0.08, 0.20)
        _siq_open_profile = (
            0.38 * self._norm(catch_and_shoot_three_rate, 0.05, 0.70)
            + 0.34 * self._norm(assisted_3pt_pct, 0.45, 0.90)
            + 0.28 * self._norm(assisted_2pt_pct, 0.35, 0.80)
        )
        _siq_tough_profile = (
            0.55 * self._norm(pull_up_three_rate, 0.03, 0.38)
            + 0.45 * self._norm(unassisted_two_rate, 0.16, 0.58)
        )
        _siq_good_offense = 0.58 * _siq_shoot_skill + 0.42 * _siq_ts
        _siq_decision = 0.44 * _siq_play + 0.32 * _siq_open_profile + 0.24 * _siq_good_offense
        _siq_tough_control = 1.0 - self._norm(_siq_tough_profile, 0.28, 0.78)

        if usage <= 20.0:
            _siq_raw = 0.84 + 0.05 * (0.70 * _siq_decision + 0.30 * _siq_tough_control)
            _siq_raw = max(0.84, min(0.89, _siq_raw))
        elif usage <= 24.0:
            _siq_raw = 0.81 + 0.05 * (0.65 * _siq_decision + 0.35 * _siq_tough_control)
            _siq_raw = max(0.81, min(0.87, _siq_raw))
        elif usage <= 28.0:
            _siq_raw = 0.74 + 0.04 * (0.65 * _siq_decision + 0.35 * _siq_tough_control)
            _siq_raw = max(0.72, min(0.79, _siq_raw))
        else:
            _siq_raw = 0.69 + 0.06 * (0.60 * _siq_decision + 0.40 * _siq_tough_control)
            _siq_raw = max(0.67, min(0.76, _siq_raw))

        # Non-shooter dampener: if a player brings very little perimeter/mid
        # skill, cap their ceiling even if they finish efficiently inside.
        _siq_non_shooter = 0.60 * _siq_fg3 + 0.40 * _siq_mid
        if usage >= 24.0 and _siq_non_shooter < 0.25:
            _siq_raw *= 0.82 + 0.18 * (_siq_non_shooter / 0.25)

        attrs["shot_iq"] = max(0.0, min(100.0, 100.0 * _siq_raw))

        # ══════════════════════════════════════════════════════════════
        # POST GAME
        # ══════════════════════════════════════════════════════════════

        # ── Post Hook ─────────────────────────────────────────────────
        # Post hook = making hook shots from the post (touch + hook skill +
        # real post opportunity).
        _ph_post = self._norm(post_burden_p36, 0.15, 5.6)
        _ph_ppp = self._norm(post_ppp, 0.72, 1.10)
        _ph_size = 0.50 * self._norm(height, 78, 84) + 0.50 * self._norm(weight, 210, 285)
        _ph_touch = self._norm(0.55 * paint_pct + 0.25 * ra_pct + 0.20 * ft_pct, 0.44, 0.70)
        _ph_close = self._norm(attrs["close_shot"], 52.0, 92.0)
        _ph_hook_tendency = self._norm(
            0.50 * (float(t.get("post_hook_left", 25)) + float(t.get("post_hook_right", 25)))
            + 0.35 * float(t.get("shoot_from_post", 25)),
            24.0,
            86.0,
        )

        _ph_raw = (
            0.28 * _ph_post
            + 0.18 * _ph_hook_tendency
            + 0.18 * _ph_touch
            + 0.16 * _ph_size
            + 0.12 * _ph_close
            + 0.08 * _ph_ppp
        )

        _PH_POS_SCALE = {"PG": 0.12, "SG": 0.20, "SF": 0.45, "PF": 0.80, "C": 1.00}
        _ph_pos_scale = _PH_POS_SCALE.get(pos, 0.45)
        if (not is_big) and height >= 79 and weight >= 220 and usage >= 25 and _ph_post >= 0.22:
            _ph_pos_scale = max(_ph_pos_scale, 0.46)

        _ph_opp = 0.62 * _ph_post + 0.38 * (_ph_size * _ph_hook_tendency)
        _ph_gate = 1.0 if _ph_opp >= 0.24 else (0.42 + 0.58 * max(0.0, _ph_opp / 0.24))

        attrs["post_hook"] = max(0.0, min(100.0, 100.0 * _ph_raw * _ph_pos_scale * _ph_gate))

        # ── Post Fade ─────────────────────────────────────────────────
        # Post fade = making fadeaways from the post (mid touch + fade craft +
        # real post opportunity).
        _pf_post = self._norm(post_burden_p36, 0.12, 5.2)
        _pf_ppp = self._norm(post_ppp, 0.72, 1.10)
        _pf_mid_touch = self._norm(0.60 * mid_pct + 0.25 * ft_pct + 0.15 * fg3_pct, 0.38, 0.60)
        _pf_mid_attr = self._norm(attrs["mid_range_shot"], 50.0, 92.0)
        _pf_close_attr = self._norm(attrs["close_shot"], 55.0, 94.0)
        _pf_fade_tendency = self._norm(
            0.50 * (float(t.get("post_fade_left", 25)) + float(t.get("post_fade_right", 25)))
            + 0.30 * float(t.get("post_face_up", 25)),
            22.0,
            86.0,
        )
        _pf_self_creation = self._norm(0.58 * unassisted_two_rate + 0.42 * self._norm(iso_burden_p36, 0.5, 8.5), 0.18, 0.78)
        _pf_size = 0.45 * self._norm(height, 77, 84) + 0.55 * self._norm(weight, 200, 275)

        _pf_raw = (
            0.22 * _pf_post
            + 0.24 * _pf_mid_touch
            + 0.22 * _pf_mid_attr
            + 0.04 * _pf_close_attr
            + 0.15 * _pf_fade_tendency
            + 0.09 * _pf_self_creation
            + 0.03 * _pf_ppp
            + 0.01 * _pf_size
        )

        _PF_POS_SCALE = {"PG": 0.18, "SG": 0.38, "SF": 0.80, "PF": 0.98, "C": 0.74}
        _pf_pos_scale = _PF_POS_SCALE.get(pos, 0.72)
        if (not is_big) and height >= 79 and weight >= 220 and usage >= 25 and _pf_post >= 0.20:
            _pf_pos_scale = max(_pf_pos_scale, 0.56)

        _pf_opp = 0.58 * _pf_post + 0.42 * (_pf_mid_touch * _pf_fade_tendency)
        _pf_gate = 1.0 if _pf_opp >= 0.22 else (0.42 + 0.58 * max(0.0, _pf_opp / 0.22))

        attrs["post_fade"] = max(0.0, min(100.0, 100.0 * _pf_raw * _pf_pos_scale * _pf_gate))

        # ── Post Control ──────────────────────────────────────────────
        # Post control = operating from the post (creating and holding deep
        # position, forcing reactions, and making basic read/passes).
        _pc_post = self._norm(post_burden_p36, 0.18, 5.6)
        _pc_ppp = self._norm(post_ppp, 0.72, 1.10)
        _pc_leverage = 0.42 * self._norm(height, 78, 84) + 0.58 * self._norm(weight, 210, 290)
        _pc_strength = 0.60 * self._norm(weight, 210, 290) + 0.40 * size_big
        _pc_security = 1.0 - self._norm(tov_pct, 0.08, 0.19)
        _pc_playmaking = self._norm(0.58 * ast_pg + 0.42 * ast_tov, 1.4, 9.2)
        _pc_create_tendency = self._norm(
            0.34 * float(t.get("post_drive", 25))
            + 0.26 * float(t.get("post_spin", 25))
            + 0.20 * float(t.get("post_back_down", 25))
            + 0.20 * float(t.get("post_face_up", 25)),
            20.0,
            84.0,
        )
        _pc_draw = self._norm(0.65 * shooting_fouls_drawn_pct + 0.35 * fta_rate, 0.03, 0.26)

        _pc_raw = (
            0.20 * _pc_post
            + 0.18 * _pc_leverage
            + 0.20 * _pc_strength
            + 0.12 * _pc_create_tendency
            + 0.08 * _pc_security
            + 0.18 * _pc_playmaking
            + 0.03 * _pc_ppp
            + 0.01 * _pc_draw
        )

        # Strong PF/C archetypes should not collapse due to limited passing.
        _pc_big_body = 0.55 * _pc_strength + 0.45 * _pc_leverage
        if pos in {"PF", "C", "F-C", "C-F"} and _pc_big_body >= 0.62:
            _pc_raw = max(_pc_raw, 0.38 + 0.28 * _pc_big_body + 0.10 * _pc_post)

        _PC_POS_SCALE = {"PG": 0.16, "SG": 0.26, "SF": 0.58, "PF": 0.90, "C": 1.00}
        _pc_pos_scale = _PC_POS_SCALE.get(pos, 0.58)
        if pos in {"PF", "C", "F-C", "C-F"}:
            _pc_pos_scale *= 1.00 + 0.06 * _pc_strength
        if (not is_big) and height >= 79 and weight >= 220 and usage >= 25 and _pc_post >= 0.20:
            _pc_pos_scale = max(_pc_pos_scale, 0.62)

        _pc_opp = 0.60 * _pc_post + 0.40 * (_pc_leverage * (0.55 + 0.45 * _pc_create_tendency))
        _pc_gate = 1.0 if _pc_opp >= 0.24 else (0.42 + 0.58 * max(0.0, _pc_opp / 0.24))

        attrs["post_control"] = max(0.0, min(100.0, 100.0 * _pc_raw * _pc_pos_scale * _pc_gate))

        # ══════════════════════════════════════════════════════════════
        # PLAYMAKING
        # ══════════════════════════════════════════════════════════════

        # ── Draw Foul ─────────────────────────────────────────────────
        # Draw foul = primarily FT rate balanced by usage.
        # High FT rate + high usage = elite, high FT rate + low usage = lower.
        _df_fta_rate = self._norm(fta_rate, 0.10, 0.62)
        _df_fta_p36 = self._norm(fta_per36, 1.6, 11.5)
        _df_usage = self._norm(usage, 10.0, 35.0)
        _df_inter = _df_fta_rate * _df_usage

        _df_base = 0.72 * _df_fta_rate + 0.28 * _df_fta_p36
        _df_raw = _df_base * (0.55 + 0.45 * _df_usage) + 0.10 * _df_usage + 0.10 * _df_inter

        # High FT-rate role finishers should grade as good, not elite.
        if _df_usage < 0.30:
            _df_raw *= 0.62 + 0.38 * (_df_usage / 0.30)

        _df_opp = 0.60 * _df_fta_rate + 0.30 * _df_usage + 0.10 * _df_inter
        _df_gate = 1.0 if _df_opp >= 0.20 else (0.62 + 0.38 * max(0.0, _df_opp / 0.20))

        if _df_fta_rate >= 0.80 and _df_usage >= 0.62:
            _df_raw = max(_df_raw, 0.78 + 0.14 * _df_fta_rate)

        _df_score = 100.0 * _df_raw * _df_gate + 2.0
        attrs["draw_foul"] = max(0.0, min(95.0, _df_score))

        # ── Ball Handle ───────────────────────────────────────────────
        # Ball handle = live dribble control + ability to run offense on-ball.
        # Use security, on-ball load, passing control, and position archetype.
        _bh_usage = self._norm(usage, 12.0, 35.0)
        _bh_creator_load = self._norm(0.58 * iso_burden_p36 + 0.42 * pnr_bh_burden_p36, 0.20, 8.50)
        _bh_unassisted_bucket = self._norm(0.72 * unassisted_two_rate + 0.28 * pull_up_three_rate, 0.16, 0.72)

        _bh_tov = 1.0 - self._norm(tov_pct, 0.09, 0.19)
        _bh_tov_p36 = 1.0 - self._norm(tov_per36, 1.4, 4.8)
        _bh_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)
        _bh_lost = 1.0 - self._norm(lost_ball_turnovers_per36, 0.20, 3.20)
        _bh_security = 0.40 * _bh_tov + 0.25 * _bh_tov_p36 + 0.20 * _bh_live + 0.15 * _bh_lost

        _bh_ast_tov = self._norm(ast_tov, 0.85, 3.40)
        _bh_ast_p36 = self._norm(ast_pg * 36.0 / max(min_pg, 1.0), 1.8, 9.0)
        _bh_pass_control = 0.70 * _bh_ast_tov + 0.30 * _bh_ast_p36

        _BH_ARCHETYPE = {
            "PG": 1.00,
            "SG": 0.92,
            "SF": 0.80,
            "PF": 0.64,
            "C": 0.48,
            "G": 0.96,
            "F": 0.76,
            "F-G": 0.88,
            "G-F": 0.92,
            "F-C": 0.62,
            "C-F": 0.56,
        }
        _bh_arch = _BH_ARCHETYPE.get(pos, 0.78)

        _bh_onball_load = 0.50 * _bh_creator_load + 0.30 * _bh_usage + 0.20 * _bh_unassisted_bucket

        _bh_raw = (
            0.35 * _bh_security
            + 0.30 * _bh_onball_load
            + 0.20 * _bh_pass_control
            + 0.15 * _bh_arch
        )

        # Need genuine on-ball role to sit in elite ball-handle bands.
        _bh_opp = 0.52 * _bh_onball_load + 0.28 * _bh_security + 0.20 * _bh_arch
        _bh_gate = 1.0 if _bh_opp >= 0.18 else (0.52 + 0.48 * max(0.0, _bh_opp / 0.18))

        # High-usage lead creators should not be heavily gate-suppressed.
        if pos in {"PG", "SG", "G", "G-F", "F-G"} and usage >= 30.0 and _bh_arch >= 0.88:
            _bh_gate = max(_bh_gate, 0.90 + 0.08 * _bh_usage)

        # Bigs with low on-ball burden should not jump into top tiers from passing alone.
        if is_big and _bh_onball_load < 0.45:
            _bh_raw *= 0.70 + 0.30 * (_bh_onball_load / 0.45)

        # High-creation guard archetypes need realistic handle floors.
        if pos in {"PG", "SG", "G", "G-F", "F-G"} and _bh_security >= 0.40:
            _bh_guard_creator = 0.55 * _bh_usage + 0.30 * _bh_unassisted_bucket + 0.15 * _bh_arch
            if _bh_guard_creator >= 0.56:
                _bh_creator_floor = 0.66 + 0.14 * _bh_guard_creator + 0.06 * _bh_creator_load
                _bh_raw = max(_bh_raw, _bh_creator_floor)

        # Heliocentric creator floor: heavy on-ball stars should still carry
        # strong handle ratings even with elevated turnover burden.
        if pos in {"PG", "SG", "G", "G-F", "F-G"} and usage >= 30.0 and (_bh_creator_load >= 0.45 or _bh_unassisted_bucket >= 0.48):
            _bh_heli_floor = 0.64 + 0.12 * _bh_creator_load + 0.06 * _bh_arch + 0.04 * _bh_usage
            _bh_raw = max(_bh_raw, _bh_heli_floor)

        # Final guard-creator floor for top-usage lead handlers.
        if pos in {"PG", "SG", "G", "G-F", "F-G"} and usage >= 28.0 and _bh_onball_load >= 0.45:
            _bh_top_guard_floor = 0.68 + 0.08 * _bh_arch + 0.08 * _bh_usage
            _bh_raw = max(_bh_raw, _bh_top_guard_floor)

        attrs["ball_handle"] = max(0.0, min(100.0, 100.0 * _bh_raw * _bh_gate))

        # ── Speed With Ball ───────────────────────────────────────────
        # Speed with ball = burst out of a standstill + pace while dribbling.
        _swb_transition = self._norm(raw_transition_p36, 0.30, 5.80)
        _swb_iso = self._norm(iso_burden_p36, 0.10, 5.20)
        _swb_pnr = self._norm(pnr_bh_burden_p36, 0.20, 8.20)
        _swb_creation = 0.56 * _swb_iso + 0.44 * _swb_pnr
        _swb_handle = self._norm(attrs["ball_handle"], 54.0, 93.0)
        _swb_security = ball_security
        _swb_usage = self._norm(usage, 12.0, 35.0)
        _swb_pace = 1.0 - self._norm(seconds_per_poss_off, 14.0, 20.0)
        _swb_size = 0.60 * (1.0 - self._norm(height, 74.0, 84.0)) + 0.40 * (1.0 - self._norm(weight, 185.0, 270.0))
        _swb_age = age_phys

        _swb_launch = 0.44 * _swb_handle + 0.36 * _swb_creation + 0.20 * _swb_size
        _swb_dribble_pace = 0.42 * _swb_transition + 0.30 * _swb_pace + 0.18 * _swb_usage + 0.10 * _swb_security
        _swb_physical = 0.72 * _swb_size + 0.28 * _swb_age

        _swb_raw = (
            0.38 * _swb_launch
            + 0.34 * _swb_dribble_pace
            + 0.17 * _swb_creation
            + 0.07 * _swb_physical
            + 0.04 * _swb_usage
        )

        # Bigs need enough handle + on-ball role to avoid inflated SWB.
        if size_big > 0.75:
            _swb_big_drag = 0.62 * _swb_handle + 0.38 * _swb_creation
            _swb_raw *= 0.56 + 0.44 * _swb_big_drag

        # Require some dribble role to reach higher tiers.
        _swb_opp = 0.42 * _swb_creation + 0.28 * _swb_handle + 0.20 * _swb_usage + 0.10 * _swb_transition
        _swb_gate = 1.0 if _swb_opp >= 0.20 else (0.48 + 0.52 * max(0.0, _swb_opp / 0.20))

        attrs["speed_with_ball"] = max(0.0, min(100.0, 100.0 * _swb_raw * _swb_gate))

        # ── Hands ─────────────────────────────────────────────────────
        # Hands = how reliably a player catches/controls the ball, especially
        # on quick actions (cuts, rolls, handoffs, spot-ups, lob catches).
        _h_roll_p36 = raw_pnr_roll_p36
        _h_cuts_p36 = raw_cuts_p36
        _h_spot_p36 = raw_spot_up_p36
        _h_handoff_p36 = raw_handoff_p36
        _h_transition_p36 = raw_transition_p36

        _h_roll = self._norm(_h_roll_p36, 0.05, 8.0)
        _h_cuts = self._norm(_h_cuts_p36, 0.05, 4.2)
        _h_spot = self._norm(_h_spot_p36, 0.30, 7.0)
        _h_handoff = self._norm(_h_handoff_p36, 0.05, 3.0)
        _h_transition = self._norm(_h_transition_p36, 0.30, 6.20)

        _h_tov = 1.0 - self._norm(tov_pct, 0.10, 0.19)
        _h_ast_tov = self._norm(ast_tov, 0.80, 3.00)
        _h_handle = self._norm(attrs["ball_handle"], 50.0, 92.0)
        _h_close = self._norm(attrs["close_shot"], 60.0, 95.0)
        _h_layup = self._norm(attrs["driving_layup"], 62.0, 95.0)
        _h_sdunk = self._norm(attrs["standing_dunk"], 25.0, 95.0)
        _h_ddunk = self._norm(attrs["driving_dunk"], 30.0, 95.0)
        _h_reb = self._norm(oreb_pg + 0.65 * dreb_pg, 2.5, 13.0)
        _h_lost = 1.0 - self._norm(lost_ball_turnovers_per36, 0.15, 3.20)
        _h_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _h_receiver = receiver_skill
        _h_live = 1.0 - self._norm(live_ball_turnover_pct, 0.22, 0.70)
        _h_bad_pass = 1.0 - self._norm(bad_pass_turnovers_per36, 0.10, 3.00)
        _h_touch_load = self._norm(possessions_used_per36, 8.0, 30.0)
        _h_usage = self._norm(usage, 14.0, 36.0)

        _h_catch_role = 0.31 * _h_roll + 0.24 * _h_cuts + 0.21 * _h_spot + 0.16 * _h_handoff + 0.08 * _h_transition
        _h_lob_context = self._norm(_h_roll_p36 + 0.55 * _h_cuts_p36 + 0.40 * _h_transition_p36, 0.40, 9.00)
        _h_flash_context = self._norm(_h_handoff_p36 + 0.45 * raw_pnr_bh_p36 + 0.25 * raw_iso_p36, 0.30, 8.20)

        # 40% Security
        _h_security = 0.34 * _h_tov + 0.26 * _h_lost + 0.20 * _h_live + 0.12 * _h_bad_pass + 0.08 * _h_ast_tov

        # 25% Reception quality
        _h_reception = 0.36 * _h_receiver + 0.24 * _h_catch_role + 0.18 * _h_recover + 0.12 * _h_lob_context + 0.10 * _h_flash_context

        # 20% Traffic control
        _h_catch_finish = 0.26 * _h_close + 0.20 * _h_layup + 0.27 * _h_sdunk + 0.27 * _h_ddunk
        _h_traffic = 0.36 * _h_reb + 0.28 * _h_roll + 0.22 * _h_catch_finish + 0.14 * size_big

        # 15% Touch reliability (touch burden is opportunity, not direct free boost).
        _h_touch_reliability = 0.60 * _h_handle + 0.25 * _h_touch_load + 0.15 * _h_ast_tov

        _h_raw = (
            0.40 * _h_security
            + 0.25 * _h_reception
            + 0.20 * _h_traffic
            + 0.15 * _h_touch_reliability
        )

        if _h_roll > 0.45 and _h_lob_context > 0.50 and _h_reception > 0.52:
            _h_raw *= 1.04

        if size_big > 0.65:
            _h_big_catch = 0.45 * _h_roll + 0.30 * _h_sdunk + 0.25 * _h_lob_context
            _h_raw *= 0.85 + 0.50 * _h_big_catch

        # Very low security should prevent high hands even with role volume.
        if _h_security < 0.35:
            _h_raw *= 0.82 + 0.18 * (_h_security / 0.35)

        # Light global normalization into NBA/2K hands ranges.
        _h_raw *= 1.10

        # Usage/touches are opportunity gates, not direct positive multipliers.
        _h_opp = 0.45 * _h_catch_role + 0.30 * _h_touch_load + 0.25 * _h_usage
        _h_gate = 1.0 if _h_opp >= 0.20 else (0.60 + 0.40 * max(0.0, _h_opp / 0.20))

        _h_value = 100.0 * _h_raw * _h_gate

        # Global lift to avoid compression in low-60s.
        _h_value += 4.0

        # Elite handler/receiver lane.
        if (is_guard or is_wing) and _h_touch_reliability >= 0.62 and _h_reception >= 0.56:
            _h_value = max(
                _h_value,
                67.0 + 9.0 * _h_touch_reliability + 8.0 * _h_reception,
            )

        # Creator reception floor for top on-ball guards/wings.
        if (is_guard or is_wing) and _h_touch_load >= 0.62 and _h_reception >= 0.54 and _h_security >= 0.50:
            _h_value = max(
                _h_value,
                69.0 + 8.0 * _h_reception + 6.0 * _h_touch_reliability,
            )

        # Roll-finisher big lane.
        if is_big and _h_roll >= 0.50 and _h_lob_context >= 0.54 and _h_catch_finish >= 0.56 and (_h_big_skill >= 0.56 or _h_touch_reliability >= 0.58):
            _h_value = max(
                _h_value,
                68.0 + 10.0 * _h_lob_context + 7.0 * _h_catch_finish,
            )

        # Skilled-big lane (Jokic archetype): passing-touch + control should
        # produce the top hands tier among bigs.
        _h_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _h_big_skill = (
            0.34 * _h_touch_reliability
            + 0.24 * _h_reception
            + 0.24 * self._norm(_h_ast_p36, 1.6, 10.0)
            + 0.18 * self._norm(post_burden_p36 + 0.55 * raw_handoff_p36, 0.20, 6.0)
        )
        if is_big and _h_big_skill >= 0.60 and _h_security >= 0.46:
            _h_value = max(
                _h_value,
                72.0 + 10.0 * (_h_big_skill - 0.60) / 0.40,
            )

        # Strong veteran wing lane (Kawhi/LeBron archetype).
        _h_wing_power = 0.36 * _h_security + 0.28 * _h_reception + 0.20 * _h_touch_reliability + 0.16 * _h_traffic
        if is_wing and weight >= 215 and _h_wing_power >= 0.56:
            _h_value = max(
                _h_value,
                69.0 + 9.0 * (_h_wing_power - 0.56) / 0.44,
            )

        if is_wing and (weight >= 210 or age >= 30.0) and _h_security >= 0.50 and _h_reception >= 0.50:
            _h_wing_floor_signal = 0.50 * _h_security + 0.30 * _h_reception + 0.20 * _h_touch_reliability
            _h_value = max(
                _h_value,
                70.0 + 8.0 * max(0.0, (_h_wing_floor_signal - 0.50) / 0.50),
            )

        if is_wing and usage >= 24.0 and _h_touch_reliability >= 0.56 and _h_security >= 0.44:
            _h_value = max(
                _h_value,
                69.0 + 7.0 * max(0.0, (_h_touch_reliability - 0.56) / 0.44),
            )

        # Guards should rate high, but typically a touch below elite hands wings/bigs.
        if is_guard:
            _h_guard_cap = 75.0 + 4.0 * _h_reception + 3.0 * _h_touch_reliability
            _h_value = min(_h_value, _h_guard_cap)

        # Defensive-only big lane (Gobert archetype): keep in mid hands band.
        _h_def_big_mid = (
            0.42 * _h_traffic
            + 0.28 * _h_reception
            + 0.18 * _h_security
            + 0.12 * _h_touch_reliability
        )
        if is_big and _h_touch_reliability < 0.52 and _h_big_skill < 0.56:
            _h_value = min(_h_value, 60.0 + 9.0 * _h_def_big_mid)
            _h_value = max(_h_value, 58.0 + 7.0 * _h_def_big_mid)

        if is_big and _h_big_skill < 0.56:
            _h_value = min(_h_value, 63.0 + 7.0 * _h_traffic + 4.0 * _h_reception)
            if _h_ast_p36 < 4.0:
                _h_value = min(_h_value, 68.0)

        # Non-playmaking bigs should sit in mid hands lanes, not elite.
        if is_big and _h_ast_p36 < 4.0 and _h_touch_reliability < 0.62:
            _h_value = min(_h_value, 67.0)
            _h_value = max(_h_value, 60.0 + 6.0 * _h_traffic)

        attrs["hands"] = max(0.0, min(100.0, _h_value))

        # ── Pass Accuracy ─────────────────────────────────────────────
        # Pass accuracy = physical delivery quality (velocity, angle, target).
        # Keep this mostly execution-driven and only lightly tied to raw assist totals.
        _pa_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _pa_pnr_p36 = pnr_bh_burden_p36
        _pa_handoff_p36 = raw_handoff_p36
        _pa_post_p36 = post_burden_p36
        _pa_transition_p36 = raw_transition_p36

        _pa_ast = self._norm(_pa_ast_p36, 2.0, 12.5)
        _pa_ast_tov = self._norm(ast_tov, 0.85, 3.50)
        _pa_tov = 1.0 - self._norm(tov_pct, 0.09, 0.20)
        _pa_bad_pass = 1.0 - self._norm(bad_pass_turnovers_per36, 0.10, 3.00)
        _pa_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)
        _pa_handle = self._norm(attrs["ball_handle"], 55.0, 92.0)
        _pa_hands = self._norm(attrs["hands"], 62.0, 95.0)
        _pa_usage = self._norm(usage, 14.0, 35.0)
        _pa_transition = self._norm(_pa_transition_p36, 0.35, 5.8)
        _pa_creator_two = self._norm(creator_two_p36, 0.40, 8.0)
        _pa_variety = (
            0.45 * self._norm(_pa_pnr_p36, 0.20, 10.0)
            + 0.25 * self._norm(_pa_handoff_p36, 0.05, 3.6)
            + 0.20 * self._norm(_pa_post_p36, 0.05, 4.8)
            + 0.10 * _pa_transition
        )
        _pa_variety = max(_pa_variety, 0.64 * _pa_creator_two + 0.36 * _pa_ast)

        _pa_delivery = 0.36 * _pa_bad_pass + 0.24 * _pa_ast_tov + 0.20 * _pa_live + 0.20 * _pa_tov
        _pa_mechanics = 0.58 * _pa_handle + 0.42 * _pa_hands

        _pa_raw = (
            0.38 * _pa_delivery
            + 0.25 * _pa_mechanics
            + 0.15 * _pa_variety
            + 0.09 * _pa_usage
            + 0.08 * _pa_ast
            + 0.05 * _pa_transition
        )

        if _pa_delivery > 0.62 and _pa_mechanics > 0.58:
            _pa_raw *= 1.02 + 0.06 * _pa_delivery

        # Slight baseline lift: most NBA rotation players can complete routine passes.
        _pa_raw *= 1.14

        # Require at least some pass load to enter top bands, but don't over-gate connectors.
        _pa_opp = 0.36 * _pa_ast + 0.32 * _pa_variety + 0.18 * _pa_usage + 0.14 * _pa_transition
        _pa_gate = 1.0 if _pa_opp >= 0.16 else (0.66 + 0.34 * max(0.0, _pa_opp / 0.16))

        _pa_value = 100.0 * _pa_raw * _pa_gate

        _pa_floor = (
            0.35
            + 0.09 * size_big
            + 0.08 * _pa_hands
            + 0.05 * self._norm(ast_tov, 0.80, 2.20)
        )
        if size_big > 0.65 and _pa_opp < 0.20:
            _pa_floor += 0.030 * _profile_noise
            _pa_floor = max(0.35, min(0.52, _pa_floor))

        _pa_value = max(_pa_value, 100.0 * min(0.58, _pa_floor))

        _pass_lead_creator = usage >= 28.0 and ast_pg >= 7.0 and ast_tov >= 1.8
        _pass_big_hub = is_big and ast_pg >= 5.8 and ast_tov >= 1.7
        _pass_pure_engine = ast_pg >= 8.8 and ast_tov >= 2.4

        if _pass_lead_creator:
            _pa_value = max(_pa_value, 70.0 + 8.0 * _pa_mechanics)
        if _pass_big_hub:
            _pa_value = max(_pa_value, 72.0 + 7.0 * _pa_delivery)
        if _pass_pure_engine:
            _pa_value = max(_pa_value, 76.0 + 6.0 * _pa_delivery)

        attrs["pass_accuracy"] = max(0.0, min(100.0, _pa_value))

        # ── Pass IQ ───────────────────────────────────────────────────
        # Pass IQ = decision quality: choosing pass timing/target/risk correctly.
        _piq_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _piq_pnr_p36 = pnr_bh_burden_p36
        _piq_iso_p36 = iso_burden_p36
        _piq_handoff_p36 = raw_handoff_p36
        _piq_post_p36 = post_burden_p36

        _piq_ast = self._norm(_piq_ast_p36, 2.0, 12.5)
        _piq_ast_tov = self._norm(ast_tov, 0.90, 3.40)
        _piq_tov = 1.0 - self._norm(tov_pct, 0.09, 0.20)
        _piq_bad_pass = 1.0 - self._norm(bad_pass_turnovers_per36, 0.10, 3.00)
        _piq_live = 1.0 - self._norm(live_ball_turnover_pct, 0.25, 0.75)
        _piq_offfoul = 1.0 - self._norm(offensive_fouls_per36, 0.06, 1.35)
        _piq_usage = self._norm(usage, 14.0, 35.0)
        _piq_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)
        _piq_shot_iq = self._norm(attrs["shot_iq"], 60.0, 95.0)
        _piq_pass_acc = self._norm(attrs["pass_accuracy"], 58.0, 96.0)

        _piq_creator_two = self._norm(creator_two_p36, 0.40, 8.0)
        _piq_creation = (
            0.40 * self._norm(_piq_pnr_p36, 0.20, 10.0)
            + 0.28 * self._norm(_piq_iso_p36, 0.10, 6.0)
            + 0.18 * self._norm(_piq_handoff_p36, 0.05, 3.6)
            + 0.14 * self._norm(_piq_post_p36, 0.05, 4.8)
        )
        _piq_creation = max(_piq_creation, 0.56 * _piq_creator_two + 0.44 * _piq_ast)
        _piq_decision_security = (
            0.32 * _piq_ast_tov
            + 0.24 * _piq_tov
            + 0.20 * _piq_bad_pass
            + 0.14 * _piq_live
            + 0.10 * _piq_offfoul
        )

        # Role-adjusted risk: creators can carry slightly more turnovers without
        # being classified as low-IQ passers.
        _piq_expected_tov = 0.095 + 0.070 * _piq_usage + 0.030 * _piq_creation
        _piq_tov_delta = max(-0.09, min(0.10, tov_pct - _piq_expected_tov))
        _piq_risk_adj = 1.0 - self._norm(_piq_tov_delta, -0.01, 0.06)

        _piq_raw = (
            0.31 * _piq_decision_security
            + 0.21 * _piq_shot_iq
            + 0.15 * _piq_creation
            + 0.11 * _piq_risk_adj
            + 0.10 * _piq_pass_acc
            + 0.07 * _piq_ast
            + 0.05 * _piq_pace
        )

        _piq_raw *= 1.08

        if _piq_decision_security > 0.62 and _piq_shot_iq > 0.58:
            _piq_raw *= 1.03

        _piq_floor = (
            0.40
            + 0.10 * _piq_shot_iq
            + 0.08 * _piq_pass_acc
            + 0.08 * _piq_ast_tov
        )

        _piq_opp = 0.30 * _piq_ast + 0.36 * _piq_creation + 0.20 * _piq_usage + 0.14 * _piq_pace
        _piq_gate = 1.0 if _piq_opp >= 0.15 else (0.68 + 0.32 * max(0.0, _piq_opp / 0.15))

        _piq_floor_cap = 0.70
        if size_big > 0.65 and _piq_opp < 0.18:
            _piq_floor += 0.028 * _profile_noise
            _piq_floor = max(0.36, min(0.54, _piq_floor))
            _piq_floor_cap = 0.56

        _piq_value = 100.0 * _piq_raw * _piq_gate
        _piq_value = max(_piq_value, 100.0 * min(_piq_floor_cap, _piq_floor))

        if _pass_lead_creator:
            _piq_value = max(_piq_value, 72.0 + 8.0 * _piq_decision_security)
        if _pass_big_hub:
            _piq_value = max(_piq_value, 74.0 + 8.0 * _piq_decision_security)
        if _pass_pure_engine:
            _piq_value = max(_piq_value, 78.0 + 7.0 * _piq_decision_security)

        attrs["pass_iq"] = max(0.0, min(100.0, _piq_value))

        # ── Pass Vision ───────────────────────────────────────────────
        # Pass vision = seeing windows/lanes early (distinct from decision or throw quality).
        _pv_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
        _pv_pnr_p36 = pnr_bh_burden_p36
        _pv_iso_p36 = iso_burden_p36
        _pv_handoff_p36 = raw_handoff_p36
        _pv_post_p36 = post_burden_p36

        _pv_ast = self._norm(_pv_ast_p36, 1.8, 12.5)
        _pv_ast_pct = self._norm(pctile_ast, 0.30, 0.99)
        _pv_usage = self._norm(usage, 14.0, 35.0)
        _pv_pace = self._norm(seconds_per_poss_off, 11.5, 18.0)

        _pv_pnr = self._norm(_pv_pnr_p36, 0.20, 10.0)
        _pv_iso = self._norm(_pv_iso_p36, 0.10, 6.0)
        _pv_handoff = self._norm(_pv_handoff_p36, 0.05, 3.6)
        _pv_post_hub = self._norm(_pv_post_p36, 0.05, 4.8)
        _pv_creation_map = 0.40 * _pv_pnr + 0.24 * _pv_iso + 0.20 * _pv_handoff + 0.16 * _pv_post_hub
        _pv_creator_two = self._norm(creator_two_p36, 0.40, 8.0)
        _pv_creation_map = max(_pv_creation_map, 0.54 * _pv_creator_two + 0.46 * _pv_ast)

        _pv_pass_iq = self._norm(attrs["pass_iq"], 58.0, 96.0)
        _pv_pass_acc = self._norm(attrs["pass_accuracy"], 58.0, 96.0)
        _pv_shot_iq = self._norm(attrs["shot_iq"], 60.0, 95.0)
        _pv_size_view = 0.62 * self._norm(height, 74, 84) + 0.38 * self._norm(weight, 185, 265)

        # Vision signal: read load + anticipation context + cognitive support.
        _pv_read_engine = (
            0.46 * _pv_creation_map
            + 0.24 * _pv_ast_pct
            + 0.14 * _pv_ast
            + 0.08 * _pv_size_view
            + 0.08 * _pv_pace
        )

        _pv_raw = (
            0.60 * _pv_read_engine
            + 0.20 * _pv_pass_iq
            + 0.08 * _pv_pass_acc
            + 0.07 * _pv_shot_iq
            + 0.05 * _pv_usage
        )

        if _pv_creation_map > 0.52 and _pv_pass_iq > 0.55:
            _pv_raw *= 1.04 + 0.04 * _pv_creation_map

        _pv_raw *= 1.12

        _pv_opp = 0.32 * _pv_ast + 0.46 * _pv_creation_map + 0.14 * _pv_usage + 0.08 * _pv_handoff
        _pv_gate = 1.0 if _pv_opp >= 0.14 else (0.66 + 0.34 * max(0.0, _pv_opp / 0.14))

        _pv_value = 100.0 * _pv_raw * _pv_gate
        if _pass_lead_creator:
            _pv_value = max(_pv_value, 72.0 + 10.0 * _pv_creation_map)
        if _pass_big_hub:
            _pv_value = max(_pv_value, 74.0 + 9.0 * _pv_creation_map)
        if _pass_pure_engine:
            _pv_value = max(_pv_value, 80.0 + 8.0 * _pv_creation_map)

        attrs["pass_vision"] = max(0.0, min(100.0, _pv_value))

        # ── Offensive Consistency ─────────────────────────────────────
        # Simplified consistency model (no game-log variance available):
        # 35% efficiency, 25% turnover control, 15% FT reliability,
        # 15% shot IQ, 10% usage/role stability.
        _oc_eff = 0.58 * self._norm(ts_pct, 0.50, 0.68) + 0.42 * self._norm(efg_pct, 0.47, 0.63)

        # Role-adjusted turnover control: high-usage creators can carry more risk.
        _oc_usage = self._norm(usage, 12.0, 35.0)
        _oc_expected_tov = 0.092 + 0.075 * _oc_usage
        _oc_tov_delta = max(-0.08, min(0.10, tov_pct - _oc_expected_tov))
        _oc_tov_role_adj = 1.0 - self._norm(_oc_tov_delta, -0.01, 0.06)
        _oc_tov_control = 0.60 * _oc_tov_role_adj + 0.40 * self._norm(ast_tov, 0.85, 3.40)

        _oc_ft = self._norm(ft_pct, 0.58, 0.92)
        _oc_shot_iq = self._norm(attrs["shot_iq"], 55.0, 96.0)
        _oc_prod = self._norm(pts_pg, 10.0, 32.0)

        # Usage stability proxy: stars with stable role + real minute load
        # generally produce more repeatable offense than low-minute roles.
        _oc_role_stability = 0.62 * self._norm(usage, 16.0, 33.0) + 0.38 * self._norm(min_pg, 18.0, 36.0)

        _oc_raw = (
            0.35 * _oc_eff
            + 0.25 * _oc_tov_control
            + 0.15 * _oc_ft
            + 0.15 * _oc_shot_iq
            + 0.10 * _oc_role_stability
        )

        # Reliability tweak so tiny samples are less extreme.
        _oc_reliability = 0.58 * self._norm(gp, 20.0, 82.0) + 0.42 * self._norm(min_pg, 16.0, 36.0)
        _oc_value = 100.0 * _oc_raw * (0.86 + 0.14 * _oc_reliability)

        # Keep role hierarchy intuitive.
        _oc_star_signal = (
            0.34 * _oc_eff
            + 0.28 * _oc_shot_iq
            + 0.22 * _oc_prod
            + 0.16 * _oc_usage
        )
        if usage >= 28.0 and pts_pg >= 24.0 and _oc_star_signal >= 0.56:
            _oc_star_tier = max(0.0, min(1.0, (_oc_star_signal - 0.56) / 0.44))
            _oc_value = max(_oc_value, 80.0 + 6.0 * _oc_star_tier)

        if usage >= 30.0 and pts_pg >= 26.0 and _oc_shot_iq >= 0.60:
            _oc_value = max(_oc_value, 81.0 + 4.0 * _oc_tov_role_adj)

        if usage <= 16.0 and min_pg <= 22.0:
            _oc_value = min(_oc_value, 78.0)

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
        _id_pos = pos.replace(" ", "")
        _id_is_guard = ("PG" in _id_pos) or ("SG" in _id_pos) or (_id_pos == "G")
        _id_is_big = ("C" in _id_pos) or ("PF" in _id_pos)
        _id_is_wing = (not _id_is_guard) and (not _id_is_big)

        # Core model weights requested:
        # IQ 30%, size/strength 25%, defensive rebound 15%,
        # foul control 15%, block presence 15%.
        _id_iq = 0.50 * _id_engage + 0.30 * _id_disc + 0.20 * _id_recover
        _id_size_strength = 0.55 * _id_strength + 0.45 * _id_size
        _id_block_presence = 0.78 * _id_blk + 0.22 * _id_rim
        _id_dreb_role = _id_dreb
        if not _id_is_big:
            _id_dreb_role = min(_id_dreb_role, 0.62)
        if (not _id_is_big) and usage >= 30:
            _id_dreb_role = min(_id_dreb_role, 0.45)

        _id_base = (
            0.30 * _id_iq
            + 0.25 * _id_size_strength
            + 0.15 * _id_dreb_role
            + 0.15 * _id_disc
            + 0.15 * _id_block_presence
        )

        if _id_is_big:
            _id_raw = (
                0.56 * _id_base
                + 0.17 * _id_size_strength
                + 0.11 * _id_block_presence
                + 0.08 * _id_dreb_role
                + 0.08 * _id_iq
            )
        elif _id_is_wing:
            _id_raw = (
                0.64 * _id_base
                + 0.14 * _id_iq
                + 0.10 * _id_size_strength
                + 0.06 * _id_block_presence
                + 0.06 * _id_dreb_role
            )
        else:
            _id_raw = (
                0.67 * _id_base
                + 0.17 * _id_iq
                + 0.08 * _id_disc
                + 0.05 * _id_block_presence
                + 0.03 * _id_dreb_role
            )

        # Interior defense opportunity gate: smaller low-rim profiles should not
        # grade like true paint anchors.
        _id_opp = 0.52 * _id_size + 0.34 * _id_block_presence + 0.14 * _id_iq
        _id_gate = 1.0 if _id_opp >= 0.34 else (0.58 + 0.42 * max(0.0, _id_opp / 0.34))

        # Elite anchor recognition for dominant rim-protecting bigs.
        if _id_is_big and _id_rim > 0.72 and _id_blk > 0.62 and _id_dreb > 0.56:
            _id_raw *= 1.07 + 0.05 * _id_rim

        # Mobile elite-big recognition (switchable rim protectors).
        if _id_is_big and _id_blk > 0.66 and _id_dreb > 0.50 and _id_iq > 0.52:
            _id_raw *= 1.06 + 0.05 * _id_blk

        # Foul-prone elite shot blockers should not collapse too far when
        # rim-protection volume is truly high.
        if _id_is_big and _id_blk > 0.58 and _id_disc < 0.35 and blk_per36 >= 1.60:
            _id_raw *= 1.08

        # Raw floor: elite shot-blocking bigs (JJJ/Myles tier).
        if _id_is_big and blk_per36 >= 1.60:
            _id_raw = max(_id_raw, 0.74)

        # Raw floor: dominant mobile rim-protectors with rebounding (Mobley tier).
        if _id_is_big and blk_per36 >= 1.75 and _id_dreb > 0.50:
            _id_raw = max(_id_raw, 0.78)

        # Elite glass + size + discipline anchors (Gobert lane).
        if _id_is_big and dreb_pg >= 7.0 and _id_size_strength >= 0.66 and _id_disc >= 0.50 and blk_per36 >= 1.20:
            _id_raw = max(_id_raw, 0.82)

        # True low-usage paint anchors (Gobert/Kessler lane).
        if _id_is_big and ("C" in _id_pos) and usage <= 18 and dreb_pg >= 7.0 and blk_per36 >= 1.35 and _id_disc >= 0.48:
            _id_raw = max(_id_raw, 0.85)

        # Hybrid high-stock bigs (JJJ lane), even with moderate foul pressure.
        if (not _id_is_guard) and blk_per36 >= 1.70 and stl_per36 >= 1.30 and usage <= 30 and _id_size_strength >= 0.50:
            _id_raw = max(_id_raw, 0.76)

        # High-minute jumbo stars with real interior event volume should not
        # sit in low interior tiers purely from role/noise.
        if (not _id_is_guard) and height >= 81 and usage >= 30 and blk_per36 >= 1.0 and _id_disc >= 0.40:
            _id_raw = max(_id_raw, 0.72)

        # High-offense guards/creators with only event steals should not grade high inside.
        if (not _id_is_big) and usage > 30 and _id_size < 0.45 and _id_blk < 0.28:
            _id_raw *= 0.88 + 0.12 * _id_disc

        # High-usage non-bigs with weak interior tools should grade clearly lower.
        if (not _id_is_big) and usage >= 31 and _id_block_presence < 0.24 and _id_iq < 0.45:
            _id_raw *= 0.80 + 0.20 * _id_disc

        # Very high-usage non-bigs with weak rim deterrence should not overgrade
        # from rebounding volume alone.
        if (not _id_is_big) and usage >= 32 and _id_blk < 0.22 and _id_iq < 0.50:
            _id_raw *= 0.68 + 0.32 * _id_disc

        # High-usage perimeter creators with low block volume are typically
        # weaker interior defenders despite size/rebound counting stats.
        if (not _id_is_big) and usage >= 32 and blk_per36 < 0.65:
            _id_raw *= 0.82 + 0.18 * _id_disc

        _id_value = 100.0 * _id_raw * _id_gate * (0.84 + 0.16 * _id_reliability)

        if _id_is_big and ("C" in _id_pos) and usage <= 18 and dreb_pg >= 7.0 and blk_per36 >= 1.35 and _id_disc >= 0.48:
            _id_value = max(_id_value, 82.0)

        attrs["interior_defense"] = max(0.0, min(100.0, _id_value))

        # ── Perimeter Defense ─────────────────────────────────────────
        # Perimeter Defense = on-ball containment, contests, and
        # navigating perimeter actions.
        # Target weighting:
        #   IQ proxy        30%
        #   Steal skill     20%
        #   Speed           20%
        #   Activity (stl)  15%
        #   Foul control    15%

        _pd_disc = 1.0 - self._norm(pf_per36, 1.5, 4.6)
        _pd_stl_activity = self._norm(stl_per36, 0.45, 2.50)
        _pd_speed = self._norm(attrs.get("speed", 50.0), 45.0, 95.0)
        _pd_hands = self._norm(attrs.get("hands", 50.0), 45.0, 95.0)
        _pd_len = self._norm(height, 73, 81)
        _pd_body = self._norm(weight, 170, 240)
        _pd_contest = self._norm(blk_per36, 0.05, 1.60)
        _pd_engage = def_engagement
        _pd_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _pd_iq = 0.60 * _pd_engage + 0.20 * _pd_len + 0.20 * _pd_contest
        _pd_steal_skill = 0.72 * _pd_stl_activity + 0.28 * _pd_hands

        _pd_base = (
            0.30 * _pd_iq
            + 0.20 * _pd_steal_skill
            + 0.20 * _pd_speed
            + 0.15 * _pd_stl_activity
            + 0.15 * _pd_disc
        )

        if is_guard:
            _pd_raw = (
                0.76 * _pd_base
                + 0.10 * _pd_speed
                + 0.08 * _pd_iq
                + 0.06 * _pd_steal_skill
            )
        elif is_big:
            _pd_raw = (
                0.72 * _pd_base
                + 0.12 * _pd_iq
                + 0.08 * _pd_len
                + 0.08 * _pd_body
            )
        else:  # wing
            _pd_raw = (
                0.76 * _pd_base
                + 0.10 * _pd_iq
                + 0.08 * _pd_len
                + 0.06 * _pd_speed
            )

        _pd_raw *= 1.08

        # Heavy offensive load usually reduces sustained on-ball effort.
        if usage > 32 and _pd_engage < 0.30:
            _pd_raw *= 0.86 + 0.14 * _pd_disc

        # High-usage low-activity perimeter defenders should grade lower.
        if usage > 29 and _pd_stl_activity < 0.42 and _pd_iq < 0.52:
            _pd_raw *= 0.82 + 0.18 * _pd_disc

        # Tiny high-usage guards are often targeted in point-of-attack defense.
        if height <= 75 and weight <= 180 and usage >= 26 and _pd_disc < 0.72:
            _pd_raw *= 0.78 + 0.22 * _pd_disc

        # True 7-foot centers should not sit in high perimeter tiers.
        if is_big and height >= 84:
            _pd_raw *= 0.82 + 0.18 * _pd_speed

        _pd_opp = 0.44 * _pd_iq + 0.34 * _pd_speed + 0.22 * _pd_len
        _pd_gate = 1.0 if _pd_opp >= 0.32 else (0.58 + 0.42 * max(0.0, _pd_opp / 0.32))

        if is_center and height >= 83:
            _pd_raw *= 0.78 + 0.22 * _pd_speed
            _pd_raw = min(_pd_raw, 0.42 + 0.26 * _pd_speed + 0.12 * _pd_iq)

        # Stopper floors.
        if (not is_big) and stl_per36 >= 1.70 and _pd_disc >= 0.78:
            _pd_raw = max(_pd_raw, 0.86)

        if (not is_big) and stl_per36 >= 1.80 and _pd_disc >= 0.40 and usage < 20:
            _pd_raw = max(_pd_raw, 0.87)

        if (not is_big) and stl_per36 >= 1.00 and _pd_disc >= 0.75 and usage < 25:
            _pd_raw = max(_pd_raw, 0.74)

        if is_guard and usage < 18 and weight >= 210 and stl_per36 >= 0.90:
            _pd_raw = max(_pd_raw, 0.82)

        _pd_value = 100.0 * _pd_raw * _pd_gate * (0.82 + 0.18 * _pd_reliability)
        if is_center and height >= 83:
            _pd_value = min(_pd_value, 44.0 + 13.0 * _pd_speed + 7.0 * _pd_iq)
        if is_center and 80 <= height < 83 and usage <= 20:
            _pd_value = min(_pd_value, 62.0)
        attrs["perimeter_defense"] = max(0.0, min(100.0, _pd_value))

        # ── Steal ─────────────────────────────────────────────────────
        # Steal = ball-takeaway skill from activity + discipline + awareness.
        # Weighting target:
        #   45% steal activity
        #   20% foul control
        #   20% awareness proxy
        #   10% position factor
        #    5% reliability
        _sl_activity = self._norm(stl_per36, 0.45, 2.60)
        _sl_disc = 1.0 - self._norm(pf_per36, 1.6, 4.8)
        _sl_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 90.0)
        _sl_awareness = 0.72 * _sl_poa + 0.28 * def_engagement
        _sl_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        if pos in {"PG", "G"}:
            _sl_pos_factor = 1.00
        elif pos in {"SG", "G-F", "F-G"}:
            _sl_pos_factor = 0.95
        elif pos in {"SF", "F"}:
            _sl_pos_factor = 0.90
        elif pos in {"PF", "F-C", "C-F"}:
            _sl_pos_factor = 0.82
        elif pos == "C":
            _sl_pos_factor = 0.72
        else:
            _sl_pos_factor = 0.88

        _sl_raw = (
            0.45 * _sl_activity
            + 0.20 * _sl_disc
            + 0.20 * _sl_awareness
            + 0.10 * _sl_pos_factor
            + 0.05 * _sl_reliability
        )

        # Reaching/gambling penalty: high event steals with low discipline
        # should not over-rate as true steal skill.
        _sl_gambler = _sl_activity * (1.0 - _sl_disc)
        if _sl_gambler > 0.46 and usage > 28:
            _sl_raw *= 0.90 + 0.10 * (1.0 - _sl_gambler)

        # Very tall true bigs are naturally capped for steal-heavy profiles.
        if is_big and height > 82 and stl_per36 < 1.10:
            _sl_raw *= 0.90

        # Event-creation floors for true perimeter disruptors.
        if (not is_big) and stl_per36 >= 1.90 and _sl_disc >= 0.62:
            _sl_raw = max(_sl_raw, 0.84)

        if (not is_big) and stl_per36 >= 1.55 and _sl_disc >= 0.70 and usage < 26:
            _sl_raw = max(_sl_raw, 0.75)

        _sl_value = 100.0 * _sl_raw

        # Top-end taper: extreme event-steal seasons should peak near low 90s,
        # not mid/high 90s, unless every supporting signal is elite.
        _sl_top_end = 0.55 * _sl_activity + 0.25 * _sl_disc + 0.20 * _sl_awareness
        if _sl_activity >= 0.92:
            _sl_top_tier = max(0.0, min(1.0, (_sl_top_end - 0.80) / 0.20))
            _sl_value = min(_sl_value, 83.0 + 4.0 * _sl_top_tier)

        attrs["steal"] = max(0.0, min(100.0, _sl_value))

        # ── Block ─────────────────────────────────────────────────────
        # Ability to protect the rim and contest shots vertically.
        # Weighted target:
        #   45% block production
        #   20% size/length
        #   15% foul control
        #   10% awareness/IQ proxy
        #   10% archetype factor
        _bl_activity = self._norm(blk_per36, 0.12, 3.20)
        _bl_height = self._norm(height, 72.0, 84.0)
        _bl_size = 0.75 * _bl_height + 0.25 * self._norm(weight, 180.0, 280.0)
        _bl_disc = 1.0 - self._norm(pf_per36, 1.6, 4.8)
        _bl_iq = 0.65 * self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0) + 0.35 * def_engagement

        if pos in {"C", "C-F", "F-C"}:
            _bl_arch = 1.00
        elif pos == "PF":
            _bl_arch = 0.92
        elif pos in {"SF", "F"}:
            _bl_arch = 0.84
        elif pos in {"SG", "G-F", "F-G"}:
            _bl_arch = 0.74
        elif pos in {"PG", "G"}:
            _bl_arch = 0.66
        else:
            _bl_arch = 0.82

        _bl_raw = (
            0.45 * _bl_activity
            + 0.20 * _bl_size
            + 0.15 * _bl_disc
            + 0.10 * _bl_iq
            + 0.10 * _bl_arch
        )

        # High-event blockers should not be over-penalized by foul rates alone.
        if is_big and _bl_activity >= 0.50:
            _bl_disc_relief = max(0.0, min(1.0, (_bl_activity - 0.50) / 0.50))
            _bl_raw += 0.05 * _bl_disc_relief * (1.0 - _bl_disc)

        _bl_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        # High-usage non-bigs with very low block events should not drift up.
        if (not is_big) and usage > 30 and blk_per36 < 0.70:
            _bl_raw *= 0.88 + 0.12 * _bl_disc

        # Tiny guards are naturally capped.
        if is_guard and height < 75 and blk_per36 < 0.55:
            _bl_raw *= 0.88

        # Very tall centers with real block volume keep strong floors.
        if is_big and blk_per36 >= 2.60:
            _bl_raw = max(_bl_raw, 0.84)
        if is_big and blk_per36 >= 1.90:
            _bl_raw = max(_bl_raw, 0.74)
        if is_big and height >= 84 and blk_per36 >= 1.60:
            _bl_raw = max(_bl_raw, 0.76)

        # Quality shot-blocking wings should sit above average.
        if is_wing and (not is_big) and blk_per36 >= 1.00 and height >= 78:
            _bl_raw = max(_bl_raw, 0.58)

        # Low-event centers should not inflate from size alone.
        if is_center and blk_per36 < 0.80:
            _bl_raw = min(_bl_raw, 0.60)

        _bl_value = 100.0 * _bl_raw * (0.84 + 0.16 * _bl_reliability)
        _bl_value += 2.5
        attrs["block"] = max(0.0, min(100.0, _bl_value))

        # ── Offensive Rebound ─────────────────────────────────────────
        # Offensive Rebound = ability to create second-chance possessions.
        # Target weighting:
        #   50% OREB production
        #   20% height/length
        #   15% strength/box-out proxy
        #   15% position/archetype
        _rb_pos = pos.replace(" ", "")
        _rb_is_guard = ("PG" in _rb_pos) or ("SG" in _rb_pos) or (_rb_pos == "G") or ("G-" in _rb_pos) or ("-G" in _rb_pos)
        _rb_is_big = ("C" in _rb_pos) or ("PF" in _rb_pos)
        _rb_is_wing = (not _rb_is_guard) and (not _rb_is_big)

        _or_prod = self._norm(oreb_pg, 0.30, 5.80)
        _or_height = self._norm(height, 72, 84)
        _or_strength = 0.75 * self._norm(weight, 175, 290) + 0.25 * (1.0 - self._norm(pf_per36, 1.8, 4.9))
        _or_pos_mod = 1.0 if _rb_is_big else (0.62 if _rb_is_wing else 0.35)
        _or_crash = 1.0 - self._norm(usage, 18, 35)
        _or_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _or_raw = (
            0.50 * _or_prod
            + 0.20 * _or_height
            + 0.15 * _or_strength
            + 0.15 * _or_pos_mod
        )

        _or_raw *= 1.09

        if _rb_is_big:
            _or_raw += 0.03 * self._norm(oreb_pg, 1.0, 4.0) + 0.02 * _or_height

        # High-usage non-bigs usually leak out for transition creation instead
        # of crashing every possession.
        if (not _rb_is_big) and usage > 31 and oreb_pg < 1.20:
            _or_raw *= 0.90 + 0.10 * _or_crash

        # Archetype floors.
        if _rb_is_big and oreb_pg >= 3.80:
            _or_raw = max(_or_raw, 0.88)

        if _rb_is_big and oreb_pg >= 3.20:
            _or_raw = max(_or_raw, 0.81)

        if _rb_is_big and oreb_pg >= 2.60:
            _or_raw = max(_or_raw, 0.74)

        # Rebounding wings/guards (Hart/Luka lane).
        if (not _rb_is_big) and oreb_pg >= 1.60 and dreb_pg >= 6.5 and height >= 77:
            _or_raw = max(_or_raw, 0.58)

        # Jumbo creator wing who still attacks weakside boards (Giannis lane).
        if _rb_is_wing and usage > 34 and oreb_pg >= 2.60 and dreb_pg >= 8.5:
            _or_raw = max(_or_raw, 0.66)

        _or_value = 100.0 * _or_raw * (0.84 + 0.16 * _or_reliability)
        if _rb_is_big:
            _or_value += 2.5
        attrs["offensive_rebound"] = max(0.0, min(100.0, _or_value))

        # ── Defensive Rebound ─────────────────────────────────────────
        # Defensive Rebound = ending defensive possessions.
        # Target weighting:
        #   50% DREB production
        #   20% height/length
        #   15% strength/box-out proxy
        #   15% position/archetype
        _dr_prod = self._norm(dreb_pg, 2.5, 13.8)
        _dr_height = self._norm(height, 72, 84)
        _dr_strength = 0.78 * self._norm(weight, 175, 295) + 0.22 * (1.0 - self._norm(pf_per36, 1.8, 4.9))
        _dr_pos_mod = 1.0 if _rb_is_big else (0.64 if _rb_is_wing else 0.38)
        _dr_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _dr_raw = (
            0.50 * _dr_prod
            + 0.20 * _dr_height
            + 0.15 * _dr_strength
            + 0.15 * _dr_pos_mod
        )

        _dr_raw *= 1.10

        if _rb_is_big:
            _dr_raw += 0.04 * self._norm(dreb_pg, 6.0, 12.0) + 0.02 * _dr_height

        # High-usage non-bigs with low rebound volume should not overgrade.
        if (not _rb_is_big) and usage > 34 and dreb_pg < 5.5:
            _dr_raw *= 0.90 + 0.10 * (1.0 - self._norm(usage, 20, 36))

        # Archetype floors.
        if _rb_is_big and dreb_pg >= 10.00:
            _dr_raw = max(_dr_raw, 0.94)

        if _rb_is_big and dreb_pg >= 9.50:
            _dr_raw = max(_dr_raw, 0.92)

        if _rb_is_big and dreb_pg >= 8.80:
            _dr_raw = max(_dr_raw, 0.87)

        if _rb_is_big and dreb_pg >= 7.50:
            _dr_raw = max(_dr_raw, 0.80)

        # Rebounding guards/wings.
        if (not _rb_is_big) and dreb_pg >= 7.0 and height >= 77:
            _dr_raw = max(_dr_raw, 0.67)

        if _rb_is_guard and dreb_pg >= 6.8 and usage < 20:
            _dr_raw = max(_dr_raw, 0.64)

        # Jumbo wing glass cleaner (Giannis archetype).
        if _rb_is_wing and height >= 82 and dreb_pg >= 8.8:
            _dr_raw = max(_dr_raw, 0.78)

        _dr_value = 100.0 * _dr_raw * (0.84 + 0.16 * _dr_reliability)
        if _rb_is_big:
            _dr_value += 3.0
        attrs["defensive_rebound"] = max(0.0, min(100.0, _dr_value))

        # ── Help Defense IQ ───────────────────────────────────────────
        # Help Defense IQ = rotation timing, tag/help decisions, and
        # team-defense positioning.
        _hd_poa = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 92.0)
        _hd_rim = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _hd_stl = self._norm(stl_per36, 0.45, 2.40)
        _hd_blk = self._norm(blk_per36, 0.15, 3.20)
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
                0.33 * _hd_poa
                + 0.17 * _hd_stl
                + 0.15 * _hd_disc
                + 0.13 * _hd_engage
                + 0.10 * _hd_balance
                + 0.06 * _hd_rim
                + 0.04 * _hd_events
                + 0.02 * _hd_recover
            )
        elif is_big:
            _hd_raw = (
                0.29 * _hd_rim
                + 0.18 * _hd_blk
                + 0.14 * _hd_disc
                + 0.13 * _hd_engage
                + 0.08 * _hd_poa
                + 0.08 * _hd_events
                + 0.05 * _hd_balance
                + 0.03 * _hd_tools
                + 0.02 * _hd_recover
            )
        else:  # wing
            _hd_raw = (
                0.23 * _hd_poa
                + 0.20 * _hd_rim
                + 0.12 * _hd_stl
                + 0.10 * _hd_blk
                + 0.12 * _hd_disc
                + 0.10 * _hd_engage
                + 0.05 * _hd_balance
                + 0.03 * _hd_tools
                + 0.03 * _hd_events
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
        if is_big and blk_per36 >= 1.60 and _hd_rim >= 0.62:
            _hd_raw = max(_hd_raw, 0.84)

        # Active weakside bigs with strong interior command (Mobley/JJJ/AD lane).
        if is_big and blk_per36 >= 1.55:
            _hd_raw = max(_hd_raw, 0.78)

        if (not is_big) and stl_per36 >= 1.10 and usage < 25:
            _hd_raw = max(_hd_raw, 0.80)

        # High-usage elite wing stoppers still provide strong team help reads.
        if is_wing and stl_per36 >= 1.70 and pf_per36 <= 2.1:
            _hd_raw = max(_hd_raw, 0.74)

        # High-IQ read/react bigs with elite glass + hands but moderate block volume.
        if is_big and stl_per36 >= 1.25 and _hd_disc > 0.45 and _hd_recover > 0.46:
            _hd_raw = max(_hd_raw, 0.74)

        # Jumbo wing/forward help lane (Giannis archetype).
        if (not is_guard) and height >= 82 and blk_per36 >= 0.95 and stl_per36 >= 0.85 and _hd_rim >= 0.50:
            _hd_raw = max(_hd_raw, 0.76)

        # Hyper-mobile rim eraser wing/big hybrids (Wemby archetype).
        if (not is_guard) and height >= 83 and blk_per36 >= 2.80 and stl_per36 >= 1.10 and _hd_rim >= 0.48:
            _hd_raw = max(_hd_raw, 0.84)

        # High-minute jumbo stars with strong stocks should not grade too low.
        if (not is_guard) and height >= 81 and usage >= 30 and blk_per36 >= 1.10 and stl_per36 >= 0.85 and _hd_disc >= 0.40:
            _hd_raw = max(_hd_raw, 0.70)

        # High-stock weakside disruptors with moderate foul pressure (JJJ archetype).
        if (not is_guard) and blk_per36 >= 1.70 and stl_per36 >= 1.35 and usage <= 30:
            _hd_raw = max(_hd_raw, 0.72)

        # Versatile low-usage wing helper lane (Draymond archetype).
        if is_wing and usage < 22 and stl_per36 >= 1.20 and blk_per36 >= 0.90:
            _hd_raw = max(_hd_raw, 0.76)

        _hd_value = 100.0 * _hd_raw * (0.82 + 0.18 * _hd_reliability)
        attrs["help_defense_iq"] = max(0.0, min(100.0, _hd_value))

        # ── Pass Perception ───────────────────────────────────────────
        # Pass Perception = anticipation of passing lanes and timely reads.
        _pp_disc = 1.0 - self._norm(pf_per36, 1.7, 4.9)
        _pp_help_rating = self._norm(attrs.get("help_defense_iq", 50.0), 45.0, 95.0)
        _pp_pdef_rating = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 95.0)
        _pp_steal_rating = self._norm(attrs.get("steal", 50.0), 45.0, 95.0)
        _pp_activity = self._norm(stl_per36 + blk_per36, 0.90, 4.40)
        _pp_experience = self._norm(age, 21.0, 33.0)
        _pp_pos = pos.replace(" ", "")
        _pp_is_guard = ("PG" in _pp_pos) or ("SG" in _pp_pos) or (_pp_pos == "G") or ("G-" in _pp_pos) or ("-G" in _pp_pos)
        _pp_is_big = ("C" in _pp_pos) or ("PF" in _pp_pos)
        _pp_is_wing = (not _pp_is_guard) and (not _pp_is_big)
        _pp_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        # IQ proxy for lane-reading: mostly perimeter + help context, with
        # discipline as a smaller tie-breaker.
        _pp_iq = 0.50 * _pp_pdef_rating + 0.35 * _pp_help_rating + 0.15 * _pp_disc

        _pp_base = (
            0.40 * _pp_iq
            + 0.25 * _pp_steal_rating
            + 0.20 * _pp_help_rating
            + 0.10 * _pp_activity
            + 0.05 * _pp_experience
        )

        if _pp_is_guard:
            _pp_raw = 0.84 * _pp_base + 0.10 * _pp_activity + 0.06 * _pp_iq
        elif _pp_is_big:
            _pp_raw = 0.86 * _pp_base + 0.08 * _pp_help_rating + 0.06 * _pp_iq
        else:  # wing
            _pp_raw = 0.84 * _pp_base + 0.09 * _pp_iq + 0.07 * _pp_activity

        _pp_raw *= 1.08

        # Event-only gambler penalty.
        _pp_gambler = _pp_activity * (1.0 - _pp_disc) * (1.0 - _pp_help_rating)
        if _pp_gambler > 0.38 and usage > 29:
            _pp_raw *= 0.88 + 0.12 * (1.0 - _pp_gambler)

        # High-usage low-IQ creators should not rate as elite lane readers.
        if usage > 31 and _pp_iq < 0.52:
            _pp_raw *= 0.84 + 0.16 * _pp_disc

        # High-usage creators with weak help context and weak weakside events
        # should not overgrade from steals alone.
        if (not _pp_is_big) and usage >= 30 and _pp_help_rating < 0.52 and blk_per36 < 0.75:
            _pp_raw *= 0.80 + 0.20 * _pp_disc

        # Floor lanes.
        if (not _pp_is_big) and stl_per36 >= 1.70 and _pp_help_rating >= 0.58:
            _pp_raw = max(_pp_raw, 0.82)

        if (not _pp_is_big) and stl_per36 >= 1.10 and _pp_help_rating >= 0.56 and usage < 25:
            _pp_raw = max(_pp_raw, 0.74)

        if _pp_is_big and _pp_help_rating >= 0.60 and stl_per36 >= 0.95:
            _pp_raw = max(_pp_raw, 0.70)

        if _pp_is_big and stl_per36 >= 1.35 and dreb_pg >= 8.8 and _pp_disc > 0.45:
            _pp_raw = max(_pp_raw, 0.74)

        if _pp_is_wing and usage < 22 and stl_per36 >= 1.20 and blk_per36 >= 0.85:
            _pp_raw = max(_pp_raw, 0.78)

        # High-IQ stopper wings/guards (Kawhi lane).
        if (not is_big) and _pp_pdef_rating >= 0.72 and _pp_disc >= 0.72 and stl_per36 >= 1.30:
            _pp_raw = max(_pp_raw, 0.74)

        # Low-usage connective point-of-attack defenders (White/Jrue lane).
        if is_guard and usage < 25 and _pp_pdef_rating >= 0.68 and _pp_help_rating >= 0.50 and _pp_disc >= 0.70 and stl_per36 >= 0.95:
            _pp_raw = max(_pp_raw, 0.72)

        # High-usage elite disruptors with real weakside activity (Shai lane).
        if is_guard and usage >= 30 and stl_per36 >= 1.70 and blk_per36 >= 0.80 and _pp_disc >= 0.55:
            _pp_raw = max(_pp_raw, 0.62)

        _pp_raw *= 0.84 + 0.16 * _pp_reliability

        # Final-stage floors after reliability scaling.
        if (not _pp_is_big) and _pp_pdef_rating >= 0.72 and _pp_disc >= 0.72 and stl_per36 >= 1.30:
            _pp_raw = max(_pp_raw, 0.66)

        if _pp_is_guard and usage < 25 and _pp_pdef_rating >= 0.68 and _pp_help_rating >= 0.50 and _pp_disc >= 0.70 and stl_per36 >= 0.95:
            _pp_raw = max(_pp_raw, 0.62)

        if _pp_is_guard and usage >= 30 and stl_per36 >= 1.70 and blk_per36 >= 0.80 and _pp_disc >= 0.55:
            _pp_raw = max(_pp_raw, 0.55)

        # Stat-driven final floors (independent of interim raw defense scales).
        if (not _pp_is_big) and stl_per36 >= 1.70 and pf_per36 <= 2.20:
            _pp_raw = max(_pp_raw, 0.62)

        if _pp_is_guard and usage < 25 and stl_per36 >= 0.95 and blk_per36 >= 0.80 and pf_per36 <= 2.20:
            _pp_raw = max(_pp_raw, 0.60)

        if _pp_is_guard and usage >= 30 and stl_per36 >= 1.70 and blk_per36 >= 0.80 and pf_per36 <= 2.50:
            _pp_raw = max(_pp_raw, 0.56)

        attrs["pass_perception"] = max(0.0, min(100.0, 100.0 * _pp_raw))

        # ── Defensive Consistency ─────────────────────────────────────
        _dc_pdef = self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 95.0)
        _dc_idef = self._norm(attrs.get("interior_defense", 50.0), 45.0, 95.0)
        _dc_help = self._norm(attrs.get("help_defense_iq", 50.0), 45.0, 95.0)
        _dc_disc = 1.0 - self._norm(self._f(f, "pf_per36", 3.0), 1.6, 4.8)
        _dc_activity = self._norm(stl_per36 + 0.85 * blk_per36, 1.0, 4.7)
        _dc_reliability = (
            0.58 * self._norm(min_pg, 12.0, 34.0)
            + 0.42 * self._norm(gp, 25.0, 82.0)
        )

        # Core reliability weighting (no defensive rebound in this model).
        _dc_base = (
            0.25 * _dc_pdef
            + 0.25 * _dc_idef
            + 0.25 * _dc_help
            + 0.20 * _dc_disc
            + 0.05 * _dc_activity
        )

        if is_guard:
            _dc_raw = _dc_base + 0.06 * (_dc_pdef - _dc_idef) + 0.03 * _dc_activity
        elif is_big:
            _dc_raw = _dc_base + 0.06 * (_dc_idef - _dc_pdef) + 0.03 * _dc_activity
        else:
            _dc_raw = _dc_base + 0.03 * _dc_help + 0.03 * _dc_activity

        _dc_raw *= 1.06

        # High-usage players with weak team-defense context are inconsistent.
        if usage > 31 and _dc_help < 0.52 and _dc_disc < 0.58:
            _dc_raw *= 0.82 + 0.18 * _dc_disc

        # Tiny high-usage guards with weak interior presence are hunted.
        if is_guard and height <= 75 and usage >= 27 and blk_per36 < 0.35:
            _dc_raw *= 0.80 + 0.20 * _dc_disc

        # Elite all-around defenders (rare 90+ lane).
        if _dc_pdef > 0.72 and _dc_idef > 0.70 and _dc_help > 0.72 and _dc_disc > 0.68:
            _dc_raw = max(_dc_raw, 0.86)

        # Elite anchor bigs.
        if is_big and _dc_idef > 0.75 and _dc_help > 0.68 and _dc_disc > 0.55:
            _dc_raw = max(_dc_raw, 0.84)

        # Good low-usage connective defenders.
        if usage < 26 and _dc_help > 0.60 and _dc_disc > 0.62 and (_dc_pdef > 0.58 or _dc_idef > 0.58):
            _dc_raw = max(_dc_raw, 0.78)

        _dc_raw *= 0.82 + 0.18 * _dc_reliability

        # Final-stage tier lanes after reliability scaling.
        if _dc_pdef > 0.72 and _dc_idef > 0.70 and _dc_help > 0.72 and _dc_disc > 0.68:
            _dc_raw = max(_dc_raw, 0.90)

        if ((_dc_pdef > 0.66 and _dc_help > 0.64) or (_dc_idef > 0.70 and _dc_help > 0.62)) and _dc_disc > 0.58:
            _dc_raw = max(_dc_raw, 0.82)

        if _dc_pdef > 0.54 and _dc_idef > 0.54 and _dc_help > 0.54 and _dc_disc > 0.45:
            _dc_raw = max(_dc_raw, 0.70)

        # Keep weak defenders in a realistic floor band.
        _dc_raw = max(_dc_raw, 0.24)

        attrs["defensive_consistency"] = max(0.0, min(100.0, 100.0 * _dc_raw))

        # ══════════════════════════════════════════════════════════════
        # PHYSICAL
        # ══════════════════════════════════════════════════════════════

        # ── Speed ─────────────────────────────────────────────────────
        # Speed = movement WITHOUT the ball, using only available DB fields.
        _sp_size = 0.58 * (1.0 - self._norm(height, 74.0, 85.0)) + 0.42 * (1.0 - self._norm(weight, 180.0, 285.0))
        _sp_age = age_phys
        _sp_pace = 1.0 - self._norm(seconds_per_poss_off, 14.0, 20.0)
        _sp_off_poss = self._norm(off_poss, 12.0, 75.0)
        _sp_second = self._norm(second_chance_off_poss_rate, 0.03, 0.22)
        _sp_def_chase = self._norm(stl_per36 + 0.35 * blk_per36, 0.8, 3.0)
        _sp_reliability = 0.55 * self._norm(min_pg, 12.0, 35.0) + 0.45 * self._norm(gp, 22.0, 82.0)

        _sp_raw = (
            0.34 * _sp_size
            + 0.22 * _sp_pace
            + 0.16 * _sp_off_poss
            + 0.10 * _sp_second
            + 0.10 * _sp_def_chase
            + 0.08 * _sp_age
        )

        _SP_POS_SCALE = {
            "PG": 1.06,
            "G": 1.05,
            "SG": 1.03,
            "G-F": 1.01,
            "F-G": 1.00,
            "SF": 0.98,
            "F": 0.97,
            "PF": 0.93,
            "F-C": 0.90,
            "C-F": 0.88,
            "C": 0.86,
        }
        _sp_raw *= _SP_POS_SCALE.get(pos, 0.97)

        # Strong physical drag for true jumbo centers.
        if is_center and height >= 83.0:
            _sp_raw *= 0.64 + 0.36 * (0.65 * _sp_pace + 0.35 * _sp_size)
        if is_center and height >= 84.0 and weight >= 270.0 and _sp_off_poss < 0.30:
            _sp_raw = min(_sp_raw, 0.28)

        # Heavy wings/guards and older jumbo players lose top-end speed.
        if (not is_big) and height >= 78.0 and weight >= 220.0:
            _sp_raw *= 0.90 + 0.10 * _sp_pace
        if age >= 35.0 and height >= 78.0 and weight >= 220.0:
            _sp_raw *= 0.88 + 0.12 * (1.0 - self._norm(age, 34.0, 40.0))

        _sp_opp = 0.45 * _sp_pace + 0.30 * _sp_off_poss + 0.15 * _sp_size + 0.10 * _sp_def_chase
        _sp_gate = 1.0 if _sp_opp >= 0.18 else (0.62 + 0.38 * max(0.0, _sp_opp / 0.18))

        _sp_value = 100.0 * _sp_raw * _sp_gate * (0.88 + 0.12 * _sp_reliability)
        attrs["speed"] = max(0.0, min(100.0, _sp_value))

        # Speed With Ball should be anchored by both raw speed and handle.
        _swb_speed = self._norm(attrs.get("speed", 50.0), 18.0, 88.0)
        _swb_handle = self._norm(attrs.get("ball_handle", 50.0), 48.0, 94.0)
        _swb_creation_role = max(0.0, min(1.0, _swb_creation))
        _swb_transition_now = self._norm(raw_transition_p36, 0.30, 5.80)

        if is_guard:
            _swb_recalc_raw = (
                0.40 * _swb_speed
                + 0.36 * _swb_handle
                + 0.16 * _swb_creation_role
                + 0.08 * _swb_transition_now
            )
        elif is_big:
            _swb_recalc_raw = (
                0.50 * _swb_speed
                + 0.24 * _swb_handle
                + 0.18 * _swb_creation_role
                + 0.08 * _swb_transition_now
            )
        else:  # wing
            _swb_recalc_raw = (
                0.45 * _swb_speed
                + 0.31 * _swb_handle
                + 0.16 * _swb_creation_role
                + 0.08 * _swb_transition_now
            )

        _swb_recalc_opp = 0.45 * _swb_handle + 0.30 * _swb_creation_role + 0.25 * _swb_speed
        _swb_recalc_gate = 1.0 if _swb_recalc_opp >= 0.22 else (0.54 + 0.46 * max(0.0, _swb_recalc_opp / 0.22))
        _swb_recalc_value = 100.0 * _swb_recalc_raw * _swb_recalc_gate

        # Keep SWB physically tethered to straight-line speed, with handle-based room.
        _swb_speed_delta_cap = 2.0 + 8.0 * self._norm(attrs.get("ball_handle", 50.0), 60.0, 95.0)
        _swb_recalc_value = min(_swb_recalc_value, attrs.get("speed", 50.0) + _swb_speed_delta_cap)

        # Strong lead creators should not sit far below their speed lane.
        if is_guard and attrs.get("ball_handle", 0.0) >= 82.0 and _swb_creation_role >= 0.55:
            _swb_recalc_value = max(_swb_recalc_value, attrs.get("speed", 50.0) - 8.0)

        # Low-handle bigs should not carry high dribble speed.
        if is_big and attrs.get("ball_handle", 0.0) <= 60.0:
            _swb_recalc_value = min(_swb_recalc_value, attrs.get("speed", 50.0) - 10.0)

        attrs["speed_with_ball"] = max(0.0, min(100.0, _swb_recalc_value))

        # ── Agility ───────────────────────────────────────────────────
        # Agility = multidirectional quickness, pivot control, and reaction.
        # Target weighting:
        #   35% speed
        #   25% ball handle
        #   25% explosiveness/first-step proxy
        #   15% body nimbleness (height/weight profile)
        _ag_pos = pos.replace(" ", "")
        _ag_is_guard = ("PG" in _ag_pos) or ("SG" in _ag_pos) or (_ag_pos == "G") or ("G-" in _ag_pos) or ("-G" in _ag_pos)
        _ag_is_big = ("C" in _ag_pos) or ("PF" in _ag_pos)
        _ag_is_wing = (not _ag_is_guard) and (not _ag_is_big)

        _ag_speed = self._norm(attrs.get("speed", 50.0), 20.0, 90.0)
        _ag_handle = self._norm(attrs.get("ball_handle", 50.0), 45.0, 95.0)
        _ag_swb = self._norm(attrs.get("speed_with_ball", 50.0), 20.0, 88.0)
        _ag_vertical = self._norm(attrs.get("vertical", 50.0), 35.0, 95.0)
        _ag_explosive = 0.45 * _ag_swb + 0.35 * _ag_vertical + 0.20 * _ag_speed
        _ag_body = (
            0.58 * (1.0 - self._norm(height, 72, 84))
            + 0.42 * (1.0 - self._norm(weight, 170, 290))
        )
        _ag_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _ag_base = (
            0.35 * _ag_speed
            + 0.25 * _ag_handle
            + 0.25 * _ag_explosive
            + 0.15 * _ag_body
        )

        if _ag_is_guard:
            _ag_raw = 0.78 * _ag_base + 0.10 * _ag_handle + 0.07 * _ag_explosive + 0.05 * _ag_body
        elif _ag_is_big:
            _ag_raw = 0.76 * _ag_base + 0.09 * _ag_speed + 0.08 * _ag_explosive + 0.07 * _ag_body
        else:  # wing
            _ag_raw = 0.78 * _ag_base + 0.09 * _ag_speed + 0.08 * _ag_handle + 0.05 * _ag_explosive

        _ag_raw *= 1.18

        # Very large centers should remain limited in rapid direction changes.
        if _ag_is_big and height >= 83 and weight >= 250:
            _ag_raw *= 0.80 + 0.20 * _ag_speed
            _ag_raw = min(_ag_raw, 0.52 + 0.22 * _ag_explosive + 0.14 * _ag_speed)

        if is_center and height >= 84 and weight >= 270:
            _ag_raw = min(_ag_raw, 0.40)

        # Heavy high-usage creators are functional but should not land in
        # elite agility bands.
        if _ag_is_wing and height >= 79 and weight >= 225 and usage >= 33:
            _ag_raw *= 0.88 + 0.12 * _ag_swb

        if _ag_is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _ag_raw *= 0.84 + 0.16 * _ag_handle

        # Archetype floors.
        if _ag_is_guard and _ag_speed > 0.78 and _ag_handle > 0.72:
            _ag_raw = max(_ag_raw, 0.83)

        if _ag_is_guard and height <= 75 and weight <= 200 and _ag_handle > 0.58:
            _ag_raw = max(_ag_raw, 0.72)

        if _ag_is_wing and _ag_speed > 0.62 and _ag_explosive > 0.58:
            _ag_raw = max(_ag_raw, 0.70)

        if _ag_is_big and height <= 83 and weight <= 255 and _ag_explosive > 0.56:
            _ag_raw = max(_ag_raw, 0.56)

        _ag_opp = 0.36 * _ag_speed + 0.28 * _ag_handle + 0.24 * _ag_explosive + 0.12 * _ag_body
        _ag_gate = 1.0 if _ag_opp >= 0.22 else (0.56 + 0.44 * max(0.0, _ag_opp / 0.22))

        _ag_value = 100.0 * _ag_raw * _ag_gate * (0.84 + 0.16 * _ag_reliability)
        _ag_value += 6.0

        if _ag_is_big and height >= 83 and weight >= 250:
            _ag_value = min(_ag_value, 58.0 + 12.0 * _ag_explosive + 9.0 * _ag_speed)

        if _ag_is_guard and height >= 78 and weight >= 220 and usage >= 30:
            _ag_value = min(_ag_value, 68.0 + 11.0 * _ag_handle + 9.0 * _ag_speed)

        attrs["agility"] = max(0.0, min(100.0, _ag_value))

        # ── Strength ──────────────────────────────────────────────────
        # Strength = ability to hold/establish physical position, absorb contact,
        # and win body-up battles.
        _str_pos = str(pos).upper()
        _str_is_guard = _str_pos in {"PG", "SG", "G", "PG-SG", "SG-PG", "G-F", "F-G"}
        _str_is_big = _str_pos in {"PF", "C", "F-C", "C-F", "PF-C", "C-PF"}
        _str_is_wing = not _str_is_guard and not _str_is_big

        _str_mass = (
            0.58 * self._norm(weight, 180, 300)
            + 0.26 * self._norm(height, 73, 85)
            + 0.16 * size_big
        )
        _str_build = (
            0.55 * self._norm(height, 74, 85)
            + 0.30 * size_big
            + 0.15 * self._norm(weight, 190, 295)
        )
        _str_anchor = self._norm(attrs.get("interior_defense", 50.0), 42.0, 92.0)
        _str_post = self._norm(attrs.get("post_control", 50.0), 35.0, 92.0)
        _str_draw = self._norm(attrs.get("draw_foul", 50.0), 40.0, 92.0)
        _str_sdunk = self._norm(attrs.get("standing_dunk", 50.0), 25.0, 95.0)
        _str_off_foul_drawn = self._norm(offensive_fouls_drawn_per36, 0.0, 0.35)
        _str_post_burden = self._norm(post_burden_p36, 0.15, 5.0)
        _str_aggr = (
            0.65 * self._norm(pf_per36, 1.6, 4.9)
            + 0.35 * _str_off_foul_drawn
        )
        _str_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _str_contact = (
            0.31 * _str_anchor
            + 0.22 * _str_post
            + 0.17 * _str_draw
            + 0.15 * _str_sdunk
            + 0.08 * _str_post_burden
            + 0.07 * _str_aggr
        )

        if _str_is_big:
            _str_arch = 0.64 + 0.24 * size_big + 0.12 * _str_contact
        elif _str_is_wing:
            _str_arch = 0.46 + 0.22 * size_big + 0.17 * _str_contact + 0.15 * self._norm(weight, 205, 255)
        else:  # guard
            _str_arch = 0.33 + 0.26 * self._norm(weight, 180, 225) + 0.16 * _str_contact + 0.12 * self._norm(height, 73, 79)

        _str_arch = max(0.0, min(1.0, _str_arch))

        # Target weighting:
        # 45% mass, 20% height/build, 15% archetype lane, 20% contact/aggression.
        _str_raw = (
            0.45 * _str_mass
            + 0.20 * _str_build
            + 0.15 * _str_arch
            + 0.20 * _str_contact
        )

        _str_raw *= 1.06

        # Light guards without physical playstyle should not drift upward.
        if _str_is_guard and weight < 190 and _str_draw < 0.45 and _str_post < 0.30:
            _str_raw *= 0.88 + 0.12 * _str_mass

        # Huge centers naturally maintain a very high strength floor.
        if is_center and weight >= 270:
            _str_raw = max(_str_raw, 0.88)

        # Physical combo bigs / wings.
        if (_str_is_big or _str_is_wing) and weight >= 240 and _str_anchor > 0.62:
            _str_raw = max(_str_raw, 0.78)

        # Powerful wings who consistently win contact and hold position.
        if _str_is_wing and weight >= 225 and height >= 78 and (_str_draw > 0.40 or dreb_pg >= 5.0):
            _str_raw = max(_str_raw, 0.60)

        # Jumbo primary wings with real glass/size should sit above average.
        if _str_is_wing and weight >= 230 and dreb_pg >= 6.0:
            _str_raw = max(_str_raw, 0.64)

        # Heavy contact wings (power-driver archetypes) should not grade as average.
        if _str_is_wing and weight >= 260 and (_str_draw >= 0.68 or _str_post >= 0.46):
            _str_raw = max(_str_raw, 0.76)

        # Powerful guards (Jrue-style) with real frame and contact tolerance.
        if _str_is_guard and weight >= 205 and _str_draw > 0.42:
            _str_raw = max(_str_raw, 0.58)

        if _str_is_guard and weight >= 215:
            _str_raw = max(_str_raw, 0.50)

        _str_value = 100.0 * _str_raw * (0.85 + 0.15 * _str_reliability)
        _str_value += 2.0
        attrs["strength"] = max(0.0, min(100.0, _str_value))

        # ── Vertical ──────────────────────────────────────────────────
        # Vertical = jump lift and explosion; weighted by athletic base,
        # contact strength, height leverage, and role/archetype context.
        _vt_ddunk = self._norm(attrs.get("driving_dunk", 50.0), 30.0, 95.0)
        _vt_sdunk = self._norm(attrs.get("standing_dunk", 50.0), 25.0, 95.0)
        _vt_blk = self._norm(blk_per36, 0.10, 2.60)
        _vt_oreb = self._norm(oreb_pg, 0.30, 4.20)
        _vt_dreb = self._norm(dreb_pg, 1.20, 11.0)
        _vt_spd = self._norm(attrs.get("speed", 50.0), 20.0, 85.0)
        _vt_agi = self._norm(attrs.get("agility", 50.0), 20.0, 88.0)
        _vt_swb = self._norm(attrs.get("speed_with_ball", 50.0), 20.0, 88.0)
        _vt_putback = self._norm(putback_rate, 0.0, 0.16)
        _vt_rim = self._norm(rim_pressure, 0.16, 0.75)
        _vt_strength = self._norm(attrs.get("strength", 50.0), 35.0, 95.0)
        _vt_height_norm = max(0.0, min(1.0, (height - 72.0) / (84.0 - 72.0)))
        _vt_height = 0.50 + 0.50 * _vt_height_norm
        _vt_age = age_phys
        _vt_reliability = (
            0.55 * self._norm(min_pg, 12.0, 35.0)
            + 0.45 * self._norm(gp, 22.0, 82.0)
        )

        _vt_roles = {
            token.strip().upper()
            for token in str(pos).replace("/", "-").split("-")
            if token.strip()
        }
        _vt_is_guard = bool(_vt_roles & {"PG", "SG", "G"})
        _vt_is_wing = bool(_vt_roles & {"SF", "F"})
        _vt_is_big = bool(_vt_roles & {"PF", "C"})

        if _vt_is_big:
            _vt_archetype_base = 1.0
        elif _vt_is_wing and _vt_is_guard:
            _vt_archetype_base = 0.83
        elif _vt_is_wing:
            _vt_archetype_base = 0.85
        else:
            _vt_archetype_base = 0.80

        # Keep the role context ranking while preventing role from overwhelming
        # true explosion for elite guard/wing athletes.
        _vt_archetype = 0.75 + 0.25 * _vt_archetype_base

        _vt_explosive = (
            0.30 * _vt_spd
            + 0.26 * _vt_agi
            + 0.14 * _vt_swb
            + 0.20 * _vt_ddunk
            + 0.10 * _vt_sdunk
        )
        _vt_impact = (
            0.34 * _vt_blk
            + 0.24 * _vt_oreb
            + 0.18 * _vt_dreb
            + 0.14 * _vt_putback
            + 0.10 * _vt_rim
        )
        _vt_athleticism = 0.70 * _vt_explosive + 0.30 * _vt_impact

        # Target weighting: 40% athleticism, 20% strength,
        # 20% height leverage, 20% role/archetype context.
        _vt_raw = (
            0.40 * _vt_athleticism
            + 0.20 * _vt_strength
            + 0.20 * _vt_height
            + 0.20 * _vt_archetype
        )

        _vt_raw *= 1.05
        _vt_raw *= 0.94 + 0.06 * _vt_age

        # Elite pop lanes.
        if (not _vt_is_big) and _vt_ddunk >= 0.74 and _vt_explosive >= 0.74:
            _vt_raw = max(_vt_raw, 0.74)

        if (not _vt_is_big) and _vt_ddunk >= 0.52 and _vt_explosive >= 0.70:
            _vt_raw = max(_vt_raw, 0.77)

        if (not _vt_is_big) and _vt_ddunk >= 0.48 and _vt_rim >= 0.30 and _vt_explosive >= 0.68:
            _vt_raw = max(_vt_raw, 0.76)

        if (not _vt_is_big) and _vt_ddunk >= 0.34 and _vt_rim >= 0.22 and _vt_explosive >= 0.68:
            _vt_raw = max(_vt_raw, 0.79)

        if (not _vt_is_big) and _vt_rim >= 0.28 and _vt_explosive >= 0.68:
            _vt_raw = max(_vt_raw, 0.81)

        if _vt_is_guard and _vt_ddunk >= 0.50 and _vt_explosive >= 0.70:
            _vt_raw = max(_vt_raw, 0.75)

        if _vt_is_guard and _vt_spd >= 0.72 and _vt_agi >= 0.78 and _vt_ddunk >= 0.44:
            _vt_raw = max(_vt_raw, 0.77)

        if _vt_is_guard and _vt_spd >= 0.70 and _vt_agi >= 0.76 and _vt_ddunk >= 0.40:
            _vt_raw = max(_vt_raw, 0.83)

        if _vt_is_guard and _vt_spd >= 0.70 and _vt_agi >= 0.76 and _vt_explosive >= 0.70:
            _vt_raw = max(_vt_raw, 0.82)

        if _vt_is_guard and _vt_spd >= 0.84 and _vt_agi >= 0.84 and _vt_rim >= 0.22:
            _vt_raw = max(_vt_raw, 0.82)

        if _vt_is_wing and _vt_ddunk >= 0.68 and _vt_explosive >= 0.70:
            _vt_raw = max(_vt_raw, 0.70)

        if _vt_is_wing and weight >= 255 and _vt_ddunk >= 0.68 and _vt_strength >= 0.72:
            _vt_raw = max(_vt_raw, 0.74)

        if _vt_is_big and _vt_blk >= 0.66 and _vt_sdunk >= 0.66:
            _vt_raw = max(_vt_raw, 0.68)

        # Ground-bound penalty for very heavy, low-pop bigs.
        if _vt_is_big and weight >= 270 and _vt_explosive <= 0.52 and _vt_blk <= 0.46:
            _vt_raw *= 0.88 + 0.12 * _vt_sdunk

        # Low-air guards should not float up from speed alone.
        if _vt_is_guard and _vt_ddunk <= 0.36 and _vt_blk <= 0.16:
            _vt_raw *= 0.90 + 0.10 * _vt_agi

        # High-usage creators with limited lift signals should not grade as high-end leapers.
        if _vt_is_guard and _vt_ddunk <= 0.44 and _vt_rim <= 0.22 and _vt_oreb <= 0.20:
            _vt_raw *= 0.82 + 0.18 * _vt_explosive

        _vt_value = 100.0 * _vt_raw * (0.85 + 0.15 * _vt_reliability)

        # Give true pop athletes a small late-stage bump so reliability
        # variance does not flatten elite jump profiles.
        if _vt_is_guard and _vt_explosive >= 0.70 and _vt_ddunk >= 0.42 and (_vt_spd >= 0.70 or _vt_agi >= 0.76):
            _vt_value += 4.0
        elif (not _vt_is_big) and _vt_explosive >= 0.68 and _vt_ddunk >= 0.48:
            _vt_value += 2.5

        _vt_value += 1.5
        attrs["vertical"] = max(0.0, min(100.0, _vt_value))

        # ── Stamina ───────────────────────────────────────────────────
        # Stamina = sustained conditioning under movement and workload stress.
        # Target weighting: 50% endurance, 20% speed/agility demand,
        # 15% body type (heavier -> more fatigue), 15% workload burden.
        _st_minutes = self._norm(min_pg, 14.0, 37.0)
        _st_availability = self._norm(gp, 12.0, 82.0)
        _st_total_load = self._norm(min_pg * gp, 500.0, 2850.0)
        _st_transition = self._norm(raw_transition_p36, 0.5, 5.5)
        _st_cuts = self._norm(raw_cuts_p36, 0.2, 2.7)
        _st_roll = self._norm(raw_pnr_roll_p36, 0.0, 4.0)
        _st_usage = self._norm(usage, 16.0, 36.0)
        _st_speed = self._norm(attrs.get("speed", 50.0), 20.0, 85.0)
        _st_agility = self._norm(attrs.get("agility", 50.0), 20.0, 88.0)
        _st_body_fatigue = self._norm(weight, 180.0, 290.0)
        _st_age = 1.0 - 0.18 * self._norm(age, 33.0, 40.0)
        _st_age = max(0.74, min(1.0, _st_age))

        _st_endurance = (
            0.44 * _st_minutes
            + 0.24 * _st_availability
            + 0.22 * _st_total_load
            + 0.10 * motor_profile
        )

        # Movement demand is a mixed signal: very quick players can sustain
        # pace when conditioned, but high-speed play also taxes stamina.
        _st_mobility = 0.56 * _st_speed + 0.44 * _st_agility
        _st_movement_demand = (
            0.46 * _st_transition
            + 0.32 * _st_cuts
            + 0.22 * _st_roll
        )
        _st_speed_agility_component = 0.68 * _st_mobility + 0.32 * (1.0 - _st_movement_demand)

        _st_workload_penalty = (
            0.54 * _st_usage
            + 0.30 * self._norm(min_pg, 30.0, 38.0)
            + 0.16 * _st_movement_demand
        )

        _st_raw = (
            0.50 * _st_endurance
            + 0.20 * _st_speed_agility_component
            + 0.15 * (1.0 - _st_body_fatigue)
            + 0.15 * (1.0 - _st_workload_penalty)
        )

        _st_raw *= _st_age
        _st_raw += 0.02

        # Ironman minute-eaters and two-way engines.
        if gp >= 64 and min_pg >= 34.0:
            _st_raw = max(_st_raw, 0.84)
        elif gp >= 56 and min_pg >= 31.0:
            _st_raw = max(_st_raw, 0.79)

        if usage >= 31.0 and min_pg >= 34.0 and gp >= 55:
            _st_raw = max(_st_raw, 0.80)

        if transition_pos >= 3.5 and min_pg >= 31.0 and gp >= 50:
            _st_raw = max(_st_raw, 0.77)

        # Heavy, high-workload bigs can hold good stamina, but not top guard bands.
        if is_big and weight >= 265.0 and _st_usage >= 0.55 and min_pg >= 30.0:
            _st_raw = min(_st_raw, 0.84)

        # Availability constraints: keep penalties, but avoid over-punishing
        # partial-season high-minute stars.
        if gp <= 10:
            _st_raw *= 0.84 + 0.16 * self._norm(min_pg, 16.0, 32.0)
        elif gp <= 20:
            _st_raw *= 0.92 + 0.08 * self._norm(min_pg, 18.0, 33.0)

        if 14 <= gp <= 35 and min_pg >= 30.0 and usage >= 30.0:
            _st_raw = max(_st_raw, 0.64)

        _st_reliability = 0.45 * self._norm(gp, 10.0, 82.0) + 0.55 * self._norm(min_pg, 14.0, 36.0)
        _st_value = 100.0 * _st_raw * (0.87 + 0.13 * _st_reliability)
        attrs["stamina"] = max(0.0, min(100.0, _st_value))

        # ══════════════════════════════════════════════════════════════
        # META
        # ══════════════════════════════════════════════════════════════

        # ── Intangibles ───────────────────────────────────────────────
        # Keep intangibles neutral by default; use in-game edits for OVR tuning.
        attrs["intangibles"] = 60.0

        # ── Hustle ────────────────────────────────────────────────────
        # Hustle is active effort and activity volume, independent of intangibles.
        # Target weighting: 40% activity, 25% stamina, 20% mobility, 15% role.
        _hs_transition = self._norm(raw_transition_p36, 0.5, 6.2)
        _hs_cuts = self._norm(raw_cuts_p36, 0.2, 3.0)
        _hs_roll = self._norm(raw_pnr_roll_p36, 0.0, 4.5)
        _hs_glass = self._norm(oreb_pg + 0.45 * dreb_pg, 2.5, 12.5)
        _hs_events = self._norm(stl_per36 + 0.85 * blk_per36, 0.8, 4.8)
        _hs_second = self._norm(second_chance_off_poss_rate, 0.03, 0.20)
        _hs_recover = self._norm(blocks_recovered_pct, 0.20, 0.75)
        _hs_loose = self._norm(loose_ball_fouls_per36, 0.0, 1.20)
        _hs_loose_drawn = self._norm(loose_ball_fouls_drawn_per36, 0.0, 1.10)
        _hs_engage = def_engagement
        _hs_stamina = self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)
        _hs_speed = self._norm(attrs.get("speed", 50.0), 20.0, 85.0)
        _hs_agility = self._norm(attrs.get("agility", 50.0), 20.0, 88.0)

        _hs_roles = {
            token.strip().upper()
            for token in str(pos).replace("/", "-").split("-")
            if token.strip()
        }
        _hs_is_guard = bool(_hs_roles & {"PG", "SG", "G"})
        _hs_is_wing = bool(_hs_roles & {"SF", "F"})
        _hs_is_big = bool(_hs_roles & {"PF", "C"})

        if _hs_is_big and not (_hs_is_guard or _hs_is_wing):
            _hs_role_factor = 0.90
        elif _hs_is_guard or _hs_is_wing:
            _hs_role_factor = 1.00
        else:
            _hs_role_factor = 0.95

        _hs_activity = (
            0.08 * _hs_transition
            + 0.05 * _hs_cuts
            + 0.04 * _hs_roll
            + 0.20 * _hs_glass
            + 0.20 * _hs_events
            + 0.12 * _hs_second
            + 0.09 * _hs_recover
            + 0.08 * _hs_loose
            + 0.08 * _hs_loose_drawn
            + 0.06 * _hs_engage
        )

        # Some feeds have sparse playtype hustle actions; in those cases,
        # avoid underestimating high-motor players.
        if _hs_transition <= 0.02 and _hs_cuts <= 0.02 and _hs_roll <= 0.02:
            _hs_activity = 0.82 * _hs_activity + 0.18 * (
                0.35 * _hs_events
                + 0.25 * _hs_glass
                + 0.20 * _hs_loose
                + 0.10 * _hs_loose_drawn
                + 0.10 * _hs_engage
            )

        _hs_mobility = 0.55 * _hs_speed + 0.45 * _hs_agility
        _hs_reliability = 0.55 * self._norm(min_pg, 12.0, 36.0) + 0.45 * self._norm(gp, 18.0, 82.0)

        _hs_raw = (
            0.40 * _hs_activity
            + 0.25 * _hs_stamina
            + 0.20 * _hs_mobility
            + 0.15 * _hs_role_factor
        )

        _hs_raw *= 0.90 + 0.10 * _hs_reliability
        _hs_raw += 0.10

        # High-motor lanes.
        if _hs_activity >= 0.72 and _hs_stamina >= 0.72:
            _hs_raw = max(_hs_raw, 0.80)

        if _hs_is_big and _hs_glass >= 0.72 and _hs_events >= 0.55:
            _hs_raw = max(_hs_raw, 0.78)

        if (_hs_is_guard or _hs_is_wing) and _hs_transition >= 0.68 and _hs_engage >= 0.60:
            _hs_raw = max(_hs_raw, 0.79)

        # Low-motor guardrails.
        if _hs_activity <= 0.35 and _hs_stamina <= 0.45:
            _hs_raw *= 0.90 + 0.10 * _hs_mobility

        attrs["hustle"] = max(0.0, min(100.0, 100.0 * _hs_raw))

        # ── Overall Durability ────────────────────────────────────────
        # Primary durability driver: multi-season games played vs possible.
        _dur_seasons = max(1.0, self._f(f, "durability_seasons_sampled", 1.0))
        _dur_played_total = self._f(f, "durability_games_played_total", gp)
        _dur_possible_total = self._f(f, "durability_games_possible_total", 82.0 * _dur_seasons)
        _dur_missed_total = self._f(f, "durability_games_missed_total", max(0.0, _dur_possible_total - _dur_played_total))
        _dur_availability_ratio = self._f(
            f,
            "durability_availability_ratio",
            (_dur_played_total / _dur_possible_total) if _dur_possible_total > 0 else 0.0,
        )

        _dur_history = max(0.0, min(1.0, _dur_availability_ratio))
        _dur_missed_penalty = self._norm(_dur_missed_total / max(_dur_seasons, 1.0), 8.0, 34.0)
        _dur_current = self._norm(gp, 20.0, 82.0)
        _dur_minutes = self._norm(min_pg, 14.0, 36.0)
        _dur_stamina = self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)

        # Blend long-window history with current-season availability so
        # recent durability gains are reflected faster.
        _dur_recent_history = 0.72 * _dur_history + 0.28 * _dur_current
        _dur_recent_history = max(0.0, min(1.0, _dur_recent_history))

        # Reliability signal used inside each tier band.
        _dur_reliability = (
            0.62 * _dur_recent_history
            + 0.18 * (1.0 - _dur_missed_penalty)
            + 0.10 * _dur_current
            + 0.06 * _dur_minutes
            + 0.04 * _dur_stamina
        )
        _dur_reliability = max(0.0, min(1.0, _dur_reliability))

        # Tiered durability bands to match requested ranges.
        # Elite: 90s, Strong: 80-90, Medium: 70-80, Low: low 70s.
        if _dur_recent_history >= 0.90:
            _dur_raw = 0.90 + 0.09 * self._norm(_dur_reliability, 0.82, 0.98)
        elif _dur_recent_history >= 0.80:
            _dur_raw = 0.80 + 0.10 * self._norm(_dur_reliability, 0.70, 0.92)
        elif _dur_recent_history >= 0.68:
            _dur_raw = 0.70 + 0.10 * self._norm(_dur_reliability, 0.56, 0.82)
        else:
            _dur_raw = 0.70 + 0.04 * self._norm(_dur_reliability, 0.42, 0.70)

        # Keep very low-availability players in the low-70s lane.
        if _dur_seasons >= 2 and _dur_history <= 0.55:
            _dur_raw = min(_dur_raw, 0.73)

        # Keep 1-year/limited samples from inflating to elite.
        if _dur_seasons < 2:
            _dur_raw = min(_dur_raw, 0.88)

        # Veteran availability exception: older stars with strong recent GP
        # should reach the strong tier even with a middling 5-year base.
        if age >= 35.0 and gp >= 68.0 and _dur_history >= 0.70:
            _dur_raw = max(_dur_raw, 0.80)

        attrs["overall_durability"] = max(0.0, min(100.0, 100.0 * _dur_raw))

        # ── Potential ─────────────────────────────────────────────────
        # Potential represents projected 2-4 year ceiling, not current impact.
        _pot_hustle = self._norm(attrs.get("hustle", 50.0), 55.0, 92.0)
        _pot_stamina = self._norm(attrs.get("stamina", 50.0), 55.0, 95.0)
        _pot_work = 0.56 * _pot_hustle + 0.44 * _pot_stamina

        _pot_off_foundation = (
            0.22 * self._norm(attrs.get("ball_handle", 50.0), 50.0, 92.0)
            + 0.22 * self._norm(attrs.get("pass_accuracy", 50.0), 50.0, 95.0)
            + 0.20 * self._norm(attrs.get("speed_with_ball", 50.0), 45.0, 92.0)
            + 0.20 * self._norm(attrs.get("three_point_shot", 50.0), 45.0, 95.0)
            + 0.16 * self._norm(attrs.get("mid_range_shot", 50.0), 45.0, 92.0)
        )
        _pot_def_foundation = (
            0.40 * self._norm(attrs.get("perimeter_defense", 50.0), 45.0, 95.0)
            + 0.34 * self._norm(attrs.get("interior_defense", 50.0), 40.0, 95.0)
            + 0.26 * self._norm(attrs.get("help_defense_iq", 50.0), 45.0, 95.0)
        )
        _pot_physical = (
            0.30 * self._norm(attrs.get("speed", 50.0), 35.0, 92.0)
            + 0.26 * self._norm(attrs.get("agility", 50.0), 35.0, 92.0)
            + 0.24 * self._norm(attrs.get("vertical", 50.0), 35.0, 95.0)
            + 0.20 * self._norm(attrs.get("strength", 50.0), 35.0, 95.0)
        )

        if is_big:
            _pot_foundation = 0.24 * _pot_off_foundation + 0.40 * _pot_def_foundation + 0.36 * _pot_physical
        elif is_guard:
            _pot_foundation = 0.46 * _pot_off_foundation + 0.20 * _pot_def_foundation + 0.34 * _pot_physical
        else:
            _pot_foundation = 0.38 * _pot_off_foundation + 0.28 * _pot_def_foundation + 0.34 * _pot_physical

        _pot_current_impact = (
            0.42 * self._norm(attrs.get("offensive_consistency", 50.0), 55.0, 95.0)
            + 0.28 * self._norm(attrs.get("defensive_consistency", 50.0), 50.0, 95.0)
            + 0.30 * _pot_foundation
        )

        _pot_age_upside = self._norm(29.0 - age, 0.0, 11.0)
        _pot_growth_window = max(0.0, min(1.0, _pot_age_upside))
        _pot_headroom = max(0.0, min(1.0, 1.0 - _pot_current_impact))
        _pot_trajectory = (0.58 * _pot_foundation + 0.42 * _pot_work)

        _pot_raw = (
            0.52 * _pot_current_impact
            + 0.20 * _pot_foundation
            + 0.28 * (_pot_growth_window * _pot_trajectory)
        )

        # Young players with real skill/work indicators should carry visible upside.
        _pot_raw += 0.20 * _pot_headroom * _pot_growth_window * (0.45 + 0.55 * _pot_trajectory)

        # Age-related ceiling compression.
        if age >= 30:
            _pot_raw *= 0.90 - 0.06 * self._norm(age, 30.0, 38.0)
            _pot_raw = min(_pot_raw, 0.82)
        if age >= 34:
            _pot_raw *= 0.92

        # Prospect / star-upside lanes.
        if age <= 23 and usage >= 25.0 and ast_pg >= 7.0 and _pot_off_foundation >= 0.62:
            _pot_raw = max(_pot_raw, 0.90)

        if age <= 24 and usage >= 24.0 and _pot_off_foundation >= 0.64:
            _pot_raw = max(_pot_raw, 0.86)

        if age <= 25 and usage >= 28.0 and ast_pg >= 6.5 and _pot_off_foundation >= 0.58:
            _pot_raw = max(_pot_raw, 0.98)

        if age <= 24 and usage >= 30.0 and _pot_off_foundation >= 0.62:
            _pot_raw = max(_pot_raw, 0.96)

        if age <= 24 and is_big and (blk_per36 >= 1.6 or dreb_pg >= 8.5) and _pot_physical >= 0.64:
            _pot_raw = max(_pot_raw, 0.84)

        if age <= 21 and _pot_foundation >= 0.56:
            _pot_raw = max(_pot_raw, 0.82)

        if age <= 24 and _pot_foundation >= 0.60 and _pot_work >= 0.62:
            _pot_raw = max(_pot_raw, 0.85)

        if age <= 27 and _pot_current_impact >= 0.70 and _pot_foundation >= 0.64:
            _pot_raw = max(_pot_raw, 0.82)

        # Calibrate into 2K-like potential bands (future ceiling scale).
        _pot_raw = 0.48 + 0.52 * _pot_raw

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

            # Keep legacy downhill lift modest so rebuilt driving_dunk logic
            # remains the primary signal source.
            if driver_signal > 0.70 and (not is_center):
                boost = (driver_signal - 0.70) / 0.30
                attrs["driving_dunk"] += 4.0 + 10.0 * boost
                attrs["driving_layup"] += 4.0 + 8.0 * boost
                attrs["speed_with_ball"] += 3.0 + 7.0 * boost
                attrs["draw_foul"] += 2.0 + 6.0 * boost

            # Low-rim perimeter scorers should not carry inflated dunk ratings
            low_rim_signal = (
                0.45 * (1.0 - self._norm(rim_pressure, 0.12, 0.36))
                + 0.35 * (1.0 - self._norm(ra_vol_p36, 0.8, 4.0))
                + 0.20 * (1.0 - self._norm(raw_transition_p36, 0.4, 3.0))
            )
            low_rim_signal = max(0.0, min(1.0, low_rim_signal))
            if low_rim_signal > 0.58 and (not is_big):
                cut = (low_rim_signal - 0.58) / 0.42
                attrs["driving_dunk"] -= 10.0 + 14.0 * cut

            # High-volume perimeter guards/wings with weak downhill profile
            # should not retain high dunk ratings in legacy seasons.
            if (not is_big) and fg3a_rate >= 0.30 and driver_signal < 0.48:
                slope = max(0.0, min(1.0, (0.48 - driver_signal) / 0.48))
                attrs["driving_dunk"] -= 8.0 + 12.0 * slope

            # Elite shooting specialists
            if shooter_signal > 0.55:
                s_boost = (shooter_signal - 0.55) / 0.45
                attrs["three_point_shot"] += 4.0 + 9.0 * s_boost

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
            raw_frac = max(0.0, min(1.0, raw / 100.0))
            frac = raw_frac
            # Power-curve boost: stretches mid-range scores upward
            # so that raw 50 → ~60 attr, raw 70 → ~82 attr
            frac = frac ** 0.75
            scaled = 25 + frac * 74
            if attr == "offensive_consistency":
                attr_min, attr_max = 60, 90
            elif attr == "defensive_consistency":
                attr_min, attr_max = 50, 95
            elif attr == "draw_foul":
                attr_min, attr_max = 25, 95
            elif attr == "hands":
                attr_min, attr_max = 60, 95
                _hands_ast_p36 = ast_pg * 36.0 / max(min_pg, 1.0)
                _hands_is_guard = pos in {"PG", "SG", "G", "G-F", "F-G"}
                _hands_is_wing = pos in {"SG", "SF", "F", "G-F", "F-G"}
                _hands_is_big = pos in {"PF", "C", "F-C", "C-F"}
                _hands_creator = self._norm(_hands_ast_p36, 2.0, 10.0)
                _hands_rebound = self._norm(oreb_pg + 0.65 * dreb_pg, 2.5, 13.0)
                _hands_usage = self._norm(usage, 16.0, 34.0)
                _hands_signal = 0.50 * raw_frac + 0.28 * _hands_creator + 0.22 * _hands_rebound

                _hands_skilled_big = _hands_is_big and _hands_ast_p36 >= 6.0
                _hands_strong_wing = _hands_is_wing and usage >= 24.0 and (weight >= 210.0 or ast_pg >= 4.0 or dreb_pg >= 5.0)
                _hands_good_guard = _hands_is_guard and usage >= 24.0
                _hands_def_big = _hands_is_big and _hands_ast_p36 < 4.0
                _hands_raw_big = _hands_is_big and _hands_ast_p36 < 2.2 and usage < 18.0

                # Elite category should land in 90+.
                _hands_elite = _hands_skilled_big
                if _hands_elite:
                    _hands_elite_tier = max(0.0, min(1.0, (_hands_signal - 0.60) / 0.40))
                    scaled = max(scaled, 90.0 + 6.0 * _hands_elite_tier)
                    attr_min = max(attr_min, 90)
                    attr_max = min(attr_max, 96)
                else:
                    # Good category should sit in 80-88.
                    _hands_good = _hands_strong_wing or _hands_good_guard
                    if _hands_good:
                        _hands_good_rank = max(
                            0.0,
                            min(
                                1.0,
                                0.42 * _hands_creator
                                + 0.33 * _hands_usage
                                + 0.25 * _hands_rebound,
                            ),
                        )
                        _hands_good_tier = max(0.0, min(1.0, (_hands_good_rank - 0.30) / 0.55))
                        scaled = max(scaled, 80.0 + 8.0 * _hands_good_tier)
                        attr_min = max(attr_min, 80)
                        attr_max = min(attr_max, 88)

                # Defensive bigs should be mid, raw bigs lower.
                if _hands_def_big:
                    attr_min = max(attr_min, 70)
                    attr_max = min(attr_max, 79)
                if _hands_raw_big:
                    attr_min = max(60, attr_min - 5)
                    attr_max = min(attr_max, 74)
            else:
                attr_min, attr_max = 25, 99
            result[attr] = max(attr_min, min(attr_max, round(scaled)))

        # Vertical pop adjustment: athletic guards and true dunkers should
        # separate from low-lift creators after global scaling.
        _res_roles = {
            token.strip().upper()
            for token in str(pos).replace("/", "-").split("-")
            if token.strip()
        }
        _res_is_guard = bool(_res_roles & {"PG", "SG", "G"})
        _res_is_big = bool(_res_roles & {"PF", "C"})
        _potential_target = self._f(f, "potential_target", -1.0)

        _v_boost = 0
        if _res_is_guard and result.get("speed", 0) >= 78 and result.get("agility", 0) >= 84 and result.get("driving_dunk", 0) >= 60:
            _v_boost = 6
        elif (not _res_is_big) and result.get("driving_dunk", 0) >= 70:
            _v_boost = 5
        elif _res_is_guard and result.get("speed", 0) >= 74 and result.get("agility", 0) >= 82 and result.get("driving_dunk", 0) >= 54:
            _v_boost = 3

        if _v_boost > 0:
            result["vertical"] = max(25, min(99, result.get("vertical", 25) + _v_boost))

        # Potential ceiling adjustment: convert developmental upside into
        # practical 2K future-ceiling bands.
        _pot_boost = 0
        if age <= 25 and usage >= 28.0 and ast_pg >= 6.5:
            _pot_boost = 14
        elif age <= 24 and usage >= 28.0:
            _pot_boost = 11
        elif age <= 24 and usage >= 24.0:
            _pot_boost = 8
        elif age <= 22:
            _pot_boost = 7

        if age >= 30:
            _pot_boost = max(0, _pot_boost - 5)

        if _pot_boost > 0:
            result["potential"] = max(25, min(99, result.get("potential", 25) + _pot_boost))

        # Align generated potential to target roster scale.
        _pot_global = 7
        if age >= 34:
            _pot_global = 5
        result["potential"] = max(25, min(99, result.get("potential", 25) + _pot_global))

        # Explicit CSV target override (pipeline-injected).
        if _potential_target >= 0.0:
            result["potential"] = max(25, min(99, round(_potential_target)))

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
