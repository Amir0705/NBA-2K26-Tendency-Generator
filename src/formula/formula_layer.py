"""Formula layer: maps feature vectors to all 99 tendency values."""
from __future__ import annotations

from typing import Any


def scale(value: float, input_range: list, output_range: list) -> float:
    """Linear interpolation with clipping."""
    in_min, in_max = input_range
    out_min, out_max = output_range
    if in_max == in_min:
        return (out_min + out_max) / 2
    normalized = (value - in_min) / (in_max - in_min)
    normalized = max(0.0, min(1.0, normalized))
    return out_min + normalized * (out_max - out_min)


def _shape_close_distribution(
    dist_close: dict[str, float],
    min_side_total_share: float = 40.0,
    min_single_side_share: float = 10.0,
    max_middle_share: float = 55.0,
) -> dict[str, float]:
    """Return a stable left/middle/right percentage split (sums to 100).

    Goals:
    - prevent close middle from being chronically dominant,
    - keep left/right from collapsing to near-zero,
    - preserve observed player bias as much as possible.
    """
    left = max(0.0, float(dist_close.get("left", 33.3)))
    middle = max(0.0, float(dist_close.get("middle", 33.4)))
    right = max(0.0, float(dist_close.get("right", 33.3)))

    total = left + middle + right
    if total <= 0:
        return {"left": 33.3, "middle": 33.4, "right": 33.3}

    left = left / total * 100.0
    middle = middle / total * 100.0
    right = right / total * 100.0

    # Preserve observed left/right bias ratio while limiting middle dominance.
    raw_side_total = left + right
    if raw_side_total <= 0:
        left_ratio = 0.5
    else:
        left_ratio = left / raw_side_total

    middle = min(middle, max_middle_share)
    side_total = max(min_side_total_share, 100.0 - middle)
    side_total = min(side_total, 100.0)
    middle = 100.0 - side_total

    left = side_total * left_ratio
    right = side_total - left

    # Prevent one side from collapsing to near-zero without forcing symmetry.
    if left < min_single_side_share:
        deficit = min_single_side_share - left
        left = min_single_side_share
        right = max(0.0, right - deficit)
    if right < min_single_side_share:
        deficit = min_single_side_share - right
        right = min_single_side_share
        left = max(0.0, left - deficit)

    # Final exact normalization.
    final_total = left + middle + right
    if final_total <= 0:
        return {"left": 33.3, "middle": 33.4, "right": 33.3}
    return {
        "left": left / final_total * 100.0,
        "middle": middle / final_total * 100.0,
        "right": right / final_total * 100.0,
    }


def _shape_five_zone_distribution(
    dist: dict[str, float],
    gamma: float = 1.25,
) -> dict[str, float]:
    """Return a normalized 5-zone percentage split after gamma shaping."""
    keys = ["left", "left_center", "center", "right_center", "right"]
    raw = [max(0.0, float(dist.get(k, 20.0))) for k in keys]
    total = sum(raw)
    if total <= 0:
        return {k: 20.0 for k in keys}

    shares = [v / total for v in raw]
    shaped = [max(1e-6, s) ** gamma for s in shares]
    shaped_total = sum(shaped)
    if shaped_total <= 0:
        return {k: 20.0 for k in keys}

    return {k: shaped[i] / shaped_total * 100.0 for i, k in enumerate(keys)}


class FormulaLayer:
    """Deterministic rule-based tendency calculator."""

    POSITION_PROFILES: dict[str, dict[str, float]] = {
        "PG": {"post_scale": 0.20, "drive_boost": 1.15, "block_scale": 0.4, "dribble_boost": 1.2},
        "SG": {"post_scale": 0.20, "drive_boost": 1.10, "block_scale": 0.5, "dribble_boost": 0.85},
        "SF": {"post_scale": 0.50, "drive_boost": 1.00, "block_scale": 0.7, "dribble_boost": 0.95},
        "PF": {"post_scale": 0.80, "drive_boost": 0.85, "block_scale": 0.9, "dribble_boost": 0.6},
        "C":  {"post_scale": 1.00, "drive_boost": 0.65, "block_scale": 1.0, "dribble_boost": 0.3},
    }

    def generate(self, features: dict[str, Any]) -> dict[str, float]:
        """
        Generate all 99 tendency values from feature vector.

        Returns canonical_name → float (0–100, pre-cap).
        """
        f = features
        pos = f.get("position", "SF")
        profile = self.POSITION_PROFILES.get(pos, self.POSITION_PROFILES["SF"])
        post_factor = profile["post_scale"]
        drive_boost = profile["drive_boost"]
        dribble_boost = profile["dribble_boost"]

        # Convenience accessors
        usg = f.get("usg_pct_proxy", 0.18)
        fga_p36 = f.get("fga_per36", 10.0)
        fga_pg = f.get("fga_per_game", max(1.0, fga_p36 * 0.75))
        fg3a_rate = f.get("fg3a_rate", 0.30)
        fta_rate = f.get("fta_rate", 0.25)
        midrange_attempts = f.get("midrange_attempts", 0.0)
        catch_and_shoot_three_rate = f.get("catch_and_shoot_three_rate", fg3a_rate * 0.55)
        pull_up_three_rate = f.get("pull_up_three_rate", fg3a_rate * 0.45)
        unassisted_two_rate = f.get("unassisted_two_rate", 0.10)
        assisted_2pt_pct = f.get("assisted_2pt_pct", 45.0)
        assisted_3pt_pct = f.get("assisted_3pt_pct", 75.0)
        putback_rate = f.get("putback_rate", 0.0)
        shooting_fouls_drawn_pct = f.get("shooting_fouls_drawn_pct", 0.0)
        live_ball_turnover_pct = f.get("live_ball_turnover_pct", 12.0)
        second_chance_off_poss_rate = f.get("second_chance_off_poss_rate", 0.08)
        seconds_per_poss_off = f.get("seconds_per_poss_off", 14.0)
        bad_pass_turnovers_p36 = f.get("bad_pass_turnovers_per36", 1.2)
        lost_ball_turnovers_p36 = f.get("lost_ball_turnovers_per36", 1.0)
        offensive_fouls_p36 = f.get("offensive_fouls_per36", 0.4)
        loose_ball_fouls_p36 = f.get("loose_ball_fouls_per36", 0.3)
        shooting_fouls_p36 = f.get("shooting_fouls_per36", 1.0)
        offensive_fouls_drawn_p36 = f.get("offensive_fouls_drawn_per36", 0.2)
        loose_ball_fouls_drawn_p36 = f.get("loose_ball_fouls_drawn_per36", 0.2)
        blocks_recovered_pct = f.get("blocks_recovered_pct", 45.0)
        ast_p36 = f.get("ast_per36", 3.0)
        pts_p36 = f.get("pts_per36", 15.0)
        stl_p36 = f.get("stl_per36", 1.0)
        blk_p36 = f.get("blk_per36", 0.5)
        pf_p36 = f.get("pf_per36", 2.5)
        oreb_pct = f.get("oreb_pct_proxy", 0.1)

        # Tracking / hustle stats (use -1.0 sentinel; fall back to position defaults)
        hustle_loose_balls = f.get("hustle_loose_balls_pg", -1.0)
        hustle_charges = f.get("hustle_charges_drawn_pg", -1.0)

        # Physical attributes
        height_inches = f.get("height_inches", 78)    # 6-6 = 78 in default
        weight_lbs = f.get("weight_lbs", 220)

        # Shooting efficiency
        fg_pct = f.get("fg_pct", 0.45)
        fg3_pct = f.get("fg3_pct", 0.35)
        ts_pct = f.get("ts_pct", 0.55)

        # Playmaking discipline
        ast_to_tov = f.get("ast_to_tov", 1.5)

        # League percentile ranks (0.0 – 1.0)
        pctile_pts = f.get("pctile_pts", 0.5)

        # Zone rates
        zra = f.get("zone_fga_rate_ra", 0.1)
        zpaint = f.get("zone_fga_rate_paint", 0.1)
        zmid_l = f.get("zone_fga_rate_mid_left", 0.0)
        zmid_c = f.get("zone_fga_rate_mid_center", 0.0)
        zmid_r = f.get("zone_fga_rate_mid_right", 0.0)
        zmid_total = zmid_l + zmid_c + zmid_r

        # Sub-zone distributions
        dist_close = f.get("sub_zone_distribution_close", {"left": 33.3, "middle": 33.4, "right": 33.3})
        dist_mid = f.get("sub_zone_distribution_mid", {
            "left": 20.0, "left_center": 20.0, "center": 20.0, "right_center": 20.0, "right": 20.0
        })
        dist_three = f.get("sub_zone_distribution_three", {
            "left": 20.0, "left_center": 20.0, "center": 20.0, "right_center": 20.0, "right": 20.0
        })

        t: dict[str, float] = {}

        # ---------------------------------------------------------------
        # Category A: Core Shooting
        # ---------------------------------------------------------------
        shooting_skill = scale(fg3a_rate + zmid_total, [0.05, 0.50], [0.6, 1.1])
        pbp_creation = scale(unassisted_two_rate, [0.04, 0.24], [0.85, 1.15])
        shot = (
            0.6 * scale(usg, [0.10, 0.35], [20, 75])
            + 0.4 * scale(fga_p36, [3, 25], [15, 75])
        ) * shooting_skill * pbp_creation
        _pts_pctile_boost = scale(pctile_pts, [0.3, 0.8], [0.95, 1.05])
        t["shot"] = min(shot * _pts_pctile_boost, 75.0)

        t["shot_under_basket"] = scale(zra, [0.0, 0.5], [0, 60])
        # Close should track rim pressure and contact drawing more than paint-share,
        # since current PBP zone splits can under-report non-rim paint attempts.
        close_mix = (
            0.70 * zra
            + 0.20 * scale(fta_rate, [0.10, 0.50], [0.0, 1.0])
            + 0.10 * scale(unassisted_two_rate, [0.04, 0.24], [0.0, 1.0])
            - 0.15 * scale(fg3a_rate, [0.15, 0.50], [0.0, 1.0])
        )
        t["shot_close"] = scale(close_mix, [0.05, 0.45], [5, 60])

        _mid_zone_volume = scale(zmid_total, [0.10, 0.42], [0, 48])
        _mid_attempt_rate = max(0.0, min(1.0, float(midrange_attempts) / max(float(fga_pg), 1.0)))
        _mid_attempt_mult = scale(_mid_attempt_rate, [0.12, 0.40], [0.85, 1.20])
        _mid_eff_mult = scale(fg_pct, [0.42, 0.56], [0.90, 1.08])
        _mid_creator_mult = scale(unassisted_two_rate, [0.04, 0.24], [0.92, 1.10])
        _mid_three_penalty = scale(fg3a_rate, [0.15, 0.55], [1.00, 0.80])
        _mid_pullup_mult = scale(pull_up_three_rate, [0.02, 0.22], [0.85, 1.08])
        _mid_interior_penalty = scale(zra + zpaint, [0.15, 0.55], [1.00, 0.84])
        _mid_big_pull_penalty = 1.0
        if pos in ("PF", "C"):
            _mid_big_pull_penalty = scale(pull_up_three_rate, [0.00, 0.12], [0.84, 1.00])
        shot_mid_range = (
            _mid_zone_volume
            * _mid_attempt_mult
            * _mid_eff_mult
            * _mid_creator_mult
            * _mid_three_penalty
            * _mid_pullup_mult
            * _mid_interior_penalty
            * _mid_big_pull_penalty
        )
        t["shot_mid_range"] = shot_mid_range

        t["spot_up_shot_mid_range"] = shot_mid_range * 0.7
        t["off_screen_shot_mid_range"] = shot_mid_range * 0.6

        fg3a_pg = f.get("fg3a_per_game", fga_p36 * fg3a_rate)
        _three_rate = scale(fg3a_rate, [0.0, 0.55], [0, 60])
        _three_vol  = scale(fg3a_pg, [1.0, 8.0], [0.5, 1.0])
        _three_eff  = scale(fg3_pct, [0.30, 0.40], [0.85, 1.10])
        shot_three  = _three_rate * _three_vol * _three_eff
        t["shot_three"] = shot_three

        # ---------------------------------------------------------------
        # Category B: Three-Point Subtypes
        # ---------------------------------------------------------------
        cs3_factor = scale(catch_and_shoot_three_rate, [0.03, 0.30], [0.65, 1.20])
        pull_up_factor = scale(pull_up_three_rate, [0.02, 0.22], [0.65, 1.25])
        t["spot_up_shot_three"] = shot_three * cs3_factor
        t["off_screen_shot_three"] = shot_three * (0.50 + 0.35 * cs3_factor)
        _transition_pace = scale(seconds_per_poss_off, [16.0, 12.0], [0.75, 1.25])
        t["transition_pull_up_three"] = (
            scale(pull_up_three_rate, [0.02, 0.20], [0, 30])
            * scale(pts_p36, [10, 30], [0.5, 1.2])
            * (1.0 if pos in ("PG", "SG", "SF") else 0.4)
            * _transition_pace
        )

        # ---------------------------------------------------------------
        # Category C: Contested / Advanced Shooting
        # ---------------------------------------------------------------
        _eff_mid = scale(fg_pct, [0.40, 0.55], [0.85, 1.15])
        _eff_three = scale(fg3_pct, [0.30, 0.42], [0.85, 1.15])
        t["contested_jumper_mid_range"] = shot_mid_range * 0.55 * _eff_mid
        t["contested_jumper_three"] = shot_three * 0.35 * _eff_three
        t["stepback_jumper_mid_range"] = scale(shot_mid_range, [0, 55], [0, 25]) * dribble_boost * _eff_mid
        _creator_stepback_three = (
            0.40 * scale(usg, [0.18, 0.35], [0.0, 1.0])
            + 0.35 * scale(ast_to_tov, [1.2, 3.2], [0.0, 1.0])
            + 0.25 * scale(fg3a_rate, [0.20, 0.60], [0.0, 1.0])
        )
        _creator_stepback_three = max(0.0, min(1.0, _creator_stepback_three))
        _pbp_stepback_creator = scale(unassisted_two_rate, [0.04, 0.24], [0.85, 1.20])

        _stepback_three_base = (
            scale(shot_three, [0, 60], [0, 30]) * dribble_boost
            + scale(fg3a_rate, [0.3, 0.6], [0, 10])
        ) * _eff_three * _pbp_stepback_creator
        _stepback_three_pre_cap = _stepback_three_base * (1.0 + 0.45 * _creator_stepback_three)
        _stepback_three_cap = (
            t["stepback_jumper_mid_range"]
            + (5.0 + round(8.0 * _creator_stepback_three))
            + scale(shot_three, [20.0, 60.0], [0.0, 20.0])
        )
        t["stepback_jumper_three"] = max(
            _stepback_three_base,
            min(_stepback_three_pre_cap, _stepback_three_cap),
        )
        t["spin_jumper"] = scale(shot_mid_range, [0, 55], [0, 15]) * dribble_boost

        # ---------------------------------------------------------------
        # Category D: Pull-Up Shooting
        # ---------------------------------------------------------------
        t["drive_pull_up_mid_range"] = scale(shot_mid_range, [0, 55], [0, 40]) * dribble_boost
        _assist_3_penalty = scale(assisted_3pt_pct, [90.0, 55.0], [0.80, 1.18])
        t["drive_pull_up_three"] = (
            scale(pull_up_three_rate, [0.0, 0.24], [0, 25])
            * dribble_boost
            * _assist_3_penalty
        )

        # ---------------------------------------------------------------
        # Category E: Finishing
        # ---------------------------------------------------------------
        t["driving_layup"] = scale(zra + zpaint, [0.1, 0.6], [30, 85])
        standing_dunk_raw = scale(zra, [0.05, 0.4], [0, 60]) * (0.5 + 0.5 * drive_boost)
        t["standing_dunk"] = standing_dunk_raw * (0.15 + 0.85 * post_factor)
        # NOTE: driving_dunk, flashy_dunk, alley_oop moved after Category H (need drive)
        # Putback: willingness to attempt putback after offensive rebound
        # 0=never, 10=rare, 20=occasional, 30=normal, 45=frequent, 55=cap
        # Primarily a big man skill — guards almost never attempt putbacks
        _pb_oreb = scale(oreb_pct, [0.05, 0.40], [0, 50])
        _pb_pos_factor = {"C": 1.0, "PF": 0.90, "SF": 0.55, "SG": 0.20, "PG": 0.15}.get(pos, 0.30)
        _pb_pbp = scale(putback_rate, [0.01, 0.09], [0, 50])
        t["putback"] = min(55.0, max(_pb_oreb * _pb_pos_factor, _pb_pbp * _pb_pos_factor))
        t["use_glass"] = scale(zpaint + zra, [0.1, 0.5], [10, 45])
        _step_signal = (
            0.55 * zpaint
            + 0.30 * zra
            + 0.15 * scale(fta_rate, [0.10, 0.50], [0.0, 1.0])
        )
        t["step_through_shot"] = (
            scale(_step_signal, [0.03, 0.40], [3, 32])
            * (0.35 + 0.65 * post_factor)
        )

        # ---------------------------------------------------------------
        # Category F: Craft Finishing — MOVED after Category H (needs drive)
        # (see below, after driving block)
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # Category G: Physical
        # ---------------------------------------------------------------
        if hustle_loose_balls >= 0 and hustle_charges >= 0:
            # Base 15 = league-floor crash value; loose-balls adds up to 15, charges up to 10
            _crash_raw = 15 + scale(hustle_loose_balls, [0, 1.5], [0, 15]) + scale(hustle_charges, [0, 0.5], [0, 10])
            t["crash"] = min(float(round(_crash_raw)), 45.0)
        else:
            _pos_crash = {"C": 22.0, "PF": 20.0, "SF": 18.0, "SG": 15.0, "PG": 15.0}
            t["crash"] = _pos_crash.get(pos, 17.0)

        # ---------------------------------------------------------------
        # Category H: Driving
        # ---------------------------------------------------------------
        # Zone-based drive signal: RA zone rate indicates rim attacks
        zone_drive = scale(zra, [0.05, 0.35], [20, 60])
        # FTA rate signals contact at the rim — true drivers draw fouls
        contact_drive = scale(fta_rate, [0.08, 0.35], [15, 55])
        # ISO possessions — self-created drives off the dribble
        iso_pos = f.get("isolation_possessions", 0.0)
        iso_drive = scale(iso_pos, [0.3, 5.0], [10, 45])
        creation_drive = scale(unassisted_two_rate, [0.03, 0.24], [8, 40])
        # PnR ball handler — drives created out of pick and roll
        pnr_bh_pos = f.get("pick_and_roll_ball_handler_possessions", 0.0)

        if pos in ("PG", "SG"):
            # Guards: blend zone rate, ISO/PnR activity, and contact rate
            pnr_drive = scale(pnr_bh_pos, [0.5, 8.0], [10, 40])
            drive = (0.26 * zone_drive + 0.28 * contact_drive
                     + 0.18 * iso_drive + 0.15 * pnr_drive + 0.13 * creation_drive) * drive_boost
        elif pos == "SF":
            # Wings: heavier on zone rate and contact
            pnr_drive = scale(pnr_bh_pos, [0.5, 5.0], [10, 30])
            drive = (0.32 * zone_drive + 0.28 * contact_drive
                     + 0.18 * iso_drive + 0.12 * pnr_drive + 0.10 * creation_drive) * drive_boost
        else:
            # Bigs: Must create their own offense to be drivers.
            big_creation = scale(iso_pos + pnr_bh_pos, [0.2, 4.0], [0.15, 1.0])
            drive = zone_drive * drive_boost * big_creation
        t["drive"] = drive

        # Spot-up drive: proportional to drive, slightly lower
        t["spot_up_drive"] = drive * 0.7

        # Off-screen drive: independent calc using off-screen possessions + drive ability
        off_scr_pos = f.get("off_screen_possessions", 0.0)
        off_scr_volume = scale(off_scr_pos, [0.2, 3.5], [12, 45])
        # Blend: how much they use off-screen plays × their drive ability
        t["off_screen_drive"] = 0.55 * off_scr_volume + 0.45 * scale(drive, [10, 55], [8, 30])
        t["drive_right"] = f.get("drive_right_bias", 50.0)

        # ---------------------------------------------------------------
        # Category E2: Dunks (moved here — needs drive from Category H)
        # ---------------------------------------------------------------
        # Driving dunk: willingness to dunk while driving to the rim
        # 0=never, 10=rare, 20=occasional, 30=normal, 40=frequent, 60=elite cap
        # Rim rate is the dominant signal — players who attack the basket dunk
        _dd_rim = scale(zra, [0.10, 0.50], [0, 45])           # dominant: rim attacking
        _dd_drive = scale(drive, [20, 55], [0, 10])            # need to drive to dunk
        if pos in ("PG", "SG"):
            # Guards: weight separates power dunkers from finesse finishers
            _dd_contact = scale(fta_rate, [0.20, 0.45], [0, 3])
            _dd_power = scale(weight_lbs, [190, 240], [0, 15])
        else:
            _dd_contact = scale(fta_rate, [0.20, 0.45], [0, 5])
            _dd_power = 0
        t["driving_dunk"] = min(60.0, (_dd_rim + _dd_drive + _dd_contact + _dd_power) * drive_boost)
        flashy_factor = {"PG": 0.6, "SG": 0.5, "SF": 0.5, "PF": 0.45, "C": 0.15}
        t["flashy_dunk"] = t["driving_dunk"] * flashy_factor.get(pos, 0.5) * drive_boost
        # Alley-oop (receiver): willingness to finish lobs with dunks
        # 0=never, 10=rare, 20=occasional, 30=normal, 45=frequent, 55=cap
        # Requires DUNKING ability — finesse bigs (Jokic) don't finish lobs
        if pos in ("C", "PF"):
            _lob_dunk_ability = (
                0.40 * scale(t["driving_dunk"], [10.0, 50.0], [0, 30])
                + 0.60 * scale(t["standing_dunk"], [15.0, 55.0], [0, 25])
            )
            _lob_dunk_gate = min(1.0, max(0.1,
                                          0.5 * (t["standing_dunk"] / 40.0)
                                          + 0.5 * scale(zra, [0.20, 0.45], [0.0, 1.0])))
        else:
            _lob_dunk_ability = (
                0.75 * scale(t["driving_dunk"], [10.0, 50.0], [0, 35])
                + 0.25 * scale(t["standing_dunk"], [15.0, 55.0], [0, 15])
            )
            _lob_dunk_gate = min(1.0, max(0.1, t["driving_dunk"] / 35.0))
        _lob_rim = scale(zra, [0.15, 0.55], [0, 10])
        _lob_size = scale(height_inches, [74, 84], [0, 8])
        t["alley_oop"] = min(55.0, (_lob_dunk_ability + _lob_rim + _lob_size) * _lob_dunk_gate)

        # ---------------------------------------------------------------
        # Category F: Craft Finishing (after Category H so drive is available)
        # ---------------------------------------------------------------
        _creator_finish = (
            0.50 * scale(usg, [0.18, 0.35], [0.0, 1.0])
            + 0.35 * scale(ast_to_tov, [1.2, 3.2], [0.0, 1.0])
            + 0.15 * scale(ast_p36, [3.0, 10.0], [0.0, 1.0])
        )
        _creator_finish = max(0.0, min(1.0, _creator_finish))
        _layup_craft_bonus = 1.0 + 0.55 * _creator_finish

        # Euro/spin/hop scale with both RA zone rate AND drive tendency
        # Good drivers should have good craft layup moves
        _rim_signal = scale(zra, [0.05, 0.35], [0, 30])
        _drive_signal = scale(drive, [15, 55], [0, 30])
        _craft_base = 0.45 * _rim_signal + 0.55 * _drive_signal
        t["euro_step_layup"] = _craft_base * _layup_craft_bonus
        t["spin_layup"] = _craft_base * _layup_craft_bonus
        t["hop_step_layup"] = _craft_base * 0.85 * _layup_craft_bonus

        # Floater is independent of drive — it's a finesse paint shot
        # Curry doesn't drive much but uses the floater well
        # Key signals: paint zone shooting (NOT rim), touch, shorter guards
        # Rim finishers (high zra) dunk/layup instead of floating
        _floater_paint = scale(zpaint, [0.05, 0.25], [0, 25])
        _floater_touch = scale(fg_pct, [0.42, 0.52], [0, 10])
        _floater_size = scale(height_inches, [80, 72], [0, 12])  # shorter = more floater
        _floater_creation = scale(usg, [0.18, 0.32], [0, 8])
        # Penalty: rim attackers finish at the basket, they don't float
        _floater_rim_penalty = scale(zra, [0.10, 0.35], [0, 22])
        # Bigs almost never use floaters
        _floater_big_penalty = 10.0 if pos in ("C", "PF") else 0.0
        t["floater"] = max(5.0, _floater_paint + _floater_touch + _floater_size
                           + _floater_creation - _floater_rim_penalty - _floater_big_penalty)

        # ---------------------------------------------------------------
        # Category I: Triple Threat
        # ---------------------------------------------------------------
        _tt_pace = scale(seconds_per_poss_off, [12.0, 16.0], [0.85, 1.20])
        _tt_assist_control = scale(assisted_2pt_pct + assisted_3pt_pct, [100.0, 170.0], [0.90, 1.10])
        t["triple_threat_pump_fake"] = (
            scale(shot_mid_range + shot_three, [0, 100], [10, 45])
            * _tt_pace
            * _tt_assist_control
        )
        t["triple_threat_jab_step"] = (
            scale(drive, [0, 60], [10, 40])
            * scale(unassisted_two_rate, [0.03, 0.24], [0.90, 1.15])
        )
        # --- triple_threat_idle: models "think time / decision time before acting"
        # LOW idle = fast decision maker (slashers, shooters, traditional bigs)
        # HIGH idle = methodical ISO creator who probes the defense (SGA, Luka, Harden)
        #
        # Step 1: Creation factor — high-USG players are ball-dominant and need more
        # time on ball to probe the defense before committing.
        # Output range [10, 40]: low-USG player starts at 10, elite ISO creator reaches 40.
        _idle_creation = scale(usg, [0.10, 0.35], [10, 40])
        # Step 2: Speed proxy — taller players are physically slower to initiate moves,
        # so they idle longer before attacking.
        # Range [0.6, 1.3]: 6-0" (72 in) guard = 0.6× multiplier (fast); 7-0" (84 in) = 1.3×.
        _idle_speed = scale(height_inches, [72, 84], [0.6, 1.3])
        # Step 3: Quick-trigger reduction — shooters (high fg3a_rate) catch-and-shoot
        # without deliberation; slashers (high zra) attack the rim immediately.
        # Coefficients: fg3a_rate × 0.4 (slightly less impactful) + zra × 0.5 (rimrunners
        # are the quickest trigger). Floor of 0.1 prevents total elimination of idle.
        _idle_quick_trigger = max(0.1, 1.0 - (fg3a_rate * 0.4 + zra * 0.5))
        # Step 4: Big rescue — traditional bigs catch and act immediately (catch→dunk/pass).
        # Exception: unicorn bigs with high AST (Jokic-type) scan the floor like guards
        # and warrant normal idle.
        # Threshold 4.0 ast/36 distinguishes playmaking bigs from rim-runners.
        # Rescue factor 0.3 pushes traditional bigs firmly into the 5–10 idle range.
        _idle_is_big = pos in ("C", "PF")
        if _idle_is_big and ast_p36 < 4.0:
            _idle_big_rescue = 0.3   # traditional big: force very low idle
        elif _idle_is_big:
            _idle_big_rescue = 1.0   # unicorn big (high AST): allow normal idle
        else:
            _idle_big_rescue = 1.0
        _idle_pace = scale(seconds_per_poss_off, [12.0, 16.0], [0.75, 1.25])
        _idle_risk = scale(live_ball_turnover_pct, [8.0, 18.0], [0.92, 1.12])
        _idle_raw = (
            _idle_creation
            * _idle_speed
            * _idle_quick_trigger
            * _idle_big_rescue
            * _idle_pace
            * _idle_risk
        )
        t["triple_threat_idle"] = max(5.0, min(50.0, _idle_raw))
        t["triple_threat_shoot"] = scale(shot_three + shot_mid_range, [0, 100], [10, 45])

        # ---------------------------------------------------------------
        # Category J: Dribble Setup
        # ---------------------------------------------------------------
        _setup_creator = (
            0.50 * scale(usg, [0.10, 0.35], [0.0, 1.0])
            + 0.30 * scale(unassisted_two_rate, [0.03, 0.24], [0.0, 1.0])
            + 0.20 * scale(pull_up_three_rate, [0.02, 0.22], [0.0, 1.0])
        )
        t["setup_with_sizeup"] = scale(_setup_creator, [0.0, 1.0], [10, 45]) * dribble_boost
        t["setup_with_hesitation"] = t["setup_with_sizeup"] * 0.9
        _no_setup_quick = (
            0.60 * scale(catch_and_shoot_three_rate, [0.03, 0.30], [0.0, 1.0])
            + 0.40 * scale(seconds_per_poss_off, [16.0, 12.0], [0.0, 1.0])
        )
        t["no_setup_dribble"] = 35 - scale(_setup_creator, [0.0, 1.0], [0, 18]) + scale(_no_setup_quick, [0.0, 1.0], [0, 10])

        # ---------------------------------------------------------------
        # Category K: Dribble Moves
        # ---------------------------------------------------------------
        # Position/size-based differentiation: guards have more elaborate ball-handling
        _gd = {"PG": 1.20, "SG": 1.05, "SF": 0.90, "PF": 0.70, "C": 0.45}.get(pos, 0.90)
        _h_dribble = scale(height_inches, [72, 84], [1.15, 0.90])  # shorter → better ball handling
        _handle_creation = (
            0.45 * scale(usg, [0.10, 0.35], [0.0, 1.0])
            + 0.35 * scale(unassisted_two_rate, [0.03, 0.24], [0.0, 1.0])
            + 0.20 * scale(pull_up_three_rate, [0.02, 0.22], [0.0, 1.0])
        )
        _handle_risk = (
            0.60 * scale(bad_pass_turnovers_p36, [0.5, 2.8], [0.90, 1.10])
            + 0.40 * scale(lost_ball_turnovers_p36, [0.4, 2.5], [0.90, 1.10])
        )
        creation_score = scale(_handle_creation, [0.0, 1.0], [5, 35]) * dribble_boost * _handle_risk
        t["driving_crossover"] = creation_score * 1.0 * _gd
        t["driving_spin"] = creation_score * 0.8
        t["driving_step_back"] = creation_score * 0.7
        t["driving_half_spin"] = creation_score * 0.7
        t["driving_double_crossover"] = creation_score * 0.7 * _gd
        t["driving_behind_the_back"] = creation_score * 0.7 * _gd * _h_dribble
        t["driving_dribble_hesitation"] = creation_score * 0.95 * _gd * _h_dribble
        t["driving_in_and_out"] = creation_score * 0.85 * _gd
        no_dribble_base = 75 - creation_score * 1.5
        pos_no_dribble = {"PG": -10, "SG": -5, "SF": 0, "PF": 5, "C": 10}
        t["no_driving_dribble_move"] = max(15.0, min(75.0, no_dribble_base + pos_no_dribble.get(pos, 0)))

        # ---------------------------------------------------------------
        # Category L: Drive Finishing
        # ---------------------------------------------------------------
        t["attack_strong_on_drive"] = min(scale(fta_rate, [0.1, 0.5], [20, 65]), 65.0)
        _sfd_bonus = scale(shooting_fouls_drawn_pct, [4.0, 16.0], [0, 12])
        t["attack_strong_on_drive"] = min(65.0, t["attack_strong_on_drive"] + _sfd_bonus)
        _dish_safety = scale(bad_pass_turnovers_p36, [2.6, 0.5], [0.85, 1.10])
        t["dish_to_open_man"] = scale(ast_p36, [1, 10], [15, 50]) * _dish_safety

        # ---------------------------------------------------------------
        # Category M: Passing
        # ---------------------------------------------------------------
        # Flashy pass driven by passing ability, gated by ast_to_tov discipline:
        # disciplined passers (high ratio) pass flashier less; riskier passers more so
        guard_flashy_bonus = 1.1 if pos in ("PG", "SG") else 1.0
        _flashy_tov = scale(ast_to_tov, [0.5, 3.5], [1.10, 0.85])
        _flashy_live_ball = scale(live_ball_turnover_pct, [8.0, 18.0], [0.92, 1.10])
        t["flashy_pass"] = (
            scale(ast_p36, [2, 12], [5, 55])
            * guard_flashy_bonus
            * _flashy_tov
            * _flashy_live_ball
        )
        _lob_vision = (
            0.60 * scale(ast_p36, [2.0, 12.0], [0.0, 1.0])
            + 0.15 * scale(ast_to_tov, [0.8, 3.5], [0.0, 1.0])
            + 0.25 * scale(usg, [0.12, 0.34], [0.0, 1.0])
        )
        _drive_collapse = scale(t.get("drive", 20.0), [15.0, 60.0], [0.0, 1.0])
        _lob_pass_pos_factor = {"PG": 1.12, "SG": 1.06, "SF": 1.03, "PF": 0.94, "C": 0.90}.get(pos, 1.0)
        _lob_pass_handle = scale(dribble_boost, [0.3, 1.2], [0.90, 1.08])
        _lob_pass_risk = scale(ast_to_tov, [0.8, 3.5], [1.02, 0.95])
        _lob_pass_score = 0.65 * _lob_vision + 0.35 * _drive_collapse
        _elite_lob_playmaker = (
            0.65 * scale(ast_p36, [6.0, 12.0], [0.0, 1.0])
            + 0.35 * scale(usg, [0.22, 0.36], [0.0, 1.0])
        )
        _elite_lob_boost = 1.0 + 0.28 * _elite_lob_playmaker
        t["alley_oop_pass"] = (
            scale(_lob_pass_score, [0.0, 1.0], [5, 45])
            * _lob_pass_pos_factor
            * _lob_pass_handle
            * _lob_pass_risk
            * _elite_lob_boost
        )

        # Elite creator floor: high-usage, high-assist engines should reliably
        # sit in the 40-50 alley-oop pass band.
        _elite_floor_score = (
            0.70 * scale(ast_p36, [7.0, 11.0], [0.0, 1.0])
            + 0.30 * scale(usg, [0.24, 0.36], [0.0, 1.0])
        )
        _elite_floor = scale(_elite_floor_score, [0.0, 1.0], [35, 50])
        if pos in ("PG", "SG", "SF") and ast_p36 >= 7.0 and usg >= 0.24:
            t["alley_oop_pass"] = max(t["alley_oop_pass"], _elite_floor)

        # ---------------------------------------------------------------
        # Category N: Post Play (17 tendencies)
        # ---------------------------------------------------------------
        # Primary signal: actual post-up possessions per game (Synergy).
        # Post-up covers BOTH scoring and playmaking from the post.
        post_up_poss = f.get("post_up_possessions", 0.0)
        post_up_ppp = f.get("post_up_ppp", 0.0)
        _post_assist_gate = scale(assisted_2pt_pct, [70.0, 35.0], [0.85, 1.15])

        # Post-up tendency: driven by real post possessions + paint zone context
        _pu_volume = scale(post_up_poss, [0.0, 6.0], [0, 40])
        _pu_paint = scale(zpaint, [0.05, 0.40], [0, 10])
        _pu_size = scale(height_inches, [76, 84], [0.75, 1.10]) * scale(weight_lbs, [200, 280], [0.85, 1.15])
        # Guard-size suppressor: small guards usually post less, but do not hard-zero
        # players with strong observed post-up volume (e.g., bigger creators).
        if height_inches < 76 and weight_lbs < 210:
            if post_up_poss < 1.8:
                _guard_penalty = 0.25
            else:
                _guard_penalty = scale(post_up_poss, [1.8, 6.0], [0.55, 0.95])
            _pu_volume *= _guard_penalty
            _pu_paint *= _guard_penalty
        t["post_up"] = min(60.0, (_pu_volume + _pu_paint) * _pu_size * _post_assist_gate)
        _pu_base = _pu_volume + _pu_paint  # pre-size volume for finesse moves

        # Big-man floor: centers/PFs always have some post presence from size alone
        if pos == "C":
            _big_floor = scale(height_inches, [80, 85], [10, 20]) * scale(weight_lbs, [230, 270], [0.8, 1.0])
            t["post_up"] = max(t["post_up"], _big_floor)
        elif pos == "PF":
            _big_floor = scale(height_inches, [80, 84], [5, 15]) * scale(weight_lbs, [230, 260], [0.8, 1.0])
            t["post_up"] = max(t["post_up"], _big_floor)

        # Shooting ability gate: separates shooters (fade/shimmy) from
        # non-shooters (hooks/drop-step). Mid-range + paint + three-point shooting.
        _post_shoot_ability = scale(zmid_total + zpaint * 0.3 + fg3_pct * 0.3, [0.05, 0.35], [0.0, 1.0])
        # Finesse gate: playmaking post moves (face-up, spin, drive)
        _finesse_gate = scale(ast_p36, [1.0, 5.0], [0.2, 1.0])
        # Hook size: taller/heavier players use hooks more
        _hook_size = scale(height_inches, [78, 84], [0.4, 1.0]) * scale(weight_lbs, [220, 280], [0.7, 1.0])

        _pu = t["post_up"]  # shorthand

        t["post_shimmy_shot"] = _pu * 0.30 * _post_shoot_ability
        t["post_face_up"] = _pu * 0.65 * _finesse_gate
        t["post_back_down"] = _pu * 0.55 * _pu_size
        t["post_aggressive_backdown"] = _pu * 0.40 * _pu_size
        # Shoot from post: requires actual shooting ability
        t["shoot_from_post"] = _pu * 0.65 * _post_shoot_ability
        # Hooks: size-driven, any big who posts up uses hooks
        t["post_hook_left"] = _pu * 0.30 * _hook_size
        t["post_hook_right"] = _pu * 0.30 * _hook_size
        # Fades: require shooting touch
        t["post_fade_left"] = _pu * 0.30 * _post_shoot_ability * _finesse_gate
        t["post_fade_right"] = _pu * 0.30 * _post_shoot_ability * _finesse_gate
        # Up and under: a finishing/layup move, correlates with rim attacking
        _uu_rim = scale(zra, [0.10, 0.45], [0.5, 1.0])
        t["post_up_and_under"] = _pu * 0.45 * _uu_rim
        t["post_hop_shot"] = _pu * 0.25 * _post_shoot_ability
        t["post_step_back_shot"] = _pu * 0.25 * _post_shoot_ability
        t["post_drive"] = _pu * 0.60 * _finesse_gate
        # Spin: finesse move for quicker/shorter players — use raw base so
        # size penalty on post_up doesn't counteract the spin height bonus
        _spin_size = scale(height_inches, [72, 84], [1.20, 0.85])
        t["post_spin"] = _pu_base * 0.35 * _spin_size * _finesse_gate
        t["post_drop_step"] = _pu * 0.35 * _pu_size
        t["post_hop_step"] = _pu * 0.25

        # ---------------------------------------------------------------
        # Category O: Playstyle Sliders
        # ---------------------------------------------------------------
        _roll_signal = (
            0.50 * scale(zra, [0.08, 0.45], [0.0, 1.0])
            + 0.25 * scale(assisted_2pt_pct, [35.0, 75.0], [0.0, 1.0])
            + 0.25 * scale(second_chance_off_poss_rate, [0.04, 0.18], [0.0, 1.0])
        )
        _pop_signal = (
            0.60 * scale(fg3a_rate, [0.10, 0.55], [0.0, 1.0])
            + 0.40 * scale(catch_and_shoot_three_rate, [0.03, 0.30], [0.0, 1.0])
        )
        t["roll_vs_pop"] = scale(_roll_signal - _pop_signal, [-1.0, 1.0], [5, 95])
        # transition_spot_up: >50 = spot up for 3, <50 = cut to basket
        # Shooters spot up; rim attackers and non-shooters cut
        _trans_volume = scale(fg3a_rate, [0.10, 0.55], [0, 30])    # how often they take 3s
        _trans_accuracy = scale(fg3_pct, [0.30, 0.40], [0, 30])    # can they actually make them
        _trans_rim_cut = scale(zra, [0.08, 0.40], [0, 30])         # rim attackers cut instead
        _trans_second_chance_push = scale(second_chance_off_poss_rate, [0.04, 0.18], [-6, 8])
        t["transition_spot_up"] = max(
            5.0,
            20 + _trans_volume + _trans_accuracy - _trans_rim_cut + _trans_second_chance_push,
        )

        # ---------------------------------------------------------------
        # Category P: Isolation
        # ---------------------------------------------------------------
        # Primary signal: actual ISO possessions per game (synergy data)
        # Players who don't ISO shouldn't have ISO tendencies
        _iso_volume = scale(iso_pos, [0.0, 7.0], [0, 30])
        _iso_unassisted = scale(unassisted_two_rate, [0.03, 0.24], [0.75, 1.30])
        _iso_assist_gate = scale(assisted_2pt_pct, [75.0, 35.0], [0.85, 1.15])
        # USG multiplier: high-usage = more willing to call their own number
        _iso_usg = scale(usg, [0.12, 0.35], [0.6, 1.25])
        # Efficiency: good scorers are bolder in ISO
        _iso_eff = scale(ts_pct, [0.50, 0.65], [0.90, 1.15])
        _iso_creator_profile = (
            0.50 * scale(usg, [0.14, 0.34], [0.0, 1.0])
            + 0.30 * scale(ast_p36, [2.0, 9.0], [0.0, 1.0])
            + 0.20 * scale(unassisted_two_rate, [0.05, 0.24], [0.0, 1.0])
        )
        _iso_role_mult = scale(_iso_creator_profile, [0.0, 1.0], [0.42, 1.05])
        _iso_drive_mult = scale(t.get("drive", 20.0), [20.0, 55.0], [0.85, 1.05])
        iso_base = (
            _iso_volume
            * _iso_usg
            * _iso_eff
            * _iso_unassisted
            * _iso_assist_gate
            * _iso_role_mult
            * _iso_drive_mult
            * 0.92
        )
        t["iso_vs_elite_defender"] = iso_base * 0.35
        t["iso_vs_good_defender"] = iso_base * 0.55
        t["iso_vs_average_defender"] = iso_base * 0.75
        t["iso_vs_poor_defender"] = iso_base * 0.95

        # ---------------------------------------------------------------
        # Category Q: Discipline
        # ---------------------------------------------------------------
        # Higher for low-usage role players (stick to playbook), lower for stars
        # to preserve freelance creation. Hard-clamped to avoid offensive stalling.
        pos_discipline = {"PG": -4, "SG": -1, "SF": 0, "PF": 2, "C": 4}
        play_discipline = (
            52
            - scale(usg, [0.10, 0.35], [0, 20])
            + scale(seconds_per_poss_off, [12.0, 16.0], [-4, 8])
            + scale(assisted_2pt_pct + assisted_3pt_pct, [100.0, 170.0], [-2, 8])
            - scale(live_ball_turnover_pct, [8.0, 18.0], [0, 8])
            + pos_discipline.get(pos, 0)
        )
        t["play_discipline"] = max(30.0, min(55.0, play_discipline))

        # ---------------------------------------------------------------
        # Category R: Defense
        # ---------------------------------------------------------------
        # Steal tendency: guards get more on-ball/interception credit per steal
        steal_pos_scale = {"PG": 1.0, "SG": 0.9, "SF": 0.85, "PF": 0.7, "C": 0.55}
        steal_scale = steal_pos_scale.get(pos, 0.85)
        _intercept_creation = scale(bad_pass_turnovers_p36, [0.5, 2.8], [0.92, 1.10])
        t["pass_interception"] = scale(stl_p36, [0.3, 2.5], [15, 55]) * steal_scale * _intercept_creation
        t["on_ball_steal"] = scale(stl_p36, [0.3, 2.5], [15, 55]) * steal_scale
        pos_contest_base = {"PG": 30, "SG": 32, "SF": 33, "PF": 35, "C": 38}
        t["contest_shot"] = (
            pos_contest_base.get(pos, 33)
            + scale(blk_p36, [0.0, 2.5], [0, 15])
            + scale(stl_p36, [0.3, 2.0], [0, 10])
        )
        block_scale = profile["block_scale"]
        raw_block = scale(blk_p36, [0.0, 3.5], [5, 55]) * scale(blocks_recovered_pct, [35.0, 60.0], [0.90, 1.12])
        t["block_shot"] = raw_block * (0.6 + 0.4 * block_scale)
        t["take_charge"] = (
            scale(offensive_fouls_drawn_p36 + loose_ball_fouls_drawn_p36, [0.1, 1.6], [5, 35]) * (1 - post_factor * 0.3)
        )

        # ---------------------------------------------------------------
        # Category S: Fouling
        # ---------------------------------------------------------------
        # Foul: how often the player commits fouls
        # Higher floor — every NBA player fouls; scale uses the full range
        _foul_shape = (
            0.60 * scale(pf_p36, [1.0, 4.5], [25, 60])
            + 0.20 * scale(loose_ball_fouls_p36, [0.0, 1.2], [20, 60])
            + 0.20 * scale(shooting_fouls_p36, [0.2, 2.5], [20, 60])
        )
        t["foul"] = _foul_shape
        # Hard foul: physicality/aggression — size + foul rate + interior play
        _hf_fouls = (
            0.50 * scale(pf_p36, [1.5, 4.5], [5, 35])
            + 0.30 * scale(offensive_fouls_p36, [0.1, 1.4], [5, 35])
            + 0.20 * scale(loose_ball_fouls_p36, [0.0, 1.2], [5, 30])
        )
        _hf_size = scale(weight_lbs, [185, 260], [0.6, 1.0])
        _hf_interior = scale(zra + zpaint, [0.10, 0.50], [0.7, 1.0])
        t["hard_foul"] = min(55.0, _hf_fouls * _hf_size * _hf_interior)

        # ---------------------------------------------------------------
        # Touches
        # ---------------------------------------------------------------
        t["touches"] = min(
            0.40 * scale(ast_p36, [0, 12], [15, 65])
            + 0.35 * scale(usg, [0.10, 0.30], [20, 60])
            + 0.25 * scale(seconds_per_poss_off, [12.0, 16.0], [15, 60]),
            65.0,
        )

        # ---------------------------------------------------------------
        # Category T: Sub-Zone Distributions (13 tendencies)
        # Close remains parent-scaled for consistency.
        # Mid/Three are chart-driven shape values and are not forced to sum
        # to their parent tendencies.
        # ---------------------------------------------------------------
        # Close sub-zones: scale by shot_close parent
        dist_close_shaped = _shape_close_distribution(dist_close)
        _close_parent = t["shot_close"]
        t["shot_close_left"] = dist_close_shaped["left"] * _close_parent / 100
        t["shot_close_middle"] = dist_close_shaped["middle"] * _close_parent / 100
        t["shot_close_right"] = dist_close_shaped["right"] * _close_parent / 100

        # Mid sub-zones: chart-driven shape (independent from shot_mid_range)
        dist_mid_shaped = _shape_five_zone_distribution(dist_mid, gamma=1.25)
        t["shot_mid_left"] = dist_mid_shaped.get("left", 20.0)
        t["shot_mid_left_center"] = dist_mid_shaped.get("left_center", 20.0)
        t["shot_mid_center"] = dist_mid_shaped.get("center", 20.0)
        t["shot_mid_right_center"] = dist_mid_shaped.get("right_center", 20.0)
        t["shot_mid_right"] = dist_mid_shaped.get("right", 20.0)

        # Three sub-zones: chart-driven shape (independent from shot_three)
        dist_three_shaped = _shape_five_zone_distribution(dist_three, gamma=1.15)
        t["shot_three_left"] = max(dist_three_shaped.get("left", 20.0), 8.0)
        t["shot_three_left_center"] = dist_three_shaped.get("left_center", 20.0)
        t["shot_three_center"] = dist_three_shaped.get("center", 20.0)
        t["shot_three_right_center"] = dist_three_shaped.get("right_center", 20.0)
        t["shot_three_right"] = max(dist_three_shaped.get("right", 20.0), 8.0)

        return t

    # ------------------------------------------------------------------
    # Legacy interface for backward-compat with stub signature
    # ------------------------------------------------------------------

    def compute(
        self, features: dict[str, float], position: str
    ) -> dict[str, int]:
        """
        Apply deterministic formulas and return integer tendency values.

        Parameters
        ----------
        features:  Feature dict from FeatureEngine.build_features.
        position:  Player position ('PG', 'SG', 'SF', 'PF', 'C').
        """
        merged = dict(features)
        merged["position"] = position
        raw = self.generate(merged)
        return {k: max(0, min(100, 5 * round(v / 5))) for k, v in raw.items()}

    def apply_locked_rules(
        self, tendencies: dict[str, int]
    ) -> dict[str, int]:
        """Enforce inter-tendency locked rules."""
        result = dict(tendencies)
        # Spot-up mid <= shot mid
        if "spot_up_shot_mid_range" in result and "shot_mid_range" in result:
            result["spot_up_shot_mid_range"] = min(
                result["spot_up_shot_mid_range"], result["shot_mid_range"]
            )
        # Off-screen mid <= shot mid
        if "off_screen_shot_mid_range" in result and "shot_mid_range" in result:
            result["off_screen_shot_mid_range"] = min(
                result["off_screen_shot_mid_range"], result["shot_mid_range"]
            )
        # Spot-Up Three <= Shot Three + 10
        if "spot_up_shot_three" in result and "shot_three" in result:
            result["spot_up_shot_three"] = min(
                result["spot_up_shot_three"], result["shot_three"] + 10
            )
        # Off-Screen Three <= Shot Three
        if "off_screen_shot_three" in result and "shot_three" in result:
            result["off_screen_shot_three"] = min(
                result["off_screen_shot_three"], result["shot_three"]
            )
        # Contested Three <= Shot Three
        if "contested_jumper_three" in result and "shot_three" in result:
            result["contested_jumper_three"] = min(
                result["contested_jumper_three"], result["shot_three"]
            )
        # No Setup Dribble absolute cap 35
        if "no_setup_dribble" in result:
            result["no_setup_dribble"] = min(result["no_setup_dribble"], 35)
        # Roll vs Pop: clamp to 5-95
        if "roll_vs_pop" in result:
            result["roll_vs_pop"] = max(5, min(95, result["roll_vs_pop"]))
        # Post hooks = 0 if post_up < 10
        if result.get("post_up", 0) < 10:
            result["post_hook_left"] = 0
            result["post_hook_right"] = 0
        return result
