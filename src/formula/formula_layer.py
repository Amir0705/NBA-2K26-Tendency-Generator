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


class FormulaLayer:
    """Deterministic rule-based tendency calculator."""

    POSITION_PROFILES: dict[str, dict[str, float]] = {
        "PG": {"post_scale": 0.05, "drive_boost": 1.15, "block_scale": 0.4, "dribble_boost": 1.2},
        "SG": {"post_scale": 0.10, "drive_boost": 1.10, "block_scale": 0.5, "dribble_boost": 0.85},
        "SF": {"post_scale": 0.30, "drive_boost": 1.00, "block_scale": 0.7, "dribble_boost": 0.95},
        "PF": {"post_scale": 0.70, "drive_boost": 0.85, "block_scale": 0.9, "dribble_boost": 0.6},
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
        fg3a_rate = f.get("fg3a_rate", 0.30)
        fta_rate = f.get("fta_rate", 0.25)
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
        shot = (
            0.6 * scale(usg, [0.10, 0.35], [20, 75])
            + 0.4 * scale(fga_p36, [3, 25], [15, 75])
        ) * shooting_skill
        _pts_pctile_boost = scale(pctile_pts, [0.3, 0.8], [0.95, 1.05])
        t["shot"] = min(shot * _pts_pctile_boost, 75.0)

        t["shot_under_basket"] = scale(zra, [0.0, 0.5], [0, 60])
        t["shot_close"] = scale(zpaint, [0.0, 0.3], [0, 60])

        shot_mid_range = scale(zmid_total, [0.0, 0.35], [0, 55])
        t["shot_mid_range"] = shot_mid_range

        t["spot_up_shot_mid_range"] = shot_mid_range * 0.7
        t["off_screen_shot_mid_range"] = shot_mid_range * 0.6

        shot_three = scale(fg3a_rate, [0.0, 0.55], [0, 60])
        t["shot_three"] = shot_three

        # ---------------------------------------------------------------
        # Category B: Three-Point Subtypes
        # ---------------------------------------------------------------
        t["spot_up_shot_three"] = shot_three * 0.85
        t["off_screen_shot_three"] = shot_three * 0.65
        t["transition_pull_up_three"] = (
            scale(fg3a_rate, [0.15, 0.45], [0, 30])
            * scale(pts_p36, [10, 30], [0.5, 1.2])
            * (1.0 if pos in ("PG", "SG", "SF") else 0.4)
        )

        # ---------------------------------------------------------------
        # Category C: Contested / Advanced Shooting
        # ---------------------------------------------------------------
        _eff_mid = scale(fg_pct, [0.40, 0.55], [0.85, 1.15])
        _eff_three = scale(fg3_pct, [0.30, 0.42], [0.85, 1.15])
        t["contested_jumper_mid_range"] = shot_mid_range * 0.55 * _eff_mid
        t["contested_jumper_three"] = shot_three * 0.35 * _eff_three
        t["stepback_jumper_mid_range"] = scale(shot_mid_range, [0, 55], [0, 25]) * dribble_boost * _eff_mid
        t["stepback_jumper_three"] = (
            scale(shot_three, [0, 60], [0, 30]) * dribble_boost
            + scale(fg3a_rate, [0.3, 0.6], [0, 10])
        ) * _eff_three
        t["spin_jumper"] = scale(shot_mid_range, [0, 55], [0, 15]) * dribble_boost

        # ---------------------------------------------------------------
        # Category D: Pull-Up Shooting
        # ---------------------------------------------------------------
        t["drive_pull_up_mid_range"] = scale(shot_mid_range, [0, 55], [0, 40]) * dribble_boost
        t["drive_pull_up_three"] = scale(fg3a_rate, [0.0, 0.4], [0, 25]) * dribble_boost

        # ---------------------------------------------------------------
        # Category E: Finishing
        # ---------------------------------------------------------------
        t["driving_layup"] = scale(zra + zpaint, [0.1, 0.6], [30, 85])
        standing_dunk_raw = scale(zra, [0.05, 0.4], [0, 60]) * (0.5 + 0.5 * drive_boost)
        t["standing_dunk"] = standing_dunk_raw * (0.15 + 0.85 * post_factor)
        t["driving_dunk"] = scale(zra, [0.05, 0.4], [0, 50]) * drive_boost
        flashy_factor = {"PG": 0.6, "SG": 0.5, "SF": 0.5, "PF": 0.45, "C": 0.15}
        t["flashy_dunk"] = t["driving_dunk"] * flashy_factor.get(pos, 0.5) * drive_boost
        t["alley_oop"] = scale(zra, [0.05, 0.35], [5, 45]) * (0.4 * post_factor + 0.6 * drive_boost)
        t["putback"] = scale(oreb_pct, [0.0, 0.3], [5, 50]) * (0.6 * post_factor + 0.4 * drive_boost)
        t["use_glass"] = scale(zpaint + zra, [0.1, 0.5], [10, 45])
        t["step_through_shot"] = scale(zpaint, [0.0, 0.25], [0, 30]) * (0.3 + 0.7 * post_factor)

        # ---------------------------------------------------------------
        # Category F: Craft Finishing
        # ---------------------------------------------------------------
        t["spin_layup"] = scale(zra, [0.05, 0.3], [0, 25]) * dribble_boost
        t["hop_step_layup"] = scale(zra, [0.05, 0.3], [0, 20]) * dribble_boost
        t["euro_step_layup"] = scale(zra, [0.05, 0.3], [0, 25]) * dribble_boost
        t["floater"] = scale(zpaint, [0.0, 0.2], [0, 25]) * dribble_boost

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
        zone_drive = scale(zra, [0.05, 0.45], [15, 55])
        # Guards create drives through ball-handling even with lower RA zone rates
        creation_drive = scale(usg, [0.15, 0.35], [10, 40]) * dribble_boost
        if pos in ("PG", "SG"):
            drive = 0.5 * zone_drive + 0.5 * creation_drive
        elif pos == "SF":
            drive = 0.6 * zone_drive + 0.4 * creation_drive
        else:
            drive = zone_drive * drive_boost  # Bigs: pure zone-based
        t["drive"] = drive
        t["spot_up_drive"] = drive * 0.7
        t["off_screen_drive"] = drive * 0.5
        t["drive_right"] = f.get("drive_right_bias", 50.0)

        # ---------------------------------------------------------------
        # Category I: Triple Threat
        # ---------------------------------------------------------------
        t["triple_threat_pump_fake"] = scale(shot_mid_range + shot_three, [0, 100], [10, 45])
        t["triple_threat_jab_step"] = scale(drive, [0, 60], [10, 40])
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
        _idle_raw = _idle_creation * _idle_speed * _idle_quick_trigger * _idle_big_rescue
        t["triple_threat_idle"] = max(5.0, min(50.0, _idle_raw))
        t["triple_threat_shoot"] = scale(shot_three + shot_mid_range, [0, 100], [10, 45])

        # ---------------------------------------------------------------
        # Category J: Dribble Setup
        # ---------------------------------------------------------------
        t["setup_with_sizeup"] = scale(usg, [0.10, 0.30], [10, 45]) * dribble_boost
        t["setup_with_hesitation"] = t["setup_with_sizeup"] * 0.9
        t["no_setup_dribble"] = 35 - scale(usg, [0.10, 0.30], [0, 20])

        # ---------------------------------------------------------------
        # Category K: Dribble Moves
        # ---------------------------------------------------------------
        # Position/size-based differentiation: guards have more elaborate ball-handling
        _gd = {"PG": 1.20, "SG": 1.05, "SF": 0.90, "PF": 0.70, "C": 0.45}.get(pos, 0.90)
        _h_dribble = scale(height_inches, [72, 84], [1.15, 0.90])  # shorter → better ball handling
        creation_score = scale(usg, [0.10, 0.30], [5, 35]) * dribble_boost
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
        t["dish_to_open_man"] = scale(ast_p36, [1, 10], [15, 50])

        # ---------------------------------------------------------------
        # Category M: Passing
        # ---------------------------------------------------------------
        # Flashy pass driven by passing ability, gated by ast_to_tov discipline:
        # disciplined passers (high ratio) pass flashier less; riskier passers more so
        guard_flashy_bonus = 1.1 if pos in ("PG", "SG") else 1.0
        _flashy_tov = scale(ast_to_tov, [0.5, 3.5], [1.10, 0.85])
        t["flashy_pass"] = scale(ast_p36, [2, 12], [5, 55]) * guard_flashy_bonus * _flashy_tov
        t["alley_oop_pass"] = scale(ast_p36, [2, 10], [5, 35]) * (0.5 + 0.5 * dribble_boost)

        # ---------------------------------------------------------------
        # Category N: Post Play (17 tendencies)
        # ---------------------------------------------------------------
        post_score = (
            scale(zpaint + zra, [0.1, 0.6], [0, 50]) * post_factor
        )
        # Height/weight size factors: taller/heavier → better at power post moves
        _h_post = scale(height_inches, [72, 84], [0.85, 1.15])
        _w_post = scale(weight_lbs, [180, 270], [0.90, 1.10])
        _size_post = _h_post * _w_post
        # Finesse moves (spin) favour shorter/quicker players
        _spin_size = scale(height_inches, [72, 84], [1.20, 0.85])

        t["post_up"] = post_score * 1.0 * _size_post
        t["post_shimmy_shot"] = post_score * 0.3
        t["post_face_up"] = post_score * 0.7
        t["post_back_down"] = post_score * (0.7 if pos == "C" else 0.6) * _size_post
        t["post_aggressive_backdown"] = post_score * (0.5 if pos == "C" else 0.4) * _size_post
        t["shoot_from_post"] = post_score * 0.8
        t["post_hook_left"] = post_score * 0.2 * _size_post
        t["post_hook_right"] = post_score * 0.2 * _size_post
        t["post_fade_left"] = post_score * 0.25 * _size_post
        t["post_fade_right"] = post_score * 0.25 * _size_post
        t["post_up_and_under"] = post_score * (0.5 if pos == "C" else 0.4)
        t["post_hop_shot"] = post_score * 0.35
        t["post_step_back_shot"] = post_score * 0.35
        t["post_drive"] = post_score * 0.65
        t["post_spin"] = post_score * 0.4 * _spin_size
        t["post_drop_step"] = post_score * (0.55 if pos == "C" else 0.4)
        t["post_hop_step"] = post_score * 0.3

        # ---------------------------------------------------------------
        # Category O: Playstyle Sliders
        # ---------------------------------------------------------------
        if pos in ("PG", "SG"):
            t["roll_vs_pop"] = 40 + scale(fg3a_rate, [0.15, 0.50], [0, 25])
        else:
            t["roll_vs_pop"] = 75 - scale(fg3a_rate, [0.0, 0.3], [0, 50])
        t["transition_spot_up"] = scale(fg3a_rate, [0.0, 0.4], [30, 70])

        # ---------------------------------------------------------------
        # Category P: Isolation
        # ---------------------------------------------------------------
        # High-efficiency scorers are more willing to take ISO opportunities
        _iso_eff = scale(ts_pct, [0.48, 0.65], [0.85, 1.20])
        iso_base = scale(usg, [0.10, 0.32], [0, 40]) * dribble_boost * _iso_eff
        t["iso_vs_elite_defender"] = iso_base * 0.5
        t["iso_vs_good_defender"] = iso_base * 0.7
        t["iso_vs_average_defender"] = iso_base * 0.85
        t["iso_vs_poor_defender"] = iso_base * 1.0

        # ---------------------------------------------------------------
        # Category Q: Discipline
        # ---------------------------------------------------------------
        pos_discipline = {"PG": -5, "SG": 0, "SF": 0, "PF": 5, "C": 10}
        play_discipline = 60 - scale(usg, [0.10, 0.35], [0, 30]) + pos_discipline.get(pos, 0)
        t["play_discipline"] = max(play_discipline, 35.0)

        # ---------------------------------------------------------------
        # Category R: Defense
        # ---------------------------------------------------------------
        # Steal tendency: guards get more on-ball/interception credit per steal
        steal_pos_scale = {"PG": 1.0, "SG": 0.9, "SF": 0.85, "PF": 0.7, "C": 0.55}
        steal_scale = steal_pos_scale.get(pos, 0.85)
        t["pass_interception"] = scale(stl_p36, [0.3, 2.5], [15, 55]) * steal_scale
        t["on_ball_steal"] = scale(stl_p36, [0.3, 2.5], [15, 55]) * steal_scale
        pos_contest_base = {"PG": 30, "SG": 32, "SF": 33, "PF": 35, "C": 38}
        t["contest_shot"] = (
            pos_contest_base.get(pos, 33)
            + scale(blk_p36, [0.0, 2.5], [0, 15])
            + scale(stl_p36, [0.3, 2.0], [0, 10])
        )
        block_scale = profile["block_scale"]
        raw_block = scale(blk_p36, [0.0, 3.5], [5, 55])
        t["block_shot"] = raw_block * (0.6 + 0.4 * block_scale)
        t["take_charge"] = (
            scale(pf_p36, [1.5, 4.0], [5, 30]) * (1 - post_factor * 0.3)
        )

        # ---------------------------------------------------------------
        # Category S: Fouling
        # ---------------------------------------------------------------
        t["foul"] = scale(pf_p36, [1.0, 4.5], [15, 55])
        hard_foul_base = scale(pf_p36, [2.0, 4.5], [5, 30])
        t["hard_foul"] = hard_foul_base * (0.3 + 0.7 * post_factor)

        # ---------------------------------------------------------------
        # Touches
        # ---------------------------------------------------------------
        t["touches"] = min(
            0.5 * scale(ast_p36, [0, 12], [15, 65])
            + 0.5 * scale(usg, [0.10, 0.30], [20, 60]),
            65.0,
        )

        # ---------------------------------------------------------------
        # Category T: Sub-Zone Distributions (13 tendencies)
        # ---------------------------------------------------------------
        t["shot_close_left"] = dist_close.get("left", 33.3)
        t["shot_close_middle"] = dist_close.get("middle", 33.4)
        t["shot_close_right"] = dist_close.get("right", 33.3)

        t["shot_mid_left"] = dist_mid.get("left", 20.0)
        t["shot_mid_left_center"] = dist_mid.get("left_center", 20.0)
        t["shot_mid_center"] = dist_mid.get("center", 20.0)
        t["shot_mid_right_center"] = dist_mid.get("right_center", 20.0)
        t["shot_mid_right"] = dist_mid.get("right", 20.0)

        t["shot_three_left"] = max(dist_three.get("left", 20.0), 8.0)
        t["shot_three_left_center"] = dist_three.get("left_center", 20.0)
        t["shot_three_center"] = dist_three.get("center", 20.0)
        t["shot_three_right_center"] = dist_three.get("right_center", 20.0)
        t["shot_three_right"] = max(dist_three.get("right", 20.0), 8.0)

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
        # Stepback three <= stepback mid + 5
        if "stepback_jumper_three" in result and "stepback_jumper_mid_range" in result:
            result["stepback_jumper_three"] = min(
                result["stepback_jumper_three"],
                result["stepback_jumper_mid_range"] + 5,
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
