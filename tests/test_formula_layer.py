"""Tests for src/formula/formula_layer.py."""
from __future__ import annotations

import pytest

from src.formula.formula_layer import FormulaLayer, scale

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")


def _all_registry_names() -> list[str]:
    with open(REGISTRY_PATH) as fh:
        return [e["canonical_name"] for e in json.load(fh)]


def _minimal_features(position: str = "SG") -> dict:
    """Baseline feature dict representing an average player."""
    return {
        "position": position,
        "usg_pct_proxy": 0.20,
        "fga_per36": 12.0,
        "fg3a_rate": 0.35,
        "fta_rate": 0.30,
        "ast_per36": 5.0,
        "pts_per36": 18.0,
        "stl_per36": 1.0,
        "blk_per36": 0.4,
        "pf_per36": 2.5,
        "oreb_pct_proxy": 0.10,
        "zone_fga_rate_ra": 0.20,
        "zone_fga_rate_paint": 0.15,
        "zone_fga_rate_mid_left": 0.08,
        "zone_fga_rate_mid_center": 0.07,
        "zone_fga_rate_mid_right": 0.08,
        "zone_fga_rate_corner3_left": 0.10,
        "zone_fga_rate_corner3_right": 0.10,
        "zone_fga_rate_above_break3": 0.22,
        "sub_zone_distribution_close": {"left": 30.0, "middle": 40.0, "right": 30.0},
        "sub_zone_distribution_mid": {
            "left": 20.0, "left_center": 20.0, "center": 20.0,
            "right_center": 20.0, "right": 20.0,
        },
        "sub_zone_distribution_three": {
            "left": 20.0, "left_center": 20.0, "center": 20.0,
            "right_center": 20.0, "right": 20.0,
        },
    }


class TestScaleFunction:
    def test_midpoint_returns_output_mid(self):
        result = scale(0.5, [0.0, 1.0], [0, 100])
        assert result == pytest.approx(50.0)

    def test_at_min_returns_output_min(self):
        result = scale(0.0, [0.0, 1.0], [10, 90])
        assert result == pytest.approx(10.0)

    def test_at_max_returns_output_max(self):
        result = scale(1.0, [0.0, 1.0], [10, 90])
        assert result == pytest.approx(90.0)

    def test_below_min_clips_to_output_min(self):
        result = scale(-1.0, [0.0, 1.0], [10, 90])
        assert result == pytest.approx(10.0)

    def test_above_max_clips_to_output_max(self):
        result = scale(2.0, [0.0, 1.0], [10, 90])
        assert result == pytest.approx(90.0)

    def test_equal_input_range_returns_midpoint(self):
        result = scale(5.0, [5.0, 5.0], [0, 100])
        assert result == pytest.approx(50.0)


class TestFormulaLayerGenerate:
    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    def test_returns_dict(self, formula):
        result = formula.generate(_minimal_features())
        assert isinstance(result, dict)

    def test_all_values_floats(self, formula):
        result = formula.generate(_minimal_features())
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"

    def test_all_values_non_negative(self, formula):
        result = formula.generate(_minimal_features())
        for k, v in result.items():
            assert v >= 0.0, f"{k} = {v} is negative"

    def test_all_registry_tendencies_produced(self, formula):
        result = formula.generate(_minimal_features())
        registry_names = _all_registry_names()
        for name in registry_names:
            assert name in result, f"Missing tendency: {name}"

    def test_position_profiles_affect_output(self, formula):
        pg_result = formula.generate(_minimal_features("PG"))
        c_result = formula.generate(_minimal_features("C"))
        # Centers should have higher post scores
        assert c_result["post_up"] > pg_result["post_up"]
        # Guards should have higher creation
        assert pg_result["driving_crossover"] > c_result["driving_crossover"]

    def test_high_usage_guard_has_high_shot(self, formula):
        f = _minimal_features("PG")
        f["usg_pct_proxy"] = 0.32
        f["fga_per36"] = 22.0
        result = formula.generate(f)
        assert result["shot"] > 50

    def test_rim_running_center_has_high_dunk(self, formula):
        f = _minimal_features("C")
        f["zone_fga_rate_ra"] = 0.40
        f["usg_pct_proxy"] = 0.18
        f["fg3a_rate"] = 0.0
        result = formula.generate(f)
        assert result["standing_dunk"] > 20
        assert result["shot_three"] < 10

    def test_high_three_rate_player_high_three_shot(self, formula):
        f = _minimal_features("SG")
        f["fg3a_rate"] = 0.55
        result = formula.generate(f)
        assert result["shot_three"] > 40

    def test_sub_zone_close_sums_to_parent(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_close_left"]
            + result["shot_close_middle"]
            + result["shot_close_right"]
        )
        assert total == pytest.approx(result["shot_close"], abs=1.0)

    def test_sub_zone_mid_sums_to_parent(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_mid_left"]
            + result["shot_mid_left_center"]
            + result["shot_mid_center"]
            + result["shot_mid_right_center"]
            + result["shot_mid_right"]
        )
        assert total == pytest.approx(result["shot_mid_range"], abs=1.0)

    def test_sub_zone_three_sums_to_parent(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_three_left"]
            + result["shot_three_left_center"]
            + result["shot_three_center"]
            + result["shot_three_right_center"]
            + result["shot_three_right"]
        )
        # With uniform distribution all zones equal, no max() floors are triggered
        assert total == pytest.approx(result["shot_three"], abs=1.0)

    def test_sub_zones_do_not_exceed_parent(self, formula):
        result = formula.generate(_minimal_features())
        parent_close = result["shot_close"]
        assert result["shot_close_left"] <= parent_close
        assert result["shot_close_middle"] <= parent_close
        assert result["shot_close_right"] <= parent_close

        parent_mid = result["shot_mid_range"]
        assert result["shot_mid_left"] <= parent_mid
        assert result["shot_mid_left_center"] <= parent_mid
        assert result["shot_mid_center"] <= parent_mid
        assert result["shot_mid_right_center"] <= parent_mid
        assert result["shot_mid_right"] <= parent_mid

        parent_three = result["shot_three"]
        assert result["shot_three_left"] <= parent_three
        assert result["shot_three_left_center"] <= parent_three
        assert result["shot_three_center"] <= parent_three
        assert result["shot_three_right_center"] <= parent_three
        assert result["shot_three_right"] <= parent_three

    def test_roll_vs_pop_guard_varies_with_three_point_rate(self, formula):
        low_3pt = _minimal_features("PG")
        low_3pt["fg3a_rate"] = 0.10
        high_3pt = _minimal_features("PG")
        high_3pt["fg3a_rate"] = 0.55
        low_r = formula.generate(low_3pt)
        high_r = formula.generate(high_3pt)
        assert high_r["roll_vs_pop"] > low_r["roll_vs_pop"]

    def test_roll_vs_pop_center_above_50(self, formula):
        f = _minimal_features("C")
        f["fg3a_rate"] = 0.0
        result = formula.generate(f)
        assert result["roll_vs_pop"] > 50

    def test_no_driving_dribble_move_inverse_of_creation(self, formula):
        low_usg = dict(_minimal_features("PG"))
        low_usg["usg_pct_proxy"] = 0.10
        high_usg = dict(_minimal_features("PG"))
        high_usg["usg_pct_proxy"] = 0.30
        low_r = formula.generate(low_usg)
        high_r = formula.generate(high_usg)
        assert low_r["no_driving_dribble_move"] > high_r["no_driving_dribble_move"]

    def test_shot_close_middle_varies_with_close_distribution(self, formula):
        # Player A: mostly-left close distribution
        f_left = _minimal_features("C")
        f_left["sub_zone_distribution_close"] = {"left": 45.0, "middle": 30.0, "right": 25.0}
        # Player B: mostly-right close distribution
        f_right = _minimal_features("C")
        f_right["sub_zone_distribution_close"] = {"left": 25.0, "middle": 30.0, "right": 45.0}
        # Player C: mostly-middle close distribution
        f_mid = _minimal_features("C")
        f_mid["sub_zone_distribution_close"] = {"left": 20.0, "middle": 45.0, "right": 35.0}

        r_left = formula.generate(f_left)
        r_right = formula.generate(f_right)
        r_mid = formula.generate(f_mid)

        # shot_close_middle should reflect the input distribution, not be a constant
        assert r_mid["shot_close_middle"] > r_left["shot_close_middle"]
        assert r_mid["shot_close_middle"] > r_right["shot_close_middle"]
        # All values must stay under the hard cap of 50
        assert r_left["shot_close_middle"] <= 50
        assert r_right["shot_close_middle"] <= 50
        assert r_mid["shot_close_middle"] <= 50

    def test_close_middle_not_pathologically_dominant(self, formula):
        f = _minimal_features("C")
        f["zone_fga_rate_paint"] = 0.30
        f["sub_zone_distribution_close"] = {"left": 10.0, "middle": 80.0, "right": 10.0}

        result = formula.generate(f)
        close_parent = result["shot_close"]

        assert result["shot_close_middle"] <= close_parent * 0.55 + 1e-6
        assert result["shot_close_left"] >= close_parent * 0.20 - 1e-6
        assert result["shot_close_right"] >= close_parent * 0.20 - 1e-6

    def test_close_left_right_bias_is_preserved(self, formula):
        f = _minimal_features("SF")
        f["zone_fga_rate_paint"] = 0.25
        f["sub_zone_distribution_close"] = {"left": 42.0, "middle": 43.0, "right": 15.0}

        result = formula.generate(f)
        assert result["shot_close_left"] > result["shot_close_right"]

    def test_shot_close_gets_mild_ra_influence(self, formula):
        low_ra = _minimal_features("C")
        low_ra["zone_fga_rate_paint"] = 0.15
        low_ra["zone_fga_rate_ra"] = 0.10

        high_ra = _minimal_features("C")
        high_ra["zone_fga_rate_paint"] = 0.15
        high_ra["zone_fga_rate_ra"] = 0.35

        low_result = formula.generate(low_ra)
        high_result = formula.generate(high_ra)

        assert high_result["shot_close"] > low_result["shot_close"]


class TestFormulaLayerCompute:
    def test_compute_returns_integers(self):
        formula = FormulaLayer()
        result = formula.compute(_minimal_features(), "SG")
        for k, v in result.items():
            assert isinstance(v, int), f"{k} is not int"

    def test_compute_values_are_multiples_of_5(self):
        formula = FormulaLayer()
        result = formula.compute(_minimal_features(), "SG")
        for k, v in result.items():
            assert v % 5 == 0, f"{k} = {v} is not a multiple of 5"

    def test_apply_locked_rules_stepback(self):
        formula = FormulaLayer()
        t = {"stepback_jumper_three": 40, "stepback_jumper_mid_range": 20}
        result = formula.apply_locked_rules(t)
        assert result["stepback_jumper_three"] <= result["stepback_jumper_mid_range"] + 5

    def test_shot_does_not_exceed_75(self):
        formula = FormulaLayer()
        for pos in ("PG", "SG", "SF", "PF", "C"):
            f = _minimal_features(pos)
            f["usg_pct_proxy"] = 0.40
            f["fga_per36"] = 30.0
            f["fg3a_rate"] = 0.60
            result = formula.generate(f)
            assert result["shot"] <= 75.0, f"{pos}: shot={result['shot']} exceeds 75"

    def test_touches_does_not_exceed_65(self):
        formula = FormulaLayer()
        for pos in ("PG", "SG", "SF", "PF", "C"):
            f = _minimal_features(pos)
            f["usg_pct_proxy"] = 0.40
            f["ast_per36"] = 15.0
            result = formula.generate(f)
            assert result["touches"] <= 65.0, f"{pos}: touches={result['touches']} exceeds 65"

    def test_apply_locked_rules_off_screen_three_le_shot_three(self):
        formula = FormulaLayer()
        t = {"off_screen_shot_three": 50, "shot_three": 30}
        result = formula.apply_locked_rules(t)
        assert result["off_screen_shot_three"] <= result["shot_three"]

    def test_apply_locked_rules_contested_three_le_shot_three(self):
        formula = FormulaLayer()
        t = {"contested_jumper_three": 45, "shot_three": 20}
        result = formula.apply_locked_rules(t)
        assert result["contested_jumper_three"] <= result["shot_three"]

    def test_apply_locked_rules_no_setup_dribble_cap_35(self):
        formula = FormulaLayer()
        t = {"no_setup_dribble": 50}
        result = formula.apply_locked_rules(t)
        assert result["no_setup_dribble"] <= 35

    def test_apply_locked_rules_roll_vs_pop_clamped(self):
        formula = FormulaLayer()
        t_low = {"roll_vs_pop": 2}
        t_high = {"roll_vs_pop": 99}
        assert formula.apply_locked_rules(t_low)["roll_vs_pop"] >= 5
        assert formula.apply_locked_rules(t_high)["roll_vs_pop"] <= 95

    def test_apply_locked_rules_post_hooks_zero_when_post_up_low(self):
        formula = FormulaLayer()
        t = {"post_up": 5, "post_hook_left": 10, "post_hook_right": 8}
        result = formula.apply_locked_rules(t)
        assert result["post_hook_left"] == 0
        assert result["post_hook_right"] == 0

    def test_low_three_point_rate_produces_low_three_tendencies(self):
        """Giannis-like big: very low three-point tendencies."""
        formula = FormulaLayer()
        f = _minimal_features("PF")
        f["fg3a_rate"] = 0.05
        f["zone_fga_rate_ra"] = 0.40
        f["zone_fga_rate_paint"] = 0.20
        f["usg_pct_proxy"] = 0.30
        result = formula.generate(f)
        assert result["shot_three"] < 10, f"shot_three={result['shot_three']}"
        assert result["spot_up_shot_three"] < 10, f"spot_up_shot_three={result['spot_up_shot_three']}"

    def test_sg_post_hooks_locked_to_zero(self):
        """Booker-like SG: post tendencies should be near 0."""
        formula = FormulaLayer()
        f = _minimal_features("SG")
        f["zone_fga_rate_ra"] = 0.15
        f["zone_fga_rate_paint"] = 0.10
        result = formula.generate(f)
        locked = formula.apply_locked_rules({k: round(v) for k, v in result.items()})
        assert locked["post_hook_left"] == 0
        assert locked["post_hook_right"] == 0


class TestFormulaBugFixes:
    """Regression tests for the 10 bug fixes."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    def test_standing_dunk_scales_with_drive_boost(self, formula):
        """PG standing_dunk < PF standing_dunk (same zra); PF has higher drive_boost."""
        base = {"zone_fga_rate_ra": 0.30, "usg_pct_proxy": 0.20,
                "fga_per36": 12.0, "fg3a_rate": 0.35, "fta_rate": 0.30,
                "ast_per36": 5.0, "pts_per36": 18.0, "stl_per36": 1.0,
                "blk_per36": 0.4, "pf_per36": 2.5, "oreb_pct_proxy": 0.10,
                "zone_fga_rate_paint": 0.15, "zone_fga_rate_mid_left": 0.08,
                "zone_fga_rate_mid_center": 0.07, "zone_fga_rate_mid_right": 0.08,
                "zone_fga_rate_corner3_left": 0.10, "zone_fga_rate_corner3_right": 0.10,
                "zone_fga_rate_above_break3": 0.22,
                "sub_zone_distribution_close": {"left": 33.3, "middle": 33.4, "right": 33.3},
                "sub_zone_distribution_mid": {"left": 20.0, "left_center": 20.0, "center": 20.0, "right_center": 20.0, "right": 20.0},
                "sub_zone_distribution_three": {"left": 20.0, "left_center": 20.0, "center": 20.0, "right_center": 20.0, "right": 20.0}}
        pg_f = dict(base, position="PG")
        pf_f = dict(base, position="PF")
        pg_r = formula.generate(pg_f)
        pf_r = formula.generate(pf_f)
        assert pg_r["standing_dunk"] < pf_r["standing_dunk"]

    def test_alley_oop_uses_drive_boost(self, formula):
        """SF with high zra should get a reasonable alley_oop (not tiny)."""
        f = _minimal_features("SF")
        f["zone_fga_rate_ra"] = 0.35
        result = formula.generate(f)
        assert result["alley_oop"] > 10

    def test_putback_scales_with_oreb_pct(self, formula):
        """High oreb_pct PF should get putback > 15."""
        f = _minimal_features("PF")
        f["oreb_pct_proxy"] = 0.25
        result = formula.generate(f)
        assert result["putback"] > 15

    def test_transition_pull_up_three_gated_for_bigs(self, formula):
        """Center with low fg3a_rate should get transition_pull_up_three < 10."""
        f = _minimal_features("C")
        f["fg3a_rate"] = 0.10
        result = formula.generate(f)
        assert result["transition_pull_up_three"] < 10

    def test_flashy_dunk_low_for_centers(self, formula):
        """Center flashy_dunk should be very low."""
        f = _minimal_features("C")
        f["zone_fga_rate_ra"] = 0.35
        result = formula.generate(f)
        assert result["flashy_dunk"] < 10

    def test_drive_right_reads_from_features(self, formula):
        """When drive_right_bias is provided, it should be used (differs from default 50.0)."""
        f = _minimal_features()
        f["drive_right_bias"] = 62.0
        result = formula.generate(f)
        assert result["drive_right"] == pytest.approx(62.0)
        assert result["drive_right"] != pytest.approx(50.0)

    def test_drive_right_defaults_to_50(self, formula):
        """Without drive_right_bias, drive_right should default to 50."""
        result = formula.generate(_minimal_features())
        assert result["drive_right"] == pytest.approx(50.0)

    def test_play_discipline_center_higher_than_pg(self, formula):
        """Center play_discipline > PG play_discipline with the same USG."""
        usg = 0.22
        pg_f = dict(_minimal_features("PG"), usg_pct_proxy=usg)
        c_f = dict(_minimal_features("C"), usg_pct_proxy=usg)
        pg_r = formula.generate(pg_f)
        c_r = formula.generate(c_f)
        assert c_r["play_discipline"] > pg_r["play_discipline"]

    def test_play_discipline_capped_to_55(self, formula):
        """play_discipline should never exceed 55 after tuning."""
        low_usg_center = _minimal_features("C")
        low_usg_center["usg_pct_proxy"] = 0.10
        result = formula.generate(low_usg_center)
        assert result["play_discipline"] <= 55.0

    def test_play_discipline_role_higher_than_star_same_position(self, formula):
        """Low-usage role player should get higher play_discipline than high-usage star."""
        role = _minimal_features("SF")
        role["usg_pct_proxy"] = 0.15
        star = _minimal_features("SF")
        star["usg_pct_proxy"] = 0.34
        role_r = formula.generate(role)
        star_r = formula.generate(star)
        assert role_r["play_discipline"] > star_r["play_discipline"]

    def test_no_driving_dribble_move_clamped(self, formula):
        """no_driving_dribble_move must be in [15, 75] for all positions."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            val = result["no_driving_dribble_move"]
            assert 15.0 <= val <= 75.0, f"{pos}: no_driving_dribble_move={val}"

    def test_all_values_non_negative_after_fixes(self, formula):
        """All tendency values must remain non-negative after bug fixes."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            for k, v in result.items():
                assert v >= 0.0, f"{pos} {k} = {v} is negative"

    def test_all_registry_tendencies_produced_after_fixes(self, formula):
        """All 99 registry tendencies must still be produced after bug fixes."""
        result = formula.generate(_minimal_features())
        registry_names = _all_registry_names()
        for name in registry_names:
            assert name in result, f"Missing tendency: {name}"


class TestDifferentiationFixes:
    """Tests verifying that hard-coded/near-constant tendencies now differentiate players."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    # --- drive_right_bias ---
    def test_drive_right_uses_drive_right_bias_when_provided(self, formula):
        """drive_right should reflect drive_right_bias from features, not always 50."""
        right_heavy = _minimal_features("PG")
        right_heavy["drive_right_bias"] = 68.0
        left_heavy = _minimal_features("PG")
        left_heavy["drive_right_bias"] = 32.0
        r_right = formula.generate(right_heavy)
        r_left = formula.generate(left_heavy)
        assert r_right["drive_right"] == pytest.approx(68.0)
        assert r_left["drive_right"] == pytest.approx(32.0)
        assert r_right["drive_right"] != r_left["drive_right"]

    def test_drive_right_defaults_50_when_no_bias_key(self, formula):
        """Without drive_right_bias key, drive_right should default to 50."""
        f = _minimal_features("SG")
        assert "drive_right_bias" not in f
        result = formula.generate(f)
        assert result["drive_right"] == pytest.approx(50.0)

    # --- triple_threat_idle ---
    def test_triple_threat_idle_high_usage_higher_than_low_usage(self, formula):
        """High-usage ISO creators should have higher triple_threat_idle (more ball-dominant)."""
        high_usg = dict(_minimal_features("PG"), usg_pct_proxy=0.33)
        low_usg = dict(_minimal_features("PG"), usg_pct_proxy=0.12)
        r_high = formula.generate(high_usg)
        r_low = formula.generate(low_usg)
        assert r_high["triple_threat_idle"] > r_low["triple_threat_idle"]

    def test_triple_threat_idle_not_constant(self, formula):
        """triple_threat_idle must not be identical for all players."""
        pg_high = dict(_minimal_features("PG"), usg_pct_proxy=0.35)
        pg_low = dict(_minimal_features("PG"), usg_pct_proxy=0.10)
        assert formula.generate(pg_high)["triple_threat_idle"] != pytest.approx(
            formula.generate(pg_low)["triple_threat_idle"]
        )

    # --- roll_vs_pop for guards ---
    def test_roll_vs_pop_guard_shooter_pops_more(self, formula):
        """Guard with high 3pt rate should get higher roll_vs_pop (prefer popping)."""
        shooter = dict(_minimal_features("PG"), fg3a_rate=0.48)
        driver = dict(_minimal_features("PG"), fg3a_rate=0.12)
        r_shooter = formula.generate(shooter)
        r_driver = formula.generate(driver)
        assert r_shooter["roll_vs_pop"] > r_driver["roll_vs_pop"]

    def test_roll_vs_pop_guard_not_always_50(self, formula):
        """Guards with different 3pt rates should not all get exactly 50."""
        extreme_shooter = dict(_minimal_features("SG"), fg3a_rate=0.55)
        result = formula.generate(extreme_shooter)
        assert result["roll_vs_pop"] != pytest.approx(50.0)

    # --- contest_shot ---
    def test_contest_shot_center_higher_than_pg_base(self, formula):
        """Center should have higher contest_shot base than PG (same block/steal stats)."""
        pg_f = _minimal_features("PG")
        c_f = _minimal_features("C")
        # Set same block/steal for fair comparison
        for f in (pg_f, c_f):
            f["blk_per36"] = 0.5
            f["stl_per36"] = 1.0
        r_pg = formula.generate(pg_f)
        r_c = formula.generate(c_f)
        assert r_c["contest_shot"] > r_pg["contest_shot"]

    def test_contest_shot_scales_with_blocks_and_steals(self, formula):
        """Player with more blocks and steals should have higher contest_shot."""
        weak_def = dict(_minimal_features("PF"), blk_per36=0.2, stl_per36=0.4)
        strong_def = dict(_minimal_features("PF"), blk_per36=2.0, stl_per36=1.8)
        r_weak = formula.generate(weak_def)
        r_strong = formula.generate(strong_def)
        assert r_strong["contest_shot"] > r_weak["contest_shot"]

    # --- hard_foul ---
    def test_hard_foul_pg_nonzero_with_high_pf(self, formula):
        """Physical PG (high pf_per36) should have non-negligible hard_foul."""
        f = dict(_minimal_features("PG"), pf_per36=4.2)
        result = formula.generate(f)
        assert result["hard_foul"] > 3.0

    def test_hard_foul_varies_with_fouls_for_guards(self, formula):
        """hard_foul should differ between high-fouling and low-fouling guards."""
        low_foul = dict(_minimal_features("PG"), pf_per36=1.5)
        high_foul = dict(_minimal_features("PG"), pf_per36=4.0)
        r_low = formula.generate(low_foul)
        r_high = formula.generate(high_foul)
        assert r_high["hard_foul"] > r_low["hard_foul"]

    # --- step_through_shot ---
    def test_step_through_shot_pg_nonzero_with_paint_work(self, formula):
        """PG with paint scoring should have non-trivial step_through_shot."""
        f = dict(_minimal_features("PG"), zone_fga_rate_paint=0.20)
        result = formula.generate(f)
        assert result["step_through_shot"] > 1.0

    def test_step_through_shot_scales_with_paint_for_guards(self, formula):
        """step_through_shot should grow with paint zone rate even for guards."""
        no_paint = dict(_minimal_features("PG"), zone_fga_rate_paint=0.0)
        heavy_paint = dict(_minimal_features("PG"), zone_fga_rate_paint=0.25)
        r_no = formula.generate(no_paint)
        r_heavy = formula.generate(heavy_paint)
        assert r_heavy["step_through_shot"] > r_no["step_through_shot"]

    # --- standing_dunk ---
    def test_standing_dunk_athletic_pg_nonzero(self, formula):
        """Rim-attacking PG should have non-trivial standing_dunk (> 3)."""
        f = dict(_minimal_features("PG"), zone_fga_rate_ra=0.35)
        result = formula.generate(f)
        assert result["standing_dunk"] > 3.0

    def test_standing_dunk_center_much_higher_than_pg(self, formula):
        """Center standing_dunk should still dominate over PG standing_dunk."""
        pg_f = dict(_minimal_features("PG"), zone_fga_rate_ra=0.30)
        c_f = dict(_minimal_features("C"), zone_fga_rate_ra=0.30)
        r_pg = formula.generate(pg_f)
        r_c = formula.generate(c_f)
        assert r_c["standing_dunk"] > r_pg["standing_dunk"]

    # --- registry completeness ---
    def test_all_99_tendencies_still_produced(self, formula):
        """All 99 registry tendencies must be produced after differentiation fixes."""
        result = formula.generate(_minimal_features())
        registry_names = _all_registry_names()
        for name in registry_names:
            assert name in result, f"Missing: {name}"
        assert len(registry_names) == 99

    def test_all_values_non_negative_differentiation(self, formula):
        """All tendency values must remain >= 0 for all positions after fixes."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            for k, v in result.items():
                assert v >= 0.0, f"{pos} {k} = {v}"


class TestImprovementPlan:
    """Tests for the 10-point improvement plan formulas."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    # --- 1. triple_threat_idle multi-factor ---
    def test_idle_big_higher_than_guard_same_usage(self, formula):
        """Tall/heavy C should idle more than a PG with identical stats."""
        c_f = dict(_minimal_features("C"), height_inches=84, weight_lbs=260)
        pg_f = dict(_minimal_features("PG"), height_inches=74, weight_lbs=190)
        r_c = formula.generate(c_f)
        r_pg = formula.generate(pg_f)
        assert r_c["triple_threat_idle"] > r_pg["triple_threat_idle"]

    def test_idle_big_low_ast_lower_than_big_high_ast(self, formula):
        """Traditional big (low AST) gets big_rescue → very low idle; unicorn big (high AST) gets higher."""
        low_ast_big = dict(_minimal_features("C"), ast_per36=2.0)
        high_ast_big = dict(_minimal_features("C"), ast_per36=6.0)
        r_low = formula.generate(low_ast_big)
        r_high = formula.generate(high_ast_big)
        assert r_high["triple_threat_idle"] > r_low["triple_threat_idle"]

    def test_idle_clamped_between_5_and_50(self, formula):
        """triple_threat_idle must stay within [5, 50] for all positions."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            val = result["triple_threat_idle"]
            assert 5.0 <= val <= 50.0, f"{pos}: triple_threat_idle={val}"

    # --- 2. crash formula ---
    def test_crash_hustle_data_present(self, formula):
        """With hustle data available, crash should scale with loose balls + charges."""
        low_hustle = dict(_minimal_features("SG"),
                          hustle_loose_balls_pg=0.1, hustle_charges_drawn_pg=0.0)
        high_hustle = dict(_minimal_features("SG"),
                           hustle_loose_balls_pg=1.3, hustle_charges_drawn_pg=0.4)
        r_low = formula.generate(low_hustle)
        r_high = formula.generate(high_hustle)
        assert r_high["crash"] > r_low["crash"]

    def test_crash_position_defaults_when_no_hustle_data(self, formula):
        """Without hustle data, crash falls back to position-based defaults."""
        c_f = _minimal_features("C")   # no hustle stats → sentinel -1.0
        pg_f = _minimal_features("PG")
        r_c = formula.generate(c_f)
        r_pg = formula.generate(pg_f)
        assert r_c["crash"] > r_pg["crash"]
        assert r_c["crash"] == pytest.approx(22.0)
        assert r_pg["crash"] == pytest.approx(15.0)

    def test_crash_capped_at_45(self, formula):
        """crash should never exceed 45 even for elite hustlers."""
        f = dict(_minimal_features("PF"),
                 hustle_loose_balls_pg=2.0, hustle_charges_drawn_pg=1.0)
        result = formula.generate(f)
        assert result["crash"] <= 45.0

    def test_crash_non_negative(self, formula):
        """crash must always be non-negative."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            assert result["crash"] >= 0.0, f"{pos}: crash={result['crash']}"

    # --- 3. height/weight as speed proxy ---
    def test_tall_heavy_player_higher_idle(self, formula):
        """Tall + heavy player should produce higher idle than short + light."""
        tall = dict(_minimal_features("SF"), height_inches=83, weight_lbs=260)
        short = dict(_minimal_features("SF"), height_inches=73, weight_lbs=185)
        assert formula.generate(tall)["triple_threat_idle"] > formula.generate(short)["triple_threat_idle"]

    def test_tall_player_more_power_post_moves(self, formula):
        """Taller center should get higher post_up than a shorter one."""
        tall_c = dict(_minimal_features("C"), height_inches=84, weight_lbs=265)
        short_c = dict(_minimal_features("C"), height_inches=78, weight_lbs=220,
                       zone_fga_rate_ra=0.35, zone_fga_rate_paint=0.20)
        tall_c.update(zone_fga_rate_ra=0.35, zone_fga_rate_paint=0.20)
        r_tall = formula.generate(tall_c)
        r_short = formula.generate(short_c)
        assert r_tall["post_up"] > r_short["post_up"]

    # --- 4. shooting efficiency for contested shots ---
    def test_high_fg_pct_yields_higher_contested_mid(self, formula):
        """High FG% player should have higher contested_jumper_mid_range."""
        high_eff = dict(_minimal_features("SG"), fg_pct=0.55)
        low_eff = dict(_minimal_features("SG"), fg_pct=0.40)
        r_high = formula.generate(high_eff)
        r_low = formula.generate(low_eff)
        assert r_high["contested_jumper_mid_range"] > r_low["contested_jumper_mid_range"]

    def test_high_fg3_pct_yields_higher_contested_three(self, formula):
        """High 3P% player should have higher contested_jumper_three."""
        high_3 = dict(_minimal_features("SG"), fg3_pct=0.42)
        low_3 = dict(_minimal_features("SG"), fg3_pct=0.30)
        r_high = formula.generate(high_3)
        r_low = formula.generate(low_3)
        assert r_high["contested_jumper_three"] > r_low["contested_jumper_three"]

    def test_high_fg_pct_yields_higher_stepback_mid(self, formula):
        """High FG% player should have higher stepback_jumper_mid_range."""
        high_eff = dict(_minimal_features("PG"), fg_pct=0.55)
        low_eff = dict(_minimal_features("PG"), fg_pct=0.40)
        assert (formula.generate(high_eff)["stepback_jumper_mid_range"] >
                formula.generate(low_eff)["stepback_jumper_mid_range"])

    def test_high_fg3_pct_yields_higher_stepback_three(self, formula):
        """High 3P% player should have higher stepback_jumper_three."""
        high_3 = dict(_minimal_features("PG"), fg3_pct=0.42)
        low_3 = dict(_minimal_features("PG"), fg3_pct=0.30)
        assert (formula.generate(high_3)["stepback_jumper_three"] >
                formula.generate(low_3)["stepback_jumper_three"])

    # --- 5. ast_to_tov for flashy pass gating ---
    def test_disciplined_passer_lower_flashy_pass(self, formula):
        """Player with high ast_to_tov (disciplined) should have lower flashy_pass."""
        disciplined = dict(_minimal_features("PG"), ast_per36=8.0, ast_to_tov=3.5)
        reckless = dict(_minimal_features("PG"), ast_per36=8.0, ast_to_tov=0.5)
        r_d = formula.generate(disciplined)
        r_r = formula.generate(reckless)
        assert r_d["flashy_pass"] < r_r["flashy_pass"]

    def test_flashy_pass_still_driven_by_assists(self, formula):
        """Regardless of ast_to_tov, more assists → more flashy_pass."""
        high_ast = dict(_minimal_features("PG"), ast_per36=12.0, ast_to_tov=1.5)
        low_ast = dict(_minimal_features("PG"), ast_per36=2.0, ast_to_tov=1.5)
        assert formula.generate(high_ast)["flashy_pass"] > formula.generate(low_ast)["flashy_pass"]

    # --- 6. ISO confidence ---
    def test_high_ts_pct_yields_higher_iso(self, formula):
        """High-TS% player should have higher ISO tendencies."""
        elite = dict(_minimal_features("SG"), ts_pct=0.65, usg_pct_proxy=0.28)
        average = dict(_minimal_features("SG"), ts_pct=0.48, usg_pct_proxy=0.28)
        r_elite = formula.generate(elite)
        r_avg = formula.generate(average)
        assert r_elite["iso_vs_average_defender"] > r_avg["iso_vs_average_defender"]
        assert r_elite["iso_vs_poor_defender"] > r_avg["iso_vs_poor_defender"]

    def test_iso_vs_elite_lower_than_vs_poor(self, formula):
        """iso_vs_elite_defender should always be less than iso_vs_poor_defender."""
        for pos in ("PG", "SG", "SF", "PF", "C"):
            result = formula.generate(_minimal_features(pos))
            assert result["iso_vs_elite_defender"] <= result["iso_vs_poor_defender"]

    # --- 7. Percentile influence on shot ---
    def test_high_percentile_boosts_shot(self, formula):
        """Top-percentile scorer should get a shot boost vs low-percentile scorer."""
        top = dict(_minimal_features("SG"), pctile_pts=0.90)
        bottom = dict(_minimal_features("SG"), pctile_pts=0.15)
        assert formula.generate(top)["shot"] > formula.generate(bottom)["shot"]

    # --- 8. Post move size diversification ---
    def test_center_higher_drop_step_than_pg(self, formula):
        """Center should have higher post_drop_step than PG with same paint zone."""
        c_f = dict(_minimal_features("C"), zone_fga_rate_ra=0.35, zone_fga_rate_paint=0.20)
        pg_f = dict(_minimal_features("PG"), zone_fga_rate_ra=0.35, zone_fga_rate_paint=0.20)
        assert formula.generate(c_f)["post_drop_step"] > formula.generate(pg_f)["post_drop_step"]

    def test_shorter_player_more_post_spin(self, formula):
        """Shorter player should have relatively higher post_spin (finesse move)."""
        short_sf = dict(_minimal_features("SF"), height_inches=74,
                        zone_fga_rate_ra=0.30, zone_fga_rate_paint=0.20)
        tall_sf = dict(_minimal_features("SF"), height_inches=82,
                       zone_fga_rate_ra=0.30, zone_fga_rate_paint=0.20)
        r_short = formula.generate(short_sf)
        r_tall = formula.generate(tall_sf)
        assert r_short["post_spin"] > r_tall["post_spin"]

    # --- 9. Dribble move position diversification ---
    def test_guard_higher_crossover_than_center(self, formula):
        """PG should have a much higher driving_crossover than C."""
        pg_f = dict(_minimal_features("PG"), usg_pct_proxy=0.22)
        c_f = dict(_minimal_features("C"), usg_pct_proxy=0.22)
        assert formula.generate(pg_f)["driving_crossover"] > formula.generate(c_f)["driving_crossover"]

    def test_guard_higher_behind_back_than_big(self, formula):
        """PG should have higher driving_behind_the_back than PF."""
        pg_f = dict(_minimal_features("PG"), usg_pct_proxy=0.22)
        pf_f = dict(_minimal_features("PF"), usg_pct_proxy=0.22)
        assert formula.generate(pg_f)["driving_behind_the_back"] > formula.generate(pf_f)["driving_behind_the_back"]

    def test_shorter_guard_more_hesitation_than_taller(self, formula):
        """Shorter guard should have more driving_dribble_hesitation than taller guard."""
        short_pg = dict(_minimal_features("PG"), height_inches=72, weight_lbs=175)
        tall_pg = dict(_minimal_features("PG"), height_inches=80, weight_lbs=210)
        r_short = formula.generate(short_pg)
        r_tall = formula.generate(tall_pg)
        assert r_short["driving_dribble_hesitation"] > r_tall["driving_dribble_hesitation"]

    # --- 10. triple_threat_idle archetype validation ---
    def test_iso_creator_higher_idle_than_pure_shooter(self, formula):
        """ISO creator (high USG, low fg3a) should idle more than pure shooter (low USG, high fg3a)."""
        iso_creator = dict(_minimal_features("SG"), usg_pct_proxy=0.30, fg3a_rate=0.20,
                           zone_fga_rate_ra=0.15)
        pure_shooter = dict(_minimal_features("SG"), usg_pct_proxy=0.15, fg3a_rate=0.55,
                            zone_fga_rate_ra=0.10)
        r_iso = formula.generate(iso_creator)
        r_shooter = formula.generate(pure_shooter)
        assert r_iso["triple_threat_idle"] > r_shooter["triple_threat_idle"]

    def test_slasher_lower_idle_than_methodical_creator(self, formula):
        """Slasher (high zra, lower USG) should idle less than methodical ball-handler."""
        slasher = dict(_minimal_features("PG"), usg_pct_proxy=0.20, zone_fga_rate_ra=0.50,
                       fg3a_rate=0.20)
        methodical = dict(_minimal_features("PG"), usg_pct_proxy=0.30, zone_fga_rate_ra=0.15,
                          fg3a_rate=0.25)
        r_slash = formula.generate(slasher)
        r_meth = formula.generate(methodical)
        assert r_slash["triple_threat_idle"] < r_meth["triple_threat_idle"]

    def test_traditional_big_very_low_idle(self, formula):
        """Traditional big (C, low AST) should get big_rescue → very low idle (≤ 10).

        10.0 is the upper bound for the 'Gobert/Capela' bucket in the expected archetype
        table (5–10 idle). The big rescue factor (0.3×) keeps these players firmly low.
        """
        trad_big = dict(_minimal_features("C"), ast_per36=1.5, fg3a_rate=0.02,
                        zone_fga_rate_ra=0.55, usg_pct_proxy=0.15)
        result = formula.generate(trad_big)
        assert result["triple_threat_idle"] <= 10.0

    def test_unicorn_big_higher_idle_than_traditional_big(self, formula):
        """Unicorn big (C, high AST like Jokic) should idle more than traditional rim-runner."""
        unicorn = dict(_minimal_features("C"), ast_per36=8.0, usg_pct_proxy=0.28,
                       fg3a_rate=0.15, zone_fga_rate_ra=0.30)
        trad = dict(_minimal_features("C"), ast_per36=1.5, usg_pct_proxy=0.12,
                    fg3a_rate=0.02, zone_fga_rate_ra=0.55)
        r_unicorn = formula.generate(unicorn)
        r_trad = formula.generate(trad)
        assert r_unicorn["triple_threat_idle"] > r_trad["triple_threat_idle"]


# ---------------------------------------------------------------------------
# Calibration tests: 5-player post tendency targets (±5 tolerance)
# ---------------------------------------------------------------------------

_POST_CALIBRATION_KEYS = [
    "post_up", "post_shimmy_shot", "post_face_up", "post_back_down",
    "post_aggressive_backdown", "shoot_from_post", "post_hook_left",
    "post_hook_right", "post_fade_left", "post_fade_right",
    "post_hop_shot", "post_step_back_shot", "post_drive", "post_spin",
    "post_drop_step",
]

_POST_TARGETS = {
    "embiid": {
        "post_up": 35, "post_shimmy_shot": 10, "post_face_up": 20,
        "post_back_down": 20, "post_aggressive_backdown": 15, "shoot_from_post": 20,
        "post_hook_left": 5, "post_hook_right": 5,
        "post_fade_left": 10, "post_fade_right": 10,
        "post_hop_shot": 10, "post_step_back_shot": 10,
        "post_drive": 20, "post_spin": 10, "post_drop_step": 10,
    },
    "wemby": {
        "post_up": 20, "post_shimmy_shot": 5, "post_face_up": 15,
        "post_back_down": 15, "post_aggressive_backdown": 10, "shoot_from_post": 15,
        "post_hook_left": 5, "post_hook_right": 5,
        "post_fade_left": 5, "post_fade_right": 5,
        "post_hop_shot": 5, "post_step_back_shot": 5,
        "post_drive": 10, "post_spin": 5, "post_drop_step": 5,
    },
    "lebron": {
        "post_up": 15, "post_shimmy_shot": 5, "post_face_up": 10,
        "post_back_down": 10, "post_aggressive_backdown": 5, "shoot_from_post": 10,
        "post_hook_left": 5, "post_hook_right": 5,
        "post_fade_left": 5, "post_fade_right": 5,
        "post_hop_shot": 5, "post_step_back_shot": 5,
        "post_drive": 10, "post_spin": 5, "post_drop_step": 5,
    },
    "luka": {
        "post_up": 10, "post_shimmy_shot": 0, "post_face_up": 5,
        "post_back_down": 5, "post_aggressive_backdown": 0, "shoot_from_post": 5,
        "post_hook_left": 0, "post_hook_right": 0,
        "post_fade_left": 0, "post_fade_right": 0,
        "post_hop_shot": 0, "post_step_back_shot": 0,
        "post_drive": 5, "post_spin": 0, "post_drop_step": 0,
    },
    "curry": {k: 0 for k in _POST_CALIBRATION_KEYS},
}


def _post_player_features(pos: str, height_inches: int, weight_lbs: int,
                           zpaint: float, zra: float) -> dict:
    """Build a minimal feature dict for a specific player's post calibration."""
    f = _minimal_features(pos)
    f.update(
        height_inches=height_inches,
        weight_lbs=weight_lbs,
        zone_fga_rate_paint=zpaint,
        zone_fga_rate_ra=zra,
        fg3a_rate=0.15,
    )
    return f


class TestPostCalibration:
    """Calibration tests ensuring post tendencies for 5 archetypes are within ±5 of targets."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    def _quantize(self, v: float) -> int:
        return max(0, min(100, 5 * round(v / 5)))

    def _check_player(self, formula, name: str, features: dict) -> None:
        raw = formula.generate(features)
        targets = _POST_TARGETS[name]
        for key in _POST_CALIBRATION_KEYS:
            computed = self._quantize(raw[key])
            target = targets[key]
            assert abs(computed - target) <= 5, (
                f"{name} {key}: computed={computed}, target={target}, "
                f"diff={abs(computed - target)} > 5"
            )

    def test_embiid_post_tendencies(self, formula):
        """Embiid (C, 7'0"/84in, 280lbs): dominant post center."""
        feats = _post_player_features("C", 84, 280, zpaint=0.30, zra=0.20)
        self._check_player(formula, "embiid", feats)

    def test_wembanyama_post_tendencies(self, formula):
        """Wembanyama (C, 7'4"/88in, 225lbs): stretch center, less post than Embiid."""
        feats = _post_player_features("C", 88, 225, zpaint=0.20, zra=0.15)
        self._check_player(formula, "wemby", feats)

    def test_lebron_post_tendencies(self, formula):
        """LeBron (SF, 6'9"/81in, 250lbs): versatile forward."""
        feats = _post_player_features("SF", 81, 250, zpaint=0.30, zra=0.20)
        self._check_player(formula, "lebron", feats)

    def test_luka_post_tendencies(self, formula):
        """Luka (PG, 6'7"/79in, 230lbs): tall, heavy guard with some post play."""
        feats = _post_player_features("PG", 79, 230, zpaint=0.30, zra=0.20)
        self._check_player(formula, "luka", feats)

    def test_curry_post_tendencies_all_zero(self, formula):
        """Curry (PG, 6'2"/74in, 185lbs): small guard — all post tendencies must be 0."""
        feats = _post_player_features("PG", 74, 185, zpaint=0.15, zra=0.05)
        raw = formula.generate(feats)
        for key in _POST_CALIBRATION_KEYS:
            computed = self._quantize(raw[key])
            assert computed == 0, (
                f"Curry {key}: expected 0, got {computed}"
            )

    def test_wemby_lower_post_up_than_embiid(self, formula):
        """Wembanyama should have lower post_up than Embiid (weight advantage)."""
        embiid = _post_player_features("C", 84, 280, zpaint=0.30, zra=0.20)
        wemby = _post_player_features("C", 88, 225, zpaint=0.20, zra=0.15)
        assert formula.generate(embiid)["post_up"] > formula.generate(wemby)["post_up"]

    def test_post_score_zero_for_size_gate_players(self, formula):
        """Players below height/weight threshold get post_score=0 (hard zeros)."""
        small_guard = _post_player_features("PG", 75, 205, zpaint=0.35, zra=0.25)
        raw = formula.generate(small_guard)
        for key in _POST_CALIBRATION_KEYS:
            assert self._quantize(raw[key]) == 0, (
                f"Small guard {key} should be 0, got {self._quantize(raw[key])}"
            )


# ---------------------------------------------------------------------------
# Rim-runner archetype calibration tests
# ---------------------------------------------------------------------------

def _rim_runner_features() -> dict:
    """Gobert/Capela-type rim-running center."""
    f = _minimal_features("C")
    f.update(
        zone_fga_rate_ra=0.45,
        zone_fga_rate_paint=0.10,
        ast_per36=1.5,
        usg_pct_proxy=0.14,
        fg3a_rate=0.02,
    )
    return f


def _jokic_features() -> dict:
    """Elite post playmaker (Jokic type): high assists, moderate paint, high usage."""
    f = _minimal_features("C")
    f.update(
        ast_per36=9.0,
        usg_pct_proxy=0.28,
        zone_fga_rate_paint=0.20,
        zone_fga_rate_ra=0.20,
        fg3a_rate=0.15,
    )
    return f


def _ad_features() -> dict:
    """Traditional post scorer (AD/Embiid type): high paint+RA, moderate assists, high usage."""
    f = _minimal_features("C")
    f.update(
        zone_fga_rate_paint=0.30,
        zone_fga_rate_ra=0.25,
        ast_per36=3.5,
        usg_pct_proxy=0.28,
        fg3a_rate=0.08,
    )
    return f


def _guard_features() -> dict:
    """Perimeter guard: low paint/RA rates."""
    f = _minimal_features("PG")
    f.update(
        zone_fga_rate_paint=0.05,
        zone_fga_rate_ra=0.15,
        ast_per36=7.0,
        usg_pct_proxy=0.22,
    )
    return f


class TestRimRunnerPostCalibration:
    """Calibration tests ensuring rim-runner post tendencies are low-moderate, not near-zero,
    while finesse moves are appropriately suppressed for low-assist players."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    # --- Rim-runner (Gobert/Capela type) ---

    def test_rim_runner_post_up_in_low_moderate_range(self, formula):
        """Rim-running center post_up should be in 10–20 range (not near-zero, not too high)."""
        raw = formula.generate(_rim_runner_features())
        assert 10 <= raw["post_up"] <= 20, (
            f"rim-runner post_up={raw['post_up']:.2f} not in [10, 20]"
        )

    def test_rim_runner_post_drop_step_present(self, formula):
        """Rim-runner should have post_drop_step > 5 (they use this move)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_drop_step"] > 5, (
            f"rim-runner post_drop_step={raw['post_drop_step']:.2f} not > 5"
        )

    def test_rim_runner_post_hook_present(self, formula):
        """Rim-runner should get some hook shot credit (left or right > 3)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_hook_left"] > 3 or raw["post_hook_right"] > 3, (
            f"rim-runner hooks: left={raw['post_hook_left']:.2f}, right={raw['post_hook_right']:.2f}"
        )

    def test_rim_runner_post_fade_suppressed(self, formula):
        """Rim-runner post_fade_left and post_fade_right should be < 5 (finesse gate)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_fade_left"] < 5, (
            f"rim-runner post_fade_left={raw['post_fade_left']:.2f} not < 5"
        )
        assert raw["post_fade_right"] < 5, (
            f"rim-runner post_fade_right={raw['post_fade_right']:.2f} not < 5"
        )

    def test_rim_runner_post_spin_suppressed(self, formula):
        """Rim-runner post_spin should be < 5 (finesse gate)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_spin"] < 5, (
            f"rim-runner post_spin={raw['post_spin']:.2f} not < 5"
        )

    def test_rim_runner_post_face_up_suppressed(self, formula):
        """Rim-runner post_face_up should be < 10 (not a face-up player)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_face_up"] < 10, (
            f"rim-runner post_face_up={raw['post_face_up']:.2f} not < 10"
        )

    def test_rim_runner_post_up_not_zero(self, formula):
        """Rim-runner post_up must be > 0 (not near-zero, unlike prior formula)."""
        raw = formula.generate(_rim_runner_features())
        assert raw["post_up"] > 0, "rim-runner post_up should be non-zero"

    # --- Elite post playmaker (Jokic type) ---

    def test_jokic_type_has_highest_post_face_up(self, formula):
        """Jokic-type (high ast) should have higher post_face_up than rim-runner."""
        jokic = formula.generate(_jokic_features())
        gobert = formula.generate(_rim_runner_features())
        assert jokic["post_face_up"] > gobert["post_face_up"], (
            f"jokic post_face_up={jokic['post_face_up']:.2f} not > "
            f"gobert post_face_up={gobert['post_face_up']:.2f}"
        )

    def test_jokic_type_high_post_drive(self, formula):
        """Jokic-type should have non-trivial post_drive (playmaking center)."""
        raw = formula.generate(_jokic_features())
        assert raw["post_drive"] > 10, (
            f"jokic post_drive={raw['post_drive']:.2f} not > 10"
        )

    # --- Traditional post scorer (AD/Embiid type) ---

    def test_ad_type_high_post_up(self, formula):
        """AD/Embiid-type should have high post_up (dominant post scorer)."""
        raw = formula.generate(_ad_features())
        assert raw["post_up"] > 20, (
            f"AD-type post_up={raw['post_up']:.2f} not > 20"
        )

    def test_ad_type_high_post_back_down(self, formula):
        """AD/Embiid-type should have high post_back_down."""
        raw = formula.generate(_ad_features())
        assert raw["post_back_down"] > 10, (
            f"AD-type post_back_down={raw['post_back_down']:.2f} not > 10"
        )

    def test_ad_type_high_post_hooks(self, formula):
        """AD/Embiid-type should have meaningful hook shot values."""
        raw = formula.generate(_ad_features())
        assert raw["post_hook_left"] > 3 and raw["post_hook_right"] > 3, (
            f"AD-type hooks: left={raw['post_hook_left']:.2f}, right={raw['post_hook_right']:.2f}"
        )

    # --- Perimeter players (guards/wings) ---

    def test_guard_near_zero_post_tendencies(self, formula):
        """Perimeter guard should have near-zero post tendencies."""
        raw = formula.generate(_guard_features())
        for key in ("post_up", "post_back_down", "post_hook_left", "post_hook_right"):
            assert raw[key] < 5, (
                f"guard {key}={raw[key]:.2f} should be near-zero"
            )

    # --- Comparison tests ---

    def test_ad_back_down_greater_than_gobert_back_down(self, formula):
        """AD-type post_back_down > Gobert-type post_back_down."""
        ad = formula.generate(_ad_features())
        gobert = formula.generate(_rim_runner_features())
        assert ad["post_back_down"] > gobert["post_back_down"]

    def test_gobert_drop_step_greater_than_guard_drop_step(self, formula):
        """Gobert-type post_drop_step > guard-type post_drop_step."""
        gobert = formula.generate(_rim_runner_features())
        guard = formula.generate(_guard_features())
        assert gobert["post_drop_step"] > guard["post_drop_step"]

    def test_all_values_non_negative_rim_runner(self, formula):
        """All tendency values must be non-negative for rim-runner archetype."""
        raw = formula.generate(_rim_runner_features())
        for k, v in raw.items():
            assert v >= 0.0, f"rim-runner {k}={v} is negative"


class TestRaBlendForShotClose:
    """Tests for RA-blend fix: higher RA rate increases shot_close."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    def test_higher_ra_rate_increases_shot_close(self, formula):
        """Higher zone_fga_rate_ra with same paint rate should increase shot_close."""
        f_low_ra = _minimal_features("C")
        f_low_ra["zone_fga_rate_paint"] = 0.10
        f_low_ra["zone_fga_rate_ra"] = 0.05

        f_high_ra = _minimal_features("C")
        f_high_ra["zone_fga_rate_paint"] = 0.10
        f_high_ra["zone_fga_rate_ra"] = 0.50  # rim-runner level

        r_low = formula.generate(f_low_ra)
        r_high = formula.generate(f_high_ra)
        assert r_high["shot_close"] > r_low["shot_close"], (
            f"High RA player should have higher shot_close: "
            f"high={r_high['shot_close']:.1f} low={r_low['shot_close']:.1f}"
        )

    def test_rim_runner_center_has_reasonable_shot_close(self, formula):
        """Rim-runner center (high RA, low paint) should not have pathologically low shot_close.

        With the RA blend, a center with high RA (0.45) and low paint (0.05) should
        have a significantly higher shot_close than what paint alone would give (~10).
        """
        f = _minimal_features("C")
        f["zone_fga_rate_ra"] = 0.45    # Gobert-like RA rate
        f["zone_fga_rate_paint"] = 0.05  # low paint (goes directly to RA)
        result = formula.generate(f)
        # Old formula (paint only): scale(0.05, [0,0.3], [0,60]) ≈ 10
        # New blend: scale(0.85*0.05 + 0.15*0.45, [0,0.3], [0,60]) ≈ 22 — much better
        assert result["shot_close"] >= 20, (
            f"Rim-runner shot_close too low (RA blend not working): {result['shot_close']:.1f}"
        )

    def test_high_paint_player_stays_stable(self, formula):
        """Player with high paint rate should still have a high shot_close."""
        f = _minimal_features("PF")
        f["zone_fga_rate_paint"] = 0.30
        f["zone_fga_rate_ra"] = 0.20
        result = formula.generate(f)
        assert result["shot_close"] >= 35, (
            f"High-paint PF shot_close too low: {result['shot_close']:.1f}"
        )


class TestCloseDistributionShaping:
    """Tests for close sub-zone distribution shaping (prevents middle dominance)."""

    @pytest.fixture(scope="class")
    def formula(self):
        return FormulaLayer()

    def test_middle_cannot_exceed_50_pct_of_parent(self, formula):
        """With middle-heavy raw distribution, shaping should reduce middle dominance.

        The 60/40 blend pulls extreme values toward uniform. With raw middle=90%,
        the shaped middle becomes ~67% (well below the raw 90%), preventing extreme dominance.
        """
        f = _minimal_features("C")
        # Extreme middle dominance (90% middle)
        f["sub_zone_distribution_close"] = {"left": 5.0, "middle": 90.0, "right": 5.0}
        result = formula.generate(f)
        parent = result["shot_close"]
        if parent > 0:
            # Shaped middle should be significantly less than the raw 90%
            # With 60/40 blend: shaped_middle ≈ 0.6*90 + 0.4*33.3 = 67.3%
            assert result["shot_close_middle"] / parent <= 0.75, (
                f"shot_close_middle ({result['shot_close_middle']:.1f}) is pathologically dominant "
                f"({result['shot_close_middle'] / parent * 100:.0f}% of shot_close {parent:.1f})"
            )

    def test_left_bias_preserved_after_shaping(self, formula):
        """Left-dominant distribution should keep shot_close_left > shot_close_right."""
        f = _minimal_features("C")
        f["sub_zone_distribution_close"] = {"left": 60.0, "middle": 30.0, "right": 10.0}
        result = formula.generate(f)
        assert result["shot_close_left"] > result["shot_close_right"], (
            f"Left bias not preserved: left={result['shot_close_left']:.1f}, "
            f"right={result['shot_close_right']:.1f}"
        )

    def test_right_bias_preserved_after_shaping(self, formula):
        """Right-dominant distribution should keep shot_close_right > shot_close_left."""
        f = _minimal_features("C")
        f["sub_zone_distribution_close"] = {"left": 10.0, "middle": 30.0, "right": 60.0}
        result = formula.generate(f)
        assert result["shot_close_right"] > result["shot_close_left"], (
            f"Right bias not preserved: left={result['shot_close_left']:.1f}, "
            f"right={result['shot_close_right']:.1f}"
        )

    def test_close_sub_zones_sum_to_parent_after_shaping(self, formula):
        """Shaped close sub-zones must still sum to shot_close."""
        for dist in [
            {"left": 5.0, "middle": 90.0, "right": 5.0},
            {"left": 60.0, "middle": 20.0, "right": 20.0},
            {"left": 33.3, "middle": 33.4, "right": 33.3},
        ]:
            f = _minimal_features("C")
            f["sub_zone_distribution_close"] = dist
            result = formula.generate(f)
            total = (
                result["shot_close_left"]
                + result["shot_close_middle"]
                + result["shot_close_right"]
            )
            assert total == pytest.approx(result["shot_close"], abs=0.1), (
                f"Sub-zone sum {total:.2f} != shot_close {result['shot_close']:.2f} for dist {dist}"
            )
