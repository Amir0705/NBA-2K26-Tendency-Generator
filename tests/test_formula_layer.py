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

    def test_sub_zone_close_sums_to_100(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_close_left"]
            + result["shot_close_middle"]
            + result["shot_close_right"]
        )
        assert total == pytest.approx(100.0, abs=1.0)

    def test_sub_zone_mid_sums_to_100(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_mid_left"]
            + result["shot_mid_left_center"]
            + result["shot_mid_center"]
            + result["shot_mid_right_center"]
            + result["shot_mid_right"]
        )
        assert total == pytest.approx(100.0, abs=1.0)

    def test_sub_zone_three_sums_to_100(self, formula):
        result = formula.generate(_minimal_features())
        total = (
            result["shot_three_left"]
            + result["shot_three_left_center"]
            + result["shot_three_center"]
            + result["shot_three_right_center"]
            + result["shot_three_right"]
        )
        assert total == pytest.approx(100.0, abs=1.0)

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
    def test_triple_threat_idle_high_usage_lower_than_low_usage(self, formula):
        """High-usage players should have lower triple_threat_idle."""
        high_usg = dict(_minimal_features("PG"), usg_pct_proxy=0.33)
        low_usg = dict(_minimal_features("PG"), usg_pct_proxy=0.12)
        r_high = formula.generate(high_usg)
        r_low = formula.generate(low_usg)
        assert r_high["triple_threat_idle"] < r_low["triple_threat_idle"]

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
