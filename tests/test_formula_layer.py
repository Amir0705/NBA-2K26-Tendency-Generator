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

    def test_roll_vs_pop_guard_is_50(self, formula):
        result = formula.generate(_minimal_features("PG"))
        assert result["roll_vs_pop"] == pytest.approx(50.0)

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
