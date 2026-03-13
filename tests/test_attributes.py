"""Tests for the AttributeCalculator."""
from __future__ import annotations

import pytest

from src.attributes.calculator import (
    ATTRIBUTE_CATEGORIES,
    ATTRIBUTE_LABELS,
    ATTRIBUTE_NAMES,
    AttributeCalculator,
)


@pytest.fixture()
def calc() -> AttributeCalculator:
    return AttributeCalculator()


def _star_guard_features() -> dict:
    """Feature dict resembling a star PG (e.g. Curry-like)."""
    return {
        "position": "PG",
        "height_inches": 74,
        "weight_lbs": 185,
        "age": 36,
        "gp": 74,
        "min_per_game": 32,
        "pts_per_game": 26.4,
        "fga_per_game": 19.5,
        "fg3a_per_game": 11.3,
        "fta_per_game": 4.4,
        "ast_per_game": 6.1,
        "oreb_per36": 0.6,
        "dreb_per36": 4.6,
        "stl_per_game": 1.0,
        "blk_per_game": 0.4,
        "tov_per_game": 2.8,
        "fg_pct": 0.450,
        "fg3_pct": 0.408,
        "ft_pct": 0.923,
        "efg_pct": 0.560,
        "ts_pct": 0.610,
        "fg3a_rate": 0.580,
        "fta_rate": 0.225,
        "usage_rate": 30.0,
        "ast_to_tov": 2.18,
        "tov_pct_proxy": 0.12,
        "plus_minus": 7.0,
        "zone_fga_rate_ra": 0.18,
        "zone_fga_rate_paint": 0.08,
        "zone_fg_pct_ra": 0.60,
        "zone_fg_pct_paint": 0.42,
        "zone_fg_pct_mid_left": 0.44,
        "zone_fg_pct_mid_center": 0.48,
        "zone_fg_pct_mid_right": 0.42,
        "zone_fga_rate_mid_left": 0.06,
        "zone_fga_rate_mid_center": 0.04,
        "zone_fga_rate_mid_right": 0.05,
        "isolation_possessions": 2.0,
        "pick_and_roll_ball_handler_possessions": 5.0,
        "pick_and_roll_rollman_possessions": 0.0,
        "post_up_possessions": 0.1,
        "cuts": 0.8,
        "transition_possessions": 3.5,
        "post_up_ppp": 0.0,
        "spot_up_ppp": 1.15,
        "pctile_pts": 0.92,
        "pctile_ast": 0.80,
        "pctile_reb": 0.30,
        "pctile_stl": 0.60,
        "pctile_blk": 0.20,
    }


def _star_center_features() -> dict:
    """Feature dict for a rim-protecting center (e.g. Gobert-like)."""
    return {
        "position": "C",
        "height_inches": 85,
        "weight_lbs": 258,
        "age": 32,
        "gp": 68,
        "min_per_game": 30,
        "pts_per_game": 14.0,
        "fga_per_game": 9.0,
        "fg3a_per_game": 0.0,
        "fta_per_game": 3.5,
        "ast_per_game": 1.3,
        "oreb_per36": 3.5,
        "dreb_per36": 9.0,
        "stl_per_game": 0.6,
        "blk_per_game": 2.1,
        "tov_per_game": 1.5,
        "fg_pct": 0.660,
        "fg3_pct": 0.0,
        "ft_pct": 0.640,
        "efg_pct": 0.660,
        "ts_pct": 0.680,
        "fg3a_rate": 0.0,
        "fta_rate": 0.39,
        "usage_rate": 16.0,
        "ast_to_tov": 0.87,
        "tov_pct_proxy": 0.10,
        "plus_minus": 5.0,
        "zone_fga_rate_ra": 0.65,
        "zone_fga_rate_paint": 0.20,
        "zone_fg_pct_ra": 0.72,
        "zone_fg_pct_paint": 0.52,
        "zone_fg_pct_mid_left": 0.0,
        "zone_fg_pct_mid_center": 0.0,
        "zone_fg_pct_mid_right": 0.0,
        "zone_fga_rate_mid_left": 0.0,
        "zone_fga_rate_mid_center": 0.0,
        "zone_fga_rate_mid_right": 0.0,
        "isolation_possessions": 0.0,
        "pick_and_roll_ball_handler_possessions": 0.0,
        "pick_and_roll_rollman_possessions": 4.5,
        "post_up_possessions": 3.0,
        "cuts": 2.0,
        "transition_possessions": 1.0,
        "post_up_ppp": 0.92,
        "spot_up_ppp": 0.0,
        "pctile_pts": 0.40,
        "pctile_ast": 0.10,
        "pctile_reb": 0.95,
        "pctile_stl": 0.30,
        "pctile_blk": 0.95,
    }


class TestAttributeScalesAndKeys:
    def test_all_37_attributes_returned(self, calc: AttributeCalculator):
        result = calc.calculate({}, {})
        assert len(result) == 37
        assert set(result.keys()) == set(ATTRIBUTE_NAMES)

    def test_values_within_25_99(self, calc: AttributeCalculator):
        feats = _star_guard_features()
        result = calc.calculate(feats, {})
        for key, val in result.items():
            assert 25 <= val <= 99, f"{key}={val} out of range"

    def test_labels_complete(self):
        assert set(ATTRIBUTE_LABELS.keys()) == set(ATTRIBUTE_NAMES)

    def test_categories_complete(self):
        assert set(ATTRIBUTE_CATEGORIES.keys()) == set(ATTRIBUTE_NAMES)


class TestGuardProfile:
    def test_three_point_high(self, calc: AttributeCalculator):
        feats = _star_guard_features()
        result = calc.calculate(feats, {"shot_three": 80})
        assert result["three_point_shot"] >= 75

    def test_speed_higher_than_center(self, calc: AttributeCalculator):
        guard = calc.calculate(_star_guard_features(), {})
        center = calc.calculate(_star_center_features(), {})
        assert guard["speed"] > center["speed"]

    def test_ball_handle_higher_than_center(self, calc: AttributeCalculator):
        guard = calc.calculate(_star_guard_features(), {})
        center = calc.calculate(_star_center_features(), {})
        assert guard["ball_handle"] > center["ball_handle"]

    def test_post_attributes_low(self, calc: AttributeCalculator):
        feats = _star_guard_features()
        result = calc.calculate(feats, {})
        assert result["post_hook"] <= 45
        assert result["post_control"] <= 45


class TestCenterProfile:
    def test_block_high(self, calc: AttributeCalculator):
        feats = _star_center_features()
        result = calc.calculate(feats, {})
        assert result["block"] >= 75

    def test_interior_defense_high(self, calc: AttributeCalculator):
        result = calc.calculate(_star_center_features(), {})
        assert result["interior_defense"] >= 70

    def test_three_point_low(self, calc: AttributeCalculator):
        result = calc.calculate(_star_center_features(), {})
        assert result["three_point_shot"] <= 40

    def test_strength_high(self, calc: AttributeCalculator):
        result = calc.calculate(_star_center_features(), {})
        assert result["strength"] >= 75

    def test_standing_dunk_high(self, calc: AttributeCalculator):
        feats = _star_center_features()
        result = calc.calculate(feats, {"standing_dunk": 70, "alley_oop": 50})
        assert result["standing_dunk"] >= 65


class TestPotentialByAge:
    def test_young_player_high_potential(self, calc: AttributeCalculator):
        feats = {"age": 20}
        result = calc.calculate(feats, {})
        assert result["potential"] >= 85

    def test_old_player_low_potential(self, calc: AttributeCalculator):
        feats = {"age": 38}
        result = calc.calculate(feats, {})
        assert result["potential"] <= 40


class TestTendencyInfluence:
    def test_driving_dunk_responds_to_tendency(self, calc: AttributeCalculator):
        feats = _star_guard_features()
        low = calc.calculate(feats, {"driving_dunk": 10, "alley_oop": 10})
        high = calc.calculate(feats, {"driving_dunk": 75, "alley_oop": 60})
        assert high["driving_dunk"] > low["driving_dunk"]

    def test_post_hook_responds_to_tendency(self, calc: AttributeCalculator):
        feats = _star_center_features()
        low = calc.calculate(feats, {"post_hook_left": 5, "post_hook_right": 5, "post_up": 10})
        high = calc.calculate(feats, {"post_hook_left": 50, "post_hook_right": 50, "post_up": 60})
        assert high["post_hook"] > low["post_hook"]
