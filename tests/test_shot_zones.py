"""Tests for src/features/shot_zones.py."""
from __future__ import annotations

import pytest

from src.features.shot_zones import (
    ShotZoneAnalyzer,
    ShotZoneBuilder,
    _bayesian_smooth,
    _classify_zone,
    ZONES,
)


def _make_shot(basic: str, area: str, made: int = 1, loc_x: int = 0) -> dict:
    return {
        "shot_zone_basic": basic,
        "shot_zone_area": area,
        "shot_made_flag": made,
        "loc_x": loc_x,
        "loc_y": 0,
        "shot_type": "2PT Field Goal",
        "action_type": "Jump Shot",
    }


class TestClassifyZone:
    def test_restricted_area(self):
        assert _classify_zone("Restricted Area", "Center(C)") == "ra"

    def test_paint_non_ra(self):
        assert _classify_zone("In The Paint (Non-RA)", "Center(C)") == "paint"

    def test_mid_range_left(self):
        assert _classify_zone("Mid-Range", "Left Side(L)") == "mid_left"

    def test_mid_range_left_center(self):
        assert _classify_zone("Mid-Range", "Left Side Center(LC)") == "mid_left"

    def test_mid_range_center(self):
        assert _classify_zone("Mid-Range", "Center(C)") == "mid_center"

    def test_mid_range_right(self):
        assert _classify_zone("Mid-Range", "Right Side(R)") == "mid_right"

    def test_mid_range_right_center(self):
        assert _classify_zone("Mid-Range", "Right Side Center(RC)") == "mid_right"

    def test_left_corner_3(self):
        assert _classify_zone("Left Corner 3", "Left Side(L)") == "corner3_left"

    def test_right_corner_3(self):
        assert _classify_zone("Right Corner 3", "Right Side(R)") == "corner3_right"

    def test_above_break_3(self):
        assert _classify_zone("Above the Break 3", "Center(C)") == "above_break3"

    def test_unknown_returns_none(self):
        assert _classify_zone("Backcourt", "Center(C)") is None


class TestBayesianSmooth:
    def test_no_attempts_returns_prior(self):
        result = _bayesian_smooth(0, 0, 0.5, prior_strength=10)
        assert result == pytest.approx(0.5)

    def test_many_attempts_dominates(self):
        # 100 makes out of 100 → should be close to 1.0
        result = _bayesian_smooth(100, 100, 0.5, prior_strength=10)
        assert result > 0.9

    def test_low_attempts_shrinks_to_prior(self):
        # 1 make out of 1 attempt — prior 0.4 → should shrink toward 0.4
        result = _bayesian_smooth(1, 1, 0.4, prior_strength=10)
        assert 0.4 <= result <= 1.0


class TestShotZoneAnalyzer:
    def test_empty_shot_chart_returns_defaults(self):
        analyzer = ShotZoneAnalyzer()
        result = analyzer.analyze([], total_minutes=100.0)
        for zone in ZONES:
            assert zone in result["zone_fga"]
            assert result["zone_fga"][zone] == 0

    def test_all_zones_present(self):
        analyzer = ShotZoneAnalyzer()
        result = analyzer.analyze([], total_minutes=100.0)
        assert set(result["zone_fga"].keys()) == set(ZONES)

    def test_ra_shots_counted(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "Center(C)", made=1)] * 5
        result = analyzer.analyze(shots, total_minutes=200.0)
        assert result["zone_fga"]["ra"] == 5
        assert result["zone_fgm"]["ra"] == 5

    def test_fga_rate_sums_to_one(self):
        analyzer = ShotZoneAnalyzer()
        shots = [
            _make_shot("Restricted Area", "Center(C)"),
            _make_shot("Mid-Range", "Center(C)"),
            _make_shot("Above the Break 3", "Center(C)"),
        ]
        result = analyzer.analyze(shots, total_minutes=100.0)
        total = sum(result["zone_fga_rate"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_sub_zone_close_sums_to_100(self):
        analyzer = ShotZoneAnalyzer()
        shots = [
            _make_shot("Restricted Area", "Center(C)", loc_x=-50),
            _make_shot("Restricted Area", "Center(C)", loc_x=0),
            _make_shot("Restricted Area", "Center(C)", loc_x=50),
        ]
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        total = sum(dist.values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_sub_zone_mid_sums_to_100(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Mid-Range", "Left Side(L)")] * 2
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_mid"]
        total = sum(dist.values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_sub_zone_three_sums_to_100(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Above the Break 3", "Center(C)")] * 3
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_three"]
        total = sum(dist.values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_empty_sub_zone_even_distribution(self):
        analyzer = ShotZoneAnalyzer()
        result = analyzer.analyze([], total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        # Should be even split ~33.3
        for v in dist.values():
            assert v == pytest.approx(100 / 3, abs=1.0)

    def test_per36_increases_with_more_shots(self):
        analyzer = ShotZoneAnalyzer()
        shots10 = [_make_shot("Restricted Area", "Center(C)")] * 10
        shots20 = [_make_shot("Restricted Area", "Center(C)")] * 20
        r10 = analyzer.analyze(shots10, total_minutes=200.0)
        r20 = analyzer.analyze(shots20, total_minutes=200.0)
        assert r20["zone_fga_per36"]["ra"] > r10["zone_fga_per36"]["ra"]


class TestShotZoneBuilder:
    def test_compute_zones_returns_dict(self):
        builder = ShotZoneBuilder()
        shots = [_make_shot("Restricted Area", "Center(C)")]
        result = builder.compute_zones(shots, total_minutes=100.0)
        assert isinstance(result, dict)
        assert "zone_fga_rate_ra" in result

    def test_distribute_from_parent_sums_to_parent(self):
        builder = ShotZoneBuilder()
        fracs = {"left": 0.4, "center": 0.4, "right": 0.2}
        dist = builder.distribute_from_parent(60, fracs)
        assert sum(dist.values()) == pytest.approx(60, abs=2)

    def test_distribute_from_parent_empty_is_even(self):
        builder = ShotZoneBuilder()
        fracs = {"left": 0.0, "center": 0.0, "right": 0.0}
        dist = builder.distribute_from_parent(60, fracs)
        assert all(v == 20 for v in dist.values())
