"""Tests for src/features/shot_zones.py."""
from __future__ import annotations

import pytest

from src.features.shot_zones import (
    ShotZoneAnalyzer,
    ShotZoneBuilder,
    _area_to_close_key,
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

    def test_close_shot_loc_x_minus100_is_left(self):
        # Empty area triggers LOC_X fallback; loc_x=-100 < -40 → left
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "", loc_x=-100)] * 4
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        assert dist["left"] == pytest.approx(100.0, abs=0.5)
        assert dist["middle"] == pytest.approx(0.0, abs=0.5)
        assert dist["right"] == pytest.approx(0.0, abs=0.5)

    def test_close_shot_loc_x_plus100_is_right(self):
        # Empty area triggers LOC_X fallback; loc_x=100 > 40 → right
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("In The Paint (Non-RA)", "", loc_x=100)] * 4
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        assert dist["right"] == pytest.approx(100.0, abs=0.5)
        assert dist["middle"] == pytest.approx(0.0, abs=0.5)
        assert dist["left"] == pytest.approx(0.0, abs=0.5)

    def test_close_shot_loc_x_zero_is_middle(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "Center(C)", loc_x=0)] * 4
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        assert dist["middle"] == pytest.approx(100.0, abs=0.5)

    def test_close_shot_loc_x_minus50_is_left(self):
        # RA shots use LOC_X; loc_x=-50 < -30 → left
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "Center(C)", loc_x=-50)] * 4
        result = analyzer.analyze(shots, total_minutes=100.0)
        dist = result["sub_zone_distribution_close"]
        assert dist["left"] == pytest.approx(100.0, abs=0.5)

    def test_close_shot_realistic_mix_no_bucket_exceeds_50(self):
        # Realistic RA data: area is always "Center(C)", LOC_X drives classification
        analyzer = ShotZoneAnalyzer()
        shots = (
            [_make_shot("Restricted Area", "Center(C)", loc_x=-60)] * 25
            + [_make_shot("Restricted Area", "Center(C)", loc_x=0)] * 45
            + [_make_shot("Restricted Area", "Center(C)", loc_x=60)] * 30
        )
        result = analyzer.analyze(shots, total_minutes=200.0)
        dist = result["sub_zone_distribution_close"]
        for bucket, value in dist.items():
            assert value <= 50.0, f"Bucket '{bucket}' exceeded 50: {value}"

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


class TestAreaToCloseKey:
    def test_left_side_returns_left(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "Left Side(L)", 0) == "left"

    def test_left_side_center_returns_left(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "Left Side Center(LC)", 0) == "left"

    def test_right_side_returns_right(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "Right Side(R)", 0) == "right"

    def test_right_side_center_returns_right(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "Right Side Center(RC)", 0) == "right"

    def test_center_returns_middle(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "Center(C)", 0) == "middle"

    def test_empty_area_fallback_left(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "", -50) == "left"

    def test_empty_area_fallback_right(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "", 50) == "right"

    def test_empty_area_fallback_middle(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "", 0) == "middle"

    def test_empty_area_fallback_boundary_left(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "", -41) == "left"

    def test_empty_area_fallback_boundary_right(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "", 41) == "right"

    def test_empty_area_fallback_at_minus30(self):
        # -30 is NOT < -30, so it should be middle
        assert _area_to_close_key("In The Paint (Non-RA)", "", -30) == "middle"

    def test_empty_area_fallback_at_plus30(self):
        # 30 is NOT > 30, so it should be middle
        assert _area_to_close_key("In The Paint (Non-RA)", "", 30) == "middle"

    def test_area_overrides_loc_x(self):
        # area="Left Side(L)" should win even with extreme positive loc_x (Paint shot)
        assert _area_to_close_key("In The Paint (Non-RA)", "Left Side(L)", 200) == "left"
        # area="Right Side(R)" should win even with extreme negative loc_x (Paint shot)
        assert _area_to_close_key("In The Paint (Non-RA)", "Right Side(R)", -200) == "right"
        # area="Center(C)" should win even with extreme loc_x (Paint shot)
        assert _area_to_close_key("In The Paint (Non-RA)", "Center(C)", -100) == "middle"

    def test_ra_ignores_area_uses_loc_x(self):
        # RA shots always use LOC_X regardless of area value
        assert _area_to_close_key("Restricted Area", "Left Side(L)", 0) == "middle"
        assert _area_to_close_key("Restricted Area", "Center(C)", -50) == "left"
        assert _area_to_close_key("Restricted Area", "Center(C)", 50) == "right"

    def test_ra_at_minus30_is_middle(self):
        # -30 is NOT < -30, so RA shot at exactly -30 should be middle
        assert _area_to_close_key("Restricted Area", "Center(C)", -30) == "middle"

    def test_ra_at_plus30_is_middle(self):
        # 30 is NOT > 30, so RA shot at exactly +30 should be middle
        assert _area_to_close_key("Restricted Area", "Center(C)", 30) == "middle"

    def test_whitespace_is_stripped(self):
        assert _area_to_close_key("In The Paint (Non-RA)", "  Left Side(L)  ", 0) == "left"


class TestCloseSubZoneAreaClassification:
    """Integration tests: area-based close-shot classification via ShotZoneAnalyzer."""

    def test_left_area_in_ra_classified_as_left(self):
        # RA shots use LOC_X; loc_x=-50 < -30 → left
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "Center(C)", loc_x=-50)] * 4
        dist = analyzer.analyze(shots, total_minutes=100.0)["sub_zone_distribution_close"]
        assert dist["left"] == pytest.approx(100.0, abs=0.5)

    def test_right_area_in_paint_classified_as_right(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("In The Paint (Non-RA)", "Right Side(R)", loc_x=0)] * 4
        dist = analyzer.analyze(shots, total_minutes=100.0)["sub_zone_distribution_close"]
        assert dist["right"] == pytest.approx(100.0, abs=0.5)

    def test_center_area_classified_as_middle(self):
        analyzer = ShotZoneAnalyzer()
        shots = [_make_shot("Restricted Area", "Center(C)", loc_x=0)] * 4
        dist = analyzer.analyze(shots, total_minutes=100.0)["sub_zone_distribution_close"]
        assert dist["middle"] == pytest.approx(100.0, abs=0.5)

    def test_empty_area_fallback_to_loc_x(self):
        analyzer = ShotZoneAnalyzer()
        shots_left = [_make_shot("Restricted Area", "", loc_x=-50)] * 3
        shots_right = [_make_shot("Restricted Area", "", loc_x=50)] * 3
        dist = analyzer.analyze(shots_left + shots_right, total_minutes=100.0)[
            "sub_zone_distribution_close"
        ]
        assert dist["left"] == pytest.approx(50.0, abs=0.5)
        assert dist["right"] == pytest.approx(50.0, abs=0.5)
        assert dist["middle"] == pytest.approx(0.0, abs=0.5)

    def test_realistic_mix_left_plus_right_greater_than_zero(self):
        # Realistic RA data: area always "Center(C)", LOC_X drives classification
        analyzer = ShotZoneAnalyzer()
        shots = (
            [_make_shot("Restricted Area", "Center(C)", loc_x=-60)] * 20
            + [_make_shot("Restricted Area", "Center(C)", loc_x=0)] * 50
            + [_make_shot("Restricted Area", "Center(C)", loc_x=60)] * 30
        )
        dist = analyzer.analyze(shots, total_minutes=200.0)["sub_zone_distribution_close"]
        assert dist["left"] == pytest.approx(20.0, abs=0.5)
        assert dist["middle"] == pytest.approx(50.0, abs=0.5)
        assert dist["right"] == pytest.approx(30.0, abs=0.5)
        assert dist["left"] + dist["right"] > 0

    def test_ra_loc_x_distribution_middle_le_50(self):
        # Typical RA distribution by LOC_X should produce middle ≤ 50
        analyzer = ShotZoneAnalyzer()
        shots = (
            [_make_shot("Restricted Area", "Center(C)", loc_x=-40)] * 30
            + [_make_shot("Restricted Area", "Center(C)", loc_x=5)] * 40
            + [_make_shot("Restricted Area", "Center(C)", loc_x=45)] * 30
        )
        dist = analyzer.analyze(shots, total_minutes=200.0)["sub_zone_distribution_close"]
        assert dist["middle"] <= 50.0

    def test_close_sub_zones_sum_to_100(self):
        analyzer = ShotZoneAnalyzer()
        shots = (
            [_make_shot("Restricted Area", "Left Side(L)")] * 10
            + [_make_shot("In The Paint (Non-RA)", "Center(C)")] * 20
            + [_make_shot("Restricted Area", "Right Side(R)")] * 10
        )
        dist = analyzer.analyze(shots, total_minutes=100.0)["sub_zone_distribution_close"]
        assert sum(dist.values()) == pytest.approx(100.0, abs=0.5)


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
