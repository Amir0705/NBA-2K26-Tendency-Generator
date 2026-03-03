"""Tests for src/features/feature_engine.py."""
from __future__ import annotations

import pytest

from src.features.feature_engine import FeatureEngine, _height_to_inches, _map_position, _per36


# ---------------------------------------------------------------------------
# Helper: mock NBA API client
# ---------------------------------------------------------------------------


class MockNBAClient:
    """Minimal mock for NBAApiClient."""

    def get_player_info(self, player_id: int) -> dict:
        return {
            "position": "Guard",
            "height": "6-3",
            "weight": "195",
            "team_id": 1610612749,
            "team_abbreviation": "MIL",
        }

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return {
            "gp": 70,
            "min": 34.0,
            "pts": 22.0,
            "fga": 17.0,
            "fgm": 8.0,
            "fg_pct": 0.471,
            "fg3a": 6.0,
            "fg3m": 2.5,
            "fg3_pct": 0.417,
            "fta": 4.5,
            "ftm": 3.8,
            "ft_pct": 0.844,
            "oreb": 0.8,
            "dreb": 4.5,
            "reb": 5.3,
            "ast": 7.1,
            "stl": 1.5,
            "blk": 0.4,
            "tov": 3.2,
            "pf": 2.1,
            "plus_minus": 6.5,
        }

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return [
            {
                "shot_zone_basic": "Restricted Area",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "Less Than 8 ft.",
                "shot_made_flag": 1,
                "loc_x": 0,
                "loc_y": 5,
                "shot_type": "2PT Field Goal",
                "action_type": "Driving Layup Shot",
            },
            {
                "shot_zone_basic": "Above the Break 3",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "24+ ft.",
                "shot_made_flag": 0,
                "loc_x": 10,
                "loc_y": 240,
                "shot_type": "3PT Field Goal",
                "action_type": "Jump Shot",
            },
        ]

    def get_league_averages(self, season: str = "2024-25") -> list:
        # Return minimal rows for percentile calculations
        return [
            {"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
             "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0},
            {"PTS": 20.0, "AST": 6.0, "REB": 7.0, "STL": 1.5, "BLK": 1.0,
             "FG3A": 5.0, "FGA": 15.0, "FTA": 4.0, "TOV": 3.0},
        ]


class TestHelpers:
    def test_height_normal(self):
        assert _height_to_inches("6-3") == 75

    def test_height_seven_footer(self):
        assert _height_to_inches("7-0") == 84

    def test_height_invalid_returns_fallback(self):
        result = _height_to_inches("")
        assert result == 78

    def test_map_position_guard(self):
        assert _map_position("Guard") == "PG"

    def test_map_position_center(self):
        assert _map_position("Center") == "C"

    def test_map_position_forward(self):
        assert _map_position("Forward") == "SF"

    def test_map_position_guard_forward(self):
        assert _map_position("Guard-Forward") == "SG"

    def test_per36_normal(self):
        # 18 pts/game in 36 min/game → 18 per36
        result = _per36(18.0, 36.0, 1)
        assert result == pytest.approx(18.0)

    def test_per36_avoids_division_by_zero(self):
        result = _per36(10.0, 0.0, 0)
        assert result >= 0.0


class TestFeatureEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        return FeatureEngine(MockNBAClient())

    @pytest.fixture(scope="class")
    def features(self, engine):
        return engine.build_features(2544)  # LeBron's player_id (mocked)

    def test_returns_dict(self, features):
        assert isinstance(features, dict)

    def test_position_present(self, features):
        assert "position" in features
        assert features["position"] in ("PG", "SG", "SF", "PF", "C")

    def test_per36_present(self, features):
        for key in ("pts_per36", "ast_per36", "reb_per36", "stl_per36", "blk_per36"):
            assert key in features, f"Missing {key}"

    def test_zone_features_present(self, features):
        from src.features.shot_zones import ZONES
        for zone in ZONES:
            assert f"zone_fga_rate_{zone}" in features

    def test_sub_zone_distributions_present(self, features):
        for key in ("sub_zone_distribution_close", "sub_zone_distribution_mid",
                    "sub_zone_distribution_three"):
            assert key in features

    def test_has_shot_chart_true(self, features):
        assert features["has_shot_chart"] is True

    def test_low_minutes_false(self, features):
        # gp=70, min=34 → not low minutes
        assert features["low_minutes"] is False

    def test_position_one_hot(self, features):
        pos = features["position"]
        pos_key = f"is_{pos.lower()}"
        assert features[pos_key] is True
        for p in ("pg", "sg", "sf", "pf", "c"):
            key = f"is_{p}"
            if p == pos.lower():
                assert features[key] is True
            else:
                assert features[key] is False

    def test_fg3a_rate_correct(self, features):
        # fg3a=6, fga=17 → 6/17 ≈ 0.353
        assert features["fg3a_rate"] == pytest.approx(6.0 / 17.0, abs=0.01)

    def test_percentiles_between_0_and_1(self, features):
        for key in ("pctile_pts", "pctile_ast", "pctile_reb"):
            assert 0.0 <= features[key] <= 1.0, f"{key} out of [0,1]"


class TestFeatureEngineEmptyShotChart:
    def test_has_shot_chart_false(self):
        class NoShotClient(MockNBAClient):
            def get_shot_chart(self, player_id, season="2024-25"):
                return []

        engine = FeatureEngine(NoShotClient())
        features = engine.build_features(1)
        assert features["has_shot_chart"] is False

    def test_zone_features_still_present(self):
        class NoShotClient(MockNBAClient):
            def get_shot_chart(self, player_id, season="2024-25"):
                return []

        engine = FeatureEngine(NoShotClient())
        features = engine.build_features(1)
        from src.features.shot_zones import ZONES
        for zone in ZONES:
            assert f"zone_fga_rate_{zone}" in features


class TestFeatureEngineLowMinutes:
    def test_low_gp_flagged(self):
        class LowGPClient(MockNBAClient):
            def get_player_stats(self, player_id, season="2024-25"):
                stats = super().get_player_stats(player_id, season)
                stats["gp"] = 3
                return stats

        engine = FeatureEngine(LowGPClient())
        features = engine.build_features(1)
        assert features["low_minutes"] is True

    def test_low_min_per_game_flagged(self):
        class LowMinClient(MockNBAClient):
            def get_player_stats(self, player_id, season="2024-25"):
                stats = super().get_player_stats(player_id, season)
                stats["min"] = 3.0
                return stats

        engine = FeatureEngine(LowMinClient())
        features = engine.build_features(1)
        assert features["low_minutes"] is True
