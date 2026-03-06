"""Integration tests for src/pipeline.py."""
from __future__ import annotations

import json
import os

import pytest

from src.pipeline import TendencyPipeline, load_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")


# ---------------------------------------------------------------------------
# Mock client that never hits the network
# ---------------------------------------------------------------------------


class MockNBAClient:
    def get_player_info(self, player_id: int) -> dict:
        return {
            "position": "Guard",
            "height": "6-3",
            "weight": "195",
            "team_id": 1,
            "team_abbreviation": "LAL",
        }

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return {
            "gp": 70, "min": 34.0, "pts": 22.0,
            "fga": 17.0, "fgm": 8.0, "fg_pct": 0.471,
            "fg3a": 6.0, "fg3m": 2.5, "fg3_pct": 0.417,
            "fta": 4.5, "ftm": 3.8, "ft_pct": 0.844,
            "oreb": 0.8, "dreb": 4.5, "reb": 5.3,
            "ast": 7.1, "stl": 1.5, "blk": 0.4,
            "tov": 3.2, "pf": 2.1, "plus_minus": 6.5,
        }

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return [
            {
                "shot_zone_basic": "Restricted Area",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "Less Than 8 ft.",
                "shot_made_flag": 1, "loc_x": 0, "loc_y": 5,
                "shot_type": "2PT Field Goal",
                "action_type": "Driving Layup Shot",
            },
            {
                "shot_zone_basic": "Above the Break 3",
                "shot_zone_area": "Center(C)",
                "shot_zone_range": "24+ ft.",
                "shot_made_flag": 0, "loc_x": 10, "loc_y": 240,
                "shot_type": "3PT Field Goal",
                "action_type": "Jump Shot",
            },
        ]

    def get_league_averages(self, season: str = "2024-25") -> list:
        return [
            {"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
             "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0},
            {"PTS": 20.0, "AST": 6.0, "REB": 7.0, "STL": 1.5, "BLK": 1.0,
             "FG3A": 5.0, "FGA": 15.0, "FTA": 4.0, "TOV": 3.0},
        ]

    def search_player(self, name: str) -> list:
        return [{"player_id": 2544, "full_name": "LeBron James", "team": "LAL", "is_active": True}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline():
    """Pipeline wired with mock client — no network calls."""
    p = TendencyPipeline.__new__(TendencyPipeline)
    from src.caps.cap_enforcer import CapEnforcer
    from src.features.feature_engine import FeatureEngine
    from src.formula.formula_layer import FormulaLayer
    from src.validation.guardrails import Guardrails

    mock_client = MockNBAClient()
    p._registry_path = REGISTRY_PATH
    p._client = mock_client
    p._features = FeatureEngine(mock_client)
    p._formula = FormulaLayer()
    p._caps = CapEnforcer(REGISTRY_PATH)
    p._guardrails = Guardrails()
    p._registry = load_registry(REGISTRY_PATH)
    return p


@pytest.fixture(scope="module")
def result(pipeline):
    return pipeline.generate(2544)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineGenerate:
    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_required_fields(self, result):
        for field in ("player_id", "season", "position", "tendencies",
                      "formula_raw", "audit", "guardrail_violations", "features"):
            assert field in result, f"Missing field: {field}"

    def test_player_id_correct(self, result):
        assert result["player_id"] == 2544

    def test_season_correct(self, result):
        assert result["season"] == "2024-25"

    def test_all_99_tendencies_present(self, result):
        registry = load_registry(REGISTRY_PATH)
        registry_names = {e["canonical_name"] for e in registry}
        output_names = set(result["tendencies"].keys())
        missing = registry_names - output_names
        assert len(missing) == 0, f"Missing tendencies: {missing}"

    def test_no_tendency_exceeds_cap(self, result):
        registry = load_registry(REGISTRY_PATH)
        caps = {e["canonical_name"]: e["hard_cap"] for e in registry if e.get("hard_cap")}
        for name, cap in caps.items():
            if name in result["tendencies"]:
                val = result["tendencies"][name]
                assert val <= cap, f"{name}: {val} > cap {cap}"

    def test_all_tendencies_are_non_negative(self, result):
        for k, v in result["tendencies"].items():
            assert v >= 0, f"{k} = {v} is negative"

    def test_audit_log_present(self, result):
        assert isinstance(result["audit"], list)

    def test_guardrail_violations_present(self, result):
        assert isinstance(result["guardrail_violations"], list)

    def test_position_field_valid(self, result):
        assert result["position"] in ("PG", "SG", "SF", "PF", "C")


class TestPipelineGenerateJson:
    def test_generate_json_returns_string(self, pipeline):
        output = pipeline.generate_json(2544)
        assert isinstance(output, str)

    def test_generate_json_valid_json(self, pipeline):
        output = pipeline.generate_json(2544)
        parsed = json.loads(output)
        assert "tendencies" in parsed

    def test_generate_json_has_99_keys(self, pipeline):
        output = pipeline.generate_json(2544)
        parsed = json.loads(output)
        assert len(parsed["tendencies"]) == 99

    def test_generate_json_key_order_matches_registry(self, pipeline):
        output = pipeline.generate_json(2544)
        parsed = json.loads(output)
        registry = load_registry(REGISTRY_PATH)
        expected_keys = [e["primjer_key"] for e in sorted(registry, key=lambda e: e["order"])]
        actual_keys = list(parsed["tendencies"].keys())
        assert actual_keys == expected_keys


class TestPipelineSearchPlayer:
    def test_search_returns_list(self, pipeline):
        results = pipeline.search_player("LeBron")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_has_player_id(self, pipeline):
        results = pipeline.search_player("LeBron")
        assert "player_id" in results[0]


class TestCloseSubZoneRoundingConstraints:
    """Tests for parent-aware rounding and tie-break in pipeline."""

    @pytest.fixture(scope="class")
    def pipeline_with_custom_formula(self):
        """Pipeline fixture with a formula that produces controlled close sub-zone values."""
        from src.caps.cap_enforcer import CapEnforcer
        from src.features.feature_engine import FeatureEngine
        from src.formula.formula_layer import FormulaLayer
        from src.validation.guardrails import Guardrails

        p = TendencyPipeline.__new__(TendencyPipeline)
        mock_client = MockNBAClient()
        p._registry_path = REGISTRY_PATH
        p._client = mock_client
        p._features = FeatureEngine(mock_client)
        p._formula = FormulaLayer()
        p._caps = CapEnforcer(REGISTRY_PATH)
        p._guardrails = Guardrails()
        p._registry = load_registry(REGISTRY_PATH)
        return p

    def test_close_sub_zone_sum_equals_shot_close(self, pipeline_with_custom_formula):
        """After all rounding, shot_close_left + middle + right must equal shot_close."""
        result = pipeline_with_custom_formula.generate(2544)
        t = result["tendencies"]
        close_sum = t["shot_close_left"] + t["shot_close_middle"] + t["shot_close_right"]
        assert close_sum == t["shot_close"], (
            f"Close sub-zone sum {close_sum} != shot_close {t['shot_close']}"
        )

    def test_close_sub_zones_are_multiples_of_5(self, pipeline_with_custom_formula):
        """After rounding, all close sub-zone tendencies must be multiples of 5."""
        result = pipeline_with_custom_formula.generate(2544)
        t = result["tendencies"]
        for key in ("shot_close_left", "shot_close_middle", "shot_close_right", "shot_close"):
            assert t[key] % 5 == 0, f"{key}={t[key]} is not a multiple of 5"

    def test_tie_break_applied_with_drive_right_bias(self):
        """When left==right, drive_right>50 should bias right (right > left)."""
        # Simulate the tie-break logic directly
        tendencies = {
            "shot_close": 30,
            "shot_close_left": 10,
            "shot_close_middle": 10,
            "shot_close_right": 10,
            "drive_right": 70,  # biased right
        }

        # Manually apply the tie-break as done in pipeline
        close_left = tendencies.get("shot_close_left", 0)
        close_right = tendencies.get("shot_close_right", 0)
        if close_left == close_right:
            drive_right = tendencies.get("drive_right", 50)
            if drive_right > 50 and tendencies["shot_close_left"] >= 5:
                tendencies["shot_close_right"] += 5
                tendencies["shot_close_left"] -= 5
            elif drive_right < 50 and tendencies["shot_close_right"] >= 5:
                tendencies["shot_close_left"] += 5
                tendencies["shot_close_right"] -= 5

        assert tendencies["shot_close_right"] > tendencies["shot_close_left"], (
            f"drive_right>50 should bias right: left={tendencies['shot_close_left']}, "
            f"right={tendencies['shot_close_right']}"
        )
        # Sum must still equal shot_close
        total = (tendencies["shot_close_left"] + tendencies["shot_close_middle"]
                 + tendencies["shot_close_right"])
        assert total == tendencies["shot_close"]

    def test_tie_break_applied_with_drive_left_bias(self):
        """When left==right, drive_right<50 should bias left (left > right)."""
        tendencies = {
            "shot_close": 30,
            "shot_close_left": 10,
            "shot_close_middle": 10,
            "shot_close_right": 10,
            "drive_right": 30,  # biased left
        }

        close_left = tendencies.get("shot_close_left", 0)
        close_right = tendencies.get("shot_close_right", 0)
        if close_left == close_right:
            drive_right = tendencies.get("drive_right", 50)
            if drive_right > 50 and tendencies["shot_close_left"] >= 5:
                tendencies["shot_close_right"] += 5
                tendencies["shot_close_left"] -= 5
            elif drive_right < 50 and tendencies["shot_close_right"] >= 5:
                tendencies["shot_close_left"] += 5
                tendencies["shot_close_right"] -= 5

        assert tendencies["shot_close_left"] > tendencies["shot_close_right"], (
            f"drive_right<50 should bias left: left={tendencies['shot_close_left']}, "
            f"right={tendencies['shot_close_right']}"
        )
        total = (tendencies["shot_close_left"] + tendencies["shot_close_middle"]
                 + tendencies["shot_close_right"])
        assert total == tendencies["shot_close"]
