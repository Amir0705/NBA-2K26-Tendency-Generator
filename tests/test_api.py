"""Tests for src/api/app.py."""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from src.pipeline import TendencyPipeline, load_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")
PRIMJER_TENDENCY_FIELDS = ("value", "label", "offset", "type", "bit_offset", "bit_length", "length")


# ---------------------------------------------------------------------------
# Mock objects (same as test_cli.py)
# ---------------------------------------------------------------------------

class MockNBAClient:
    def get_player_info(self, player_id: int) -> dict:
        return {"position": "Guard", "height": "6-3", "weight": "185",
                "team_id": 1610612744, "team_abbreviation": "GSW"}

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return {"gp": 74, "min": 33.5, "pts": 26.4, "fga": 19.0, "fgm": 8.9,
                "fg_pct": 0.468, "fg3a": 11.6, "fg3m": 5.1, "fg3_pct": 0.438,
                "fta": 4.5, "ftm": 4.1, "ft_pct": 0.916, "oreb": 0.5,
                "dreb": 4.6, "reb": 5.1, "ast": 6.1, "stl": 0.7, "blk": 0.2,
                "tov": 3.0, "pf": 2.3, "plus_minus": 6.2}

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return [{"shot_zone_basic": "Above the Break 3", "shot_zone_area": "Center(C)",
                 "shot_zone_range": "24+ ft.", "shot_made_flag": 1,
                 "loc_x": 10, "loc_y": 240, "shot_type": "3PT Field Goal",
                 "action_type": "Jump Shot"}]

    def get_league_averages(self, season: str = "2024-25") -> list:
        return [{"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
                 "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0}]

    def search_player(self, name: str) -> list:
        db = {
            "curry": [{"player_id": 201939, "full_name": "Stephen Curry",
                       "team": "GSW", "is_active": True}],
            "james": [{"player_id": 2544, "full_name": "LeBron James",
                       "team": "LAL", "is_active": True}],
        }
        name_lower = name.lower()
        for key, players in db.items():
            if key in name_lower:
                return players
        return []

    def get_team_roster(self, team_abbreviation: str, season: str = "2024-25") -> list:
        if team_abbreviation.upper() == "GSW":
            return [
                {"player_id": 201939, "full_name": "Stephen Curry", "position": "PG"},
            ]
        return []


def _make_pipeline():
    from src.caps.cap_enforcer import CapEnforcer
    from src.features.feature_engine import FeatureEngine
    from src.formula.formula_layer import FormulaLayer
    from src.hybrid.combiner import HybridCombiner
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
    p._combiner = HybridCombiner(formula_layer=p._formula, predictor=None, confidence_scorer=None)
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    import src.api.app as app_module
    pipeline = _make_pipeline()
    app_module._pipeline = pipeline
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


class TestSearchEndpoint:
    def test_found(self, client):
        resp = client.get("/search/Curry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Curry"
        assert len(data["results"]) >= 1
        assert data["results"][0]["player_id"] == 201939

    def test_not_found_returns_empty(self, client):
        resp = client.get("/search/NONEXISTENTXYZ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []


class TestGenerateByName:
    def test_generates_tendencies(self, client):
        resp = client.get("/generate/Stephen Curry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_name"] == "Stephen Curry"
        assert data["player_id"] == 201939
        assert "tendencies" in data
        assert len(data["tendencies"]) == 99

    def test_not_found_returns_404(self, client):
        resp = client.get("/generate/NOBODYX")
        assert resp.status_code == 404

    def test_response_has_required_fields(self, client):
        resp = client.get("/generate/Stephen Curry")
        data = resp.json()
        for field in ("player_name", "player_id", "position", "team", "season", "tendencies"):
            assert field in data, f"Missing field: {field}"

    def test_season_param(self, client):
        resp = client.get("/generate/Stephen Curry?season=2023-24")
        assert resp.status_code == 200
        data = resp.json()
        assert data["season"] == "2023-24"


class TestGenerateById:
    def test_generates_tendencies(self, client):
        resp = client.get("/generate/id/201939")
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_id"] == 201939
        assert "tendencies" in data
        assert len(data["tendencies"]) == 99


class TestTeamEndpoint:
    def test_generates_team(self, client):
        resp = client.get("/team/GSW")
        assert resp.status_code == 200
        data = resp.json()
        assert data["team"] == "GSW"
        assert data["season"] == "2024-25"
        assert data["roster_season"] == "2025-26"
        assert isinstance(data["players"], list)
        assert data["player_count"] == len(data["players"])
        assert data["generated_count"] == data["player_count"]
        assert data["total_players"] >= data["generated_count"]
        assert data["failed_count"] == data["total_players"] - data["generated_count"]

    def test_invalid_team_returns_404(self, client):
        resp = client.get("/team/XYZ")
        assert resp.status_code == 404

    def test_team_season_param(self, client):
        resp = client.get("/team/GSW?season=2023-24")
        assert resp.status_code == 200
        data = resp.json()
        assert data["season"] == "2023-24"

    def test_team_roster_season_param(self, client):
        resp = client.get("/team/GSW?season=2024-25&roster_season=2023-24")
        assert resp.status_code == 200
        data = resp.json()
        assert data["season"] == "2024-25"
        assert data["roster_season"] == "2023-24"


class TestTeamPlayerEndpoint:
    def test_generates_player_on_team(self, client):
        resp = client.get("/team/GSW/Stephen Curry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_name"] == "Stephen Curry"
        assert "tendencies" in data

    def test_player_not_on_team_returns_404(self, client):
        resp = client.get("/team/GSW/LeBron James")
        assert resp.status_code == 404

    def test_invalid_team_returns_404(self, client):
        resp = client.get("/team/XYZ/Stephen Curry")
        assert resp.status_code == 404


class TestTendencyValues:
    def test_all_values_in_range(self, client):
        resp = client.get("/generate/Stephen Curry")
        data = resp.json()
        for key, entry in data["tendencies"].items():
            val = entry["value"]
            assert 0 <= val <= 100, f"{key}: {val} out of [0,100]"

    def test_all_values_have_label(self, client):
        resp = client.get("/generate/Stephen Curry")
        data = resp.json()
        for key, entry in data["tendencies"].items():
            assert "label" in entry, f"Missing label for {key}"
            assert "value" in entry, f"Missing value for {key}"

    def test_tendency_entry_has_all_primjer_fields(self, client):
        """Each tendency entry must include all primjer.txt-compatible fields."""
        resp = client.get("/generate/Stephen Curry")
        data = resp.json()
        for key, entry in data["tendencies"].items():
            for field in PRIMJER_TENDENCY_FIELDS:
                assert field in entry, f"Tendency '{key}' missing field '{field}'"

    def test_team_tendency_entry_has_all_primjer_fields(self, client):
        """Team endpoint tendency entries must also include all primjer.txt-compatible fields."""
        resp = client.get("/team/GSW")
        data = resp.json()
        for player in data["players"]:
            for key, entry in player["tendencies"].items():
                for field in PRIMJER_TENDENCY_FIELDS:
                    assert field in entry, (
                        f"Player '{player['player_name']}' tendency '{key}' missing field '{field}'"
                    )
