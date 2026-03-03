"""Tests for src/cli.py."""
from __future__ import annotations

import json
import os
import sys

import pytest

from src.pipeline import TendencyPipeline, load_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")


# ---------------------------------------------------------------------------
# Mock objects
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
                {"player_id": 1629029, "full_name": "Jordan Poole", "position": "SG"},
            ]
        return []


def _make_pipeline():
    """Build a pipeline with the mock client."""
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
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestSafeFilename:
    def test_basic(self):
        from src.cli import _safe_filename
        assert _safe_filename("Stephen Curry") == "stephen_curry_tendencies.json"

    def test_apostrophe(self):
        from src.cli import _safe_filename
        assert _safe_filename("D'Angelo Russell") == "dangelo_russell_tendencies.json"

    def test_dots(self):
        from src.cli import _safe_filename
        assert _safe_filename("J.R. Smith") == "jr_smith_tendencies.json"


class TestResolvePlayer:
    def test_exact_match_preferred(self):
        from src.cli import _resolve_player
        pipeline = _make_pipeline()
        result = _resolve_player(pipeline, "Stephen Curry")
        assert result is not None
        assert result["player_id"] == 201939

    def test_returns_none_when_not_found(self):
        from src.cli import _resolve_player
        pipeline = _make_pipeline()
        result = _resolve_player(pipeline, "NONEXISTENT PLAYER XYZ")
        assert result is None


# ---------------------------------------------------------------------------
# Tests for CLI main function
# ---------------------------------------------------------------------------

class TestCLISearch:
    def test_search_output(self, capsys, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        main(["--search", "Curry"])
        captured = capsys.readouterr()
        assert "Stephen Curry" in captured.out
        assert "201939" in captured.out

    def test_search_no_results(self, capsys, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        main(["--search", "ZZZNOBODY"])
        captured = capsys.readouterr()
        assert "No players found" in captured.out


class TestCLISinglePlayer:
    def test_generates_file(self, tmp_path, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        out_dir = str(tmp_path / "output")
        main(["Stephen Curry", "--output-dir", out_dir])
        expected = os.path.join(out_dir, "stephen_curry_tendencies.json")
        assert os.path.isfile(expected)
        with open(expected, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "tendencies" in data
        assert data["player_name"] == "Stephen Curry"

    def test_player_not_found_exits(self, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        with pytest.raises(SystemExit):
            main(["NONEXISTENT PLAYER XYZ", "--output-dir", "/tmp/test_out"])


class TestCLITeam:
    def test_team_generates_files(self, tmp_path, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        out_dir = str(tmp_path / "output")
        main(["--team", "GSW", "--output-dir", out_dir])
        team_dir = os.path.join(out_dir, "GSW")
        assert os.path.isdir(team_dir)
        files = os.listdir(team_dir)
        assert len(files) >= 1

    def test_invalid_team_exits(self, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        with pytest.raises(SystemExit):
            main(["--team", "XYZ", "--output-dir", "/tmp/test_out"])

    def test_no_args_exits(self, monkeypatch):
        from src.cli import main
        pipeline = _make_pipeline()
        monkeypatch.setattr("src.cli._build_pipeline", lambda season: pipeline)
        with pytest.raises(SystemExit):
            main([])
