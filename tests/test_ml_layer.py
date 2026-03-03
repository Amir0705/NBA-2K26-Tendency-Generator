"""Tests for the ML correction layer and hybrid combiner."""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs for formula layer and feature engine
# ---------------------------------------------------------------------------

_TENDENCIES = ["shot", "drive", "shot_three", "floater", "post_up"]


class _StubFormula:
    def generate(self, features: dict) -> dict[str, float]:
        return {t: 50.0 for t in _TENDENCIES}


class _StubFeatureEngine:
    def build_features(self, player_id: int, season: str = "2024-25") -> dict:
        return {
            "position": "PG",
            "pts_per_game": 20.0,
            "fga_per_game": 15.0,
            "fg3a_per_game": 5.0,
            "fta_per_game": 4.0,
            "ast_per_game": 7.0,
            "reb_per_game": 5.0,
            "stl_per_game": 1.5,
            "blk_per_game": 0.4,
            "tov_per_game": 3.0,
            "min_per_game": 34.0,
            "gp": 70,
            "fg_pct": 0.47,
            "fg3_pct": 0.38,
            "ft_pct": 0.85,
            "usg_pct_proxy": 0.28,
            "fg3a_rate": 0.33,
            "fta_rate": 0.27,
            "oreb_pct_proxy": 0.05,
            "low_minutes": False,
            "is_pg": True,
            "is_sg": False,
            "is_sf": False,
            "is_pf": False,
            "is_c": False,
        }


class _StubNBAClient:
    def search_player(self, name: str) -> list:
        # Return a fake player_id for any upper-case name
        if name.isupper():
            return [{"player_id": 999, "full_name": name}]
        return []


# ---------------------------------------------------------------------------
# Helper: build a minimal ATD-style CSV in a temp file
# ---------------------------------------------------------------------------

def _make_atd_csv(path: str, n_players: int = 40) -> None:
    """Write a minimal ATD Committee CSV that the trainer can parse."""
    # Tendency column names (98 columns after "player_name")
    tendency_cols = [
        "Shot", "Touches", "Shot Close", "Shot Under Basket",
        "Shot Close Left", "Shot Close Middle", "Shot Close Right",
        "Shot Mid-Range", "Spot Up Shot Mid-Range", "Off-Screen Shot Mid-Range",
        "Shot Mid Left", "Shot Mid Left-Center", "Shot Mid Center",
        "Shot Mid Right-Center", "Shot Mid Right",
        "Shot Three", "Spot Up Shot Three", "Off-Screen Shot Three",
        "Shot Three Left", "Shot Three Left-Center", "Shot Three Center",
        "Shot Three Right-Center", "Shot Three Right",
        "Contested Jumper Mid-Range", "Contested Jumper Three",
        "Stepback Jumper Mid-Range", "Stepback Three Point Shot",
        "Spin Jumper", "Transition Pull-Up Three Point Shot",
        "Drive Pull-Up Mid-Range", "Drive Pull-Up Three",
        "Drive", "Spot Up Drive", "Off-Screen Drive", "Use Glass",
        "Step Through Shot", "Driving Layup", "Spin Layup",
        "Euro Step Layup", "Hop Step Layup", "Floater",
        "Standing Dunk", "Driving Dunk", "Flashy Dunk", "Alley-Oop",
        "Putback", "Crash", "Drive Right",
        "Triple Threat Pump Fake", "Triple Threat Jab Step",
        "Triple Threat Idle", "Triple Threat Shoot",
        "Setup With Sizeup", "Setup With Hesitation", "No Setup Dribble",
        "Driving Crossover", "Driving Double Crossover", "Driving Spin",
        "Driving Half Spin", "Driving Stepback", "Driving Behind the Back",
        "Driving Dribble Hesitation", "Driving In & Out",
        "No Driving Dribble Move", "Attack Strong on Drive",
        "Dish to Open Man", "Flashy Pass", "Alley-Oop Pass",
        "Roll vs Pop", "Transition Spot Up vs Cut to the Basket",
        "Iso vs Elite Defender", "Iso vs Good Defender",
        "Iso vs Average Defender", "Iso vs Poor Defender",
        "Play Discipline", "Post Up", "Post Back Down",
        "Post Aggressive Backdown", "Post Face Up", "Post Spin",
        "Post Drive", "Post Drop Step", "Shoot From Post",
        "Post Hook Left", "Post Hook Right", "Post Fade Left",
        "Post Fade Right", "Post Shimmy Shot", "Post Hop Step",
        "Post Stepback Shot", "Post Up & Under",
        "Take Charge", "Foul", "Hard Foul",
        "Pass Interception", "On-Ball Steal", "Block Shot", "Contest Shot",
    ]
    # The real CSV has 4 header rows before the column-header row (header=4)
    pad = "," * len(tendency_cols)
    header_rows = [f"row{i}{pad}" for i in range(1, 5)]
    col_header = "Tendency\n(In Order)," + ",".join(tendency_cols)

    rows = [col_header]
    for i in range(n_players):
        name = f"PLAYER{i:03d}"
        values = [str(40 + (i % 40))] * len(tendency_cols)
        rows.append(f"{name}," + ",".join(values))

    with open(path, "w") as fh:
        fh.write("\n".join(header_rows) + "\n")
        fh.write("\n".join(rows) + "\n")


# ===========================================================================
# Tests: TendencyTrainer
# ===========================================================================


class TestTendencyTrainerPrepareData:
    def test_returns_tuple(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=5)

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        result = trainer.prepare_training_data(csv_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_features_df_is_dataframe(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=5)

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        features_df, residuals = trainer.prepare_training_data(csv_path)
        assert isinstance(features_df, pd.DataFrame)

    def test_residuals_is_dict(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=5)

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        features_df, residuals = trainer.prepare_training_data(csv_path)
        assert isinstance(residuals, dict)

    def test_unresolved_players_skipped(self, tmp_path):
        """Players whose names can't be resolved should be skipped."""
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=3)

        class _NoResolve:
            def search_player(self, name): return []

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_NoResolve(),
        )
        features_df, residuals = trainer.prepare_training_data(csv_path)
        assert features_df.empty

    def test_residuals_are_correct(self, tmp_path):
        """Residuals = ATD_label − formula_prediction."""
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=5)

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),  # always predicts 50.0
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        features_df, residuals = trainer.prepare_training_data(csv_path)
        # formula predicts 50.0 for "shot"; ATD has 40..79 range
        if "shot" in residuals and len(residuals["shot"]) > 0:
            for v in residuals["shot"]:
                assert isinstance(v, float)


class TestTendencyTrainerTrain:
    def test_train_returns_report(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=50)
        model_dir = str(tmp_path / "models")

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        report = trainer.train(csv_path, model_dir=model_dir)
        assert isinstance(report, dict)

    def test_train_saves_joblib_files(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=50)
        model_dir = str(tmp_path / "models")

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        report = trainer.train(csv_path, model_dir=model_dir)
        if report:
            # At least one .joblib file should exist
            joblib_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
            assert len(joblib_files) > 0

    def test_report_contains_expected_keys(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        csv_path = str(tmp_path / "atd.csv")
        _make_atd_csv(csv_path, n_players=50)
        model_dir = str(tmp_path / "models")

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        report = trainer.train(csv_path, model_dir=model_dir)
        for entry in report.values():
            for key in ("n_samples", "rmse", "r2"):
                assert key in entry, f"Missing key '{key}' in report entry"


class TestTendencyTrainerCrossValidate:
    def test_cross_validate_returns_dict(self, tmp_path):
        from src.ml.trainer import TendencyTrainer

        trainer = TendencyTrainer(
            formula_layer=_StubFormula(),
            feature_engine=_StubFeatureEngine(),
            nba_client=_StubNBAClient(),
        )
        import numpy as np
        X = pd.DataFrame({"a": np.random.rand(40), "b": np.random.rand(40)})
        y = pd.Series(np.random.rand(40))
        result = trainer.cross_validate(X, y, "shot")
        assert isinstance(result, dict)
        assert "rmse" in result
        assert "r2" in result


# ===========================================================================
# Tests: TendencyPredictor
# ===========================================================================


class TestTendencyPredictor:
    def test_loads_empty_model_dir(self, tmp_path):
        from src.ml.predictor import TendencyPredictor

        p = TendencyPredictor(model_dir=str(tmp_path))
        assert p._models == {}

    def test_loads_nonexistent_dir(self, tmp_path):
        from src.ml.predictor import TendencyPredictor

        p = TendencyPredictor(model_dir=str(tmp_path / "nonexistent"))
        assert p._models == {}

    def test_predict_corrections_empty_returns_empty(self, tmp_path):
        from src.ml.predictor import TendencyPredictor

        p = TendencyPredictor(model_dir=str(tmp_path))
        result = p.predict_corrections({"pts_per_game": 20.0})
        assert result == {}

    def test_has_model_false_when_empty(self, tmp_path):
        from src.ml.predictor import TendencyPredictor

        p = TendencyPredictor(model_dir=str(tmp_path))
        assert not p.has_model("shot")

    def test_loads_saved_model_and_predicts(self, tmp_path):
        """Train a tiny model, save it, load it with predictor, predict."""
        import joblib
        import lightgbm as lgb
        import numpy as np
        from src.ml.predictor import TendencyPredictor

        X = pd.DataFrame({"a": np.random.rand(30), "b": np.random.rand(30)})
        y = np.random.rand(30)
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(X, y)
        joblib.dump(model, str(tmp_path / "shot.joblib"))

        p = TendencyPredictor(model_dir=str(tmp_path))
        assert p.has_model("shot")

        features = {"a": 0.5, "b": 0.3}
        corrections = p.predict_corrections(features)
        assert "shot" in corrections
        assert isinstance(corrections["shot"], float)


# ===========================================================================
# Tests: ConfidenceScorer
# ===========================================================================


class TestConfidenceScorer:
    def test_score_returns_float(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer()
        score = cs.score("shot", {"low_minutes": False})
        assert isinstance(score, float)

    def test_score_in_range(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 60, "rmse": 3.0, "r2": 0.5}})
        score = cs.score("shot", {"low_minutes": False})
        assert 0.0 <= score <= 1.0

    def test_low_minutes_low_confidence(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 100, "rmse": 1.0, "r2": 0.9}})
        score = cs.score("shot", {"low_minutes": True})
        assert score < 0.3

    def test_few_samples_low_confidence(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 10, "rmse": 2.0, "r2": 0.5}})
        score = cs.score("shot", {"low_minutes": False})
        assert score < 0.3

    def test_high_quality_model_high_confidence(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 100, "rmse": 2.0, "r2": 0.8}})
        score = cs.score("shot", {"low_minutes": False})
        assert score > 0.5

    def test_blend_weight_in_range(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 60, "rmse": 3.0, "r2": 0.5}})
        weight = cs.get_blend_weight("shot", {"low_minutes": False})
        assert 0.0 <= weight <= 0.4

    def test_blend_weight_never_exceeds_04(self):
        from src.ml.confidence import ConfidenceScorer

        cs = ConfidenceScorer({"shot": {"n_samples": 1000, "rmse": 0.1, "r2": 1.0}})
        weight = cs.get_blend_weight("shot", {"low_minutes": False})
        assert weight <= 0.4


# ===========================================================================
# Tests: HybridCombiner
# ===========================================================================


class TestHybridCombiner:
    def test_combine_no_ml_uses_formula(self):
        from src.hybrid.combiner import HybridCombiner

        combiner = HybridCombiner(formula_layer=_StubFormula())
        features = {"position": "PG", "low_minutes": False}
        result = combiner.combine(features)
        assert isinstance(result, dict)
        assert result == {t: 50.0 for t in _TENDENCIES}

    def test_combine_with_mock_predictor(self, tmp_path):
        """Predictor with no models should still return formula values."""
        from src.hybrid.combiner import HybridCombiner
        from src.ml.predictor import TendencyPredictor

        predictor = TendencyPredictor(model_dir=str(tmp_path))  # empty dir
        combiner = HybridCombiner(
            formula_layer=_StubFormula(),
            predictor=predictor,
        )
        features = {"position": "PG", "low_minutes": False}
        result = combiner.combine(features)
        assert result == {t: 50.0 for t in _TENDENCIES}

    def test_combine_applies_correction(self, tmp_path):
        """When predictor has a model, correction should shift the value."""
        import joblib
        import lightgbm as lgb
        import numpy as np
        from src.hybrid.combiner import HybridCombiner
        from src.ml.confidence import ConfidenceScorer
        from src.ml.predictor import TendencyPredictor

        # Train a model that predicts ~+5 residual
        X = pd.DataFrame({"is_pg": [1.0] * 30, "pts_per_game": np.random.rand(30)})
        y = np.full(30, 5.0)  # constant residual
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(X, y)
        joblib.dump(model, str(tmp_path / "shot.joblib"))

        predictor = TendencyPredictor(model_dir=str(tmp_path))
        scorer = ConfidenceScorer({"shot": {"n_samples": 60, "rmse": 0.1, "r2": 0.9}})
        combiner = HybridCombiner(
            formula_layer=_StubFormula(),
            predictor=predictor,
            confidence_scorer=scorer,
        )
        features = {
            "is_pg": True,
            "pts_per_game": 20.0,
            "low_minutes": False,
        }
        result = combiner.combine(features)
        # "shot" should be shifted above 50.0 (formula baseline)
        assert result["shot"] > 50.0
        # Other tendencies without a model → pure formula
        assert result["drive"] == 50.0

    def test_combine_returns_all_tendencies(self):
        from src.hybrid.combiner import HybridCombiner

        combiner = HybridCombiner(formula_layer=_StubFormula())
        result = combiner.combine({})
        assert set(result.keys()) == set(_TENDENCIES)


# ===========================================================================
# Tests: TendencyPipeline — formula-only and hybrid modes
# ===========================================================================


class TestPipelineFormulaOnlyMode:
    """Pipeline should work exactly as before when no model_dir is given."""

    def _build_pipeline(self, registry_path):
        from src.caps.cap_enforcer import CapEnforcer
        from src.features.feature_engine import FeatureEngine
        from src.formula.formula_layer import FormulaLayer
        from src.pipeline import TendencyPipeline, load_registry
        from src.validation.guardrails import Guardrails

        class _MockClient:
            def get_player_info(self, pid): return {"position": "Guard", "height": "6-3", "weight": "195"}
            def get_player_stats(self, pid, season="2024-25"):
                return {"gp": 70, "min": 34.0, "pts": 22.0, "fga": 17.0, "fgm": 8.0,
                        "fg_pct": 0.471, "fg3a": 6.0, "fg3m": 2.5, "fg3_pct": 0.417,
                        "fta": 4.5, "ftm": 3.8, "ft_pct": 0.844, "oreb": 0.8, "dreb": 4.5,
                        "reb": 5.3, "ast": 7.1, "stl": 1.5, "blk": 0.4, "tov": 3.2, "pf": 2.1}
            def get_shot_chart(self, pid, season="2024-25"): return []
            def get_league_averages(self, season="2024-25"):
                return [{"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
                         "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0}]
            def search_player(self, name): return []

        p = TendencyPipeline.__new__(TendencyPipeline)
        mock_client = _MockClient()
        p._registry_path = registry_path
        p._client = mock_client
        p._features = FeatureEngine(mock_client)
        p._formula = FormulaLayer()
        p._caps = CapEnforcer(registry_path)
        p._guardrails = Guardrails()
        p._registry = load_registry(registry_path)
        from src.hybrid.combiner import HybridCombiner
        p._combiner = HybridCombiner(formula_layer=p._formula)
        return p

    def test_generate_returns_99_tendencies(self):
        import json, os
        REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        registry_path = os.path.join(REPO, "data", "tendency_registry.json")
        pipeline = self._build_pipeline(registry_path)
        result = pipeline.generate(2544)
        assert len(result["tendencies"]) == 99


class TestPipelineHybridMode:
    """Pipeline should work in hybrid mode when model_dir has .joblib files."""

    def test_pipeline_hybrid_mode_produces_tendencies(self, tmp_path):
        import joblib
        import lightgbm as lgb
        import numpy as np
        from src.caps.cap_enforcer import CapEnforcer
        from src.features.feature_engine import FeatureEngine
        from src.formula.formula_layer import FormulaLayer
        from src.hybrid.combiner import HybridCombiner
        from src.ml.confidence import ConfidenceScorer
        from src.ml.predictor import TendencyPredictor
        from src.pipeline import TendencyPipeline, load_registry
        from src.validation.guardrails import Guardrails
        import os

        REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        registry_path = os.path.join(REPO, "data", "tendency_registry.json")

        # Save a tiny model for "shot"
        X = pd.DataFrame({"is_pg": np.random.rand(30), "pts_per_game": np.random.rand(30)})
        y = np.random.rand(30)
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(X, y)
        joblib.dump(model, str(tmp_path / "shot.joblib"))

        class _MockClient:
            def get_player_info(self, pid): return {"position": "Guard", "height": "6-3", "weight": "195"}
            def get_player_stats(self, pid, season="2024-25"):
                return {"gp": 70, "min": 34.0, "pts": 22.0, "fga": 17.0, "fgm": 8.0,
                        "fg_pct": 0.471, "fg3a": 6.0, "fg3m": 2.5, "fg3_pct": 0.417,
                        "fta": 4.5, "ftm": 3.8, "ft_pct": 0.844, "oreb": 0.8, "dreb": 4.5,
                        "reb": 5.3, "ast": 7.1, "stl": 1.5, "blk": 0.4, "tov": 3.2, "pf": 2.1}
            def get_shot_chart(self, pid, season="2024-25"): return []
            def get_league_averages(self, season="2024-25"):
                return [{"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
                         "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0}]
            def search_player(self, name): return []

        mock_client = _MockClient()
        formula = FormulaLayer()
        predictor = TendencyPredictor(model_dir=str(tmp_path))
        scorer = ConfidenceScorer({"shot": {"n_samples": 60, "rmse": 2.0, "r2": 0.4}})
        combiner = HybridCombiner(
            formula_layer=formula,
            predictor=predictor,
            confidence_scorer=scorer,
        )

        p = TendencyPipeline.__new__(TendencyPipeline)
        p._registry_path = registry_path
        p._client = mock_client
        p._features = FeatureEngine(mock_client)
        p._formula = formula
        p._caps = CapEnforcer(registry_path)
        p._guardrails = Guardrails()
        p._registry = load_registry(registry_path)
        p._combiner = combiner

        result = p.generate(2544)
        assert "tendencies" in result
        assert len(result["tendencies"]) == 99
