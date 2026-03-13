"""ML predictor — loads trained residual-correction models and applies them."""
from __future__ import annotations

import os


class TendencyPredictor:
    """Loads trained models and generates residual corrections."""

    def __init__(self, model_dir: str = "models/") -> None:
        """
        Initialise predictor by loading all .joblib models from *model_dir*.

        Parameters
        ----------
        model_dir: Directory containing persisted model files.
        """
        self._models: dict = {}
        self._load_models(model_dir)

    def _load_models(self, model_dir: str) -> None:
        """Load all .joblib models from *model_dir*."""
        if not os.path.isdir(model_dir):
            return
        try:
            import joblib
        except ImportError:
            return
        for fname in os.listdir(model_dir):
            if fname.endswith(".joblib"):
                tendency_name = fname[:-7]  # strip ".joblib"
                path = os.path.join(model_dir, fname)
                try:
                    self._models[tendency_name] = joblib.load(path)
                except Exception:  # noqa: BLE001
                    pass

    def predict_corrections(self, features: dict) -> dict[str, float]:
        """
        For each loaded model predict the residual correction.

        Returns {tendency_name: predicted_residual}.
        """
        if not self._models:
            return {}
        import pandas as pd

        corrections: dict[str, float] = {}
        for tendency_name, model in self._models.items():
            try:
                flat = _flatten_features(features)
                X = pd.DataFrame([flat])
                # Align columns to what the model was trained on
                if hasattr(model, "feature_name_"):
                    X = X.reindex(columns=model.feature_name_, fill_value=0.0)
                elif hasattr(model, "feature_names_in_"):
                    X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)
                pred = model.predict(X)[0]
                corrections[tendency_name] = float(pred)
            except Exception:  # noqa: BLE001
                pass
        return corrections

    def has_model(self, tendency_name: str) -> bool:
        """Check if a trained model exists for this tendency."""
        return tendency_name in self._models


def _flatten_features(features: dict) -> dict[str, float]:
    """Flatten nested feature dicts (e.g. sub_zone_distribution_*) to scalar."""
    flat: dict[str, float] = {}
    for k, v in features.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}__{sub_k}"] = float(sub_v) if sub_v is not None else 0.0
        elif isinstance(v, bool):
            flat[k] = float(v)
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
        # skip non-numeric (e.g. position string) — handled via one-hot flags
    return flat
