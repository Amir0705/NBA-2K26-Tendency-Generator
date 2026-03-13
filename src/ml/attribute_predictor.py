"""Attribute ML predictor — applies trained residual corrections to formula attributes."""
from __future__ import annotations

import json
import os
from typing import Any


class AttributePredictor:
    """Loads trained attribute models and generates residual corrections."""

    def __init__(self, model_dir: str = "models/attributes/") -> None:
        self._models: dict = {}
        self._improvement_threshold = 1.0  # only use models that improved by at least this
        self._max_correction = 8.0  # clamp corrections to ±this value
        self._beneficial_attrs: set[str] = set()
        self._load_models(model_dir)

    def _load_models(self, model_dir: str) -> None:
        """Load models and the training report to filter beneficial ones."""
        if not os.path.isdir(model_dir):
            return
        try:
            import joblib
        except ImportError:
            return

        # Load training report to know which models actually helped
        report_path = os.path.join(model_dir, "training_report.json")
        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            for attr_name, info in report.items():
                if info.get("improvement", 0) > self._improvement_threshold:
                    self._beneficial_attrs.add(attr_name)

        # Load model files — only keep beneficial ones
        for fname in os.listdir(model_dir):
            if fname.endswith(".joblib"):
                attr_name = fname[:-7]
                if attr_name not in self._beneficial_attrs:
                    continue
                path = os.path.join(model_dir, fname)
                try:
                    self._models[attr_name] = joblib.load(path)
                except Exception:  # noqa: BLE001
                    pass

    @property
    def available_attributes(self) -> list[str]:
        """Return list of attributes that have beneficial ML corrections."""
        return sorted(self._models.keys())

    def predict_corrections(self, features: dict) -> dict[str, float]:
        """Predict residual corrections for all beneficial attributes."""
        if not self._models:
            return {}
        import pandas as pd

        flat = _flatten_features(features)
        # Add position one-hot flags
        pos = features.get("position", "SF")
        for pos_label in ("PG", "SG", "SF", "PF", "C"):
            flat[f"pos_{pos_label}"] = 1.0 if pos == pos_label else 0.0

        corrections: dict[str, float] = {}
        for attr_name, model in self._models.items():
            try:
                X = pd.DataFrame([flat])
                if hasattr(model, "feature_name_"):
                    X = X.reindex(columns=model.feature_name_, fill_value=0.0)
                elif hasattr(model, "feature_names_in_"):
                    X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)
                pred = model.predict(X)[0]
                corrections[attr_name] = float(pred)
            except Exception:  # noqa: BLE001
                pass
        return corrections

    def apply_corrections(
        self, attributes: dict[str, int], features: dict
    ) -> dict[str, int]:
        """Apply ML corrections to formula attributes and return corrected dict."""
        corrections = self.predict_corrections(features)
        corrected = dict(attributes)
        for attr_name, correction in corrections.items():
            if attr_name in corrected:
                # Clamp correction to prevent wild swings
                clamped = max(-self._max_correction, min(self._max_correction, correction))
                raw = corrected[attr_name] + clamped
                corrected[attr_name] = max(25, min(99, int(round(raw))))
        return corrected

    def has_models(self) -> bool:
        """Check if any beneficial models are loaded."""
        return len(self._models) > 0


def _flatten_features(features: dict) -> dict[str, float]:
    """Flatten nested feature dicts to scalar float values."""
    flat: dict[str, float] = {}
    for k, v in features.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}__{sub_k}"] = float(sub_v) if sub_v is not None else 0.0
        elif isinstance(v, bool):
            flat[k] = float(v)
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
    return flat
