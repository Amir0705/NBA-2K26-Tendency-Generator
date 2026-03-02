"""ML predictor stub — Phase 4 implementation."""
from __future__ import annotations

from typing import Any


class TendencyPredictor:
    """Loads trained models and generates tendency predictions."""

    def __init__(self, model_dir: str, registry_path: str) -> None:
        """
        Initialise predictor by loading models from *model_dir*.

        Parameters
        ----------
        model_dir:     Directory containing persisted model files.
        registry_path: Path to data/tendency_registry.json.
        """
        raise NotImplementedError("Phase 4 implementation")

    def predict(
        self, features: dict[str, float], tendency_name: str
    ) -> int:
        """
        Return integer tendency prediction for one tendency.

        Parameters
        ----------
        features:      Feature dict from FeatureEngine.
        tendency_name: canonical_name of the tendency to predict.
        """
        raise NotImplementedError("Phase 4 implementation")

    def predict_all(
        self, features: dict[str, float]
    ) -> dict[str, int]:
        """
        Predict all 99 tendencies at once.

        Returns
        -------
        Dict of canonical_name → integer value (pre-cap).
        """
        raise NotImplementedError("Phase 4 implementation")
