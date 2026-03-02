"""ML model trainer stub — Phase 4 implementation."""
from __future__ import annotations

from typing import Any


class TendencyTrainer:
    """Trains per-tendency regression models from historical data."""

    def __init__(
        self,
        registry_path: str,
        model_dir: str,
    ) -> None:
        """
        Initialise trainer.

        Parameters
        ----------
        registry_path: Path to data/tendency_registry.json.
        model_dir:     Directory where trained models are persisted.
        """
        raise NotImplementedError("Phase 4 implementation")

    def train(
        self,
        training_data: list[dict[str, Any]],
        tendency_name: str,
    ) -> None:
        """
        Train a model for *tendency_name* from *training_data*.

        Parameters
        ----------
        training_data: List of {features, label} dicts.
        tendency_name: canonical_name of the target tendency.
        """
        raise NotImplementedError("Phase 4 implementation")

    def train_all(self, training_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Train models for all 99 tendencies; return metadata dict."""
        raise NotImplementedError("Phase 4 implementation")

    def save(self, output_dir: str) -> None:
        """Persist trained models to *output_dir*."""
        raise NotImplementedError("Phase 4 implementation")
