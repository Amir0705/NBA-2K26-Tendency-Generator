"""Confidence scoring stub — Phase 4 implementation."""
from __future__ import annotations

from typing import Any


class ConfidenceScorer:
    """Assigns confidence scores to ML-generated tendency predictions."""

    def __init__(self, registry_path: str) -> None:
        """
        Initialise scorer.

        Parameters
        ----------
        registry_path: Path to data/tendency_registry.json.
        """
        raise NotImplementedError("Phase 4 implementation")

    def score(
        self,
        predictions: dict[str, int],
        features: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute per-tendency confidence scores in [0.0, 1.0].

        Parameters
        ----------
        predictions: Dict of canonical_name → predicted value.
        features:    Feature dict used to generate the predictions.

        Returns
        -------
        Dict of canonical_name → confidence float.
        """
        raise NotImplementedError("Phase 4 implementation")

    def overall(self, scores: dict[str, float]) -> float:
        """Return a single overall confidence score (weighted mean)."""
        raise NotImplementedError("Phase 4 implementation")
