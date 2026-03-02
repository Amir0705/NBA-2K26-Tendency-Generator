"""Hybrid combiner stub — Phase 5 implementation."""
from __future__ import annotations

from typing import Any


class HybridCombiner:
    """
    Merges formula-layer and ML-layer tendency predictions into a
    single authoritative set of values.
    """

    def __init__(
        self,
        registry_path: str,
        formula_weight: float = 0.5,
        ml_weight: float = 0.5,
    ) -> None:
        """
        Initialise combiner with blend weights.

        Parameters
        ----------
        registry_path:  Path to data/tendency_registry.json.
        formula_weight: Relative weight given to formula predictions.
        ml_weight:      Relative weight given to ML predictions.
        """
        raise NotImplementedError("Phase 5 implementation")

    def combine(
        self,
        formula_preds: dict[str, int],
        ml_preds: dict[str, int],
        confidence: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """
        Blend formula and ML predictions into final tendency values.

        Parameters
        ----------
        formula_preds: Canonical-name → value from FormulaLayer.
        ml_preds:      Canonical-name → value from TendencyPredictor.
        confidence:    Optional per-tendency confidence weights.

        Returns
        -------
        Dict of canonical_name → blended integer value (pre-cap).
        """
        raise NotImplementedError("Phase 5 implementation")

    def explain(
        self,
        formula_preds: dict[str, int],
        ml_preds: dict[str, int],
        final: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Return per-tendency explanation of how the blend was applied."""
        raise NotImplementedError("Phase 5 implementation")
