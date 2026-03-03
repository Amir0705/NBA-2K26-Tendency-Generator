"""Hybrid combiner — blends formula predictions with ML residual corrections."""
from __future__ import annotations

from typing import Any


class HybridCombiner:
    """
    Merges formula-layer and ML residual corrections into a single
    authoritative set of tendency values.
    """

    def __init__(
        self,
        formula_layer: Any,
        predictor: Any | None = None,
        confidence_scorer: Any | None = None,
    ) -> None:
        """
        Initialise combiner.

        Parameters
        ----------
        formula_layer:     FormulaLayer instance (always used as baseline).
        predictor:         TendencyPredictor instance (optional).
        confidence_scorer: ConfidenceScorer instance (optional).
        """
        self._formula = formula_layer
        self._predictor = predictor
        self._confidence = confidence_scorer

    def combine(self, features: dict) -> dict[str, float]:
        """
        Combine formula + ML predictions.

        final = formula_value + (ml_weight * ml_correction)

        If no ML model exists for a tendency, use pure formula value.

        Parameters
        ----------
        features: Feature dict from FeatureEngine.

        Returns
        -------
        Dict of canonical_name → float tendency value (pre-cap).
        """
        formula_values: dict[str, float] = self._formula.generate(features)

        if self._predictor is None:
            return formula_values

        corrections = self._predictor.predict_corrections(features)
        if not corrections:
            return formula_values

        result: dict[str, float] = {}
        for name, formula_val in formula_values.items():
            if self._predictor.has_model(name):
                correction = corrections.get(name)
                if correction is not None:
                    if self._confidence is not None:
                        weight = self._confidence.get_blend_weight(name, features)
                    else:
                        weight = 0.2  # conservative default
                    result[name] = formula_val + weight * correction
                else:
                    result[name] = formula_val
            else:
                result[name] = formula_val
        return result
