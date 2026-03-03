"""Confidence scoring for ML residual corrections."""
from __future__ import annotations


_MIN_SAMPLES_HIGH = 50
_MIN_SAMPLES_LOW = 30
_R2_HIGH = 0.3
_R2_LOW = 0.1


class ConfidenceScorer:
    """Assigns confidence scores to ML residual corrections."""

    def __init__(self, training_report: dict | None = None) -> None:
        """
        Initialise scorer.

        Parameters
        ----------
        training_report: Output of TendencyTrainer.train() — dict of
                         {tendency_name: {n_samples, rmse, r2}}.
        """
        self._report: dict = training_report or {}

    def score(self, tendency_name: str, features: dict) -> float:
        """
        Return confidence score (0.0 to 1.0) for ML correction.

        High confidence (> 0.7):
        - Model R² > 0.3 on CV
        - Player has sufficient data (not low_minutes)
        - Training had >= 50 samples

        Low confidence (< 0.3):
        - Model R² < 0.1
        - Player is low-minutes / rookie
        - Training had < 30 samples

        Medium (0.3–0.7): interpolated.
        """
        report = self._report.get(tendency_name, {})
        n_samples = report.get("n_samples", 0)
        r2 = report.get("r2", 0.0)

        # Player-level quality signal
        low_minutes = bool(features.get("low_minutes", False))
        if low_minutes or n_samples < _MIN_SAMPLES_LOW:
            return 0.1

        # Interpolate R² signal
        if r2 >= _R2_HIGH:
            r2_score = 1.0
        elif r2 <= _R2_LOW:
            r2_score = 0.0
        else:
            r2_score = (r2 - _R2_LOW) / (_R2_HIGH - _R2_LOW)

        # Interpolate sample-count signal
        if n_samples >= _MIN_SAMPLES_HIGH:
            sample_score = 1.0
        else:
            sample_score = (n_samples - _MIN_SAMPLES_LOW) / (
                _MIN_SAMPLES_HIGH - _MIN_SAMPLES_LOW
            )

        raw = 0.6 * r2_score + 0.4 * sample_score
        # Map raw 0-1 to [0.1, 0.9] to avoid extremes
        return 0.1 + raw * 0.8

    def get_blend_weight(self, tendency_name: str, features: dict) -> float:
        """
        Return weight for ML correction (0.0 = pure formula, 1.0 = full ML).

        Typically in the 0.0–0.4 range (formula is trusted more).
        """
        confidence = self.score(tendency_name, features)
        # Scale confidence [0.1, 0.9] → [0.0, 0.4]
        return max(0.0, (confidence - 0.1) / 0.8) * 0.4
