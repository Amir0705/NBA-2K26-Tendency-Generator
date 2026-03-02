"""Feature engineering pipeline stub — Phase 3 implementation."""
from __future__ import annotations

from typing import Any


class FeatureEngine:
    """Transforms raw NBA stats into model-ready tendency features."""

    def __init__(self, registry_path: str) -> None:
        """
        Initialise engine with the tendency registry.

        Parameters
        ----------
        registry_path: Path to data/tendency_registry.json.
        """
        raise NotImplementedError("Phase 3 implementation")

    def build_features(self, player_stats: dict[str, Any]) -> dict[str, float]:
        """
        Convert raw per-game / advanced stats into tendency feature values.

        Parameters
        ----------
        player_stats: Output from NBAApiClient.get_player_stats.

        Returns
        -------
        Dict mapping canonical tendency name → computed feature value.
        """
        raise NotImplementedError("Phase 3 implementation")

    def normalise(self, features: dict[str, float]) -> dict[str, float]:
        """Normalise raw feature values to the [0, 100] scale."""
        raise NotImplementedError("Phase 3 implementation")
