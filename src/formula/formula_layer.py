"""Formula layer stub — Phase 3 implementation."""
from __future__ import annotations

from typing import Any


class FormulaLayer:
    """Deterministic rule-based tendency calculator."""

    def __init__(self, registry_path: str, scales_csv_path: str) -> None:
        """
        Load registry and scale definitions.

        Parameters
        ----------
        registry_path:  Path to data/tendency_registry.json.
        scales_csv_path: Path to NBA_2K_Tendency_Scales.csv.
        """
        raise NotImplementedError("Phase 3 implementation")

    def compute(
        self, features: dict[str, float], position: str
    ) -> dict[str, int]:
        """
        Apply deterministic formulas to produce raw tendency values.

        Parameters
        ----------
        features:  Feature dict from FeatureEngine.build_features.
        position:  Player position ('PG', 'SG', 'SF', 'PF', 'C').

        Returns
        -------
        Dict of canonical_name → integer tendency value (pre-cap).
        """
        raise NotImplementedError("Phase 3 implementation")

    def apply_locked_rules(
        self, tendencies: dict[str, int]
    ) -> dict[str, int]:
        """
        Enforce inter-tendency locked rules from the scales CSV notes
        (e.g. Spot-Up Mid ≤ Shot Mid).
        """
        raise NotImplementedError("Phase 3 implementation")
