"""Shot-zone feature builder stub — Phase 3 implementation."""
from __future__ import annotations

from typing import Any


class ShotZoneBuilder:
    """Computes shot-zone sub-tendency features from shot-chart data."""

    def __init__(self) -> None:
        """Initialise zone definitions."""
        raise NotImplementedError("Phase 3 implementation")

    def compute_zones(
        self, shot_chart: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Aggregate shot-chart rows into zone-level tendency scores.

        Parameters
        ----------
        shot_chart: Raw shot rows from NBAApiClient.get_shot_chart.

        Returns
        -------
        Dict mapping sub-zone canonical_name → computed score [0, 100].
        """
        raise NotImplementedError("Phase 3 implementation")

    def distribute_from_parent(
        self, parent_value: int, zone_fractions: dict[str, float]
    ) -> dict[str, int]:
        """
        Distribute a parent tendency value across sub-zones using
        observed shot fractions from *zone_fractions*.
        """
        raise NotImplementedError("Phase 3 implementation")
