"""Community feedback store stub — Phase 8 implementation."""
from __future__ import annotations

from typing import Any


class FeedbackStore:
    """Persists and retrieves community tendency-correction feedback."""

    def __init__(self, store_path: str) -> None:
        """
        Initialise the feedback store.

        Parameters
        ----------
        store_path: Path to the JSON file (or database URI) used for storage.
        """
        raise NotImplementedError("Phase 8 implementation")

    def submit(
        self,
        player_id: int,
        tendency_name: str,
        suggested_value: int,
        reviewer: str | None = None,
        notes: str = "",
    ) -> str:
        """
        Record a new feedback entry.

        Returns the generated feedback ID.
        """
        raise NotImplementedError("Phase 8 implementation")

    def get_for_player(self, player_id: int) -> list[dict[str, Any]]:
        """Return all feedback entries for *player_id*."""
        raise NotImplementedError("Phase 8 implementation")

    def aggregate(
        self, player_id: int, tendency_name: str
    ) -> dict[str, Any]:
        """
        Aggregate feedback for a specific tendency.

        Returns dict with: mean_value, vote_count, suggested_values.
        """
        raise NotImplementedError("Phase 8 implementation")
