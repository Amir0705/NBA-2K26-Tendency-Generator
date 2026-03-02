"""NBA API client stub — Phase 2 implementation."""
from __future__ import annotations

from typing import Any


class NBAApiClient:
    """Fetches live player stats from the nba_api library."""

    def __init__(self, cache_dir: str | None = None) -> None:
        """
        Initialise the client.

        Parameters
        ----------
        cache_dir: Optional directory path for response caching.
        """
        raise NotImplementedError("Phase 2 implementation")

    def get_player_stats(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Retrieve per-game and advanced stats for *player_id*.

        Returns a dict with keys: per_game, advanced, shot_chart.
        """
        raise NotImplementedError("Phase 2 implementation")

    def get_shot_chart(
        self, player_id: int, season: str = "2024-25"
    ) -> list[dict[str, Any]]:
        """Return raw shot-chart rows for the given player and season."""
        raise NotImplementedError("Phase 2 implementation")

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search for players by name; returns list of matching records."""
        raise NotImplementedError("Phase 2 implementation")
