"""Excel exporter stub — Phase 6 implementation."""
from __future__ import annotations

from typing import Any


def export_player_excel(
    tendencies_dict: dict[str, int],
    registry: list[dict[str, Any]],
    output_path: str,
    player_name: str = "Player",
) -> None:
    """
    Write tendency values to a formatted Excel (.xlsx) workbook.

    Parameters
    ----------
    tendencies_dict: canonical_name → integer value.
    registry:        Ordered registry entries.
    output_path:     Destination .xlsx file path.
    player_name:     Used as the worksheet tab name.
    """
    raise NotImplementedError("Phase 6 implementation")


def export_bulk_excel(
    players: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Write tendency values for multiple players to a single Excel workbook,
    one worksheet per player.

    Parameters
    ----------
    players:     List of {name, tendencies} dicts.
    registry:    Ordered registry entries.
    output_path: Destination .xlsx file path.
    """
    raise NotImplementedError("Phase 6 implementation")
