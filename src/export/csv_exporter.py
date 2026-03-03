"""CSV exporter stub — Phase 6 implementation."""
from __future__ import annotations

from typing import Any


def export_player_csv(
    tendencies_dict: dict[str, int],
    registry: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Write tendency values to a CSV file.

    Parameters
    ----------
    tendencies_dict: canonical_name → integer value.
    registry:        Ordered registry entries.
    output_path:     Destination file path.
    """
    raise NotImplementedError("Phase 6 implementation")


def export_bulk_csv(
    players: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Write tendency values for multiple players to a single CSV file.

    Parameters
    ----------
    players:     List of {name, tendencies} dicts.
    registry:    Ordered registry entries.
    output_path: Destination file path.
    """
    raise NotImplementedError("Phase 6 implementation")
