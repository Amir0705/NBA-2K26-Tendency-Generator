"""Input guardrails stub — Phase 7 implementation."""
from __future__ import annotations

from typing import Any


def validate_player_input(payload: dict[str, Any]) -> bool:
    """
    Validate an incoming player generation request payload.

    Expected keys: player_id (int), season (str), position (str).

    Returns True on success; raises ValueError with details on failure.
    """
    raise NotImplementedError("Phase 7 implementation")


def sanitise_tendencies(tendencies: dict[str, Any]) -> dict[str, int]:
    """
    Coerce and sanitise raw tendency values.

    - Converts string numerics to int.
    - Clamps values to [0, 100].
    - Drops unrecognised keys.

    Returns cleaned dict of canonical_name → int.
    """
    raise NotImplementedError("Phase 7 implementation")
