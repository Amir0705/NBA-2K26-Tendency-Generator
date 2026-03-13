"""Cap enforcer: clamps tendency values to their hard caps."""
from __future__ import annotations

import json
from typing import Any


class CapEnforcer:
    """Enforces hard caps defined in tendency_registry.json."""

    def __init__(self, registry_path: str) -> None:
        """Load cap rules from tendency_registry.json."""
        with open(registry_path, encoding="utf-8") as fh:
            self._registry: list[dict[str, Any]] = json.load(fh)
        self._caps: dict[str, int] = {
            entry["canonical_name"]: entry["hard_cap"]
            for entry in self._registry
            if entry.get("hard_cap") is not None
        }

    def hard_clamp(
        self, value: int, tendency_name: str, position: str | None = None
    ) -> tuple[int, int]:
        """
        Clamp *value* to the hard cap for *tendency_name*.

        Parameters
        ----------
        value:          raw tendency value (0–100).
        tendency_name:  canonical_name of the tendency.
        position:       (reserved for future position-based overrides).

        Returns
        -------
        (clamped_value, delta)  — delta = clamped_value - original value (≤ 0).
        """
        cap = self._caps.get(tendency_name)
        if cap is None:
            return (value, 0)
        clamped = min(value, cap)
        return (clamped, clamped - value)

    def enforce_all(
        self, tendencies_dict: dict[str, int]
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """
        Apply hard caps to every tendency in *tendencies_dict*.

        Parameters
        ----------
        tendencies_dict: mapping of canonical_name → raw value.

        Returns
        -------
        (clamped_dict, audit_log)

        Each audit_log entry:
        {tendency, pre_cap, post_cap, delta, cap_applied, reason}
        """
        clamped: dict[str, int] = {}
        audit: list[dict[str, Any]] = []

        for name, raw in tendencies_dict.items():
            clamped_val, delta = self.hard_clamp(raw, name)
            cap = self._caps.get(name)
            clamped[name] = clamped_val
            audit.append(
                {
                    "tendency": name,
                    "pre_cap": raw,
                    "post_cap": clamped_val,
                    "delta": delta,
                    "cap_applied": cap,
                    "reason": f"hard_cap={cap}" if delta < 0 else "within_cap",
                }
            )

        return clamped, audit
