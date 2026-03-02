"""Schema validator for exported tendency JSON."""
from __future__ import annotations

import json
import os
from typing import Any

# Locate primjer.txt relative to this file's package tree
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PRIMJER_PATH = os.path.join(_REPO_ROOT, "primjer.txt")


def _load_reference() -> dict[str, Any]:
    with open(_PRIMJER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def validate_export(json_output: str) -> bool:
    """
    Validate a JSON export string against the primjer.txt reference.

    Checks performed:
    - Valid JSON with top-level 'tendencies' key.
    - Key count matches reference (99).
    - All values are integers in [0, 100].
    - Key order matches reference order.

    Returns True on success; raises ValueError with details on failure.
    """
    try:
        out_data = json.loads(json_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Output is not valid JSON: {exc}") from exc

    if "tendencies" not in out_data:
        raise ValueError("Output JSON missing top-level 'tendencies' key.")

    reference = _load_reference()
    ref_keys = list(reference["tendencies"].keys())
    out_tendencies: dict[str, Any] = out_data["tendencies"]
    out_keys = list(out_tendencies.keys())

    if len(out_keys) != len(ref_keys):
        raise ValueError(
            f"Key count mismatch: expected {len(ref_keys)}, got {len(out_keys)}."
        )

    misordered = [
        (i, out_keys[i], ref_keys[i])
        for i in range(len(ref_keys))
        if out_keys[i] != ref_keys[i]
    ]
    if misordered:
        details = "; ".join(
            f"pos {i}: got '{o}' expected '{r}'"
            for i, o, r in misordered[:5]
        )
        raise ValueError(f"Key order mismatch: {details}")

    out_of_range = [
        k
        for k, v in out_tendencies.items()
        if not isinstance(v.get("value"), int) or not (0 <= v["value"] <= 100)
    ]
    if out_of_range:
        raise ValueError(
            f"Values out of range [0–100] for keys: {out_of_range[:10]}"
        )

    return True
