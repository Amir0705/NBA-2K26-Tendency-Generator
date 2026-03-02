"""JSON exporter: reconstructs a primjer.txt-compatible JSON string."""
from __future__ import annotations

import json
from typing import Any


def export_player_json(
    tendencies_dict: dict[str, int], registry: list[dict[str, Any]]
) -> str:
    """
    Build an ordered JSON string matching the primjer.txt structure.

    Parameters
    ----------
    tendencies_dict:
        Mapping of canonical_name → integer value.
    registry:
        Ordered list of registry entries (from tendency_registry.json).

    Returns
    -------
    JSON string with 2-space indent; keys follow primjer.txt order.

    Each tendency entry contains:
      value, label, offset, type ("bitfield"), bit_offset,
      bit_length (7), length (null).
    """
    ordered: dict[str, Any] = {}
    for entry in sorted(registry, key=lambda e: e["order"]):
        canon = entry["canonical_name"]
        primjer_key = entry["primjer_key"]
        value = tendencies_dict.get(canon, 0)
        ordered[primjer_key] = {
            "value": value,
            "label": entry["primjer_label"],
            "offset": entry["offset"],
            "type": "bitfield",
            "bit_offset": entry["bit_offset"],
            "bit_length": 7,
            "length": None,
        }

    return json.dumps({"tendencies": ordered}, indent=2, ensure_ascii=False)


def validate_against_primjer(output: str, reference_path: str) -> bool:
    """
    Compare *output* JSON structure against the primjer.txt reference.

    Checks: key count, key order, value ranges (0–100).

    Returns True on success; raises ValueError with details on failure.
    """
    with open(reference_path, encoding="utf-8") as fh:
        reference = json.load(fh)

    ref_tendencies: dict[str, Any] = reference["tendencies"]
    ref_keys = list(ref_tendencies.keys())

    try:
        out_data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Output is not valid JSON: {exc}") from exc

    if "tendencies" not in out_data:
        raise ValueError("Output JSON missing top-level 'tendencies' key.")

    out_tendencies: dict[str, Any] = out_data["tendencies"]
    out_keys = list(out_tendencies.keys())

    if len(out_keys) != len(ref_keys):
        raise ValueError(
            f"Key count mismatch: output has {len(out_keys)}, "
            f"reference has {len(ref_keys)}."
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
