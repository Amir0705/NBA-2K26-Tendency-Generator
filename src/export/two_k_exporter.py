"""2K-style full player JSON exporter using a fixed template structure."""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

from src.attributes.calculator import ATTRIBUTE_LABELS

_TEMPLATE_PATH = os.path.join("data", "export_templates", "player_template.json")

_ATTRIBUTE_ALIASES = {
    "mid_range_shot": "Mid Range",
    "three_point_shot": "Three Point",
    "ball_handle": "Ball Control",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _load_template() -> dict[str, Any]:
    with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("2K export template must be a JSON object")
    return data


def _set_names(payload: dict[str, Any], player_name: str, player_id: int | None, team: str | None) -> None:
    full_name = (player_name or "").strip() or "Unknown Player"
    parts = [p for p in full_name.split(" ") if p]
    first = parts[0] if parts else full_name
    last = parts[-1] if len(parts) > 1 else ""

    payload["fullName"] = full_name
    payload["firstName"] = first
    payload["lastName"] = last
    if player_id is not None:
        payload["id"] = int(player_id)
    if team:
        payload["team"] = f"Current {team}"

    categories = payload.get("categories", {})
    if isinstance(categories, dict):
        vitals = categories.get("Vitals")
        if isinstance(vitals, dict):
            vitals["First Name"] = first
            if last:
                vitals["Last Name"] = last
            if team:
                vitals["Current Team"] = team


def _replace_tendencies(
    payload: dict[str, Any],
    tendencies: dict[str, int],
    registry: list[dict[str, Any]],
) -> None:
    label_to_value: dict[str, int] = {}
    for row in registry:
        canonical = str(row.get("canonical_name", ""))
        label = str(row.get("primjer_label", ""))
        if not canonical or not label:
            continue
        value = tendencies.get(canonical)
        if value is None:
            continue
        label_to_value[_normalize_key(label)] = int(max(0, min(100, value)))

    categories = payload.get("categories", {})
    if not isinstance(categories, dict):
        return
    target = categories.get("Tendencies")
    if not isinstance(target, dict):
        return

    for key in list(target.keys()):
        norm = _normalize_key(key)
        if norm in label_to_value:
            target[key] = label_to_value[norm]


def _replace_attributes(payload: dict[str, Any], attributes: dict[str, int]) -> None:
    by_name: dict[str, int] = {}
    for canonical, value in attributes.items():
        label = _ATTRIBUTE_ALIASES.get(canonical, ATTRIBUTE_LABELS.get(canonical, canonical))
        by_name[_normalize_key(label)] = int(max(0, min(99, value)))

    categories = payload.get("categories", {})
    if not isinstance(categories, dict):
        return
    target = categories.get("Attributes")
    if not isinstance(target, dict):
        return

    for key in list(target.keys()):
        norm = _normalize_key(key)
        if norm in by_name:
            target[key] = by_name[norm]


def export_player_2k_json(
    *,
    player_name: str,
    player_id: int | None,
    team: str | None,
    tendencies: dict[str, int],
    attributes: dict[str, int],
    registry: list[dict[str, Any]],
) -> str:
    """Return a full 2K-style JSON string with generated tendencies/attributes."""
    payload = copy.deepcopy(_load_template())
    _set_names(payload, player_name=player_name, player_id=player_id, team=team)
    _replace_tendencies(payload, tendencies=tendencies, registry=registry)
    _replace_attributes(payload, attributes=attributes)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
