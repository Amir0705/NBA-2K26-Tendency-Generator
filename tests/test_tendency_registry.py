"""Tests for data/tendency_registry.json integrity."""
from __future__ import annotations

import json
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")
PRIMJER_PATH = os.path.join(REPO, "primjer.txt")

SUB_ZONE_LABELS = {
    # label → parent canonical_name, parent hard_cap
    "Shot Close Left": ("shot_close", 60),
    "Shot Close Middle": ("shot_close", 60),
    "Shot Close Right": ("shot_close", 60),
    "Shot Mid Left": ("shot_mid_range", 55),
    "Shot Mid Left-Center": ("shot_mid_range", 55),
    "Shot Mid Center": ("shot_mid_range", 55),
    "Shot Mid Right-Center": ("shot_mid_range", 55),
    "Shot Mid Right": ("shot_mid_range", 55),
    "Shot Three Left": ("shot_three", 60),
    "Shot Three Left-Center": ("shot_three", 60),
    "Shot Three Center": ("shot_three", 60),
    "Shot Three Right-Center": ("shot_three", 60),
    "Shot Three Right": ("shot_three", 60),
}


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def primjer() -> dict:
    with open(PRIMJER_PATH, encoding="utf-8") as fh:
        return json.load(fh)["tendencies"]


def test_registry_is_list(registry):
    assert isinstance(registry, list), "Registry must be a list (ordered)"


def test_registry_length_matches_primjer(registry, primjer):
    assert len(registry) == len(primjer), (
        f"Registry has {len(registry)} entries; primjer.txt has {len(primjer)}"
    )


def test_all_primjer_keys_present(registry, primjer):
    registry_keys = {e["primjer_key"] for e in registry}
    for key in primjer:
        assert key in registry_keys, f"primjer key {key!r} missing from registry"


def test_offsets_match_primjer(registry, primjer):
    for entry in registry:
        pkey = entry["primjer_key"]
        assert entry["offset"] == primjer[pkey]["offset"], (
            f"Offset mismatch for {pkey}: "
            f"registry={entry['offset']} primjer={primjer[pkey]['offset']}"
        )
        assert entry["bit_offset"] == primjer[pkey]["bit_offset"], (
            f"bit_offset mismatch for {pkey}: "
            f"registry={entry['bit_offset']} primjer={primjer[pkey]['bit_offset']}"
        )


def test_order_field_is_sequential(registry):
    for i, entry in enumerate(registry, start=1):
        assert entry["order"] == i, (
            f"Entry {entry['canonical_name']} has order={entry['order']}, expected {i}"
        )


def test_sub_zone_caps_are_parent_minus_10(registry):
    cap_by_canonical = {e["canonical_name"]: e["hard_cap"] for e in registry}
    for entry in registry:
        label = entry["primjer_label"]
        if label in SUB_ZONE_LABELS:
            parent_canon, parent_cap = SUB_ZONE_LABELS[label]
            expected_cap = parent_cap - 10
            assert entry["hard_cap"] == expected_cap, (
                f"{label}: expected hard_cap={expected_cap}, got {entry['hard_cap']}"
            )
            assert entry["parent_tendency"] == parent_canon, (
                f"{label}: expected parent={parent_canon!r}, got {entry['parent_tendency']!r}"
            )


def test_sub_zone_is_sub_zone_true(registry):
    for entry in registry:
        if entry["primjer_label"] in SUB_ZONE_LABELS:
            assert entry["is_sub_zone"] is True, (
                f"{entry['primjer_label']} should have is_sub_zone=True"
            )


def test_non_sub_zone_is_sub_zone_false(registry):
    for entry in registry:
        if entry["primjer_label"] not in SUB_ZONE_LABELS:
            assert entry["is_sub_zone"] is False, (
                f"{entry['primjer_label']} should have is_sub_zone=False"
            )


def test_canonical_names_unique(registry):
    seen = {}
    for entry in registry:
        cn = entry["canonical_name"]
        assert cn not in seen, (
            f"Duplicate canonical_name {cn!r}: entries {seen[cn]} and {entry['order']}"
        )
        seen[cn] = entry["order"]


def test_hard_caps_in_valid_range(registry):
    for entry in registry:
        cap = entry["hard_cap"]
        assert 0 < cap <= 100, (
            f"{entry['canonical_name']}: hard_cap={cap} out of range"
        )
