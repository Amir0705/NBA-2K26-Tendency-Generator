"""Tests for src/export/json_exporter.py."""
from __future__ import annotations

import json
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")
PRIMJER_PATH = os.path.join(REPO, "primjer.txt")

from src.export.json_exporter import export_player_json, validate_against_primjer


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def primjer_keys() -> list[str]:
    with open(PRIMJER_PATH, encoding="utf-8") as fh:
        return list(json.load(fh)["tendencies"].keys())


@pytest.fixture(scope="module")
def sample_tendencies(registry) -> dict[str, int]:
    """Build a sample dict: every tendency set to its hard_cap value."""
    return {entry["canonical_name"]: entry["hard_cap"] for entry in registry}


class TestExportPlayerJson:
    def test_returns_string(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        assert isinstance(result, str)

    def test_valid_json(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_tendencies_key(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        assert "tendencies" in parsed

    def test_correct_key_count(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        assert len(parsed["tendencies"]) == 99

    def test_key_order_matches_primjer(self, sample_tendencies, registry, primjer_keys):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        output_keys = list(parsed["tendencies"].keys())
        assert output_keys == primjer_keys, "Key order does not match primjer.txt"

    def test_entry_has_required_fields(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        first_entry = next(iter(parsed["tendencies"].values()))
        for field in ("value", "label", "offset", "type", "bit_offset", "bit_length", "length"):
            assert field in first_entry, f"Entry missing field {field!r}"

    def test_type_is_bitfield(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        for entry in parsed["tendencies"].values():
            assert entry["type"] == "bitfield"

    def test_bit_length_is_7(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        for entry in parsed["tendencies"].values():
            assert entry["bit_length"] == 7

    def test_length_is_null(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        parsed = json.loads(result)
        for entry in parsed["tendencies"].values():
            assert entry["length"] is None

    def test_value_comes_from_input(self, registry):
        tendencies = {entry["canonical_name"]: 42 for entry in registry}
        result = export_player_json(tendencies, registry)
        parsed = json.loads(result)
        for entry in parsed["tendencies"].values():
            assert entry["value"] == 42

    def test_missing_canonical_defaults_to_zero(self, registry):
        result = export_player_json({}, registry)
        parsed = json.loads(result)
        for entry in parsed["tendencies"].values():
            assert entry["value"] == 0

    def test_uses_2_space_indent(self, sample_tendencies, registry):
        result = export_player_json(sample_tendencies, registry)
        # 2-space indent: first nested line starts with '  '
        lines = result.split("\n")
        assert any(line.startswith("  ") for line in lines)


class TestValidateAgainstPrimjer:
    def test_valid_output_returns_true(self, sample_tendencies, registry):
        output = export_player_json(sample_tendencies, registry)
        assert validate_against_primjer(output, PRIMJER_PATH) is True

    def test_wrong_key_count_raises(self, registry):
        output = json.dumps({"tendencies": {}})
        with pytest.raises(ValueError, match="count"):
            validate_against_primjer(output, PRIMJER_PATH)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            validate_against_primjer("not json", PRIMJER_PATH)

    def test_missing_tendencies_key_raises(self, registry):
        output = json.dumps({"wrong_key": {}})
        with pytest.raises(ValueError, match="tendencies"):
            validate_against_primjer(output, PRIMJER_PATH)

    def test_out_of_range_value_raises(self, registry):
        with open(PRIMJER_PATH) as fh:
            ref = json.load(fh)
        bad_tendencies = {k: {"value": 200, "label": "", "offset": 0,
                               "type": "bitfield", "bit_offset": 0,
                               "bit_length": 7, "length": None}
                          for k in ref["tendencies"]}
        output = json.dumps({"tendencies": bad_tendencies})
        with pytest.raises(ValueError, match="range"):
            validate_against_primjer(output, PRIMJER_PATH)
