"""Tests for src/ingest/csv_loaders.py."""
from __future__ import annotations

import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALES_PATH = os.path.join(REPO, "NBA_2K_Tendency_Scales.csv")
ATD_PATH = os.path.join(
    REPO, "ATD Committee Roster Edits  - Tendency Test Edit - PlainSightToSee.csv"
)

from src.ingest.csv_loaders import load_atd_csv, load_scales_csv


# ---------------------------------------------------------------------------
# load_scales_csv
# ---------------------------------------------------------------------------

class TestLoadScalesCsv:
    def test_returns_dict(self):
        result = load_scales_csv(SCALES_PATH)
        assert isinstance(result, dict)

    def test_non_empty(self):
        result = load_scales_csv(SCALES_PATH)
        assert len(result) > 0

    def test_known_key_shot(self):
        result = load_scales_csv(SCALES_PATH)
        assert "Shot" in result

    def test_known_key_touch(self):
        result = load_scales_csv(SCALES_PATH)
        assert "Touch" in result

    def test_entry_has_required_fields(self):
        result = load_scales_csv(SCALES_PATH)
        entry = result["Shot"]
        for field in ("order", "name", "definition", "scale_bands", "typical_range", "hard_cap", "notes"):
            assert field in entry, f"Field {field!r} missing from 'Shot' entry"

    def test_shot_hard_cap_is_75(self):
        result = load_scales_csv(SCALES_PATH)
        assert result["Shot"]["hard_cap"] == 75

    def test_all_entries_have_int_order(self):
        result = load_scales_csv(SCALES_PATH)
        for name, entry in result.items():
            assert isinstance(entry["order"], int), f"order is not int for {name!r}"

    def test_84_tendencies_loaded(self):
        result = load_scales_csv(SCALES_PATH)
        # The Scales CSV has 84 tendency rows (excludes sub-zones and Driving Layup
        # which are additional entries only present in primjer.txt)
        assert len(result) == 84

    def test_raises_on_missing_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_scales_csv("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# load_atd_csv
# ---------------------------------------------------------------------------

class TestLoadAtdCsv:
    def test_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        result = load_atd_csv(ATD_PATH)
        assert isinstance(result, pd.DataFrame)

    def test_has_player_name_column(self):
        pytest.importorskip("pandas")
        result = load_atd_csv(ATD_PATH)
        assert "player_name" in result.columns

    def test_has_tendency_columns(self):
        pytest.importorskip("pandas")
        result = load_atd_csv(ATD_PATH)
        # Should have columns for at least a few tendencies
        cols_lower = [c.lower() for c in result.columns]
        assert any("shot" in c for c in cols_lower), (
            "Expected at least one 'shot'-related column in ATD DataFrame"
        )

    def test_non_empty_rows(self):
        pytest.importorskip("pandas")
        result = load_atd_csv(ATD_PATH)
        assert len(result) > 0

    def test_raises_on_missing_file(self):
        pytest.importorskip("pandas")
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            load_atd_csv("/nonexistent/atd.csv")
