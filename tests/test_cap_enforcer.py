"""Tests for src/caps/cap_enforcer.py."""
from __future__ import annotations

import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(REPO, "data", "tendency_registry.json")

from src.caps.cap_enforcer import CapEnforcer


@pytest.fixture(scope="module")
def enforcer() -> CapEnforcer:
    return CapEnforcer(REGISTRY_PATH)


class TestHardClamp:
    def test_value_above_cap_is_reduced(self, enforcer):
        # shot hard cap is 75; passing 90 should return 75
        clamped, delta = enforcer.hard_clamp(90, "shot")
        assert clamped == 75
        assert delta == -15

    def test_value_at_cap_unchanged(self, enforcer):
        clamped, delta = enforcer.hard_clamp(75, "shot")
        assert clamped == 75
        assert delta == 0

    def test_value_below_cap_unchanged(self, enforcer):
        clamped, delta = enforcer.hard_clamp(50, "shot")
        assert clamped == 50
        assert delta == 0

    def test_unknown_tendency_passes_through(self, enforcer):
        clamped, delta = enforcer.hard_clamp(99, "nonexistent_tendency")
        assert clamped == 99
        assert delta == 0

    def test_sub_zone_cap(self, enforcer):
        # shot_close_left hard cap is 50
        clamped, delta = enforcer.hard_clamp(60, "shot_close_left")
        assert clamped == 50
        assert delta == -10

    def test_no_driving_dribble_move_cap_85(self, enforcer):
        clamped, delta = enforcer.hard_clamp(100, "no_driving_dribble_move")
        assert clamped == 85
        assert delta == -15

    def test_driving_layup_cap_85(self, enforcer):
        clamped, delta = enforcer.hard_clamp(100, "driving_layup")
        assert clamped == 85
        assert delta == -15


class TestEnforceAll:
    def test_returns_tuple_of_two(self, enforcer):
        result = enforcer.enforce_all({"shot": 50})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_clamped_dict_has_all_keys(self, enforcer):
        inputs = {"shot": 80, "touches": 70, "shot_close": 65}
        clamped, _ = enforcer.enforce_all(inputs)
        assert set(clamped.keys()) == set(inputs.keys())

    def test_over_cap_values_are_clamped(self, enforcer):
        clamped, audit = enforcer.enforce_all({"shot": 90, "touches": 70})
        assert clamped["shot"] == 75      # hard cap 75
        assert clamped["touches"] == 65   # hard cap 65

    def test_within_cap_values_unchanged(self, enforcer):
        clamped, _ = enforcer.enforce_all({"shot": 50, "touches": 40})
        assert clamped["shot"] == 50
        assert clamped["touches"] == 40

    def test_audit_log_length_matches_input(self, enforcer):
        inputs = {"shot": 90, "touches": 30, "drive": 70}
        _, audit = enforcer.enforce_all(inputs)
        assert len(audit) == 3

    def test_audit_log_entry_fields(self, enforcer):
        _, audit = enforcer.enforce_all({"shot": 90})
        entry = audit[0]
        for field in ("tendency", "pre_cap", "post_cap", "delta", "cap_applied", "reason"):
            assert field in entry, f"Audit entry missing field {field!r}"

    def test_audit_reason_within_cap(self, enforcer):
        _, audit = enforcer.enforce_all({"shot": 50})
        assert audit[0]["reason"] == "within_cap"

    def test_audit_reason_hard_cap_applied(self, enforcer):
        _, audit = enforcer.enforce_all({"shot": 90})
        assert "hard_cap" in audit[0]["reason"]

    def test_sub_zone_cap_enforced_in_bulk(self, enforcer):
        inputs = {
            "shot_close_left": 60,
            "shot_close_middle": 55,
            "shot_three_center": 65,
        }
        clamped, _ = enforcer.enforce_all(inputs)
        assert clamped["shot_close_left"] == 50
        assert clamped["shot_close_middle"] == 50
        assert clamped["shot_three_center"] == 50
