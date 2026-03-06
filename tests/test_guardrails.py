"""Tests for src/validation/guardrails.py."""
from __future__ import annotations

import pytest

from src.validation.guardrails import Guardrails, sanitise_tendencies, validate_player_input


def _base_tendencies() -> dict:
    """Return a minimal passing tendency dict (99 entries)."""
    import json
    import os
    registry_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "tendency_registry.json"
    )
    with open(registry_path) as fh:
        registry = json.load(fh)
    t = {e["canonical_name"]: 30 for e in registry}
    # Keep sub-zone families aligned to their parent tendency (30)
    for k in ("shot_close_left", "shot_close_middle", "shot_close_right"):
        t[k] = 10
    for k in ("shot_mid_left", "shot_mid_left_center", "shot_mid_center",
               "shot_mid_right_center", "shot_mid_right"):
        t[k] = 6
    for k in ("shot_three_left", "shot_three_left_center", "shot_three_center",
               "shot_three_right_center", "shot_three_right"):
        t[k] = 6
    return t


class TestGuardrailsCheck:
    @pytest.fixture(scope="class")
    def guardrails(self):
        return Guardrails()

    def test_valid_tendencies_no_violations(self, guardrails):
        t = _base_tendencies()
        t["shot_three"] = 30
        t["spot_up_shot_three"] = 25
        violations = guardrails.check(t)
        # Filter out warnings
        errors = [v for v in violations if "warning" not in v["action_taken"]]
        assert len(errors) == 0

    def test_spot_up_three_corrected(self, guardrails):
        t = _base_tendencies()
        t["shot_three"] = 50
        t["spot_up_shot_three"] = 5  # should be >= 50*0.3 = 15
        violations = guardrails.check(t)
        assert any(v["tendency"] == "spot_up_shot_three" for v in violations)
        assert t["spot_up_shot_three"] >= 50 * 0.3

    def test_post_hooks_corrected_when_post_up_low(self, guardrails):
        t = _base_tendencies()
        t["post_up"] = 5
        t["post_hook_left"] = 20
        t["post_hook_right"] = 20
        violations = guardrails.check(t)
        assert t["post_hook_left"] <= 5
        assert t["post_hook_right"] <= 5

    def test_stepback_three_not_hard_corrected(self, guardrails):
        t = _base_tendencies()
        t["stepback_jumper_three"] = 40
        t["stepback_jumper_mid_range"] = 10
        violations = guardrails.check(t)
        assert not any(v["tendency"] == "stepback_jumper_three" for v in violations)
        assert t["stepback_jumper_three"] == 40

    def test_sub_zone_normalization(self, guardrails):
        t = _base_tendencies()
        # Make sub-zones sum way too high relative to parent (shot_close=30)
        t["shot_close_left"] = 80
        t["shot_close_middle"] = 80
        t["shot_close_right"] = 80
        guardrails.check(t)
        total = t["shot_close_left"] + t["shot_close_middle"] + t["shot_close_right"]
        assert total == pytest.approx(t["shot_close"], abs=1.0)
        assert t["shot_close_left"] <= t["shot_close"]
        assert t["shot_close_middle"] <= t["shot_close"]
        assert t["shot_close_right"] <= t["shot_close"]

    def test_at_least_30_nonzero(self, guardrails):
        t = {k: 0 for k in _base_tendencies()}
        violations = guardrails.check(t)
        assert any("at least 30" in v["rule"] for v in violations)

    def test_violations_have_required_fields(self, guardrails):
        t = _base_tendencies()
        t["shot_three"] = 50
        t["spot_up_shot_three"] = 1
        violations = guardrails.check(t)
        for v in violations:
            for field in ("rule", "tendency", "value", "expected", "action_taken"):
                assert field in v, f"Missing field {field}"


class TestValidatePlayerInput:
    def test_valid_input_returns_true(self):
        assert validate_player_input(
            {"player_id": 2544, "season": "2024-25", "position": "SF"}
        ) is True

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            validate_player_input({"player_id": 2544, "season": "2024-25"})

    def test_invalid_player_id_type_raises(self):
        with pytest.raises(ValueError, match="player_id"):
            validate_player_input(
                {"player_id": "abc", "season": "2024-25", "position": "SG"}
            )

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="position"):
            validate_player_input(
                {"player_id": 1, "season": "2024-25", "position": "XX"}
            )


class TestSanitiseTendencies:
    def test_clamps_above_100(self):
        result = sanitise_tendencies({"shot": 150})
        assert result["shot"] == 100

    def test_clamps_below_0(self):
        result = sanitise_tendencies({"shot": -10})
        assert result["shot"] == 0

    def test_converts_string_float(self):
        result = sanitise_tendencies({"shot": "45.7"})
        assert result["shot"] == 45

    def test_drops_non_string_keys(self):
        result = sanitise_tendencies({1: 50, "shot": 30})
        assert 1 not in result
        assert "shot" in result

    def test_drops_non_numeric_values(self):
        result = sanitise_tendencies({"shot": "abc", "drive": 30})
        assert "shot" not in result
        assert result["drive"] == 30


class TestIdleDisciplineGuardrail:
    """Tests for the idle ↔ discipline guardrail (6f)."""

    @pytest.fixture(scope="class")
    def guardrails(self):
        return Guardrails()

    def test_idle_and_pump_fake_sum_too_high_corrected(self, guardrails):
        """When idle + pump_fake > 75, triple_threat_idle is reduced."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 55
        t["triple_threat_pump_fake"] = 30
        violations = guardrails.check(t)
        assert any(v["tendency"] == "triple_threat_idle" for v in violations)
        assert t["triple_threat_idle"] + t["triple_threat_pump_fake"] <= 75

    def test_idle_and_pump_fake_within_limit_no_violation(self, guardrails):
        """When idle + pump_fake <= 75, no correction occurs."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 30
        t["triple_threat_pump_fake"] = 40
        orig_idle = t["triple_threat_idle"]
        violations = guardrails.check(t)
        # No correction should have been applied to triple_threat_idle for this rule:
        # the sum is 70 which is within the 75 limit, so idle must be unchanged
        assert t["triple_threat_idle"] == orig_idle

    def test_idle_clamped_to_zero_minimum(self, guardrails):
        """triple_threat_idle should not go below 0 after guardrail correction."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 40
        t["triple_threat_pump_fake"] = 75
        guardrails.check(t)
        assert t["triple_threat_idle"] >= 0.0

    def test_violation_has_required_fields(self, guardrails):
        """Guardrail violation must have all required fields."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 60
        t["triple_threat_pump_fake"] = 20
        violations = guardrails.check(t)
        idle_v = [v for v in violations if v["tendency"] == "triple_threat_idle"]
        assert len(idle_v) >= 1
        for v in idle_v:
            for field in ("rule", "tendency", "value", "expected", "action_taken"):
                assert field in v


class TestIdlePlayDisciplineGuardrail:
    """Tests for the idle ↔ play_discipline guardrail (6g)."""

    @pytest.fixture(scope="class")
    def guardrails(self):
        return Guardrails()

    def test_idle_and_play_discipline_sum_too_high_corrected(self, guardrails):
        """When idle + play_discipline > 75, triple_threat_idle is reduced."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 50
        t["triple_threat_pump_fake"] = 20   # keep pump_fake low so 6f doesn't trigger
        t["play_discipline"] = 40
        violations = guardrails.check(t)
        assert any(v["tendency"] == "triple_threat_idle" and "play_discipline" in v["rule"]
                   for v in violations)
        assert t["triple_threat_idle"] + t["play_discipline"] <= 75

    def test_idle_and_play_discipline_within_limit_no_violation(self, guardrails):
        """When idle + play_discipline <= 75, no 6g correction occurs."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 30
        t["triple_threat_pump_fake"] = 20
        t["play_discipline"] = 40
        orig_idle = t["triple_threat_idle"]
        guardrails.check(t)
        # Sum is 70 which is within the 75 limit — idle must be unchanged
        assert t["triple_threat_idle"] == orig_idle

    def test_idle_clamped_to_zero_minimum_by_discipline(self, guardrails):
        """triple_threat_idle should not go below 0 after the discipline guardrail."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 30
        t["triple_threat_pump_fake"] = 10
        t["play_discipline"] = 80
        guardrails.check(t)
        assert t["triple_threat_idle"] >= 0.0

    def test_discipline_violation_has_required_fields(self, guardrails):
        """Guardrail 6g violation must include all required metadata fields."""
        t = _base_tendencies()
        t["triple_threat_idle"] = 55
        t["triple_threat_pump_fake"] = 10
        t["play_discipline"] = 30
        violations = guardrails.check(t)
        disc_v = [v for v in violations
                  if v["tendency"] == "triple_threat_idle" and "play_discipline" in v["rule"]]
        assert len(disc_v) >= 1
        for v in disc_v:
            for field in ("rule", "tendency", "value", "expected", "action_taken"):
                assert field in v


class TestSubZoneParentAwareNormalization:
    """Tests close sub-zone family normalization (mid/three remain chart-driven)."""

    @pytest.fixture(scope="class")
    def guardrails(self):
        return Guardrails()

    def test_close_sub_zones_normalize_to_parent(self, guardrails):
        """Close sub-zones that don't sum to parent should be corrected to parent."""
        t = _base_tendencies()
        t["shot_close"] = 40
        # Sub-zones sum to 90, parent is 40 — well outside tolerance
        t["shot_close_left"] = 30
        t["shot_close_middle"] = 30
        t["shot_close_right"] = 30
        violations = guardrails.check(t)
        total = t["shot_close_left"] + t["shot_close_middle"] + t["shot_close_right"]
        assert abs(total - 40) <= max(5.0, 40 * 0.1), (
            f"Expected sum ≈40 after normalization, got {total}"
        )
        assert any("shot_close" in v["rule"] for v in violations)

    def test_mid_sub_zones_normalize_to_parent(self, guardrails):
        """Mid sub-zones are parent-normalized by guardrails when far off."""
        t = _base_tendencies()
        t["shot_mid_range"] = 25
        # Sub-zones sum to 100 (5 × 20), parent is 25 — outside tolerance
        for k in ("shot_mid_left", "shot_mid_left_center", "shot_mid_center",
                  "shot_mid_right_center", "shot_mid_right"):
            t[k] = 20
        violations = guardrails.check(t)
        total = sum(t[k] for k in ("shot_mid_left", "shot_mid_left_center", "shot_mid_center",
                                    "shot_mid_right_center", "shot_mid_right"))
        assert abs(total - 25) <= max(5.0, 25 * 0.1), (
            f"Expected sum ≈25 after normalization, got {total}"
        )
        assert any("shot_mid" in v.get("rule", "") for v in violations)

    def test_three_sub_zones_normalize_to_parent(self, guardrails):
        """Three sub-zones are parent-normalized by guardrails when far off."""
        t = _base_tendencies()
        t["shot_three"] = 20
        # Sub-zones sum to 100 (5 × 20), parent is 20 — outside tolerance
        for k in ("shot_three_left", "shot_three_left_center", "shot_three_center",
                  "shot_three_right_center", "shot_three_right"):
            t[k] = 20
        violations = guardrails.check(t)
        total = sum(t[k] for k in ("shot_three_left", "shot_three_left_center",
                                    "shot_three_center", "shot_three_right_center",
                                    "shot_three_right"))
        assert abs(total - 20) <= max(5.0, 20 * 0.1), (
            f"Expected sum ≈20 after normalization, got {total}"
        )
        assert any("shot_three" in v.get("rule", "") for v in violations)

    def test_close_sub_zones_already_summing_to_parent_no_violation(self, guardrails):
        """No violation when close sub-zones already sum to parent."""
        t = _base_tendencies()
        t["shot_close"] = 30
        t["shot_close_left"] = 10
        t["shot_close_middle"] = 10
        t["shot_close_right"] = 10
        violations = guardrails.check(t)
        close_violations = [v for v in violations
                            if "shot_close sub-zones" in v.get("tendency", "")]
        assert len(close_violations) == 0
