"""Cross-tendency consistency guardrails."""
from __future__ import annotations

from typing import Any


class Guardrails:
    """Runs consistency checks on generated tendency dicts."""

    def check(self, tendencies: dict[str, float]) -> list[dict[str, Any]]:
        """
        Run all consistency rules.

        Returns list of violations; each entry:
        {rule, tendency, value, expected, action_taken}

        Auto-corrects values in-place and records corrections.
        """
        violations: list[dict[str, Any]] = []

        def _fix(name: str, new_val: float, rule: str, expected: str) -> None:
            old = tendencies.get(name)
            tendencies[name] = new_val
            violations.append(
                {
                    "rule": rule,
                    "tendency": name,
                    "value": old,
                    "expected": expected,
                    "action_taken": f"corrected to {new_val:.1f}",
                }
            )

        # 1. If Shot Three > 40, Spot Up Three >= Shot Three * 0.3
        shot_three = tendencies.get("shot_three", 0)
        spot_up_three = tendencies.get("spot_up_shot_three", 0)
        if shot_three > 40 and spot_up_three < shot_three * 0.3:
            _fix(
                "spot_up_shot_three",
                shot_three * 0.3,
                "shot_three > 40 → spot_up_shot_three >= shot_three * 0.3",
                f">= {shot_three * 0.3:.1f}",
            )

        # 2. Spot Up Mid within 15 of Shot Mid
        shot_mid = tendencies.get("shot_mid_range", 0)
        spot_up_mid = tendencies.get("spot_up_shot_mid_range", 0)
        if abs(spot_up_mid - shot_mid) > 15:
            target = max(0.0, shot_mid - 15)
            _fix(
                "spot_up_shot_mid_range",
                target,
                "spot_up_shot_mid_range within 15 of shot_mid_range",
                f"within 15 of {shot_mid:.1f}",
            )

        # 3. Off Screen Mid within 15 of Shot Mid
        off_screen_mid = tendencies.get("off_screen_shot_mid_range", 0)
        if abs(off_screen_mid - shot_mid) > 15:
            target = max(0.0, shot_mid - 15)
            _fix(
                "off_screen_shot_mid_range",
                target,
                "off_screen_shot_mid_range within 15 of shot_mid_range",
                f"within 15 of {shot_mid:.1f}",
            )

        # 4. If Post Up < 10, post hooks must be 0
        post_up = tendencies.get("post_up", 0)
        if post_up < 10:
            for hook in ("post_hook_left", "post_hook_right"):
                if tendencies.get(hook, 0) > 0:
                    _fix(hook, 0.0, "post_up < 10 → post hooks = 0", "= 0")

        # 5a. No Setup Dribble absolute cap 35
        no_setup = tendencies.get("no_setup_dribble", 0)
        if no_setup > 35:
            _fix("no_setup_dribble", 35.0, "no_setup_dribble absolute cap 35", "<= 35")

        # 5b. Roll vs Pop: avoid 0/100 extremes (5–95)
        roll_pop = tendencies.get("roll_vs_pop", 50)
        if roll_pop < 5:
            _fix("roll_vs_pop", 5.0, "roll_vs_pop avoid extreme low", ">= 5")
        elif roll_pop > 95:
            _fix("roll_vs_pop", 95.0, "roll_vs_pop avoid extreme high", "<= 95")

        # 5c. Spot-Up Three should not exceed Shot Three + 10
        spot_up_three = tendencies.get("spot_up_shot_three", 0)
        shot_three = tendencies.get("shot_three", 0)
        if spot_up_three > shot_three + 10:
            _fix(
                "spot_up_shot_three",
                shot_three + 10.0,
                "spot_up_shot_three <= shot_three + 10",
                f"<= {shot_three + 10}",
            )

        # 5d. Off-Screen Three <= Shot Three
        off_screen_three = tendencies.get("off_screen_shot_three", 0)
        if off_screen_three > shot_three:
            _fix(
                "off_screen_shot_three",
                float(shot_three),
                "off_screen_shot_three <= shot_three",
                f"<= {shot_three}",
            )

        # 5e. Contested Jumper Three <= Shot Three
        contested_three = tendencies.get("contested_jumper_three", 0)
        if contested_three > shot_three:
            _fix(
                "contested_jumper_three",
                float(shot_three),
                "contested_jumper_three <= shot_three",
                f"<= {shot_three}",
            )

        # 5f. Idle ↔ discipline guardrail:
        # triple_threat_idle + triple_threat_pump_fake should not both be very high —
        # idling while also being pump-fake disciplined is contradictory
        triple_idle = tendencies.get("triple_threat_idle", 0)
        triple_pump = tendencies.get("triple_threat_pump_fake", 0)
        if triple_idle + triple_pump > 75:
            _fix(
                "triple_threat_idle",
                max(0.0, 75.0 - triple_pump),
                "triple_threat_idle + triple_threat_pump_fake <= 75",
                f"<= {max(0.0, 75.0 - triple_pump):.1f}",
            )

        # 6g. Idle ↔ play_discipline guardrail:
        # High idle + high play_discipline is pathological in 2K — the AI holds the
        # ball AND rigidly follows the play call, running down the shot clock.
        # Rule: triple_threat_idle + play_discipline <= 75.
        # Re-read idle in case 6f already reduced it.
        triple_idle_cur = tendencies.get("triple_threat_idle", 0)
        play_discipline_val = tendencies.get("play_discipline", 0)
        if triple_idle_cur + play_discipline_val > 75:
            _fix(
                "triple_threat_idle",
                max(0.0, 75.0 - play_discipline_val),
                "triple_threat_idle + play_discipline <= 75",
                f"<= {max(0.0, 75.0 - play_discipline_val):.1f}",
            )

        # 7. Sub-zone families should sum close to their parent tendency.
        # This preserves formula intent: sub-zones are a distribution of the
        # parent, not an independent 0-100 family.
        sub_zone_families = [
            (
                ["shot_close_left", "shot_close_middle", "shot_close_right"],
                "shot_close sub-zones",
                "shot_close",
            ),
        ]
        for family_keys, family_name, parent_key in sub_zone_families:
            parent_val = max(0.0, float(tendencies.get(parent_key, 0.0)))
            total = sum(max(0.0, tendencies.get(k, 0.0)) for k in family_keys)
            lower = parent_val * 0.8
            upper = parent_val * 1.2

            if not (lower <= total <= upper):
                n = len(family_keys)
                if parent_val <= 0:
                    for k in family_keys:
                        tendencies[k] = 0.0
                elif total <= 0:
                    even = parent_val / n
                    for k in family_keys:
                        tendencies[k] = even
                else:
                    for k in family_keys:
                        tendencies[k] = max(0.0, tendencies.get(k, 0.0)) / total * parent_val
                violations.append(
                    {
                        "rule": f"{family_name} sum close to {parent_key}",
                        "tendency": family_name,
                        "value": total,
                        "expected": f"within ±20% of {parent_key} ({parent_val:.1f})",
                        "action_taken": f"normalized to {parent_key} ({parent_val:.1f})",
                    }
                )

        # 7. At least 30 of 99 tendencies should be > 0
        nonzero = sum(1 for v in tendencies.values() if v > 0)
        if nonzero < 30:
            violations.append(
                {
                    "rule": "at least 30 tendencies > 0",
                    "tendency": "_all_",
                    "value": nonzero,
                    "expected": ">= 30",
                    "action_taken": "warning only",
                }
            )

        # 8. No more than 20% of tendencies at their cap
        # (Cap values not known here — this is a soft check on 100%)
        at_100 = sum(1 for v in tendencies.values() if v >= 100)
        total_t = len(tendencies)
        if total_t > 0 and at_100 / total_t > 0.20:
            violations.append(
                {
                    "rule": "no more than 20% of tendencies at cap",
                    "tendency": "_all_",
                    "value": at_100 / total_t,
                    "expected": "<= 0.20",
                    "action_taken": "warning only",
                }
            )

        return violations


# ---------------------------------------------------------------------------
# Legacy function stubs kept for backward compat
# ---------------------------------------------------------------------------


def validate_player_input(payload: dict[str, Any]) -> bool:
    """
    Validate an incoming player generation request payload.

    Expected keys: player_id (int), season (str), position (str).

    Returns True on success; raises ValueError with details on failure.
    """
    missing = [k for k in ("player_id", "season", "position") if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if not isinstance(payload["player_id"], int):
        raise ValueError("player_id must be an integer")
    if not isinstance(payload["season"], str):
        raise ValueError("season must be a string")
    if payload["position"] not in ("PG", "SG", "SF", "PF", "C"):
        raise ValueError(f"Invalid position: {payload['position']!r}")
    return True


def sanitise_tendencies(tendencies: dict[str, Any]) -> dict[str, int]:
    """
    Coerce and sanitise raw tendency values.

    - Converts string numerics to int.
    - Clamps values to [0, 100].
    - Drops unrecognised keys (only str keys kept).
    """
    result: dict[str, int] = {}
    for k, v in tendencies.items():
        if not isinstance(k, str):
            continue
        try:
            int_val = int(float(v))
        except (TypeError, ValueError):
            continue
        result[k] = max(0, min(100, 5 * round(int_val / 5)))
    return result
