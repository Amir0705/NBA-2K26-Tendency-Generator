"""
Run attribute generation for 5 diverse players and compare against
known NBA 2K26 approximate ratings for accuracy evaluation.
"""
import json
import sys
from src.pipeline import TendencyPipeline

# 5 diverse players: star guard, star wing, star big, role player guard, role player big
PLAYERS = [
    "Stephen Curry",
    "LeBron James",
    "Nikola Jokic",
    "Anthony Edwards",
    "Giannis Antetokounmpo",
]

# Approximate real 2K26 attributes (based on 2K25 known ratings as reference)
# These are ballpark targets to evaluate if our generator is in the right range
EXPECTED_RANGES = {
    "Stephen Curry": {
        "three_point_shot": (90, 99),
        "mid_range_shot": (85, 95),
        "ball_handle": (88, 96),
        "speed": (72, 82),
        "driving_layup": (78, 88),
        "pass_accuracy": (82, 92),
        "pass_vision": (80, 92),
        "interior_defense": (25, 50),
        "block": (25, 40),
        "standing_dunk": (25, 40),
        "strength": (30, 55),
        "post_hook": (25, 50),
        "free_throw": (88, 99),
        "steal": (65, 82),
    },
    "LeBron James": {
        "driving_layup": (85, 97),
        "driving_dunk": (80, 95),
        "three_point_shot": (72, 85),
        "pass_vision": (82, 96),
        "pass_accuracy": (78, 92),
        "speed": (65, 82),
        "strength": (75, 95),
        "interior_defense": (55, 78),
        "block": (50, 72),
        "steal": (55, 75),
        "ball_handle": (72, 88),
    },
    "Nikola Jokic": {
        "pass_vision": (88, 99),
        "pass_accuracy": (85, 97),
        "post_control": (80, 97),
        "post_hook": (70, 90),
        "three_point_shot": (65, 82),
        "speed": (30, 55),
        "ball_handle": (55, 78),
        "interior_defense": (55, 80),
        "defensive_rebound": (80, 97),
        "strength": (75, 95),
        "block": (35, 60),
        "standing_dunk": (40, 65),
    },
    "Anthony Edwards": {
        "driving_dunk": (85, 97),
        "driving_layup": (80, 95),
        "three_point_shot": (72, 88),
        "mid_range_shot": (72, 88),
        "speed": (80, 95),
        "ball_handle": (75, 90),
        "steal": (55, 75),
        "strength": (55, 75),
        "vertical": (80, 97),
    },
    "Giannis Antetokounmpo": {
        "driving_dunk": (90, 99),
        "driving_layup": (85, 99),
        "standing_dunk": (80, 97),
        "speed": (70, 88),
        "strength": (85, 99),
        "interior_defense": (75, 95),
        "block": (70, 92),
        "three_point_shot": (50, 72),
        "ball_handle": (60, 82),
        "pass_vision": (55, 80),
        "vertical": (78, 97),
        "free_throw": (45, 70),
    },
}


def main():
    pipe = TendencyPipeline()
    results = {}
    failures = {}

    for name in PLAYERS:
        print(f"\n{'='*70}")
        print(f"Generating for: {name}")
        print(f"{'='*70}")
        try:
            result = pipe.generate(name, season="2024-25")
            results[name] = result
            attrs = result.get("attributes", {})
            pos = result.get("position", "?")

            print(f"Position: {pos}")
            print(f"\n{'Attribute':<28} {'Value':>5}  {'Expected':>14}  {'Status':>8}")
            print("-" * 60)

            expected = EXPECTED_RANGES.get(name, {})
            fail_count = 0
            for attr_name in sorted(attrs.keys()):
                val = attrs[attr_name]
                if attr_name in expected:
                    lo, hi = expected[attr_name]
                    if val < lo:
                        status = f"TOO LOW (exp {lo}-{hi})"
                        fail_count += 1
                    elif val > hi:
                        status = f"TOO HIGH (exp {lo}-{hi})"
                        fail_count += 1
                    else:
                        status = "OK"
                    print(f"  {attr_name:<26} {val:>5}  [{lo:>3}-{hi:>3}]        {status}")
                else:
                    print(f"  {attr_name:<26} {val:>5}")

            if fail_count > 0:
                failures[name] = fail_count
                print(f"\n  >>> {fail_count} attribute(s) OUT OF RANGE for {name}")
            else:
                print(f"\n  >>> All checked attributes in range for {name}")

        except Exception as e:
            print(f"ERROR generating for {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_checked = 0
    total_failed = 0
    for name in PLAYERS:
        if name in results:
            expected = EXPECTED_RANGES.get(name, {})
            attrs = results[name].get("attributes", {})
            checked = len(expected)
            failed = failures.get(name, 0)
            total_checked += checked
            total_failed += failed
            status = "PASS" if failed == 0 else "FAIL"
            print(f"  {name:<30} {checked:>3} checked, {failed:>3} failed  [{status}]")
        else:
            print(f"  {name:<30} ERROR - no results")

    print(f"\n  Total: {total_checked} checks, {total_failed} failures")
    accuracy = ((total_checked - total_failed) / total_checked * 100) if total_checked > 0 else 0
    print(f"  Accuracy: {accuracy:.1f}%")

    # Dump detailed results for analysis
    print("\n\nDETAILED ATTRIBUTE VALUES:")
    for name in PLAYERS:
        if name in results:
            attrs = results[name].get("attributes", {})
            print(f"\n{name} ({results[name].get('position', '?')}):")
            for attr_name in [
                "driving_layup", "standing_dunk", "driving_dunk", "close_shot",
                "mid_range_shot", "three_point_shot", "free_throw",
                "post_hook", "post_fade", "post_control",
                "draw_foul", "shot_iq", "ball_handle", "speed_with_ball",
                "hands", "pass_accuracy", "pass_iq", "pass_vision",
                "offensive_consistency",
                "interior_defense", "perimeter_defense", "steal", "block",
                "offensive_rebound", "defensive_rebound",
                "help_defense_iq", "pass_perception", "defensive_consistency",
                "speed", "agility", "strength", "vertical",
                "stamina", "intangibles", "hustle", "overall_durability", "potential",
            ]:
                print(f"  {attr_name:<28} {attrs.get(attr_name, 'N/A'):>5}")


if __name__ == "__main__":
    main()
