"""Estimate new formula MAE by applying calibration to old formula values.

Back-computes raw score from old output, applies tapered calibration,
then recomputes the output. This captures the calibration+tapering effect
without needing to re-run the pipeline.

NOTE: potential and steal have formula rewrites that this can't simulate.
"""
import json
import math

_RAW_CALIBRATION = {
    "hustle": 22,
    "mid_range_shot": 20,
    "post_hook": 20,
    "interior_defense": 19,
    "close_shot": 18,
    "post_fade": 17,
    "pass_iq": 16,
    "post_control": 15,
    "hands": 14,
    "ball_handle": 14,
    "vertical": 9,
    "stamina": 9,
    "pass_accuracy": 8,
    "driving_dunk": 7,
    "three_point_shot": 7,
    "block": 6,
    "potential": 5,
    "defensive_rebound": 5,
    "free_throw": 4,
    "driving_layup": 4,
    "strength": 3,
    "steal": -12,
    "offensive_rebound": -12,
    "intangibles": -4,
    "shot_iq": -4,
    "speed_with_ball": -2,
}


def output_to_raw(output: int) -> float:
    """Back-compute raw score from 25-99 output."""
    frac = (output - 25) / 74.0
    frac = max(0.001, min(0.999, frac))
    raw = (frac ** (1.0 / 0.75)) * 100
    return raw


def raw_to_output(raw: float) -> int:
    frac = max(0.0, min(1.0, raw / 100.0))
    frac = frac ** 0.75
    scaled = 25 + frac * 74
    return max(25, min(99, round(scaled)))


def apply_tapered_cal(raw: float, cal: float) -> float:
    if cal > 0:
        cal *= max(0.0, 1.0 - max(0.0, raw - 55) / 45)
    elif cal < 0:
        cal *= max(0.0, 1.0 - max(0.0, 35 - raw) / 35)
    return raw + cal


with open("models/attributes/training_report.json") as f:
    report = json.load(f)

print(f"{'Attribute':<25} {'Old MAE':>8} {'New MAE':>8} {'Delta':>7} {'Old Bias':>9} {'New Bias':>9} {'Result':>8}")
print("-" * 80)

total_old = 0
total_new = 0
n_better = 0
n_worse = 0
n_same = 0
n_attrs = 0

for attr in sorted(report.keys()):
    info = report[attr]
    pp = info.get("per_player", {})
    cal = _RAW_CALIBRATION.get(attr, 0.0)

    old_errs = []
    new_errs = []
    old_biases = []
    new_biases = []

    for player, vals in pp.items():
        old_f = vals["formula"]
        real = vals["real_2k"]

        # Skip potential and steal (formula rewrites can't be simulated)
        if attr in ("potential", "steal"):
            old_errs.append(abs(old_f - real))
            new_errs.append(abs(old_f - real))  # placeholder
            old_biases.append(real - old_f)
            new_biases.append(real - old_f)
            continue

        # Back-compute raw, apply calibration, recompute
        raw = output_to_raw(old_f)
        new_raw = apply_tapered_cal(raw, cal)
        new_f = raw_to_output(new_raw)

        old_errs.append(abs(old_f - real))
        new_errs.append(abs(new_f - real))
        old_biases.append(real - old_f)
        new_biases.append(real - new_f)

    old_mae = sum(old_errs) / len(old_errs)
    new_mae = sum(new_errs) / len(new_errs)
    old_bias = sum(old_biases) / len(old_biases)
    new_bias = sum(new_biases) / len(new_biases)
    delta = new_mae - old_mae
    total_old += old_mae
    total_new += new_mae
    n_attrs += 1

    if delta < -0.5:
        result = "BETTER"
        n_better += 1
    elif delta > 0.5:
        result = "WORSE"
        n_worse += 1
    else:
        result = ""
        n_same += 1

    print(f"  {attr:<23} {old_mae:>8.1f} {new_mae:>8.1f} {delta:>+7.1f} {old_bias:>+9.1f} {new_bias:>+9.1f} {result:>8}")

# For potential/steal, note they're placeholders
print(f"\nOverall avg MAE: {total_old/n_attrs:.1f} -> {total_new/n_attrs:.1f}")
print(f"Better: {n_better}  Same: {n_same}  Worse: {n_worse}")
print("\n* potential and steal show OLD values (formula rewrites can't be simulated here)")
print("  potential: old bias +18.0 -> expected ~+3 after rewrite")
print("  steal: old bias -8.4 -> expected ~-3 after rewrite + calibration")
