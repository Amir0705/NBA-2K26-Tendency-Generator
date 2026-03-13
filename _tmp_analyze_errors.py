"""Analyze formula errors across all 38 players to find systematic biases."""
import json

with open("models/attributes/training_report.json") as f:
    report = json.load(f)

print(f"{'Attribute':<25} {'Avg Err':>8} {'Avg Bias':>9} {'Direction':>10} {'Worst Player':>25} {'Worst Err':>10}")
print("-" * 95)

# Collect per-attribute stats
attr_stats = []
for attr in sorted(report.keys()):
    info = report[attr]
    pp = info.get("per_player", {})
    
    errors = []
    biases = []  # positive = formula too low, negative = formula too high
    worst_player = ""
    worst_err = 0
    
    for player, vals in pp.items():
        residual = vals["real_2k"] - vals["formula"]
        errors.append(abs(residual))
        biases.append(residual)
        if abs(residual) > abs(worst_err):
            worst_err = residual
            worst_player = player
    
    avg_err = sum(errors) / len(errors) if errors else 0
    avg_bias = sum(biases) / len(biases) if biases else 0
    direction = "TOO LOW" if avg_bias > 2 else ("TOO HIGH" if avg_bias < -2 else "~OK")
    
    attr_stats.append((attr, avg_err, avg_bias, direction, worst_player, worst_err))
    print(f"  {attr:<23} {avg_err:>8.1f} {avg_bias:>+9.1f} {direction:>10} {worst_player:>25} {worst_err:>+10}")

print()
print("=" * 95)
print("SYSTEMATIC BIASES (formula consistently wrong in one direction):")
print("=" * 95)

too_low = [(a, e, b) for a, e, b, d, _, _ in attr_stats if d == "TOO LOW"]
too_high = [(a, e, b) for a, e, b, d, _, _ in attr_stats if d == "TOO HIGH"]

print(f"\nFormula TOO LOW ({len(too_low)} attrs) — needs boost:")
for attr, err, bias in sorted(too_low, key=lambda x: x[2], reverse=True):
    print(f"  {attr:<25} avg bias: {bias:>+6.1f}  avg error: {err:.1f}")

print(f"\nFormula TOO HIGH ({len(too_high)} attrs) — needs reduction:")
for attr, err, bias in sorted(too_high, key=lambda x: x[2]):
    print(f"  {attr:<25} avg bias: {bias:>+6.1f}  avg error: {err:.1f}")

# Now show distribution of errors for the worst attributes
print()
print("=" * 95)
print("DISTRIBUTION for worst attributes:")
print("=" * 95)

for attr in ["hustle", "potential", "steal", "interior_defense", "post_hook", 
             "close_shot", "mid_range_shot", "offensive_consistency", "draw_foul"]:
    if attr not in report:
        continue
    pp = report[attr]["per_player"]
    residuals = [(p, v["real_2k"] - v["formula"], v["formula"], v["real_2k"]) 
                 for p, v in pp.items()]
    residuals.sort(key=lambda x: x[1])
    print(f"\n{attr} (formula_MAE={report[attr]['formula_mae']}):")
    for player, res, formula, real in residuals:
        bar = "+" * max(0, res) + "-" * max(0, -res)
        print(f"  {player:<30} formula={formula:>3} real={real:>3} diff={res:>+4} {bar}")
