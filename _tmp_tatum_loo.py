"""Show Tatum's LOO (leave-one-out) predictions — honest out-of-sample estimate."""
import json

with open("models/attributes/training_report.json") as f:
    report = json.load(f)

print(f"{'Attribute':<25} {'Formula':>8} {'Real 2K':>8} {'LOO Corr':>9} {'LOO Result':>11} {'LOO Err':>8}")
print("-" * 75)
total_formula_err = 0
total_loo_err = 0
n = 0
for attr in sorted(report.keys()):
    info = report[attr]
    pp = info.get("per_player", {})
    t = pp.get("Jayson Tatum")
    if t:
        loo_corrected = t["formula"] + t["loo_predicted_correction"]
        formula_err = abs(t["real_2k"] - t["formula"])
        loo_err = abs(t["real_2k"] - loo_corrected)
        total_formula_err += formula_err
        total_loo_err += loo_err
        n += 1
        print(f"  {attr:<23} {t['formula']:>8} {t['real_2k']:>8} {t['loo_predicted_correction']:>+9.1f} {loo_corrected:>11.0f} {t['real_2k']-loo_corrected:>+8.0f}")
print("-" * 75)
print(f"  Formula avg abs error: {total_formula_err/n:.1f}")
print(f"  LOO avg abs error:     {total_loo_err/n:.1f}")
print(f"  (LOO = ML prediction when Tatum is EXCLUDED from training)")
