"""Test new formula calibration across all training players."""
import socket
socket.setdefaulttimeout(15)

import json
import sys
from src.pipeline import TendencyPipeline
from src.attributes.calculator import AttributeCalculator

r = json.load(open("models/attributes/training_report.json"))

# Get all player names
all_attrs = list(r.keys())
players = sorted(set(
    player for attr in all_attrs
    for player in r[attr].get("per_player", {})
))

p = TendencyPipeline()
calc = AttributeCalculator()

# Collect results per attribute
attr_old_errs: dict[str, list] = {a: [] for a in all_attrs}
attr_new_errs: dict[str, list] = {a: [] for a in all_attrs}

success = 0
fail = 0

for name in players:
    try:
        result = p.generate(name, season="2024-25")
        features = result.get("features", {})
        tendencies = result.get("tendencies", {})
        if not features:
            print(f"  SKIP {name}: no features", file=sys.stderr)
            fail += 1
            continue

        new_attrs = calc.calculate(features, tendencies)
        
        for attr in all_attrs:
            pp = r[attr].get("per_player", {})
            if name not in pp:
                continue
            td = pp[name]
            old_f = td["formula"]
            real = td["real_2k"]
            new_f = new_attrs.get(attr)
            if new_f is None:
                continue
            
            attr_old_errs[attr].append(abs(old_f - real))
            attr_new_errs[attr].append(abs(new_f - real))
        
        success += 1
        print(f"  OK {name}", file=sys.stderr)
    except Exception as e:
        fail += 1
        print(f"  FAIL {name}: {e}", file=sys.stderr)

print(f"\nProcessed {success}/{len(players)} players (failed: {fail})")
print(f"\n{'Attribute':<25} {'Old MAE':>8} {'New MAE':>8} {'Delta':>7} {'Result':>8}")
print("-" * 60)

total_old = 0
total_new = 0
n_better = 0
n_worse = 0
n_same = 0

for attr in sorted(all_attrs):
    old_e = attr_old_errs[attr]
    new_e = attr_new_errs[attr]
    if not old_e:
        continue
    old_mae = sum(old_e) / len(old_e)
    new_mae = sum(new_e) / len(new_e)
    delta = new_mae - old_mae
    total_old += old_mae
    total_new += new_mae
    
    if delta < -0.5:
        result = "BETTER"
        n_better += 1
    elif delta > 0.5:
        result = "WORSE"
        n_worse += 1
    else:
        result = ""
        n_same += 1
    
    print(f"  {attr:<23} {old_mae:>8.1f} {new_mae:>8.1f} {delta:>+7.1f} {result:>8}")

n_attrs = len([a for a in all_attrs if attr_old_errs[a]])
print(f"\nOverall avg MAE: {total_old/n_attrs:.1f} -> {total_new/n_attrs:.1f}")
print(f"Better: {n_better}  Same: {n_same}  Worse: {n_worse}")
