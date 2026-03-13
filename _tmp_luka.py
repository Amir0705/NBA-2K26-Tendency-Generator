"""Generate Luka Doncic and evaluate."""
import socket
socket.setdefaulttimeout(15)

from src.pipeline import TendencyPipeline
from src.attributes.calculator import AttributeCalculator

p = TendencyPipeline()
result = p.generate('Luka Doncic', season='2024-25')
features = result.get('features', {})
tendencies = result.get('tendencies', {})

calc = AttributeCalculator()
attrs = calc.calculate(features, tendencies)
pipeline_attrs = result.get('attributes', {})

pos = features.get('position', '?')
ht = features.get('height_inches', '?')
wt = features.get('weight_lbs', '?')
age = features.get('age', '?')
ppg = features.get('pts_per_game', 0)
apg = features.get('ast_per_game', 0)
rpg = features.get('reb_per_game', 0)
usg = features.get('usage_rate', 0)
fg = features.get('fg_pct', 0)
fg3 = features.get('fg3_pct', 0)
ft = features.get('ft_pct', 0)
ts = features.get('ts_pct', 0)

print(f"Player: Luka Doncic")
print(f"Position: {pos}  Height: {ht}in  Weight: {wt}lbs  Age: {age}")
print(f"PPG: {ppg:.1f}  APG: {apg:.1f}  RPG: {rpg:.1f}  USG: {usg:.1f}")
print(f"FG%: {fg:.3f}  3P%: {fg3:.3f}  FT%: {ft:.3f}  TS%: {ts:.3f}")
print()
print(f"{'Attribute':<25} {'Formula':>8} {'Pipeline':>9}")
print("-" * 45)

for a in sorted(attrs.keys()):
    f_val = attrs[a]
    p_val = pipeline_attrs.get(a, '?')
    print(f"  {a:<23} {f_val:>8} {p_val:>9}")

finishing = [attrs[x] for x in ['driving_layup','driving_dunk','close_shot','standing_dunk']]
shooting = [attrs[x] for x in ['mid_range_shot','three_point_shot','free_throw']]
playmaking = [attrs[x] for x in ['ball_handle','pass_accuracy','pass_iq','pass_vision']]
defense = [attrs[x] for x in ['interior_defense','perimeter_defense','steal','block']]
physical = [attrs[x] for x in ['speed','agility','strength','vertical','stamina']]
all_vals = list(attrs.values())

print(f"\nAvg all attrs: {sum(all_vals)/len(all_vals):.1f}")
print(f"Finishing avg: {sum(finishing)/len(finishing):.1f}")
print(f"Shooting avg:  {sum(shooting)/len(shooting):.1f}")
print(f"Playmaking avg:{sum(playmaking)/len(playmaking):.1f}")
print(f"Defense avg:   {sum(defense)/len(defense):.1f}")
print(f"Physical avg:  {sum(physical)/len(physical):.1f}")
