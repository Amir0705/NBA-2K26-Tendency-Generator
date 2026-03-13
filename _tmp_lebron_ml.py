"""Compare LeBron: formula-only vs formula+ML (simulated)."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline
from src.ml.attribute_predictor import AttributePredictor

p = TendencyPipeline()
result = p.generate('LeBron James', season='2024-25')
formula_attrs = result['attributes']
features = result['features']

ml = AttributePredictor(model_dir='models/attributes/')
corrections = ml.predict_corrections(features)
ml_attrs = ml.apply_corrections(formula_attrs, features)

print(f"LeBron James — pos={features.get('position')}, age={features.get('age')}")
print(f"  pts={features.get('pts_per_game',0):.1f}  ast={features.get('ast_per_game',0):.1f}  reb={features.get('reb_per_game',0):.1f}  usage={features.get('usage_rate',0):.1f}")
print(f"  stl_per36={features.get('stl_per36',0):.2f}  blk_per36={features.get('blk_per36',0):.2f}\n")

print(f"{'Attribute':<28} {'Formula':>8} {'ML':>8} {'Correction':>11} {'Clamped':>8}")
print("-" * 68)
for attr in sorted(formula_attrs.keys()):
    f_val = formula_attrs[attr]
    m_val = ml_attrs[attr]
    corr = corrections.get(attr, 0.0)
    diff = m_val - f_val
    flag = "" if diff == 0 else " *"
    print(f"  {attr:<26} {f_val:>8} {m_val:>8} {corr:>+11.1f} {diff:>+8}{flag}")

changed = sum(1 for a in formula_attrs if ml_attrs[a] != formula_attrs[a])
print(f"\nAttributes changed by ML: {changed}/{len(formula_attrs)}")
