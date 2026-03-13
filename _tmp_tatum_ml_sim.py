"""Simulate ML corrections on Tatum WITHOUT enabling ML in the pipeline."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline
from src.ml.attribute_predictor import AttributePredictor

p = TendencyPipeline()
result = p.generate('Jayson Tatum', season='2024-25')
formula_attrs = result['attributes']
features = result['features']

# Manually load ML predictor and apply corrections
ml = AttributePredictor(model_dir='models/attributes/')
print(f"ML models loaded: {ml.has_models()}")
print(f"Beneficial attributes: {ml.available_attributes}")
print(f"Total beneficial models: {len(ml.available_attributes)}\n")

corrections = ml.predict_corrections(features)
ml_attrs = ml.apply_corrections(formula_attrs, features)

print(f"{'Attribute':<28} {'Formula':>8} {'ML':>8} {'Correction':>11} {'Change':>7}")
print("-" * 65)
for attr in sorted(formula_attrs.keys()):
    f_val = formula_attrs[attr]
    m_val = ml_attrs[attr]
    corr = corrections.get(attr, 0.0)
    diff = m_val - f_val
    flag = "" if diff == 0 else " *"
    print(f"  {attr:<26} {f_val:>8} {m_val:>8} {corr:>+11.1f} {diff:>+7}{flag}")

changed = sum(1 for a in formula_attrs if ml_attrs[a] != formula_attrs[a])
print(f"\nAttributes changed by ML: {changed}/{len(formula_attrs)}")
