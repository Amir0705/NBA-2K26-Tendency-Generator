"""Quick test: compare formula-only vs ML-corrected attributes for Tatum."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.formula.formula_layer import FormulaLayer
from src.ingest.nba_api_client import NBAApiClient
from src.ml.attribute_predictor import AttributePredictor

CACHE_DIR = os.path.join("data", "cache")

client = NBAApiClient(cache_dir=CACHE_DIR)
feature_engine = FeatureEngine(client)
formula_layer = FormulaLayer()
calculator = AttributeCalculator()
predictor = AttributePredictor(model_dir="models/attributes/")

# Load 2K ground truth for comparison
with open("data/2k_exports/jayson_tatum.txt", encoding="utf-8") as f:
    tatum_2k = json.load(f)
real_attrs = tatum_2k["categories"]["Attributes"]

# Lookup Tatum
results = client.search_player("Jayson Tatum")
player_id = results[0]["player_id"]
print(f"Jayson Tatum (ID: {player_id})")

# Build features
features = feature_engine.build_features(player_id, season="2024-25")

# Generate tendencies
tendencies = formula_layer.generate(features)

# Formula-only attributes
formula_attrs = calculator.calculate(features, tendencies)

# ML-corrected attributes
corrected_attrs = predictor.apply_corrections(formula_attrs, features)

# Show comparison
print(f"\n{'Attribute':<25} {'2K Real':>8} {'Formula':>8} {'ML Corr':>8} {'F Δ':>6} {'ML Δ':>6}")
print("-" * 70)

attr_map = {
    "Driving Layup": "driving_layup",
    "Standing Dunk": "standing_dunk",
    "Driving Dunk": "driving_dunk",
    "Close Shot": "close_shot",
    "Mid Range": "mid_range_shot",
    "Three Point": "three_point_shot",
    "Free Throw": "free_throw",
    "Post Hook": "post_hook",
    "Post Fade": "post_fade",
    "Post Control": "post_control",
    "Draw Foul": "draw_foul",
    "Shot IQ": "shot_iq",
    "Ball Control": "ball_handle",
    "Speed With Ball": "speed_with_ball",
    "Hands": "hands",
    "Passing Accuracy": "pass_accuracy",
    "Passing IQ": "pass_iq",
    "Passing Vision": "pass_vision",
    "Offensive Consistency": "offensive_consistency",
    "Interior Defense": "interior_defense",
    "Perimeter Defense": "perimeter_defense",
    "Steal": "steal",
    "Block": "block",
    "Offensive Rebound": "offensive_rebound",
    "Defensive Rebound": "defensive_rebound",
    "Help Defense IQ": "help_defense_iq",
    "Passing Perception": "pass_perception",
    "Defensive Consistency": "defensive_consistency",
    "Speed": "speed",
    "Agility": "agility",
    "Strength": "strength",
    "Vertical": "vertical",
    "Stamina": "stamina",
    "Intangibles": "intangibles",
    "Hustle": "hustle",
    "Potential": "potential",
}

total_formula_err = 0
total_ml_err = 0
count = 0

for display_name, canonical in attr_map.items():
    real = real_attrs.get(display_name, 0)
    formula = formula_attrs.get(canonical, 0)
    ml = corrected_attrs.get(canonical, 0)
    f_delta = formula - real
    ml_delta = ml - real
    marker = " ✓" if abs(ml_delta) < abs(f_delta) else (" ✗" if abs(ml_delta) > abs(f_delta) else "")
    print(f"{display_name:<25} {real:>8} {formula:>8} {ml:>8} {f_delta:>+6} {ml_delta:>+6}{marker}")
    total_formula_err += abs(f_delta)
    total_ml_err += abs(ml_delta)
    count += 1

print("-" * 70)
print(f"{'Total Abs Error':<25} {'':>8} {total_formula_err:>8} {total_ml_err:>8}")
print(f"{'Mean Abs Error':<25} {'':>8} {total_formula_err/count:>8.1f} {total_ml_err/count:>8.1f}")

print(f"\nML models used: {predictor.available_attributes}")
