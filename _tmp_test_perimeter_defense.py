"""Benchmark: perimeter_defense formula"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
calc   = AttributeCalculator()
client = NBAApiClient(cache_dir=CACHE_DIR)
engine = FeatureEngine(client)

PLAYERS = [
    ("Alex Caruso",       "1629173"),
    ("Derrick White",     "1629684"),
    ("Jrue Holiday",      "203200"),
    ("Luguentz Dort",     "1629631"),
    ("Kawhi Leonard",     "202695"),
    ("Anthony Davis",     "203076"),
    ("Draymond Green",    "203110"),
    ("Luka Doncic",       "1629029"),
    ("Trae Young",        "1629027"),
    ("LaMelo Ball",       "1630163"),
    ("Stephen Curry",     "201939"),
]

print(f"{'Player':<22} {'PD':>4}   stl36  disc  usage   h    w")
print("-" * 65)
for name, pid in PLAYERS:
    try:
        feats = engine.build_multiseasonal_features(pid, s0_season="2025-26")
        attrs = calc.calculate(feats)
        pd_val = attrs.get("perimeter_defense", 0)

        stl36  = feats.get("stl_per36", 0)
        pf36   = feats.get("pf_per36",  0)
        usage  = feats.get("usage_rate", 0)
        h      = feats.get("height_inches", 0)
        w      = feats.get("weight_lbs", 0)
        print(f"{name:<22} {pd_val:>4.0f}  {stl36:5.2f}  {pf36:5.2f}  {usage:5.1f}  {h:3.0f}  {w:3.0f}")
    except Exception as e:
        print(f"{name:<22}  ERR  {e}")
