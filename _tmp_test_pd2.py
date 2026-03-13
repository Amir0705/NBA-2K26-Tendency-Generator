from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Alex Caruso",
    "Derrick White",
    "Jrue Holiday",
    "Luguentz Dort",
    "Kawhi Leonard",
    "Anthony Davis",
    "Draymond Green",
    "Luka Doncic",
    "Trae Young",
    "LaMelo Ball",
    "Stephen Curry",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc   = AttributeCalculator()

print(f"{'Player':<22} {'PD':>4}   stl36  pf36  usage   h    w")
print("-" * 65)
for name in PLAYERS:
    matches = client.search_player(name)
    if not matches:
        print(f"{name:<22}  not found")
        continue
    pid  = int(matches[0]["player_id"])
    feats = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    attrs = calc.calculate(feats, tendencies={})
    pd_val = attrs.get("perimeter_defense", 0)
    stl36  = float(feats.get("stl_per36", 0))
    pf36   = float(feats.get("pf_per36",  0))
    usage  = float(feats.get("usage_rate", 0))
    h      = float(feats.get("height_inches", 0))
    w      = float(feats.get("weight_lbs", 0))
    print(f"{name:<22} {pd_val:>4.0f}  {stl36:5.2f}  {pf36:5.2f}  {usage:5.1f}  {h:3.0f}  {w:3.0f}")
