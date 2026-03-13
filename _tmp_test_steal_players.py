from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Alex Caruso",
    "Kawhi Leonard",
    "Derrick White",
    "Jrue Holiday",
    "Luguentz Dort",
    "Draymond Green",
    "Anthony Davis",
    "Luka Doncic",
    "LaMelo Ball",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<22} {'STL':>4}   {'stl36':>5} {'stl/g':>5} {'pf36':>5} {'usage':>5} {'pos':>3}")
print("-" * 72)
for name in PLAYERS:
    matches = client.search_player(name)
    if not matches:
        print(f"{name:<22} not found")
        continue
    pid = int(matches[0]["player_id"])
    feats = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    attrs = calc.calculate(feats, tendencies={})
    print(
        f"{name:<22} {attrs.get('steal', 0):>4}   "
        f"{float(feats.get('stl_per36', 0)):5.2f} "
        f"{float(feats.get('stl_per_game', 0)):5.2f} "
        f"{float(feats.get('pf_per36', 0)):5.2f} "
        f"{float(feats.get('usage_rate', 0)):5.1f} "
        f"{str(feats.get('position', '?')):>3}"
    )
