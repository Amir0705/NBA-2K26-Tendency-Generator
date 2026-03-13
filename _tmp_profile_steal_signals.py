from src.ingest.nba_api_client import NBAApiClient
from src.features.feature_engine import FeatureEngine

PLAYERS = [
    "Alex Caruso",
    "Dyson Daniels",
    "Matisse Thybulle",
    "Herb Jones",
    "Kris Dunn",
    "Dejounte Murray",
    "Shai Gilgeous-Alexander",
    "Derrick White",
    "Jrue Holiday",
    "Luguentz Dort",
    "Kawhi Leonard",
    "Draymond Green",
    "Luka Doncic",
    "Stephen Curry",
    "Trae Young",
    "Nikola Jokic",
    "Rudy Gobert",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)

print(f"{'Player':<26} {'stl':>5} {'stl36':>6} {'pf36':>6} {'usage':>6} {'min':>5} {'gp':>4} {'h':>3} {'w':>3} {'pos':>3}")
print('-' * 88)
for name in PLAYERS:
    matches = client.search_player(name)
    if not matches:
        print(f"{name:<26} not found")
        continue
    pid = int(matches[0]['player_id'])
    feats = engine.build_multiseasonal_features(pid, s0_season='2025-26')
    print(
        f"{name:<26} "
        f"{float(feats.get('stl_per_game',0)):5.2f} "
        f"{float(feats.get('stl_per36',0)):6.2f} "
        f"{float(feats.get('pf_per36',0)):6.2f} "
        f"{float(feats.get('usage_rate',0)):6.1f} "
        f"{float(feats.get('min_per_game',0)):5.1f} "
        f"{float(feats.get('gp',0)):4.0f} "
        f"{float(feats.get('height_inches',0)):3.0f} "
        f"{float(feats.get('weight_lbs',0)):3.0f} "
        f"{str(feats.get('position','?')):>3}"
    )
