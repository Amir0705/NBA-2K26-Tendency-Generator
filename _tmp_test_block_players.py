from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Victor Wembanyama",
    "Anthony Davis",
    "Rudy Gobert",
    "Evan Mobley",
    "Myles Turner",
    "Brook Lopez",
    "Jaren Jackson Jr.",
    "Bam Adebayo",
    "Draymond Green",
    "Giannis Antetokounmpo",
    "Nikola Jokic",
    "Derrick White",
    "Alex Caruso",
    "Luka Doncic",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<24} {'BLK':>4}   {'blk36':>6} {'blk/g':>6} {'pf36':>6} {'dreb36':>7} {'h':>3} {'pos':>3}")
print("-" * 92)
for name in PLAYERS:
    matches = client.search_player(name)
    if not matches:
        print(f"{name:<24} not found")
        continue
    pid = int(matches[0]["player_id"])
    feats = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    attrs = calc.calculate(feats, tendencies={})
    print(
        f"{name:<24} {attrs.get('block', 0):>4}   "
        f"{float(feats.get('blk_per36', 0)):6.2f} "
        f"{float(feats.get('blk_per_game', 0)):6.2f} "
        f"{float(feats.get('pf_per36', 0)):6.2f} "
        f"{float(feats.get('dreb_per36', 0)):7.2f} "
        f"{float(feats.get('height_inches', 0)):3.0f} "
        f"{str(feats.get('position', '?')):>3}"
    )
