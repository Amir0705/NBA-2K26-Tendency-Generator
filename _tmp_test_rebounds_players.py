from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Rudy Gobert",
    "Nikola Jokic",
    "Anthony Davis",
    "Evan Mobley",
    "Ivica Zubac",
    "Domantas Sabonis",
    "Giannis Antetokounmpo",
    "Bam Adebayo",
    "Jaren Jackson Jr.",
    "Draymond Green",
    "Josh Hart",
    "Luka Doncic",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<24} {'ORB':>4} {'DRB':>4}   {'oreb36':>7} {'dreb36':>7} {'usage':>6} {'h':>3} {'pos':>3}")
print("-" * 86)
for name in PLAYERS:
    m = client.search_player(name)
    if not m:
        print(f"{name:<24} not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    print(
        f"{name:<24} {a.get('offensive_rebound',0):>4} {a.get('defensive_rebound',0):>4}   "
        f"{float(f.get('oreb_per36',0)):7.2f} {float(f.get('dreb_per36',0)):7.2f} "
        f"{float(f.get('usage_rate',0)):6.1f} {float(f.get('height_inches',0)):3.0f} {str(f.get('position','?')):>3}"
    )
