from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Ja Morant",
    "Anthony Edwards",
    "Shai Gilgeous-Alexander",
    "LeBron James",
    "Giannis Antetokounmpo",
    "Zion Williamson",
    "Luka Doncic",
    "Evan Mobley",
    "Anthony Davis",
    "Rudy Gobert",
    "Nikola Jokic",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(
    f"{'Player':<26} {'VERT':>4} {'DD':>3} {'SD':>3} {'SPD':>3} {'AGI':>3} {'BLK':>3} {'ORB':>3}   {'h':>3} {'w':>3} {'blk36':>6} {'oreb36':>7} {'pos':>3}"
)
print("-" * 128)
for name in PLAYERS:
    match = client.search_player(name)
    if not match:
        print(f"{name:<26} not found")
        continue
    pid = int(match[0]["player_id"])
    features = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    attrs = calc.calculate(features, tendencies={})
    print(
        f"{name:<26} {attrs.get('vertical', 0):>4} {attrs.get('driving_dunk', 0):>3} {attrs.get('standing_dunk', 0):>3} "
        f"{attrs.get('speed', 0):>3} {attrs.get('agility', 0):>3} {attrs.get('block', 0):>3} {attrs.get('offensive_rebound', 0):>3}   "
        f"{float(features.get('height_inches', 0)):3.0f} {float(features.get('weight_lbs', 0)):3.0f} "
        f"{float(features.get('blk_per36', 0)):6.2f} {float(features.get('oreb_per36', 0)):7.2f} {str(features.get('position', '?')):>3}"
    )
