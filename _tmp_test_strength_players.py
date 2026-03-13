from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Rudy Gobert",
    "Nikola Jokic",
    "Joel Embiid",
    "Anthony Davis",
    "Giannis Antetokounmpo",
    "LeBron James",
    "Draymond Green",
    "Kawhi Leonard",
    "Jayson Tatum",
    "Luka Doncic",
    "Jimmy Butler",
    "Bam Adebayo",
    "Jrue Holiday",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<22} {'STR':>4} {'ID':>3} {'ORB':>3} {'DRB':>3}   {'h':>3} {'w':>3} {'oreb36':>7} {'dreb36':>7} {'pos':>3}")
print("-" * 106)
for name in PLAYERS:
    m = client.search_player(name)
    if not m:
        print(f"{name:<22} not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    print(
        f"{name:<22} {a.get('strength',0):>4} {a.get('interior_defense',0):>3} {a.get('offensive_rebound',0):>3} {a.get('defensive_rebound',0):>3}   "
        f"{float(f.get('height_inches',0)):3.0f} {float(f.get('weight_lbs',0)):3.0f} {float(f.get('oreb_per36',0)):7.2f} {float(f.get('dreb_per36',0)):7.2f} {str(f.get('position','?')):>3}"
    )
