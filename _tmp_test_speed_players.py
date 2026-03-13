from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Ja Morant",
    "Tyrese Maxey",
    "Anthony Edwards",
    "Shai Gilgeous-Alexander",
    "Stephen Curry",
    "Luka Doncic",
    "Jayson Tatum",
    "LeBron James",
    "Giannis Antetokounmpo",
    "Evan Mobley",
    "Anthony Davis",
    "Rudy Gobert",
    "Nikola Jokic",
    "Brook Lopez",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<24} {'SPD':>4} {'SWB':>4}   {'trans':>6} {'rimP':>5} {'usage':>5} {'h':>3} {'w':>3} {'pos':>3}")
print("-" * 96)
for name in PLAYERS:
    m = client.search_player(name)
    if not m:
        print(f"{name:<24} not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    rim_p = float(f.get("zone_fga_rate_ra", 0)) + float(f.get("zone_fga_rate_paint", 0))
    print(
        f"{name:<24} {a.get('speed',0):>4} {a.get('speed_with_ball',0):>4}   "
        f"{float(f.get('transition_possessions',0)):6.2f} {rim_p:5.2f} {float(f.get('usage_rate',0)):5.1f} "
        f"{float(f.get('height_inches',0)):3.0f} {float(f.get('weight_lbs',0)):3.0f} {str(f.get('position','?')):>3}"
    )
