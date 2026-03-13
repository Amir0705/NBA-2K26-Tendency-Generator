from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Ja Morant",
    "Tyrese Maxey",
    "De'Aaron Fox",
    "Shai Gilgeous-Alexander",
    "Stephen Curry",
    "Kyrie Irving",
    "Luka Doncic",
    "Anthony Edwards",
    "Jayson Tatum",
    "Kawhi Leonard",
    "Draymond Green",
    "Giannis Antetokounmpo",
    "Anthony Davis",
    "Evan Mobley",
    "Rudy Gobert",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<22} {'AGI':>4} {'SPD':>4} {'SWB':>4} {'PD':>3}   {'h':>3} {'w':>3} {'usage':>5} {'stl36':>5} {'blk36':>5} {'pos':>3}")
print("-" * 104)
for name in PLAYERS:
    m = client.search_player(name)
    if not m:
        print(f"{name:<22} not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    print(
        f"{name:<22} {a.get('agility',0):>4} {a.get('speed',0):>4} {a.get('speed_with_ball',0):>4} {a.get('perimeter_defense',0):>3}   "
        f"{float(f.get('height_inches',0)):3.0f} {float(f.get('weight_lbs',0)):3.0f} {float(f.get('usage_rate',0)):5.1f} "
        f"{float(f.get('stl_per36',0)):5.2f} {float(f.get('blk_per36',0)):5.2f} {str(f.get('position','?')):>3}"
    )
