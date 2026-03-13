from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Alex Caruso",
    "Derrick White",
    "Jrue Holiday",
    "Luguentz Dort",
    "Kawhi Leonard",
    "Draymond Green",
    "Rudy Gobert",
    "Anthony Davis",
    "Evan Mobley",
    "Nikola Jokic",
    "Shai Gilgeous-Alexander",
    "Luka Doncic",
    "LaMelo Ball",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<24} {'PP':>4} {'HD':>3} {'PD':>3} {'DC':>3} {'STL':>3}   {'stl36':>5} {'blk36':>5} {'pf36':>5} {'usage':>5} {'pos':>3}")
print("-" * 110)
for name in PLAYERS:
    m = client.search_player(name)
    if not m:
        print(f"{name:<24} not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    print(
        f"{name:<24} {a.get('pass_perception', 0):>4} "
        f"{a.get('help_defense_iq', 0):>3} {a.get('perimeter_defense', 0):>3} {a.get('defensive_consistency', 0):>3} {a.get('steal', 0):>3}   "
        f"{float(f.get('stl_per36', 0)):5.2f} {float(f.get('blk_per36', 0)):5.2f} {float(f.get('pf_per36', 0)):5.2f} {float(f.get('usage_rate', 0)):5.1f} {str(f.get('position', '?')):>3}"
    )
