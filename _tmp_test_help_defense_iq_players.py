from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Draymond Green",
    "Rudy Gobert",
    "Anthony Davis",
    "Evan Mobley",
    "Jaren Jackson Jr.",
    "Bam Adebayo",
    "Giannis Antetokounmpo",
    "Nikola Jokic",
    "Derrick White",
    "Jrue Holiday",
    "Alex Caruso",
    "Luguentz Dort",
    "Kawhi Leonard",
    "Luka Doncic",
    "Stephen Curry",
    "Trae Young",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<22} {'HDIQ':>5} {'PD':>3} {'ID':>3} {'STL':>3} {'BLK':>3}   {'stl36':>5} {'blk36':>5} {'pf36':>5} {'usage':>5} {'pos':>3}")
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
        f"{name:<22} {a.get('help_defense_iq', 0):>5} "
        f"{a.get('perimeter_defense', 0):>3} {a.get('interior_defense', 0):>3} {a.get('steal', 0):>3} {a.get('block', 0):>3}   "
        f"{float(f.get('stl_per36', 0)):5.2f} {float(f.get('blk_per36', 0)):5.2f} {float(f.get('pf_per36', 0)):5.2f} {float(f.get('usage_rate', 0)):5.1f} {str(f.get('position', '?')):>3}"
    )
