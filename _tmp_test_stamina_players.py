from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    "Mikal Bridges",
    "Josh Hart",
    "Anthony Edwards",
    "Shai Gilgeous-Alexander",
    "Luka Doncic",
    "Nikola Jokic",
    "Rudy Gobert",
    "Evan Mobley",
    "LeBron James",
    "Stephen Curry",
    "Joel Embiid",
    "Kawhi Leonard",
    "Kristaps Porzingis",
    "Giannis Antetokounmpo",
    "Zion Williamson",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

print(f"{'Player':<24} {'STA':>4} {'MPG':>5} {'GP':>3} {'USG':>5} {'TRN':>5} {'CUT':>5} {'ROLL':>5} {'REB36':>6} {'AGE':>4} {'POS':>3}")
print("-" * 120)
for name in PLAYERS:
    match = client.search_player(name)
    if not match:
        print(f"{name:<24} not found")
        continue

    pid = int(match[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})

    reb36 = float(f.get("oreb_per36", 0.0) or 0.0) + float(f.get("dreb_per36", 0.0) or 0.0)
    print(
        f"{name:<24} {a.get('stamina', 0):>4} {float(f.get('min_per_game', 0.0)):5.2f} {int(float(f.get('gp', 0.0))):>3} "
        f"{float(f.get('usage_rate', 0.0)):5.1f} {float(f.get('transition_possessions', 0.0)):5.2f} "
        f"{float(f.get('cuts', 0.0)):5.2f} {float(f.get('pick_and_roll_rollman_possessions', 0.0)):5.2f} "
        f"{reb36:6.2f} {float(f.get('age', 0.0)):4.1f} {str(f.get('position', '?')):>3}"
    )
