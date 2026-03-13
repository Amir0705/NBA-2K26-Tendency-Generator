from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

players = [
    "Rudy Gobert",
    "Evan Mobley",
    "Jaren Jackson Jr.",
    "Anthony Davis",
    "Bam Adebayo",
    "Brook Lopez",
    "Myles Turner",
    "Draymond Green",
    "Nikola Jokic",
    "Luka Doncic",
    "Stephen Curry",
    "Trae Young",
    "LaMelo Ball",
    "Tyrese Haliburton",
]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

for name in players:
    matches = client.search_player(name)
    if not matches:
        print(f"{name:20} not found")
        continue
    pid = int(matches[0]["player_id"])
    feats = engine.build_multiseasonal_features(pid, s0_season="2025-26")
    attrs = calc.calculate(feats, tendencies={})
    print(
        f"{name:20} int_def={attrs['interior_defense']:2} "
        f"blk36={float(feats.get('blk_per36', 0)):.2f} "
        f"dreb36={float(feats.get('dreb_per36', 0)):.2f} "
        f"pf36={float(feats.get('pf_per36', 0)):.2f} "
        f"h={float(feats.get('height_inches', 0)):.0f} "
        f"w={float(feats.get('weight_lbs', 0)):.0f}"
    )
