"""Quick test: run pipeline on first 5 players to verify everything works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.formula.formula_layer import FormulaLayer
from src.ingest.nba_api_client import NBAApiClient
from src.ml.attribute_trainer import load_2k_exports, _name_from_filename

EXPORT_DIR = "data/2k_exports"
SEASON = "2024-25"

client = NBAApiClient(cache_dir="data/cache")
fe = FeatureEngine(client)
fl = FormulaLayer()
calc = AttributeCalculator()

players_2k = load_2k_exports(EXPORT_DIR)
print(f"Loaded {len(players_2k)} exports")

for p in players_2k[:5]:
    name = p["full_name"]
    print(f"  {name}...", end=" ", flush=True)
    try:
        results = client.search_player(name)
        if not results:
            alt = _name_from_filename(p["filename"])
            results = client.search_player(alt)
        if not results:
            print(f"NOT FOUND")
            continue
        pid = int(results[0]["player_id"])
        feats = fe.build_features(pid, season=SEASON)
        tends = fl.generate(feats)
        attrs = calc.calculate(feats, tends)
        real_3pt = p["attributes"].get("three_point_shot", "?")
        form_3pt = attrs.get("three_point_shot", "?")
        print(f"OK (3PT: formula={form_3pt}, 2K={real_3pt})")
    except Exception as e:
        print(f"ERROR: {e}")
