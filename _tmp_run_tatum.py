"""Run Tatum through the full pipeline and show attributes vs 2K real values."""
import json
import socket
import sys
import os

socket.setdefaulttimeout(15)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import TendencyPipeline


def main():
    pipe = TendencyPipeline()

    print("Running Tatum through pipeline...")
    result = pipe.generate("Jayson Tatum", season="2024-25")

    attrs = result.get("attributes", {})
    print(f"\nPlayer: {result['player_name']}")
    print(f"Position: {result['position']}")
    print()

    # Load 2K real values for comparison
    export_path = os.path.join("data", "2k_exports", "jayson_tatum.txt")
    real_2k = {}
    if os.path.isfile(export_path):
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        from src.ml.attribute_trainer import _EXPORT_TO_CANONICAL
        for export_name, canonical in _EXPORT_TO_CANONICAL.items():
            if export_name in data["categories"]["Attributes"]:
                real_2k[canonical] = data["categories"]["Attributes"][export_name]

    # Also compute formula-only attributes for comparison
    from src.attributes.calculator import AttributeCalculator
    from src.features.feature_engine import FeatureEngine
    from src.formula.formula_layer import FormulaLayer
    from src.ingest.nba_api_client import NBAApiClient

    client = NBAApiClient(cache_dir="data/cache")
    fe = FeatureEngine(client)
    fl = FormulaLayer()
    calc = AttributeCalculator()

    pid = client.search_player("Jayson Tatum")[0]["player_id"]
    features = fe.build_features(pid, season="2024-25")
    tendencies = fl.generate(features)
    formula_attrs = calc.calculate(features, tendencies)

    print(f"{'Attribute':<25} {'Formula':>8} {'ML Corrected':>12} {'Real 2K':>8} {'Δ Formula':>10} {'Δ ML':>6}")
    print("-" * 75)

    total_formula_err = 0
    total_ml_err = 0
    count = 0

    for attr in sorted(attrs.keys()):
        formula_val = formula_attrs.get(attr, "?")
        ml_val = attrs.get(attr, "?")
        real_val = real_2k.get(attr, "?")

        if isinstance(real_val, int) and isinstance(formula_val, int) and isinstance(ml_val, int):
            d_formula = real_val - formula_val
            d_ml = real_val - ml_val
            total_formula_err += abs(d_formula)
            total_ml_err += abs(d_ml)
            count += 1
            print(f"  {attr:<23} {formula_val:>8} {ml_val:>12} {real_val:>8} {d_formula:>+10} {d_ml:>+6}")
        else:
            print(f"  {attr:<23} {formula_val:>8} {ml_val:>12} {real_val!s:>8}")

    if count > 0:
        print("-" * 75)
        print(f"  {'AVG ABS ERROR':<23} {total_formula_err/count:>8.1f} {total_ml_err/count:>12.1f}")
        print(f"  {'TOTAL ABS ERROR':<23} {total_formula_err:>8} {total_ml_err:>12}")


if __name__ == "__main__":
    main()
