"""Run attribute ML training from 2K editor exports."""
import json
import socket
import sys
import os

# Global socket timeout to prevent indefinite hangs on NBA API calls
socket.setdefaulttimeout(15)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.formula.formula_layer import FormulaLayer
from src.ingest.nba_api_client import NBAApiClient
from src.ml.attribute_trainer import AttributeTrainer

EXPORT_DIR = os.path.join("data", "2k_exports")
MODEL_DIR = os.path.join("models", "attributes")
SEASON = "2024-25"
CACHE_DIR = os.path.join("data", "cache")

def main():
    print("=" * 70)
    print("ATTRIBUTE ML TRAINING")
    print("=" * 70)
    print(f"Export dir: {EXPORT_DIR}")
    print(f"Model dir:  {MODEL_DIR}")
    print(f"Season:     {SEASON}")
    print()

    # Initialize components
    client = NBAApiClient(cache_dir=CACHE_DIR)
    feature_engine = FeatureEngine(client)
    formula_layer = FormulaLayer()
    calculator = AttributeCalculator()

    trainer = AttributeTrainer(
        feature_engine=feature_engine,
        attribute_calculator=calculator,
        nba_client=client,
        formula_layer=formula_layer,
    )

    report = trainer.train(
        export_dir=EXPORT_DIR,
        model_dir=MODEL_DIR,
        season=SEASON,
        cached_only=True,
    )

    if not report:
        print("Training failed — no models produced.")
        return

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    better = sum(1 for r in report.values() if r["improvement"] > 0.5)
    same = sum(1 for r in report.values() if -0.5 <= r["improvement"] <= 0.5)
    worse = sum(1 for r in report.values() if r["improvement"] < -0.5)

    print(f"Models trained: {len(report)}")
    print(f"  BETTER (ML helps):  {better}")
    print(f"  SAME:               {same}")
    print(f"  WORSE (ML hurts):   {worse}")
    print()

    # Show top improvements
    by_improvement = sorted(report.items(), key=lambda x: x[1]["improvement"], reverse=True)
    print("Top 10 improvements:")
    for attr, info in by_improvement[:10]:
        print(f"  {attr:<25} formula_MAE={info['formula_mae']:5.1f} → "
              f"corrected_MAE={info['corrected_mae']:5.1f}  "
              f"(improvement: {info['improvement']:+5.1f})")

    print("\nBottom 5 (ML hurts):")
    for attr, info in by_improvement[-5:]:
        print(f"  {attr:<25} formula_MAE={info['formula_mae']:5.1f} → "
              f"corrected_MAE={info['corrected_mae']:5.1f}  "
              f"(improvement: {info['improvement']:+5.1f})")


if __name__ == "__main__":
    main()
