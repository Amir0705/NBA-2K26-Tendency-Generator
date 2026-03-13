from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Nikola Jokic",
    "Luka Doncic",
    "Giannis Antetokounmpo",
    "Anthony Edwards",
    "Jalen Brunson",
    "James Harden",
    "LaMelo Ball",
    "Draymond Green",
    "Rudy Gobert",
]


def main() -> None:
    client = NBAApiClient(cache_dir="data/cache")
    engine = FeatureEngine(client)
    calc = AttributeCalculator()

    for name in PLAYERS:
        print("=" * 60)
        print(name)
        matches = client.search_player(name)
        if not matches:
            print("  not found")
            continue

        pid = int(matches[0]["player_id"])
        try:
            features = engine.build_multiseasonal_features(pid, s0_season="2025-26")
            source = "multiseason"
        except Exception:
            features = engine.build_features(pid, season="2024-25")
            source = "fallback-2024-25"

        attrs = calc.calculate(features, tendencies={})

        print(f"  source={source}  pos={features.get('position')}  gp={int(float(features.get('gp', 0)))}")
        print(
            "  "
            f"off_cons={attrs['offensive_consistency']}  "
            f"shot_iq={attrs['shot_iq']}  "
            f"ts={float(features.get('ts_pct', 0)):.3f}  "
            f"usage={float(features.get('usage_rate', 0)):.1f}  "
            f"pts={float(features.get('pts_per_game', 0)):.1f}"
        )


if __name__ == "__main__":
    main()