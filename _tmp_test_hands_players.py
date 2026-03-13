from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Luka Doncic",
    "Nikola Jokic",
    "Giannis Antetokounmpo",
    "Jalen Brunson",
    "Anthony Edwards",
    "Joel Embiid",
    "Rudy Gobert",
    "Ja Morant",
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
        min_pg = max(float(features.get("min_per_game", 0.0)), 1.0)
        roll_p36 = float(features.get("pick_and_roll_rollman_possessions", 0.0)) * 36.0 / min_pg
        cuts_p36 = float(features.get("cuts", 0.0)) * 36.0 / min_pg
        spot_p36 = float(features.get("spot_up_possessions", 0.0)) * 36.0 / min_pg

        print(f"  source={source}  pos={features.get('position')}  gp={int(float(features.get('gp', 0)))}")
        print(
            "  "
            f"hands={attrs['hands']}  "
            f"tov_pct={float(features.get('tov_pct_proxy', 0)):.3f}  "
            f"roll_p36={roll_p36:.2f}  "
            f"cuts_p36={cuts_p36:.2f}  "
            f"spot_p36={spot_p36:.2f}"
        )


if __name__ == "__main__":
    main()