from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Ja Morant",
    "Anthony Edwards",
    "Jalen Brunson",
    "Luka Doncic",
    "Giannis Antetokounmpo",
    "James Harden",
    "Nikola Jokic",
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
        min_pg = max(float(features.get("min_per_game", 0.0)), 1.0)
        transition_p36 = float(features.get("transition_possessions", 0.0)) * 36.0 / min_pg
        rim_pressure = float(features.get("zone_fga_rate_ra", 0.0)) + float(features.get("zone_fga_rate_paint", 0.0))

        print(f"  source={source}  pos={features.get('position')}  gp={int(float(features.get('gp', 0)))}")
        print(
            "  "
            f"speed_with_ball={attrs['speed_with_ball']}  "
            f"ball_handle={attrs['ball_handle']}  "
            f"transition_p36={transition_p36:.2f}  "
            f"rim_pressure={rim_pressure:.3f}  "
            f"driving_dunk={attrs['driving_dunk']}"
        )


if __name__ == "__main__":
    main()