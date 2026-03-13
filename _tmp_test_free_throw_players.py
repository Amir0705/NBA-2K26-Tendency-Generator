from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Stephen Curry",
    "Luka Doncic",
    "Anthony Edwards",
    "LeBron James",
    "Nikola Jokic",
    "Nikola Vucevic",
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
            f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
            source = "multiseason"
        except Exception:
            f = engine.build_features(pid, season="2024-25")
            source = "fallback-2024-25"

        a = calc.calculate(f, tendencies={})
        est_attempts = float(f.get("fta_per_game", 0)) * float(f.get("gp", 0))

        print(f"  source={source}  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"free_throw={a['free_throw']}  "
            f"ft_pct={float(f.get('ft_pct', 0)):.3f}  "
            f"fta_pg={float(f.get('fta_per_game', 0)):.2f}  "
            f"est_total_fta={est_attempts:.1f}"
        )


if __name__ == "__main__":
    main()
