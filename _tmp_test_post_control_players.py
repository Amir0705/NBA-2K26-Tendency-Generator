from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Nikola Jokic",
    "Joel Embiid",
    "Nikola Vucevic",
    "Rudy Gobert",
    "LeBron James",
    "Luka Doncic",
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
        f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
        a = calc.calculate(f, tendencies={})
        post_p36 = float(f.get("post_up_possessions", 0)) * 36.0 / max(float(f.get("min_per_game", 1)), 1.0)

        print(f"  source=multiseason  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"post_control={a['post_control']}  "
            f"post_poss_p36={post_p36:.2f}  "
            f"post_ppp={float(f.get('post_up_ppp', 0)):.3f}  "
            f"close_shot={a['close_shot']}  "
            f"post_hook={a['post_hook']}  "
            f"post_fade={a['post_fade']}"
        )


if __name__ == "__main__":
    main()
