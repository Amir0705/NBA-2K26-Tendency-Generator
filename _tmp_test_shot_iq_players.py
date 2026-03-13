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
        mid_pct = (
            float(f.get("zone_fg_pct_mid_left", 0))
            + float(f.get("zone_fg_pct_mid_center", 0))
            + float(f.get("zone_fg_pct_mid_right", 0))
        ) / 3.0

        print(f"  source={source}  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"shot_iq={a['shot_iq']}  "
            f"ts_pct={float(f.get('ts_pct', 0)):.3f}  "
            f"efg_pct={float(f.get('efg_pct', 0)):.3f}  "
            f"fg3_pct={float(f.get('fg3_pct', 0)):.3f}  "
            f"mid_pct={mid_pct:.3f}  "
            f"usage={float(f.get('usage_rate', 0)):.1f}"
        )


if __name__ == "__main__":
    main()
