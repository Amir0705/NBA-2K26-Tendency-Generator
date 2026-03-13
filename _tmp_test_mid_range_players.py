from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Luka Doncic",
    "Rudy Gobert",
    "Ja Morant",
    "Anthony Edwards",
    "LeBron James",
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
        mid_rate = (
            float(f.get("zone_fga_rate_mid_left", 0))
            + float(f.get("zone_fga_rate_mid_center", 0))
            + float(f.get("zone_fga_rate_mid_right", 0))
        )

        print(f"  source={source}  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"mid_range_shot={a['mid_range_shot']}  "
            f"mid_pct={mid_pct:.3f}  "
            f"mid_rate={mid_rate:.3f}  "
            f"ft_pct={float(f.get('ft_pct', 0)):.3f}  "
            f"fg3a_rate={float(f.get('fg3a_rate', 0)):.3f}"
        )


if __name__ == "__main__":
    main()
