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
        rim_pressure = float(f.get("zone_fga_rate_ra", 0)) + float(f.get("zone_fga_rate_paint", 0))

        print(f"  source={source}  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"close_shot={a['close_shot']}  "
            f"paint_pct={float(f.get('zone_fg_pct_paint', 0)):.3f}  "
            f"ra_pct={float(f.get('zone_fg_pct_ra', 0)):.3f}  "
            f"rim_pressure={rim_pressure:.3f}  "
            f"ft_pct={float(f.get('ft_pct', 0)):.3f}"
        )


if __name__ == "__main__":
    main()
