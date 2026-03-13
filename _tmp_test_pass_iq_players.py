from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = [
    "Nikola Jokic",
    "Luka Doncic",
    "LeBron James",
    "LaMelo Ball",
    "Jalen Brunson",
    "James Harden",
    "Draymond Green",
    "Ja Morant",
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
        ast_p36 = float(features.get("ast_per_game", 0.0)) * 36.0 / min_pg
        pnr_p36 = float(features.get("pick_and_roll_ball_handler_possessions", 0.0)) * 36.0 / min_pg
        iso_p36 = float(features.get("isolation_possessions", 0.0)) * 36.0 / min_pg

        print(f"  source={source}  pos={features.get('position')}  gp={int(float(features.get('gp', 0)))}")
        print(
            "  "
            f"pass_iq={attrs['pass_iq']}  "
            f"pass_accuracy={attrs['pass_accuracy']}  "
            f"ast_p36={ast_p36:.2f}  "
            f"ast/tov={float(features.get('ast_to_tov', 0)):.2f}  "
            f"tov_pct={float(features.get('tov_pct_proxy', 0)):.3f}  "
            f"pnr_p36={pnr_p36:.2f}  "
            f"iso_p36={iso_p36:.2f}"
        )


if __name__ == "__main__":
    main()