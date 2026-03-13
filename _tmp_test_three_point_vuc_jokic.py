from __future__ import annotations

import socket
import sqlite3

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(10)

PLAYERS = ["Nikola Vucevic", "Nikola Jokic"]


def _has_cache(pid: int, season: str) -> bool:
    try:
        conn = sqlite3.connect("data/cache/nba_cache.db")
        k1 = f"player_stats:{pid}:{season}"
        k2 = f"shot_chart:{pid}:{season}"
        ok1 = conn.execute("SELECT 1 FROM cache WHERE key=?", (k1,)).fetchone() is not None
        ok2 = conn.execute("SELECT 1 FROM cache WHERE key=?", (k2,)).fetchone() is not None
        conn.close()
        return ok1 and ok2
    except Exception:
        return False


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
        if _has_cache(pid, "2025-26") and _has_cache(pid, "2024-25"):
            try:
                f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
                source = "multiseason"
            except Exception:
                f = engine.build_features(pid, season="2024-25")
                source = "fallback-2024-25"
        else:
            f = engine.build_features(pid, season="2024-25")
            source = "fallback-2024-25"

        a = calc.calculate(f, tendencies={})
        above_break = float(f.get("zone_fga_rate_above_break3", 0))
        corner = float(f.get("zone_fga_rate_corner3_left", 0)) + float(f.get("zone_fga_rate_corner3_right", 0))

        print(f"  source={source}  pos={f.get('position')}  gp={int(float(f.get('gp', 0)))}")
        print(
            "  "
            f"three_point_shot={a['three_point_shot']}  "
            f"fg3_pct={float(f.get('fg3_pct', 0)):.3f}  "
            f"fg3a_pg={float(f.get('fg3a_per_game', 0)):.2f}  "
            f"fg3a_rate={float(f.get('fg3a_rate', 0)):.3f}  "
            f"AB3_rate={above_break:.3f}  corner3_rate={corner:.3f}"
        )


if __name__ == "__main__":
    main()
