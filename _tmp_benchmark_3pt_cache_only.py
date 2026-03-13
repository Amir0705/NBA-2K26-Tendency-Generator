from __future__ import annotations

import sqlite3

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

PLAYERS = [
    ("Stephen Curry", 201939),
    ("Luka Doncic", 1629029),
    ("Anthony Edwards", 1630162),
    ("LeBron James", 2544),
    ("Nikola Jokic", 203999),
    ("Nikola Vucevic", 202696),
    ("Rudy Gobert", 203497),
    ("Ja Morant", 1629630),
]
SEASONS = ("2025-26", "2024-25", "2023-24")


def has_cache(keys: set[str], pid: int, season: str) -> bool:
    return (
        f"player_stats:{pid}:{season}" in keys
        and f"shot_chart:{pid}:{season}" in keys
    )


def main() -> None:
    conn = sqlite3.connect("data/cache/nba_cache.db")
    keys = {r[0] for r in conn.execute("SELECT key FROM cache").fetchall()}

    client = NBAApiClient(cache_dir="data/cache")
    engine = FeatureEngine(client)
    engine._league_averages = []  # avoid any API call for percentile context
    calc = AttributeCalculator()

    for name, pid in PLAYERS:
        full = all(has_cache(keys, pid, s) for s in SEASONS)
        if full:
            f = engine.build_multiseasonal_features(pid, s0_season="2025-26")
            source = "multiseason"
        elif has_cache(keys, pid, "2024-25"):
            f = engine.build_features(pid, season="2024-25")
            source = "cache-2024-25"
        else:
            print(f"{name:16} | no usable cache")
            continue

        a = calc.calculate(f, tendencies={})
        print(
            f"{name:16} | {source:12} | 3PT={a['three_point_shot']:>2} "
            f"| fg3%={float(f.get('fg3_pct', 0)):.3f} "
            f"| fg3a/36={float(f.get('fg3a_per36', 0)):.2f} "
            f"| gp={int(float(f.get('gp', 0))):>2}"
        )


if __name__ == "__main__":
    main()
