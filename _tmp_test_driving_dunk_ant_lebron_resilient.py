from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(8)

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

for name in ("Anthony Edwards", "LeBron James"):
    print("=" * 50)
    print(name)
    try:
        matches = client.search_player(name)
        if not matches:
            print("not found")
            continue
        player_id = int(matches[0]["player_id"])
        print("player_id:", player_id)

        try:
            features = engine.build_multiseasonal_features(player_id, s0_season="2025-26")
            used = "multiseason(2025-26/2024-25/2023-24)"
        except Exception:
            # Network fallback: at least produce a deterministic test output.
            features = engine.build_features(player_id, season="2024-25")
            used = "single-season(2024-25 fallback)"
        attrs = calc.calculate(features, tendencies={})

        rim_pressure = float(features.get("zone_fga_rate_ra", 0)) + float(features.get("zone_fga_rate_paint", 0))
        print("source:", used)
        print("position:", features.get("position"))
        print("gp:", int(float(features.get("gp", 0))))
        print("rim_pressure:", round(rim_pressure, 3))
        print("ra_pct:", round(float(features.get("zone_fg_pct_ra", 0)), 3))
        print("transition:", round(float(features.get("transition_possessions", 0)), 3))
        print("driving_dunk:", attrs["driving_dunk"])
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", type(exc).__name__, exc)
