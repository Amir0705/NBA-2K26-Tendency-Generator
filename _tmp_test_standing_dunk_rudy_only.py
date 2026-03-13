from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(8)

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

matches = client.search_player("Rudy Gobert")
if not matches:
    print("Rudy Gobert not found")
    raise SystemExit(1)

player_id = int(matches[0]["player_id"])
print("player_id:", player_id)

try:
    f = engine.build_multiseasonal_features(player_id, s0_season="2025-26")
    a = calc.calculate(f, tendencies={})
    print("position:", f.get("position"))
    print("gp:", f.get("gp"))
    print("ra_rate:", round(float(f.get("zone_fga_rate_ra", 0)), 3))
    print("ra_per36:", round(float(f.get("zone_fga_per36_ra", 0)), 3))
    print("oreb_per36:", round(float(f.get("oreb_per36", 0)), 3))
    print("roll_poss:", round(float(f.get("pick_and_roll_rollman_possessions", 0)), 3))
    print("standing_dunk:", a["standing_dunk"])
except Exception as exc:  # noqa: BLE001
    print("FAILED:", type(exc).__name__, exc)
