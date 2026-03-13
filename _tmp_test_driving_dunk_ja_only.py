from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(12)

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

matches = client.search_player("Ja Morant")
if not matches:
    print("Ja Morant not found")
    raise SystemExit(1)

player_id = int(matches[0]["player_id"])
print("player_id:", player_id)

f = engine.build_multiseasonal_features(player_id, s0_season="2025-26")
a = calc.calculate(f, tendencies={})

rim_pressure = float(f.get("zone_fga_rate_ra", 0)) + float(f.get("zone_fga_rate_paint", 0))
print("position:", f.get("position"))
print("gp:", f.get("gp"))
print("rim_pressure:", round(rim_pressure, 3))
print("ra_pct:", round(float(f.get("zone_fg_pct_ra", 0)), 3))
print("ra_per36:", round(float(f.get("zone_fga_per36_ra", 0)), 3))
print("transition_poss:", round(float(f.get("transition_possessions", 0)), 3))
print("iso+pnr_bh:", round(float(f.get("isolation_possessions", 0)) + float(f.get("pick_and_roll_ball_handler_possessions", 0)), 3))
print("fta_rate:", round(float(f.get("fta_rate", 0)), 3))
print("driving_dunk:", a["driving_dunk"])
