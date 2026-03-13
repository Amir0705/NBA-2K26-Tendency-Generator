from __future__ import annotations

import socket

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(15)

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

for name in ("Anthony Edwards", "LeBron James"):
    matches = client.search_player(name)
    if not matches:
        print(f"{name}: not found")
        continue

    player_id = int(matches[0]["player_id"])
    features = engine.build_multiseasonal_features(player_id, s0_season="2025-26")
    attrs = calc.calculate(features, tendencies={})

    rim_pressure = float(features.get("zone_fga_rate_ra", 0)) + float(features.get("zone_fga_rate_paint", 0))

    print(
        f"{name}: driving_dunk={attrs['driving_dunk']}, "
        f"pos={features.get('position')}, gp={int(float(features.get('gp', 0)))}, "
        f"rim_pressure={rim_pressure:.3f}, ra_pct={float(features.get('zone_fg_pct_ra', 0)):.3f}, "
        f"transition={float(features.get('transition_possessions', 0)):.3f}"
    )
