from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

name = "Jrue Holiday"
client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

match = client.search_player(name)[0]
pid = int(match["player_id"])
f = engine.build_multiseasonal_features(pid, s0_season="2025-26")

pos = str(f.get("position", "SF")).upper()
is_big = pos in {"PF", "C"}

stl_per36 = float(f.get("stl_per36", 0.0))
blk_per36 = float(f.get("blk_per36", 0.0))
dreb_pg = float(f.get("dreb_per36", 0.0))
pf_per36 = float(f.get("pf_per36", 0.0))
usage = float(f.get("usage_rate", 0.0))
min_pg = float(f.get("min_per_game", 0.0))
gp = float(f.get("gp", 0.0))
height = float(f.get("height_inches", 0.0))
weight = float(f.get("weight_lbs", 0.0))

norm = calc._norm
size_big = max(
    1.0 if is_big else 0.0,
    0.60 * norm(height, 79, 84) + 0.40 * norm(weight, 215, 265),
)
size_big = min(1.0, size_big)

def_engagement = max(0.0, 1.0 - norm(usage, 18, 35))

dc_stl = norm(stl_per36, 0.55, 2.30)
dc_blk = norm(blk_per36, 0.10, 2.60)
dc_dreb = norm(dreb_pg, 2.8, 10.8)
dc_disc = 1.0 - norm(pf_per36, 1.6, 4.8)
dc_stock = norm(stl_per36 + 0.85 * blk_per36, 1.0, 4.7)
dc_reliability = 0.58 * norm(min_pg, 12, 34) + 0.42 * norm(gp, 25, 82)
dc_poa = 0.33 * dc_stl + 0.28 * dc_disc + 0.21 * def_engagement + 0.18 * norm(height, 73, 80)

print("name", name)
print("position", pos, "is_big", is_big)
print("height", height, "weight", weight)
print("gp", gp, "min", min_pg, "usage", usage)
print("stl36", stl_per36, "blk36", blk_per36, "dreb36", dreb_pg, "pf36", pf_per36)
print("dc_stl", round(dc_stl,3), "dc_blk", round(dc_blk,3), "dc_dreb", round(dc_dreb,3))
print("dc_disc", round(dc_disc,3), "dc_stock", round(dc_stock,3), "dc_reliability", round(dc_reliability,3), "dc_poa", round(dc_poa,3))
print("stopper_gate", (not is_big) and dc_disc > 0.68 and dc_poa > 0.52 and dc_stock > 0.34 and usage < 26)
print("veteran_gate", (not is_big) and dc_disc > 0.70 and dc_stl > 0.38 and usage < 24 and dc_reliability > 0.65)
