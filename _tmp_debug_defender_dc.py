from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

names = ["Luguentz Dort", "Alex Caruso", "Jrue Holiday"]

client = NBAApiClient(cache_dir="data/cache")
engine = FeatureEngine(client)
calc = AttributeCalculator()

for name in names:
    m = client.search_player(name)
    if not m:
        print(name, "not found")
        continue
    pid = int(m[0]["player_id"])
    f = engine.build_multiseasonal_features(pid, s0_season="2025-26")

    pos = str(f.get("position", "SF")).upper()
    is_guard = pos in {"PG", "SG"}
    is_big = pos in {"PF", "C"}

    height = float(f.get("height_inches", 78.0))
    weight = float(f.get("weight_lbs", 220.0))
    usage = float(f.get("usage_rate", 18.0))
    stl_per36 = float(f.get("stl_per36", 0.0))
    blk_per36 = float(f.get("blk_per36", 0.0))
    dreb_pg = float(f.get("dreb_per36", 0.0))
    pf_per36 = float(f.get("pf_per36", 3.0))
    min_pg = float(f.get("min_per_game", 0.0))
    gp = float(f.get("gp", 0.0))

    n = calc._norm
    size_big = max(
        1.0 if is_big else 0.0,
        0.60 * n(height, 79, 84) + 0.40 * n(weight, 215, 265),
    )
    size_big = min(1.0, size_big)
    dc_stl = n(stl_per36, 0.55, 2.30)
    dc_blk = n(blk_per36, 0.10, 2.60)
    dc_dreb = n(dreb_pg, 2.8, 10.8)
    dc_disc = 1.0 - n(pf_per36, 1.6, 4.8)
    dc_engage = max(0.0, 1.0 - n(usage, 18, 35))
    dc_stock = n(stl_per36 + 0.85 * blk_per36, 1.0, 4.7)
    dc_poa = 0.33 * dc_stl + 0.28 * dc_disc + 0.21 * dc_engage + 0.18 * n(height, 73, 80)
    dc_anchor = 0.42 * dc_blk + 0.27 * dc_dreb + 0.21 * size_big + 0.10 * dc_disc
    dc_rel = 0.58 * n(min_pg, 12, 34) + 0.42 * n(gp, 25, 82)

    attrs = calc.calculate(f, tendencies={})

    print("-" * 70)
    print(name, "pos", pos, "is_guard", is_guard, "is_big", is_big)
    print("def_cons", attrs["defensive_consistency"])
    print("usage", round(usage, 2), "stl36", round(stl_per36, 2), "blk36", round(blk_per36, 2), "dreb36", round(dreb_pg, 2), "pf36", round(pf_per36, 2))
    print("dc_disc", round(dc_disc, 3), "dc_poa", round(dc_poa, 3), "dc_stock", round(dc_stock, 3), "dc_anchor", round(dc_anchor, 3), "dc_rel", round(dc_rel, 3))
