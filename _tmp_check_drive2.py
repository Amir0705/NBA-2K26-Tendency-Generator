from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()
for name in ["Stephen Curry", "LeBron James", "Anthony Edwards", "Rudy Gobert", "Donovan Mitchell", "Klay Thompson"]:
    try:
        r = pipe.generate(name, season="2024-25")
        f = r.get("features", {})
        pos = r.get("position", "?")
        zra = f.get("zone_fga_rate_ra", 0)
        usg = f.get("usage_rate", 0)
        iso = f.get("isolation_possessions", 0)
        pnr = f.get("pick_and_roll_ball_handler_possessions", 0)
        off_scr = f.get("off_screen_possessions", 0)
        fta_rate = f.get("fta_rate", 0)
        print(
            f"{name} ({pos}): zra={zra:.3f}  usg={usg:.1f}%  "
            f"iso={iso:.1f}  pnr_bh={pnr:.1f}  off_scr={off_scr:.1f}  "
            f"fta_rate={fta_rate:.3f}"
        )
    except Exception as e:
        print(f"{name}: ERROR {e}")
