from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()
for name in ["Stephen Curry", "LeBron James", "Anthony Edwards", "Rudy Gobert"]:
    try:
        r = pipe.generate(name, season="2024-25")
        t = r.get("tendencies", {})
        f = r.get("features", {})
        pos = r.get("position", "?")
        zra = f.get("zone_fga_rate_ra", 0)
        usg = f.get("usage_rate", 0)
        print(
            f"{name} ({pos}): drive={t.get('drive','?')}  "
            f"spot_up_drive={t.get('spot_up_drive','?')}  "
            f"off_screen_drive={t.get('off_screen_drive','?')}  "
            f"zra={zra:.3f}  usg={usg:.1f}%"
        )
    except Exception as e:
        print(f"{name}: ERROR {e}")
