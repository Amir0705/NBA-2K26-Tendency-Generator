from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()
for name in ["Stephen Curry", "LeBron James", "Anthony Edwards", "Donovan Mitchell", "Rudy Gobert", "Kyrie Irving"]:
    try:
        r = pipe.generate(name, season="2024-25")
        t = r.get("tendencies", {})
        f = r.get("features", {})
        pos = r.get("position", "?")
        drive = t.get("drive", "?")
        print(
            f"{name} ({pos}): drive={drive}  "
            f"euro={t.get('euro_step_layup','?')}  spin={t.get('spin_layup','?')}  "
            f"hop={t.get('hop_step_layup','?')}  floater={t.get('floater','?')}  "
            f"zra={f.get('zone_fga_rate_ra',0):.3f}  zpaint={f.get('zone_fga_rate_paint',0):.3f}"
        )
    except Exception as e:
        print(f"{name}: ERROR {e}")
