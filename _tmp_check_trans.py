from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()

players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Alex Caruso", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Giannis Antetokounmpo", "2024-25"),
    ("Mikal Bridges", "2024-25"),
    ("Klay Thompson", "2024-25"),
    ("Kyrie Irving", "2024-25"),
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        fg3a = f.get("fg3a_rate", 0)
        fg3_pct = f.get("fg3_pct", 0)
        zra = f.get("zone_fga_rate_ra", 0)
        pos = f.get("position", "?")
        trans = round(t.get("transition_spot_up", -1))
        print(f"{name:25s} ({pos})  fg3a_rate={fg3a:.2f}  fg3%={fg3_pct:.3f}  zra={zra:.3f}  transition_spot_up={trans}")
    except Exception as e:
        print(f"{name:25s} ERROR: {e}")
