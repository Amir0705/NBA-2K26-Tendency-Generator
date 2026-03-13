from src.pipeline import TendencyPipeline
pipe = TendencyPipeline()
players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Giannis Antetokounmpo", "2024-25"),
    ("Nikola Jokic", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Draymond Green", "2024-25"),
    ("Anthony Davis", "2024-25"),
    ("Luka Doncic", "2024-25"),
    ("Alex Caruso", "2024-25"),
    ("Joel Embiid", "2024-25"),
    ("Shai Gilgeous-Alexander", "2024-25"),
]
for name, season in players:
    try:
        r = pipe.generate(name, season)
        t = r["tendencies"]
        f = r["features"]
        pos = f.get("position", "?")
        pf = f.get("pf_per36", 0)
        ht = f.get("height_inches", 0)
        wt = f.get("weight_lbs", 0)
        foul = round(t.get("foul", -1))
        hard = round(t.get("hard_foul", -1))
        print(f"{name:28s} ({pos}) pf36={pf:.1f}  ht={ht}  wt={wt}  foul={foul}  hard_foul={hard}")
    except Exception as e:
        print(f"{name}: ERROR: {e}")
