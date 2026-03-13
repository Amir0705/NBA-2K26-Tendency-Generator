from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()

players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Giannis Antetokounmpo", "2024-25"),
    ("Anthony Davis", "2024-25"),
    ("Zion Williamson", "2024-25"),
    ("Ja Morant", "2024-25"),
    ("Kyrie Irving", "2024-25"),
    ("Nikola Jokic", "2024-25"),
    ("Alex Caruso", "2024-25"),
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        pos = f.get("position", "?")
        zra = f.get("zone_fga_rate_ra", 0)
        ht = f.get("height_inches", 0)
        dd = round(t.get("driving_dunk", -1))
        sd = round(t.get("standing_dunk", -1))
        oop = round(t.get("alley_oop", -1))
        print(f"{name:25s} ({pos})  ht={ht}  zra={zra:.3f}  dr_dunk={dd}  st_dunk={sd}  alley_oop={oop}")
    except Exception as e:
        print(f"{name:25s} ERROR: {e}")
