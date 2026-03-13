from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()

players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Alex Caruso", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Shai Gilgeous-Alexander", "2024-25"),
    ("Luka Doncic", "2024-25"),
    ("Mikal Bridges", "2024-25"),
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        iso_e = round(t.get("iso_vs_elite_defender", -1))
        iso_g = round(t.get("iso_vs_good_defender", -1))
        iso_a = round(t.get("iso_vs_average_defender", -1))
        iso_p = round(t.get("iso_vs_poor_defender", -1))
        usg = f.get("usg_pct_proxy", 0)
        iso_poss = f.get("isolation_possessions", 0)
        print(f"{name:25s} USG={usg:.2f}  ISO_poss={iso_poss:.1f}  "
              f"elite={iso_e}  good={iso_g}  avg={iso_a}  poor={iso_p}")
    except Exception as e:
        print(f"{name:25s} ERROR: {e}")
