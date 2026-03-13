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
    ("Shai Gilgeous-Alexander", "2024-25"),
    ("Donovan Mitchell", "2024-25"),
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        pos = f.get("position", "?")
        zra = f.get("zone_fga_rate_ra", 0)
        usg = f.get("usg_pct_proxy", 0)
        fta = f.get("fta_rate", 0)
        drv = round(t.get("drive", -1))
        dd = round(t.get("driving_dunk", -1))
        sd = round(t.get("standing_dunk", -1))
        fd = round(t.get("flashy_dunk", -1))
        ht = f.get("height_inches", 0)
        wt = f.get("weight_lbs", 0)
        blk = f.get("blk_per36", 0)
        print(f"{name:25s} ({pos})  zra={zra:.3f}  fta={fta:.3f}  drive={drv}  ht={ht}  wt={wt}  blk={blk:.1f}  dr_dunk={dd}  st_dunk={sd}  flashy={fd}")
    except Exception as e:
        print(f"{name:25s} ERROR: {e}")
