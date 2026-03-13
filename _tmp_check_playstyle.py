from src.pipeline import TendencyPipeline
pipe = TendencyPipeline()

players = [
    ("Alex Caruso", "2024-25"),
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Nikola Jokic", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Shai Gilgeous-Alexander", "2024-25"),
    ("Luka Doncic", "2024-25"),
    ("Draymond Green", "2024-25"),
    ("Mikal Bridges", "2024-25"),
]

for name, season in players:
    try:
        r = pipe.generate(name, season)
        f = r["features"]
        ps = r.get("play_styles", {})
        pos = f.get("position", "?")
        usg = f.get("usg_pct_proxy", 0) * 100
        fg3a = f.get("fg3a_per_game", 0)
        fg3p = f.get("fg3_pct", 0)
        fg3r = f.get("fg3a_rate", 0)
        ast = f.get("ast_per_game", 0)
        stl = f.get("stl_per36", 0)
        spot = f.get("spot_up_possessions", 0)
        pnr_bh = f.get("pick_and_roll_ball_handler_possessions", 0)
        cuts = f.get("cuts", 0)

        pris = r.get("play_style_priorities", [])
        scores = r.get("play_style_scores", {})
        top5 = list(scores.items())[:5]

        print(f"\n{name} ({pos}) usg={usg:.0f}% fg3a={fg3a:.1f} fg3%={fg3p:.3f} fg3r={fg3r:.2f} ast={ast:.1f} stl={stl:.1f} spot={spot:.1f} pnr_bh={pnr_bh:.1f} cuts={cuts:.1f}")
        print(f"  Priorities: {pris}")
        print(f"  Top scores: {', '.join(f'{k}={v}' for k,v in top5)}")
    except Exception as e:
        print(f"{name}: ERROR: {e}")
