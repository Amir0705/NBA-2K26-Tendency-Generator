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
    ("Klay Thompson", "2024-25"),
]

for name, season in players:
    try:
        r = pipe.generate(name, season)
        f = r["features"]
        t = r["tendencies"]
        pos = f.get("position", "?")
        fg3a = f.get("fg3a_per_game", 0)
        fg3p = f.get("fg3_pct", 0)
        fg3r = f.get("fg3a_rate", 0)

        s3 = t.get("shot_three", 0)
        su3 = t.get("spot_up_shot_three", 0)
        os3 = t.get("off_screen_shot_three", 0)
        c3 = t.get("contested_jumper_three", 0)
        sb3 = t.get("stepback_jumper_three", 0)
        tp3 = t.get("transition_pull_up_three", 0)

        print(f"{name:28s} ({pos}) fg3a={fg3a:4.1f} fg3%={fg3p:.3f} rate={fg3r:.2f}  "
              f"shot3={s3:5.1f} spot={su3:5.1f} off={os3:5.1f} cont={c3:5.1f} step={sb3:5.1f} trans={tp3:5.1f}")
    except Exception as e:
        print(f"{name}: ERROR: {e}")
