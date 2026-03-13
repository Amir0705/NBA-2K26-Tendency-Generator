from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()

players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Alex Caruso", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Giannis Antetokounmpo", "2024-25"),
    ("Anthony Davis", "2024-25"),
    ("Steven Adams", "2024-25"),
    ("Ivica Zubac", "2024-25"),
    ("Kyrie Irving", "2024-25"),
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        oreb = f.get("oreb_pct_proxy", 0)
        pos = f.get("position", "?")
        pb = round(t.get("putback", -1))
        print(f"{name:25s} ({pos})  oreb_pct={oreb:.3f}  putback={pb}")
    except Exception as e:
        print(f"{name:25s} ERROR: {e}")
