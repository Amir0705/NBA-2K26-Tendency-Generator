from src.pipeline import TendencyPipeline
p = TendencyPipeline()
r = p.generate("Nikola Jokic", "2024-25")
f = r["features"]
t = r["tendencies"]
print(f"Jokic ({f['position']}) oreb_pct={f.get('oreb_pct_proxy',0):.3f} putback={round(t['putback'])}")
