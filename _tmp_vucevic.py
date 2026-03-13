import sys, os, socket
socket.setdefaulttimeout(15)
sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()
result = pipe.generate("Nikola Vucevic", season="2024-25")

# Print key features that drive close_shot, mid_range, pass
f = result.get("features", {})
print("=== KEY FEATURES ===")
for k in sorted(f.keys()):
    if any(x in k for x in ["paint", "ra_", "mid_", "ast", "fg_pct", "fg3_pct",
                              "ts_pct", "efg", "pts_per", "usage", "tov",
                              "position", "height", "weight", "age", "min_per",
                              "dish", "pctile_ast"]):
        print(f"  {k}: {f[k]}")

print("\n=== ATTRIBUTES ===")
attrs = result.get("attributes", {})
targets = ["close_shot", "mid_range_shot", "pass_accuracy", "pass_iq", "pass_vision",
           "three_point_shot", "driving_layup", "post_hook", "post_fade", "post_control"]
for a in targets:
    print(f"  {a}: {attrs.get(a, '?')}")
