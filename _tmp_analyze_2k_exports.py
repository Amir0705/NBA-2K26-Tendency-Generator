"""Analyze all 59 2K player exports to understand attribute distributions."""
import json
import os
from collections import defaultdict
from pathlib import Path

EXPORT_DIR = Path(r"c:\Users\basic\Desktop\NBA-2K26-Tendency-Generator-fresh\data\2k_exports")

# 2K position map
POS_MAP = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "SG/SF"}

# Gameplay attributes (exclude durability)
GAMEPLAY_ATTRS = [
    "Agility", "Ball Control", "Block", "Close Shot", "Defensive Consistency",
    "Defensive Rebound", "Draw Foul", "Driving Dunk", "Driving Layup",
    "Free Throw", "Hands", "Help Defense IQ", "Hustle", "Intangibles",
    "Interior Defense", "Mid Range", "Offensive Consistency", "Offensive Rebound",
    "Passing Accuracy", "Passing IQ", "Passing Perception", "Passing Vision",
    "Perimeter Defense", "Post Control", "Post Fade", "Post Hook", "Potential",
    "Shot IQ", "Speed", "Speed With Ball", "Stamina", "Standing Dunk",
    "Steal", "Strength", "Three Point", "Vertical",
]

players = []
for f in sorted(EXPORT_DIR.glob("*.txt")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        attrs = data["categories"]["Attributes"]
        vitals = data["categories"]["Vitals"]
        tendencies = data["categories"]["Tendencies"]
        pos_num = vitals.get("Position", 0)
        pos = POS_MAP.get(pos_num, str(pos_num))
        
        player = {
            "name": data.get("fullName", f.stem),
            "file": f.name,
            "position": pos,
            "age": vitals.get("Age", 0),
            "height": vitals.get("Height", 0) if vitals.get("Height", 0) > 100 else 0,  # raw encoding
            "weight": vitals.get("Weight", 0) if vitals.get("Weight", 0) < 400 else 0,
            "attributes": {k: v for k, v in attrs.items() if k in GAMEPLAY_ATTRS},
            "tendencies": tendencies,
        }
        players.append(player)
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

print(f"\n{'='*80}")
print(f"LOADED {len(players)} PLAYERS")
print(f"{'='*80}")

# ── Group by position ──
by_pos = defaultdict(list)
for p in players:
    by_pos[p["position"]].append(p)

print(f"\nPosition breakdown: {dict((k, len(v)) for k, v in sorted(by_pos.items()))}")

# ── Attribute ranges by position ──
print(f"\n{'='*80}")
print("ATTRIBUTE RANGES BY POSITION (min / avg / max)")
print(f"{'='*80}")

for pos in ["PG", "SG", "SF", "PF", "C"]:
    pos_players = by_pos.get(pos, [])
    if not pos_players:
        continue
    print(f"\n── {pos} ({len(pos_players)} players) ──")
    print(f"  Players: {', '.join(p['name'] for p in pos_players)}")
    print(f"  {'Attribute':<25} {'Min':>5} {'Avg':>5} {'Max':>5}  {'Min Player':<20} {'Max Player':<20}")
    print(f"  {'-'*90}")
    for attr in GAMEPLAY_ATTRS:
        vals = [(p["attributes"].get(attr, 0), p["name"]) for p in pos_players]
        vals_only = [v[0] for v in vals]
        min_val = min(vals_only)
        max_val = max(vals_only)
        avg_val = sum(vals_only) / len(vals_only)
        min_player = [v[1] for v in vals if v[0] == min_val][0]
        max_player = [v[1] for v in vals if v[0] == max_val][0]
        print(f"  {attr:<25} {min_val:>5} {avg_val:>5.0f} {max_val:>5}  {min_player:<20} {max_player:<20}")

# ── Overall attribute distribution (all players) ──
print(f"\n{'='*80}")
print("GLOBAL ATTRIBUTE STATS (ALL 59 PLAYERS)")
print(f"{'='*80}")
print(f"  {'Attribute':<25} {'Min':>5} {'P25':>5} {'Med':>5} {'P75':>5} {'Max':>5} {'Avg':>5}")
print(f"  {'-'*70}")

import statistics
for attr in GAMEPLAY_ATTRS:
    vals = sorted([p["attributes"].get(attr, 0) for p in players])
    n = len(vals)
    p25 = vals[n // 4]
    med = vals[n // 2]
    p75 = vals[3 * n // 4]
    print(f"  {attr:<25} {vals[0]:>5} {p25:>5} {med:>5} {p75:>5} {vals[-1]:>5} {sum(vals)/n:>5.0f}")

# ── Key patterns: What separates stars from role players? ──
print(f"\n{'='*80}")
print("TOP 10 vs BOTTOM 10 PLAYERS BY KEY ATTRIBUTES")
print(f"{'='*80}")

for attr in ["Three Point", "Driving Layup", "Ball Control", "Perimeter Defense", 
             "Interior Defense", "Speed", "Strength", "Shot IQ", "Post Control",
             "Draw Foul", "Offensive Consistency"]:
    ranked = sorted(players, key=lambda p: p["attributes"].get(attr, 0), reverse=True)
    print(f"\n  {attr}:")
    top5 = ', '.join(p['name'] + ' (' + str(p['attributes'].get(attr, 0)) + ')' for p in ranked[:5])
    bot5 = ', '.join(p['name'] + ' (' + str(p['attributes'].get(attr, 0)) + ')' for p in ranked[-5:])
    print(f"    TOP 5:    {top5}")
    print(f"    BOTTOM 5: {bot5}")

# ── Tendency vs Attribute correlation spotcheck ──
print(f"\n{'='*80}")
print("TENDENCY vs ATTRIBUTE SPOTCHECK")
print(f"{'='*80}")

# How does Shot Three tendency correlate with Three Point attribute?
print(f"\n  Shot Three Tendency vs Three Point Attribute:")
print(f"  {'Player':<25} {'Pos':>3} {'Shot Three T':>12} {'3PT Attr':>8}")
print(f"  {'-'*55}")
for p in sorted(players, key=lambda x: x["attributes"].get("Three Point", 0), reverse=True)[:15]:
    t_three = p["tendencies"].get("Shot Three", 0)
    a_three = p["attributes"].get("Three Point", 0)
    print(f"  {p['name']:<25} {p['position']:>3} {t_three:>12} {a_three:>8}")

print(f"\n  Driving Layup Tendency vs Driving Layup Attribute:")
print(f"  {'Player':<25} {'Pos':>3} {'Drv Layup T':>12} {'DrvLayup A':>10}")
print(f"  {'-'*55}")
for p in sorted(players, key=lambda x: x["attributes"].get("Driving Layup", 0), reverse=True)[:15]:
    t_lay = p["tendencies"].get("Driving Layup Tendency", 0)
    a_lay = p["attributes"].get("Driving Layup", 0)
    print(f"  {p['name']:<25} {p['position']:>3} {t_lay:>12} {a_lay:>10}")

# ── How 2K handles the "overall" puzzle attributes ──
print(f"\n{'='*80}")
print("INTANGIBLES & POTENTIAL ANALYSIS")
print(f"{'='*80}")
print(f"  {'Player':<25} {'Pos':>3} {'Intang':>6} {'Potent':>6} {'OC':>4} {'DC':>4} {'ShotIQ':>6}")
print(f"  {'-'*60}")
for p in sorted(players, key=lambda x: x["attributes"].get("Intangibles", 0), reverse=True):
    print(f"  {p['name']:<25} {p['position']:>3} "
          f"{p['attributes'].get('Intangibles', 0):>6} "
          f"{p['attributes'].get('Potential', 0):>6} "
          f"{p['attributes'].get('Offensive Consistency', 0):>4} "
          f"{p['attributes'].get('Defensive Consistency', 0):>4} "
          f"{p['attributes'].get('Shot IQ', 0):>6}")
