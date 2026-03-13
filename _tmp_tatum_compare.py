"""Compare Tatum generated vs 2K reference attributes."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline

p = TendencyPipeline()
result = p.generate('Jayson Tatum', season='2024-25')
gen = result.get('attributes', {})

ref = {
    "driving_layup": 89, "standing_dunk": 45, "driving_dunk": 87, "close_shot": 86,
    "mid_range_shot": 83, "three_point_shot": 90, "free_throw": 84, "shot_iq": 86,
    "post_hook": 64, "post_fade": 73, "post_control": 72,
    "draw_foul": 77, "ball_handle": 90, "speed_with_ball": 81, "hands": 80,
    "pass_accuracy": 87, "pass_iq": 87, "pass_vision": 85,
    "offensive_consistency": 90, "defensive_consistency": 59,
    "interior_defense": 46, "perimeter_defense": 68, "steal": 50, "block": 37,
    "help_defense_iq": 48, "pass_perception": 59,
    "offensive_rebound": 40, "defensive_rebound": 80,
    "speed": 78, "agility": 73, "strength": 66, "vertical": 76, "stamina": 99,
    "intangibles": 91, "hustle": 62, "overall_durability": 96, "potential": 99,
}

total_err = 0
count = 0
print(f"{'Attribute':<28} {'Gen':>5} {'2K':>5} {'Diff':>6}")
print("-" * 48)
for attr in sorted(ref.keys()):
    g = gen.get(attr, 0)
    r = ref[attr]
    diff = g - r
    total_err += abs(diff)
    count += 1
    flag = "" if abs(diff) <= 3 else (" <<<" if abs(diff) > 7 else " <")
    print(f"  {attr:<26} {g:>5} {r:>5} {diff:>+6}{flag}")

print(f"\nMAE: {total_err/count:.1f}  |  Total attrs: {count}")
print(f"Within 3: {sum(1 for a in ref if abs(gen.get(a,0)-ref[a])<=3)}/{count}")
print(f"Within 5: {sum(1 for a in ref if abs(gen.get(a,0)-ref[a])<=5)}/{count}")
print(f"Within 7: {sum(1 for a in ref if abs(gen.get(a,0)-ref[a])<=7)}/{count}")
