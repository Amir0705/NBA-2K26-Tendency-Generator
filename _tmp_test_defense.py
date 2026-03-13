"""Compare defensive attributes across player archetypes."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline

p = TendencyPipeline()

# Defense-relevant attributes to compare
DEF_ATTRS = [
    "perimeter_defense", "interior_defense", "steal", "block",
    "help_defense_iq", "pass_perception", "defensive_consistency",
    "hustle", "hands", "speed", "agility",
]

players = ['Luka Doncic', 'Alex Caruso', 'Anthony Edwards', 'Jayson Tatum']

results = {}
for name in players:
    result = p.generate(name, season='2024-25')
    results[name] = result.get('attributes', {})

# Print comparison table
header = f"{'Attribute':<25}" + "".join(f"{n:>15}" for n in players)
print(header)
print("-" * len(header))
for attr in DEF_ATTRS:
    row = f"{attr:<25}"
    for name in players:
        val = results[name].get(attr, 0)
        row += f"{val:>15}"
    print(row)
print()

# Also print non-defense attrs to make sure we didn't break them
print("\n=== Luka Full Attributes ===")
for k, v in sorted(results['Luka Doncic'].items()):
    print(f"  {k:<28} {v}")
