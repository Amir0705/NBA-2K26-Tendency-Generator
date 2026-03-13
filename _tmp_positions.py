"""Quick check of player positions and key feature values."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline

p = TendencyPipeline()

for name in ['Luka Doncic', 'Alex Caruso', 'Anthony Edwards', 'Jayson Tatum']:
    result = p.generate(name, season='2024-25')
    f = result.get('features', {})
    print(f"{name}: pos={f.get('position')}, is_guard=PG/SG, size_big={max(1.0 if f.get('position') in ('PF','C') else 0.0, 0.60 * max(0,min(1,(f.get('height_inches',78)-79)/(84-79))) + 0.40 * max(0,min(1,(f.get('weight_lbs',220)-215)/(265-215)))):.3f}")
