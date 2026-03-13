"""Compare defensive signals between good and bad defenders."""
import socket
socket.setdefaulttimeout(15)
from src.pipeline import TendencyPipeline

p = TendencyPipeline()

for name in ['Luka Doncic', 'Alex Caruso', 'Anthony Edwards', 'Jayson Tatum']:
    result = p.generate(name, season='2024-25')
    f = result.get('features', {})
    print(f"{name}:")
    print(f"  stl_pg={f.get('stl_per_game',0):.1f}  stl_per36={f.get('stl_per36',0):.2f}  pctile_stl={f.get('pctile_stl',0):.2f}")
    print(f"  blk_pg={f.get('blk_per_game',0):.1f}  blk_per36={f.get('blk_per36',0):.2f}  pctile_blk={f.get('pctile_blk',0):.2f}")
    print(f"  pf_per36={f.get('pf_per36',0):.2f}  tov_per36={f.get('tov_per36',0):.2f}  tov_pct={f.get('tov_pct_proxy',0):.3f}")
    print(f"  min_pg={f.get('min_per_game',0):.1f}  usage={f.get('usage_rate',0):.1f}  ast_tov={f.get('ast_to_tov',0):.2f}")
    print(f"  dreb_per36={f.get('dreb_per36',0):.2f}  height={f.get('height_inches',0)}  weight={f.get('weight_lbs',0)}")
    print()
