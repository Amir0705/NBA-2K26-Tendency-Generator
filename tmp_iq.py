from src.pipeline import TendencyPipeline
pipe = TendencyPipeline()
players = [
    (1629029, 'Luka'),
    (203935, 'Smart'),
    (201939, 'Curry'),
    (2544, 'LeBron'),
    (203999, 'Jokic'),
    (203507, 'Giannis'),
    (1630162, 'Edwards'),
    (201935, 'Harden'),
]
for pid, name in players:
    try:
        r = pipe.generate(pid, season='2024-25')
        a = r.get('attributes', {})
        f = r.get('features', {})
        siq = a.get('shot_iq')
        hdiq = a.get('help_defense_iq')
        ts = f.get('ts_pct', 0)
        efg = f.get('efg_pct', 0)
        usg = f.get('usage_rate', 0)
        tov = f.get('tov_pct_proxy', 0)
        blk = f.get('blk_per_game', 0)
        dreb = f.get('dreb_per36', 0)
        stl = f.get('stl_per_game', 0)
        mn = f.get('min_per_game', 0)
        age = f.get('age', 0)
        print(f"{name:10s} shot_iq={siq:3d} help_def_iq={hdiq:3d}  ts={ts:.3f} efg={efg:.3f} usg={usg:.1f} tov%={tov:.3f} blk={blk} dreb={dreb:.1f} stl={stl} min={mn} age={age}")
    except Exception as e:
        print(f"{name:10s} ERROR: {e}")
