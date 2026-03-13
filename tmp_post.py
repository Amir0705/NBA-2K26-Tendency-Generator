from src.pipeline import TendencyPipeline
pipe = TendencyPipeline()
players = [
    (1629029, 'Luka'),
    (2544, 'LeBron'),
    (203999, 'Jokic'),
    (203507, 'Giannis'),
    (201935, 'Harden'),
    (201939, 'Curry'),
    (1630162, 'Edwards'),
]
for pid, name in players:
    try:
        r = pipe.generate(pid, season='2024-25')
        a = r.get('attributes', {})
        f = r.get('features', {})
        t = r.get('tendencies', {})
        ph = a.get('post_hook')
        pf = a.get('post_fade')
        pc = a.get('post_control')
        post_pos = f.get('post_up_possessions', 0)
        post_ppp = f.get('post_up_ppp', 0)
        mid_pct = (f.get('zone_fg_pct_mid_left',0) + f.get('zone_fg_pct_mid_center',0) + f.get('zone_fg_pct_mid_right',0)) / 3
        t_hook_l = t.get('post_hook_left', 0)
        t_hook_r = t.get('post_hook_right', 0)
        t_fade = t.get('post_fade', 0)
        t_post_up = t.get('post_up', 25)
        t_spin = t.get('post_spin', 0)
        t_drive = t.get('post_drive', 0)
        height = f.get('height_inches', 0)
        weight = f.get('weight_lbs', 0)
        pos = f.get('position', '')
        print(f"{name:10s} hook={ph:3d} fade={pf:3d} ctrl={pc:3d} | pos={pos} ht={height} wt={weight}")
        print(f"           post_pos={post_pos:.1f} post_ppp={post_ppp:.2f} mid%={mid_pct:.3f}")
        print(f"           t_hook_l={t_hook_l} t_hook_r={t_hook_r} t_fade={t_fade} t_post_up={t_post_up} t_spin={t_spin} t_drive={t_drive}")
        print()
    except Exception as e:
        print(f"{name:10s} ERROR: {e}")
