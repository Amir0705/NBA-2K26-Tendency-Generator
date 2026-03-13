from src.pipeline import TendencyPipeline

pipe = TendencyPipeline()

players = [
    ("Stephen Curry", "2024-25"),
    ("LeBron James", "2024-25"),
    ("Luka Doncic", "2024-25"),
    ("Nikola Jokic", "2024-25"),
    ("Rudy Gobert", "2024-25"),
    ("Giannis Antetokounmpo", "2024-25"),
    ("Anthony Davis", "2024-25"),
    ("Joel Embiid", "2024-25"),
    ("Anthony Edwards", "2024-25"),
    ("Bam Adebayo", "2024-25"),
    ("Karl-Anthony Towns", "2024-25"),
    ("DeMar DeRozan", "2024-25"),
    ("Julius Randle", "2024-25"),
]

post_keys = [
    "post_up", "shoot_from_post", "post_hook_left", "post_hook_right",
    "post_fade_left", "post_fade_right", "post_up_and_under",
    "post_face_up", "post_back_down", "post_drive", "post_spin",
    "post_drop_step", "post_hop_step",
]

for name, season in players:
    try:
        result = pipe.generate(name, season)
        t = result["tendencies"]
        f = result["features"]
        pos = f.get("position", "?")
        ht = f.get("height_inches", 0)
        wt = f.get("weight_lbs", 0)
        zra = f.get("zone_fga_rate_ra", 0)
        zpaint = f.get("zone_fga_rate_paint", 0)
        ast = f.get("ast_per36", 0)
        post_poss = f.get("post_up_possessions", 0)
        post_ppp = f.get("post_up_ppp", 0)
        zmid_l = f.get("zone_fga_rate_mid_left", 0)
        zmid_c = f.get("zone_fga_rate_mid_center", 0)
        zmid_r = f.get("zone_fga_rate_mid_right", 0)
        zmid = zmid_l + zmid_c + zmid_r
        print(f"\n{name} ({pos}) ht={ht} wt={wt} zra={zra:.3f} zpaint={zpaint:.3f} zmid={zmid:.3f} ast={ast:.1f} post_poss={post_poss:.2f}")
        vals = {k: round(t.get(k, -1)) for k in post_keys}
        print(f"  post_up={vals['post_up']}  shoot_post={vals['shoot_from_post']}  "
              f"hook_L={vals['post_hook_left']}  hook_R={vals['post_hook_right']}  "
              f"fade_L={vals['post_fade_left']}  fade_R={vals['post_fade_right']}  "
              f"up_under={vals['post_up_and_under']}")
        print(f"  face_up={vals['post_face_up']}  back_down={vals['post_back_down']}  "
              f"drive={vals['post_drive']}  spin={vals['post_spin']}  "
              f"drop_step={vals['post_drop_step']}  hop_step={vals['post_hop_step']}")
    except Exception as e:
        print(f"{name}: ERROR: {e}")
