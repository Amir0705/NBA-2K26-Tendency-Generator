"""
Test script: Multi-season driving layup attribute preview.
Players: Luka Doncic, Rudy Gobert
Seasons: 2025-26 (s0), 2024-25 (s1), 2023-24 (s2)

Dynamic weights:
  s0 games < 20  → [0.35, 0.45, 0.20]
  s0 games 20-40 → [0.45, 0.35, 0.20]
  s0 games 40+   → [0.55, 0.30, 0.15]
"""
import socket
socket.setdefaulttimeout(5)  # short timeout — we only want cached data

import sqlite3

from src.ingest.nba_api_client import NBAApiClient
from src.features.feature_engine import FeatureEngine

# ── helpers ──────────────────────────────────────────────────────────────────

SEASONS = ["2025-26", "2024-25", "2023-24"]

NON_BLEND_KEYS = {
    "position", "height_inches", "weight_lbs", "age",
    "has_shot_chart", "low_minutes", "games_played",
    "sub_zone_distribution_close", "sub_zone_distribution_mid",
    "sub_zone_distribution_three", "drive_right_bias",
    "is_pg", "is_sg", "is_sf", "is_pf", "is_c",
}


def get_weights(s0_gp: int) -> tuple[float, float, float]:
    """Dynamic weights based on s0 games played."""
    if s0_gp < 20:
        return 0.35, 0.45, 0.20
    if s0_gp < 40:
        return 0.45, 0.35, 0.20
    return 0.55, 0.30, 0.15


def blend_features(f0: dict, f1: dict, f2: dict) -> dict:
    """Weighted blend of 3 season feature dicts."""
    s0_gp = int(f0.get("gp", f0.get("games_played", 0)))
    w0, w1, w2 = get_weights(s0_gp)

    # Renormalise if seasons are missing
    available = []
    if f0:
        available.append((w0, f0))
    if f1:
        available.append((w1, f1))
    if f2:
        available.append((w2, f2))

    total_w = sum(w for w, _ in available)

    blended = {}
    # Start from s0 for non-numeric/non-blend keys
    base = f0 or (f1 or f2)
    for k in NON_BLEND_KEYS:
        if k in base:
            blended[k] = base[k]

    # Blend all numeric keys
    all_keys = set()
    for _, f in available:
        all_keys.update(f.keys())

    for k in all_keys:
        if k in NON_BLEND_KEYS:
            continue
        vals = []
        weights = []
        for w, f in available:
            v = f.get(k)
            if v is not None:
                try:
                    vals.append(float(v))
                    weights.append(w)
                except (TypeError, ValueError):
                    pass
        if vals:
            norm = sum(weights)
            blended[k] = sum(v * w for v, w in zip(vals, weights)) / norm

    return blended


def _norm(value: float, lo: float, hi: float) -> float:
    """Linear 0–1 normalisation clamped to [0, 1]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _smooth_penalty(rim_pressure: float, cutoff: float = 0.10, full_ok: float = 0.15) -> float:
    """Returns 1.0 when rim_pressure >= full_ok, ramps down to 0.60 at 0."""
    if rim_pressure >= full_ok:
        return 1.0
    t = max(0.0, rim_pressure / full_ok)
    return 0.60 + 0.40 * t


POSITION_SCALE = {"PG": 1.00, "SG": 0.97, "SF": 0.90, "PF": 0.83, "C": 0.70}

# Final rating is 25–99 scale
ATTR_MIN = 25.0
ATTR_MAX = 99.0


def driving_layup_score(f: dict) -> dict:
    """
    Proposed multi-signal driving layup formula.
    Returns dict with score and all component values for inspection.
    """
    pos = str(f.get("position", "SF")).upper()

    ra_pct      = float(f.get("zone_fg_pct_ra", 0))
    ra_rate     = float(f.get("zone_fga_rate_ra", 0))
    paint_rate  = float(f.get("zone_fga_rate_paint", 0))
    rim_pressure = ra_rate + paint_rate
    fta_rate    = float(f.get("fta_rate", 0))
    ft_pct      = float(f.get("ft_pct", 0))
    ra_per36    = float(f.get("zone_fga_per36_ra", 0))

    min_pg  = float(f.get("min_per_game", 1))
    iso_pos = float(f.get("isolation_possessions", 0))
    pnr_bh  = float(f.get("pick_and_roll_ball_handler_possessions", 0))
    # Convert from per-game possessions to per-36 equivalent
    creation_p36 = (iso_pos + pnr_bh) * 36.0 / max(min_pg, 1.0)

    # Component scores (0–1 normalised)
    c_ra_pct      = _norm(ra_pct,         0.52, 0.78)   # elite: 78%+ (true elite RA finishers)
    c_rim_press   = _norm(rim_pressure,   0.08, 0.50)   # elite: 50%+ of FGA at rim
    c_fta_rate    = _norm(fta_rate,       0.10, 0.45)   # elite: 0.45 FTA/FGA
    c_ft_pct      = _norm(ft_pct,         0.60, 0.85)   # elite finisher touch
    c_creation    = _norm(creation_p36,   0.5,  8.0)    # playmaker rim creation
    c_ra_vol      = _norm(ra_per36,       0.5,  6.0)    # absolute rim volume

    raw = (
        0.38 * c_ra_pct
        + 0.22 * c_rim_press
        + 0.15 * c_fta_rate
        + 0.12 * c_ft_pct
        + 0.08 * c_creation
        + 0.05 * c_ra_vol
    )

    # Position ceiling
    pos_scale = POSITION_SCALE.get(pos, 0.90)

    # Volume penalty (smooth)
    vol_penalty = _smooth_penalty(rim_pressure)

    adjusted = raw * pos_scale * vol_penalty

    # Scale to 25–99
    score = ATTR_MIN + adjusted * (ATTR_MAX - ATTR_MIN)
    score = max(ATTR_MIN, min(ATTR_MAX, score))

    return {
        "score": round(score, 1),
        "raw_0_1": round(raw, 4),
        "pos_scale": pos_scale,
        "vol_penalty": round(vol_penalty, 3),
        "--- inputs ---": "---",
        "ra_pct":       round(ra_pct, 3),
        "rim_pressure": round(rim_pressure, 3),
        "fta_rate":     round(fta_rate, 3),
        "ft_pct":       round(ft_pct, 3),
        "creation_p36": round(creation_p36, 2),
        "ra_per36":     round(ra_per36, 2),
        "--- components (0-1) ---": "---",
        "c_ra_pct":    round(c_ra_pct, 3),
        "c_rim_press": round(c_rim_press, 3),
        "c_fta_rate":  round(c_fta_rate, 3),
        "c_ft_pct":    round(c_ft_pct, 3),
        "c_creation":  round(c_creation, 3),
        "c_ra_vol":    round(c_ra_vol, 3),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def _is_season_cached(player_id: int, season: str) -> bool:
    """Return True only if player_stats AND shot_chart are both in the local cache."""
    try:
        conn = sqlite3.connect("data/cache/nba_cache.db")
        for key in (f"player_stats:{player_id}:{season}", f"shot_chart:{player_id}:{season}"):
            row = conn.execute("SELECT 1 FROM cache WHERE key=?", (key,)).fetchone()
            if not row:
                conn.close()
                return False
        conn.close()
        return True
    except Exception:
        return False


def run_player(name: str, client: NBAApiClient, engine: FeatureEngine) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Find player
    matches = client.search_player(name)
    if not matches:
        print(f"  ERROR: player not found")
        return
    player_id = matches[0]["player_id"]

    # Fetch features per season — only use cached data, skip on any network error
    season_features: list[dict] = []
    for season in SEASONS:
        if not _is_season_cached(player_id, season):
            print(f"  [{season}]  SKIPPED (not in cache)")
            season_features.append({})
            continue
        try:
            f = engine.build_features(player_id, season=season)
            gp = int(f.get("gp", f.get("games_played", 0)))
            print(f"  [{season}]  GP={gp:>3}  RA_pct={f.get('zone_fg_pct_ra',0):.3f}"
                  f"  rim_press={f.get('zone_fga_rate_ra',0)+f.get('zone_fga_rate_paint',0):.3f}"
                  f"  FTA_rate={f.get('fta_rate',0):.3f}  FT%={f.get('ft_pct',0):.3f}")
            season_features.append(f)
        except Exception as e:
            print(f"  [{season}]  FAILED: {type(e).__name__}: {e}")
            season_features.append({})

    f0, f1, f2 = (season_features + [{}, {}, {}])[:3]

    # --- Per-season individual scores ---
    print(f"\n  Per-season driving layup scores:")
    for season, fdict in zip(SEASONS, season_features):
        if fdict:
            res = driving_layup_score(fdict)
            print(f"    {season}  →  {res['score']:>5.1f}  (raw={res['raw_0_1']:.3f})")
        else:
            print(f"    {season}  →  N/A")

    # --- Blended ---
    s0_gp = int(f0.get("gp", f0.get("games_played", 0)))
    w0, w1, w2 = get_weights(s0_gp)
    print(f"\n  Weights: s0(2025-26)={w0}  s1(2024-25)={w1}  s2(2023-24)={w2}  "
          f"(s0 GP={s0_gp})")

    blended = blend_features(f0, f1, f2)
    result = driving_layup_score(blended)

    print(f"\n  ── BLENDED DRIVING LAYUP ──")
    for k, v in result.items():
        if "---" in str(k):
            print(f"\n    {v}")
        else:
            print(f"    {k:<22} {v}")

    print(f"\n  ★  FINAL DRIVING LAYUP RATING: {result['score']:.1f}")


def main() -> None:
    cache_dir = "data/cache"
    client = NBAApiClient(cache_dir=cache_dir)
    engine = FeatureEngine(nba_client=client)

    # NOTE: Rudy Gobert not in cache (API timeout).
    # Using 3 contrasting profiles instead for formula validation:
    #   Luka Doncic   -- high-usage driving guard/wing
    #   Giannis       -- elite power driver as a big
    #   Evan Mobley   -- mobile center who rarely drives (should score low)
    run_player("Luka Doncic", client, engine)
    run_player("Giannis Antetokounmpo", client, engine)
    run_player("Evan Mobley", client, engine)


if __name__ == "__main__":
    main()
