"""Quick validation for the new driving dunk formula (Luka + Ja)."""
from __future__ import annotations

import socket
from typing import Any

from src.attributes.calculator import AttributeCalculator
from src.features.feature_engine import FeatureEngine
from src.ingest.nba_api_client import NBAApiClient

socket.setdefaulttimeout(20)


def _norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _age_phys(age: float) -> float:
    val = 1.0 - 0.012 * max(0.0, age - 27.0) - 0.008 * max(0.0, 22.0 - age)
    return max(0.75, min(1.0, val))


def _driving_dunk_components(features: dict[str, Any]) -> dict[str, float]:
    pos = str(features.get("position", "SF")).upper()
    height = float(features.get("height_inches", 78))
    weight = float(features.get("weight_lbs", 220))
    age = float(features.get("age", 25))
    min_pg = float(features.get("min_per_game", 1))

    rim_pressure = float(features.get("zone_fga_rate_ra", 0)) + float(features.get("zone_fga_rate_paint", 0))
    ra_pct = float(features.get("zone_fg_pct_ra", 0))
    ra_per36 = float(features.get("zone_fga_per36_ra", 0))
    transition_pos = float(features.get("transition_possessions", 0))
    iso_pos = float(features.get("isolation_possessions", 0))
    pnr_bh = float(features.get("pick_and_roll_ball_handler_possessions", 0))
    fta_rate = float(features.get("fta_rate", 0))

    age_phys = _age_phys(age)
    pos_scale = {"PG": 1.00, "SG": 0.95, "SF": 0.82, "PF": 0.72, "C": 0.60}.get(pos, 0.90)

    c_rim = _norm(rim_pressure, 0.10, 0.55)
    c_ra_pct = _norm(ra_pct, 0.56, 0.76)
    c_ra_vol = _norm(ra_per36, 0.60, 7.5)
    c_transition = _norm(transition_pos, 0.20, 4.5)
    c_creation = _norm((iso_pos + pnr_bh) * 36.0 / max(min_pg, 1.0), 0.60, 9.5)
    c_fta = _norm(fta_rate, 0.12, 0.50)
    c_size = 0.60 * _norm(height, 75, 84) + 0.40 * _norm(weight, 180, 260)
    c_pop = max(0.0, min(1.0, 0.60 * c_size + 0.40 * age_phys))
    c_burst = (
        0.55 * (1.0 - _norm(height, 75, 84))
        + 0.45 * (1.0 - _norm(weight, 180, 260))
    )
    c_burst = max(0.0, min(1.0, c_burst))

    raw = (
        0.24 * c_rim
        + 0.18 * c_ra_vol
        + 0.16 * c_transition
        + 0.14 * c_creation
        + 0.10 * c_burst
        + 0.08 * c_pop
        + 0.06 * c_fta
        + 0.04 * c_ra_pct
    )

    opp = 0.45 * c_rim + 0.30 * c_ra_vol + 0.25 * c_transition
    gate = 1.0 if opp >= 0.30 else (0.55 + 0.45 * max(0.0, opp / 0.30))

    return {
        "pos_scale": pos_scale,
        "c_rim": c_rim,
        "c_ra_pct": c_ra_pct,
        "c_ra_vol": c_ra_vol,
        "c_transition": c_transition,
        "c_creation": c_creation,
        "c_fta": c_fta,
        "c_burst": c_burst,
        "c_pop": c_pop,
        "raw": raw,
        "opp": opp,
        "gate": gate,
        "raw_out_of_100": 100.0 * raw * pos_scale * gate,
    }


def run_player(name: str, client: NBAApiClient, engine: FeatureEngine, calc: AttributeCalculator) -> None:
    print("\n" + "=" * 64)
    print(name)
    print("=" * 64)

    matches = client.search_player(name)
    if not matches:
        print("Player not found")
        return

    player_id = int(matches[0]["player_id"])
    features = engine.build_multiseasonal_features(player_id, s0_season="2025-26")
    attrs = calc.calculate(features, tendencies={})
    comp = _driving_dunk_components(features)

    print(
        f"position={features.get('position')}  gp(blended)={features.get('gp')}  "
        f"height={features.get('height_inches')}  weight={features.get('weight_lbs')}"
    )
    print(
        f"rim_pressure={float(features.get('zone_fga_rate_ra',0))+float(features.get('zone_fga_rate_paint',0)):.3f}  "
        f"ra_pct={float(features.get('zone_fg_pct_ra',0)):.3f}  ra_per36={float(features.get('zone_fga_per36_ra',0)):.3f}"
    )
    print(
        f"transition_poss={float(features.get('transition_possessions',0)):.3f}  "
        f"iso+pnr_bh={float(features.get('isolation_possessions',0))+float(features.get('pick_and_roll_ball_handler_possessions',0)):.3f}  "
        f"fta_rate={float(features.get('fta_rate',0)):.3f}"
    )

    print("components:")
    for key in [
        "pos_scale", "c_rim", "c_ra_pct", "c_ra_vol", "c_transition", "c_creation",
        "c_fta", "c_burst", "c_pop", "raw", "opp", "gate", "raw_out_of_100",
    ]:
        print(f"  {key:>14}: {comp[key]:.4f}")

    print(f"NEW driving_dunk rating: {attrs['driving_dunk']}")


def main() -> None:
    client = NBAApiClient(cache_dir="data/cache")
    engine = FeatureEngine(client)
    calc = AttributeCalculator()

    for name in ("Luka Doncic", "Ja Morant"):
        try:
            run_player(name, client, engine, calc)
        except Exception as exc:  # noqa: BLE001
            print("\n" + "=" * 64)
            print(name)
            print("=" * 64)
            print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
