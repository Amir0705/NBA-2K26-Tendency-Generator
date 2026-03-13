"""Quick validation for the new standing dunk formula (Luka + Rudy)."""
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


def _standing_dunk_components(features: dict[str, Any]) -> dict[str, float]:
    pos = str(features.get("position", "SF")).upper()
    height = float(features.get("height_inches", 78))
    weight = float(features.get("weight_lbs", 220))
    min_pg = float(features.get("min_per_game", 0))
    pnr_roll = float(features.get("pick_and_roll_rollman_possessions", 0))
    oreb_pg = float(features.get("oreb_per36", 0))
    blk_per36 = float(features.get("blk_per36", 0))
    ra_rate = float(features.get("zone_fga_rate_ra", 0))
    ra_per36 = float(features.get("zone_fga_per36_ra", 0))

    pos_scale = {"PG": 0.20, "SG": 0.30, "SF": 0.55, "PF": 0.82, "C": 1.00}.get(pos, 0.55)
    sd_size = 0.55 * _norm(height, 78, 84) + 0.45 * _norm(weight, 205, 275)
    sd_roll_p36 = pnr_roll * 36.0 / max(min_pg, 1.0)
    c_roll = _norm(sd_roll_p36, 0.20, 7.0)
    c_oreb = _norm(oreb_pg, 0.40, 4.0)
    c_ra_vol = _norm(ra_per36, 0.40, 7.0)
    c_ra_rate = _norm(ra_rate, 0.08, 0.55)
    c_blk = _norm(blk_per36, 0.20, 2.8)

    raw = (
        0.34 * sd_size
        + 0.20 * c_roll
        + 0.18 * c_oreb
        + 0.14 * c_ra_vol
        + 0.08 * c_ra_rate
        + 0.06 * c_blk
    )
    opp = 0.55 * c_ra_rate + 0.45 * c_roll
    gate = 1.0 if opp >= 0.20 else (0.55 + 0.45 * max(0.0, opp / 0.20))

    return {
        "pos_scale": pos_scale,
        "size": sd_size,
        "roll_p36": sd_roll_p36,
        "c_roll": c_roll,
        "c_oreb": c_oreb,
        "c_ra_vol": c_ra_vol,
        "c_ra_rate": c_ra_rate,
        "c_blk": c_blk,
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
    comp = _standing_dunk_components(features)

    print(
        f"position={features.get('position')}  gp(blended)={features.get('gp')}  "
        f"height={features.get('height_inches')}  weight={features.get('weight_lbs')}"
    )
    print(
        f"ra_rate={float(features.get('zone_fga_rate_ra', 0)):.3f}  "
        f"ra_per36={float(features.get('zone_fga_per36_ra', 0)):.3f}  "
        f"oreb_per36={float(features.get('oreb_per36', 0)):.3f}  "
        f"roll_poss={float(features.get('pick_and_roll_rollman_possessions', 0)):.3f}"
    )

    print("components:")
    for key in [
        "pos_scale", "size", "roll_p36", "c_roll", "c_oreb", "c_ra_vol",
        "c_ra_rate", "c_blk", "raw", "opp", "gate", "raw_out_of_100",
    ]:
        print(f"  {key:>14}: {comp[key]:.4f}")

    print(f"NEW standing_dunk rating: {attrs['standing_dunk']}")


def main() -> None:
    client = NBAApiClient(cache_dir="data/cache")
    engine = FeatureEngine(client)
    calc = AttributeCalculator()

    for name in ("Luka Doncic", "Rudy Gobert"):
        try:
            run_player(name, client, engine, calc)
        except Exception as exc:  # noqa: BLE001
            print("\n" + "=" * 64)
            print(name)
            print("=" * 64)
            print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
