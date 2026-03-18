"""Ingest PBP Stats contextual profiles for modern seasons (2016-17 → 2024-25).

Strategy: The PBPStatsClient fetches the *full season table* at once and
caches it.  We warm the totals + shot-summary caches with 2 HTTP requests
per season, then iterate every player in the totals table (no additional
requests) and write a row to pbp_profiles.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Activate PBP Stats (required by PBPStatsClient.enabled check).
os.environ.setdefault("N2K_PBP_ONLY", "1")

from scripts.ingest.db import (
    create_schema,
    get_connection,
    is_done,
    log_done,
    log_empty,
    log_error,
)

PBP_MIN_YEAR = 2016
PBP_MAX_YEAR = 2024   # 2024-25

PBP_PROFILE_COLUMNS = [
    "player_id",
    "season",
    "gp",
    "min_pg",
    "pts_pg",
    "fga_pg",
    "fgm_pg",
    "fg_pct",
    "fg3a_pg",
    "fg3m_pg",
    "fg3_pct",
    "fta_pg",
    "ftm_pg",
    "ft_pct",
    "ast_pg",
    "reb_pg",
    "oreb_pg",
    "dreb_pg",
    "stl_pg",
    "blk_pg",
    "tov_pg",
    "pf_pg",
    "catch_and_shoot_three_rate",
    "pull_up_three_rate",
    "unassisted_two_rate",
    "assisted_2pt_pct",
    "assisted_3pt_pct",
    "putback_rate",
    "rim_fga_share_pbp",
    "mid_fga_share_pbp",
    "at_rim_frequency",
    "short_mid_frequency",
    "long_mid_frequency",
    "corner3_frequency",
    "arc3_frequency",
    "at_rim_accuracy",
    "short_mid_accuracy",
    "long_mid_accuracy",
    "corner3_accuracy",
    "arc3_accuracy",
    "pbp_usage_rate",
    "total_poss",
    "off_poss",
    "seconds_per_poss_off",
    "second_chance_off_poss_rate",
    "live_ball_turnover_pct",
    "shooting_fouls_drawn_pct",
    "three_pt_fouls_drawn_pct",
    "bad_pass_turnovers",
    "lost_ball_turnovers",
    "blocks_recovered_pct",
    "offensive_fouls",
    "loose_ball_fouls",
    "shooting_fouls",
    "offensive_fouls_drawn",
    "loose_ball_fouls_drawn",
    "def_poss",
    "name",
    "team_id",
    "team_abbreviation",
]


def _season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _f(val: object, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def _get_profile_with_retry(
    pbp: object,
    player_id: int,
    season: str,
    retries: int = 2,
    pause: float = 0.35,
) -> dict:
    """Fetch a player PBP profile with small retry for transient 5xx/timeouts."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # PBPStatsClient has get_player_profile; duck-typed here.
            return pbp.get_player_profile(player_id, season=season)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(pause * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    return {}


def ingest_pbp(
    *,
    min_year: int = PBP_MIN_YEAR,
    max_year: int = PBP_MAX_YEAR,
    force: bool = False,
) -> None:
    from src.ingest.cache import Cache  # noqa: PLC0415
    from src.ingest.pbpstats_client import PBPStatsClient  # noqa: PLC0415

    cache_dir = os.path.join(
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "data", "cache",
    )
    pbp = PBPStatsClient(cache=Cache(cache_dir))

    conn = get_connection()
    create_schema(conn)

    seasons = [_season(y) for y in range(max_year, min_year - 1, -1)]
    print(f"[pbp_profiles] {len(seasons)} seasons ({seasons[-1]} → {seasons[0]})")

    for season in seasons:
        if not force and is_done(conn, "pbp_profiles", season):
            print(f"  {season}  ✓ already done")
            continue

        print(f"  {season}  warming totals cache …", end="", flush=True)
        try:
            totals_table = pbp.get_player_totals_table(season)
            time.sleep(0.4)
        except Exception as exc:  # noqa: BLE001
            log_error(conn, "pbp_profiles", season, f"totals fetch: {exc}")
            print(f" ERROR (totals): {exc}")
            continue

        if not totals_table:
            log_empty(conn, "pbp_profiles", season)
            print(" → empty")
            continue

        # Warm shot-summary cache via a dummy player_id probe (uses cached result
        # for all subsequent players in the same season).
        first_pid = int(totals_table[0].get("EntityId") or 0)
        print(" shot-summary …", end="", flush=True)
        try:
            _get_profile_with_retry(pbp, first_pid, season=season)
            time.sleep(0.4)
        except Exception as exc:  # noqa: BLE001
            log_error(conn, "pbp_profiles", season, f"shot-summary fetch: {exc}")
            print(f" ERROR (shot-summary): {exc}")
            continue

        rows: list[tuple] = []
        for totals_row in totals_table:
            pid = int(totals_row.get("EntityId") or 0)
            if pid <= 0:
                continue
            try:
                p = _get_profile_with_retry(pbp, pid, season=season)
            except Exception:  # noqa: BLE001
                continue
            if not p:
                continue

            rows.append((
                pid,
                season,
                _f(p.get("gp")),
                _f(p.get("min")),
                _f(p.get("pts")),
                _f(p.get("fga")),
                _f(p.get("fgm")),
                _f(p.get("fg_pct")),
                _f(p.get("fg3a")),
                _f(p.get("fg3m")),
                _f(p.get("fg3_pct")),
                _f(p.get("fta")),
                _f(p.get("ftm")),
                _f(p.get("ft_pct")),
                _f(p.get("ast")),
                _f(p.get("reb")),
                _f(p.get("oreb")),
                _f(p.get("dreb")),
                _f(p.get("stl")),
                _f(p.get("blk")),
                _f(p.get("tov")),
                _f(p.get("pf")),
                _f(p.get("catch_and_shoot_three_rate")),
                _f(p.get("pull_up_three_rate")),
                _f(p.get("unassisted_two_rate")),
                _f(p.get("assisted_2pt_pct")),
                _f(p.get("assisted_3pt_pct")),
                _f(p.get("putback_rate")),
                _f(p.get("rim_fga_share_pbp")),
                _f(p.get("mid_fga_share_pbp")),
                _f(p.get("at_rim_frequency")),
                _f(p.get("short_mid_frequency")),
                _f(p.get("long_mid_frequency")),
                _f(p.get("corner3_frequency")),
                _f(p.get("arc3_frequency")),
                _f(p.get("at_rim_accuracy")),
                _f(p.get("short_mid_accuracy")),
                _f(p.get("long_mid_accuracy")),
                _f(p.get("corner3_accuracy")),
                _f(p.get("arc3_accuracy")),
                _f(p.get("pbp_usage_rate")),
                _f(p.get("total_poss")),
                _f(p.get("off_poss")),
                _f(p.get("seconds_per_poss_off")),
                _f(p.get("second_chance_off_poss_rate")),
                _f(p.get("live_ball_turnover_pct")),
                _f(p.get("shooting_fouls_drawn_pct")),
                _f(p.get("three_pt_fouls_drawn_pct")),
                _f(p.get("bad_pass_turnovers")),
                _f(p.get("lost_ball_turnovers")),
                _f(p.get("blocks_recovered_pct")),
                _f(p.get("offensive_fouls")),
                _f(p.get("loose_ball_fouls")),
                _f(p.get("shooting_fouls")),
                _f(p.get("offensive_fouls_drawn")),
                _f(p.get("loose_ball_fouls_drawn")),
                _f(p.get("def_poss")),
                str(p.get("name") or ""),
                int(_f(p.get("team_id"))),
                str(p.get("team_abbreviation") or ""),
            ))

        if not rows:
            log_empty(conn, "pbp_profiles", season)
            print(" → empty (no valid profiles)")
            continue

        if len(rows[0]) != len(PBP_PROFILE_COLUMNS):
            raise ValueError(
                f"pbp_profiles row width mismatch: got {len(rows[0])}, expected {len(PBP_PROFILE_COLUMNS)}"
            )

        column_sql = ", ".join(PBP_PROFILE_COLUMNS)
        value_sql = ", ".join(["%s"] * len(PBP_PROFILE_COLUMNS))
        update_columns = [c for c in PBP_PROFILE_COLUMNS if c not in {"player_id", "season"}]
        update_sql = ",\n                ".join(f"{c} = excluded.{c}" for c in update_columns)

        conn.executemany(
            f"""
            INSERT INTO pbp_profiles ({column_sql})
            VALUES ({value_sql})
            ON CONFLICT (player_id, season) DO UPDATE SET
                {update_sql},
                fetched_at = now()
            """,
            rows,
        )
        log_done(conn, "pbp_profiles", season)
        print(f" → saved {len(rows)} players")

    conn.close()
    print("[pbp_profiles] complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PBP profiles into DuckDB warehouse")
    parser.add_argument("--min-year", type=int, default=PBP_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=PBP_MAX_YEAR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ingest_pbp(min_year=args.min_year, max_year=args.max_year, force=args.force)
