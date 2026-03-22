"""Ingest per-game + advanced box scores for all seasons (2000-01 → 2025-26).

Source: nba_api  LeagueDashPlayerStats  (PerGame  +  Advanced)
Output: player_seasons table  +  player_info  (name/position only — bios come later)

One API call pair per season = 52 total network requests for 26 seasons.
"""
from __future__ import annotations

import os
import sys
import time

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.ingest.db import (
    create_schema,
    get_connection,
    is_done,
    log_done,
    log_error,
)

# seconds between each nba_api request (keep well below ban threshold)
_RATE_LIMIT = 0.7

MIN_START_YEAR = 2000
MAX_START_YEAR = 2025   # 2025-26 season currently available for basic stats ingest


def _season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _f(val: object, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def ingest_box_scores(
    *,
    min_year: int = MIN_START_YEAR,
    max_year: int = MAX_START_YEAR,
    force: bool = False,
    rate_limit: float = _RATE_LIMIT,
) -> None:
    """Ingest box scores for every season in [min_year, max_year]."""
    conn = get_connection()
    create_schema(conn)

    seasons = [_season(y) for y in range(max_year, min_year - 1, -1)]
    print(f"[box_scores] {len(seasons)} seasons to process ({seasons[-1]} → {seasons[0]})")

    for season in seasons:
        task_key = season
        if not force and is_done(conn, "box_scores", task_key):
            print(f"  {season}  ✓ already done")
            continue

        print(f"  {season}  fetching base stats …", end="", flush=True)
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats  # noqa: PLC0415

            # ── Per-game base stats ────────────────────────────────────
            ep_base = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                timeout=35,
            )
            time.sleep(rate_limit)
            base_rows = ep_base.get_normalized_dict().get("LeagueDashPlayerStats", [])

            # ── Advanced stats (usg, ts, efg, ortg, drtg) ─────────────
            print(" adv …", end="", flush=True)
            ep_adv = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=35,
            )
            time.sleep(rate_limit)
            adv_rows = ep_adv.get_normalized_dict().get("LeagueDashPlayerStats", [])
            adv_by_pid = {
                int(r.get("PLAYER_ID") or 0): r
                for r in adv_rows
                if r.get("PLAYER_ID")
            }

            inserted = 0
            season_rows: list[tuple] = []
            info_rows: list[tuple] = []

            for row in base_rows:
                pid = int(row.get("PLAYER_ID") or 0)
                if pid <= 0:
                    continue

                adv = adv_by_pid.get(pid, {})
                fga = _f(row.get("FGA"))
                fg3a = _f(row.get("FG3A"))
                fg3a_rate = (fg3a / fga) if fga > 0 else 0.0

                # USG_PCT from nba_api is already 0-100; normalise to 0-1
                usg_raw = _f(adv.get("USG_PCT"))
                usg = usg_raw / 100.0 if usg_raw > 1.0 else usg_raw

                season_rows.append((
                    pid,
                    season,
                    int(row.get("GP") or 0),
                    _f(row.get("MIN")),
                    _f(row.get("PTS")),
                    fga,
                    _f(row.get("FGM")),
                    _f(row.get("FG_PCT")),
                    fg3a,
                    _f(row.get("FG3M")),
                    _f(row.get("FG3_PCT")),
                    fg3a_rate,
                    _f(row.get("FTA")),
                    _f(row.get("FTM")),
                    _f(row.get("FT_PCT")),
                    _f(row.get("OREB")),
                    _f(row.get("DREB")),
                    _f(row.get("REB")),
                    _f(row.get("AST")),
                    _f(row.get("STL")),
                    _f(row.get("BLK")),
                    _f(row.get("TOV")),
                    _f(row.get("PF")),
                    usg,
                    _f(adv.get("TS_PCT")),
                    _f(adv.get("EFG_PCT")),
                    _f(adv.get("OFF_RATING")),
                    _f(adv.get("DEF_RATING")),
                ))

                name = str(row.get("PLAYER_NAME") or "").strip()
                pos = str(row.get("PLAYER_POSITION") or "").strip()
                if "/" in pos:
                    pos = pos.split("/")[0].strip()
                info_rows.append((pid, name, pos))
                inserted += 1

            # Batch upsert player_seasons
            conn.executemany(
                """
                INSERT INTO player_seasons (
                    player_id, season, gp, min_pg, pts_pg, fga_pg, fgm_pg, fg_pct,
                    fg3a_pg, fg3m_pg, fg3_pct, fg3a_rate,
                    fta_pg, ftm_pg, ft_pct, oreb_pg, dreb_pg, reb_pg,
                    ast_pg, stl_pg, blk_pg, tov_pg, pf_pg,
                    usg_pct, ts_pct, efg_pct, ortg, drtg
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, season) DO UPDATE SET
                    gp = excluded.gp, min_pg = excluded.min_pg,
                    pts_pg = excluded.pts_pg, fga_pg = excluded.fga_pg,
                    fgm_pg = excluded.fgm_pg, fg_pct = excluded.fg_pct,
                    fg3a_pg = excluded.fg3a_pg, fg3m_pg = excluded.fg3m_pg,
                    fg3_pct = excluded.fg3_pct, fg3a_rate = excluded.fg3a_rate,
                    fta_pg = excluded.fta_pg, ftm_pg = excluded.ftm_pg,
                    ft_pct = excluded.ft_pct, oreb_pg = excluded.oreb_pg,
                    dreb_pg = excluded.dreb_pg, reb_pg = excluded.reb_pg,
                    ast_pg = excluded.ast_pg, stl_pg = excluded.stl_pg,
                    blk_pg = excluded.blk_pg, tov_pg = excluded.tov_pg,
                    pf_pg = excluded.pf_pg, usg_pct = excluded.usg_pct,
                    ts_pct = excluded.ts_pct, efg_pct = excluded.efg_pct,
                    ortg = excluded.ortg, drtg = excluded.drtg,
                    fetched_at = now()
                """,
                season_rows,
            )

            # Upsert player_info (name + position only; bios added later)
            conn.executemany(
                """
                INSERT INTO player_info (player_id, full_name, position)
                VALUES (%s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET
                    full_name = CASE WHEN excluded.full_name != '' THEN excluded.full_name
                                     ELSE player_info.full_name END,
                    position  = CASE WHEN excluded.position  != '' THEN excluded.position
                                      ELSE player_info.position  END,
                    fetched_at = now()
                """,
                info_rows,
            )

            log_done(conn, "box_scores", task_key)
            print(f" → saved {inserted} players")

        except Exception as exc:  # noqa: BLE001
            log_error(conn, "box_scores", task_key, str(exc))
            print(f" ERROR: {exc}")

    conn.close()
    print("[box_scores] complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NBA box scores into DuckDB warehouse")
    parser.add_argument("--min-year", type=int, default=MIN_START_YEAR)
    parser.add_argument("--max-year", type=int, default=MAX_START_YEAR)
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already done")
    parser.add_argument("--rate-limit", type=float, default=_RATE_LIMIT)
    args = parser.parse_args()

    ingest_box_scores(
        min_year=args.min_year,
        max_year=args.max_year,
        force=args.force,
        rate_limit=args.rate_limit,
    )
