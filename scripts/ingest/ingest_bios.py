"""Ingest player bio/physical data (CommonPlayerInfo) for all players in warehouse.

Reads the unique set of player_ids already in player_seasons, skips any that
already have height_in populated in player_info, and fetches the remainder.

This is the slowest step: ~4 500 players × 0.7 s ≈ 52 min on a cold run.
After the first full run the resume logic makes reruns nearly instant.
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.ingest.db import (
    create_schema,
    get_connection,
    is_done,
    log_done,
    log_empty,
    log_error,
)

_RATE_LIMIT = 0.7
_HEIGHT_RE = re.compile(r"^(\d+)-(\d+)$")


def _parse_height(raw: str) -> float | None:
    """Convert '6-7' → 79.0 inches; return None if unparseable."""
    m = _HEIGHT_RE.match(str(raw or "").strip())
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return float(feet * 12 + inches)


def _f(val: object, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def _get_player_info_with_retry(
    commonplayerinfo: object,
    player_id: int,
    retries: int = 2,
    pause: float = 0.75,
) -> list[dict]:
    """Fetch one player bio row with retry for transient network failures."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            ep = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30)  # type: ignore[attr-defined]
            return ep.get_normalized_dict().get("CommonPlayerInfo", [])
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(pause * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    return []


def ingest_bios(
    *,
    force: bool = False,
    rate_limit: float = _RATE_LIMIT,
    limit: int | None = None,
) -> None:
    from nba_api.stats.endpoints import commonplayerinfo  # noqa: PLC0415

    conn = get_connection()
    create_schema(conn)

    # Players already having height populated are not reprocessed (unless force).
    if force:
        pids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT player_id FROM player_seasons ORDER BY player_id"
            ).fetchall()
        ]
    else:
        pids = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT ps.player_id
                FROM player_seasons ps
                LEFT JOIN player_info pi ON pi.player_id = ps.player_id
                WHERE pi.height_in IS NULL
                ORDER BY ps.player_id
                """
            ).fetchall()
        ]

    if limit is not None:
        pids = pids[:limit]

    print(f"[bios] {len(pids)} players to fetch (est. ~{len(pids) * rate_limit / 60:.0f} min)")

    done_count = 0
    skip_count = 0
    err_count = 0

    for pid in pids:
        task_key = str(pid)
        if not force and is_done(conn, "bios", task_key):
            skip_count += 1
            continue

        try:
            rows = _get_player_info_with_retry(commonplayerinfo, pid)
            time.sleep(rate_limit)

            if not rows:
                log_empty(conn, "bios", task_key)
                continue

            r = rows[0]
            height = _parse_height(r.get("HEIGHT") or "")
            weight_raw = r.get("WEIGHT") or ""
            try:
                weight = float(str(weight_raw).strip()) if weight_raw else None
            except ValueError:
                weight = None

            birthdate = str(r.get("BIRTHDATE") or "").strip()[:10]  # "YYYY-MM-DD…" → trim
            position = str(r.get("POSITION") or "").strip()
            full_name = str(r.get("DISPLAY_FIRST_LAST") or "").strip()

            conn.execute(
                """
                INSERT INTO player_info (player_id, full_name, position, height_in, weight_lbs, birthdate)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET
                    full_name  = CASE WHEN excluded.full_name  != '' THEN excluded.full_name
                                      ELSE player_info.full_name  END,
                    position   = CASE WHEN excluded.position   != '' THEN excluded.position
                                      ELSE player_info.position   END,
                    height_in  = COALESCE(excluded.height_in,  player_info.height_in),
                    weight_lbs = COALESCE(excluded.weight_lbs, player_info.weight_lbs),
                    birthdate  = CASE WHEN excluded.birthdate  != '' THEN excluded.birthdate
                                      ELSE player_info.birthdate  END,
                    fetched_at = now()
                """,
                [pid, full_name, position, height, weight, birthdate],
            )
            log_done(conn, "bios", task_key)
            done_count += 1
            if done_count % 100 == 0:
                print(f"  … {done_count} bios fetched so far")

        except Exception as exc:  # noqa: BLE001
            log_error(conn, "bios", task_key, str(exc))
            err_count += 1
            print(f"  pid={pid} ERROR: {exc}")

    conn.close()
    print(f"[bios] complete. fetched={done_count}, skipped={skip_count}, errors={err_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest player bios into DuckDB warehouse")
    parser.add_argument("--force", action="store_true", help="Re-fetch all, ignore cache")
    parser.add_argument("--rate-limit", type=float, default=_RATE_LIMIT)
    parser.add_argument("--limit", type=int, default=None, help="Max players to fetch (testing)")
    args = parser.parse_args()

    ingest_bios(force=args.force, rate_limit=args.rate_limit, limit=args.limit)
