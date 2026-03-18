"""Continuously retry failed pbp_shots ingest entities until error count is zero.

This runner targets only failed player-season entities in ingest_log and retries
get-shots with additional attempts/backoff. It keeps running until all errors are
cleared, then marks season-level pbp_shots entities as done when applicable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

from scripts.ingest.db import get_connection, log_done, log_empty, log_error
from scripts.ingest.ingest_pbp_shots import _f, _i, _row_value, _s, _shots_with_retry
from src.ingest.cache import Cache
from src.ingest.pbpstats_client import PBPStatsClient


def _entity_error_count(conn: Any) -> int:
    row = conn.execute(
        """
        select count(*)
        from ingest_log
        where task='pbp_shots'
          and status='error'
          and position(chr(58) in entity) > 0
        """
    ).fetchone()
    return int(row[0] if row else 0)


def _season_error_count(conn: Any) -> int:
    row = conn.execute(
        """
        select count(*)
        from ingest_log
        where task='pbp_shots'
          and status='error'
          and position(chr(58) in entity) = 0
        """
    ).fetchone()
    return int(row[0] if row else 0)


def _next_error_entities(conn: Any, limit: int) -> list[str]:
    rows = conn.execute(
        """
        select entity
        from ingest_log
        where task='pbp_shots'
          and status='error'
          and position(chr(58) in entity) > 0
        order by fetched_at asc
        limit %s
        """,
        [limit],
    ).fetchall()
    return [str(r[0]) for r in rows]


def _upsert_player_shots(conn: Any, payload_rows: list[tuple[Any, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO player_shots (
            shot_uid, player_id, season, game_id, period,
            time_remaining, team_id, opponent_team_id,
            x, y, shot_value, shot_distance, made
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (shot_uid) DO UPDATE SET
            game_id          = excluded.game_id,
            period           = excluded.period,
            time_remaining   = excluded.time_remaining,
            team_id          = excluded.team_id,
            opponent_team_id = excluded.opponent_team_id,
            x                = excluded.x,
            y                = excluded.y,
            shot_value       = excluded.shot_value,
            shot_distance    = excluded.shot_distance,
            made             = excluded.made,
            fetched_at       = now()
        """,
        payload_rows,
    )


def _reconcile_season_markers(conn: Any) -> None:
    seasons = conn.execute(
        """
        select distinct split_part(entity, ':', 1) as season
        from ingest_log
        where task='pbp_shots'
          and position(chr(58) in entity) > 0
        """
    ).fetchall()

    for row in seasons:
        season = str(row[0])
        remaining = conn.execute(
            """
            select count(*)
            from ingest_log
            where task='pbp_shots'
              and status='error'
              and entity like %s
            """,
            [f"{season}:%"],
        ).fetchone()[0]
        if int(remaining) == 0:
            log_done(conn, "pbp_shots", season)
        else:
            log_error(conn, "pbp_shots", season, f"{remaining} player-season errors remaining")


def _retry_one_entity(
    conn: Any,
    pbp: PBPStatsClient,
    entity: str,
    attempts_per_entity: int,
) -> tuple[bool, int]:
    season, pid_text = entity.split(":", 1)
    player_id = int(pid_text)

    last_exc: Exception | None = None
    rows: list[dict[str, Any]] = []

    for attempt in range(max(1, attempts_per_entity)):
        try:
            rows = _shots_with_retry(pbp, player_id, season)
            if rows:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        if attempt < max(1, attempts_per_entity) - 1:
            time.sleep(0.7 * (attempt + 1))

    if not rows and last_exc is not None:
        log_error(conn, "pbp_shots", entity, str(last_exc))
        return False, 0

    if not rows:
        log_empty(conn, "pbp_shots", entity)
        return True, 0

    payload_rows: list[tuple[Any, ...]] = []
    for idx, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue

        payload_json = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        shot_uid = hashlib.sha1(
            f"{season}:{player_id}:{idx}:{payload_json}".encode("utf-8")
        ).hexdigest()

        payload_rows.append(
            (
                shot_uid,
                player_id,
                season,
                _s(_row_value(raw, "game_id", "GameId", default="")),
                _i(_row_value(raw, "period", "Period", default=0), 0),
                _s(_row_value(raw, "time_remaining", "timeRemaining", "Clock", default="")),
                _i(_row_value(raw, "team_id", "teamId", "OffTeamId", default=0), 0),
                _i(_row_value(raw, "opponent_team_id", "opponentTeamId", "DefTeamId", default=0), 0),
                _f(_row_value(raw, "x", "loc_x", "LocX", default=0.0), 0.0),
                _f(_row_value(raw, "y", "loc_y", "LocY", default=0.0), 0.0),
                _i(_row_value(raw, "shot_value", "shotValue", "ShotValue", default=0), 0),
                _f(_row_value(raw, "shot_distance", "shotDistance", "ShotDistance", default=0.0), 0.0),
                _i(_row_value(raw, "made", "is_made", "made_flag", "shot_made_flag", default=0), 0),
            )
        )

    if not payload_rows:
        log_empty(conn, "pbp_shots", entity)
        return True, 0

    _upsert_player_shots(conn, payload_rows)
    log_done(conn, "pbp_shots", entity)
    return True, len(payload_rows)


def run_until_100(batch_size: int, attempts_per_entity: int, round_sleep_seconds: float) -> None:
    os.environ.setdefault("N2K_PBP_ONLY", "1")

    conn = get_connection()
    pbp = PBPStatsClient(cache=Cache(os.path.join("data", "cache")))

    round_no = 0
    while True:
        round_no += 1
        entity_errors_before = _entity_error_count(conn)
        season_errors_before = _season_error_count(conn)

        print(
            f"round={round_no} entity_errors={entity_errors_before} season_errors={season_errors_before}",
            flush=True,
        )

        if entity_errors_before <= 0:
            _reconcile_season_markers(conn)
            season_errors_after = _season_error_count(conn)
            print(
                f"complete entity_errors=0 season_errors={season_errors_after}",
                flush=True,
            )
            break

        entities = _next_error_entities(conn, batch_size)
        if not entities:
            _reconcile_season_markers(conn)
            print("no_error_entities_found", flush=True)
            break

        done = 0
        failed = 0
        inserted = 0

        for entity in entities:
            ok, row_count = _retry_one_entity(conn, pbp, entity, attempts_per_entity)
            if ok:
                done += 1
                inserted += row_count
            else:
                failed += 1

        _reconcile_season_markers(conn)

        entity_errors_after = _entity_error_count(conn)
        season_errors_after = _season_error_count(conn)

        print(
            (
                "round_summary "
                f"done={done} failed={failed} inserted={inserted} "
                f"entity_errors={entity_errors_after} season_errors={season_errors_after}"
            ),
            flush=True,
        )

        if round_sleep_seconds > 0:
            time.sleep(round_sleep_seconds)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry pbp_shots errors until 100% completion")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--attempts-per-entity", type=int, default=5)
    parser.add_argument("--round-sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    run_until_100(
        batch_size=args.batch_size,
        attempts_per_entity=args.attempts_per_entity,
        round_sleep_seconds=args.round_sleep_seconds,
    )
