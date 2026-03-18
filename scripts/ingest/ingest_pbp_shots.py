"""Ingest raw PBP shot-event rows for modern seasons (2016-17 → 2024-25).

This captures the get-shots payload at player-season level so warehouse reads can
fully replace `get_player_shots()` without live API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ.setdefault("N2K_PBP_ONLY", "1")

from scripts.ingest.db import (  # noqa: E402
    create_schema,
    get_connection,
    is_done,
    log_done,
    log_empty,
    log_error,
)

PBP_MIN_YEAR = 2016
PBP_MAX_YEAR = 2024


def _season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _s(v: Any) -> str:
    return str(v or "").strip()


def _row_value(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return default


def _shots_with_retry(
    pbp: object,
    player_id: int,
    season: str,
    retries: int = 2,
) -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return pbp.get_player_shots(player_id=player_id, season=season)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(0.6 * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    return []


def ingest_pbp_shots(
    *,
    min_year: int = PBP_MIN_YEAR,
    max_year: int = PBP_MAX_YEAR,
    force: bool = False,
    player_limit: int | None = None,
) -> None:
    from src.ingest.cache import Cache  # noqa: PLC0415
    from src.ingest.pbpstats_client import PBPStatsClient  # noqa: PLC0415

    cache_dir = os.path.join(
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "data",
        "cache",
    )
    pbp = PBPStatsClient(cache=Cache(cache_dir))

    conn = get_connection()
    create_schema(conn)

    seasons = [_season(y) for y in range(max_year, min_year - 1, -1)]
    print(f"[pbp_shots] {len(seasons)} seasons ({seasons[-1]} → {seasons[0]})")

    for season in seasons:
        print(f"  {season} totals …", end="", flush=True)
        try:
            totals = pbp.get_player_totals_table(season)
        except Exception as exc:  # noqa: BLE001
            log_error(conn, "pbp_shots", season, f"totals fetch: {exc}")
            print(f" ERROR (totals): {exc}")
            continue

        if not totals:
            log_empty(conn, "pbp_shots", season)
            print(" → empty")
            continue

        pids: list[int] = []
        for row in totals:
            pid = _i(row.get("EntityId"), 0)
            if pid > 0:
                pids.append(pid)
        pids = sorted(set(pids))
        if player_limit is not None and player_limit > 0:
            pids = pids[:player_limit]
        print(f" players={len(pids)}")

        season_inserted = 0
        season_players_done = 0
        season_players_empty = 0
        season_players_error = 0

        for pid in pids:
            entity = f"{season}:{pid}"
            if (not force) and is_done(conn, "pbp_shots", entity):
                continue

            try:
                rows = _shots_with_retry(pbp, pid, season)
                if force:
                    conn.execute(
                        "DELETE FROM player_shots WHERE season = %s AND player_id = %s",
                        [season, pid],
                    )

                if not rows:
                    log_empty(conn, "pbp_shots", entity)
                    season_players_empty += 1
                    continue

                payload_rows: list[tuple] = []
                for idx, raw in enumerate(rows):
                    if not isinstance(raw, dict):
                        continue

                    payload_json = json.dumps(raw, sort_keys=True, separators=(",", ":"))
                    uid_src = f"{season}:{pid}:{idx}:{payload_json}"
                    shot_uid = hashlib.sha1(uid_src.encode("utf-8")).hexdigest()

                    payload_rows.append(
                        (
                            shot_uid,
                            pid,
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
                    season_players_empty += 1
                    continue

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
                log_done(conn, "pbp_shots", entity)
                season_inserted += len(payload_rows)
                season_players_done += 1

            except Exception as exc:  # noqa: BLE001
                log_error(conn, "pbp_shots", entity, str(exc))
                season_players_error += 1

        # Mark season complete only if no per-player errors remain.
        if season_players_error == 0:
            log_done(conn, "pbp_shots", season)
        else:
            log_error(
                conn,
                "pbp_shots",
                season,
                f"{season_players_error} player(s) failed",
            )

        print(
            f"    {season}: shots={season_inserted}, players_done={season_players_done}, "
            f"empty={season_players_empty}, errors={season_players_error}"
        )

    conn.close()
    print("[pbp_shots] complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw PBP player shots into DuckDB")
    parser.add_argument("--min-year", type=int, default=PBP_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=PBP_MAX_YEAR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--player-limit", type=int, default=None)
    args = parser.parse_args()

    ingest_pbp_shots(
        min_year=args.min_year,
        max_year=args.max_year,
        force=args.force,
        player_limit=args.player_limit,
    )
