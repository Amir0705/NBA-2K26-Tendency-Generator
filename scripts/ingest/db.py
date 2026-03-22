"""PostgreSQL warehouse — schema creation and connection management.

All ingestion scripts import from here so the DB URL and schema stay
in one place.  Set DATABASE_URL to a Postgres connection string before
running any ingest script.
"""
from __future__ import annotations

import os
from typing import Any


class PostgresWarehouseConnection:
    """Thin adapter exposing DuckDB-like helpers on top of psycopg."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, query: str, params: list[Any] | tuple[Any, ...] | None = None) -> Any:
        return self._conn.execute(query, params)

    def executemany(self, query: str, params_seq: list[tuple[Any, ...]] | list[list[Any]]) -> Any:
        with self._conn.cursor() as cur:
            cur.executemany(query, params_seq)
            return cur

    def close(self) -> None:
        self._conn.close()

# ---------------------------------------------------------------------------
# DDL — each statement separated by the magic comment so we can split easily
# ---------------------------------------------------------------------------
_TABLES: list[str] = [
    # ------------------------------------------------------------------
    # player_info  — one row per player (bio / physical data)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_info (
        player_id   INTEGER PRIMARY KEY,
        full_name   TEXT    NOT NULL,
        position    TEXT    DEFAULT '',
        height_in   REAL,        -- stored as decimal inches (e.g. 79.0 for 6-7)
        weight_lbs  REAL,
        birthdate   TEXT    DEFAULT '',
        age         INTEGER,
        years_pro   INTEGER,
        draft       TEXT    DEFAULT '',
        school      TEXT    DEFAULT '',
        fetched_at  TIMESTAMP DEFAULT current_timestamp
    )
    """,

    # Backward-compatible adds for existing DBs created before demographics columns.
    """
    ALTER TABLE player_info
    ADD COLUMN IF NOT EXISTS age INTEGER
    """,
    """
    ALTER TABLE player_info
    ADD COLUMN IF NOT EXISTS years_pro INTEGER
    """,
    """
    ALTER TABLE player_info
    ADD COLUMN IF NOT EXISTS draft TEXT DEFAULT ''
    """,
    """
    ALTER TABLE player_info
    ADD COLUMN IF NOT EXISTS school TEXT DEFAULT ''
    """,

    # ------------------------------------------------------------------
    # player_seasons  — per-game stats per player × season
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_seasons (
        player_id   INTEGER NOT NULL,
        season      TEXT    NOT NULL,  -- '2024-25'
        gp          INTEGER,
        min_pg      REAL,
        pts_pg      REAL,
        fga_pg      REAL,
        fgm_pg      REAL,
        fg_pct      REAL,
        fg3a_pg     REAL,
        fg3m_pg     REAL,
        fg3_pct     REAL,
        fg3a_rate   REAL,           -- fg3a / fga  (0-1)
        fta_pg      REAL,
        ftm_pg      REAL,
        ft_pct      REAL,
        oreb_pg     REAL,
        dreb_pg     REAL,
        reb_pg      REAL,
        ast_pg      REAL,
        stl_pg      REAL,
        blk_pg      REAL,
        tov_pg      REAL,
        pf_pg       REAL,
        usg_pct     REAL,           -- 0-1 fraction
        ts_pct      REAL,
        efg_pct     REAL,
        ortg        REAL,
        drtg        REAL,
        fetched_at  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (player_id, season)
    )
    """,

    # ------------------------------------------------------------------
    # pbp_profiles  — PBP Stats contextual signals (modern seasons ≥ 2016-17)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS pbp_profiles (
        player_id                   INTEGER NOT NULL,
        season                      TEXT    NOT NULL,
        -- base per-game outputs mirrored from get_player_profile
        gp                          REAL DEFAULT 0,
        min_pg                      REAL DEFAULT 0,
        pts_pg                      REAL DEFAULT 0,
        fga_pg                      REAL DEFAULT 0,
        fgm_pg                      REAL DEFAULT 0,
        fg_pct                      REAL DEFAULT 0,
        fg3a_pg                     REAL DEFAULT 0,
        fg3m_pg                     REAL DEFAULT 0,
        fg3_pct                     REAL DEFAULT 0,
        fta_pg                      REAL DEFAULT 0,
        ftm_pg                      REAL DEFAULT 0,
        ft_pct                      REAL DEFAULT 0,
        ast_pg                      REAL DEFAULT 0,
        reb_pg                      REAL DEFAULT 0,
        oreb_pg                     REAL DEFAULT 0,
        dreb_pg                     REAL DEFAULT 0,
        stl_pg                      REAL DEFAULT 0,
        blk_pg                      REAL DEFAULT 0,
        tov_pg                      REAL DEFAULT 0,
        pf_pg                       REAL DEFAULT 0,
        -- shot-type rates (0-1 fractions of FGA)
        catch_and_shoot_three_rate  REAL DEFAULT 0,
        pull_up_three_rate          REAL DEFAULT 0,
        unassisted_two_rate         REAL DEFAULT 0,
        assisted_2pt_pct            REAL DEFAULT 0,   -- 0-100
        assisted_3pt_pct            REAL DEFAULT 0,   -- 0-100
        putback_rate                REAL DEFAULT 0,
        rim_fga_share_pbp           REAL DEFAULT 0,
        mid_fga_share_pbp           REAL DEFAULT 0,
        -- shot-zone frequencies (0-1 fractions of FGA)
        at_rim_frequency            REAL DEFAULT 0,
        short_mid_frequency         REAL DEFAULT 0,
        long_mid_frequency          REAL DEFAULT 0,
        corner3_frequency           REAL DEFAULT 0,
        arc3_frequency              REAL DEFAULT 0,
        -- shot-zone accuracy (0-1 FG%)
        at_rim_accuracy             REAL DEFAULT 0,
        short_mid_accuracy          REAL DEFAULT 0,
        long_mid_accuracy           REAL DEFAULT 0,
        corner3_accuracy            REAL DEFAULT 0,
        arc3_accuracy               REAL DEFAULT 0,
        -- possession / tempo signals
        pbp_usage_rate              REAL DEFAULT 0,   -- 0-100
        total_poss                  REAL DEFAULT 0,
        off_poss                    REAL DEFAULT 0,
        seconds_per_poss_off        REAL DEFAULT 0,
        second_chance_off_poss_rate REAL DEFAULT 0,
        live_ball_turnover_pct      REAL DEFAULT 0,
        shooting_fouls_drawn_pct    REAL DEFAULT 0,
        three_pt_fouls_drawn_pct    REAL DEFAULT 0,
        bad_pass_turnovers          REAL DEFAULT 0,
        lost_ball_turnovers         REAL DEFAULT 0,
        blocks_recovered_pct        REAL DEFAULT 0,
        offensive_fouls             REAL DEFAULT 0,
        loose_ball_fouls            REAL DEFAULT 0,
        shooting_fouls              REAL DEFAULT 0,
        offensive_fouls_drawn       REAL DEFAULT 0,
        loose_ball_fouls_drawn      REAL DEFAULT 0,
        def_poss                    REAL DEFAULT 0,
        name                        TEXT DEFAULT '',
        team_id                     INTEGER DEFAULT 0,
        team_abbreviation           TEXT DEFAULT '',
        fetched_at                  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (player_id, season)
    )
    """,

    # ------------------------------------------------------------------
    # player_shots  — raw PBP shot-event rows (modern seasons)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_shots (
        shot_uid         TEXT PRIMARY KEY,
        player_id        INTEGER NOT NULL,
        season           TEXT    NOT NULL,
        game_id          TEXT    DEFAULT '',
        period           INTEGER DEFAULT 0,
        time_remaining   TEXT    DEFAULT '',
        team_id          INTEGER DEFAULT 0,
        opponent_team_id INTEGER DEFAULT 0,
        x                REAL DEFAULT 0,
        y                REAL DEFAULT 0,
        shot_value       INTEGER DEFAULT 0,
        shot_distance    REAL DEFAULT 0,
        made             INTEGER DEFAULT 0,
        fetched_at       TIMESTAMP DEFAULT now()
    )
    """,

    # ------------------------------------------------------------------
    # team_rosters  — one row per team × season × player
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS team_rosters (
        team_abbr   TEXT    NOT NULL,
        season      TEXT    NOT NULL,
        player_id   INTEGER NOT NULL,
        full_name   TEXT    DEFAULT '',
        position    TEXT    DEFAULT '',
        fetched_at  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (team_abbr, season, player_id)
    )
    """,

    # ------------------------------------------------------------------
    # ingest_log  — progress tracking for resumable ingestion
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        task        TEXT NOT NULL,   -- 'box_scores', 'pbp_profiles', 'pbp_shots', 'rosters', 'bios'
        entity      TEXT NOT NULL,   -- season / 'TEAM:season' / str(player_id)
        status      TEXT NOT NULL DEFAULT 'pending',  -- 'done' | 'error' | 'empty'
        error_msg   TEXT DEFAULT '',
        fetched_at  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (task, entity)
    )
    """,
]




# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_connection() -> Any:
    """Return a psycopg connection to the Postgres warehouse (autocommit)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable not set. "
            "Export it before running any ingest script."
        )
    import psycopg  # noqa: PLC0415
    conn = psycopg.connect(url, autocommit=True, prepare_threshold=None)
    return PostgresWarehouseConnection(conn)


def create_schema(conn: Any) -> None:
    """Create all warehouse tables if they do not already exist."""
    for ddl in _TABLES:
        stmt = ddl.strip()
        if stmt:
            conn.execute(stmt)


def log_done(conn: Any, task: str, entity: str) -> None:
    conn.execute(
        """
        INSERT INTO ingest_log (task, entity, status, fetched_at)
        VALUES (%s, %s, 'done', now())
        ON CONFLICT (task, entity) DO UPDATE SET
            status = 'done',
            error_msg = '',
            fetched_at = now()
        """,
        [task, entity],
    )


def log_error(conn: Any, task: str, entity: str, msg: str) -> None:
    conn.execute(
        """
        INSERT INTO ingest_log (task, entity, status, error_msg, fetched_at)
        VALUES (%s, %s, 'error', %s, now())
        ON CONFLICT (task, entity) DO UPDATE SET
            status = 'error',
            error_msg = excluded.error_msg,
            fetched_at = now()
        """,
        [task, entity, str(msg)[:1000]],
    )


def log_empty(conn: Any, task: str, entity: str) -> None:
    conn.execute(
        """
        INSERT INTO ingest_log (task, entity, status, fetched_at)
        VALUES (%s, %s, 'empty', now())
        ON CONFLICT (task, entity) DO UPDATE SET
            status = 'empty',
            fetched_at = now()
        """,
        [task, entity],
    )


def is_done(conn: Any, task: str, entity: str) -> bool:
    """Return True if this task/entity was already successfully completed."""
    row = conn.execute(
        "SELECT status FROM ingest_log WHERE task = %s AND entity = %s",
        [task, entity],
    ).fetchone()
    return row is not None and row[0] in ("done", "empty")
