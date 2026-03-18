"""Ingest team rosters (CommonTeamRoster) for all seasons 2000-01 → 2024-25.

Strategy: 30 current franchises × 25 seasons = up to 750 nba_api requests.
Historical franchise abbreviations (VAN, SEA, NJN, CHH, NOH, NOK) share a
team_id with their successor; the API returns rows for the seasons they
existed so non-existent combos return empty data (logged as 'empty').

Each row is unique on (team_abbr, season, player_id).
"""
from __future__ import annotations

import os
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

MIN_START_YEAR = 2000
MAX_START_YEAR = 2024

# All franchises that played at any point in 2000-01 to 2024-25.
# Historical abbreviations are included; we use the abbreviation that was
# current during each era so the roster data is queryable by that abbr.
_ALL_ABBRS: list[str] = [
    "ATL", "BOS", "BKN", "NJN",
    "CHA", "CHH",
    "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND",
    "LAC", "LAL",
    "MEM", "VAN",
    "MIA", "MIL", "MIN",
    "NOP", "NOH", "NOK",
    "NYK", "OKC", "SEA",
    "ORL", "PHI", "PHX", "POR",
    "SAC", "SAS", "TOR", "UTA",
    "WAS",
]

# Eras each historical abbreviation was actually used.
_ABBR_ERA: dict[str, tuple[int, int]] = {
    "CHH": (2000, 2004),   # Charlotte Hornets → relocated to NO 2002-03 (last roster 2001-02)
    "VAN": (2000, 2000),   # Vancouver Grizzlies → Memphis from 2001-02
    "SEA": (2000, 2007),   # Seattle SuperSonics → OKC from 2008-09
    "NJN": (2000, 2011),   # New Jersey Nets → Brooklyn from 2012-13
    "NOH": (2002, 2012),   # New Orleans Hornets (various names)
    "NOK": (2005, 2007),   # New Orleans/Oklahoma City Hornets (Katrina era)
}

_RATE_LIMIT = 0.7


def _season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _abbr_active(abbr: str, year: int) -> bool:
    """Return True if this abbreviation was in use during start-year = year."""
    era = _ABBR_ERA.get(abbr)
    if era is None:
        return True   # current franchise — always active
    return era[0] <= year <= era[1]


def _get_team_roster_with_retry(
    commonteamroster: object,
    team_id: int,
    season: str,
    retries: int = 2,
    pause: float = 0.75,
) -> list[dict]:
    """Fetch one team-season roster with retry for transient network failures."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            ep = commonteamroster.CommonTeamRoster(  # type: ignore[attr-defined]
                team_id=team_id,
                season=season,
                timeout=30,
            )
            return ep.get_normalized_dict().get("CommonTeamRoster", [])
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(pause * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    return []


def ingest_rosters(
    *,
    min_year: int = MIN_START_YEAR,
    max_year: int = MAX_START_YEAR,
    force: bool = False,
    rate_limit: float = _RATE_LIMIT,
) -> None:
    from nba_api.stats.endpoints import commonteamroster  # noqa: PLC0415

    conn = get_connection()
    create_schema(conn)

    years = list(range(min_year, max_year + 1))

    total_calls = sum(
        1
        for abbr in _ALL_ABBRS
        for year in years
        if _abbr_active(abbr, year)
    )
    print(f"[rosters] up to {total_calls} requests ({_season(min_year)} → {_season(max_year)})")

    for abbr in _ALL_ABBRS:
        for year in range(max_year, min_year - 1, -1):
            if not _abbr_active(abbr, year):
                continue

            season = _season(year)
            task_key = f"{abbr}:{season}"

            if not force and is_done(conn, "rosters", task_key):
                continue

            try:
                rows_raw = _get_team_roster_with_retry(
                    commonteamroster,
                    team_id=_team_id(abbr),
                    season=season,
                )
                time.sleep(rate_limit)

                if not rows_raw:
                    log_empty(conn, "rosters", task_key)
                    print(f"  {task_key}  empty")
                    continue

                rows: list[tuple] = []
                for r in rows_raw:
                    pid_raw = r.get("PLAYER_ID") or r.get("PlayerID")
                    if pid_raw is None:
                        continue
                    try:
                        pid = int(pid_raw)
                    except (TypeError, ValueError):
                        continue
                    name = str(r.get("PLAYER") or r.get("PlayerName") or "").strip()
                    pos = str(r.get("POSITION") or "").strip()
                    rows.append((abbr, season, pid, name, pos))

                conn.executemany(
                    """
                    INSERT INTO team_rosters (team_abbr, season, player_id, full_name, position)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_abbr, season, player_id) DO UPDATE SET
                        full_name  = excluded.full_name,
                        position   = excluded.position,
                        fetched_at = now()
                    """,
                    rows,
                )
                log_done(conn, "rosters", task_key)
                print(f"  {task_key}  → {len(rows)} players")

            except Exception as exc:  # noqa: BLE001
                log_error(conn, "rosters", task_key, str(exc))
                print(f"  {task_key}  ERROR: {exc}")

    conn.close()
    print("[rosters] complete.")


# Inline the mapping (same as nba_api_client.py) so this script is self-contained.
_TEAM_ID_BY_ABBR: dict[str, int] = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHH": 1610612766, "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742,
    "DEN": 1610612743, "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745,
    "IND": 1610612754, "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "VAN": 1610612763, "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NOH": 1610612740, "NOK": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "SEA": 1610612760, "ORL": 1610612753, "PHI": 1610612755,
    "PHX": 1610612756, "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764, "NJN": 1610612751,
}


def _team_id(abbr: str) -> int:
    tid = _TEAM_ID_BY_ABBR.get(abbr.upper())
    if tid is None:
        raise KeyError(f"Unknown team abbreviation: {abbr!r}")
    return tid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NBA team rosters into DuckDB warehouse")
    parser.add_argument("--min-year", type=int, default=MIN_START_YEAR)
    parser.add_argument("--max-year", type=int, default=MAX_START_YEAR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=_RATE_LIMIT)
    args = parser.parse_args()

    ingest_rosters(
        min_year=args.min_year,
        max_year=args.max_year,
        force=args.force,
        rate_limit=args.rate_limit,
    )
