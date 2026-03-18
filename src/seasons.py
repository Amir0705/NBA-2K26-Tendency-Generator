"""Season parsing and era routing helpers."""
from __future__ import annotations

import re

SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")
MIN_START_YEAR = 2000
MAX_START_YEAR = 2025
PBP_MIN_START_YEAR = 2016
DEFAULT_SEASON = "2025-26"


def season_start_year(season: str) -> int:
    """Return the season start year for a YYYY-YY season string."""
    text = str(season or "").strip()
    match = SEASON_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid season format '{season}'. Expected YYYY-YY.")

    start_year = int(match.group(1))
    end_suffix = int(match.group(2))
    if (start_year + 1) % 100 != end_suffix:
        raise ValueError(f"Invalid season value '{season}'. End year must match start year + 1.")
    return start_year


def is_supported_season(season: str) -> bool:
    """Return whether season is in the supported range [2000-01, 2025-26]."""
    try:
        year = season_start_year(season)
    except ValueError:
        return False
    return MIN_START_YEAR <= year <= MAX_START_YEAR


def normalize_season(season: str, *, default: str = DEFAULT_SEASON) -> str:
    """Validate and normalize season string, returning default when empty."""
    text = str(season or "").strip() or default
    _ = season_start_year(text)
    if not is_supported_season(text):
        raise ValueError(
            f"Unsupported season '{text}'. Supported range is {MIN_START_YEAR}-01 through {MAX_START_YEAR}-{str(MAX_START_YEAR + 1)[-2:]}."
        )
    return text


def data_mode_for_season(season: str) -> str:
    """Return 'pbp' for modern seasons and 'legacy' for historical seasons."""
    start = season_start_year(season)
    return "pbp" if start >= PBP_MIN_START_YEAR else "legacy"


def prior_seasons(season: str, count: int = 2) -> list[str]:
    """Return up to `count` valid prior seasons within supported range."""
    start = season_start_year(season)
    out: list[str] = []
    for i in range(1, max(0, count) + 1):
        year = start - i
        if year < MIN_START_YEAR:
            break
        out.append(f"{year}-{str(year + 1)[-2:]}")
    return out
