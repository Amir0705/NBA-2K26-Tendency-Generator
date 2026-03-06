"""Shot-zone feature builder — maps shot chart data to 8 canonical zones."""
from __future__ import annotations

from typing import Any

# Default league-average FG% per zone used for Bayesian smoothing
_LEAGUE_AVG_FG_PCT: dict[str, float] = {
    "ra": 0.64,
    "paint": 0.40,
    "mid_left": 0.38,
    "mid_center": 0.39,
    "mid_right": 0.38,
    "corner3_left": 0.38,
    "corner3_right": 0.38,
    "above_break3": 0.36,
}

# 8 canonical zones
ZONES = list(_LEAGUE_AVG_FG_PCT.keys())


def _classify_zone(shot_zone_basic: str, shot_zone_area: str) -> str | None:
    """Map NBA shot zone strings to a canonical zone ID."""
    basic = shot_zone_basic.strip()
    area = shot_zone_area.strip()
    if basic == "Restricted Area":
        return "ra"
    if basic == "In The Paint (Non-RA)":
        return "paint"
    if basic == "Mid-Range":
        if area in ("Left Side(L)", "Left Side Center(LC)"):
            return "mid_left"
        if area in ("Right Side(R)", "Right Side Center(RC)"):
            return "mid_right"
        return "mid_center"
    if basic == "Left Corner 3":
        return "corner3_left"
    if basic == "Right Corner 3":
        return "corner3_right"
    if basic == "Above the Break 3":
        return "above_break3"
    return None  # backcourt or unknown


def _normalize_distribution(counts: dict[str, float], keys: list[str]) -> dict[str, float]:
    """Normalize a count dict to sum to 100; fall back to even split."""
    total = sum(counts.get(k, 0) for k in keys)
    if total <= 0:
        even = 100.0 / len(keys)
        return {k: even for k in keys}
    return {k: counts.get(k, 0) / total * 100 for k in keys}


class ShotZoneAnalyzer:
    """Computes shot-zone sub-tendency features from shot-chart data."""

    PRIOR_STRENGTH = 10

    def analyze(
        self, shot_chart_rows: list[dict[str, Any]], total_minutes: float
    ) -> dict[str, Any]:
        """
        Aggregate shot-chart rows into zone-level features.

        Returns a dict with keys:
          zone_fga, zone_fgm, zone_fga_rate, zone_fg_pct, zone_fga_per36,
          zone_pref_vs_league, sub_zone_distribution_close,
          sub_zone_distribution_mid, sub_zone_distribution_three
        """
        zone_fga: dict[str, int] = {z: 0 for z in ZONES}
        zone_fgm: dict[str, int] = {z: 0 for z in ZONES}

        # Sub-zone accumulators
        close_counts: dict[str, int] = {"left": 0, "middle": 0, "right": 0}
        mid_area_counts: dict[str, int] = {
            "left": 0, "left_center": 0, "center": 0, "right_center": 0, "right": 0
        }
        three_area_counts: dict[str, int] = {
            "left": 0, "left_center": 0, "center": 0, "right_center": 0, "right": 0
        }

        for shot in shot_chart_rows:
            basic = shot.get("shot_zone_basic", "")
            area = shot.get("shot_zone_area", "")
            made = int(shot.get("shot_made_flag", 0))
            loc_x = shot.get("loc_x", 0) or 0

            zone = _classify_zone(basic, area)
            if zone:
                zone_fga[zone] += 1
                zone_fgm[zone] += made

            # Sub-zone: close (ra + paint → left/middle/right)
            if basic in ("Restricted Area", "In The Paint (Non-RA)"):
                close_key = _area_to_close_key(basic, area, loc_x)
                close_counts[close_key] += 1

            # Sub-zone: mid-range (5-way area split)
            if basic == "Mid-Range":
                area_key = _area_to_mid_key(area)
                mid_area_counts[area_key] += 1

            # Sub-zone: three-point (5-way area split)
            if basic in ("Above the Break 3", "Left Corner 3", "Right Corner 3"):
                area_key = _area_to_three_key(basic, area)
                three_area_counts[area_key] += 1

        total_fga = max(sum(zone_fga.values()), 1)
        total_min = max(total_minutes, 1)

        zone_fga_rate = {z: zone_fga[z] / total_fga for z in ZONES}
        zone_fga_per36 = {z: zone_fga[z] / total_min * 36 for z in ZONES}
        zone_fg_pct = {
            z: _bayesian_smooth(
                zone_fgm[z],
                zone_fga[z],
                _LEAGUE_AVG_FG_PCT[z],
                self.PRIOR_STRENGTH,
            )
            for z in ZONES
        }
        # League-average zone rate (uniform prior)
        league_avg_rate = 1.0 / len(ZONES)
        zone_pref_vs_league = {
            z: zone_fga_rate[z] / league_avg_rate for z in ZONES
        }

        mid_keys = ["left", "left_center", "center", "right_center", "right"]
        three_keys = ["left", "left_center", "center", "right_center", "right"]
        close_keys = ["left", "middle", "right"]

        return {
            "zone_fga": zone_fga,
            "zone_fgm": zone_fgm,
            "zone_fga_rate": zone_fga_rate,
            "zone_fg_pct": zone_fg_pct,
            "zone_fga_per36": zone_fga_per36,
            "zone_pref_vs_league": zone_pref_vs_league,
            "sub_zone_distribution_close": _normalize_distribution(close_counts, close_keys),
            "sub_zone_distribution_mid": _normalize_distribution(mid_area_counts, mid_keys),
            "sub_zone_distribution_three": _normalize_distribution(three_area_counts, three_keys),
        }


def _bayesian_smooth(
    makes: int, attempts: int, prior_rate: float, prior_strength: int = 10
) -> float:
    """Empirical Bayes shrinkage toward the prior rate."""
    return (makes + prior_strength * prior_rate) / (attempts + prior_strength)


def _area_to_close_key(basic: str, area: str, loc_x: float) -> str:
    """Map a close-range shot to left/middle/right sub-zone.

    For Restricted Area shots, the NBA API always returns area="Center(C)"
    so we must use LOC_X coordinates. For Paint (Non-RA) shots, the API
    provides proper area breakdown (Left Side, Right Side, etc.), so we
    use area as primary with LOC_X fallback.
    """
    # Restricted Area: always use LOC_X (area is always "Center(C)")
    basic = basic.strip()
    if basic == "Restricted Area":
        if loc_x < -30:
            return "left"
        if loc_x > 30:
            return "right"
        return "middle"

    # Paint (Non-RA): use shot_zone_area as primary classifier
    area = area.strip()
    if area in ("Left Side(L)", "Left Side Center(LC)"):
        return "left"
    if area in ("Right Side(R)", "Right Side Center(RC)"):
        return "right"
    if area == "Center(C)":
        return "middle"

    # Fallback: use LOC_X when area is missing or unrecognized
    if loc_x < -30:
        return "left"
    if loc_x > 30:
        return "right"
    return "middle"


def _area_to_mid_key(area: str) -> str:
    area = area.strip()
    if area == "Left Side(L)":
        return "left"
    if area == "Left Side Center(LC)":
        return "left_center"
    if area == "Right Side Center(RC)":
        return "right_center"
    if area == "Right Side(R)":
        return "right"
    return "center"  # Center(C) or unknown


def _area_to_three_key(basic: str, area: str) -> str:
    if basic == "Left Corner 3":
        return "left"
    if basic == "Right Corner 3":
        return "right"
    # Above the Break 3 — use area
    area = area.strip()
    if area in ("Left Side Center(LC)", "Left Side(L)"):
        return "left_center"
    if area in ("Right Side Center(RC)", "Right Side(R)"):
        return "right_center"
    return "center"


# ---------------------------------------------------------------------------
# Legacy class kept for backward compat with existing stub interface
# ---------------------------------------------------------------------------


class ShotZoneBuilder:
    """Computes shot-zone sub-tendency features from shot-chart data."""

    def __init__(self) -> None:
        """Initialise zone definitions."""
        self._analyzer = ShotZoneAnalyzer()

    def compute_zones(
        self, shot_chart: list[dict[str, Any]], total_minutes: float = 0.0
    ) -> dict[str, float]:
        """
        Aggregate shot-chart rows into zone-level tendency scores.

        Returns a dict mapping canonical sub-zone names → score [0, 100].
        """
        result = self._analyzer.analyze(shot_chart, total_minutes)
        out: dict[str, float] = {}
        for zone, rate in result["zone_fga_rate"].items():
            out[f"zone_fga_rate_{zone}"] = rate
        for zone, pct in result["zone_fg_pct"].items():
            out[f"zone_fg_pct_{zone}"] = pct
        return out

    def distribute_from_parent(
        self, parent_value: int, zone_fractions: dict[str, float]
    ) -> dict[str, int]:
        """
        Distribute a parent tendency value across sub-zones using
        observed shot fractions from *zone_fractions*.
        """
        total_frac = sum(zone_fractions.values())
        if total_frac <= 0:
            even = parent_value // len(zone_fractions)
            return {k: even for k in zone_fractions}
        return {
            k: round(parent_value * v / total_frac)
            for k, v in zone_fractions.items()
        }
