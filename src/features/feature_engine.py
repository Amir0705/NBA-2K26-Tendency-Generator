"""Feature engineering: transforms raw NBA stats into a feature vector."""
from __future__ import annotations

import datetime
from typing import Any

from src.features.shot_zones import ShotZoneAnalyzer, ZONES

# Position one-hot keys
_POSITIONS = ("PG", "SG", "SF", "PF", "C")

# Map from nba_api position strings to canonical 2-letter codes
_POSITION_MAP = {
    "guard": "PG",
    "guard-forward": "SG",
    "forward-guard": "SF",
    "forward": "SF",
    "forward-center": "PF",
    "center-forward": "PF",
    "center": "C",
}


def _map_position(raw: str) -> str:
    """Map nba_api position string to canonical position code."""
    key = raw.strip().lower()
    return _POSITION_MAP.get(key, "SF")


def _height_to_inches(height_str: str) -> int:
    """Convert 'ft-in' string (e.g. '6-6') to total inches."""
    try:
        parts = height_str.replace('"', "").split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except Exception:  # noqa: BLE001
        return 78  # league-average fallback (~6-6)


def _per36(stat_per_game: float, minutes_per_game: float) -> float:
    """Compute per-36-minute rate from per-game values; avoids division by zero."""
    return stat_per_game * 36.0 / max(minutes_per_game, 1.0)


def _percentile(value: float, all_values: list[float]) -> float:
    """Return empirical percentile (0–1) of *value* in *all_values*."""
    if not all_values:
        return 0.5
    below = sum(1 for v in all_values if v < value)
    return below / len(all_values)


class FeatureEngine:
    """Transforms raw NBA stats into model-ready tendency features."""

    def __init__(self, nba_client: Any) -> None:
        """
        Initialise engine.

        Parameters
        ----------
        nba_client: NBAApiClient instance (or any duck-typed equivalent).
        """
        self._client = nba_client
        self._league_averages: list[dict] | None = None
        self._zone_analyzer = ShotZoneAnalyzer()

    def build_features(
        self, player_id: int, season: str = "2024-25"
    ) -> dict[str, Any]:
        """
        Build complete feature vector for a player.

        Parameters
        ----------
        player_id: NBA player ID.
        season:    Season string (e.g. "2024-25").

        Returns
        -------
        Feature dict ready for FormulaLayer.generate().
        """
        # Fetch raw data
        info = self._client.get_player_info(player_id)
        stats = self._client.get_player_stats(player_id, season=season)
        shot_chart = self._client.get_shot_chart(player_id, season=season)

        # --- Player info ---
        position = _map_position(info.get("position", ""))
        height_inches = _height_to_inches(info.get("height", ""))
        weight_lbs = int(info.get("weight") or 0)

        # --- Age and experience ---
        birthdate_str = info.get("birthdate", "")
        try:
            bd = datetime.datetime.strptime(birthdate_str[:10], "%Y-%m-%d")
            age = (datetime.datetime.now() - bd).days / 365.25
        except (ValueError, TypeError):
            age = 27.0  # league average fallback

        season_exp = int(info.get("season_exp", 0))
        drive_right_bias = 50.0  # default; feature exists so formula can read it

        # --- Volume stats (per game) ---
        gp = int(stats.get("gp", 0))
        min_pg = float(stats.get("min", 0.0))
        pts_pg = float(stats.get("pts", 0.0))
        fga_pg = float(stats.get("fga", 0.0))
        fg3a_pg = float(stats.get("fg3a", 0.0))
        fta_pg = float(stats.get("fta", 0.0))
        ast_pg = float(stats.get("ast", 0.0))
        reb_pg = float(stats.get("reb", 0.0))
        oreb_pg = float(stats.get("oreb", 0.0))
        dreb_pg = float(stats.get("dreb", 0.0))
        stl_pg = float(stats.get("stl", 0.0))
        blk_pg = float(stats.get("blk", 0.0))
        tov_pg = float(stats.get("tov", 0.0))
        pf_pg = float(stats.get("pf", 0.0))

        # --- Shooting efficiency ---
        fg_pct = float(stats.get("fg_pct", 0.0))
        fg3_pct = float(stats.get("fg3_pct", 0.0))
        ft_pct = float(stats.get("ft_pct", 0.0))
        efg_pct = (
            (float(stats.get("fgm", 0.0)) + 0.5 * float(stats.get("fg3m", 0.0)))
            / max(fga_pg, 0.001)
        )
        ts_pct = pts_pg / max(2 * (fga_pg + 0.44 * fta_pg), 0.001)

        # --- Per-36 rates ---
        per36 = lambda s: _per36(s, min_pg)  # noqa: E731
        pts_p36 = per36(pts_pg)
        fga_p36 = per36(fga_pg)
        fg3a_p36 = per36(fg3a_pg)
        fta_p36 = per36(fta_pg)
        ast_p36 = per36(ast_pg)
        reb_p36 = per36(reb_pg)
        oreb_p36 = per36(oreb_pg)
        dreb_p36 = per36(dreb_pg)
        stl_p36 = per36(stl_pg)
        blk_p36 = per36(blk_pg)
        tov_p36 = per36(tov_pg)
        pf_p36 = per36(pf_pg)

        # --- Derived ratios ---
        # USG% proxy: fraction of team possessions used while on court
        # ~2.2 possessions per minute is the empirical team rate
        possessions_used = fga_pg + 0.44 * fta_pg + tov_pg
        team_possessions_while_on = min_pg * 2.2  # ~2.2 poss/min for a team
        usg_pct_proxy = possessions_used / max(team_possessions_while_on, 1.0)

        league_avg_ast_p36 = 5.0  # approximate
        ast_pct_proxy = ast_p36 / max(league_avg_ast_p36, 0.001)
        tov_pct_proxy = tov_pg / max(fga_pg + 0.44 * fta_pg + tov_pg, 0.001)
        ast_to_tov = ast_pg / max(tov_pg, 0.1)
        fg3a_rate = fg3a_pg / max(fga_pg, 1)
        fta_rate = fta_pg / max(fga_pg, 1)
        oreb_pct_proxy = oreb_pg / max(reb_pg, 1)

        # --- Shot zone features ---
        total_minutes = min_pg * gp
        has_shot_chart = len(shot_chart) > 0
        zone_data = self._zone_analyzer.analyze(shot_chart, total_minutes)

        # --- League percentiles ---
        league_rows = self._get_league_averages(season)
        pctile_pts = _percentile(pts_pg, [r.get("PTS", 0) for r in league_rows])
        pctile_ast = _percentile(ast_pg, [r.get("AST", 0) for r in league_rows])
        pctile_reb = _percentile(reb_pg, [r.get("REB", 0) for r in league_rows])
        pctile_stl = _percentile(stl_pg, [r.get("STL", 0) for r in league_rows])
        pctile_blk = _percentile(blk_pg, [r.get("BLK", 0) for r in league_rows])
        league_fg3a_rates = [
            r.get("FG3A", 0) / max(r.get("FGA", 1), 1) for r in league_rows
        ]
        league_fta_rates = [
            r.get("FTA", 0) / max(r.get("FGA", 1), 1) for r in league_rows
        ]
        pctile_fg3a_rate = _percentile(fg3a_rate, league_fg3a_rates)
        pctile_fta_rate = _percentile(fta_rate, league_fta_rates)
        pctile_tov = _percentile(tov_pg, [r.get("TOV", 0) for r in league_rows])

        # --- Position one-hot ---
        pos_flags = {f"is_{p.lower()}": (position == p) for p in _POSITIONS}

        features: dict[str, Any] = {
            # Player info
            "position": position,
            "height_inches": height_inches,
            "weight_lbs": weight_lbs,
            # Age and experience
            "age": age,
            "season_exp": season_exp,
            "drive_right_bias": drive_right_bias,
            # Volume stats
            "pts_per_game": pts_pg,
            "fga_per_game": fga_pg,
            "fg3a_per_game": fg3a_pg,
            "fta_per_game": fta_pg,
            "ast_per_game": ast_pg,
            "reb_per_game": reb_pg,
            "stl_per_game": stl_pg,
            "blk_per_game": blk_pg,
            "tov_per_game": tov_pg,
            "min_per_game": min_pg,
            "gp": gp,
            # Per-36 rates
            "pts_per36": pts_p36,
            "fga_per36": fga_p36,
            "fg3a_per36": fg3a_p36,
            "fta_per36": fta_p36,
            "ast_per36": ast_p36,
            "reb_per36": reb_p36,
            "oreb_per36": oreb_p36,
            "dreb_per36": dreb_p36,
            "stl_per36": stl_p36,
            "blk_per36": blk_p36,
            "tov_per36": tov_p36,
            "pf_per36": pf_p36,
            # Shooting efficiency
            "fg_pct": fg_pct,
            "fg3_pct": fg3_pct,
            "ft_pct": ft_pct,
            "efg_pct": efg_pct,
            "ts_pct": ts_pct,
            # Derived ratios
            "usg_pct_proxy": usg_pct_proxy,
            "ast_pct_proxy": ast_pct_proxy,
            "tov_pct_proxy": tov_pct_proxy,
            "ast_to_tov": ast_to_tov,
            "fg3a_rate": fg3a_rate,
            "fta_rate": fta_rate,
            "oreb_pct_proxy": oreb_pct_proxy,
            # Shot zone features
            **{f"zone_fga_rate_{z}": zone_data["zone_fga_rate"][z] for z in ZONES},
            **{f"zone_fg_pct_{z}": zone_data["zone_fg_pct"][z] for z in ZONES},
            **{f"zone_fga_per36_{z}": zone_data["zone_fga_per36"][z] for z in ZONES},
            **{f"zone_pref_vs_league_{z}": zone_data["zone_pref_vs_league"][z] for z in ZONES},
            "sub_zone_distribution_close": zone_data["sub_zone_distribution_close"],
            "sub_zone_distribution_mid": zone_data["sub_zone_distribution_mid"],
            "sub_zone_distribution_three": zone_data["sub_zone_distribution_three"],
            # Percentiles
            "pctile_pts": pctile_pts,
            "pctile_ast": pctile_ast,
            "pctile_reb": pctile_reb,
            "pctile_stl": pctile_stl,
            "pctile_blk": pctile_blk,
            "pctile_fg3a_rate": pctile_fg3a_rate,
            "pctile_fta_rate": pctile_fta_rate,
            "pctile_tov": pctile_tov,
            # Data quality flags
            "has_shot_chart": has_shot_chart,
            "low_minutes": (gp < 5 or min_pg < 5),
            "games_played": gp,
        }
        features.update(pos_flags)
        return features

    def _get_league_averages(self, season: str) -> list[dict]:
        """Return cached league averages, fetching if necessary."""
        if self._league_averages is None:
            try:
                rows = self._client.get_league_averages(season=season)
                self._league_averages = rows if isinstance(rows, list) else []
            except Exception:  # noqa: BLE001
                self._league_averages = []
        return self._league_averages

    # Keep old interface for backward compatibility
    def normalise(self, features: dict[str, float]) -> dict[str, float]:
        """Identity pass-through (values already normalised in build_features)."""
        return features
