"""Feature engineering: transforms raw NBA stats into a feature vector."""
from __future__ import annotations

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


def _compute_age(birthdate_str: str) -> int:
    """Return player age from a YYYY-MM-DD (or ISO) birthdate string."""
    try:
        from datetime import date
        bd = birthdate_str[:10]  # 'YYYY-MM-DD...'
        parts = bd.split("-")
        born = date(int(parts[0]), int(parts[1]), int(parts[2]))
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:  # noqa: BLE001
        return 25  # league-average fallback


def _percentile(value: float, all_values: list[float]) -> float:
    """Return empirical percentile (0–1) of *value* in *all_values*."""
    if not all_values:
        return 0.5
    below = sum(1 for v in all_values if v < value)
    return below / len(all_values)


def _first_numeric(stats: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    """Return first numeric value found under the provided keys."""
    for key in keys:
        if key in stats:
            try:
                return float(stats.get(key, default))
            except (TypeError, ValueError):
                continue
    return float(default)


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
        try:
            playtypes = self._client.get_player_playtypes(player_id, season=season)
        except Exception:  # noqa: BLE001
            playtypes = {}

        # --- Player info ---
        position = _map_position(info.get("position", ""))
        height_inches = _height_to_inches(info.get("height", ""))
        weight_lbs = int(info.get("weight") or 0)

        # Age
        age = _compute_age(info.get("birthdate", ""))

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

        # Optional possession-style signals (if available in upstream stats feed)
        usage_rate_pct = _first_numeric(
            stats,
            ("usage_rate", "usage_rate_pct", "usg_pct", "USG_PCT"),
            default=usg_pct_proxy * 100.0,
        )
        isolation_possessions = float(playtypes.get("isolation_possessions", 0.0) or 0.0)
        pnr_ball_handler_possessions = float(
            playtypes.get("pick_and_roll_ball_handler_possessions", 0.0) or 0.0
        )
        pnr_rollman_possessions = float(
            playtypes.get("pick_and_roll_rollman_possessions", 0.0) or 0.0
        )
        post_up_possessions = float(playtypes.get("post_up_possessions", 0.0) or 0.0)
        cuts_possessions = float(playtypes.get("cuts", 0.0) or 0.0)
        handoff_possessions = float(playtypes.get("handoff_possessions", 0.0) or 0.0)
        spot_up_possessions = float(playtypes.get("spot_up_possessions", 0.0) or 0.0)
        off_screen_possessions = float(playtypes.get("off_screen_possessions", 0.0) or 0.0)
        transition_possessions = float(playtypes.get("transition_possessions", 0.0) or 0.0)

        # PPP efficiency data
        isolation_ppp = float(playtypes.get("isolation_ppp", 0.0) or 0.0)
        pnr_ball_handler_ppp = float(playtypes.get("pick_and_roll_ball_handler_ppp", 0.0) or 0.0)
        pnr_rollman_ppp = float(playtypes.get("pick_and_roll_rollman_ppp", 0.0) or 0.0)
        post_up_ppp = float(playtypes.get("post_up_ppp", 0.0) or 0.0)
        spot_up_ppp = float(playtypes.get("spot_up_ppp", 0.0) or 0.0)

        # Fall back to optional upstream stats fields if Synergy data is unavailable.
        if isolation_possessions <= 0.0:
            isolation_possessions = _first_numeric(
                stats,
                ("isolation_possessions", "iso_possessions", "isolation_poss"),
            )
        if pnr_ball_handler_possessions <= 0.0:
            pnr_ball_handler_possessions = _first_numeric(
                stats,
                (
                    "pick_and_roll_ball_handler_possessions",
                    "pnr_ball_handler_possessions",
                    "pnr_bh_possessions",
                ),
            )
        if pnr_rollman_possessions <= 0.0:
            pnr_rollman_possessions = _first_numeric(
                stats,
                (
                    "pick_and_roll_rollman_possessions",
                    "pnr_rollman_possessions",
                ),
            )
        if post_up_possessions <= 0.0:
            post_up_possessions = _first_numeric(
                stats,
                ("post_up_possessions", "post_ups", "post_possessions"),
            )
        if cuts_possessions <= 0.0:
            cuts_possessions = _first_numeric(
                stats,
                ("cuts", "cut_possessions", "cut_poss"),
            )
        if handoff_possessions <= 0.0:
            handoff_possessions = _first_numeric(
                stats,
                ("handoff_possessions", "handoff_poss", "dhos"),
            )
        midrange_attempts = _first_numeric(
            stats,
            ("midrange_attempts", "mid_range_attempts", "midrange_fga"),
        )

        # --- Shot zone features ---
        total_minutes = min_pg * gp
        has_shot_chart = len(shot_chart) > 0
        zone_data = self._zone_analyzer.analyze(shot_chart, total_minutes)

        # --- Drive right bias: % of close-range shots from right side (LOC_X > 0) ---
        close_shots = [
            s for s in shot_chart
            if s.get("shot_zone_basic", "") in ("Restricted Area", "In The Paint (Non-RA)")
        ]
        if close_shots:
            right_shots = sum(
                1 for s in close_shots if (s.get("loc_x", 0) or 0) > 0
            )
            drive_right_bias = 25.0 + (right_shots / len(close_shots)) * 50.0
        else:
            drive_right_bias = 50.0  # neutral fallback

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
            "age": age,
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
            "usage_rate": usage_rate_pct,
            "isolation_possessions": isolation_possessions,
            "pick_and_roll_ball_handler_possessions": pnr_ball_handler_possessions,
            "pick_and_roll_rollman_possessions": pnr_rollman_possessions,
            "post_up_possessions": post_up_possessions,
            "cuts": cuts_possessions,
            "handoff_possessions": handoff_possessions,
            "spot_up_possessions": spot_up_possessions,
            "off_screen_possessions": off_screen_possessions,
            "transition_possessions": transition_possessions,
            "isolation_ppp": isolation_ppp,
            "pick_and_roll_ball_handler_ppp": pnr_ball_handler_ppp,
            "pick_and_roll_rollman_ppp": pnr_rollman_ppp,
            "post_up_ppp": post_up_ppp,
            "spot_up_ppp": spot_up_ppp,
            "midrange_attempts": midrange_attempts,
            # Shot zone features
            **{f"zone_fga_rate_{z}": zone_data["zone_fga_rate"][z] for z in ZONES},
            **{f"zone_fg_pct_{z}": zone_data["zone_fg_pct"][z] for z in ZONES},
            **{f"zone_fga_per36_{z}": zone_data["zone_fga_per36"][z] for z in ZONES},
            **{f"zone_pref_vs_league_{z}": zone_data["zone_pref_vs_league"][z] for z in ZONES},
            "sub_zone_distribution_close": zone_data["sub_zone_distribution_close"],
            "sub_zone_distribution_mid": zone_data["sub_zone_distribution_mid"],
            "sub_zone_distribution_three": zone_data["sub_zone_distribution_three"],
            "drive_right_bias": drive_right_bias,
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

    # ------------------------------------------------------------------
    # Multi-season blending
    # ------------------------------------------------------------------

    @staticmethod
    def _prior_seasons(s0: str) -> list[str]:
        """Return [s1, s2] given s0 season string (e.g. '2025-26' → ['2024-25', '2023-24'])."""
        try:
            start_year = int(s0.split("-")[0])
            def _fmt(y: int) -> str:
                return f"{y}-{str(y + 1)[-2:]}"
            return [_fmt(start_year - 1), _fmt(start_year - 2)]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _dynamic_weights(s0_gp: int) -> tuple[float, float, float]:
        """Dynamic season weights based on s0 games played."""
        if s0_gp < 20:
            return 0.35, 0.45, 0.20
        if s0_gp < 40:
            return 0.45, 0.35, 0.20
        return 0.55, 0.30, 0.15

    # Keys that are never numerically blended (always taken from s0)
    _NON_BLEND_KEYS: frozenset[str] = frozenset({
        "position", "height_inches", "weight_lbs", "age",
        "has_shot_chart", "low_minutes", "games_played", "gp",
        "sub_zone_distribution_close", "sub_zone_distribution_mid",
        "sub_zone_distribution_three", "drive_right_bias",
        "is_pg", "is_sg", "is_sf", "is_pf", "is_c",
    })

    def build_multiseasonal_features(
        self, player_id: int, s0_season: str = "2025-26"
    ) -> dict[str, Any]:
        """
        Build a weighted multi-season feature vector for *player_id*.

        Fetches s0, s1, s2 independently and blends numeric features
        using dynamic weights driven by s0 games played.  Non-numeric
        identity fields (position, height, age, etc.) are always taken
        from s0.  Seasons absent from cache/API are silently dropped and
        the remaining weights are renormalised.

        Parameters
        ----------
        player_id:  NBA player ID.
        s0_season:  Current / most-recent season string (e.g. "2025-26").

        Returns
        -------
        Blended feature dict compatible with FormulaLayer.generate() and
        AttributeCalculator.calculate().
        """
        prior = self._prior_seasons(s0_season)
        seasons = [s0_season] + prior  # [s0, s1, s2]

        # --- collect per-season feature dicts ---
        season_features: list[tuple[float, dict[str, Any]]] = []
        raw_weights = [1.0, 1.0, 1.0]  # placeholders; resolved after s0 GP

        raw_f0: dict[str, Any] = {}
        for i, season in enumerate(seasons):
            try:
                f = self.build_features(player_id, season=season)
                season_features.append((raw_weights[i], f))
                if i == 0:
                    raw_f0 = f
            except Exception:  # noqa: BLE001
                pass  # season unavailable — skip silently

        if not season_features:
            # Absolute fallback: nothing fetched at all
            return self.build_features(player_id, season=s0_season)

        # Resolve weights now that we know s0 GP
        s0_gp = int(raw_f0.get("gp", raw_f0.get("games_played", 0)))
        w0, w1, w2 = self._dynamic_weights(s0_gp)
        all_weights = [w0, w1, w2]

        # Rebuild with correct weights, only for seasons that succeeded
        weighted: list[tuple[float, dict[str, Any]]] = []
        for i, (_, f) in enumerate(season_features):
            weighted.append((all_weights[i], f))

        total_w = sum(w for w, _ in weighted)

        # --- blend numeric features ---
        all_keys: set[str] = set()
        for _, f in weighted:
            all_keys.update(f.keys())

        blended: dict[str, Any] = {}

        # Non-blend keys: from s0 (raw_f0) first, fallback to first available
        base = raw_f0 if raw_f0 else weighted[0][1]
        for k in self._NON_BLEND_KEYS:
            if k in base:
                blended[k] = base[k]

        for k in all_keys:
            if k in self._NON_BLEND_KEYS:
                continue
            vals: list[float] = []
            ws: list[float] = []
            for w, f in weighted:
                v = f.get(k)
                if v is not None:
                    try:
                        vals.append(float(v))
                        ws.append(w)
                    except (TypeError, ValueError):
                        pass
            if vals:
                norm = sum(ws)
                blended[k] = sum(v * w for v, w in zip(vals, ws)) / norm

        # Carry across any non-blend keys missed above (e.g. position flags)
        for k, v in base.items():
            if k not in blended:
                blended[k] = v

        return blended
