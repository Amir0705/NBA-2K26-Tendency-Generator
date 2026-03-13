"""ML training pipeline — learns residual corrections from ATD Committee CSV."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# ATD CSV column name → canonical tendency name mapping
# ---------------------------------------------------------------------------
_ATD_COL_TO_CANONICAL: dict[str, str] = {
    "Shot": "shot",
    "Touches": "touches",
    "Shot Close": "shot_close",
    "Shot Under Basket": "shot_under_basket",
    "Shot Close Left": "shot_close_left",
    "Shot Close Middle": "shot_close_middle",
    "Shot Close Right": "shot_close_right",
    "Shot Mid-Range": "shot_mid_range",
    "Spot Up Shot Mid-Range": "spot_up_shot_mid_range",
    "Off-Screen Shot Mid-Range": "off_screen_shot_mid_range",
    "Shot Mid Left": "shot_mid_left",
    "Shot Mid Left-Center": "shot_mid_left_center",
    "Shot Mid Center": "shot_mid_center",
    "Shot Mid Right-Center": "shot_mid_right_center",
    "Shot Mid Right": "shot_mid_right",
    "Shot Three": "shot_three",
    "Spot Up Shot Three": "spot_up_shot_three",
    "Off-Screen Shot Three": "off_screen_shot_three",
    "Shot Three Left": "shot_three_left",
    "Shot Three Left-Center": "shot_three_left_center",
    "Shot Three Center": "shot_three_center",
    "Shot Three Right-Center": "shot_three_right_center",
    "Shot Three Right": "shot_three_right",
    "Contested Jumper Mid-Range": "contested_jumper_mid_range",
    "Contested Jumper Three": "contested_jumper_three",
    "Stepback Jumper Mid-Range": "stepback_jumper_mid_range",
    "Stepback Three Point Shot": "stepback_jumper_three",
    "Spin Jumper": "spin_jumper",
    "Transition Pull-Up Three Point Shot": "transition_pull_up_three",
    "Drive Pull-Up Mid-Range": "drive_pull_up_mid_range",
    "Drive Pull-Up Three": "drive_pull_up_three",
    "Use Glass": "use_glass",
    "Step Through Shot": "step_through_shot",
    "Driving Layup": "driving_layup",
    "Spin Layup": "spin_layup",
    "Euro Step Layup": "euro_step_layup",
    "Hop Step Layup": "hop_step_layup",
    "Floater": "floater",
    "Standing Dunk": "standing_dunk",
    "Driving Dunk": "driving_dunk",
    "Flashy Dunk": "flashy_dunk",
    "Alley-Oop": "alley_oop",
    "Putback": "putback",
    "Crash": "crash",
    "Drive Right": "drive_right",
    "Triple Threat Pump Fake": "triple_threat_pump_fake",
    "Triple Threat Jab Step": "triple_threat_jab_step",
    "Triple Threat Idle": "triple_threat_idle",
    "Triple Threat Shoot": "triple_threat_shoot",
    "Setup With Sizeup": "setup_with_sizeup",
    "Setup With Hesitation": "setup_with_hesitation",
    "No Setup Dribble": "no_setup_dribble",
    "Drive": "drive",
    "Spot Up Drive": "spot_up_drive",
    "Off-Screen Drive": "off_screen_drive",
    "Driving Crossover": "driving_crossover",
    "Driving Double Crossover": "driving_double_crossover",
    "Driving Spin": "driving_spin",
    "Driving Half Spin": "driving_half_spin",
    "Driving Stepback": "driving_step_back",
    "Driving Behind the Back": "driving_behind_the_back",
    "Driving Dribble Hesitation": "driving_dribble_hesitation",
    "Driving In & Out": "driving_in_and_out",
    "No Driving Dribble Move": "no_driving_dribble_move",
    "Attack Strong on Drive": "attack_strong_on_drive",
    "Dish to Open Man": "dish_to_open_man",
    "Flashy Pass": "flashy_pass",
    "Alley-Oop Pass": "alley_oop_pass",
    "Roll vs Pop": "roll_vs_pop",
    "Transition Spot Up vs Cut to the Basket": "transition_spot_up",
    "Iso vs Elite Defender": "iso_vs_elite_defender",
    "Iso vs Good Defender": "iso_vs_good_defender",
    "Iso vs Average Defender": "iso_vs_average_defender",
    "Iso vs Poor Defender": "iso_vs_poor_defender",
    "Play Discipline": "play_discipline",
    "Post Up": "post_up",
    "Post Back Down": "post_back_down",
    "Post Aggressive Backdown": "post_aggressive_backdown",
    "Post Face Up": "post_face_up",
    "Post Spin": "post_spin",
    "Post Drive": "post_drive",
    "Post Drop Step": "post_drop_step",
    "Shoot From Post": "shoot_from_post",
    "Post Hook Left": "post_hook_left",
    "Post Hook Right": "post_hook_right",
    "Post Fade Left": "post_fade_left",
    "Post Fade Right": "post_fade_right",
    "Post Shimmy Shot": "post_shimmy_shot",
    "Post Hop Step": "post_hop_step",
    "Post Stepback Shot": "post_step_back_shot",
    "Post Up & Under": "post_up_and_under",
    "Take Charge": "take_charge",
    "Foul": "foul",
    "Hard Foul": "hard_foul",
    "Pass Interception": "pass_interception",
    "On-Ball Steal": "on_ball_steal",
    "Block Shot": "block_shot",
    "Contest Shot": "contest_shot",
}

# NBA API player name → player_id lookup (populated lazily via nba_client)
_MIN_TRAINING_SAMPLES = 30


def _flatten_features(features: dict) -> dict[str, float]:
    """Flatten nested feature dicts to scalar float values."""
    from src.ml.predictor import _flatten_features as _ff
    return _ff(features)


class TendencyTrainer:
    """Trains per-tendency residual models from ATD community labels."""

    def __init__(
        self,
        formula_layer: Any,
        feature_engine: Any,
        nba_client: Any,
    ) -> None:
        """
        Initialise trainer.

        Parameters
        ----------
        formula_layer:  FormulaLayer instance for computing formula predictions.
        feature_engine: FeatureEngine instance for building player features.
        nba_client:     NBAApiClient instance for resolving player IDs.
        """
        self._formula = formula_layer
        self._features = feature_engine
        self._client = nba_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        atd_csv_path: str,
        season: str = "2024-25",
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """
        Build training dataset from ATD Committee CSV.

        Steps
        -----
        1. Load ATD CSV (community labels for ~400-500 players).
        2. For each player that can be resolved to an NBA player_id:
           a. Build features using FeatureEngine.
           b. Generate formula predictions using FormulaLayer.
           c. Compute residuals: ATD_label − formula_prediction.
        3. Return (feature_df, residuals_dict).

        Returns
        -------
        features_df : pd.DataFrame  — one row per resolved player.
        residuals   : dict[tendency_name, pd.Series]  — residual per player.
        """
        atd_df = self._load_atd_csv(atd_csv_path)

        feature_rows: list[dict[str, float]] = []
        residual_rows: list[dict[str, float]] = []

        for _, row in atd_df.iterrows():
            player_name: str = str(row["player_name"]).strip()
            player_id = self._resolve_player_id(player_name)
            if player_id is None:
                continue

            # Build features (skip if API call fails)
            try:
                features = self._features.build_features(player_id, season=season)
            except Exception:  # noqa: BLE001
                continue

            # Formula predictions
            try:
                formula_preds = self._formula.generate(features)
            except Exception:  # noqa: BLE001
                continue

            flat = _flatten_features(features)
            residuals: dict[str, float] = {}
            for atd_col, canonical in _ATD_COL_TO_CANONICAL.items():
                if atd_col not in row.index:
                    continue
                label = row[atd_col]
                if pd.isna(label):
                    continue
                formula_val = formula_preds.get(canonical)
                if formula_val is None:
                    continue
                residuals[canonical] = float(label) - float(formula_val)

            if not residuals:
                continue

            feature_rows.append(flat)
            residual_rows.append(residuals)

        if not feature_rows:
            return pd.DataFrame(), {}

        features_df = pd.DataFrame(feature_rows).fillna(0.0)
        residuals_df = pd.DataFrame(residual_rows)

        residuals_dict: dict[str, pd.Series] = {
            col: residuals_df[col].dropna()
            for col in residuals_df.columns
        }
        return features_df, residuals_dict

    def train(
        self,
        atd_csv_path: str,
        model_dir: str = "models/",
    ) -> dict[str, dict[str, Any]]:
        """
        Train one LightGBM model per tendency that has enough data.

        - Target: residual (ATD − formula).
        - Only train when >= 30 non-null samples exist.
        - Save models to model_dir/{tendency_name}.joblib.
        - Return training report: {tendency: {n_samples, rmse, r2}}.
        """
        import joblib
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error, r2_score

        os.makedirs(model_dir, exist_ok=True)
        features_df, residuals_dict = self.prepare_training_data(atd_csv_path)
        if features_df.empty:
            return {}

        report: dict[str, dict[str, Any]] = {}
        for tendency_name, residuals in residuals_dict.items():
            valid_idx = residuals.dropna().index
            if len(valid_idx) < _MIN_TRAINING_SAMPLES:
                continue

            X = features_df.loc[features_df.index.intersection(valid_idx)].fillna(0.0)
            y = residuals.loc[X.index]

            cv_metrics = self.cross_validate(X, y, tendency_name)

            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1,
            )
            model.fit(X, y)

            y_pred = model.predict(X)
            train_rmse = float(mean_squared_error(y, y_pred) ** 0.5)
            train_r2 = float(r2_score(y, y_pred))

            model_path = os.path.join(model_dir, f"{tendency_name}.joblib")
            joblib.dump(model, model_path)

            report[tendency_name] = {
                "n_samples": len(X),
                "rmse": cv_metrics.get("rmse", train_rmse),
                "r2": cv_metrics.get("r2", train_r2),
                "train_rmse": train_rmse,
                "train_r2": train_r2,
            }

        return report

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tendency_name: str,
    ) -> dict[str, float]:
        """5-fold CV to estimate model quality."""
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses: list[float] = []
        r2s: list[float] = []

        X_arr = X.values
        y_arr = y.values

        for train_idx, val_idx in kf.split(X_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            if len(X_val) == 0:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmses.append(float(mean_squared_error(y_val, preds) ** 0.5))
            r2s.append(float(r2_score(y_val, preds)))

        return {
            "rmse": sum(rmses) / len(rmses) if rmses else float("inf"),
            "r2": sum(r2s) / len(r2s) if r2s else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_atd_csv(self, path: str) -> pd.DataFrame:
        """
        Load the ATD Committee CSV and return a tidy DataFrame with columns:
        player_name + one column per tendency (ATD CSV display name).
        """
        raw = pd.read_csv(path, header=4)
        # First column is "Tendency\n(In Order)" containing player/team names
        name_col = raw.columns[0]
        raw = raw.rename(columns={name_col: "player_name"})

        # Drop rows that look like team headers (no numeric tendency values)
        tendency_cols = [c for c in raw.columns if c != "player_name"]
        raw[tendency_cols] = raw[tendency_cols].apply(pd.to_numeric, errors="coerce")
        has_any_value = raw[tendency_cols].notna().any(axis=1)
        raw = raw[has_any_value & raw["player_name"].notna()].copy()
        return raw.reset_index(drop=True)

    def _resolve_player_id(self, player_name: str) -> int | None:
        """
        Attempt to resolve a player name to an NBA player_id via nba_client.
        Returns None if not found.
        """
        try:
            results = self._client.search_player(player_name)
            if results:
                return int(results[0]["player_id"])
        except Exception:  # noqa: BLE001
            pass
        return None
