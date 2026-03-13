"""Attribute ML trainer — learns residual corrections from 2K editor exports.

Reads real 2K attribute values from exported player JSON files,
compares them to our formula-generated attributes, and trains
per-attribute LightGBM models to predict the residual (2K_real - formula).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# 2K export attribute name → our canonical name
_EXPORT_TO_CANONICAL: dict[str, str] = {
    "Driving Layup": "driving_layup",
    "Standing Dunk": "standing_dunk",
    "Driving Dunk": "driving_dunk",
    "Close Shot": "close_shot",
    "Mid Range": "mid_range_shot",
    "Three Point": "three_point_shot",
    "Free Throw": "free_throw",
    "Post Hook": "post_hook",
    "Post Fade": "post_fade",
    "Post Control": "post_control",
    "Draw Foul": "draw_foul",
    "Shot IQ": "shot_iq",
    "Ball Control": "ball_handle",
    "Speed With Ball": "speed_with_ball",
    "Hands": "hands",
    "Passing Accuracy": "pass_accuracy",
    "Passing IQ": "pass_iq",
    "Passing Vision": "pass_vision",
    "Offensive Consistency": "offensive_consistency",
    "Interior Defense": "interior_defense",
    "Perimeter Defense": "perimeter_defense",
    "Steal": "steal",
    "Block": "block",
    "Offensive Rebound": "offensive_rebound",
    "Defensive Rebound": "defensive_rebound",
    "Help Defense IQ": "help_defense_iq",
    "Passing Perception": "pass_perception",
    "Defensive Consistency": "defensive_consistency",
    "Speed": "speed",
    "Agility": "agility",
    "Strength": "strength",
    "Vertical": "vertical",
    "Stamina": "stamina",
    "Intangibles": "intangibles",
    "Hustle": "hustle",
    "Potential": "potential",
}

# Position map from 2K integer codes
_POS_MAP = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "SG"}

# Minimum samples to train a model
_MIN_TRAINING_SAMPLES = 10


def _flatten_features(features: dict) -> dict[str, float]:
    """Flatten nested feature dicts to scalar float values."""
    flat: dict[str, float] = {}
    for k, v in features.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}__{sub_k}"] = float(sub_v) if sub_v is not None else 0.0
        elif isinstance(v, bool):
            flat[k] = float(v)
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
    return flat


def _name_from_filename(filename: str) -> str:
    """Convert filename to player name for NBA API lookup."""
    stem = Path(filename).stem
    # Handle special cases
    stem = stem.replace("_jr", " Jr.")
    stem = stem.replace("dangelo", "D'Angelo")
    stem = stem.replace("c.j._mccullum", "CJ McCollum")
    stem = stem.replace("lamelo", "LaMelo")
    stem = stem.replace("giannis_antetokumpo", "Giannis Antetokounmpo")
    stem = stem.replace("JD_davison", "JD Davison")
    stem = stem.replace("scotty_pippen", "Scotty Pippen")
    # General: underscores to spaces, title case
    name = stem.replace("_", " ").strip()
    # Title case but preserve already-cased names
    if name == name.lower():
        name = name.title()
    return name


def load_2k_exports(export_dir: str) -> list[dict[str, Any]]:
    """Load all 2K editor exports and extract attributes + metadata."""
    players = []
    export_path = Path(export_dir)

    for f in sorted(export_path.glob("*.txt")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            attrs_raw = data["categories"]["Attributes"]
            vitals = data["categories"]["Vitals"]
            tendencies = data["categories"]["Tendencies"]

            # Extract only gameplay attributes (skip durability)
            attrs = {}
            for export_name, canonical in _EXPORT_TO_CANONICAL.items():
                if export_name in attrs_raw:
                    attrs[canonical] = int(attrs_raw[export_name])

            pos_num = vitals.get("Position", 0)
            player = {
                "filename": f.name,
                "full_name": data.get("fullName", _name_from_filename(f.name)),
                "position": _POS_MAP.get(pos_num, "SF"),
                "attributes": attrs,
                "tendencies": tendencies,
                "weight": vitals.get("Weight", 0),
            }
            players.append(player)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    return players


class AttributeTrainer:
    """Trains per-attribute residual models from 2K editor exports."""

    def __init__(self, feature_engine: Any, attribute_calculator: Any,
                 nba_client: Any, formula_layer: Any) -> None:
        self._features = feature_engine
        self._calculator = attribute_calculator
        self._client = nba_client
        self._formula = formula_layer

    def prepare_training_data(
        self,
        export_dir: str,
        season: str = "2024-25",
        cached_only: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build training dataset from 2K exports.

        Returns
        -------
        features_df:  Feature vectors (one row per player).
        formula_df:   Formula-predicted attributes.
        target_df:    Real 2K attributes (ground truth).
        """
        players_2k = load_2k_exports(export_dir)
        print(f"Loaded {len(players_2k)} 2K exports")

        feature_rows: list[dict[str, float]] = []
        formula_rows: list[dict[str, int]] = []
        target_rows: list[dict[str, int]] = []
        player_names: list[str] = []

        for p in players_2k:
            name = p["full_name"]
            print(f"  Processing {name}...", end=" ", flush=True)

            # When cached_only, skip players whose data isn't cached
            if cached_only:
                player_id = self._resolve_player(name, p["filename"])
                if player_id is None:
                    print("SKIP (not found)")
                    continue
                cache_key = f"player_stats:{player_id}:{season}"
                if self._client._cache and self._client._cache.get(cache_key) is None:
                    print("SKIP (not cached)")
                    continue

            try:
                result = self._process_one_player(p, season)
            except Exception as e:
                print(f"SKIP (error: {e})")
                continue

            if result is None:
                print()  # newline after "SKIP (not found)" or "SKIP (timeout)"
                continue

            flat, formula_attrs = result
            feature_rows.append(flat)
            formula_rows.append(formula_attrs)
            target_rows.append(p["attributes"])
            player_names.append(name)
            print("OK")

        print(f"\nSuccessfully processed {len(feature_rows)}/{len(players_2k)} players")

        features_df = pd.DataFrame(feature_rows, index=player_names).fillna(0.0)
        formula_df = pd.DataFrame(formula_rows, index=player_names).fillna(0)
        target_df = pd.DataFrame(target_rows, index=player_names).fillna(0)

        return features_df, formula_df, target_df

    def train(
        self,
        export_dir: str,
        model_dir: str = "models/attributes/",
        season: str = "2024-25",
        cached_only: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Train one LightGBM model per attribute.

        Target: residual = 2K_real - formula_predicted.
        """
        import joblib
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold, cross_val_predict

        os.makedirs(model_dir, exist_ok=True)

        features_df, formula_df, target_df = self.prepare_training_data(
            export_dir, season=season, cached_only=cached_only,
        )

        if features_df.empty:
            print("No training data!")
            return {}

        # Save training data for inspection
        training_data_path = os.path.join(model_dir, "training_data.json")
        training_snapshot = {
            "players": list(features_df.index),
            "n_players": len(features_df),
            "feature_columns": list(features_df.columns),
            "n_features": len(features_df.columns),
        }
        with open(training_data_path, "w") as f:
            json.dump(training_snapshot, f, indent=2)

        # Get common attribute columns
        attr_cols = sorted(set(formula_df.columns) & set(target_df.columns))
        print(f"\nTraining models for {len(attr_cols)} attributes...")
        print(f"Features: {len(features_df.columns)} columns, {len(features_df)} players\n")

        report: dict[str, dict[str, Any]] = {}

        for attr_name in attr_cols:
            formula_vals = formula_df[attr_name].astype(float)
            target_vals = target_df[attr_name].astype(float)
            residuals = target_vals - formula_vals

            n_valid = residuals.notna().sum()
            if n_valid < _MIN_TRAINING_SAMPLES:
                print(f"  {attr_name:<25} SKIP (only {n_valid} samples)")
                continue

            X = features_df.fillna(0.0)
            y = residuals

            # Heavy regularization to prevent memorization with small sample sizes
            model = lgb.LGBMRegressor(
                n_estimators=30,
                max_depth=2,
                num_leaves=4,
                learning_rate=0.08,
                min_child_samples=8,
                reg_alpha=2.0,
                reg_lambda=5.0,
                subsample=0.7,
                colsample_bytree=0.3,
                verbose=-1,
            )

            # 5-fold CV for honest error estimate
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            try:
                cv_preds = cross_val_predict(model, X, y, cv=kf)
                cv_mae = float(mean_absolute_error(y, cv_preds))
                cv_rmse = float(mean_squared_error(y, cv_preds) ** 0.5)
                # Corrected predictions = formula + predicted_residual
                corrected = formula_vals.values + cv_preds
                baseline_mae = float(mean_absolute_error(target_vals, formula_vals))
                corrected_mae = float(mean_absolute_error(target_vals, corrected))
                improvement = baseline_mae - corrected_mae
            except Exception as e:
                print(f"  {attr_name:<25} CV FAILED: {e}")
                continue

            # Train final model on all data
            model.fit(X, y)

            model_path = os.path.join(model_dir, f"{attr_name}.joblib")
            joblib.dump(model, model_path)

            # Per-player breakdown
            per_player = {}
            for i, player in enumerate(features_df.index):
                per_player[player] = {
                    "formula": int(formula_vals.iloc[i]),
                    "real_2k": int(target_vals.iloc[i]),
                    "residual": int(residuals.iloc[i]),
                    "cv_predicted_correction": round(float(cv_preds[i]), 1),
                    "corrected": int(round(formula_vals.iloc[i] + cv_preds[i])),
                }

            report[attr_name] = {
                "n_samples": int(n_valid),
                "formula_mae": round(baseline_mae, 1),
                "corrected_mae": round(corrected_mae, 1),
                "improvement": round(improvement, 1),
                "loo_residual_mae": round(cv_mae, 1),
                "loo_residual_rmse": round(cv_rmse, 1),
                "per_player": per_player,
            }

            status = "BETTER" if improvement > 0.5 else ("SAME" if improvement > -0.5 else "WORSE")
            print(f"  {attr_name:<25} formula_MAE={baseline_mae:5.1f}  "
                  f"corrected_MAE={corrected_mae:5.1f}  "
                  f"improvement={improvement:+5.1f}  [{status}]")

        # Save full report
        report_path = os.path.join(model_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")

        return report

    def _process_one_player(
        self, p: dict[str, Any], season: str
    ) -> tuple[dict[str, float], dict[str, int]] | None:
        """Process a single player. Returns (flat_features, formula_attrs) or None."""
        player_id = self._resolve_player(p["full_name"], p["filename"])
        if player_id is None:
            print("SKIP (not found)", end="")
            return None

        features = self._features.build_features(player_id, season=season)
        tendencies = self._formula.generate(features)
        formula_attrs = self._calculator.calculate(features, tendencies)

        flat = _flatten_features(features)
        pos = features.get("position", "SF")
        for pos_label in ("PG", "SG", "SF", "PF", "C"):
            flat[f"pos_{pos_label}"] = 1.0 if pos == pos_label else 0.0

        return flat, formula_attrs

    def _resolve_player(self, name: str, filename: str) -> int | None:
        """Resolve player name to NBA API player ID."""
        try:
            results = self._client.search_player(name)
            if results:
                return int(results[0]["player_id"])
        except Exception:
            pass

        # Try filename-based name
        alt_name = _name_from_filename(filename)
        if alt_name != name:
            try:
                results = self._client.search_player(alt_name)
                if results:
                    return int(results[0]["player_id"])
            except Exception:
                pass

        return None
