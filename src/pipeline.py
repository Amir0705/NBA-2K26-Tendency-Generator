"""Top-level tendency generation pipeline."""
from __future__ import annotations

import json
import os
from typing import Any

from src.caps.cap_enforcer import CapEnforcer
from src.export.json_exporter import export_player_json
from src.features.feature_engine import FeatureEngine
from src.formula.formula_layer import FormulaLayer
from src.hybrid.combiner import HybridCombiner
from src.ingest.nba_api_client import NBAApiClient
from src.ml.confidence import ConfidenceScorer
from src.ml.predictor import TendencyPredictor
from src.validation.guardrails import Guardrails

_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "data", "tendency_registry.json"
)


def _round_to_5(x: float) -> int:
    """Round to nearest multiple of 5, clamped to [0, 100]."""
    return max(0, min(100, 5 * round(x / 5)))


def load_registry(registry_path: str) -> list[dict[str, Any]]:
    """Load tendency registry JSON."""
    with open(registry_path, encoding="utf-8") as fh:
        return json.load(fh)


class TendencyPipeline:
    """Wires together the full NBA 2K26 tendency generation pipeline."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        registry_path: str = _DEFAULT_REGISTRY,
        model_dir: str | None = None,
        training_report: dict | None = None,
    ) -> None:
        self._registry_path = os.path.abspath(registry_path)
        self._client = NBAApiClient(cache_dir=cache_dir)
        self._features = FeatureEngine(self._client)
        self._formula = FormulaLayer()
        self._caps = CapEnforcer(self._registry_path)
        self._guardrails = Guardrails()
        self._registry = load_registry(self._registry_path)

        # Build hybrid combiner if model_dir contains trained models
        predictor: TendencyPredictor | None = None
        confidence: ConfidenceScorer | None = None
        if model_dir and os.path.isdir(model_dir):
            predictor = TendencyPredictor(model_dir=model_dir)
            confidence = ConfidenceScorer(training_report=training_report)

        self._combiner = HybridCombiner(
            formula_layer=self._formula,
            predictor=predictor if (predictor and predictor._models) else None,
            confidence_scorer=confidence,
        )

    def generate(self, player_id: int, season: str = "2024-25") -> dict[str, Any]:
        """
        Full pipeline: Stats → Features → Formula → Guardrails → Caps.

        Returns
        -------
        {
            'player_name': str,
            'player_id': int,
            'season': str,
            'position': str,
            'tendencies': {canonical_name: int},
            'formula_raw': {canonical_name: float},
            'audit': [cap clamp entries],
            'guardrail_violations': [violation entries],
            'features': {feature summary},
        }
        """
        errors: list[str] = []

        # Step 1: Build features
        try:
            features = self._features.build_features(player_id, season=season)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Feature extraction failed: {exc}")
            features = self._fallback_features(player_id)

        # Step 2: Apply formula (or hybrid formula+ML)
        try:
            combiner = getattr(self, "_combiner", None)
            formula_raw = (
                combiner.combine(features) if combiner is not None
                else self._formula.generate(features)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Formula error: {exc}")
            formula_raw = {}

        # Step 3: Round to nearest multiple of 5
        rounded = {k: _round_to_5(v) for k, v in formula_raw.items()}

        # Step 4: Guardrail checks (operates on float dict)
        guardrail_input = dict(formula_raw)
        try:
            violations = self._guardrails.check(guardrail_input)
            # Apply corrected values back
            for k, v in guardrail_input.items():
                rounded[k] = _round_to_5(v)
        except Exception as exc:  # noqa: BLE001
            violations = []
            errors.append(f"Guardrail error: {exc}")

        # Step 4a: Parent-aware redistribution — ensure sub-zone sums equal parent
        # (Must run after guardrail re-rounding to avoid being overwritten)
        _sub_zone_families = [
            (
                "shot_close_left", "shot_close_middle", "shot_close_right",
                "shot_close",
            ),
            (
                "shot_mid_left", "shot_mid_left_center", "shot_mid_center",
                "shot_mid_right_center", "shot_mid_right",
                "shot_mid_range",
            ),
            (
                "shot_three_left", "shot_three_left_center", "shot_three_center",
                "shot_three_right_center", "shot_three_right",
                "shot_three",
            ),
        ]
        for _family in _sub_zone_families:
            _parent_key = _family[-1]
            _child_keys = list(_family[:-1])
            _parent_val = rounded.get(_parent_key, 0)
            _child_sum = sum(rounded.get(_k, 0) for _k in _child_keys)
            if _child_sum != _parent_val and _child_sum > 0:
                _diff = _parent_val - _child_sum
                _largest = max(_child_keys, key=lambda _k: rounded.get(_k, 0))
                rounded[_largest] = rounded.get(_largest, 0) + _diff

        # Step 4b: Tie-break — use drive_right bias when left == right for close shots
        _close_left = rounded.get("shot_close_left", 0)
        _close_right = rounded.get("shot_close_right", 0)
        if _close_left == _close_right:
            _drive_right = rounded.get("drive_right", 50)
            if _drive_right > 50 and _close_left >= 5:
                rounded["shot_close_right"] += 5
                rounded["shot_close_left"] -= 5
            elif _drive_right < 50 and _close_right >= 5:
                rounded["shot_close_left"] += 5
                rounded["shot_close_right"] -= 5

        # Step 5: Cap enforcement
        try:
            capped, audit = self._caps.enforce_all(rounded)
        except Exception as exc:  # noqa: BLE001
            capped = rounded
            audit = []
            errors.append(f"Cap enforcement error: {exc}")

        # Step 6: Gather player info
        player_name = ""
        position = features.get("position", "")
        try:
            info = self._client.get_player_info(player_id)
            # Try to get name from search (not ideal but works)
        except Exception:  # noqa: BLE001
            info = {}

        result: dict[str, Any] = {
            "player_name": player_name,
            "player_id": player_id,
            "season": season,
            "position": position,
            "tendencies": capped,
            "formula_raw": formula_raw,
            "audit": audit,
            "guardrail_violations": violations,
            "features": {
                k: v
                for k, v in features.items()
                if not isinstance(v, dict)
            },
        }
        if errors:
            result["errors"] = errors
        return result

    def _fallback_features(self, player_id: int) -> dict[str, Any]:
        """Build baseline features when live API stats are unavailable."""
        position = "SF"
        try:
            info = self._client.get_player_info(player_id)
            raw_position = str(info.get("position", "")).upper()
            if "PG" in raw_position or raw_position == "GUARD":
                position = "PG"
            elif "SG" in raw_position:
                position = "SG"
            elif "PF" in raw_position:
                position = "PF"
            elif "C" == raw_position or "CENTER" in raw_position:
                position = "C"
        except Exception:  # noqa: BLE001
            pass

        return {
            "position": position,
            "usg_pct_proxy": 0.20,
            "fga_per36": 12.0,
            "fg3a_rate": 0.35,
            "fta_rate": 0.30,
            "ast_per36": 5.0,
            "pts_per36": 18.0,
            "stl_per36": 1.0,
            "blk_per36": 0.4,
            "pf_per36": 2.5,
            "oreb_pct_proxy": 0.10,
            "zone_fga_rate_ra": 0.20,
            "zone_fga_rate_paint": 0.15,
            "zone_fga_rate_mid_left": 0.08,
            "zone_fga_rate_mid_center": 0.07,
            "zone_fga_rate_mid_right": 0.08,
            "sub_zone_distribution_close": {
                "left": 30.0,
                "middle": 40.0,
                "right": 30.0,
            },
            "sub_zone_distribution_mid": {
                "left": 20.0,
                "left_center": 20.0,
                "center": 20.0,
                "right_center": 20.0,
                "right": 20.0,
            },
            "sub_zone_distribution_three": {
                "left": 20.0,
                "left_center": 20.0,
                "center": 20.0,
                "right_center": 20.0,
                "right": 20.0,
            },
        }

    def generate_json(self, player_id: int, season: str = "2024-25") -> str:
        """Generate and return primjer.txt-compatible JSON string."""
        result = self.generate(player_id, season=season)
        output = export_player_json(result.get("tendencies", {}), self._registry)
        errors = result.get("errors")
        if errors:
            payload = json.loads(output)
            payload["errors"] = errors
            output = json.dumps(payload, indent=2, ensure_ascii=False)
        return output

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search for a player by name."""
        return self._client.search_player(name)
