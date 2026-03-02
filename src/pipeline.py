"""Top-level tendency generation pipeline."""
from __future__ import annotations

import json
import os
from typing import Any

from src.caps.cap_enforcer import CapEnforcer
from src.export.json_exporter import export_player_json
from src.features.feature_engine import FeatureEngine
from src.formula.formula_layer import FormulaLayer
from src.ingest.nba_api_client import NBAApiClient
from src.validation.guardrails import Guardrails

_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "data", "tendency_registry.json"
)


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
    ) -> None:
        self._registry_path = os.path.abspath(registry_path)
        self._client = NBAApiClient(cache_dir=cache_dir)
        self._features = FeatureEngine(self._client)
        self._formula = FormulaLayer()
        self._caps = CapEnforcer(self._registry_path)
        self._guardrails = Guardrails()
        self._registry = load_registry(self._registry_path)

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
            return {
                "player_id": player_id,
                "season": season,
                "error": f"Feature extraction failed: {exc}",
                "tendencies": {},
            }

        # Step 2: Apply formula
        try:
            formula_raw = self._formula.generate(features)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Formula error: {exc}")
            formula_raw = {}

        # Step 3: Round to integers
        rounded = {k: round(v) for k, v in formula_raw.items()}

        # Step 4: Guardrail checks (operates on float dict)
        guardrail_input = dict(formula_raw)
        try:
            violations = self._guardrails.check(guardrail_input)
            # Apply corrected values back
            for k, v in guardrail_input.items():
                rounded[k] = round(v)
        except Exception as exc:  # noqa: BLE001
            violations = []
            errors.append(f"Guardrail error: {exc}")

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

    def generate_json(self, player_id: int, season: str = "2024-25") -> str:
        """Generate and return primjer.txt-compatible JSON string."""
        result = self.generate(player_id, season=season)
        return export_player_json(result.get("tendencies", {}), self._registry)

    def search_player(self, name: str) -> list[dict[str, Any]]:
        """Search for a player by name."""
        return self._client.search_player(name)
