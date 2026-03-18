"""Firestore-backed feedback storage and aggregation."""
from __future__ import annotations

import importlib
import time
import uuid
from typing import Any


class FirebaseFeedbackStore:
    """Persist editor feedback in Firestore."""

    @staticmethod
    def _firestore_module() -> Any:
        return importlib.import_module("firebase_admin.firestore")

    def __init__(self, collection_name: str = "feedback_entries") -> None:
        self._db = self._firestore_module().client()
        self._col = self._db.collection(collection_name)

    def submit(
        self,
        player_id: int,
        tendency_name: str,
        suggested_value: int,
        reviewer: str | None = None,
        notes: str = "",
    ) -> str:
        tendency = str(tendency_name).strip()
        if not tendency:
            raise ValueError("tendency_name is required")

        try:
            pid = int(player_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("player_id must be an integer") from exc

        try:
            value = int(round(float(suggested_value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("suggested_value must be a number") from exc
        value = max(0, min(100, value))

        entry_id = str(uuid.uuid4())
        record = {
            "id": entry_id,
            "created_at": int(time.time()),
            "player_id": pid,
            "tendency_name": tendency,
            "suggested_value": value,
            "reviewer": (reviewer or "").strip() or None,
            "notes": (notes or "").strip(),
        }
        self._col.document(entry_id).set(record)
        return entry_id

    def get_for_player(self, player_id: int) -> list[dict[str, Any]]:
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            return []

        docs = self._col.where("player_id", "==", pid).stream()
        out: list[dict[str, Any]] = []
        for doc in docs:
            data = doc.to_dict() or {}
            if isinstance(data, dict):
                out.append(data)
        out.sort(key=lambda e: int(e.get("created_at", 0)), reverse=True)
        return out

    def aggregate(self, player_id: int, tendency_name: str) -> dict[str, Any]:
        tendency = str(tendency_name).strip()
        if not tendency:
            return {"mean_value": None, "vote_count": 0, "suggested_values": []}

        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            return {"mean_value": None, "vote_count": 0, "suggested_values": []}

        docs = self._col.where("player_id", "==", pid).where("tendency_name", "==", tendency).stream()
        values: list[int] = []
        for doc in docs:
            data = doc.to_dict() or {}
            try:
                values.append(int(data.get("suggested_value", 0)))
            except (TypeError, ValueError):
                continue

        if not values:
            return {"mean_value": None, "vote_count": 0, "suggested_values": []}

        mean = sum(values) / max(1, len(values))
        return {
            "mean_value": round(mean, 2),
            "vote_count": len(values),
            "suggested_values": values,
        }

    def aggregate_for_player(self, player_id: int) -> dict[str, dict[str, Any]]:
        """Return aggregates for all tendency keys for *player_id* in one query."""
        entries = self.get_for_player(player_id)
        by_tendency: dict[str, list[int]] = {}
        for entry in entries:
            tendency = str(entry.get("tendency_name", "")).strip()
            if not tendency:
                continue
            try:
                value = int(entry.get("suggested_value", 0))
            except (TypeError, ValueError):
                continue
            by_tendency.setdefault(tendency, []).append(value)

        out: dict[str, dict[str, Any]] = {}
        for tendency, values in by_tendency.items():
            if not values:
                continue
            mean = sum(values) / max(1, len(values))
            out[tendency] = {
                "mean_value": round(mean, 2),
                "vote_count": len(values),
                "suggested_values": values,
            }
        return out
