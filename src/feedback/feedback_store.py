"""JSON-backed feedback storage and aggregation utilities."""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any
from threading import Lock


class FeedbackStore:
    """Persists and retrieves community tendency-correction feedback."""

    def __init__(self, store_path: str) -> None:
        """
        Initialise the feedback store.

        Parameters
        ----------
        store_path: Path to the JSON file (or database URI) used for storage.
        """
        self._store_path = store_path
        self._lock = Lock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        directory = os.path.dirname(self._store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.isfile(self._store_path):
            self._write_store({"entries": []})

    def _read_store(self) -> dict[str, Any]:
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {"entries": []}
            entries = data.get("entries", [])
            if not isinstance(entries, list):
                entries = []
            return {"entries": entries}
        except FileNotFoundError:
            return {"entries": []}
        except json.JSONDecodeError:
            return {"entries": []}

    def _write_store(self, data: dict[str, Any]) -> None:
        with open(self._store_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def submit(
        self,
        player_id: int,
        tendency_name: str,
        suggested_value: int,
        reviewer: str | None = None,
        notes: str = "",
    ) -> str:
        """
        Record a new feedback entry.

        Returns the generated feedback ID.
        """
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

        with self._lock:
            data = self._read_store()
            data.setdefault("entries", []).append(record)
            self._write_store(data)

        return entry_id

    def get_for_player(self, player_id: int) -> list[dict[str, Any]]:
        """Return all feedback entries for *player_id*."""
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            return []

        with self._lock:
            data = self._read_store()
            entries = data.get("entries", [])
            result = [e for e in entries if int(e.get("player_id", -1)) == pid]

        result.sort(key=lambda e: int(e.get("created_at", 0)), reverse=True)
        return result

    def aggregate(
        self, player_id: int, tendency_name: str
    ) -> dict[str, Any]:
        """
        Aggregate feedback for a specific tendency.

        Returns dict with: mean_value, vote_count, suggested_values.
        """
        tendency = str(tendency_name).strip()
        if not tendency:
            return {"mean_value": None, "vote_count": 0, "suggested_values": []}

        entries = self.get_for_player(player_id)
        values: list[int] = []
        for entry in entries:
            if str(entry.get("tendency_name", "")).strip() != tendency:
                continue
            try:
                values.append(int(entry.get("suggested_value", 0)))
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
