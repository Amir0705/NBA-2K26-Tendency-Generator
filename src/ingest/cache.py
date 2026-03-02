"""Disk-based SQLite response cache."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any


class Cache:
    """Simple disk-backed JSON cache for API responses."""

    def __init__(self, cache_dir: str) -> None:
        """
        Initialise the cache.

        Parameters
        ----------
        cache_dir: Directory where cached responses are stored.
        """
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "nba_cache.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Return cached value for *key*, or None if missing or expired."""
        now = time.time()
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value_str, expires_at = row
        if expires_at is not None and now > expires_at:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return json.loads(value_str)

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        """Store *value* under *key* with an optional TTL."""
        expires_at = time.time() + ttl_seconds
        value_str = json.dumps(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value_str, expires_at),
        )
        self._conn.commit()

    def invalidate(self, key: str) -> None:
        """Remove the cached entry for *key*."""
        self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        """Remove all cached entries."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
