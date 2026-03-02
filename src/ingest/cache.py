"""Disk-based response cache stub — Phase 2 implementation."""
from __future__ import annotations

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
        raise NotImplementedError("Phase 2 implementation")

    def get(self, key: str) -> Any | None:
        """Return cached value for *key*, or None if not cached."""
        raise NotImplementedError("Phase 2 implementation")

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        """Store *value* under *key* with an optional TTL."""
        raise NotImplementedError("Phase 2 implementation")

    def invalidate(self, key: str) -> None:
        """Remove the cached entry for *key*."""
        raise NotImplementedError("Phase 2 implementation")

    def clear(self) -> None:
        """Remove all cached entries."""
        raise NotImplementedError("Phase 2 implementation")
