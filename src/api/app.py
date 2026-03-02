"""FastAPI application stub — Phase 7 implementation."""
from __future__ import annotations

# App is constructed lazily to avoid importing fastapi at module level
# when it may not be installed.

def create_app():
    """
    Create and return the FastAPI application instance.

    Endpoints (all Phase 7 implementation):
    - POST /generate      — Generate tendencies for a player.
    - GET  /player/{id}   — Retrieve cached tendencies for a player.
    - POST /export/json   — Export tendencies as primjer.txt-compatible JSON.
    - POST /export/csv    — Export tendencies as CSV.
    - POST /feedback      — Submit community feedback.
    - GET  /health        — Health check.
    """
    raise NotImplementedError("Phase 7 implementation")


# Provide a module-level `app` only if fastapi is available, so that
# uvicorn can discover it without crashing when not yet implemented.
try:
    from fastapi import FastAPI as _FastAPI  # noqa: F401
    app = None  # placeholder; real app wired in Phase 7
except ImportError:
    app = None
