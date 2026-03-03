"""FastAPI web server for the NBA 2K26 Tendency Generator."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.export.json_exporter import export_player_json
from src.pipeline import TendencyPipeline

_VALID_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

_pipeline: TendencyPipeline | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[type-arg]
    global _pipeline  # noqa: PLW0603
    _pipeline = TendencyPipeline(cache_dir=".cache")
    yield


app = FastAPI(title="NBA 2K26 Tendency Generator", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline() -> TendencyPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def _build_tendency_response(
    result: dict[str, Any],
    player_name: str,
    player_id: int,
    team: str,
    season: str,
) -> dict[str, Any]:
    """Build the structured tendency response matching the spec format."""
    registry = _get_pipeline()._registry
    tendencies_dict = result.get("tendencies", {})
    json_str = export_player_json(tendencies_dict, registry)
    payload = json.loads(json_str)
    ordered_tendencies: dict[str, Any] = {}
    for key, entry in payload.get("tendencies", {}).items():
        ordered_tendencies[key] = {
            "value": entry["value"],
            "label": entry["label"],
        }
    return {
        "player_name": player_name,
        "player_id": player_id,
        "position": result.get("position", ""),
        "team": team,
        "season": season,
        "tendencies": ordered_tendencies,
    }


@app.get("/")
def health_check() -> dict[str, str]:
    """Health check / welcome endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/search/{name}")
def search_player(name: str) -> dict[str, Any]:
    """Search for players by name."""
    pipeline = _get_pipeline()
    results = pipeline.search_player(name)
    return {"query": name, "results": results}


@app.get("/generate/id/{player_id}")
def generate_by_id(player_id: int, season: str = "2024-25") -> dict[str, Any]:
    """Generate tendencies by player ID."""
    pipeline = _get_pipeline()
    try:
        result = pipeline.generate(player_id, season=season)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc

    try:
        info = pipeline._client.get_player_info(player_id)
        team = info.get("team_abbreviation", "")
    except Exception:  # noqa: BLE001
        team = ""

    # Try to resolve player name from search
    player_name = result.get("player_name", "")
    if not player_name:
        player_name = f"Player {player_id}"

    return _build_tendency_response(result, player_name, player_id, team, season)


@app.get("/generate/{player_name}")
def generate_by_name(player_name: str, season: str = "2024-25") -> dict[str, Any]:
    """Generate tendencies for a player by name."""
    pipeline = _get_pipeline()
    results = pipeline.search_player(player_name)
    if not results:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    # Prefer exact match
    name_lower = player_name.lower()
    match = next((r for r in results if r.get("full_name", "").lower() == name_lower), results[0])
    pid = match["player_id"]
    full_name = match["full_name"]
    team = match.get("team", "")

    try:
        result = pipeline.generate(pid, season=season)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc

    return _build_tendency_response(result, full_name, pid, team, season)


@app.get("/team/{team_abbr}")
def generate_team(team_abbr: str, season: str = "2024-25") -> dict[str, Any]:
    """Generate tendencies for all players on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    players: list[dict[str, Any]] = []
    for player in roster:
        pid = player["player_id"]
        full_name = player["full_name"]
        try:
            result = pipeline.generate(pid, season=season)
            registry = pipeline._registry
            tendencies_dict = result.get("tendencies", {})
            json_str = export_player_json(tendencies_dict, registry)
            payload = json.loads(json_str)
            ordered: dict[str, Any] = {}
            for key, entry in payload.get("tendencies", {}).items():
                ordered[key] = {"value": entry["value"], "label": entry["label"]}
            players.append({
                "player_name": full_name,
                "player_id": pid,
                "position": result.get("position", ""),
                "tendencies": ordered,
            })
        except Exception:  # noqa: BLE001
            continue

    return {
        "team": abbr,
        "season": season,
        "player_count": len(players),
        "players": players,
    }


@app.get("/team/{team_abbr}/{player_name}")
def generate_team_player(
    team_abbr: str, player_name: str, season: str = "2024-25"
) -> dict[str, Any]:
    """Generate tendencies for a specific player on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    name_lower = player_name.lower()
    match = next(
        (p for p in roster if p.get("full_name", "").lower() == name_lower),
        None,
    )
    if match is None:
        # Fall back to partial match
        match = next(
            (p for p in roster if name_lower in p.get("full_name", "").lower()),
            None,
        )
    if match is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    pid = match["player_id"]
    full_name = match["full_name"]
    try:
        result = pipeline.generate(pid, season=season)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc

    return _build_tendency_response(result, full_name, pid, abbr, season)


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
