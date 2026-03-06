"""FastAPI web server for the NBA 2K26 Tendency Generator."""
from __future__ import annotations

import json
import os
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.export.csv_exporter import export_player_csv, export_team_csv
from src.export.excel_exporter import export_player_excel, export_team_excel
from src.export.json_exporter import export_player_json
from src.pipeline import TendencyPipeline

_VALID_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}
_DEFAULT_TEAM_ROSTER_SEASON = "2025-26"

_pipeline: TendencyPipeline | None = None

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")


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

# Serve frontend static files
_frontend_abs = os.path.normpath(_FRONTEND_DIR)
if os.path.isdir(_frontend_abs):
    app.mount("/static", StaticFiles(directory=_frontend_abs), name="static")


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
            "offset": entry["offset"],
            "type": entry["type"],
            "bit_offset": entry["bit_offset"],
            "bit_length": entry["bit_length"],
            "length": entry["length"],
        }
    guardrails = result.get("guardrail_violations", [])
    errors = result.get("errors", [])
    audit = result.get("audit", [])
    return {
        "player_name": player_name,
        "player_id": player_id,
        "position": result.get("position", ""),
        "team": team,
        "season": season,
        "tendencies": ordered_tendencies,
        "debug": {
            "guardrail_count": len(guardrails),
            "error_count": len(errors),
            "guardrail_violations": guardrails,
            "errors": errors,
            "audit_sample": audit[:20],
            "feature_summary": result.get("features", {}),
        },
    }


def _safe_filename(name: str) -> str:
    """Convert a name into a safe ASCII filename segment."""
    return name.lower().replace(" ", "_").replace("/", "_")


@app.get("/")
def root() -> Response:
    """Serve the web frontend."""
    index_path = os.path.join(_frontend_abs, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return Response(
        content='{"status":"ok","version":"1.0.0"}',
        media_type="application/json",
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
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
def generate_team(
    team_abbr: str,
    season: str = "2024-25",
    roster_season: str = _DEFAULT_TEAM_ROSTER_SEASON,
) -> dict[str, Any]:
    """Generate tendencies for all players on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    players: list[dict[str, Any]] = []
    total_players = len(roster)
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
                ordered[key] = {
                    "value": entry["value"],
                    "label": entry["label"],
                    "offset": entry["offset"],
                    "type": entry["type"],
                    "bit_offset": entry["bit_offset"],
                    "bit_length": entry["bit_length"],
                    "length": entry["length"],
                }
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
        "roster_season": roster_season,
        "total_players": total_players,
        "generated_count": len(players),
        "failed_count": max(0, total_players - len(players)),
        "player_count": len(players),
        "players": players,
    }


@app.get("/team/{team_abbr}/{player_name}")
def generate_team_player(
    team_abbr: str,
    player_name: str,
    season: str = "2024-25",
    roster_season: str = _DEFAULT_TEAM_ROSTER_SEASON,
) -> dict[str, Any]:
    """Generate tendencies for a specific player on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
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


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

def _resolve_player(player_name: str, season: str, pipeline: TendencyPipeline) -> tuple[str, str, dict[str, int]]:
    """Search, generate, and return (full_name, position, canonical tendencies)."""
    results = pipeline.search_player(player_name)
    if not results:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    name_lower = player_name.lower()
    match = next((r for r in results if r.get("full_name", "").lower() == name_lower), results[0])
    pid = match["player_id"]
    full_name = match["full_name"]
    try:
        result = pipeline.generate(pid, season=season)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc
    position = result.get("position", "")
    tendencies = result.get("tendencies", {})
    return full_name, position, tendencies


@app.get("/export/csv/{player_name}")
def export_csv_player(player_name: str, season: str = "2024-25") -> Response:
    """Export a single player's tendencies as a CSV file."""
    pipeline = _get_pipeline()
    full_name, position, tendencies = _resolve_player(player_name, season, pipeline)
    csv_str = export_player_csv(full_name, tendencies, pipeline._registry, position)
    filename = f"{_safe_filename(full_name)}_tendencies.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/excel/{player_name}")
def export_excel_player(player_name: str, season: str = "2024-25") -> Response:
    """Export a single player's tendencies as an Excel file."""
    pipeline = _get_pipeline()
    full_name, position, tendencies = _resolve_player(player_name, season, pipeline)
    xlsx_bytes = export_player_excel(full_name, tendencies, pipeline._registry, position)
    filename = f"{_safe_filename(full_name)}_tendencies.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/csv/team/{team_abbr}")
def export_csv_team(
    team_abbr: str,
    season: str = "2024-25",
    roster_season: str = _DEFAULT_TEAM_ROSTER_SEASON,
) -> Response:
    """Export a full team's tendencies as a CSV file."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    team_data: list[dict[str, Any]] = []
    for player in roster:
        try:
            result = pipeline.generate(player["player_id"], season=season)
            team_data.append({
                "player_name": player["full_name"],
                "position": result.get("position", ""),
                "tendencies": result.get("tendencies", {}),
            })
        except Exception:  # noqa: BLE001
            continue
    csv_str = export_team_csv(team_data, pipeline._registry)
    filename = f"{abbr}_roster_tendencies.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/excel/team/{team_abbr}")
def export_excel_team(
    team_abbr: str,
    season: str = "2024-25",
    roster_season: str = _DEFAULT_TEAM_ROSTER_SEASON,
) -> Response:
    """Export a full team's tendencies as an Excel file."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    team_data: list[dict[str, Any]] = []
    for player in roster:
        try:
            result = pipeline.generate(player["player_id"], season=season)
            team_data.append({
                "player_name": player["full_name"],
                "position": result.get("position", ""),
                "tendencies": result.get("tendencies", {}),
            })
        except Exception:  # noqa: BLE001
            continue
    xlsx_bytes = export_team_excel(abbr, team_data, pipeline._registry)
    filename = f"{abbr}_roster_tendencies.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/json/team/{team_abbr}")
def export_json_team_zip(
    team_abbr: str,
    season: str = "2024-25",
    roster_season: str = _DEFAULT_TEAM_ROSTER_SEASON,
) -> Response:
    """Export a team's tendencies as a ZIP with one JSON file per player."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for player in roster:
            pid = player["player_id"]
            full_name = player["full_name"]
            try:
                result = pipeline.generate(pid, season=season)
                tendencies = result.get("tendencies", {})
                player_json = export_player_json(tendencies, pipeline._registry)
                member_name = f"{_safe_filename(full_name)}_{pid}_tendencies.json"
                zf.writestr(member_name, player_json)
            except Exception:  # noqa: BLE001
                continue

    filename = f"{abbr}_roster_tendencies_json.zip"
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
