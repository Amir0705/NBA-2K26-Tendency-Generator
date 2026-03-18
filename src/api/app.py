"""FastAPI web server for the NBA 2K26 ATD Committee Helper Tool."""
from __future__ import annotations

import json
import os
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.auth_store import AuthStore
from src.api.firebase_auth_store import FirebaseAuthStore
from src.export.excel_exporter import export_player_excel, export_team_excel, export_team_attributes_excel
from src.export.json_exporter import export_player_json
from src.export.two_k_exporter import export_player_2k_json
from src.attributes.calculator import ATTRIBUTE_LABELS, ATTRIBUTE_CATEGORIES
from src.feedback.feedback_store import FeedbackStore
from src.feedback.firebase_feedback_store import FirebaseFeedbackStore
from src.pipeline import TendencyPipeline
from src.seasons import DEFAULT_SEASON, data_mode_for_season, normalize_season

_VALID_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
    "CHH", "NJN", "NOH", "NOK", "SEA", "VAN",
}

_pipeline: TendencyPipeline | None = None
_auth_store: Any = None
_feedback_store: Any = None

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[type-arg]
    global _pipeline, _auth_store, _feedback_store  # noqa: PLW0603
    cache_dir = "data/cache" if os.path.isdir("data/cache") else ".cache"
    _pipeline = TendencyPipeline(cache_dir=cache_dir)

    use_firebase = (os.environ.get("N2K_USE_FIREBASE", "0").strip().lower() in {"1", "true", "yes"})
    if use_firebase:
        try:
            _auth_store = FirebaseAuthStore()
            _feedback_store = FirebaseFeedbackStore()
        except Exception:
            _auth_store = AuthStore(users_path=os.path.join("data", "auth", "users.json"))
            _feedback_store = FeedbackStore(store_path=os.path.join("data", "feedback", "entries.json"))
    else:
        _auth_store = AuthStore(users_path=os.path.join("data", "auth", "users.json"))
        _feedback_store = FeedbackStore(store_path=os.path.join("data", "feedback", "entries.json"))
    yield


app = FastAPI(title="NBA 2K26 ATD Committee Helper Tool", version="1.0.0", lifespan=lifespan)

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


def _get_auth_store() -> Any:
    if _auth_store is None:
        raise HTTPException(status_code=503, detail="Auth store not initialized")
    return _auth_store


def _get_feedback_store() -> Any:
    if _feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    return _feedback_store


def _extract_bearer_token(request: Request) -> str:
    auth_header = str(request.headers.get("authorization", ""))
    if not auth_header.lower().startswith("bearer "):
        return ""
    return auth_header[7:].strip()


def _get_request_user(request: Request) -> dict[str, Any]:
    existing = getattr(request.state, "user", None)
    if isinstance(existing, dict):
        return existing
    token = _extract_bearer_token(request)
    user = _get_auth_store().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    request.state.user = user
    return user


def _require_role(request: Request, allowed_roles: set[str]) -> dict[str, Any]:
    user = _get_request_user(request)
    role = str(user.get("role", "")).lower()
    if role not in allowed_roles:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user


def _apply_feedback_learning(player_id: int, result: dict[str, Any]) -> dict[str, Any]:
    tendencies = result.get("tendencies", {})
    if not isinstance(tendencies, dict) or not tendencies:
        return result

    store = _get_feedback_store()
    adjusted: dict[str, int] = {}
    applied_count = 0
    for tendency_key, base_value in tendencies.items():
        try:
            base = int(round(float(base_value)))
        except (TypeError, ValueError):
            adjusted[tendency_key] = 0
            continue

        agg = store.aggregate(player_id=player_id, tendency_name=tendency_key)
        votes = int(agg.get("vote_count") or 0)
        mean_value = agg.get("mean_value")
        if votes <= 0 or mean_value is None:
            adjusted[tendency_key] = max(0, min(100, base))
            continue

        alpha = min(0.40, 0.08 * votes)
        learned = int(round(base * (1.0 - alpha) + float(mean_value) * alpha))
        learned = max(0, min(100, learned))
        adjusted[tendency_key] = learned
        if learned != base:
            applied_count += 1

    next_result = dict(result)
    next_result["tendencies"] = adjusted
    if applied_count > 0:
        next_result["feedback_learning"] = {
            "applied_count": applied_count,
        }
    return next_result


def _to_canonical_tendency_key(raw_key: str, valid_keys: set[str], registry: list[dict[str, Any]]) -> str:
    key = str(raw_key or "").strip()
    if not key:
        return ""
    if key in valid_keys:
        return key

    for row in registry:
        canonical = str(row.get("canonical_name", "")).strip()
        primjer_key = str(row.get("primjer_key", "")).strip()
        primjer_label = str(row.get("primjer_label", "")).strip()
        scales_name = str(row.get("scales_csv_name", "")).strip()
        if key == primjer_key or key == primjer_label or key == scales_name:
            if canonical in valid_keys:
                return canonical
    return ""


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if (
        path in {"/", "/app", "/admin", "/health", "/auth/login"}
        or path.startswith("/static")
        or path.startswith("/export/")
        or path.startswith("/docs")
        or path.startswith("/openapi.json")
        or path == "/favicon.ico"
    ):
        return await call_next(request)

    token = _extract_bearer_token(request)
    user = _get_auth_store().get_user_by_token(token)
    if not user:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    request.state.user = user
    return await call_next(request)


def _get_pipeline() -> TendencyPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def _require_supported_season(season: str) -> str:
    """Validate and normalize season query strings."""
    try:
        return normalize_season(season, default=DEFAULT_SEASON)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_roster_season(season: str, roster_season: str | None) -> str:
    """Use selected season for roster by default unless override is provided."""
    candidate = str(roster_season or "").strip() or season
    return _require_supported_season(candidate)


@app.post("/auth/login")
def auth_login(payload: dict[str, Any]) -> dict[str, Any]:
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password are required")

    user = _get_auth_store().login(username=username, password=password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return user


@app.get("/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    return _get_request_user(request)


@app.post("/auth/change-password")
def auth_change_password(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    user = _get_request_user(request)
    new_password = str(payload.get("new_password", "")).strip()
    if len(new_password) < 4:
        raise HTTPException(status_code=400, detail="new_password must be at least 4 characters")

    _get_auth_store().change_password(username=str(user.get("username", "")), new_password=new_password)
    return {"ok": True}


@app.get("/auth/users")
def auth_list_users(request: Request) -> dict[str, Any]:
    _require_role(request, {"admin"})
    return {"users": _get_auth_store().list_users()}


@app.post("/auth/users")
def auth_create_user(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _require_role(request, {"admin"})
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    role = str(payload.get("role", "viewer")).strip().lower()
    full_name = str(payload.get("full_name", "")).strip() or None

    try:
        created = _get_auth_store().create_user(
            username=username,
            password=password,
            role=role,
            full_name=full_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "user": created}


@app.put("/auth/users/{username}")
def auth_update_user(username: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _require_role(request, {"admin"})

    full_name = payload.get("full_name")
    if full_name is not None:
        full_name = str(full_name)

    role = payload.get("role")
    if role is not None:
        role = str(role).strip().lower()

    reset_password = payload.get("reset_password")
    if reset_password is not None:
        reset_password = str(reset_password)

    must_change_password = payload.get("must_change_password")
    if must_change_password is not None:
        must_change_password = bool(must_change_password)

    try:
        updated = _get_auth_store().update_user(
            username=username,
            full_name=full_name,
            role=role,
            reset_password=reset_password,
            must_change_password=must_change_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True, "user": updated}


@app.delete("/auth/users/{username}")
def auth_delete_user(username: str, request: Request) -> dict[str, Any]:
    current = _require_role(request, {"admin"})
    try:
        _get_auth_store().delete_user(username=username, acting_username=str(current.get("username", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/feedback/submit")
def feedback_submit(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    user = _require_role(request, {"editor", "admin"})
    try:
        player_id = int(payload.get("player_id"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="player_id is required") from exc

    corrections = payload.get("corrections", {})
    if not isinstance(corrections, dict) or not corrections:
        raise HTTPException(status_code=400, detail="corrections must be a non-empty object")

    notes = str(payload.get("notes", "")).strip()
    reviewer = str(user.get("username", "editor"))
    store = _get_feedback_store()
    pipeline = _get_pipeline()

    # Resolve canonical tendency keys from player output and registry metadata.
    try:
        feedback_season = _require_supported_season(str(payload.get("season", DEFAULT_SEASON)))
        generated = pipeline.generate(player_id, season=feedback_season)
    except Exception:
        generated = {"tendencies": {}}
    valid_keys = set((generated.get("tendencies") or {}).keys())
    registry = pipeline._registry

    ids: list[str] = []
    for tendency_name, suggested_value in corrections.items():
        canonical = _to_canonical_tendency_key(
            raw_key=str(tendency_name),
            valid_keys=valid_keys,
            registry=registry,
        )
        if not canonical:
            raise HTTPException(status_code=400, detail=f"Unknown tendency key: {tendency_name}")
        try:
            fid = store.submit(
                player_id=player_id,
                tendency_name=canonical,
                suggested_value=int(round(float(suggested_value))),
                reviewer=reviewer,
                notes=notes,
            )
            ids.append(fid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid correction for {tendency_name}: {exc}") from exc

    return {"ok": True, "saved_count": len(ids), "feedback_ids": ids}


@app.get("/feedback/player/{player_id}")
def feedback_for_player(player_id: int, request: Request) -> dict[str, Any]:
    _require_role(request, {"editor", "admin"})
    entries = _get_feedback_store().get_for_player(player_id)
    return {"player_id": player_id, "entries": entries}


@app.get("/feedback/summary/{player_id}/{tendency_key}")
def feedback_summary(player_id: int, tendency_key: str, request: Request) -> dict[str, Any]:
    _require_role(request, {"editor", "admin"})
    summary = _get_feedback_store().aggregate(player_id, tendency_key)
    return {"player_id": player_id, "tendency_key": tendency_key, **summary}


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

    # Build structured attributes dict
    raw_attrs = result.get("attributes", {})
    attributes: dict[str, Any] = {}
    for key, value in raw_attrs.items():
        attributes[key] = {
            "value": value,
            "label": ATTRIBUTE_LABELS.get(key, key),
            "category": ATTRIBUTE_CATEGORIES.get(key, ""),
        }

    return {
        "player_name": player_name,
        "player_id": player_id,
        "position": result.get("position", ""),
        "team": team,
        "season": season,
        "data_mode": data_mode_for_season(season),
        "play_style_count": result.get("play_style_count", 0),
        "play_style_usage_rate": result.get("play_style_usage_rate", 0.0),
        "play_style_priorities": result.get("play_style_priorities", []),
        "play_style_weights": result.get("play_style_weights", {}),
        "play_style_scores": result.get("play_style_scores", {}),
        "tendencies": ordered_tendencies,
        "attributes": attributes,
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
    """Serve login page as the public entrypoint."""
    index_path = os.path.join(_frontend_abs, "login.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return Response(
        content='{"status":"ok","version":"1.0.0"}',
        media_type="application/json",
    )


@app.get("/app")
def app_page() -> Response:
    """Serve the main generation app page."""
    index_path = os.path.join(_frontend_abs, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return Response(
        content='{"status":"ok","version":"1.0.0"}',
        media_type="application/json",
    )


@app.get("/admin")
def admin_page() -> Response:
    """Serve the admin user-management page."""
    index_path = os.path.join(_frontend_abs, "admin.html")
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
def generate_by_id(player_id: int, season: str = DEFAULT_SEASON) -> dict[str, Any]:
    """Generate tendencies by player ID."""
    pipeline = _get_pipeline()
    season = _require_supported_season(season)
    try:
        result = pipeline.generate(player_id, season=season)
        result = _apply_feedback_learning(player_id=player_id, result=result)
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
def generate_by_name(player_name: str, season: str = DEFAULT_SEASON) -> dict[str, Any]:
    """Generate tendencies for a player by name."""
    pipeline = _get_pipeline()
    season = _require_supported_season(season)
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
        result = _apply_feedback_learning(player_id=pid, result=result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc

    return _build_tendency_response(result, full_name, pid, team, season)


@app.get("/team/{team_abbr}")
def generate_team(
    team_abbr: str,
    season: str = DEFAULT_SEASON,
    roster_season: str | None = None,
) -> dict[str, Any]:
    """Generate tendencies for all players on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    season = _require_supported_season(season)
    roster_season = _resolve_roster_season(season, roster_season)

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
            result = _apply_feedback_learning(player_id=pid, result=result)
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
                "play_style_count": result.get("play_style_count", 0),
                "play_style_usage_rate": result.get("play_style_usage_rate", 0.0),
                "play_style_priorities": result.get("play_style_priorities", []),
                "play_style_weights": result.get("play_style_weights", {}),
                "play_style_scores": result.get("play_style_scores", {}),
                "tendencies": ordered,
                "attributes": {
                    k: {"value": v, "label": ATTRIBUTE_LABELS.get(k, k), "category": ATTRIBUTE_CATEGORIES.get(k, "")}
                    for k, v in result.get("attributes", {}).items()
                },
            })
        except Exception:  # noqa: BLE001
            continue

    return {
        "team": abbr,
        "season": season,
        "roster_season": roster_season,
        "data_mode": data_mode_for_season(season),
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
    season: str = DEFAULT_SEASON,
    roster_season: str | None = None,
) -> dict[str, Any]:
    """Generate tendencies for a specific player on a team."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    season = _require_supported_season(season)
    roster_season = _resolve_roster_season(season, roster_season)

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
        result = _apply_feedback_learning(player_id=pid, result=result)
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
        result = _apply_feedback_learning(player_id=pid, result=result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc
    position = result.get("position", "")
    tendencies = result.get("tendencies", {})
    return full_name, position, tendencies


def _resolve_player_full(
    player_name: str,
    season: str,
    pipeline: TendencyPipeline,
) -> tuple[str, int, str, dict[str, Any]]:
    """Search, generate, and return full player generation context."""
    results = pipeline.search_player(player_name)
    if not results:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    name_lower = player_name.lower()
    match = next((r for r in results if r.get("full_name", "").lower() == name_lower), results[0])
    pid = int(match["player_id"])
    full_name = str(match.get("full_name") or player_name)
    team = str(match.get("team") or "")
    try:
        result = pipeline.generate(pid, season=season)
        result = _apply_feedback_learning(player_id=pid, result=result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate tendencies: {exc}") from exc
    return full_name, pid, team, result


@app.get("/export/excel/{player_name}")
def export_excel_player(player_name: str, season: str = DEFAULT_SEASON) -> Response:
    """Export a single player's tendencies as an Excel file."""
    pipeline = _get_pipeline()
    season = _require_supported_season(season)
    full_name, position, tendencies = _resolve_player(player_name, season, pipeline)
    xlsx_bytes = export_player_excel(full_name, tendencies, pipeline._registry, position)
    filename = f"{_safe_filename(full_name)}_tendencies.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/excel/team/{team_abbr}")
def export_excel_team(
    team_abbr: str,
    season: str = DEFAULT_SEASON,
    roster_season: str | None = None,
) -> Response:
    """Export a full team's tendencies as an Excel file."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    season = _require_supported_season(season)
    roster_season = _resolve_roster_season(season, roster_season)

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    team_data: list[dict[str, Any]] = []
    for player in roster:
        try:
            result = pipeline.generate(player["player_id"], season=season)
            result = _apply_feedback_learning(player_id=player["player_id"], result=result)
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


@app.get("/export/excel/team/{team_abbr}/attributes")
def export_excel_team_attributes(
    team_abbr: str,
    season: str = DEFAULT_SEASON,
    roster_season: str | None = None,
) -> Response:
    """Export a full team's attributes as an Excel file."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    season = _require_supported_season(season)
    roster_season = _resolve_roster_season(season, roster_season)

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")
    team_data: list[dict[str, Any]] = []
    for player in roster:
        try:
            result = pipeline.generate(player["player_id"], season=season)
            result = _apply_feedback_learning(player_id=player["player_id"], result=result)
            team_data.append({
                "player_name": player["full_name"],
                "position": result.get("position", ""),
                "attributes": result.get("attributes", {}),
            })
        except Exception:  # noqa: BLE001
            continue
    xlsx_bytes = export_team_attributes_excel(abbr, team_data)
    filename = f"{abbr}_roster_attributes.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/2k/{player_name}")
def export_2k_player(player_name: str, season: str = DEFAULT_SEASON) -> Response:
    """Export a single player as a full 2K-style JSON using the template format."""
    pipeline = _get_pipeline()
    season = _require_supported_season(season)
    full_name, pid, team, result = _resolve_player_full(player_name, season, pipeline)
    payload = export_player_2k_json(
        player_name=full_name,
        player_id=pid,
        team=team,
        tendencies=result.get("tendencies", {}),
        attributes=result.get("attributes", {}),
        registry=pipeline._registry,
    )
    filename = f"{_safe_filename(full_name)}_2k_export.json"
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/2k/team/{team_abbr}")
def export_2k_team_zip(
    team_abbr: str,
    season: str = DEFAULT_SEASON,
    roster_season: str | None = None,
) -> Response:
    """Export a team's players as a ZIP of full 2K-style JSON files."""
    abbr = team_abbr.upper()
    if abbr not in _VALID_TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    season = _require_supported_season(season)
    roster_season = _resolve_roster_season(season, roster_season)

    pipeline = _get_pipeline()
    roster = pipeline._client.get_team_roster(abbr, season=roster_season)
    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_abbr}' not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for player in roster:
            pid = int(player["player_id"])
            full_name = str(player.get("full_name") or pid)
            try:
                result = pipeline.generate(pid, season=season)
                result = _apply_feedback_learning(player_id=pid, result=result)
                player_json = export_player_2k_json(
                    player_name=full_name,
                    player_id=pid,
                    team=abbr,
                    tendencies=result.get("tendencies", {}),
                    attributes=result.get("attributes", {}),
                    registry=pipeline._registry,
                )
                member_name = f"{_safe_filename(full_name)}_{pid}_2k_export.json"
                zf.writestr(member_name, player_json)
            except Exception:  # noqa: BLE001
                continue

    filename = f"{abbr}_roster_2k_export.zip"
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
