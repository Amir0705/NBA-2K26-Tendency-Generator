"""Excel exporter — Phase 3 implementation."""
from __future__ import annotations

import io
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Colour constants
_HEADER_BG = "0F3460"   # dark blue
_RED_FILL = "FFCCCC"
_YELLOW_FILL = "FFFFCC"
_GREEN_FILL = "CCFFCC"
_DARK_GREEN_FILL = "66BB6A"


def _value_fill(value: int) -> PatternFill:
    """Return a PatternFill based on the tendency value range."""
    if value <= 20:
        colour = _RED_FILL
    elif value <= 40:
        colour = _YELLOW_FILL
    elif value <= 60:
        colour = _GREEN_FILL
    else:
        colour = _DARK_GREEN_FILL
    return PatternFill(start_color=colour, end_color=colour, fill_type="solid")


def _write_player_sheet(
    ws: Any,
    player_name: str,
    tendencies_dict: dict[str, int],
    registry: list[dict[str, Any]],
) -> None:
    """Populate a worksheet with a single player's tendencies."""
    ws.title = player_name[:31]  # Excel sheet names are max 31 chars

    # Header row
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color=_HEADER_BG, end_color=_HEADER_BG, fill_type="solid")
    for col, heading in enumerate(["Tendency", "Value", "Category"], start=1):
        cell = ws.cell(row=1, column=col, value=heading)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Data rows
    for row_idx, entry in enumerate(sorted(registry, key=lambda e: e["order"]), start=2):
        canon = entry["canonical_name"]
        label = entry["primjer_label"]
        category = entry.get("category", "")
        value = tendencies_dict.get(canon, 0)

        ws.cell(row=row_idx, column=1, value=label)
        val_cell = ws.cell(row=row_idx, column=2, value=value)
        val_cell.fill = _value_fill(value)
        val_cell.alignment = Alignment(horizontal="center")
        ws.cell(row=row_idx, column=3, value=category)

    # Auto-size columns
    for col_idx, col_cells in enumerate(ws.columns, start=1):
        max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 40)

    ws.freeze_panes = "A2"


def export_player_excel(
    player_name: str,
    tendencies_dict: dict[str, int],
    registry: list[dict[str, Any]],
    position: str = "",
) -> bytes:
    """Export single player tendencies to Excel bytes.

    Parameters
    ----------
    player_name:     Human-readable player name.
    tendencies_dict: canonical_name → integer value.
    registry:        Ordered registry entries (for labels and categories).
    position:        Optional player position.

    Returns
    -------
    xlsx file content as bytes.
    """
    wb = Workbook()
    ws = wb.active
    _write_player_sheet(ws, player_name, tendencies_dict, registry)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_team_excel(
    team_abbr: str,
    team_data: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> bytes:
    """Export team tendencies to Excel bytes.

    Creates a workbook with:
    - A summary sheet listing all players with key tendencies.
    - One sheet per player with full tendencies.

    Parameters
    ----------
    team_abbr:  Team abbreviation (used in the summary sheet title).
    team_data:  List of {player_name, position, tendencies: {canonical_name: int}}.
    registry:   Ordered registry entries.

    Returns
    -------
    xlsx file content as bytes.
    """
    wb = Workbook()
    summary_ws = wb.active
    summary_ws.title = "Summary"

    # Pick a handful of key tendencies for the summary sheet
    key_canons = [
        "shot_three", "shot_mid_range", "shot_close", "driving_layup",
        "on_ball_defense", "post_up",
    ]
    key_entries = [e for e in registry if e["canonical_name"] in key_canons]
    key_entries.sort(key=lambda e: e["order"])

    # Summary header
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color=_HEADER_BG, end_color=_HEADER_BG, fill_type="solid")
    summary_headers = ["Player", "Position"] + [e["primjer_label"] for e in key_entries]
    for col, heading in enumerate(summary_headers, start=1):
        cell = summary_ws.cell(row=1, column=col, value=heading)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Summary rows
    for row_idx, player in enumerate(team_data, start=2):
        tendencies = player.get("tendencies", {})
        summary_ws.cell(row=row_idx, column=1, value=player.get("player_name", ""))
        summary_ws.cell(row=row_idx, column=2, value=player.get("position", ""))
        for col_idx, entry in enumerate(key_entries, start=3):
            val = tendencies.get(entry["canonical_name"], 0)
            cell = summary_ws.cell(row=row_idx, column=col_idx, value=val)
            cell.fill = _value_fill(val)
            cell.alignment = Alignment(horizontal="center")

    for col_idx, col_cells in enumerate(summary_ws.columns, start=1):
        max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
        summary_ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 40)
    summary_ws.freeze_panes = "A2"

    # Per-player sheets
    for player in team_data:
        ws = wb.create_sheet()
        _write_player_sheet(
            ws,
            player.get("player_name", "Player"),
            player.get("tendencies", {}),
            registry,
        )

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Legacy file-based helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def export_bulk_excel(
    players: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write tendency values for multiple players to a single Excel workbook.

    Parameters
    ----------
    players:     List of {name, tendencies} dicts.
    registry:    Ordered registry entries.
    output_path: Destination .xlsx file path.
    """
    team_data = [
        {
            "player_name": p.get("name", ""),
            "position": p.get("position", ""),
            "tendencies": p.get("tendencies", {}),
        }
        for p in players
    ]
    content = export_team_excel("", team_data, registry)
    with open(output_path, "wb") as fh:
        fh.write(content)
