"""CSV exporter — Phase 3 implementation."""
from __future__ import annotations

import csv
import io
from typing import Any


def export_player_csv(
    player_name: str,
    tendencies_dict: dict[str, int],
    registry: list[dict[str, Any]],
    position: str = "",
) -> str:
    """Export single player tendencies to CSV string.

    Parameters
    ----------
    player_name:     Human-readable player name.
    tendencies_dict: canonical_name → integer value.
    registry:        Ordered registry entries (for labels and categories).
    position:        Optional player position.

    Returns
    -------
    CSV as a UTF-8 string with columns: Tendency, Value, Category.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Tendency", "Value", "Category"])
    for entry in sorted(registry, key=lambda e: e["order"]):
        canon = entry["canonical_name"]
        label = entry["primjer_label"]
        category = entry.get("category", "")
        value = tendencies_dict.get(canon, 0)
        writer.writerow([label, value, category])
    return output.getvalue()


def export_team_csv(
    team_data: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> str:
    """Export team tendencies to CSV string.

    Parameters
    ----------
    team_data: List of {player_name, position, tendencies: {canonical_name: int}}.
    registry:  Ordered registry entries (for labels).

    Returns
    -------
    CSV with players as rows and tendencies as columns.
    """
    sorted_entries = sorted(registry, key=lambda e: e["order"])
    output = io.StringIO()
    writer = csv.writer(output)
    header = ["Player", "Position"] + [e["primjer_label"] for e in sorted_entries]
    writer.writerow(header)
    for player in team_data:
        tendencies = player.get("tendencies", {})
        row = [player.get("player_name", ""), player.get("position", "")]
        row += [tendencies.get(e["canonical_name"], 0) for e in sorted_entries]
        writer.writerow(row)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Legacy file-based helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def export_bulk_csv(
    players: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write tendency values for multiple players to a single CSV file.

    Parameters
    ----------
    players:     List of {name, tendencies} dicts.
    registry:    Ordered registry entries.
    output_path: Destination file path.
    """
    team_data = [
        {
            "player_name": p.get("name", ""),
            "position": p.get("position", ""),
            "tendencies": p.get("tendencies", {}),
        }
        for p in players
    ]
    content = export_team_csv(team_data, registry)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(content)
