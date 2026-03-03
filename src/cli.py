"""Command-line interface for the NBA 2K26 Tendency Generator."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

_VALID_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}


def _safe_filename(name: str) -> str:
    """Convert a player name to a safe filename slug."""
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "") + "_tendencies.json"


def _build_pipeline(season: str) -> Any:
    from src.pipeline import TendencyPipeline  # noqa: PLC0415
    return TendencyPipeline(cache_dir=".cache")


def _resolve_player(pipeline: Any, name: str) -> dict | None:
    """Return the first search result for *name*, or None."""
    results = pipeline.search_player(name)
    if not results:
        return None
    # Prefer exact match (case-insensitive) over partial match
    name_lower = name.lower()
    for r in results:
        if r.get("full_name", "").lower() == name_lower:
            return r
    return results[0]


def _generate_and_save(
    pipeline: Any,
    player_id: int,
    player_name: str,
    season: str,
    out_path: str,
) -> dict | None:
    """Generate tendencies and write JSON file. Returns result dict or None on error."""
    try:
        result = pipeline.generate(player_id, season=season)
        result["player_name"] = player_name
        from src.export.json_exporter import export_player_json  # noqa: PLC0415
        registry = pipeline._registry
        tendencies = result.get("tendencies", {})
        json_str = export_player_json(tendencies, registry)
        payload = json.loads(json_str)
        payload["player_name"] = player_name
        payload["player_id"] = player_id
        payload["season"] = season
        payload["position"] = result.get("position", "")
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR generating {player_name}: {exc}", file=sys.stderr)
        return None


def _print_summary(players: list[dict]) -> None:
    """Print a summary table of generated players."""
    if not players:
        return
    header = f"{'Player':<25} {'Pos':<5} {'Shot':>5} {'Drive':>6} {'Shot3':>6} {'Pass':>5}"
    print("\n" + header)
    print("-" * len(header))
    for p in players:
        name = p.get("player_name", "Unknown")[:24]
        pos = p.get("position", "?")[:4]
        t = p.get("tendencies", {})
        shot = t.get("shot_tendency", "?")
        drive = t.get("drive_tendency", "?")
        shot3 = t.get("shot_three_tendency", "?")
        pass_ = t.get("pass_tendency", "?")
        print(f"{name:<25} {pos:<5} {shot!s:>5} {drive!s:>6} {shot3!s:>6} {pass_!s:>5}")


def cmd_search(args: argparse.Namespace) -> None:
    """Handle --search flag."""
    pipeline = _build_pipeline(args.season)
    results = pipeline.search_player(args.search)
    if not results:
        print(f"No players found matching '{args.search}'.")
        return
    print(f"Found {len(results)} player(s) matching '{args.search}':")
    for r in results:
        status = "active" if r.get("is_active") else "inactive"
        print(f"  ID={r['player_id']:>7}  {r['full_name']:<30} {r.get('team', '?'):<4} ({status})")


def cmd_single(args: argparse.Namespace) -> None:
    """Handle single-player generation."""
    pipeline = _build_pipeline(args.season)
    player_info = _resolve_player(pipeline, args.player_name)
    if player_info is None:
        print(f"Player '{args.player_name}' not found.", file=sys.stderr)
        sys.exit(1)

    player_id = player_info["player_id"]
    player_name = player_info["full_name"]
    filename = _safe_filename(player_name)
    out_path = os.path.join(args.output_dir, filename)

    print(f"Generating tendencies for {player_name} (ID={player_id}, season={args.season})...")
    result = _generate_and_save(pipeline, player_id, player_name, args.season, out_path)
    if result is None:
        sys.exit(1)

    print(f"Saved → {out_path}")
    _print_summary([result])


def cmd_team(args: argparse.Namespace) -> None:
    """Handle --team flag."""
    team_abbr = args.team.upper()
    if team_abbr not in _VALID_TEAMS:
        print(f"Invalid team abbreviation '{team_abbr}'. Valid teams: {', '.join(sorted(_VALID_TEAMS))}", file=sys.stderr)
        sys.exit(1)

    pipeline = _build_pipeline(args.season)
    roster = pipeline._client.get_team_roster(team_abbr, season=args.season)
    if not roster:
        print(f"No roster found for team '{team_abbr}'.", file=sys.stderr)
        sys.exit(1)

    team_dir = os.path.join(args.output_dir, team_abbr)
    os.makedirs(team_dir, exist_ok=True)

    total = len(roster)
    results: list[dict] = []
    for i, player in enumerate(roster, 1):
        player_id = player["player_id"]
        player_name = player["full_name"]
        print(f"Generating {i}/{total}: {player_name}...")
        filename = _safe_filename(player_name)
        out_path = os.path.join(team_dir, filename)
        result = _generate_and_save(pipeline, player_id, player_name, args.season, out_path)
        if result is not None:
            result["player_name"] = player_name
            results.append(result)
        else:
            print(f"  Skipping {player_name} due to error.")

    print(f"\nGenerated {len(results)}/{total} players → {team_dir}/")
    _print_summary(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="NBA 2K26 Tendency Generator CLI",
    )
    parser.add_argument(
        "player_name",
        nargs="?",
        default=None,
        help="Player name for single-player generation (e.g. 'Stephen Curry')",
    )
    parser.add_argument("--team", metavar="ABBR", help="Generate tendencies for an entire team (e.g. GSW)")
    parser.add_argument("--search", metavar="NAME", help="Search players by name without generating tendencies")
    parser.add_argument("--season", default="2024-25", metavar="YYYY-YY", help="NBA season (default: 2024-25)")
    parser.add_argument("--output-dir", default="output", metavar="PATH", help="Output directory (default: output/)")

    args = parser.parse_args(argv)

    if args.search:
        cmd_search(args)
    elif args.team:
        cmd_team(args)
    elif args.player_name:
        cmd_single(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
