"""Evaluate one attribute across a fixed player test set.

Usage examples:
  python scripts/attribute_rework_eval.py --attribute driving_dunk --season 2024-25
  python scripts/attribute_rework_eval.py --attribute mid_range_shot --season 2024-25 --players "Jayson Tatum,Devin Booker,Kevin Durant"
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingest.nba_api_client import _strip_accents
from src.pipeline import TendencyPipeline
from src.seasons import normalize_season

DEFAULT_PLAYERS = [
    "Anthony Edwards",
    "Ja Morant",
    "Luka Doncic",
    "LeBron James",
    "Shai Gilgeous-Alexander",
    "Jayson Tatum",
    "Stephen Curry",
    "Giannis Antetokounmpo",
    "Nikola Jokic",
    "Joel Embiid",
]

EXTENDED_PLAYERS = [
    "Anthony Edwards",
    "Ja Morant",
    "Luka Doncic",
    "LeBron James",
    "Shai Gilgeous-Alexander",
    "Jayson Tatum",
    "Stephen Curry",
    "Giannis Antetokounmpo",
    "Nikola Jokic",
    "Joel Embiid",
    "Kevin Durant",
    "Kawhi Leonard",
    "DeMar DeRozan",
    "Zion Williamson",
    "Karl-Anthony Towns",
    "Domantas Sabonis",
    "Bam Adebayo",
    "Rudy Gobert",
    "Tyrese Haliburton",
    "De'Aaron Fox",
]


def _parse_players(raw: str | None, pool_size: int) -> list[str]:
    if not raw:
        return list(EXTENDED_PLAYERS if pool_size >= 20 else DEFAULT_PLAYERS)
    return [p.strip() for p in raw.split(",") if p.strip()]


def _resolve_player_id(pipeline: TendencyPipeline, query: str) -> tuple[int, str]:
    if query.isdigit():
        return int(query), query

    matches = pipeline.search_player(query)
    if not matches:
        raise ValueError(f"Player '{query}' not found")

    q = _strip_accents(query.lower())
    exact = next(
        (m for m in matches if _strip_accents(str(m.get("full_name", "")).lower()) == q),
        matches[0],
    )
    return int(exact["player_id"]), str(exact.get("full_name") or query)


def _to_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one attribute across 10 players.")
    parser.add_argument("--attribute", required=True, help="Canonical attribute key, e.g. driving_dunk")
    parser.add_argument("--season", default="2024-25", help="Season in YYYY-YY format")
    parser.add_argument(
        "--pool-size",
        type=int,
        choices=[10, 20],
        default=10,
        help="Use built-in default pool size when --players is omitted.",
    )
    parser.add_argument(
        "--players",
        default=None,
        help="Comma-separated names/ids. If omitted, uses a default 10-player mix.",
    )
    args = parser.parse_args()

    season = normalize_season(args.season)
    players = _parse_players(args.players, args.pool_size)

    print(
        f"Starting attribute benchmark for '{args.attribute}' in {season} across {len(players)} players...",
        flush=True,
    )

    pipeline = TendencyPipeline()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for idx, query in enumerate(players, start=1):
        try:
            print(f"[{idx}/{len(players)}] Resolving and generating: {query}", flush=True)
            player_id, resolved_name = _resolve_player_id(pipeline, query)
            result = pipeline.generate(player_id, season=season)

            attrs = result.get("attributes", {}) or {}
            features = result.get("features", {}) or {}

            if args.attribute not in attrs:
                failures.append(f"{resolved_name}: missing attribute '{args.attribute}'")
                continue

            rows.append(
                {
                    "name": resolved_name,
                    "id": player_id,
                    "position": str(result.get("position", "")),
                    "value": int(attrs.get(args.attribute, 0)),
                    "age": round(_to_num(features.get("age", 0))),
                    "gp": int(round(_to_num(features.get("gp", features.get("games_played", 0))))),
                    "pts_pg": round(_to_num(features.get("pts_per_game", 0)), 1),
                }
            )
            print(
                f"[{idx}/{len(players)}] Done: {resolved_name} -> {args.attribute}={rows[-1]['value']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{query}: {exc}")
            print(f"[{idx}/{len(players)}] Failed: {query} ({exc})", flush=True)

    if not rows:
        print("No successful player evaluations.")
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  - {f}")
        return

    rows.sort(key=lambda r: r["value"], reverse=True)

    print(f"\nAttribute benchmark: {args.attribute} | season={season} | players={len(rows)}")
    print("-" * 86)
    print(f"{'PLAYER':28} {'ID':>10} {'POS':>5} {'VAL':>5} {'AGE':>5} {'GP':>5} {'PTS':>6}")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['name'][:28]:28} "
            f"{row['id']:>10} "
            f"{row['position'][:5]:>5} "
            f"{row['value']:>5} "
            f"{int(row['age']):>5} "
            f"{row['gp']:>5} "
            f"{row['pts_pg']:>6.1f}"
        )

    values = [int(r["value"]) for r in rows]
    print("-" * 86)
    print(
        "Summary: "
        f"min={min(values)} "
        f"max={max(values)} "
        f"avg={statistics.mean(values):.1f} "
        f"median={statistics.median(values):.1f}"
    )

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
