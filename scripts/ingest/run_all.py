"""Orchestrator — run all ingestion steps in order.

Usage:
    python -m scripts.ingest.run_all               # full run
    python -m scripts.ingest.run_all --skip-bios   # skip slow bio step
    python -m scripts.ingest.run_all --only box    # run one step
    python -m scripts.ingest.run_all --min-year 2016 --max-year 2025
    python -m scripts.ingest.run_all --force       # ignore done-flags
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── Helpers ────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA DuckDB warehouse — full ingest")
    parser.add_argument("--min-year", type=int, default=2000, help="First season start year")
    parser.add_argument("--max-year", type=int, default=2025, help="Last season start year")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already done")
    parser.add_argument("--skip-bios", action="store_true", help="Skip the slow bio step")
    parser.add_argument(
        "--only",
        choices=["box", "pbp", "pbp-shots", "rosters", "bios"],
        default=None,
        help="Run only a single step",
    )
    parser.add_argument("--rate-limit", type=float, default=0.7, help="Seconds between API calls")
    args = parser.parse_args()

    wall_start = time.time()

    # Determine which steps to run
    steps_all = ["box", "pbp", "pbp-shots", "rosters", "bios"]
    if args.only:
        steps = [args.only]
    elif args.skip_bios:
        steps = [s for s in steps_all if s != "bios"]
    else:
        steps = steps_all

    pbp_min = max(args.min_year, 2016)   # PBP data not available before 2016-17

    for step in steps:
        step_start = time.time()

        if step == "box":
            _section("Box Scores  (nba_api  LeagueDashPlayerStats)")
            from scripts.ingest.ingest_box_scores import ingest_box_scores  # noqa: PLC0415
            ingest_box_scores(
                min_year=args.min_year,
                max_year=args.max_year,
                force=args.force,
                rate_limit=args.rate_limit,
            )

        elif step == "pbp":
            if pbp_min > args.max_year:
                print("[pbp] no PBP seasons in requested range — skipping")
                continue
            _section(f"PBP Profiles  (api.pbpstats.com  {pbp_min}–{args.max_year})")
            os.environ.setdefault("N2K_PBP_ONLY", "1")
            from scripts.ingest.ingest_pbp import ingest_pbp  # noqa: PLC0415
            ingest_pbp(
                min_year=pbp_min,
                max_year=args.max_year,
                force=args.force,
            )

        elif step == "pbp-shots":
            if pbp_min > args.max_year:
                print("[pbp_shots] no PBP seasons in requested range — skipping")
                continue
            _section(f"PBP Raw Shots  (api.pbpstats.com  {pbp_min}–{args.max_year})")
            os.environ.setdefault("N2K_PBP_ONLY", "1")
            from scripts.ingest.ingest_pbp_shots import ingest_pbp_shots  # noqa: PLC0415
            ingest_pbp_shots(
                min_year=pbp_min,
                max_year=args.max_year,
                force=args.force,
            )

        elif step == "rosters":
            _section("Team Rosters  (nba_api  CommonTeamRoster)")
            from scripts.ingest.ingest_rosters import ingest_rosters  # noqa: PLC0415
            ingest_rosters(
                min_year=args.min_year,
                max_year=args.max_year,
                force=args.force,
                rate_limit=args.rate_limit,
            )

        elif step == "bios":
            _section("Player Bios  (nba_api  CommonPlayerInfo)")
            from scripts.ingest.ingest_bios import ingest_bios  # noqa: PLC0415
            ingest_bios(
                force=args.force,
                rate_limit=args.rate_limit,
            )

        print(f"\n  [{step}] finished in {_elapsed(step_start)}")

    print(f"\n{'═' * 60}")
    print(f"  All steps complete.  Total time: {_elapsed(wall_start)}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
