"""CSV loaders for NBA 2K tendency data."""
from __future__ import annotations

import csv
import json
from typing import Any


def load_scales_csv(filepath: str) -> dict[str, Any]:
    """
    Parse NBA_2K_Tendency_Scales.csv (semicolon-delimited).

    Returns dict keyed by tendency name with fields:
    order, name, definition, scale_bands, typical_range, hard_cap, notes.

    Raises ValueError on schema mismatch.
    """
    expected_columns = [
        "Order",
        "Tendency",
        "Definition",
        "Scale bands (0-100 meaning)",
        "Typical NBA range",
        "Hard cap",
        "Notes / locked rules",
    ]

    result: dict[str, Any] = {}
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)

        # Normalise header for comparison (strip BOM / whitespace)
        clean_header = [h.strip().lstrip("\ufeff") for h in header]
        if len(clean_header) < len(expected_columns):
            raise ValueError(
                f"Schema mismatch: expected {len(expected_columns)} columns, "
                f"got {len(clean_header)}.  Header: {clean_header}"
            )

        for row in reader:
            if not row or not row[0].strip():
                continue
            # Some rows have a pipe-prefixed line number — strip it
            order_raw = row[0].strip().lstrip("|")
            if not order_raw.isdigit():
                continue
            if len(row) < 7:
                row += [""] * (7 - len(row))
            name = row[1].strip()
            if not name:
                continue
            try:
                hard_cap_val: int | None = int(row[5].strip())
            except (ValueError, IndexError):
                hard_cap_val = None
            result[name] = {
                "order": int(order_raw),
                "name": name,
                "definition": row[2].strip(),
                "scale_bands": row[3].strip(),
                "typical_range": row[4].strip(),
                "hard_cap": hard_cap_val,
                "notes": row[6].strip() if len(row) > 6 else "",
            }

    if not result:
        raise ValueError(f"No data rows parsed from {filepath}")
    return result


def load_atd_csv(filepath: str) -> "import pandas; pandas.DataFrame":
    """
    Parse ATD Committee CSV.

    Validates that a player-name column and tendency columns are present.
    Reports basic data quality stats to stdout.
    Returns a pandas DataFrame.

    Raises ValueError if required structure is missing.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for load_atd_csv") from exc

    # Find the header row: it contains 'Tendency\n(In Order)' in col 0
    df_raw = pd.read_csv(
        filepath, header=None, dtype=str, encoding="utf-8", encoding_errors="replace"
    )

    header_row_idx: int | None = None
    for i, row in df_raw.iterrows():
        if str(row.iloc[0]).strip().replace("\n", "") in (
            "Tendency(In Order)",
            "Tendency\n(In Order)",
        ):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError(
            "Required header row ('Tendency\\n(In Order)') not found in ATD CSV."
        )

    # Re-read with the correct header
    df = pd.read_csv(
        filepath,
        header=header_row_idx,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
    )
    df = df.rename(columns={df.columns[0]: "player_name"})

    # Drop fully-empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Basic data quality report
    total = len(df)
    empty_name = df["player_name"].isna().sum()
    print(
        f"[load_atd_csv] {total} rows loaded; "
        f"{empty_name} rows with missing player_name; "
        f"{len(df.columns)} columns."
    )

    return df
