#!/usr/bin/env python3
"""Replace checkpoint-specific tuning columns in a Figure 4 unit table.

The expensive real-trace scorer uses the tuning CSV only as metadata; it does
not affect any movie score.  This utility lets a selected twin's freshly
computed SF/TF groups replace the historical metadata after shard merging,
without re-scoring the 100,000-movie matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_tuning(
    *,
    unit_feature_table: Path,
    new_unit_tuning_csv: Path,
    out_path: Path,
    previous_unit_tuning_csv: Path | None = None,
) -> dict:
    input_unit_feature_sha256 = _sha256(unit_feature_table)
    new_unit_tuning_sha256 = _sha256(new_unit_tuning_csv)
    previous_unit_tuning_sha256 = (
        _sha256(previous_unit_tuning_csv)
        if previous_unit_tuning_csv is not None
        else None
    )
    base = pd.read_csv(unit_feature_table)
    tuning = pd.read_csv(new_unit_tuning_csv)
    if "unit_index" not in base or "unit_index" not in tuning:
        raise ValueError("Both tables must contain unit_index.")
    if base["unit_index"].duplicated().any():
        raise ValueError("unit_feature_table contains duplicate unit_index values.")
    if tuning["unit_index"].duplicated().any():
        raise ValueError("new_unit_tuning_csv contains duplicate unit_index values.")

    previous_columns: set[str] = set()
    if previous_unit_tuning_csv is not None:
        previous = pd.read_csv(previous_unit_tuning_csv, nrows=0)
        previous_columns.update(previous.columns)
    # New and previous tuning products use the same schema in production, but
    # take the union so a schema migration cannot leave stale columns behind.
    tuning_columns = (previous_columns | set(tuning.columns)) - {
        "unit_index",
        "unit_label",
    }
    stale_columns = sorted(tuning_columns & set(base.columns))
    clean = base.drop(columns=stale_columns)
    payload = tuning.drop(columns=["unit_label"], errors="ignore")
    merged = clean.merge(
        payload,
        on="unit_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged.shape[0] != base.shape[0]:
        raise RuntimeError("Replacing tuning metadata changed the unit row count.")
    missing = sorted(set(base["unit_index"]) - set(tuning["unit_index"]))
    if missing:
        raise ValueError(
            f"New tuning table is missing {len(missing)} matrix units; first={missing[:8]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    report = {
        "analysis": "fig4_replace_unit_feature_tuning",
        "input_unit_feature_table": str(unit_feature_table.resolve()),
        "input_unit_feature_table_sha256": input_unit_feature_sha256,
        "new_unit_tuning_csv": str(new_unit_tuning_csv.resolve()),
        "new_unit_tuning_csv_sha256": new_unit_tuning_sha256,
        "previous_unit_tuning_csv": (
            str(previous_unit_tuning_csv.resolve())
            if previous_unit_tuning_csv is not None
            else None
        ),
        "previous_unit_tuning_csv_sha256": previous_unit_tuning_sha256,
        "output_unit_feature_table": str(out_path.resolve()),
        "output_unit_feature_table_sha256": _sha256(out_path),
        "n_units": int(merged.shape[0]),
        "removed_stale_columns": stale_columns,
        "inserted_tuning_columns": [
            column for column in tuning.columns if column not in {"unit_index", "unit_label"}
        ],
    }
    report_path = out_path.with_name(f"{out_path.stem}_tuning_provenance.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-feature-table", type=Path, required=True)
    parser.add_argument("--new-unit-tuning-csv", type=Path, required=True)
    parser.add_argument("--previous-unit-tuning-csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Output CSV; defaults to replacing --unit-feature-table in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.unit_feature_table
    report = replace_tuning(
        unit_feature_table=args.unit_feature_table,
        new_unit_tuning_csv=args.new_unit_tuning_csv,
        previous_unit_tuning_csv=args.previous_unit_tuning_csv,
        out_path=out_path,
    )
    print(
        f"Updated {report['n_units']} units in {report['output_unit_feature_table']} "
        f"({len(report['inserted_tuning_columns'])} tuning columns)."
    )


if __name__ == "__main__":
    main()
