#!/usr/bin/env python3
"""Compare a replayed Figure 4 real-trace matrix against a reference cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MOVIE_ARRAYS = (
    "ssi_matrix.npy",
    "expected_spikes_matrix.npy",
    "mean_rate_matrix.npy",
    "population_ssi.npy",
)
KEYED_TABLES = (
    ("image_feature_table.csv", "image_index"),
    ("trace_feature_table.csv", "trace_bank_index"),
    ("movie_feature_table.csv", "movie_index"),
)
STABILIZED_ARRAYS = (
    "stabilized_ssi_by_image.npy",
    "stabilized_expected_spikes_by_image.npy",
    "stabilized_mean_rate_by_image.npy",
    "stabilized_population_ssi_by_image.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_keyed_indices(table: pd.DataFrame, key: str, *, table_name: str) -> pd.Series:
    if key not in table.columns:
        raise ValueError(f"{table_name} must contain {key!r}.")
    keys = pd.to_numeric(table[key], errors="raise").astype(int)
    if keys.duplicated().any():
        duplicates = keys[keys.duplicated()].head(8).astype(str).tolist()
        raise ValueError(f"{table_name} has duplicate {key} values: {', '.join(duplicates)}")
    return pd.Series(np.arange(table.shape[0], dtype=int), index=keys.to_numpy())


def comparison_stats(candidate: np.ndarray, expected: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    if candidate.shape != expected.shape:
        return {
            "status": "fail",
            "reason": f"shape mismatch: {candidate.shape} != {expected.shape}",
            "candidate_shape": tuple(candidate.shape),
            "reference_shape": tuple(expected.shape),
        }
    candidate64 = np.asarray(candidate, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    diff = np.abs(candidate64 - expected64)
    finite = np.isfinite(diff)
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(expected64), np.finfo(np.float64).tiny)
    rel = np.where(finite, diff / denom, np.nan)
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    passed = bool(np.allclose(candidate64, expected64, atol=float(atol), rtol=float(rtol), equal_nan=True))
    return {
        "status": "pass" if passed else "fail",
        "candidate_shape": tuple(candidate.shape),
        "reference_shape": tuple(expected.shape),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def compare_row_array(
    *,
    name: str,
    candidate_dir: Path,
    reference_dir: Path,
    candidate_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    key: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    candidate_path = candidate_dir / name
    reference_path = reference_dir / name
    if not candidate_path.exists():
        return {"name": name, "status": "missing-candidate", "path": candidate_path}
    if not reference_path.exists():
        return {"name": name, "status": "missing-reference", "path": reference_path}

    candidate = np.load(candidate_path)
    reference = np.load(reference_path, mmap_mode="r")
    if candidate.shape[0] != candidate_table.shape[0]:
        return {
            "name": name,
            "status": "fail",
            "reason": f"candidate row count {candidate.shape[0]} != {candidate_table.shape[0]} {key} rows",
        }
    if reference.shape[0] != reference_table.shape[0]:
        return {
            "name": name,
            "status": "fail",
            "reason": f"reference row count {reference.shape[0]} != {reference_table.shape[0]} {key} rows",
        }

    reference_lookup = load_keyed_indices(reference_table, key, table_name=f"{reference_dir.name}/{key} table")
    candidate_keys = pd.to_numeric(candidate_table[key], errors="raise").astype(int).to_numpy()
    missing = [int(value) for value in candidate_keys if int(value) not in reference_lookup.index]
    if missing:
        return {
            "name": name,
            "status": "fail",
            "reason": f"candidate {key} values missing from reference",
            "missing_keys": missing[:16],
        }
    reference_indices = reference_lookup.loc[candidate_keys].to_numpy(dtype=int)
    expected = np.asarray(reference[reference_indices])
    return {"name": name, **comparison_stats(candidate, expected, atol=atol, rtol=rtol)}


def values_match(candidate: pd.Series, expected: pd.Series, *, atol: float, rtol: float) -> tuple[bool, float | None]:
    if pd.api.types.is_numeric_dtype(candidate) and pd.api.types.is_numeric_dtype(expected):
        candidate_values = pd.to_numeric(candidate, errors="coerce").to_numpy(dtype=np.float64)
        expected_values = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=np.float64)
        diff = np.abs(candidate_values - expected_values)
        finite = np.isfinite(diff)
        max_abs = float(np.nanmax(diff[finite])) if np.any(finite) else 0.0
        return (
            bool(np.allclose(candidate_values, expected_values, atol=float(atol), rtol=float(rtol), equal_nan=True)),
            max_abs,
        )
    return bool(candidate.astype(str).equals(expected.astype(str))), None


def compare_keyed_table(
    *,
    name: str,
    key: str,
    candidate_dir: Path,
    reference_dir: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    candidate_path = candidate_dir / name
    reference_path = reference_dir / name
    if not candidate_path.exists():
        return {"name": name, "status": "missing-candidate", "path": candidate_path}
    if not reference_path.exists():
        return {"name": name, "status": "missing-reference", "path": reference_path}
    candidate = pd.read_csv(candidate_path)
    reference = pd.read_csv(reference_path)
    if candidate.columns.tolist() != reference.columns.tolist():
        return {
            "name": name,
            "status": "fail",
            "reason": "column mismatch",
            "candidate_shape": tuple(candidate.shape),
            "reference_shape": tuple(reference.shape),
            "candidate_only_columns": sorted(set(candidate.columns) - set(reference.columns)),
            "reference_only_columns": sorted(set(reference.columns) - set(candidate.columns)),
        }
    if key not in candidate.columns or key not in reference.columns:
        return {"name": name, "status": "fail", "reason": f"missing key column {key!r}"}

    reference_lookup = load_keyed_indices(reference, key, table_name=f"{reference_dir.name}/{name}")
    candidate_keys = pd.to_numeric(candidate[key], errors="raise").astype(int).to_numpy()
    missing = [int(value) for value in candidate_keys if int(value) not in reference_lookup.index]
    if missing:
        return {
            "name": name,
            "status": "fail",
            "reason": f"candidate {key} values missing from reference",
            "missing_keys": missing[:16],
        }
    expected = reference.iloc[reference_lookup.loc[candidate_keys].to_numpy(dtype=int)].reset_index(drop=True)
    candidate = candidate.reset_index(drop=True)
    mismatches: list[dict[str, Any]] = []
    for column in candidate.columns:
        matched, max_abs = values_match(candidate[column], expected[column], atol=atol, rtol=rtol)
        if not matched:
            row: dict[str, Any] = {"column": column}
            if max_abs is not None:
                row["max_abs_diff"] = max_abs
            mismatches.append(row)
    return {
        "name": name,
        "status": "pass" if not mismatches else "fail",
        "candidate_shape": tuple(candidate.shape),
        "reference_shape": tuple(reference.shape),
        "key": key,
        "mismatched_columns": mismatches[:16],
        "n_mismatched_columns": len(mismatches),
    }


def compare_units(candidate_dir: Path, reference_dir: Path) -> dict[str, Any]:
    candidate_path = candidate_dir / "unit_feature_table.csv"
    reference_path = reference_dir / "unit_feature_table.csv"
    if not candidate_path.exists() or not reference_path.exists():
        return {"name": "unit_feature_table.csv", "status": "skipped"}
    candidate = pd.read_csv(candidate_path)
    reference = pd.read_csv(reference_path)
    status = "pass" if candidate.shape[0] == reference.shape[0] else "fail"
    out: dict[str, Any] = {
        "name": "unit_feature_table.csv",
        "status": status,
        "candidate_rows": int(candidate.shape[0]),
        "reference_rows": int(reference.shape[0]),
    }
    if "unit_label" in candidate.columns and "unit_label" in reference.columns and candidate.shape[0] == reference.shape[0]:
        out["unit_labels_match"] = bool(candidate["unit_label"].astype(str).equals(reference["unit_label"].astype(str)))
        if not out["unit_labels_match"]:
            out["status"] = "fail"
    return out


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    reference_dir = Path(args.reference_dir)
    candidate_movies = pd.read_csv(require(candidate_dir / "movie_feature_table.csv"))
    reference_movies = pd.read_csv(require(reference_dir / "movie_feature_table.csv"))
    if "movie_index" not in candidate_movies.columns:
        raise ValueError(f"{candidate_dir / 'movie_feature_table.csv'} must contain movie_index.")
    if "movie_index" not in reference_movies.columns:
        raise ValueError(f"{reference_dir / 'movie_feature_table.csv'} must contain movie_index.")

    comparisons: list[dict[str, Any]] = []
    for name in MOVIE_ARRAYS:
        comparisons.append(
            compare_row_array(
                name=name,
                candidate_dir=candidate_dir,
                reference_dir=reference_dir,
                candidate_table=candidate_movies,
                reference_table=reference_movies,
                key="movie_index",
                atol=float(args.atol),
                rtol=float(args.rtol),
            )
        )
    for name, key in KEYED_TABLES:
        comparisons.append(
            compare_keyed_table(
                name=name,
                key=key,
                candidate_dir=candidate_dir,
                reference_dir=reference_dir,
                atol=float(args.atol),
                rtol=float(args.rtol),
            )
        )

    candidate_stabilized = candidate_dir / "stabilized_movie_feature_table.csv"
    reference_stabilized = reference_dir / "stabilized_movie_feature_table.csv"
    if candidate_stabilized.exists() and reference_stabilized.exists():
        candidate_images = pd.read_csv(candidate_stabilized)
        reference_images = pd.read_csv(reference_stabilized)
        for name in STABILIZED_ARRAYS:
            comparisons.append(
                compare_row_array(
                    name=name,
                    candidate_dir=candidate_dir,
                    reference_dir=reference_dir,
                    candidate_table=candidate_images,
                    reference_table=reference_images,
                    key="image_index",
                    atol=float(args.atol),
                    rtol=float(args.rtol),
                )
            )
    comparisons.append(compare_units(candidate_dir, reference_dir))

    failing_statuses = {"fail", "missing-candidate", "missing-reference"}
    passed = not any(row.get("status") in failing_statuses for row in comparisons)
    report = {
        "analysis": "fig4_real_trace_matrix_output_comparison",
        "candidate_dir": candidate_dir,
        "reference_dir": reference_dir,
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "candidate_movies": int(candidate_movies.shape[0]),
        "reference_movies": int(reference_movies.shape[0]),
        "passed": passed,
        "comparisons": comparisons,
    }
    if args.report_json is not None:
        write_json(Path(args.report_json), report)

    print("FIGURE 4 REAL-TRACE MATRIX COMPARISON")
    print(f"candidate: {candidate_dir}")
    print(f"reference: {reference_dir}")
    for row in comparisons:
        label = row["name"]
        status = row["status"]
        detail = ""
        if "max_abs_diff" in row:
            detail = f" max_abs={row['max_abs_diff']:.6g} max_rel={row['max_rel_diff']:.6g}"
        elif "reason" in row:
            detail = f" {row['reason']}"
        elif row.get("n_mismatched_columns"):
            detail = f" mismatched_columns={row['n_mismatched_columns']}"
        print(f"  [{status}] {label}{detail}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
