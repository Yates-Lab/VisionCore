#!/usr/bin/env python3
"""Merge image-sharded BackImage real-trace SSI matrix outputs.

Each shard contains the same selected-image, trace, unit, and trace-xy tables,
plus a slice of image-major movie rows. This script rebuilds the full
image-by-trace matrix in `movie_index` order and writes the merged bank consumed
by the Figure 4 refresh jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MATRIX_FILES = (
    "ssi_matrix.npy",
    "expected_spikes_matrix.npy",
    "mean_rate_matrix.npy",
    "population_ssi.npy",
)

PROVENANCE_PATHS = (
    ("rr100_version",),
    ("bin_seconds",),
    ("n_timepoints",),
    ("patch_size_px",),
    ("source_csv",),
    ("unit_tuning_csv",),
    ("trace_time_contract",),
    ("model_provenance", "model", "checkpoint_sha256"),
    ("model_provenance", "model", "dataset_configs_sha256"),
    ("model_provenance", "rr100_population_spec_json_sha256"),
    ("model_provenance", "rr100_population_spec_npz_sha256"),
    ("model_provenance", "stimulus", "model_history_frames"),
    ("model_provenance", "stimulus", "model_input_rate_hz"),
    ("model_provenance", "stimulus", "model_output_rate_hz"),
    ("model_provenance", "stimulus", "supervision_phase"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Shard summary lacks required provenance {'.'.join(path)}")
        value = value[key]
    return value


def assert_matching_provenance(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    first = summaries[0]
    for path in PROVENANCE_PATHS:
        expected = nested_value(first, path)
        for shard_index, summary in enumerate(summaries[1:], start=1):
            observed = nested_value(summary, path)
            if observed != expected:
                raise ValueError(
                    f"Shard {shard_index} provenance differs at {'.'.join(path)}: "
                    f"{observed!r} versus {expected!r}."
                )
        reference[".".join(path)] = expected
    return reference


def require_member(shard_dir: Path, name: str) -> Path:
    path = shard_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing shard member: {path}")
    return path


def assert_same_table(first: pd.DataFrame, other: pd.DataFrame, *, name: str) -> None:
    left = first.reset_index(drop=True)
    right = other.reset_index(drop=True)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        raise ValueError(f"{name} differs across shards: shape/columns mismatch.")
    if not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
        raise ValueError(f"{name} differs across shards.")


def load_reference_tables(shard_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    first = shard_dirs[0]
    image_table = pd.read_csv(require_member(first, "image_feature_table.csv"))
    trace_table = pd.read_csv(require_member(first, "trace_feature_table.csv"))
    unit_table = pd.read_csv(require_member(first, "unit_feature_table.csv"))
    trace_xy = np.load(require_member(first, "trace_xy.npy"))

    for shard_dir in shard_dirs[1:]:
        assert_same_table(
            image_table,
            pd.read_csv(require_member(shard_dir, "image_feature_table.csv")),
            name="image_feature_table.csv",
        )
        assert_same_table(
            trace_table,
            pd.read_csv(require_member(shard_dir, "trace_feature_table.csv")),
            name="trace_feature_table.csv",
        )
        assert_same_table(
            unit_table,
            pd.read_csv(require_member(shard_dir, "unit_feature_table.csv")),
            name="unit_feature_table.csv",
        )
        other_xy = np.load(require_member(shard_dir, "trace_xy.npy"))
        if trace_xy.shape != other_xy.shape or not np.array_equal(trace_xy, other_xy):
            raise ValueError("trace_xy.npy differs across shards.")

    return image_table, trace_table, unit_table, trace_xy


def allocate_arrays(shard_dir: Path, *, n_movies: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    filled: dict[str, np.ndarray] = {}
    for name in MATRIX_FILES:
        sample = np.load(require_member(shard_dir, name))
        shape = (int(n_movies), *sample.shape[1:])
        arrays[name] = np.empty(shape, dtype=sample.dtype)
        filled[name] = np.zeros((int(n_movies),), dtype=bool)
    return arrays, filled


def copy_shard_arrays(
    *,
    shard_dir: Path,
    movies: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    filled: dict[str, np.ndarray],
    n_movies: int,
) -> None:
    movie_index = movies["movie_index"].astype(int).to_numpy()
    if movie_index.min(initial=0) < 0 or movie_index.max(initial=-1) >= int(n_movies):
        raise ValueError(f"{shard_dir} has movie_index outside expected 0-{int(n_movies) - 1}.")
    if movie_index.size != np.unique(movie_index).size:
        raise ValueError(f"{shard_dir} has duplicate movie_index values.")
    matrix_row = (
        movies["matrix_row_index"].astype(int).to_numpy()
        if "matrix_row_index" in movies.columns
        else np.arange(movies.shape[0], dtype=int)
    )

    for name in MATRIX_FILES:
        values = np.load(require_member(shard_dir, name))
        if values.shape[0] != movies.shape[0]:
            raise ValueError(f"{name} rows do not match movie table in {shard_dir}.")
        if matrix_row.min(initial=0) < 0 or matrix_row.max(initial=-1) >= values.shape[0]:
            raise ValueError(f"{shard_dir} has matrix_row_index outside {name} rows.")
        if filled[name][movie_index].any():
            duplicated = movie_index[filled[name][movie_index]][0]
            raise ValueError(f"{name} would overwrite duplicate movie_index {int(duplicated)}.")
        arrays[name][movie_index] = values[matrix_row]
        filled[name][movie_index] = True


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_dir} already exists and is not empty. Pass --force.")
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path(path) for path in args.shard_dirs]
    if not shard_dirs:
        raise ValueError("At least one shard directory is required.")
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    summaries = [load_json(require_member(path, "summary.json")) for path in shard_dirs]
    common_provenance = assert_matching_provenance(summaries)
    image_table, trace_table, unit_table, trace_xy = load_reference_tables(shard_dirs)
    n_images = int(image_table.shape[0])
    n_traces = int(trace_table.shape[0])
    n_units = int(unit_table.shape[0])
    n_movies = n_images * n_traces

    movie_parts: list[pd.DataFrame] = []
    arrays, filled = allocate_arrays(shard_dirs[0], n_movies=n_movies)
    for shard_dir in shard_dirs:
        movies = pd.read_csv(require_member(shard_dir, "movie_feature_table.csv"))
        copy_shard_arrays(
            shard_dir=shard_dir,
            movies=movies,
            arrays=arrays,
            filled=filled,
            n_movies=n_movies,
        )
        movie_parts.append(movies)

    merged_movie = (
        pd.concat(movie_parts, ignore_index=True)
        .sort_values("movie_index", kind="mergesort")
        .reset_index(drop=True)
    )
    if merged_movie.shape[0] != n_movies:
        raise ValueError(f"Merged movie table has {merged_movie.shape[0]} rows; expected {n_movies}.")
    if merged_movie["movie_index"].astype(int).nunique() != n_movies:
        raise ValueError("Merged movie table does not cover every movie_index exactly once.")
    movie_index = merged_movie["movie_index"].astype(int).to_numpy()
    if not np.array_equal(movie_index, np.arange(n_movies, dtype=int)):
        raise ValueError("Merged movie rows are not in exact contiguous movie_index order.")
    expected_image = movie_index // n_traces
    expected_trace = movie_index % n_traces
    if not np.array_equal(merged_movie["image_index"].astype(int), expected_image):
        raise ValueError("Merged image_index is inconsistent with image-major movie_index.")
    if not np.array_equal(merged_movie["trace_index"].astype(int), expected_trace):
        raise ValueError("Merged trace_index is inconsistent with image-major movie_index.")

    for name, values in arrays.items():
        if not filled[name].all():
            missing = np.flatnonzero(~filled[name])[:5].astype(int).tolist()
            raise ValueError(f"{name} has unfilled movie rows after merge, starting with {missing}.")
        expected_shape = (n_movies,) if name == "population_ssi.npy" else (n_movies, n_units)
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
        if not np.all(np.isfinite(values)):
            count = int(values.size - np.count_nonzero(np.isfinite(values)))
            raise ValueError(f"{name} contains {count} non-finite entries.")
        if np.any(values < 0):
            raise ValueError(f"{name} contains negative entries (minimum {values.min():g}).")
        np.save(out_dir / name, values)

    image_table.to_csv(out_dir / "image_feature_table.csv", index=False)
    trace_table.to_csv(out_dir / "trace_feature_table.csv", index=False)
    unit_table.to_csv(out_dir / "unit_feature_table.csv", index=False)
    merged_movie.to_csv(out_dir / "movie_feature_table.csv", index=False)
    np.save(out_dir / "trace_xy.npy", trace_xy)
    if (shard_dirs[0] / "trace_bank_metric_summary.csv").exists():
        pd.read_csv(shard_dirs[0] / "trace_bank_metric_summary.csv").to_csv(
            out_dir / "trace_bank_metric_summary.csv",
            index=False,
        )

    summary = {
        "analysis": "backimage_real_trace_ssi_matrix_merged_shards",
        "shard_dirs": shard_dirs,
        "n_shards": len(shard_dirs),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_units": n_units,
        "n_movies": n_movies,
        "validated_common_provenance": common_provenance,
        "integrity_checks": {
            "complete_image_major_movie_index": True,
            "matrix_shapes_match_coordinates": True,
            "all_matrix_entries_finite": True,
            "all_matrix_entries_nonnegative": True,
        },
        "shard_summaries": summaries,
        "outputs": {name.removesuffix(".npy"): out_dir / name for name in MATRIX_FILES}
        | {
            "movie_feature_table": out_dir / "movie_feature_table.csv",
            "image_feature_table": out_dir / "image_feature_table.csv",
            "trace_feature_table": out_dir / "trace_feature_table.csv",
            "unit_feature_table": out_dir / "unit_feature_table.csv",
            "trace_xy": out_dir / "trace_xy.npy",
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
