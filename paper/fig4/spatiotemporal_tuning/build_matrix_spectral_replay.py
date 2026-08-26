#!/usr/bin/env python3
"""Attach exact rendered-movie spectra to an audited response matrix.

The large Figure-4 matrix already contains the model's responses to a
complete natural-image x filtered-fixation factorial design.  Replaying those
same movies through the model a second time is both wasteful and a potential
source of drift.  This command therefore computes only the renderer-faithful
SF x TF x orientation predictors, then joins them to the cached moving and
stabilized responses after strict image, trace, unit, filter, and checkpoint
checks.

The output uses the same compact archive contract as ``run_retinal_causal_chain``
so downstream mechanism plots can consume either source without special cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import (  # noqa: E402
    render_movies,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (  # noqa: E402
    frequency_grid,
)
from paper.fig4.spatiotemporal_tuning.run_retinal_causal_chain import (  # noqa: E402
    interpolate_tuning_temporal,
    load_signed_projection_controls,
    load_tuning_tensors,
    mode_to_grid_matrix,
    movie_power_cube,
    spectral_predictors,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch  # noqa: E402


EPS = 1e-12
PREDICTOR_NAMES = (
    "total_dynamic_power",
    "joint_signed_rate_drive",
    "joint_passband_power",
    "tf_marginal_power",
    "sf_orientation_marginal_power",
    "separable_passband_power",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-trace-batch-size", type=int, default=4)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument(
        "--image-stop",
        type=int,
        default=0,
        help="Exclusive matrix-image row; zero uses every remaining image.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_shards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    shards = summary.get("shard_summaries", [])
    if isinstance(shards, list) and shards:
        return [value for value in shards if isinstance(value, dict)]
    if isinstance(summary.get("model_provenance"), dict):
        return [summary]
    raise ValueError("matrix summary contains no shard provenance")


def _unique_nested(shards: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    values: list[Any] = []
    for shard in shards:
        value: Any = shard
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if value not in (None, "", {}):
            values.append(value)
    encoded = {json.dumps(value, sort_keys=True) for value in values}
    if len(encoded) != 1:
        raise ValueError(f"matrix shards disagree on {'.'.join(keys)}")
    return values[0]


def _load_matrix_responses(
    matrix_dir: Path,
    *,
    n_images: int,
    n_traces: int,
    n_units: int,
) -> dict[str, np.ndarray]:
    moving_files = {
        "mean_rate": "mean_rate_matrix.npy",
        "expected_spikes": "expected_spikes_matrix.npy",
        "map_ssi": "ssi_matrix.npy",
    }
    stable_files = {
        "mean_rate": "stabilized_mean_rate_by_image.npy",
        "expected_spikes": "stabilized_expected_spikes_by_image.npy",
        "map_ssi": "stabilized_ssi_by_image.npy",
    }
    output: dict[str, np.ndarray] = {}
    for key, filename in moving_files.items():
        value = np.load(matrix_dir / filename, mmap_mode="r")
        expected = (n_images * n_traces, n_units)
        if value.shape != expected:
            raise ValueError(f"{filename} has shape {value.shape}; expected {expected}")
        output[f"moving_{key}"] = np.asarray(value).reshape(n_images, n_traces, n_units)
    for key, filename in stable_files.items():
        value = np.load(matrix_dir / filename)
        expected = (n_images, n_units)
        if value.shape != expected:
            raise ValueError(f"{filename} has shape {value.shape}; expected {expected}")
        output[f"stable_{key}"] = np.asarray(value)
    return output


def _matrix_contract(matrix_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = matrix_dir / "summary.json"
    baseline_path = matrix_dir / "stabilized_baseline_summary.json"
    if not summary_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(
            "matrix spectral replay requires merged summary.json and a completed "
            "stabilized_baseline_summary.json"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    shards = _summary_shards(summary)
    checkpoint = str(
        _unique_nested(shards, ("model_provenance", "model", "checkpoint_sha256"))
    )
    baseline_checkpoint = str(
        baseline.get("model_provenance", {}).get("model", {}).get(
            "checkpoint_sha256", ""
        )
    )
    if checkpoint != baseline_checkpoint:
        raise ValueError("moving matrix and stabilized baseline use different checkpoints")
    trace_provenance = _unique_nested(shards, ("trace_bank", "trace_provenance"))
    trace_filter = trace_provenance.get("filter", {})
    if "zero-phase" not in str(trace_filter.get("kind", "")).lower():
        raise ValueError("matrix traces were not continuously zero-phase filtered")
    if not bool(
        trace_provenance.get("spectral_filter_qc", {}).get(
            "stopband_suppression_gate", False
        )
    ):
        raise ValueError("matrix trace-filter suppression gate did not pass")
    return summary, {
        "checkpoint_sha256": checkpoint,
        "dataset_configs_sha256": str(
            _unique_nested(
                shards,
                ("model_provenance", "model", "dataset_configs_sha256"),
            )
        ),
        "trace_provenance": trace_provenance,
        "model_provenance": shards[0].get("model_provenance", {}),
        "n_timepoints": int(shards[0]["n_timepoints"]),
        "bin_seconds": float(shards[0]["bin_seconds"]),
    }


def main() -> int:
    args = parse_args()
    matrix_dir = args.matrix_dir.resolve()
    out_dir = args.out_dir.resolve()
    archive_path = out_dir / "causal_chain_shard.npz"
    summary_out = out_dir / "summary.json"
    if (archive_path.exists() or summary_out.exists()) and not args.force:
        raise FileExistsError(f"outputs already exist in {out_dir}; pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_summary, provenance = _matrix_contract(matrix_dir)
    # Matrix rows are image-major and use these tables in their stored row
    # order.  Never silently sort a table away from its response arrays.
    scored_image_table = matrix_dir / "scored_image_feature_table.csv"
    image_table_path = (
        scored_image_table
        if scored_image_table.exists()
        else matrix_dir / "image_feature_table.csv"
    )
    all_images = pd.read_csv(image_table_path).reset_index(drop=True)
    trace_table = pd.read_csv(matrix_dir / "trace_feature_table.csv").reset_index(drop=True)
    unit_table = pd.read_csv(matrix_dir / "unit_feature_table.csv").reset_index(drop=True)
    traces = np.load(matrix_dir / "trace_xy.npy")
    n_total_images = len(all_images)
    n_traces = len(trace_table)
    n_matrix_units = len(unit_table)
    trace_index_column = (
        "trace_index" if "trace_index" in trace_table else "trace_bank_index"
    )
    if trace_index_column not in trace_table:
        raise ValueError("trace_feature_table lacks a matrix-row index column")
    if not np.array_equal(
        trace_table[trace_index_column].to_numpy(dtype=int), np.arange(n_traces)
    ):
        raise ValueError("trace_feature_table row order is not the matrix trace order")
    if not np.array_equal(
        unit_table.unit_index.to_numpy(dtype=int), np.arange(n_matrix_units)
    ):
        raise ValueError("unit_feature_table row order is not the response-unit order")
    if traces.shape != (n_traces, int(provenance["n_timepoints"]), 2):
        raise ValueError(
            f"trace_xy has shape {traces.shape}; expected "
            f"{(n_traces, int(provenance['n_timepoints']), 2)}"
        )
    declared_n_images = matrix_summary.get("n_images")
    if declared_n_images is None:
        declared_n_images = matrix_summary.get("image_shard", {}).get(
            "n_scored_images"
        )
    if declared_n_images is None:
        raise ValueError("matrix summary does not declare its scored image count")
    if int(declared_n_images) != n_total_images:
        raise ValueError("matrix summary and image table disagree")
    image_start = max(0, int(args.image_start))
    image_stop = (
        n_total_images
        if int(args.image_stop) <= 0
        else min(n_total_images, int(args.image_stop))
    )
    if image_start >= image_stop:
        raise ValueError("requested image shard is empty")
    image_rows = np.arange(image_start, image_stop, dtype=int)
    images = all_images.iloc[image_rows].reset_index(drop=True)
    n_images = len(images)

    tuning = load_tuning_tensors(args.tuning_table, output_rate_hz=240.0)
    unit_indices = np.asarray(tuning["unit_indices"], dtype=int)
    if len(unit_indices) == 0 or unit_indices.min() < 0 or unit_indices.max() >= n_matrix_units:
        raise ValueError("periodic-tuning unit indices do not fit the response matrix")
    responses = _load_matrix_responses(
        matrix_dir,
        n_images=n_total_images,
        n_traces=n_traces,
        n_units=n_matrix_units,
    )
    responses = {
        key: value[image_rows]
        for key, value in responses.items()
    }
    frame_rate_hz = 1.0 / float(provenance["bin_seconds"])
    if not np.isclose(frame_rate_hz, 240.0):
        raise ValueError(f"matrix is not native 240 Hz: {frame_rate_hz:g} Hz")
    n_units = len(unit_indices)
    shape = (n_images, n_traces, 2, n_units)
    mean_rate = np.empty(shape, dtype=np.float32)
    expected_spikes = np.empty(shape, dtype=np.float32)
    map_ssi = np.empty(shape, dtype=np.float32)
    for key, destination in (
        ("mean_rate", mean_rate),
        ("expected_spikes", expected_spikes),
        ("map_ssi", map_ssi),
    ):
        stable = responses[f"stable_{key}"][:, unit_indices]
        moving = responses[f"moving_{key}"][:, :, unit_indices]
        if key == "mean_rate":
            # The matrix stores expected spikes per native output bin.  The
            # causal-chain archive contract expresses mean_rate in spikes/s.
            stable = frame_rate_hz * stable
            moving = frame_rate_hz * moving
        destination[:, :, 0] = stable[:, None, :]
        destination[:, :, 1] = moving

    predictors = {
        name: np.zeros(shape, dtype=np.float32) for name in PREDICTOR_NAMES
    }
    grid = frequency_grid()
    distributor, resolved_modes = mode_to_grid_matrix(
        np.asarray(grid["kxy"]),
        np.asarray(tuning["spatial_cpd"]),
        np.asarray(tuning["orientation_deg"]),
    )
    temporal_hz: np.ndarray | None = None
    signed: np.ndarray | None = None
    passband: np.ndarray | None = None
    signed_controls: dict[str, np.ndarray] | None = None
    average_power: np.ndarray | None = None
    canvas_cache: dict[Any, Any] = {}
    batch_size = max(1, int(args.render_trace_batch_size))
    for image_local, image_row in images.iterrows():
        patch, _ = extract_patch(
            image_row,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        for start in range(0, n_traces, batch_size):
            stop = min(start + batch_size, n_traces)
            movies = render_movies(
                patch,
                np.asarray(traces[start:stop], dtype=np.float32),
                device=str(args.device),
            )
            for offset, movie in enumerate(movies):
                frequency, cube = movie_power_cube(
                    movie,
                    flat_index=np.asarray(grid["flat_index"], dtype=int),
                    mode_to_grid=distributor,
                    n_spatial=len(tuning["spatial_cpd"]),
                    n_orientation=len(tuning["orientation_deg"]),
                    frame_rate_hz=frame_rate_hz,
                )
                if temporal_hz is None:
                    temporal_hz = frequency
                    signed = interpolate_tuning_temporal(
                        tuning["signed_rate_sensitivity"],
                        tuning["temporal_hz"],
                        temporal_hz,
                        normalize=False,
                    )
                    passband = interpolate_tuning_temporal(
                        tuning["phase_rms"],
                        tuning["temporal_hz"],
                        temporal_hz,
                        normalize=True,
                    )
                    signed_controls = load_signed_projection_controls(
                        args.tuning_table.parent
                        / "population_signed_projection_controls_240hz_matrix60.npz",
                        signed,
                        temporal_hz,
                        source=args.tuning_table,
                    )
                    average_power = np.zeros((2, *cube.shape), dtype=np.float64)
                elif not np.array_equal(temporal_hz, frequency):
                    raise RuntimeError("temporal spectrum grid changed across movies")
                projection = spectral_predictors(
                    cube,
                    signed,
                    passband,
                    signed_controls=signed_controls,
                )
                trace_local = start + offset
                for name, value in projection.items():
                    predictors[name][image_local, trace_local, 1] = value
                average_power[1] += cube
        print(
            f"matrix spectral replay image {image_local + 1}/{n_images}",
            flush=True,
        )

    if temporal_hz is None or average_power is None:
        raise RuntimeError("no rendered spectra were computed")
    average_power /= float(n_images * n_traces)
    trace_indices = trace_table[trace_index_column].to_numpy(dtype=int)
    image_indices = images.image_index.to_numpy(dtype=int)
    if len(np.unique(trace_indices)) != n_traces or len(np.unique(image_indices)) != n_images:
        raise ValueError("image and trace indices must be unique")
    empty = np.empty((0,), dtype=np.float32)
    np.savez_compressed(
        archive_path,
        image_indices=image_indices,
        trace_indices=trace_indices,
        motion_scales=np.asarray((0.0, 1.0), dtype=np.float32),
        unit_indices=unit_indices,
        spatial_cpd=np.asarray(tuning["spatial_cpd"], dtype=np.float32),
        temporal_hz=np.asarray(temporal_hz, dtype=np.float32),
        orientation_deg=np.asarray(tuning["orientation_deg"], dtype=np.float32),
        mean_rate=mean_rate,
        expected_spikes=expected_spikes,
        map_ssi=map_ssi,
        average_power=average_power.astype(np.float32),
        example_power=empty,
        example_rate_maps=empty,
        example_movie_frames=empty,
        example_trace_xy=empty,
        **predictors,
    )
    report = {
        "analysis": "exact rendered spectral replay joined to completed response matrix",
        "contract": (
            "Every spectral predictor is computed from the exact natural image and "
            "filtered 250-ms trace whose selected-twin response is stored in the source "
            "factorial matrix; stabilized responses are scored once per image and broadcast."
        ),
        "model_label": str(args.model_label),
        "checkpoint_sha256": str(provenance["checkpoint_sha256"]),
        "dataset_configs_sha256": str(provenance["dataset_configs_sha256"]),
        "model_provenance": provenance["model_provenance"],
        "source_matrix": str(matrix_dir),
        "source_matrix_summary_sha256": sha256(matrix_dir / "summary.json"),
        "source_tuning_table": str(args.tuning_table.resolve()),
        "source_tuning_table_sha256": sha256(args.tuning_table),
        "source_image_table": str(image_table_path),
        "source_baseline_summary_sha256": sha256(
            matrix_dir / "stabilized_baseline_summary.json"
        ),
        "response_reuse": (
            "moving and stabilized responses are read from the audited matrix; the twin "
            "is not evaluated a second time"
        ),
        "mean_rate_units": (
            "spikes/s; source expected-spikes-per-bin values multiplied by frame_rate_hz"
        ),
        "spectrum": (
            "exact 151-pixel renderer; spatial Tukey; temporal mean removed; "
            "DPSS NW=1.5 K=2; full SFxTFxorientation projection"
        ),
        "trace_kind": "filtered",
        "trace_transform": "identity",
        "behavior": "fixed all-zero vector in the source response matrix",
        "trace_filter": provenance["trace_provenance"].get("filter", {}),
        "trace_filter_qc": provenance["trace_provenance"].get(
            "spectral_filter_qc", {}
        ),
        "response_independent_selection": {
            "image": provenance["trace_provenance"].get("image_selection", {}),
            "trace": provenance["trace_provenance"].get("trace_selection", {}),
        },
        "motion_scales": [0.0, 1.0],
        "n_images": int(n_images),
        "n_total_matrix_images": int(n_total_images),
        "n_traces": int(n_traces),
        "trace_index_column": str(trace_index_column),
        "n_units": int(n_units),
        "n_timepoints": int(provenance["n_timepoints"]),
        "frame_rate_hz": float(frame_rate_hz),
        "n_resolved_fourier_modes": int(np.count_nonzero(resolved_modes)),
        "image_shard": {
            "start": int(image_start),
            "stop": int(image_stop),
            "matrix_rows": image_rows.tolist(),
            "image_indices": image_indices.tolist(),
        },
        "archive": str(archive_path),
    }
    summary_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
