#!/usr/bin/env python3
"""Replay centered drift-only Figure 4 trajectories at fixed amplitude scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_OUT_DIR = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling"
SCALE_FACTORS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)


def matrix_trace_time_contract(
    matrix_dir: Path,
    trace_xy: np.ndarray,
    *,
    model_output_rate_hz: int,
) -> dict[str, Any]:
    """Recover and validate the retained-trace timing used by the matrix.

    ``trace_xy.npy`` stays on the acquired eye-trace grid (120 Hz for the
    BackImage bank), even when a native-240 twin was scored. The matrix scorer
    interpolates those samples onto the model output grid internally. A
    controlled replay must pass the same source-bin duration or it silently
    compresses the physical fixation interval by two.
    """
    summary_path = Path(matrix_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing matrix timing provenance: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = dict(summary.get("trace_time_contract") or {})
    if not contract:
        shard_contracts = [
            dict(item.get("trace_time_contract") or {})
            for item in (summary.get("shard_summaries") or [])
        ]
        shard_contracts = [item for item in shard_contracts if item]
        if shard_contracts:
            contract = shard_contracts[0]
            if any(item != contract for item in shard_contracts[1:]):
                raise ValueError(
                    f"Merged matrix shards disagree on trace timing in {summary_path}"
                )
    source_rate_hz = int(contract.get("source_trace_rate_hz", 0))
    source_samples = int(contract.get("source_trace_samples", 0))
    declared_output_rate_hz = int(contract.get("model_output_rate_hz", 0))
    if source_rate_hz < 1 or source_samples < 1 or declared_output_rate_hz < 1:
        raise ValueError(f"Incomplete matrix trace_time_contract in {summary_path}")
    if np.asarray(trace_xy).ndim != 3 or np.asarray(trace_xy).shape[1:] != (
        source_samples,
        2,
    ):
        raise ValueError(
            "trace_xy.npy disagrees with the matrix timing contract: "
            f"shape={np.asarray(trace_xy).shape}, expected [trace, {source_samples}, 2]."
        )
    if declared_output_rate_hz != int(model_output_rate_hz):
        raise ValueError(
            "Controlled replay model output rate differs from the matrix: "
            f"{model_output_rate_hz} versus {declared_output_rate_hz} Hz."
        )
    if declared_output_rate_hz < source_rate_hz or (
        declared_output_rate_hz % source_rate_hz
    ):
        raise ValueError(
            "Matrix trace rate must divide model output rate; got "
            f"{source_rate_hz} -> {declared_output_rate_hz} Hz."
        )
    factor = declared_output_rate_hz // source_rate_hz
    expected_scored = source_samples * factor
    declared_scored = int(contract.get("scored_trace_samples", expected_scored))
    if declared_scored != expected_scored:
        raise ValueError(
            "Matrix scored_trace_samples is inconsistent with its rates: "
            f"{declared_scored} versus {expected_scored}."
        )
    return {
        **contract,
        "source_trace_rate_hz": source_rate_hz,
        "source_trace_samples": source_samples,
        "model_output_rate_hz": declared_output_rate_hz,
        "scored_samples_per_source_trace_sample": factor,
        "scored_trace_samples": expected_scored,
        "analysis_interval_seconds": source_samples / float(source_rate_hz),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_quantile_indices(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    order = candidates[np.argsort(values[candidates], kind="mergesort")]
    chunks = np.array_split(order, int(count))
    return np.asarray([chunk[len(chunk) // 2] for chunk in chunks if len(chunk)], dtype=int)


def selections(matrix_dir: Path, n_images: int, n_traces: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    images = pd.read_csv(matrix_dir / "image_feature_table.csv")
    images["matrix_image_row"] = np.arange(len(images), dtype=int)
    images = images.sort_values("image_index").reset_index(drop=True)
    traces = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    traces["matrix_trace_row"] = np.arange(len(traces), dtype=int)
    traces = traces.sort_values("trace_bank_index").reset_index(drop=True)
    trace_xy = np.load(matrix_dir / "trace_xy.npy")
    image_candidates = np.flatnonzero(
        pd.to_numeric(images["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float) >= 0.2
    )
    if image_candidates.size < n_images:
        raise ValueError(f"Only {image_candidates.size} contour images are available")
    # Cover the contour-strength distribution rather than selecting only its easiest tail.
    image_indices = choose_quantile_indices(
        pd.to_numeric(images["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float),
        image_candidates,
        n_images,
    )
    drift_candidates = np.flatnonzero(
        pd.to_numeric(traces["rendered_n_microsaccade_events"], errors="coerce").fillna(0).to_numpy(dtype=float) == 0
    )
    trace_indices = choose_quantile_indices(
        pd.to_numeric(traces["rendered_path_length_arcmin"], errors="coerce").to_numpy(dtype=float),
        drift_candidates,
        n_traces,
    )
    selected_images = images.iloc[image_indices].copy().reset_index(drop=True)
    selected_traces = traces.iloc[trace_indices].copy().reset_index(drop=True)
    trace_rows = selected_traces["matrix_trace_row"].to_numpy(dtype=int)
    return (
        selected_images,
        selected_traces,
        np.asarray(trace_xy[trace_rows], dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--rr100-version", default=RR100_VERSION)
    parser.add_argument("--mcfarland-outputs", type=Path, default=ROOT / "scripts/mcfarland_outputs_mono.pkl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = out_dir / "controlled_scaling_response.npz"
    if output_npz.exists() and not bool(args.force):
        print(f"Using existing {output_npz}")
        return 0
    images, traces, trace_xy = selections(matrix_dir, int(args.n_images), int(args.n_traces))
    images.to_csv(out_dir / "selected_images.csv", index=False)
    traces.to_csv(out_dir / "selected_traces.csv", index=False)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=Path(args.checkpoint),
        dataset_configs=Path(args.dataset_configs),
        population_spec_dir=Path(args.population_spec_dir),
        rr100_version=str(args.rr100_version),
        device=str(args.device),
        strict=True,
        mcfarland_outputs=Path(args.mcfarland_outputs),
    )
    trace_time_contract = matrix_trace_time_contract(
        matrix_dir,
        trace_xy,
        model_output_rate_hz=int(scorer.output_rate_hz),
    )
    centered = trace_xy - np.mean(trace_xy, axis=1, keepdims=True)
    scaled_traces = [centered[trace_idx] * scale for trace_idx in range(len(traces)) for scale in SCALE_FACTORS]
    shape = (len(images), len(traces), len(SCALE_FACTORS), scorer.n_units)
    ssi = np.full(shape, np.nan, dtype=np.float32)
    expected = np.full(shape, np.nan, dtype=np.float32)
    mean_rate = np.full(shape, np.nan, dtype=np.float32)
    population_ssi = np.full(shape[:-1], np.nan, dtype=np.float32)
    n_timepoints = int(trace_time_contract["source_trace_samples"])
    source_trace_rate_hz = int(trace_time_contract["source_trace_rate_hz"])
    bin_seconds = 1.0 / float(source_trace_rate_hz)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    for image_ordinal, image_row in images.iterrows():
        patch, _patch_meta = extract_patch(image_row, canvas_cache=canvas_cache, patch_size_px=540)
        unit_ssi, unit_expected, unit_rate, pop_ssi = scorer.score_traces_for_patch(
            patch,
            scaled_traces,
            trace_batch_size=int(args.trace_batch_size),
            frame_batch_size=int(args.frame_batch_size),
            n_timepoints=n_timepoints,
            bin_seconds=bin_seconds,
        )
        ssi[image_ordinal] = unit_ssi.reshape(len(traces), len(SCALE_FACTORS), scorer.n_units)
        expected[image_ordinal] = unit_expected.reshape(len(traces), len(SCALE_FACTORS), scorer.n_units)
        mean_rate[image_ordinal] = unit_rate.reshape(len(traces), len(SCALE_FACTORS), scorer.n_units)
        population_ssi[image_ordinal] = pop_ssi.reshape(len(traces), len(SCALE_FACTORS))
        print(f"controlled scaling image {image_ordinal + 1}/{len(images)}", flush=True)
    np.savez_compressed(
        output_npz,
        ssi=ssi,
        expected_spikes=expected,
        mean_rate=mean_rate,
        population_ssi=population_ssi,
        scale_factors=SCALE_FACTORS,
        selected_image_index=images["image_index"].to_numpy(dtype=np.int32),
        selected_trace_index=traces["trace_bank_index"].to_numpy(dtype=np.int32),
        base_trace_xy=centered.astype(np.float32),
        base_trace_source_rate_hz=np.asarray(float(source_trace_rate_hz)),
        frame_rate_hz=np.asarray(float(scorer.output_rate_hz)),
        duration_s=np.asarray(float(trace_time_contract["analysis_interval_seconds"])),
    )
    image_rows = images["matrix_image_row"].to_numpy(dtype=int)
    trace_rows = traces["matrix_trace_row"].to_numpy(dtype=int)
    scale_one_index = int(np.flatnonzero(np.isclose(SCALE_FACTORS, 1.0))[0])
    matrix_images = len(pd.read_csv(matrix_dir / "image_feature_table.csv"))
    matrix_traces = len(pd.read_csv(matrix_dir / "trace_feature_table.csv"))
    cached_ssi = np.load(matrix_dir / "ssi_matrix.npy").reshape(
        matrix_images, matrix_traces, scorer.n_units
    )
    cached_expected = np.load(matrix_dir / "expected_spikes_matrix.npy").reshape(
        matrix_images, matrix_traces, scorer.n_units
    )
    selected_cached_ssi = cached_ssi[np.ix_(image_rows, trace_rows, np.arange(scorer.n_units))]
    selected_cached_expected = cached_expected[
        np.ix_(image_rows, trace_rows, np.arange(scorer.n_units))
    ]
    cached_stabilized_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")[image_rows]
    cached_stabilized_expected = np.load(
        matrix_dir / "stabilized_expected_spikes_by_image.npy"
    )[image_rows]
    replay_validation = {
        "scale1_max_abs_ssi_error_vs_cached": float(np.max(np.abs(ssi[:, :, scale_one_index] - selected_cached_ssi))),
        "scale1_max_abs_expected_spikes_error_vs_cached": float(
            np.max(np.abs(expected[:, :, scale_one_index] - selected_cached_expected))
        ),
        "scale0_max_abs_ssi_error_vs_cached_stabilized": float(
            np.max(np.abs(ssi[:, :, 0] - cached_stabilized_ssi[:, None, :]))
        ),
        "scale0_max_abs_expected_spikes_error_vs_cached_stabilized": float(
            np.max(np.abs(expected[:, :, 0] - cached_stabilized_expected[:, None, :]))
        ),
    }
    (out_dir / "replay_validation.json").write_text(
        json.dumps(replay_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "analysis": "fig4_controlled_centered_trajectory_scaling",
        "matrix_dir": str(matrix_dir.resolve()),
        "output_npz": str(output_npz.resolve()),
        "output_sha256": sha256_file(output_npz),
        "scale_factors": SCALE_FACTORS.tolist(),
        "selection": "equal-quantile representatives of contour coherence and drift-only raw path length",
        "centering": "subtract each trace's sample mean before scaling (original cached traces are already centered to numerical precision)",
        "max_scaled_trace_radius_deg": float(
            np.max(np.linalg.norm(centered, axis=-1)) * float(np.max(SCALE_FACTORS))
        ),
        "renderer_patch_margin_deg": float((540 - 151) / (2.0 * 37.50476617)),
        "n_images": len(images),
        "n_traces": len(traces),
        "n_units": scorer.n_units,
        "time_contract": {
            **trace_time_contract,
            "source_bin_seconds": bin_seconds,
        },
        "replay_validation": replay_validation,
        "model": scorer.provenance,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
