#!/usr/bin/env python3
"""Generate a BackImage real-trace x image RR100 SSI matrix from source inputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

UPSTREAM_DIR = Path(__file__).resolve().parent
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from real_trace_matrix.core import (
    build_native_snippet_trace_bank,
    annotate_selected_image_flags,
    image_candidate_rows,
    image_sampling_summary,
    load_backimage_eyepos_by_session,
    load_source_rows,
    microsaccade_event_count,
    sample_image_rows,
    sample_trace_items,
    score_matrix,
    trace_bank_metadata_row,
    trace_bank_metric_summary_rows,
    trace_items_from_table_and_array,
    write_csv,
    write_json,
    write_unit_feature_table,
)
from real_trace_matrix.model import RealTraceMatrixScorer


ROOT = Path(__file__).resolve().parents[3]
RUN_STEM = "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_SOURCE_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_UNIT_TUNING_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/active_sensing_movie_information" / RUN_STEM
CHECKPOINT_ENV = "FIG4_TWIN_CHECKPOINT"
MODEL_CHECKPOINT_FILENAME = "epoch=147-val_bps_overall=0.5702.ckpt"
STAGED_MODEL_CHECKPOINT_PATH = ROOT / "outputs/artifacts/model_checkpoints/fig4_twin" / MODEL_CHECKPOINT_FILENAME


def default_checkpoint_path() -> Path:
    if CHECKPOINT_ENV in os.environ:
        return Path(os.environ[CHECKPOINT_ENV])
    return STAGED_MODEL_CHECKPOINT_PATH
DEFAULT_DATASET_CONFIGS = Path(
    os.environ.get(
        "FIG4_DATASET_CONFIGS",
        str(UPSTREAM_DIR / "dataset_configs" / "multi_basic_120_long.yaml"),
    )
)
DEFAULT_POPULATION_SPEC_DIR = Path(
    os.environ.get(
        "FIG4_RR100_POPULATION_SPEC_DIR",
        str(ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"),
    )
)
DEFAULT_MCFARLAND_OUTPUTS = os.environ.get("FIG4_MCFARLAND_OUTPUTS")


def parse_session_filter(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def filter_source_rows(rows: pd.DataFrame, session_filter: list[str]) -> pd.DataFrame:
    if not session_filter:
        return rows
    if "session" not in rows.columns:
        raise ValueError("--session-filter requires a 'session' column in --source-csv.")
    mask = rows["session"].astype(str).isin(session_filter)
    filtered = rows.loc[mask].copy()
    if filtered.empty:
        available = sorted(rows["session"].astype(str).dropna().unique().tolist())
        preview = ", ".join(available[:8])
        suffix = "..." if len(available) > 8 else ""
        raise ValueError(
            "No source rows matched --session-filter "
            f"{','.join(session_filter)!r}. Available sessions include: {preview}{suffix}"
        )
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--unit-tuning-csv", type=Path, default=DEFAULT_UNIT_TUNING_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_VERSION)
    parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint_path())
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument(
        "--mcfarland-outputs",
        type=Path,
        default=Path(DEFAULT_MCFARLAND_OUTPUTS) if DEFAULT_MCFARLAND_OUTPUTS else None,
    )
    parser.add_argument(
        "--session-filter",
        type=str,
        default="",
        help="Comma-separated session names to keep from --source-csv before image/trace sampling.",
    )
    parser.add_argument(
        "--replay-matrix-dir",
        type=Path,
        default=None,
        help="Replay selected image/trace tables from a historical matrix dir instead of resampling from source.",
    )
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--benchmark-n-images", type=int, default=2)
    parser.add_argument("--benchmark-n-traces", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--image-contrast-quantile", type=float, default=0.75)
    parser.add_argument("--image-min-orientation-coherence", type=float, default=0.0)
    parser.add_argument("--image-min-drift-anisotropy", type=float, default=0.0)
    parser.add_argument("--min-strong-contour-images", type=int, default=0)
    parser.add_argument("--strong-contour-orientation-coherence-min", type=float, default=0.5)
    parser.add_argument("--max-trace-path-length-arcmin", type=float, default=350.0)
    parser.add_argument("--trace-scale-metric", type=str, default="rendered_path_length_arcmin")
    parser.add_argument("--trace-sampling", choices=("quantile", "random"), default="quantile")
    parser.add_argument("--min-microsaccade-traces", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark-frame-batch-sizes", type=str, default="8,32,64")
    parser.add_argument("--benchmark-trace-batch-sizes", type=str, default="1,8,16")
    parser.add_argument("--pilot-frame-batch-size", type=int, default=16)
    parser.add_argument("--pilot-trace-batch-size", type=int, default=8)
    parser.add_argument("--image-shard-start", type=int, default=0)
    parser.add_argument("--image-shard-stop", type=int, default=0)
    parser.add_argument("--trace-shard-start", type=int, default=0)
    parser.add_argument("--trace-shard-stop", type=int, default=0)
    parser.add_argument("--skip-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def feature_rows_from_items(items: list[dict[str, Any]], *, scale_metric: str, n_timepoints: int) -> list[dict[str, Any]]:
    return [
        trace_bank_metadata_row(item, idx, n_timepoints=int(n_timepoints), scale_metric=str(scale_metric))
        for idx, item in enumerate(items)
    ]


def build_trace_bank(args: argparse.Namespace, rows: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_rows = rows.drop_duplicates("source_row").copy()
    if "n_samples" in trace_rows.columns:
        trace_rows = trace_rows[pd.to_numeric(trace_rows["n_samples"], errors="coerce") >= int(args.n_timepoints)].copy()
    sessions = trace_rows["session"].astype(str).dropna().unique().tolist()
    eyepos_by_session = load_backimage_eyepos_by_session(sessions)
    bank, meta = build_native_snippet_trace_bank(
        trace_rows,
        eyepos_by_session,
        int(args.n_timepoints),
        dt=float(args.bin_seconds),
        microsaccade_speed_threshold_dps=None,
        microsaccade_threshold_z=6.0,
        microsaccade_pad_frames=1,
    )
    eligible: list[dict[str, Any]] = []
    for item in bank:
        item["observed_rms_arcmin"] = float(item["observed_rms_deg"]) * 60.0
        item["path_length_arcmin"] = float(item["path_length_deg"]) * 60.0
        path_arcmin = float(item.get("rendered_path_length_arcmin", item["path_length_arcmin"]))
        if path_arcmin <= float(args.max_trace_path_length_arcmin):
            eligible.append(item)
    meta.update(
        {
            "trace_bank_rows_before_path_filter": int(len(bank)),
            "trace_bank_eligible_rows": int(len(eligible)),
            "max_trace_path_length_arcmin": float(args.max_trace_path_length_arcmin),
        }
    )
    return eligible, meta


def replay_selection(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    replay_dir = Path(args.replay_matrix_dir)
    image_path = replay_dir / "image_feature_table.csv"
    trace_path = replay_dir / "trace_feature_table.csv"
    trace_xy_path = replay_dir / "trace_xy.npy"
    for path in (image_path, trace_path, trace_xy_path):
        if not path.exists():
            raise FileNotFoundError(f"Replay input is missing: {path}")

    image_table = pd.read_csv(image_path)
    trace_table_all = pd.read_csv(trace_path)
    trace_xy_all = np.load(trace_xy_path)
    if "image_index" not in image_table.columns:
        raise ValueError(f"{image_path} must contain image_index.")
    if int(trace_xy_all.shape[0]) != int(trace_table_all.shape[0]):
        raise ValueError(
            f"{trace_xy_path} rows ({trace_xy_all.shape[0]}) do not match "
            f"{trace_path} rows ({trace_table_all.shape[0]})."
        )

    image_table.to_csv(out_dir / "image_feature_table.csv", index=False)
    image_start = max(0, int(args.image_shard_start))
    default_image_stop = image_start + int(args.n_images)
    image_stop = int(args.image_shard_stop) if int(args.image_shard_stop) > 0 else default_image_stop
    image_stop = min(image_stop, int(image_table.shape[0]))
    if image_start >= image_stop:
        raise ValueError(
            f"Empty replay image shard: start={image_start}, stop={image_stop}, "
            f"n_reference_images={image_table.shape[0]}."
        )
    score_images = image_table.iloc[image_start:image_stop].copy().reset_index(drop=True)
    score_images.to_csv(out_dir / "scored_image_feature_table.csv", index=False)

    trace_start = max(0, int(args.trace_shard_start))
    default_trace_stop = trace_start + int(args.n_traces)
    trace_stop = int(args.trace_shard_stop) if int(args.trace_shard_stop) > 0 else default_trace_stop
    trace_stop = min(trace_stop, int(trace_table_all.shape[0]))
    if trace_start >= trace_stop:
        raise ValueError(
            f"Empty replay trace shard: start={trace_start}, stop={trace_stop}, "
            f"n_reference_traces={trace_table_all.shape[0]}."
        )
    trace_table = trace_table_all.iloc[trace_start:trace_stop].copy().reset_index(drop=True)
    trace_xy = np.asarray(trace_xy_all[trace_start:trace_stop], dtype=np.float32)
    traces = trace_items_from_table_and_array(trace_table, trace_xy, n_timepoints=int(args.n_timepoints))
    trace_table.to_csv(out_dir / "trace_feature_table.csv", index=False)
    write_csv(out_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(trace_table.to_dict("records")))

    return {
        "source_filter": {
            "mode": "replay",
            "replay_matrix_dir": replay_dir,
            "source_csv": None,
        },
        "image_sampling": {
            "mode": "replay",
            "reference_matrix_dir": replay_dir,
            "n_reference_images": int(image_table.shape[0]),
            "n_images": int(score_images.shape[0]),
            "image_start": int(image_start),
            "image_stop": int(image_stop),
            "global_image_indices": score_images["image_index"].astype(int).to_list(),
        },
        "trace_sampling": {
            "mode": "replay",
            "reference_matrix_dir": replay_dir,
            "n_reference_traces": int(trace_table_all.shape[0]),
            "n_traces": int(len(traces)),
            "trace_start": int(trace_start),
            "trace_stop": int(trace_stop),
            "trace_scale_metric": str(args.trace_scale_metric),
            "trace_bank_eligible_rows": int(trace_table_all.shape[0]),
            "selected_microsaccade_traces": int(sum(microsaccade_event_count(item) > 0 for item in traces)),
        },
        "trace_bank": {
            "trace_bank_snippet_policy": "replay_trace_xy",
            "trace_bank_native_snippet_n_timepoints": int(args.n_timepoints),
            "trace_xy": trace_xy_path,
            "trace_feature_table": trace_path,
        },
        "image_start": int(image_start),
        "image_stop": int(image_stop),
        "image_table": image_table,
        "score_images": score_images,
        "traces": traces,
        "trace_index_offset": int(trace_start),
        "movie_index_stride": int(trace_table_all.shape[0]),
    }


def main() -> int:
    args = parse_args()
    if not bool(args.skip_benchmark) or bool(args.benchmark_only):
        raise NotImplementedError(
            "Benchmark search is not ported in the clean Fig. 4 scorer; pass --skip-benchmark "
            "and explicit --pilot-frame-batch-size/--pilot-trace-batch-size."
        )

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"{out_dir} already exists and is not empty. Pass --force to append/overwrite files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replay_matrix_dir is not None:
        selection = replay_selection(args, out_dir)
    else:
        rng = np.random.default_rng(int(args.seed))
        rows = load_source_rows(Path(args.source_csv))
        rows_before_filter = int(rows.shape[0])
        session_filter = parse_session_filter(args.session_filter)
        rows = filter_source_rows(rows, session_filter)
        rows_after_filter = int(rows.shape[0])
        image_candidates = image_candidate_rows(
            rows,
            contrast_quantile=float(args.image_contrast_quantile),
            n_timepoints=int(args.n_timepoints),
            min_orientation_coherence=float(args.image_min_orientation_coherence),
            min_drift_anisotropy=float(args.image_min_drift_anisotropy),
        )
        images = sample_image_rows(
            image_candidates,
            int(args.n_images),
            rng=rng,
            min_strong_contour_images=int(args.min_strong_contour_images),
            strong_contour_orientation_coherence_min=float(args.strong_contour_orientation_coherence_min),
        )
        image_table = annotate_selected_image_flags(
            images.copy().reset_index(drop=True),
            reliable_min=max(0.2, float(args.image_min_orientation_coherence)),
            strong_min=float(args.strong_contour_orientation_coherence_min),
        )
        image_table.insert(0, "image_index", np.arange(image_table.shape[0], dtype=int))
        image_table.to_csv(out_dir / "image_feature_table.csv", index=False)

        shard_start = max(0, int(args.image_shard_start))
        shard_stop = int(args.image_shard_stop) if int(args.image_shard_stop) > 0 else int(image_table.shape[0])
        shard_stop = min(shard_stop, int(image_table.shape[0]))
        if shard_start >= shard_stop:
            raise ValueError(
                f"Empty image shard: start={shard_start}, stop={shard_stop}, n_images={image_table.shape[0]}."
            )
        score_images = image_table.iloc[shard_start:shard_stop].copy().reset_index(drop=True)
        score_images.to_csv(out_dir / "scored_image_feature_table.csv", index=False)

        trace_bank, trace_bank_meta = build_trace_bank(args, rows)
        traces = sample_trace_items(
            trace_bank,
            int(args.n_traces),
            metric=str(args.trace_scale_metric),
            sampling=str(args.trace_sampling),
            rng=rng,
            min_microsaccade_traces=int(args.min_microsaccade_traces),
        )
        trace_rows = feature_rows_from_items(
            traces,
            scale_metric=str(args.trace_scale_metric),
            n_timepoints=int(args.n_timepoints),
        )
        write_csv(out_dir / "trace_feature_table.csv", trace_rows)
        write_csv(out_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(trace_rows))
        selection = {
            "source_filter": {
                "session_filter": session_filter,
                "rows_before_filter": rows_before_filter,
                "rows_after_filter": rows_after_filter,
            },
            "image_sampling": image_sampling_summary(images, n_candidates=image_candidates.shape[0], args=args),
            "trace_sampling": {
                "n_traces": int(args.n_traces),
                "trace_sampling": str(args.trace_sampling),
                "trace_scale_metric": str(args.trace_scale_metric),
                "max_trace_path_length_arcmin": float(args.max_trace_path_length_arcmin),
                "trace_bank_eligible_rows": int(len(trace_bank)),
                "min_microsaccade_traces": int(args.min_microsaccade_traces),
                "selected_microsaccade_traces": int(sum(microsaccade_event_count(item) > 0 for item in traces)),
            },
            "trace_bank": trace_bank_meta,
            "image_start": int(shard_start),
            "image_stop": int(shard_stop),
            "image_table": image_table,
            "score_images": score_images,
            "traces": traces,
            "trace_index_offset": 0,
            "movie_index_stride": None,
        }

    image_table = selection["image_table"]
    score_images = selection["score_images"]
    traces = selection["traces"]

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=Path(args.checkpoint_path),
        dataset_configs=Path(args.dataset_configs),
        population_spec_dir=Path(args.population_spec_dir),
        rr100_version=str(args.rr100_version),
        device=str(args.device),
        mcfarland_outputs=Path(args.mcfarland_outputs) if args.mcfarland_outputs is not None else None,
    )
    if not np.isfinite(float(args.bin_seconds)) or float(args.bin_seconds) <= 0.0:
        raise ValueError(
            f"Trace bin_seconds must be positive, got {args.bin_seconds}."
        )
    source_trace_rate_hz = int(round(1.0 / float(args.bin_seconds)))
    if scorer.output_rate_hz < source_trace_rate_hz or (
        scorer.output_rate_hz % source_trace_rate_hz
    ):
        raise ValueError(
            "Selected twin output rate must be an integer multiple of the retained "
            f"trace rate; got {source_trace_rate_hz} -> {scorer.output_rate_hz} Hz."
        )
    scored_samples_per_source = scorer.output_rate_hz // source_trace_rate_hz
    scored_timepoints = int(args.n_timepoints) * scored_samples_per_source
    write_unit_feature_table(
        out_dir / "unit_feature_table.csv",
        scorer.rr_unit_rows,
        Path(args.unit_tuning_csv),
        int(scorer.n_units),
    )

    pilot_frame_batch = max(1, int(args.pilot_frame_batch_size))
    pilot_trace_batch = max(1, int(args.pilot_trace_batch_size))
    pilot_timing = score_matrix(
        scorer=scorer,
        image_rows=score_images,
        trace_items=traces,
        frame_batch_size=pilot_frame_batch,
        trace_batch_size=pilot_trace_batch,
        n_timepoints=int(args.n_timepoints),
        bin_seconds=float(args.bin_seconds),
        patch_size_px=int(args.patch_size_px),
        write_outputs=True,
        out_dir=out_dir,
        trace_index_offset=int(selection["trace_index_offset"]),
        movie_index_stride=selection["movie_index_stride"],
    )
    payload = {
        "analysis": "backimage_real_trace_ssi_matrix",
        "source_csv": None if args.replay_matrix_dir is not None else Path(args.source_csv),
        "replay_matrix_dir": Path(args.replay_matrix_dir) if args.replay_matrix_dir is not None else None,
        "unit_tuning_csv": Path(args.unit_tuning_csv),
        "out_dir": out_dir,
        "rr100_version": str(args.rr100_version),
        "n_timepoints": int(args.n_timepoints),
        "bin_seconds": float(args.bin_seconds),
        "trace_time_contract": {
            "source_trace_rate_hz": source_trace_rate_hz,
            "source_trace_samples": int(args.n_timepoints),
            "model_output_rate_hz": int(scorer.output_rate_hz),
            "scored_samples_per_source_trace_sample": int(
                scored_samples_per_source
            ),
            "scored_trace_samples": scored_timepoints,
            "scored_bin_seconds": 1.0 / float(scorer.output_rate_hz),
            "analysis_interval_seconds": (
                float(args.n_timepoints) / float(source_trace_rate_hz)
            ),
            "resampling": (
                "endpoint-anchored linear interpolation with held boundaries"
                if scored_samples_per_source > 1
                else "none"
            ),
        },
        "patch_size_px": int(args.patch_size_px),
        "source_filter": selection["source_filter"],
        "image_sampling": selection["image_sampling"],
        "image_shard": {
            "start": int(selection["image_start"]),
            "stop": int(selection["image_stop"]),
            "n_total_images": int(image_table.shape[0]),
            "n_scored_images": int(score_images.shape[0]),
            "global_image_indices": score_images["image_index"].astype(int).to_list(),
        },
        "trace_sampling": selection["trace_sampling"],
        "trace_bank": selection["trace_bank"],
        "pilot": {
            "frame_batch_size": int(pilot_frame_batch),
            "trace_batch_size": int(pilot_trace_batch),
            **pilot_timing,
        },
        "model_provenance": scorer.provenance,
        "outputs": {
            "ssi_matrix": out_dir / "ssi_matrix.npy",
            "expected_spikes_matrix": out_dir / "expected_spikes_matrix.npy",
            "mean_rate_matrix": out_dir / "mean_rate_matrix.npy",
            "population_ssi": out_dir / "population_ssi.npy",
            "movie_feature_table": out_dir / "movie_feature_table.csv",
            "image_feature_table": out_dir / "image_feature_table.csv",
            "scored_image_feature_table": out_dir / "scored_image_feature_table.csv",
            "trace_feature_table": out_dir / "trace_feature_table.csv",
            "trace_xy": out_dir / "trace_xy.npy",
            "unit_feature_table": out_dir / "unit_feature_table.csv",
        },
        "contract": (
            "Rows are image-major image x trace movies. SSI is corrected time-resolved spatial SSI "
            "from full twin rate maps after applying the RR100 population view. Traces are unscaled "
            "center-cropped real BackImage snippets on their retained source grid; native-rate "
            "twins receive endpoint-anchored interpolation without changing the physical interval."
        ),
    }
    write_json(out_dir / "summary.json", payload)
    print(f"[fig4-real-trace-matrix] wrote matrix shard to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
