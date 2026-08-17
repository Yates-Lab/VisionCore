#!/usr/bin/env python3
"""Recover true source prehistory and build validated true/held trajectory banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    DT_S,
    FRAME_RATE_HZ,
    LEGACY_MATRIX_DIR,
    N_LAGS,
    N_PRECEDING,
    N_SCORED,
    OUT_DIR,
    SOURCE_CSV,
    corrected_source_indices,
    held_time_indices,
    lag_windows_from_sequence,
    sha256_file,
    validate_monotone_causal_windows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-matrix-dir", type=Path, default=LEGACY_MATRIX_DIR)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_eye_positions(sessions: list[str]) -> dict[str, np.ndarray]:
    from paper.fig4.upstream.real_trace_matrix.core import load_backimage_eyepos_by_session

    return load_backimage_eyepos_by_session(sessions)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    bank_dir = out_dir / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path(args.legacy_matrix_dir)
    table = pd.read_csv(legacy_dir / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    stored = np.load(legacy_dir / "trace_xy.npy").astype(np.float32)
    source_rows = pd.read_csv(args.source_csv).reset_index().rename(columns={"index": "source_row"})
    eye_by_session = load_eye_positions(table["session"].astype(str).unique().tolist())

    n = len(table)
    true_xy = np.full((n, N_PRECEDING + N_SCORED, 2), np.nan, dtype=np.float32)
    held_xy = np.full_like(true_xy, np.nan)
    source_index = np.full((n, N_PRECEDING + N_SCORED), -1, dtype=np.int64)
    true_rows: list[dict[str, object]] = []
    held_rows: list[dict[str, object]] = []
    invalid_reasons: list[str] = []

    for index, row in table.iterrows():
        session = str(row["session"])
        scored_start = int(row["snippet_global_start"])
        scored_stop = int(row["snippet_global_stop"])
        window_start = int(row["source_window_global_start"])
        window_stop = int(row["source_window_global_stop"])
        reason = ""
        if scored_stop - scored_start != N_SCORED:
            reason = "scored_interval_not_40_samples"
        elif scored_start - N_PRECEDING < window_start:
            reason = "insufficient_preceding_samples_inside_source_fixation_window"
        elif scored_stop > window_stop:
            reason = "scored_interval_exceeds_source_fixation_window"
        eyepos = eye_by_session[session]
        indices = corrected_source_indices(scored_start)
        if not reason and (indices[0] < 0 or indices[-1] >= len(eyepos)):
            reason = "source_indices_out_of_recording_bounds"
        raw_full = np.asarray(eyepos[indices], dtype=np.float64) if not reason else np.empty((0, 2))
        if not reason and (raw_full.shape != (N_PRECEDING + N_SCORED, 2) or not np.isfinite(raw_full).all()):
            reason = "nonfinite_or_bad_shape_source_history"
        if reason:
            invalid_reasons.append(reason)
            true_rows.append({
                "trajectory_id": int(row["trace_bank_index"]), "source_session": session,
                "source_trial_fixation": int(row["trial_idx"]), "valid_history": False,
                "reason_invalid": reason,
            })
            continue

        # The historical builder centered the scored 40 samples by their own
        # mean.  Apply that one translation to the complete 71-sample segment.
        scored_raw = raw_full[N_PRECEDING:]
        common_translation = np.mean(scored_raw, axis=0)
        transformed = (raw_full - common_translation[None]).astype(np.float32)
        reconstruction_error = float(np.max(np.abs(transformed[N_PRECEDING:] - stored[index])))
        true_xy[index] = transformed
        held_xy[index, :N_PRECEDING] = transformed[N_PRECEDING]
        held_xy[index, N_PRECEDING:] = transformed[N_PRECEDING:]
        source_index[index] = indices
        source_meta = source_rows.iloc[int(row["source_row"])]
        if str(source_meta["session"]) != session or int(source_meta["global_start"]) != window_start:
            raise AssertionError(f"Source-row provenance mismatch for trajectory {index}")
        common = {
            "trajectory_id": int(row["trace_bank_index"]),
            "source_row": int(row["source_row"]),
            "source_session": session,
            "source_trial_fixation": int(row["trial_idx"]),
            "source_window_start_index": window_start,
            "source_window_end_index_exclusive": window_stop,
            "source_start_index": int(indices[0]),
            "scored_start_index": scored_start,
            "source_end_index_exclusive": scored_stop,
            "number_of_preceding_samples": N_PRECEDING,
            "source_sampling_hz": FRAME_RATE_HZ,
            "downsampling": "none; native 120 Hz samples",
            "filtering": "none",
            "validity_mask": "all 71 recovered samples finite and inside selected source fixation window",
            "event_mask_processing": "not applied to coordinates; cached interval event label retained for stratification",
            "coordinate_transform": "subtract mean of the scored 40 source samples once from full history+score",
            "translation_x_deg": float(common_translation[0]),
            "translation_y_deg": float(common_translation[1]),
            "reconstruction_error_vs_original_40_samples": reconstruction_error,
            "valid_history": True,
            "reason_invalid": "",
            "cached_interval_has_microsaccade": int(row["rendered_n_microsaccade_events"]) > 0,
            "rendered_path_length_arcmin": float(row["rendered_path_length_arcmin"]),
        }
        true_rows.append({**common, "bank": "real_trace_true_history_v1", "prefix_type": "real source samples"})
        held_rows.append({
            **common,
            "bank": "real_trace_held_initial_history_v1",
            "prefix_type": "synthetic hold at scored sample e[0]",
            "source_start_index": "synthetic",
        })

    valid = np.isfinite(true_xy).all(axis=(1, 2))
    if not np.all(valid):
        raise RuntimeError(f"Primary source recovery lost {np.sum(~valid)} trajectories: {pd.Series(invalid_reasons).value_counts().to_dict()}")
    max_error = float(max(float(row["reconstruction_error_vs_original_40_samples"]) for row in true_rows))
    if max_error != 0.0:
        raise AssertionError(f"Stored snippet reconstruction is not exact: {max_error}")

    # Global automatic causal checks.  The mapping is identical across true
    # trajectories up to a constant source-index offset.
    true_example_indices = source_index[0]
    true_windows = lag_windows_from_sequence(true_example_indices)
    true_output_times = true_example_indices[N_PRECEDING:]
    true_validation = validate_monotone_causal_windows(true_windows, true_output_times)
    held_indices = held_time_indices()
    held_windows = lag_windows_from_sequence(held_indices)
    held_output_times = held_indices[N_PRECEDING:]
    held_validation = validate_monotone_causal_windows(held_windows, held_output_times)
    if any(true_validation[key] for key in true_validation if key != "total_outputs"):
        raise AssertionError(true_validation)
    if any(held_validation[key] for key in held_validation if key != "total_outputs"):
        raise AssertionError(held_validation)
    true_boundary_step = np.linalg.norm(true_xy[:, N_PRECEDING] - true_xy[:, N_PRECEDING - 1], axis=1)
    source_boundary_step = np.asarray([
        np.linalg.norm(
            eye_by_session[str(row["session"])][int(row["snippet_global_start"])]
            - eye_by_session[str(row["session"])][int(row["snippet_global_start"]) - 1]
        )
        for _, row in table.iterrows()
    ])
    boundary_error = float(np.max(np.abs(true_boundary_step - source_boundary_step)))
    held_boundary_step = np.linalg.norm(held_xy[:, N_PRECEDING] - held_xy[:, N_PRECEDING - 1], axis=1)

    np.savez_compressed(
        bank_dir / "corrected_history_trajectory_banks.npz",
        true_history_xy=true_xy,
        held_initial_history_xy=held_xy,
        true_source_indices=source_index,
        held_relative_time_indices=held_indices,
        stored_scored_trace_xy=stored,
        n_preceding=np.asarray(N_PRECEDING, dtype=np.int32),
        n_lags=np.asarray(N_LAGS, dtype=np.int32),
        frame_rate_hz=np.asarray(FRAME_RATE_HZ),
    )
    pd.DataFrame(true_rows).to_csv(out_dir / "true_history_trajectory_table.csv", index=False)
    pd.DataFrame(held_rows).to_csv(out_dir / "held_prefix_trajectory_table.csv", index=False)
    validation_rows = [
        {
            "bank": "real_trace_true_history_v1",
            "total_outputs": n * N_SCORED,
            "outputs_with_future_samples": true_validation["outputs_with_future_samples"] * n,
            "outputs_with_nonmonotonic_source_indices": true_validation["outputs_with_nonmonotonic_indices"] * n,
            "outputs_with_artificial_boundary_discontinuity": 0 if boundary_error <= 1e-6 else n,
            "trajectories_with_incomplete_history": 0,
            "trajectories_excluded": 0,
            "maximum_reconstruction_error": max_error,
            "maximum_boundary_step_error_vs_source_deg": boundary_error,
        },
        {
            "bank": "real_trace_held_initial_history_v1",
            "total_outputs": n * N_SCORED,
            "outputs_with_future_samples": held_validation["outputs_with_future_samples"] * n,
            "outputs_with_nonmonotonic_source_indices": held_validation["outputs_with_nonmonotonic_indices"] * n,
            "outputs_with_artificial_boundary_discontinuity": int(np.sum(held_boundary_step != 0.0)),
            "trajectories_with_incomplete_history": 0,
            "trajectories_excluded": 0,
            "maximum_reconstruction_error": float(np.max(np.abs(held_xy[:, N_PRECEDING:] - stored))),
            "maximum_boundary_step_error_vs_source_deg": "not_applicable_synthetic_hold",
        },
    ]
    pd.DataFrame(validation_rows).to_csv(out_dir / "history_validation.csv", index=False)
    write_json(
        bank_dir / "bank_manifest.json",
        {
            "primary_bank": "real_trace_true_history_v1",
            "control_bank": "real_trace_held_initial_history_v1",
            "legacy_name": "legacy_wrapped_prefix",
            "source_csv": Path(args.source_csv),
            "source_csv_sha256": sha256_file(Path(args.source_csv)),
            "legacy_trace_xy": legacy_dir / "trace_xy.npy",
            "legacy_trace_xy_sha256": sha256_file(legacy_dir / "trace_xy.npy"),
            "n_trajectories": n,
            "n_preceding_samples": N_PRECEDING,
            "n_scored_samples": N_SCORED,
            "all_histories_valid": True,
            "max_reconstruction_error": max_error,
            "max_true_boundary_step_error_vs_source_deg": boundary_error,
            "max_held_boundary_step_deg": float(np.max(held_boundary_step)),
            "output_npz": bank_dir / "corrected_history_trajectory_banks.npz",
            "output_npz_sha256": sha256_file(bank_dir / "corrected_history_trajectory_banks.npz"),
        },
    )
    print(f"Recovered {n} exact true histories; max reconstruction error={max_error:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
