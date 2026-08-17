#!/usr/bin/env python3
"""Rerun time-resolved grating responses for eight predeclared RR100 units."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR, sha256_file, write_json
from paper.fig4.spatiotemporal_tuning.run_grating_probe import (
    DENSE_TF_HZ,
    FRAME_RATE_HZ,
    make_grating_movie,
    score_scalar_traces,
)
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = OUT_DIR / "first_pass_v1" / "tf_timecourse_pilot"
N_FRAMES = int(3.0 * FRAME_RATE_HZ)


def closest_joint_median(frame: pd.DataFrame) -> int:
    sf0 = float(np.median(np.log2(frame.existing_sf_pref_cpd)))
    tf0 = float(np.median(np.log2(frame.tf_pref_hz)))
    distance = (np.log2(frame.existing_sf_pref_cpd) - sf0) ** 2 + (np.log2(frame.tf_pref_hz) - tf0) ** 2
    return int(frame.loc[distance.sort_values(kind="mergesort").index[0], "unit_index"])


def select_units(table: pd.DataFrame) -> pd.DataFrame:
    """Eight categories fixed from the existing scalar probe before new traces."""
    selected: list[tuple[str, int]] = []
    for group in ("low_sf", "high_sf"):
        interior = table.loc[(table.figure4_sf_group == group) & ~table.tf_peak_boundary].copy()
        selected.append((f"{group}_interior_low_tf", int(interior.sort_values(["tf_pref_hz", "unit_index"]).iloc[0].unit_index)))
        selected.append((f"{group}_interior_high_tf", int(interior.sort_values(["tf_pref_hz", "unit_index"]).iloc[-1].unit_index)))
    for group in ("low_sf", "high_sf"):
        interior = table.loc[(table.figure4_sf_group == group) & ~table.tf_peak_boundary].copy()
        selected.append((f"{group}_joint_median", closest_joint_median(interior)))
    left = table.loc[table.tf_peak_censoring == "left"].copy()
    left_target = float(np.median(left.existing_sf_pref_cpd))
    selected.append(("left_censored", int(left.loc[(left.existing_sf_pref_cpd - left_target).abs().idxmin()].unit_index)))
    right = table.loc[table.tf_peak_censoring == "right"].copy()
    right_target = float(np.median(right.existing_sf_pref_cpd))
    selected.append(("right_censored", int(right.loc[(right.existing_sf_pref_cpd - right_target).abs().idxmin()].unit_index)))
    # Extremely unlikely duplicates are filled deterministically without seeing new data.
    used: set[int] = set()
    rows = []
    for category, unit_id in selected:
        if unit_id in used:
            replacement = int(table.loc[~table.unit_index.isin(used)].sort_values("unit_index").iloc[0].unit_index)
            unit_id = replacement
            category += "_deduplicated_fill"
        used.add(unit_id)
        row = table.loc[table.unit_index == unit_id].iloc[0].to_dict()
        row["selection_category"] = category
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--max-conditions", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/analysis/per_unit_tuning.csv"
    table = pd.read_csv(source).sort_values("unit_index").reset_index(drop=True)
    selected = select_units(table)
    selected.to_csv(OUT / "selected_units.csv", index=False)
    shape = (len(selected), len(DENSE_TF_HZ), N_FRAMES)
    traces_path = OUT / "rate_traces.npy"
    complete_path = OUT / "completed.npy"
    if traces_path.exists():
        rate = np.lib.format.open_memmap(traces_path, mode="r+")
        if rate.shape != shape:
            raise ValueError(f"Existing timecourse shape {rate.shape}, expected {shape}")
    else:
        rate = np.lib.format.open_memmap(traces_path, mode="w+", dtype=np.float32, shape=shape)
        rate[:] = np.nan
        rate.flush()
    complete = np.load(complete_path).astype(bool) if complete_path.exists() else np.zeros(shape[:2], bool)
    np.save(complete_path, complete)
    pending = list(zip(*np.where(~complete)))
    if int(args.max_conditions) > 0:
        pending = pending[: int(args.max_conditions)]
    if not pending:
        print("No pending TF time-course conditions", flush=True)
        return 0
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=str(args.device), strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    for run_index, (selection_index, tf_index) in enumerate(pending, start=1):
        row = selected.iloc[int(selection_index)]
        movie = make_grating_movie(
            orientation_deg=float(row.nearest_probe_orientation_deg),
            spatial_cpd=float(row.nearest_probe_sf_cpd),
            temporal_hz=float(DENSE_TF_HZ[tf_index]),
            phase_rad=0.0, duration_s=3.0,
        )
        all_traces = score_scalar_traces(scorer, movie, frame_batch_size=int(args.frame_batch_size))
        if all_traces.shape[0] != N_FRAMES:
            raise ValueError(all_traces.shape)
        rate[selection_index, tf_index] = all_traces[:, int(row.unit_index)]
        rate.flush()
        complete[selection_index, tf_index] = True
        np.save(complete_path, complete)
        print(
            f"TF trace {run_index}/{len(pending)} complete={complete.sum()}/{complete.size}: "
            f"u{int(row.unit_index):03d}, {DENSE_TF_HZ[tf_index]:g} Hz",
            flush=True,
        )
    write_json(
        OUT / "manifest.json",
        {
            "analysis": "eight_unit_output_tf_timecourse_pilot",
            "status": "pilot",
            "selection_contract": "categories selected from the existing scalar grating probe before time-course scoring",
            "n_units": len(selected), "n_temporal_frequencies": len(DENSE_TF_HZ),
            "temporal_hz": DENSE_TF_HZ, "duration_s": 3.0, "frame_rate_hz": FRAME_RATE_HZ,
            "phase_rad": 0.0, "n_completed": int(complete.sum()),
            "checkpoint": MODEL_CHECKPOINT_PATH, "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "device": str(args.device),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
