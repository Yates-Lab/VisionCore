#!/usr/bin/env python3
"""Analyze and plot the eight-unit temporal-frequency time-course pilot."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR, write_json
from paper.fig4.spatiotemporal_tuning.run_grating_probe import DENSE_TF_HZ, FRAME_RATE_HZ


FIRST = OUT_DIR / "first_pass_v1"
OUT = FIRST / "tf_timecourse_pilot"
DATA = FIRST / "plot_data"
LOW = "#007C83"
HIGH = "#D55E00"


def amplitude(trace: np.ndarray, tf: float, discard: int) -> float:
    values = np.asarray(trace, float)[discard:]
    time = np.arange(discard, len(trace), dtype=float) / FRAME_RATE_HZ
    design = np.column_stack((np.sin(2 * np.pi * tf * time), np.cos(2 * np.pi * tf * time), np.ones_like(time)))
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(beta[0], beta[1]))


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    selected = pd.read_csv(OUT / "selected_units.csv")
    completed = np.load(OUT / "completed.npy").astype(bool)
    if not np.all(completed):
        raise RuntimeError(f"TF time-course pilot incomplete: {completed.sum()}/{completed.size}")
    rate = np.asarray(np.load(OUT / "rate_traces.npy"), float)
    discards = {"267 ms": 32, "500 ms": 60, "1000 ms": 120}
    rows = []
    for selection_index, unit_row in selected.iterrows():
        for label, discard in discards.items():
            values = np.asarray([amplitude(rate[selection_index, i], float(tf), discard) for i, tf in enumerate(DENSE_TF_HZ)])
            peak = int(np.argmax(values))
            for tf_index, (tf, value) in enumerate(zip(DENSE_TF_HZ, values)):
                rows.append({
                    "unit_index": int(unit_row.unit_index), "selection_category": unit_row.selection_category,
                    "sf_group": unit_row.figure4_sf_group, "discard_label": label, "discard_frames": discard,
                    "temporal_hz": tf, "response_amplitude": value, "peak_tf_hz": float(DENSE_TF_HZ[peak]),
                    "peak_boundary": bool(peak in (0, len(DENSE_TF_HZ) - 1)),
                })
    tuning = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(DATA / "tf_timecourse_tuning_by_discard.csv", index=False)
    peak_table = tuning.groupby(["unit_index", "selection_category", "sf_group", "discard_label"], as_index=False).first()[
        ["unit_index", "selection_category", "sf_group", "discard_label", "peak_tf_hz", "peak_boundary"]
    ]
    peak_table.to_csv(DATA / "tf_timecourse_peak_summary.csv", index=False)

    fig = plt.figure(figsize=(13.4, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    representatives = []
    for group in ("low_sf", "high_sf"):
        row = selected.loc[selected.selection_category == f"{group}_joint_median"].iloc[0]
        representatives.append(row)
    colors = {"267 ms": "#111827", "500 ms": "#2563EB", "1000 ms": "#8B5CF6"}
    for row_index, unit_row in enumerate(representatives):
        unit_id = int(unit_row.unit_index)
        group_color = LOW if unit_row.figure4_sf_group == "low_sf" else HIGH
        ax = fig.add_subplot(grid[row_index, 0])
        for label, color in colors.items():
            sub = tuning.loc[(tuning.unit_index == unit_id) & (tuning.discard_label == label)].sort_values("temporal_hz")
            y = sub.response_amplitude.to_numpy(float)
            ax.plot(sub.temporal_hz, y / max(y.max(), 1e-12), "o-", ms=3, color=color, label=f"discard {label}")
        ax.set(xscale="log", xlabel="temporal frequency (Hz)", ylabel="normalized fitted modulation", title=f"{'A' if row_index == 0 else 'E'}  u{unit_id:03d} {unit_row.figure4_sf_group.replace('_', ' ')} tuning")
        ax.legend(frameon=False, fontsize=7)
        existing_tf = float(unit_row.tf_pref_hz)
        tf_choices = [0.8, float(DENSE_TF_HZ[np.argmin(np.abs(DENSE_TF_HZ - existing_tf))]), 47.2]
        for col, tf in enumerate(tf_choices, start=1):
            tf_index = int(np.argmin(np.abs(DENSE_TF_HZ - tf)))
            trace = rate[selected.index[selected.unit_index == unit_id][0], tf_index]
            time = np.arange(len(trace)) / FRAME_RATE_HZ
            stimulus = np.sin(-2 * np.pi * float(DENSE_TF_HZ[tf_index]) * time)
            ax = fig.add_subplot(grid[row_index, col])
            ax.plot(time, (trace - np.mean(trace)) / max(np.std(trace), 1e-8), color=group_color, lw=1, label="model output (z)")
            ax.plot(time, stimulus - 2.8, color="#6B7280", lw=.8, alpha=.8, label="stimulus phase")
            ax.axvspan(0, 32 / FRAME_RATE_HZ, color="#E5E7EB", alpha=.7)
            ax.set(xlim=(0, 1.25), xlabel="time (s)", ylabel="normalized signal", title=f"{chr(65 + row_index * 4 + col)}  {DENSE_TF_HZ[tf_index]:g} Hz")
            if row_index == 0 and col == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Figure E — Output TF maxima can mix sustained modulation, onset history, and boundary behavior", fontsize=15, weight="bold")
    export(fig, "fig_E_output_tf_timecourses")

    wide = peak_table.pivot(index="unit_index", columns="discard_label", values="peak_tf_hz")
    changed_267_to_1000 = int(np.count_nonzero(~np.isclose(wide["267 ms"], wide["1000 ms"])))
    stats = {
        "status": "eight_predeclared_unit_pilot_single_phase",
        "selected_unit_ids": selected.unit_index.astype(int).tolist(),
        "n_units_whose_peak_changed_267ms_to_1000ms": changed_267_to_1000,
        "n_units": len(selected),
        "highest_tf_hz": float(DENSE_TF_HZ[-1]),
        "samples_per_cycle_at_highest_tf": float(FRAME_RATE_HZ / DENSE_TF_HZ[-1]),
        "interpretation": "Peak instability with discard or a boundary maximum prevents treating ft* as an identified sustained preference.",
    }
    write_json(FIRST / "tf_timecourse_statistics.json", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
