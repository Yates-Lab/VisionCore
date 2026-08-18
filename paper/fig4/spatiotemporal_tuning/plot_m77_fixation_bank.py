#!/usr/bin/env python3
"""Quality-control figure for the filtered/raw M77 fixation bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_bank", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = pd.read_csv(args.trace_bank / "trace_table.csv").sort_values("trace_index")
    filtered = np.load(args.trace_bank / "trace_xy_filtered.npy")
    raw = np.load(args.trace_bank / "trace_xy_raw.npy")
    if filtered.shape != raw.shape or filtered.shape != (len(table), 300, 2):
        raise ValueError("fixation-bank table and trace arrays disagree")
    examples = np.unique(np.round(np.linspace(0, len(table) - 1, 4)).astype(int))
    figure = plt.figure(figsize=(15.0, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    path_axis = figure.add_subplot(grid[0, 0])
    for index in examples:
        value = filtered[index, 60:] * 60.0
        path_axis.plot(value[:, 0], value[:, 1], lw=1.1, label=f"trace {index}")
    path_axis.set_aspect("equal", adjustable="datalim")
    path_axis.set(xlabel="horizontal position (arcmin)", ylabel="vertical position (arcmin)", title="A  One-second scored trajectories")
    path_axis.legend(frameon=False, fontsize=8)

    time_axis = figure.add_subplot(grid[0, 1:])
    time = (np.arange(300) - 60) / 240.0
    index = int(examples[len(examples) // 2])
    time_axis.plot(time, raw[index, :, 0] * 60, color="0.65", lw=1, label="raw-resampled x")
    time_axis.plot(time, filtered[index, :, 0] * 60, color="#1479B8", lw=1.5, label="filtered x")
    time_axis.plot(time, raw[index, :, 1] * 60, color="#E9A3A3", lw=1, label="raw-resampled y")
    time_axis.plot(time, filtered[index, :, 1] * 60, color="#C94343", lw=1.5, label="filtered y")
    time_axis.axvspan(-0.25, 0, color="0.9", label="real history")
    time_axis.axvline(0, color="black", lw=0.8)
    time_axis.set(xlabel="time from scored interval (s)", ylabel="centered eye position (arcmin)", title="B  Filtering precedes native-240 sampling")
    time_axis.legend(frameon=False, ncol=3, fontsize=8)

    spectrum_axis = figure.add_subplot(grid[1, 0])
    spectra = {}
    for label, trace, color in (("raw-resampled", raw, "0.45"), ("100-Hz-passband filtered", filtered, "#276FBF")):
        frequency, power = welch(
            trace[:, 60:, :],
            fs=240.0,
            window="hann",
            nperseg=240,
            noverlap=0,
            axis=1,
            detrend="constant",
            scaling="density",
        )
        radial = power.sum(axis=-1)
        center = np.median(radial, axis=0)
        low, high = np.quantile(radial, (0.25, 0.75), axis=0)
        spectrum_axis.plot(frequency[1:], center[1:], color=color, lw=1.8, label=label)
        spectrum_axis.fill_between(frequency[1:], low[1:], high[1:], color=color, alpha=0.16)
        spectra[label] = center
    spectrum_axis.set_yscale("log")
    spectrum_axis.set(xlabel="frequency (Hz)", ylabel="position PSD (deg²/Hz)", title="C  Filtered versus raw sensitivity")
    spectrum_axis.legend(frameon=False, fontsize=8)

    distribution_axis = figure.add_subplot(grid[1, 1])
    distribution_axis.hist(table.analysis_path_length_deg * 60, bins=18, color="#3A86A8", alpha=0.85)
    distribution_axis.set(xlabel="one-second path length (arcmin)", ylabel="traces", title="D  Motion-range coverage")

    session_axis = figure.add_subplot(grid[1, 2])
    counts = table.session.value_counts().sort_values()
    session_axis.barh(np.arange(len(counts)), counts, color="#5A9E6F")
    session_axis.set_yticks(np.arange(len(counts)), counts.index.str.replace("_", " "), fontsize=6.5)
    session_axis.set(xlabel="selected traces", title=f"E  Session coverage · {len(counts)} sessions")
    figure.suptitle("M77 production fixation bank · 60-frame history + 240-frame analysis", fontsize=15, fontweight="semibold")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "m77_fixation_bank_qc.png"
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    summary = {
        "n_traces": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "median_path_length_arcmin": float(np.median(table.analysis_path_length_deg) * 60),
        "median_filtered_vs_raw_rms_arcmin": float(np.median(table.filtered_vs_raw_rms_difference_deg) * 60),
        "figure": str(path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
