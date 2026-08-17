#!/usr/bin/env python3
"""Compare first-layer temporal filters across native-240 normalization runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = ROOT / "outputs/dekel240_diagnostics"
OUT = ROOT / "outputs/dekel240_paper/training_monitor/native240_lrn"
SAMPLING_RATE = 240.0

REPORTS = {
    "M73 · GroupNorm\nepoch 235 · 0.5801 bps": DIAGNOSTICS
    / "D240M73c_dekel_native240_freqmasked_floor0p5_nostructuralreg_mlp_s201"
    / "epoch_235/first_layer_report.json",
    "M75 · pure LRN\nepoch 55 · 0.3411 bps": DIAGNOSTICS
    / "D240M75c_dekel_native240_freqmasked_floor0p5_lrn_s201"
    / "epoch_055/first_layer_report.json",
    "M76 · GroupNorm→LRN\nepoch 207 · 0.5741 bps": DIAGNOSTICS
    / "D240M76c_dekel_native240_freqmasked_floor0p5_historical_gnlrn_presplit_s201"
    / "epoch_207/first_layer_report.json",
    "M77 · GroupNorm→LRN α=.1\nepoch 127 · 0.5501 bps": DIAGNOSTICS
    / "D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201"
    / "epoch_127/first_layer_report.json",
}


def prepare(report_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text())
    temporal = np.asarray(report["temporal_components"], dtype=float)
    temporal = temporal / np.maximum(np.max(np.abs(temporal), axis=1, keepdims=True), 1e-12)
    sign = np.sign(temporal[np.arange(len(temporal)), np.argmax(np.abs(temporal), axis=1)])
    temporal = temporal * sign[:, None]
    spectrum = np.abs(np.fft.rfft(temporal, axis=1)) ** 2
    spectrum = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-12)
    frequency = np.fft.rfftfreq(temporal.shape[1], d=1.0 / SAMPLING_RATE)
    centroid = (spectrum * frequency[None]).sum(axis=1)
    order = np.argsort(centroid)
    return report, temporal[order], spectrum[order], frequency


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        len(REPORTS),
        3,
        figsize=(10.8, 9.6),
        gridspec_kw={"width_ratios": [1.35, 1.35, 1.1]},
        constrained_layout=True,
    )
    summary = {}
    for row, (label, path) in enumerate(REPORTS.items()):
        report, temporal, spectrum, frequency = prepare(path)
        time_ms = np.arange(temporal.shape[1]) * 1000.0 / SAMPLING_RATE
        axes[row, 0].imshow(
            temporal,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            extent=[time_ms[0], time_ms[-1], temporal.shape[0] - 0.5, -0.5],
        )
        axes[row, 1].imshow(
            spectrum,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=np.quantile(spectrum, 0.96),
            extent=[frequency[0], frequency[-1], spectrum.shape[0] - 0.5, -0.5],
        )
        median = np.median(spectrum, axis=0)
        lo, hi = np.quantile(spectrum, [0.1, 0.9], axis=0)
        axes[row, 2].fill_between(frequency, lo, hi, color="#8AA8B8", alpha=0.25, lw=0)
        axes[row, 2].plot(frequency, median, color="#245F75", lw=1.8)
        axes[row, 2].axvline(60, color="#C75B39", ls=(0, (3, 2)), lw=0.9)
        axes[row, 2].set_xlim(0, 120)
        axes[row, 2].set_ylim(bottom=0)
        axes[row, 0].set_ylabel(f"{label}\nfilter (slow → fast)")
        axes[row, 1].set_yticks([])
        metrics = (
            f"rank-1 {report['rank1_fraction_mean']:.2f}\n"
            f">60 Hz {100 * report['temporal_high_frequency_fraction_mean']:.2f}%\n"
            f"high-SF {100 * report['spatial_high_frequency_fraction_mean']:.2f}%"
        )
        axes[row, 2].text(
            0.98,
            0.95,
            metrics,
            transform=axes[row, 2].transAxes,
            va="top",
            ha="right",
            fontsize=8,
            color="#31383E",
        )
        summary[label] = {
            "report": str(path),
            "rank1_fraction_mean": report["rank1_fraction_mean"],
            "temporal_high_frequency_fraction_mean": report[
                "temporal_high_frequency_fraction_mean"
            ],
            "spatial_high_frequency_fraction_mean": report[
                "spatial_high_frequency_fraction_mean"
            ],
        }

    axes[0, 0].set_title("normalized temporal profiles", fontsize=11, weight="bold")
    axes[0, 1].set_title("per-filter temporal power", fontsize=11, weight="bold")
    axes[0, 2].set_title("median and 10–90% range", fontsize=11, weight="bold")
    for axis in axes[-1, :2]:
        axis.set_xlabel("history position (ms)" if axis is axes[-1, 0] else "frequency (Hz)")
    axes[-1, 2].set_xlabel("frequency (Hz)")
    for axis in axes[:, 2]:
        axis.set_ylabel("power fraction")
        axis.grid(axis="y", color="#E4E8EB", lw=0.6)
    fig.suptitle(
        "Native-240 first layers: moderate LRN retains organized low-TF filters",
        fontsize=14,
        weight="bold",
    )
    path = OUT / "native240_normalization_filter_comparison.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    (OUT / "native240_normalization_filter_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(path)


if __name__ == "__main__":
    main()
