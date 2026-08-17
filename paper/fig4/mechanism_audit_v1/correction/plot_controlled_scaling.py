#!/usr/bin/env python3
"""Plot corrected controlled scaling entirely from saved CSV outputs."""

from __future__ import annotations

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

from paper.fig4.mechanism_audit_v1.correction.common import CONTROLLED_DIR


LOW = "#2F6B9A"
HIGH = "#D97706"
GROUPS = {
    "low_sf_lt0p5": ("low SF (n=71)", LOW, "o"),
    "high_sf_ge0p5": ("high SF (n=29)", HIGH, "s"),
}
BANKS = {
    "real_trace_true_history_v1": "Real preceding history",
    "real_trace_held_initial_history_v1": "Held-initial prehistory",
}


def main() -> int:
    data = pd.read_csv(CONTROLLED_DIR / "corrected_controlled_scaling_curves.csv")
    plot_dir = CONTROLLED_DIR / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(plot_dir / "fig_controlled_scaling_corrected.csv", index=False)

    y_values = np.concatenate(
        (
            data["ci95_low_units_images_trajectories_boot"].to_numpy(dtype=float),
            data["ci95_high_units_images_trajectories_boot"].to_numpy(dtype=float),
            np.asarray([0.0]),
        )
    )
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    span = max(y_max - y_min, 1.0)
    ylim = (y_min - 0.10 * span, y_max + 0.16 * span)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharex=True, sharey=True, constrained_layout=True)
    for ax, (bank, title) in zip(axes, BANKS.items()):
        for group, (label, color, marker) in GROUPS.items():
            sub = data[data["bank"].eq(bank) & data["sf_group"].eq(group)].sort_values(
                "trajectory_amplitude_x"
            )
            x = sub["trajectory_amplitude_x"].to_numpy(dtype=float)
            y = sub["ssi_percent_vs_bank_scale0"].to_numpy(dtype=float)
            lo = sub["ci95_low_units_images_trajectories_boot"].to_numpy(dtype=float)
            hi = sub["ci95_high_units_images_trajectories_boot"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lo, hi - y)),
                color=color,
                marker=marker,
                ms=5,
                lw=1.8,
                capsize=2.5,
            )
            ax.text(x[-1] + 0.05, y[-1], label, color=color, va="center", fontsize=8.5)
        ax.axhline(0.0, color="#111827", lw=0.75)
        ax.axvline(1.0, color="#4B5563", lw=0.9, ls=":")
        ax.text(1.0, ylim[1] - 0.035 * (ylim[1] - ylim[0]), "measured 1×", ha="center", va="top", fontsize=8)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("retinal trajectory amplitude (× measured trajectory)")
        ax.set_xlim(-0.08, 3.65)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SSI change vs matched 0× (%)")
    fig.suptitle(
        "Does varying only within-window FEM amplitude reproduce the movement-scale interaction?",
        weight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            CONTROLLED_DIR / f"fig_controlled_scaling_corrected.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
