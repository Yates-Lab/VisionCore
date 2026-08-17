#!/usr/bin/env python3
"""Plot the saved preliminary true-history-only correction checkpoint."""

from __future__ import annotations

import json
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

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR


PREVIEW_DIR = OUT_DIR / "preliminary_true_only"
STYLES = {
    "legacy_wrapped_prefix": ("legacy wrapped", "#C2413B", ":", "x"),
    "real_trace_true_history_v1": ("true history", "#167D75", "-", "o"),
}


def main() -> int:
    curves = pd.read_csv(PREVIEW_DIR / "preliminary_true_only_curves.csv")
    stats = json.loads((PREVIEW_DIR / "statistics.json").read_text(encoding="utf-8"))
    checkpoint_label = (
        "COMPLETE primary true-history result"
        if stats["status"] == "COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING"
        else "PRELIMINARY true-history checkpoint"
    )
    specs = [
        ("drift_only", "low_sf_lt0p5", "A  Drift only — low SF (n=71)", "#2F6B9A"),
        ("drift_only", "high_sf_ge0p5", "B  Drift only — high SF (n=29)", "#D97706"),
        ("microsaccade", "low_sf_lt0p5", "C  Microsaccade — low SF", "#2F6B9A"),
        ("microsaccade", "high_sf_ge0p5", "D  Microsaccade — high SF", "#D97706"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8), sharey="row", constrained_layout=True)
    for ax, (context, group, title, title_color) in zip(axes.flat, specs):
        for bank, (label, color, ls, marker) in STYLES.items():
            sub = curves[
                curves["bank"].eq(bank)
                & curves["context"].eq(context)
                & curves["sf_group"].eq(group)
            ].sort_values("bin_index")
            x = sub["path_median_arcmin"].to_numpy(float)
            y = sub["ssi_percent_vs_stabilized"].to_numpy(float)
            lo = sub["ci95_low_paired_image_boot"].to_numpy(float)
            hi = sub["ci95_high_paired_image_boot"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((y - lo, hi - y)),
                color=color,
                ls=ls,
                marker=marker,
                ms=4,
                lw=1.6,
                capsize=2,
                label=label,
            )
        ax.axhline(0.0, color="#111827", lw=0.75)
        ax.set_title(title, loc="left", color=title_color, weight="bold")
        ax.set_xlabel("trajectory path length (arcmin)")
        ax.set_ylabel("SSI change vs stabilization (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{checkpoint_label} — {stats['n_images']}/100 images, all trajectories and units\n"
        f"qualitative signature: {stats['provisional_signature']} (held-prefix control pending)",
        weight="bold",
    )
    stems = ["fig_preliminary_true_only"]
    if stats["status"] == "COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING":
        stems.append("fig_true_history_primary")
    for stem in stems:
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                PREVIEW_DIR / f"{stem}.{suffix}",
                dpi=300 if suffix == "png" else None,
            )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
