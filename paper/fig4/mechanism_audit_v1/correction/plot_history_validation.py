#!/usr/bin/env python3
"""Plot predetermined examples validating true, held, and legacy histories."""

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

from paper.fig4.mechanism_audit_v1.correction.common import (
    DT_S,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    N_SCORED,
    OUT_DIR,
)


TRUE_COLOR = "#1B7F79"
HELD_COLOR = "#6B7280"
LEGACY_COLOR = "#C2413B"


def predetermined_ids(legacy: np.ndarray) -> np.ndarray:
    """Nearest members to fixed wrap-speed quantiles; selected before plotting."""
    speed = np.linalg.norm(legacy[:, 31] - legacy[:, 0], axis=1) / DT_S
    order = np.argsort(speed, kind="mergesort")
    targets = np.linspace(0.1, 0.9, 5)
    indices = np.clip(np.rint(targets * (len(order) - 1)).astype(int), 0, len(order) - 1)
    return order[indices]


def main() -> int:
    (OUT_DIR / "plot_data").mkdir(parents=True, exist_ok=True)
    bank_path = OUT_DIR / "banks/corrected_history_trajectory_banks.npz"
    with np.load(bank_path) as archive:
        true_xy = np.asarray(archive["true_history_xy"])
        held_xy = np.asarray(archive["held_initial_history_xy"])
        legacy = np.asarray(archive["stored_scored_trace_xy"])
    ids = predetermined_ids(legacy)
    rows: list[dict[str, float | int]] = []
    for trajectory_id in ids:
        wrap_speed = float(np.linalg.norm(legacy[trajectory_id, 31] - legacy[trajectory_id, 0]) / DT_S)
        rows.append({"trajectory_id": int(trajectory_id), "legacy_wrap_speed_deg_s": wrap_speed})
    pd.DataFrame(rows).to_csv(OUT_DIR / "plot_data/history_validation_examples.csv", index=False)

    output_samples = np.asarray([0, 19, 39], dtype=int)
    causal_windows = pd.DataFrame(
        {
            "output_sample": output_samples,
            "output_time_ms": output_samples * DT_S * 1000.0,
            "oldest_input_sample_relative_to_scored_onset": output_samples - N_PRECEDING,
            "oldest_input_time_ms": (output_samples - N_PRECEDING) * DT_S * 1000.0,
            "newest_input_sample_relative_to_scored_onset": output_samples,
        }
    )
    causal_windows.to_csv(OUT_DIR / "plot_data/history_validation_causal_windows.csv", index=False)

    time_corrected_ms = np.arange(-N_PRECEDING, N_SCORED) * DT_S * 1000.0
    legacy_sequence = np.concatenate((legacy[:, :32], legacy), axis=1)
    # The duplicated scored e[0] is legacy sequence index 32 and defines time 0.
    time_legacy_ms = np.arange(-(N_PRECEDING + 1), N_SCORED) * DT_S * 1000.0
    fig, axes = plt.subplots(len(ids), 3, figsize=(11.2, 8.2), sharex="col", constrained_layout=True)
    for row_index, trajectory_id in enumerate(ids):
        wrap_speed = rows[row_index]["legacy_wrap_speed_deg_s"]
        for col, (data, time_ms, color, title) in enumerate(
            (
                (true_xy[trajectory_id], time_corrected_ms, TRUE_COLOR, "True source history"),
                (held_xy[trajectory_id], time_corrected_ms, HELD_COLOR, "Held-initial control"),
                (legacy_sequence[trajectory_id], time_legacy_ms, LEGACY_COLOR, "Legacy wrapped prefix"),
            )
        ):
            ax = axes[row_index, col]
            ax.axvspan(0, 325, color="#E5E7EB", alpha=0.55, lw=0)
            ax.plot(time_ms, data[:, 0] * 60.0, color=color, lw=1.15, label="x")
            ax.plot(time_ms, data[:, 1] * 60.0, color=color, lw=1.15, ls="--", label="y")
            ax.axvline(0, color="#111827", lw=0.75)
            if col == 2:
                # The copied prefix ends between legacy sequence samples 31 and 32.
                ax.scatter([0], [data[32, 0] * 60.0], s=18, facecolor="white", edgecolor=LEGACY_COLOR, zorder=3)
                ax.annotate(
                    "artificial wrap",
                    xy=(0, data[32, 0] * 60.0),
                    xytext=(18, 13),
                    textcoords="offset points",
                    fontsize=7,
                    color=LEGACY_COLOR,
                    arrowprops={"arrowstyle": "->", "color": LEGACY_COLOR, "lw": 0.7},
                )
            if row_index == 0:
                ax.set_title(title, fontsize=10, color=color, weight="bold")
                # Explicitly show the 32 retinal samples supplied to early,
                # middle, and late scored outputs.  The time spans are the
                # same, while the legacy panel makes clear that the content
                # inside its early/middle windows is wrapped/noncausal.
                for bracket_index, window in causal_windows.iterrows():
                    start = float(window["oldest_input_time_ms"])
                    stop = float(window["output_time_ms"])
                    y = 0.88 - 0.10 * bracket_index
                    transform = ax.get_xaxis_transform()
                    ax.plot([start, stop], [y, y], color="#111827", lw=0.65, transform=transform)
                    ax.plot([start, start], [y - 0.018, y + 0.018], color="#111827", lw=0.65, transform=transform)
                    ax.plot([stop, stop], [y - 0.018, y + 0.018], color="#111827", lw=0.65, transform=transform)
                    ax.text(
                        stop,
                        y + 0.012,
                        ("early", "middle", "late")[bracket_index],
                        transform=transform,
                        ha="right",
                        va="bottom",
                        fontsize=5.8,
                        color="#111827",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3},
                    )
            if col == 0:
                ax.set_ylabel(f"trace {trajectory_id}\neye pos. (arcmin)", fontsize=8)
            if row_index == len(ids) - 1:
                ax.set_xlabel("time from scored onset (ms)")
            ax.text(
                0.98,
                0.05,
                f"legacy wrap {wrap_speed:.1f} deg/s",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#374151",
            )
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(
        "Causal-history correction: five predeclared wrap-speed quantiles",
        fontsize=13,
        weight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_history_validation.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
