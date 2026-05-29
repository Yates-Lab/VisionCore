"""Eye-movement selection and trace visualization.

This is step 1 of the production pipeline.  It selects real eye-trace windows
with either no detected microsaccades or exactly one detected microsaccade, and
writes atlas figures that make that classification auditable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .common import DT, extract_fixrsvp_eye_traces, load_digital_twin
from .io_utils import write_csv
from .retinal_examples import TraceExample, select_trace_examples, trace_example_row


def plot_trace_selection_atlas(
    rows: list[dict[str, Any]],
    traces: dict[str, np.ndarray],
    path: Path,
    *,
    title: str,
) -> None:
    """Plot x/y position, 2D path, and speed for selected trace windows."""
    n = len(rows)
    fig, axs = plt.subplots(
        n,
        3,
        figsize=(13.0, max(4.5, 2.25 * n)),
        squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.28},
    )
    fig.subplots_adjust(top=0.965)
    for r, row in enumerate(rows):
        trace = traces[row["example_id"]]
        t = np.arange(trace.shape[0]) * DT
        speed = np.linalg.norm(np.diff(trace, axis=0, prepend=trace[:1]), axis=1) / DT
        color = "#4c78a8" if row["kind"] == "fixation" else "#f58518"

        axs[r, 0].plot(t, trace[:, 0] * 60.0, lw=1.0, label="x")
        axs[r, 0].plot(t, trace[:, 1] * 60.0, lw=1.0, label="y")
        if row["event_onset"] is not None and np.isfinite(row["event_onset"]):
            axs[r, 0].axvline(t[int(row["event_onset"])], color="#111111", lw=1.0, ls="--")
        axs[r, 0].text(
            0.01,
            0.94,
            f"{row['example_id']} | events={row['n_events_in_window']}",
            transform=axs[r, 0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
        )
        axs[r, 0].set_ylabel("pos\narcmin", fontsize=8)

        axs[r, 1].plot(trace[:, 0] * 60.0, trace[:, 1] * 60.0, lw=0.9)
        axs[r, 1].scatter(trace[0, 0] * 60.0, trace[0, 1] * 60.0, s=12, color="green")
        axs[r, 1].scatter(trace[-1, 0] * 60.0, trace[-1, 1] * 60.0, s=12, color="red")
        axs[r, 1].set_aspect("equal", adjustable="datalim")

        axs[r, 2].plot(t, speed, color="0.25", lw=1.0)
        axs[r, 2].axhline(row["threshold_deg_s"], color="#d62728", lw=0.8, ls=":")
        if row["event_onset"] is not None and np.isfinite(row["event_onset"]):
            axs[r, 2].axvline(t[int(row["event_onset"])], color="#111111", lw=1.0, ls="--")
        axs[r, 2].set_ylabel("speed\ndeg/s", fontsize=8)

        if r == 0:
            axs[r, 0].set_title("x/y trace", fontsize=10)
            axs[r, 1].set_title("2D path", fontsize=10)
            axs[r, 2].set_title("speed + threshold", fontsize=10)
        if r == n - 1:
            axs[r, 0].set_xlabel("time (s)")
            axs[r, 1].set_xlabel("x pos (arcmin)")
            axs[r, 2].set_xlabel("time (s)")
        for ax in axs[r]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.grid(color="0.92", lw=0.6)
    fig.suptitle(title, y=0.992, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_trace_selection_step(
    *,
    figure_dir: Path,
    metadata_dir: Path,
    seed: int,
    n_examples_per_kind: int,
    t_max: int,
    stride: int,
    model: Any | None = None,
) -> list[TraceExample]:
    """Select traces from the real dataset and write trace QC figures."""
    if model is None:
        model, _info, _device = load_digital_twin()
    eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=t_max)
    examples = select_trace_examples(
        eye_traces,
        durations,
        t_max=t_max,
        n_each=n_examples_per_kind,
        seed=seed,
        stride=stride,
    )
    rows = [trace_example_row(example) for example in examples]
    traces = {example.example_id: example.trace for example in examples}
    write_csv(rows, metadata_dir / "01_trace_examples.csv")
    plot_trace_selection_atlas(
        rows,
        traces,
        figure_dir / "01_eye_trace_selection_atlas.pdf",
        title="Selected real eye traces: fixation-only and one-microsaccade windows",
    )
    for kind in ("fixation", "microsaccade"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        plot_trace_selection_atlas(
            kind_rows,
            traces,
            figure_dir / f"01_{kind}_trace_selection.pdf",
            title=f"Selected {kind} windows",
        )
    return examples
