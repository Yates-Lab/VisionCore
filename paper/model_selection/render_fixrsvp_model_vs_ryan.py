#!/usr/bin/env python3
"""Render a Figure-3 FixRSVP comparison for a selected twin and Ryan's twin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RYAN = ROOT / "outputs/cache/fig3_digitaltwin.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ryan-cache", type=Path, default=DEFAULT_RYAN)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", default="Allen_2022-02-16")
    parser.add_argument("--unit-index", type=int, default=105)
    return parser.parse_args()


def load_sessions(path: Path) -> dict[str, dict]:
    """Load either a dill/pickle cache or the evaluator's NumPy-saved list."""
    try:
        with path.open("rb") as stream:
            values = dill.load(stream)
    except Exception:
        values = np.load(path, allow_pickle=True)
        if isinstance(values, np.ndarray):
            values = values.tolist()
    return {str(value["session"]): value for value in values}


def aligned_table(ryan: dict[str, dict], model: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for session in sorted(set(ryan).intersection(model)):
        left, right = ryan[session], model[session]
        left_units = np.asarray(left["neuron_mask"], dtype=int)
        right_units = np.asarray(right["neuron_mask"], dtype=int)
        if not np.array_equal(left_units, right_units):
            raise RuntimeError(f"Neuron masks differ for {session}")
        if np.asarray(left["robs_used"]).shape != np.asarray(right["robs_used"]).shape:
            raise RuntimeError(f"Observation shapes differ for {session}")
        for local, source_unit in enumerate(left_units):
            rows.append({
                "session": session,
                "subject": session.split("_")[0],
                "source_unit_index": int(source_unit),
                "ryan_rho": float(np.asarray(left["rhos"])[local]),
                "model_rho": float(np.asarray(right["rhos"])[local]),
                "ryan_ccnorm": float(np.asarray(left["ccnorm"])[local]),
                "model_ccnorm": float(np.asarray(right["ccnorm"])[local]),
                "ccmax": float(np.asarray(left["ccmax"])[local]),
                "ryan_variance_explained": float(np.asarray(left["ve_model"])[local]),
                "model_variance_explained": float(np.asarray(right["ve_model"])[local]),
            })
    return pd.DataFrame(rows)


def finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else None,
        "q25": float(np.quantile(values, 0.25)) if len(values) else None,
        "q75": float(np.quantile(values, 0.75)) if len(values) else None,
    }


def scatter_comparison(
    ax,
    table: pd.DataFrame,
    left: str,
    right: str,
    label: str,
    model_label: str,
) -> None:
    values = table[[left, right]].to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    lo = float(min(-0.05, np.quantile(values, 0.01)))
    hi = float(max(0.1, np.quantile(values, 0.99)))
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=1, zorder=0)
    ax.scatter(values[:, 0], values[:, 1], s=7, alpha=0.22, color="#2d6fa3", edgecolors="none")
    ax.scatter(
        [np.median(values[:, 0])], [np.median(values[:, 1])],
        marker="D", s=48, color="#b8412d", edgecolors="white", linewidths=0.7,
        zorder=3,
    )
    ax.set(
        xlabel=f"Ryan twin {label}",
        ylabel=f"{model_label} {label}",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    ax.text(
        0.03, 0.97,
        f"n={len(values):,}\nmedian {np.median(values[:, 0]):.3f} → {np.median(values[:, 1]):.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )


def exemplar(
    ax,
    ryan: dict[str, dict],
    model: dict[str, dict],
    session: str,
    unit_index: int,
    model_label: str,
) -> dict:
    left, right = ryan[session], model[session]
    units = np.asarray(left["neuron_mask"], dtype=int)
    match = np.flatnonzero(units == int(unit_index))
    if not len(match):
        raise ValueError(f"Unit {unit_index} is not in the Figure-3 population for {session}")
    local = int(match[0])
    observed = np.asarray(left["robs_mean"])[:, local]
    ryan_prediction = np.asarray(left["rhat_mean"])[:, local]
    model_prediction = np.asarray(right["rhat_mean"])[:, local]
    valid = np.isfinite(observed) & np.isfinite(ryan_prediction) & np.isfinite(model_prediction)
    time_ms = 1000 * np.arange(len(observed)) / 120.0
    ax.plot(time_ms[valid], observed[valid], color="0.15", lw=1.8, label="data")
    ax.plot(time_ms[valid], ryan_prediction[valid], color="#d06a2e", lw=1.3, label="Ryan")
    ax.plot(time_ms[valid], model_prediction[valid], color="#2d6fa3", lw=1.3, label=model_label)
    ax.set(xlabel="time in FixRSVP trial (ms)", ylabel="mean spikes / 120-Hz bin")
    ax.set_title(f"{session}, unit {unit_index}", loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=3)
    return {
        "session": session,
        "source_unit_index": int(unit_index),
        "ryan_rho": float(np.asarray(left["rhos"])[local]),
        "model_rho": float(np.asarray(right["rhos"])[local]),
        "ryan_ccnorm": float(np.asarray(left["ccnorm"])[local]),
        "model_ccnorm": float(np.asarray(right["ccnorm"])[local]),
        "ryan_variance_explained": float(np.asarray(left["ve_model"])[local]),
        "model_variance_explained": float(np.asarray(right["ve_model"])[local]),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ryan = load_sessions(args.ryan_cache)
    model = load_sessions(args.model_cache)
    table = aligned_table(ryan, model)
    table.to_csv(args.output_dir / "paired_unit_metrics.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(10.5, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.55), wspace=0.42)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    scatter_comparison(
        axes[0], table, "ryan_rho", "model_rho", "PSTH correlation", args.model_label,
    )
    scatter_comparison(
        axes[1], table,
        "ryan_variance_explained", "model_variance_explained",
        "single-trial variance explained", args.model_label,
    )
    exemplar_metrics = exemplar(
        axes[2], ryan, model, args.session, args.unit_index, args.model_label,
    )
    fig.tight_layout()
    stem = f"{args.model_label.lower()}_vs_ryan_fixrsvp"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=240)
    fig.savefig(args.output_dir / f"{stem}.pdf")
    plt.close(fig)

    metric_names = (
        "ryan_rho", "model_rho", "ryan_ccnorm", "model_ccnorm",
        "ryan_variance_explained", "model_variance_explained",
    )
    report = {
        "ryan_cache": str(args.ryan_cache.resolve()),
        "model_cache": str(args.model_cache.resolve()),
        "model_label": args.model_label,
        "n_sessions": int(table.session.nunique()),
        "n_cells": int(len(table)),
        "all_cells": {
            name: finite_summary(table[name].to_numpy()) for name in metric_names
        },
        "reliable_cells": {
            name: finite_summary(table.loc[table.ccmax > 0.85, name].to_numpy())
            for name in metric_names
        },
        "exemplar": exemplar_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
