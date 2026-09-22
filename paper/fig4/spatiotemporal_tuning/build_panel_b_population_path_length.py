#!/usr/bin/env python3
"""Reduce the completed Figure-4 replay to one population path-length curve.

Every continuously filtered fixation is pooled.  The estimand is computed over
all units in the exact-CID response matrix: population rate is total expected
spikes and population SSI is total spatial information divided by total
expected spikes.  No tuning or microsaccade labels enter this analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.population_response import (
    _population_arrays,
    assert_selected_matrix,
    configure,
    crossed_population_bootstrap,
    load_matrix,
    matrix_trace_filter,
    population_effects,
    quantile_bins,
)


BLUE = "#0072B2"
PURPLE = "#6A51A3"
PATH_COLUMN = "rendered_path_length_arcmin"
EPS = np.finfo(np.float64).eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def summarize(
    path_length: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    labels = quantile_bins(path_length, int(n_bins))
    reduced = _population_arrays(
        arrays, np.arange(arrays["moving_spikes"].shape[2], dtype=int)
    )
    rows: list[dict[str, object]] = []
    for bin_index in range(int(n_bins)):
        traces = np.flatnonzero(labels == bin_index)
        center, low, high = crossed_population_bootstrap(
            reduced,
            traces,
            n_bootstrap=int(n_bootstrap),
            rng=np.random.default_rng(int(seed) + 1000 * bin_index),
        )
        for outcome_index, outcome in enumerate(("rate", "SSI")):
            rows.append(
                {
                    "outcome": outcome,
                    "bin_index": int(bin_index),
                    "x_median": float(np.median(path_length[traces])),
                    "x_low": float(np.min(path_length[traces])),
                    "x_high": float(np.max(path_length[traces])),
                    "effect_percent": float(center[outcome_index]),
                    "ci_low": float(low[outcome_index]),
                    "ci_high": float(high[outcome_index]),
                    "n_traces": int(len(traces)),
                    "n_units": int(arrays["moving_spikes"].shape[2]),
                }
            )
    return pd.DataFrame(rows)


def unit_level_effects(
    path_length: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    n_bins: int,
) -> dict[str, np.ndarray]:
    """Return matched per-unit rate and SSI modulation in each path bin."""
    labels = quantile_bins(path_length, int(n_bins))
    stable_rate = np.mean(np.asarray(arrays["stable_rate"], dtype=float), axis=0)
    stable_spikes = np.asarray(arrays["stable_spikes"], dtype=float)
    stable_information = stable_spikes * np.asarray(
        arrays["stable_ssi"], dtype=float
    )
    stable_ssi = np.sum(stable_information, axis=0) / np.maximum(
        np.sum(stable_spikes, axis=0), EPS
    )
    rate_percent = np.empty((int(n_bins), len(stable_rate)), dtype=float)
    ssi_percent = np.empty_like(rate_percent)
    x_median = np.empty(int(n_bins), dtype=float)
    n_traces = np.empty(int(n_bins), dtype=int)
    for bin_index in range(int(n_bins)):
        traces = np.flatnonzero(labels == bin_index)
        if len(traces) == 0:
            raise ValueError(f"path-length bin {bin_index} is empty")
        moving_rate = np.mean(
            np.asarray(arrays["moving_rate"][:, traces], dtype=float), axis=(0, 1)
        )
        moving_spikes = np.asarray(
            arrays["moving_spikes"][:, traces], dtype=float
        )
        moving_information = moving_spikes * np.asarray(
            arrays["moving_ssi"][:, traces], dtype=float
        )
        moving_ssi = np.sum(moving_information, axis=(0, 1)) / np.maximum(
            np.sum(moving_spikes, axis=(0, 1)), EPS
        )
        rate_percent[bin_index] = (
            100.0 * (moving_rate - stable_rate) / np.maximum(stable_rate, EPS)
        )
        ssi_percent[bin_index] = (
            100.0 * (moving_ssi - stable_ssi) / np.maximum(stable_ssi, EPS)
        )
        x_median[bin_index] = float(np.median(path_length[traces]))
        n_traces[bin_index] = int(len(traces))
    if not np.isfinite(rate_percent).all() or not np.isfinite(ssi_percent).all():
        raise ValueError("unit-level Panel-B effects contain non-finite values")
    return {
        "rate_percent": rate_percent,
        "ssi_percent": ssi_percent,
        "x_median": x_median,
        "n_traces": n_traces,
        "unit_indices": np.arange(len(stable_rate), dtype=int),
    }


def summarize_unit_distributions(
    effects: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    for outcome, key in (("rate", "rate_percent"), ("SSI", "ssi_percent")):
        values = np.asarray(effects[key], dtype=float)
        for bin_index, row in enumerate(values):
            q05, q25, q50, q75, q95 = np.quantile(row, quantiles)
            rows.append(
                {
                    "outcome": outcome,
                    "bin_index": int(bin_index),
                    "x_median": float(effects["x_median"][bin_index]),
                    "q05": float(q05),
                    "q25": float(q25),
                    "median": float(q50),
                    "q75": float(q75),
                    "q95": float(q95),
                    "fraction_positive": float(np.mean(row > 0)),
                    "n_units": int(len(row)),
                }
            )
    return pd.DataFrame(rows)


def draw(
    path: Path,
    summary: pd.DataFrame,
    path_length: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    reduced = _population_arrays(
        arrays, np.arange(arrays["moving_spikes"].shape[2], dtype=int)
    )
    per_trace = np.asarray(
        [
            population_effects(reduced, np.asarray([trace_index], dtype=int))
            for trace_index in range(len(path_length))
        ],
        dtype=float,
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.1, 2.75), constrained_layout=True)
    for column, (axis, outcome, title, color) in enumerate(
        zip(
            axes,
            ("rate", "SSI"),
            ("firing-rate change", "single-spike information change"),
            (BLUE, PURPLE),
        )
    ):
        axis.scatter(
            path_length,
            per_trace[:, column],
            s=6,
            color=color,
            alpha=0.055,
            edgecolor="none",
            rasterized=True,
        )
        frame = summary.loc[summary.outcome.eq(outcome)].sort_values("bin_index")
        center = frame.effect_percent.to_numpy(dtype=float)
        axis.errorbar(
            frame.x_median,
            center,
            yerr=np.vstack(
                (
                    center - frame.ci_low.to_numpy(dtype=float),
                    frame.ci_high.to_numpy(dtype=float) - center,
                )
            ),
            fmt="o-",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=1.7,
            ms=4.5,
            capsize=2.2,
        )
        axis.axhline(0, color="0.5", lw=0.75)
        axis.set_title(title)
        axis.set_xlabel("filtered fixation path length (arcmin)")
        axis.set_ylabel("motion − stabilized (%)")
        axis.grid(axis="y", alpha=0.16)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    configure()
    if int(args.n_bins) < 3:
        raise ValueError("n-bins must be at least three")
    if int(args.n_bootstrap) < 1:
        raise ValueError("n-bootstrap must be positive")
    arrays, trace_table, matrix_summary = load_matrix(args.matrix_dir)
    model_label, checkpoint_sha256 = assert_selected_matrix(
        matrix_summary, model_spec=args.model_spec
    )
    trace_filter = matrix_trace_filter(matrix_summary)
    if trace_filter is None:
        raise ValueError("Panel B requires continuously filtered eye traces")
    if PATH_COLUMN not in trace_table:
        raise ValueError(f"trace table lacks {PATH_COLUMN!r}")
    path_length = trace_table[PATH_COLUMN].to_numpy(dtype=float)
    if not np.all(np.isfinite(path_length)):
        raise ValueError("path length contains non-finite values")
    summary = summarize(
        path_length,
        arrays,
        n_bins=int(args.n_bins),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    effects = unit_level_effects(
        path_length,
        arrays,
        n_bins=int(args.n_bins),
    )
    unit_summary = summarize_unit_distributions(effects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "binned_curves.csv", index=False)
    unit_summary.to_csv(
        args.out_dir / "unit_distribution_summary.csv", index=False
    )
    np.savez_compressed(
        args.out_dir / "unit_effects.npz",
        **effects,
    )
    figure_path = args.out_dir / "panel_b_population_path_length.png"
    draw(figure_path, summary, path_length, arrays)
    report = {
        "analysis": "pooled Figure-4 population response versus path length",
        "model_label": model_label,
        "checkpoint_sha256": checkpoint_sha256,
        "matrix_dir": str(args.matrix_dir.resolve()),
        "n_images": int(matrix_summary["n_images"]),
        "n_traces": int(matrix_summary["n_traces"]),
        "n_units": int(matrix_summary["n_units"]),
        "unit_selection": "every unit in the exact-CID response matrix",
        "fixation_selection": "all filtered fixations pooled; no microsaccade stratification",
        "motion_coordinate": PATH_COLUMN,
        "trace_filter": trace_filter,
        "n_bins": int(args.n_bins),
        "n_bootstrap": int(args.n_bootstrap),
        "estimator": "spike-weighted population; matched stabilized images and units; crossed image-by-trace-by-unit bootstrap",
        "unit_distribution": {
            "archive": str((args.out_dir / "unit_effects.npz").resolve()),
            "summary": str(
                (args.out_dir / "unit_distribution_summary.csv").resolve()
            ),
            "estimand": (
                "per-unit percent modulation in each equal-count path-length bin; "
                "rate averages matched images and traces, and SSI is pooled by each "
                "unit's expected-spike mass before comparison with its stabilized baseline"
            ),
            "display_contract": (
                "box is interquartile range, center is per-unit median, whiskers are "
                "5th and 95th percentiles, and extreme units remain in the archived data"
            ),
        },
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
