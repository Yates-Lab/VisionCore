#!/usr/bin/env python3
"""Test whether M66's FixRSVP deficit is a temporal phase/lag error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RYAN = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
DEFAULT_M66 = ROOT / "outputs/dekel240_paper/final/cache_fig3/fig3_digitaltwin.pkl"
DEFAULT_OUTPUT = ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_temporal_alignment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ryan", type=Path, default=DEFAULT_RYAN)
    parser.add_argument("--m66", type=Path, default=DEFAULT_M66)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-trials", type=int, default=20)
    parser.add_argument("--maximum-shift-bins", type=float, default=3.0)
    parser.add_argument("--shift-step-bins", type=float, default=0.25)
    return parser.parse_args()


def load_sessions(path: Path) -> dict[str, dict]:
    with path.open("rb") as stream:
        rows = dill.load(stream)
    return {str(row["session"]): row for row in rows}


def pearson(left, right, minimum=20) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < minimum:
        return np.nan
    left = left[valid]
    right = right[valid]
    if np.var(left) <= 0 or np.var(right) <= 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def shifted_correlation(data, prediction, eligible, shift_bins) -> float:
    """Correlate data[t] with linearly interpolated prediction[t + shift].

    Positive shifts therefore mean that the model response occurs later than
    the data response (the model lags the data).
    """
    data = np.asarray(data, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    time = np.arange(len(data), dtype=float)
    source_valid = np.isfinite(prediction)
    if source_valid.sum() < 2:
        return np.nan
    sample_at = time + float(shift_bins)
    in_bounds = (
        eligible
        & np.isfinite(data)
        & (sample_at >= time[source_valid].min())
        & (sample_at <= time[source_valid].max())
    )
    shifted = np.full_like(data, np.nan, dtype=float)
    shifted[in_bounds] = np.interp(
        sample_at[in_bounds], time[source_valid], prediction[source_valid]
    )
    return pearson(data, shifted)


def finite_summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else None,
        "q25": float(np.quantile(values, 0.25)) if len(values) else None,
        "q75": float(np.quantile(values, 0.75)) if len(values) else None,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ryan = load_sessions(args.ryan.resolve())
    m66 = load_sessions(args.m66.resolve())
    if set(ryan) != set(m66):
        raise RuntimeError("Session sets differ")

    shifts = np.arange(
        -args.maximum_shift_bins,
        args.maximum_shift_bins + args.shift_step_bins / 2,
        args.shift_step_bins,
    )
    ryan_curves = []
    m66_curves = []
    rows = []
    for session in sorted(ryan):
        left = ryan[session]
        right = m66[session]
        units = np.asarray(left["neuron_mask"], dtype=int)
        if not np.array_equal(units, np.asarray(right["neuron_mask"], dtype=int)):
            raise RuntimeError(f"Neuron masks differ for {session}")
        robs = np.asarray(left["robs_used"], dtype=float)
        dfs = np.asarray(left["dfs_used"], dtype=float)
        data_valid = np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)
        n_valid = data_valid.sum(axis=0)
        data_mean = np.asarray(left["robs_mean"], dtype=float)
        ryan_mean = np.asarray(left["rhat_mean"], dtype=float)
        m66_mean = np.asarray(right["rhat_mean"], dtype=float)
        for local, source_unit in enumerate(units):
            eligible = n_valid[:, local] >= args.minimum_trials
            curve_ryan = np.asarray(
                [
                    shifted_correlation(
                        data_mean[:, local], ryan_mean[:, local], eligible, shift
                    )
                    for shift in shifts
                ]
            )
            curve_m66 = np.asarray(
                [
                    shifted_correlation(
                        data_mean[:, local], m66_mean[:, local], eligible, shift
                    )
                    for shift in shifts
                ]
            )
            zero_index = int(np.argmin(np.abs(shifts)))
            if not np.isfinite(curve_ryan[zero_index]) or not np.isfinite(
                curve_m66[zero_index]
            ):
                continue
            best_ryan = int(np.nanargmax(curve_ryan))
            best_m66 = int(np.nanargmax(curve_m66))
            common = eligible & np.isfinite(ryan_mean[:, local]) & np.isfinite(
                m66_mean[:, local]
            )
            valid_indices = np.flatnonzero(common)
            top_overlap_ryan = np.nan
            top_overlap_m66 = np.nan
            if len(valid_indices) >= 20:
                n_top = max(3, int(np.ceil(0.10 * len(valid_indices))))
                data_values = data_mean[valid_indices, local]
                ryan_values = ryan_mean[valid_indices, local]
                m66_values = m66_mean[valid_indices, local]
                data_top = set(valid_indices[np.argpartition(data_values, -n_top)[-n_top:]])
                ryan_top = set(valid_indices[np.argpartition(ryan_values, -n_top)[-n_top:]])
                m66_top = set(valid_indices[np.argpartition(m66_values, -n_top)[-n_top:]])
                top_overlap_ryan = len(data_top & ryan_top) / n_top
                top_overlap_m66 = len(data_top & m66_top) / n_top
            rows.append(
                {
                    "session": session,
                    "subject": session.split("_")[0],
                    "source_unit_index": int(source_unit),
                    "ryan_zero_lag": float(curve_ryan[zero_index]),
                    "m66_zero_lag": float(curve_m66[zero_index]),
                    "ryan_best_shift_bins": float(shifts[best_ryan]),
                    "m66_best_shift_bins": float(shifts[best_m66]),
                    "ryan_best_lag_gain": float(
                        curve_ryan[best_ryan] - curve_ryan[zero_index]
                    ),
                    "m66_best_lag_gain": float(
                        curve_m66[best_m66] - curve_m66[zero_index]
                    ),
                    "ryan_m66_prediction_correlation": pearson(
                        ryan_mean[:, local][common], m66_mean[:, local][common]
                    ),
                    "ryan_top10_overlap": top_overlap_ryan,
                    "m66_top10_overlap": top_overlap_m66,
                }
            )
            ryan_curves.append(curve_ryan)
            m66_curves.append(curve_m66)

    import pandas as pd

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "paired_temporal_alignment.csv", index=False)
    ryan_curves = np.asarray(ryan_curves)
    m66_curves = np.asarray(m66_curves)
    ryan_median_curve = np.nanmedian(ryan_curves, axis=0)
    m66_median_curve = np.nanmedian(m66_curves, axis=0)
    ryan_global_best = int(np.nanargmax(ryan_median_curve))
    m66_global_best = int(np.nanargmax(m66_median_curve))

    session_best = []
    for session, indices in table.groupby("session").groups.items():
        indices = np.asarray(list(indices), dtype=int)
        ryan_curve = np.nanmedian(ryan_curves[indices], axis=0)
        m66_curve = np.nanmedian(m66_curves[indices], axis=0)
        session_best.append(
            {
                "session": session,
                "n_units": int(len(indices)),
                "ryan_best_shift_bins": float(shifts[np.nanargmax(ryan_curve)]),
                "m66_best_shift_bins": float(shifts[np.nanargmax(m66_curve)]),
                "ryan_zero_lag": float(ryan_curve[np.argmin(np.abs(shifts))]),
                "m66_zero_lag": float(m66_curve[np.argmin(np.abs(shifts))]),
                "m66_best_global_lag_value": float(np.nanmax(m66_curve)),
            }
        )
    pd.DataFrame(session_best).to_csv(args.output_dir / "session_temporal_alignment.csv", index=False)

    report = {
        "shift_definition": (
            "corr(data[t], prediction[t+shift]); positive means model lags data"
        ),
        "bin_duration_ms": 1000 / 120,
        "native_m66_frame_duration_ms": 1000 / 240,
        "n_units": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "global_lag_scan": {
            "shift_bins": shifts.tolist(),
            "shift_ms": (1000 * shifts / 120).tolist(),
            "ryan_median_correlation": ryan_median_curve.tolist(),
            "m66_median_correlation": m66_median_curve.tolist(),
            "ryan_best_shift_bins": float(shifts[ryan_global_best]),
            "m66_best_shift_bins": float(shifts[m66_global_best]),
            "ryan_best_vs_zero_gain": float(
                ryan_median_curve[ryan_global_best]
                - ryan_median_curve[np.argmin(np.abs(shifts))]
            ),
            "m66_best_vs_zero_gain": float(
                m66_median_curve[m66_global_best]
                - m66_median_curve[np.argmin(np.abs(shifts))]
            ),
        },
        "unit_specific": {
            "ryan_best_shift_bins": finite_summary(table.ryan_best_shift_bins),
            "m66_best_shift_bins": finite_summary(table.m66_best_shift_bins),
            "ryan_best_lag_gain": finite_summary(table.ryan_best_lag_gain),
            "m66_best_lag_gain": finite_summary(table.m66_best_lag_gain),
            "ryan_fraction_best_within_half_bin": float(
                np.mean(np.abs(table.ryan_best_shift_bins) <= 0.5)
            ),
            "m66_fraction_best_within_half_bin": float(
                np.mean(np.abs(table.m66_best_shift_bins) <= 0.5)
            ),
        },
        "event_identity": {
            "ryan_m66_prediction_correlation": finite_summary(
                table.ryan_m66_prediction_correlation
            ),
            "ryan_top10_time_bin_overlap_with_data": finite_summary(
                table.ryan_top10_overlap
            ),
            "m66_top10_time_bin_overlap_with_data": finite_summary(
                table.m66_top10_overlap
            ),
            "chance_top10_overlap": 0.10,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45), constrained_layout=True)
    shift_ms = 1000 * shifts / 120
    axes[0].plot(shift_ms, ryan_median_curve, label="Ryan", color="#d06a2e")
    axes[0].plot(shift_ms, m66_median_curve, label="M66", color="#2d6fa3")
    axes[0].axvline(0, color="0.55", linewidth=1)
    axes[0].axvline(1000 / 240, color="0.7", linewidth=0.8, linestyle="--")
    axes[0].axvline(-1000 / 240, color="0.7", linewidth=0.8, linestyle="--")
    axes[0].set(
        title="A  Population lag scan",
        xlabel="model lag relative to data (ms)",
        ylabel="median PSTH correlation",
    )
    axes[0].legend(frameon=False)

    bins = np.arange(-3.125, 3.126, 0.25)
    axes[1].hist(
        table.ryan_best_shift_bins,
        bins=bins,
        alpha=0.55,
        label="Ryan",
        color="#d06a2e",
    )
    axes[1].hist(
        table.m66_best_shift_bins,
        bins=bins,
        alpha=0.55,
        label="M66",
        color="#2d6fa3",
    )
    axes[1].axvline(0, color="0.55", linewidth=1)
    axes[1].set(
        title="B  Best shift for each neuron",
        xlabel="model lag relative to data (120-Hz bins)",
        ylabel="neurons",
    )
    axes[1].legend(frameon=False)
    stem = args.output_dir / "temporal_lag_scan"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
