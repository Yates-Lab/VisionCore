#!/usr/bin/env python3
"""Analyze the compact step-and-hold and constant-velocity pilot."""

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

from paper.fig4.mechanism_audit_v1.correction.common import LEGACY_MATRIX_DIR, OUT_DIR, write_json


FIRST = OUT_DIR / "first_pass_v1"
OUT = FIRST / "synthetic_history_pilot"
DATA = FIRST / "plot_data"
LOW = "#007C83"
HIGH = "#D55E00"
EPS = 1e-8


def contributions_by_image(ssi: np.ndarray, expected: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.sum(ssi[..., units] * expected[..., units], axis=(-2, -1)),
        np.sum(expected[..., units], axis=(-2, -1)),
    )


def bootstrap_delta(point: tuple[np.ndarray, np.ndarray], baseline: tuple[np.ndarray, np.ndarray], seed: int) -> tuple[float, float, float]:
    point_value = point[0].sum() / max(point[1].sum(), EPS)
    baseline_value = baseline[0].sum() / max(baseline[1].sum(), EPS)
    point_est = 100 * (point_value - baseline_value) / baseline_value
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(point[0]), size=(10000, len(point[0])))
    boot_p = point[0][ids].sum(axis=1) / np.maximum(point[1][ids].sum(axis=1), EPS)
    boot_b = baseline[0][ids].sum(axis=1) / np.maximum(baseline[1][ids].sum(axis=1), EPS)
    boot = 100 * (boot_p - boot_b) / np.maximum(boot_b, EPS)
    return float(point_est), *[float(x) for x in np.percentile(boot, [2.5, 97.5])]


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    with np.load(OUT / "synthetic_history_response.npz") as archive:
        step_ssi = np.asarray(archive["step_ssi"], float)
        step_expected = np.asarray(archive["step_expected_spikes"], float)
        velocity_ssi = np.asarray(archive["velocity_ssi"], float)
        velocity_expected = np.asarray(archive["velocity_expected_spikes"], float)
        amplitudes = np.asarray(archive["amplitudes_deg"], float)
        lags = np.asarray(archive["step_lag_frames"], int)
        velocities = np.asarray(archive["velocities_deg_s"], float)
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(unit.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low SF": np.flatnonzero(sf < .5), "high SF": np.flatnonzero(sf >= .5)}
    rows = []
    velocity_rows = []
    for group_index, (group, units) in enumerate(groups.items()):
        step_num, step_den = contributions_by_image(step_ssi, step_expected, units)
        # Contributions leave image × amplitude × lag; directions/units are pooled.
        for amplitude_index, amplitude in enumerate(amplitudes):
            for lag_index, lag in enumerate(lags):
                point = (step_num[:, amplitude_index, lag_index], step_den[:, amplitude_index, lag_index])
                baseline = (step_num[:, 0, lag_index], step_den[:, 0, lag_index])
                value, low, high = bootstrap_delta(point, baseline, 10100 + group_index * 1000 + amplitude_index * 100 + lag_index)
                rows.append({
                    "sf_group": group, "step_amplitude_deg": amplitude,
                    "time_since_step_ms": 1000 * lag / 120.0,
                    "ssi_percent_vs_stationary": value, "ci95_low": low, "ci95_high": high,
                })
        velocity_num, velocity_den = contributions_by_image(velocity_ssi, velocity_expected, units)
        for velocity_index, velocity in enumerate(velocities):
            point = (velocity_num[:, velocity_index], velocity_den[:, velocity_index])
            baseline = (velocity_num[:, 0], velocity_den[:, 0])
            value, low, high = bootstrap_delta(point, baseline, 11100 + group_index * 100 + velocity_index)
            velocity_rows.append({
                "sf_group": group, "velocity_deg_s": velocity,
                "ssi_percent_vs_stationary": value, "ci95_low": low, "ci95_high": high,
            })
    step = pd.DataFrame(rows)
    velocity = pd.DataFrame(velocity_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    step.to_csv(DATA / "step_hold_pilot.csv", index=False)
    velocity.to_csv(DATA / "constant_velocity_pilot.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    heatmaps = []
    for ax, (group, color) in zip(axes[:2], (("low SF", LOW), ("high SF", HIGH))):
        sub = step.loc[step.sf_group == group]
        matrix = sub.pivot(index="step_amplitude_deg", columns="time_since_step_ms", values="ssi_percent_vs_stationary").sort_index().sort_index(axis=1)
        heatmaps.append(matrix.to_numpy())
    limit = max(abs(np.nanmin(np.concatenate([x.ravel() for x in heatmaps]))), abs(np.nanmax(np.concatenate([x.ravel() for x in heatmaps]))))
    for ax, (group, _), matrix in zip(axes[:2], (("low SF", LOW), ("high SF", HIGH)), heatmaps):
        im = ax.imshow(
            matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit,
            extent=[lags.min() * 1000 / 120, lags.max() * 1000 / 120, amplitudes.min() * 60, amplitudes.max() * 60],
        )
        ax.set(xlabel="time since step (ms)", ylabel="step amplitude (arcmin)", title=f"{'A' if group == 'low SF' else 'B'}  Step-and-hold: {group}")
    fig.colorbar(im, ax=axes[:2], label="SSI change vs stationary (%)", shrink=.82)
    ax = axes[2]
    for group, color, marker in (("low SF", LOW, "o"), ("high SF", HIGH, "s")):
        sub = velocity.loc[velocity.sf_group == group].sort_values("velocity_deg_s")
        y = sub.ssi_percent_vs_stationary.to_numpy(float)
        ax.errorbar(sub.velocity_deg_s, y, yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)), color=color, marker=marker, capsize=2, label=group)
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="constant retinal velocity (deg/s)", ylabel="SSI change vs stationary (%)", title="C  Sustained constant velocity")
    ax.legend(frameon=False)
    fig.suptitle("Figure F — Transient displacement and sustained velocity are separable model inputs", fontsize=14, weight="bold")
    export(fig, "fig_F_synthetic_history")

    maxima = {}
    for group in groups:
        sub = step.loc[(step.sf_group == group) & (step.step_amplitude_deg > 0)]
        best = sub.loc[sub.ssi_percent_vs_stationary.idxmax()]
        vsub = velocity.loc[(velocity.sf_group == group) & (velocity.velocity_deg_s > 0)]
        vbest = vsub.loc[vsub.ssi_percent_vs_stationary.idxmax()]
        maxima[group] = {
            "step_best_percent": float(best.ssi_percent_vs_stationary),
            "step_best_amplitude_deg": float(best.step_amplitude_deg),
            "step_best_time_ms": float(best.time_since_step_ms),
            "velocity_best_percent": float(vbest.ssi_percent_vs_stationary),
            "velocity_best_deg_s": float(vbest.velocity_deg_s),
        }
    write_json(
        FIRST / "synthetic_history_statistics.json",
        {
            "status": "eight_image_single_output_direction_averaged_pilot",
            "step_grid": {"amplitudes_deg": amplitudes, "lag_frames": lags},
            "velocity_grid_deg_s": velocities,
            "maxima": maxima,
            "limitation": "single-output designed inputs; not yet the full 40-output natural-movie dose response",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
