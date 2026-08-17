#!/usr/bin/env python3
"""Fit and visualize layerwise SF x TF tuning from the dense grating probe."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
LOW = "#007C83"
HIGH = "#D55E00"
EPS = 1e-12

REFERENCE_QUARTILE_PEAKS = np.asarray(
    [[1.3, 17.0], [2.3, 15.0], [3.4, 7.7], [5.0, 6.7]], dtype=float
)
REFERENCE_QUARTILE_COUNTS = np.asarray([22, 21, 21, 21], dtype=int)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def load_layout() -> dict[str, np.ndarray]:
    with np.load(RAW / "layout_and_grid.npz") as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def phase_orientation_surface(values: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return channel x SF x TF marginal and channel x SF x TF x orientation."""
    x = np.asarray(values, dtype=np.float64)
    if metric == "f0_signed_mean":
        oriented = np.nanmean(x, axis=-1)
        collapsed = np.nanmean(oriented, axis=-1)
    else:
        oriented = np.sqrt(np.nanmean(x**2, axis=-1))
        collapsed = np.nanmean(oriented, axis=-1)
    return collapsed, oriented


def gaussian_surface(params: np.ndarray, log_sf: np.ndarray, log_tf: np.ndarray) -> np.ndarray:
    baseline, amplitude, mu_sf, sigma_sf, mu_tf, sigma_tf = params
    return baseline + amplitude * np.exp(
        -0.5 * ((log_sf - mu_sf) / sigma_sf) ** 2
        -0.5 * ((log_tf - mu_tf) / sigma_tf) ** 2
    )


def fit_surface(
    surface: np.ndarray,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    *,
    allow_negative_amplitude: bool = False,
) -> dict[str, float | bool | str]:
    sf_grid, tf_grid = np.meshgrid(np.log2(spatial_cpd), np.log2(temporal_hz), indexing="ij")
    y = np.asarray(surface, dtype=float)
    keep = np.isfinite(y)
    if keep.sum() < 20:
        return {"fit_ok": False, "fit_error": "too_few_finite_points"}
    yy = y[keep]
    lo = float(np.percentile(yy, 5))
    hi = float(np.percentile(yy, 98))
    dynamic = max(hi - lo, float(np.ptp(yy)), EPS)
    peak = np.unravel_index(int(np.nanargmax(y)), y.shape)
    trough = np.unravel_index(int(np.nanargmin(y)), y.shape)
    initials = [np.asarray([lo, max(float(np.nanmax(y)) - lo, EPS), sf_grid[peak], 0.9, tf_grid[peak], 1.0])]
    if allow_negative_amplitude:
        initials.append(
            np.asarray(
                [hi, -max(hi - float(np.nanmin(y)), EPS), sf_grid[trough], 0.9, tf_grid[trough], 1.0]
            )
        )
    sf_min, sf_max = float(np.log2(spatial_cpd[0])), float(np.log2(spatial_cpd[-1]))
    tf_min, tf_max = float(np.log2(temporal_hz[0])), float(np.log2(temporal_hz[-1]))
    amplitude_lower = -5 * dynamic if allow_negative_amplitude else 0.0
    lower = np.asarray([float(np.nanmin(y)) - 2 * dynamic, amplitude_lower, sf_min - 1.0, 0.15, tf_min - 1.0, 0.15])
    upper = np.asarray([float(np.nanmax(y)) + dynamic, 5 * dynamic, sf_max + 1.0, 4.0, tf_max + 1.0, 4.0])
    try:
        candidates = [
            least_squares(
                lambda p: gaussian_surface(p, sf_grid[keep], tf_grid[keep]) - yy,
                initial,
                bounds=(lower, upper),
                max_nfev=1000,
            )
            for initial in initials
        ]
        result = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    except Exception as exc:
        return {"fit_ok": False, "fit_error": type(exc).__name__}
    prediction = gaussian_surface(result.x, sf_grid[keep], tf_grid[keep])
    residual_ss = float(np.sum((yy - prediction) ** 2))
    total_ss = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - residual_ss / max(total_ss, EPS)
    baseline, amplitude, mu_sf, sigma_sf, mu_tf, sigma_tf = result.x
    sf_censor = "left" if mu_sf <= sf_min else "right" if mu_sf >= sf_max else "none"
    tf_censor = "left" if mu_tf <= tf_min else "right" if mu_tf >= tf_max else "none"
    signal_fraction = float(abs(amplitude) / max(abs(baseline) + abs(amplitude), EPS))
    valid = bool(result.success and amplitude > EPS and r2 >= 0.15 and signal_fraction >= 0.02)
    return {
        "fit_ok": bool(result.success),
        "fit_error": "",
        "valid_tuned": valid,
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "preferred_sf_cpd": float(2**mu_sf),
        "preferred_tf_hz": float(2**mu_tf),
        "preferred_speed_dps": float(2 ** (mu_tf - mu_sf)),
        "sf_sigma_octaves": float(sigma_sf),
        "tf_sigma_octaves": float(sigma_tf),
        "sf_censoring": sf_censor,
        "tf_censoring": tf_censor,
        "r2": float(r2),
        "signal_fraction": signal_fraction,
        "surface_min": float(np.nanmin(y)),
        "surface_max": float(np.nanmax(y)),
    }


def fit_all(layout: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    stages = layout["stages"].astype(str)
    starts = layout["stage_starts"].astype(int)
    stops = layout["stage_stops"].astype(int)
    sf = layout["spatial_cpd"].astype(float)
    tf = layout["temporal_hz"].astype(float)
    rows = []
    surfaces: dict[tuple[str, str], np.ndarray] = {}
    for metric in ("f0_signed_mean", "f1_amplitude", "f2_amplitude", "temporal_ac_rms"):
        values = np.load(RAW / f"{metric}.npy", mmap_mode="r")
        for stage_i, stage in enumerate(stages):
            collapsed, _ = phase_orientation_surface(values[starts[stage_i] : stops[stage_i]], metric)
            surfaces[(stage, metric)] = collapsed.astype(np.float32)
            for channel in range(len(collapsed)):
                fit = fit_surface(
                    collapsed[channel],
                    sf,
                    tf,
                    allow_negative_amplitude=metric == "f0_signed_mean",
                )
                rows.append({"stage": stage, "channel": channel, "response_metric": metric, **fit})
            print(f"fit {metric}: {stage} ({len(collapsed)} channels)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "layerwise_channel_sftf_fits.csv.gz", index=False)
    return table, surfaces


def assign_rr100_quartiles(fits: pd.DataFrame) -> pd.DataFrame:
    rr = fits.loc[
        fits.stage.eq("rr100") & fits.response_metric.eq("f0_signed_mean") & fits.valid_tuned.fillna(False)
    ].copy()
    rr["sf_quartile"] = pd.qcut(
        rr.preferred_sf_cpd.rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    rr.to_csv(DATA / "rr100_f0_sf_quartiles.csv", index=False)
    return rr


def normalized_fitted_surface(row: pd.Series, sf_grid: np.ndarray, tf_grid: np.ndarray) -> np.ndarray:
    log_sf, log_tf = np.meshgrid(np.log2(sf_grid), np.log2(tf_grid), indexing="ij")
    params = np.asarray(
        [row.baseline, row.amplitude, np.log2(row.preferred_sf_cpd), row.sf_sigma_octaves, np.log2(row.preferred_tf_hz), row.tf_sigma_octaves]
    )
    value = gaussian_surface(params, log_sf, log_tf)
    low, high = np.percentile(value, [5, 98])
    return np.clip((value - low) / max(high - low, EPS), 0, 1)


def quartile_mean_surfaces(rr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    fine_sf = np.geomspace(0.8, 10.0, 100)
    fine_tf = np.geomspace(0.5, 32.0, 110)
    surfaces = {}
    rows = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        frame = rr.loc[rr.sf_quartile.eq(quartile)]
        stack = np.stack([normalized_fitted_surface(row, fine_sf, fine_tf) for _, row in frame.iterrows()])
        mean = stack.mean(axis=0)
        peak = np.unravel_index(int(np.argmax(mean)), mean.shape)
        surfaces[quartile] = mean
        rows.append(
            {
                "sf_quartile": quartile,
                "n_units": int(len(frame)),
                "mean_surface_peak_sf_cpd": float(fine_sf[peak[0]]),
                "mean_surface_peak_tf_hz": float(fine_tf[peak[1]]),
                "median_unit_preferred_sf_cpd": float(frame.preferred_sf_cpd.median()),
                "median_unit_preferred_tf_hz": float(frame.preferred_tf_hz.median()),
                "median_unit_preferred_speed_dps": float(frame.preferred_speed_dps.median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "rr100_f0_quartile_surface_summary.csv", index=False)
    return fine_sf, fine_tf, surfaces, summary


def figure_rr100_quartiles(
    fine_sf: np.ndarray, fine_tf: np.ndarray, surfaces: dict[str, np.ndarray], summary: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for index, quartile in enumerate(("Q1", "Q2", "Q3", "Q4")):
        ax = axes[index]
        value = surfaces[quartile]
        row = summary.loc[summary.sf_quartile.eq(quartile)].iloc[0]
        mesh = ax.pcolormesh(fine_sf, fine_tf, value.T, shading="auto", cmap="viridis", vmin=0, vmax=1)
        ax.contour(fine_sf, fine_tf, value.T, levels=[0.68], colors=[plt.get_cmap("tab10")(index)], linewidths=2)
        ax.set(xscale="log", yscale="log", xticks=[1, 2, 4, 8], yticks=[.5, 1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(
            f"SF {quartile}  n={int(row.n_units)}\npeak {row.mean_surface_peak_sf_cpd:.2g} cpd / {row.mean_surface_peak_tf_hz:.2g} Hz",
            loc="left",
            weight="bold",
        )
        ax.set_xlabel("SF (cycles/deg)")
    axes[0].set_ylabel("TF (Hz)")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="mean within-unit normalized fitted F0", shrink=.86)
    fig.suptitle(
        "Independent RR100 F0 reconstruction — ordering reproduced; absolute postdoc fit not reproduced",
        fontsize=14,
        weight="bold",
    )
    export(fig, "figure_1_rr100_f0_quartile_surfaces")


def figure_population_surfaces(
    surfaces: dict[tuple[str, str], np.ndarray], layout: dict[str, np.ndarray]
) -> None:
    stages = [
        "temporal_frontend",
        "stem_preactivation",
        "post_stem_splitrelu",
        "resblock1_preactivation",
        "resblock1_output",
        "resblock2_output",
        "convgru",
        "rr100",
    ]
    labels = ["frontend", "stem pre", "stem SplitReLU", "RB1 pre", "RB1 out", "RB2 out", "ConvGRU", "RR100"]
    sf = layout["spatial_cpd"].astype(float)
    tf = layout["temporal_hz"].astype(float)
    fig, axes = plt.subplots(2, len(stages), figsize=(16.0, 6.5), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for row_i, metric in enumerate(("f1_amplitude", "f0_signed_mean")):
        for col_i, (stage, label) in enumerate(zip(stages, labels)):
            value = np.asarray(surfaces[(stage, metric)], dtype=float)
            if metric == "f0_signed_mean":
                value = value - np.nanmin(value, axis=(1, 2), keepdims=True)
            value = np.clip(value, 0, None)
            value /= np.maximum(np.nanmax(value, axis=(1, 2), keepdims=True), EPS)
            mean = np.nanmean(value, axis=0)
            mesh = axes[row_i, col_i].pcolormesh(sf, tf, mean.T, shading="auto", cmap="viridis", vmin=0, vmax=1)
            axes[row_i, col_i].set(
                xscale="log",
                yscale="log",
                xticks=[.4, 1, 4, 16],
                yticks=[.4, 1, 4, 16, 51.2],
                title=label,
            )
            axes[row_i, col_i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axes[row_i, col_i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if row_i == 1:
                axes[row_i, col_i].set_xlabel("SF (cpd)")
        axes[row_i, 0].set_ylabel(("F1 phase-locked" if row_i == 0 else "F0 mean response") + "\nTF (Hz)")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=.76, label="mean within-channel normalized response")
    fig.suptitle("Measured SF×TF representation through the complete trained model", fontsize=14, weight="bold")
    export(fig, "figure_2a_layerwise_mean_sftf_surfaces")


def figure_preference_emergence(fits: pd.DataFrame) -> None:
    stages = [
        "temporal_frontend",
        "stem_preactivation",
        "post_stem_splitrelu",
        "resblock1_preactivation",
        "resblock1_output",
        "resblock2_output",
        "convgru",
        "rr100",
    ]
    labels = ["frontend", "stem pre", "stem SplitReLU", "RB1 pre", "RB1 out", "RB2 out", "ConvGRU", "RR100"]
    fig, axes = plt.subplots(2, len(stages), figsize=(16.0, 6.8), sharex=True, sharey=True, constrained_layout=True)
    for row_i, metric in enumerate(("f1_amplitude", "f0_signed_mean")):
        for col_i, (stage, label) in enumerate(zip(stages, labels)):
            ax = axes[row_i, col_i]
            frame = fits.loc[
                fits.stage.eq(stage)
                & fits.response_metric.eq(metric)
                & fits.valid_tuned.fillna(False)
                & fits.sf_censoring.eq("none")
                & fits.tf_censoring.eq("none")
            ]
            if len(frame):
                ax.scatter(frame.preferred_sf_cpd, frame.preferred_tf_hz, s=10, alpha=.45, color="#2563EB" if row_i == 0 else "#D55E00")
            for speed in (1, 4, 16):
                sf_line = np.geomspace(.4, 16, 100)
                ax.plot(sf_line, speed * sf_line, color="0.75", lw=.6, ls=":" if speed != 4 else "--")
            ax.set(xscale="log", yscale="log", xlim=(.35, 19), ylim=(.35, 60), xticks=[.4, 1, 4, 16], yticks=[.4, 1, 4, 16, 51.2])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_title(f"{label}\n{len(frame)} valid interior")
            if row_i == 1:
                ax.set_xlabel("preferred SF")
        axes[row_i, 0].set_ylabel(("F1 modulation" if row_i == 0 else "F0 mean") + "\npreferred TF (Hz)")
    fig.suptitle("Where joint SF×TF tuning emerges (diagonals are constant retinal speed)", fontsize=14, weight="bold")
    export(fig, "figure_2b_layerwise_sftf_preference_emergence")


def stage_summary(fits: pd.DataFrame, layout: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for stage, n_channels in zip(layout["stages"].astype(str), layout["stage_channels"].astype(int)):
        for metric in ("f0_signed_mean", "f1_amplitude", "f2_amplitude", "temporal_ac_rms"):
            frame = fits.loc[fits.stage.eq(stage) & fits.response_metric.eq(metric)]
            valid = frame.valid_tuned.fillna(False)
            interior = valid & frame.sf_censoring.eq("none") & frame.tf_censoring.eq("none")
            rows.append(
                {
                    "stage": stage,
                    "response_metric": metric,
                    "n_channels": int(n_channels),
                    "n_valid_tuned": int(valid.sum()),
                    "n_valid_interior": int(interior.sum()),
                    "fraction_valid_tuned": float(valid.mean()),
                    "median_r2": float(frame.r2.median()),
                    "median_preferred_sf_cpd_interior": float(frame.loc[interior, "preferred_sf_cpd"].median()),
                    "median_preferred_tf_hz_interior": float(frame.loc[interior, "preferred_tf_hz"].median()),
                    "median_preferred_speed_dps_interior": float(frame.loc[interior, "preferred_speed_dps"].median()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "layerwise_sftf_stage_summary.csv", index=False)
    return result


def main() -> int:
    configure()
    DATA.mkdir(parents=True, exist_ok=True)
    layout = load_layout()
    completed = np.load(RAW / "completed.npy", mmap_mode="r")
    if not bool(np.all(completed)):
        raise RuntimeError(f"Layerwise probe incomplete: {int(completed.sum())}/{completed.size}")
    fits, layer_surfaces = fit_all(layout)
    summary = stage_summary(fits, layout)
    rr = assign_rr100_quartiles(fits)
    if len(rr) < 20:
        raise RuntimeError(f"RR100 validation failed: only {len(rr)} valid interior F0 fits")
    fine_sf, fine_tf, quartile_surfaces, quartile_summary = quartile_mean_surfaces(rr)
    figure_rr100_quartiles(fine_sf, fine_tf, quartile_surfaces, quartile_summary)
    figure_population_surfaces(layer_surfaces, layout)
    figure_preference_emergence(fits)
    observed = quartile_summary[["mean_surface_peak_sf_cpd", "mean_surface_peak_tf_hz"]].to_numpy(float)
    validation = {
        "n_rr100_valid_positive_dynamic_f0_interior": int(len(rr)),
        "quartile_counts": quartile_summary.n_units.to_numpy(int),
        "observed_quartile_mean_surface_peaks_sf_tf": observed,
        "independent_reference_counts": REFERENCE_QUARTILE_COUNTS,
        "independent_reference_peaks_sf_tf": REFERENCE_QUARTILE_PEAKS,
        "log2_peak_rmse_vs_reference": float(np.sqrt(np.mean((np.log2(observed) - np.log2(REFERENCE_QUARTILE_PEAKS)) ** 2))),
        "validity_rule": "best of positive-bump and negative-dip separable log-Gaussian F0 models; positive fitted dynamic amplitude, R2>=0.15, and absolute signal fraction>=0.02. Boundary peaks remain eligible and censoring is reported separately.",
        "stage_summary": summary.to_dict(orient="records"),
    }
    write_json(OUT / "tuning_statistics.json", validation)
    print(json.dumps({key: value for key, value in validation.items() if key != "stage_summary"}, indent=2, default=lambda x: x.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
