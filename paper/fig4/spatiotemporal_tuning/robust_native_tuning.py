#!/usr/bin/env python3
"""Fit and inspect native-240 RR100 joint SF/TF tuning surfaces.

This postprocessor is deliberately stricter than the historical argmax code:
it requires a cycle-valid, approximately half-octave grid, fits a continuous
2-D log-Gaussian with a robust loss, records boundary censoring, and renders
the observed curves beside the fit.  It never turns a boundary bin into an
uncaveated preferred frequency.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


EPS = 1e-10
SQRT2 = math.sqrt(2.0)
DEFAULT_PPD = 37.50476617
DEFAULT_IMAGE_SIZE = 101
DEFAULT_RATE_HZ = 240.0


def recommended_native_grid(
    *,
    ppd: float = DEFAULT_PPD,
    image_size: int = DEFAULT_IMAGE_SIZE,
    frame_rate_hz: float = DEFAULT_RATE_HZ,
) -> dict[str, Any]:
    """Return a half-octave grid that stays clear of both Nyquist limits."""
    fov_deg = float(image_size) / float(ppd)
    spatial_nyquist = 0.5 * float(ppd)
    temporal_nyquist = 0.5 * float(frame_rate_hz)
    # Anchor the half-octave lattice to the aperture's fundamental rather than
    # to a global 0.5-cpd lattice.  On the 35-pixel crop, the old lattice jumped
    # from an invalid 0.66-cycle ramp straight to 1.32 cycles, needlessly
    # censoring units whose peak lies near one cycle across the model input.
    # The final guard-frequency sample prevents the high end from being almost
    # a full half-octave below the safe Nyquist margin.
    spatial_fundamental = 1.0 / fov_deg
    spatial_guard = 0.85 * spatial_nyquist
    spatial = spatial_fundamental * SQRT2 ** np.arange(14, dtype=np.float64)
    spatial = spatial[spatial < spatial_guard]
    if spatial_guard / spatial[-1] >= 2.0**0.3:
        spatial = np.concatenate((spatial, [spatial_guard]))
    temporal = SQRT2 ** np.arange(14, dtype=np.float64)
    if len(spatial) < 5:
        raise ValueError("aperture and pixel grid leave fewer than five valid SF bins")
    if temporal[-1] >= 0.8 * temporal_nyquist:
        raise ValueError("highest temporal frequency is too close to temporal Nyquist")
    return {
        "spatial_cpd": spatial,
        "temporal_hz": np.concatenate(([0.0], temporal)),
        "dynamic_temporal_hz": temporal,
        "orientation_deg": np.asarray([0.0, 45.0, 90.0, 135.0]),
        "duration_s": 2.5,
        "discard_frames": 60,
        "n_dynamic_phases": 2,
        "n_static_phases": 4,
        "ppd": float(ppd),
        "image_size": int(image_size),
        "frame_rate_hz": float(frame_rate_hz),
        "fov_deg": fov_deg,
        "spatial_nyquist_cpd": spatial_nyquist,
        "temporal_nyquist_hz": temporal_nyquist,
        "minimum_cycles_across_aperture": float(spatial[0] * fov_deg),
        "minimum_pixels_per_spatial_cycle": float(ppd / spatial[-1]),
        "minimum_frames_per_temporal_cycle": float(frame_rate_hz / temporal[-1]),
        "spatial_grid_anchor": "one cycle across the model input aperture",
        "minimum_analyzed_cycles": float(
            temporal[0] * (2.5 - 60.0 / float(frame_rate_hz))
        ),
    }


def _surface_prediction(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    baseline, log_amplitude, mux, muy, log_sx, log_sy, rho_raw = params
    amplitude = math.exp(float(log_amplitude))
    sx, sy = math.exp(float(log_sx)), math.exp(float(log_sy))
    rho = math.tanh(float(rho_raw))
    dx, dy = (x - mux) / sx, (y - muy) / sy
    exponent = -0.5 * (dx * dx + dy * dy - 2.0 * rho * dx * dy) / max(
        1.0 - rho * rho, 1e-4
    )
    return baseline + amplitude * np.exp(exponent)


def _quadratic_prediction(
    coefficients: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Evaluate ``c + bx*x + by*y + qxx*x² + qxy*x*y + qyy*y²``."""
    return (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
    )


def fit_local_quadratic_peak(
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    response: np.ndarray,
    *,
    omitted_local_index: int | None = None,
) -> dict[str, Any]:
    """Interpolate the *local* joint SF/TF maximum in log-frequency space.

    A global parametric surface can fit most sampled bins well while placing
    its center away from the largest observed responses.  This estimator asks
    the narrower question required for a preferred-frequency annotation: it
    finds the sampled joint maximum and fits a full quadratic only to its 3x3
    neighborhood.  The vertex is accepted only when the local surface is
    concave and the vertex remains inside the neighboring sample rectangle.

    ``omitted_local_index`` supports a delete-one-point stability audit while
    keeping the same peak neighborhood.  It is deliberately not a global
    goodness-of-fit statistic.
    """
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    temporal = np.asarray(temporal_hz, dtype=np.float64)
    values = np.asarray(response, dtype=np.float64)
    if values.shape != (len(temporal), len(spatial)):
        raise ValueError(f"response shape {values.shape} != {(len(temporal), len(spatial))}")
    if np.any(spatial <= 0) or np.any(temporal <= 0):
        raise ValueError("joint dynamic tuning grids must be strictly positive")
    if not np.all(np.diff(spatial) > 0) or not np.all(np.diff(temporal) > 0):
        raise ValueError("frequency grids must be strictly increasing")
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"peak_status": "insufficient_data"}

    discrete_flat = int(np.nanargmax(values))
    tf_index, sf_index = np.unravel_index(discrete_flat, values.shape)
    base = {
        "discrete_peak_sf_cpd": float(spatial[sf_index]),
        "discrete_peak_tf_hz": float(temporal[tf_index]),
        "discrete_peak_sf_index": int(sf_index),
        "discrete_peak_tf_index": int(tf_index),
    }
    if sf_index in (0, len(spatial) - 1) or tf_index in (0, len(temporal) - 1):
        return {"peak_status": "boundary", "peak_censored": True, **base}

    rows = np.arange(tf_index - 1, tf_index + 2)
    columns = np.arange(sf_index - 1, sf_index + 2)
    x0, y0 = np.log2(spatial[sf_index]), np.log2(temporal[tf_index])
    local_x: list[float] = []
    local_y: list[float] = []
    local_z: list[float] = []
    local_grid_indices: list[tuple[int, int]] = []
    for row in rows:
        for column in columns:
            local_x.append(float(np.log2(spatial[column]) - x0))
            local_y.append(float(np.log2(temporal[row]) - y0))
            local_z.append(float(values[row, column]))
            local_grid_indices.append((int(row), int(column)))
    x = np.asarray(local_x)
    y = np.asarray(local_y)
    z = np.asarray(local_z)
    use = np.isfinite(z)
    if omitted_local_index is not None:
        if omitted_local_index < 0 or omitted_local_index >= len(z):
            raise ValueError("omitted_local_index must address the 3x3 neighborhood")
        use[int(omitted_local_index)] = False
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    if np.count_nonzero(use) < 7 or np.linalg.matrix_rank(design[use]) < 6:
        return {"peak_status": "insufficient_local_data", "peak_censored": True, **base}
    coefficients = np.linalg.lstsq(design[use], z[use], rcond=None)[0]
    hessian = np.asarray(
        [
            [2.0 * coefficients[3], coefficients[4]],
            [coefficients[4], 2.0 * coefficients[5]],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    prediction = design @ coefficients
    denominator = float(np.sum(np.square(z[use] - np.mean(z[use]))))
    local_r2 = 1.0 - float(np.sum(np.square(z[use] - prediction[use]))) / max(
        denominator, EPS
    )
    diagnostic = {
        **base,
        "local_fit_r2": local_r2,
        "local_hessian_eigenvalue_min": float(eigenvalues[0]),
        "local_hessian_eigenvalue_max": float(eigenvalues[-1]),
        "local_quadratic_coefficients": coefficients,
        "local_observed": z,
        "local_fitted": prediction,
        "local_grid_indices": np.asarray(local_grid_indices, dtype=int),
    }
    # Strict concavity is the condition under which the stationary point is a
    # local maximum rather than a saddle or minimum.
    curvature_tolerance = max(float(np.ptp(z[use])), EPS) * 1e-8
    if eigenvalues[-1] >= -curvature_tolerance:
        return {"peak_status": "nonconcave", "peak_censored": True, **diagnostic}
    vertex = -np.linalg.solve(hessian, coefficients[1:3])
    x_low = float(np.log2(spatial[sf_index - 1]) - x0)
    x_high = float(np.log2(spatial[sf_index + 1]) - x0)
    y_low = float(np.log2(temporal[tf_index - 1]) - y0)
    y_high = float(np.log2(temporal[tf_index + 1]) - y0)
    vertex_inside = x_low <= vertex[0] <= x_high and y_low <= vertex[1] <= y_high
    if not vertex_inside:
        return {
            "peak_status": "vertex_outside_neighborhood",
            "peak_censored": True,
            "peak_offset_sf_octaves": float(vertex[0]),
            "peak_offset_tf_octaves": float(vertex[1]),
            **diagnostic,
        }
    return {
        "peak_status": "ok",
        "peak_censored": False,
        "preferred_sf_cpd": float(2.0 ** (x0 + vertex[0])),
        "preferred_tf_hz": float(2.0 ** (y0 + vertex[1])),
        "peak_offset_sf_octaves": float(vertex[0]),
        "peak_offset_tf_octaves": float(vertex[1]),
        **diagnostic,
    }


def fit_log_gaussian_surface(
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    response: np.ndarray,
) -> dict[str, Any]:
    """Robust continuous fit with explicit discrete and fitted boundary flags."""
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    temporal = np.asarray(temporal_hz, dtype=np.float64)
    values = np.asarray(response, dtype=np.float64)
    if values.shape != (len(temporal), len(spatial)):
        raise ValueError(f"response shape {values.shape} != {(len(temporal), len(spatial))}")
    if np.any(spatial <= 0) or np.any(temporal <= 0):
        raise ValueError("joint dynamic tuning grids must be strictly positive")
    if not np.all(np.diff(spatial) > 0) or not np.all(np.diff(temporal) > 0):
        raise ValueError("frequency grids must be strictly increasing")
    log_sf, log_tf = np.log2(spatial), np.log2(temporal)
    xx, yy = np.meshgrid(log_sf, log_tf)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 12:
        return {"fit_status": "insufficient_data"}
    z = values[finite]
    span = float(np.max(z) - np.min(z))
    if span <= max(EPS, 1e-6 * float(np.max(np.abs(z)))):
        return {"fit_status": "flat"}
    discrete_flat = int(np.nanargmax(values))
    tf_index, sf_index = np.unravel_index(discrete_flat, values.shape)
    baseline0 = max(float(np.nanpercentile(z, 5)), 0.0)
    amplitude0 = max(float(np.nanmax(z) - baseline0), EPS)
    initial = np.asarray(
        [baseline0, math.log(amplitude0), log_sf[sf_index], log_tf[tf_index], 0.0, 0.0, 0.0]
    )
    upper_response = max(float(np.nanmax(z)) * 2.0, amplitude0 * 2.0, EPS)
    lower = np.asarray(
        [0.0, math.log(EPS), log_sf[0], log_tf[0], math.log(0.20), math.log(0.20), -1.8]
    )
    upper = np.asarray(
        [upper_response, math.log(upper_response * 10.0), log_sf[-1], log_tf[-1], math.log(5.0), math.log(5.0), 1.8]
    )
    scale = max(0.1 * span, EPS)
    result = least_squares(
        lambda params: (_surface_prediction(params, xx[finite], yy[finite]) - z) / scale,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=4000,
    )
    predicted = _surface_prediction(result.x, xx, yy)
    residual = z - predicted[finite]
    denominator = float(np.sum(np.square(z - np.mean(z))))
    r2 = 1.0 - float(np.sum(np.square(residual))) / max(denominator, EPS)
    baseline, log_amplitude, mux, muy, log_sx, log_sy, rho_raw = result.x
    weights = np.clip(values - np.nanpercentile(z, 10), 0.0, None)
    weight_sum = float(np.nansum(weights))
    weighted_sf = (
        float(2.0 ** (np.nansum(weights * xx) / weight_sum)) if weight_sum > EPS else np.nan
    )
    weighted_tf = (
        float(2.0 ** (np.nansum(weights * yy) / weight_sum)) if weight_sum > EPS else np.nan
    )
    sf_step = float(np.median(np.diff(log_sf)))
    tf_step = float(np.median(np.diff(log_tf)))
    low_sf_censored = sf_index == 0 or mux - log_sf[0] < 0.25 * sf_step
    high_sf_censored = sf_index == len(spatial) - 1 or log_sf[-1] - mux < 0.25 * sf_step
    low_tf_censored = tf_index == 0 or muy - log_tf[0] < 0.25 * tf_step
    high_tf_censored = tf_index == len(temporal) - 1 or log_tf[-1] - muy < 0.25 * tf_step
    discrete_boundary = sf_index in (0, len(spatial) - 1) or tf_index in (0, len(temporal) - 1)
    fitted_boundary = low_sf_censored or high_sf_censored or low_tf_censored or high_tf_censored
    censoring_labels = [
        label
        for label, active in (
            ("low_sf", low_sf_censored),
            ("high_sf", high_sf_censored),
            ("low_tf", low_tf_censored),
            ("high_tf", high_tf_censored),
        )
        if active
    ]
    return {
        "fit_status": "ok" if result.success else "optimizer_failed",
        "fit_success": bool(result.success),
        "fit_message": str(result.message),
        "fit_r2": r2,
        "baseline": float(baseline),
        "amplitude": float(math.exp(log_amplitude)),
        "preferred_sf_cpd": float(2.0**mux),
        "preferred_tf_hz": float(2.0**muy),
        "sf_bandwidth_sigma_octaves": float(math.exp(log_sx)),
        "tf_bandwidth_sigma_octaves": float(math.exp(log_sy)),
        "sf_tf_log_correlation": float(math.tanh(rho_raw)),
        "weighted_center_sf_cpd": weighted_sf,
        "weighted_center_tf_hz": weighted_tf,
        "discrete_peak_sf_cpd": float(spatial[sf_index]),
        "discrete_peak_tf_hz": float(temporal[tf_index]),
        "discrete_peak_boundary": bool(discrete_boundary),
        "fitted_peak_boundary": bool(fitted_boundary),
        "peak_censored": bool(discrete_boundary or fitted_boundary),
        "low_sf_censored": bool(low_sf_censored),
        "high_sf_censored": bool(high_sf_censored),
        "low_tf_censored": bool(low_tf_censored),
        "high_tf_censored": bool(high_tf_censored),
        "peak_censoring": "+".join(censoring_labels) if censoring_labels else "none",
        "response_prominence_fraction": span / max(float(np.nanmax(z)), EPS),
        "observed": values,
        "fitted": predicted,
    }


def _complete_surface(subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial = np.sort(subset.spatial_cpd.unique().astype(float))
    temporal = np.sort(subset.temporal_hz.unique().astype(float))
    pivot = subset.pivot_table(
        index="temporal_hz", columns="spatial_cpd", values="response_amp_rms", aggfunc="mean"
    ).reindex(index=temporal, columns=spatial)
    return spatial, temporal, pivot.to_numpy(dtype=float)


def analyze_grouped_table(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    required = {
        "unit_index",
        "probe_orientation_deg",
        "spatial_cpd",
        "temporal_hz",
        "response_amp_rms",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"grouped tuning table lacks {missing}")
    values = table.copy()
    for column in required - {"unit_index"}:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values["unit_index"] = pd.to_numeric(values["unit_index"], errors="raise").astype(int)
    values = values[values.temporal_hz.gt(0)].copy()
    rows: list[dict[str, Any]] = []
    surfaces: dict[int, dict[str, Any]] = {}
    for unit_index, unit in values.groupby("unit_index", sort=True):
        orientation_scores = unit.groupby("probe_orientation_deg")["response_amp_rms"].apply(
            lambda x: float(np.sqrt(np.nanmean(np.square(x.to_numpy(dtype=float)))))
        )
        best_orientation = float(orientation_scores.idxmax())
        selected = unit[np.isclose(unit.probe_orientation_deg, best_orientation)]
        spatial, temporal, response = _complete_surface(selected)
        fit = fit_log_gaussian_surface(spatial, temporal, response)
        record: dict[str, Any] = {
            "unit_index": int(unit_index),
            "best_orientation_deg": best_orientation,
            "n_spatial_frequencies": int(len(spatial)),
            "n_temporal_frequencies": int(len(temporal)),
            "surface_completeness": float(np.mean(np.isfinite(response))),
        }
        record.update({key: value for key, value in fit.items() if key not in {"observed", "fitted"}})
        rows.append(record)
        surfaces[int(unit_index)] = {
            "spatial_cpd": spatial,
            "temporal_hz": temporal,
            **fit,
        }
    return pd.DataFrame(rows), surfaces


def _select_examples(summary: pd.DataFrame, maximum: int = 12) -> list[int]:
    good = summary[
        summary.fit_status.eq("ok")
        & summary.fit_r2.ge(0.35)
        & ~summary.peak_censored.astype(bool)
    ].sort_values("preferred_sf_cpd")
    if good.empty:
        good = summary[summary.fit_status.eq("ok")].sort_values("preferred_sf_cpd")
    if len(good) <= maximum:
        return good.unit_index.astype(int).tolist()
    indices = np.round(np.linspace(0, len(good) - 1, maximum)).astype(int)
    return good.iloc[indices].unit_index.astype(int).tolist()


def _plot_example_pages(
    summary: pd.DataFrame,
    surfaces: dict[int, dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    units = _select_examples(summary)
    outputs: list[str] = []

    def compact_frequency(value: float) -> str:
        return f"{value:.3g}"

    for page_index, start in enumerate(range(0, len(units), 4), start=1):
        page_units = units[start : start + 4]
        figure, axes = plt.subplots(len(page_units), 3, figsize=(11.5, 2.8 * len(page_units)), squeeze=False)
        for row_index, unit_index in enumerate(page_units):
            surface = surfaces[unit_index]
            spatial = surface["spatial_cpd"]
            temporal = surface["temporal_hz"]
            observed = surface["observed"]
            fitted = surface["fitted"]
            vmax = max(float(np.nanmax(observed)), EPS)
            axes[row_index, 0].imshow(
                observed / vmax, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1
            )
            axes[row_index, 0].contour(fitted / vmax, levels=[0.25, 0.5, 0.75], colors="cyan", linewidths=0.8)
            # The native grid is half-octave spaced. Label octave landmarks
            # rather than printing all 14 TF values into a small heatmap.
            sf_tick_index = np.arange(0, len(spatial), 2, dtype=int)
            tf_tick_index = np.unique(
                np.r_[np.arange(0, len(temporal), 2, dtype=int), len(temporal) - 1]
            )
            axes[row_index, 0].set_xticks(
                sf_tick_index,
                [compact_frequency(spatial[index]) for index in sf_tick_index],
                rotation=0,
            )
            axes[row_index, 0].set_yticks(
                tf_tick_index,
                [compact_frequency(temporal[index]) for index in tf_tick_index],
            )
            axes[row_index, 0].set_ylabel(f"unit {unit_index}\nTF (Hz)")
            axes[row_index, 0].set_xlabel("SF (cpd)")
            axes[row_index, 0].set_title(
                f"observed + fit; $R^2$={surface['fit_r2']:.2f}\nori {summary.loc[summary.unit_index.eq(unit_index), 'best_orientation_deg'].iloc[0]:g} deg"
            )
            observed_sf = np.sqrt(np.nanmean(np.square(observed), axis=0))
            fitted_sf = np.sqrt(np.nanmean(np.square(fitted), axis=0))
            axes[row_index, 1].plot(spatial, observed_sf, "o-", label="observed")
            axes[row_index, 1].plot(spatial, fitted_sf, "-", label="fit")
            axes[row_index, 1].axvline(surface["preferred_sf_cpd"], color="0.4", linestyle=":")
            axes[row_index, 1].set_xscale("log", base=2)
            axes[row_index, 1].set_xlabel("SF (cpd)")
            axes[row_index, 1].set_ylabel("RMS amplitude")
            axes[row_index, 1].grid(alpha=0.2)
            if row_index == 0:
                axes[row_index, 1].legend(frameon=False)
            observed_tf = np.sqrt(np.nanmean(np.square(observed), axis=1))
            fitted_tf = np.sqrt(np.nanmean(np.square(fitted), axis=1))
            axes[row_index, 2].plot(temporal, observed_tf, "o-", label="observed")
            axes[row_index, 2].plot(temporal, fitted_tf, "-", label="fit")
            axes[row_index, 2].axvline(surface["preferred_tf_hz"], color="0.4", linestyle=":")
            axes[row_index, 2].set_xscale("log", base=2)
            axes[row_index, 2].set_xlabel("TF (Hz)")
            axes[row_index, 2].set_ylabel("RMS amplitude")
            axes[row_index, 2].grid(alpha=0.2)
        figure.suptitle(
            "Native-240 joint tuning: actual curves and continuous fits\n"
            "cyan contours are fitted, not argmax interpolation",
            fontsize=13,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.965])
        path = out_dir / f"robust_tuning_examples_page{page_index}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def _plot_population(summary: pd.DataFrame, out_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), constrained_layout=True)
    okay = summary.fit_status.eq("ok")
    axes[0, 0].hist(summary.loc[okay, "fit_r2"], bins=np.linspace(-0.2, 1.0, 25), color="#4c78a8")
    axes[0, 0].axvline(0.35, color="0.3", linestyle=":")
    axes[0, 0].set_xlabel("joint log-Gaussian fit $R^2$")
    axes[0, 0].set_ylabel("units")
    axes[0, 0].set_title("Fit quality")
    axes[0, 1].scatter(
        summary.loc[okay, "discrete_peak_sf_cpd"],
        summary.loc[okay, "preferred_sf_cpd"],
        c=summary.loc[okay, "fit_r2"], cmap="viridis", vmin=0, vmax=1,
    )
    axes[0, 1].plot([0.4, 18], [0.4, 18], color="0.7", linestyle=":")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log", base=2)
    axes[0, 1].set_xlabel("discrete peak SF (cpd)")
    axes[0, 1].set_ylabel("continuous fitted SF (cpd)")
    axes[0, 1].set_title("Interpolation check")
    axes[1, 0].scatter(
        summary.loc[okay, "discrete_peak_tf_hz"],
        summary.loc[okay, "preferred_tf_hz"],
        c=summary.loc[okay, "fit_r2"], cmap="viridis", vmin=0, vmax=1,
    )
    axes[1, 0].plot([0.8, 100], [0.8, 100], color="0.7", linestyle=":")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log", base=2)
    axes[1, 0].set_xlabel("discrete peak TF (Hz)")
    axes[1, 0].set_ylabel("continuous fitted TF (Hz)")
    axes[1, 0].set_title("Interpolation check")
    fractions = [
        float(np.mean(summary.fit_status.ne("ok"))),
        float(np.mean(summary.low_sf_censored.astype("boolean").fillna(True).to_numpy(dtype=bool))),
        float(np.mean(summary.low_tf_censored.astype("boolean").fillna(True).to_numpy(dtype=bool))),
        float(np.mean(
            summary[["high_sf_censored", "high_tf_censored"]]
            .astype("boolean")
            .fillna(True)
            .any(axis=1)
            .to_numpy(dtype=bool)
        )),
        float(np.mean(summary.fit_r2.fillna(-np.inf).lt(0.35))),
    ]
    labels = ["fit failed", "low-SF\ncensored", "low-TF\ncensored", "high-edge\ncensored", "$R^2<.35$"]
    axes[1, 1].bar(np.arange(5), fractions, color=["#999999", "#4c78a8", "#72b7b2", "#e45756", "#f58518"])
    axes[1, 1].set_xticks(np.arange(5), labels)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("fraction of units")
    axes[1, 1].set_title("Acceptance diagnostics")
    figure.suptitle("Native-240 tuning quality control", fontsize=14)
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grouped_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Periodic-probe provenance; defaults to a sibling periodic_tuning_provenance.json",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.grouped_csv)
    summary, surfaces = analyze_grouped_table(table)
    summary.to_csv(args.out_dir / "robust_tuning_summary.csv", index=False)
    examples = _plot_example_pages(summary, surfaces, args.out_dir)
    _plot_population(summary, args.out_dir / "robust_tuning_population_qc.png")
    provenance_path = args.provenance
    if provenance_path is None:
        sibling = args.grouped_csv.parent / "periodic_tuning_provenance.json"
        provenance_path = sibling if sibling.exists() else None
    if provenance_path is not None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        grid = provenance["grid"]
        expected_sf = np.asarray(grid["spatial_cpd"], dtype=float)
        expected_tf = np.asarray(grid["temporal_hz"], dtype=float)
        observed_sf = np.sort(table.spatial_cpd.unique().astype(float))
        observed_tf = np.sort(table.temporal_hz.unique().astype(float))
        if not np.allclose(observed_sf, expected_sf, rtol=0, atol=1e-10):
            raise RuntimeError("Grouped SF grid disagrees with periodic-probe provenance")
        if not np.allclose(observed_tf, expected_tf, rtol=0, atol=1e-10):
            raise RuntimeError("Grouped TF grid disagrees with periodic-probe provenance")
    else:
        grid = recommended_native_grid()
    boundary_fraction = float(
        np.mean(summary.peak_censored.astype("boolean").fillna(True).to_numpy(dtype=bool))
    )
    fit_fraction = float(np.mean(summary.fit_status.eq("ok") & summary.fit_r2.ge(0.35)))
    low_edge_fraction = float(np.mean(
        summary[["low_sf_censored", "low_tf_censored"]]
        .astype("boolean").fillna(True).any(axis=1).to_numpy(dtype=bool)
    ))
    high_edge_fraction = float(np.mean(
        summary[["high_sf_censored", "high_tf_censored"]]
        .astype("boolean").fillna(True).any(axis=1).to_numpy(dtype=bool)
    ))
    gate = {
        "n_units": int(len(summary)),
        "fraction_acceptable_fit": fit_fraction,
        "fraction_peak_censored": boundary_fraction,
        "fraction_low_edge_censored": low_edge_fraction,
        "fraction_high_edge_censored": high_edge_fraction,
        "status": (
            "review_required"
            if fit_fraction < 0.6 or high_edge_fraction > 0.2
            else (
                "grid_supported_with_lowpass_censoring"
                if low_edge_fraction > 0.2
                else "grid_supported"
            )
        ),
        "acceptance_rule": "acceptable fit fraction >=0.6 and high-edge censored fraction <=0.2; low-edge censoring is reported as unresolved low-pass tuning because sub-cycle SF bins are forbidden",
        "recommended_native_grid": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in grid.items()
        },
        "source_grouped_csv": str(args.grouped_csv.resolve()),
        "source_provenance": str(provenance_path.resolve()) if provenance_path is not None else None,
        "example_figures": examples,
    }
    (args.out_dir / "robust_tuning_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
