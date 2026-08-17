#!/usr/bin/env python3
"""Build the non-response first-pass q, aperture, retinal, and frontend audits."""

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
import torch
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.conv_layers import _get_window
from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    OUT_DIR,
    SCALE_FACTORS,
    sha256_file,
    write_json,
)
from paper.fig4.spatiotemporal_tuning.run_grating_probe import (
    IMAGE_SIZE,
    PPD as GRATING_PPD,
    SPATIAL_CPDS,
    WINDOW_SIGMA_FRAC,
    make_grating_movie,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    OUT_SIZE,
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)
from paper.fig4.upstream.run_real_trace_matrix import MODEL_CHECKPOINT_PATH


FIRST_DIR = OUT_DIR / "first_pass_v1"
DATA_DIR = FIRST_DIR / "plot_data"
LOW_COLOR = "#007C83"
HIGH_COLOR = "#D55E00"
NEUTRAL = "#4B5563"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIRST_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def pooled_interval(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x) & (x > 0)]
    return tuple(float(v) for v in np.quantile(x, [0.05, 0.5, 0.95])) if x.size else (math.nan,) * 3


def select_hand_audit_units(units: pd.DataFrame) -> np.ndarray:
    """Predeclared coverage: SF extremes/median plus available censoring cases."""
    chosen: list[int] = []
    for group in ("low_sf", "high_sf"):
        frame = units.loc[units.figure4_sf_group == group].sort_values("existing_sf_pref_cpd")
        interior = frame.loc[~frame.tf_peak_boundary]
        if len(interior):
            for quantile in (0.0, 0.5, 1.0):
                target = float(interior.existing_sf_pref_cpd.quantile(quantile))
                index = int((interior.existing_sf_pref_cpd - target).abs().idxmin())
                chosen.append(int(units.loc[index, "unit_index"]))
        for censoring in ("left", "right"):
            candidates = frame.loc[frame.tf_peak_censoring == censoring]
            if len(candidates):
                chosen.append(int(candidates.iloc[len(candidates) // 2].unit_index))
        # Fill deterministically to five using SF-ordered unused units.
        for unit_id in frame.unit_index.astype(int):
            if sum(units.set_index("unit_index").loc[x, "figure4_sf_group"] == group for x in chosen) >= 5:
                break
            if int(unit_id) not in chosen:
                chosen.append(int(unit_id))
    # Retain exactly five per historical group.
    out: list[int] = []
    for group in ("low_sf", "high_sf"):
        group_ids = [x for x in chosen if units.set_index("unit_index").loc[x, "figure4_sf_group"] == group]
        out.extend(group_ids[:5])
    return np.asarray(out, dtype=int)


def build_q_audit() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    analysis_dir = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/analysis"
    probe_dir = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/grating_probe"
    units = pd.read_csv(analysis_dir / "per_unit_tuning.csv").sort_values("unit_index").reset_index(drop=True)
    identities = pd.read_csv(probe_dir / "rr100_unit_identity.csv").sort_values("unit_index").reset_index(drop=True)
    traces = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    metrics = np.load(analysis_dir / "trajectory_unit_metrics.npz")
    projected = np.asarray(metrics["projected_speed_deg_s"], float)
    drift = np.asarray(metrics["drift_only"], bool)
    fit_q = np.asarray(metrics["q"], float)
    expected_q = projected * units.existing_sf_pref_cpd.to_numpy(float)[None] / units.tf_pref_hz.to_numpy(float)[None]
    q_arithmetic_max_error = float(np.nanmax(np.abs(fit_q - expected_q)))
    expected_q_relative_error = float(
        np.nanmax(np.abs(fit_q - expected_q) / np.maximum(np.abs(expected_q), 1e-12))
    )
    units["predicted_speed_nearest_probe_deg_s"] = units.tf_pref_hz / units.nearest_probe_sf_cpd

    support_rows: list[dict] = []
    selected_controlled = pd.read_csv(
        ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling/selected_traces.csv"
    ).trace_bank_index.to_numpy(int)
    for unit_idx, row in units.iterrows():
        if bool(row.tf_peak_boundary):
            for condition in ("drift_only", "microsaccade", "controlled_scaling"):
                support_rows.append({
                    "unit_index": int(row.unit_index), "sf_group": row.figure4_sf_group,
                    "condition": condition, "tf_censored": True,
                    "q_p05": np.nan, "q_median": np.nan, "q_p95": np.nan, "brackets_q1": False,
                })
            continue
        values_by_condition = {
            "drift_only": fit_q[drift, unit_idx],
            "microsaccade": fit_q[~drift, unit_idx],
            "controlled_scaling": (
                projected[selected_controlled, unit_idx, None]
                * SCALE_FACTORS[None]
                / float(row.predicted_speed_deg_s)
            ).ravel(),
        }
        for condition, values in values_by_condition.items():
            p05, median, p95 = pooled_interval(values)
            support_rows.append({
                "unit_index": int(row.unit_index), "sf_group": row.figure4_sf_group,
                "condition": condition, "tf_censored": False,
                "q_p05": p05, "q_median": median, "q_p95": p95,
                "brackets_q1": bool(p05 <= 1.0 <= p95),
            })
    support = pd.DataFrame(support_rows)
    support.to_csv(DATA_DIR / "q_support.csv", index=False)

    aperture_extent_deg = (IMAGE_SIZE - 1) / GRATING_PPD
    two_sigma_extent_deg = 2.0 * WINDOW_SIGMA_FRAC * IMAGE_SIZE / GRATING_PPD
    aperture = pd.DataFrame({
        "spatial_cpd": SPATIAL_CPDS,
        "wavelength_deg": 1.0 / SPATIAL_CPDS,
        "sample_center_extent_deg": aperture_extent_deg,
        "gaussian_2sigma_extent_deg": two_sigma_extent_deg,
        "cycles_across_sample_centers": SPATIAL_CPDS * aperture_extent_deg,
        "cycles_across_gaussian_2sigma": SPATIAL_CPDS * two_sigma_extent_deg,
    })
    aperture.to_csv(DATA_DIR / "grating_aperture_identifiability.csv", index=False)

    hand_ids = select_hand_audit_units(units)
    drift_ids = np.flatnonzero(drift)
    median_path = float(np.median(traces.loc[drift, "rendered_path_length_arcmin"]))
    trace_id = int(drift_ids[np.argmin(np.abs(traces.loc[drift, "rendered_path_length_arcmin"].to_numpy(float) - median_path))])
    hand = units.loc[units.unit_index.isin(hand_ids)].merge(identities, on=["unit_index", "unit_label"], how="left")
    hand["trajectory_id"] = trace_id
    hand["raw_2d_mean_speed_deg_s"] = float(traces.loc[trace_id, "rendered_speed_mean_deg_s"])
    hand["projected_speed_deg_s"] = [projected[trace_id, int(x)] for x in hand.unit_index]
    hand["q_recomputed"] = hand.existing_sf_pref_cpd * hand.projected_speed_deg_s / hand.tf_pref_hz
    hand["q_nearest_probe_sf"] = hand.nearest_probe_sf_cpd * hand.projected_speed_deg_s / hand.tf_pref_hz
    hand["predicted_speed_nearest_probe_deg_s"] = hand.tf_pref_hz / hand.nearest_probe_sf_cpd
    hand["units_check"] = "cycles/deg × deg/s = cycles/s = Hz"
    hand.to_csv(DATA_DIR / "q_hand_audit_10_units.csv", index=False)

    # Figure A: actual aperture, tuning-derived speed, and sampled q support.
    fig = plt.figure(figsize=(12.4, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(4, 5, height_ratios=[1.15, 1.0, 1.0, 2.25])
    shown_sfs = [0.0125, 0.05, 0.2, 0.8, 3.2]
    for col, sf in enumerate(shown_sfs):
        ax = fig.add_subplot(grid[0, col])
        frame = make_grating_movie(
            orientation_deg=0.0, spatial_cpd=sf, temporal_hz=0.0, phase_rad=0.0, duration_s=0.1
        )[-1]
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        cycles = sf * aperture_extent_deg
        ax.set_title(f"{sf:g} cpd\n{cycles:.3g} cycles across aperture", fontsize=9)
        ax.set_axis_off()

    ax = fig.add_subplot(grid[1, :2])
    interior = ~units.tf_peak_boundary.to_numpy(bool)
    for group, color, label in (("low_sf", LOW_COLOR, "low SF"), ("high_sf", HIGH_COLOR, "high SF")):
        mask = (units.figure4_sf_group == group).to_numpy() & interior
        ax.scatter(units.loc[mask, "existing_sf_pref_cpd"], units.loc[mask, "tf_pref_hz"], s=24, color=color, label=label)
        bound = (units.figure4_sf_group == group).to_numpy() & ~interior
        ax.scatter(units.loc[bound, "existing_sf_pref_cpd"], units.loc[bound, "tf_pref_hz"], s=30, facecolors="none", edgecolors=color, marker="s")
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="fitted SF (cpd)", ylabel="TF maximum (Hz)", title="B  Output SF and TF estimates")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 2:])
    bins = np.logspace(-1.5, 4.5, 34)
    for group, color, label in (("low_sf", LOW_COLOR, "low SF"), ("high_sf", HIGH_COLOR, "high SF")):
        vals = units.loc[(units.figure4_sf_group == group) & ~units.tf_peak_boundary, "predicted_speed_deg_s"]
        ax.hist(vals, bins=bins, histtype="step", lw=2, color=color, label=label)
    drift_speed = traces.loc[drift, "rendered_speed_mean_deg_s"].to_numpy(float)
    ms_speed = traces.loc[~drift, "rendered_speed_mean_deg_s"].to_numpy(float)
    for vals, color, label in ((drift_speed, "#6B7280", "drift mean speed"), (ms_speed, "#8B5CF6", "microsaccade-window mean speed")):
        ax.axvspan(np.quantile(vals, .05), np.quantile(vals, .95), color=color, alpha=.12)
        ax.axvline(np.median(vals), color=color, lw=1.5, label=label)
    ax.set(xscale="log", xlabel=r"speed (deg s$^{-1}$)", ylabel="unit count", title=r"C  $v^*=f_t^*/f_s^*$ versus measured motion")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = fig.add_subplot(grid[2, :])
    curves = pd.read_csv(analysis_dir / "temporal_tuning_curves.csv")
    for group, color in (("low_sf", LOW_COLOR), ("high_sf", HIGH_COLOR)):
        candidates = units.loc[(units.figure4_sf_group == group) & ~units.tf_peak_boundary].copy()
        center_sf = np.median(np.log2(candidates.existing_sf_pref_cpd))
        center_tf = np.median(np.log2(candidates.tf_pref_hz))
        candidates["distance"] = (
            (np.log2(candidates.existing_sf_pref_cpd) - center_sf) ** 2
            + (np.log2(candidates.tf_pref_hz) - center_tf) ** 2
        )
        unit_id = int(candidates.sort_values(["distance", "unit_index"]).iloc[0].unit_index)
        c = curves.loc[curves.unit_index == unit_id].sort_values("temporal_hz")
        y = c.response_amplitude_rms.to_numpy(float)
        ax.plot(c.temporal_hz, y / np.nanmax(y), "o-", color=color, label=f"{group.replace('_', ' ')} representative u{unit_id:03d}")
    ax.set(xscale="log", xlabel="temporal frequency (Hz)", ylabel="normalized phase-RMS response", title="D  Representative output temporal tuning (existing 3-s probe)")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[3, :])
    natural = support.loc[support.condition == "drift_only"].merge(
        units[["unit_index", "existing_sf_pref_cpd"]], on="unit_index"
    ).sort_values(["tf_censored", "existing_sf_pref_cpd", "unit_index"]).reset_index(drop=True)
    for y, row in natural.iterrows():
        color = LOW_COLOR if row.sf_group == "low_sf" else HIGH_COLOR
        if row.tf_censored:
            ax.scatter([1e-4], [y], marker="x", s=18, color=color)
        else:
            ax.plot([row.q_p05, row.q_p95], [y, y], color=color, lw=1)
            ax.scatter([row.q_median], [y], color=color, s=10)
    ax.axvline(1.0, color="black", lw=1.2, ls="--")
    ax.set(xscale="log", xlim=(1e-4, 1e3), xlabel=r"sampled $q=f_s|v_\perp|/f_t^*$ (drift only)", ylabel="units sorted by fitted SF", title="E  Natural trajectories barely test a common q range; × marks censored TF")
    ax.set_yticks([])
    fig.suptitle("Figure A — Why the output-unit q coordinate is not yet a valid mechanism test", fontsize=15, weight="bold")
    save_figure(fig, "fig_A_q_audit")

    summary: dict[str, object] = {
        "q_arithmetic_max_abs_error": q_arithmetic_max_error,
        "q_arithmetic_max_relative_error": expected_q_relative_error,
        "aperture_extent_sample_centers_deg": aperture_extent_deg,
        "aperture_gaussian_2sigma_extent_deg": two_sigma_extent_deg,
        "n_tf_interior": int((~units.tf_peak_boundary).sum()),
        "n_tf_censored": int(units.tf_peak_boundary.sum()),
        "hand_audit_unit_ids": hand_ids.tolist(),
        "hand_audit_trajectory_id": trace_id,
        "median_predicted_speed_fitted_sf_all_low_deg_s": float(units.loc[units.figure4_sf_group == "low_sf", "predicted_speed_deg_s"].median()),
        "median_predicted_speed_fitted_sf_all_high_deg_s": float(units.loc[units.figure4_sf_group == "high_sf", "predicted_speed_deg_s"].median()),
        "median_predicted_speed_fitted_sf_interior_low_deg_s": float(units.loc[(units.figure4_sf_group == "low_sf") & ~units.tf_peak_boundary, "predicted_speed_deg_s"].median()),
        "median_predicted_speed_fitted_sf_interior_high_deg_s": float(units.loc[(units.figure4_sf_group == "high_sf") & ~units.tf_peak_boundary, "predicted_speed_deg_s"].median()),
        "median_predicted_speed_nearest_sf_interior_low_deg_s": float(units.loc[(units.figure4_sf_group == "low_sf") & ~units.tf_peak_boundary, "predicted_speed_nearest_probe_deg_s"].median()),
        "median_predicted_speed_nearest_sf_interior_high_deg_s": float(units.loc[(units.figure4_sf_group == "high_sf") & ~units.tf_peak_boundary, "predicted_speed_nearest_probe_deg_s"].median()),
    }
    for condition in ("drift_only", "microsaccade", "controlled_scaling"):
        for group in ("low_sf", "high_sf"):
            x = support.loc[(support.condition == condition) & (support.sf_group == group) & ~support.tf_censored]
            summary[f"{condition}_{group}_identifiable_n"] = int(len(x))
            summary[f"{condition}_{group}_fraction_bracketing_q1"] = float(x.brackets_q1.mean()) if len(x) else math.nan
    for group in ("low_sf", "high_sf"):
        vals = fit_q[:, units.figure4_sf_group.to_numpy() == group]
        summary[f"natural_{group}_pooled_q_p05_p95"] = list(pooled_interval(vals)[::2])
    low_range = summary["natural_low_sf_pooled_q_p05_p95"]
    high_range = summary["natural_high_sf_pooled_q_p05_p95"]
    summary["natural_common_pooled_q_range"] = [max(low_range[0], high_range[0]), min(low_range[1], high_range[1])]
    write_json(FIRST_DIR / "q_audit_statistics.json", summary)
    return summary


def effective_frontend_weights() -> np.ndarray:
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    original = state["model.frontend.temporal_conv.conv.parametrizations.weight.original"].detach().cpu()
    window = _get_window("hann", original.shape[2], power=0.25).view(1, 1, -1, 1, 1)
    return (original * window).numpy()[:, 0, :, 0, 0]


def render_history_movie(patch: np.ndarray, history: np.ndarray) -> np.ndarray:
    image = _standardize_uint_like(patch)
    repeated = np.broadcast_to(image[None], (len(history), *image.shape)).copy()
    eye = torch.from_numpy(np.asarray(history, np.float32))
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            torch.from_numpy(repeated), eye_norm, out_size=OUT_SIZE, scale_factor=1.0, torch=torch
        )
    return shifted.numpy().astype(np.float32)


def trajectory_aligned_slice(movie: np.ndarray, scored_trace: np.ndarray) -> np.ndarray:
    centered = scored_trace - np.mean(scored_trace, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    eye_axis = vh[0]
    pixel_axis = np.asarray([eye_axis[0], -eye_axis[1]], float)
    pixel_axis /= np.linalg.norm(pixel_axis)
    half = 0.46 * min(movie.shape[1:])
    positions = np.linspace(-half, half, movie.shape[2])
    center = 0.5 * (np.asarray(movie.shape[1:]) - 1)
    x = center[1] + positions * pixel_axis[0]
    y = center[0] + positions * pixel_axis[1]
    return np.stack([map_coordinates(frame, [y, x], order=1, mode="nearest") for frame in movie])


def xt_power(xt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(xt, float) - np.mean(xt, axis=0, keepdims=True)
    data *= np.hanning(data.shape[0])[:, None] * np.hanning(data.shape[1])[None]
    ft = np.fft.fftfreq(data.shape[0], d=1.0 / 120.0)
    fs = np.fft.fftfreq(data.shape[1], d=1.0 / PPD)
    power = np.abs(np.fft.fft2(data)) ** 2
    tmask = ft >= 0
    return np.fft.fftshift(fs), ft[tmask], np.fft.fftshift(power[tmask], axes=1)


def build_retinal_frontend_audit() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    traces = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    coherence = pd.to_numeric(images.image_orientation_coherence, errors="coerce").to_numpy(float)
    image_candidates = np.flatnonzero(np.isfinite(coherence) & (coherence >= 0.2))
    image_target = float(np.median(coherence[image_candidates]))
    image_row_idx = int(image_candidates[np.argmin(np.abs(coherence[image_candidates] - image_target))])
    drift = traces.rendered_n_microsaccade_events.to_numpy(float) == 0
    path = traces.rendered_path_length_arcmin.to_numpy(float)
    trace_candidates = np.flatnonzero(drift)
    trace_target = float(np.median(path[trace_candidates]))
    trace_id = int(trace_candidates[np.argmin(np.abs(path[trace_candidates] - trace_target))])
    image_id = int(images.iloc[image_row_idx].image_index)
    patch, _ = extract_patch(images.iloc[image_row_idx], canvas_cache={}, patch_size_px=540)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base = np.asarray(archive["true_history_xy"][trace_id], np.float32)
    e0 = base[N_PRECEDING].copy()
    scored_displacement = base[N_PRECEDING:] - e0
    scales = np.asarray([0.0, 0.5, 1.0, 2.0], float)
    movies: dict[float, np.ndarray] = {}
    slices: dict[float, np.ndarray] = {}
    powers: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scale in scales:
        history = base.copy()
        history[N_PRECEDING:] = e0 + scale * scored_displacement
        movie = render_history_movie(patch, history)[N_PRECEDING:]
        movies[float(scale)] = movie
        slices[float(scale)] = trajectory_aligned_slice(movie, history[N_PRECEDING:])
        powers[float(scale)] = xt_power(slices[float(scale)])
    np.savez_compressed(
        DATA_DIR / "retinal_xt_pilot.npz",
        scales=scales,
        xt_slices=np.stack([slices[float(s)] for s in scales]),
        representative_image_id=image_id,
        representative_trace_id=trace_id,
        scored_trace_xy=base[N_PRECEDING:],
    )

    weights = effective_frontend_weights()
    freq_dense = np.linspace(0, 60, 1201)
    response = np.asarray([
        np.abs(np.sum(kernel[None] * np.exp(-2j * np.pi * freq_dense[:, None] * np.arange(len(kernel))[None] / 120.0), axis=1))
        for kernel in weights
    ])
    frontend_rows = []
    for channel, kernel in enumerate(weights):
        peak_idx = int(np.argmax(response[channel]))
        half = 0.5 * response[channel, peak_idx]
        above = freq_dense[response[channel] >= half]
        frontend_rows.append({
            "channel": channel, "dc_gain": float(abs(kernel.sum())),
            "peak_frequency_hz": float(freq_dense[peak_idx]),
            "half_height_low_hz": float(above.min()), "half_height_high_hz": float(above.max()),
            "kernel_sum": float(kernel.sum()), "kernel_l1": float(np.abs(kernel).sum()),
        })
    pd.DataFrame(frontend_rows).to_csv(DATA_DIR / "frontend_filter_summary.csv", index=False)
    pd.DataFrame(
        [{"channel": c, "lag_ms": -1000.0 * lag / 120.0, "weight": weights[c, lag]} for c in range(len(weights)) for lag in range(weights.shape[1])]
    ).to_csv(DATA_DIR / "frontend_kernels.csv", index=False)
    pd.DataFrame(
        [{"channel": c, "temporal_hz": f, "magnitude": response[c, i]} for c in range(len(weights)) for i, f in enumerate(freq_dense)]
    ).to_csv(DATA_DIR / "frontend_frequency_response.csv", index=False)

    fig = plt.figure(figsize=(13.2, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.2, 1.2, 1.05])
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(_standardize_uint_like(patch), cmap="gray")
    ax.set_title(f"A  Representative image {image_id}")
    ax.set_axis_off()
    ax = fig.add_subplot(grid[0, 1])
    trace = base[N_PRECEDING:]
    ax.plot(60 * trace[:, 0], 60 * trace[:, 1], color=NEUTRAL, lw=1.5)
    ax.scatter(60 * trace[0, 0], 60 * trace[0, 1], color=LOW_COLOR, s=25, label="start")
    ax.set(aspect="equal", xlabel="horizontal eye position (arcmin)", ylabel="vertical eye position (arcmin)", title=f"B  Drift trajectory {trace_id}")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[0, 2:])
    temporal_rows = []
    temporal_by_scale = {}
    for scale in scales:
        fs, ft, p = powers[float(scale)]
        temporal = p.sum(axis=1)
        temporal[0] = 0.0
        temporal_by_scale[float(scale)] = temporal
    common_power = max(float(np.max(x)) for x in temporal_by_scale.values())
    for scale in scales:
        fs, ft, _ = powers[float(scale)]
        temporal = temporal_by_scale[float(scale)] / max(common_power, 1e-12)
        ax.plot(ft, temporal, label=f"{scale:g}×")
        temporal_rows.extend({"scale": scale, "temporal_hz": f, "power_relative_to_common_max": v} for f, v in zip(ft, temporal))
    pd.DataFrame(temporal_rows).to_csv(DATA_DIR / "retinal_temporal_power.csv", index=False)
    ax.set(xlim=(0, 60), xlabel="temporal frequency (Hz)", ylabel="fraction of trajectory-aligned slice power", title="C  More motion spreads retinal power across temporal frequency")
    ax.legend(frameon=False, ncol=4, fontsize=8)

    for col, scale in enumerate(scales):
        ax = fig.add_subplot(grid[1, col])
        xt = slices[float(scale)]
        ax.imshow(xt.T, cmap="gray", aspect="auto", origin="lower", extent=[0, 1000 * (len(xt) - 1) / 120, -0.5 * len(xt[0]) / PPD, 0.5 * len(xt[0]) / PPD])
        ax.set_title(f"D{col + 1}  {scale:g}× movement")
        ax.set_xlabel("time (ms)")
        if col == 0:
            ax.set_ylabel("trajectory-aligned position (deg)")

    common_spectrum_max = max(float(np.max(powers[scale][2])) for scale in (0.5, 1.0, 2.0))
    common_log_min = math.log10(max(common_spectrum_max * 1e-7, 1e-30))
    common_log_max = math.log10(max(common_spectrum_max, 1e-30))
    for col, scale in enumerate((0.5, 1.0, 2.0)):
        ax = fig.add_subplot(grid[2, col])
        fs, ft, p = powers[scale]
        select_f = np.abs(fs) <= 12
        image = np.log10(p[:, select_f] + common_spectrum_max * 1e-8)
        ax.imshow(image, origin="lower", aspect="auto", extent=[fs[select_f][0], fs[select_f][-1], ft[0], ft[-1]], cmap="magma", vmin=common_log_min, vmax=common_log_max)
        ax.set(xlabel="spatial frequency (cpd)", title=f"E{col + 1}  {scale:g}× input power")
        if col == 0:
            ax.set_ylabel("temporal frequency (Hz)")
    ax = fig.add_subplot(grid[2, 3])
    colors = plt.cm.viridis(np.linspace(.15, .9, len(weights)))
    lags_ms = -1000 * np.arange(weights.shape[1]) / 120.0
    for channel, (kernel, color) in enumerate(zip(weights, colors)):
        ax.plot(lags_ms, kernel, "o-", ms=3, color=color, label=f"channel {channel + 1}")
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="input lag (ms; 0=current)", ylabel="effective weight", title="F  Learned temporal kernels")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(grid[3, :2])
    for channel, color in enumerate(colors):
        ax.plot(freq_dense, response[channel] / max(response[channel].max(), 1e-12), color=color, label=f"channel {channel + 1}")
    ax.set(xlim=(0, 60), xlabel="temporal frequency (Hz)", ylabel="normalized magnitude", title="G  Learned frontend frequency responses")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = fig.add_subplot(grid[3, 2:])
    fs, ft, p = powers[1.0]
    input_temporal = p.sum(axis=1)
    for channel, color in enumerate(colors):
        h = np.interp(ft, freq_dense, response[channel])
        transmitted = input_temporal * h**2
        transmitted /= max(transmitted.sum(), 1e-12)
        ax.plot(ft, transmitted, color=color, label=f"channel {channel + 1}")
    ax.set(xlim=(0, 60), xlabel="temporal frequency (Hz)", ylabel="normalized input power × |H(f)|²", title="H  Motion-created power transmitted at 1×")
    fig.suptitle("Figure B — Retinal motion creates temporal structure that the learned frontend filters", fontsize=15, weight="bold")
    save_figure(fig, "fig_B_retinal_to_frontend")

    summary = {
        "representative_selection_rule": "median contour-qualified image coherence; median drift-only path length",
        "representative_image_id": image_id,
        "representative_trace_id": trace_id,
        "trajectory_path_arcmin": float(traces.loc[trace_id, "rendered_path_length_arcmin"]),
        "frontend_channels": frontend_rows,
        "retinal_power_scope": "40-sample scored interval; true prefix preserved but not included in Fourier estimate",
        "controlled_scaling": "true prefix unchanged; scored displacement relative to e0 scaled",
    }
    write_json(FIRST_DIR / "retinal_frontend_statistics.json", summary)
    return summary


def main() -> int:
    FIRST_DIR.mkdir(parents=True, exist_ok=True)
    q = build_q_audit()
    retinal = build_retinal_frontend_audit()
    write_json(
        FIRST_DIR / "static_audit_manifest.json",
        {
            "analysis": "fig4_first_pass_static_mechanism_audits",
            "status": "pilot",
            "q": q,
            "retinal_frontend": retinal,
            "checkpoint": MODEL_CHECKPOINT_PATH,
            "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "figures": ["fig_A_q_audit", "fig_B_retinal_to_frontend"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
