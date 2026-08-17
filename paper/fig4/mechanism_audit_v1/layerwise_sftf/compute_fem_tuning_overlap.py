#!/usr/bin/env python3
"""Bridge corrected FEM retinal spectra to measured layerwise SF x TF tuning."""

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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
PHASE_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
IMAGE_POWER = ROOT / "outputs/figures/fig4/mechanism_audit_v1/targeted_frontend_v1/exact_arrays/natural_image_mode_power.npz"
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=float)
FRAME_RATE_HZ = 120.0
LOW = "#007C83"
HIGH = "#D55E00"
EPS = 1e-30
SPECTRUM_METHOD_VERSION = "trajectory_phase_hann_twosided_v2_no_velocity_alias"


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


def nearest_axis_orientation(kxy: np.ndarray, orientations_deg: np.ndarray) -> np.ndarray:
    normal_deg = np.degrees(np.arctan2(kxy[:, 1], kxy[:, 0]))
    bar_deg = np.mod(normal_deg - 90.0, 180.0)
    distance = np.abs(((bar_deg[:, None] - orientations_deg[None] + 90.0) % 180.0) - 90.0)
    return np.argmin(distance, axis=1)


def log_bin_indices(radial: np.ndarray, spatial_cpd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.log2(spatial_cpd)
    edges = np.concatenate(
        ([centers[0] - 0.5 * (centers[1] - centers[0])], 0.5 * (centers[:-1] + centers[1:]), [centers[-1] + 0.5 * (centers[-1] - centers[-2])])
    )
    index = np.digitize(np.log2(radial), edges) - 1
    keep = (index >= 0) & (index < len(spatial_cpd))
    return index, keep


def log_edges(centers_linear: np.ndarray) -> np.ndarray:
    centers = np.log2(np.asarray(centers_linear, dtype=float))
    return np.concatenate(
        (
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        )
    )


def instantaneous_temporal_frequency(
    kxy_cycles_per_degree: np.ndarray,
    velocity_degrees_per_second: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Return |k dot v| in Hz for mode x trajectory x time arrays."""
    return float(scale) * np.abs(
        np.einsum(
            "md,ntd->nmt",
            np.asarray(kxy_cycles_per_degree, dtype=float),
            np.asarray(velocity_degrees_per_second, dtype=float),
            optimize=True,
        )
    )


def continuous_phase_power(signal: np.ndarray, temporal_hz: np.ndarray) -> np.ndarray:
    """Two-sided non-DC power at requested positive frequencies.

    signal is trajectory x mode x time.  A Hann taper and mean removal match
    the finite 40-sample Figure 4 interval without pretending it is stationary.
    """
    n_time = signal.shape[-1]
    centered = signal - signal.mean(axis=-1, keepdims=True)
    window = np.hanning(n_time)
    window /= math.sqrt(float(np.sum(window**2)))
    time_s = np.arange(n_time, dtype=float) / FRAME_RATE_HZ
    positive = np.exp(-2j * np.pi * temporal_hz[:, None] * time_s[None]) * window[None]
    negative = np.conj(positive)
    cpos = np.einsum("nmt,ft->nmf", centered, positive, optimize=True)
    cneg = np.einsum("nmt,ft->nmf", centered, negative, optimize=True)
    return (np.abs(cpos) ** 2 + np.abs(cneg) ** 2).mean(axis=0)


def compute_retinal_spectrum(layout: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cache = RAW / "corrected_fem_retinal_sftf_spectrum.npz"
    if cache.exists():
        with np.load(cache) as archive:
            cached = {key: np.asarray(archive[key]) for key in archive.files}
        version = str(cached.get("method_version", np.asarray("")).item())
        if version == SPECTRUM_METHOD_VERSION:
            return cached
        # The original cache already contained the correct finite-trajectory
        # carrier spectrum as a secondary diagnostic, but exposed the
        # instantaneous |k.v| histogram as `retinal_motion_power`.  Migrate the
        # scientifically valid field in place rather than silently reusing the
        # invalid primary alias.
        if "finite_window_phase_modulation_power" in cached:
            phase = np.asarray(cached["finite_window_phase_modulation_power"])
            instantaneous = np.asarray(
                cached.get("instantaneous_ft_occupancy", cached["retinal_motion_power"])
            )
            cached.update(
                {
                    "retinal_motion_power": phase,
                    "trajectory_phase_modulation_power": phase,
                    "deprecated_instantaneous_velocity_histogram": instantaneous,
                    "retinal_in_grid_power": phase.sum(axis=(1, 2, 3)),
                    "method_version": np.asarray(SPECTRUM_METHOD_VERSION),
                    "definition": np.asarray(
                        "primary: mean selected-image Fourier power times the selected-trace "
                        "two-sided Hann-tapered finite-trajectory spectrum of demeaned "
                        "exp(-i2pi k dot scale[e(t)-e(0)]); the instantaneous |k dot v| "
                        "histogram is retained only as a deprecated diagnostic and is not a PSD"
                    ),
                }
            )
            cached.pop("instantaneous_ft_occupancy", None)
            np.savez_compressed(cache, **cached)
            return cached
    sf = layout["spatial_cpd"].astype(float)
    tf = layout["temporal_hz"].astype(float)
    orientation = layout["orientation_deg"].astype(float)
    with np.load(PHASE_OUT / "exact_arrays/exact_phase_spatial_metrics.npz") as archive:
        image_ids = np.asarray(archive["selected_image_index"], dtype=int)
        trace_ids = np.asarray(archive["selected_trace_index"], dtype=int)
    with np.load(IMAGE_POWER) as archive:
        all_image_ids = np.asarray(archive["image_ids"], dtype=int)
        image_matrix = np.asarray(archive["per_image_mode_power"], dtype=float)
        kxy = np.asarray(archive["kxy"], dtype=float)
    image_rows = [int(np.flatnonzero(all_image_ids == value)[0]) for value in image_ids]
    mean_image_power = image_matrix[image_rows].mean(axis=0)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        scored = np.asarray(archive["stored_scored_trace_xy"][trace_ids], dtype=float)
    displacement = scored - scored[:, :1]
    velocity = np.diff(scored, axis=1) * FRAME_RATE_HZ
    radial = np.linalg.norm(kxy, axis=1)
    sf_bin, keep = log_bin_indices(radial, sf)
    ori_bin = nearest_axis_orientation(kxy, orientation)
    # The retinal signal carried by image Fourier mode k is
    # I_k exp(-i 2pi k.X(t)).  Its temporal spectrum depends on the complete
    # displacement trajectory and cannot be recovered from a histogram of
    # instantaneous |k.v(t)| values.  We retain that histogram only as a
    # deprecated diagnostic so old results can be identified during audits.
    kinematic = np.zeros((len(SCALES), len(sf), len(tf), len(orientation)), dtype=np.float64)
    phase_spectrum = np.zeros_like(kinematic)
    temporal_edges = log_edges(tf)
    kinematic_total_weight = np.zeros(len(SCALES), dtype=np.float64)
    kinematic_in_grid_weight = np.zeros(len(SCALES), dtype=np.float64)
    chunk = 256
    kept_indices = np.flatnonzero(keep)
    for start in range(0, len(kept_indices), chunk):
        mode_ids = kept_indices[start : start + chunk]
        dot = np.einsum("md,ntd->nmt", kxy[mode_ids], displacement, optimize=True)
        instantaneous_hz = instantaneous_temporal_frequency(kxy[mode_ids], velocity)
        for scale_i, scale in enumerate(SCALES):
            phase_signal = np.exp(-2j * np.pi * float(scale) * dot)
            phase_power = continuous_phase_power(phase_signal, tf)
            weighted = phase_power * mean_image_power[mode_ids, None]
            for local_i, mode_id in enumerate(mode_ids):
                phase_spectrum[scale_i, sf_bin[mode_id], :, ori_bin[mode_id]] += weighted[local_i]
                weight = float(mean_image_power[mode_id])
                hz = float(scale) * instantaneous_hz[:, local_i].reshape(-1)
                kinematic_total_weight[scale_i] += weight * len(hz)
                positive = hz > 0
                if not np.any(positive):
                    continue
                temporal_bin = np.digitize(np.log2(hz[positive]), temporal_edges) - 1
                counts = np.bincount(
                    temporal_bin[(temporal_bin >= 0) & (temporal_bin < len(tf))],
                    minlength=len(tf),
                )[: len(tf)]
                kinematic[scale_i, sf_bin[mode_id], :, ori_bin[mode_id]] += weight * counts
                kinematic_in_grid_weight[scale_i] += weight * counts.sum()
        if (start // chunk + 1) % 10 == 0:
            print(f"FEM spectrum exact modes {min(start + chunk, len(kept_indices))}/{len(kept_indices)}", flush=True)
    np.savez_compressed(
        cache,
        retinal_motion_power=phase_spectrum.astype(np.float32),
        trajectory_phase_modulation_power=phase_spectrum.astype(np.float32),
        deprecated_instantaneous_velocity_histogram=kinematic.astype(np.float32),
        finite_window_phase_modulation_power=phase_spectrum.astype(np.float32),
        scales=SCALES,
        spatial_cpd=sf,
        temporal_hz=tf,
        orientation_deg=orientation,
        selected_image_index=image_ids,
        selected_trace_index=trace_ids,
        n_exact_fourier_modes=np.asarray(len(kept_indices)),
        kinematic_total_weight=kinematic_total_weight,
        kinematic_in_grid_weight=kinematic_in_grid_weight,
        retinal_in_grid_power=phase_spectrum.sum(axis=(1, 2, 3)),
        method_version=np.asarray(SPECTRUM_METHOD_VERSION),
        definition=np.asarray("primary: mean selected-image Fourier power times selected-trace two-sided Hann-tapered finite-trajectory spectrum of demeaned exp(-i2pi k dot scale[e(t)-e(0)]); the instantaneous |k dot v| histogram is retained only as a deprecated diagnostic and is not a PSD"),
    )
    with np.load(cache) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def phase_aggregate(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "f0_signed_mean":
        return np.nanmean(values, axis=-1)
    return np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2, axis=-1))


def tuning_for_overlap(oriented: np.ndarray, metric: str) -> np.ndarray:
    value = np.asarray(oriented, dtype=float)
    if metric == "f0_signed_mean":
        value = np.clip(value - np.nanmin(value, axis=(1, 2, 3), keepdims=True), 0, None)
    else:
        value = np.clip(value, 0, None)
    return value / np.maximum(value.sum(axis=(1, 2, 3), keepdims=True), EPS)


def compute_overlaps(layout: dict[str, np.ndarray], retinal: np.ndarray) -> pd.DataFrame:
    stages = layout["stages"].astype(str)
    starts = layout["stage_starts"].astype(int)
    stops = layout["stage_stops"].astype(int)
    rows = []
    for metric in ("f0_signed_mean", "f1_amplitude"):
        array = np.load(RAW / f"{metric}.npy", mmap_mode="r")
        for stage_i, stage in enumerate(stages):
            oriented = phase_aggregate(array[starts[stage_i] : stops[stage_i]], metric)
            tuning = tuning_for_overlap(oriented, metric)
            overlap = np.einsum("stfo,ctfo->cs", retinal, tuning, optimize=True)
            norm = overlap / np.maximum(overlap.max(axis=1, keepdims=True), EPS)
            optimum = np.argmax(overlap[:, 1:], axis=1) + 1
            for channel in range(len(overlap)):
                for scale_i, scale in enumerate(SCALES):
                    rows.append(
                        {
                            "stage": stage,
                            "channel": channel,
                            "response_metric": metric,
                            "scale": float(scale),
                            "retinal_tuning_overlap": float(overlap[channel, scale_i]),
                            "normalized_overlap": float(norm[channel, scale_i]),
                            "predicted_optimum_scale": float(SCALES[optimum[channel]]),
                            "optimum_censoring": "right" if optimum[channel] == len(SCALES) - 1 else "none",
                            "spectrum_method_version": SPECTRUM_METHOD_VERSION,
                        }
                    )
            print(f"FEM overlap {metric}: {stage}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "layerwise_corrected_fem_tuning_overlap.csv.gz", index=False)
    return result


def quartile_prediction(layout: dict[str, np.ndarray], retinal: np.ndarray, overlaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quartiles = pd.read_csv(DATA / "rr100_f0_sf_quartiles.csv")
    rr_overlap = overlaps.loc[
        overlaps.stage.eq("rr100") & overlaps.response_metric.eq("f0_signed_mean")
    ].merge(quartiles[["channel", "sf_quartile"]], on="channel", validate="many_to_one")
    predicted = rr_overlap.groupby(["sf_quartile", "scale"], as_index=False).normalized_overlap.mean()
    predicted["spectrum_method_version"] = SPECTRUM_METHOD_VERSION

    with np.load(PHASE_OUT / "exact_arrays/exact_phase_spatial_metrics.npz") as archive:
        ssi = np.asarray(archive["ssi"], dtype=float)
        expected = np.asarray(archive["expected_spikes"], dtype=float)
        scales = np.asarray(archive["scales"], dtype=float)
    actual_rows = []
    unit_rows = []
    for quartile, frame in quartiles.groupby("sf_quartile"):
        units = frame.channel.to_numpy(int)
        numerator = np.sum(ssi[..., units] * expected[..., units], axis=(0, 1, 3))
        denominator = np.sum(expected[..., units], axis=(0, 1, 3))
        curve = numerator / np.maximum(denominator, EPS)
        percent = 100 * (curve - curve[0]) / max(abs(curve[0]), EPS)
        for scale, value, raw in zip(scales, percent, curve):
            actual_rows.append({"sf_quartile": quartile, "scale": float(scale), "ssi_percent_vs_0x": float(value), "ssi": float(raw)})
    for _, row in quartiles.iterrows():
        unit = int(row.channel)
        numerator = np.sum(ssi[..., unit] * expected[..., unit], axis=(0, 1))
        denominator = np.sum(expected[..., unit], axis=(0, 1))
        observed = numerator / np.maximum(denominator, EPS)
        observed_change = observed - observed[0]
        pred = rr_overlap.loc[rr_overlap.channel.eq(unit)].sort_values("scale").normalized_overlap.to_numpy(float)
        if np.std(pred[1:]) > 0 and np.std(observed_change[1:]) > 0:
            rho = float(spearmanr(pred[1:], observed_change[1:]).statistic)
        else:
            rho = math.nan
        unit_rows.append(
            {
                "unit_index": unit,
                "sf_quartile": row.sf_quartile,
                "predicted_overlap_optimum_scale": float(SCALES[1 + np.argmax(pred[1:])]),
                "observed_ssi_optimum_scale": float(scales[np.argmax(observed)]),
                "spearman_predicted_overlap_vs_ssi_change_moving_scales": rho,
            }
        )
    actual = pd.DataFrame(actual_rows)
    unit = pd.DataFrame(unit_rows)
    predicted.to_csv(DATA / "rr100_quartile_fem_overlap_prediction.csv", index=False)
    actual.to_csv(DATA / "rr100_quartile_observed_ssi.csv", index=False)
    unit.to_csv(DATA / "rr100_unit_fem_overlap_vs_ssi.csv", index=False)
    return predicted, actual


def figure_bridge(
    layout: dict[str, np.ndarray], retinal: np.ndarray, predicted: pd.DataFrame, actual: pd.DataFrame
) -> None:
    sf = layout["spatial_cpd"].astype(float)
    tf = layout["temporal_hz"].astype(float)
    collapsed = retinal.sum(axis=-1)
    display = collapsed / np.maximum(collapsed.sum(axis=(1, 2), keepdims=True), EPS)
    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    for col, scale_i in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(grid[0, col])
        value = display[scale_i]
        mesh = ax.pcolormesh(sf, tf, value.T, shading="auto", cmap="magma", norm=matplotlib.colors.LogNorm(vmin=max(float(value[value > 0].min()), 1e-8), vmax=float(value.max())))
        ax.set(xscale="log", yscale="log", xticks=[.4, 1, 4, 16], yticks=[.4, 1, 4, 16, 51.2])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f"{SCALES[scale_i]:g}× corrected FEM")
        ax.set_xlabel("image SF (cpd)")
        if col == 0:
            ax.set_ylabel("generated TF (Hz)")
    fig.colorbar(mesh, ax=[fig.axes[i] for i in range(4)], label="normalized trajectory-phase spectral power", shrink=.8)
    ax = fig.add_subplot(grid[1, :2])
    colors = {"Q1": "#0072B2", "Q2": "#009E73", "Q3": "#CC79A7", "Q4": "#D55E00"}
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        frame = predicted.loc[predicted.sf_quartile.eq(quartile)].sort_values("scale")
        ax.plot(frame.scale, frame.normalized_overlap, marker="o", lw=2, color=colors[quartile], label=quartile)
    ax.set(xlabel="trajectory amplitude", ylabel="mean normalized retinal-spectrum × RR100-tuning overlap", title="Prediction from Rucci transform + grating tuning")
    ax.grid(alpha=.18); ax.legend(frameon=False, ncol=4)
    ax = fig.add_subplot(grid[1, 2:])
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        frame = actual.loc[actual.sf_quartile.eq(quartile)].sort_values("scale")
        ax.plot(frame.scale, frame.ssi_percent_vs_0x, marker="o", lw=2, color=colors[quartile], label=quartile)
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="trajectory amplitude", ylabel="SSI change vs stabilized (%)", title="Observed exact natural-movie SSI")
    ax.grid(alpha=.18); ax.legend(frameon=False, ncol=4)
    fig.suptitle("Eye movements provide the SF→TF input; measured model tuning predicts which channels they engage", fontsize=14, weight="bold")
    export(fig, "figure_3_corrected_fem_to_tuning_to_ssi_bridge")


def main() -> int:
    configure()
    DATA.mkdir(parents=True, exist_ok=True)
    with np.load(RAW / "layout_and_grid.npz") as archive:
        layout = {key: np.asarray(archive[key]) for key in archive.files}
    spectrum = compute_retinal_spectrum(layout)
    retinal = np.asarray(spectrum["retinal_motion_power"], dtype=float)
    overlaps = compute_overlaps(layout, retinal)
    predicted, actual = quartile_prediction(layout, retinal, overlaps)
    figure_bridge(layout, retinal, predicted, actual)
    unit = pd.read_csv(DATA / "rr100_unit_fem_overlap_vs_ssi.csv")
    temporal_marginal = retinal.sum(axis=(1, 3))
    temporal_sum = temporal_marginal.sum(axis=1)
    temporal_centroid = np.divide(
        (temporal_marginal * layout["temporal_hz"][None]).sum(axis=1),
        temporal_sum,
        out=np.full(len(SCALES), np.nan),
        where=temporal_sum > 0,
    )
    temporal_mode = np.full(len(SCALES), np.nan)
    temporal_mode[1:] = layout["temporal_hz"][np.argmax(temporal_marginal[1:], axis=1)]
    deprecated = np.asarray(spectrum["deprecated_instantaneous_velocity_histogram"], dtype=float)
    cosine = []
    for scale_i in range(len(SCALES)):
        a = retinal[scale_i].ravel()
        b = deprecated[scale_i].ravel()
        cosine.append(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS)))
    summary = {
        "n_exact_fourier_modes": int(spectrum["n_exact_fourier_modes"]),
        "n_selected_images": int(len(spectrum["selected_image_index"])),
        "n_selected_corrected_drift_trajectories": int(len(spectrum["selected_trace_index"])),
        "scales": SCALES,
        "trajectory_phase_temporal_centroid_hz_by_scale": temporal_centroid,
        "trajectory_phase_temporal_mode_hz_by_scale": temporal_mode,
        "kinematic_fraction_falling_inside_measured_tf_grid_by_scale": np.divide(
            spectrum["kinematic_in_grid_weight"],
            spectrum["kinematic_total_weight"],
            out=np.zeros(len(SCALES)),
            where=spectrum["kinematic_total_weight"] > 0,
        ),
        "cosine_similarity_trajectory_phase_spectrum_vs_deprecated_velocity_histogram_by_scale": cosine,
        "spectrum_method_version": SPECTRUM_METHOD_VERSION,
        "median_unit_curve_spearman": float(unit.spearman_predicted_overlap_vs_ssi_change_moving_scales.median()),
        "fraction_units_positive_curve_spearman": float((unit.spearman_predicted_overlap_vs_ssi_change_moving_scales > 0).mean()),
        "prediction_warning": "For early linear stages the overlap is a power-transfer summary. For nonlinear layers it is a descriptive passband-overlap index and is validated only by comparison with exact natural-movie responses/SSI.",
    }
    write_json(OUT / "fem_tuning_overlap_statistics.json", summary)
    print(json.dumps(summary, indent=2, default=lambda x: x.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
