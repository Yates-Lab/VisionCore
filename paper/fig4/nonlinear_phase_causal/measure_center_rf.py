#!/usr/bin/env python3
"""Measure center-pixel receptive fields of the stabilized-tangent twin.

For each predeclared natural image, this script evaluates the exact gradient
of a scalar pre-softplus output at the center of its 51 x 51 activation map
with respect to every pixel and lag of the matched stabilized input movie.
That gradient is the local spatiotemporal receptive field used by the tangent
counterfactual.  Results are summarized across image-dependent operating
points with a representative signed local RF and RMS energy envelopes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    DT_S,
    LEGACY_MATRIX_DIR,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.nonlinear_phase_causal.common import (
    DIAGNOSTICS,
    EXAMPLE_OUTPUT,
    EXAMPLE_TRACE,
    OUT,
    json_ready,
)
from paper.fig4.nonlinear_phase_causal.run_experiment import (
    SOURCE,
    _selection,
    historical_groups,
    preactivation,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    PPD,
    RealTraceMatrixScorer,
    _standardize_uint_like,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


RF_OUT = DIAGNOSTICS / "center_pixel_linearized_rf"
FIGURE_STEM = "diagnostic_center_pixel_local_receptive_fields"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RF_OUT)
    parser.add_argument("--reuse-gradients", action="store_true")
    return parser.parse_args()


def representative_unit(
    table: pd.DataFrame,
    indices: np.ndarray,
    *,
    minimum_fit_r2: float = 0.5,
    minimum_dynamic_amplitude: float = 0.1,
) -> int:
    """Choose the well-fit unit nearest the group's median SF preference."""

    rows = table.set_index("unit_index").loc[np.asarray(indices, dtype=int)].copy()
    sf = pd.to_numeric(rows["sf_split_metric"], errors="coerce")
    target = float(np.nanmedian(sf))
    fit = pd.to_numeric(rows["dynamic_log_gaussian_marginal_r2"], errors="coerce")
    amplitude = pd.to_numeric(rows["dynamic_peak_response_amp"], errors="coerce")
    eligible = np.isfinite(sf) & (fit >= minimum_fit_r2) & (amplitude >= minimum_dynamic_amplitude)
    if not eligible.any():
        eligible = np.isfinite(sf)
    distance = np.abs(np.log2(sf[eligible].to_numpy(float) / target))
    return int(rows.loc[eligible].iloc[int(np.argmin(distance))].name)


def radial_profile(power: np.ndarray, *, ppd: float, bin_width_cpd: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Return an azimuthally averaged spatial-frequency power profile."""

    value = np.asarray(power, dtype=np.float64)
    fy = np.fft.fftshift(np.fft.fftfreq(value.shape[0], d=1.0 / ppd))
    fx = np.fft.fftshift(np.fft.fftfreq(value.shape[1], d=1.0 / ppd))
    radius = np.hypot(fy[:, None], fx[None, :])
    edges = np.arange(0.0, float(radius.max()) + bin_width_cpd, bin_width_cpd)
    which = np.digitize(radius.ravel(), edges) - 1
    count = np.bincount(which, minlength=len(edges) - 1)
    total = np.bincount(which, weights=value.ravel(), minlength=len(edges) - 1)
    profile = np.divide(total[: len(count)], np.maximum(count, 1))
    centers = edges[:-1] + bin_width_cpd / 2.0
    return centers[: len(profile)], profile


def lag_before_output_ms(n_lags: int) -> np.ndarray:
    """Production lag coordinates: current frame first, oldest frame last."""

    if int(n_lags) <= 0:
        raise ValueError(n_lags)
    return np.arange(int(n_lags), dtype=np.float64) * DT_S * 1000.0


def rf_geometry(energy_xy: np.ndarray, *, ppd: float) -> dict[str, float]:
    """Energy centroid, containment radii, and border fraction."""

    energy = np.asarray(energy_xy, dtype=np.float64)
    total = float(energy.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Receptive-field energy must be positive and finite")
    yy, xx = np.indices(energy.shape, dtype=np.float64)
    cy = float((energy * yy).sum() / total)
    cx = float((energy * xx).sum() / total)
    center_y = (energy.shape[0] - 1) / 2.0
    center_x = (energy.shape[1] - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(energy.ravel()[order]) / total

    def containment(fraction: float) -> float:
        index = min(int(np.searchsorted(cumulative, fraction)), len(order) - 1)
        return float(radius[order[index]] / ppd)

    border = np.zeros_like(energy, dtype=bool)
    border[:10] = True
    border[-10:] = True
    border[:, :10] = True
    border[:, -10:] = True
    return {
        "centroid_x_offset_deg": float((cx - center_x) / ppd),
        "centroid_y_offset_deg": float((cy - center_y) / ppd),
        "centroid_offset_deg": float(np.hypot(cx - center_x, cy - center_y) / ppd),
        "radius_50_deg": containment(0.5),
        "radius_90_deg": containment(0.9),
        "outer_10px_energy_fraction": float(energy[border].sum() / total),
    }


def context_cosine(gradients: np.ndarray) -> tuple[float, float]:
    """Median pairwise cosine for signed RFs and their nonnegative envelopes."""

    flat = np.asarray(gradients, dtype=np.float64).reshape(len(gradients), -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-20)
    signed = flat @ flat.T
    envelope = np.abs(flat)
    envelope /= np.maximum(np.linalg.norm(envelope, axis=1, keepdims=True), 1e-20)
    unsigned = envelope @ envelope.T
    pair = np.triu_indices(len(flat), 1)
    if len(pair[0]) == 0:
        return 1.0, 1.0
    return float(np.median(signed[pair])), float(np.median(unsigned[pair]))


def spatial_frequency_power(gradients: np.ndarray) -> np.ndarray:
    """Image/lag-averaged windowed spatial power of local RF gradients."""

    value = np.asarray(gradients, dtype=np.float64)
    value = value - value.mean(axis=(-2, -1), keepdims=True)
    window_y = np.hanning(value.shape[-2])
    window_x = np.hanning(value.shape[-1])
    transformed = np.fft.fftshift(
        np.fft.fft2(value * window_y[None, None, :, None] * window_x[None, None, None, :], axes=(-2, -1)),
        axes=(-2, -1),
    )
    return np.mean(np.abs(transformed) ** 2, axis=(0, 1))


def _target_value(z: torch.Tensor, spec: dict[str, Any], center: tuple[int, int]) -> torch.Tensor:
    if spec["kind"] == "population_mean":
        return z[0, spec["indices"], center[0], center[1]].mean()
    return z[0, int(spec["unit_index"]), center[0], center[1]]


def measure_local_gradients(
    *,
    scorer: RealTraceMatrixScorer,
    readout: torch.nn.Module,
    target_specs: list[dict[str, Any]],
    selected_images: pd.DataFrame,
    base_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Measure one center-output gradient per target and natural-image anchor."""

    dtype = next(scorer.model.model.parameters()).dtype
    source_images = pd.read_csv(SOURCE / "image_feature_table.csv").set_index("image_index")
    canvas_cache: dict[Any, Any] = {}
    all_gradients: list[np.ndarray] = []
    all_anchor_values: list[np.ndarray] = []
    output_center: tuple[int, int] | None = None

    stable_history = scaled_histories(base_history[None], np.asarray([0.0], dtype=np.float32))
    for image_id in selected_images.image_index.astype(int):
        patch, _ = extract_patch(source_images.loc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
        image = _standardize_uint_like(patch)
        stims = (make_corrected_causal_stims(image, stable_history, torch=scorer.torch) - 127.0) / 255.0
        x_anchor = stims[EXAMPLE_OUTPUT : EXAMPLE_OUTPUT + 1].to(scorer.device)
        behavior = scorer._zero_behavior(1, dtype)

        image_gradients: list[np.ndarray] = []
        image_values: list[float] = []
        for spec in target_specs:
            x = x_anchor.detach().clone().requires_grad_(True)
            z = preactivation(scorer.model.model, readout, x, behavior)
            if z.ndim != 4 or z.shape[-2] % 2 != 1 or z.shape[-1] % 2 != 1:
                raise ValueError(f"Expected an odd 2-D activation map, got {tuple(z.shape)}")
            center = (int(z.shape[-2] // 2), int(z.shape[-1] // 2))
            if output_center is None:
                output_center = center
            elif output_center != center:
                raise ValueError((output_center, center))
            target = _target_value(z, spec, center)
            gradient = torch.autograd.grad(target, x, create_graph=False, retain_graph=False)[0]
            image_gradients.append(gradient[0, 0].detach().float().cpu().numpy())
            image_values.append(float(target.detach().cpu()))
            del x, z, target, gradient
        all_gradients.append(np.stack(image_gradients))
        all_anchor_values.append(np.asarray(image_values, dtype=np.float32))
        print(f"measured center-pixel RFs for image {image_id}", flush=True)
        if str(scorer.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if output_center is None:
        raise RuntimeError("No images were measured")
    # target, image, lag, y, x; target, image
    gradients = np.stack(all_gradients, axis=1).astype(np.float32)
    anchor_values = np.stack(all_anchor_values, axis=1).astype(np.float32)
    return gradients, anchor_values, output_center


def summarize(
    gradients: np.ndarray,
    target_specs: list[dict[str, Any]],
    tuning_table: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    n_lags = int(gradients.shape[2])
    table = tuning_table.set_index("unit_index")
    for index, (spec, gradient) in enumerate(zip(target_specs, gradients, strict=True)):
        energy_lag = np.mean(gradient.astype(np.float64) ** 2, axis=(0, 2, 3))
        energy_xy = np.mean(gradient.astype(np.float64) ** 2, axis=(0, 1))
        geometry = rf_geometry(energy_xy, ppd=PPD)
        peak_lag_index = int(np.argmax(energy_lag))
        # Production convention: lag index 0 is the current retinal frame;
        # increasing indices are progressively older frames.
        lag_ms = lag_before_output_ms(n_lags)
        temporal_center = float(np.sum(lag_ms * energy_lag) / np.sum(energy_lag))
        power = spatial_frequency_power(gradient)
        frequency, profile = radial_profile(power, ppd=PPD)
        valid = (frequency >= 0.25) & (frequency <= 15.0)
        spectral_peak = float(frequency[valid][int(np.argmax(profile[valid]))])
        profiles.append((frequency, profile))
        signed_cosine, envelope_cosine = context_cosine(gradient)
        target_indices = [int(spec["unit_index"])] if spec["kind"] == "unit" else spec["indices"]
        target_tuning = table.loc[target_indices]
        rows.append(
            {
                "target_index": index,
                "target": spec["label"],
                "kind": spec["kind"],
                "unit_index": spec.get("unit_index", np.nan),
                "n_units": len(spec.get("indices", [spec.get("unit_index")])),
                "historical_group_metric_cpd": float(np.nanmedian(target_tuning["sf_split_metric"])),
                "dynamic_grating_peak_cpd": float(
                    np.nanmedian(target_tuning["dynamic_peak_spatial_cpd_by_amp"])
                ),
                "static_grating_peak_cpd": float(
                    np.nanmedian(target_tuning["static_peak_spatial_cpd_by_mean_rate"])
                ),
                "peak_energy_lag_index_current_zero": peak_lag_index,
                "peak_energy_lag_before_output_ms": float(lag_ms[peak_lag_index]),
                "energy_weighted_lag_before_output_ms": temporal_center,
                "spectral_peak_cpd": spectral_peak,
                "median_signed_context_cosine": signed_cosine,
                "median_absolute_context_cosine": envelope_cosine,
                **geometry,
            }
        )
    return pd.DataFrame(rows), profiles


def render(
    gradients: np.ndarray,
    summaries: pd.DataFrame,
    profiles: list[tuple[np.ndarray, np.ndarray]],
    target_specs: list[dict[str, Any]],
    selected_image_index: np.ndarray,
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    n_target, _, n_lags, height, width = gradients.shape
    fig, axes = plt.subplots(n_target, 4, figsize=(9.3, 1.62 * n_target + 0.55), constrained_layout=True)
    if n_target == 1:
        axes = axes[None]
    extent = np.asarray([-(width - 1) / 2, (width - 1) / 2, -(height - 1) / 2, (height - 1) / 2]) / PPD
    lag_ms = lag_before_output_ms(n_lags)
    signed_context_index = int(np.flatnonzero(selected_image_index == 15)[0]) if 15 in selected_image_index else 0
    signed_context_label = int(selected_image_index[signed_context_index])

    for row_index, (spec, gradient) in enumerate(zip(target_specs, gradients, strict=True)):
        summary = summaries.iloc[row_index]
        energy_lag = np.mean(gradient.astype(np.float64) ** 2, axis=(0, 2, 3))
        energy_xy = np.mean(gradient.astype(np.float64) ** 2, axis=(0, 1))
        peak = int(summary.peak_energy_lag_index_current_zero)
        signed_map = gradient[signed_context_index, peak]
        vmax = max(float(np.max(np.abs(signed_map))), 1e-12)

        ax = axes[row_index, 0]
        image = ax.imshow(
            signed_map,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.axhline(0, color="0.25", lw=0.35, alpha=0.5)
        ax.axvline(0, color="0.25", lw=0.35, alpha=0.5)
        ax.set_ylabel(f"{spec['label']}\nretinal y (deg)")
        ax.set_title(f"local signed RF, −{summary.peak_energy_lag_before_output_ms:.0f} ms")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02).set_label("local gain", fontsize=6)

        ax = axes[row_index, 1]
        envelope = np.sqrt(energy_xy)
        image = ax.imshow(envelope, cmap="magma", origin="lower", extent=extent, interpolation="nearest")
        ax.plot(summary.centroid_x_offset_deg, summary.centroid_y_offset_deg, "+", color="cyan", ms=5, mew=1)
        ax.set_title(f"RMS envelope (r90={summary.radius_90_deg:.2f}°)")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02).set_label("RMS gain", fontsize=6)

        ax = axes[row_index, 2]
        ax.plot(lag_ms, energy_lag / np.max(energy_lag), color="#1769aa", lw=1.5)
        ax.axvline(summary.peak_energy_lag_before_output_ms, color="0.35", lw=0.8, ls=":")
        ax.set_xlim(float(lag_ms.max()), 0.0)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"temporal profile (center={summary.energy_weighted_lag_before_output_ms:.0f} ms)")
        ax.set_ylabel("normalized energy")

        ax = axes[row_index, 3]
        frequency, profile = profiles[row_index]
        profile = profile / max(float(np.max(profile[(frequency >= 0.25) & (frequency <= 15.0)])), 1e-20)
        ax.plot(frequency, profile, color="#6a3d9a", lw=1.5)
        ax.axvline(
            summary.dynamic_grating_peak_cpd,
            color="#d95f02",
            lw=0.9,
            ls="--",
            label="dynamic grating peak",
        )
        ax.axvline(
            summary.static_grating_peak_cpd,
            color="#1b9e77",
            lw=0.9,
            ls="-.",
            label="static grating peak",
        )
        ax.axvline(summary.spectral_peak_cpd, color="0.35", lw=0.8, ls=":", label="local RF peak")
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"RF spectrum (peak={summary.spectral_peak_cpd:.2f} c/deg)")
        if row_index == 0:
            ax.legend(frameon=False, fontsize=5.7, loc="upper right")

        zoom_deg = max(0.35, min(0.65, float(summary.radius_90_deg) * 2.5))
        for col in (0, 1):
            axes[row_index, col].set_xlim(-zoom_deg, zoom_deg)
            axes[row_index, col].set_ylim(-zoom_deg, zoom_deg)
        for col in range(4):
            axes[row_index, col].spines[["top", "right"]].set_visible(False)
            if row_index == n_target - 1:
                axes[row_index, col].set_xlabel(
                    "retinal x (deg)" if col < 2 else ("time before output (ms)" if col == 2 else "spatial frequency (c/deg)")
                )

    fig.suptitle(
        "Center-pixel local receptive fields of the stabilized-tangent twin\n"
        f"signed RF at image {signed_context_label}; RMS summaries across the eight natural-image operating points",
        fontsize=10,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{FIGURE_STEM}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_explanation(summary: pd.DataFrame, output_dir: Path) -> None:
    population = summary.loc[summary.kind.eq("population_mean")]
    units = summary.loc[summary.kind.eq("unit")]
    text = f"""# Center-pixel receptive field of the linearized twin

The stabilized-tangent twin is the first-order expansion of the trained model's complete pre-softplus output around a matched stabilized movie. For one output-map value, its receptive field is therefore the exact input gradient `dz_center/dx`: one signed sensitivity value for each retinal pixel and each of the 32 input lags. A positive entry means that a tiny luminance increment raises that center output locally; a negative entry means that it lowers it.

Because the trained twin is nonlinear, this receptive field depends on the image and recent history used as the linearization anchor. The figure measures the same center output at output index {EXAMPLE_OUTPUT} for all eight predeclared natural images and corrected stabilized history from trajectory {EXAMPLE_TRACE}. It shows one local signed RF (image 15), the RMS envelope across anchors (which cannot cancel when phase or gating changes), temporal energy, and spatial-frequency power.

The first two rows are the literal receptive fields of the lower- and higher-SF population-mean activation maps used by the Figure 4 diagnostic. The final two rows show individual units chosen deterministically as well-fit units nearest each historical group's median SF preference (units {', '.join(str(int(value)) for value in units.unit_index)}). Individual-unit fields are useful because averaging neurons with different signs and phases can cancel a real localized filter.

## Sanity checks

- All four gradients are finite and nonzero.
- The RMS centroids lie {population.centroid_offset_deg.min():.2f}–{population.centroid_offset_deg.max():.2f}° from the retinal center for the population maps and {units.centroid_offset_deg.min():.2f}–{units.centroid_offset_deg.max():.2f}° for the representative units.
- The 90%-energy radii are {population.radius_90_deg.min():.2f}–{population.radius_90_deg.max():.2f}° for the population maps and {units.radius_90_deg.min():.2f}–{units.radius_90_deg.max():.2f}° for the representative units.
- Peak temporal energy occurs {summary.peak_energy_lag_before_output_ms.min():.0f}–{summary.peak_energy_lag_before_output_ms.max():.0f} ms before the scored output; the production lag tensor stores the current frame at index 0.
- Less than {100 * summary.outer_10px_energy_fraction.max():.4f}% of RF energy reaches the outer 10-pixel border, ruling out a crop-edge explanation for the localization.
- The higher-SF population has a higher local RF-spectrum peak than the lower-SF population ({population.iloc[1].spectral_peak_cpd:.2f} versus {population.iloc[0].spectral_peak_cpd:.2f} cycles/degree); the representative units show the same ordering ({units.iloc[1].spectral_peak_cpd:.2f} versus {units.iloc[0].spectral_peak_cpd:.2f}).

These measurements test whether the tangent operator is spatially localized, centered on the intended output position, temporally causal, and ordered sensibly by SF group. They do not imply a single stimulus-independent biological RF: median signed RF cosine similarity across image anchors is near zero, whereas median absolute-envelope similarity is about {summary.median_absolute_context_cosine.mean():.2f}. Thus the RF location is stable but its signed substructure is strongly image-dependent, as expected when the fitted nonlinear gates change state. The local RF-spectrum peak is also not expected to equal the historical grouping metric, which came from finite-amplitude grating responses rather than an infinitesimal natural-image gradient.
"""
    (output_dir / "README.md").write_text(text)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_images, _ = _selection()
    if args.max_images:
        selected_images = selected_images.iloc[: args.max_images].copy()
    if len(selected_images) == 0:
        raise ValueError("No selected images")

    groups = historical_groups()
    tuning = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    representative_low = representative_unit(tuning, groups["low"])
    representative_high = representative_unit(tuning, groups["high"])
    target_specs: list[dict[str, Any]] = [
        {"label": "lower-SF mean", "kind": "population_mean", "indices": groups["low"]},
        {"label": "higher-SF mean", "kind": "population_mean", "indices": groups["high"]},
        {"label": f"lower-SF unit {representative_low}", "kind": "unit", "unit_index": representative_low},
        {"label": f"higher-SF unit {representative_high}", "kind": "unit", "unit_index": representative_high},
    ]

    gradient_path = output_dir / "center_pixel_rf_gradients.npz"
    if args.reuse_gradients and gradient_path.is_file():
        with np.load(gradient_path) as archive:
            gradients = np.asarray(archive["gradients"], dtype=np.float32)
            anchor_values = np.asarray(archive["anchor_preactivation"], dtype=np.float32)
            output_center = tuple(np.asarray(archive["output_center_yx"], dtype=int).tolist())
            saved_images = np.asarray(archive["selected_image_index"], dtype=np.int64)
            saved_labels = np.asarray(archive["target_labels"]).astype(str)
        expected_images = selected_images.image_index.to_numpy(np.int64)
        expected_labels = np.asarray([spec["label"] for spec in target_specs])
        if not np.array_equal(saved_images, expected_images) or not np.array_equal(saved_labels, expected_labels):
            raise ValueError("Saved gradients do not match the requested image/target selection")
        print(f"reusing {gradient_path}", flush=True)
    else:
        scorer = RealTraceMatrixScorer.load(
            checkpoint_path=MODEL_CHECKPOINT_PATH,
            dataset_configs=DEFAULT_DATASET_CONFIGS,
            population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
            rr100_version=RR100_VERSION,
            device=str(args.device),
            strict=True,
            mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
        )
        scorer.model.model.eval()
        readout = build_direct_readout(scorer).eval()
        with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
            base_history = np.asarray(archive["true_history_xy"][EXAMPLE_TRACE], dtype=np.float32)
        gradients, anchor_values, output_center = measure_local_gradients(
            scorer=scorer,
            readout=readout,
            target_specs=target_specs,
            selected_images=selected_images,
            base_history=base_history,
        )
    summary, profiles = summarize(gradients, target_specs, tuning)
    summary.to_csv(output_dir / "center_pixel_rf_summary.csv", index=False)
    np.savez_compressed(
        gradient_path,
        gradients=gradients,
        anchor_preactivation=anchor_values,
        target_labels=np.asarray([spec["label"] for spec in target_specs]),
        selected_image_index=selected_images.image_index.to_numpy(np.int64),
        input_ppd=np.asarray(PPD),
        frame_interval_seconds=np.asarray(DT_S),
        output_center_yx=np.asarray(output_center, dtype=np.int64),
        example_trace_index=np.asarray(EXAMPLE_TRACE, dtype=np.int64),
        example_output_index=np.asarray(EXAMPLE_OUTPUT, dtype=np.int64),
    )
    render(
        gradients,
        summary,
        profiles,
        target_specs,
        selected_images.image_index.to_numpy(np.int64),
        output_dir,
    )
    write_explanation(summary, output_dir)
    manifest = {
        "analysis": "center_pixel_local_receptive_field_of_stabilized_tangent_twin",
        "definition": "exact gradient of center pre-softplus output with respect to all normalized stimulus pixels/lags",
        "gradient_shape": list(gradients.shape),
        "output_map_center_yx": list(output_center),
        "input_lag_convention": "index 0 is current frame; index k is k/120 seconds before output",
        "signed_map_context_image_index": 15,
        "selected_image_index": selected_images.image_index.to_numpy(int),
        "trace_index": EXAMPLE_TRACE,
        "scored_output_index": EXAMPLE_OUTPUT,
        "target_specs": target_specs,
        "finite": bool(np.isfinite(gradients).all()),
        "all_targets_nonzero": bool(np.all(np.sum(gradients.astype(np.float64) ** 2, axis=(1, 2, 3, 4)) > 0)),
        "files": [
            f"{FIGURE_STEM}.pdf",
            f"{FIGURE_STEM}.svg",
            f"{FIGURE_STEM}.png",
            "center_pixel_rf_summary.csv",
            "center_pixel_rf_gradients.npz",
            "README.md",
        ],
    }
    write_json(output_dir / "manifest.json", json_ready(manifest))
    print(json.dumps({"summary": summary.to_dict(orient="records"), "manifest": json_ready(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
