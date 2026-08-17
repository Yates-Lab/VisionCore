#!/usr/bin/env python3
"""Measure and render center-pixel local RFs for every RR100 unit.

The atlas evaluates the exact pre-softplus gradient for all 100 unit outputs
at the center of their 51 x 51 maps.  It uses all eight predeclared natural
images at a declared FEM-amplitude operating point, retains a central crop of
each full 32-lag gradient, and accumulates full-field energy for localization
checks.  Figures show all units without population averaging and a unit x
image grid that exposes context dependence directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.nonlinear_phase_causal.common import DIAGNOSTICS, EXAMPLE_OUTPUT, EXAMPLE_TRACE, json_ready
from paper.fig4.nonlinear_phase_causal.measure_center_rf import (
    context_cosine,
    lag_before_output_ms,
    radial_profile,
    rf_geometry,
    spatial_frequency_power,
)
from paper.fig4.nonlinear_phase_causal.run_experiment import SOURCE, _selection, preactivation
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import PPD, RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


DEFAULT_OUT = DIAGNOSTICS / "rr100_center_rf_atlas_1x"
GRADIENT_FILE = "rr100_center_rf_crops.npz"
CROP_RADIUS_PIX = 32
DISPLAY_RADIUS_DEG = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vjp-chunk-size", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--anchor-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse-gradients", action="store_true")
    return parser.parse_args()


def orientation_metrics(power: np.ndarray, *, ppd: float) -> tuple[float, float]:
    """Fourier orientation selectivity and preferred orientation in degrees."""

    value = np.asarray(power, dtype=np.float64)
    fy = np.fft.fftshift(np.fft.fftfreq(value.shape[0], d=1.0 / ppd))
    fx = np.fft.fftshift(np.fft.fftfreq(value.shape[1], d=1.0 / ppd))
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(grid_y, grid_x)
    theta = np.arctan2(grid_y, grid_x)
    valid = (radius >= 0.5) & (radius <= 15.0) & np.isfinite(value)
    weight = np.where(valid, value, 0.0)
    total = float(weight.sum())
    if total <= 0:
        return np.nan, np.nan
    vector = np.sum(weight * np.exp(2j * theta)) / total
    return float(np.abs(vector)), float((np.degrees(np.angle(vector)) / 2.0) % 180.0)


def temporal_separability(local_gradient: np.ndarray) -> float:
    """Fraction of local spatiotemporal RF energy in its first SVD mode."""

    matrix = np.asarray(local_gradient, dtype=np.float64).reshape(local_gradient.shape[0], -1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    return float(energy[0] / np.maximum(energy.sum(), 1e-20))


def select_context_examples(summary: pd.DataFrame, n_per_group: int = 8) -> np.ndarray:
    """Select units at evenly spaced local-SF ranks within each group."""

    selected: list[int] = []
    for group in ("lower SF", "higher SF"):
        rows = summary.loc[summary.rf_group.eq(group)].sort_values(
            ["local_rf_spectral_peak_cpd", "unit_index"]
        )
        positions = np.linspace(0, len(rows) - 1, min(n_per_group, len(rows))).round().astype(int)
        selected.extend(rows.iloc[positions].unit_index.astype(int).tolist())
    return np.asarray(selected, dtype=np.int64)


def _movie_anchor(
    *,
    scorer: RealTraceMatrixScorer,
    source_row: pd.Series,
    canvas_cache: dict[Any, Any],
    anchor_history: np.ndarray,
) -> torch.Tensor:
    patch, _ = extract_patch(source_row, canvas_cache=canvas_cache, patch_size_px=540)
    image = _standardize_uint_like(patch)
    stims = (make_corrected_causal_stims(image, anchor_history, torch=scorer.torch) - 127.0) / 255.0
    return stims[EXAMPLE_OUTPUT : EXAMPLE_OUTPUT + 1].to(scorer.device)


def all_unit_vjps(
    *,
    scorer: RealTraceMatrixScorer,
    readout: torch.nn.Module,
    x_anchor: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return cropped gradients, full spatial energy, and temporal energy."""

    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    behavior = scorer._zero_behavior(1, dtype)
    x = x_anchor.detach().clone().requires_grad_(True)
    z = preactivation(model, readout, x, behavior)
    if z.ndim != 4 or z.shape[-2:] != (51, 51):
        raise ValueError(tuple(z.shape))
    center = (int(z.shape[-2] // 2), int(z.shape[-1] // 2))
    center_values = z[0, :, center[0], center[1]]
    n_units = int(center_values.numel())
    n_lags, height, width = map(int, x.shape[-3:])
    crop_y = slice(height // 2 - CROP_RADIUS_PIX, height // 2 + CROP_RADIUS_PIX + 1)
    crop_x = slice(width // 2 - CROP_RADIUS_PIX, width // 2 + CROP_RADIUS_PIX + 1)
    crop_size = 2 * CROP_RADIUS_PIX + 1
    crops = np.empty((n_units, n_lags, crop_size, crop_size), dtype=np.float32)
    spatial_energy = np.empty((n_units, height, width), dtype=np.float64)
    temporal_energy = np.empty((n_units, n_lags), dtype=np.float64)

    identity = torch.eye(n_units, device=center_values.device, dtype=center_values.dtype)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_units, chunk_size):
        stop = min(start + chunk_size, n_units)
        gradient = torch.autograd.grad(
            center_values,
            x,
            grad_outputs=identity[start:stop],
            retain_graph=stop < n_units,
            create_graph=False,
            is_grads_batched=True,
        )[0][:, 0, 0]
        value = gradient.detach().float().cpu().numpy()
        crops[start:stop] = value[:, :, crop_y, crop_x]
        spatial_energy[start:stop] = np.mean(value.astype(np.float64) ** 2, axis=1)
        temporal_energy[start:stop] = np.mean(value.astype(np.float64) ** 2, axis=(2, 3))
        print(f"  VJP units {start:03d}–{stop - 1:03d}", flush=True)
        del gradient, value
    return crops, spatial_energy, temporal_energy, center


def measure_all_contexts(
    *,
    scorer: RealTraceMatrixScorer,
    readout: torch.nn.Module,
    selected_images: pd.DataFrame,
    base_history: np.ndarray,
    anchor_scale: float,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    source_images = pd.read_csv(SOURCE / "image_feature_table.csv").set_index("image_index")
    anchor_history = scaled_histories(
        base_history[None], np.asarray([float(anchor_scale)], dtype=np.float32)
    )
    canvas_cache: dict[Any, Any] = {}
    crops: list[np.ndarray] = []
    spatial_energies: list[np.ndarray] = []
    temporal_energies: list[np.ndarray] = []
    centers: list[tuple[int, int]] = []
    for image_id in selected_images.image_index.astype(int):
        print(f"image {image_id}", flush=True)
        anchor = _movie_anchor(
            scorer=scorer,
            source_row=source_images.loc[image_id],
            canvas_cache=canvas_cache,
            anchor_history=anchor_history,
        )
        crop, spatial, temporal, center = all_unit_vjps(
            scorer=scorer,
            readout=readout,
            x_anchor=anchor,
            chunk_size=chunk_size,
        )
        crops.append(crop)
        spatial_energies.append(spatial)
        temporal_energies.append(temporal)
        centers.append(center)
        del anchor
        if str(scorer.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if len(set(centers)) != 1:
        raise ValueError(centers)
    return {
        "gradient_crops": np.stack(crops).astype(np.float32),
        "spatial_energy": np.mean(spatial_energies, axis=0).astype(np.float64),
        "temporal_energy_by_image": np.stack(temporal_energies).astype(np.float64),
        "selected_image_index": selected_images.image_index.to_numpy(np.int64),
        "output_center_yx": np.asarray(centers[0], dtype=np.int64),
        "input_crop_radius_pixels": np.asarray(CROP_RADIUS_PIX, dtype=np.int64),
        "anchor_scale": np.asarray(float(anchor_scale), dtype=np.float32),
    }


def summarize(payload: dict[str, np.ndarray], tuning: pd.DataFrame) -> pd.DataFrame:
    crops = np.asarray(payload["gradient_crops"], dtype=np.float32)  # image, unit, lag, y, x
    spatial_energy = np.asarray(payload["spatial_energy"], dtype=np.float64)
    temporal_by_image = np.asarray(payload["temporal_energy_by_image"], dtype=np.float64)
    temporal = temporal_by_image.mean(axis=0)
    lag_ms = lag_before_output_ms(crops.shape[2])
    source_image_ids = np.asarray(payload["selected_image_index"], dtype=int)
    signed_context = int(np.flatnonzero(source_image_ids == 15)[0]) if 15 in source_image_ids else 0
    rows: list[dict[str, Any]] = []
    crop_slice = slice(
        spatial_energy.shape[-1] // 2 - CROP_RADIUS_PIX,
        spatial_energy.shape[-1] // 2 + CROP_RADIUS_PIX + 1,
    )
    for unit in range(crops.shape[1]):
        geometry = rf_geometry(spatial_energy[unit], ppd=PPD)
        peak_lag = int(np.argmax(temporal[unit]))
        power = spatial_frequency_power(crops[:, unit])
        frequency, profile = radial_profile(power, ppd=PPD)
        valid = (frequency >= 0.5) & (frequency <= 15.0)
        spectral_peak = float(frequency[valid][int(np.argmax(profile[valid]))])
        osi, orientation = orientation_metrics(power, ppd=PPD)
        signed_cosine, absolute_cosine = context_cosine(crops[:, unit])
        full_energy = float(spatial_energy[unit].sum())
        crop_energy = float(spatial_energy[unit, crop_slice, crop_slice].sum())
        rows.append(
            {
                "unit_index": unit,
                "peak_lag_index_current_zero": peak_lag,
                "peak_lag_before_output_ms": float(lag_ms[peak_lag]),
                "energy_weighted_lag_before_output_ms": float(
                    np.sum(lag_ms * temporal[unit]) / np.maximum(np.sum(temporal[unit]), 1e-20)
                ),
                "local_rf_spectral_peak_cpd": spectral_peak,
                "local_rf_orientation_selectivity": osi,
                "local_rf_preferred_orientation_deg": orientation,
                "local_rf_temporal_separability": temporal_separability(crops[signed_context, unit]),
                "median_signed_context_cosine": signed_cosine,
                "median_absolute_context_cosine": absolute_cosine,
                "stored_crop_energy_fraction": crop_energy / np.maximum(full_energy, 1e-20),
                **geometry,
            }
        )
    summary = pd.DataFrame(rows)
    columns = [
        "unit_index",
        "unit_label",
        "sf_split_metric",
        "sf_rank_low_to_high",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
        "static_peak_spatial_cpd_by_mean_rate",
        "dynamic_peak_spatial_cpd_by_amp",
        "dynamic_amp_weighted_sf_cpd",
        "static_rate_weighted_sf_cpd",
    ]
    summary = summary.merge(tuning[columns], on="unit_index", how="left", validate="one_to_one")
    summary["rf_group"] = np.where(summary.sf_split_metric < 0.5, "lower SF", "higher SF")
    return summary.sort_values("unit_index").reset_index(drop=True)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _unit_map(
    ax: plt.Axes,
    value: np.ndarray,
    row: pd.Series,
    *,
    title: str,
    group_border: bool = True,
) -> None:
    extent = np.asarray([-CROP_RADIUS_PIX, CROP_RADIUS_PIX] * 2, dtype=float) / PPD
    vmax = max(float(np.max(np.abs(value))), 1e-12)
    ax.imshow(value, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", extent=extent)
    ax.axhline(0, color="0.25", lw=0.25, alpha=0.45)
    ax.axvline(0, color="0.25", lw=0.25, alpha=0.45)
    ax.set_xlim(-DISPLAY_RADIUS_DEG, DISPLAY_RADIUS_DEG)
    ax.set_ylim(-DISPLAY_RADIUS_DEG, DISPLAY_RADIUS_DEG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=2.0)
    if group_border:
        color = "#0077b6" if row.rf_group == "lower SF" else "#d55e00"
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.1)


def atlas_figures(payload: dict[str, np.ndarray], summary: pd.DataFrame, output_dir: Path) -> None:
    _style()
    crops = np.asarray(payload["gradient_crops"], dtype=np.float32)
    image_ids = np.asarray(payload["selected_image_index"], dtype=int)
    anchor_scale = float(np.asarray(payload["anchor_scale"]).item())
    context = int(np.flatnonzero(image_ids == 15)[0]) if 15 in image_ids else 0
    ordered = summary.sort_values(["sf_split_metric", "unit_index"]).unit_index.astype(int).to_numpy()
    rows_per_page, cols_per_page = 4, 5
    per_page = rows_per_page * cols_per_page
    pdf_path = output_dir / "rr100_center_rf_atlas_multipage.pdf"
    with PdfPages(pdf_path) as pdf:
        for page, start in enumerate(range(0, len(ordered), per_page), start=1):
            units = ordered[start : start + per_page]
            fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(7.4, 6.2), constrained_layout=True)
            for ax, unit in zip(axes.flat, units, strict=False):
                row = summary.set_index("unit_index").loc[unit]
                peak = int(row.peak_lag_index_current_zero)
                group_short = "L" if row.rf_group == "lower SF" else "H"
                title = (
                    f"u{unit:03d}  {group_short} | group={row.sf_split_metric:.2g}\n"
                    f"local peak={row.local_rf_spectral_peak_cpd:.1f} c/°"
                )
                _unit_map(ax, crops[context, unit, peak], row, title=title)
            for ax in axes.flat[len(units) :]:
                ax.axis("off")
            fig.suptitle(
                f"RR100 center-pixel local RF atlas — {anchor_scale:g}× FEM, image {image_ids[context]} — page {page}/5\n"
                "sorted by historical SF metric; each panel independently normalized; blue=lower SF, orange=higher SF",
                fontsize=9,
                fontweight="bold",
            )
            pdf.savefig(fig, bbox_inches="tight")
            for extension in ("png", "svg"):
                kwargs = {"dpi": 300} if extension == "png" else {}
                fig.savefig(output_dir / f"rr100_center_rf_atlas_page_{page}.{extension}", bbox_inches="tight", **kwargs)
            plt.close(fig)

    fig, axes = plt.subplots(10, 10, figsize=(12.0, 12.0), constrained_layout=True)
    for ax, unit in zip(axes.flat, ordered, strict=True):
        row = summary.set_index("unit_index").loc[unit]
        peak = int(row.peak_lag_index_current_zero)
        _unit_map(ax, crops[context, unit, peak], row, title=f"u{unit:03d} | {row.local_rf_spectral_peak_cpd:.1f} c/°")
    fig.suptitle(
        f"All RR100 center-pixel local RFs at {anchor_scale:g}× FEM, image {image_ids[context]}\n"
        "sorted by historical SF metric; independently normalized; blue=lower SF, orange=higher SF",
        fontsize=12,
        fontweight="bold",
    )
    for extension in ("pdf", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"rr100_center_rf_contact_sheet.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def context_figure(payload: dict[str, np.ndarray], summary: pd.DataFrame, output_dir: Path) -> np.ndarray:
    _style()
    crops = np.asarray(payload["gradient_crops"], dtype=np.float32)
    image_ids = np.asarray(payload["selected_image_index"], dtype=int)
    anchor_scale = float(np.asarray(payload["anchor_scale"]).item())
    selected = select_context_examples(summary, n_per_group=8)
    lookup = summary.set_index("unit_index")
    fig, axes = plt.subplots(
        len(selected),
        len(image_ids),
        figsize=(10.0, 15.5),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, unit in enumerate(selected):
        row = lookup.loc[int(unit)]
        peak = int(row.peak_lag_index_current_zero)
        for column, image_id in enumerate(image_ids):
            title = f"image {image_id}" if row_index == 0 else ""
            _unit_map(
                axes[row_index, column],
                crops[column, unit, peak],
                row,
                title=title,
                group_border=False,
            )
            if column == 0:
                axes[row_index, column].set_ylabel(
                    f"u{unit:03d} {row.rf_group}\nlocal peak {row.local_rf_spectral_peak_cpd:.1f} c/°",
                    fontsize=6.2,
                )
    fig.suptitle(
        f"Natural-image dependence of local RR100 receptive fields at {anchor_scale:g}× FEM\n"
        "16 units selected at evenly spaced local-SF ranks within each group; each map independently normalized",
        fontsize=10,
        fontweight="bold",
    )
    for extension in ("pdf", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"rr100_rf_across_natural_images.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return selected


def summary_figure(
    summary: pd.DataFrame, output_dir: Path, *, anchor_scale: float
) -> dict[str, float]:
    _style()
    colors = np.where(summary.rf_group.eq("lower SF"), "#0077b6", "#d55e00")
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.7), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(summary.sf_split_metric, summary.local_rf_spectral_peak_cpd, c=colors, s=15, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("historical SF grouping metric (c/°)")
    ax.set_ylabel("local RF spectrum peak (c/°)")
    ax.set_title("Local spectrum versus prior grouping")

    ax = axes[0, 1]
    for group, color in (("lower SF", "#0077b6"), ("higher SF", "#d55e00")):
        values = summary.loc[summary.rf_group.eq(group), "local_rf_spectral_peak_cpd"]
        ax.hist(values, bins=np.arange(0, 15.5, 0.75), histtype="step", lw=1.5, color=color, label=group)
    ax.set_xlabel("local RF spectrum peak (c/°)")
    ax.set_ylabel("units")
    ax.set_title("RR100 local spatial spectra")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    ax.scatter(summary.local_rf_spectral_peak_cpd, summary.local_rf_orientation_selectivity, c=colors, s=15)
    ax.set_xlabel("local RF spectrum peak (c/°)")
    ax.set_ylabel("Fourier orientation selectivity")
    ax.set_title("Spectral organization")

    ax = axes[1, 0]
    ax.hist(summary.radius_90_deg, bins=18, color="0.35")
    ax.set_xlabel("90%-energy radius (deg)")
    ax.set_ylabel("units")
    ax.set_title("Spatial localization")

    ax = axes[1, 1]
    ax.hist(summary.energy_weighted_lag_before_output_ms, bins=18, color="#4c78a8")
    ax.set_xlabel("energy-weighted lag (ms)")
    ax.set_ylabel("units")
    ax.set_title("Temporal localization")

    ax = axes[1, 2]
    ax.scatter(
        summary.median_signed_context_cosine,
        summary.median_absolute_context_cosine,
        c=colors,
        s=15,
    )
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel("signed RF similarity across images")
    ax.set_ylabel("absolute-envelope similarity")
    ax.set_title("Context dependence")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"RR100 center-pixel local RF summary at {anchor_scale:g}× FEM",
        fontsize=10,
        fontweight="bold",
    )
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"rr100_center_rf_summary.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    low = summary.loc[summary.rf_group.eq("lower SF")]
    high = summary.loc[summary.rf_group.eq("higher SF")]
    return {
        "lower_local_spectral_peak_median_cpd": float(low.local_rf_spectral_peak_cpd.median()),
        "higher_local_spectral_peak_median_cpd": float(high.local_rf_spectral_peak_cpd.median()),
        "radius90_median_deg": float(summary.radius_90_deg.median()),
        "radius90_iqr_low_deg": float(summary.radius_90_deg.quantile(0.25)),
        "radius90_iqr_high_deg": float(summary.radius_90_deg.quantile(0.75)),
        "centroid_offset_median_deg": float(summary.centroid_offset_deg.median()),
        "energy_weighted_lag_median_ms": float(summary.energy_weighted_lag_before_output_ms.median()),
        "orientation_selectivity_median": float(summary.local_rf_orientation_selectivity.median()),
        "orientation_selectivity_fraction_above_0_2": float(
            np.mean(summary.local_rf_orientation_selectivity > 0.2)
        ),
        "signed_context_cosine_median": float(summary.median_signed_context_cosine.median()),
        "absolute_context_cosine_median": float(summary.median_absolute_context_cosine.median()),
        "crop_energy_fraction_min": float(summary.stored_crop_energy_fraction.min()),
    }


def write_report(
    stats: dict[str, float], selected: np.ndarray, output_dir: Path, *, anchor_scale: float
) -> None:
    report = f"""# RR100 center-pixel receptive-field atlas

This atlas removes the two main ambiguities in the initial diagnostic: it shows every RR100 unit without population averaging, and it repeats 16 units across all eight natural-image operating points. Each signed map is independently normalized because local-gradient amplitudes vary substantially across cells.

## What was measured

For every unit and image, the exact local RF is the gradient of that unit's center pre-softplus output, `dz[unit,25,25]/dx`, with respect to the 32 × 151 × 151 retinal input at the {anchor_scale:g}× FEM operating point. Full-field energy was retained for localization metrics; the central 65 × 65 region was stored for signed visualization. The stored crop contains at least {100 * stats['crop_energy_fraction_min']:.4f}% of every unit's RF energy.

## Population summary

- Median 90%-energy radius: {stats['radius90_median_deg']:.3f}° (IQR {stats['radius90_iqr_low_deg']:.3f}–{stats['radius90_iqr_high_deg']:.3f}°).
- Median centroid offset from the intended center: {stats['centroid_offset_median_deg']:.3f}°.
- Median energy-weighted temporal lag: {stats['energy_weighted_lag_median_ms']:.1f} ms.
- Median local RF-spectrum peak: {stats['lower_local_spectral_peak_median_cpd']:.2f} c/° in the lower-SF group and {stats['higher_local_spectral_peak_median_cpd']:.2f} c/° in the higher-SF group.
- Median Fourier orientation selectivity: {stats['orientation_selectivity_median']:.3f}; {100 * stats['orientation_selectivity_fraction_above_0_2']:.1f}% of units exceed 0.2.
- Median signed similarity of the same unit's RF across natural images: {stats['signed_context_cosine_median']:.3f}; median absolute-envelope similarity: {stats['absolute_context_cosine_median']:.3f}.

The last comparison is important: many local signed RFs are not fixed Gabor filters. Their localized envelope is substantially more stable than their positive/negative substructure. This is expected for a gradient through a model whose SplitReLU routes, normalizations, and recurrent gates depend on the natural-image anchor, but the atlas lets that claim be inspected rather than inferred from two examples.

The 16-unit context panel uses units {', '.join(f'u{unit:03d}' for unit in selected)} selected at evenly spaced local-spectrum ranks within each historical SF group; it is not a best-looking subset.
"""
    (output_dir / "README.md").write_text(report)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / GRADIENT_FILE
    selected_images, _ = _selection()
    if args.max_images:
        selected_images = selected_images.iloc[: args.max_images].copy()
    expected_images = selected_images.image_index.to_numpy(np.int64)
    if float(args.anchor_scale) < 0:
        raise ValueError("--anchor-scale must be nonnegative")

    if args.reuse_gradients and gradient_path.is_file():
        with np.load(gradient_path) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        if not np.array_equal(payload["selected_image_index"], expected_images):
            raise ValueError("Saved gradients use a different image selection")
        if not np.isclose(float(np.asarray(payload["anchor_scale"]).item()), float(args.anchor_scale)):
            raise ValueError("Saved gradients use a different FEM anchor scale")
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
        payload = measure_all_contexts(
            scorer=scorer,
            readout=readout,
            selected_images=selected_images,
            base_history=base_history,
            anchor_scale=float(args.anchor_scale),
            chunk_size=int(args.vjp_chunk_size),
        )
        np.savez_compressed(gradient_path, **payload)

    tuning = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    summary = summarize(payload, tuning)
    summary.to_csv(output_dir / "rr100_center_rf_summary.csv", index=False)
    atlas_figures(payload, summary, output_dir)
    selected = context_figure(payload, summary, output_dir)
    stats = summary_figure(summary, output_dir, anchor_scale=float(args.anchor_scale))
    write_report(stats, selected, output_dir, anchor_scale=float(args.anchor_scale))
    manifest = {
        "analysis": "all_rr100_center_pixel_local_receptive_fields",
        "definition": "exact VJP rows dz[unit,25,25]/dx at the declared FEM-amplitude natural-image anchors",
        "n_units": 100,
        "n_images": len(expected_images),
        "anchor_scale": float(args.anchor_scale),
        "selected_image_index": expected_images,
        "gradient_crop_shape": list(payload["gradient_crops"].shape),
        "full_spatial_energy_shape": list(payload["spatial_energy"].shape),
        "lag_convention": "index 0 current; index k is k/120 seconds before output",
        "display_radius_deg": DISPLAY_RADIUS_DEG,
        "context_example_units": selected,
        "summary_statistics": stats,
        "finite": bool(
            np.isfinite(payload["gradient_crops"]).all()
            and np.isfinite(payload["spatial_energy"]).all()
            and np.isfinite(payload["temporal_energy_by_image"]).all()
        ),
    }
    write_json(output_dir / "manifest.json", json_ready(manifest))
    print(json.dumps(json_ready(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
