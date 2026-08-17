#!/usr/bin/env python3
"""Expose signed temporal structure in every RR100 natural-context Jacobian.

The existing receptive-field atlas summarizes temporal *energy*, which cannot
distinguish monophasic from biphasic sensitivity.  This analysis instead
diagonalizes the 32 x 32 temporal Gram matrix of each complete local Jacobian.
It shows the signed temporal modes themselves, reports how much of the full
space-time derivative each mode explains, and measures whether those modes are
stable across the eight natural-image operating points.

The decomposition is descriptive, not a claim that a context-dependent CNN
Jacobian is a biological receptive field.  In particular, biphasic scores are
only interpretable when the leading mode explains substantial energy and is
stable across contexts; the figures expose all three quantities together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ATLAS_DIR = (
    ROOT
    / "outputs/figures/fig4/nonlinear_phase_causal_v1/diagnostics/rr100_center_rf_atlas_1x"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/figures/fig4/nonlinear_phase_causal_v1/diagnostics/rr100_signed_temporal_1x"
)
DT_MS = 1000.0 / 120.0
PPD = 37.50476617
LOW_COLOR = "#007C83"
HIGH_COLOR = "#D55E00"
NEUTRAL = "#4B5563"
CONTEXT_COLOR = "#9CA3AF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=ATLAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--interactive-output",
        type=Path,
        help="Optional destination for an inline signed-temporal unit explorer.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def temporal_decomposition(
    crops: np.ndarray, n_components: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonically signed temporal eigenvectors and energy fractions."""

    value = np.asarray(crops, dtype=np.float64)
    if value.ndim != 5:
        raise ValueError(f"Expected image x unit x lag x y x x, got {value.shape}")
    n_image, n_unit, n_lag, height, width = value.shape
    matrix = value.reshape(n_image * n_unit, n_lag, height * width)
    gram = matrix @ np.swapaxes(matrix, 1, 2)
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue, axis=1)[:, ::-1]
    eigenvalue = np.take_along_axis(eigenvalue, order, axis=1)
    eigenvector = np.take_along_axis(eigenvector, order[:, None, :], axis=2)
    eigenvalue = np.maximum(eigenvalue, 0.0)
    temporal = np.swapaxes(eigenvector[:, :, :n_components], 1, 2)

    # The paired temporal/spatial SVD modes have arbitrary joint sign.  Orient
    # every temporal mode so its largest-magnitude sample is positive.
    peak_index = np.argmax(np.abs(temporal), axis=2)
    peak_value = np.take_along_axis(temporal, peak_index[:, :, None], axis=2)[:, :, 0]
    temporal *= np.where(peak_value < 0.0, -1.0, 1.0)[:, :, None]
    fraction = eigenvalue / np.maximum(eigenvalue.sum(axis=1, keepdims=True), 1e-20)
    return (
        temporal.reshape(n_image, n_unit, n_components, n_lag),
        fraction.reshape(n_image, n_unit, n_lag),
    )


def signed_metrics(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return opponent-energy balance, opponent peak ratio, and robust crossings."""

    value = np.asarray(trace, dtype=np.float64)
    positive = np.sum(np.maximum(value, 0.0) ** 2, axis=-1)
    negative = np.sum(np.minimum(value, 0.0) ** 2, axis=-1)
    balance = 2.0 * np.minimum(positive, negative) / np.maximum(positive + negative, 1e-20)
    opponent_ratio = np.abs(np.min(value, axis=-1)) / np.maximum(np.max(value, axis=-1), 1e-20)

    flat = value.reshape(-1, value.shape[-1])
    crossings = np.zeros(flat.shape[0], dtype=np.int64)
    for row, current in enumerate(flat):
        keep = np.abs(current) >= 0.1 * np.max(np.abs(current))
        signs = np.sign(current[keep])
        crossings[row] = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    return balance, opponent_ratio, crossings.reshape(value.shape[:-1])


def peak_pixel_traces(crops: np.ndarray) -> np.ndarray:
    """Return the signed time course at each Jacobian's highest-energy pixel."""

    value = np.asarray(crops, dtype=np.float64)
    n_image, n_unit, n_lag, height, width = value.shape
    flat = value.reshape(n_image * n_unit, n_lag, height * width)
    spatial_energy = np.sum(flat**2, axis=1)
    peak = np.argmax(spatial_energy, axis=1)
    trace = np.take_along_axis(flat, peak[:, None, None], axis=2)[:, :, 0]
    dominant = trace[np.arange(len(trace)), np.argmax(np.abs(trace), axis=1)]
    trace *= np.where(dominant < 0.0, -1.0, 1.0)[:, None]
    trace /= np.maximum(np.max(np.abs(trace), axis=1, keepdims=True), 1e-20)
    return trace.reshape(n_image, n_unit, n_lag)


def context_stability(temporal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median pairwise overlap for rank-one, rank-two, and rank-four temporal modes."""

    n_image, n_unit = temporal.shape[:2]
    rank1 = np.empty(n_unit, dtype=np.float64)
    rank2 = np.empty(n_unit, dtype=np.float64)
    rank4 = np.empty(n_unit, dtype=np.float64)
    for unit in range(n_unit):
        overlap: dict[int, list[float]] = {1: [], 2: [], 4: []}
        for left in range(n_image):
            for right in range(left + 1, n_image):
                for rank in overlap:
                    left_basis = temporal[left, unit, :rank].T
                    right_basis = temporal[right, unit, :rank].T
                    overlap[rank].append(
                        float(np.sum((left_basis.T @ right_basis) ** 2) / rank)
                    )
        rank1[unit] = np.median(overlap[1])
        rank2[unit] = np.median(overlap[2])
        rank4[unit] = np.median(overlap[4])
    return rank1, rank2, rank4


def unit_summary(
    *,
    temporal: np.ndarray,
    fraction: np.ndarray,
    peak_trace: np.ndarray,
    atlas_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build context-level and per-unit signed temporal metrics."""

    balance, opponent_ratio, crossings = signed_metrics(temporal)
    peak_balance, peak_opponent_ratio, peak_crossings = signed_metrics(peak_trace)
    rank1_stability, rank2_stability, rank4_stability = context_stability(temporal)
    n_image, n_unit, n_component = temporal.shape[:3]

    context_rows: list[dict[str, Any]] = []
    for image in range(n_image):
        for unit in range(n_unit):
            row: dict[str, Any] = {
                "context_index": image,
                "unit_index": unit,
                "rank1_energy_fraction": float(fraction[image, unit, 0]),
                "rank2_cumulative_energy_fraction": float(fraction[image, unit, :2].sum()),
                "rank4_cumulative_energy_fraction": float(fraction[image, unit, :4].sum()),
                "peak_pixel_biphasic_balance": float(peak_balance[image, unit]),
                "peak_pixel_opponent_peak_ratio": float(peak_opponent_ratio[image, unit]),
                "peak_pixel_robust_zero_crossings": int(peak_crossings[image, unit]),
            }
            for component in range(n_component):
                prefix = f"component{component + 1}"
                row[f"{prefix}_biphasic_balance"] = float(balance[image, unit, component])
                row[f"{prefix}_opponent_peak_ratio"] = float(
                    opponent_ratio[image, unit, component]
                )
                row[f"{prefix}_robust_zero_crossings"] = int(
                    crossings[image, unit, component]
                )
            context_rows.append(row)
    context = pd.DataFrame(context_rows)

    metadata = atlas_summary.sort_values("unit_index").reset_index(drop=True)
    unit_rows: list[dict[str, Any]] = []
    for unit in range(n_unit):
        current = context.loc[context.unit_index.eq(unit)]
        component1_balance = current.component1_biphasic_balance.to_numpy()
        component1_crossings = current.component1_robust_zero_crossings.to_numpy()
        peak_balance_unit = current.peak_pixel_biphasic_balance.to_numpy()
        peak_crossings_unit = current.peak_pixel_robust_zero_crossings.to_numpy()
        unit_rows.append(
            {
                "unit_index": unit,
                "unit_label": metadata.loc[unit, "unit_label"],
                "rf_group": metadata.loc[unit, "rf_group"],
                "sf_split_metric": float(metadata.loc[unit, "sf_split_metric"]),
                "median_rank1_energy_fraction": float(current.rank1_energy_fraction.median()),
                "median_rank2_cumulative_energy_fraction": float(
                    current.rank2_cumulative_energy_fraction.median()
                ),
                "median_rank4_cumulative_energy_fraction": float(
                    current.rank4_cumulative_energy_fraction.median()
                ),
                "median_component1_biphasic_balance": float(np.median(component1_balance)),
                "component1_biphasic_context_fraction": float(
                    np.mean((component1_balance >= 0.5) & (component1_crossings >= 1))
                ),
                "median_component1_opponent_peak_ratio": float(
                    current.component1_opponent_peak_ratio.median()
                ),
                "median_component1_robust_zero_crossings": float(
                    current.component1_robust_zero_crossings.median()
                ),
                "median_peak_pixel_biphasic_balance": float(np.median(peak_balance_unit)),
                "peak_pixel_biphasic_context_fraction": float(
                    np.mean((peak_balance_unit >= 0.5) & (peak_crossings_unit >= 1))
                ),
                "rank1_temporal_context_overlap": float(rank1_stability[unit]),
                "rank2_temporal_subspace_overlap": float(rank2_stability[unit]),
                "rank4_temporal_subspace_overlap": float(rank4_stability[unit]),
            }
        )
    return context, pd.DataFrame(unit_rows)


def normalized_median_traces(temporal: np.ndarray) -> np.ndarray:
    median = np.median(temporal[:, :, 0], axis=0)
    median /= np.maximum(np.max(np.abs(median), axis=1, keepdims=True), 1e-20)
    return median


def group_color(group: str) -> str:
    return LOW_COLOR if group == "lower SF" else HIGH_COLOR


def plot_trace(
    ax: plt.Axes,
    traces: np.ndarray,
    lag_ms: np.ndarray,
    *,
    color: str,
    title: str,
    annotation: str,
) -> None:
    value = np.asarray(traces, dtype=np.float64)
    value = value / np.maximum(np.max(np.abs(value), axis=1, keepdims=True), 1e-20)
    for trace in value:
        ax.plot(-lag_ms, trace, color=CONTEXT_COLOR, alpha=0.48, lw=0.65)
    median = np.median(value, axis=0)
    median /= max(float(np.max(np.abs(median))), 1e-20)
    ax.plot(-lag_ms, median, color=color, lw=1.5)
    ax.axhline(0.0, color="0.78", lw=0.55)
    ax.axvline(0.0, color="0.88", lw=0.5)
    ax.set_xlim(-lag_ms[-1], 0.0)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title(title, loc="left", pad=1.5)
    ax.text(0.02, 0.03, annotation, transform=ax.transAxes, fontsize=6.4, color="0.28")


def save_all_unit_atlas(
    temporal: np.ndarray,
    unit: pd.DataFrame,
    output_dir: Path,
    lag_ms: np.ndarray,
) -> None:
    fig, axes = plt.subplots(10, 10, figsize=(20, 19), sharex=True, sharey=True)
    for unit_index, ax in enumerate(axes.flat):
        row = unit.iloc[unit_index]
        plot_trace(
            ax,
            temporal[:, unit_index, 0],
            lag_ms,
            color=group_color(str(row.rf_group)),
            title=f"{row.unit_label}",
            annotation=(
                f"E1={row.median_rank1_energy_fraction:.2f}  "
                f"B={row.median_component1_biphasic_balance:.2f}  "
                f"C={row.rank1_temporal_context_overlap:.2f}"
            ),
        )
        if unit_index // 10 == 9:
            ax.set_xticks([-250, -125, 0])
        else:
            ax.set_xticks([])
        if unit_index % 10 == 0:
            ax.set_yticks([-1, 0, 1])
        else:
            ax.set_yticks([])
    fig.supxlabel("time relative to scored response (ms)")
    fig.supylabel("signed leading temporal mode (peak-normalized)")
    fig.suptitle(
        "All RR100 signed temporal modes at 1× FEM\n"
        "grey = 8 natural-image tangents; color = median; E1 = rank-1 energy, "
        "B = biphasic balance, C = context overlap",
        y=1.002,
    )
    fig.tight_layout(pad=0.65)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"all_units_signed_temporal_atlas.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    with PdfPages(output_dir / "all_units_signed_temporal_multipage.pdf") as pdf:
        for start in range(0, len(unit), 20):
            fig, axes = plt.subplots(5, 4, figsize=(11, 12), sharex=True, sharey=True)
            for offset, ax in enumerate(axes.flat):
                unit_index = start + offset
                if unit_index >= len(unit):
                    ax.axis("off")
                    continue
                row = unit.iloc[unit_index]
                plot_trace(
                    ax,
                    temporal[:, unit_index, 0],
                    lag_ms,
                    color=group_color(str(row.rf_group)),
                    title=f"{row.unit_label}  ({row.rf_group})",
                    annotation=(
                        f"E1={row.median_rank1_energy_fraction:.2f}, "
                        f"B={row.median_component1_biphasic_balance:.2f}, "
                        f"C={row.rank1_temporal_context_overlap:.2f}"
                    ),
                )
            fig.supxlabel("time relative to response (ms)")
            fig.supylabel("signed leading temporal mode")
            fig.suptitle(f"RR100 units {start:03d}–{min(start + 19, len(unit) - 1):03d}")
            fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_summary_figure(
    context: pd.DataFrame,
    unit: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))

    ax = axes[0, 0]
    rank = np.arange(1, 9)
    cumulative = np.column_stack(
        [
            context.rank1_energy_fraction.to_numpy(),
            context.rank2_cumulative_energy_fraction.to_numpy(),
            context.rank4_cumulative_energy_fraction.to_numpy(),
        ]
    )
    # Compute all eight ranks from the saved per-context spectrum elsewhere is
    # unnecessary for the headline: show the measured landmarks explicitly.
    x = np.asarray([1, 2, 4])
    q = np.quantile(cumulative, [0.25, 0.5, 0.75], axis=0)
    ax.fill_between(x, q[0], q[2], color=NEUTRAL, alpha=0.16)
    ax.plot(x, q[1], marker="o", color=NEUTRAL)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of temporal modes")
    ax.set_ylabel("fraction of Jacobian energy")
    ax.set_title("No single temporal filter dominates")

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 31)
    ax.hist(
        context.component1_biphasic_balance,
        bins=bins,
        alpha=0.72,
        color=NEUTRAL,
        label="leading SVD mode",
    )
    ax.hist(
        context.peak_pixel_biphasic_balance,
        bins=bins,
        histtype="step",
        lw=1.5,
        color="#7C3AED",
        label="highest-energy pixel",
    )
    ax.axvline(0.5, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("opponent-sign energy balance (0 mono, 1 balanced)")
    ax.set_ylabel("context × unit Jacobians")
    ax.set_title("Most leading modes are monophasic")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for column, label, color in (
        ("rank1_temporal_context_overlap", "rank 1", NEUTRAL),
        ("rank2_temporal_subspace_overlap", "rank 2", "#7C3AED"),
        ("rank4_temporal_subspace_overlap", "rank 4", "#B7791F"),
    ):
        ax.hist(unit[column], bins=np.linspace(0, 1, 21), histtype="step", lw=1.5, label=label, color=color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("median temporal-subspace overlap across images")
    ax.set_ylabel("units")
    ax.set_title("The leading mode is context dependent")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for group in ("lower SF", "higher SF"):
        current = unit.loc[unit.rf_group.eq(group)]
        ax.scatter(
            current.median_rank1_energy_fraction,
            current.median_component1_biphasic_balance,
            c=group_color(group),
            s=22 + 34 * current.rank1_temporal_context_overlap,
            alpha=0.72,
            edgecolor="none",
            label=group,
        )
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0, 1)
    ax.set_xlabel("median rank-1 energy fraction")
    ax.set_ylabel("median biphasic balance")
    ax.set_title("Few units are both clean and biphasic")
    ax.legend(frameon=False)

    fig.suptitle("Signed temporal audit of RR100 natural-context Jacobians (1× FEM)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"signed_temporal_summary.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def save_temporal_heatmap(
    temporal: np.ndarray,
    unit: pd.DataFrame,
    output_dir: Path,
    lag_ms: np.ndarray,
) -> None:
    trace = normalized_median_traces(temporal)
    order = np.lexsort(
        (
            unit.unit_index.to_numpy(),
            -unit.median_component1_biphasic_balance.to_numpy(),
            unit.rf_group.ne("lower SF").to_numpy(),
        )
    )
    fig, ax = plt.subplots(figsize=(8.2, 10.0))
    image = ax.imshow(
        trace[order, ::-1],
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(-lag_ms[-1], 0, len(order) - 0.5, -0.5),
    )
    ax.set_xlabel("time relative to scored response (ms)")
    ax.set_ylabel("unit (within SF group, sorted by biphasic balance)")
    ax.set_title("Median signed leading temporal mode for every RR100 unit")
    lower_count = int(np.sum(unit.iloc[order].rf_group.eq("lower SF")))
    ax.axhline(lower_count - 0.5, color="black", lw=1.0)
    ax.text(1.01, 0.98, "lower SF", transform=ax.transAxes, va="top", color=LOW_COLOR)
    ax.text(1.01, 0.02, "higher SF", transform=ax.transAxes, va="bottom", color=HIGH_COLOR)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.08)
    colorbar.set_label("signed sensitivity (peak-normalized)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"all_units_signed_temporal_heatmap.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def reconstruct_spatial_mode(
    crop: np.ndarray, temporal_mode: np.ndarray, energy_fraction: float
) -> np.ndarray:
    matrix = np.asarray(crop, dtype=np.float64).reshape(crop.shape[0], -1)
    total_energy = float(np.sum(matrix**2))
    singular = np.sqrt(max(total_energy * float(energy_fraction), 1e-20))
    spatial = temporal_mode @ matrix / singular
    return spatial.reshape(crop.shape[-2:])


def representative_units(unit: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    clean = unit.loc[
        (unit.median_rank1_energy_fraction >= unit.median_rank1_energy_fraction.median())
        & (unit.rank1_temporal_context_overlap >= unit.rank1_temporal_context_overlap.median())
    ]
    biphasic = clean.sort_values(
        ["median_component1_biphasic_balance", "median_rank1_energy_fraction"],
        ascending=False,
    ).head(4)
    used = set(biphasic.unit_index.astype(int))
    monophasic = clean.loc[
        (~clean.unit_index.isin(used)) & (clean.median_component1_biphasic_balance < 0.12)
    ].sort_values(
        ["median_rank1_energy_fraction", "rank1_temporal_context_overlap"], ascending=False
    ).head(4)
    used.update(monophasic.unit_index.astype(int))
    unstable = unit.loc[~unit.unit_index.isin(used)].sort_values(
        ["median_rank1_energy_fraction", "rank1_temporal_context_overlap"], ascending=True
    ).head(4)
    ids = np.concatenate(
        [
            biphasic.unit_index.to_numpy(dtype=int),
            monophasic.unit_index.to_numpy(dtype=int),
            unstable.unit_index.to_numpy(dtype=int),
        ]
    )
    labels = ["cleaner / most biphasic"] * len(biphasic)
    labels += ["cleaner / monophasic"] * len(monophasic)
    labels += ["least separable / unstable"] * len(unstable)
    return ids, labels


def save_representatives(
    crops: np.ndarray,
    temporal: np.ndarray,
    fraction: np.ndarray,
    unit: pd.DataFrame,
    output_dir: Path,
    lag_ms: np.ndarray,
) -> None:
    ids, categories = representative_units(unit)
    n_row = len(ids)
    fig, axes = plt.subplots(n_row, 5, figsize=(13.5, 2.0 * n_row))
    for row_index, (unit_index, category) in enumerate(zip(ids, categories)):
        metrics = unit.iloc[unit_index]
        median = normalized_median_traces(temporal)[unit_index]
        similarity = temporal[:, unit_index, 0] @ median
        context = int(np.argmax(np.abs(similarity)))
        for component, column in ((0, 0), (1, 2)):
            spatial = reconstruct_spatial_mode(
                crops[context, unit_index],
                temporal[context, unit_index, component],
                fraction[context, unit_index, component],
            )
            vmax = max(float(np.max(np.abs(spatial))), 1e-20)
            extent = np.asarray([-spatial.shape[1] / 2, spatial.shape[1] / 2,
                                 spatial.shape[0] / 2, -spatial.shape[0] / 2]) / PPD
            axes[row_index, column].imshow(
                spatial,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=extent,
                interpolation="nearest",
            )
            axes[row_index, column].axhline(0, color="0.7", lw=0.4)
            axes[row_index, column].axvline(0, color="0.7", lw=0.4)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])

        plot_trace(
            axes[row_index, 1],
            temporal[:, unit_index, 0],
            lag_ms,
            color=group_color(str(metrics.rf_group)),
            title="",
            annotation=f"E1={metrics.median_rank1_energy_fraction:.2f}, B={metrics.median_component1_biphasic_balance:.2f}",
        )
        plot_trace(
            axes[row_index, 3],
            temporal[:, unit_index, 1],
            lag_ms,
            color="#7C3AED",
            title="",
            annotation=f"E1:2={metrics.median_rank2_cumulative_energy_fraction:.2f}",
        )
        cumulative = np.cumsum(fraction[:, unit_index], axis=1)
        axes[row_index, 4].plot(
            np.arange(1, 9), cumulative[:, :8].T, color=CONTEXT_COLOR, alpha=0.45, lw=0.7
        )
        axes[row_index, 4].plot(
            np.arange(1, 9), np.median(cumulative[:, :8], axis=0), color=NEUTRAL, lw=1.5
        )
        axes[row_index, 4].set_ylim(0, 1.02)
        axes[row_index, 4].set_xticks([1, 2, 4, 8])
        axes[row_index, 4].set_ylabel("cum. energy")
        axes[row_index, 0].set_ylabel(f"{metrics.unit_label}\n{category}")

    titles = ("spatial mode 1", "signed temporal mode 1", "spatial mode 2", "signed temporal mode 2", "temporal rank")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for ax in axes[-1, (1, 3)]:
        ax.set_xlabel("time (ms)")
    axes[-1, 4].set_xlabel("modes")
    fig.suptitle(
        "Representative space–time decompositions: the spatial mess and temporal ambiguity are coupled",
        y=1.001,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"representative_spatiotemporal_modes.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_interactive_explorer(
    path: Path,
    *,
    temporal: np.ndarray,
    unit: pd.DataFrame,
    lag_ms: np.ndarray,
) -> None:
    """Write a compact, self-contained unit selector for the signed traces."""

    traces = temporal[:, :, 0].copy()
    traces /= np.maximum(np.max(np.abs(traces), axis=2, keepdims=True), 1e-20)
    payload = {
        "time": np.round(-lag_ms, 3).tolist(),
        "trace": np.round(traces.transpose(1, 0, 2), 4).tolist(),
        "unit": unit.to_dict(orient="records"),
    }
    data = json.dumps(payload, separators=(",", ":"))
    fragment = r'''<div id="rr100-temporal-explorer">
  <div class="rr-summary viz-row" aria-live="polite">
    <span><strong id="rr-label">u000</strong> <span id="rr-group" class="text-muted"></span></span>
    <span>rank-1 energy <strong id="rr-energy"></strong></span>
    <span>biphasic balance <strong id="rr-balance"></strong></span>
    <span>context overlap <strong id="rr-overlap"></strong></span>
  </div>
  <svg id="rr-chart" role="img" aria-labelledby="rr-chart-title rr-chart-desc" viewBox="0 0 720 285">
    <title id="rr-chart-title">Signed leading temporal mode across natural-image contexts</title>
    <desc id="rr-chart-desc">Eight context traces and their median, from 258 milliseconds before the scored response to the response time.</desc>
  </svg>
  <div id="rr-grid" class="rr-grid" role="group" aria-label="Select an RR100 unit"></div>
  <div class="rr-legend text-small text-muted"><span>Cell fill: monophasic</span><span class="rr-ramp" aria-hidden="true"></span><span>balanced biphasic</span></div>
</div>
<style>
#rr100-temporal-explorer { color: var(--foreground); width: 100%; }
#rr100-temporal-explorer .rr-summary { justify-content: space-between; margin-bottom: 0.5rem; }
#rr100-temporal-explorer #rr-chart { width: 100%; height: auto; overflow: visible; }
#rr100-temporal-explorer .rr-axis { stroke: var(--border); stroke-width: 1; }
#rr100-temporal-explorer .rr-zero { stroke: var(--muted-foreground); stroke-width: 1; opacity: 0.45; }
#rr100-temporal-explorer .rr-context { fill: none; stroke: var(--muted-foreground); stroke-width: 1.2; opacity: 0.42; }
#rr100-temporal-explorer .rr-median { fill: none; stroke: var(--viz-series-1); stroke-width: 3; }
#rr100-temporal-explorer .rr-text { fill: var(--muted-foreground); font-size: 12px; }
#rr100-temporal-explorer .rr-grid { display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 0.25rem; margin-top: 0.25rem; }
#rr100-temporal-explorer .rr-unit { min-width: 0; padding-inline: 0; }
#rr100-temporal-explorer .rr-unit[data-band="1"] { background: color-mix(in srgb, var(--viz-series-2) 12%, transparent); }
#rr100-temporal-explorer .rr-unit[data-band="2"] { background: color-mix(in srgb, var(--viz-series-2) 25%, transparent); }
#rr100-temporal-explorer .rr-unit[data-band="3"] { background: color-mix(in srgb, var(--viz-series-2) 42%, transparent); }
#rr100-temporal-explorer .rr-unit[data-band="4"] { background: color-mix(in srgb, var(--viz-series-2) 60%, transparent); }
#rr100-temporal-explorer .rr-legend { display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-top: 0.5rem; }
#rr100-temporal-explorer .rr-ramp { width: 8rem; height: 0.7rem; background: linear-gradient(90deg, transparent, var(--viz-series-2)); }
@media (max-width: 520px) {
  #rr100-temporal-explorer .rr-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
  #rr100-temporal-explorer .rr-summary { justify-content: flex-start; }
}
</style>
<script>
(() => {
  const root = document.getElementById("rr100-temporal-explorer");
  const DATA = __RR100_DATA__;
  const grid = root.querySelector("#rr-grid");
  const svg = root.querySelector("#rr-chart");
  const ns = "http://www.w3.org/2000/svg";
  const W = 720, H = 285, left = 48, right = 18, top = 15, bottom = 38;
  const x = t => left + (t + 258.333) / 258.333 * (W - left - right);
  const y = v => top + (1.08 - v) / 2.16 * (H - top - bottom);
  const make = (tag, attrs = {}) => {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  };
  const linePath = values => values.map((v, i) => `${i ? "L" : "M"}${x(DATA.time[i]).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
  const axis = make("g");
  [-1, 0, 1].forEach(v => {
    const line = make("line", {x1: left, x2: W-right, y1: y(v), y2: y(v), class: v === 0 ? "rr-zero" : "rr-axis"});
    axis.appendChild(line);
    const text = make("text", {x: left-9, y: y(v)+4, "text-anchor": "end", class: "rr-text"});
    text.textContent = v;
    axis.appendChild(text);
  });
  [-250, -125, 0].forEach(v => {
    const text = make("text", {x: x(v), y: H-12, "text-anchor": "middle", class: "rr-text"});
    text.textContent = v;
    axis.appendChild(text);
  });
  const xlabel = make("text", {x: (left+W-right)/2, y: H-1, "text-anchor": "middle", class: "rr-text"});
  xlabel.textContent = "time relative to scored response (ms)";
  axis.appendChild(xlabel);
  svg.appendChild(axis);
  const traceLayer = make("g");
  svg.appendChild(traceLayer);

  function median(values) {
    const sorted = values.slice().sort((a,b) => a-b);
    return (sorted[3] + sorted[4]) / 2;
  }
  function selectUnit(index) {
    const meta = DATA.unit[index];
    root.querySelector("#rr-label").textContent = meta.unit_label;
    root.querySelector("#rr-group").textContent = `(${meta.rf_group})`;
    root.querySelector("#rr-energy").textContent = meta.median_rank1_energy_fraction.toFixed(2);
    root.querySelector("#rr-balance").textContent = meta.median_component1_biphasic_balance.toFixed(2);
    root.querySelector("#rr-overlap").textContent = meta.rank1_temporal_context_overlap.toFixed(2);
    traceLayer.replaceChildren();
    DATA.trace[index].forEach(values => traceLayer.appendChild(make("path", {d: linePath(values), class: "rr-context"})));
    const med = DATA.time.map((_, lag) => median(DATA.trace[index].map(values => values[lag])));
    const scale = Math.max(...med.map(Math.abs), 1e-9);
    traceLayer.appendChild(make("path", {d: linePath(med.map(v => v / scale)), class: "rr-median"}));
    grid.querySelectorAll("button").forEach((button, i) => button.setAttribute("aria-pressed", i === index ? "true" : "false"));
  }
  DATA.unit.forEach((meta, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn viz-tile rr-unit";
    button.textContent = String(index).padStart(3, "0");
    button.dataset.band = String(Math.min(4, Math.floor(meta.median_component1_biphasic_balance * 5)));
    button.setAttribute("aria-label", `${meta.unit_label}, biphasic balance ${meta.median_component1_biphasic_balance.toFixed(2)}`);
    button.addEventListener("click", () => selectUnit(index));
    grid.appendChild(button);
  });
  selectUnit(0);
})();
</script>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment.replace("__RR100_DATA__", data))


def write_readme(
    output_dir: Path,
    *,
    context: pd.DataFrame,
    unit: pd.DataFrame,
    anchor_scale: float,
) -> None:
    strong_component1 = (
        (context.component1_biphasic_balance >= 0.5)
        & (context.component1_robust_zero_crossings >= 1)
    )
    strong_peak = (
        (context.peak_pixel_biphasic_balance >= 0.5)
        & (context.peak_pixel_robust_zero_crossings >= 1)
    )
    separability_balance_rho = unit[
        ["median_rank1_energy_fraction", "median_component1_biphasic_balance"]
    ].corr(method="spearman").iloc[0, 1]
    stability_balance_rho = unit[
        ["rank1_temporal_context_overlap", "median_component1_biphasic_balance"]
    ].corr(method="spearman").iloc[0, 1]
    lines = [
        "# RR100 signed temporal Jacobian audit",
        "",
        f"- Operating point: {anchor_scale:g}× FEM, eight natural-image contexts, all 100 units.",
        f"- Median rank-1 space–time energy fraction: {context.rank1_energy_fraction.median():.3f}.",
        f"- Median rank-2 cumulative energy fraction: {context.rank2_cumulative_energy_fraction.median():.3f}.",
        f"- Median rank-4 cumulative energy fraction: {context.rank4_cumulative_energy_fraction.median():.3f}.",
        f"- Median leading-mode biphasic balance: {context.component1_biphasic_balance.median():.3f}.",
        f"- Strongly biphasic leading modes: {100 * strong_component1.mean():.1f}% of unit × context Jacobians.",
        f"- Strongly biphasic peak-pixel traces: {100 * strong_peak.mean():.1f}% of unit × context Jacobians.",
        f"- Median rank-1 temporal context overlap: {unit.rank1_temporal_context_overlap.median():.3f}.",
        f"- Median rank-4 temporal-subspace overlap: {unit.rank4_temporal_subspace_overlap.median():.3f}.",
        f"- Units whose leading mode is strongly biphasic in at least half of contexts: {int(np.sum(unit.component1_biphasic_context_fraction >= 0.5))}/100.",
        f"- Across units, biphasic balance versus rank-1 separability: Spearman rho = {separability_balance_rho:.3f}.",
        f"- Across units, biphasic balance versus rank-1 context overlap: Spearman rho = {stability_balance_rho:.3f}.",
        "",
        "The biphasic threshold is operational: opponent-sign energy balance >= 0.5 and at least one zero crossing after excluding samples below 10% of the dominant peak. The continuous traces and scores should be interpreted together.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_plotting()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.atlas_dir / "rr100_center_rf_crops.npz") as archive:
        crops = np.asarray(archive["gradient_crops"], dtype=np.float32)
        selected_image_index = np.asarray(archive["selected_image_index"], dtype=np.int64)
        anchor_scale = float(archive["anchor_scale"])
    atlas_summary = pd.read_csv(args.atlas_dir / "rr100_center_rf_summary.csv")
    if crops.shape != (8, 100, 32, 65, 65):
        raise ValueError(crops.shape)
    if not np.isclose(anchor_scale, 1.0):
        raise ValueError(f"Expected 1x FEM gradients, found {anchor_scale:g}x")

    temporal, fraction = temporal_decomposition(crops)
    peak_trace = peak_pixel_traces(crops)
    context, unit = unit_summary(
        temporal=temporal,
        fraction=fraction,
        peak_trace=peak_trace,
        atlas_summary=atlas_summary,
    )
    context["image_index"] = selected_image_index[context.context_index.to_numpy(dtype=int)]
    context.to_csv(args.output_dir / "context_signed_temporal_metrics.csv", index=False)
    unit.to_csv(args.output_dir / "unit_signed_temporal_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / "signed_temporal_modes.npz",
        temporal_modes=temporal.astype(np.float32),
        temporal_energy_fraction=fraction.astype(np.float32),
        peak_pixel_traces=peak_trace.astype(np.float32),
        selected_image_index=selected_image_index,
        lag_before_output_ms=np.arange(crops.shape[2], dtype=np.float64) * DT_MS,
    )

    lag_ms = np.arange(crops.shape[2], dtype=np.float64) * DT_MS
    save_all_unit_atlas(temporal, unit, args.output_dir, lag_ms)
    save_summary_figure(context, unit, args.output_dir)
    save_temporal_heatmap(temporal, unit, args.output_dir, lag_ms)
    save_representatives(crops, temporal, fraction, unit, args.output_dir, lag_ms)
    write_readme(
        args.output_dir,
        context=context,
        unit=unit,
        anchor_scale=anchor_scale,
    )
    if args.interactive_output is not None:
        write_interactive_explorer(
            args.interactive_output,
            temporal=temporal,
            unit=unit,
            lag_ms=lag_ms,
        )

    manifest = {
        "analysis": "signed temporal SVD of exact pre-softplus center-pixel Jacobians",
        "source": str(args.atlas_dir / "rr100_center_rf_crops.npz"),
        "shape": list(crops.shape),
        "anchor_scale": anchor_scale,
        "selected_image_index": selected_image_index.tolist(),
        "dt_ms": DT_MS,
        "biphasic_balance_definition": "2 * min(positive energy, negative energy) / total signed-trace energy",
        "strong_biphasic_definition": "balance >= 0.5 and >= 1 zero crossing among samples >= 10% max absolute amplitude",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
