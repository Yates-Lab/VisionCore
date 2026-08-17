#!/usr/bin/env python3
"""Test whether the 1x-FEM RR100 local Jacobians share a compact spatial basis.

The analysis deliberately gives the receptive-field interpretation two chances:

1. the signed spatial snapshot used in the all-unit atlas (one population-defined
   peak lag per unit), and
2. the optimal rank-one spatial mode of each complete 32-lag local Jacobian.

Both representations are spatially demeaned and L2-normalized before PCA so that
large-gradient units cannot dominate the basis. Leave-one-image-out projection
measures whether a basis learned at seven natural-image operating points explains
the eighth, rather than merely compressing the images used to fit it.
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
    / "outputs/figures/fig4/nonlinear_phase_causal_v1/diagnostics/rr100_rf_dimensionality_1x"
)
PPD = 37.50476617
DISPLAY_RADIUS_DEG = 0.4
RANKS = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=ATLAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def normalize_spatial_maps(maps: np.ndarray) -> np.ndarray:
    """Spatially demean and unit-normalize maps on their last two axes."""

    value = np.asarray(maps, dtype=np.float64)
    flat = value.reshape(*value.shape[:-2], -1)
    flat = flat - flat.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(flat, axis=-1, keepdims=True)
    return (flat / np.maximum(norm, 1e-20)).reshape(value.shape)


def normalize_spatiotemporal_volumes(volumes: np.ndarray) -> np.ndarray:
    """Remove per-lag spatial DC, then normalize each complete space-time volume."""

    value = np.asarray(volumes, dtype=np.float64)
    spatial = value.reshape(*value.shape[:-2], -1)
    spatial = spatial - spatial.mean(axis=-1, keepdims=True)
    flat = spatial.reshape(*value.shape[:2], -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-20)
    return flat.reshape(value.shape)


def dominant_spatial_modes(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the optimal rank-one spatial mode and its energy fraction."""

    value = np.asarray(crops, dtype=np.float64)
    if value.ndim != 5:
        raise ValueError(f"Expected image x unit x lag x y x x, got {value.shape}")
    n_image, n_unit, n_lag, height, width = value.shape
    matrix = value.reshape(n_image * n_unit, n_lag, height * width)
    gram = matrix @ np.swapaxes(matrix, 1, 2)
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    leading_temporal = eigenvector[:, :, -1]
    spatial = np.einsum("nt,ntp->np", leading_temporal, matrix, optimize=True)
    leading_energy = np.maximum(eigenvalue[:, -1], 0.0)
    spatial /= np.sqrt(np.maximum(leading_energy, 1e-20))[:, None]
    separability = leading_energy / np.maximum(np.maximum(eigenvalue, 0.0).sum(axis=1), 1e-20)
    return spatial.reshape(n_image, n_unit, height, width), separability.reshape(
        n_image, n_unit
    )


def orient_modes_to_snapshots(modes: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    """Resolve SVD sign using the corresponding peak-lag signed snapshot."""

    value = np.asarray(modes, dtype=np.float64).copy()
    reference = np.asarray(snapshots, dtype=np.float64)
    dot = np.sum(value * reference, axis=(-2, -1), keepdims=True)
    return value * np.where(dot < 0.0, -1.0, 1.0)


def exact_pca(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact uncentered PCA components and fractional map energy."""

    value = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    _, singular, components = np.linalg.svd(value, full_matrices=False)
    energy = singular**2
    return components, energy / np.maximum(energy.sum(), 1e-20)


def dimensionality_metrics(fraction: np.ndarray) -> dict[str, float]:
    cumulative = np.cumsum(np.asarray(fraction, dtype=np.float64))
    result: dict[str, float] = {
        "participation_ratio": float(1.0 / np.maximum(np.sum(fraction**2), 1e-20))
    }
    for threshold in (0.5, 0.8, 0.9):
        result[f"rank_{int(100 * threshold)}pct"] = int(np.searchsorted(cumulative, threshold) + 1)
    return result


def crossed_variance_decomposition(maps: np.ndarray) -> dict[str, float]:
    """Decompose a balanced image x unit array into main effects and interaction."""

    value = np.asarray(maps, dtype=np.float64)
    if value.ndim < 3:
        raise ValueError(value.shape)
    flat = value.reshape(value.shape[0], value.shape[1], -1)
    grand = flat.mean(axis=(0, 1))
    total = float(np.sum((flat - grand) ** 2))
    image_mean = flat.mean(axis=1)
    unit_mean = flat.mean(axis=0)
    image_ss = float(flat.shape[1] * np.sum((image_mean - grand) ** 2))
    unit_ss = float(flat.shape[0] * np.sum((unit_mean - grand) ** 2))
    additive = image_mean[:, None] + unit_mean[None] - grand
    interaction_ss = float(np.sum((flat - additive) ** 2))
    denominator = max(total, 1e-20)
    return {
        "image_fraction": image_ss / denominator,
        "unit_fraction": unit_ss / denominator,
        "image_by_unit_fraction": interaction_ss / denominator,
    }


def leave_one_image_out_recovery(
    maps: np.ndarray, image_ids: np.ndarray, ranks: np.ndarray = RANKS
) -> pd.DataFrame:
    """Project each held-out image through PCA learned on all other images."""

    value = np.asarray(maps, dtype=np.float64)
    n_image, n_unit = value.shape[:2]
    flat = value.reshape(n_image, n_unit, -1)
    rows: list[dict[str, Any]] = []
    for held in range(n_image):
        train = np.delete(flat, held, axis=0).reshape((n_image - 1) * n_unit, -1)
        _, _, components = np.linalg.svd(train, full_matrices=False)
        scores = flat[held] @ components[: int(ranks.max())].T
        denominator = np.maximum(np.sum(flat[held] ** 2, axis=1), 1e-20)
        for rank in ranks:
            recovered = np.sum(scores[:, : int(rank)] ** 2, axis=1) / denominator
            rows.extend(
                {
                    "heldout_image_index": int(image_ids[held]),
                    "unit_index": int(unit),
                    "rank": int(rank),
                    "recovered_energy_fraction": float(recovered[unit]),
                }
                for unit in range(n_unit)
            )
    return pd.DataFrame(rows)


def leave_one_image_out_same_unit_span(
    maps: np.ndarray, image_ids: np.ndarray, ranks: tuple[int, ...] = (1, 2, 4, 7)
) -> pd.DataFrame:
    """Ask whether seven contexts of one unit span its eighth-context map."""

    value = np.asarray(maps, dtype=np.float64)
    n_image, n_unit = value.shape[:2]
    flat = value.reshape(n_image, n_unit, -1)
    if max(ranks) > n_image - 1:
        raise ValueError("Same-unit rank cannot exceed the number of training images")
    rows: list[dict[str, Any]] = []
    for held in range(n_image):
        for unit in range(n_unit):
            train = np.delete(flat[:, unit], held, axis=0)
            _, _, components = np.linalg.svd(train, full_matrices=False)
            scores = flat[held, unit] @ components[: max(ranks)].T
            denominator = max(float(np.sum(flat[held, unit] ** 2)), 1e-20)
            for rank in ranks:
                rows.append(
                    {
                        "heldout_image_index": int(image_ids[held]),
                        "unit_index": int(unit),
                        "rank": int(rank),
                        "recovered_energy_fraction": float(np.sum(scores[:rank] ** 2) / denominator),
                    }
                )
    return pd.DataFrame(rows)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _signed_map(ax: plt.Axes, value: np.ndarray, title: str = "") -> None:
    vmax = max(float(np.max(np.abs(value))), 1e-12)
    half_y = (value.shape[0] - 1) / (2 * PPD)
    half_x = (value.shape[1] - 1) / (2 * PPD)
    ax.imshow(
        value,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=(-half_x, half_x, -half_y, half_y),
    )
    ax.axhline(0, color="0.35", lw=0.25, alpha=0.4)
    ax.axvline(0, color="0.35", lw=0.25, alpha=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=1.5)


def component_figure(
    components: dict[str, np.ndarray], shape: tuple[int, int], output_dir: Path
) -> None:
    _style()
    fig, axes = plt.subplots(4, 6, figsize=(8.2, 5.6), constrained_layout=True)
    labels = (("snapshot", "Peak-lag snapshots"), ("dominant_mode", "Best temporal rank-one modes"))
    for block, (key, label) in enumerate(labels):
        for column in range(12):
            ax = axes[2 * block + column // 6, column % 6]
            value = components[key][column].reshape(shape).copy()
            index = np.unravel_index(np.argmax(np.abs(value)), value.shape)
            if value[index] < 0:
                value = -value
            _signed_map(ax, value, title=f"PC {column + 1}")
            if column == 0:
                ax.set_ylabel(label, fontsize=7.2)
    fig.suptitle(
        "Leading spatial PCA modes across all 800 RR100 local Jacobians",
        fontsize=11,
        fontweight="bold",
    )
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output_dir / f"rr100_rf_pca_components.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def summary_figure(
    fractions: dict[str, np.ndarray],
    context_fraction: np.ndarray,
    recovery: pd.DataFrame,
    decompositions: pd.DataFrame,
    output_dir: Path,
) -> None:
    _style()
    colors = {"snapshot": "#0072b2", "dominant_mode": "#d55e00"}
    labels = {"snapshot": "peak-lag snapshot", "dominant_mode": "best temporal mode"}
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.0), constrained_layout=True)

    ax = axes[0]
    for key in ("snapshot", "dominant_mode"):
        cumulative = np.cumsum(fractions[key])
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=colors[key], label=labels[key])
    ax.plot(
        np.arange(1, len(context_fraction) + 1),
        np.cumsum(context_fraction),
        color="0.3",
        linestyle="--",
        label="snapshot, one image",
    )
    for level in (0.5, 0.8, 0.9):
        ax.axhline(level, color="0.8", lw=0.6)
    ax.set_xlim(1, 130)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("cumulative normalized energy")
    ax.set_title("In-sample dimensionality")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    grouped = recovery.groupby(["representation", "rank"]).recovered_energy_fraction
    for key in ("snapshot", "dominant_mode"):
        med = grouped.median().loc[key]
        low = grouped.quantile(0.25).loc[key]
        high = grouped.quantile(0.75).loc[key]
        rank = med.index.to_numpy(float)
        ax.plot(rank, med, marker="o", ms=3, color=colors[key], label=labels[key])
        ax.fill_between(rank, low, high, color=colors[key], alpha=0.18, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS, [str(rank) for rank in RANKS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("rank learned from seven images")
    ax.set_ylabel("held-out map energy recovered")
    ax.set_title("Leave-one-image-out generalization")

    ax = axes[2]
    categories = ["unit", "image", "image × unit"]
    x = np.arange(len(categories))
    width = 0.36
    for offset, key in ((-width / 2, "snapshot"), (width / 2, "dominant_mode")):
        row = decompositions.set_index("representation").loc[key]
        values = [row.unit_fraction, row.image_fraction, row.image_by_unit_fraction]
        ax.bar(x + offset, values, width=width, color=colors[key], label=labels[key])
        for xpos, value in zip(x + offset, values, strict=True):
            ax.text(xpos, value + 0.025, f"{100 * value:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("fraction of normalized-map variance")
    ax.set_title("What identifies a local RF?")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "RR100 local Jacobians are compressible within one image but not stable across images",
        fontsize=10.5,
        fontweight="bold",
    )
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output_dir / f"rr100_rf_pca_summary.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def heldout_contact_sheets(
    snapshots: np.ndarray,
    image_ids: np.ndarray,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Render every unit before and after honest cross-image PCA compression."""

    _style()
    target = int(np.flatnonzero(image_ids == 15)[0]) if 15 in image_ids else 0
    train = np.delete(snapshots, target, axis=0).reshape(-1, np.prod(snapshots.shape[-2:]))
    _, _, components = np.linalg.svd(train, full_matrices=False)
    test = snapshots[target].reshape(snapshots.shape[1], -1)
    ordered = summary.sort_values(["sf_split_metric", "unit_index"]).unit_index.to_numpy(int)
    lookup = summary.set_index("unit_index")
    reconstructions = {
        rank: ((test @ components[:rank].T) @ components[:rank]).reshape(
            snapshots.shape[1], *snapshots.shape[-2:]
        )
        for rank in (16, 64)
    }
    for rank, reconstruction in reconstructions.items():
        fig, axes = plt.subplots(10, 10, figsize=(11.5, 11.5), constrained_layout=True)
        for ax, unit in zip(axes.flat, ordered, strict=True):
            row = lookup.loc[int(unit)]
            _signed_map(ax, reconstruction[unit], title=f"u{unit:03d}")
            color = "#0077b6" if row.rf_group == "lower SF" else "#d55e00"
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(0.8)
        fig.suptitle(
            f"All RR100 image-{image_ids[target]} maps reconstructed at rank {rank}\n"
            "basis learned from the other seven images; maps independently normalized",
            fontsize=11,
            fontweight="bold",
        )
        for suffix in ("pdf", "png"):
            kwargs = {"dpi": 300} if suffix == "png" else {}
            fig.savefig(
                output_dir / f"rr100_rf_heldout_rank{rank}_contact_sheet.{suffix}",
                bbox_inches="tight",
                **kwargs,
            )
        plt.close(fig)

    pdf_path = output_dir / "rr100_rf_raw_vs_low_rank_multipage.pdf"
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(ordered), 10):
            units = ordered[start : start + 10]
            fig, axes = plt.subplots(
                len(units), 3, figsize=(5.7, 1.5 * len(units)), constrained_layout=True
            )
            for row_index, unit in enumerate(units):
                row = lookup.loc[int(unit)]
                maps = (
                    snapshots[target, unit],
                    reconstructions[16][unit],
                    reconstructions[64][unit],
                )
                for column, (value, label) in enumerate(
                    zip(maps, ("raw", "rank 16", "rank 64"), strict=True)
                ):
                    _signed_map(axes[row_index, column], value, title=label if row_index == 0 else "")
                axes[row_index, 0].set_ylabel(
                    f"u{unit:03d} | {'L' if row.rf_group == 'lower SF' else 'H'}", fontsize=7
                )
            fig.suptitle(
                f"Held-out image {image_ids[target]}: raw versus cross-image PCA reconstruction",
                fontsize=10,
                fontweight="bold",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    atlas_dir = args.atlas_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(atlas_dir / "rr100_center_rf_crops.npz") as archive:
        crops = np.asarray(archive["gradient_crops"], dtype=np.float32)
        image_ids = np.asarray(archive["selected_image_index"], dtype=int)
        anchor_scale = float(np.asarray(archive["anchor_scale"]).item())
    summary = pd.read_csv(atlas_dir / "rr100_center_rf_summary.csv").sort_values("unit_index")
    peaks = summary.peak_lag_index_current_zero.to_numpy(int)

    display_radius_px = int(round(DISPLAY_RADIUS_DEG * PPD))
    center = crops.shape[-1] // 2
    spatial_slice = slice(center - display_radius_px, center + display_radius_px + 1)
    local = crops[:, :, :, spatial_slice, spatial_slice]
    snapshots = np.stack(
        [local[image, np.arange(local.shape[1]), peaks] for image in range(local.shape[0])]
    )
    modes, temporal_separability = dominant_spatial_modes(local)
    modes = orient_modes_to_snapshots(modes, snapshots)
    representations = {
        "snapshot": normalize_spatial_maps(snapshots),
        "dominant_mode": normalize_spatial_maps(modes),
    }

    components: dict[str, np.ndarray] = {}
    fractions: dict[str, np.ndarray] = {}
    dimensionality_rows: list[dict[str, Any]] = []
    recovery_frames: list[pd.DataFrame] = []
    same_unit_frames: list[pd.DataFrame] = []
    decomposition_rows: list[dict[str, Any]] = []
    for key, value in representations.items():
        component, fraction = exact_pca(value.reshape(-1, *value.shape[-2:]))
        components[key] = component
        fractions[key] = fraction
        dimensionality_rows.append(
            {"representation": key, "scope": "all_images", **dimensionality_metrics(fraction)}
        )
        recovery = leave_one_image_out_recovery(value, image_ids)
        recovery.insert(0, "representation", key)
        recovery_frames.append(recovery)
        same_unit = leave_one_image_out_same_unit_span(value, image_ids)
        same_unit.insert(0, "representation", key)
        same_unit_frames.append(same_unit)
        decomposition_rows.append(
            {"representation": key, **crossed_variance_decomposition(value)}
        )

    target = int(np.flatnonzero(image_ids == 15)[0]) if 15 in image_ids else 0
    _, context_fraction = exact_pca(representations["snapshot"][target])
    dimensionality_rows.append(
        {
            "representation": "snapshot",
            "scope": f"image_{image_ids[target]}",
            **dimensionality_metrics(context_fraction),
        }
    )
    dimensionality = pd.DataFrame(dimensionality_rows)
    recovery = pd.concat(recovery_frames, ignore_index=True)
    same_unit = pd.concat(same_unit_frames, ignore_index=True)
    full_spatiotemporal = normalize_spatiotemporal_volumes(local)
    full_span = leave_one_image_out_same_unit_span(full_spatiotemporal, image_ids)
    full_span.insert(0, "representation", "full_spatiotemporal")
    same_unit = pd.concat([same_unit, full_span], ignore_index=True)
    decompositions = pd.DataFrame(decomposition_rows)
    dimensionality.to_csv(output_dir / "rr100_rf_dimensionality_summary.csv", index=False)
    recovery.to_csv(output_dir / "rr100_rf_heldout_reconstruction.csv", index=False)
    same_unit.to_csv(output_dir / "rr100_rf_same_unit_context_span.csv", index=False)
    decompositions.to_csv(output_dir / "rr100_rf_variance_decomposition.csv", index=False)
    np.savez_compressed(
        output_dir / "rr100_rf_pca_components.npz",
        snapshot_components=components["snapshot"],
        snapshot_fraction=fractions["snapshot"],
        dominant_mode_components=components["dominant_mode"],
        dominant_mode_fraction=fractions["dominant_mode"],
        image15_snapshot_fraction=context_fraction,
        temporal_separability=temporal_separability,
        image_index=image_ids,
    )

    component_figure(components, snapshots.shape[-2:], output_dir)
    summary_figure(fractions, context_fraction, recovery, decompositions, output_dir)
    heldout_contact_sheets(representations["snapshot"], image_ids, summary, output_dir)

    selected = recovery.loc[recovery["rank"].isin([16, 64])].groupby(
        ["representation", "rank"]
    ).recovered_energy_fraction
    medians = selected.median()
    same_unit_rank7 = same_unit.loc[same_unit["rank"].eq(7)].groupby(
        "representation"
    ).recovered_energy_fraction.median()
    metrics = {
        "analysis": "PCA dimensionality of all RR100 local Jacobians",
        "analysis_scope": (
            "population-pooled PCA over eight contexts; not the earlier per-unit PCA over "
            "approximately 5,000 high-response Jacobians"
        ),
        "anchor_scale": anchor_scale,
        "n_images": int(crops.shape[0]),
        "n_units": int(crops.shape[1]),
        "display_crop_shape": list(snapshots.shape[-2:]),
        "normalization": "spatial mean removed and each map L2-normalized",
        "pca": "uncentered exact SVD after per-map normalization",
        "cross_validation": "leave one natural image out; basis fit to all units in the other seven",
        "dimensionality": dimensionality.to_dict(orient="records"),
        "variance_decomposition": decompositions.to_dict(orient="records"),
        "heldout_recovery_median": {
            f"{representation}_rank{rank}": float(value)
            for (representation, rank), value in medians.items()
        },
        "same_unit_seven_context_span_heldout_median": {
            str(representation): float(value)
            for representation, value in same_unit_rank7.items()
        },
        "median_temporal_rank1_energy_fraction": float(np.median(temporal_separability)),
        "finite": bool(
            all(np.isfinite(value).all() for value in representations.values())
            and np.isfinite(recovery.recovered_energy_fraction).all()
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")

    snapshot_dim = dimensionality.loc[
        (dimensionality.representation == "snapshot") & (dimensionality.scope == "all_images")
    ].iloc[0]
    mode_dim = dimensionality.loc[dimensionality.representation == "dominant_mode"].iloc[0]
    snapshot_decomp = decompositions.set_index("representation").loc["snapshot"]
    mode_decomp = decompositions.set_index("representation").loc["dominant_mode"]
    report = f"""# Dimensionality of RR100 local Jacobians at 1x FEM

This analysis asks whether one population-wide basis compresses 100 units across eight contexts. It is **not** the same analysis as the earlier CNN work, which fit a separate space-time PCA to roughly 5,000 high-response Jacobians for each neuron. Consequently, the pooled PCA cannot establish that a Gabor-like unit-specific subspace is absent.

After removing each map's spatial mean and normalizing its L2 energy, the pooled 800 peak-lag maps require rank {int(snapshot_dim.rank_50pct)}, {int(snapshot_dim.rank_80pct)}, and {int(snapshot_dim.rank_90pct)} to retain 50%, 80%, and 90% of their energy. Replacing each 32-lag Jacobian by its optimal rank-one spatial mode changes those ranks to {int(mode_dim.rank_50pct)}, {int(mode_dim.rank_80pct)}, and {int(mode_dim.rank_90pct)}.

The honest cross-image test is more revealing. A rank-16 basis learned from seven images recovers a median {100 * medians.loc[('snapshot', 16)]:.1f}% of a held-out peak-lag map and {100 * medians.loc[('dominant_mode', 16)]:.1f}% of a held-out optimal spatial mode. Rank 64 raises this to {100 * medians.loc[('snapshot', 64)]:.1f}% and {100 * medians.loc[('dominant_mode', 64)]:.1f}%, respectively.

The strongest available fixed-unit check is unfavorable but data-limited: the complete seven-dimensional span of one unit's maps at the other seven images recovers a median of only {100 * same_unit_rank7.loc['snapshot']:.1f}% of its held-out snapshot, {100 * same_unit_rank7.loc['dominant_mode']:.1f}% of its held-out optimal spatial mode, and {100 * same_unit_rank7.loc['full_spatiotemporal']:.1f}% of its full space-time Jacobian. This is inconsistent with a simple fixed quadrature pair on these eight contexts, but eight distinct natural images are too sparse to reproduce the earlier per-unit experiment.

In a balanced image x unit decomposition, stable unit identity explains {100 * snapshot_decomp.unit_fraction:.1f}% of snapshot variance, image identity explains {100 * snapshot_decomp.image_fraction:.1f}%, and the image-by-unit interaction accounts for {100 * snapshot_decomp.image_by_unit_fraction:.1f}%. For the best temporal spatial modes those fractions are {100 * mode_decomp.unit_fraction:.1f}%, {100 * mode_decomp.image_fraction:.1f}%, and {100 * mode_decomp.image_by_unit_fraction:.1f}%. The median first spatiotemporal SVD mode itself captures only {100 * np.median(temporal_separability):.1f}% of each Jacobian's energy.

Interpretation: the maps have a moderately compressible population basis within one operating point, but the eight-context results do not support a simple fixed-filter account. The leading population components can look more organized than individual maps because PCA denoises and pools them. A fair comparison to the earlier encouraging CNN result requires per-unit space-time PCA over hundreds to thousands of response-selected Jacobians; until that is run, absence of a Gabor-like per-unit subspace remains unresolved. These results challenge mechanistic receptive-field interpretation of the twin, not its separately measured predictive accuracy.
"""
    (output_dir / "README.md").write_text(report)
    print(json.dumps(_json_ready(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
