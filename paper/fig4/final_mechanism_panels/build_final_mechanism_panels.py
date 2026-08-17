#!/usr/bin/env python3
"""Build the final Figure 4 mechanism panels D--F from validated caches only.

This script is deliberately plotting-only.  It does not import or instantiate
the model.  Every value comes from validated corrected-history outputs already
saved by the mechanism analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[3]
MECHANISM = ROOT / "outputs/figures/fig4/mechanism_audit_v1"
SFTF = MECHANISM / "layerwise_sftf_v1"
PHASE = MECHANISM / "phase_spatial_followup_v1"
SSI = MECHANISM / "ssi_mechanism_v2"
OUT = ROOT / "outputs/figures/fig4/final_mechanism_panels"
DATA = OUT / "plot_data"

FEM_SOURCE = SFTF / "exact_arrays/corrected_fem_retinal_sftf_spectrum.npz"
PREDICTION_SOURCE = SSI / "plot_data/authoritative_population_prediction_vs_ssi.csv"
SSI_CURVE_SOURCE = PHASE / "plot_data/exact_final_ssi_curves.csv"
METRICS_SOURCE = PHASE / "exact_arrays/exact_phase_spatial_metrics.npz"
MAP_SOURCE = PHASE / "exact_arrays/representative_signed_maps.npz"
CONTRAST_SOURCE = SSI / "plot_data/rate_contrast_diagnostics.csv"

LOW = "#0072B2"
HIGH = "#D55E00"
GROUP_COLOR = {"lower SF": LOW, "higher SF": HIGH}
GROUP_COUNT = {"lower SF": 71, "higher SF": 29}
GROUP_KEY = {"lower SF": "low", "higher SF": "high"}
GROUP_FROM_SOURCE = {"low SF": "lower SF", "high SF": "higher SF"}
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0])
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def geometric_edges(centers: np.ndarray) -> np.ndarray:
    value = np.log2(np.asarray(centers, dtype=float))
    edges = np.r_[
        value[0] - (value[1] - value[0]) / 2,
        (value[:-1] + value[1:]) / 2,
        value[-1] + (value[-1] - value[-2]) / 2,
    ]
    return 2**edges


def load_panel_d() -> tuple[pd.DataFrame, dict[str, object]]:
    with np.load(FEM_SOURCE) as archive:
        scales = archive["scales"].astype(float)
        sf = archive["spatial_cpd"].astype(float)
        tf = archive["temporal_hz"].astype(float)
        orientation = archive["orientation_deg"].astype(float)
        if "trajectory_phase_modulation_power" in archive.files:
            phase_power = archive["trajectory_phase_modulation_power"].astype(float)
        elif "finite_window_phase_modulation_power" in archive.files:
            # Backward-compatible migration of caches written before the
            # trajectory-phase field became the primary public contract.
            phase_power = archive["finite_window_phase_modulation_power"].astype(float)
        else:
            raise RuntimeError(
                "FEM cache has only instantaneous |k.v| occupancy, which is not a temporal PSD"
            )
        total_power = phase_power.sum(axis=(1, 2, 3))
        definition = str(archive["definition"].item())
        method_version = (
            str(archive["method_version"].item())
            if "method_version" in archive.files
            else "legacy_finite_trajectory_phase_spectrum"
        )
        image_ids = archive["selected_image_index"].astype(int)
        trace_ids = archive["selected_trace_index"].astype(int)
    rows = []
    for scale in (1.0, 3.0):
        si = int(np.flatnonzero(np.isclose(scales, scale))[0])
        collapsed = phase_power[si].sum(axis=-1)
        percent = 100.0 * collapsed / total_power[si]
        for sf_i, sf_value in enumerate(sf):
            for tf_i, tf_value in enumerate(tf):
                rows.append(
                    {
                        "movement_scale": scale,
                        "spatial_frequency_cpd": sf_value,
                        "temporal_frequency_hz": tf_value,
                        "trajectory_phase_power_raw": collapsed[sf_i, tf_i],
                        "trajectory_phase_power_percent_in_grid": percent[sf_i, tf_i],
                    }
                )
    table = pd.DataFrame(rows)
    metadata = {
        "definition": definition,
        "method_version": method_version,
        "orientation_bins_collapsed_deg": orientation.tolist(),
        "n_selected_images": len(image_ids),
        "selected_image_indices": image_ids.tolist(),
        "n_selected_corrected_trajectories": len(trace_ids),
        "selected_trace_indices": trace_ids.tolist(),
        "normalization": "percent of trajectory-phase dynamic power represented on the plotted SFxTFxorientation grid, separately for each motion scale",
    }
    return table, metadata


def load_panel_e() -> pd.DataFrame:
    predicted = pd.read_csv(PREDICTION_SOURCE).rename(
        columns={"figure4_sf_group": "group_key"}
    )
    if "spectrum_method_version" not in predicted or not predicted.spectrum_method_version.astype(str).str.startswith("trajectory_phase").all():
        raise RuntimeError(
            "Stale Panel E prediction: rerun FEM overlap with complete trajectory-phase spectra"
        )
    predicted["sf_group"] = predicted.group_key.map({"low": "lower SF", "high": "higher SF"})
    observed = pd.read_csv(SSI_CURVE_SOURCE)
    observed["sf_group"] = observed.sf_group.map(GROUP_FROM_SOURCE)
    observed = observed.rename(
        columns={
            "ssi_percent_vs_0x": "ssi_percent_vs_stabilized",
            "ci95_low": "ssi_percent_ci95_low",
            "ci95_high": "ssi_percent_ci95_high",
        }
    )
    result = predicted[
        ["sf_group", "scale", "raw_prediction", "prediction_normalized"]
    ].merge(
        observed[
            [
                "sf_group",
                "scale",
                "ssi_percent_vs_stabilized",
                "ssi_percent_ci95_low",
                "ssi_percent_ci95_high",
            ]
        ],
        on=["sf_group", "scale"],
        validate="one_to_one",
    )
    result["n_units"] = result.sf_group.map(GROUP_COUNT)
    result["is_measured_fem_amplitude"] = np.isclose(result.scale, 1.0)
    result["is_predicted_optimum"] = False
    result["is_observed_ssi_optimum"] = False
    for group, frame in result.groupby("sf_group"):
        result.loc[frame.prediction_normalized.idxmax(), "is_predicted_optimum"] = True
        result.loc[frame.ssi_percent_vs_stabilized.idxmax(), "is_observed_ssi_optimum"] = True
    return result


def summarize_panel_e_claim(table: pd.DataFrame) -> dict[str, object]:
    """Record the deliberately limited claim supported by Panel E."""

    high = table.loc[table.sf_group.eq("higher SF")].set_index("scale")
    low = table.loc[table.sf_group.eq("lower SF")].set_index("scale")
    return {
        "claim_boundary": (
            "SFxTF overlap captures only coarse sampled-peak ordering; it does not "
            "predict the useful movement range, SSI sign or magnitude, or the "
            "higher-SF reversal"
        ),
        "normalization": (
            "each population is normalized separately to its own sampled maximum; "
            "curve heights are not comparable across populations"
        ),
        "highest_sampled_overlap_scale": {
            "lower SF": float(low.prediction_normalized.idxmax()),
            "higher SF": float(high.prediction_normalized.idxmax()),
        },
        "highest_sampled_ssi_scale": {
            "lower SF": float(low.ssi_percent_vs_stabilized.idxmax()),
            "higher SF": float(high.ssi_percent_vs_stabilized.idxmax()),
        },
        "higher_sf_curve": {
            f"{scale:g}x": {
                "prediction_normalized": float(high.loc[scale, "prediction_normalized"]),
                "ssi_percent_vs_stabilized": float(
                    high.loc[scale, "ssi_percent_vs_stabilized"]
                ),
            }
            for scale in (0.5, 1.0, 2.0, 3.0)
        },
        "higher_sf_at_3x": {
            "prediction_normalized": float(high.loc[3.0, "prediction_normalized"]),
            "ssi_percent_vs_stabilized": float(high.loc[3.0, "ssi_percent_vs_stabilized"]),
            "ssi_percent_ci95_low": float(high.loc[3.0, "ssi_percent_ci95_low"]),
            "ssi_percent_ci95_high": float(high.loc[3.0, "ssi_percent_ci95_high"]),
        },
        "structural_limitation": (
            "the nonnegative overlap metric cannot predict a below-stabilization "
            "SSI reversal by construction"
        ),
    }


def weighted_population_ssi(
    ssi: np.ndarray, expected: np.ndarray, image_i: int, trace_i: int, scale_i: int, units: np.ndarray
) -> float:
    numerator = float(np.sum(ssi[image_i, trace_i, scale_i, units] * expected[image_i, trace_i, scale_i, units]))
    denominator = float(np.sum(expected[image_i, trace_i, scale_i, units]))
    return numerator / max(denominator, EPS)


def load_panel_f() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with np.load(MAP_SOURCE) as maps:
        map_scales = maps["scales"].astype(float)
        image_id = int(maps["image_index"])
        trace_id = int(maps["trace_index"])
        scored_output = int(maps["scored_output_index"])
        selection_rule = str(maps["selection_rule"].item())
        population_maps = {
            "lower SF": maps["final_low_population_map"].astype(float),
            "higher SF": maps["final_high_population_map"].astype(float),
        }
    with np.load(METRICS_SOURCE) as metrics:
        all_scales = metrics["scales"].astype(float)
        image_ids = metrics["selected_image_index"].astype(int)
        trace_ids = metrics["selected_trace_index"].astype(int)
        unit_ssi = metrics["ssi"].astype(float)
        expected = metrics["expected_spikes"].astype(float)
        groups = {
            "lower SF": metrics["low_unit_indices"].astype(int),
            "higher SF": metrics["high_unit_indices"].astype(int),
        }
    image_i = int(np.flatnonzero(image_ids == image_id)[0])
    trace_i = int(np.flatnonzero(trace_ids == trace_id)[0])
    requested = {"lower SF": (0.0, 2.0), "higher SF": (0.0, 1.0, 3.0)}
    map_rows = []
    summary_rows = []
    for group, display_scales in requested.items():
        for scale in display_scales:
            map_i = int(np.flatnonzero(np.isclose(map_scales, scale))[0])
            scale_i = int(np.flatnonzero(np.isclose(all_scales, scale))[0])
            rate = population_maps[group][map_i]
            gain = rate / max(float(rate.mean()), EPS)
            exact_ssi = weighted_population_ssi(
                unit_ssi, expected, image_i, trace_i, scale_i, groups[group]
            )
            summary_rows.append(
                {
                    "sf_group": group,
                    "n_units": len(groups[group]),
                    "movement_scale": scale,
                    "image_index": image_id,
                    "trace_index": trace_id,
                    "scored_output_index": scored_output,
                    "exact_population_ssi_bits": exact_ssi,
                    "displayed_population_map_mean_rate": float(rate.mean()),
                    "displayed_population_map_variance_of_g": float(np.var(gain)),
                }
            )
            yy, xx = np.indices(rate.shape)
            for y, x, raw, normalized in zip(
                yy.ravel(), xx.ravel(), rate.ravel(), gain.ravel()
            ):
                map_rows.append(
                    {
                        "sf_group": group,
                        "movement_scale": scale,
                        "x_index": int(x),
                        "y_index": int(y),
                        "population_mean_rate": raw,
                        "g_mean_normalized_population_rate": normalized,
                    }
                )
    contrast = pd.read_csv(CONTRAST_SOURCE)
    contrast = contrast.loc[contrast.scale.gt(0)].copy()
    contrast["sf_group"] = contrast.figure4_sf_group.map(
        {"low": "lower SF", "high": "higher SF"}
    )
    contrast = contrast[
        [
            "image_index",
            "trace_index",
            "sf_group",
            "scale",
            "delta_cv2",
            "delta_ssi",
        ]
    ].rename(
        columns={
            "scale": "movement_scale",
            "delta_cv2": "delta_normalized_spatial_variance",
            "delta_ssi": "delta_exact_ssi_bits",
        }
    )
    pearson = float(
        pearsonr(
            contrast.delta_normalized_spatial_variance,
            contrast.delta_exact_ssi_bits,
        ).statistic
    )
    spearman = float(
        spearmanr(
            contrast.delta_normalized_spatial_variance,
            contrast.delta_exact_ssi_bits,
        ).statistic
    )
    metadata = {
        "representative_image_index": image_id,
        "representative_trace_index": trace_id,
        "displayed_scored_output_index_zero_based": scored_output,
        "selection_rule": selection_rule,
        "map_definition": "group-mean final rate map at the saved representative scored output, divided by its own spatial mean; one common display scale is used across all five maps",
        "ssi_definition": "exact expected-spike-weighted mean of per-unit spatial SSI over all 40 scored outputs for the same representative movie condition",
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "n_inset_observations": len(contrast),
    }
    if not math.isclose(pearson, 0.9042535924702411, rel_tol=0, abs_tol=5e-12):
        raise AssertionError(pearson)
    if not math.isclose(spearman, 0.9363687596559565, rel_tol=0, abs_tol=5e-12):
        raise AssertionError(spearman)
    return pd.DataFrame(map_rows), pd.DataFrame(summary_rows), contrast, metadata


def panel_letter(container, letter: str) -> None:
    container.text(0.004, 0.99, letter, fontsize=12.5, fontweight="bold", va="top", ha="left")


def draw_panel_d(container, table: pd.DataFrame) -> None:
    container.suptitle(
        "Retinal motion transforms spatial structure\ninto temporal modulation",
        x=0.09,
        y=0.99,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="semibold",
    )
    panel_letter(container, "D")
    grid = container.add_gridspec(
        1, 3, width_ratios=[1, 1, 0.055], left=0.12, right=0.96, bottom=0.18, top=0.76, wspace=0.12
    )
    axes = [container.add_subplot(grid[0, i]) for i in range(2)]
    cax = container.add_subplot(grid[0, 2])
    sf = np.sort(table.spatial_frequency_cpd.unique())
    tf = np.sort(table.temporal_frequency_hz.unique())
    sf_edges, tf_edges = geometric_edges(sf), geometric_edges(tf)
    positive = table.loc[table.trajectory_phase_power_percent_in_grid.gt(0), "trajectory_phase_power_percent_in_grid"].to_numpy(float)
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    mesh = None
    for ax, scale in zip(axes, (1.0, 3.0)):
        frame = table.loc[table.movement_scale.eq(scale)]
        value = frame.pivot(
            index="temporal_frequency_hz",
            columns="spatial_frequency_cpd",
            values="trajectory_phase_power_percent_in_grid",
        ).reindex(index=tf, columns=sf).to_numpy(float)
        value[value <= 0] = np.nan
        mesh = ax.pcolormesh(
            sf_edges,
            tf_edges,
            value,
            cmap="magma",
            norm=norm,
            shading="flat",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(sf_edges[0], sf_edges[-1])
        ax.set_ylim(tf_edges[0], tf_edges[-1])
        ax.set_xticks([0.5, 2, 8, 16], ["0.5", "2", "8", "16"])
        ax.set_yticks([0.5, 2, 8, 32], ["0.5", "2", "8", "32"])
        ax.minorticks_off()
        ax.set_title("1× measured motion" if scale == 1 else "3× motion", pad=4)
        ax.set_xlabel("Spatial frequency (cycles/deg)")
    axes[0].set_ylabel("Temporal frequency (Hz)")
    axes[1].tick_params(labelleft=False)
    axes[1].annotate(
        "higher temporal\nfrequency",
        xy=(2.2, 19),
        xytext=(2.2, 3.0),
        ha="center",
        va="center",
        color="white",
        fontsize=6.6,
        arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.0},
    )
    container.text(
        0.50,
        0.835,
        r"$R_{\mathbf{k}}(t)=I_{\mathbf{k}}e^{-i2\pi\mathbf{k}\cdot\mathbf{X}(t)}$",
        ha="center",
        va="center",
        fontsize=9.0,
    )
    assert mesh is not None
    colorbar = container.colorbar(mesh, cax=cax)
    colorbar.set_label("Trajectory-phase dynamic power\n(% in plotted grid; log scale)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.5)


def draw_panel_e(container, table: pd.DataFrame) -> None:
    container.suptitle(
        "Spatiotemporal overlap captures coarse peak ordering,\nnot the useful movement range",
        x=0.12,
        y=0.99,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="semibold",
    )
    panel_letter(container, "E")
    grid = container.add_gridspec(
        2, 1, left=0.19, right=0.96, bottom=0.14, top=0.80, hspace=0.14
    )
    top = container.add_subplot(grid[0, 0])
    bottom = container.add_subplot(grid[1, 0], sharex=top)
    for ax in (top, bottom):
        ax.axvline(1.0, color="#8A8F98", lw=0.8, ls=":", zorder=0)
        ax.grid(axis="y", color="#E5E7EB", lw=0.65, zorder=0)
    for group in ("lower SF", "higher SF"):
        frame = table.loc[table.sf_group.eq(group)].sort_values("scale")
        color = GROUP_COLOR[group]
        top.plot(frame.scale, frame.prediction_normalized, color=color, lw=1.8)
        optimum = frame.loc[frame.is_predicted_optimum].iloc[0]
        top.scatter(
            optimum.scale,
            optimum.prediction_normalized,
            s=44,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            zorder=4,
        )
        bottom.fill_between(
            frame.scale,
            frame.ssi_percent_ci95_low,
            frame.ssi_percent_ci95_high,
            color=color,
            alpha=0.11,
            lw=0,
        )
        bottom.plot(frame.scale, frame.ssi_percent_vs_stabilized, color=color, lw=1.8)
        observed = frame.loc[frame.is_observed_ssi_optimum].iloc[0]
        bottom.scatter(
            observed.scale,
            observed.ssi_percent_vs_stabilized,
            s=38,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    top.set_ylabel("SF×TF spectral overlap\n(within-group maximum = 1)")
    top.set_ylim(-0.05, 1.09)
    top.set_yticks([0, 0.5, 1.0])
    top.tick_params(labelbottom=False)
    top.text(1.02, 0.06, "1× measured\nFEM amplitude", color="#6B7280", fontsize=6.5, ha="left", va="bottom")
    top.annotate(
        "Higher SF (n=29)",
        xy=(0.55, 0.95),
        xytext=(0.12, 0.97),
        color=HIGH,
        fontsize=7.0,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": HIGH, "lw": 0.8},
    )
    top.annotate(
        "Lower SF (n=71)",
        xy=(0.55, 0.82),
        xytext=(0.12, 0.66),
        color=LOW,
        fontsize=7.0,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": LOW, "lw": 0.8},
    )
    high_3x = table.loc[
        table.sf_group.eq("higher SF") & np.isclose(table.scale, 3.0)
    ].iloc[0]
    top.annotate(
        f"3× overlap = {high_3x.prediction_normalized:.3f}\n(continued predicted engagement)",
        xy=(high_3x.scale, high_3x.prediction_normalized),
        xytext=(1.62, 0.73),
        color=HIGH,
        fontsize=6.2,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": HIGH, "lw": 0.8},
    )
    bottom.axhline(0, color="#4B5563", lw=0.75)
    bottom.set_ylabel("Exact SSI change\nfrom stabilization (%)")
    bottom.set_xlabel("Trajectory amplitude (× measured FEM)")
    bottom.set_xticks(SCALES, ["0", "0.5", "1", "2", "3"])
    bottom.annotate(
        "Lower SF",
        xy=(2.8, 22.2),
        xytext=(2.32, 27.0),
        color=LOW,
        fontsize=7.0,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": LOW, "lw": 0.8},
    )
    bottom.annotate(
        f"3× SSI = {high_3x.ssi_percent_vs_stabilized:.1f}%\n(observed reversal)",
        xy=(high_3x.scale, high_3x.ssi_percent_vs_stabilized),
        xytext=(1.98, -11.7),
        color=HIGH,
        fontsize=6.5,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": HIGH, "lw": 0.8},
    )
    bottom.set_ylim(-18, 34)
    top.text(0.98, 0.04, "open: highest sampled overlap", transform=top.transAxes, ha="right", fontsize=6.3, color="#4B5563")
    top.text(0.98, 0.13, "curves normalized separately", transform=top.transAxes, ha="right", fontsize=6.3, color="#6B7280")
    bottom.text(0.98, 0.92, "filled: highest sampled SSI", transform=bottom.transAxes, ha="right", fontsize=6.3, color="#4B5563")


def draw_panel_f(
    container,
    maps: pd.DataFrame,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    container.suptitle(
        "Motion increases SSI by increasing\nnormalized spatial contrast",
        x=0.09,
        y=0.99,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="semibold",
    )
    panel_letter(container, "F")
    outer = container.add_gridspec(
        1, 2, width_ratios=[1.55, 1.08], left=0.07, right=0.98, bottom=0.14, top=0.80, wspace=0.23
    )
    map_grid = outer[0, 0].subgridspec(2, 3, wspace=0.08, hspace=0.34)
    inset = container.add_subplot(outer[0, 1])
    active_axes = []
    display = {
        ("lower SF", 0.0): (0, 0, "Stabilized"),
        ("lower SF", 2.0): (0, 1, "2× movement"),
        ("higher SF", 0.0): (1, 0, "Stabilized"),
        ("higher SF", 1.0): (1, 1, "1× movement"),
        ("higher SF", 3.0): (1, 2, "3× movement"),
    }
    gain_values = maps.g_mean_normalized_population_rate.to_numpy(float)
    vmin, vmax = np.quantile(gain_values, [0.002, 0.998])
    image = None
    for (group, scale), (row, column, title) in display.items():
        ax = container.add_subplot(map_grid[row, column])
        frame = maps.loc[maps.sf_group.eq(group) & maps.movement_scale.eq(scale)]
        value = frame.pivot(
            index="y_index", columns="x_index", values="g_mean_normalized_population_rate"
        ).sort_index().sort_index(axis=1).to_numpy(float)
        image = ax.imshow(
            value,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        exact = float(
            summary.loc[
                summary.sf_group.eq(group) & summary.movement_scale.eq(scale),
                "exact_population_ssi_bits",
            ].iloc[0]
        )
        ax.set_title(title, fontsize=7.4, pad=2)
        ax.set_xlabel(f"exact SSI = {exact:.5f} bits", fontsize=6.4, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color(GROUP_COLOR[group])
        if column == 0:
            ax.set_ylabel(
                f"{'Lower' if group == 'lower SF' else 'Higher'} SF\n(n={GROUP_COUNT[group]})",
                color=GROUP_COLOR[group],
                fontweight="semibold",
                fontsize=7.4,
                labelpad=5,
            )
        active_axes.append(ax)
    blank = container.add_subplot(map_grid[0, 2])
    blank.axis("off")
    assert image is not None
    colorbar = container.colorbar(
        image,
        ax=active_axes,
        location="bottom",
        fraction=0.055,
        pad=0.14,
        aspect=32,
    )
    colorbar.set_label(r"Mean-normalized population rate  $g(x,y)=r(x,y)/\overline{r}_{xy}$", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.3)

    for group in ("lower SF", "higher SF"):
        frame = contrast.loc[contrast.sf_group.eq(group)]
        inset.scatter(
            frame.delta_normalized_spatial_variance,
            frame.delta_exact_ssi_bits,
            s=7,
            alpha=0.22,
            color=GROUP_COLOR[group],
            edgecolor="none",
        )
    inset.axhline(0, color="#9CA3AF", lw=0.65)
    inset.axvline(0, color="#9CA3AF", lw=0.65)
    inset.set_xlabel(r"Δ normalized spatial variance,  $\Delta\,\mathrm{Var}_{xy}[g]$")
    inset.set_ylabel("Δ exact SSI (bits)")
    inset.set_title("All held-out movies", fontsize=7.8, pad=3)
    inset.text(0.04, 0.95, "Lower SF", transform=inset.transAxes, color=LOW, fontsize=6.8, fontweight="semibold", va="top")
    inset.text(0.04, 0.88, "Higher SF", transform=inset.transAxes, color=HIGH, fontsize=6.8, fontweight="semibold", va="top")
    inset.text(
        0.96,
        0.06,
        f"Pearson r = {metadata['pearson_r']:.3f}\nSpearman ρ = {metadata['spearman_rho']:.3f}",
        transform=inset.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#111827",
    )


def write_caption(
    panel_e_metadata: dict[str, object], panel_f_metadata: dict[str, object]
) -> None:
    high_3x = panel_e_metadata["higher_sf_at_3x"]
    caption = f"""# Figure 4D–F caption draft

**D, Retinal motion transforms spatial structure into temporal modulation.** Image-power-weighted trajectory-phase spectra under the corrected measured fixation histories at 1× and 3× trajectory amplitude. For image Fourier mode $\\mathbf{{k}}$, the translated retinal carrier was $R_\\mathbf{{k}}(t)=I_\\mathbf{{k}}\\exp[-i2\\pi\\mathbf{{k}}\\cdot\\mathbf{{X}}(t)]$; its two-sided, mean-removed finite-trajectory temporal spectrum was then estimated and pooled by spatial-frequency and orientation bin. This preserves the temporal ordering and displacement correlations that broaden power under drift or Brownian-like motion. Increasing movement amplitude redistributes spatial image power across temporal frequencies. Both heat maps use identical logarithmic axes and one shared logarithmic scale; 1× denotes the measured FEM amplitude.

**E, Spatiotemporal overlap captures only a coarse ordering of motion-scale preference.** Top, overlap between the exact corrected-FEM spectrum and measured RR100 SF×TF tuning for the historical lower-SF (n=71) and higher-SF (n=29) populations. Curves are normalized separately to their sampled maxima, so their heights cannot be compared across populations; open circles mark the highest sampled overlap, not sharply resolved optima. The higher-SF maximum occurs at a smaller sampled amplitude than the lower-SF maximum (1× versus 3×), matching only the ordinal separation of the SSI maxima (1× versus 2×). Bottom, exact corrected natural-movie SSI change relative to matched stabilization for the same populations (95% crossed image–trajectory bootstrap intervals); filled circles mark the highest sampled SSI. The higher-SF overlap remains {high_3x['prediction_normalized']:.3f} at 3× while SSI falls to {high_3x['ssi_percent_vs_stabilized']:.2f}% (95% CI {high_3x['ssi_percent_ci95_low']:.2f} to {high_3x['ssi_percent_ci95_high']:.2f}), and the lower-SF overlap maximum misses the observed 2× maximum. Because overlap is nonnegative, it cannot predict a below-stabilization reversal by construction. Thus tuning overlap does not predict the useful movement range, SSI sign, SSI magnitude, or the higher-SF reversal; it identifies only a coarse retinal spectral-matching component.

**F, Motion increases SSI by increasing normalized spatial contrast.** Mean-normalized final population rate maps, $g(x,y)=r(x,y)/\\overline{{r}}_{{xy}}$, for the predeclared representative image (index {panel_f_metadata['representative_image_index']}) and corrected trajectory nearest the median selected path length (index {panel_f_metadata['representative_trace_index']}). Maps show saved output {panel_f_metadata['displayed_scored_output_index_zero_based']} and share one color scale across all conditions. Values below maps are exact expected-spike-weighted population SSI over all 40 scored outputs for the same movie. Lower-SF spatial contrast increases at 2×; higher-SF contrast increases at 1× and is redistributed or lost at 3×. Inset, across all held-out normal-moving image–trajectory conditions and nonzero amplitudes, change in normalized spatial variance closely tracks change in exact SSI (Pearson r=0.904; Spearman $\\rho$=0.936). Mean-rate effects are not shown here and are reserved for Extended Data.

Panels use only validated saved outputs from the corrected-history analysis. No unique recurrent circuit localization is implied.
"""
    (OUT / "CAPTION_DRAFT.md").write_text(caption)


def verify(
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f_maps: pd.DataFrame,
    panel_f_summary: pd.DataFrame,
    panel_f_contrast: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    panel_e_claim = summarize_panel_e_claim(panel_e)
    high_3x = panel_e_claim["higher_sf_at_3x"]
    checks = {
        "model_executed": False,
        "panel_d_scales": sorted(panel_d.movement_scale.unique().tolist()),
        "panel_e_scales": sorted(panel_e.scale.unique().tolist()),
        "panel_e_group_counts": panel_e.groupby("sf_group").n_units.first().astype(int).to_dict(),
        "panel_e_predicted_optimum": panel_e.loc[panel_e.is_predicted_optimum].set_index("sf_group").scale.to_dict(),
        "panel_e_observed_optimum": panel_e.loc[panel_e.is_observed_ssi_optimum].set_index("sf_group").scale.to_dict(),
        "panel_e_interpretation": "coarse_peak_ordering_not_useful_range",
        "panel_e_higher_sf_at_3x": high_3x,
        "panel_e_high_sf_prediction_observation_mismatch": bool(
            high_3x["prediction_normalized"] > 0.9
            and high_3x["ssi_percent_vs_stabilized"] < 0.0
        ),
        "panel_f_conditions": panel_f_summary.groupby("sf_group").movement_scale.apply(list).to_dict(),
        "panel_f_map_shape": [51, 51],
        "panel_f_common_color_scale": metadata["panel_f_common_color_scale"],
        "panel_f_pearson_r": metadata["panel_f"]["pearson_r"],
        "panel_f_spearman_rho": metadata["panel_f"]["spearman_rho"],
        "all_plotting_values_finite": bool(
            np.isfinite(panel_d.select_dtypes("number")).all().all()
            and np.isfinite(panel_e.select_dtypes("number")).all().all()
            and np.isfinite(panel_f_maps.select_dtypes("number")).all().all()
            and np.isfinite(panel_f_summary.select_dtypes("number")).all().all()
            and np.isfinite(panel_f_contrast.select_dtypes("number")).all().all()
        ),
    }
    production_files = [
        OUT / f"figure4_panel_{panel}_{stem}.{suffix}"
        for panel, stem in (
            ("d", "retinal_sftf"),
            ("e", "tuning_prediction"),
            ("f", "spatial_contrast"),
        )
        for suffix in ("svg", "pdf", "png")
    ]
    checks["n_production_panels"] = 3
    checks["all_nine_production_exports_present"] = all(path.is_file() for path in production_files)
    png_dpi = {}
    for path in production_files:
        if path.suffix == ".png":
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (float("nan"), float("nan")))
                png_dpi[path.name] = [float(dpi[0]), float(dpi[1])]
    checks["png_export_dpi"] = png_dpi
    checks["all_png_exports_are_600_dpi"] = all(
        abs(axis_dpi - 600.0) < 0.01 for pair in png_dpi.values() for axis_dpi in pair
    )
    assert checks["panel_d_scales"] == [1.0, 3.0]
    assert checks["panel_e_scales"] == SCALES.tolist()
    assert checks["panel_e_group_counts"] == GROUP_COUNT
    assert checks["panel_e_predicted_optimum"] == {"higher SF": 1.0, "lower SF": 3.0}
    assert checks["panel_e_observed_optimum"] == {"higher SF": 1.0, "lower SF": 2.0}
    assert np.isclose(high_3x["prediction_normalized"], 0.9548224765632176)
    assert np.isclose(high_3x["ssi_percent_vs_stabilized"], -8.979430317021252)
    assert checks["panel_e_high_sf_prediction_observation_mismatch"]
    assert checks["panel_f_conditions"] == {"higher SF": [0.0, 1.0, 3.0], "lower SF": [0.0, 2.0]}
    assert checks["all_plotting_values_finite"]
    assert checks["all_nine_production_exports_present"]
    assert checks["all_png_exports_are_600_dpi"]
    checks["status"] = "pass"
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")


def main() -> int:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    sources = [
        FEM_SOURCE,
        PREDICTION_SOURCE,
        SSI_CURVE_SOURCE,
        METRICS_SOURCE,
        MAP_SOURCE,
        CONTRAST_SOURCE,
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    panel_d, panel_d_metadata = load_panel_d()
    panel_e = load_panel_e()
    panel_e_metadata = summarize_panel_e_claim(panel_e)
    panel_f_maps, panel_f_summary, panel_f_contrast, panel_f_metadata = load_panel_f()
    panel_d.to_csv(DATA / "panel_d_exact_corrected_fem_phase_power.csv.gz", index=False)
    panel_e.to_csv(DATA / "panel_e_prediction_and_exact_ssi.csv", index=False)
    panel_f_maps.to_csv(DATA / "panel_f_representative_normalized_rate_maps.csv.gz", index=False)
    panel_f_summary.to_csv(DATA / "panel_f_representative_exact_ssi.csv", index=False)
    panel_f_contrast.to_csv(DATA / "panel_f_spatial_variance_vs_exact_ssi.csv.gz", index=False)

    # Record the one common Panel F color scale in the exact plotting metadata.
    all_gain = panel_f_maps.g_mean_normalized_population_rate.to_numpy(float)
    common_scale = [float(value) for value in np.quantile(all_gain, [0.002, 0.998])]
    metadata = {
        "analysis": "final_figure4_mechanism_panels_d_f_saved_outputs_only",
        "model_executed": False,
        "historical_population_definition": {
            "lower SF": {"rule": "sf_split_metric < 0.5", "n_units": 71},
            "higher SF": {"rule": "sf_split_metric >= 0.5", "n_units": 29},
        },
        "panel_d": panel_d_metadata,
        "panel_e": panel_e_metadata,
        "panel_f": panel_f_metadata,
        "panel_f_common_color_scale": {
            "vmin": common_scale[0],
            "vmax": common_scale[1],
            "rule": "pooled 0.2nd and 99.8th percentiles across all five displayed mean-normalized maps",
        },
        "source_files": {str(path.relative_to(ROOT)): {"sha256": sha256(path)} for path in sources},
        "outputs": {
            "production_panels": [
                "figure4_panel_d_retinal_sftf.[svg|pdf|png]",
                "figure4_panel_e_tuning_prediction.[svg|pdf|png]",
                "figure4_panel_f_spatial_contrast.[svg|pdf|png]",
            ],
            "png_dpi": 600,
        },
    }
    (OUT / "plotting_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    # Individually editable production panels.
    fig = plt.figure(figsize=(5.15, 4.55), facecolor="white")
    draw_panel_d(fig, panel_d)
    export(fig, "figure4_panel_d_retinal_sftf")

    fig = plt.figure(figsize=(4.25, 4.55), facecolor="white")
    draw_panel_e(fig, panel_e)
    export(fig, "figure4_panel_e_tuning_prediction")

    fig = plt.figure(figsize=(7.15, 4.55), facecolor="white")
    draw_panel_f(fig, panel_f_maps, panel_f_summary, panel_f_contrast, panel_f_metadata)
    export(fig, "figure4_panel_f_spatial_contrast")

    # An earlier diagnostic strip is deliberately removed: the deliverable is
    # exactly three editable production panels, to be placed by the paper's
    # master Figure 4 layout rather than reflowed by a second plotting system.
    for suffix in ("svg", "pdf", "png"):
        (OUT / f"figure4_mechanism_panels_d_f.{suffix}").unlink(missing_ok=True)

    write_caption(panel_e_metadata, panel_f_metadata)
    verify(panel_d, panel_e, panel_f_maps, panel_f_summary, panel_f_contrast, metadata)
    print(json.dumps({"output": str(OUT), "status": "complete", "model_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
