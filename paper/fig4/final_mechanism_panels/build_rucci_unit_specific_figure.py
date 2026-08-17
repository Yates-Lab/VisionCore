#!/usr/bin/env python3
"""Build an interpretable unit-specific M66 retinal-motion mechanism figure.

The figure deliberately separates four claims:

1. an output unit's measured SF x TF tuning under the standard grating probe;
2. retinal AC power from unfiltered measured eye motion on the same axes;
3. the unit's actual spatial response under matched stabilization and motion;
4. the population relationship between preferred TF and the causal
   moving-versus-stabilized SSI effect.

No branch-averaged spectral proxy is used.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "rucci_unit_specific"
)
DEFAULT_VIS = Path(
    "/home/jake/.codex/visualizations/2026/08/14/"
    "019ffe32-dd4b-7ab1-ba87-15e22155ada2/"
    "rucci-unit-specific.html"
)

TUNING_DIR = ROOT / (
    "outputs/dekel240_paper/final/fig4_real_trace/frequency_tuning_m66"
)
RETINAL_POWER = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "motion_power_100_movies/native240_retinal_power_20_movies_arrays.npz"
)
MAP_CACHE = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/schematic_maps_m66/cache/"
    "schematic_rr100_final_maps.npz"
)
MAP_METRICS = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/schematic_maps_m66/"
    "schematic_rr100_final_map_unit_metrics.csv"
)
TRACE_BANK = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_trace_bank_merged"
)

EXEMPLAR_UNIT = 63
TF_DISPLAY_MAX = 60.0
SF_DISPLAY_MIN = 0.05
SF_DISPLAY_MAX = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VIS)
    parser.add_argument("--unit", type=int, default=EXEMPLAR_UNIT)
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def tuning_grid(unit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    grouped = pd.read_csv(TUNING_DIR / "frequency_tuning_grouped.csv")
    summary = pd.read_csv(TUNING_DIR / "frequency_tuning_summary.csv")
    selected = grouped.loc[
        grouped.unit_index.eq(int(unit)) & grouped.temporal_hz.gt(0.0)
    ].copy()
    # The standard probe measures all four orientations.  Taking the maximum
    # exposes the unit's joint SF x TF envelope without averaging away its
    # orientation preference.
    table = (
        selected.groupby(["spatial_cpd", "temporal_hz"], sort=True)
        .response_amp_rms.max()
        .unstack()
    )
    sf = table.index.to_numpy(dtype=float)
    tf = table.columns.to_numpy(dtype=float)
    response = table.to_numpy(dtype=float)
    response /= max(float(np.nanmax(response)), 1e-30)
    row = summary.loc[summary.unit_index.eq(int(unit))].iloc[0]
    meta = {
        "peak_sf_cpd": float(row.dynamic_peak_spatial_cpd_by_amp),
        "peak_tf_hz": float(row.dynamic_peak_temporal_hz_by_amp),
        "peak_orientation_deg": float(row.dynamic_peak_orientation_deg_by_amp),
        "peak_response_amp": float(row.dynamic_peak_response_amp),
    }
    return sf, tf, response, meta


def tuning_on_grid(
    sf: np.ndarray,
    tf: np.ndarray,
    response: np.ndarray,
    sf_query: np.ndarray,
    tf_query: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (np.log10(tf), np.log10(sf)),
        response.T,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    query_tf, query_sf = np.meshgrid(tf_query, sf_query, indexing="ij")
    points = np.column_stack((np.log10(query_tf.ravel()), np.log10(query_sf.ravel())))
    return interpolator(points).reshape(query_tf.shape)


def load_retinal_power() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(RETINAL_POWER, allow_pickle=False) as archive:
        sf_edges = np.asarray(archive["sf_edges"], dtype=float)
        tf_edges = np.asarray(archive["tf_edges"], dtype=float)
        power = np.asarray(archive["retinal_power_scale_1p0"], dtype=float)
    sf = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    tf = 0.5 * (tf_edges[:-1] + tf_edges[1:])
    return sf, tf, np.maximum(power, 0.0)


def load_maps(unit: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(MAP_CACHE, allow_pickle=False) as archive:
        maps = np.asarray(archive["final_maps"], dtype=float)
        condition_ids = archive["condition_id"].astype(str).tolist()
    stable = maps[condition_ids.index("endpoint_stabilized_final"), int(unit)]
    moving = maps[condition_ids.index("real_trace_final"), int(unit)]
    metrics = pd.read_csv(MAP_METRICS).set_index("unit_index").loc[int(unit)]
    meta = {
        "stable_ssi": float(metrics.stable_final_map_ssi_bits_per_spike),
        "moving_ssi": float(metrics.real_final_map_ssi_bits_per_spike),
        "delta_ssi": float(metrics.real_minus_stable_map_ssi),
        "sf_split_metric": float(metrics.sf_split_metric),
    }
    return stable, moving, meta


def population_effects() -> pd.DataFrame:
    units = pd.read_csv(TRACE_BANK / "unit_feature_table.csv")
    n_units = len(units)
    n_images = len(pd.read_csv(TRACE_BANK / "image_feature_table.csv"))
    n_traces = len(pd.read_csv(TRACE_BANK / "trace_feature_table.csv"))
    moving_ssi = np.load(TRACE_BANK / "ssi_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_expected = np.load(
        TRACE_BANK / "expected_spikes_matrix.npy", mmap_mode="r"
    ).reshape(n_images, n_traces, n_units)
    stable_ssi = np.load(TRACE_BANK / "stabilized_ssi_by_image.npy", mmap_mode="r")
    stable_expected = np.load(
        TRACE_BANK / "stabilized_expected_spikes_by_image.npy", mmap_mode="r"
    )
    moving_denominator = np.sum(moving_expected, axis=(0, 1), dtype=np.float64)
    moving = np.sum(
        moving_ssi * moving_expected, axis=(0, 1), dtype=np.float64
    ) / np.maximum(moving_denominator, 1e-30)
    stable_denominator = np.sum(stable_expected, axis=0, dtype=np.float64)
    stable = np.sum(
        stable_ssi * stable_expected, axis=0, dtype=np.float64
    ) / np.maximum(stable_denominator, 1e-30)
    effect = 100.0 * (moving - stable) / np.maximum(stable, 1e-30)

    tuning = pd.read_csv(TUNING_DIR / "frequency_tuning_summary.csv")
    result = tuning[[
        "unit_index",
        "dynamic_peak_spatial_cpd_by_amp",
        "dynamic_peak_temporal_hz_by_amp",
    ]].copy()
    result["stable_ssi"] = stable
    result["moving_ssi"] = moving
    result["ssi_change_percent"] = effect
    result["expected_spikes"] = moving_denominator
    result["active"] = (
        np.isfinite(effect)
        & (stable > 1e-5)
        & (moving_denominator > 1e-8)
    )
    return result


def bootstrap_group_medians(values: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    rows = []
    for tf, frame in values.groupby("dynamic_peak_temporal_hz_by_amp", sort=True):
        y = frame.ssi_change_percent.to_numpy(dtype=float)
        draws = np.empty(4000, dtype=float)
        for index in range(len(draws)):
            draws[index] = float(np.median(y[rng.integers(0, len(y), len(y))]))
        rows.append(
            {
                "preferred_tf_hz": float(tf),
                "n_units": int(len(y)),
                "median_ssi_change_percent": float(np.median(y)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def render(
    out_dir: Path,
    *,
    unit: int,
    tuning_sf: np.ndarray,
    tuning_tf: np.ndarray,
    tuning_response: np.ndarray,
    tuning_meta: dict[str, float],
    retinal_sf: np.ndarray,
    retinal_tf: np.ndarray,
    retinal_power: np.ndarray,
    stable_map: np.ndarray,
    moving_map: np.ndarray,
    map_meta: dict[str, float],
    population: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> tuple[Path, dict[str, float]]:
    configure()
    fig = plt.figure(figsize=(10.4, 6.7), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.08),
        height_ratios=(1.0, 0.96),
        left=0.075,
        right=0.975,
        bottom=0.105,
        top=0.885,
        wspace=0.30,
        hspace=0.43,
    )

    # A: full output-unit tuning, not a stem or channel average.
    ax_tuning = fig.add_subplot(grid[0, 0])
    sf_dense = np.geomspace(max(SF_DISPLAY_MIN, tuning_sf.min()), SF_DISPLAY_MAX, 150)
    tf_dense = np.geomspace(max(0.2, tuning_tf.min()), min(TF_DISPLAY_MAX, tuning_tf.max()), 150)
    tuning_dense = tuning_on_grid(
        tuning_sf, tuning_tf, tuning_response, sf_dense, tf_dense
    )
    contour = ax_tuning.contourf(
        sf_dense,
        tf_dense,
        tuning_dense,
        levels=np.linspace(0.0, 1.0, 11),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    query_tf, query_sf = np.meshgrid(tuning_tf, tuning_sf, indexing="xy")
    ax_tuning.scatter(
        query_sf.ravel(),
        query_tf.ravel(),
        s=6,
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.8,
        zorder=3,
    )
    ax_tuning.scatter(
        [tuning_meta["peak_sf_cpd"]],
        [tuning_meta["peak_tf_hz"]],
        marker="*",
        s=92,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.75,
        zorder=4,
    )
    ax_tuning.set_xscale("log")
    ax_tuning.set_yscale("log")
    ax_tuning.set_xlim(SF_DISPLAY_MIN, SF_DISPLAY_MAX)
    ax_tuning.set_ylim(0.2, TF_DISPLAY_MAX)
    ax_tuning.set_xticks([0.05, 0.2, 0.8, 3.2, 12.0], ["0.05", "0.2", "0.8", "3.2", "12"])
    ax_tuning.set_yticks([0.2, 0.8, 3.2, 12.8, 47.2], ["0.2", "0.8", "3.2", "12.8", "47"])
    ax_tuning.set_xlabel("spatial frequency (cycles/deg)")
    ax_tuning.set_ylabel("temporal frequency (Hz)")
    ax_tuning.set_title(
        f"A  Output unit u{unit:03d} prefers "
        f"{tuning_meta['peak_sf_cpd']:g} c/deg at {tuning_meta['peak_tf_hz']:g} Hz",
        loc="left",
        pad=6,
    )
    colorbar = fig.colorbar(contour, ax=ax_tuning, fraction=0.046, pad=0.03)
    colorbar.set_label("normalized grating-response amplitude")
    colorbar.set_ticks([0.0, 0.5, 1.0])

    # B: measured retinal power with this exact unit's half-max tuning contour.
    ax_power = fig.add_subplot(grid[0, 1])
    tf_mask = retinal_tf <= TF_DISPLAY_MAX
    sf_mask = (retinal_sf >= SF_DISPLAY_MIN) & (retinal_sf <= SF_DISPLAY_MAX)
    power = retinal_power[np.ix_(tf_mask, sf_mask)]
    sf_display = retinal_sf[sf_mask]
    tf_display = retinal_tf[tf_mask]
    positive = power[power > 0]
    power_floor = max(float(np.quantile(positive, 0.02)), float(np.max(positive)) * 1e-5)
    power_ceiling = float(np.quantile(positive, 0.995))
    power_contour = ax_power.contourf(
        sf_display,
        tf_display,
        np.maximum(power, power_floor),
        levels=np.geomspace(power_floor, power_ceiling, 12),
        cmap="magma",
        norm=LogNorm(vmin=power_floor, vmax=power_ceiling),
        extend="both",
    )
    tuning_on_retina = tuning_on_grid(
        tuning_sf, tuning_tf, tuning_response, sf_display, tf_display
    )
    # Draw a dark underlay and light overlay so the contour stays visible at
    # every retinal-power level.
    ax_power.contour(
        sf_display,
        tf_display,
        tuning_on_retina,
        levels=[0.5],
        colors="#222222",
        linewidths=2.6,
    )
    ax_power.contour(
        sf_display,
        tf_display,
        tuning_on_retina,
        levels=[0.5],
        colors="white",
        linewidths=1.35,
    )
    ax_power.scatter(
        [tuning_meta["peak_sf_cpd"]],
        [tuning_meta["peak_tf_hz"]],
        marker="*",
        s=92,
        color="#56B4E9",
        edgecolor="#222222",
        linewidth=0.7,
        zorder=4,
    )
    plotted_power = float(np.sum(power))
    halfmax_fraction = float(
        np.sum(power[tuning_on_retina >= 0.5]) / max(plotted_power, 1e-30)
    )
    ax_power.text(
        0.03,
        0.96,
        f"{100 * halfmax_fraction:.0f}% of plotted retinal AC power\n"
        "falls inside the unit's half-max contour",
        transform=ax_power.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7.3,
        bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 2.5},
    )
    ax_power.set_xlim(SF_DISPLAY_MIN, SF_DISPLAY_MAX)
    ax_power.set_ylim(0.0, TF_DISPLAY_MAX)
    ax_power.set_xscale("log")
    ax_power.set_xticks([0.05, 0.2, 0.8, 3.2, 12.0], ["0.05", "0.2", "0.8", "3.2", "12"])
    ax_power.set_yticks([0, 12, 24, 36, 48, 60])
    ax_power.set_xlabel("spatial frequency (cycles/deg)")
    ax_power.set_ylabel("temporal frequency (Hz)")
    ax_power.set_title(
        "B  Measured motion supplies power inside this unit's passband",
        loc="left",
        pad=6,
    )
    power_bar = fig.colorbar(power_contour, ax=ax_power, fraction=0.046, pad=0.03)
    power_bar.set_label("retinal AC power (log scale)")
    power_bar.set_ticks([])

    # C: the actual full-model response maps for the same unit.
    parent = fig.add_subplot(grid[1, 0])
    parent.set_axis_off()
    parent.set_title(
        f"C  Measured motion sharpens u{unit:03d}'s spatial response",
        loc="left",
        pad=6,
    )
    map_grid = grid[1, 0].subgridspec(1, 3, wspace=0.08)
    stable_gain = stable_map / max(float(np.mean(stable_map)), 1e-30)
    moving_gain = moving_map / max(float(np.mean(moving_map)), 1e-30)
    delta_gain = moving_gain - stable_gain
    gain_values = np.concatenate((stable_gain.ravel(), moving_gain.ravel()))
    gain_max = float(np.quantile(gain_values, 0.995))
    delta_limit = float(np.quantile(np.abs(delta_gain), 0.995))
    axes_maps = [fig.add_subplot(map_grid[0, index]) for index in range(3)]
    for axis, image, title, ssi_value in (
        (axes_maps[0], stable_gain, "stabilized", map_meta["stable_ssi"]),
        (axes_maps[1], moving_gain, "measured motion", map_meta["moving_ssi"]),
    ):
        axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=gain_max,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(
            0.04,
            0.96,
            f"{title}\nSSI {ssi_value:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=6.8,
            bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 2.0},
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes_maps[2].imshow(
        delta_gain,
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit),
    )
    axes_maps[2].set_xticks([])
    axes_maps[2].set_yticks([])
    axes_maps[2].text(
        0.04,
        0.96,
        "motion − stable\nnormalized rate",
        transform=axes_maps[2].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6.8,
        bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 2.0},
    )
    for spine in axes_maps[2].spines.values():
        spine.set_visible(False)
    axes_maps[2].text(
        0.98,
        0.03,
        f"+{100 * map_meta['delta_ssi'] / map_meta['stable_ssi']:.0f}%",
        transform=axes_maps[2].transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=8.2,
        fontweight="semibold",
        bbox={"facecolor": "#111111", "alpha": 0.70, "edgecolor": "none", "pad": 2.5},
    )
    parent.text(
        0.5,
        -0.065,
        "mean-normalized predicted rate; matched natural image and endpoint",
        transform=parent.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="#555555",
    )

    # D: the causal effect across the full bank, organized by independently
    # measured temporal preference.
    ax_population = fig.add_subplot(grid[1, 1])
    active = population.loc[population.active].copy()
    rho, p_value = spearmanr(
        np.log10(active.dynamic_peak_temporal_hz_by_amp.to_numpy(dtype=float)),
        active.ssi_change_percent.to_numpy(dtype=float),
    )
    rng = np.random.default_rng(20260816)
    x = active.dynamic_peak_temporal_hz_by_amp.to_numpy(dtype=float)
    jitter = np.exp(rng.uniform(-0.075, 0.075, len(active)))
    ax_population.scatter(
        x * jitter,
        active.ssi_change_percent,
        s=17,
        color="#777777",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    summary_x = group_summary.preferred_tf_hz.to_numpy(dtype=float)
    summary_y = group_summary.median_ssi_change_percent.to_numpy(dtype=float)
    summary_low = summary_y - group_summary.ci95_low.to_numpy(dtype=float)
    summary_high = group_summary.ci95_high.to_numpy(dtype=float) - summary_y
    ax_population.plot(summary_x, summary_y, color="#0072B2", linewidth=1.4, zorder=2)
    ax_population.errorbar(
        summary_x,
        summary_y,
        yerr=np.vstack((summary_low, summary_high)),
        fmt="o",
        color="#0072B2",
        markerfacecolor="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=5.8,
        linewidth=1.2,
        capsize=2.4,
        zorder=3,
    )
    exemplar = active.loc[active.unit_index.eq(int(unit))].iloc[0]
    ax_population.scatter(
        [float(exemplar.dynamic_peak_temporal_hz_by_amp)],
        [float(exemplar.ssi_change_percent)],
        marker="D",
        s=54,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax_population.annotate(
        f"u{unit:03d}",
        xy=(
            float(exemplar.dynamic_peak_temporal_hz_by_amp),
            float(exemplar.ssi_change_percent),
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=7.2,
        color="#333333",
    )
    ax_population.axhline(0.0, color="#888888", linewidth=0.75, zorder=0)
    ax_population.set_xscale("log")
    ax_population.set_xticks(summary_x, [f"{value:g}" for value in summary_x])
    ax_population.set_xlim(0.14, 19.0)
    ax_population.set_xlabel("model unit's preferred temporal frequency (Hz)")
    ax_population.set_ylabel("SSI change: measured motion vs stabilized (%)")
    ax_population.set_title(
        "D  Faster-tuned units benefit more from measured retinal motion",
        loc="left",
        pad=6,
    )
    ax_population.text(
        0.03,
        0.97,
        f"Spearman ρ={rho:.2f}, p={p_value:.1e}\n"
        f"{len(active)} active units; 100 images × 1,000 traces",
        transform=ax_population.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )
    ax_population.text(
        0.98,
        0.03,
        "gray: units   blue: median ± 95% bootstrap",
        transform=ax_population.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color="#555555",
    )

    fig.suptitle(
        "Measured retinal motion engages a temporally tuned M66 output unit",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.975,
        0.025,
        "M66: native 240-Hz retinal input, 120-Hz scored output · tuning: full-model grating probe · "
        "retinal power: 20 movies, unfiltered eye traces",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#555555",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "rucci_unit_specific.png"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"rucci_unit_specific.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return output, {
        "halfmax_retinal_power_fraction_within_60_hz": halfmax_fraction,
        "preferred_tf_ssi_spearman_rho": float(rho),
        "preferred_tf_ssi_spearman_p": float(p_value),
        "n_active_units": int(len(active)),
        "exemplar_population_ssi_change_percent": float(exemplar.ssi_change_percent),
    }


def write_visualization(path: Path, png: Path, unit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    path.write_text(
        f'''<div id="rucci-unit-specific" style="width:100%;color:var(--foreground)">
  <figure style="margin:0;display:grid;gap:6px">
    <img src="data:image/png;base64,{encoded}" alt="Four-panel M66 mechanism figure. Output unit u{unit:03d} has joint spatial-temporal grating tuning, measured retinal motion overlaps its half-maximum passband, its spatial response becomes more selective under motion, and faster-tuned units show larger motion-induced SSI increases across the population." style="width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:var(--card)">
  </figure>
</div>
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    unit = int(args.unit)
    tuning_sf, tuning_tf, tuning_response, tuning_meta = tuning_grid(unit)
    retinal_sf, retinal_tf, retinal_power = load_retinal_power()
    stable_map, moving_map, map_meta = load_maps(unit)
    population = population_effects()
    active = population.loc[population.active].copy()
    group_summary = bootstrap_group_medians(active)
    png, figure_metrics = render(
        args.out_dir,
        unit=unit,
        tuning_sf=tuning_sf,
        tuning_tf=tuning_tf,
        tuning_response=tuning_response,
        tuning_meta=tuning_meta,
        retinal_sf=retinal_sf,
        retinal_tf=retinal_tf,
        retinal_power=retinal_power,
        stable_map=stable_map,
        moving_map=moving_map,
        map_meta=map_meta,
        population=population,
        group_summary=group_summary,
    )
    population.to_csv(args.out_dir / "per_unit_motion_effects.csv", index=False)
    group_summary.to_csv(args.out_dir / "preferred_tf_group_summary.csv", index=False)
    report = {
        "analysis": "unit-specific M66 retinal-motion mechanism figure",
        "exemplar_unit": unit,
        "tuning": {
            **tuning_meta,
            "source": str(TUNING_DIR / "frequency_tuning_grouped.csv"),
            "aggregation": "maximum response amplitude across the four probe orientations",
        },
        "retinal_power": {
            "source": str(RETINAL_POWER),
            "n_movies": 20,
            "eye_trace_filtering": "none",
            "display_temporal_range_hz": [0.0, TF_DISPLAY_MAX],
        },
        "response_maps": {
            **map_meta,
            "source": str(MAP_CACHE),
        },
        "population_effect": {
            "source": str(TRACE_BANK),
            "estimand": "per-unit expected-spike-weighted SSI over 100 images x 1,000 traces",
            **figure_metrics,
        },
        "claim_boundary": (
            "Measured motion versus stabilization is causal at the retinal input; "
            "the preferred-TF population relationship and spectral overlap are associations, "
            "not a frequency-band ablation."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_visualization(args.visualization, png, unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
