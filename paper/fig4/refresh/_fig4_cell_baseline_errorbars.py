#!/usr/bin/env python3
"""Standalone Panel C for SF>=0.5 with bootstrap CIs and p-value brackets."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
from VisionCore.paths import VISIONCORE_ROOT as ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
from _fig4_style import ORANGE
from _fig4_component_2d_surface import (
    MATRIX_DIR,
    OUT_DIR,
    _assign_bins,
    _compute_component_metrics,
    _json_ready,
    _quantile_edges,
)
from _fig4_component_path_baseline import (
    _cell_matched_baseline,
)
from _fig4_geometry_story import (
    _add_bracket,
    _format_axis,
    _format_p_label,
    _x_broken_log,
)
from _fig4_trace_schematics import (
    accumulate_population_movie_rows,
    axis_delta_deg,
    baseline_rows_by_image,
    finite_ratio,
    load_dataset,
    ratio_delta_stats,
)


OUT_STEM = "backimage_real_trace_panel_c_aligned_sf_ge_0p5_cell_baseline_errorbars"
SF_METRIC_COL = "sf_split_metric"
SF_MIN_CPD = 0.50
CONTOUR_COHERENCE_MIN = 0.20
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
N_COMPONENT_BINS = 8
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
LOWER_MIN_POS = 45.0
LOWER_MAX_POS = 180.0
LOWER_TICKS = [0, 50, 65, 90, 120, 160]
EPS = 1e-12

COMPONENT_STYLES = {
    "across_path_arcmin": ("across contour", "-", "o"),
    "along_path_arcmin": ("along contour", (0, (4.2, 2.0)), "s"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _quantile_edges_from_probs(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("Cannot bin an empty metric.")
    probs = np.asarray(quantiles, dtype=np.float64)
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError("component bin quantiles must contain at least two values.")
    if not (np.isclose(probs[0], 0.0) and np.isclose(probs[-1], 1.0)):
        raise ValueError("component bin quantiles must start at 0 and end at 1.")
    if np.any(np.diff(probs) <= 0):
        raise ValueError("component bin quantiles must be strictly increasing.")
    edges = np.quantile(finite, probs)
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _selected_unit_images(
    unit: pd.DataFrame,
    image: pd.DataFrame,
    *,
    match_max_deg: float = MATCH_MAX_DEG,
) -> dict[int, np.ndarray]:
    sf = pd.to_numeric(unit[SF_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    pref = pd.to_numeric(unit["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(unit["prior_orientation_selectivity_index"], errors="coerce").to_numpy(dtype=float)
    unit_index = unit["unit_index"].astype(int).to_numpy()

    image_indices = image["image_index"].astype(int).to_numpy()
    image_axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    contour_mask = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= CONTOUR_COHERENCE_MIN)

    selected: dict[int, np.ndarray] = {}
    for idx, unit_id in enumerate(unit_index):
        if not (
            math.isfinite(float(sf[idx]))
            and float(sf[idx]) >= SF_MIN_CPD
            and math.isfinite(float(pref[idx]))
            and math.isfinite(float(osi[idx]))
            and float(osi[idx]) >= MIN_OSI
        ):
            continue
        delta = axis_delta_deg(image_axis, float(pref[idx]))
        keep = contour_mask & np.isfinite(delta) & (delta <= match_max_deg)
        images = image_indices[keep]
        if images.size:
            selected[int(unit_id)] = images.astype(int)
    return selected


def _compute_panel(
    data: dict[str, Any],
    *,
    match_max_deg: float = MATCH_MAX_DEG,
    component_bin_quantiles: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = _compute_component_metrics(data)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    unit_to_images = _selected_unit_images(data["unit"], data["image"], match_max_deg=match_max_deg)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows: list[dict[str, Any]] = []
    for metric_col, (metric_label, _linestyle, _marker) in COMPONENT_STYLES.items():
        if component_bin_quantiles is None:
            edges = _quantile_edges(metrics.loc[drift_mask, metric_col].to_numpy(dtype=float), N_COMPONENT_BINS)
        else:
            edges = _quantile_edges_from_probs(
                metrics.loc[drift_mask, metric_col].to_numpy(dtype=float),
                component_bin_quantiles,
            )
        bins = _assign_bins(metrics[metric_col].to_numpy(dtype=float), edges)
        n_component_bins = len(edges) - 1

        rows.append(
            {
                "component_metric": metric_col,
                "component_metric_label": metric_label,
                "context": "stabilized",
                "component_bin": "stabilized_zero_motion",
                "component_bin_order": 0,
                "component_median_arcmin": 0.0,
                "n_movie_rows_global": 0,
                "n_movie_samples": 0,
                "n_images_contributing": 0,
                "moving_population_ssi_bits_per_spike": float("nan"),
                "cell_baseline_population_ssi_bits_per_spike": float("nan"),
                "ssi_percent_vs_cell_baseline": 0.0,
                "population_delta_ci95_low_image_boot": float("nan"),
                "population_delta_ci95_high_image_boot": float("nan"),
                "population_delta_percent_ci95_low_image_boot": float("nan"),
                "population_delta_percent_ci95_high_image_boot": float("nan"),
                "population_delta_p_image_bootstrap_sign": float("nan"),
            }
        )

        for bin_index in range(n_component_bins):
            row_mask = drift_mask & (bins == bin_index)
            moving_pop = accumulate_population_movie_rows(
                ssi=data["ssi"],
                expected=data["expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            cell_pop = _cell_matched_baseline(
                stabilized_ssi=data["stabilized_ssi"],
                stabilized_expected=data["stabilized_expected"],
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            moving_ssi = finite_ratio(float(moving_pop["information_numerator_bits"]), float(moving_pop["expected_spikes"]))
            cell_ssi = finite_ratio(float(cell_pop["information_numerator_bits"]), float(cell_pop["expected_spikes"]))
            delta_stats = ratio_delta_stats(
                moving_pop["per_image_num"],
                moving_pop["per_image_den"],
                cell_pop["per_image_num"],
                cell_pop["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            delta_low = float(delta_stats["population_delta_ci95_low_image_boot"])
            delta_high = float(delta_stats["population_delta_ci95_high_image_boot"])
            global_rows = metrics[row_mask]
            rows.append(
                {
                    "component_metric": metric_col,
                    "component_metric_label": metric_label,
                    "context": "drift_only",
                    "component_bin": f"drift_only_q{bin_index + 1:02d}",
                    "component_bin_order": int(bin_index + 1),
                    "component_min_arcmin": float(edges[bin_index]),
                    "component_max_arcmin": float(edges[bin_index + 1]),
                    "component_quantile_min": float(component_bin_quantiles[bin_index])
                    if component_bin_quantiles is not None
                    else float(bin_index / N_COMPONENT_BINS),
                    "component_quantile_max": float(component_bin_quantiles[bin_index + 1])
                    if component_bin_quantiles is not None
                    else float((bin_index + 1) / N_COMPONENT_BINS),
                    "component_median_arcmin": float(np.nanmedian(global_rows[metric_col])),
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "n_movie_samples": int(moving_pop["n_movie_samples"]),
                    "n_images_contributing": int(moving_pop["n_images_contributing"]),
                    "moving_population_ssi_bits_per_spike": moving_ssi,
                    "cell_baseline_population_ssi_bits_per_spike": cell_ssi,
                    "ssi_percent_vs_cell_baseline": _pct_delta(moving_ssi, cell_ssi),
                    "population_delta_ci95_low_image_boot": delta_low,
                    "population_delta_ci95_high_image_boot": delta_high,
                    "population_delta_percent_ci95_low_image_boot": 100.0 * delta_low / cell_ssi
                    if math.isfinite(delta_low) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                    else float("nan"),
                    "population_delta_percent_ci95_high_image_boot": 100.0 * delta_high / cell_ssi
                    if math.isfinite(delta_high) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                    else float("nan"),
                    "population_delta_p_image_bootstrap_sign": float(
                        delta_stats["population_delta_p_image_bootstrap_sign"]
                    ),
                }
            )

    metadata = {
        "n_selected_units": int(len(unit_to_images)),
        "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "n_contour_images": int(
            (
                pd.to_numeric(data["image"]["image_orientation_coherence"], errors="coerce").ge(CONTOUR_COHERENCE_MIN)
                & pd.to_numeric(data["image"]["image_edge_axis_deg"], errors="coerce").notna()
            ).sum()
        ),
        "match_max_deg": float(match_max_deg),
    }
    return pd.DataFrame(rows), metadata


def _plot_panel(
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    ax: Any | None = None,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.15, 4.35))
        do_tight_layout = True
    else:
        fig = ax.figure
        do_tight_layout = False
    finite_cols = [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]
    vals = [0.0]
    for col in finite_cols:
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    if ylim is None:
        ylim = (lo - 0.13 * span, hi + 0.32 * span)

    first_rows: dict[str, pd.Series] = {}
    for metric_col, (label, linestyle, marker) in COMPONENT_STYLES.items():
        rows = summary[summary["component_metric"].eq(metric_col)].copy()
        zero = rows[rows["context"].eq("stabilized")]
        drift = rows[rows["context"].eq("drift_only")].sort_values("component_median_arcmin")
        if drift.empty:
            continue
        x = _x_broken_log(drift["component_median_arcmin"], min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)
        y = drift["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([y - ci_low, ci_high - y])
        ax.plot(x, y, color=ORANGE, linestyle=linestyle, linewidth=2.1, label=label, zorder=3)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=ORANGE,
            linestyle="none",
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=1.6,
            elinewidth=1.2,
            capsize=2.0,
            zorder=4,
        )
        if not zero.empty:
            ax.scatter(
                [0.0],
                [0.0],
                marker=marker,
                s=32,
                facecolors="white",
                edgecolors=ORANGE,
                linewidths=1.35,
                zorder=5,
            )
        first_rows[metric_col] = drift.iloc[0]

    _format_axis(ax, ticks=LOWER_TICKS, min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    ax.set_ylim(*ylim)
    if title is None:
        title = (
            f"SF >= {SF_MIN_CPD:.2f}; unit-contour match <= {metadata.get('match_max_deg', MATCH_MAX_DEG):g} deg\n"
            f"coh >= {CONTOUR_COHERENCE_MIN:.2f}; "
            f"{metadata['n_selected_units']} units, {metadata['n_selected_unit_image_pairs']} pairs"
        )
    ax.set_title(title, fontsize=12.0, pad=8)
    ax.set_ylabel("SSI residual\n(% vs cell-matched stabilized)")
    ax.set_xlabel("component path length (arcmin; log scale after break)")
    ax.legend(frameon=False, fontsize=8.4, loc="lower left")

    across_first = first_rows.get("across_path_arcmin")
    along_first = first_rows.get("along_path_arcmin")
    if across_first is not None and along_first is not None:
        x1_raw = max(
            float(across_first["component_median_arcmin"]),
            float(along_first["component_median_arcmin"]),
        )
        x1 = float(_x_broken_log([x1_raw], min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)[0])
        y_lo, y_top = ax.get_ylim()
        y_span = max(y_top - y_lo, 1.0)
        label = (
            "across "
            + _format_p_label(float(across_first["population_delta_p_image_bootstrap_sign"]))
            + "\nalong "
            + _format_p_label(float(along_first["population_delta_p_image_bootstrap_sign"]))
        )
        _add_bracket(
            ax,
            x0=0.0,
            x1=x1,
            y=y_top - 0.11 * y_span,
            text=label,
            color=ORANGE,
            linestyle="-",
            text_x=x1 + 0.10,
            text_ha="left",
        )

    ax.spines[["top", "right"]].set_visible(False)
    if do_tight_layout:
        fig.tight_layout()
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    summary, metadata = _compute_panel(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"

    summary.to_csv(csv_path, index=False)
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_panel_c_sf05_cell_baseline_errorbars",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "png": png_path,
                "pdf": pdf_path,
                "values_csv": csv_path,
                "summary_json": json_path,
            },
            "selection": {
                **metadata,
                "sf_metric_col": SF_METRIC_COL,
                "sf_min_cpd": SF_MIN_CPD,
                "contour_coherence_min": CONTOUR_COHERENCE_MIN,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
            },
            "binning": {
                "component_drift_bins": N_COMPONENT_BINS,
            },
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired image bootstrap of moving-vs-cell-matched-stabilized ratio delta",
            },
        },
    )
    fig = _plot_panel(summary, metadata)
    fig.savefig(png_path, dpi=230, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
