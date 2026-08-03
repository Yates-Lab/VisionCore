#!/usr/bin/env python3
"""Across-vs-along bootstrap contrasts for Panel C high component-path tails."""

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

import _fig4_cell_baseline_errorbars as panel_c
from _fig4_component_2d_surface import (
    _assign_bins,
    _compute_component_metrics,
)
from _fig4_component_path_baseline import (
    _cell_matched_baseline,
)
from _fig4_trace_schematics import (
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    finite_ratio,
)


MATCH_MAX_DEG = 15.0
BIN_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95, 1.0)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
OUT_STEM = "backimage_real_trace_panel_c_across_along_tail_contrast_sf05_match15"
EPS = 1e-12


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(panel_c._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _population_for_mask(
    data: dict[str, Any],
    *,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> dict[str, Any]:
    moving = accumulate_population_movie_rows(
        ssi=data["ssi"],
        expected=data["expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    cell = _cell_matched_baseline(
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    moving_ssi = finite_ratio(float(moving["information_numerator_bits"]), float(moving["expected_spikes"]))
    cell_ssi = finite_ratio(float(cell["information_numerator_bits"]), float(cell["expected_spikes"]))
    return {
        "moving": moving,
        "cell": cell,
        "moving_ssi": moving_ssi,
        "cell_ssi": cell_ssi,
        "ssi_percent_vs_cell_baseline": panel_c._pct_delta(moving_ssi, cell_ssi),
    }


def _residual_percent(num: float, den: float, base_num: float, base_den: float) -> float:
    moving_ssi = finite_ratio(float(num), float(den))
    cell_ssi = finite_ratio(float(base_num), float(base_den))
    if not (math.isfinite(moving_ssi) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS):
        return float("nan")
    return 100.0 * (moving_ssi / cell_ssi - 1.0)


def _bootstrap_residual_difference(
    across: dict[str, Any],
    along: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    a_mov_num = np.asarray(across["moving"]["per_image_num"], dtype=np.float64)
    a_mov_den = np.asarray(across["moving"]["per_image_den"], dtype=np.float64)
    a_base_num = np.asarray(across["cell"]["per_image_num"], dtype=np.float64)
    a_base_den = np.asarray(across["cell"]["per_image_den"], dtype=np.float64)
    l_mov_num = np.asarray(along["moving"]["per_image_num"], dtype=np.float64)
    l_mov_den = np.asarray(along["moving"]["per_image_den"], dtype=np.float64)
    l_base_num = np.asarray(along["cell"]["per_image_num"], dtype=np.float64)
    l_base_den = np.asarray(along["cell"]["per_image_den"], dtype=np.float64)
    ok = (
        np.isfinite(a_mov_num)
        & np.isfinite(a_mov_den)
        & np.isfinite(a_base_num)
        & np.isfinite(a_base_den)
        & np.isfinite(l_mov_num)
        & np.isfinite(l_mov_den)
        & np.isfinite(l_base_num)
        & np.isfinite(l_base_den)
        & ((a_mov_den > EPS) | (a_base_den > EPS) | (l_mov_den > EPS) | (l_base_den > EPS))
    )
    indices = np.flatnonzero(ok)
    observed = float(across["ssi_percent_vs_cell_baseline"] - along["ssi_percent_vs_cell_baseline"])
    if indices.size <= 1:
        return {
            "across_minus_along_percent_point": observed,
            "contrast_ci95_low_image_boot": float("nan"),
            "contrast_ci95_high_image_boot": float("nan"),
            "contrast_p_image_bootstrap_sign": float("nan"),
            "n_bootstrap_images": int(indices.size),
        }

    sample = rng.integers(0, indices.size, size=(int(N_BOOTSTRAP), indices.size))
    sampled = indices[sample]

    def boot_resid(mov_num: np.ndarray, mov_den: np.ndarray, base_num: np.ndarray, base_den: np.ndarray) -> np.ndarray:
        moving_ratio = np.nansum(mov_num[sampled], axis=1) / np.maximum(np.nansum(mov_den[sampled], axis=1), EPS)
        cell_ratio = np.nansum(base_num[sampled], axis=1) / np.maximum(np.nansum(base_den[sampled], axis=1), EPS)
        return 100.0 * (moving_ratio / np.maximum(cell_ratio, EPS) - 1.0)

    boot_values = boot_resid(a_mov_num, a_mov_den, a_base_num, a_base_den)
    boot_values -= boot_resid(l_mov_num, l_mov_den, l_base_num, l_base_den)
    lo, hi = np.nanpercentile(boot_values, [2.5, 97.5])
    below = (float(np.count_nonzero(boot_values <= 0.0)) + 1.0) / (float(int(N_BOOTSTRAP)) + 1.0)
    above = (float(np.count_nonzero(boot_values >= 0.0)) + 1.0) / (float(int(N_BOOTSTRAP)) + 1.0)
    p = min(1.0, 2.0 * min(below, above))
    return {
        "across_minus_along_percent_point": observed,
        "contrast_ci95_low_image_boot": float(lo),
        "contrast_ci95_high_image_boot": float(hi),
        "contrast_p_image_bootstrap_sign": float(min(max(p, 0.0), 1.0)),
        "n_bootstrap_images": int(indices.size),
    }


def _contrast_row(
    data: dict[str, Any],
    *,
    label: str,
    description: str,
    across_mask: np.ndarray,
    along_mask: np.ndarray,
    across_range: tuple[float, float],
    along_range: tuple[float, float],
    metrics: pd.DataFrame,
    row_image_index: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    across = _population_for_mask(
        data,
        row_image_index=row_image_index,
        row_mask=across_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    along = _population_for_mask(
        data,
        row_image_index=row_image_index,
        row_mask=along_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    stats = _bootstrap_residual_difference(across, along, rng=rng)
    across_values = pd.to_numeric(metrics.loc[across_mask, "across_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    along_values = pd.to_numeric(metrics.loc[along_mask, "along_path_arcmin"], errors="coerce").to_numpy(dtype=float)
    return {
        "contrast": label,
        "description": description,
        "across_path_min_arcmin": float(across_range[0]),
        "across_path_max_arcmin": float(across_range[1]),
        "along_path_min_arcmin": float(along_range[0]),
        "along_path_max_arcmin": float(along_range[1]),
        "across_path_median_arcmin": float(np.nanmedian(across_values)),
        "along_path_median_arcmin": float(np.nanmedian(along_values)),
        "across_n_movie_rows_global": int(np.count_nonzero(across_mask)),
        "along_n_movie_rows_global": int(np.count_nonzero(along_mask)),
        "across_n_movie_samples": int(across["moving"]["n_movie_samples"]),
        "along_n_movie_samples": int(along["moving"]["n_movie_samples"]),
        "across_ssi_percent_vs_cell_baseline": float(across["ssi_percent_vs_cell_baseline"]),
        "along_ssi_percent_vs_cell_baseline": float(along["ssi_percent_vs_cell_baseline"]),
        **stats,
    }


def _plot_contrasts(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    x = np.arange(frame.shape[0], dtype=float)
    y = frame["across_minus_along_percent_point"].to_numpy(dtype=float)
    lo = frame["contrast_ci95_low_image_boot"].to_numpy(dtype=float)
    hi = frame["contrast_ci95_high_image_boot"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    color = "#D55E00"
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.4,
        markersize=6.0,
        capsize=3.0,
        elinewidth=1.4,
    )
    for xpos, row in zip(x, frame.itertuples(index=False), strict=True):
        ax.text(
            xpos + 0.06,
            float(row.contrast_ci95_high_image_boot) + 0.7,
            f"p={float(row.contrast_p_image_bootstrap_sign):.3g}",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["separate\nq95-q100", "matched\narcmin range"])
    ax.set_ylabel("Across - along SSI residual\n(percentage points)")
    ax.set_title("Panel C high-path tail contrast\nSF >= 0.50, match <= 15 deg, coh >= 0.20", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="0.90", linewidth=0.8)
    fig.tight_layout()
    return fig


def main() -> None:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    unit_to_images = panel_c._selected_unit_images(data["unit"], data["image"], match_max_deg=MATCH_MAX_DEG)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    across_edges = panel_c._quantile_edges_from_probs(
        metrics.loc[drift_mask, "across_path_arcmin"].to_numpy(dtype=float),
        BIN_QUANTILES,
    )
    along_edges = panel_c._quantile_edges_from_probs(
        metrics.loc[drift_mask, "along_path_arcmin"].to_numpy(dtype=float),
        BIN_QUANTILES,
    )
    across_bins = _assign_bins(metrics["across_path_arcmin"].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics["along_path_arcmin"].to_numpy(dtype=float), along_edges)

    separate_across_mask = drift_mask & (across_bins == len(across_edges) - 2)
    separate_along_mask = drift_mask & (along_bins == len(along_edges) - 2)
    common_low = max(float(across_edges[-2]), float(along_edges[-2]))
    common_high = min(float(across_edges[-1]), float(along_edges[-1]))
    across_path = metrics["across_path_arcmin"].to_numpy(dtype=float)
    along_path = metrics["along_path_arcmin"].to_numpy(dtype=float)
    matched_across_mask = drift_mask & (across_path >= common_low) & (across_path <= common_high)
    matched_along_mask = drift_mask & (along_path >= common_low) & (along_path <= common_high)

    rows = [
        _contrast_row(
            data,
            label="separate_q95_q100",
            description="Across and along each use their own q95-q100 component-path tail bin.",
            across_mask=separate_across_mask,
            along_mask=separate_along_mask,
            across_range=(float(across_edges[-2]), float(across_edges[-1])),
            along_range=(float(along_edges[-2]), float(along_edges[-1])),
            metrics=metrics,
            row_image_index=row_image_index,
            baseline_lookup=baseline_lookup,
            unit_to_images=unit_to_images,
            n_images=n_images,
            rng=rng,
        ),
        _contrast_row(
            data,
            label="matched_absolute_tail_range",
            description="Across and along use the same absolute component-path range given by the overlap of their q95-q100 tails.",
            across_mask=matched_across_mask,
            along_mask=matched_along_mask,
            across_range=(common_low, common_high),
            along_range=(common_low, common_high),
            metrics=metrics,
            row_image_index=row_image_index,
            baseline_lookup=baseline_lookup,
            unit_to_images=unit_to_images,
            n_images=n_images,
            rng=rng,
        ),
    ]
    frame = pd.DataFrame(rows)
    panel_c.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = panel_c.OUT_DIR / f"{OUT_STEM}.csv"
    json_path = panel_c.OUT_DIR / f"{OUT_STEM}_summary.json"
    png_path = panel_c.OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = panel_c.OUT_DIR / f"{OUT_STEM}.pdf"
    frame.to_csv(csv_path, index=False)
    fig = _plot_contrasts(frame)
    fig.savefig(png_path, dpi=230, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _write_json(
        json_path,
        {
            "analysis": OUT_STEM,
            "matrix_dir": panel_c.MATRIX_DIR,
            "out_dir": panel_c.OUT_DIR,
            "outputs": {
                "png": png_path,
                "pdf": pdf_path,
                "values_csv": csv_path,
                "summary_json": json_path,
            },
            "selection": {
                "sf_metric_col": panel_c.SF_METRIC_COL,
                "sf_min_cpd": panel_c.SF_MIN_CPD,
                "contour_coherence_min": panel_c.CONTOUR_COHERENCE_MIN,
                "min_osi": panel_c.MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "n_selected_units": int(len(unit_to_images)),
                "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            },
            "binning": {
                "component_bin_quantiles": BIN_QUANTILES,
                "across_q95_q100_edges": [float(across_edges[-2]), float(across_edges[-1])],
                "along_q95_q100_edges": [float(along_edges[-2]), float(along_edges[-1])],
                "matched_absolute_tail_range": [common_low, common_high],
            },
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired image bootstrap of the across-minus-along residual-percent contrast",
            },
        },
    )
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
