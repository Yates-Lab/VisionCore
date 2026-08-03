#!/usr/bin/env python3
"""Panel I (displayed letter; module/file keep the "h" name for history):
unwrapped FEM position-spread profile by local edge coherence.

Originally Panel H; the whole figure's G/H/I shifted to H/I/J once a new
panel G (contour-normal/parallel decomposition, reserved/placeholder) was
inserted between F and the RMS-excursion panel -- see
_fig4_ssi_common.py's draw_contour_components_panel and EF_INSET_*
constants.

This regenerates a compact, wider-bin version of
``b_position_spread_unwrapped_overlay_by_edge_coherence_zoomed`` from the saved
profile table produced by the BackImage contour-motion component analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _fig4_panel_header


from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths
from _fig4_style import INK, configure_matplotlib
OUT_DIR = _paths.PANELS_DIR
PROFILE_CSV = _paths.EDGE_COHERENCE_PROFILES_CSV
BASELINE_CSV = _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV

WIDE_COHERENCE_BANDS = (
    (0.0, 0.2, "0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 1.0, "0.8-1"),
)
COLORS = ("#8DAF8C", "#5D966D", "#2D7C5C", "#0E4E3D")
GRID = "#d8dde3"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _parse_bin_edges(label: str) -> tuple[float, float]:
    lo, hi = str(label).split("-", maxsplit=1)
    return float(lo), float(hi)


def _wide_band_for_source_bin(bin_low: float, bin_high: float) -> tuple[int, float, float, str] | None:
    eps = 1e-9
    for order, (lo, hi, label) in enumerate(WIDE_COHERENCE_BANDS):
        if bin_low >= lo - eps and bin_high <= hi + eps:
            return order, lo, hi, label
    return None


def _add_wide_band_columns(values: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    edges = values["coherence_bin"].astype(str).map(_parse_bin_edges)
    values["source_bin_low"] = [edge[0] for edge in edges]
    values["source_bin_high"] = [edge[1] for edge in edges]
    bands = [
        _wide_band_for_source_bin(float(row.source_bin_low), float(row.source_bin_high))
        for row in values.itertuples(index=False)
    ]
    values["wide_band_order"] = [band[0] if band is not None else np.nan for band in bands]
    values["wide_bin_low"] = [band[1] if band is not None else np.nan for band in bands]
    values["wide_bin_high"] = [band[2] if band is not None else np.nan for band in bands]
    values["wide_coherence_bin"] = [band[3] if band is not None else None for band in bands]
    return values


def _weighted_rms(values: np.ndarray | pd.Series, weights: np.ndarray | pd.Series) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.sum(w[valid] * vals[valid] ** 2) / np.sum(w[valid])))


def load_panel_values(profile_csv: Path = PROFILE_CSV) -> pd.DataFrame:
    if not profile_csv.exists():
        raise FileNotFoundError(profile_csv)
    values = pd.read_csv(profile_csv)
    required = ["coherence_bin", "relative_angle_deg", "rms_arcmin", "n_windows"]
    missing = [col for col in required if col not in values.columns]
    if missing:
        raise ValueError(f"Missing required columns in {profile_csv}: {missing}")
    values = _add_wide_band_columns(values)
    values["relative_angle_deg"] = pd.to_numeric(values["relative_angle_deg"], errors="coerce")
    values["rms_arcmin"] = pd.to_numeric(values["rms_arcmin"], errors="coerce")
    values["n_windows"] = pd.to_numeric(values["n_windows"], errors="coerce")
    values = values[
        values["wide_coherence_bin"].notna()
        & np.isfinite(values["relative_angle_deg"])
        & np.isfinite(values["rms_arcmin"])
        & np.isfinite(values["n_windows"])
    ].copy()
    if values.empty:
        raise ValueError(f"No selected coherence-bin values in {profile_csv}")

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["wide_band_order", "wide_bin_low", "wide_bin_high", "wide_coherence_bin", "relative_angle_deg"]
    for key, sub in values.groupby(group_cols, sort=True):
        order, bin_low, bin_high, label, angle = key
        weights = sub["n_windows"].to_numpy(dtype=float)
        rms = sub["rms_arcmin"].to_numpy(dtype=float)
        valid = np.isfinite(weights) & np.isfinite(rms) & (weights > 0)
        if not np.any(valid):
            continue
        total_n = float(np.sum(weights[valid]))
        merged_rms = _weighted_rms(rms, weights)
        rows.append(
            {
                "wide_band_order": int(order),
                "wide_coherence_bin": str(label),
                "wide_bin_low": float(bin_low),
                "wide_bin_high": float(bin_high),
                "relative_angle_deg": float(angle),
                "rms_arcmin": merged_rms,
                "n_windows": int(round(total_n)),
                "n_source_bins": int(sub["coherence_bin"].nunique()),
                "source_bins": ",".join(sorted(sub["coherence_bin"].astype(str).unique())),
            }
        )

    merged = pd.DataFrame(rows).sort_values(["wide_band_order", "relative_angle_deg"])
    if merged.empty:
        raise ValueError(f"No merged wider-bin values in {profile_csv}")
    return merged


def load_random_orientation_reference(baseline_csv: Path = BASELINE_CSV) -> pd.DataFrame:
    if not baseline_csv.exists():
        raise FileNotFoundError(baseline_csv)
    values = pd.read_csv(baseline_csv)
    required = [
        "coherence_bin",
        "relative_angle_deg",
        "n_windows",
        "random_orientation_median_rms_arcmin",
        "random_orientation_ci95_low_arcmin",
        "random_orientation_ci95_high_arcmin",
    ]
    missing = [col for col in required if col not in values.columns]
    if missing:
        raise ValueError(f"Missing required columns in {baseline_csv}: {missing}")
    values = _add_wide_band_columns(values)
    numeric_cols = [
        "relative_angle_deg",
        "n_windows",
        "random_orientation_median_rms_arcmin",
        "random_orientation_ci95_low_arcmin",
        "random_orientation_ci95_high_arcmin",
    ]
    for col in numeric_cols:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    values = values[
        values["wide_coherence_bin"].notna()
        & np.isfinite(values["relative_angle_deg"])
        & np.isfinite(values["n_windows"])
        & np.isfinite(values["random_orientation_median_rms_arcmin"])
    ].copy()
    if values.empty:
        raise ValueError(f"No selected random-orientation values in {baseline_csv}")

    angle_rows: list[dict[str, float | int | str]] = []
    group_cols = ["wide_band_order", "wide_bin_low", "wide_bin_high", "wide_coherence_bin", "relative_angle_deg"]
    for key, sub in values.groupby(group_cols, sort=True):
        order, bin_low, bin_high, label, angle = key
        weights = sub["n_windows"].to_numpy(dtype=float)
        total_n = float(np.sum(weights[np.isfinite(weights) & (weights > 0)]))
        angle_rows.append(
            {
                "wide_band_order": int(order),
                "wide_coherence_bin": str(label),
                "wide_bin_low": float(bin_low),
                "wide_bin_high": float(bin_high),
                "relative_angle_deg": float(angle),
                "random_orientation_median_rms_arcmin": _weighted_rms(
                    sub["random_orientation_median_rms_arcmin"], weights
                ),
                "random_orientation_ci95_low_arcmin": _weighted_rms(
                    sub["random_orientation_ci95_low_arcmin"], weights
                ),
                "random_orientation_ci95_high_arcmin": _weighted_rms(
                    sub["random_orientation_ci95_high_arcmin"], weights
                ),
                "n_windows": int(round(total_n)),
                "n_source_bins": int(sub["coherence_bin"].nunique()),
                "source_bins": ",".join(sorted(sub["coherence_bin"].astype(str).unique())),
            }
        )
    angle_values = pd.DataFrame(angle_rows)
    if angle_values.empty:
        raise ValueError(f"No merged random-orientation values in {baseline_csv}")

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["wide_band_order", "wide_bin_low", "wide_bin_high", "wide_coherence_bin"]
    for key, sub in angle_values.groupby(group_cols, sort=True):
        order, bin_low, bin_high, label = key
        weights = sub["n_windows"].to_numpy(dtype=float)
        rows.append(
            {
                "wide_band_order": int(order),
                "wide_coherence_bin": str(label),
                "wide_bin_low": float(bin_low),
                "wide_bin_high": float(bin_high),
                "random_orientation_median_rms_arcmin": _weighted_rms(
                    sub["random_orientation_median_rms_arcmin"], weights
                ),
                "random_orientation_ci95_low_arcmin": _weighted_rms(
                    sub["random_orientation_ci95_low_arcmin"], weights
                ),
                "random_orientation_ci95_high_arcmin": _weighted_rms(
                    sub["random_orientation_ci95_high_arcmin"], weights
                ),
                "n_windows": int(round(float(sub["n_windows"].dropna().iloc[0]))),
                "n_angles": int(sub["relative_angle_deg"].nunique()),
                "n_source_bins": int(sub["n_source_bins"].max()),
                "source_bins": str(sub["source_bins"].dropna().iloc[0]),
            }
        )
    reference = pd.DataFrame(rows).sort_values("wide_band_order")
    if reference.empty:
        raise ValueError(f"No flat random-orientation references in {baseline_csv}")
    return reference


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "I",
    title: str = "Real FEM spread is\ncontour-anisotropic",
    values: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = load_panel_values() if values is None else values.copy()
    try:
        reference = load_random_orientation_reference() if reference is None else reference.copy()
    except Exception:
        reference = pd.DataFrame()

    observed_handles = []
    band_labels = [band[2] for band in WIDE_COHERENCE_BANDS]
    for bin_name, color in zip(band_labels, COLORS, strict=True):
        ref = reference[reference["wide_coherence_bin"].astype(str).eq(bin_name)] if not reference.empty else pd.DataFrame()
        if not ref.empty:
            ref_row = ref.iloc[0]
            y = float(ref_row["random_orientation_median_rms_arcmin"])
            lo = float(ref_row["random_orientation_ci95_low_arcmin"])
            hi = float(ref_row["random_orientation_ci95_high_arcmin"])
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                ax.fill_between([0.0, 180.0], [lo, lo], [hi, hi], color=color, alpha=0.055, lw=0, zorder=0)
            if np.isfinite(y):
                ax.plot([0.0, 180.0], [y, y], color=color, lw=0.9, ls=(0, (3.2, 2.2)), alpha=0.72, zorder=1)

        sub = values[values["wide_coherence_bin"].astype(str).eq(bin_name)].sort_values("relative_angle_deg")
        if sub.empty:
            continue
        n_windows = int(pd.to_numeric(sub["n_windows"], errors="coerce").dropna().iloc[0])
        label_text = f"{bin_name} (n={n_windows})"
        (line,) = ax.plot(
            sub["relative_angle_deg"],
            sub["rms_arcmin"],
            color=color,
            lw=1.75,
            label=label_text,
            zorder=3,
        )
        observed_handles.append(line)

    ax.axvline(90.0, color="#7d858c", lw=0.75, ls=":", zorder=2)
    ax.set_xlim(0.0, 180.0)
    y_arrays = [values["rms_arcmin"].to_numpy(dtype=float)]
    if not reference.empty:
        for col in [
            "random_orientation_median_rms_arcmin",
            "random_orientation_ci95_low_arcmin",
            "random_orientation_ci95_high_arcmin",
        ]:
            y_arrays.append(reference[col].to_numpy(dtype=float))
    finite_y = np.concatenate(y_arrays)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        lo, hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        pad = max(0.06, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    # The two "parallel" ticks sit exactly at the data/axes edges, so their
    # default center-aligned labels overhang past the axes on their outer
    # side -- bbox_inches="tight" then grows the saved page to include that
    # overhang, which pushed this panel's real width past its compositing
    # box and into its neighbor. Align each edge label's outer edge to its
    # own tick instead, so nothing extends past the axes box.
    xticklabels = ax.get_xticklabels()
    xticklabels[0].set_ha("left")
    xticklabels[-1].set_ha("right")
    ax.set_xlabel("angle from local edge")
    ax.set_ylabel("position spread RMS (arcmin)", labelpad=2.0)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _fig4_panel_header.draw_bottom_row_header(ax, label, title, title_linespacing=_fig4_panel_header.PANEL_TITLE_LINESPACING, color=INK)
    legend_handles = observed_handles
    if not reference.empty:
        from matplotlib.lines import Line2D

        legend_handles = [
            *legend_handles,
            Line2D([0], [0], color="#4b4f55", lw=0.9, ls=(0, (3.2, 2.2)), label="ori-scrambled ref."),
        ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=5.2,
        loc="upper left",
        title="edge coherence",
        title_fontsize=5.6,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    _fig4_panel_header.align_bottom_row_xlabel(ax)
    _clean_axis(ax)
    return values


def build_panel(
    out_dir: Path = OUT_DIR,
    *,
    figsize: tuple[float, float] = (2.35, 2.25),
    label: str = "I",
    title: str = "Real FEM spread is\ncontour-aligned",
) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    reference = load_random_orientation_reference()
    values.to_csv(out_dir / "panel_f_unwrapped_edge_coherence_values.csv", index=False)
    reference.to_csv(out_dir / "panel_f_unwrapped_edge_coherence_random_orientation_reference.csv", index=False)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = _fig4_panel_header.add_bottom_row_axes(fig)
    draw_panel(ax, label=label, title=title, values=values, reference=reference)
    paths = {
        "png": out_dir / "panel_f_unwrapped_edge_coherence.png",
        "pdf": out_dir / "panel_f_unwrapped_edge_coherence.pdf",
        "svg": out_dir / "panel_f_unwrapped_edge_coherence.svg",
    }
    fig.savefig(paths["png"], dpi=220, transparent=True)
    fig.savefig(paths["pdf"], transparent=True)
    fig.savefig(paths["svg"], transparent=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build_panel(args.out_dir)
    for key in ("png", "pdf", "svg"):
        print(paths[key])


if __name__ == "__main__":
    main()
