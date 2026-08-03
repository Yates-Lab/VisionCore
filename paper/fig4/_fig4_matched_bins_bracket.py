#!/usr/bin/env python3
"""Panel G: aligned high-SF component paths with bootstrap CIs and bracket stats.

This is a compact, figure-local renderer for the precomputed outputs from
``make_backimage_panel_c_sf05_match15_matched_bins_bracket.py`` (upstream; not in this repo).
It keeps the error bars and final-bin across-minus-along contrast without
rerunning the expensive panel-C bootstrap computation during every figure build.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_cell_baseline_errorbars as panel_c  # noqa: E402


OUT_DIR = _paths.PANELS_DIR
SOURCE_STEM = "backimage_real_trace_panel_c_aligned_sf_ge_0p5_match15_matched_bins_bracket"
VALUES_CSV = _paths.MATCHED_BINS_BRACKET_VALUES_CSV
CONTRAST_CSV = _paths.MATCHED_BINS_BRACKET_LAST_BIN_CONTRAST_CSV
SUMMARY_JSON = _paths.MATCHED_BINS_BRACKET_SUMMARY_JSON
UPSTREAM_SCRIPT = (
    "upstream://make_backimage_panel_c_sf05_match15_matched_bins_bracket.py"
)

ORANGE = "#D55E00"
INK = "#111111"
GRAY = "#6B6F75"
BROKEN_AXIS_BREAK_LEFT = 0.27
BROKEN_AXIS_BREAK_RIGHT = 0.82
BROKEN_AXIS_BREAK_CENTER = 0.545


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_panel_values(
    values_csv: Path = VALUES_CSV,
    contrast_csv: Path = CONTRAST_CSV,
    summary_json: Path = SUMMARY_JSON,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    if not values_csv.exists():
        raise FileNotFoundError(values_csv)
    if not contrast_csv.exists():
        raise FileNotFoundError(contrast_csv)
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)

    values = pd.read_csv(values_csv)
    contrast = pd.read_csv(contrast_csv).iloc[0].to_dict()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    return values, contrast, summary


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_provenance(
    *,
    values_csv: Path = VALUES_CSV,
    contrast_csv: Path = CONTRAST_CSV,
    summary_json: Path = SUMMARY_JSON,
) -> dict:
    summary: dict = {}
    contrast: dict = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    if contrast_csv.exists():
        contrast = pd.read_csv(contrast_csv).iloc[0].to_dict()
    return {
        "panels": ["G"],
        "source_script": _relative(UPSTREAM_SCRIPT),
        "source_values_csv": _relative(values_csv),
        "source_contrast_csv": _relative(contrast_csv),
        "source_summary_json": _relative(summary_json),
        "selection": summary.get("selection", {}),
        "binning": summary.get("binning", {}),
        "bootstrap": summary.get("bootstrap", {}),
        "last_bin_across_along_contrast": summary.get("last_bin_across_along_contrast", contrast),
    }


def _source_context_from_contrast(contrast: dict) -> dict | None:
    text = contrast.get("last_bin_standard_drift_component_path_context")
    if not isinstance(text, str) or not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _add_component_reference_bar(ax: plt.Axes, context: dict | None) -> None:
    if not context:
        return
    low = float(context.get("q25_arcmin", np.nan))
    high = float(context.get("q75_arcmin", np.nan))
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        return
    x_low, x_high = panel_c._x_broken_log(
        [low, high],
        min_pos=panel_c.LOWER_MIN_POS,
        max_pos=panel_c.LOWER_MAX_POS,
    )
    ax.axvspan(float(x_low), float(x_high), facecolor="#7c7c7c", edgecolor="none", alpha=0.12, zorder=0)


def _add_vertical_bracket(
    ax: plt.Axes,
    *,
    x: float,
    y0: float,
    y1: float,
    label: str,
    color: str = ORANGE,
) -> None:
    low, high = sorted([float(y0), float(y1)])
    tick = 0.10
    ax.plot([x, x], [low, high], color=color, lw=1.15, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [low, low], color=color, lw=1.15, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [high, high], color=color, lw=1.15, clip_on=False, zorder=7)
    ax.text(
        x + 0.045,
        0.5 * (low + high),
        label,
        ha="left",
        va="center",
        fontsize=5.8,
        color=color,
        linespacing=0.95,
        zorder=8,
    )


def _remove_upstream_break_label(ax: plt.Axes) -> None:
    for text in list(ax.texts):
        if text.get_text() == "//":
            text.remove()


def _draw_broken_x_axis(ax: plt.Axes) -> None:
    _remove_upstream_break_label(ax)
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    ax.plot(
        [x_left, BROKEN_AXIS_BREAK_LEFT],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    ax.plot(
        [BROKEN_AXIS_BREAK_RIGHT, x_right],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    for offset in (-0.040, 0.040):
        ax.plot(
            [
                BROKEN_AXIS_BREAK_CENTER + offset - 0.035,
                BROKEN_AXIS_BREAK_CENTER + offset + 0.035,
            ],
            [-0.033, 0.033],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )


def _component_xlim_right() -> float:
    tick_x = panel_c._x_broken_log(
        panel_c.LOWER_TICKS,
        min_pos=panel_c.LOWER_MIN_POS,
        max_pos=panel_c.LOWER_MAX_POS,
    )
    return float(np.nanmax(tick_x) + 0.20)


def _set_panel_ylim(ax: plt.Axes, values: pd.DataFrame, contrast: dict) -> None:
    vals = [0.0]
    for col in [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(values.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    for key in ["contrast_ci95_low_image_boot", "contrast_ci95_high_image_boot"]:
        val = float(contrast.get(key, np.nan))
        if np.isfinite(val):
            vals.append(val)
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.16 * span, hi + 0.36 * span)


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "G",
    title: str = "Aligned high-SF components",
    values: pd.DataFrame | None = None,
    contrast: dict[str, float] | None = None,
    summary: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    if values is None or contrast is None or summary is None:
        values, contrast, summary = load_panel_values()
    else:
        values = values.copy()
        contrast = dict(contrast)
        summary = dict(summary)

    _set_panel_ylim(ax, values, contrast)
    first_rows: dict[str, pd.Series] = {}
    last_rows: dict[str, pd.Series] = {}

    for metric_col, (series_label, linestyle, marker) in panel_c.COMPONENT_STYLES.items():
        drift = values[
            values["component_metric"].eq(metric_col) & values["context"].eq("drift_only")
        ].sort_values("component_bin_order")
        if drift.empty:
            continue
        x = panel_c._x_broken_log(
            drift["plot_median_arcmin"],
            min_pos=panel_c.LOWER_MIN_POS,
            max_pos=panel_c.LOWER_MAX_POS,
        )
        y = drift["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([y - ci_low, ci_high - y])
        is_across = metric_col == "across_path_arcmin"
        line_zorder = 7 if is_across else 3
        marker_zorder = 8 if is_across else 4
        zero_zorder = 9 if is_across else 5
        ax.plot(x, y, color=ORANGE, linestyle=linestyle, linewidth=1.65, label=series_label, zorder=line_zorder)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=ORANGE,
            linestyle="none",
            marker=marker,
            markersize=3.6,
            markerfacecolor="white",
            markeredgewidth=1.0,
            elinewidth=0.9,
            capsize=1.8,
            zorder=marker_zorder,
        )
        ax.scatter(
            [0.0],
            [0.0],
            marker=marker,
            s=24,
            facecolors="white",
            edgecolors=ORANGE,
            linewidths=1.1,
            zorder=zero_zorder,
        )
        first_rows[metric_col] = drift.iloc[0]
        last_rows[metric_col] = drift.iloc[-1]

    panel_c._format_axis(
        ax,
        ticks=panel_c.LOWER_TICKS,
        min_pos=panel_c.LOWER_MIN_POS,
        max_pos=panel_c.LOWER_MAX_POS,
    )
    ax.set_xlim(-0.12, _component_xlim_right())
    _draw_broken_x_axis(ax)
    _add_component_reference_bar(ax, _source_context_from_contrast(contrast))
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_title(f"{label}  {title}", loc="left", fontsize=8.8, fontweight="bold", pad=4, color=INK)
    ax.set_ylabel("SSI change (%)")
    ax.set_xlabel("component path (arcmin)")
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in item else "along" for item in legend_labels]
        ax.legend(
            handles,
            short_labels,
            frameon=False,
            fontsize=5.9,
            loc="lower left",
            handlelength=1.8,
            borderaxespad=0.2,
        )

    selection = summary.get("selection", {})
    n_units = selection.get("n_selected_units")
    n_pairs = selection.get("n_selected_unit_image_pairs")
    if n_units is not None and n_pairs is not None:
        ax.text(
            0.985,
            0.875,
            f"{int(n_units)} units\n{int(n_pairs)} pairs",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.7,
            color=GRAY,
            linespacing=1.0,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )

    if {"across_path_arcmin", "along_path_arcmin"}.issubset(first_rows):
        across_first = first_rows["across_path_arcmin"]
        along_first = first_rows["along_path_arcmin"]
        x1_raw = max(float(across_first["plot_median_arcmin"]), float(along_first["plot_median_arcmin"]))
        x1 = float(panel_c._x_broken_log([x1_raw], min_pos=panel_c.LOWER_MIN_POS, max_pos=panel_c.LOWER_MAX_POS)[0])
        y_lo, y_hi = ax.get_ylim()
        y_span = max(y_hi - y_lo, 1.0)
        text = (
            "near 0\n"
            f"across {panel_c._format_p_label(float(across_first['population_delta_p_image_bootstrap_sign']))}\n"
            f"along {panel_c._format_p_label(float(along_first['population_delta_p_image_bootstrap_sign']))}"
        )
        panel_c._add_bracket(
            ax,
            x0=0.0,
            x1=x1,
            y=y_hi - 0.185 * y_span,
            text=text,
            color=ORANGE,
            linestyle="-",
            text_x=x1 + 0.07,
            text_ha="left",
        )

    if {"across_path_arcmin", "along_path_arcmin"}.issubset(last_rows):
        across_last = last_rows["across_path_arcmin"]
        along_last = last_rows["along_path_arcmin"]
        x_last = float(
            panel_c._x_broken_log(
                [float(across_last["plot_median_arcmin"])],
                min_pos=panel_c.LOWER_MIN_POS,
                max_pos=panel_c.LOWER_MAX_POS,
            )[0]
        )
        diff = float(contrast.get("across_minus_along_percent_point", np.nan))
        p_label = panel_c._format_p_label(float(contrast.get("contrast_p_image_bootstrap_sign", np.nan)))
        _add_vertical_bracket(
            ax,
            x=x_last + 0.22,
            y0=float(across_last["ssi_percent_vs_cell_baseline"]),
            y1=float(along_last["ssi_percent_vs_cell_baseline"]),
            label=f"{diff:+.1f} pp\n{p_label}",
        )

    ax.spines[["top", "right"]].set_visible(False)
    return values, contrast, summary


def build_panel(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values, contrast, summary = load_panel_values()
    values.to_csv(out_dir / "panel_g_matched_bins_bracket_values.csv", index=False)
    pd.DataFrame([contrast]).to_csv(out_dir / "panel_g_matched_bins_bracket_contrast.csv", index=False)
    (out_dir / "panel_g_matched_bins_bracket_provenance.json").write_text(
        json.dumps(load_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(2.45, 2.35), constrained_layout=False)
    draw_panel(ax, values=values, contrast=contrast, summary=summary)
    fig.tight_layout(pad=0.55)
    paths = {
        "png": out_dir / "_fig4_matched_bins_bracket.png",
        "pdf": out_dir / "_fig4_matched_bins_bracket.pdf",
        "svg": out_dir / "_fig4_matched_bins_bracket.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
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
