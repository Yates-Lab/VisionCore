"""Broken log x-axis for the path-length panels, and the series drawn on it.

Panel B's x axis is a zero anchor followed by a log-spaced positive range, with
a `//` break drawn between them: path length 0 (the stabilized condition) has
to sit on the same axis as 88-180 arcmin, and a plain log scale cannot show
zero. `x_broken_log` is the mapping from arcmin to that axis's coordinates;
every tick, series and limit on the axis has to go through it.

Extracted from `_fig4_geometry_story_cde8bins`, which defines these five
functions alongside ~640 lines of trace-bank analysis that the panels never
run -- importing it for the plotting half dragged the analysis half (and
`_fig4_component_2d_surface`, `_fig4_component_path_baseline` and
`_fig4_trace_schematics` behind it) onto the figure build path.
`refresh/_fig4_geometry_story_sf075.py` carries an older copy of three of these
five, predating the bootstrap-CI error bars, and has no `ylim_series` at all. It
is deliberately left alone: pointing it here would silently add error bars it
never drew.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The positive (log-spaced) half of the broken axis, in arcmin, and the ticks
# drawn on it. 0.0 is the zero anchor left of the break.
B_MIN_POS = 88.0
B_MAX_POS = 180.0
B_TICKS = (0.0, 90.0, 105.0, 120.0, 150.0, 175.0)


def x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def format_broken_axis(
    ax: plt.Axes,
    *,
    ticks: tuple[float, ...],
    min_pos: float,
    max_pos: float,
    xlabel: str,
    show_xlabel: bool = True,
) -> None:
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(x_broken_log(list(ticks), min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    if show_xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def plot_b_series(ax: plt.Axes, frame: pd.DataFrame, *, color: str) -> None:
    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=28,
        facecolors="white",
        edgecolors=color,
        linewidths=1.35,
        zorder=5,
    )
    for context, filled in [("drift_only", False), ("microsaccade", True)]:
        sub = frame[frame["context"].eq(context)].sort_values("path_bin_order")
        if sub.empty:
            continue
        x = x_broken_log(sub["path_median_arcmin"], min_pos=B_MIN_POS, max_pos=B_MAX_POS)
        y = sub["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        if "ssi_percent_ci95_low_image_boot" in sub.columns:
            ci_low = pd.to_numeric(sub["ssi_percent_ci95_low_image_boot"], errors="coerce").to_numpy(dtype=float)
            ci_high = pd.to_numeric(sub["ssi_percent_ci95_high_image_boot"], errors="coerce").to_numpy(dtype=float)
            has_ci = np.isfinite(ci_low) & np.isfinite(ci_high) & np.isfinite(y)
            if np.any(has_ci):
                yerr_low = np.clip(y[has_ci] - ci_low[has_ci], 0.0, None)
                yerr_high = np.clip(ci_high[has_ci] - y[has_ci], 0.0, None)
                ax.errorbar(
                    x[has_ci],
                    y[has_ci],
                    yerr=[yerr_low, yerr_high],
                    color=color,
                    linestyle="none",
                    elinewidth=1.1,
                    capsize=0,
                    zorder=3,
                )
        ax.plot(x, y, color=color, linewidth=1.75, zorder=2)
        ax.scatter(
            x,
            y,
            marker="o",
            s=24,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.25,
            zorder=4,
        )


def ylim_series(frame: pd.DataFrame, col: str = "ssi_percent_vs_cell_baseline") -> list[pd.Series]:
    """Point estimate plus bootstrap CI bounds (if present), so shared y-limits
    aren't clipping the error bars this function's callers go on to draw."""
    series = [frame[col]]
    for ci_col in ("ssi_percent_ci95_low_image_boot", "ssi_percent_ci95_high_image_boot"):
        if ci_col in frame.columns:
            series.append(frame[ci_col])
    return series


def shared_ylim(values: list[pd.Series], *, pad_low: float = 0.12, pad_high: float = 0.14) -> tuple[float, float]:
    arrs = [pd.to_numeric(series, errors="coerce").to_numpy(dtype=float) for series in values if not series.empty]
    vals = [0.0]
    for arr in arrs:
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    return lo - pad_low * span, hi + pad_high * span
