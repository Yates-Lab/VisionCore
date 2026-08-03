"""Significance brackets and p-value labels shared across figure-4 panels.

These three helpers were the only reason `_fig4_option_sheet` (and through it
panel E) imported `_fig4_geometry_story` and `_fig4_matched_bins_bracket` --
two refresh-path analysis modules that between them pull in roughly 4,000
lines of trace-bank machinery the panel never calls. Extracting them here cuts
that edge without changing a pixel.

The horizontal bracket spans a range on the x axis with the label above it;
the vertical one spans a range on the y axis with the label to its right.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from _fig4_style import ORANGE


def format_p_label(value: float) -> str:
    if not math.isfinite(float(value)):
        return "p=n/a"
    if float(value) < 0.001:
        return "p<0.001"
    return f"p={float(value):.3f}"


def add_bracket(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    text: str,
    color: str,
    linestyle: str | tuple[int, tuple[float, ...]] = "-",
    text_x: float | None = None,
    text_ha: str = "center",
) -> None:
    tick = 0.7
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color, lw=1.0, ls=linestyle, zorder=6)
    ax.text(
        0.5 * (x0 + x1) if text_x is None else text_x,
        y + 0.45,
        text,
        ha=text_ha,
        va="bottom",
        color=color,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        zorder=7,
    )


def add_vertical_bracket(
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
