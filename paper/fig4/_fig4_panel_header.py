"""Shared typography for SSI figure panel headers."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

INK = "#111111"
GRAY = "#6B6F75"

PANEL_LETTER_SIZE = 10.4
PANEL_TITLE_SIZE = 8.2
PANEL_SUBTITLE_SIZE = 5.35
PANEL_TITLE_OFFSET_PT = 17.5
PANEL_TITLE_LINESPACING = 1.15
PANEL_TITLE_LINE_STEP = 0.052
TOP_ROW_TITLE_Y_OFFSET_PT = 4.9
MIDDLE_ROW_TITLE_LINESPACING = PANEL_TITLE_LINESPACING
BOTTOM_ROW_AXES_BOX = (0.18, 0.16, 0.78, 0.64)
BOTTOM_ROW_HEADER_LETTER_X = -BOTTOM_ROW_AXES_BOX[0] / BOTTOM_ROW_AXES_BOX[2]
BOTTOM_ROW_TITLE_TOP_Y = 1.150
BOTTOM_ROW_XLABEL_Y = -0.130
MIDDLE_ROW_AXES_BOX = (0.17, 0.15, 0.78, 0.72)
MIDDLE_ROW_HEADER_LETTER_X = -MIDDLE_ROW_AXES_BOX[0] / MIDDLE_ROW_AXES_BOX[2]
MIDDLE_ROW_TITLE_TOP_Y = 1.111
MIDDLE_ROW_XLABEL_Y = -0.086
MIDDLE_ROW_YLABEL_X = -0.110


def add_bottom_row_axes(fig: plt.Figure) -> plt.Axes:
    """Add an axes with a shared baseline for the composited bottom row."""
    return fig.add_axes(list(BOTTOM_ROW_AXES_BOX))


def add_middle_row_axes(fig: plt.Figure) -> plt.Axes:
    """Add an axes with a shared template for middle-row analytic panels."""
    return fig.add_axes(list(MIDDLE_ROW_AXES_BOX))


def draw_bottom_row_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    title_top_y: float = BOTTOM_ROW_TITLE_TOP_Y,
    **kwargs: object,
) -> None:
    """Draw a bottom-row header inside a fixed three-line title reserve."""
    n_lines = title.count("\n") + 1
    y = title_top_y - (n_lines - 1) * PANEL_TITLE_LINE_STEP
    kwargs.setdefault("letter_x", BOTTOM_ROW_HEADER_LETTER_X)
    draw_panel_header(ax, letter, title, y=y, **kwargs)


def draw_middle_row_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    title_top_y: float = MIDDLE_ROW_TITLE_TOP_Y,
    **kwargs: object,
) -> None:
    """Draw a middle-row header inside a shared title/subtitle reserve."""
    n_lines = title.count("\n") + 1
    y = title_top_y - (n_lines - 1) * PANEL_TITLE_LINE_STEP
    kwargs.setdefault("letter_x", MIDDLE_ROW_HEADER_LETTER_X)
    draw_panel_header(ax, letter, title, y=y, **kwargs)


def align_bottom_row_xlabel(ax: plt.Axes, *, y: float = BOTTOM_ROW_XLABEL_Y) -> None:
    """Pin bottom-row x-axis titles to one shared text baseline zone."""
    ax.xaxis.set_label_coords(0.5, y)


def align_middle_row_xlabel(ax: plt.Axes, *, y: float = MIDDLE_ROW_XLABEL_Y) -> None:
    """Pin middle-row x-axis titles to one shared text baseline zone."""
    ax.xaxis.set_label_coords(0.5, y)


def align_middle_row_ylabel(ax: plt.Axes, *, x: float = MIDDLE_ROW_YLABEL_X) -> None:
    """Pull middle-row y-axis titles closer to their axes."""
    ax.yaxis.set_label_coords(x, 0.5)


def draw_panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    subtitle: str | None = None,
    y: float = 1.045,
    letter_x: float = 0.0,
    letter_y_offset_pt: float = 0.0,
    title_x: float | None = None,
    title_offset_pt: float = PANEL_TITLE_OFFSET_PT,
    title_size: float = PANEL_TITLE_SIZE,
    subtitle_size: float = PANEL_SUBTITLE_SIZE,
    letter_size: float = PANEL_LETTER_SIZE,
    color: str = INK,
    title_color: str | None = None,
    subtitle_color: str = GRAY,
    title_linespacing: float = PANEL_TITLE_LINESPACING,
    title_y_offset: float = 0.0,
    title_y_offset_pt: float = 0.0,
    subtitle_linespacing: float = 1.02,
    subtitle_gap: float = 0.020,
) -> None:
    """Draw a consistent letter/title/subtitle block.

    The title offset defaults to physical points rather than an axes fraction,
    so the letter-title gap stays visually constant across differently sized
    independently rendered panel PDFs.
    """
    n_lines = title.count("\n") + 1
    letter_y = y + (n_lines - 1) * PANEL_TITLE_LINE_STEP
    letter_transform = ax.transAxes
    if letter_y_offset_pt:
        letter_transform = ax.transAxes + ScaledTranslation(
            0.0,
            letter_y_offset_pt / 72.0,
            ax.figure.dpi_scale_trans,
        )
    ax.text(
        letter_x,
        letter_y,
        letter,
        transform=letter_transform,
        ha="left",
        va="bottom",
        fontsize=letter_size,
        fontweight="bold",
        color=color,
        clip_on=False,
    )
    title_transform = ax.transAxes
    title_x_pos = title_x
    title_dx_pt = 0.0
    if title_x_pos is None:
        title_x_pos = letter_x
        title_dx_pt = title_offset_pt
    if title_dx_pt or title_y_offset_pt:
        title_transform = ax.transAxes + ScaledTranslation(
            title_dx_pt / 72.0,
            title_y_offset_pt / 72.0,
            ax.figure.dpi_scale_trans,
        )
    title_y = y + title_y_offset
    ax.text(
        title_x_pos,
        title_y,
        title,
        transform=title_transform,
        ha="left",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or color,
        linespacing=title_linespacing,
        clip_on=False,
    )
    if subtitle:
        ax.text(
            title_x_pos,
            title_y - subtitle_gap,
            subtitle,
            transform=title_transform,
            ha="left",
            va="top",
            fontsize=subtitle_size,
            color=subtitle_color,
            linespacing=subtitle_linespacing,
            clip_on=False,
        )
