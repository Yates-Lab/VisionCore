#!/usr/bin/env python3
"""Drawing code for the schematic panels (displayed A and C).

This began as the whole figure's entry point -- one script that composed every
panel onto a shared canvas. `generate_figure4.py` replaced that: each panel is
now rendered independently and composited with pypdf, so this module is a
library, not a script. Its callers are `panel_a_motion_schematic` and
`panel_c_contour_relative_stimulus`, which between them use exactly four names
from here: `configure_matplotlib`, `read_schematic_payload`, `draw_panel_a`
and `draw_panel_b`.

What is left is panel-level: the two panel builders, the layout boxes they
place things in, and the shared chrome (headers, movie cube, model icon,
colorbar, placeholders) they draw with. The layer underneath -- loading the
cached schematic payload and annotating a stimulus crop axis -- is
`_fig4_schematic_axes`, which this module imports and never the other way
round. The old entry point (`build_figure`/`main`), the story-panel drawing,
and the layout-box SVG exporter are gone.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from matplotlib import patches

import _fig4_panel_header


from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths
from _fig4_style import BLUE, GRAY, INK, ORANGE, PALE_GRID
# Re-exported, not used here: panel_a and panel_c reach it as
# `figure.configure_matplotlib()`.
from _fig4_style import configure_matplotlib  # noqa: F401
# `read_schematic_payload` is re-exported: panel_a and panel_c reach it as
# `figure.read_schematic_payload()`.
# `read_schematic_payload` is re-exported, not used here: panel_a and panel_c
# reach it as `figure.read_schematic_payload()`.
from _fig4_schematic_axes import read_schematic_payload  # noqa: F401
from _fig4_schematic_axes import (
    CROP_BORDER_LW,
    CYAN,
    D_CONNECTOR_LW,
    TRACE_COLOR,
    ZOOM_BOX,
    _finite_float,
    add_center_zoom_box_to_crop_axis,
    add_center_zoom_to_zoom_connectors,
    add_contour_axis_line_to_crop_axis,
    add_contour_window_parent_overlay,
    add_contour_window_to_crop_axis,
    add_lower_left_image_label,
    add_roi_to_crop_connectors,
    add_trajectory_span_arrows_to_crop_axis,
    add_upper_left_image_label,
    add_zoomed_crop_view,
    contour_window_metadata,
    data_width_for_physical_aspect,
    draw_plain_crop,
    hide_axis_completely,
    restore_trace_orientation,
    schematic_response_maps,
    stimulus_canvas_aspect,
    trace_fit_center_zoom_metadata,
)

FIG4_DIR = ROOT / "paper" / "fig4"

try:  # noqa: E402
    import _fig4_contour_schematic as ssi_schematic

    SCHEMATIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    ssi_schematic = None
    SCHEMATIC_IMPORT_ERROR = exc

try:  # noqa: E402
    import _fig4_path_bins

    PANEL_BCEF_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    _fig4_path_bins = None
    PANEL_BCEF_IMPORT_ERROR = exc

# panel_e / panel_f / panel_g used to be imported here too, for the composite
# entry point that `generate_figure4.py` replaced. Those imports made
# `panel_a -> _fig4_ssi_common -> panel_e/f/g` a genuine import cycle, held
# together only by the try/except: nothing in this module calls them any more.

try:  # noqa: E402
    import _fig4_coherence_gallery

    PANEL_D_GALLERY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    _fig4_coherence_gallery = None
    PANEL_D_GALLERY_IMPORT_ERROR = exc

OUT_DIR = _paths.FIG_DIR
STORY_STEM = "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins"
PANEL_BCEF_STEM = "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15"
PANEL_B_VALUES_CSV = _paths.STORY_PANEL_B_VALUES_CSV
COMPONENT_VALUES_CSV = _paths.STORY_COMPONENT_VALUES_CSV

EYE_TRAJECTORY_COLOR = TRACE_COLOR
UNIT_TUNING_COLOR = TRACE_COLOR
PLACEHOLDER_FILL = "#F6F7F8"
PLACEHOLDER_EDGE = "#B9BFC6"
D_FULL_IMAGE_LABEL_Y = 0.659
D_CROP_IMAGE_LABEL_X_FRAC = 0.63
D_CROP_IMAGE_LABEL_Y = 0.556
D_ZOOM_CROP_BORDER_LW = 1.9
D_TRACE_LABEL_FS = 6.2
D_SUBHEAD_FS = 7.0
D_ZOOM_OVERLAY_SCALE = 0.88
# E/F now live as insets inside D's own axes (data coords, D's xlim/ylim are
# (0, 1)) rather than as separate gridspec cells -- this is the region that
# used to hold D's unfinished "Contour-carried signal" placeholder, which
# moved out to its own panel (see draw_contour_components_panel).
# Heights/widths at scale 1.0 make E/F's physical footprint (inside D's
# axes) match B/C's exactly -- 0.355/0.4454 of D's data-x/y range ==
# B/C's real gridspec-cell width/height (1.955in / 1.337in), computed
# directly via gridspec instantiation. AXES_SHRINK (see B/C below, applied
# via _shrink_axes_center there) scales both by the same factor here,
# re-centered on the same midpoint each occupied at scale 1.0, to keep B/C
# and E/F matched to each other at whatever size AXES_SHRINK picks. Back at
# 1.0 (full size) -- an earlier ~10% shrink read as too small in review.
_EF_FULL_X, _EF_FULL_W = 0.620, 0.355
_EF_FULL_F_Y, _EF_FULL_F_H = 0.020, 0.4454
_EF_FULL_E_Y, _EF_FULL_E_H = 0.660, 0.4454
AXES_SHRINK = 1.0
EF_INSET_W = _EF_FULL_W * AXES_SHRINK
EF_INSET_X = _EF_FULL_X + (_EF_FULL_W - EF_INSET_W) / 2
EF_INSET_F_H = _EF_FULL_F_H * AXES_SHRINK
EF_INSET_F_Y = _EF_FULL_F_Y + (_EF_FULL_F_H - EF_INSET_F_H) / 2
EF_INSET_E_H = _EF_FULL_E_H * AXES_SHRINK
EF_INSET_E_Y = _EF_FULL_E_Y + (_EF_FULL_E_H - EF_INSET_E_H) / 2
# The E/F gap (E_Y - (F_Y + F_H)) has to clear both F's title (~0.047 above
# its own axes box) and E's x-tick labels/xlabel (~0.144 below its own axes
# box) or the two collide -- measured via get_tightbbox, not visually
# obvious from the nominal axes boxes alone; shrinking only widens that
# clearance. D's row has ~0.18 of slack above y=1 and below y=0 (the
# gridspec hspace to the neighboring rows) to park overflow in.
FIGURE_SIZE_IN = (8.5, 11.0)
MAIN_GRID_KWARGS = {
    "left": 0.060,
    "right": 0.982,
    "top": 0.930,
    "bottom": 0.045,
    "width_ratios": [1.18, 1.18, 0.90],
    "height_ratios": [1.24, 1.16, 0.94],
    "hspace": 0.190,
    "wspace": 0.160,
}
RIGHT_PANEL_HSPACE = 0.400
# A and D are self-drawn (axis off) and don't need MAIN_GRID_KWARGS['left'] --
# that margin exists for H's automatic y-tick labels/ylabel, which A/D don't
# have. Give A/D their own tighter left edge instead of the shared gridspec
# column boundary; see _wide_panel_axes.
WIDE_PANEL_LEFT = 0.016


PANEL_BOX_LABELS = {
    "A": "Motion schematic",
    "B": "Low-SF units",
    "C": "High-SF units",
    "D": "Unit tuning interacts with local image content",
    "E": "Low-SF aligned",
    "F": "High-SF aligned",
    "G": "Local contour detail (crop ref. + zoom, from D)",
    "H": "Aligned high-SF RMS excursion",
    "I": "Position spread",
    "J": "Trace-contour match advantage",
}


def draw_panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    y: float = 1.025,
    title_linespacing: float = _fig4_panel_header.MIDDLE_ROW_TITLE_LINESPACING,
    title_y_offset: float = 0.0,
    title_y_offset_pt: float = 0.0,
) -> None:
    _fig4_panel_header.draw_panel_header(
        ax,
        letter,
        title,
        y=y,
        title_linespacing=title_linespacing,
        title_y_offset=title_y_offset,
        title_y_offset_pt=title_y_offset_pt,
    )


def set_panel_title(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    color: str = INK,
    fontsize: float = 8.6,
    pad: float = 3.0,
    linespacing: float = 1.0,
) -> None:
    ax.set_title(
        f"{label}  {title}",
        loc="left",
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        pad=pad,
        linespacing=linespacing,
    )


def placeholder_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    sublabel: str | None = None,
    hatch: str = "///",
    edgecolor: str = PLACEHOLDER_EDGE,
    label_size: float = 7.0,
) -> None:
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=PLACEHOLDER_FILL,
            edgecolor=edgecolor,
            linewidth=0.9,
            hatch=hatch,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.012 if sublabel else 0.0),
        label,
        ha="center",
        va="center",
        fontsize=label_size,
        color="#343A40",
        fontweight="bold",
        linespacing=1.1,
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sublabel,
            ha="center",
            va="center",
            fontsize=max(label_size - 1.2, 5.2),
            color=GRAY,
            linespacing=1.08,
        )


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="data",
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0, mutation_scale=12),
    )


def draw_movie_cube(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    jittered: bool,
) -> None:
    dx = 0.070
    dy = 0.052
    top = patches.Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor="#F1F5F6",
        edgecolor=CYAN,
        lw=1.4,
        hatch="//" if jittered else "--",
    )
    side = patches.Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor="#E7ECEE",
        edgecolor=CYAN,
        lw=1.4,
        hatch="//" if jittered else "--",
    )
    ax.add_patch(top)
    ax.add_patch(side)
    placeholder_box(
        ax,
        x,
        y,
        w,
        h,
        "movie\nplaceholder",
        hatch="//" if jittered else "--",
        edgecolor=CYAN,
        label_size=6.3,
    )
    if label:
        ax.text(x + w / 2, y + h + dy + 0.035, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    ax.annotate(
        "",
        xy=(x + w + dx * 0.72, y - 0.040),
        xytext=(x + 0.020, y - 0.040),
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=GRAY, mutation_scale=9),
    )
    ax.text(x + 0.5 * w, y - 0.066, "267 ms", ha="center", va="top", fontsize=6.0, color=GRAY)


def draw_schematic_movie_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    schematic_payload: dict | None,
    trace_key: str,
    trace_color: str,
    fallback_jittered: bool,
) -> None:
    if ssi_schematic is None or schematic_payload is None:
        draw_movie_cube(ax, x, y + 0.035, w * 0.82, h * 0.56, label=label, jittered=fallback_jittered)
        return
    try:
        cube_ax = ax.inset_axes([x, y, w, h], transform=ax.transData)
        ssi_schematic.add_visual_model_input_cube(
            cube_ax,
            schematic_payload.get("patch"),
            schematic_payload.get("contour_axis_image_deg", 10.352312),
            show_motion_labels=False,
            show_model_labels=False,
            source_image=schematic_payload.get("stimulus_model_source_patch"),
            trace_xy=schematic_payload.get(trace_key),
            trace_path_color=trace_color,
            show_motion_overlay=False,
        )
        if label:
            ax.text(x + w / 2, y + h + 0.008, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    except Exception:
        draw_movie_cube(ax, x, y + 0.035, w * 0.82, h * 0.56, label=label, jittered=fallback_jittered)


def draw_model_icon(ax: plt.Axes, x: float, y: float, sx: float = 1.0, sy: float | None = None) -> None:
    """Draw the compact single-unit-readout icon: a small feedforward
    network sketch (the "model") feeding into the position-readout diamond
    stack (unchanged from before -- still built from the same rf_x anchor).

    ``sx``/``sy`` scale the icon's horizontal and vertical extent
    independently (``sy`` defaults to ``sx``) so it can be squeezed
    horizontally without shrinking its vertical presence next to the taller
    movie-cube and response-map images either side of it.
    """
    sy = sx if sy is None else sy
    net_w = 0.075 * sx
    net_h = 0.150 * sy
    left_x, right_x = x, x + net_w
    left_ys = [y + f * net_h for f in (0.0, 0.33, 0.67, 1.0)]
    right_ys = [y + f * net_h for f in (0.12, 0.50, 0.88)]
    for ly in left_ys:
        for ry in right_ys:
            ax.plot([left_x, right_x], [ly, ry], color="#B9BFC6", lw=0.5, alpha=0.75, zorder=1)
    for nx, nys in ((left_x, left_ys), (right_x, right_ys)):
        for ny in nys:
            ax.scatter([nx], [ny], s=26 * sx, color=GRAY, zorder=2, edgecolor="none")

    label_x = x + net_w * 0.5
    ax.text(label_x, y + net_h + 0.050 * sy, "single unit", ha="center", va="bottom", fontsize=5.4, color=INK)
    ax.text(label_x, y + net_h + 0.030 * sy, "readout", ha="center", va="bottom", fontsize=5.4, color=INK)
    ax.plot(
        [label_x, label_x], [y + net_h * 0.60, y + net_h + 0.026 * sy], color=INK, lw=0.6, zorder=3
    )

    rf_x = x + 0.204 * sx
    add_flow_arrow(ax, (right_x + 0.006, y + 0.077 * sy), (rf_x - 0.010, y + 0.077 * sy))
    for j in range(3):
        ax.add_patch(
            patches.Rectangle(
                (rf_x + 0.016 * j * sx, y + 0.037 * sy + 0.021 * j * sy),
                0.049 * sx,
                0.080 * sy,
                facecolor="#EFF7EF",
                edgecolor="#49834E",
                linewidth=0.75,
                alpha=0.75,
            )
        )
    ax.plot([rf_x + 0.017 * sx, rf_x + 0.083 * sx], [y + 0.047 * sy, y + 0.123 * sy], color="#267335", lw=1.0)
    ax.plot(rf_x + 0.050 * sx, y + 0.084 * sy, marker="o", ms=2.4, color="#267335")
    ax.text(rf_x + 0.058 * sx, y + 0.134 * sy, "x,y", fontsize=4.8, color="#267335")
    ax.text(rf_x + 0.040 * sx, y + 0.017 * sy, "one response\nper position", ha="center", va="top", fontsize=4.7, color=GRAY)


# Must match _fig4_contour_schematic.py's own choice for
# PANEL_B_ACTIVATION_MAP_STYLE == "mean_centered_diverging" (the style this
# figure actually uses) -- that module doesn't expose its colormap/limits
# through add_spatial_activation_map's return value, so the colorbar here is
# built independently from the same vmin/vmax already computed for it.
RESPONSE_MAP_CMAP = "RdBu_r"


def add_response_map_colorbar(ax: plt.Axes, x: float, y: float, h: float, vlim: tuple[float, float] | None) -> None:
    """A slim vertical colorbar to the right of a response map, aligned to
    its full height. This is a per-pixel firing-rate scale (mean-centered
    for display), NOT the map's scalar SSI -- see draw_response_placeholder
    for that separate number. Labeled with its real units so it doesn't get
    mistaken for one.
    """
    if vlim is None:
        return
    vmin, vmax = vlim
    cbar_w = 0.020
    cbar_ax = ax.inset_axes([x, y, cbar_w, h], transform=ax.transData)
    sm = mpl_cm.ScalarMappable(cmap=RESPONSE_MAP_CMAP, norm=mpl_colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=[vmin, 0.0, vmax])
    cbar.ax.set_yticklabels([f"{vmin:+.2f}", "0", f"{vmax:+.2f}"], fontsize=4.4)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2.0, width=0.5, pad=1.0)
    ax.text(
        x + cbar_w / 2,
        y + h + 0.010,
        "Δ rate\n(spikes/s)",
        ha="center",
        va="bottom",
        fontsize=4.6,
        color=INK,
        linespacing=1.05,
    )


def draw_response_placeholder(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    real_map=None,
    map_vlim: tuple[float, float] | None = None,
    ssi_bits_per_spike: float | None = None,
) -> None:
    if ssi_schematic is not None and real_map is not None:
        img_w = data_width_for_physical_aspect(ax, h, 1.0)
        map_ax = ax.inset_axes([x, y, img_w, h], transform=ax.transData)
        ssi_schematic.add_spatial_activation_map(
            map_ax,
            "fem",
            real_map=real_map,
            map_vlim=map_vlim,
        )
        ax.text(
            x + img_w / 2,
            y - 0.020,
            label.replace("\n", " "),
            ha="center",
            va="top",
            fontsize=5.7,
            color=GRAY,
        )
        # SSI is one scalar per map (bits/spike), not a per-pixel quantity --
        # that's what the colorbar to the right shows instead (firing rate).
        if ssi_bits_per_spike is not None and math.isfinite(ssi_bits_per_spike):
            ax.text(
                x + img_w / 2,
                y - 0.048,
                f"SSI = {ssi_bits_per_spike:.2f} bits/spike",
                ha="center",
                va="top",
                fontsize=6.0,
                color=INK,
                fontweight="bold",
            )
        add_response_map_colorbar(ax, x + img_w + 0.018, y, h, map_vlim)
        return
    placeholder_box(
        ax,
        x,
        y,
        w,
        h,
        label,
        sublabel="activation map\nfrom real run",
        hatch="xx",
        label_size=6.7,
    )


def panel_a_default_layout_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Default Panel A block-layout geometry, as named boxes ``(x, y, w, h)``
    in this panel's own 0..1 axes-fraction coordinates (bottom-left origin,
    matching ``ax.set_xlim(0, 1)``/``ax.set_ylim(0, 1)`` in draw_panel_b).

    This is the single source of truth draw_panel_b falls back to when no
    ``layout_overrides`` is given, and what
    panels/panel_a_layout_boxes.py exports as an editable SVG template (one
    rect per box, drawn over a raster preview of the current render) and
    re-imports after manual dragging/resizing -- see that module's
    docstring for the whole round trip.

    v3 note: these fractions are tuned directly against this panel's own
    measured box (not against the reference PDF's internal proportions --
    the reference is a hand-edited Illustrator artifact and isn't a
    reliable source for sub-panel layout).
    """
    movie_w, movie_h = 0.376, 0.496
    content_y_shift = 0.079
    top_label_y, bottom_label_y = 0.900 + content_y_shift, 0.450 + content_y_shift
    # movie_overlap_above: how far the movie box's top edge sits above its
    # own row label's baseline. Was pushed to +0.100 as a one-off test of
    # overlapping the label on purpose, then back down to -0.025 (clear of
    # it), then halfway back up to split the difference.
    movie_overlap_above = 0.0375
    movie_x = 0.045
    top_movie_y = top_label_y + movie_overlap_above - movie_h
    bottom_movie_y = bottom_label_y + movie_overlap_above - movie_h

    # icon_sx/icon_sy are draw_model_icon's own scale factors (for v2's
    # matplotlib reproduction of the icon; v3 stamps a real vector asset
    # instead -- see panels/panel_a_motion_schematic.py -- but still uses
    # this box's w/h to size and place it). icon_w/icon_h are just those
    # scale factors converted to the same box units everything else uses.
    icon_sx, icon_sy = 0.42, 1.05
    icon_w = 0.285 * icon_sx  # matches draw_model_icon's own right-edge extent (rf box end) at scale sx
    icon_h = 0.150 * icon_sy
    map_w, map_h = 0.260, 0.290
    # Wider gap than movie->icon: the icon's own "one response per position"
    # caption and the response map's caption both live in this gap and
    # collide if it's too tight.
    icon_x = movie_x + movie_w + 0.025
    map_x = icon_x + icon_w + 0.060
    # Response maps anchor at the pre-overlap label position (not the movie
    # box's own top edge, which can sit above or below the label depending
    # on movie_overlap_above) so the map/colorbar never gets dragged around
    # by the cube's own vertical position -- see draw_panel_b.
    map_top_anchor = 0.025
    top_map_y = top_label_y - map_top_anchor - map_h
    bottom_map_y = bottom_label_y - map_top_anchor - map_h
    top_icon_y = top_movie_y + 0.1275 * movie_h
    bottom_icon_y = bottom_movie_y + 0.1275 * movie_h

    return {
        "label_fem": (movie_x, top_label_y, 0.150, 0.001),
        "label_stable": (movie_x, bottom_label_y, 0.150, 0.001),
        "movie_fem": (movie_x, top_movie_y, movie_w, movie_h),
        "movie_stable": (movie_x, bottom_movie_y, movie_w, movie_h),
        "icon_fem": (icon_x, top_icon_y, icon_w, icon_h),
        "icon_stable": (icon_x, bottom_icon_y, icon_w, icon_h),
        "map_fem": (map_x, top_map_y, map_w, map_h),
        "map_stable": (map_x, bottom_map_y, map_w, map_h),
    }


def draw_panel_b(
    ax: plt.Axes,
    *,
    schematic_payload: dict | None = None,
    include_network_icon: bool = True,
    layout_overrides: dict[str, tuple[float, float, float, float]] | None = None,
    header_label: str = "A",
    header_title: str = "FEMs sharpen spatial coding",
    header_y: float = 1.010,
    header_title_y_offset: float = 0.0,
    header_title_y_offset_pt: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Draw Panel A's two movie-cube/network-icon/response-map rows.

    ``include_network_icon=False`` skips drawing the matplotlib
    single-unit-readout icon (and its icon->map flow arrow, which the
    extracted icon already includes as its own trailing arrow) so a caller
    can stamp a vector asset there instead -- see
    panels/panel_a_motion_schematic.py and
    panels/_fig4_network_icon.py. The returned dict always reports
    where that icon slot is, in this axes' own 0..1 data coordinates, keyed
    by row ("fem"/"stable") with x/y/w/h/sx/sy fields, regardless of whether
    it was actually drawn.

    ``layout_overrides`` replaces individual boxes from
    panel_a_default_layout_boxes() by name ("label_fem", "label_stable",
    "movie_fem", "movie_stable", "icon_fem", "icon_stable", "map_fem",
    "map_stable") -- see panels/panel_a_layout_boxes.py. Each box is fully
    independent once overridden (e.g. moving movie_fem does not also drag
    icon_fem along with it) -- the *defaults* are what's derived from the
    movie box via fixed gaps, not a live relationship.
    """
    hide_axis_completely(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    real_maps, real_map_vlim, real_map_ssi = schematic_response_maps(schematic_payload)
    draw_panel_header(
        ax,
        header_label,
        header_title,
        y=header_y,
        title_y_offset=header_title_y_offset,
        title_y_offset_pt=header_title_y_offset_pt,
    )

    boxes = {**panel_a_default_layout_boxes(), **(layout_overrides or {})}
    label_fem_x, top_label_y = boxes["label_fem"][:2]
    label_stable_x, bottom_label_y = boxes["label_stable"][:2]
    top_movie_x, top_movie_y, top_movie_w, top_movie_h = boxes["movie_fem"]
    bottom_movie_x, bottom_movie_y, bottom_movie_w, bottom_movie_h = boxes["movie_stable"]
    top_icon_x, top_icon_y, top_icon_w, top_icon_h = boxes["icon_fem"]
    bottom_icon_x, bottom_icon_y, bottom_icon_w, bottom_icon_h = boxes["icon_stable"]
    top_map_x, top_map_y, top_map_w, top_map_h = boxes["map_fem"]
    bottom_map_x, bottom_map_y, bottom_map_w, bottom_map_h = boxes["map_stable"]

    # draw_model_icon (v2's matplotlib reproduction of the icon; v3 stamps a
    # real vector asset over this same slot instead) takes scale factors,
    # not a box -- recover them from each row's own icon box so an override
    # still sizes it correctly.
    top_icon_sx, top_icon_sy = top_icon_w / 0.285, top_icon_h / 0.150
    bottom_icon_sx, bottom_icon_sy = bottom_icon_w / 0.285, bottom_icon_h / 0.150

    ax.text(label_fem_x, top_label_y, "FEM jittered movie", fontsize=11.0, ha="left", va="bottom")
    ax.text(label_stable_x, bottom_label_y, "Stabilized movie", fontsize=11.0, ha="left", va="bottom")

    draw_schematic_movie_block(
        ax,
        top_movie_x,
        top_movie_y,
        top_movie_w,
        top_movie_h,
        schematic_payload=schematic_payload,
        trace_key="stimulus_real_trace_lag32",
        trace_color=EYE_TRAJECTORY_COLOR,
        fallback_jittered=True,
    )
    if include_network_icon:
        draw_model_icon(ax, top_icon_x, top_icon_y, sx=top_icon_sx, sy=top_icon_sy)
        add_flow_arrow(
            ax,
            (top_icon_x + top_icon_w + 0.008, top_icon_y + 0.077 * top_icon_sy),
            (top_map_x - 0.008, top_icon_y + 0.077 * top_icon_sy),
        )
    draw_response_placeholder(
        ax,
        top_map_x,
        top_map_y,
        top_map_w,
        top_map_h,
        "FEM response\nmap",
        real_map=real_maps.get("fem"),
        map_vlim=real_map_vlim,
        ssi_bits_per_spike=real_map_ssi.get("fem"),
    )

    draw_schematic_movie_block(
        ax,
        bottom_movie_x,
        bottom_movie_y,
        bottom_movie_w,
        bottom_movie_h,
        schematic_payload=schematic_payload,
        trace_key="stimulus_endpoint_stabilized_trace_lag32",
        trace_color=GRAY,
        fallback_jittered=False,
    )
    if include_network_icon:
        draw_model_icon(ax, bottom_icon_x, bottom_icon_y, sx=bottom_icon_sx, sy=bottom_icon_sy)
        add_flow_arrow(
            ax,
            (bottom_icon_x + bottom_icon_w + 0.008, bottom_icon_y + 0.077 * bottom_icon_sy),
            (bottom_map_x - 0.008, bottom_icon_y + 0.077 * bottom_icon_sy),
        )
    draw_response_placeholder(
        ax,
        bottom_map_x,
        bottom_map_y,
        bottom_map_w,
        bottom_map_h,
        "stabilized response\nmap",
        real_map=real_maps.get("stable"),
        map_vlim=real_map_vlim,
        ssi_bits_per_spike=real_map_ssi.get("stable"),
    )

    return {
        "fem": {"x": top_icon_x, "y": top_icon_y, "w": top_icon_w, "h": top_icon_h, "sx": top_icon_sx, "sy": top_icon_sy},
        "stable": {
            "x": bottom_icon_x,
            "y": bottom_icon_y,
            "w": bottom_icon_w,
            "h": bottom_icon_h,
            "sx": bottom_icon_sx,
            "sy": bottom_icon_sy,
        },
    }


def panel_d_default_layout_boxes(
    ax: plt.Axes, schematic_payload: dict | None = None
) -> dict[str, tuple[float, float, float, float]]:
    """Default Panel D block-layout geometry, as named boxes ``(x, y, w, h)``
    in this panel's own 0..1 axes-fraction coordinates -- the single source
    of truth draw_panel_a falls back to when no ``layout_overrides`` is
    given, and what panels/panel_d_layout_boxes.py exports/imports as an
    editable SVG template (same pattern as Panel A's
    panel_a_default_layout_boxes()/panel_a_layout_boxes.py).

    full_stimulus/crop's widths are derived from their heights via
    data_width_for_physical_aspect (so the real image isn't stretched),
    which needs ``ax`` already positioned/limited as draw_panel_a leaves it
    -- this is why, unlike Panel A's boxes, this function takes ``ax``
    rather than being computable in isolation.
    """
    full_x, full_y, full_h = 0.012, 0.545, 0.420
    full_w = data_width_for_physical_aspect(ax, full_h, stimulus_canvas_aspect(schematic_payload))
    crop_y, crop_h = 0.500, 0.385
    crop_w = data_width_for_physical_aspect(ax, crop_h, 1.0)
    # Small overlap with the full-stimulus image, cascaded-photos style.
    crop_x = full_x + full_w - 0.075
    return {
        "full_stimulus": (full_x, full_y, full_w, full_h),
        "crop": (crop_x, crop_y, crop_w, crop_h),
        "gallery": (0.060, 0.075, 0.540, 0.200),
    }


def draw_panel_a(
    ax: plt.Axes,
    *,
    schematic_payload: dict | None = None,
    draw_ef_insets: bool = True,
    panel_b_values: pd.DataFrame | None = None,
    ef_ylim: tuple[float, float] = (-32, 24),
    header_label: str = "D",
    header_title: str = "Local contours define\nthe relevant image axis",
    header_y: float = 1.050,
    header_title_y_offset: float = 0.0,
    xlim: tuple[float, float] = (0.0, 1.0),
    layout_overrides: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    hide_axis_completely(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    draw_panel_header(ax, header_label, header_title, y=header_y, title_y_offset=header_title_y_offset)

    boxes = {**panel_d_default_layout_boxes(ax, schematic_payload), **(layout_overrides or {})}
    full_x, full_y, full_w, full_h = boxes["full_stimulus"]
    crop_x, crop_y, crop_w, crop_h = boxes["crop"]
    window_metadata = contour_window_metadata(schematic_payload)
    axis_image_deg = _finite_float((schematic_payload or {}).get("contour_axis_image_deg"), 10.352312)
    motion_eye: dict | None = None
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            synthetic_left = ssi_schematic.make_synthetic_left_side(
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
            )
            motion_eye = restore_trace_orientation(synthetic_left.get("eye"))
            window_metadata = trace_fit_center_zoom_metadata(window_metadata, motion_eye)
        except Exception:
            motion_eye = None
    has_real_schematic = False
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            full_ax = ax.inset_axes([full_x, full_y, full_w, full_h], transform=ax.transData)
            ssi_schematic.add_source_overview(
                full_ax,
                schematic_payload["stimulus_canvas"],
                schematic_payload["stimulus_crop_center_xy"],
                schematic_payload["stimulus_crop_size_px"],
                label=False,
            )
            full_ax.set_anchor("NW")
            full_ax.set_zorder(2)
            crop_ax = ax.inset_axes([crop_x, crop_y, crop_w, crop_h], transform=ax.transData)
            crop_ax.set_zorder(4)
            draw_plain_crop(
                crop_ax,
                schematic_payload.get("patch"),
                trace_xy_px=motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None,
                trace_color=EYE_TRAJECTORY_COLOR,
            )
            crop_ax.set_anchor("NW")
            crop_box_edge = getattr(ssi_schematic, "FIG3_CYAN", CYAN)
            crop_x1, crop_y1 = crop_ax.get_xlim()[1], crop_ax.get_ylim()[0]
            crop_ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    crop_x1,
                    crop_y1,
                    fill=False,
                    edgecolor=crop_box_edge,
                    linewidth=CROP_BORDER_LW,
                    zorder=14,
                )
            )
            add_contour_window_to_crop_axis(crop_ax, window_metadata)
            add_center_zoom_box_to_crop_axis(crop_ax, window_metadata)
            add_roi_to_crop_connectors(
                ax,
                full_ax,
                crop_ax,
                schematic_payload["stimulus_crop_center_xy"],
                schematic_payload["stimulus_crop_size_px"],
                color=crop_box_edge,
            )

            zoom_w = crop_w * D_ZOOM_OVERLAY_SCALE
            zoom_h = crop_h * D_ZOOM_OVERLAY_SCALE
            zoom_x = crop_x + crop_w * 0.68
            zoom_y = crop_y - crop_h * 0.060
            zoom_ax = ax.inset_axes([zoom_x, zoom_y, zoom_w, zoom_h], transform=ax.transData)
            zoom_ax.set_zorder(8)
            add_zoomed_crop_view(
                zoom_ax,
                schematic_payload,
                motion_eye,
                window_metadata,
                trace_color=EYE_TRAJECTORY_COLOR,
                border_color=ZOOM_BOX,
                border_lw=D_ZOOM_CROP_BORDER_LW,
            )
            add_center_zoom_to_zoom_connectors(ax, crop_ax, zoom_ax, window_metadata, color=ZOOM_BOX)
            add_contour_axis_line_to_crop_axis(zoom_ax, window_metadata, zoomed=True)
            add_trajectory_span_arrows_to_crop_axis(zoom_ax, window_metadata, motion_eye)
            zoom_ax.set_anchor("NW")
            has_real_schematic = True
        except Exception:
            has_real_schematic = False

    if not has_real_schematic:
        placeholder_box(ax, full_x, full_y, full_w, full_h, "full stimulus", sublabel="image asset", hatch="...", label_size=6.3)
        roi = (full_x + 0.096, full_y + 0.109, 0.042, 0.068)
        ax.add_patch(patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor=CYAN, linewidth=0.95))
        placeholder_box(ax, crop_x, crop_y, crop_w, crop_h, "model window", sublabel="151 x 151 crop", hatch="///", label_size=6.9)
        for start_y, end_y in [(roi[1] + roi[3], crop_y + crop_h), (roi[1], crop_y)]:
            ax.plot(
                [roi[0] + roi[2], crop_x],
                [start_y, end_y],
                color=CYAN,
                lw=D_CONNECTOR_LW,
                ls=(0, (3, 3)),
                alpha=0.58,
            )
        add_contour_window_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
    else:
        add_upper_left_image_label(full_ax, "full stimulus")
        add_lower_left_image_label(crop_ax, "gaze-centered\npatch")

    # D's lower block reinforces the local-image-content axis; trajectory
    # spread is only split quantitatively in G.
    gallery_x, gallery_y, gallery_w, gallery_h = boxes["gallery"]
    ax.text(
        gallery_x,
        gallery_y + gallery_h + 0.082,
        "Per fixation: local contour axis\nand strength (coherence) vary",
        fontsize=7.8,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.06,
    )
    gallery_kwargs = dict(
        x0=gallery_x,
        y0=gallery_y,
        w=gallery_w,
        h=gallery_h,
        gap=0.028,
        header_y=gallery_y + gallery_h + 0.030,
        header_text=None,
    )
    drew_gallery = False
    if _fig4_coherence_gallery is not None:
        try:
            drew_gallery = _fig4_coherence_gallery.draw_gallery(ax, **gallery_kwargs)
        except Exception:
            drew_gallery = False
    if not drew_gallery:
        if _fig4_coherence_gallery is not None:
            _fig4_coherence_gallery.draw_gallery_placeholder(ax, **gallery_kwargs)
        else:
            ax.text(gallery_x, gallery_y + 0.225, "local edge coherence", fontsize=D_SUBHEAD_FS, color=GRAY, ha="left")

    if not draw_ef_insets:
        return boxes

    # E/F: same path-length dose curves as always, now living inside D's
    # axes (where the unfinished "Contour-carried signal" placeholder used
    # to be) rather than a separate gridspec cell -- see EF_INSET_* constants.
    ax_e = ax.inset_axes([EF_INSET_X, EF_INSET_E_Y, EF_INSET_W, EF_INSET_E_H], transform=ax.transData)
    draw_panel_bcef_or_placeholder(
        ax_e,
        panel_b_values if panel_b_values is not None else pd.DataFrame(),
        label="E",
        title="Low-SF aligned units",
        sf_group="low_lt0p5",
        relation="contour_matched",
        color=BLUE,
        ylabel="SSI change (%)",
        ylim=ef_ylim,
    )
    ax_f = ax.inset_axes([EF_INSET_X, EF_INSET_F_Y, EF_INSET_W, EF_INSET_F_H], transform=ax.transData)
    draw_panel_bcef_or_placeholder(
        ax_f,
        panel_b_values if panel_b_values is not None else pd.DataFrame(),
        label="F",
        title="High-SF aligned units",
        sf_group="high_ge0p75",
        relation="contour_matched",
        color=ORANGE,
        ylabel="SSI change (%)",
        ylim=ef_ylim,
    )
    return boxes


def format_placeholder_plot(
    ax: plt.Axes,
    *,
    label: str,
    title: str,
    color: str,
    ylabel: str | None,
    xlabel: str,
    ylim: tuple[float, float],
) -> None:
    set_panel_title(ax, label, title, color=color)
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_xlim(-0.12, 6.25)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1.0, 3.1, 4.0, 5.0, 6.0])
    ax.set_xticklabels(["0", "90", "105", "120", "150", "175"])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    ax.plot([x_left, 0.27], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    ax.plot([0.82, x_right], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    for offset in (-0.040, 0.040):
        ax.plot(
            [0.545 + offset - 0.035, 0.545 + offset + 0.035],
            [-0.033, 0.033],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )
    ax.text(
        0.50,
        0.54,
        "data panel placeholder",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )
    ax.grid(True, color=PALE_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_bcef_or_placeholder(
    ax: plt.Axes,
    panel_b: pd.DataFrame,
    *,
    label: str,
    title: str,
    sf_group: str,
    relation: str,
    color: str,
    ylabel: str | None,
    ylim: tuple[float, float],
    show_microsaccade_legend: bool = False,
) -> None:
    if _fig4_path_bins is not None and not panel_b.empty:
        try:
            _fig4_path_bins.draw_panel(
                ax,
                values=panel_b,
                label=label,
                title=title,
                sf_group=sf_group,
                relation=relation,
                color=color,
                ylabel=ylabel,
                ylim=ylim,
                show_microsaccade_legend=show_microsaccade_legend,
            )
            return
        except Exception:
            ax.clear()
    format_placeholder_plot(
        ax,
        label=label,
        title=title,
        color=color,
        ylabel=ylabel,
        xlabel="path length (arcmin)",
        ylim=ylim,
    )


