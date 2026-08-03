#!/usr/bin/env python3
"""Generate a draft SSI multipanel figure scaffold.

This version is composition-first: it preserves the layout and storyboard of
the supplied PDF sketch, pulls in existing BackImage story-figure plotting code
where available, and leaves the remaining result/data slots as explicit
placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from matplotlib import patches

import _fig4_panel_header


from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

FIG4_DIR = ROOT / "paper" / "fig4"

import _fig4_geometry_story_sf075 as story_panels  # noqa: E402

try:  # noqa: E402
    import _fig4_contour_schematic as ssi_schematic

    SCHEMATIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    ssi_schematic = None
    SCHEMATIC_IMPORT_ERROR = exc

try:  # noqa: E402
    import panel_bd_path_bins

    PANEL_BCEF_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_bd_path_bins = None
    PANEL_BCEF_IMPORT_ERROR = exc

try:  # noqa: E402
    import panel_e_rms_excursion

    PANEL_G_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_e_rms_excursion = None
    PANEL_G_IMPORT_ERROR = exc

try:  # noqa: E402
    import panel_f_unwrapped_edge_coherence
    import panel_g_match_advantage as panel_i_match_advantage

    PANEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_f_unwrapped_edge_coherence = None
    panel_i_match_advantage = None
    PANEL_IMPORT_ERROR = exc

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

BLUE = "#0072B2"
CYAN = "#00A9C8"
ORANGE = "#D55E00"
TRACE_COLOR = "#8B1E3F"
EYE_TRAJECTORY_COLOR = TRACE_COLOR
UNIT_TUNING_COLOR = TRACE_COLOR
GRAY = "#6B6F75"
INK = "#111111"
PALE_GRID = "#E7E7E7"
PLACEHOLDER_FILL = "#F6F7F8"
PLACEHOLDER_EDGE = "#B9BFC6"
CONTOUR_WINDOW = "#E6A700"
ZOOM_BOX = "#E8118A"  # vivid magenta -- center-zoom marker box, connectors, and the crop border all use this
CROP_BORDER_LW = 2.6
CENTER_ZOOM_HALF_DEG = 0.25
CENTER_ZOOM_TRACE_PAD_DEG = 0.04
CONTOUR_AXIS_DASH = (0, (2.1, 1.5))
D_IMAGE_LABEL_FS = 5.8
D_FULL_IMAGE_LABEL_Y = 0.659
D_CROP_IMAGE_LABEL_X_FRAC = 0.63
D_CROP_IMAGE_LABEL_Y = 0.556
D_CENTER_ZOOM_BOX_LW = 1.05
D_ZOOM_CROP_BORDER_LW = 1.9
D_TRACE_LABEL_FS = 6.2
D_SUBHEAD_FS = 7.0
D_CONNECTOR_LW = 0.95
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


def _wide_panel_axes(fig: plt.Figure, gs, row: int) -> plt.Axes:
    """Add the row-spanning A/D axes shifted flush to WIDE_PANEL_LEFT instead
    of MAIN_GRID_KWARGS['left'], as a rigid translation (same width/height as
    gs[row, :2] would give). A rigid shift, rather than stretching the left
    edge out to WIDE_PANEL_LEFT while keeping the right edge fixed, keeps
    every width-derived constant tuned elsewhere in D (EF_INSET_W matching
    B/C's width, crop/zoom sizing, etc.) numerically exact -- it just opens a
    correspondingly small gap between A/D and the column to their right
    (B/C, G) where MAIN_GRID_KWARGS['left'] used to be closed up."""
    pos = gs[row, :2].get_position(fig)
    return fig.add_axes([WIDE_PANEL_LEFT, pos.y0, pos.width, pos.height])


def _shrink_axes_center(ax: plt.Axes, factor: float = AXES_SHRINK) -> None:
    """Scale an axes' box down by ``factor`` around its own center, keeping
    its aspect ratio (both dimensions scaled equally). Used for B/C so they
    stay matched to E/F, which are shrunk the same way via EF_INSET_*."""
    pos = ax.get_position()
    cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
    new_w, new_h = pos.width * factor, pos.height * factor
    ax.set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])
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


def read_existing_story_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load precomputed BackImage panel values used by the figure."""
    panel_b = pd.read_csv(PANEL_B_VALUES_CSV) if PANEL_B_VALUES_CSV.exists() else pd.DataFrame()
    component = pd.read_csv(COMPONENT_VALUES_CSV) if COMPONENT_VALUES_CSV.exists() else pd.DataFrame()
    return panel_b, component


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_methods_provenance(out_dir: Path = OUT_DIR) -> dict:
    provenance: dict[str, object] = {
        "figure": "ssi_figure_v2",
        "outputs": {
            "png": _relative(out_dir / "ssi_figure_v2.png"),
            "pdf": _relative(out_dir / "ssi_figure_v2.pdf"),
            "svg": _relative(out_dir / "ssi_figure_v2.svg"),
            "panel_boxes_svg": _relative(out_dir / "ssi_figure_v2_panel_boxes.svg"),
        },
        "panels": {},
    }
    panels = provenance["panels"]
    panels["D"] = {
        "source_script": _relative(FIG4_DIR / "_fig4_ssi_common.py"),
        "standalone_script": _relative(
            FIG4_DIR / "panel_c_contour_relative_stimulus.py"
        ),
        "contour_window_radius_source": "stimulus_row.image_patch_radius_px",
        "contour_window_radius_interpretation": (
            "BackImage reviewed windows store image_patch_radius_px=38 px; with MODEL_PPD=37.50476617 "
            "this is 1.01 deg, drawn as a 1 deg local contour-axis/coherence window."
        ),
        "center_zoom_rule": (
            f"minimum +/-{CENTER_ZOOM_HALF_DEG:g} deg, expanded by "
            f"{CENTER_ZOOM_TRACE_PAD_DEG:g} deg beyond the selected trace extent when needed"
        ),
        "coherence_gallery": {
            "status": "real image crops, one per COHERENCE_ORDER bin",
            "source_script": _relative(
                FIG4_DIR / "_fig4_coherence_gallery.py"
            ),
            "cache_builder_script": _relative(
                FIG4_DIR / "refresh" / "build_coherence_gallery_cache.py"
            ),
            "cache_provenance_json": _relative(
                ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache" / "coherence_gallery_provenance.json"
            ),
            "note": (
                "Replaces D's former short/long-path trace legend; the real traces moved to G, where "
                "there's room to show them at a legible scale."
            ),
        },
    }
    if panel_bd_path_bins is not None:
        try:
            panels["B_C_E_F"] = panel_bd_path_bins.load_provenance()
        except Exception as exc:
            panels["B_C_E_F"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["B_C_E_F"] = {"status": "panel import failed", "error": repr(PANEL_BCEF_IMPORT_ERROR)}

    panels["G"] = {
        "status": "real image assets (crop + zoom) and a schematic decomposition, no population-level result",
        "source_script": _relative(FIG4_DIR / "_fig4_ssi_common.py"),
        "note": (
            "Reference copy of D's 151x151 crop plus the zoomed local-contour aperture that used to be "
            "drawn as D's third cascaded image; moved here so it isn't squeezed for width against D's "
            "E/F insets. Below that, this same example trace is decomposed into across-/along-contour "
            "components (draw_rms_excursion_explainer) -- the same split H's dose curve reports as "
            "separate lines, using this one trace's real geometry, not H's population statistic."
        ),
    }

    if panel_e_rms_excursion is not None:
        try:
            panels["H"] = panel_e_rms_excursion.load_provenance()
        except Exception as exc:
            panels["H"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["H"] = {"status": "panel import failed", "error": repr(PANEL_G_IMPORT_ERROR)}

    panels["I"] = {
        "source_script": _relative(FIG4_DIR / "panel_f_unwrapped_edge_coherence.py"),
        "source_profile_csv": _relative(panel_f_unwrapped_edge_coherence.PROFILE_CSV)
        if panel_f_unwrapped_edge_coherence is not None
        else None,
        "source_random_orientation_baseline_csv": _relative(panel_f_unwrapped_edge_coherence.BASELINE_CSV)
        if panel_f_unwrapped_edge_coherence is not None
        else None,
        "metric_interpretation": "Position-spread RMS profile from covariance/position cloud, not unsigned path length.",
    }
    panels["J"] = {
        "source_script": _relative(FIG4_DIR / "panel_g_match_advantage.py"),
        "source_coherence_summary_csv": _relative(panel_i_match_advantage.COHERENCE_SUMMARY_CSV)
        if panel_i_match_advantage is not None
        else None,
        "metric_interpretation": (
            "Observed-minus-random-rotation model SSI prediction (RMS excursion axis) by local edge "
            "coherence; replaced the descriptive drift-cloud/edge-alignment panel "
            "(panels/panel_i_edge_alignment.py, kept unwired for reference) once the random-rotation "
            "null showed the coherence-dependent correlation is also model-beneficial relative to chance."
        ),
    }
    return provenance


def _svg_number(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _make_layout_axes(fig: plt.Figure) -> dict[str, plt.Axes]:
    gs = fig.add_gridspec(3, 3, **MAIN_GRID_KWARGS)
    axes: dict[str, plt.Axes] = {}
    axes["A"] = _wide_panel_axes(fig, gs, 0)

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    axes["B"] = fig.add_subplot(gs_b_right[0, 0])
    _shrink_axes_center(axes["B"])
    axes["C"] = fig.add_subplot(gs_b_right[1, 0])
    _shrink_axes_center(axes["C"])

    axes["D"] = _wide_panel_axes(fig, gs, 1)
    axes["D"].set_xlim(0.0, 1.0)
    axes["D"].set_ylim(0.0, 1.0)
    axes["E"] = axes["D"].inset_axes(
        [EF_INSET_X, EF_INSET_E_Y, EF_INSET_W, EF_INSET_E_H], transform=axes["D"].transData
    )
    axes["F"] = axes["D"].inset_axes(
        [EF_INSET_X, EF_INSET_F_Y, EF_INSET_W, EF_INSET_F_H], transform=axes["D"].transData
    )
    axes["G"] = fig.add_subplot(gs[1, 2])

    axes["H"] = fig.add_subplot(gs[2, 0])
    axes["I"] = fig.add_subplot(gs[2, 1])
    axes["J"] = fig.add_subplot(gs[2, 2])
    return axes


def _figure_bbox_to_svg_box(bbox, *, page_w: float, page_h: float) -> dict[str, float]:
    return {
        "x": float(bbox.x0 * page_w),
        "y": float((1.0 - bbox.y1) * page_h),
        "width": float(bbox.width * page_w),
        "height": float(bbox.height * page_h),
    }


def _axis_position_box(ax: plt.Axes, *, page_w: float, page_h: float) -> dict[str, float]:
    return _figure_bbox_to_svg_box(ax.get_position(), page_w=page_w, page_h=page_h)


def _set_axis_text_extent_style(ax: plt.Axes, label: str) -> None:
    """Add representative axis text, excluding panel titles/headings."""
    if label in {"A", "D", "G"}:
        ax.set_axis_off()
        return
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    if label in {"B", "C", "E", "F"}:
        ax.set_xlim(-0.12, 6.10)
        ax.set_xticks([0.0, 1.16, 2.26, 3.21, 4.80, 5.90])
        ax.set_xticklabels(["0", "90", "105", "120", "150", "175"])
        ax.set_yticks([-20.0, 0.0, 20.0, 40.0])
        ax.set_xlabel("path length (arcmin)")
        if label == "B":
            ax.set_ylabel("SSI change (%)")
    elif label == "H":
        ax.set_xlim(-0.12, 5.73)
        ax.set_xticks([0.0, 0.91, 3.63, 5.23])
        ax.set_xticklabels(["0", "1", "2", "3"])
        ax.set_yticks([-15.0, 0.0, 15.0])
        ax.set_xlabel("component RMS excursion (arcmin)")
        ax.set_ylabel("SSI change (%)")
    elif label == "I":
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks([0.0, 90.0, 180.0])
        ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
        ax.set_yticks([1.7, 1.9, 2.1])
        ax.set_xlabel("angle from local edge")
        ax.set_ylabel("position spread RMS (arcmin)")
    elif label == "J":
        ax.set_xlim(-0.12, 3.12)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0-.2", ".2-.5", ".5-.8", ".8-1"])
        ax.set_yticks([-0.2, 0.0, 0.2])
        ax.set_xlabel("local edge coherence")
        ax.set_ylabel("observed - random rotated (pp SSI)")


def compute_panel_box_layout() -> dict[str, dict[str, dict[str, float]]]:
    """Return current panel axes and axis-text boxes in SVG point coordinates."""
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    axes = _make_layout_axes(fig)
    for label, ax in axes.items():
        _set_axis_text_extent_style(ax, label)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes_boxes: dict[str, dict[str, float]] = {}
    axis_text_boxes: dict[str, dict[str, float]] = {}
    for label, ax in axes.items():
        axes_boxes[label] = _axis_position_box(ax, page_w=page_w, page_h=page_h)
        if label in {"A", "D"}:
            axis_text_boxes[label] = axes_boxes[label]
        else:
            bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            axis_text_boxes[label] = _figure_bbox_to_svg_box(bbox, page_w=page_w, page_h=page_h)
    plt.close(fig)
    return {"axes": axes_boxes, "axis_text": axis_text_boxes}


def write_panel_boxes_svg(out_dir: Path = OUT_DIR) -> Path:
    """Write an editable empty-box SVG matching the current panel layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    boxes = compute_panel_box_layout()
    axes_boxes = boxes["axes"]
    axis_text_boxes = boxes["axis_text"]
    path = out_dir / "ssi_figure_v2_panel_boxes.svg"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_svg_number(page_w)}pt" '
            f'height="{_svg_number(page_h)}pt" viewBox="0 0 {_svg_number(page_w)} {_svg_number(page_h)}">'
        ),
        '  <title>SSI figure v2 editable panel bounding boxes</title>',
        (
            "  <desc>Blue boxes are raw panel axes; orange boxes include tick labels and axis labels, "
            "but exclude panel headings.</desc>"
        ),
        '  <rect id="page" x="0" y="0" width="100%" height="100%" fill="white" stroke="#d0d4d9" stroke-width="0.75"/>',
        '  <g id="panel-axis-text-boxes" fill="none" stroke="#D55E00" stroke-width="1.0" stroke-dasharray="2.5 2.5">',
    ]
    for label in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        box = axis_text_boxes[label]
        lines.append(f'    <g id="panel-{label}-axis-text" data-panel="{label}" data-box-type="axis-text">')
        lines.append(
            '      <rect '
            f'id="panel-{label}-axis-text-box" '
            f'x="{_svg_number(box["x"])}" '
            f'y="{_svg_number(box["y"])}" '
            f'width="{_svg_number(box["width"])}" '
            f'height="{_svg_number(box["height"])}" />'
        )
        lines.append("    </g>")
    lines.extend(
        [
            "  </g>",
            '  <g id="panel-axes-boxes" fill="none" stroke="#0072B2" stroke-width="1.25" stroke-dasharray="5 3">',
        ]
    )
    for label in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        box = axes_boxes[label]
        lines.append(f'    <g id="panel-{label}" data-panel="{label}" data-box-type="axes">')
        lines.append(
            '      <rect '
            f'id="panel-{label}-box" '
            f'x="{_svg_number(box["x"])}" '
            f'y="{_svg_number(box["y"])}" '
            f'width="{_svg_number(box["width"])}" '
            f'height="{_svg_number(box["height"])}" />'
        )
        lines.append(
            '      <text '
            f'id="panel-{label}-label" '
            f'x="{_svg_number(box["x"] + 5.0)}" '
            f'y="{_svg_number(box["y"] + 12.0)}" '
            'font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="9" '
            'font-weight="700" '
            'fill="#111111" '
            'stroke="none">'
            f'{label}  {PANEL_BOX_LABELS[label]}</text>'
        )
        lines.append("    </g>")
    lines.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_schematic_payload() -> dict | None:
    """Panel A's schematic payload.

    The bare `except: return None` this replaces swallowed every failure --
    a missing cache, a corrupt npz, a genuine bug in the loader -- and turned
    all of them into a silently substituted synthetic schematic. Errors now
    propagate; only explicitly-enabled degraded rendering yields None.
    """
    if ssi_schematic is None:
        if _paths.ALLOW_MISSING:
            return None
        raise _paths.Fig4MissingInput(
            "panel A's contour schematic module failed to import, so the panel "
            "cannot be drawn from real data.\n"
            f"  underlying error: {SCHEMATIC_IMPORT_ERROR!r}"
        )
    return ssi_schematic.load_real_payload()


def _finite_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def contour_window_metadata(payload: dict | None) -> dict[str, float | str]:
    """Return display metadata for the local contour-axis/coherence aperture."""
    default_ppd = 37.50476617
    if ssi_schematic is not None:
        default_ppd = _finite_float(getattr(ssi_schematic, "MODEL_PPD", default_ppd), default_ppd)

    row = {}
    if isinstance(payload, dict) and isinstance(payload.get("stimulus_row"), dict):
        row = payload["stimulus_row"]

    radius_px = _finite_float(row.get("image_patch_radius_px"), default_ppd)
    crop_size_px = _finite_float((payload or {}).get("stimulus_crop_size_px"), 151.0)
    if crop_size_px <= 0:
        crop_size_px = 151.0
    ppd = _finite_float(row.get("ppd"), default_ppd)
    if ppd <= 0:
        ppd = default_ppd
    radius_deg = radius_px / ppd
    zoom_half_px = CENTER_ZOOM_HALF_DEG * ppd
    axis_image_deg = _finite_float((payload or {}).get("contour_axis_image_deg"), 10.352312)
    coherence = _finite_float(row.get("image_orientation_coherence"), float("nan"))
    radius_label = "1 deg radius" if 0.93 <= radius_deg <= 1.08 else f"{radius_deg:.2f} deg radius"
    return {
        "radius_px": radius_px,
        "radius_deg": radius_deg,
        "radius_fraction": min(0.48, max(0.08, radius_px / crop_size_px)),
        "crop_size_px": crop_size_px,
        "ppd": ppd,
        "center_zoom_half_deg": CENTER_ZOOM_HALF_DEG,
        "center_zoom_half_px": zoom_half_px,
        "center_zoom_fraction": min(0.48, max(0.03, zoom_half_px / crop_size_px)),
        "center_zoom_mode": "minimum",
        "axis_image_deg": axis_image_deg,
        "coherence": coherence,
        "radius_label": radius_label,
    }


def trace_fit_center_zoom_metadata(
    metadata: dict[str, float | str],
    motion_eye: object | None,
) -> dict[str, float | str]:
    """Expand the center zoom just enough to contain the displayed traces."""
    if not isinstance(motion_eye, dict):
        return metadata

    trace_extents: list[float] = []
    for key in ("small_xy_px", "large_xy_px"):
        try:
            trace = np.asarray(motion_eye.get(key), dtype=np.float64)
        except Exception:
            continue
        if trace.ndim != 2 or trace.shape[1] != 2 or trace.size == 0:
            continue
        finite = trace[np.isfinite(trace)]
        if finite.size:
            trace_extents.append(float(np.nanmax(np.abs(finite))))
    if not trace_extents:
        return metadata

    updated = dict(metadata)
    ppd = _finite_float(updated.get("ppd"), 37.50476617)
    crop_size_px = _finite_float(updated.get("crop_size_px"), 151.0)
    min_half_px = CENTER_ZOOM_HALF_DEG * ppd
    trace_max_px = max(trace_extents)
    half_px = max(min_half_px, trace_max_px + CENTER_ZOOM_TRACE_PAD_DEG * ppd)
    updated["center_zoom_half_px"] = half_px
    updated["center_zoom_half_deg"] = half_px / ppd if ppd > 0 else CENTER_ZOOM_HALF_DEG
    updated["center_zoom_fraction"] = min(0.48, max(0.03, half_px / crop_size_px))
    updated["center_zoom_trace_max_px"] = trace_max_px
    updated["center_zoom_trace_max_deg"] = trace_max_px / ppd if ppd > 0 else float("nan")
    updated["center_zoom_mode"] = "trace_fit" if half_px > min_half_px else "minimum"
    return updated


def center_zoom_label(metadata: dict[str, float | str]) -> str:
    half_deg = _finite_float(metadata.get("center_zoom_half_deg"), CENTER_ZOOM_HALF_DEG)
    if abs(half_deg - CENTER_ZOOM_HALF_DEG) < 0.005:
        return f"central +/-{CENTER_ZOOM_HALF_DEG:g} deg"
    return f"central +/-{half_deg:.2f} deg"


def data_width_for_physical_aspect(ax: plt.Axes, height: float, width_over_height: float) -> float:
    """Convert a desired physical aspect into parent-axis data coordinates."""
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    axis_w = float(bbox.width) * float(fig_w)
    axis_h = float(bbox.height) * float(fig_h)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = abs(float(x1) - float(x0))
    yspan = abs(float(y1) - float(y0))
    if axis_w <= 0 or axis_h <= 0 or xspan <= 0 or yspan <= 0:
        return float(height) * float(width_over_height)
    return float(width_over_height) * float(height) * (axis_h / axis_w) * (xspan / yspan)


def stimulus_canvas_aspect(payload: dict | None) -> float:
    canvas = (payload or {}).get("stimulus_canvas")
    try:
        height, width = canvas.shape[:2]
        if float(height) > 0:
            return float(width) / float(height)
    except Exception:
        pass
    return 16.0 / 9.0


def _axis_vector_image(axis_image_deg: float) -> tuple[float, float]:
    if ssi_schematic is not None and hasattr(ssi_schematic, "axis_vector_image"):
        try:
            vec = ssi_schematic.axis_vector_image(float(axis_image_deg))
            return float(vec[0]), float(vec[1])
        except Exception:
            pass
    theta = math.radians(float(axis_image_deg))
    return math.cos(theta), math.sin(theta)


def add_contour_window_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Draw the real 1-deg contour-analysis aperture in crop pixel coordinates."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    norm = math.hypot(dx, dy)
    if norm <= 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / norm, dy / norm

    crop_ax.add_patch(
        patches.Circle(
            (center, center),
            radius,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.2)),
            zorder=8,
        )
    )
    local_half = radius * 0.84
    crop_ax.plot(
        [center - dx * local_half, center + dx * local_half],
        [center - dy * local_half, center + dy * local_half],
        color="white",
        lw=1.5,
        ls=CONTOUR_AXIS_DASH,
        alpha=0.95,
        solid_capstyle="round",
        zorder=9,
    )


def add_contour_axis_line_to_crop_axis(
    crop_ax: plt.Axes,
    metadata: dict[str, float | str],
    *,
    zoomed: bool = False,
) -> None:
    """Draw only the local contour axis line, useful for the zoomed crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    half = radius * 0.84
    if zoomed:
        half = min(half, _finite_float(metadata.get("center_zoom_half_px"), half) * 0.78)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    norm = math.hypot(dx, dy)
    if norm <= 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / norm, dy / norm
    crop_ax.plot(
        [center - dx * half, center + dx * half],
        [center - dy * half, center + dy * half],
        color="white",
        lw=1.5,
        ls=CONTOUR_AXIS_DASH,
        alpha=0.95,
        solid_capstyle="round",
        zorder=22,
    )


def add_unit_tuning_indicator_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Draw the example unit's preferred orientation on the local crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    direction = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    half = radius * 0.34
    xs = [center - direction[0] * half, center + direction[0] * half]
    ys = [center - direction[1] * half, center + direction[1] * half]
    crop_ax.plot(xs, ys, color="white", lw=3.0, alpha=0.80, solid_capstyle="round", zorder=15)
    crop_ax.plot(xs, ys, color=UNIT_TUNING_COLOR, lw=1.65, solid_capstyle="round", zorder=16)
    crop_ax.scatter(
        [center],
        [center],
        s=26,
        facecolor="white",
        edgecolor=UNIT_TUNING_COLOR,
        linewidth=1.1,
        zorder=17,
    )


def add_trace_path_without_marker(ax: plt.Axes, center: np.ndarray, trace_xy_px: object, color: str, *, lw: float, zorder: float) -> None:
    """Draw the eye trajectory path without the endpoint dot used upstream."""
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 2:
        return
    points = np.asarray(center, dtype=np.float64)[None, :] + trace
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="white",
        lw=float(lw) + 0.85,
        alpha=0.72,
        solid_capstyle="round",
        zorder=zorder - 0.1,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=float(lw),
        alpha=0.92,
        solid_capstyle="round",
        zorder=zorder,
    )


def add_contour_axis_labels_to_crop_axis(
    crop_ax: plt.Axes,
    metadata: dict[str, float | str],
    *,
    zoomed: bool = False,
) -> None:
    """Small in-crop labels for the contour-relative frame used by G."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    label_span = radius
    if zoomed:
        label_span = min(radius, _finite_float(metadata.get("center_zoom_half_px"), radius) * 0.82)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    tangent = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm <= 0:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    label_style = dict(
        fontsize=5.4,
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="black", edgecolor="none", alpha=0.45),
        zorder=30,
    )
    along = np.array([center, center], dtype=np.float64) - tangent * label_span * 0.56 - normal * label_span * 0.24
    across = np.array([center, center], dtype=np.float64) - normal * label_span * 0.52 + tangent * label_span * 0.22
    crop_ax.text(float(along[0]), float(along[1]), "along", **label_style)
    crop_ax.text(float(across[0]), float(across[1]), "across", **label_style)


def add_trajectory_span_arrows_to_crop_axis(
    crop_ax: plt.Axes,
    metadata: dict[str, float | str],
    motion_eye: dict | None,
) -> None:
    """Draw double-headed along/across arrows spanning the displayed trace."""
    if not isinstance(motion_eye, dict) or "large_xy_px" not in motion_eye:
        return
    trace = np.asarray(motion_eye["large_xy_px"], dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return
    trace = trace[np.all(np.isfinite(trace), axis=1)]
    if trace.shape[0] < 2:
        return

    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = np.array([0.5 * (n - 1.0), 0.5 * (n - 1.0)], dtype=np.float64)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    tangent = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm <= 0:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    along = trace @ tangent
    across = trace @ normal
    along_span = max(float(np.nanmax(along) - np.nanmin(along)), 1.5)
    across_span = max(float(np.nanmax(across) - np.nanmin(across)), 1.5)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), 18.8) * 0.82

    along_min = max(float(np.nanmin(along) - 0.08 * along_span), -half_px)
    along_max = min(float(np.nanmax(along) + 0.08 * along_span), half_px)
    along_cross = float(np.nanmedian(across) - 0.34 * across_span)
    along_cross = float(np.clip(along_cross, -half_px * 0.62, half_px * 0.62))

    across_min = max(float(np.nanmin(across) - 0.12 * across_span), -half_px)
    across_max = min(float(np.nanmax(across) + 0.12 * across_span), half_px)
    across_along = float(np.nanmax(along) + 0.34 * along_span)
    across_along = float(np.clip(across_along, -half_px * 0.62, half_px * 0.76))

    def point(a: float, c: float) -> tuple[float, float]:
        p = center + tangent * a + normal * c
        return float(p[0]), float(p[1])

    def display_angle(vector: np.ndarray) -> float:
        origin_disp = crop_ax.transData.transform(center)
        end_disp = crop_ax.transData.transform(center + vector)
        angle = math.degrees(math.atan2(end_disp[1] - origin_disp[1], end_disp[0] - origin_disp[0]))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        return angle

    def shift_up(point_xy: tuple[float, float], amount_px: float) -> tuple[float, float]:
        return (point_xy[0], point_xy[1] - amount_px)

    def shift_right(point_xy: tuple[float, float], amount_px: float) -> tuple[float, float]:
        return (point_xy[0] + amount_px, point_xy[1])

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        for color, lw, alpha, zorder in [("black", 2.7, 0.62, 27), ("white", 1.25, 0.98, 28)]:
            crop_ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, mutation_scale=8, shrinkA=0, shrinkB=0),
                alpha=alpha,
                zorder=zorder,
            )

    along_up_shift = min(half_px * 0.24, 4.6)
    across_right_shift = min(half_px * 0.16, 3.4)
    along_start = shift_up(point(along_min, along_cross), along_up_shift)
    along_end = shift_up(point(along_max, along_cross), along_up_shift)
    across_start = shift_right(point(across_along, across_min), across_right_shift)
    across_end = shift_right(point(across_along, across_max), across_right_shift)
    arrow(along_start, along_end)
    arrow(across_start, across_end)

    label_style = dict(
        fontsize=5.8,
        color="white",
        ha="center",
        va="center",
        rotation_mode="anchor",
        clip_on=False,
        zorder=31,
    )
    along_label = shift_up(point(0.5 * (along_min + along_max), along_cross - 0.15 * across_span), along_up_shift)
    crop_ax.text(*along_label, "along", rotation=display_angle(tangent), **label_style)
    crop_ax.text(
        1.075,
        0.46,
        "across",
        transform=crop_ax.transAxes,
        color=INK,
        rotation=display_angle(normal),
        **{key: value for key, value in label_style.items() if key != "color"},
    )


def restore_trace_orientation(motion_eye: dict | None) -> dict | None:
    """Undo the extra display rotation baked into the "large"/long-path
    trace by _fig4_contour_schematic.py's selected_real_panel_a_trace_pair
    (PANEL_A_LARGE_TRACE_ROTATION_DEG = 90 deg, PANEL_A_SMALL_TRACE_ROTATION_DEG
    = 0 deg) for that module's own Panel A schematic. D/G want the traces at
    their real recorded orientation, not that rotation, so rotate back by the
    same angle in reverse. Only applies when the real trace bank was used
    (large_trace_index >= 0); the synthetic fallback trace was never rotated.
    """
    if not isinstance(motion_eye, dict) or ssi_schematic is None:
        return motion_eye
    if motion_eye.get("large_trace_index", -1) is None or motion_eye.get("large_trace_index", -1) < 0:
        return motion_eye
    rotate = getattr(ssi_schematic, "rotate_trace_xy_px", None)
    if rotate is None:
        return motion_eye
    restored = dict(motion_eye)
    large_rotation_deg = getattr(ssi_schematic, "PANEL_A_LARGE_TRACE_ROTATION_DEG", 0.0)
    if large_rotation_deg and "large_xy_px" in restored:
        restored["large_xy_px"] = rotate(restored["large_xy_px"], -large_rotation_deg)
    small_rotation_deg = getattr(ssi_schematic, "PANEL_A_SMALL_TRACE_ROTATION_DEG", 0.0)
    if small_rotation_deg and "small_xy_px" in restored:
        restored["small_xy_px"] = rotate(restored["small_xy_px"], -small_rotation_deg)
    return restored


def add_center_zoom_box_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Mark the tight central zoom window on the 151 px crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    crop_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=ZOOM_BOX,
            linewidth=D_CENTER_ZOOM_BOX_LW,
            zorder=12,
        )
    )


def draw_plain_crop(
    crop_ax: plt.Axes,
    patch: object,
    *,
    trace_xy_px: object | None = None,
    trace_color: str | None = None,
) -> int:
    """Render a crop image directly instead of via ssi_schematic.add_stimulus.

    add_stimulus always draws a red+blue trace pair (or, with motion_eye=None,
    a red/blue placeholder double-arrow) with colors hardcoded to its own
    module -- there's no way to show only one trace, or to recolor it, through
    that function. This replicates just its image/border rendering, then
    optionally draws a single trace in a caller-chosen color via
    ssi_schematic.add_panel_a_trace_path directly. Returns the crop's pixel
    size (patches are square).
    """
    image = ssi_schematic.normalize_image(patch)
    n = int(image.shape[0])
    crop_ax.imshow(image, cmap="gray", interpolation="bicubic")
    crop_ax.set_aspect("equal", adjustable="box")
    hide_axis_completely(crop_ax)
    crop_ax.set_xlim(0, n - 1)
    crop_ax.set_ylim(n - 1, 0)
    crop_ax.add_patch(patches.Rectangle((0, 0), n - 1, n - 1, fill=False, lw=1.0, ec=INK))
    if trace_xy_px is not None and trace_color is not None:
        axis_center = np.array([0.5 * (n - 1), 0.5 * (n - 1)], dtype=np.float64)
        add_trace_path_without_marker(crop_ax, axis_center, trace_xy_px, trace_color, lw=1.85, zorder=4)
    return n


def add_upper_left_image_label(image_ax: plt.Axes, label: str) -> None:
    """Caption an inset image from its rendered upper-left corner."""
    x_left = float(image_ax.get_xlim()[0])
    y_top = float(image_ax.get_ylim()[1])
    image_ax.annotate(
        label,
        xy=(x_left, y_top),
        xycoords="data",
        xytext=(0, 3.6),
        textcoords="offset points",
        fontsize=D_IMAGE_LABEL_FS,
        color=GRAY,
        ha="left",
        va="bottom",
        linespacing=0.95,
        annotation_clip=False,
        clip_on=False,
        zorder=40,
    )


def add_lower_left_image_label(image_ax: plt.Axes, label: str) -> None:
    """Caption an inset image from its rendered lower-left corner."""
    x_left = float(image_ax.get_xlim()[0])
    y_bottom = float(image_ax.get_ylim()[0])
    image_ax.annotate(
        label,
        xy=(x_left, y_bottom),
        xycoords="data",
        xytext=(0, -3.6),
        textcoords="offset points",
        fontsize=D_IMAGE_LABEL_FS,
        color=GRAY,
        ha="left",
        va="top",
        linespacing=0.95,
        annotation_clip=False,
        clip_on=False,
        zorder=40,
    )


def add_zoomed_crop_view(
    zoom_ax: plt.Axes,
    schematic_payload: dict,
    motion_eye: dict | None,
    metadata: dict[str, float | str],
    *,
    trace_color: str = TRACE_COLOR,
    border_color: str = ZOOM_BOX,
    border_lw: float = CROP_BORDER_LW,
) -> None:
    """Draw the crop again, zoomed to the central +/-0.25 deg window."""
    if ssi_schematic is None:
        raise RuntimeError("SSI schematic helpers are unavailable")
    trace_xy_px = motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None
    draw_plain_crop(zoom_ax, schematic_payload.get("patch"), trace_xy_px=trace_xy_px, trace_color=trace_color)
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    zoom_ax.set_xlim(center - half_px, center + half_px)
    zoom_ax.set_ylim(center + half_px, center - half_px)
    zoom_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=border_color,
            linewidth=border_lw,
            zorder=20,
        )
    )


def add_center_zoom_parent_overlay(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    """Fallback/connector geometry for the tight central zoom window."""
    center_x = crop_x + 0.5 * crop_w
    center_y = crop_y + 0.5 * crop_h
    half_fraction = _finite_float(metadata.get("center_zoom_fraction"), CENTER_ZOOM_HALF_DEG * 37.50476617 / 151.0)
    ax.add_patch(
        patches.Rectangle(
            (center_x - crop_w * half_fraction, center_y - crop_h * half_fraction),
            2.0 * crop_w * half_fraction,
            2.0 * crop_h * half_fraction,
            fill=False,
            edgecolor=ZOOM_BOX,
            linewidth=0.75,
            zorder=24,
        )
    )


def add_zoom_connectors(
    ax: plt.Axes,
    source_box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
    *,
    color: str = CYAN,
) -> None:
    """Connect the visible source zoom box to the next crop stage."""
    source_x, source_y, source_w, source_h = source_box
    target_x, target_y, _target_w, target_h = target_box
    ax.plot(
        [source_x + source_w, target_x],
        [source_y + source_h, target_y + target_h],
        color=color,
        lw=D_CONNECTOR_LW,
        ls=(0, (3, 3)),
        alpha=0.58,
        zorder=15,
    )
    ax.plot(
        [source_x + source_w, target_x],
        [source_y, target_y],
        color=color,
        lw=D_CONNECTOR_LW,
        ls=(0, (3, 3)),
        alpha=0.58,
        zorder=15,
    )


def add_roi_to_crop_connectors(
    ax: plt.Axes,
    full_ax: plt.Axes,
    crop_ax: plt.Axes,
    crop_center_xy: object,
    crop_size_px: object,
    *,
    color: str = CYAN,
) -> None:
    """Connect the source-image ROI square to the rendered crop border."""
    center = np.asarray(crop_center_xy, dtype=np.float64).reshape(-1)
    if center.size < 2:
        return
    size = _finite_float(crop_size_px, np.nan)
    if not np.isfinite(size) or size <= 0:
        return

    cx, cy = float(center[0]), float(center[1])
    right = cx + size / 2.0
    top = cy - size / 2.0
    bottom = cy + size / 2.0

    crop_x_left = float(crop_ax.get_xlim()[0])
    crop_y_bottom, crop_y_top = [float(v) for v in crop_ax.get_ylim()]
    endpoint_pairs = [
        ((right, top), (crop_x_left, crop_y_top)),
        ((right, bottom), (crop_x_left, crop_y_bottom)),
    ]
    for source_xy, target_xy in endpoint_pairs:
        connector = patches.ConnectionPatch(
            xyA=source_xy,
            xyB=target_xy,
            coordsA="data",
            coordsB="data",
            axesA=full_ax,
            axesB=crop_ax,
            arrowstyle="-",
            color=color,
            lw=D_CONNECTOR_LW,
            ls=(0, (3, 3)),
            alpha=0.58,
            clip_on=False,
            zorder=3,
        )
        ax.add_artist(connector)


def add_center_zoom_to_zoom_connectors(
    ax: plt.Axes,
    crop_ax: plt.Axes,
    zoom_ax: plt.Axes,
    metadata: dict[str, float | str],
    *,
    color: str = ZOOM_BOX,
) -> None:
    """Connect the center zoom box to the rendered zoomed-crop border."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    if not np.isfinite(half_px) or half_px <= 0:
        return

    source_right = center + half_px
    source_top = center - half_px
    source_bottom = center + half_px
    target_left = float(zoom_ax.get_xlim()[0])
    target_bottom, target_top = [float(v) for v in zoom_ax.get_ylim()]
    endpoint_pairs = [
        ((source_right, source_top), (target_left, target_top)),
        ((source_right, source_bottom), (target_left, target_bottom)),
    ]
    for source_xy, target_xy in endpoint_pairs:
        connector = patches.ConnectionPatch(
            xyA=source_xy,
            xyB=target_xy,
            coordsA="data",
            coordsB="data",
            axesA=crop_ax,
            axesB=zoom_ax,
            arrowstyle="-",
            color=color,
            lw=D_CONNECTOR_LW,
            ls=(0, (3, 3)),
            alpha=0.70,
            clip_on=False,
            zorder=7,
        )
        ax.add_artist(connector)


def add_contour_window_parent_overlay(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    """Fallback aperture overlay when the real crop inset is unavailable."""
    center_x = crop_x + 0.5 * crop_w
    center_y = crop_y + 0.5 * crop_h
    radius_fraction = _finite_float(metadata.get("radius_fraction"), 38.0 / 151.0)
    radius_x = crop_w * radius_fraction
    radius_y = crop_h * radius_fraction
    theta = math.radians(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    dx = math.cos(theta)
    dy = -math.sin(theta)
    line_scale = min(0.46 / max(abs(dx), 1e-6), 0.46 / max(abs(dy), 1e-6))
    local_scale = radius_fraction * 0.84

    ax.plot(
        [crop_x + crop_w * (0.5 - dx * line_scale), crop_x + crop_w * (0.5 + dx * line_scale)],
        [crop_y + crop_h * (0.5 - dy * line_scale), crop_y + crop_h * (0.5 + dy * line_scale)],
        color=CYAN,
        lw=0.95,
        ls=(0, (4, 3)),
        zorder=18,
    )
    ax.add_patch(
        patches.Ellipse(
            (center_x, center_y),
            width=2.0 * radius_x,
            height=2.0 * radius_y,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=0.9,
            linestyle=(0, (3.0, 2.0)),
            zorder=19,
        )
    )
    ax.plot(
        [crop_x + crop_w * (0.5 - dx * local_scale), crop_x + crop_w * (0.5 + dx * local_scale)],
        [crop_y + crop_h * (0.5 - dy * local_scale), crop_y + crop_h * (0.5 + dy * local_scale)],
        color=CYAN,
        lw=1.3,
        solid_capstyle="round",
        zorder=20,
    )
    ax.plot([center_x, center_x + radius_x], [center_y, center_y], color=CONTOUR_WINDOW, lw=0.65, zorder=21)
    ax.scatter([center_x], [center_y], s=15, facecolor="white", edgecolor=INK, linewidth=0.65, zorder=22)


def add_contour_window_callout(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    radius_fraction = _finite_float(metadata.get("radius_fraction"), 38.0 / 151.0)
    xy = (crop_x + 0.50 * crop_w, crop_y + (0.50 + radius_fraction * 0.82) * crop_h)
    text_xy = (crop_x + 0.52 * crop_w, crop_y + crop_h + 0.040)
    ax.annotate(
        f"local contour estimate\n{metadata['radius_label']}; axis + coherence",
        xy=xy,
        xytext=text_xy,
        xycoords="data",
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#343A40",
        linespacing=1.08,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.82),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color=CONTOUR_WINDOW, mutation_scale=7, shrinkA=2, shrinkB=2),
        zorder=30,
    )


def schematic_response_maps(
    payload: dict | None,
) -> tuple[dict[str, object], tuple[float, float] | None, dict[str, float]]:
    """Real/stabilized response maps for panel A, plus the scalar SSI (bits
    per spike) for each -- schematic_rr100_final_map_unit_metrics.csv's
    real_final_map_ssi_bits_per_spike / stable_final_map_ssi_bits_per_spike
    columns for whichever unit choose_right_panel_real_unit selected. SSI is
    one number per map, not a per-pixel quantity -- the map's own pixels are
    the model's predicted firing rate (spikes/s), mean-centered for display.
    """
    if ssi_schematic is None or payload is None:
        return {}, None, {}
    maps = payload.get("schematic_rr100_final_maps")
    condition_ids = payload.get("schematic_rr100_final_condition_id")
    unit_row = ssi_schematic.choose_right_panel_real_unit(payload)
    if maps is None or condition_ids is None or unit_row is None:
        return {}, None, {}
    try:
        ids = [str(x) for x in condition_ids]
        real_idx = ids.index("real_trace_final")
        stable_idx = ids.index("endpoint_stabilized_final")
        unit_idx = int(unit_row["unit_index"])
        real_maps = {
            "fem": maps[real_idx, unit_idx],
            "stable": maps[stable_idx, unit_idx],
        }
        ssi_bits_per_spike = {
            "fem": _finite_float(unit_row.get("real_final_map_ssi_bits_per_spike"), float("nan")),
            "stable": _finite_float(unit_row.get("stable_final_map_ssi_bits_per_spike"), float("nan")),
        }
        return real_maps, ssi_schematic.panel_b_map_pair_limits(real_maps.values()), ssi_bits_per_spike
    except Exception:
        return {}, None, {}


def shared_story_ylim(frame: pd.DataFrame, *, fallback: tuple[float, float]) -> tuple[float, float]:
    if frame.empty or "ssi_percent_vs_cell_baseline" not in frame:
        return fallback
    return story_panels._shared_ylim([frame["ssi_percent_vs_cell_baseline"]])


def first_int(frame: pd.DataFrame, column: str) -> int | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.iloc[0]) if not values.empty else None


def add_support_note(ax: plt.Axes, frame: pd.DataFrame) -> None:
    n_units = first_int(frame, "n_selected_units")
    n_pairs = first_int(frame, "n_selected_unit_image_pairs")
    if n_units is None or n_pairs is None:
        return
    ax.text(
        0.985,
        0.930,
        f"{n_units} units\n{n_pairs} pairs",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=GRAY,
        linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )


def hide_axis_completely(ax: plt.Axes) -> None:
    """``ax.set_axis_off()`` hides ticks/spines visually, but the default
    tick-label Text artists it leaves behind (e.g. an untouched 0-1 axes
    still carries '0.0'..'1.0' tick labels) keep reporting real bounding
    boxes to ``fig.get_tightbbox()`` -- matplotlib's axis-off draw path
    skips *drawing* them but doesn't check visibility when measuring tight
    bbox. In this file's v3 per-panel builds that made ``bbox_inches="tight"``
    silently grow a panel's saved page well past its intended size (e.g.
    panel D grew 0.51in taller from an invisible x-axis sitting below its
    own y=0), pushing it into whichever neighbor happened to be composited
    on top. Clearing the ticks outright (not just hiding the axis) removes
    the phantom Text artists so nothing is left to measure.
    """
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])


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


def draw_spread_example(
    example_ax: plt.Axes,
    *,
    broad_across: bool,
    color: str = EYE_TRAJECTORY_COLOR,
) -> None:
    hide_axis_completely(example_ax)
    example_ax.set_xlim(-1.0, 1.0)
    example_ax.set_ylim(-0.72, 0.72)
    example_ax.plot([-0.88, 0.88], [0.0, 0.0], color=GRAY, lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=1)
    t = np.linspace(0.0, 1.0, 120)
    x = -0.80 + 1.60 * t
    if broad_across:
        y = 0.33 * np.sin(2.0 * np.pi * t - 0.45) + 0.16 * np.sin(5.0 * np.pi * t + 0.25)
    else:
        y = 0.055 * np.sin(4.0 * np.pi * t + 0.20) + 0.025 * np.sin(11.0 * np.pi * t)
    example_ax.plot(x, y, color=color, lw=1.45, zorder=3)
    example_ax.scatter([x[0]], [y[0]], s=15, facecolor="white", edgecolor=INK, linewidth=0.65, zorder=4)
    example_ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-7], y[-7]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=8),
        zorder=5,
    )


def draw_d_trajectory_spread_explainer(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
) -> None:
    """Lower Panel-D key: the geometry G used to explain before the H plot."""
    legend_w = min(0.20, w * 0.28)
    legend_x = x0
    legend_y_top = y0 + h * 0.78
    sample_x = legend_x + 0.008
    text_x = legend_x + 0.075

    ax.plot([sample_x, sample_x + 0.055], [legend_y_top, legend_y_top], color=GRAY, lw=1.0, ls=(0, (4, 3)))
    ax.text(text_x, legend_y_top, "image axis", fontsize=6.0, color=INK, ha="left", va="center")

    y_unit = legend_y_top - h * 0.25
    ax.plot([sample_x, sample_x + 0.055], [y_unit, y_unit], color=UNIT_TUNING_COLOR, lw=1.45)
    ax.scatter([sample_x + 0.028], [y_unit], s=18, facecolor="white", edgecolor=UNIT_TUNING_COLOR, linewidth=1.0, zorder=4)
    ax.text(text_x, y_unit + 0.012, "unit tuning", fontsize=6.0, color=INK, ha="left", va="center")
    ax.text(text_x, y_unit - 0.035, "pref. orientation", fontsize=5.1, color=GRAY, ha="left", va="center")

    y_eye = legend_y_top - h * 0.55
    eye_x = np.array([sample_x, sample_x + 0.018, sample_x + 0.035, sample_x + 0.055])
    eye_y = np.array([y_eye, y_eye + 0.018, y_eye - 0.006, y_eye + 0.010])
    ax.plot(eye_x, eye_y, color=EYE_TRAJECTORY_COLOR, lw=1.45)
    ax.scatter([eye_x[0]], [eye_y[0]], s=18, facecolor=EYE_TRAJECTORY_COLOR, edgecolor=EYE_TRAJECTORY_COLOR, zorder=4)
    ax.annotate(
        "",
        xy=(eye_x[-1], eye_y[-1]),
        xytext=(eye_x[-2], eye_y[-2]),
        arrowprops=dict(arrowstyle="-|>", color=EYE_TRAJECTORY_COLOR, lw=1.0, mutation_scale=8),
    )
    ax.text(text_x, y_eye, "eye trajectory", fontsize=6.0, color=INK, ha="left", va="center")

    box_x = x0 + legend_w + 0.030
    box_y = y0 + h * 0.03
    box_w = max(0.10, x0 + w - box_x)
    box_h = h * 0.92
    ax.add_patch(
        patches.FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            facecolor="none",
            edgecolor=GRAY,
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            alpha=0.85,
            zorder=0,
        )
    )
    ax.text(
        box_x + box_w / 2,
        box_y + box_h * 0.88,
        "same path length,\ndifferent spread",
        fontsize=5.8,
        color=INK,
        ha="center",
        va="center",
        linespacing=0.95,
    )
    col_w = box_w * 0.41
    left_x = box_x + box_w * 0.08
    right_x = box_x + box_w * 0.52
    label_y = box_y + box_h * 0.70
    for label_x, heading, subheading in [
        (left_x + col_w / 2, "mostly\nalong", "low across spread"),
        (right_x + col_w / 2, "broad\nacross", "high across spread"),
    ]:
        ax.text(
            label_x,
            label_y,
            heading,
            fontsize=5.7,
            color=EYE_TRAJECTORY_COLOR,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=0.90,
        )
        ax.text(label_x, label_y - box_h * 0.15, subheading, fontsize=5.0, color=INK, ha="center", va="center")

    ex_y = box_y + box_h * 0.21
    ex_h = box_h * 0.31
    left_ax = ax.inset_axes([left_x, ex_y, col_w, ex_h], transform=ax.transData)
    draw_spread_example(left_ax, broad_across=False)
    right_ax = ax.inset_axes([right_x, ex_y, col_w, ex_h], transform=ax.transData)
    draw_spread_example(right_ax, broad_across=True)
    ax.text(left_x + col_w / 2, box_y + box_h * 0.10, "path length = x", fontsize=5.1, color=INK, ha="center")
    ax.text(right_x + col_w / 2, box_y + box_h * 0.10, "path length = x", fontsize=5.1, color=INK, ha="center")


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


def draw_rms_excursion_explainer(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    motion_eye: dict | None,
    axis_image_deg: float,
) -> bool:
    """Decompose G's own example trace into across-/along-contour components,
    the same split H's dose curve reports as separate lines. Real geometry
    from this trace (not H's population statistic) -- illustrative, not a
    stand-in for H's actual numbers.
    """
    if not isinstance(motion_eye, dict) or "large_xy_px" not in motion_eye:
        return False
    diagram_ax = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    hide_axis_completely(diagram_ax)

    dx, dy = _axis_vector_image(axis_image_deg)
    norm = math.hypot(dx, dy)
    tangent = np.array([dx, dy]) / norm if norm > 0 else np.array([1.0, 0.0])
    normal = np.array([-tangent[1], tangent[0]])
    trace_px = np.asarray(motion_eye["large_xy_px"], dtype=np.float64)
    along_px = trace_px @ tangent
    across_px = trace_px @ normal
    along_px = along_px - along_px.mean()
    across_px = across_px - across_px.mean()
    ppd = _finite_float(getattr(ssi_schematic, "MODEL_PPD", None), 37.50476617)
    along_arcmin = along_px / ppd * 60.0
    across_arcmin = across_px / ppd * 60.0
    rms_along = float(np.sqrt(np.mean(along_arcmin**2)))
    rms_across = float(np.sqrt(np.mean(across_arcmin**2)))

    half_span = max(float(np.abs(along_arcmin).max()), float(np.abs(across_arcmin).max()), rms_along, rms_across) * 1.5
    diagram_ax.set_xlim(-half_span, half_span)
    diagram_ax.set_ylim(-half_span * 1.18, half_span)
    # Not aspect="equal": the panel is much wider than tall, and forcing
    # equal data/physical scaling squeezes the along-axis (x) bracket down
    # to a sliver -- its arrowheads then overlap into an illegible blob.
    # This is a schematic, not a ruler, so a stretched trace is an
    # acceptable trade for both brackets rendering at a legible size.
    diagram_ax.axhline(0.0, color=CYAN, lw=1.3, ls=(0, (4, 3)), alpha=0.85, zorder=2)
    diagram_ax.plot(along_arcmin, across_arcmin, color=EYE_TRAJECTORY_COLOR, lw=1.3, zorder=3)
    diagram_ax.scatter([0], [0], s=14, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=4)

    bracket_y = -half_span * 1.08
    diagram_ax.annotate(
        "",
        xy=(rms_along, bracket_y),
        xytext=(-rms_along, bracket_y),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK),
    )
    diagram_ax.text(
        0.0, bracket_y - half_span * 0.10, f"along: {rms_along:.1f}'", ha="center", va="top", fontsize=5.8, color=INK
    )

    bracket_x = half_span * 1.06
    diagram_ax.annotate(
        "",
        xy=(bracket_x, rms_across),
        xytext=(bracket_x, -rms_across),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK),
    )
    diagram_ax.text(
        bracket_x + half_span * 0.10,
        0.0,
        f"across: {rms_across:.1f}'",
        ha="left",
        va="center",
        fontsize=5.8,
        color=INK,
        rotation=90,
    )
    return True


def panel_g_default_layout_boxes(
    ax: plt.Axes, schematic_payload: dict | None = None
) -> dict[str, tuple[float, float, float, float]]:
    """Default Panel G block-layout geometry, as named boxes ``(x, y, w, h)``
    -- the panel_d_default_layout_boxes() counterpart for G. crop/zoom's
    widths derive from height via data_width_for_physical_aspect, hence the
    ``ax`` argument (see that function's own docstring).
    """
    crop_x, crop_y, crop_h = 0.060, 0.560, 0.320
    crop_w = data_width_for_physical_aspect(ax, crop_h, 1.0)
    zoom_h = 0.857 * crop_h  # matches D's original zoom_h / crop_h ratio (0.330 / 0.385)
    zoom_w = data_width_for_physical_aspect(ax, zoom_h, 1.0)
    zoom_x = crop_x + crop_w * (1.0 - 0.262)  # matches D's original (crop_w - 0.055) / crop_w overlap fraction
    zoom_y = crop_y - 0.117 * crop_h  # matches D's original (crop_y - zoom_y) / crop_h offset fraction
    return {
        "crop": (crop_x, crop_y, crop_w, crop_h),
        "zoom": (zoom_x, zoom_y, zoom_w, zoom_h),
        "rms_explainer": (0.14, 0.145, 0.72, 0.180),
    }


def draw_contour_components_panel(
    ax: plt.Axes,
    *,
    label: str = "G",
    title: str = "Effects of FEMs depend\non both content and tuning",
    schematic_payload: dict | None = None,
    layout_overrides: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Reference crop plus the zoomed local-contour aperture, moved from D.

    D's cascade used to be full stimulus -> 151x151 crop -> zoomed detail,
    all three crowded into one row and competing with D's E/F insets for
    width. The zoomed detail now lives here instead; a copy of the 151x151
    crop comes along as a reference (D keeps its own, independent copy) so
    the zoom isn't shown floating with no visual anchor. The crop/zoom
    overlap is kept proportionally the same as it was in D -- as a fraction
    of the crop's own size, since this axes has a different physical aspect
    than D's (tall and narrow here vs. D's short and wide).
    """
    hide_axis_completely(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    set_panel_title(ax, label, title, fontsize=7.0, pad=2, linespacing=1.05)

    boxes = {**panel_g_default_layout_boxes(ax, schematic_payload), **(layout_overrides or {})}
    crop_x, crop_y, crop_w, crop_h = boxes["crop"]
    zoom_x, zoom_y, zoom_w, zoom_h = boxes["zoom"]

    window_metadata = contour_window_metadata(schematic_payload)
    axis_image_deg = _finite_float((schematic_payload or {}).get("contour_axis_image_deg"), 10.352312)
    motion_eye: dict | None = None
    has_real_schematic = False
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            crop_ax = ax.inset_axes([crop_x, crop_y, crop_w, crop_h], transform=ax.transData)
            crop_ax.set_zorder(4)
            synthetic_left = ssi_schematic.make_synthetic_left_side(
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
            )
            motion_eye = restore_trace_orientation(synthetic_left.get("eye"))
            window_metadata = trace_fit_center_zoom_metadata(window_metadata, motion_eye)
            # Only the long-path trace, in TRACE_COLOR (not add_stimulus's
            # hardcoded red+blue pair) -- see draw_plain_crop.
            draw_plain_crop(
                crop_ax,
                schematic_payload.get("patch"),
                trace_xy_px=motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None,
                trace_color=TRACE_COLOR,
            )
            crop_ax.set_anchor("NW")
            crop_box_edge = getattr(ssi_schematic, "FIG3_CYAN", CYAN)
            crop_x1, crop_y1 = crop_ax.get_xlim()[1], crop_ax.get_ylim()[0]
            crop_ax.add_patch(
                patches.Rectangle(
                    (0, 0), crop_x1, crop_y1, fill=False, edgecolor=crop_box_edge, linewidth=CROP_BORDER_LW, zorder=14
                )
            )
            add_contour_window_to_crop_axis(crop_ax, window_metadata)
            add_center_zoom_box_to_crop_axis(crop_ax, window_metadata)

            zoom_ax = ax.inset_axes([zoom_x, zoom_y, zoom_w, zoom_h], transform=ax.transData)
            zoom_ax.set_zorder(6)
            add_zoomed_crop_view(zoom_ax, schematic_payload, motion_eye, window_metadata)
            zoom_ax.set_anchor("NW")
            has_real_schematic = True
        except Exception:
            has_real_schematic = False

    if not has_real_schematic:
        placeholder_box(
            ax, crop_x, crop_y, crop_w, crop_h, "model window", sublabel="151 x 151 crop\n(reference)", hatch="///", label_size=6.4
        )
        placeholder_box(
            ax, zoom_x, zoom_y, zoom_w, zoom_h, "trace zoom", sublabel=center_zoom_label(window_metadata), hatch="\\\\\\", label_size=6.0
        )
        add_contour_window_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
        add_center_zoom_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
    else:
        ax.text(
            crop_x + 0.34 * crop_w, crop_y - 0.028, "151 x 151 crop", fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top"
        )
        ax.text(
            zoom_x + zoom_w / 2, zoom_y - 0.030, center_zoom_label(window_metadata), fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top"
        )

    zoom_fraction = _finite_float(window_metadata.get("center_zoom_fraction"), CENTER_ZOOM_HALF_DEG * 37.50476617 / 151.0)
    zoom_source_box = (
        crop_x + crop_w * (0.5 - zoom_fraction),
        crop_y + crop_h * (0.5 - zoom_fraction),
        2.0 * crop_w * zoom_fraction,
        2.0 * crop_h * zoom_fraction,
    )
    add_zoom_connectors(ax, zoom_source_box, (zoom_x, zoom_y, zoom_w, zoom_h), color=ZOOM_BOX)

    ax.text(
        0.02,
        0.40,
        "Reference crop + zoomed aperture from D.",
        fontsize=6.2,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.15,
    )

    # Lower half: decompose this same trace into the across-/along-contour
    # components H's dose curve reports as separate lines, so a reader sees
    # what "across" and "along" mean geometrically before H uses them.
    ax.text(0.02, 0.335, "RMS excursion, by direction", fontsize=7.0, color=GRAY, ha="left")
    rms_x, rms_y, rms_w, rms_h = boxes["rms_explainer"]
    drew_explainer = draw_rms_excursion_explainer(
        ax, x0=rms_x, y0=rms_y, w=rms_w, h=rms_h, motion_eye=motion_eye, axis_image_deg=axis_image_deg
    )
    if not drew_explainer:
        ax.text(
            0.50,
            0.18,
            "trace decomposition\nunavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            color=GRAY,
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


def draw_story_path_panel(
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
) -> None:
    frame = pd.DataFrame()
    if not panel_b.empty:
        frame = panel_b[panel_b["sf_group"].eq(sf_group) & panel_b["relation"].eq(relation)].copy()

    if frame.empty:
        format_placeholder_plot(
            ax,
            label=label,
            title=title,
            color=color,
            ylabel=ylabel,
            xlabel="path length (arcmin)",
            ylim=ylim,
        )
        return

    story_panels._plot_b_series(ax, frame, color=color)
    story_panels._format_broken_axis(
        ax,
        ticks=story_panels.B_TICKS,
        min_pos=story_panels.B_MIN_POS,
        max_pos=story_panels.B_MAX_POS,
        xlabel="path length (arcmin)",
    )
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*ylim)
    set_panel_title(ax, label, title, color=color)
    if ylabel:
        ax.set_ylabel(ylabel)
    add_support_note(ax, frame)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)


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
    if panel_bd_path_bins is not None and not panel_b.empty:
        try:
            panel_bd_path_bins.draw_panel(
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


def draw_story_component_panel(
    ax: plt.Axes,
    component: pd.DataFrame,
    *,
    label: str = "H",
    relation: str = "contour_matched",
    color: str = ORANGE,
    ylim: tuple[float, float] = (-32, 24),
) -> None:
    frame = pd.DataFrame()
    if not component.empty:
        frame = component[component["relation"].eq(relation)].copy()

    if frame.empty:
        format_placeholder_plot(
            ax,
            label=label,
            title="Aligned high-SF components",
            color=INK,
            ylabel="SSI change (%)",
            xlabel="component path (arcmin)",
            ylim=ylim,
        )
        ax.text(
            0.05,
            0.08,
            "reserve: across-contour and\nalong-contour series",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
            color=GRAY,
        )
        return

    story_panels._plot_component_series(ax, frame, color=color)
    story_panels._format_broken_axis(
        ax,
        ticks=story_panels.LOWER_TICKS,
        min_pos=story_panels.LOWER_MIN_POS,
        max_pos=story_panels.LOWER_MAX_POS,
        xlabel="component path (arcmin)",
    )
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*ylim)
    set_panel_title(ax, label, "Aligned high-SF components", color=INK, fontsize=8.3)
    ax.set_ylabel("SSI change (%)")
    add_support_note(ax, frame)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in item else "along" for item in labels]
        ax.legend(
            handles,
            short_labels,
            frameon=False,
            fontsize=6.1,
            loc="lower left",
            handlelength=1.8,
            borderaxespad=0.2,
        )
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)


def draw_polar_placeholder(ax: plt.Axes) -> None:
    set_panel_title(ax, "H", "FEM paths align with local contours", fontsize=9.0, pad=10)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks([0, 1.5708, 3.1416, 4.7124])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel", "orthogonal"], fontsize=7)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["1", "2"], fontsize=7)
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, color="#D8DEE5", lw=0.8)
    ax.spines["polar"].set_linewidth(1.1)
    ax.text(
        0.5,
        0.5,
        "polar distribution\nplaceholder",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )


def draw_path_alignment_placeholder(ax: plt.Axes, *, error: Exception | None = None) -> None:
    set_panel_title(ax, "I", "Real FEMs are anisotropic\nnear local contours", fontsize=7.8, pad=5)
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(1.65, 2.55)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ax.set_xlabel("angle from local edge")
    ax.set_ylabel("position spread RMS (arcmin)")
    message = "unwrapped profile\nplaceholder"
    if error is not None:
        message = "panel generation\nfailed; placeholder"
    ax.text(
        0.50,
        0.54,
        message,
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


def draw_edge_alignment_placeholder(ax: plt.Axes, *, error: Exception | None = None) -> None:
    set_panel_title(
        ax,
        "J",
        "Real (vs. randomly rotated)\nFEM trajectories preferentially\nbenefit aligned high-SF units",
        fontsize=7.4,
        pad=5,
    )
    ax.axhline(0.0, color="0.35", lw=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("observed - random rotated (pp SSI)")
    message = "coherence-binned\nresult placeholder"
    if error is not None:
        message = "panel generation\nfailed; placeholder"
    ax.text(
        0.50,
        0.54,
        message,
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


def draw_generated_panel_or_placeholder(
    ax: plt.Axes,
    module: object | None,
    fallback,
) -> None:
    if module is None:
        fallback(ax, error=PANEL_IMPORT_ERROR)
        return
    try:
        module.draw_panel(ax)
    except Exception as exc:
        ax.clear()
        fallback(ax, error=exc)


def draw_panel_g_or_fallback(
    ax: plt.Axes,
    component_values: pd.DataFrame,
    *,
    component_ylim: tuple[float, float],
) -> None:
    if panel_e_rms_excursion is not None:
        try:
            panel_e_rms_excursion.draw_panel(ax)
            return
        except Exception:
            ax.clear()
    draw_story_component_panel(ax, pd.DataFrame(), ylim=component_ylim)


def draw_overall_title(fig: plt.Figure, *, has_existing_panels: bool) -> None:
    fig.text(
        0.50,
        0.986,
        "Single-spike information and contour-relative FEM",
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )


def build_figure(out_dir: Path) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_b_values, component_values = read_existing_story_values()
    schematic_payload = read_schematic_payload()
    has_existing_panels = not panel_b_values.empty and panel_e_rms_excursion is not None
    if panel_bd_path_bins is not None and not panel_b_values.empty:
        bc_ylim = panel_bd_path_bins.shared_ylim_for(panel_b_values, relations=("strong_contours_no_osi",))
        ef_ylim = panel_bd_path_bins.shared_ylim_for(panel_b_values, relations=("contour_matched",))
    else:
        bc_ylim = shared_story_ylim(panel_b_values, fallback=(-20, 48))
        ef_ylim = bc_ylim
    aligned_component = (
        component_values[component_values["relation"].eq("contour_matched")].copy()
        if not component_values.empty
        else pd.DataFrame()
    )
    component_ylim = shared_story_ylim(aligned_component, fallback=(-32, 24))

    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    draw_overall_title(fig, has_existing_panels=has_existing_panels)
    gs = fig.add_gridspec(3, 3, **MAIN_GRID_KWARGS)

    ax_b = _wide_panel_axes(fig, gs, 0)
    draw_panel_b(ax_b, schematic_payload=schematic_payload)

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    ax_b1 = fig.add_subplot(gs_b_right[0, 0])
    _shrink_axes_center(ax_b1)
    draw_panel_bcef_or_placeholder(
        ax_b1,
        panel_b_values,
        label="B",
        title="Low-SF units",
        sf_group="low_lt0p5",
        relation="strong_contours_no_osi",
        color=BLUE,
        ylabel="SSI change (%)",
        ylim=bc_ylim,
        show_microsaccade_legend=True,
    )
    ax_b2 = fig.add_subplot(gs_b_right[1, 0])
    _shrink_axes_center(ax_b2)
    draw_panel_bcef_or_placeholder(
        ax_b2,
        panel_b_values,
        label="C",
        title="High-SF units",
        sf_group="high_ge0p75",
        relation="strong_contours_no_osi",
        color=ORANGE,
        ylabel="SSI change (%)",
        ylim=bc_ylim,
    )

    ax_a = _wide_panel_axes(fig, gs, 1)
    draw_panel_a(ax_a, schematic_payload=schematic_payload, panel_b_values=panel_b_values, ef_ylim=ef_ylim)

    ax_g_new = fig.add_subplot(gs[1, 2])
    draw_contour_components_panel(ax_g_new, schematic_payload=schematic_payload)

    ax_c1 = fig.add_subplot(gs[2, 0])
    draw_panel_g_or_fallback(ax_c1, component_values, component_ylim=component_ylim)
    ax_c2 = fig.add_subplot(gs[2, 1])
    draw_generated_panel_or_placeholder(ax_c2, panel_f_unwrapped_edge_coherence, draw_path_alignment_placeholder)
    ax_c3 = fig.add_subplot(gs[2, 2])
    draw_generated_panel_or_placeholder(ax_c3, panel_i_match_advantage, draw_edge_alignment_placeholder)

    paths = {
        "png": out_dir / "ssi_figure_v2.png",
        "pdf": out_dir / "ssi_figure_v2.pdf",
        "svg": out_dir / "ssi_figure_v2.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    paths["panel_boxes_svg"] = write_panel_boxes_svg(out_dir)
    provenance_path = out_dir / "ssi_figure_v2_methods_provenance.json"
    provenance_path.write_text(
        json.dumps(collect_methods_provenance(out_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["methods_provenance_json"] = provenance_path
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the draft SSI figure v2 scaffold.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for rendered figure files.")
    args = parser.parse_args()
    paths = build_figure(args.out_dir)
    for key in ("png", "pdf", "svg", "panel_boxes_svg", "methods_provenance_json"):
        print(paths[key])


if __name__ == "__main__":
    main()
