#!/usr/bin/env python3
"""Standalone build for Panel A (motion/model schematic + response maps).

v3 architecture: every panel is its own independently-rendered figure,
composited onto the final page at a measured position (see
compose_ssi_figure_v3.py) instead of being hand-placed via nested
inset_axes inside one shared Figure -- that nesting is what caused this
session's recurring collision bugs (a neighboring panel's opaque axes
background painting over this panel's text). The actual drawing logic is
unchanged and still lives in _fig4_ssi_common.draw_panel_b; this
wrapper just gives it its own canvas, sized from the measured reference
layout, and its own save step.

The single-unit-readout network icon is the one exception: instead of
letting draw_panel_b draw its own matplotlib reproduction (busy criss-cross
connection lines, no cube->network entry arrow -- see
panels/_fig4_network_icon.py's docstring), this build stamps in
the actual vector icon cropped from ssi_figure_v2_3.pdf's Panel A, scaled
to fit the same slot draw_panel_b would have used. The movie cubes and
response maps stay matplotlib-drawn since they carry real per-run data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_ssi_common as figure  # noqa: E402

import _fig4_panel_header  # noqa: E402
import _fig4_layout as layout  # noqa: E402
from _fig4_network_icon import OUT_PDF as NETWORK_ICON_PDF, extract_network_icon  # noqa: E402

OUT_DIR = _paths.PANELS_DIR
DEFAULT_FIGSIZE = layout.PANEL_BOXES["A"][2:4]

# Written by panels/panel_a_layout_boxes.py's import step after a human
# drags/resizes the exported box template; read back here so build_panel()
# picks up manual layout edits without any code change.
LAYOUT_OVERRIDES_JSON = _paths.PANEL_A_LAYOUT_OVERRIDES_JSON

# draw_panel_b's own ax.set_xlim/ylim is exactly (0, 1) and its header/
# labels are positioned in ax.transAxes, some of them (e.g. the panel
# letter+title, at y=1.010) deliberately just past that top edge. Previously
# bbox_inches="tight" grew the saved PDF's page to include that overflow,
# but a variable-size page defeats the point-mapping the icon-stamping step
# below needs. Instead, ax is placed inside the figure with a margin on
# each side (AX_BOX, figure-fraction) so transAxes' own [0, 1] box already
# has room for that overflow within a *fixed*, known figsize page -- no
# tight-bbox cropping, no post-hoc offset to solve for.
AX_BOX = (0.006, 0.006, 0.988, 0.925)  # left, bottom, width, height (figure-fraction)
HEADER_Y = 1.033
TITLE_Y_OFFSET = -0.0146
TITLE_Y_OFFSET_PT = _fig4_panel_header.TOP_ROW_TITLE_Y_OFFSET_PT
OVERRIDE_CONTENT_Y_SHIFT = 0.049


def data_frac_to_page_pt(x_frac: float, y_frac: float, figsize: tuple[float, float]) -> tuple[float, float]:
    """draw_panel_b's 0..1 axes-fraction -> this panel's saved-PDF points
    (bottom-left origin, matching pypdf/PDF convention). AX_BOX places the
    axes inside the figure with a margin (see above), so this is an affine
    map, not a straight scale -- used both for stamping the network icon
    and for panels/panel_a_layout_boxes.py's box export/import.
    """
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_pt = ax_left * panel_w_pt + x_frac * ax_w * panel_w_pt
    y_pt = ax_bottom * panel_h_pt + y_frac * ax_h * panel_h_pt
    return x_pt, y_pt


def page_pt_to_data_frac(x_pt: float, y_pt: float, figsize: tuple[float, float]) -> tuple[float, float]:
    """Inverse of data_frac_to_page_pt."""
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_frac = (x_pt - ax_left * panel_w_pt) / (ax_w * panel_w_pt)
    y_frac = (y_pt - ax_bottom * panel_h_pt) / (ax_h * panel_h_pt)
    return x_frac, y_frac


def load_layout_overrides() -> dict[str, tuple[float, float, float, float]] | None:
    if not LAYOUT_OVERRIDES_JSON.exists():
        return None
    raw = json.loads(LAYOUT_OVERRIDES_JSON.read_text(encoding="utf-8"))
    return {
        name: (box[0], box[1] + OVERRIDE_CONTENT_Y_SHIFT, box[2], box[3])
        for name, box in raw.items()
    }


def _ensure_network_icon() -> Path:
    if not NETWORK_ICON_PDF.exists():
        extract_network_icon()
    return NETWORK_ICON_PDF


def _stamp_network_icons(
    panel_pdf: Path,
    icon_pdf: Path,
    icon_slots: dict[str, dict[str, float]],
    figsize: tuple[float, float],
) -> None:
    ax_w, ax_h = AX_BOX[2], AX_BOX[3]
    x_scale = ax_w * figsize[0] * 72.0  # data-fraction (0..1) -> points, since xlim is (0, 1)
    y_scale = ax_h * figsize[1] * 72.0

    def to_pt(x_frac: float, y_frac: float) -> tuple[float, float]:
        return data_frac_to_page_pt(x_frac, y_frac, figsize)

    icon_reader = PdfReader(str(icon_pdf))
    icon_page = icon_reader.pages[0]
    icon_native_w = float(icon_page.mediabox.width)
    icon_native_h = float(icon_page.mediabox.height)
    icon_aspect = icon_native_w / icon_native_h

    writer = PdfWriter()
    writer.append(str(panel_pdf))
    base_page = writer.pages[0]

    for slot in icon_slots.values():
        x0_pt, y0_slot_pt = to_pt(slot["x"], slot["y"])
        w_pt = slot["w"] * x_scale
        # Preserve the icon's own aspect ratio rather than stretching it to
        # draw_model_icon's node-cluster height, which was tuned for a
        # visually denser matplotlib drawing, not this icon's real proportions.
        h_pt = w_pt / icon_aspect
        center_y_pt = y0_slot_pt + (slot["h"] * y_scale) / 2.0
        y0_pt = center_y_pt - h_pt / 2.0

        icon_reader_local = PdfReader(str(icon_pdf))
        icon_page_local = icon_reader_local.pages[0]
        transform = Transformation().scale(w_pt / icon_native_w, h_pt / icon_native_h).translate(tx=x0_pt, ty=y0_pt)
        base_page.merge_transformed_page(icon_page_local, transform)

    with open(panel_pdf, "wb") as f:
        writer.write(f)


def build_panel(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    out_dir: Path = OUT_DIR,
    *,
    panel_label: str = "A",
    panel_title: str = "FEMs sharpen spatial coding",
) -> Path:
    figure.configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    schematic_payload = figure.read_schematic_payload()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(AX_BOX))
    ax.set_axis_off()
    icon_slots = figure.draw_panel_b(
        ax,
        schematic_payload=schematic_payload,
        include_network_icon=False,
        layout_overrides=load_layout_overrides(),
        header_label=panel_label,
        header_title=panel_title,
        header_y=HEADER_Y,
        header_title_y_offset=TITLE_Y_OFFSET,
        header_title_y_offset_pt=TITLE_Y_OFFSET_PT,
    )

    out_path = out_dir / "panel_a.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)

    icon_pdf = _ensure_network_icon()
    _stamp_network_icons(out_path, icon_pdf, icon_slots, figsize)

    return out_path


def main() -> None:
    path = build_panel()
    print(path)


if __name__ == "__main__":
    main()
