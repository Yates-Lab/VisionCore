#!/usr/bin/env python3
"""Standalone build for Panel D (contour-relative stimulus + crop +
coherence gallery).

v3 architecture: every panel is its own independently-rendered figure,
composited onto the final page at a measured position (see
compose_ssi_figure_v3.py). E/F used to live as insets inside D's own axes
(there was unused space there); now that D is independently sized/placed,
E and F become their own top-level panels too (see
panels/panel_bd_path_bins.py's single-panel build), so this only draws
D itself -- draw_ef_insets=False. The drawing logic is otherwise unchanged
and still lives in _fig4_ssi_common.draw_panel_a.

AX_BOX reserves headroom for the panel letter+title (drawn via
draw_panel_header, above the axes' own y=1 edge) inside a *fixed* figsize
page, instead of letting bbox_inches="tight" grow the saved page to fit --
same fix as Panel A/G, needed so panel_d_layout_boxes.py's box export/
import has a deterministic axes-fraction -> page-point mapping to invert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_ssi_common as figure  # noqa: E402

import _fig4_layout as layout  # noqa: E402

OUT_DIR = _paths.PANELS_DIR
DEFAULT_FIGSIZE = layout.PANEL_BOXES["D"][2:4]

LAYOUT_OVERRIDES_JSON = _paths.PANEL_D_LAYOUT_OVERRIDES_JSON

AX_BOX = (0.0, 0.0, 1.0, 0.86)  # left, bottom, width, height (figure-fraction)
TITLE_Y_OFFSET = 0.020


def data_frac_to_page_pt(x_frac: float, y_frac: float, figsize: tuple[float, float]) -> tuple[float, float]:
    """See panels/panel_a_motion_schematic.py's function of the same name --
    identical purpose, just against this panel's own AX_BOX."""
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_pt = ax_left * panel_w_pt + x_frac * ax_w * panel_w_pt
    y_pt = ax_bottom * panel_h_pt + y_frac * ax_h * panel_h_pt
    return x_pt, y_pt


def page_pt_to_data_frac(x_pt: float, y_pt: float, figsize: tuple[float, float]) -> tuple[float, float]:
    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    ax_left, ax_bottom, ax_w, ax_h = AX_BOX
    x_frac = (x_pt - ax_left * panel_w_pt) / (ax_w * panel_w_pt)
    y_frac = (y_pt - ax_bottom * panel_h_pt) / (ax_h * panel_h_pt)
    return x_frac, y_frac


def load_layout_overrides() -> dict[str, tuple[float, float, float, float]] | None:
    if not LAYOUT_OVERRIDES_JSON.exists():
        return None
    raw = json.loads(LAYOUT_OVERRIDES_JSON.read_text(encoding="utf-8"))
    return {name: tuple(box) for name, box in raw.items()}


def build_panel(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    out_dir: Path = OUT_DIR,
    *,
    panel_label: str = "D",
    panel_title: str = "Local contours define\nthe relevant image axis",
) -> Path:
    figure.configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    schematic_payload = figure.read_schematic_payload()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(AX_BOX))
    ax.set_axis_off()
    figure.draw_panel_a(
        ax,
        schematic_payload=schematic_payload,
        draw_ef_insets=False,
        layout_overrides=load_layout_overrides(),
        header_label=panel_label,
        header_title=panel_title,
        header_title_y_offset=TITLE_Y_OFFSET,
    )

    out_path = out_dir / "panel_c.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def compute_current_boxes(figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> dict[str, tuple[float, float, float, float]]:
    """The resolved (default-merged-with-override) boxes, without a full
    build_panel() side effect -- used by panel_d_layout_boxes.py's exporter,
    which calls build_panel() separately to get a fresh background render."""
    figure.configure_matplotlib()
    schematic_payload = figure.read_schematic_payload()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(list(AX_BOX))
    ax.set_axis_off()
    boxes = figure.draw_panel_a(
        ax,
        schematic_payload=schematic_payload,
        draw_ef_insets=False,
        layout_overrides=load_layout_overrides(),
    )
    plt.close(fig)
    return boxes


def main() -> None:
    path = build_panel()
    print(path)


if __name__ == "__main__":
    main()
