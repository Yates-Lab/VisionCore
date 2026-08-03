#!/usr/bin/env python3
"""Extract the single-unit-readout network icon straight out of
ssi_figure_v2_3.pdf's Panel A -- the Illustrator-fixed reference artwork --
instead of continuing to hand-tune _fig4_ssi_common.draw_model_icon's
matplotlib reproduction of it.

The reference draws this icon once per row (FEM jittered / stabilized) at
identical geometry, just translated vertically; both instances were checked
by eye and are pixel-identical, so only the first (FEM) row is extracted and
reused for both rows in the v3 composite. It also includes an entry arrow
(cube -> network) that _fig4_ssi_common.draw_model_icon never drew, so
using the extracted version is a strict improvement over the matplotlib
reproduction, not just a style change.

Bounding box (`ICON_BBOX_PT`, PDF points, origin top-left of the page) was
measured by hand against this file's own vector drawings/text spans -- see
the exploration in this session -- padded by ~1-2pt on each side.

Does NOT extract the movie-cube icon: unlike the network diagram, the cube
already renders real per-run stimulus/trace data via
_fig4_contour_schematic.add_visual_model_input_cube, so
freezing it to this one reference PDF's snapshot would silently discard that
per-run data instead of copying a purely-illustrative asset.

Run: uv run python paper/fig4/_fig4_network_icon.py
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

REFERENCE_PDF = _paths.LAYOUT_REFERENCE_PDF
OUT_PDF = _paths.PANEL_A_NETWORK_ICON_PDF
PROVENANCE_JSON = _paths.PANEL_A_NETWORK_ICON_PROVENANCE_JSON

# (x0, y0, x1, y1) in PDF points, top-left origin, measured from the FEM
# (first) row's copy of the icon in ssi_figure_v2_3.pdf.
ICON_BBOX_PT = (158.0, 91.0, 279.0, 172.0)


def extract_network_icon(
    pdf_path: Path = REFERENCE_PDF,
    bbox_pt: tuple[float, float, float, float] = ICON_BBOX_PT,
    out_pdf: Path = OUT_PDF,
) -> Path:
    src = fitz.open(str(pdf_path))
    clip = fitz.Rect(*bbox_pt)

    out_dir = out_pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    new_doc = fitz.open()
    new_page = new_doc.new_page(width=clip.width, height=clip.height)
    new_page.show_pdf_page(new_page.rect, src, 0, clip=clip)
    new_doc.save(str(out_pdf))
    new_doc.close()
    src.close()

    provenance = {
        "asset": "panel_a_network_icon",
        "source_pdf": str(pdf_path.relative_to(ROOT)),
        "source_bbox_pt_top_left_origin": list(bbox_pt),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
        "output_size_pt": [clip.width, clip.height],
        "note": "cropped verbatim from ssi_figure_v2_3.pdf's Panel A FEM row; "
        "the stabilized row's copy is geometrically identical and is not "
        "extracted separately.",
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_pdf


def main() -> None:
    path = extract_network_icon()
    print(path)


if __name__ == "__main__":
    main()
