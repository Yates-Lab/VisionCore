"""Panel bounding boxes for ssi_figure_v3's page layout.

PANEL_BOXES[letter] = (x_in, y_in, w_in, h_in), inches, y measured from
the page's TOP edge; convert to PDF-native bottom-up points when
compositing: ty = page_h_pt - y_in*72 - h_in*72.

This file is written by either of two independent tools -- re-deriving
it from the reference PDF discards any hand edits made by the other:
    extract_reference_layout.py (upstream; not in this repo)
        (re-measures every box from ssi_figure_v2_3.pdf from scratch)
    page_layout_boxes.py (upstream; not in this repo) import
        (reads back a hand-dragged/resized SVG export of these same boxes)
"""

from __future__ import annotations

PAGE_SIZE_IN = (8.5000, 11.0000)

PANEL_BOXES: dict[str, tuple[float, float, float, float]] = {
    "A": (0.0944, 0.3722, 5.6278, 3.7417),
    "B": (6.0403, 0.6528, 2.1333, 1.6875),
    "C": (6.0403, 2.4458, 2.1333, 1.6486),
    "D": (0.0944, 4.1931, 3.2806, 3.3750),
    "E": (3.5222, 4.2500, 2.0903, 1.6056),
    "F": (3.5153, 5.9833, 2.0861, 1.5847),
    "G": (5.7917, 4.2250, 2.4861, 3.3722),
    "H": (0.0000, 7.7056, 2.9306, 3.0306),
    "I": (3.0056, 7.7056, 2.7583, 3.0139),
    "J": (5.8889, 7.7083, 2.5556, 3.0278),
}
