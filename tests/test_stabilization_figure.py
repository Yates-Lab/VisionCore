"""Check the installed stabilization-control figure layout after rendering."""
import unittest
from pathlib import Path

import numpy as np
import pymupdf

ABLATIONS = ["Retinal only", "History-local", "Trial", "Global"]
# Panels B, C, and D, left to right.
PANELS = (["Full", *ABLATIONS], ["PSTH", "Full", *ABLATIONS], ABLATIONS)


class StabilizationFigureTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "manuscript/figures/stabilization_control.pdf"
        self.document = pymupdf.open(path)
        self.addCleanup(self.document.close)
        self.page = self.document[0]
        # The schematic occupies the top half and repeats the condition names.
        self.middle = self.page.rect.height / 2

    def split(self, items, position):
        """Group left-to-right items into the three score panels."""
        items = sorted(items, key=position)
        self.assertEqual(len(items), sum(map(len, PANELS)))
        edges = np.cumsum([0, *map(len, PANELS)])
        return [items[start:stop] for start, stop in zip(edges[:-1], edges[1:])]

    def test_condition_order_in_score_panels(self):
        names = {name for panel in PANELS for name in panel}
        spans = [span for block in self.page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]
                 if span["text"] in names and span["bbox"][1] > self.middle]
        for labels, expected in zip(self.split(spans, lambda s: s["bbox"][2]), PANELS):
            with self.subTest(panel=expected):
                self.assertEqual([s["text"] for s in labels], expected)

    def test_box_spacing_is_uniform_in_score_panels(self):
        boxes = [d["rect"] for d in self.page.get_drawings()
                 if "f" in d["type"] and d["fill"] != (1, 1, 1)
                 and d["rect"].width > 0 and d["rect"].height > 0 and d["rect"].y0 > self.middle]
        for panel, expected in zip(self.split(boxes, lambda r: r.x0), PANELS):
            with self.subTest(panel=expected):
                gaps = np.diff([(r.x0 + r.x1) / 2 for r in panel])
                np.testing.assert_allclose(gaps, gaps[0], atol=1e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
