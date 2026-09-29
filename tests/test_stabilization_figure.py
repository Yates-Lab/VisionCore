"""Check the installed Extended Data Figure 3 layout after rendering."""
import unittest
from pathlib import Path

import numpy as np
import pymupdf


class StabilizationFigureTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "manuscript/figures/stabilization_control.pdf"
        self.document = pymupdf.open(path)
        self.addCleanup(self.document.close)
        self.page = self.document[0]

    def test_condition_order_in_both_panels(self):
        order = ["Full", "Retinal", "Global", "Trial", "History-local"]
        spans = [span for block in self.page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]
                 if span["text"] in {*order, "PSTH", "Retinal-only"}]
        for panel, expected in enumerate((order, ["PSTH", *order])):
            with self.subTest(panel=panel):
                labels = [s for s in spans if (s["bbox"][2] > self.page.rect.width / 2) == bool(panel)]
                self.assertEqual([s["text"] for s in sorted(labels, key=lambda s: s["bbox"][2])], expected)

    def test_box_spacing_is_uniform_in_both_panels(self):
        boxes = [d["rect"] for d in self.page.get_drawings()
                 if d["type"] == "f" and d["fill"] != (1, 1, 1)
                 and d["rect"].width > 0 and d["rect"].height > 0]
        for panel, count in enumerate((5, 6)):
            with self.subTest(panel=panel):
                centers = sorted((r.x0 + r.x1) / 2 for r in boxes
                                 if (r.x0 > self.page.rect.width / 2) == bool(panel))
                self.assertEqual(len(centers), count)
                gaps = np.diff(centers)
                np.testing.assert_allclose(gaps, gaps[0], atol=1e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
