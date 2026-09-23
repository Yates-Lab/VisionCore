from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

FIG2_DIR = Path(__file__).resolve().parents[1] / "paper" / "fig2"
sys.path.insert(0, str(FIG2_DIR))

import generate_figure2 as figure2
import generate_panel_example as panel_example


DECOMP = {
    "bin_centers": np.linspace(0.05, 0.95, 8),
    "cum_crate": np.linspace(4.6, 1.3, 8),
    "Ctotal": 6.5,
    "Cpsth": 1.15,
    "sigma_int": 1.9,
}


class Figure2LabelTest(unittest.TestCase):
    def test_variance_labels_use_explicit_sigma_notation(self) -> None:
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        panel_example.plot_unaccounted_variance_panel(ax, decomp=DECOMP)
        # The composed panel uses body-size math labels, with smaller scripts.
        for text in ax.texts:
            text.set_fontsize(8.8)
        text_by_label = {text.get_text(): text for text in ax.texts}

        expected = {
            "FEM variability\n" + r"($\sigma^2_{\mathrm{FEM}}$)",
            r"Stimulus variability ($\sigma^2_{\mathrm{PSTH}}$)",
            "Corrected residual\n" + r"variability ($\sigma^2_{\mathrm{res}}$)",
            "$\\sigma^2_{\\mathrm{rate}} = \\sigma^2_{\\mathrm{PSTH}} + "
            "\\sigma^2_{\\mathrm{FEM}}$",
        }
        self.assertLessEqual(expected, text_by_label.keys())

        renderer = fig.canvas.get_renderer()
        nearby = expected | {
            "Total variability",
            "Uncorrected\nresidual",
            "Trajectories\nmatched",
        }
        boxes = {
            label: text_by_label[label].get_window_extent(renderer)
            for label in nearby
        }
        for i, label in enumerate(nearby):
            for other in list(nearby)[i + 1:]:
                self.assertFalse(
                    boxes[label].overlaps(boxes[other]),
                    f"{label!r} overlaps {other!r}",
                )

        band_bottom = ax.transData.transform((0.0, DECOMP["Ctotal"] - DECOMP["Cpsth"]))[1]
        band_top = ax.transData.transform((0.0, DECOMP["Ctotal"]))[1]
        for label in (r"Stimulus variability ($\sigma^2_{\mathrm{PSTH}}$)",):
            self.assertGreater(boxes[label].y0, band_bottom + 2)
            self.assertLess(boxes[label].y1, band_top - 2)
        residual = text_by_label["Corrected residual\n" + r"variability ($\sigma^2_{\mathrm{res}}$)"]
        self.assertEqual(residual.get_color(), "k")
        uncorrected = text_by_label["Uncorrected\nresidual"]
        self.assertAlmostEqual(uncorrected.get_window_extent(renderer).x1, ax.bbox.x1)
        self.assertEqual(uncorrected.get_ha(), "right")
        equation = next(text for label, text in text_by_label.items() if label.startswith(r"$\sigma^2_{\mathrm{rate}}"))
        floor_y = ax.transData.transform((0, DECOMP["sigma_int"]))[1]
        self.assertLess(residual.get_window_extent(renderer).y1, floor_y)
        self.assertGreater(equation.get_window_extent(renderer).y0, floor_y)
        equation_box = equation.get_window_extent(renderer)
        self.assertAlmostEqual((equation_box.x0 + equation_box.x1) / 2, (ax.bbox.x0 + ax.bbox.x1) / 2)
        arrows = [text for text in ax.texts if getattr(text, "arrow_patch", None)]
        self.assertTrue(any(text.xy == (0.92, DECOMP["sigma_int"]) and text.xyann == (0.92, 0) for text in arrows))
        plt.close(fig)

        captured = {}

        def fem_panel(ax, data):
            captured["axis"] = ax
            return ax.figure, ax

        with (
            patch.object(figure2, "_plot_eye_rate_panel"),
            patch.object(figure2, "_compute_unaccounted_curve", return_value=DECOMP),
            patch.object(figure2, "plot_fem_fraction", side_effect=fem_panel),
            patch.object(figure2, "_plot_compact_cov_decomp", return_value=[]),
            patch.object(figure2, "_expand_axes_group_from_lower_right"),
            patch.object(figure2, "_center_top_residual"),
            patch.object(figure2, "plot_fano_population"),
            patch.object(figure2, "plot_nc_violin"),
            patch.object(
                figure2,
                "_plot_pr_comparison",
                side_effect=lambda fig, spec, data: fig.add_subplot(spec),
            ),
            patch.object(
                figure2,
                "_plot_subspace_schematic",
                side_effect=lambda fig, spec: fig.add_subplot(spec, projection="3d"),
            ),
            patch.object(
                figure2,
                "_plot_subspace_alignment_vs_shuffle",
                side_effect=lambda fig, spec, data: fig.add_subplot(spec),
            ),
        ):
            with patch.dict(os.environ, {"VISIONCORE_MIN_FIGURE_FONT_PT": "8.8"}):
                figure2.compose(prepared_data={}, return_png_bytes=True)

        figure = captured["axis"].figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        b_axis = next(ax for ax in figure.axes if ax.get_title(loc="left") == "B")
        for text in b_axis.texts:
            if "$" in text.get_text():
                self.assertEqual(text.get_fontsize(), b_axis.yaxis.label.get_fontsize())
        uncorrected = next(text for text in b_axis.texts if text.get_text() == "Uncorrected\nresidual")
        self.assertAlmostEqual(uncorrected.get_fontsize(), 0.9 * b_axis.yaxis.label.get_fontsize())
        f_axis = next(ax for ax in figure.axes if ax.get_title(loc="left") == "F")
        self.assertGreater(
            captured["axis"].xaxis.label.get_window_extent(renderer).y0,
            f_axis._left_title.get_window_extent(renderer).y1 + 4,
            "Panel C xlabel crowds panel F title",
        )
        self.assertEqual(captured["axis"].xaxis.label.get_fontsize(), b_axis.yaxis.label.get_fontsize())
        self.assertEqual(
            captured["axis"].get_xlabel(),
            "FEM fraction of rate variance "
            "($\\sigma^2_{\\mathrm{FEM}} / \\sigma^2_{\\mathrm{rate}}$)",
        )


if __name__ == "__main__":
    unittest.main()
