"""Layout and statistics contracts for the stabilization-control supplement."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.supp_stabilization_control import analysis, figure


def synthetic_scores(n=40):
    rng = np.random.default_rng(0)
    arrays = {"sessions": np.repeat(["A", "B"], n // 2), "population": np.ones(n, bool)}
    for name, keys in figure.PANEL_CONDITIONS.items():
        for offset, key in enumerate(keys):
            arrays[f"{name}_{key}"] = .5 - .02 * offset + rng.normal(0, .02, n)
        arrays[f"{name}_mask"] = np.ones(n, bool)
    return arrays


def synthetic_example(n=200, endpoint=150, n_lags=60):
    rng = np.random.default_rng(1)
    example = {"time_ms": np.arange(n) / .24, "endpoint": endpoint,
               "gaze_arcmin": np.cumsum(rng.normal(0, .3, (n, 2)), axis=0)}
    for offset, condition in enumerate(figure.STABILIZATION):
        example["movie_" + condition] = rng.integers(0, 255, (n_lags, 35, 35), dtype=np.uint8)
        if condition != "full":
            example["frozen_arcmin_" + condition] = np.array([offset, -offset], float)
    return example


class FigureTest(unittest.TestCase):
    def test_panels_show_every_condition_in_order(self):
        fig = figure.plot(synthetic_scores(), synthetic_example())
        self.addCleanup(plt.close, fig)
        panels = [*figure.PANEL_CONDITIONS.values(), figure.ABLATIONS]
        for ax, keys in zip(fig.axes[-3:], panels):
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                             [figure.LABELS[key] for key in keys])

    def test_example_shows_one_cube_per_stabilization_scope(self):
        fig = figure.plot(synthetic_scores(), synthetic_example())
        self.addCleanup(plt.close, fig)
        cubes = [ax for ax in fig.axes if len(ax.images) == 3]
        self.assertEqual(len(cubes), len(figure.STABILIZATION))
        titles = [ax.texts[0].get_text() for ax in cubes]
        self.assertEqual(titles, [figure.LABELS[key] for key in figure.STABILIZATION])

    def test_paired_cost_is_computed_within_unit(self):
        arrays = synthetic_scores()
        arrays["fraction_global"] = arrays["fraction_full"] - .25
        fig = figure.plot(arrays, synthetic_example())
        self.addCleanup(plt.close, fig)
        self.assertEqual(fig.axes[-1].texts[len(figure.ABLATIONS) - 1].get_text(), "-0.250")

    def test_cropped_whisker_is_rejected(self):
        arrays = synthetic_scores()
        arrays["ccnorm_global"] = arrays["ccnorm_global"] - 1.
        with self.assertRaises(AssertionError):
            figure.plot(arrays, synthetic_example())


class SummaryTest(unittest.TestCase):
    def test_summary_masks_nonfinite_units_and_pairs_within_unit(self):
        arrays = synthetic_scores()
        metrics = {name: {key: arrays[f"{name}_{key}"].copy() for key in keys}
                   for name, keys in figure.PANEL_CONDITIONS.items()}
        metrics["fraction"]["trial"][0] = np.nan
        calls = []

        def bootstrap(first, second, sessions, mask):
            calls.append(int(mask.sum()))
            return {"median": float(np.median((second - first)[mask])), "ci_low": 0., "ci_high": 0.}

        reports, out = analysis.summarize(
            metrics, arrays["sessions"], arrays["population"], bootstrap)
        # One unit lacks a single-trial score, so it leaves both panels.
        self.assertEqual(reports["ccnorm"]["n_units"], 39)
        self.assertEqual(reports["fraction"]["n_units"], 39)
        self.assertFalse(out["ccnorm_mask"][0])
        self.assertFalse(out["fraction_mask"][0])
        self.assertIn("global_minus_trial", reports["fraction"]["contrasts"])
        tex = analysis.statistics_tex(reports)
        self.assertIn("StabDTrialToGlobal", tex)
        # Panel D pairs every ablation with Full.
        for key in figure.ABLATIONS:
            self.assertIn(key + "_minus_full", reports["fraction"]["contrasts"])
            self.assertIn("\\StabDFullTo" + key.capitalize() + "CI}", tex)


if __name__ == "__main__":
    unittest.main()
