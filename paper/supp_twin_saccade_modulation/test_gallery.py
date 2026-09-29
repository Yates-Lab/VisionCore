"""Candidate review must use the cached cohort and the composite's A–D conventions."""
import json
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paper.supp_twin_saccade_modulation.analysis import CACHE, STATS
from paper.supp_twin_saccade_modulation.plot import rank_example_rows


class GalleryTest(unittest.TestCase):
    def test_top50_are_exact_ranked_cached_units_not_selection_top10(self):
        from paper.supp_twin_saccade_modulation.gallery import candidates
        folders = sorted((CACHE / 'sessions').iterdir())
        rows = [row for folder in folders for row in json.loads((folder / 'results.json').read_text())['cohort']]
        expected = rank_example_rows(rows)
        selected = candidates()
        self.assertEqual(len(expected), 316)
        self.assertEqual(len(selected), 50)
        self.assertEqual([(r['session'], r['cid']) for r in selected],
                         [(r['session'], r['cid']) for r in expected[:50]])
        self.assertEqual((selected[1]['session'], selected[1]['cid']), ('Allen_2022-04-06', 93))
        self.assertEqual((selected[29]['session'], selected[29]['cid']), ('Allen_2022-04-06', 98))
        self.assertEqual(len({r['session'] for r in selected}), 7)
        self.assertEqual(len(json.loads((STATS / 'selection.json').read_text())['examples']), 10)

    def test_page_uses_composite_maps_psth_event_bands_and_identifiers(self):
        from paper.supp_twin_saccade_modulation.gallery import page
        time = np.arange(59, 193) / 240
        mask = np.ones((2, len(time)), bool)
        observed = np.ones_like(mask, float)
        observed[0, 1] = 5
        full = np.full(mask.shape, 240.)
        full[1, 1] = 1500
        maps = dict(trials=[3, 4], time_s=time, mask=mask,
                    observed=observed, full=full,
                    gain=np.full(mask.shape, -2.), additive=np.full(mask.shape, 3.),
                    markers=[(0, .3, False), (1, .5, True)])
        a = dict(event_lags_ms=np.arange(-24, 49)/240*1000)
        means = {'gain': np.linspace(-20, 15, 73), 'additive': np.linspace(-5, 3, 73)}
        intervals = {k: np.stack([v-3, v+2]) for k, v in means.items()}
        row = dict(session='Allen_2022-04-06', cid=98, ccnorm=.9, ccmax=.95)
        fig = page(row, 30, maps, a['event_lags_ms'], means, intervals, 50, 3.)
        try:
            self.assertEqual(len(fig.axes), 9)
            np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), observed * 240)
            self.assertIs(fig.axes[0].images[0].norm, fig.axes[1].images[0].norm)
            np.testing.assert_array_equal(fig.axes[0].images[0].get_cmap()(np.linspace(0, 1, 256)),
                                          fig.axes[1].images[0].get_cmap()(np.linspace(0, 1, 256)))
            self.assertEqual((fig.axes[1].images[0].norm.vmin,
                              fig.axes[1].images[0].norm.vmax, fig.axes[1].images[0].norm.clip),
                             (0, 500, True))
            np.testing.assert_array_equal(fig.axes[1].images[0].get_array().data, full)
            self.assertEqual(len(fig.axes[0].collections), 1)
            self.assertEqual(len(fig.axes[0].images), 1)
            for ax in fig.axes[:4]:
                self.assertEqual(ax.get_xlabel(), 'Trial time (s)')
            for ax in fig.axes[5:7]:
                self.assertIn('Saccade-aligned', ax.get_title())
                self.assertEqual(ax.get_xlabel(), 'Time from saccade onset (ms)')
                self.assertIsNone(ax.get_legend())
            self.assertEqual([ax.get_xlabel() for ax in fig.axes[7:]],
                             ['Rate (spikes/s)', 'Signed effect (spikes/s)'])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for pair, bar in ((fig.axes[:2], fig.axes[7]), (fig.axes[2:4], fig.axes[8])):
                self.assertAlmostEqual(bar.get_position().x0, pair[0].get_position().x0)
                self.assertAlmostEqual(bar.get_position().x1, pair[1].get_position().x1)
                self.assertGreater(min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in pair),
                                   bar.get_window_extent(renderer).y1 + 2)
                self.assertGreater(bar.xaxis.label.get_window_extent(renderer).y0,
                                   max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes[4:7]) + 8)
            self.assertIn('Rank 30 / page 30', fig.texts[0].get_text())
            self.assertIn('Current example', fig.texts[0].get_text())
            self.assertIn('CID 98', fig.texts[0].get_text())
            self.assertIn('50 supported events', fig.texts[0].get_text())
            self.assertEqual(fig.axes[4].lines[0].get_ydata()[0], 336)
            self.assertEqual(fig.axes[4].lines[1].get_ydata()[0], 366)
            self.assertEqual(fig.axes[5].get_xlim(), (-100, 200))
            self.assertEqual(len(fig.axes[5].collections), 1)
            self.assertEqual(fig.axes[5].get_ylim(), fig.axes[6].get_ylim())
            low, high = fig.axes[5].get_ylim()
            self.assertAlmostEqual(low, -high)
            for name in ('gain', 'additive'):
                self.assertLess(low, min(means[name].min(), intervals[name].min()))
                self.assertGreater(high, max(means[name].max(), intervals[name].max()))
            for ax in fig.axes[:4]:
                self.assertEqual(ax.collections[-1].get_label(), 'Saccade onset')
                np.testing.assert_array_equal(ax.collections[-1].get_offsets(), [[.3, 0], [.5, 1]])
                self.assertEqual(ax.collections[-1].get_sizes()[0], 10)
            self.assertEqual((fig.axes[2].images[0].norm.vmin,
                              fig.axes[2].images[0].norm.vmax), (-3, 3))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
