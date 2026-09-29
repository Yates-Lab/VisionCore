import unittest
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
import numpy as np

from paper.supp_twin_saccade_modulation import figure
from paper.supp_twin_saccade_modulation.analysis import CACHE, STATS
from paper.supp_twin_saccade_modulation.figure import (
    matched_psth, check_trial_indices, draw_population,
    draw_saccade_markers, draw_example_pathway,
)


class AssemblyTest(unittest.TestCase):
    def test_psth_uses_matched_display_bins_and_native_rates(self):
        maps = dict(mask=np.array([[1, 1, 0, 1], [1, 0, 1, 1]], bool),
                    observed=np.array([[1., 2., np.nan, 4.], [3., np.nan, 5., 6.]]),
                    full=np.array([[480., 960., np.nan, 1920.], [1440., np.nan, 2400., 2880.]]))
        result = matched_psth(maps, bins=1)
        np.testing.assert_array_equal(result['valid_trials'], [2, 1, 1, 2])
        np.testing.assert_allclose(result['observed_raw_hz'], [480, 480, 1200, 1200])
        np.testing.assert_allclose(result['predicted_raw_hz'], [960, 960, 2400, 2400])
        np.testing.assert_array_equal(result['mask'], maps['mask'])

    def test_smoothing_ignores_missing_bins_and_does_not_bridge_gaps(self):
        maps = dict(mask=np.array([[1, 0, 0, 0, 1]], bool),
                    observed=np.array([[1., np.nan, np.nan, np.nan, 5.]]),
                    full=np.array([[480., np.nan, np.nan, np.nan, 2400.]]))
        result = matched_psth(maps, bins=3)
        np.testing.assert_array_equal(result['valid_trials'], [1, 0, 0, 0, 1])
        np.testing.assert_allclose(result['observed_smooth_hz'][[0, 4]], [240, 1200])
        np.testing.assert_allclose(result['predicted_smooth_hz'][[0, 4]], [480, 2400])
        self.assertTrue(np.isnan(result['observed_smooth_hz'][1:4]).all())
        self.assertTrue(np.isnan(result['predicted_raw_hz'][1:4]).all())

    def test_trial_spread_is_sample_sd_after_per_trial_smoothing_with_missing_support(self):
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], bool)
        obs = np.array([[0., 0., 3., np.nan, np.nan], [0., 0., 0., np.nan, np.nan]])
        pred = np.array([[0., 0., 6., np.nan, np.nan], [0., 0., 0., np.nan, np.nan]])
        maps = dict(mask=mask, observed=obs, full=pred)
        got = matched_psth(maps, bins=3)
        self.assertAlmostEqual(got['observed_std_hz'][1], 240 / np.sqrt(2))
        self.assertAlmostEqual(got['predicted_std_hz'][1], 6 / (3 * np.sqrt(2)))
        self.assertEqual(got['observed_std_hz'][0], 0)
        self.assertTrue(np.isnan(got['observed_std_hz'][3:]).all())
        self.assertTrue(np.isnan(got['predicted_std_hz'][3:]).all())
        mask[1, 2] = False
        obs[1, 2] = pred[1, 2] = np.nan
        got = matched_psth(maps, bins=3)
        self.assertTrue(np.isnan(got['observed_std_hz'][2]))
        self.assertTrue(np.isnan(got['predicted_std_hz'][2]))
        self.assertEqual(got['observed_std_hz'][0], 0)
        self.assertEqual(got['valid_trials'][2], 1)

    def test_default_outside_top10_is_eligible_and_invalid_ids_fail(self):
        self.assertEqual((figure.SESSION, figure.CID), ('Allen_2022-04-06', 98))
        selected = json.loads((STATS / 'selection.json').read_text())
        self.assertNotIn(98, [r['cid'] for r in selected['examples'] if r['session'] == figure.SESSION])
        row = figure.select_example(figure.SESSION, 98)
        self.assertEqual((row['ordinal'], row['supported_events']), (90, 50))
        self.assertAlmostEqual(row['ccnorm'], .8615716118773447)
        with self.assertRaisesRegex(ValueError, 'ineligible|not in cached cohort'):
            figure.select_example(figure.SESSION, -999)
        with self.assertRaisesRegex(ValueError, 'ineligible|not in cached cohort'):
            figure.select_example(figure.SESSION, next(r['cid'] for r in json.loads((CACHE / 'sessions' / figure.SESSION / 'results.json').read_text())['cohort'] if not r['example_eligible'] or r['supported_events'] < 20))

    def test_compact_canvas_psth_bands_and_population_widths(self):
        fig, grid = figure.make_figure()
        try:
            self.assertEqual(tuple(fig.get_size_inches()), (8, 10.5))
            self.assertEqual((grid[2, :7].colspan.stop, grid[2, 7:].colspan.start), (7, 7))
            ax = fig.add_subplot(grid[1, :4])
            time = np.arange(3) / 240
            psth = dict(observed_smooth_hz=np.array([1., 2., 3.]), predicted_smooth_hz=np.array([3., 4., 5.]),
                        observed_std_hz=np.array([3., 1., 0.]), predicted_std_hz=np.array([1., 1., 1.]))
            figure.draw_psth(ax, time, psth)
            self.assertEqual(len(ax.collections), 2)
            self.assertEqual(len(ax.lines), 2)
            self.assertLessEqual(ax.get_ylim()[0], -2)
            self.assertEqual(ax.get_legend_handles_labels()[1], ['Observed', 'Full prediction'])
        finally:
            plt.close(fig)

    def test_trial_rates_share_capped_scale_mask_and_extent_without_mutating_sources(self):
        fig, axes = plt.subplots(1, 2)
        values = np.array([[0., 1., np.nan, 5.], [2., 3., 4., 0.]])
        mask = np.array([[1, 1, 0, 1], [1, 1, 1, 1]], bool)
        full = np.array([[0., 300., np.nan, 1400.], [100., 240., 500., 0.]])
        maps = dict(observed=values.copy(), full=full.copy(), mask=mask)
        try:
            norm = figure.trial_rate_norm(maps)
            images = [figure.draw_trial_rate(ax, maps, key, (0, 4/240, 1.5, -.5), norm)
                      for ax, key in zip(axes, ('observed', 'full'))]
            self.assertIs(type(norm), Normalize)
            self.assertEqual((norm.vmin, norm.vmax, norm.clip), (0, 500, True))
            self.assertIs(images[0].norm, images[1].norm)
            np.testing.assert_array_equal(images[0].get_cmap()(np.linspace(0, 1, 256)),
                                          images[1].get_cmap()(np.linspace(0, 1, 256)))
            for image, expected in zip(images, (values * 240, full)):
                shown = image.get_array()
                np.testing.assert_array_equal(shown.data[mask], expected[mask])
                np.testing.assert_array_equal(np.ma.getmaskarray(shown), ~mask)
                np.testing.assert_array_equal(image.get_extent(), (0, 4/240, 1.5, -.5))
                self.assertEqual(image.get_cmap()(np.ma.masked), to_rgba('#e9e9e9'))
            self.assertEqual(images[0].norm(480), .96)
            self.assertEqual(images[0].norm(1200), 1)
            self.assertEqual(images[1].norm(1400), 1)
            np.testing.assert_array_equal(images[0].get_cmap()(norm([500, 1200])),
                                          [images[0].get_cmap()(1.)] * 2)
            np.testing.assert_array_equal(maps['observed'], values)
            np.testing.assert_array_equal(maps['full'], full)
            self.assertEqual(len(axes[0].collections), 0)
        finally:
            plt.close(fig)

    def test_top_pair_colorbars_span_pairs_with_label_and_title_clearance(self):
        fig, grid = figure.make_figure()
        top = figure.make_top_grid(grid)
        try:
            axes = [fig.add_subplot(top[0, 3*j:3*j+3]) for j in range(4)]
            for ax in axes:
                ax.set_xlabel('Trial time (s)')
            bars = figure.add_trial_colorbars(fig, axes,
                       axes[0].imshow([[0., 1200.]], norm=Normalize(0, 500, clip=True), aspect='auto'),
                       axes[2].imshow([[-3., 3.]], aspect='auto'))
            b = fig.add_subplot(grid[1, :4]); b.set_title('Stimulus-aligned PSTH')
            c = fig.add_subplot(grid[1, 4:8]); c.set_title('Saccade-aligned\ngain pathway')
            d = fig.add_subplot(grid[1, 8:]); d.set_title('Saccade-aligned\nadditive pathway')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars[0].ax.get_xticklabels()[-1].get_text(), '500+')
            self.assertEqual([bar.ax.get_xlabel() for bar in bars],
                             ['Rate (spikes/s)', 'Signed effect (spikes/s)'])
            for pair, bar in zip((axes[:2], axes[2:]), bars):
                left, right = pair
                self.assertAlmostEqual(bar.ax.get_position().x0, left.get_position().x0)
                self.assertAlmostEqual(bar.ax.get_position().x1, right.get_position().x1)
                self.assertGreater(min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in pair),
                                   bar.ax.get_window_extent(renderer).y1 + 2)
                self.assertLess(min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in pair)
                                - bar.ax.get_window_extent(renderer).y1, 35)
                self.assertGreater(bar.ax.xaxis.label.get_window_extent(renderer).y0,
                                   max(t.get_window_extent(renderer).y1 for t in (b.title, c.title, d.title)) + 8)
        finally:
            plt.close(fig)

    def test_second_row_artist_bounds_clear_neighbors_at_print_size(self):
        fig, grid = figure.make_figure()
        time = np.arange(59, 193) / 240
        lags = np.arange(-24, 49) / 240 * 1000
        psth = {f'{key}_{suffix}_hz': np.full(len(time), value)
                for key, value in (('observed', 30.), ('predicted', 50.))
                for suffix in ('smooth', 'std')}
        means = {'gain': np.linspace(-20, 15, len(lags)), 'additive': np.linspace(-5, 3, len(lags))}
        intervals = {key: np.stack((value-2, value+2)) for key, value in means.items()}
        try:
            axes = figure.make_example_axes(fig, grid)
            figure.draw_psth(axes[0], time, psth)
            axes[0].set_xlim(time[0], .8)
            for ax, key in zip(axes[1:], ('gain', 'additive')):
                draw_example_pathway(ax, lags, key, means, intervals)
                ax.set_ylabel('')
            for letter, ax in zip('BCD', axes):
                ax.text(-.09, 1.09, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for left, right in zip(axes, axes[1:]):
                left_boxes = [left.get_window_extent(renderer),
                              left.xaxis.label.get_window_extent(renderer),
                              *(tick.get_window_extent(renderer) for tick in left.get_xticklabels() if tick.get_visible()),
                              *(tick.get_window_extent(renderer) for tick in left.get_yticklabels() if tick.get_visible())]
                right_boxes = [right.get_window_extent(renderer),
                               right.yaxis.label.get_window_extent(renderer),
                               *(tick.get_window_extent(renderer) for tick in right.get_xticklabels() if tick.get_visible()),
                               *(tick.get_window_extent(renderer) for tick in right.get_yticklabels() if tick.get_visible())]
                gap = (min(box.x0 for box in right_boxes) - max(box.x1 for box in left_boxes)) / fig.dpi * 72
                self.assertGreater(gap, 3.6)
                self.assertLess(gap, 14.4)
            for ax in axes:
                letter = ax.texts[-1]
                self.assertFalse(letter.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)))
        finally:
            plt.close(fig)

    def test_compact_panel_letters_and_titles_do_not_collide(self):
        fig, grid = figure.make_figure()
        lags = np.arange(-24, 49) / 240 * 1000
        population = dict(gain_hz=np.ones((2, len(lags))), additive_hz=-np.ones((2, len(lags))), lags_ms=lags)
        areas = figure.integrated_effects(lags, population['gain_hz'], population['additive_hz'])
        try:
            axes = figure.draw_population_summary(fig, grid, population, areas, 2.)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in (axes[0], axes[1], axes[2], axes[3]):
                letter = next(t for t in ax.texts if t.get_text() in 'EFGH')
                self.assertFalse(letter.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)))
        finally:
            plt.close(fig)

    def test_native_trial_mapping_accepts_float_roundoff(self):
        time = np.arange(59, 193) / 240
        a = dict(trial_ids=np.full(len(time), 7), times=np.arange(len(time)) / 240,
                 psth_indices=np.arange(len(time)))
        check_trial_indices(a, dict(trials=[7], time_s=time))

    def test_integrated_balance_includes_200ms_and_leaves_zero_undefined(self):
        lags = np.array([-100., 0., 100., 200., 204.])
        gain = np.array([[999., 2., 2., 2., 999.],
                         [-999., -4., -4., -4., -999.],
                         [999., 2., -2., 2., 999.],
                         [999., 0., 0., 0., 999.]])
        try:
            result = figure.integrated_effects(lags, gain, gain)
        except ValueError as exc:
            self.fail(f'0–200 ms integration missing: {exc}')
        np.testing.assert_array_equal(result['integration_lags_ms'], [0, 100, 200])
        np.testing.assert_allclose(result['gain_enhancing_area_spikes'], [.4, 0, .2, 0])
        np.testing.assert_allclose(result['gain_suppressive_area_spikes'], [0, .8, .2, 0])
        np.testing.assert_allclose(result['gain_absolute_area_spikes'], [.4, .8, .4, 0])
        self.assertIn('gain_signed_balance', result)
        np.testing.assert_allclose(result['gain_signed_balance'][:3], [1, -1, 0])
        self.assertTrue(np.isnan(result['gain_signed_balance'][3]))
        self.assertNotIn('gain_weaker_sign_fraction', result)
        np.testing.assert_array_equal(result['additive_signed_balance'][:3],
                                      result['gain_signed_balance'][:3])
        self.assertIn('balance_bin_edges', result)
        np.testing.assert_array_equal(result['balance_bin_edges'], np.linspace(-1, 1, 11))
        np.testing.assert_array_equal(result['gain_balance_bin_counts'], [1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        np.testing.assert_allclose(result['gain_balance_bin_fractions'],
                                   np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1]) / 3)
        self.assertEqual(result['gain_balance_bin_counts'].sum(), 3)  # zero balance undefined
        self.assertAlmostEqual(result['gain_balance_bin_fractions'].sum(), 1)
        self.assertIn('strength_bin_edges_spikes', result)
        edges = result['strength_bin_edges_spikes']
        np.testing.assert_allclose(np.diff(edges), .05, rtol=0, atol=1e-14)
        self.assertEqual(edges[0], 0)
        self.assertGreaterEqual(edges[-1], .8)
        np.testing.assert_array_equal(result['gain_strength_bin_counts'],
                                      np.histogram([.4, .8, .4, 0], bins=edges)[0])
        self.assertEqual(result['gain_strength_bin_counts'][0], 1)
        self.assertEqual(result['gain_strength_bin_counts'][-1], 1)  # highest endpoint
        self.assertEqual(result['gain_strength_bin_counts'].sum(), 4)  # includes zero
        np.testing.assert_allclose(result['gain_strength_bin_fractions'],
                                   result['gain_strength_bin_counts']/4)
        self.assertAlmostEqual(result['gain_strength_bin_fractions'].sum(), 1)

    def test_uniform_saccade_markers_keep_all_locations_and_label(self):
        fig, axes = plt.subplots(1, 4)
        markers = [(0, .25, True), (2, .5, False)]
        try:
            for ax in axes:
                draw_saccade_markers(ax, markers)
                ax.legend()
                self.assertEqual(ax.get_legend_handles_labels()[1], ['Saccade onset'])
                points = ax.collections[0]
                np.testing.assert_array_equal(points.get_offsets(), [[.25, 0], [.5, 2]])
                self.assertEqual(len(points.get_facecolors()), 0)
                self.assertEqual(points.get_sizes()[0], 10)
                self.assertLessEqual(points.get_linewidths()[0], .8)
                self.assertEqual(len(points.get_paths()[0].vertices), 4)  # right-pointing triangle
                self.assertEqual(np.count_nonzero(np.isclose(points.get_paths()[0].vertices[:, 0], .5)), 2)
                self.assertEqual(np.count_nonzero(np.isclose(points.get_paths()[0].vertices[:, 0], -.5)), 2)
                self.assertGreater(points.get_edgecolors()[0, 1], points.get_edgecolors()[0, 0])
        finally:
            plt.close(fig)

    def test_example_pathway_restores_full_display_without_changing_trace_or_band(self):
        fig, ax = plt.subplots()
        try:
            lags = np.array([-100., 0., 75., 150., 200.])
            draw_example_pathway(ax, lags, 'gain',
                                 {'gain': np.arange(5.)}, {'gain': np.array([np.arange(5.) - 1,
                                                                              np.arange(5.) + 1])})
            self.assertEqual(ax.get_xlim(), (-100, 200))
            self.assertEqual(ax.get_xlabel(), 'Time from saccade onset (ms)')
            self.assertIn('Saccade-aligned', ax.get_title())
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), lags)
            self.assertEqual(len(ax.collections), 1)  # full bootstrap band
            self.assertEqual(len(ax.patches), 0)
            self.assertEqual(len(ax.lines), 3)  # trace and zero references
            self.assertIn('spikes/s', ax.get_ylabel())
            self.assertEqual(ax.get_legend_handles_labels()[1], ['Gain effect'])
        finally:
            plt.close(fig)

    def test_balance_scatter_and_histogram_share_mapping_and_linear_scales(self):
        gain = np.array([-1., 1.])
        additive = np.array([0., 1.])
        result = dict(gain_signed_balance=gain, gain_absolute_area_spikes=np.array([0., .15]),
                      additive_signed_balance=additive, additive_absolute_area_spikes=np.array([.05, .30]))
        bins = np.linspace(-1, 1, 11)
        result['balance_bin_edges'] = bins
        for name, values in (('gain', gain), ('additive', additive)):
            counts = np.histogram(values, bins=bins)[0]
            result[f'{name}_balance_bin_counts'] = counts
            result[f'{name}_balance_bin_fractions'] = counts / len(values)
        strength_edges = np.arange(7) * .05
        result['strength_bin_edges_spikes'] = strength_edges
        for name in ('gain', 'additive'):
            counts = np.histogram(result[f'{name}_absolute_area_spikes'], bins=strength_edges)[0]
            result[f'{name}_strength_bin_counts'] = counts
            result[f'{name}_strength_bin_fractions'] = counts / 2
        fig, axes = plt.subplots(2, 3)
        scatters, histograms, strength_hists = axes[:, 0], axes[:, 1], axes[:, 2]
        try:
            try:
                figure.draw_area_scatter(scatters, histograms, strength_hists, result)
            except TypeError as exc:
                self.fail(f'right marginal axes missing: {exc}')
            fig.canvas.draw()
            self.assertEqual(scatters[0].get_ylim(), scatters[1].get_ylim())
            self.assertEqual(histograms[0].get_ylim(), histograms[1].get_ylim())
            self.assertEqual(strength_hists[0].get_xlim(), strength_hists[1].get_xlim())
            for scatter, hist, right, name in zip(scatters, histograms, strength_hists, ('gain', 'additive')):
                self.assertEqual(scatter.get_xlim(), hist.get_xlim())
                self.assertEqual(scatter.get_xlim(), (-1, 1))
                self.assertEqual(scatter.get_ylim()[0], 0)
                self.assertGreater(scatter.get_ylim()[1], .30)
                self.assertEqual(scatter.get_ylabel(), 'Total (spikes/saccade)')
                self.assertEqual(scatter.get_yscale(), 'linear')
                self.assertEqual(hist.get_yscale(), 'linear')
                self.assertEqual(scatter.get_xlabel(), '')
                self.assertFalse(any(label.get_visible() for label in scatter.get_xticklabels()))
                self.assertEqual(hist.get_xlabel(), 'Signed balance')
                self.assertEqual(hist.get_ylabel(), '')  # Caption and right marginal label identify fractions.
                self.assertEqual([label.get_text() for label in hist.get_xticklabels()],
                                 ['−1\nNegative only', '0\nBalanced', '+1\nPositive only'])
                np.testing.assert_array_equal(scatter.lines[0].get_xdata(), [0, 0])
                self.assertEqual(len(scatter.lines), 1)
                np.testing.assert_array_equal(scatter.collections[0].get_offsets(),
                                              np.column_stack((result[f'{name}_signed_balance'],
                                                               result[f'{name}_absolute_area_spikes'])))
                self.assertFalse(scatter.collections[0].get_clip_on())
                np.testing.assert_allclose([patch.get_height() for patch in hist.patches],
                                           result[f'{name}_balance_bin_fractions'])
                np.testing.assert_allclose([patch.get_x() for patch in hist.patches], bins[:-1])
                self.assertEqual(right.get_ylim(), scatter.get_ylim())
                self.assertEqual(right.get_yscale(), 'linear')
                self.assertEqual(right.get_xlabel(), 'Fraction')
                self.assertFalse(any(label.get_visible() for label in right.get_yticklabels()))
                np.testing.assert_allclose([patch.get_y() for patch in right.patches], strength_edges[:-1])
                np.testing.assert_allclose([patch.get_height() for patch in right.patches], .05)
                np.testing.assert_allclose([patch.get_width() for patch in right.patches],
                                           result[f'{name}_strength_bin_fractions'])
        finally:
            plt.close(fig)

    def test_population_rows_pair_each_heatmap_with_its_scatter(self):
        from paper.supp_twin_saccade_modulation import figure
        self.assertTrue(hasattr(figure, 'draw_population_summary'))
        lags = np.array([0., 50., 100., 150., 200.])
        gain = np.array([[-2., -2., 3., 3., 3.]])
        additive = gain / 10
        population = dict(gain_hz=gain, additive_hz=additive, lags_ms=lags)
        self.assertTrue(hasattr(figure, 'integrated_effects'))
        areas = figure.integrated_effects(lags, gain, additive)
        fig = plt.figure(figsize=(14, 16))
        try:
            grid = fig.add_gridspec(4, 12, left=.075, right=.96, top=.97, bottom=.08,
                                    height_ratios=[2.1, 1.15, 1.8, 1.8], hspace=.34, wspace=.85)
            axes = figure.draw_population_summary(fig, grid, population, areas, 4.)
            fig.canvas.draw()
            for heatmap, scatter, matrix in ((axes[0], axes[1], gain),
                                             (axes[2], axes[3], additive)):
                np.testing.assert_array_equal(heatmap.images[0].get_array(), matrix)
                self.assertEqual(heatmap.get_subplotspec().rowspan.start,
                                 scatter.get_subplotspec().get_topmost_subplotspec().rowspan.start)
                self.assertLess(heatmap.get_position().x1, scatter.get_position().x0)
                self.assertEqual(heatmap.get_subplotspec().colspan.start, 0)
                self.assertEqual(heatmap.get_subplotspec().colspan.stop, 7)
                self.assertEqual(scatter.get_subplotspec().get_topmost_subplotspec().colspan.start, 7)
                self.assertEqual(scatter.get_subplotspec().get_topmost_subplotspec().colspan.stop, 12)
                self.assertGreater(heatmap.get_position().width, scatter.get_position().width)
            self.assertGreater(axes[0].get_position().y0, axes[2].get_position().y1)
            self.assertEqual(axes[1].get_xlim(), axes[3].get_xlim())
            self.assertEqual(axes[1].get_ylim(), axes[3].get_ylim())
            self.assertEqual(len(axes), 8)
            for scatter, hist, right, map_ax in ((axes[1], axes[4], axes[6], axes[0]),
                                                 (axes[3], axes[5], axes[7], axes[2])):
                self.assertEqual(scatter.get_xlim(), hist.get_xlim())
                self.assertAlmostEqual(scatter.get_position().x0, hist.get_position().x0)
                self.assertAlmostEqual(scatter.get_position().x1, hist.get_position().x1)
                self.assertGreater(scatter.get_position().y0, hist.get_position().y1)
                self.assertAlmostEqual(hist.get_position().y0, map_ax.get_position().y0)
                self.assertLess(hist.get_position().height, scatter.get_position().height / 3)
                self.assertGreater(right.get_position().x0, scatter.get_position().x1)
                self.assertAlmostEqual(right.get_position().y0, scatter.get_position().y0)
                self.assertAlmostEqual(right.get_position().y1, scatter.get_position().y1)
                self.assertEqual(right.get_ylim(), scatter.get_ylim())
                renderer = fig.canvas.get_renderer()
                self.assertGreater(scatter.yaxis.label.get_window_extent(renderer).x0,
                                   map_ax.get_window_extent(renderer).x1 + 5)
        finally:
            plt.close(fig)

    def test_population_uses_cached_rows_without_separator(self):
        gain = np.array([[1., -2.], [3., 4.]])
        additive = np.array([[5., 6.], [-7., 8.]])
        lags = np.array([-100., 200.])
        fig, axes = plt.subplots(1, 2)
        try:
            draw_population(axes, gain, additive, lags, 9.)
            np.testing.assert_array_equal(axes[0].images[0].get_array(), gain)
            np.testing.assert_array_equal(axes[1].images[0].get_array(), additive)
            for ax in axes:
                self.assertEqual(ax.get_xlabel(), 'Time from saccade onset (ms)')
                self.assertEqual(len(ax.lines), 1)
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 0])
                self.assertEqual((ax.images[0].norm.vmin, ax.images[0].norm.vmax), (-9., 9.))
                self.assertEqual(ax.get_xlim(), (-100, 200))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
