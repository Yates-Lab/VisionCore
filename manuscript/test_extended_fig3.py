"""Regression checks for the frozen-input Extended Data Figure 3 renderer."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

HERE = VISIONCORE_ROOT / 'manuscript'
DATA = HERE / 'analysis/extended_fig3'
SOURCE = (VISIONCORE_ROOT / 'outputs/stats/fig4_orientation_fix/bundle/figure4/'
          'all_available_population_spec/all_available_units.csv')


class ExtendedFigure3Test(unittest.TestCase):
    def test_frozen_inputs_match_main_cohort_and_three_bin_psth(self):
        self.assertTrue((DATA / 'plot_inputs.npz').is_file(), 'missing durable ED3 plot inputs')
        from manuscript.render_extended_fig3 import load_inputs
        saved, maps, psth, population, areas, means, intervals = load_inputs()
        with (DATA / 'complete_725_units.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        plotted = list(zip(population['sessions'].tolist(), population['cids'].tolist()))
        cohort = {(r['session'], int(r['cid'])) for r in rows}
        self.assertEqual(len(plotted), 725)
        self.assertEqual(len(set(plotted)), 725)
        self.assertEqual(set(plotted), cohort)
        self.assertEqual(len(set(population['sessions'])), 15)
        if SOURCE.exists():
            with SOURCE.open() as stream:
                authoritative = {(r['session'], int(r['cid'])) for r in csv.DictReader(stream)}
            self.assertEqual(cohort, authoritative)
        self.assertEqual(len(maps['trials']), 56)
        self.assertEqual(len(saved['example_event_ids']), 53)
        self.assertEqual(len(means['gain']), len(population['lags_ms']))
        for name in ('observed', 'predicted'):
            np.testing.assert_allclose(psth[f'{name}_raw_hz'], saved[f'example_psth_{name}_raw_hz'], equal_nan=True)
        from paper.supp_twin_saccade_modulation.figure import matched_psth
        for key, value in matched_psth(maps, bins=3).items():
            np.testing.assert_allclose(psth[key], value, equal_nan=True)
        for name, total, balance in [('gain', .4055278928989235, .29607536232097054),
                                     ('additive', .09154268640559168, .8694631214298871)]:
            self.assertAlmostEqual(float(np.median(areas[f'{name}_absolute_area_spikes'])), total)
            self.assertAlmostEqual(float(np.median(abs(areas[f'{name}_signed_balance']))), balance)

    def test_statistics_export_uses_plotted_arrays_and_metadata(self):
        from manuscript import render_extended_fig3 as ed3
        self.assertTrue(hasattr(ed3, 'statistics_tex'), 'missing ED3 stats export')
        saved, maps, _, population, areas, _, _ = ed3.load_inputs()
        metadata = json.loads((DATA / 'full_A_to_H_B_three_bin_725_metadata.json').read_text())
        text = ed3.statistics_tex(saved, population, areas, metadata)
        for key, value in [('EDThreeTrials', '56'), ('EDThreeUnits', '725'),
                           ('EDThreeSmoothingBins', 'three'), ('EDThreeSmoothingMs', '12.5'),
                           ('EDThreeBootstrapDraws', '400'), ('EDThreeGainBalance', '0.296'),
                           ('EDThreeGainTotal', '0.406'), ('EDThreeAdditiveBalance', '0.869'),
                           ('EDThreeAdditiveTotal', '0.092')]:
            self.assertIn(f'\\newcommand{{\\{key}}}{{{value}}}\n', text)
        changed = {key: value.copy() for key, value in areas.items()}
        changed['gain_absolute_area_spikes'] += .1
        altered = ed3.statistics_tex(saved, population, changed, metadata)
        self.assertIn('\\newcommand{\\EDThreeGainTotal}{0.506}', altered)
        with self.assertRaisesRegex(ValueError, 'smoothing'):
            ed3.statistics_tex(saved, population, areas, dict(metadata, psth_smoothing_ms=99))

    def test_statistics_stale_check_never_writes(self):
        from manuscript import render_extended_fig3 as ed3
        self.assertTrue(hasattr(ed3, 'sync_statistics'), 'missing ED3 stale check')
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'extended_fig3_stats.tex'
            ed3.sync_statistics(target)
            expected = target.read_bytes()
            ed3.sync_statistics(target, check=True)
            target.write_text('stale stats\n')
            with self.assertRaisesRegex(ValueError, 'stale'):
                ed3.sync_statistics(target, check=True)
            self.assertEqual(target.read_bytes(), b'stale stats\n')
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'stale'):
                ed3.sync_statistics(target, check=True)
            self.assertFalse(target.exists())
            self.assertIn(b'EDThreeGainTotal', expected)

    def test_caption_and_makefile_bind_generated_statistics(self):
        tex = (HERE / 'main.tex').read_text()
        self.assertIn('\\input{extended_fig3_stats}', tex)
        caption = tex[tex.index('\\includegraphics[width=\\linewidth]{figures/extended_fig3.pdf}'):
                      tex.index('\\label{fig:extraretinal_saccade_modulation}')]
        for macro in ('Trials', 'Units', 'SmoothingBins', 'SmoothingMs', 'BootstrapDraws',
                      'GainBalance', 'GainTotal', 'AdditiveBalance', 'AdditiveTotal'):
            self.assertIn('\\EDThree' + macro, caption)
        make = (HERE / 'Makefile').read_text()
        self.assertIn('PYTHONPATH=.. $(SCI_PYTHON) render_extended_fig3.py --stats-only', make)
        self.assertIn('PYTHONPATH=.. $(SCI_PYTHON) render_extended_fig3.py --check-stats', make)

    def test_callout_and_appendix_follow_first_citation_order(self):
        tex = (HERE / 'main.tex').read_text()
        lead = ("Most of the model's single-trial predictive advantage persisted when the explicit "
                'extraretinal inputs were removed.')
        callout = ('We found that the extraretinal pathway learned structured modulation linked to '
                   'saccades (\\ref{fig:extraretinal_saccade_modulation}).')
        follow = 'However, zeroing the eye position and velocity terms cost very little in terms of captured variance'
        self.assertIn(lead + ' ' + callout + ' ' + follow, tex)
        self.assertLess(tex.index(callout), tex.index('\\ref{fig:stabilization_control}'))
        image = tex.index('\\includegraphics[width=\\linewidth]{figures/extended_fig3.pdf}')
        continued = tex.index('\\ContinuedFloat', image)
        saccade_label = tex.index('\\label{fig:extraretinal_saccade_modulation}', continued)
        stabilization_image = tex.index('\\includegraphics[width=\\linewidth]{figures/stabilization_control.pdf}')
        stabilization_label = tex.index('\\label{fig:stabilization_control}', stabilization_image)
        self.assertLess(image, continued)
        self.assertLess(continued, saccade_label)
        self.assertLess(saccade_label, stabilization_image)
        self.assertLess(stabilization_image, stabilization_label)

    def test_build_selection_needs_no_selected_analysis_bundle(self):
        env = dict(os.environ, VISIONCORE_MANUSCRIPT_SOURCE_ROOT='/nonexistent/selected-analysis')
        result = subprocess.run([sys.executable, str(HERE / 'render_figures.py'), 'extended3'],
                                cwd=VISIONCORE_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_renderer_writes_figure_without_provisional_heading(self):
        self.assertTrue((HERE / 'render_extended_fig3.py').is_file(), 'missing ED3 renderer')
        from manuscript.render_extended_fig3 import render
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'extended_fig3.pdf'
            render(output)
            self.assertTrue(output.is_file())
            text = subprocess.check_output(['pdftotext', str(output), '-'], text=True)
            self.assertIn('Stimulus-aligned PSTH', text)
            self.assertNotIn('Provisional', text)


if __name__ == '__main__':
    unittest.main()
