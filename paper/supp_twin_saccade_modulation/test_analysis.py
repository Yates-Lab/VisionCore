import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from paper.supp_twin_saccade_modulation.analysis import (
    cohort_rows, crop_maps, supported_effects, sort_population,
    validate_session_cache, provenance_paths,
)
from paper.supp_twin_saccade_modulation.plot import rank_example_rows


class CohortTest(unittest.TestCase):
    def test_ordinal_join_and_missing_scores_remain_in_population(self):
        aligned = [{'session': 'S', 'neuron_mask': np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    'rate_hz': np.array([3., 4., 1.] + [3.] * 8),
                    'psth_r2': np.array([.2, .3, .5] + [.2] * 8)}]
        scores = [{'session': 'S', 'neuron_mask': np.array([0]),
                   'ccnorm': np.array([.91]), 'ccmax': np.array([.9])}]
        rows = cohort_rows(aligned, scores, {'S': list(range(100, 111))})
        self.assertEqual([(r['cid'], r['ccnorm']) for r in rows[:2]], [(101, None), (100, .91)])
        self.assertEqual([r['example_eligible'] for r in rows[:2]], [False, True])
        self.assertEqual(len(rows), 10)


class ProvenanceTest(unittest.TestCase):
    def test_pinned_model_spec_and_manuscript_pointer_are_in_provenance(self):
        from paper.supp_twin_saccade_modulation.analysis import VISIONCORE_ROOT
        inputs = {key: Path('/example') / key for key in ('checkpoint', 'config', 'aligned', 'scores', 'pointer', 'spec')}
        inputs['pointer'] = VISIONCORE_ROOT / 'manuscript/analysis/selected_model_bundle.json'
        inputs['spec'] = Path('/example/rank1/model/no_phase_model.yaml')
        paths = {str(path) for path in provenance_paths(inputs)}
        self.assertTrue(any(path.endswith('selected_model_bundle.json') for path in paths))
        self.assertTrue(any(path.endswith('/rank1/model/no_phase_model.yaml') for path in paths))


class ResumeTest(unittest.TestCase):
    def test_corrupt_result_rejects_completed_manifest(self):
        from paper.supp_twin_saccade_modulation.analysis import save_json, sha256
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'components.npz').write_bytes(b'components')
            (root / 'results.json').write_text('{"valid":true}')
            save_json(root / 'manifest.json', dict(inputs={'x': 'a'}, source_hashes={'y': 'b'},
                      components_sha256=sha256(root / 'components.npz'),
                      results_sha256=sha256(root / 'results.json')))
            validate_session_cache(root, {'x': 'a'}, {'y': 'b'})
            (root / 'results.json').write_text('{"valid":false}')
            with self.assertRaises(AssertionError):
                validate_session_cache(root, {'x': 'a'}, {'y': 'b'})


class RankingTest(unittest.TestCase):
    def test_example_ranking_filters_support_and_breaks_ties_by_session_cid(self):
        rows = [dict(session=s, cid=c, ccnorm=score, example_eligible=ok,
                     supported_events=n) for s, c, score, ok, n in
                [('Z', 1, .98, True, 19), ('B', 2, .92, True, 20),
                 ('A', 1, .92, True, 20), ('C', 3, None, False, 30)]]
        self.assertEqual([(r['session'], r['cid']) for r in rank_example_rows(rows)],
                         [('A', 1), ('B', 2)])


class MapsTest(unittest.TestCase):
    def test_reach_endpoint_trim_common_invalid_lead_preserve_holes_and_time(self):
        t = np.r_[np.arange(6), np.arange(6), np.arange(4)] / 5
        trial = np.repeat([11, 12, 13], [6, 6, 4])
        valid = np.ones(len(t), bool)
        valid[[0, 1, 6, 7, 9, 10]] = False
        vals = {'full': np.arange(len(t), dtype=float), 'gain': np.arange(len(t), dtype=float)}
        maps = crop_maps(t, trial, valid, vals, .8, 5)
        self.assertEqual(maps['trials'], [11, 12])
        np.testing.assert_allclose(maps['time_s'], [.4, .6, .8])
        self.assertTrue(np.isnan(maps['full'][1, 1]))
        self.assertTrue(np.isfinite(maps['full'][0, 1]))
        self.assertEqual(maps['excluded_short'], 1)
        self.assertEqual(maps['invalid_endpoint_trials'], [12])


class EffectsTest(unittest.TestCase):
    def test_only_whole_shared_events_count_and_empty_session_is_valid(self):
        a = {'robs': np.zeros((6, 1)), 'dfs': np.ones((6, 1)),
             'r0': np.zeros((6, 1)), 'rfull': np.zeros((6, 1)),
             'r_gain': np.arange(6, dtype=float)[:, None], 'r_additive': np.ones((6, 1))}
        a['dfs'][4, 0] = 0
        result = supported_effects(a, np.array([[0, 1, 2], [3, 4, 5]]), 1, 0)
        self.assertEqual(result['n_events'], 1)
        np.testing.assert_allclose(result['gain'], [[0, 1, 2]])
        self.assertEqual(supported_effects(a, np.empty((0, 3), int), 1, 0)['n_events'], 0)

    def test_positive_gain_peak_sorted_earliest_bottom_with_nonpositive_separate(self):
        lags = np.array([-100, 0, 50, 100, 200])
        rows = [{'session': 'S', 'cid': cid} for cid in [1, 2, 3, 4]]
        gain = np.array([[0, -10, 2, 1, 0], [0, 1, 2, 4, 0],
                         [0, -3, -1, 0, -2], [0, 3, 1, 0, 0]], float)
        additive = np.arange(20).reshape(4, 5)
        ordered, g, a = sort_population(rows, gain, additive, lags)
        self.assertEqual([r['cid'] for r in ordered], [2, 1, 4, 3])
        np.testing.assert_array_equal(a, additive[[1, 0, 3, 2]])
        self.assertIsNone(ordered[-1]['positive_gain_peak_ms'])
        self.assertEqual([r['positive_gain_peak_ms'] for r in ordered[:-1]], [100, 50, 0])


if __name__ == '__main__':
    unittest.main()
