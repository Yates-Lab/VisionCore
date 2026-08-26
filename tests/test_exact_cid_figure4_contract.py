from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest
from paper.fig4.spatiotemporal_tuning.build_exact_cid_figure4_contract import _crossed_yu_examples, _population_spec
from paper.fig4.upstream.real_trace_matrix.model import population_unit_rows

class _PopulationView:

    def __init__(self, membership: np.ndarray, meta: dict) -> None:
        self.membership = membership
        self.cluster_membership = membership.copy()
        self.meta = meta
        self.name = 'exact-test'
        self.n_units = membership.shape[0]

def test_exact_cid_population_spec_is_one_to_one_and_unpooled(tmp_path) -> None:
    selected = pd.DataFrame({'unit_index': [0, 1, 2], 'source_unit_index': [10, 20, 30], 'canonical_channel': [1, 4, 7], 'session': ['a', 'b', 'c'], 'cid': [11, 22, 33]})
    release = {'source_provenance': {'n_canonical_units': 9, 'checkpoint_sha256': 'abc'}, 'source_measurement': str(tmp_path / 'measurement')}
    npz_path, json_path = _population_spec(selected, release, tmp_path, 'exact-test')
    with np.load(npz_path, allow_pickle=False) as archive:
        membership = archive['membership']
    metadata = json.loads(json_path.read_text(encoding='utf-8'))
    assert membership.shape == (3, 9)
    np.testing.assert_array_equal(membership.sum(axis=1), np.ones(3))
    np.testing.assert_array_equal(np.argmax(membership, axis=1), [1, 4, 7])
    assert metadata['pooling_mode'] == 'exact_identity'
    assert metadata['rr_clustering'] is False
    canonical = [{'channel': index, 'available': True} for index in range(9)]
    rows = population_unit_rows(_PopulationView(membership, metadata), canonical)
    assert [row['population_input_channel'] for row in rows] == [1, 4, 7]
    assert all((row['population_active'] for row in rows))

def test_crossed_examples_preserve_yu_coordinate_chain_of_custody() -> None:
    fits = pd.DataFrame({'unit_index': np.arange(12), 'source_unit_index': np.arange(100, 112), 'preferred_sf_cpd': np.geomspace(1.0, 8.0, 12), 'preferred_tf_hz': np.geomspace(32.0, 1.0, 12), 'full_support_r2': np.linspace(0.81, 0.99, 12), 'recorded_data_preferred_sf_cpd': np.geomspace(1.0, 8.0, 12), 'crossed_group': ['low recorded SF / high twin TF'] * 4 + ['middle'] * 4 + ['high recorded SF / low twin TF'] * 4, 'crossed_extremity_octaves': [4, 3, 2, 1] + [np.nan] * 4 + [1, 2, 3, 4], 'high_mode_count': 1, 'contrast_surface_corr': 0.99, 'peak_delta_f0_expected_count': 0.1})
    selected = _crossed_yu_examples(fits)
    assert set(selected.role) == {'low SF / high TF', 'high SF / low TF'}
    low = selected.loc[selected.role.eq('low SF / high TF')].iloc[0]
    high = selected.loc[selected.role.eq('high SF / low TF')].iloc[0]
    assert low.preferred_sf_cpd < high.preferred_sf_cpd
    assert low.recorded_data_preferred_sf_cpd < high.recorded_data_preferred_sf_cpd
    assert low.preferred_tf_hz > high.preferred_tf_hz
    assert low.example_selection_policy == 'crossed-extremity leader'
    assert high.example_selection_policy == 'log-SF/log-TF group medoid'
    high_group = fits.loc[fits.crossed_group.eq('high recorded SF / low twin TF')].copy()
    log_sf = np.log2(high_group.preferred_sf_cpd.to_numpy(dtype=float))
    log_tf = np.log2(high_group.preferred_tf_hz.to_numpy(dtype=float))
    distances = np.hypot(log_sf - np.median(log_sf), log_tf - np.median(log_tf))
    assert int(high.source_unit_index) == int(high_group.iloc[int(np.argmin(distances))].source_unit_index)
