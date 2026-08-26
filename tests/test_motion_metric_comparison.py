from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest
from paper.fig4.spatiotemporal_tuning import compare_motion_metrics as comparison

def test_quantile_bins_assign_every_trace_once_with_balanced_counts() -> None:
    values = np.asarray((8.0, 1.0, 3.0, 4.0, 2.0, 7.0, 6.0, 5.0))
    labels = comparison.quantile_bins(values, 3)
    assert labels.shape == values.shape
    assert set(labels.tolist()) == {0, 1, 2}
    counts = np.bincount(labels)
    assert int(counts.max() - counts.min()) <= 1
    assert np.max(values[labels == 0]) <= np.min(values[labels == 1])
    assert np.max(values[labels == 1]) <= np.min(values[labels == 2])

def test_crossed_bootstrap_preserves_constant_effect() -> None:
    values = np.full((12, 5), 7.5)
    center, low, high = comparison.crossed_bootstrap(values, n_bootstrap=50, rng=np.random.default_rng(4))
    assert center == 7.5
    assert low == 7.5
    assert high == 7.5

def test_high_sf_tail_audit_distinguishes_plateau_from_continued_acceleration() -> None:
    base = pd.DataFrame({'metric': ['path_length'] * 5, 'outcome': ['rate'] * 5, 'sf_group': ['high SF'] * 5, 'context': ['microsaccade'] * 5, 'bin_index': np.arange(5), 'x_median': [40, 70, 100, 130, 160], 'effect_median_percent': [5, 14, 22, 28, 28.5]})
    plateau = comparison.high_sf_rate_tail_audit(base)
    assert plateau['highest_motion_bin_does_not_show_indefinite_acceleration']
    base.loc[4, 'effect_median_percent'] = 40.0
    accelerating = comparison.high_sf_rate_tail_audit(base)
    assert not accelerating['highest_motion_bin_does_not_show_indefinite_acceleration']

def test_high_sf_tail_contrast_uses_crossed_image_trace_resampling() -> None:
    effect_by_bin = np.repeat([10.0, 20.0, 30.0, 40.0, 35.0], 3)
    moving_spikes = np.broadcast_to(1.0 + effect_by_bin[None, :, None] / 100.0, (2, 15, 1)).copy()
    arrays = {'moving_spikes': moving_spikes, 'moving_ssi': np.ones_like(moving_spikes), 'stable_spikes': np.ones((2, 1)), 'stable_ssi': np.ones((2, 1))}
    groups = {'high SF': np.asarray([0])}
    report = comparison.high_sf_rate_tail_contrast(arrays, groups, np.asarray(['microsaccade'] * 15), np.arange(15, dtype=float), n_bins=5, n_bootstrap=50, seed=2)
    assert report['upper_tail_contrast_percent'] == pytest.approx(0.0)
    assert report['upper_tail_contrast_crossed_ci95'] == pytest.approx([0.0, 0.0])

def test_population_effect_uses_spike_weighted_ssi_and_matched_baseline() -> None:
    reduced = {'moving_spikes': np.asarray([[2.0, 4.0], [2.0, 4.0]]), 'moving_information': np.asarray([[1.0, 2.0], [1.0, 2.0]]), 'stable_spikes': np.asarray([2.0, 2.0]), 'stable_information': np.asarray([0.5, 0.5])}
    rate, ssi = comparison.population_effects(reduced, np.asarray((0, 1)))
    assert rate == 50.0
    assert ssi == 100.0

def test_trace_context_separates_detected_microsaccades() -> None:
    table = pd.DataFrame({'rendered_n_microsaccade_events': [0, 2, 0, 1]})
    context = comparison.trace_context(table)
    assert context.tolist() == ['drift_only', 'microsaccade', 'drift_only', 'microsaccade']

def test_measured_sf_groups_requires_validated_trusted_coordinates(tmp_path) -> None:
    table = pd.DataFrame({'unit_index': np.arange(24), 'audit_category': ['trusted'] * 24, 'validated_tuning': [True] * 24, 'validated_preferred_sf_cpd': np.arange(1, 25, dtype=float), 'data_preferred_sf_cpd': np.arange(24, 0, -1, dtype=float)})
    path = tmp_path / 'tuning.csv'
    table.to_csv(path, index=False)
    groups = comparison.measured_sf_groups(path, 24)
    assert groups['sf_column'] == 'validated_preferred_sf_cpd'
    assert set(np.asarray(groups['all active']).tolist()) == set(range(24))
    assert set(np.asarray(groups['low SF']).tolist()) == set(range(8))
    assert set(np.asarray(groups['high SF']).tolist()) == set(range(16, 24))

def test_measured_sf_groups_rejects_unchecked_fallback(tmp_path) -> None:
    path = tmp_path / 'unchecked.csv'
    pd.DataFrame({'unit_index': np.arange(24), 'data_preferred_sf_cpd': np.arange(24)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='unchecked tuning fallbacks are forbidden'):
        comparison.measured_sf_groups(path, 24)

def test_population_bootstrap_resamples_units() -> None:
    moving_spikes = np.asarray([[[1.0, 3.0]]])
    arrays = {'moving_spikes': moving_spikes, 'moving_ssi': np.ones_like(moving_spikes), 'stable_spikes': np.ones((1, 2)), 'stable_ssi': np.ones((1, 2))}
    reduced = comparison._population_arrays(arrays, np.asarray([0, 1]))
    center, low, high = comparison.crossed_population_bootstrap(reduced, np.asarray([0]), n_bootstrap=500, rng=np.random.default_rng(11))
    assert center[0] == pytest.approx(100.0)
    assert low[0] < center[0] < high[0]

def test_matrix_filter_gate_requires_continuous_zero_phase_provenance() -> None:
    summary = {'shard_summaries': [{'trace_bank': {'trace_provenance': {'filter': {'kind': 'zero-phase analysis IIR before 240-Hz sampling', 'passband_hz': 20.0, 'stopband_hz': 30.0}}}}]}
    contract = comparison.matrix_trace_filter(summary)
    assert contract['passband_hz'] == 20.0
    assert comparison.matrix_trace_filter({'shard_summaries': [{}]}) is None

def test_matrix_filter_gate_accepts_a_single_shard_summary() -> None:
    summary = {'model_provenance': {'model': {'checkpoint_sha256': 'abc'}}, 'trace_bank': {'trace_provenance': {'filter': {'kind': 'zero-phase analysis IIR before 240-Hz sampling', 'passband_hz': 20.0}}}}
    assert comparison.matrix_shard_summaries(summary) == [summary]
    assert comparison.matrix_trace_filter(summary)['passband_hz'] == 20.0

def test_matrix_replay_selection_requires_matching_response_independent_contracts() -> None:
    contract = {'image_selection': {'kind': 'evenly spaced rows across image table'}, 'trace_selection': {'kind': 'evenly spaced rows across fixation bank', 'maximum_path_length_arcmin': 350.0}}
    summary = {'shard_summaries': [{'trace_bank': {'trace_provenance': contract}}, {'trace_bank': {'trace_provenance': contract}}]}
    assert comparison.matrix_replay_selection(summary) == contract
    summary['shard_summaries'][1]['trace_bank']['trace_provenance'] = {**contract, 'trace_selection': {'kind': 'response-selected'}}
    with pytest.raises(ValueError, match='different replay selections'):
        comparison.matrix_replay_selection(summary)

def test_selected_model_gate_rejects_a_matrix_from_another_checkpoint(tmp_path) -> None:
    model_spec = tmp_path / 'selected.yaml'
    model_spec.write_text('checkpoint_sha256: selected-digest\n', encoding='utf-8')
    summary = {'shard_summaries': [{'model_provenance': {'model': {'checkpoint_path': '/models/other/epoch.ckpt', 'checkpoint_sha256': 'other-digest'}}}]}
    with pytest.raises(ValueError, match='does not match the selected model'):
        comparison.assert_selected_matrix(summary, model_spec=model_spec)
    summary['shard_summaries'][0]['model_provenance']['model']['checkpoint_sha256'] = 'selected-digest'
    label, digest = comparison.assert_selected_matrix(summary, model_spec=model_spec)
    assert label == 'other'
    assert digest == 'selected-digest'

def test_validated_tuning_provenance_must_be_release_ready_and_checkpoint_matched(tmp_path) -> None:
    tuning = tmp_path / 'tuning.csv'
    tuning.write_text('unit_index,validated_preferred_sf_cpd\n0,1\n', encoding='utf-8')
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps({'release_ready': True, 'checkpoint_sha256': 'selected', 'trusted_tuning_contract': {'coordinate_assay': 'exact_cid_yu_sf_tf', 'sf_column': 'validated_preferred_sf_cpd'}}), encoding='utf-8')
    path, payload = comparison.assert_validated_tuning_provenance(tuning, None, checkpoint_sha256='selected')
    assert path == summary.resolve()
    assert payload['release_ready']
    with pytest.raises(ValueError, match='different checkpoints'):
        comparison.assert_validated_tuning_provenance(tuning, None, checkpoint_sha256='other')
