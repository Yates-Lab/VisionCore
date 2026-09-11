from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from paper.fig4.spatiotemporal_tuning.build_panel_b_population_path_length import summarize, summarize_unit_distributions, unit_level_effects
from paper.fig4.spatiotemporal_tuning._figure4_renderer import PANEL_LAYOUT, _direct_mechanism_values, _draw_panel_c_power, _draw_panel_e_population, _draw_panel_h_normalized

def _matrix_arrays() -> dict[str, np.ndarray]:
    moving_spikes = np.asarray([[[1.1, 2.2], [1.2, 2.4], [1.3, 2.6], [1.4, 2.8]], [[1.1, 2.2], [1.2, 2.4], [1.3, 2.6], [1.4, 2.8]]], dtype=float)
    moving_ssi = np.full_like(moving_spikes, 0.6)
    stable_spikes = np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=float)
    stable_ssi = np.full_like(stable_spikes, 0.5)
    return {'moving_rate': moving_spikes, 'moving_spikes': moving_spikes, 'moving_ssi': moving_ssi, 'stable_rate': stable_spikes, 'stable_spikes': stable_spikes, 'stable_ssi': stable_ssi}

def test_panel_b_population_summary_has_no_tuning_or_motion_context_split() -> None:
    frame = summarize(np.asarray([1.0, 2.0, 3.0, 4.0]), _matrix_arrays(), n_bins=4, n_bootstrap=20, seed=3)
    assert set(frame.outcome) == {'rate', 'SSI'}
    assert len(frame) == 8
    assert 'sf_group' not in frame
    assert 'context' not in frame
    assert frame.n_units.eq(2).all()

def test_panel_b_unit_distributions_preserve_per_unit_effects() -> None:
    effects = unit_level_effects(np.asarray([1.0, 2.0, 3.0, 4.0]), _matrix_arrays(), n_bins=4)
    np.testing.assert_allclose(effects['rate_percent'], np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]]))
    np.testing.assert_allclose(effects['ssi_percent'], 20.0)
    summary = summarize_unit_distributions(effects)
    assert set(summary.outcome) == {'rate', 'SSI'}
    assert summary.n_units.eq(2).all()
    assert summary.fraction_positive.eq(1.0).all()

def test_direct_mechanism_values_use_percent_modulation_and_within_unit_ranks() -> None:
    shape = (2, 4, 2, 2)
    rate = np.ones(shape, dtype=float)
    rate[:, :, 1] = 1.5
    spikes = rate.copy()
    ssi = np.full(shape, 0.5, dtype=float)
    ssi[:, :, 1] = 0.6
    power = np.zeros(shape, dtype=float)
    for trace in range(4):
        power[:, trace, 1] = float(trace + 1)
    values = _direct_mechanism_values({'motion_scales': np.asarray([0.0, 1.0]), 'mean_rate': rate, 'expected_spikes': spikes, 'map_ssi': ssi, 'joint_passband_power': power})
    np.testing.assert_allclose(values['rate_percent'], 50.0)
    np.testing.assert_allclose(values['ssi_percent'], 20.0)
    expected = np.asarray([12.5, 37.5, 62.5, 87.5])
    np.testing.assert_allclose(values['passband_percentile'][:, 0], expected)
    np.testing.assert_allclose(values['passband_percentile'][:, 1], expected)

def test_revised_layout_preserves_locked_page_and_eight_distinct_panels() -> None:
    assert tuple(PANEL_LAYOUT) == tuple('ABCDEFGH')
    for x, y, width, height in PANEL_LAYOUT.values():
        assert x >= 0 and y >= 0 and (width > 0) and (height > 0)
        assert x + width <= 12.0
        assert y + height <= 10.0

def test_panel_c_uses_equal_mass_conditional_power_without_passbands() -> None:
    spatial = np.asarray([1.0, 2.0, 4.0])
    temporal = np.asarray([1.0, 4.0, 16.0])
    tuning = {'measured': [{'spatial': spatial, 'temporal': temporal}, {'spatial': spatial, 'temporal': temporal}]}
    log_power = np.stack((np.asarray([[-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5], [-2.0, -3.0, -4.0]]), np.asarray([[-2.0, -1.0, -0.5], [-2.5, -1.5, -1.0], [-3.0, -2.0, -1.5]])))
    metrics = {'spatial': spatial, 'temporal': temporal, 'log_power': log_power, 'display_limits': (-4.0, -0.5), 'integrals': np.ones(2), 'regime_code': np.asarray([0, 0, 1, 1]), 'per_trace_power_centroid_hz': np.asarray([1.5, 2.0, 6.0, 8.0]), 'speed_deg_s': np.asarray([0.4, 0.6, 6.0, 9.0]), 'path_length_arcmin': np.asarray([20.0, 30.0, 300.0, 500.0]), 'microsaccade_count': np.asarray([0, 1, 0, 1])}
    figure = Figure(figsize=(4.16, 2.78))
    report = _draw_panel_c_power(figure, tuning, metrics)
    assert report['data_dependent'] is True
    assert report['equal_dynamic_mass_before_comparison'] is True
    assert report['passband_contours_drawn'] is False
    assert report['selected_by_microsaccade_label'] is False
    assert any((text.get_text() == 'C' for text in figure.texts))

def test_panel_e_draws_one_authoritative_contour_per_released_unit() -> None:
    tuning = np.asarray([1.5, 3.0])
    summary = pd.DataFrame({'unit_index': [0, 1], 'source_unit_index': [10, 11], 'audit_category': ['trusted', 'trusted'], 'validated_tuning': [True, True], 'validated_preferred_sf_cpd': tuning, 'validated_preferred_tf_hz': [8.0, 4.0], 'exact_twin_yu_preferred_sf_cpd': tuning, 'exact_twin_yu_preferred_tf_hz': [8.0, 4.0]})
    fits = pd.DataFrame({'unit_index': [0, 1], 'source_unit_index': [10, 11], 'preferred_sf_cpd': tuning, 'preferred_tf_hz': [8.0, 4.0], 'selected_model': ['R0', 'R0'], 'optimizer_success': [True, True], 'full_support_r2': [0.9, 0.9], 'sigma_s': [0.8, 0.8], 'zeta_s': [0.0, 0.0], 'sigma_t': [0.8, 0.8], 'zeta_t': [0.0, 0.0], 'q': [0.0, 0.0], 'measured_min_sf_cpd': [1.0, 1.0], 'measured_max_sf_cpd': [8.0, 8.0], 'measured_min_tf_hz': [1.0, 1.0], 'measured_max_tf_hz': [64.0, 64.0]})
    example_fits = fits.set_index('unit_index')
    figure = Figure(figsize=(3.0, 2.72))
    report = _draw_panel_e_population(figure, summary, example_fits, fits, population_policy='validated')
    assert report['n_passband_contours'] == 2
    assert report['n_filled_passbands'] == 2
    assert 0 < report['density_maximum_overlap_fraction'] <= 1
    assert 0 < report['density_darkest_gray'] < 1
    assert 'no KDE' in report['density_definition']
    assert report['contours_share_authoritative_code_with_panels_d_and_f'] is True

@pytest.mark.parametrize('has_phase', [False, True])
def test_panel_h_uses_gain_invariant_natural_unit_trajectories(has_phase) -> None:
    trajectory = {'stage_names': np.asarray(('S1 + phase', '+ S2', '+ S3 / output')), 'unit_temporal_modulation_points': np.asarray([[2.0, 3.0], [5.0, 7.0], [9.0, 11.0]]), 'unit_ssi_delta_bits_per_spike': np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])}
    summary = {'n_images': 4, 'n_traces': 10, 'n_image_trace_pairs': 40, 'readout_trajectory': {'final_stage_is_ordinary_model': True}, 'identity_checks': {'ordinary_output_max_abs': 1e-06}, 'cached_G_output_checks': {}}
    summary['readout_trajectory']['has_phase_branch'] = has_phase
    if not has_phase:
        trajectory['stage_names'][0] = 'S1'
    figure = Figure(figsize=(4.22, 2.72))
    report = _draw_panel_h_normalized(figure, trajectory, summary, n_bootstrap=20, seed=3)
    assert report['mean_rate_gain_plotted'] is False
    assert report['final_stage_is_ordinary_model'] is True
    assert report['cumulative_stage_labels'][-1] == '+ S3 / output'
    assert 'movie-wide mean' in report['normalization']
    assert ('phase' in report['intermediate_definition']) is has_phase
