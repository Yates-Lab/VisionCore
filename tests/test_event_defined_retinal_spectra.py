import numpy as np
import pandas as pd
import pytest

from paper.fig4.spatiotemporal_tuning.eye_trace_filter import anti_alias_eye_position, filter_qc_passed, gaussian_filter_spec
from paper.fig4.spatiotemporal_tuning.refilter_fixation_bank import classify_window, audit_session_events
from paper.fig4.spatiotemporal_tuning.build_rucci_ensemble_power import event_defined_regimes


def test_gaussian_preserves_monotonic_transient_and_rejects_high_frequency_noise():
    time = np.arange(-1, 2, 1/540)
    movement = (1 + np.tanh((time-.5)/.006)) / 2
    frame = pd.DataFrame({'t_ephys': time, 'dpi_i': movement,
                          'dpi_j': np.sin(2*np.pi*180*time), 'valid': True})
    result = anti_alias_eye_position(frame, start_ephys=0, stop_ephys=1,
        target_rate_hz=240, passband_hz=20, stopband_hz=30, padding_seconds=.6,
        filter_family='gaussian', gaussian_sigma_seconds=.006)
    output = result['filtered_position_px']
    assert output.shape == (240, 2)
    assert np.min(np.diff(output[:, 0])) >= -1e-12
    assert output[:, 0].min() >= 0 and output[:, 0].max() <= 1 + 1e-12
    assert np.max(np.abs(output[:, 1])) < 1e-5
    with pytest.raises(ValueError, match='padding'):
        anti_alias_eye_position(frame, start_ephys=0, stop_ephys=1,
            target_rate_hz=240, passband_hz=20, stopband_hz=30, padding_seconds=.01,
            filter_family='gaussian')


def test_gaussian_cannot_inherit_an_elliptic_stopband_pass():
    provenance = {'filter': gaussian_filter_spec(),
                  'spectral_filter_qc': {'stopband_suppression_gate': True}}
    assert not filter_qc_passed(provenance)


def event_audit(events=()):
    return {'events': pd.DataFrame(events, columns=['event_id', 'start_time', 'end_time', 'amplitude_deg', 'verified']),
            'time': np.arange(0, 5, .001), 'valid': np.ones(5000, bool),
            'unassigned': np.zeros(5000, bool), 'coverage_start': 0,
            'coverage_stop': 5, 'screen_threshold_deg_s': 5}


def test_event_classes_exclude_large_boundary_and_unmatched_movements():
    assert classify_window(event_audit(), 1, 2)['event_class'] == 'drift'
    micro = event_audit([(0, 1.4, 1.44, .5, True)])
    assert classify_window(micro, 1, 2)['event_class'] == 'microsaccade'
    for event in [(0, 1.4, 1.44, 1.0, True), (0, .98, 1.02, .5, True),
                  (0, 1.4, 1.44, .5, False)]:
        assert classify_window(event_audit([event]), 1, 2)['event_class'] == 'excluded'
    micro['unassigned'][1700] = True
    assert classify_window(micro, 1, 2)['event_class'] == 'excluded'
    missing = event_audit()
    missing['coverage_start'] = 1.1
    assert classify_window(missing, 1, 2)['event_class'] == 'excluded'


def test_event_regimes_do_not_select_on_spectral_centroid_and_require_both_animals():
    frame = pd.DataFrame({'event_class': ['drift', 'microsaccade', 'drift', 'microsaccade', 'excluded'],
        'event_coverage_pass': [True]*4+[False], 'unmatched_rapid_motion': [False]*5,
        'verified_microsaccade_count': [0, 1, 0, 2, 0],
        'verified_microsaccade_max_amplitude_deg': [0, .4, 0, .9, 0]})
    animals = np.array(['A', 'A', 'B', 'B', 'A'])
    codes, report = event_defined_regimes(frame, animals)
    np.testing.assert_array_equal(codes, [0, 1, 0, 1, -1])
    assert report['selected_by_spectral_centroid'] is False
    assert report['animal_counts'] == {'A': [1, 1], 'B': [1, 1]}
    frame.loc[3, 'event_class'] = 'excluded'
    with pytest.raises(ValueError, match='each animal'):
        event_defined_regimes(frame, animals)


def test_saved_indices_survive_dropped_ddpi_rows_and_affine_calibration(tmp_path):
    import json
    time = np.arange(4000)/1000
    xy = np.zeros((4000, 2))
    transform = np.array([[1.1, .03], [-.04, .9]])
    events = []
    for start, delta in [(1000, [.3, 0]), (2000, [0, .4]), (3000, [.3, .4])]:
        stop = start+50
        ramp = np.clip((np.arange(4000)-start)/50, 0, 1)
        xy += ramp[:, None]*delta
        vector = np.asarray(delta) @ transform
        events.append({'start_idx': start, 'end_idx': stop, 'mu': (start+25)/1000,
            'start_time': start/1000, 'end_time': stop/1000,
            'start_x': 0, 'start_y': 0, 'end_x': vector[0], 'end_y': vector[1]})
    ddpi = pd.DataFrame({'t_ephys': time, 'dpi_j': xy[:, 0]*40,
                         'dpi_i': -xy[:, 1]*40, 'valid': True})
    ddpi = ddpi.drop(index=np.arange(100, 300))
    path = tmp_path/'saccades.json'
    path.write_text(json.dumps(events))
    result = audit_session_events(ddpi, path, ppd=40)
    assert result['events'].verified.all()
    np.testing.assert_allclose(result['events'].amplitude_deg, [.3, .4, .5], atol=1e-12)
    np.testing.assert_allclose(result['current_to_saved_coordinate_transform'], transform, atol=1e-12)
