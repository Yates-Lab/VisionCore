from __future__ import annotations

import numpy as np

from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad


def test_speed_threshold_mad_is_robust_to_small_drift():
    t = np.arange(64, dtype=np.float32)
    trace = np.column_stack([0.0002 * t, 0.0001 * t]).astype(np.float32)

    threshold = speed_threshold_mad(trace)

    assert threshold > 0.0
    assert threshold < 1.0


def test_microsaccade_event_detector_marks_high_speed_step():
    trace = np.zeros((32, 2), dtype=np.float32)
    trace[12:, 0] = 0.25

    events, event_mask, threshold = detect_microsaccade_events(trace, threshold_deg_s=10.0)

    assert threshold == 10.0
    assert len(events) == 1
    assert events[0]["onset"] == 12
    assert events[0]["offset"] == 12
    assert event_mask[12]


def test_microsaccade_event_detector_can_pad_events():
    trace = np.zeros((32, 2), dtype=np.float32)
    trace[12:, 0] = 0.25

    events, event_mask, _threshold = detect_microsaccade_events(
        trace,
        threshold_deg_s=10.0,
        pad_samples=2,
    )

    assert len(events) == 1
    assert events[0]["onset"] == 10
    assert events[0]["offset"] == 14
    assert np.all(event_mask[10:15])
