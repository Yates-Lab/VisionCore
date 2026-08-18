import json

import numpy as np

from paper.fig4.spatiotemporal_tuning.build_real_fixation_bank import (
    ANALYSIS_SAMPLES,
    HISTORY_SAMPLES,
    complete_fixation_windows,
    guarded_fixations,
    ij_pixels_to_centered_xy_degrees,
    quantile_session_balanced_selection,
    trace_metrics,
)


def test_guarded_fixations_discards_corrupt_long_detector_event(tmp_path):
    path = tmp_path / "saccades.json"
    path.write_text(
        json.dumps(
            [
                {"start_time": 0.2, "end_time": 10**9},
                {"start_time": 0.5, "end_time": 0.55},
            ]
        )
    )
    fixations = guarded_fixations(
        [{"trial_idx": 3, "trial_start_ephys": 0.0, "trial_stop_ephys": 1.0}],
        path,
        guard_seconds=0.05,
    )
    assert len(fixations) == 2
    assert fixations[0]["fixation_stop_ephys"] == 0.45
    assert np.isclose(fixations[1]["fixation_start_ephys"], 0.60)


def test_complete_windows_keep_scored_intervals_nonoverlapping() -> None:
    fixation = {
        "fixation_start_ephys": 10.0,
        "fixation_stop_ephys": 13.6,
        "fixation_index": 4,
    }
    windows = complete_fixation_windows(
        fixation, required_seconds=1.25, stride_seconds=1.0
    )
    assert [row["window_start_ephys"] for row in windows] == [10.0, 11.0, 12.0]
    # The scored interval begins 250 ms into each window and lasts one second.
    scored = [(row["window_start_ephys"] + 0.25, row["window_stop_ephys"]) for row in windows]
    assert all(left[1] <= right[0] for left, right in zip(scored[:-1], scored[1:]))
    longer = {**fixation, "fixation_stop_ephys": 13.8}
    longer_windows = complete_fixation_windows(
        longer, required_seconds=1.25, stride_seconds=1.0
    )
    assert [row["window_start_ephys"] for row in longer_windows] == [10.0, 11.0, 12.0]


def test_pixel_conversion_uses_right_up_xy_and_analysis_center():
    position = np.asarray(
        [[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]], dtype=float
    )
    result = ij_pixels_to_centered_xy_degrees(
        position, ppd=2.0, center_from=slice(1, 3)
    )
    np.testing.assert_allclose(result[:, 0], [-1.5, -0.5, 0.5])
    np.testing.assert_allclose(result[:, 1], [0.75, 0.25, -0.25])


def test_trace_metrics_use_only_scored_interval():
    trace = np.zeros((HISTORY_SAMPLES + ANALYSIS_SAMPLES, 2), dtype=float)
    trace[:HISTORY_SAMPLES, 0] = 100.0
    metrics = trace_metrics(trace)
    assert metrics["analysis_rms_radius_deg"] == 0.0
    assert metrics["analysis_path_length_deg"] == 0.0
    assert metrics["analysis_speed_peak_deg_s"] == 0.0


def test_trace_metrics_exposes_residual_saccade_like_peak_speed():
    trace = np.zeros((HISTORY_SAMPLES + ANALYSIS_SAMPLES, 2), dtype=float)
    trace[HISTORY_SAMPLES + 10 :, 0] = 0.25
    metrics = trace_metrics(trace)
    assert np.isclose(metrics["analysis_speed_peak_deg_s"], 60.0)


def test_quantile_selection_covers_path_distribution_and_sessions():
    rows = []
    for index in range(30):
        rows.append(
            {
                "session": f"s{index % 5}",
                "analysis_path_length_deg": float(index),
            }
        )
    selected = quantile_session_balanced_selection(rows, 10, seed=4)
    values = np.asarray([row["analysis_path_length_deg"] for row in selected])
    assert len(selected) == 10
    assert values.min() <= 2
    assert values.max() >= 27
    assert len({row["session"] for row in selected}) >= 4
