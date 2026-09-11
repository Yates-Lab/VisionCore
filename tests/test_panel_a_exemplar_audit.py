from __future__ import annotations

from argparse import Namespace
import json

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning import audit_panel_a_exemplars as audit


def test_map_statistics_reports_rate_and_bits_per_spike() -> None:
    maps = np.asarray([[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 4.0]]]])

    result = audit.map_statistics(maps, output_rate_hz=240.0)

    np.testing.assert_allclose(result["rate_spikes_s"], [[240.0, 240.0]])
    np.testing.assert_allclose(result["ssi_bits_per_spike"], [[0.0, 2.0]])


def test_example_selection_rejects_extreme_percentage_winner() -> None:
    metrics = pd.DataFrame(
        {
            "unit_index": [1, 2],
            "stable_rate_spikes_s": [1.0, 4.0],
            "motion_rate_spikes_s": [10.0, 6.0],
            "rate_change_percent": [900.0, 50.0],
            "stable_ssi_bits_per_spike": [0.1, 0.2],
            "ssi_change_bits_per_spike": [0.8, 0.05],
            "ssi_change_percent": [800.0, 25.0],
            "normalized_map_rms_change": [2.0, 0.8],
        }
    )
    args = Namespace(
        minimum_stable_rate_spikes_s=0.5,
        minimum_stable_ssi=0.05,
        minimum_rate_change_percent=25.0,
        maximum_rate_change_percent=100.0,
        minimum_ssi_change_bits=0.02,
        maximum_ssi_change_bits=0.15,
        minimum_ssi_change_percent=10.0,
        maximum_ssi_change_percent=80.0,
        maximum_motion_rate_spikes_s=20.0,
        minimum_normalized_map_rms_change=0.12,
    )

    selected, tier = audit.choose_example(metrics, args)

    assert int(selected.unit_index) == 2
    assert tier == "bounded disclosed endpoint-clarity gates"


def test_example_selection_searches_predeclared_population_unit_set() -> None:
    metrics = pd.DataFrame(
        {
            "unit_index": [1, 2],
            "stable_rate_spikes_s": [4.0, 4.0],
            "motion_rate_spikes_s": [5.2, 6.0],
            "rate_change_percent": [30.0, 50.0],
            "stable_ssi_bits_per_spike": [0.2, 0.2],
            "ssi_change_bits_per_spike": [0.03, 0.06],
            "ssi_change_percent": [15.0, 30.0],
            "normalized_map_rms_change": [0.2, 0.8],
            "population_joint_rank_score": [4.0, 3.5],
        }
    )
    args = Namespace(
        minimum_stable_rate_spikes_s=0.5,
        minimum_stable_ssi=0.05,
        minimum_rate_change_percent=25.0,
        maximum_rate_change_percent=100.0,
        minimum_ssi_change_bits=0.02,
        maximum_ssi_change_bits=0.15,
        minimum_ssi_change_percent=10.0,
        maximum_ssi_change_percent=80.0,
        maximum_motion_rate_spikes_s=20.0,
        minimum_normalized_map_rms_change=0.12,
        population_unit_candidates=2,
    )

    selected, tier = audit.choose_example(metrics, args)

    assert int(selected.unit_index) == 2
    assert tier == "top-2 population-robust units + bounded disclosed endpoint-clarity gates"


def test_window_metrics_are_endpoint_relative() -> None:
    window = np.asarray([[10.0, -2.0], [10.25, -2.0], [10.5, -2.0]])

    result = audit._window_metrics(window, rate_hz=4.0)

    assert result["window_span_deg"] == 0.5
    assert result["window_path_length_deg"] == 0.5
    assert result["window_peak_speed_deg_s"] == 1.0


def test_trace_window_selection_uses_fixation_bank_identity_not_rucci_rows() -> None:
    traces = np.zeros((3, 8, 2), dtype=np.float32)
    traces[:, :, 0] = np.linspace(0.0, 0.6, 8)
    table = pd.DataFrame(
        {
            "trace_index": [0, 1, 2],
            "target_rate_hz": [8.0, 8.0, 8.0],
            "saved_microsaccade_count": [0, 1, 0],
            "session": ["Allen_day", "Logan_day", "Allen_other"],
        }
    )

    selected, windows = audit.select_trace_windows(
        traces,
        table,
        history_samples=3,
        per_subject=2,
        minimum_span=0.1,
        maximum_span=1.0,
        maximum_peak_speed=10.0,
    )

    assert set(selected.subject) == {"Allen", "Logan"}
    assert set(selected.trace_index).issubset(windows)
    assert np.all(np.isfinite(selected.power_centroid_hz))


def test_population_unit_effects_can_use_panel_b_matrix(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "n_images": 2,
                "n_traces": 2,
                "validated_common_provenance": {"bin_seconds": 1.0 / 240.0},
            }
        )
    )
    stable_rate = np.asarray([[1.0, 4.0], [1.0, 4.0]])
    stable_ssi = np.asarray([[0.10, 0.20], [0.10, 0.20]])
    moving_rate = np.asarray(
        [[1.2, 5.0], [1.4, 6.0], [1.3, 5.5], [1.5, 6.5]]
    )
    moving_ssi = np.asarray(
        [[0.11, 0.24], [0.12, 0.26], [0.13, 0.25], [0.14, 0.27]]
    )
    np.save(tmp_path / "mean_rate_matrix.npy", moving_rate)
    np.save(tmp_path / "ssi_matrix.npy", moving_ssi)
    np.save(tmp_path / "stabilized_mean_rate_by_image.npy", stable_rate)
    np.save(tmp_path / "stabilized_ssi_by_image.npy", stable_ssi)

    result = audit.population_unit_effects_from_matrix(tmp_path)

    assert result.unit_index.tolist() == [0, 1]
    assert np.all(result.population_rate_change_percent > 0)
    assert np.isclose(result.loc[0, "population_rate_change_spikes_s"], 0.35)
    assert np.all(result.population_ssi_change_percent > 0)
    assert np.all(result.population_joint_rank_score > 0)


def test_population_matrix_checkpoint_must_match_panel_a_model(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"selected-model")
    digest = audit.sha256(checkpoint)
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "summary.json").write_text(
        json.dumps(
            {
                "validated_common_provenance": {
                    "model_provenance.model.checkpoint_sha256": digest
                }
            }
        )
    )

    assert audit.assert_matrix_checkpoint(matrix, checkpoint) == digest

    checkpoint.write_bytes(b"different-model")
    with np.testing.assert_raises_regex(ValueError, "different checkpoints"):
        audit.assert_matrix_checkpoint(matrix, checkpoint)
