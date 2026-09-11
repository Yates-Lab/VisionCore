from __future__ import annotations

import json

import numpy as np
import pytest
from paper.fig4.upstream.real_trace_matrix.model import RESPONSE_UNITS

from paper.fig4.spatiotemporal_tuning.build_matrix_spectral_replay import (
    _load_matrix_responses,
    _matrix_contract,
    _pack_response_conditions,
)


def _model_provenance(checkpoint: str = "a" * 64) -> dict:
    return {
        "response_units": dict(RESPONSE_UNITS),
        "model": {
            "checkpoint_sha256": checkpoint,
            "dataset_configs_sha256": "b" * 64,
        }
    }


def test_load_matrix_responses_preserves_image_major_trace_order(tmp_path):
    n_images, n_traces, n_units = 2, 3, 4
    moving = np.arange(n_images * n_traces * n_units, dtype=np.float32).reshape(
        n_images * n_traces, n_units
    )
    stable = np.arange(n_images * n_units, dtype=np.float32).reshape(
        n_images, n_units
    )
    for filename in (
        "mean_rate_matrix.npy",
        "expected_spikes_matrix.npy",
        "ssi_matrix.npy",
    ):
        np.save(tmp_path / filename, moving)
    for filename in (
        "stabilized_mean_rate_by_image.npy",
        "stabilized_expected_spikes_by_image.npy",
        "stabilized_ssi_by_image.npy",
    ):
        np.save(tmp_path / filename, stable)

    result = _load_matrix_responses(
        tmp_path,
        n_images=n_images,
        n_traces=n_traces,
        n_units=n_units,
    )

    assert result["moving_mean_rate"].shape == (2, 3, 4)
    np.testing.assert_array_equal(result["moving_mean_rate"][1, 0], moving[3])
    np.testing.assert_array_equal(result["stable_map_ssi"], stable)


def test_condition_archive_preserves_hz_and_full_movie_counts():
    moving_hz = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    stable_hz = np.arange(30, 38, dtype=np.float32).reshape(2, 4)
    duration_seconds = 60 / 240
    responses = {
        "moving_mean_rate": moving_hz,
        "stable_mean_rate": stable_hz,
        "moving_expected_spikes": moving_hz * duration_seconds,
        "stable_expected_spikes": stable_hz * duration_seconds,
        "moving_map_ssi": np.full_like(moving_hz, 0.2),
        "stable_map_ssi": np.full_like(stable_hz, 0.1),
    }
    units = np.array([2, 0])

    result = _pack_response_conditions(responses, units)

    assert result["mean_rate"].shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(result["mean_rate"][:, :, 1], moving_hz[:, :, units])
    np.testing.assert_array_equal(
        result["mean_rate"][:, :, 0],
        np.broadcast_to(stable_hz[:, None, units], (2, 3, 2)),
    )
    np.testing.assert_array_equal(
        result["expected_spikes"], result["mean_rate"] * duration_seconds
    )


def test_matrix_contract_requires_matching_checkpoint_and_filtered_gate(tmp_path):
    trace_provenance = {
        "filter": {"kind": "continuous zero-phase filter"},
        "spectral_filter_qc": {"stopband_suppression_gate": True},
    }
    shard = {
        "model_provenance": _model_provenance(),
        "trace_bank": {"trace_provenance": trace_provenance},
        "n_timepoints": 60,
        "bin_seconds": 1.0 / 240.0,
    }
    (tmp_path / "summary.json").write_text(
        json.dumps({"shard_summaries": [shard]}), encoding="utf-8"
    )
    (tmp_path / "stabilized_baseline_summary.json").write_text(
        json.dumps({"model_provenance": _model_provenance()}), encoding="utf-8"
    )

    _, provenance = _matrix_contract(tmp_path)

    assert provenance["checkpoint_sha256"] == "a" * 64
    assert provenance["n_timepoints"] == 60
    assert provenance["trace_provenance"] == trace_provenance


def test_matrix_contract_rejects_checkpoint_mismatch(tmp_path):
    shard = {
        "model_provenance": _model_provenance(),
        "trace_bank": {
            "trace_provenance": {
                "filter": {"kind": "continuous zero-phase filter"},
                "spectral_filter_qc": {"stopband_suppression_gate": True},
            }
        },
        "n_timepoints": 60,
        "bin_seconds": 1.0 / 240.0,
    }
    (tmp_path / "summary.json").write_text(
        json.dumps({"shard_summaries": [shard]}), encoding="utf-8"
    )
    (tmp_path / "stabilized_baseline_summary.json").write_text(
        json.dumps({"model_provenance": _model_provenance("c" * 64)}),
        encoding="utf-8",
    )

    try:
        _matrix_contract(tmp_path)
    except ValueError as error:
        assert "different checkpoints" in str(error)
    else:
        raise AssertionError("checkpoint mismatch was not rejected")


def test_matrix_contract_rejects_undeclared_response_units(tmp_path):
    model = _model_provenance()
    del model["response_units"]
    (tmp_path / "summary.json").write_text(json.dumps({"shard_summaries": [{"model_provenance": model}]}))
    (tmp_path / "stabilized_baseline_summary.json").write_text(json.dumps({"model_provenance": model}))
    with pytest.raises(ValueError, match="response.units"):
        _matrix_contract(tmp_path)
