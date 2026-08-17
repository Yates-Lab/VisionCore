import json

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning.analyze_native_controlled_scaling import (
    paired_image_interval,
    percent_from_zero,
    pooled_components,
    tuning_groups,
)
from paper.fig4.spatiotemporal_tuning.run_controlled_scaling import (
    matrix_trace_time_contract,
)


def test_matrix_trace_time_contract_preserves_source_interval(tmp_path):
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "summary.json").write_text(json.dumps({
        "trace_time_contract": {
            "source_trace_rate_hz": 120,
            "source_trace_samples": 40,
            "model_output_rate_hz": 240,
            "scored_trace_samples": 80,
        }
    }))
    contract = matrix_trace_time_contract(
        matrix,
        np.zeros((3, 40, 2), dtype=np.float32),
        model_output_rate_hz=240,
    )
    assert contract["scored_samples_per_source_trace_sample"] == 2
    assert contract["scored_trace_samples"] == 80
    assert contract["analysis_interval_seconds"] == 1 / 3


def test_matrix_trace_time_contract_rejects_compressed_trace_shape(tmp_path):
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "summary.json").write_text(json.dumps({
        "trace_time_contract": {
            "source_trace_rate_hz": 120,
            "source_trace_samples": 40,
            "model_output_rate_hz": 240,
            "scored_trace_samples": 80,
        }
    }))
    try:
        matrix_trace_time_contract(
            matrix,
            np.zeros((3, 80, 2), dtype=np.float32),
            model_output_rate_hz=240,
        )
    except ValueError as error:
        assert "disagrees with the matrix timing contract" in str(error)
    else:
        raise AssertionError("A wrongly expanded retained trace was accepted")


def test_matrix_trace_time_contract_reads_merged_shard_provenance(tmp_path):
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    contract = {
        "source_trace_rate_hz": 120,
        "source_trace_samples": 40,
        "model_output_rate_hz": 240,
        "scored_trace_samples": 80,
    }
    (matrix / "summary.json").write_text(json.dumps({
        "analysis": "merged",
        "shard_summaries": [
            {"trace_time_contract": contract},
            {"trace_time_contract": contract},
        ],
    }))
    observed = matrix_trace_time_contract(
        matrix,
        np.zeros((3, 40, 2), dtype=np.float32),
        model_output_rate_hz=240,
    )
    assert observed["analysis_interval_seconds"] == 1 / 3


def test_controlled_scaling_pools_expected_spike_weighted_ssi():
    # Two images, one trace, two scales, two units.  The high-weight unit must
    # dominate; this catches accidental arithmetic averaging across units.
    ssi = np.asarray([
        [[ [1.0, 10.0], [2.0, 10.0] ]],
        [[ [1.0, 10.0], [2.0, 10.0] ]],
    ])
    expected = np.asarray([
        [[ [100.0, 1.0], [100.0, 1.0] ]],
        [[ [100.0, 1.0], [100.0, 1.0] ]],
    ])
    numerator, denominator = pooled_components(ssi, expected, np.asarray([0, 1]))
    result = percent_from_zero(numerator, denominator)
    expected_percent = 100.0 * ((210.0 / 101.0) - (110.0 / 101.0)) / (110.0 / 101.0)
    np.testing.assert_allclose(result, [0.0, expected_percent])


def test_controlled_scaling_bootstrap_is_paired_and_zero_is_exact():
    numerator = np.asarray([[1.0, 1.1], [2.0, 2.4], [3.0, 3.9]])
    denominator = np.ones_like(numerator)
    low, high = paired_image_interval(numerator, denominator, n_bootstrap=200, seed=3)
    assert low[0] == 0.0
    assert high[0] == 0.0
    assert low[1] > 0.0


def test_controlled_scaling_uses_weighted_center_sf_tertiles(tmp_path):
    path = tmp_path / "robust.csv"
    pd.DataFrame({
        "unit_index": [4, 1, 3, 0, 2, 5, 6],
        "low_sf_censored": [False, True, False, True, False, False, False],
        "weighted_center_sf_cpd": [4.0, 1.0, 3.0, 0.5, 2.0, 5.0, 6.0],
    }).to_csv(path, index=False)
    groups = tuning_groups(path, 7, mode="weighted_center_tertiles")
    np.testing.assert_array_equal(groups["lower-SF tertile"], [0, 1])
    np.testing.assert_array_equal(groups["middle-SF tertile"], [2, 3, 4])
    np.testing.assert_array_equal(groups["higher-SF tertile"], [5, 6])
