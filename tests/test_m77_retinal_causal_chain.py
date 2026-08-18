from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from paper.fig4.upstream.real_trace_matrix.model import _embed_time_lags
from paper.fig4.spatiotemporal_tuning.run_m77_retinal_causal_chain import (
    folded_dpss_mode_power,
    interpolate_tuning_temporal,
    lagged_movie_view,
    map_ssi,
    spectral_predictors,
    tuning_tensors,
)


def _complete_tuning_table() -> pd.DataFrame:
    rows = []
    for unit in (3, 7):
        for spatial in (1.0, 2.0):
            for orientation in (0.0, 90.0):
                static = unit + spatial + orientation / 90.0
                rows.append(
                    {
                        "unit_index": unit,
                        "spatial_cpd": spatial,
                        "temporal_hz": 0.0,
                        "probe_orientation_deg": orientation,
                        "mean_rate": static,
                        "response_amp_rms": 0.0,
                    }
                )
                for temporal in (2.0, 8.0):
                    rows.append(
                        {
                            "unit_index": unit,
                            "spatial_cpd": spatial,
                            "temporal_hz": temporal,
                            "probe_orientation_deg": orientation,
                            "mean_rate": static + temporal / 2.0,
                            "response_amp_rms": temporal + spatial,
                        }
                    )
    return pd.DataFrame(rows)


def test_tuning_tensor_is_signed_relative_to_matched_static_control() -> None:
    result = tuning_tensors(_complete_tuning_table())
    assert result["signed_rate_sensitivity"].shape == (2, 2, 2, 2)
    np.testing.assert_allclose(
        result["signed_rate_sensitivity"][:, :, 0], 240.0
    )
    np.testing.assert_allclose(
        result["signed_rate_sensitivity"][:, :, 1], 960.0
    )
    np.testing.assert_allclose(
        result["normalized_phase_rms"].sum(axis=(1, 2, 3)), 1.0
    )


def test_temporal_interpolation_preserves_nodes_and_censors_outside_bank() -> None:
    tuning = np.zeros((1, 1, 2, 1), dtype=float)
    tuning[0, 0, :, 0] = (2.0, 8.0)
    target = np.asarray((1.0, 2.0, 4.0, 8.0, 16.0))
    result = interpolate_tuning_temporal(
        tuning, np.asarray((2.0, 8.0)), target, normalize=False
    )
    np.testing.assert_allclose(result[0, 0, :, 0], (0.0, 2.0, 5.0, 8.0, 0.0))


def test_folded_dpss_static_component_is_numerically_zero() -> None:
    value = np.ones((4, 240), dtype=np.complex128) * (3.0 + 2.0j)
    _, power = folded_dpss_mode_power(value, 240.0)
    assert float(np.max(np.abs(power))) < 1e-20


def test_folded_dpss_recovers_complex_temporal_frequency() -> None:
    time = np.arange(240) / 240.0
    value = np.exp(2j * np.pi * 17.0 * time)[None]
    frequency, power = folded_dpss_mode_power(value, 240.0)
    peak = float(frequency[np.argmax(power[0])])
    assert abs(peak - 17.0) <= 1.0


def test_joint_projection_distinguishes_equal_marginals() -> None:
    power = np.zeros((2, 2, 1), dtype=float)
    power[0, 0, 0] = 1.0
    power[1, 1, 0] = 1.0
    tuning = np.zeros((2, 2, 2, 1), dtype=float)
    tuning[0, 0, 0, 0] = 1.0
    tuning[1, 0, 1, 0] = 1.0
    result = spectral_predictors(power, tuning, tuning)
    np.testing.assert_allclose(result["total_dynamic_power"], (2.0, 2.0))
    np.testing.assert_allclose(result["joint_passband_power"], (1.0, 0.0))
    # Both units have the same SF and TF marginals; only the joint tensor can
    # distinguish their conjunctions.
    np.testing.assert_allclose(
        result["sf_orientation_marginal_power"], (0.5, 0.5)
    )
    np.testing.assert_allclose(result["tf_marginal_power"], (0.5, 0.5))


def test_map_ssi_is_zero_for_uniform_map_and_positive_for_sharp_map() -> None:
    maps = np.asarray(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[4.0, 0.0], [0.0, 0.0]]],
        ]
    )
    information, mean = map_ssi(maps)
    np.testing.assert_allclose(mean[:, 0], (1.0, 1.0))
    assert information[0, 0] == 0.0
    assert information[1, 0] == 2.0


def test_strided_lag_view_exactly_matches_training_embedder() -> None:
    movie = torch.arange(9 * 3 * 2, dtype=torch.float32).reshape(9, 3, 2)
    expected = _embed_time_lags(movie, n_lags=4, torch=torch)
    observed = lagged_movie_view(movie, 4)
    torch.testing.assert_close(observed, expected)
