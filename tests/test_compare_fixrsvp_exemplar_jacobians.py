import numpy as np
import torch

from paper.model_selection.compare_fixrsvp_exemplar_jacobians import (
    _aligned_observation,
    _exact_jacobian,
    _resolve_rows,
)


def test_native_pair_rows_follow_the_scored_120hz_coordinates():
    endpoint_map = {
        "good_trials": np.asarray([10, 12]),
        "endpoint_kind": "native_pair",
        "endpoint_trials": np.asarray([10, 10, 12, 12]),
        "endpoint_psth": np.asarray([3, 4, 3, 4]),
        "pair_start": np.asarray([20, 22, 40, 42]),
        "pair_end": np.asarray([21, 23, 41, 43]),
        "lags": np.asarray([0, 1, 2]),
    }
    trials, endpoints = _resolve_rows(
        endpoint_map, np.asarray([0, 1]), np.asarray([4, 3])
    )
    np.testing.assert_array_equal(trials, [10, 12])
    np.testing.assert_array_equal(endpoints, [[22, 23], [40, 41]])


def test_aligned_observation_sums_native_pair_counts():
    robs = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    endpoints = np.asarray([[1, 2], [5, 6]])
    observed = _aligned_observation({"robs": robs}, endpoints, 1)
    np.testing.assert_array_equal(
        observed,
        [float(robs[1, 1] + robs[2, 1]), float(robs[5, 1] + robs[6, 1])],
    )


class _SumModel(torch.nn.Module):
    log_input = False

    def forward(self, stimulus, dataset_idx, behavior, history, output_behavior):
        return stimulus.sum(dim=(1, 2, 3, 4), keepdim=False)[:, None] + 1.0


def test_native_pair_jacobian_uses_one_shared_movie_timeline():
    # One-pixel, positive movie makes the analytic derivative transparent.
    stimulus = torch.arange(1, 11, dtype=torch.float32).reshape(10, 1, 1, 1)
    dset = {
        "stim": stimulus,
        "behavior": torch.zeros(10, 2),
        "robs": torch.zeros(10, 1),
    }
    endpoints = np.asarray([[4, 5], [7, 8]])
    jacobian, rate, effective_lags = _exact_jacobian(
        _SumModel(),
        dset,
        endpoints,
        np.asarray([0, 1, 2]),
        dataset_idx=0,
        unit_index=0,
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(effective_lags, [0, 1, 2, 3])
    expected_rate = np.asarray(
        [
            stimulus[[4, 3, 2]].sum().item()
            + stimulus[[5, 4, 3]].sum().item()
            + 2.0,
            stimulus[[7, 6, 5]].sum().item()
            + stimulus[[8, 7, 6]].sum().item()
            + 2.0,
        ]
    )
    np.testing.assert_allclose(rate, expected_rate)
    expected_weights = np.asarray([1.0, 2.0, 2.0, 1.0])
    np.testing.assert_allclose(
        jacobian[:, :, 0, 0],
        expected_weights[None, :] / expected_rate[:, None],
        rtol=1e-6,
    )
