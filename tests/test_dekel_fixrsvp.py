import numpy as np
import torch

from eval.eval_stack_utils import ccnorm_split_half_variable_trials
from paper.model_selection.evaluate_dekel_fixrsvp import (
    _audit_shared_ccnorm,
    _ccnorm_on_fixed_support,
    _data_score_mask,
    _predict_endpoints,
    _valid_score_mask,
    _validate_observations,
    _variance_explained_float64,
    figure3_endpoint_coordinates,
)
from paper.model_selection.evaluate_true240_fixrsvp import (
    _native_data_support,
    _native_pair_indices,
    _representative_examples,
)


def test_figure3_coordinates_drop_cross_trial_block_and_use_block_start():
    # Global phase-1 endpoints are [1, 3, 5].  The first block crosses from
    # trial 10 to trial 11 and must not become a spurious one-bin trial.  The
    # remaining causal blocks use their block-start Figure-3 coordinate.
    trial = np.array([10, 11, 11, 11, 11, 11])
    psth = np.array([8, 0, 1, 2, 3, 4])
    endpoints, coordinates = figure3_endpoint_coordinates(
        trial, psth, np.array([1, 3, 5]), factor=2
    )
    np.testing.assert_array_equal(endpoints, [3, 5])
    np.testing.assert_array_equal(coordinates, [0, 1])


def test_true240_pairing_matches_canonical_global_rebin_origin():
    # The legacy 120-Hz loader reshapes the entire source tensor starting at
    # row zero.  Trial-local PSTH parity would choose [1, 2] and [3, 4], but
    # the canonical global pairs are [0, 1], [2, 3], [4, 5]; after rejecting
    # cross-trial blocks, only [2, 3] remains.
    trial = np.array([10, 11, 11, 11, 11, 12, 12])
    psth = np.array([7, 0, 1, 2, 3, 0, 2])
    starts, ends = _native_pair_indices(trial, psth)
    np.testing.assert_array_equal(starts, [2])
    np.testing.assert_array_equal(ends, [3])


def test_native_support_intersects_figure3_and_native_filters():
    robs = np.ones((1, 4, 1), dtype=float)
    native_dfs = np.asarray([[[1.0], [0.0], [1.0], [1.0]]])
    reference_dfs = np.asarray([[[1.0], [0.0]]])

    support = _native_data_support(robs, native_dfs, reference_dfs)

    np.testing.assert_array_equal(
        support,
        np.asarray([[[True], [False], [False], [False]]]),
    )


def test_observation_alignment_requires_shape_mask_and_values():
    reference = {
        "robs_used": np.array([[[1.0], [np.nan], [2.0]]], dtype=np.float32)
    }
    exact = {"robs_used": reference["robs_used"].copy()}
    assert _validate_observations(exact, reference)["exact_match"]

    extra = {"robs_used": np.array([[[1.0], [0.0], [2.0]]], dtype=np.float32)}
    check = _validate_observations(extra, reference)
    assert not check["exact_match"]
    assert not check["finite_mask_match"]

    shifted = {"robs_used": np.array([[[1.0], [np.nan], [3.0]]], dtype=np.float32)}
    check = _validate_observations(shifted, reference)
    assert not check["exact_match"]
    assert check["max_abs_difference"] == 1.0


def test_valid_score_mask_rejects_nan_filters_and_missing_predictions():
    robs = np.array([[[1.0], [2.0], [3.0], [4.0], [np.nan]]])
    rhat = np.array([[[10.0], [20.0], [np.nan], [40.0], [50.0]]])
    dfs = np.array([[[1.0], [np.nan], [1.0], [0.0], [1.0]]])
    mask = _valid_score_mask(robs, rhat, dfs)
    np.testing.assert_array_equal(mask.ravel(), [True, False, False, False, False])


def test_valid_score_mask_requires_identical_shapes():
    with np.testing.assert_raises(ValueError):
        _valid_score_mask(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 2)))


def test_data_score_mask_treats_nan_filter_as_invalid_even_with_finite_data():
    robs = np.array([[[1.0], [2.0], [3.0]]])
    dfs = np.array([[[1.0], [np.nan], [0.0]]])
    np.testing.assert_array_equal(_data_score_mask(robs, dfs).ravel(), [True, False, False])


def test_ccnorm_audit_uses_shared_noise_ceiling_and_exact_identity():
    rng = np.random.default_rng(7)
    time_signal = np.sin(np.linspace(0, 3 * np.pi, 30))[None, :, None]
    robs = 2.0 + time_signal + 0.25 * rng.standard_normal((40, 30, 2))
    dfs = np.ones_like(robs)
    # A finite response paired with a NaN filter must not enter CCnorm.
    dfs[0, 0, 0] = np.nan
    prediction_a = np.broadcast_to(2.0 + 0.9 * time_signal, robs.shape).copy()
    prediction_b = np.broadcast_to(2.0 + 0.7 * time_signal, robs.shape).copy()

    candidate = _ccnorm_on_fixed_support(robs, prediction_a, dfs)
    reference = {
        "robs_used": robs,
        "rhat_used": prediction_b,
        "dfs_used": dfs,
        "ccnorm": np.zeros(2),
        "ccabs": np.zeros(2),
        "ccmax": np.zeros(2),
    }
    current = {
        "ccnorm_support": candidate["support"],
        "ccmax": candidate["ccmax"],
        "ccnorm_unstable": candidate["unstable"],
    }
    audited_reference, audit = _audit_shared_ccnorm(current, reference)

    assert audit["support_exact_match"]
    assert audit["ccmax_exact_match"]
    assert audit["stability_mask_exact_match"]
    np.testing.assert_allclose(candidate["ccmax"], audited_reference["ccmax"], rtol=0, atol=0)
    # Positive affine changes leave the PSTH correlation unchanged.  On the
    # shared data-only ceiling they must therefore leave CCnorm unchanged too.
    np.testing.assert_allclose(
        candidate["ccabs"], audited_reference["ccabs"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        candidate["ccnorm"], audited_reference["ccnorm"], rtol=0, atol=1e-12
    )
    valid = np.isfinite(candidate["ccnorm"])
    np.testing.assert_allclose(
        candidate["ccnorm"][valid],
        (candidate["ccabs"] / candidate["ccmax"])[valid],
        rtol=0,
        atol=1e-12,
    )


def test_ccnorm_stability_exclusion_is_model_independent():
    rng = np.random.default_rng(17)
    signal = np.sin(np.linspace(0, 4 * np.pi, 36))[None, :, None]
    robs = 2.0 + signal + 0.55 * rng.standard_normal((42, 36, 3))
    dfs = np.ones_like(robs)
    prediction_good = np.broadcast_to(2.0 + signal, robs.shape).copy()
    prediction_poor = rng.standard_normal(robs.shape)

    good = _ccnorm_on_fixed_support(robs, prediction_good, dfs)
    poor = _ccnorm_on_fixed_support(robs, prediction_poor, dfs)

    np.testing.assert_array_equal(good["unstable"], poor["unstable"])
    np.testing.assert_allclose(good["seed_delta"], poor["seed_delta"], equal_nan=True)


def test_ccnorm_audit_rejects_missing_prediction_on_data_support():
    robs = np.ones((40, 30, 1))
    rhat = np.ones_like(robs)
    dfs = np.ones_like(robs)
    rhat[0, 0, 0] = np.nan
    with np.testing.assert_raises(RuntimeError):
        _ccnorm_on_fixed_support(robs, rhat, dfs)


def test_ccnorm_utility_sanitizes_numeric_masks_without_mutating_inputs():
    rng = np.random.default_rng(11)
    signal = np.sin(np.linspace(0, 2 * np.pi, 30))[None, :, None]
    robs = 2.0 + signal + 0.2 * rng.standard_normal((40, 30, 1))
    rhat = np.broadcast_to(2.0 + signal, robs.shape).copy()
    dfs = np.ones_like(robs)
    dfs[:, 4, :] = np.nan
    dfs[:, 9, :] = 0.0
    robs_before = robs.copy()
    rhat_before = rhat.copy()
    explicit = np.isfinite(dfs) & (dfs != 0)

    numeric_result = ccnorm_split_half_variable_trials(
        robs, rhat, dfs, n_splits=5, rng=3
    )
    boolean_result = ccnorm_split_half_variable_trials(
        robs, rhat, explicit, n_splits=5, rng=3
    )

    for numeric, boolean in zip(numeric_result, boolean_result):
        np.testing.assert_allclose(numeric, boolean, equal_nan=True)
    np.testing.assert_array_equal(robs, robs_before)
    np.testing.assert_array_equal(rhat, rhat_before)


def test_variance_explained_is_invariant_to_float32_storage():
    rng = np.random.default_rng(23)
    observation = rng.poisson(0.25, size=(80, 120, 3)).astype(np.float32)
    prediction = (0.1 + 0.7 * observation + 0.05 * rng.random(observation.shape)).astype(
        np.float32
    )
    observation[:, :4] = np.nan
    prediction[:, :4] = np.nan

    from_float32 = _variance_explained_float64(prediction, observation)
    from_float64 = _variance_explained_float64(
        prediction.astype(np.float64), observation.astype(np.float64)
    )

    np.testing.assert_allclose(from_float32, from_float64, rtol=0, atol=0)


def test_representative_examples_cover_typical_and_model_advantage_cases():
    candidate = np.asarray([0.15, 0.30, 0.45, 0.60, 0.75])
    reference = np.asarray([0.35, 0.35, 0.40, 0.55, 0.65])
    result = {
        "session": "session",
        "ccabs": candidate,
        "ccnorm": candidate,
    }
    ryan = {
        "session": "session",
        "ccabs": reference,
        "ccnorm": reference,
    }

    chosen = _representative_examples([result], [ryan], n_examples=4)

    assert [row[0] for row in chosen] == [
        "strong candidate",
        "typical",
        "candidate advantage",
        "Twin advantage",
    ]
    assert len({row[3] for row in chosen}) == 4


def test_predict_endpoints_uses_spike_trained_behavior_contract():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_input = False
            self.received = None

        def forward(self, stim, dataset_idx, behavior=None, history=None):
            self.received = (stim, dataset_idx, behavior, history)
            return torch.ones(stim.shape[0], 1)

    model = RecordingModel()
    dset = {
        "stim": torch.arange(6 * 4, dtype=torch.float32).reshape(6, 1, 2, 2),
        "behavior": torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3),
    }
    indices = np.array([2, 4], dtype=np.int64)
    prediction = _predict_endpoints(
        model,
        dset,
        indices,
        torch.tensor([0, 1]),
        dataset_idx=3,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert prediction.shape == (2, 1)
    stim, dataset_idx, behavior, history = model.received
    assert stim.shape == (2, 1, 2, 2, 2)
    assert dataset_idx == 3
    assert history is None
    torch.testing.assert_close(behavior, dset["behavior"][indices])
