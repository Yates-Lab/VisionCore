from __future__ import annotations

import inspect
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import readout_preactivation
from paper.fig4.mechanism_audit_v1.registration_mechanism import pq_semantics as pq


def coordinate_basis() -> np.ndarray:
    return np.eye(pq.N_CHANNELS, pq.RANK, dtype=np.float32)


def test_basis_and_leverage_invariants() -> None:
    basis = pq.validate_basis(coordinate_basis())
    leverage = pq.leverage_scores(basis)
    np.testing.assert_allclose(leverage[: pq.RANK], 1.0)
    np.testing.assert_allclose(leverage[pq.RANK :], 0.0)
    assert leverage.sum() == pytest.approx(8.0)
    assert pq.effective_participating_channels(leverage) == pytest.approx(8.0)


def test_distributed_leverage_has_larger_effective_count() -> None:
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.standard_normal((pq.N_CHANNELS, pq.RANK)))
    leverage = pq.leverage_scores(q.astype(np.float32))
    assert leverage.sum() == pytest.approx(8.0, abs=2e-5)
    assert pq.effective_participating_channels(leverage) > 60


def test_invalid_basis_is_rejected_without_silent_rotation() -> None:
    invalid = coordinate_basis()
    invalid[:, 1] = invalid[:, 0]
    with pytest.raises(ValueError, match="not orthonormal"):
        pq.validate_basis(invalid)


def test_projection_energy_separates_total_fraction_from_per_dimension() -> None:
    basis = coordinate_basis()
    value = np.zeros((2, pq.N_CHANNELS, 2, 2), dtype=np.float32)
    value[:, 0] = 2.0
    value[:, 10] = 3.0
    total, p_energy, q_energy, sites = pq.projection_energy(value, basis)
    assert sites == 8
    assert p_energy / sites == pytest.approx(4.0)
    assert q_energy / sites == pytest.approx(9.0)
    row = pq.energy_row(
        fold=0,
        contrast="synthetic",
        analysis="movement",
        scale=1.0,
        total_sum=total,
        p_sum=p_energy,
        q_sum=q_energy,
        sites=sites,
        definition="test",
    )
    assert row["candidate_p_total_fraction"] == pytest.approx(4 / 13)
    assert row["candidate_p_energy_per_dimension"] == pytest.approx(4 / 8)
    assert row["complementary_q_energy_per_dimension"] == pytest.approx(9 / 120)
    assert row["p_to_q_per_dimension_energy_ratio"] == pytest.approx((4 / 8) / (9 / 120))


def test_variance_energy_identity_preserves_spatial_mean_map() -> None:
    basis = coordinate_basis()
    values = np.zeros((3, pq.N_CHANNELS, 2, 2), dtype=np.float32)
    values[0, 0] = 1.0
    values[1, 0] = 2.0
    values[2, 0] = 3.0
    raw_total, raw_p, _, _ = pq.projection_energy(values, basis)
    total, p_energy, q_energy = pq._difference_energy_from_raw_moments(
        raw_total, raw_p, values.mean(axis=0), basis, len(values)
    )
    direct = np.square(values - values.mean(axis=0, keepdims=True)).sum() / len(values)
    assert total == pytest.approx(direct)
    assert p_energy == pytest.approx(direct)
    assert q_energy == pytest.approx(0.0)


def test_variance_decomposition_includes_covariance_and_cancellation() -> None:
    result = pq.variance_decomposition(
        np.asarray([4.0, 4.0]),
        np.asarray([1.0, 4.0]),
        np.asarray([1.0, -3.0]),
    )
    np.testing.assert_allclose(result["variance_full"], [7.0, 2.0])
    np.testing.assert_allclose(
        result["p_fraction_of_full_variance"]
        + result["q_fraction_of_full_variance"]
        + result["covariance_fraction_of_full_variance"],
        1.0,
    )
    assert result["covariance_fraction_of_full_variance"][1] < 0


def test_exact_tiled_linear_readout_adds_bias_once() -> None:
    generator = torch.Generator().manual_seed(9)
    h = torch.randn(3, pq.N_CHANNELS, 6, 6, generator=generator)
    basis = torch.as_tensor(coordinate_basis())
    h_p = pq._project(h, basis)
    h_q = h - h_p
    feature = torch.randn(5, pq.N_CHANNELS, generator=generator)
    space = torch.randn(5, 2, 2, generator=generator)
    bias = torch.randn(5, generator=generator)
    zero = torch.zeros_like(bias)
    full = readout_preactivation(h, feature, bias, space)
    a_p = readout_preactivation(h_p, feature, zero, space)
    a_q = readout_preactivation(h_q, feature, zero, space)
    np.testing.assert_allclose(
        full.numpy(),
        (bias[None, :, None, None] + a_p + a_q).numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_weight_reliance_is_projector_not_native_channel_selection() -> None:
    weights = np.zeros((3, pq.N_CHANNELS), dtype=np.float64)
    weights[0, 0] = 1.0
    weights[1, 9] = 2.0
    weights[2, 0] = 1.0
    weights[2, 9] = 1.0
    reliance = pq.weight_reliance(weights, coordinate_basis())
    np.testing.assert_allclose(reliance, [1.0, 0.0, 0.5])


def test_cross_moments_match_direct_per_unit_statistics() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(4, 3, 2, 2))
    y = 0.4 * x + rng.normal(size=x.shape)
    moments = pq.CrossMoments(3)
    moments.add(x[:2], y[:2])
    moments.add(x[2:], y[2:])
    var_x, var_y, covariance = moments.values()
    flattened_x = np.moveaxis(x, 1, 0).reshape(3, -1)
    flattened_y = np.moveaxis(y, 1, 0).reshape(3, -1)
    np.testing.assert_allclose(var_x, flattened_x.var(axis=1))
    np.testing.assert_allclose(var_y, flattened_y.var(axis=1))
    np.testing.assert_allclose(
        covariance,
        ((flattened_x - flattened_x.mean(axis=1, keepdims=True)) *
         (flattened_y - flattened_y.mean(axis=1, keepdims=True))).mean(axis=1),
    )


def test_map_recovery_is_complete_unclipped_effect_recovery() -> None:
    target = np.asarray([[[[1.0, 3.0]]], [[[2.0, 4.0]]]])
    reference = np.zeros((1, 1, 2))
    accumulator = pq.MapRecoveryAccumulator(n_units=1)
    weight = np.ones((2, 1))
    accumulator.add(target * 0.5, target, reference, weight=weight)
    assert accumulator.recovery()[0] == pytest.approx(0.75)
    accumulator = pq.MapRecoveryAccumulator(n_units=1)
    accumulator.add(target * -1.0, target, reference, weight=weight)
    assert accumulator.recovery()[0] < 0.0


def test_primary_map_recovery_uses_paired_expected_spikes_and_keeps_equal_robustness() -> None:
    target = np.asarray([[[[1.0]]], [[[10.0]]]])
    prediction = np.zeros_like(target)
    reference = np.zeros((1, 1, 1))
    weight = np.asarray([[100.0], [1.0]])
    accumulator = pq.MapRecoveryAccumulator(n_units=1)
    accumulator.add(prediction, target, reference, weight=weight)
    assert accumulator.recovery()[0] == pytest.approx(0.0)
    assert accumulator.recovery_equal_unit()[0] == pytest.approx(0.0)
    # A non-proportional prediction makes the two aggregation schemes differ.
    accumulator = pq.MapRecoveryAccumulator(n_units=1)
    prediction = np.asarray([[[[1.0]]], [[[0.0]]]])
    accumulator.add(prediction, target, reference, weight=weight)
    assert accumulator.recovery()[0] > accumulator.recovery_equal_unit()[0]


def test_equal_unit_population_recovery_is_mean_of_unit_recoveries() -> None:
    accumulator = pq.MapRecoveryAccumulator(n_units=2)
    accumulator.residual_equal[:] = [0.0, 100.0]
    accumulator.reference_effect_equal[:] = [1.0, 100.0]
    # Unit recoveries are 1 and 0, independent of their effect magnitude.
    assert pq._population_recovery_equal(accumulator, np.asarray([0, 1])) == pytest.approx(0.5)


def test_outer_training_mean_pairs_exclude_crossed_test_block() -> None:
    for fold_index in range(4):
        fold = pq.load_fold(fold_index)
        assert set(pq._training_pairs(fold_index)).isdisjoint(fold.test.pairs)
        assert len(pq._training_pairs(fold_index)) == 108


def test_module_has_no_core_model_forward_or_checkpoint_loader() -> None:
    source = inspect.getsource(pq)
    forbidden = ("load_twin", "load_model", "core(", "checkpoint.load", "run_exact_subset")
    assert not any(token in source for token in forbidden)


def test_pq_probe_features_reconstruct_native_channel_features() -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(size=(20, pq.N_CHANNELS))
    split = pq._split_pq_features(x, coordinate_basis())
    p_coordinates = split["candidate_p_coordinates_rank8"]
    q_channels = split["complementary_q_reconstructed_channels_rank120"]
    reconstruction = p_coordinates @ coordinate_basis().T + q_channels
    np.testing.assert_allclose(reconstruction, x, atol=1e-12)


def test_fixed_ridge_probe_uses_train_fit_and_predicts_linear_target() -> None:
    rng = np.random.default_rng(16)
    train_x = rng.normal(size=(200, 5))
    test_x = rng.normal(size=(50, 5))
    weights = rng.normal(size=(5, 2))
    offset = np.asarray([2.0, -3.0])
    train_y = train_x @ weights + offset
    prediction = pq.ridge_predict(train_x, train_y, test_x, alpha=1e-8)
    np.testing.assert_allclose(prediction, test_x @ weights + offset, atol=1e-7)


def test_probe_documentation_rejects_unseen_class_claim() -> None:
    source = inspect.getsource(pq.run_probe_stage)
    assert "unseen_image_identity_generalization_claimed" in source
    assert "crossed_outer_train_to_unseen_test_images_and_trajectories" in source


def test_cuda_request_runner_uses_shared_lock_budget_deadline_and_debit() -> None:
    source = inspect.getsource(pq.run_readout_requests)
    for token in ("exclusive_gpu_lock", "load_budget", "budget_deadline", "record_gpu_time"):
        assert token in source


def test_stage_marker_requires_input_fingerprint_and_product_hashes() -> None:
    source = inspect.getsource(pq._stage_done)
    assert "input_fingerprint" in source
    assert "sha256_file" in source
    assert "schema_version" in source


def test_training_mean_subcache_is_bound_to_state_cache_and_its_own_hash() -> None:
    source = inspect.getsource(pq._training_mean_state)
    assert "state_cache_fingerprint" in source
    assert "fold_assignments_sha256" in source
    assert "mean_state_sha256" in source
    assert "np.isfinite(mean).all()" in source


def test_probe_feature_subcache_is_fold_scoped_and_state_fingerprinted() -> None:
    source = inspect.getsource(pq._probe_feature_cache)
    assert 'f"fold_{fold}"' in source
    assert "state_cache_fingerprint" in source
    assert "fold_assignments_sha256" in source
    assert "product_sha256" in source


def test_consolidation_does_not_raw_glob_supporting_arrays() -> None:
    source = inspect.getsource(pq.consolidate)
    assert 'INTERMEDIATE.glob("fold_*/*/variance/variance_supporting_arrays.npz")' not in source
    assert 'path.parent / "variance_supporting_arrays.npz" for path in variance_paths' in source


def test_fold_observational_metadata_reads_crossed_map_cache_not_global_archive() -> None:
    source = inspect.getsource(pq._unit_observational_metadata)
    assert "load_fold(fold).test.pairs" in source
    assert "MAP_CACHE" in source
    assert "SOURCE_EXACT" not in source


def test_per_unit_contrast_recovery_uses_direct_a_to_b_states() -> None:
    source = inspect.getsource(pq.run_readout_stage)
    assert "contrast_delta = h_b - h_a" in source
    assert "direct_a_to_b_" in source
    assert '"defining_contrast_movement_effect_decomposition"' in source
    assert "literal h(" in source


def test_state_semantic_decomposition_is_fold_test_only_and_detects_p_motion(
    tmp_path, monkeypatch
) -> None:
    state_path = tmp_path / "states.h5"
    h = np.zeros((2, 2, len(pq.SCALES), 2, pq.N_CHANNELS, 2, 2), dtype=np.float32)
    for image in range(2):
        for trajectory in range(2):
            for scale_i, scale in enumerate(pq.SCALES):
                h[image, trajectory, scale_i, :, 0] = float(scale)  # motion is entirely P
                h[image, trajectory, scale_i, :, 10] = float(image + trajectory)  # content/Q
    with h5py.File(state_path, "w") as handle:
        handle.create_dataset("h", data=h)

    split = SimpleNamespace(
        image_positions=(0, 1),
        trajectory_positions=(0, 1),
        pairs=[(0, 0), (0, 1), (1, 0), (1, 1)],
    )
    monkeypatch.setattr(pq, "load_fold", lambda _: SimpleNamespace(test=split))
    with h5py.File(state_path, "r") as state:
        table, motion_ms, motion_variance = pq._state_semantic_decomposition(
            state=state,
            fold=0,
            contrast_key="low_0_to_2",
            basis=coordinate_basis(),
        )
    movement = table[table.analysis == "movement_change_from_stabilization"]
    np.testing.assert_allclose(movement.candidate_p_total_fraction, 1.0)
    np.testing.assert_allclose(movement.complementary_q_total_fraction, 0.0)
    assert motion_ms[0] > 0
    assert motion_ms[10] == pytest.approx(0.0)
    assert motion_variance[0] == pytest.approx(0.0)  # constant contrast delta
    assert set(table.analysis) == {
        "movement_change_from_stabilization",
        "stabilized_visual_content_all_records",
        "stabilized_visual_content_image_means",
        "trajectory_specific_at_fixed_image_scale_frame",
    }
