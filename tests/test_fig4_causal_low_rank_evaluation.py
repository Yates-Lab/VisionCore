from __future__ import annotations

import h5py
import numpy as np
import torch

from paper.fig4.mechanism_audit_v1.causal_low_rank.evaluate_subspaces import (
    _attach_state_delta_from_donor,
    _derangement,
    _readout_svd_basis,
    _shuffled_initial_values,
    _unit_observational_metadata,
    deterministic_validation_elbow,
    recovery_r2,
)


def test_effect_recovery_is_not_clipped() -> None:
    assert recovery_r2(1.0, 1.0) == 0.0
    assert recovery_r2(0.0, 1.0) == 1.0
    assert recovery_r2(3.0, 1.0) == -2.0


def test_validation_elbow_is_deterministic_and_never_uses_test_values() -> None:
    curve = {1: 0.10, 2: 0.38, 4: 0.62, 8: 0.70, 16: 0.74, 32: 0.76}
    first = deterministic_validation_elbow(curve)
    second = deterministic_validation_elbow(dict(reversed(list(curve.items()))))
    assert first == second
    assert first["ambiguous"] is False
    assert first["elbow_rank"] in curve
    assert len(first["adjacent_ranks"]) == 2


def test_flat_validation_curve_is_predeclared_ambiguous() -> None:
    result = deterministic_validation_elbow({1: 0.5, 2: 0.5, 4: 0.5, 8: 0.5})
    assert result["ambiguous"] is True
    assert result["elbow_rank"] is None
    assert result["adjacent_ranks"] == [1, 2, 4, 8]


def test_zero_effect_denominator_returns_nan() -> None:
    assert np.isnan(recovery_r2(0.0, 0.0))


def test_high_population_readout_svd_supports_same_rank_32_baseline() -> None:
    basis = _readout_svd_basis("high", 32)
    assert basis.shape == (128, 32)
    np.testing.assert_allclose(basis.T @ basis, np.eye(32), atol=1e-5)


def test_shuffled_target_permutation_is_pair_level_derangement() -> None:
    first = _derangement(102, 20260812)
    second = _derangement(102, 20260812)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(102))
    assert np.all(first != np.arange(102))


def test_shuffled_target_attaches_donor_delta_but_preserves_target_anchors(tmp_path) -> None:
    cache = tmp_path / "states.h5"
    with h5py.File(cache, "w") as state:
        values = np.zeros((1, 2, 5, 2, 3, 2, 2), dtype=np.float32)
        values[0, 1, 0] = 11.0
        values[0, 1, 2] = 17.0
        state.create_dataset("h", data=values)
        target_a = torch.full((2, 3, 2, 2), 101.0)
        target_b = torch.full((2, 3, 2, 2), 107.0)
        batch = {"h_a": target_a.clone(), "h_b": target_b.clone()}
        _attach_state_delta_from_donor(
            batch,
            state,
            donor_pair=(0, 1),
            scale_a=0.0,
            scale_b=1.0,
            frames=np.asarray([0, 1]),
            device="cpu",
        )
    torch.testing.assert_close(batch["h_a"], target_a)
    torch.testing.assert_close(batch["h_b"], target_b)
    torch.testing.assert_close(
        batch["intervention_delta_h"], torch.full_like(target_a, 6.0)
    )


def test_shuffled_target_uses_three_independent_preregistered_starts() -> None:
    pca = torch.eye(128)
    starts, seeds = _shuffled_initial_values(pca, 4, 1234, "cpu")
    assert seeds == [1235, 1236, 1237]
    assert len(starts) == 3
    torch.testing.assert_close(starts[1], pca[:, :4])
    assert not torch.allclose(starts[0], starts[2])
    for start in starts:
        torch.testing.assert_close(start.T @ start, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_per_unit_saved_metadata_contains_scale_preference_and_reversal() -> None:
    metadata = _unit_observational_metadata()
    assert metadata["sf_split_metric"].shape == (100,)
    assert metadata["optimal_movement_scale"].shape == (100,)
    assert metadata["ssi_reversal_1_to_3_bits"].shape == (100,)
    assert metadata["has_1x_to_3x_reversal"].dtype == np.bool_
