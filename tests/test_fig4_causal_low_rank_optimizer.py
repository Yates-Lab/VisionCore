from __future__ import annotations

import h5py
import numpy as np
import torch

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    N_FRAMES,
    normalize_rate,
    qr_basis,
    softplus_rate,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.optimize_subspaces import (
    _TRAIN_RESIDENT_CACHE,
    Denominators,
    batch_from_cache,
    batch_from_training_resident,
    evaluate_validation_resident,
    load_training_resident,
    loss_for_basis,
    select_training_minibatch_indices,
)


def test_map_objective_backpropagates_through_qr_and_shared_projector() -> None:
    generator = torch.Generator().manual_seed(20260812)
    batch, channels, rank, units, hidden_size, map_size = 2, 6, 2, 3, 7, 5
    feature = torch.randn(units, channels, generator=generator)
    space = torch.randn(units, hidden_size - map_size + 1, hidden_size - map_size + 1, generator=generator)
    bias = torch.randn(units, generator=generator)
    h_a = torch.randn(batch, channels, hidden_size, hidden_size, generator=generator)
    h_b = torch.randn(batch, channels, hidden_size, hidden_size, generator=generator)
    z_a = torch.nn.functional.conv2d(
        torch.nn.functional.conv2d(h_a, feature[:, :, None, None]),
        space[:, None],
        groups=units,
    ) + bias[None, :, None, None]
    z_b = torch.nn.functional.conv2d(
        torch.nn.functional.conv2d(h_b, feature[:, :, None, None]),
        space[:, None],
        groups=units,
    ) + bias[None, :, None, None]
    g_a = normalize_rate(softplus_rate(z_a))
    g_b = normalize_rate(softplus_rate(z_b))
    expected_a = softplus_rate(z_a).mean((-2, -1)) / 120.0
    expected_b = softplus_rate(z_b).mean((-2, -1)) / 120.0
    weight = 0.5 * (expected_a + expected_b)
    denominator = float(((g_b - g_a).square() * weight[..., None, None]).sum())
    data = {
        "h_a": h_a,
        "h_b": h_b,
        "preactivation_a": z_a,
        "preactivation_b": z_b,
        "gain_a": g_a,
        "gain_b": g_b,
        "expected_spikes_a": expected_a,
        "expected_spikes_b": expected_b,
    }
    parameter = torch.nn.Parameter(torch.randn(channels, rank, generator=generator))
    loss, sufficiency, necessity = loss_for_basis(
        data,
        qr_basis(parameter),
        {"feature": feature, "space": space, "bias": bias},
        Denominators(denominator, denominator, batch),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(sufficiency)
    assert torch.isfinite(necessity)
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert float(parameter.grad.norm()) > 0


def test_resident_validation_is_exactly_the_complete_objective() -> None:
    generator = torch.Generator().manual_seed(17)
    records, channels, rank, units, hidden_size = 5, 6, 2, 3, 7
    feature = torch.randn(units, channels, generator=generator)
    space = torch.randn(units, 3, 3, generator=generator)
    bias = torch.randn(units, generator=generator)
    h_a = torch.randn(records, channels, hidden_size, hidden_size, generator=generator)
    h_b = torch.randn(records, channels, hidden_size, hidden_size, generator=generator)

    def endpoint(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.nn.functional.conv2d(
            torch.nn.functional.conv2d(state, feature[:, :, None, None]),
            space[:, None],
            groups=units,
        ) + bias[None, :, None, None]
        rate = softplus_rate(z)
        return z, normalize_rate(rate), rate.mean((-2, -1)) / 120.0

    z_a, g_a, expected_a = endpoint(h_a)
    z_b, g_b, expected_b = endpoint(h_b)
    weight = 0.5 * (expected_a + expected_b)
    denominator = float(((g_b - g_a).square() * weight[..., None, None]).sum())
    data = {
        "h_a": h_a,
        "h_b": h_b,
        "preactivation_a": z_a,
        "preactivation_b": z_b,
        "gain_a": g_a,
        "gain_b": g_b,
        "expected_spikes_a": expected_a,
        "expected_spikes_b": expected_b,
    }
    bases = torch.stack(
        [qr_basis(torch.randn(channels, rank, generator=generator)) for _ in range(3)]
    )
    denominators = Denominators(denominator, denominator, records)
    observed = evaluate_validation_resident(
        data,
        bases,
        {"feature": feature, "space": space, "bias": bias},
        denominators,
    )
    expected = [
        [float(value) for value in loss_for_basis(data, basis, {"feature": feature, "space": space, "bias": bias}, denominators)]
        for basis in bases
    ]
    expected_array = torch.tensor(expected, dtype=torch.float64).numpy()
    assert torch.equal(torch.as_tensor(observed[0]), torch.as_tensor(expected_array[:, 0]))
    assert torch.equal(torch.as_tensor(observed[1]), torch.as_tensor(expected_array[:, 1]))
    assert torch.equal(torch.as_tensor(observed[2]), torch.as_tensor(expected_array[:, 2]))


def test_cpu_resident_training_batches_equal_direct_cache_reads(tmp_path) -> None:
    rng = np.random.default_rng(2301)
    state_path = tmp_path / "state.h5"
    maps_path = tmp_path / "maps.h5"
    n_images, n_trajectories, n_scales, n_units = 2, 3, 5, 6
    state_shape = (n_images, n_trajectories, n_scales, N_FRAMES, 3, 2, 2)
    gain_shape = (n_images, n_trajectories, n_scales, N_FRAMES, n_units, 2, 2)
    expected_shape = (n_images, n_trajectories, n_scales, N_FRAMES, n_units)
    with h5py.File(state_path, "w") as state:
        state.create_dataset(
            "h", data=rng.normal(size=state_shape).astype(np.float16)
        )
    with h5py.File(maps_path, "w") as maps:
        maps.create_dataset(
            "gain", data=rng.normal(size=gain_shape).astype(np.float16)
        )
        maps.create_dataset(
            "expected_spikes",
            data=rng.uniform(size=expected_shape).astype(np.float16),
        )

    pairs = [(1, 2), (0, 1), (1, 0)]
    units = np.asarray([4, 1, 5], dtype=np.int64)
    _TRAIN_RESIDENT_CACHE.clear()
    with h5py.File(state_path, "r") as state, h5py.File(maps_path, "r") as maps:
        resident = load_training_resident(
            state, maps, pairs, 0.5, 3.0, units
        )
        assert resident.pairs == tuple(pairs)
        for name in (
            "h_a",
            "h_b",
            "gain_a",
            "gain_b",
            "expected_spikes_a",
            "expected_spikes_b",
        ):
            assert getattr(resident, name).dtype == np.float32

        # Deliberately unsorted input verifies the same frame-sort convention
        # as the historical HDF5 minibatch path for every pair-list index.
        frames = np.asarray([31, 2, 17, 5, 29, 0, 11, 7])
        for pair_index, pair in enumerate(pairs):
            direct = batch_from_cache(
                state, maps, [pair], 0.5, 3.0, frames, units, "cpu"
            )
            observed = batch_from_training_resident(
                resident, pair_index, frames, "cpu"
            )
            assert observed.keys() == direct.keys()
            for name in direct:
                assert torch.equal(observed[name], direct[name]), name

        # The full provenance/slice key reuses a split across ranks, whereas a
        # changed pair order has changed index semantics and must not alias it.
        reused = load_training_resident(state, maps, pairs, 0.5, 3.0, units)
        reordered = load_training_resident(
            state, maps, list(reversed(pairs)), 0.5, 3.0, units
        )
        assert reused is resident
        assert reordered is not resident


def test_training_minibatch_rng_and_index_semantics_are_unchanged() -> None:
    observed_rng = np.random.default_rng(20260812)
    expected_rng = np.random.default_rng(20260812)
    for _ in range(12):
        observed_pair, observed_frames = select_training_minibatch_indices(
            observed_rng, n_pairs=102, frame_batch=8
        )
        expected_pair = int(expected_rng.integers(102))
        expected_frames = np.sort(
            expected_rng.choice(N_FRAMES, size=8, replace=False)
        )
        assert observed_pair == expected_pair
        assert np.array_equal(observed_frames, expected_frames)
        assert len(observed_frames) == 8
        assert np.all(np.diff(observed_frames) > 0)
