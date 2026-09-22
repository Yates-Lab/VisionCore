"""Regression tests for fresh homogeneous batches on every epoch."""

import pytest
import torch

from training.samplers import ByDatasetBatchSampler


BATCH = 64
SIZES = {"sessA": 5000, "sessB": 3000, "sessC": 2000}


class _Tag(torch.utils.data.Dataset):
    def __init__(self, n, idx):
        self.n, self.idx = n, idx

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": i, "dataset_idx": self.idx}


def make_sampler(seed=0, shuffle=True, batch=BATCH, fixed_random=False):
    name2idx = {name: i for i, name in enumerate(SIZES)}
    cat = torch.utils.data.ConcatDataset(
        [_Tag(n, name2idx[name]) for name, n in SIZES.items()]
    )
    sampler = ByDatasetBatchSampler(
        cat,
        name2idx,
        batch,
        contrast_scores=None,
        warmup_steps=8000,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
        fixed_random=fixed_random,
    )
    return sampler


def epochs(sampler, n):
    return [[tuple(batch) for batch in sampler] for _ in range(n)]


def test_successive_epochs_are_not_identical():
    first, second = epochs(make_sampler(), 2)
    assert first != second


def test_truncated_epoch_prefix_changes():
    first, second = epochs(make_sampler(), 2)
    assert first[:8] != second[:8]


def test_coverage_grows_across_epochs():
    passes = epochs(make_sampler(), 8)
    seen_first = {i for batch in passes[0] for i in batch}
    seen_all = {i for epoch in passes for batch in epoch for i in batch}
    assert len(seen_all) > 1.5 * len(seen_first)


def test_curriculum_step_still_varies_the_draw():
    sampler = make_sampler()
    first = [tuple(batch) for batch in sampler]
    sampler.set_step(4096)
    after = [tuple(batch) for batch in sampler]
    assert first != after


def test_first_pass_is_reproducible_across_instances():
    a = make_sampler(seed=7)
    b = make_sampler(seed=7)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


def test_different_seeds_give_different_draws():
    a = make_sampler(seed=1)
    b = make_sampler(seed=2)
    assert [tuple(x) for x in a] != [tuple(x) for x in b]


def test_set_epoch_reproduces_uninterrupted_resume_sequence():
    uninterrupted = make_sampler(seed=11)
    expected = epochs(uninterrupted, 9)[8]

    resumed = make_sampler(seed=11)
    resumed.set_epoch(8)
    actual = [tuple(batch) for batch in resumed]

    assert actual == expected


def test_batches_remain_homogeneous_and_full_size():
    sampler = make_sampler()
    bounds = sampler.cat.cumulative_sizes

    def dataset_index(global_idx):
        return next(j for j, end in enumerate(bounds) if global_idx < end)

    for _ in range(3):
        for batch in sampler:
            assert len(batch) == BATCH
            assert len({dataset_index(i) for i in batch}) == 1
            assert all(0 <= i < sum(SIZES.values()) for i in batch)


def test_unshuffled_batches_are_deterministic_nonrepeating_and_interleaved():
    sampler = make_sampler(shuffle=False)
    first = [tuple(batch) for batch in sampler]
    second = [tuple(batch) for batch in sampler]
    assert first == second
    assert len(first) == sum(size // BATCH for size in SIZES.values())

    flattened = [idx for batch in first for idx in batch]
    assert len(flattened) == len(set(flattened))

    bounds = sampler.cat.cumulative_sizes
    def dataset_index(global_idx):
        return next(j for j, end in enumerate(bounds) if global_idx < end)

    # The first round contains one sequential batch from every session.
    assert [dataset_index(first[i][0]) for i in range(len(SIZES))] == [0, 1, 2]
    assert first[0] == tuple(range(0, BATCH))
    assert first[3] == tuple(range(BATCH, 2 * BATCH))


def test_fixed_random_evaluation_is_representative_and_reproducible():
    sampler = make_sampler(shuffle=False, fixed_random=True, seed=19)
    first = [tuple(batch) for batch in sampler]
    second = [tuple(batch) for batch in sampler]
    assert first == second
    assert first != [tuple(batch) for batch in make_sampler(shuffle=False)]
    flattened = [idx for batch in first for idx in batch]
    assert len(flattened) == len(set(flattened))

    # A different fixed seed changes the subset/order without changing its size.
    other = [tuple(batch) for batch in make_sampler(shuffle=False, fixed_random=True, seed=20)]
    assert first != other
    assert len(first) == len(other)


def test_sample_scores_change_training_frequency_without_affecting_batch_shape():
    name2idx = {"sessA": 0}
    cat = torch.utils.data.ConcatDataset([_Tag(1000, 0)])
    weights = torch.ones(1000)
    weights[500:] = 9.0
    sampler = ByDatasetBatchSampler(
        cat,
        name2idx,
        batch_size=64,
        sample_scores={"sessA": weights},
        shuffle=True,
        drop_last=True,
        seed=3,
    )

    samples = [idx for _ in range(12) for batch in sampler for idx in batch]
    assert all(len(batch) == 64 for batch in make_sampler())
    assert sum(idx >= 500 for idx in samples) / len(samples) > 0.82


def test_uniform_dataset_sampling_balances_unequal_recordings():
    name2idx = {"small": 0, "large": 1}
    cat = torch.utils.data.ConcatDataset([_Tag(100, 0), _Tag(900, 1)])
    sampler = ByDatasetBatchSampler(
        cat,
        name2idx,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        seed=17,
        dataset_sampling="uniform",
    )
    batches = list(sampler)
    small = sum(batch[0] < 100 for batch in batches)
    assert 0.4 < small / len(batches) < 0.6


def test_dataset_sampling_mode_is_validated():
    cat = torch.utils.data.ConcatDataset([_Tag(10, 0)])
    with pytest.raises(ValueError, match="dataset_sampling"):
        ByDatasetBatchSampler(
            cat,
            {"sess": 0},
            batch_size=2,
            dataset_sampling="inverse_size",
        )
