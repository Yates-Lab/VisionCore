"""Regression tests for fresh homogeneous batches on every epoch."""

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


def make_sampler(seed=0, shuffle=True, batch=BATCH):
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


def test_first_pass_is_reproducible_across_instances():
    a = make_sampler(seed=7)
    b = make_sampler(seed=7)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


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
