"""`ByDatasetBatchSampler` must draw fresh batches every epoch.

Regression test for a defect found on 2026-08-05 by watching arm E1b's
validation BPS flatline. `__iter__` seeded its generator on `self.seed +
self._step`, and `_step` advances only via `set_step`, whose only caller is
`CurriculumCallback` -- registered solely under `--enable_curriculum`. With
curriculum off, which is the default and what every model-selection arm uses,
`_step` stayed 0 and *every epoch yielded the identical batch sequence*.

Combined with `limit_train_batches`, that meant a run trained on one epoch's
worth of samples repeated for its whole duration: E1b saw ~131k samples 61
times instead of 8M samples once, and the arm measured nothing about
homogeneous batching.

The defect was invisible to the existing 2-epoch smoke test, which checked that
the path *trains* -- it does. Only comparing batch indices across epochs shows
it, which is what these tests do.

This sampler is constructed only when `homogeneous_batches=True`; the default
cross-session path builds a plain shuffling DataLoader and was never affected.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from training.samplers import ByDatasetBatchSampler  # noqa: E402


BATCH = 64
SIZES = {"sessA": 5000, "sessB": 3000, "sessC": 2000}


class _Tag(torch.utils.data.Dataset):
    """Stands in for the Tag wrapper `MultiDatasetDM._mk_loader` builds."""

    def __init__(self, n, idx):
        self.n, self.idx = n, idx

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": i, "dataset_idx": self.idx}


def make_sampler(seed=0, shuffle=True, batch=BATCH):
    name2idx = {name: i for i, name in enumerate(SIZES)}
    cat = torch.utils.data.ConcatDataset(
        [_Tag(n, name2idx[name]) for name, n in SIZES.items()])
    sampler = ByDatasetBatchSampler(
        cat, name2idx, batch, contrast_scores=None, warmup_steps=8000,
        shuffle=shuffle, drop_last=True, seed=seed)
    return sampler, name2idx


def epochs(sampler, n):
    """`n` successive passes, as `n` training epochs would produce."""
    return [[tuple(b) for b in sampler] for _ in range(n)]


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------
def test_successive_epochs_are_not_identical():
    """The regression itself: epoch 2 must not repeat epoch 1."""
    sampler, _ = make_sampler()
    first, second = epochs(sampler, 2)
    assert first != second


def test_repetition_is_not_hidden_by_truncation():
    """limit_train_batches takes a prefix, so the *prefix* must differ too.

    A run with `limit_train_batches=512` only ever sees the first 512 batches
    of each epoch. If just the tail varied, every truncated epoch would still
    be identical and the defect would survive in exactly the configuration
    that triggered it.
    """
    sampler, _ = make_sampler()
    first, second = epochs(sampler, 2)
    prefix = min(8, len(first))
    assert first[:prefix] != second[:prefix]


def test_coverage_grows_across_epochs():
    """More epochs must mean more of the dataset seen, which is the point."""
    sampler, _ = make_sampler()
    passes = epochs(sampler, 8)

    seen_first = {i for batch in passes[0] for i in batch}
    seen_all = {i for p in passes for batch in p for i in batch}
    assert len(seen_all) > 1.5 * len(seen_first)


def test_curriculum_step_still_varies_the_draw():
    """`set_step` must keep working; the epoch term is additional, not a
    replacement."""
    sampler, _ = make_sampler()
    first = [tuple(b) for b in sampler]
    sampler.set_step(4096)
    after = [tuple(b) for b in sampler]
    assert first != after


# ---------------------------------------------------------------------------
# What must not have changed
# ---------------------------------------------------------------------------
def test_first_pass_is_reproducible_across_instances():
    """Two samplers with the same seed agree on epoch 1.

    The fix varies the draw *within* a run's lifetime without making runs
    irreproducible: a fresh sampler at a given seed still replays the same
    first epoch, so a re-run of a seeded job is unchanged.
    """
    a, _ = make_sampler(seed=7)
    b, _ = make_sampler(seed=7)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


def test_different_seeds_give_different_draws():
    a, _ = make_sampler(seed=1)
    b, _ = make_sampler(seed=2)
    assert [tuple(x) for x in a] != [tuple(x) for x in b]


def test_every_batch_comes_from_exactly_one_dataset():
    """The sampler's whole contract: one session per optimizer step."""
    sampler, _ = make_sampler()
    bounds = sampler.cat.cumulative_sizes

    def which(global_idx):
        for j, end in enumerate(bounds):
            if global_idx < end:
                return j
        raise AssertionError("index past the end of the ConcatDataset")

    for _ in range(3):                       # holds on every epoch, not just the first
        for batch in sampler:
            assert len({which(i) for i in batch}) == 1


def test_batches_are_full_size():
    sampler, _ = make_sampler()
    for batch in sampler:
        assert len(batch) == BATCH


def test_indices_stay_in_range():
    sampler, _ = make_sampler()
    total = sum(SIZES.values())
    for _ in range(2):
        for batch in sampler:
            assert all(0 <= i < total for i in batch)
