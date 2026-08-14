"""Three-way train/val/test splitting.

The paper model used a two-way 80/20 split, so checkpoint selection and
reported performance drew on the same trials. That is tolerable for one model
but optimistically biased for a family of N (see
`paper/model_selection/protocol.py`). These tests pin the additive design: a
``test_split`` key that, when absent, leaves the two-way path byte-identical,
because figures 1-4 depend on the existing split.
"""
import numpy as np
import pytest
import torch

from models.data.datasets import DictDataset
from models.data.loading import get_embedded_datasets, resolve_split_fractions
from models.data.splitting import (
    split_inds_by_trial,
    split_inds_by_trial_train_val_test,
)


N_TRIALS = 40
BINS_PER_TRIAL = 12
SEED = 1002


def _make_dset(n_trials=N_TRIALS, bins=BINS_PER_TRIAL, n_units=5, seed=0):
    """Synthetic single-session DictDataset with clean trial structure."""
    rng = np.random.default_rng(seed)
    n = n_trials * bins
    trial_inds = np.repeat(np.arange(n_trials), bins)
    return DictDataset({
        "stim": torch.from_numpy(rng.normal(size=(n, 1, 4, 4)).astype(np.float32)),
        "robs": torch.from_numpy(rng.poisson(1.0, size=(n, n_units)).astype(np.float32)),
        "dfs": torch.ones(n, n_units),
        "trial_inds": torch.from_numpy(trial_inds.astype(np.int64)),
    })


def _trials_of(dset, inds):
    return set(dset["trial_inds"][inds].tolist())


# ---------------------------------------------------------------------------
# The splitter itself
# ---------------------------------------------------------------------------
def test_three_way_split_covers_all_indices_without_overlap():
    dset = _make_dset()
    inds = torch.arange(len(dset))

    tr, va, te = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)

    assert len(tr) + len(va) + len(te) == len(inds)
    joined = torch.cat([tr, va, te]).sort().values
    assert torch.equal(joined, inds)


def test_three_way_split_never_shares_a_trial_across_splits():
    """Leakage across splits would let a validation trial's neighbours train."""
    dset = _make_dset()
    inds = torch.arange(len(dset))

    tr, va, te = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)
    t_tr, t_va, t_te = _trials_of(dset, tr), _trials_of(dset, va), _trials_of(dset, te)

    assert not (t_tr & t_va)
    assert not (t_tr & t_te)
    assert not (t_va & t_te)
    assert len(t_tr | t_va | t_te) == N_TRIALS


def test_three_way_split_respects_requested_proportions():
    dset = _make_dset()
    inds = torch.arange(len(dset))

    tr, va, te = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)

    assert len(_trials_of(dset, tr)) == int(N_TRIALS * 0.70)
    assert len(_trials_of(dset, va)) == int(N_TRIALS * 0.85) - int(N_TRIALS * 0.70)


def test_three_way_split_is_deterministic_given_the_seed():
    dset = _make_dset()
    inds = torch.arange(len(dset))

    a = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)
    b = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)

    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_three_way_test_trials_are_held_out_of_the_two_way_val_set():
    """Both splitters permute trials identically under one seed, so the 70/15/15
    train set nests inside the 80/20 train set. Documents that moving to the
    three-way split does not train on anything the two-way split held out."""
    dset = _make_dset()
    inds = torch.arange(len(dset))

    tr2, _ = split_inds_by_trial(dset, inds, 0.80, seed=SEED)
    tr3, _, _ = split_inds_by_trial_train_val_test(dset, inds, 0.70, 0.15, seed=SEED)

    assert _trials_of(dset, tr3) <= _trials_of(dset, tr2)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def test_absent_test_split_resolves_to_the_two_way_fractions():
    train_frac, test_frac = resolve_split_fractions({"train_val_split": 0.8})
    assert train_frac == 0.8
    assert test_frac is None


def test_present_test_split_resolves_to_three_fractions():
    train_frac, test_frac = resolve_split_fractions(
        {"train_val_split": 0.70, "test_split": 0.15})
    assert train_frac == 0.70
    assert test_frac == 0.15


def test_null_test_split_is_treated_as_absent():
    """`test_split: null` in YAML must not silently switch splitters."""
    train_frac, test_frac = resolve_split_fractions(
        {"train_val_split": 0.8, "test_split": None})
    assert test_frac is None


def test_impossible_split_fractions_are_rejected():
    with pytest.raises(ValueError):
        resolve_split_fractions({"train_val_split": 0.9, "test_split": 0.2})


# ---------------------------------------------------------------------------
# get_embedded_datasets: the default path must not move
# ---------------------------------------------------------------------------
def _embed(dsets, test_split):
    return get_embedded_datasets(
        sess=None,
        types=dsets,
        keys_lags={"robs": 0, "stim": [0, 1, 2], "dfs": 0},
        train_val_split=0.80 if test_split is None else 0.70,
        cids=None,
        seed=SEED,
        pre_func=lambda x: x,
        test_split=test_split,
    )


def test_default_path_returns_two_datasets_with_untouched_indices():
    """Byte-identical guard: with no test_split, served indices must match the
    two-way splitter exactly. Figures 1-4 depend on this."""
    dsets = [_make_dset(seed=1)]
    train_dset, val_dset = _embed(dsets, test_split=None)

    valid = dsets[0]["dfs"].any(dim=1).nonzero(as_tuple=True)[0]
    expect_tr, expect_va = split_inds_by_trial(dsets[0], valid, 0.80, seed=SEED)

    assert torch.equal(train_dset.dset_inds[0], expect_tr)
    assert torch.equal(val_dset.dset_inds[0], expect_va)


def test_test_split_returns_a_third_dataset():
    dsets = [_make_dset(seed=1)]
    out = _embed(dsets, test_split=0.15)

    assert len(out) == 3
    train_dset, val_dset, test_dset = out
    assert len(test_dset) > 0
    assert len(train_dset) > len(val_dset)


def test_test_split_keeps_the_three_served_index_sets_disjoint():
    dsets = [_make_dset(seed=1)]
    train_dset, val_dset, test_dset = _embed(dsets, test_split=0.15)

    tr = set(train_dset.dset_inds[0].tolist())
    va = set(val_dset.dset_inds[0].tolist())
    te = set(test_dset.dset_inds[0].tolist())

    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)


def test_test_split_applies_per_sub_dataset():
    """Multiple stimulus types are split independently; each must contribute."""
    dsets = [_make_dset(seed=1), _make_dset(seed=2)]
    _, _, test_dset = _embed(dsets, test_split=0.15)

    assert len(test_dset.dset_inds) == 2
    assert all(len(x) > 0 for x in test_dset.dset_inds)
