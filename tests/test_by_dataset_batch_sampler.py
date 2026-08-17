"""Regression tests for fresh homogeneous batches on every epoch."""

import ast
from pathlib import Path

import torch

from training.samplers import ByDatasetBatchSampler
from paper.model_selection.evaluate_dekel_split import iter_session_split_batches


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


def test_validation_epoch_hook_is_unique_and_synchronizes_sampler():
    """Prevent a duplicate Lightning hook from silently dropping resume sync."""
    path = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "pl_modules"
        / "multidataset_model.py"
    )
    module = ast.parse(path.read_text())
    model_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MultiDatasetModel"
    )
    hooks = [
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "on_validation_epoch_start"
    ]
    assert len(hooks) == 1
    assert "_sync_loader_sampler_epoch" in ast.unparse(hooks[0])


def test_full_split_evaluation_visits_every_example_once():
    class _EvalDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return {"sample_index": index}

    class _EvalDM:
        names = ["session"]
        val_dsets = {"session": _EvalDataset()}
        test_dsets = {"session": _EvalDataset()}
        batch = 2
        workers = 0

    batches = list(iter_session_split_batches(_EvalDM(), "val"))
    assert [dataset_idx for dataset_idx, _, _ in batches] == [0, 0, 0]
    visited = torch.cat([batch["sample_index"] for _, _, batch in batches])
    assert torch.equal(visited, torch.arange(5))
