"""MultiDatasetDM exposure of the optional third (test) split.

These tests avoid loading real sessions: they drive `setup` with a stubbed
`prepare_data` so the wiring is testable without a GPU or the data volume.
"""
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from training.pl_modules import MultiDatasetDM


class _Stub(Dataset):
    def __init__(self, n=8, n_units=3):
        self.n = n
        self.n_units = n_units

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "robs": torch.zeros(self.n_units),
            "stim": torch.zeros(1, 2, 2),
            "row": torch.tensor(i),
        }

    def cast(self, dtype, target_keys=None):
        """No-op stand-in for DictDataset.cast."""
        return self


@pytest.fixture
def stub_data(monkeypatch):
    """Patch the loaders `MultiDatasetDM.setup` imports at call time."""
    cfgs = [
        {"session": "Allen_2022-02-16", "types": ["backimage"], "train_val_split": 0.8},
        {"session": "Logan_2020-01-06", "types": ["backimage"], "train_val_split": 0.8},
    ]

    def fake_load_dataset_configs(cfg_dir):
        import copy
        return copy.deepcopy(cfgs)

    def fake_prepare_data(cfg, strict=True, return_test=False):
        if return_test:
            return _Stub(), _Stub(4), _Stub(4), cfg
        return _Stub(), _Stub(4), cfg

    def fake_remove_pixel_norm(cfg):
        return cfg, False

    monkeypatch.setitem(
        sys.modules, "models.config_loader",
        types.SimpleNamespace(load_dataset_configs=fake_load_dataset_configs))
    monkeypatch.setitem(
        sys.modules, "DataYatesV1.utils.data.loading",
        types.SimpleNamespace(remove_pixel_norm=fake_remove_pixel_norm))
    monkeypatch.setitem(
        sys.modules, "models.data",
        types.SimpleNamespace(prepare_data=fake_prepare_data))
    return cfgs


def _make_dm(**kw):
    return MultiDatasetDM(
        cfg_dir="unused.yaml", max_ds=2, batch=2, workers=0,
        steps_per_epoch=4, dset_dtype="float32", **kw)


def test_setup_without_test_split_exposes_no_test_datasets(stub_data):
    dm = _make_dm()
    dm.setup()

    assert dm.train_dsets and dm.val_dsets
    assert dm.test_dsets == {}


@pytest.mark.parametrize("distributed", [False, True])
def test_limited_mixed_validation_represents_every_session(monkeypatch, distributed):
    dm = _make_dm()
    dm.name2idx = {f"session{i}": i for i in range(3)}
    dm.val_dsets = {name: _Stub(100) for name in dm.name2idx}

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: distributed)
    if distributed:
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    loader = dm.val_dataloader()

    def prefix():
        items = []
        for batch_number, groups in enumerate(loader):
            for group in groups:
                items.extend(zip(
                    group["dataset_idx"].tolist(),
                    group["row"].tolist(),
                ))
            if batch_number == 2:
                break
        return items

    first = prefix()
    assert {dataset_idx for dataset_idx, _row in first} == {0, 1, 2}
    assert len(first) == len(set(first))
    assert prefix() == first


def test_test_dataloader_fails_loudly_when_no_test_split_was_configured(stub_data):
    """Silently returning the validation loader here would report selection
    numbers as test numbers, which is the exact bias the split exists to fix."""
    dm = _make_dm()
    dm.setup()

    with pytest.raises(RuntimeError, match="test_split"):
        dm.test_dataloader()


def test_setup_with_test_split_exposes_one_test_dataset_per_session(stub_data):
    for cfg in stub_data:
        cfg["test_split"] = 0.15
    dm = _make_dm()
    dm.setup()

    assert set(dm.test_dsets) == {"Allen_2022-02-16", "Logan_2020-01-06"}


def test_test_dataloader_is_built_when_configured(stub_data):
    for cfg in stub_data:
        cfg["test_split"] = 0.15
    dm = _make_dm()
    dm.setup()

    loader = dm.test_dataloader()
    assert len(next(iter(loader))) > 0


def test_setup_can_select_an_exact_session(stub_data):
    dm = _make_dm(selected_sessions=["Logan_2020-01-06"])
    dm.setup()
    assert dm.names == ["Logan_2020-01-06"]
    assert set(dm.train_dsets) == {"Logan_2020-01-06"}


def test_setup_rejects_unknown_selected_session(stub_data):
    dm = _make_dm(selected_sessions=["Allen_2099-01-01"])
    with pytest.raises(ValueError, match="absent"):
        dm.setup()


def test_named_sampling_weight_allows_sessions_without_that_bank():
    dm = _make_dm(stimulus_sampling_weights={"gratings": 4.0})
    dm.cfgs = [
        {"types": ["backimage", "gratings"]},
        {"types": ["backimage"]},
    ]
    dm.names = ["with_gratings", "without_gratings"]
    dm.train_dsets = {
        "with_gratings": SimpleNamespace(
            inds=torch.tensor([[0, 1], [1, 2], [1, 3]])
        ),
        "without_gratings": SimpleNamespace(
            inds=torch.tensor([[0, 4], [0, 5]])
        ),
    }

    scores = dm._build_stimulus_sampling_scores()

    assert scores["with_gratings"].tolist() == [1.0, 4.0, 4.0]
    assert scores["without_gratings"].tolist() == [1.0, 1.0]
