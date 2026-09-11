"""Focused contracts for the production native-240-Hz model pipeline."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models import build_model
from models.config_loader import load_config
from models.data.datasets import CombinedEmbeddedDataset, DictDataset
from models.modules.readout import (
    DynamicGaussianReadout,
    SparseGaussianLowRankReadout,
    SparseGaussianReadout,
)
from training.pl_modules.multidataset_model import (
    MultiDatasetModel,
    set_trainable_model_components,
)
from training.train_multidataset import apply_stimulus_sampling_weight_overrides


ROOT = Path(__file__).resolve().parents[1]


def test_stage_two_trains_only_deep_and_phase_readouts():
    holder = nn.Module()
    holder.convnet = nn.Linear(3, 3)
    holder.modulator = nn.Linear(3, 3)
    holder.readouts = nn.ModuleList([nn.Linear(3, 2)])
    holder.phase_readouts = nn.ModuleList([nn.Linear(3, 2)])

    set_trainable_model_components(holder, ["readouts", "phase_readouts"])
    for name, parameter in holder.named_parameters():
        expected = name.startswith(("readouts.", "phase_readouts."))
        assert parameter.requires_grad is expected


def test_sparse_gaussian_exactly_embeds_ordinary_gaussian():
    torch.manual_seed(3)
    ordinary = DynamicGaussianReadout(4, 2)
    sparse = SparseGaussianReadout(4, 2, spatial_shape=(5, 5))
    scale = 5.0
    with torch.no_grad():
        sparse.features.weight.copy_(ordinary.features.weight * scale)
        sparse.spatial_weights.fill_(1.0 / scale)
        sparse.mean.copy_(ordinary.mean)
        sparse.std.copy_(ordinary.std)
        sparse.theta.copy_(ordinary.theta)
        sparse.bias.copy_(ordinary.bias)
    stimulus = torch.randn(3, 4, 5, 5)
    assert torch.allclose(sparse(stimulus), ordinary(stimulus), atol=1.0e-6)


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_phase_readout_rank_is_configurable(rank):
    readout = SparseGaussianLowRankReadout(
        4, 3, rank=rank, bias=False, spatial_shape=(7, 7)
    )
    assert readout.rank == rank
    assert readout(torch.randn(2, 4, 7, 7)).shape == (2, 3)


def test_rank_two_phase_readout_is_exactly_embedded_in_rank_four():
    torch.manual_seed(281)
    source = SparseGaussianLowRankReadout(
        4, 3, rank=2, bias=False, spatial_shape=(7, 7), output_scale=7.0
    )
    target = SparseGaussianLowRankReadout(
        4,
        3,
        rank=4,
        bias=False,
        spatial_shape=(7, 7),
        zero_features=True,
        output_scale=7.0,
    )
    model = nn.Module()
    model.readouts = None
    model.phase_readouts = nn.ModuleList([target])
    checkpoint = {
        f"model.phase_readouts.0.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    holder = SimpleNamespace(model=model)
    selected, _ = MultiDatasetModel._compatible_pretrained_state(
        holder, checkpoint, load_heads=True
    )
    model.load_state_dict(selected, strict=False)

    stimulus = torch.randn(5, 4, 7, 7)
    assert holder._low_rank_migrated_readouts == {"phase_readouts.0"}
    assert torch.allclose(target(stimulus), source(stimulus), atol=1.0e-6)


@pytest.mark.parametrize("floor", [0.0, 0.75])
def test_gaussian_to_rank_two_preserves_predictions_and_extra_factor_learns(floor):
    torch.manual_seed(201)
    source = DynamicGaussianReadout(4, 3, initial_std=0.5)
    target = SparseGaussianLowRankReadout(
        4, 3, rank=2, spatial_shape=(9, 9), migration_std_floor=floor
    )
    model = nn.Module()
    model.readouts = nn.ModuleList([target])
    model.phase_readouts = None
    checkpoint = {
        f"model.readouts.0.{key}": value.clone()
        for key, value in source.state_dict().items()
    }
    holder = SimpleNamespace(model=model)
    selected, _ = MultiDatasetModel._compatible_pretrained_state(
        holder, checkpoint, load_heads=True
    )
    model.load_state_dict(selected, strict=True)
    stimulus = torch.randn(5, 4, 9, 9)
    assert torch.allclose(target(stimulus), source(stimulus), atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(target.spatial_weights, dim=(1, 2, 3)), torch.ones(3))
    assert torch.count_nonzero(target.features.weight.reshape(3, 2, 4)[:, 1]) == 0
    target(stimulus).square().sum().backward()
    assert torch.count_nonzero(target.features.weight.grad.reshape(3, 2, 4)[:, 1]) > 0


@pytest.mark.parametrize("rank", [1, 2])
def test_no_phase_curriculum_models_and_frozen_stage(rank):
    configs = [{"session": "test", "cids": [0, 1, 2]}]
    for stage in (2, 3):
        config = load_config(ROOT / f"experiments/model_configs/dekel_native240_no_phase_rank{rank}_stage{stage}.yaml")
        model = build_model(config, configs)
        assert model.phase_readouts is None
        assert getattr(model.readouts[0], "rank", 1) == rank
        if stage == 2:
            set_trainable_model_components(model, config["trainable_components"])
            assert all(p.requires_grad == name.startswith("readouts.") for name, p in model.named_parameters())


def test_stage_configs_build_native_240_models():
    dataset_configs = [{"session": "test", "cids": [0, 1, 2]}]
    stage_paths = [
        ROOT / "experiments/model_configs/dekel_native240_curriculum_stage1_core.yaml",
        ROOT / "experiments/model_configs/dekel_native240_curriculum_stage2_sparse_readout.yaml",
        ROOT / "experiments/model_configs/dekel_native240_curriculum_stage3_finetune.yaml",
    ]
    models = [build_model(load_config(path), dataset_configs) for path in stage_paths]
    assert all(model.sampling_rate == 240 for model in models)
    assert models[0].phase_readouts is None
    assert models[1].phase_readouts[0].rank == 4
    assert models[2].phase_readouts[0].rank == 4


def test_stimulus_sampling_is_explicit_and_not_loss_weighting():
    original = {"stimulus_sampling_weights": {"gratings": 2.0}}
    resolved = apply_stimulus_sampling_weight_overrides(
        original, ["gratings=16", "backimage=0.5"]
    )
    assert original == {"stimulus_sampling_weights": {"gratings": 2.0}}
    assert resolved["stimulus_sampling_weights"] == {
        "gratings": 16.0,
        "backimage": 0.5,
    }
    assert "stimulus_loss_weights" not in resolved


def test_combined_dataset_exposes_stimulus_bank_for_sampling():
    first = DictDataset({"x": torch.arange(4, dtype=torch.float32)[:, None]})
    second = DictDataset({"x": torch.arange(10, 14, dtype=torch.float32)[:, None]})
    dataset = CombinedEmbeddedDataset(
        [first, second],
        [torch.tensor([0, 2]), torch.tensor([1, 3])],
        {"x": None},
    )
    batch = dataset[torch.tensor([0, 1, 2, 3])]
    assert batch["stimulus_type_idx"].tolist() == [0, 0, 1, 1]
