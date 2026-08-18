"""Contracts for the ordinary-stem / SO(2)-spatial M77 pilot."""

from pathlib import Path

import pytest
import torch
import yaml

escnn = pytest.importorskip("escnn")
from escnn import nn as enn

from models import build_model
from models.modules.dekel import HammingConv3d
from models.modules.dekel_so2 import SO2DekelCore
from training.regularizers import create_regularizers


ROOT = Path(__file__).resolve().parents[1]


def _small_core() -> SO2DekelCore:
    return SO2DekelCore(
        {
            "initial_channels": 1,
            "temporal_support": 5,
            "temporal_channels": 8,
            "spatial_fields": [2, 2, 2],
            "spatial_kernels": [3, 3, 3],
            "maximum_frequency": 2,
            "orientation_samples": 12,
            "norm_groups": [8, 1, 1, 1],
            "normalization": {
                "type": "groupnorm_lrn_presplit",
                "lrn_size": 3,
                "lrn_alpha": 0.1,
            },
            "input_size": [9, 9],
            "scaffold_size": 3,
            "frequency_mask": {},
        }
    )


def test_ordinary_stem_so2_suffix_shapes_and_gradients():
    core = _small_core()
    value = torch.randn(2, 1, 5, 9, 9, requires_grad=True)
    stages = core.forward_stages(value)
    output = core(value)

    # m <= 2 has five real Fourier coefficients per field.  Each stage has
    # two fields and retains positive/negative branches: 2 * 2 * 5 = 20.
    assert [tuple(stage.shape) for stage in stages] == [
        (2, 20, 9, 9),
        (2, 20, 5, 5),
        (2, 20, 3, 3),
    ]
    assert output.shape == (2, 60, 1, 3, 3)
    assert isinstance(core.temporal_conv, HammingConv3d)
    assert core.temporal_conv.conv.out_channels == 8

    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_spatial_suffix_is_equivariant_at_grid_exact_quarter_turn():
    core = _small_core().eval()
    # Test only the SO(2) suffix.  The preceding ordinary Conv3d is deliberately
    # not claimed to be rotation equivariant.
    stem = enn.GeometricTensor(torch.randn(2, 16, 9, 9), core.stem_type)
    quarter_turn = core.space.fibergroup.element(torch.pi / 2)

    def suffix(value):
        stage1 = core.stage1_nonlinearity(core.stage1_conv(value))
        stage2 = core.stage2_nonlinearity(core.stage2_conv(core.pool1(stage1)))
        return core.stage3_nonlinearity(core.stage3_conv(core.pool2(stage2)))

    rotate_after = suffix(stem).transform(quarter_turn).tensor
    rotate_before = suffix(stem.transform(quarter_turn)).tensor
    relative_error = (rotate_after - rotate_before).norm() / rotate_after.norm()
    assert relative_error < 2.0e-5


def test_production_pilot_config_builds_compressed_fieldwise_behavior_model():
    path = (
        ROOT
        / "experiments/model_configs"
        / "dekel_m77_8temporal_so2m3_lessreg_mlp_behavior.yaml"
    )
    config = yaml.safe_load(path.read_text())
    model = build_model(config, [{"cids": list(range(5)), "behavior_dim": 42}])
    core = model.convnet

    assert isinstance(core, SO2DekelCore)
    assert core.maximum_frequency == 3
    assert core.temporal_conv.conv.out_channels == 8
    assert core.spatial_fields == (8, 8, 8)
    assert core.modulation_field_size == 7
    assert core.modulation_num_fields == 48
    assert core.get_output_channels() == 336
    assert model.modulator.feature_dim == 336
    assert model.modulator.modulation_field_size == 7
    assert model.modulator.scale_layer.out_features == 48

    spatial_coefficients = sum(
        module.weights.numel()
        for module in (core.stage1_conv, core.stage2_conv, core.stage3_conv)
    )
    assert spatial_coefficients == 50_816

    regularizers = create_regularizers(config, list(model.named_parameters()))
    smoothness = next(
        item for item in regularizers if item.name == "first_layer_spatial_smoothness"
    )
    assert smoothness.lmbda == pytest.approx(2.5e-4)
    assert smoothness.param_names == ["convnet.temporal_conv.conv.weight"]


def test_behavior_film_repeats_one_gain_over_each_so2_field():
    path = (
        ROOT
        / "experiments/model_configs"
        / "dekel_m77_8temporal_so2m3_lessreg_mlp_behavior.yaml"
    )
    config = yaml.safe_load(path.read_text())
    model = build_model(config, [{"cids": [0], "behavior_dim": 42}])
    modulator = model.modulator
    with torch.no_grad():
        modulator.scale_layer.weight.zero_()
        modulator.scale_layer.bias.copy_(
            torch.linspace(-0.5, 0.5, modulator.scale_layer.out_features)
        )
    features = torch.ones(1, 336, 1, 3, 3)
    modulated = modulator(features, torch.zeros(1, 42))[:, :336]
    gains = modulated[0, :, 0, 0, 0].reshape(48, 7)
    assert torch.equal(gains, gains[:, :1].expand_as(gains))

