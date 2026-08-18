"""Contract tests for the production M77-style feed-forward core."""

from pathlib import Path

import pytest
import torch
import yaml

from models import build_model
from models.data.transforms import make_pipeline
from models.modules.dekel import DekelCore, HammingConv3d
from training.regularizers import Regularizer, create_regularizers, laplacian_penalty


ROOT = Path(__file__).resolve().parents[1]


def _small_core(**overrides):
    config = {
        "initial_channels": 1,
        "temporal_support": 5,
        "temporal_channels": 2,
        "spatial_channels": [4, 4, 4],
        "spatial_kernels": [3, 3, 3],
        "norm_groups": [2, 2, 2, 2],
        "input_size": [9, 9],
        "scaffold_size": 3,
    }
    config.update(overrides)
    return DekelCore(config)


def test_core_shapes_split_relu_and_large_field_path():
    core = _small_core().eval()
    native = torch.randn(2, 1, 5, 9, 9)
    large = torch.randn(1, 1, 5, 17, 17)
    with torch.no_grad():
        stage1, stage2, stage3 = core.forward_stages(native)
        ordinary = core(native)
        native_map = core.forward_spatial_map(native)
        large_map = core.forward_spatial_map(large)

    assert stage1.shape == (2, 8, 9, 9)
    assert stage2.shape == (2, 8, 5, 5)
    assert stage3.shape == (2, 8, 3, 3)
    assert ordinary.shape == (2, 24, 1, 3, 3)
    assert torch.equal(native_map, ordinary)
    assert large_map.shape == (1, 24, 1, 5, 5)


def test_core_rejects_wrong_history_or_crop():
    core = _small_core()
    with pytest.raises(ValueError, match="exactly 5"):
        core(torch.randn(1, 1, 4, 9, 9))
    with pytest.raises(ValueError, match="9x9"):
        core(torch.randn(1, 1, 5, 11, 11))


@pytest.mark.parametrize(
    "normalization",
    ["groupnorm", "lrn", "groupnorm_lrn", "groupnorm_lrn_presplit"],
)
def test_normalization_variants_are_finite(normalization):
    core = _small_core(
        normalization={"type": normalization, "lrn_size": 5, "lrn_alpha": 0.1}
    ).eval()
    with torch.no_grad():
        output = core(torch.randn(2, 1, 5, 9, 9))
    assert output.shape == (2, 24, 1, 3, 3)
    assert torch.isfinite(output).all()


def test_frequency_mask_suppresses_temporal_nyquist_and_backpropagates():
    plain = HammingConv3d(1, 1, (60, 7, 7))
    masked = HammingConv3d(
        1,
        1,
        (60, 7, 7),
        frequency_mask_axes=(-3,),
        frequency_window="hann",
        frequency_fft_pad=2,
    )
    alternating = torch.where(
        torch.arange(60) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0)
    ).view(1, 1, 60, 1, 1).expand_as(plain.conv.weight)
    with torch.no_grad():
        plain.conv.weight.copy_(alternating)
        masked.conv.weight.copy_(alternating)

    plain_power = torch.fft.rfft(plain.weight.mean(dim=(-2, -1)).squeeze()).abs().square()
    masked_power = torch.fft.rfft(masked.weight.mean(dim=(-2, -1)).squeeze()).abs().square()
    assert masked_power[-1] < 1.0e-4 * plain_power[-1]
    masked.weight.square().mean().backward()
    assert torch.isfinite(masked.conv.weight.grad).all()


def test_center_crop_preserves_uint8_and_alignment():
    value = torch.arange(3 * 51 * 51, dtype=torch.int64).reshape(3, 51, 51).byte()
    crop = make_pipeline([{"center_crop": {"size": 35}}])(value)
    assert crop.dtype == torch.uint8
    assert torch.equal(crop, value[..., 8:43, 8:43])


def test_laplacian_and_ramp_then_constant_contract():
    smooth = torch.ones(2, 3, 9, 9)
    checker = torch.tensor(
        [[(-1.0) ** (i + j) for j in range(9)] for i in range(9)]
    ).expand(2, 3, -1, -1)
    assert laplacian_penalty(smooth, [-2, -1]) < laplacian_penalty(
        checker, [-2, -1]
    )

    parameter = torch.nn.Parameter(checker.clone())
    regularizer = Regularizer(
        {
            "name": "spatial_smoothness",
            "type": "laplacian",
            "lambda": 1.0e-3,
            "apply_to": ["weight"],
            "dims": [-2, -1],
            "schedule": {
                "kind": "ramp_then_constant",
                "start_epoch": 4,
                "end_epoch": 20,
            },
        },
        [("weight", parameter)],
    )
    assert regularizer.get_schedule_weight(3) == 0.0
    assert regularizer.get_schedule_weight(12) == pytest.approx(5.0e-4)
    assert regularizer.get_schedule_weight(20) == pytest.approx(1.0e-3)
    assert regularizer.get_schedule_weight(200) == pytest.approx(1.0e-3)


@pytest.mark.parametrize(
    ("filename", "expected_lambda"),
    [
        ("dekel_m77_4temporal_spatialsmooth_low_mlp_behavior.yaml", 5.0e-4),
        ("dekel_m77_4temporal_spatialsmooth_high_mlp_behavior.yaml", 1.0e-3),
    ],
)
def test_four_filter_configs_build_exact_production_ablation(
    filename, expected_lambda
):
    path = ROOT / "experiments/model_configs" / filename
    config = yaml.safe_load(path.read_text())
    model = build_model(config, [{"cids": list(range(5)), "behavior_dim": 42}])
    regs = create_regularizers(config, list(model.named_parameters()))
    smoothness = next(reg for reg in regs if reg.name == "first_layer_spatial_smoothness")

    assert config["convnet"]["params"]["temporal_channels"] == 4
    assert config["convnet"]["params"]["norm_groups"][0] == 4
    assert config["convnet"]["params"]["spatial_channels"] == [84, 84, 84]
    assert config["convnet"]["params"]["frequency_mask"]["temporal"] is True
    assert config["convnet"]["params"]["frequency_mask"]["stem_spatial"] is True
    assert config["convnet"]["params"]["normalization"]["type"] == "groupnorm_lrn_presplit"
    assert model.recurrent.__class__ is torch.nn.Identity
    assert model.convnet.stage1_conv.conv.in_channels == 8
    assert smoothness.lmbda == pytest.approx(expected_lambda)
    assert smoothness.param_names == ["convnet.temporal_conv.conv.weight"]

    rate = model(
        torch.randn(1, 1, 60, 35, 35),
        dataset_idx=0,
        behavior=torch.randn(1, 42),
    )
    assert rate.shape == (1, 5)
    assert torch.isfinite(rate).all() and torch.all(rate > 0)


def test_four_filter_sweep_changes_only_spatial_smoothness_strength():
    config_dir = ROOT / "experiments/model_configs"
    low = yaml.safe_load(
        (config_dir / "dekel_m77_4temporal_spatialsmooth_low_mlp_behavior.yaml").read_text()
    )
    high = yaml.safe_load(
        (config_dir / "dekel_m77_4temporal_spatialsmooth_high_mlp_behavior.yaml").read_text()
    )
    low["regularization"][0]["lambda"] = high["regularization"][0]["lambda"]
    assert low == high


def test_original_twin_config_still_builds():
    config = yaml.safe_load(
        (ROOT / "paper/model_selection/configs/width1.yaml").read_text()
    )
    model = build_model(config, [{"cids": [0, 1]}])
    assert model.convnet.__class__.__name__ == "ResNet"
    assert model.recurrent.__class__.__name__ == "ConvGRU"
