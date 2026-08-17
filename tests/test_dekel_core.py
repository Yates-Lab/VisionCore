"""Focused tests for the feed-forward Dekel core and its structured priors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from models import build_model
from models.data.datasets import CombinedEmbeddedDataset, DictDataset
from models.data.loading import causal_supervision_bin, subsample_embedded_supervision
from models.data.transforms import make_pipeline
from models.modules.dekel import DekelCore, HammingConv3d, SignPairNormReLU
from training.regularizers import Regularizer, create_regularizers, laplacian_penalty
from paper.model_selection.evaluate_rebinned_bps import adjacent_pair_positions


ROOT = Path(__file__).resolve().parents[1]


def _teacher_core():
    return DekelCore(
        {
            "initial_channels": 1,
            "temporal_support": 60,
            "temporal_channels": 4,
            "spatial_channels": [24, 24, 24],
            "spatial_kernels": [15, 11, 9],
            "norm_groups": [4, 8, 6, 4],
            "input_size": [35, 35],
        }
    )


def test_dekel_core_reproduces_teacher_stage_and_scaffold_shapes():
    core = _teacher_core()
    x = torch.randn(2, 1, 60, 35, 35)
    with torch.no_grad():
        stage1, stage2, stage3 = core.forward_stages(x)
        out = core(x)

    assert stage1.shape == (2, 48, 35, 35)
    assert stage2.shape == (2, 48, 18, 18)
    assert stage3.shape == (2, 48, 9, 9)
    assert out.shape == (2, 144, 1, 9, 9)
    assert core.get_output_channels() == 144


def test_dekel_spatial_map_matches_native_forward_and_retains_large_field():
    core = _teacher_core().eval()
    native = torch.randn(1, 1, 60, 35, 35)
    large = torch.randn(1, 1, 60, 151, 151)

    with torch.no_grad():
        ordinary = core(native)
        native_map = core.forward_spatial_map(native)
        large_map = core.forward_spatial_map(large)

    assert torch.equal(native_map, ordinary)
    assert large_map.shape == (1, 144, 1, 38, 38)
    assert torch.isfinite(large_map).all()


def test_dekel_core_rejects_wrong_history_or_crop():
    core = _teacher_core()
    with pytest.raises(ValueError, match="exactly 60"):
        core(torch.randn(1, 1, 59, 35, 35))
    with pytest.raises(ValueError, match="35x35"):
        core(torch.randn(1, 1, 60, 51, 51))


def test_surround_successor_preserves_all_warm_start_tensor_shapes():
    """Changing only aperture/scaffold geometry must permit a complete load."""
    narrow_config = yaml.safe_load(
        (ROOT / "experiments/model_configs/dekel_m16_output_behavior_gain_additive_dual84.yaml").read_text()
    )
    surround_config = yaml.safe_load(
        (ROOT / "experiments/model_configs/dekel_m27_surround51_dual84_fixedvision.yaml").read_text()
    )
    datasets = [{"cids": [3, 7, 11], "behavior_dim": 42}]
    narrow = build_model(narrow_config, datasets)
    surround = build_model(surround_config, datasets)

    assert surround.convnet.input_size == (51, 51)
    assert surround.convnet.scaffold_size == 13
    assert {
        name: tuple(value.shape) for name, value in narrow.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in surround.state_dict().items()
    }

    purefilm_narrow = build_model(
        yaml.safe_load(
            (ROOT / "experiments/model_configs/dekel_m28_biasless_purefilm_dual84.yaml").read_text()
        ),
        datasets,
    )
    purefilm_surround = build_model(
        yaml.safe_load(
            (ROOT / "experiments/model_configs/dekel_m29_surround51_biasless_purefilm_dual84.yaml").read_text()
        ),
        datasets,
    )
    assert purefilm_surround.convnet.input_size == (51, 51)
    assert purefilm_surround.convnet.scaffold_size == 13
    assert {
        name: tuple(value.shape)
        for name, value in purefilm_narrow.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in purefilm_surround.state_dict().items()
    }

    ryan_grid_narrow = build_model(
        yaml.safe_load(
            (ROOT / "experiments/model_configs/dekel_m22_readout_outputbehavior_refine.yaml").read_text()
        ),
        datasets,
    )
    ryan_grid_surround = build_model(
        yaml.safe_load(
            (ROOT / "experiments/model_configs/dekel_m31_surround51_ryanbehavior_fixedvision.yaml").read_text()
        ),
        datasets,
    )
    assert {
        name: tuple(value.shape)
        for name, value in ryan_grid_narrow.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in ryan_grid_surround.state_dict().items()
    }

    purefilm_joint = build_model(
        yaml.safe_load(
            (ROOT / "experiments/model_configs/dekel_m30_surround51_jointrefine.yaml").read_text()
        ),
        datasets,
    )
    assert {
        name: tuple(value.shape)
        for name, value in purefilm_surround.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in purefilm_joint.state_dict().items()
    }


def test_sign_pair_groupnorm_groups_contain_complete_opponent_pairs():
    layer = SignPairNormReLU(4, num_groups=4)
    x = torch.randn(3, 4, 7, 7)
    paired = torch.stack((x, -x), dim=2).flatten(1, 2)
    normalized = layer.norm(paired)
    grouped = normalized.reshape(3, 4, 2, 7, 7)
    assert torch.allclose(grouped.mean(dim=(2, 3, 4)), torch.zeros(3, 4), atol=1e-5)
    assert torch.all(layer(x) >= 0)


@pytest.mark.parametrize(
    "normalization",
    ["groupnorm", "lrn", "groupnorm_lrn", "groupnorm_lrn_presplit"],
)
def test_dekel_normalization_variants_preserve_shapes_and_finite_outputs(normalization):
    core = DekelCore(
        {
            "initial_channels": 1,
            "temporal_support": 5,
            "temporal_channels": 2,
            "spatial_channels": [4, 4, 4],
            "spatial_kernels": [3, 3, 3],
            "norm_groups": [2, 2, 2, 2],
            "input_size": [9, 9],
            "scaffold_size": 3,
            "normalization": {"type": normalization, "lrn_size": 5},
        }
    ).eval()
    with torch.no_grad():
        output = core(torch.randn(2, 1, 5, 9, 9))
    assert output.shape == (2, 24, 1, 3, 3)
    assert torch.isfinite(output).all()
    assert core.temporal_nonlinearity.normalization == normalization


def test_lrn_is_spatially_local_while_groupnorm_uses_the_whole_aperture():
    local = SignPairNormReLU(2, 2, normalization="lrn", lrn_size=5).eval()
    global_norm = SignPairNormReLU(2, 2, normalization="groupnorm").eval()
    baseline = torch.zeros(1, 2, 5, 5)
    baseline[0, 0, 2, 2] = 1.0
    remote_energy = baseline.clone()
    remote_energy[0, 0, 0, 0] = 100.0

    with torch.no_grad():
        local_baseline = local(baseline)[0, 0, 2, 2]
        local_remote = local(remote_energy)[0, 0, 2, 2]
        global_baseline = global_norm(baseline)[0, 0, 2, 2]
        global_remote = global_norm(remote_energy)[0, 0, 2, 2]

    assert torch.equal(local_baseline, local_remote)
    assert not torch.isclose(global_baseline, global_remote)


def test_first_layer_window_is_spatial_only():
    core = _teacher_core()
    mask = core.temporal_conv.spatial_window
    assert mask.shape == (1, 1, 1, 7, 7)
    assert mask[0, 0, 0, 3, 3] == pytest.approx(1.0)
    assert mask[0, 0, 0, 0, 0] < 0.1
    # Broadcasting, rather than a temporal window, is intentional.
    assert core.temporal_conv.weight.shape[2] == 60


def test_optional_frequency_mask_suppresses_temporal_nyquist_energy():
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
        torch.arange(60) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).view(1, 1, 60, 1, 1).expand_as(plain.conv.weight)
    with torch.no_grad():
        plain.conv.weight.copy_(alternating)
        masked.conv.weight.copy_(alternating)

    plain_temporal = plain.weight.mean(dim=(-2, -1)).squeeze()
    masked_temporal = masked.weight.mean(dim=(-2, -1)).squeeze()
    plain_nyquist = torch.fft.rfft(plain_temporal).abs().square()[-1]
    masked_nyquist = torch.fft.rfft(masked_temporal).abs().square()[-1]
    assert masked_nyquist < 1.0e-4 * plain_nyquist

    masked.weight.square().mean().backward()
    assert masked.conv.weight.grad is not None
    assert torch.isfinite(masked.conv.weight.grad).all()


def test_first_layer_svd_diagnostic_does_not_cancel_opponent_space():
    core = _teacher_core()
    temporal, spatial, separability = core.first_layer_separable_components()
    assert temporal.shape == (4, 60)
    assert spatial.shape == (4, 7, 7)
    assert separability.shape == (4,)
    assert torch.isfinite(temporal).all()
    assert torch.all((separability >= 0) & (separability <= 1))


def test_center_crop_is_dtype_preserving_and_centered():
    x = torch.arange(3 * 51 * 51, dtype=torch.uint8).reshape(3, 51, 51)
    crop = make_pipeline([{"center_crop": {"size": 35}}])(x)
    assert crop.shape == (3, 35, 35)
    assert crop.dtype == torch.uint8
    assert torch.equal(crop, x[..., 8:43, 8:43])


def test_resampled_pipeline_matches_lower_rate_behavior_contract_at_endpoints():
    x = torch.tensor(
        [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]
    )
    transform = make_pipeline(
        [
            {
                "resampled_pipeline": {
                    "factor": 2,
                    "reduction": "mean",
                    "ops": [{"diff": {"axis": 0}}, {"symlog": {}}],
                }
            }
        ]
    )
    pooled = x.reshape(2, 2, 2).mean(dim=1)
    expected_lower_rate = torch.diff(
        pooled, dim=0, prepend=pooled[:1]
    )
    expected_lower_rate = torch.sign(expected_lower_rate) * torch.log1p(
        expected_lower_rate.abs()
    )
    restored = transform(x)
    assert restored.shape == x.shape
    assert torch.allclose(restored[1::2], expected_lower_rate)
    assert torch.allclose(restored[0::2], expected_lower_rate)


def test_resampled_pipeline_preserves_odd_native_length():
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    transform = make_pipeline(
        [{"resampled_pipeline": {"factor": 2, "reduction": "mean", "ops": []}}]
    )
    restored = transform(x)
    assert restored.shape == x.shape
    assert torch.equal(restored[:4, 0], torch.tensor([0.5, 0.5, 2.5, 2.5]))
    assert restored[-1, 0] == 2.5


def test_dual_grid_eye_basis_exactly_matches_legacy_downsample_then_transform():
    time = torch.arange(120, dtype=torch.float32)
    eyepos = torch.stack(
        [torch.sin(time / 11.0) + time / 300.0, torch.cos(time / 17.0)],
        dim=1,
    )
    behavior_ops = [
        {"diff": {"axis": 0}},
        {"maxnorm": {}},
        {"symlog": {}},
        {
            "temporal_basis": {
                "num_delta_funcs": 0,
                "num_cosine_funcs": 10,
                "history_bins": 50,
                "causal": False,
                "log_spacing": False,
                "peak_range_ms": [30, 200],
                "normalize": True,
            }
        },
        {"splitrelu": {"split_dim": 1, "trainable_gain": False}},
    ]
    lower_rate_eye = eyepos.reshape(60, 2, 2).mean(dim=1)
    legacy = make_pipeline(behavior_ops)(lower_rate_eye)
    dual_grid = make_pipeline(
        [
            {
                "resampled_pipeline": {
                    "factor": 2,
                    "reduction": "mean",
                    "ops": behavior_ops,
                }
            }
        ]
    )(eyepos)
    assert dual_grid.shape == (120, 40)
    assert torch.equal(dual_grid[1::2], legacy)


def test_laplacian_penalty_prefers_smooth_kernels():
    smooth = torch.ones(2, 3, 9, 9)
    checker = torch.tensor(
        [[(-1.0) ** (i + j) for j in range(9)] for i in range(9)]
    ).expand(2, 3, -1, -1)
    assert laplacian_penalty(smooth, [-2, -1]) < laplacian_penalty(
        checker, [-2, -1]
    )


def test_ramp_then_constant_delays_then_preserves_structural_prior():
    param = torch.nn.Parameter(torch.ones(2, 3, 5, 5))
    reg = Regularizer(
        {
            "name": "delayed_smoothness",
            "type": "laplacian",
            "lambda": 1.0e-3,
            "apply_to": ["weight"],
            "dims": [-2, -1],
            "schedule": {
                "kind": "ramp_then_constant",
                "start_epoch": 2,
                "end_epoch": 10,
            },
        },
        [("weight", param)],
    )
    assert reg.get_schedule_weight(1) == 0.0
    assert reg.get_schedule_weight(2) == 0.0
    assert reg.get_schedule_weight(6) == pytest.approx(5.0e-4)
    assert reg.get_schedule_weight(10) == pytest.approx(1.0e-3)
    assert reg.get_schedule_weight(50) == pytest.approx(1.0e-3)


def test_regularizer_start_anchor_distinguishes_two_modulator_paths():
    inherited = torch.nn.Parameter(torch.ones(2, 2))
    output = torch.nn.Parameter(torch.ones(2, 2))
    reg = Regularizer(
        {
            "name": "inherited_behavior_only",
            "type": "l2",
            "lambda": 1.0e-5,
            "apply_to": ["^modulator/weight"],
        },
        [
            ("modulator.encoder.weight", inherited),
            ("output_modulator.encoder.weight", output),
        ],
    )

    assert reg.param_names == ["modulator.encoder.weight"]


def test_regularizer_start_anchor_distinguishes_base_and_auxiliary_readouts():
    base_std = torch.nn.Parameter(torch.zeros(2, 2))
    auxiliary_std_scale = torch.nn.Parameter(torch.zeros(2, 2))
    reg = Regularizer(
        {
            "name": "base_width_floor",
            "type": "proximal_clamp_min",
            "lambda": 0.5,
            "apply_to": ["^readouts/std"],
        },
        [
            ("readouts.0.std", base_std),
            ("auxiliary_readouts.0.std_scale_raw", auxiliary_std_scale),
        ],
    )

    assert reg.param_names == ["readouts.0.std"]


def test_competitive_prox_preserves_strongest_and_shrinks_weaker_groups():
    param = torch.nn.Parameter(
        torch.stack(
            (
                torch.full((2, 2), 3.0),
                torch.full((2, 2), 1.0),
                torch.full((2, 2), 0.1),
            )
        ).unsqueeze(0)
    )
    before = param.detach().clone()
    reg = Regularizer(
        {
            "name": "competition",
            "type": "proximal_sparsity_dekel",
            "lambda": 1.0,
            "apply_to": ["weight"],
            "group_dims": [-2, -1],
            "competition_dims": [1],
        },
        [("weight", param)],
    )
    reg.prox(epoch=0, lr=0.1)
    assert torch.allclose(param[:, 0], before[:, 0], atol=1e-6)
    assert torch.all(param[:, 1].abs() < before[:, 1].abs())
    assert torch.all(param[:, 2].abs() < before[:, 2].abs())


@pytest.mark.parametrize("floor", [0.25, 0.5])
def test_gaussian_readout_width_floor_is_enforced_exactly(floor):
    std = torch.nn.Parameter(torch.tensor([[0.01, 0.1], [0.4, 0.8]]))
    reg = Regularizer(
        {
            "name": "gaussian_width_floor",
            "type": "proximal_clamp_min",
            "lambda": floor,
            "apply_to": ["readouts/std"],
            "schedule": {"kind": "constant"},
        },
        [("readouts.0.std", std)],
    )

    reg.prox(epoch=0, lr=5.0e-4)

    assert torch.all(std >= floor)
    assert std[0, 0].item() == pytest.approx(floor)
    assert std[-1, -1].item() == pytest.approx(0.8)


def test_competitive_prox_skips_inactive_multisession_readout():
    active = torch.nn.Parameter(torch.tensor([[[3.0], [1.0]]]))
    inactive = torch.nn.Parameter(torch.tensor([[[3.0], [1.0]]]))
    optimizer = torch.optim.SGD([active, inactive], lr=0.1)
    active.grad = torch.ones_like(active)
    inactive.grad = None
    inactive_before = inactive.detach().clone()
    reg = Regularizer(
        {
            "name": "competition",
            "type": "proximal_sparsity_dekel",
            "lambda": 1.0,
            "apply_to": ["features"],
            "group_dims": [-1],
            "competition_dims": [1],
        },
        [("readout0.features", active), ("readout1.features", inactive)],
    )

    reg.prox(epoch=0, lr=0.1, optimizer=optimizer)

    assert active[0, 1, 0] < 1.0
    assert torch.equal(inactive, inactive_before)


def test_full_config_builds_with_nonrecurrent_behavior_and_exact_targets():
    path = ROOT / "experiments/model_configs/dekel_teacherwidth_mlp_behavior.yaml"
    config = yaml.safe_load(path.read_text())
    model = build_model(config, [{"cids": list(range(5))}])
    regs = create_regularizers(config, list(model.named_parameters()))
    by_name = {reg.name: reg.param_names for reg in regs}

    assert by_name["first_layer_spatial_smoothness"] == [
        "convnet.temporal_conv.conv.weight"
    ]
    assert by_name["hidden_spatial_smoothness"] == [
        "convnet.stage1_conv.conv.weight",
        "convnet.stage2_conv.conv.weight",
        "convnet.stage3_conv.conv.weight",
    ]
    assert by_name["hidden_competitive_sparsity"] == by_name[
        "hidden_spatial_smoothness"
    ]
    assert by_name["readout_competitive_sparsity"] == [
        "readouts.0.features.weight"
    ]
    assert model.recurrent.__class__ is torch.nn.Identity

    stimulus = torch.randn(2, 1, 60, 35, 35)
    behavior = torch.randn(2, 42)
    prediction = model(stimulus, dataset_idx=0, behavior=behavior)
    assert prediction.shape == (2, 5)
    assert torch.isfinite(prediction).all()
    assert (prediction > 0).all()
    prediction.mean().backward()
    assert model.convnet.temporal_conv.conv.weight.grad is not None
    assert model.modulator.encoder.mlp[0].linear.weight.grad is not None


def test_lowprox_sweep_preserves_architecture_and_smoothness_priors():
    primary = yaml.safe_load(
        (ROOT / "experiments/model_configs/dekel_multisession_mlp_behavior.yaml").read_text()
    )
    lowprox = yaml.safe_load(
        (
            ROOT
            / "experiments/model_configs/dekel_multisession_mlp_behavior_lowprox.yaml"
        ).read_text()
    )
    assert lowprox["convnet"] == primary["convnet"]
    assert lowprox["modulator"] == primary["modulator"]

    primary_regs = {spec["name"]: spec for spec in primary["regularization"]}
    lowprox_regs = {spec["name"]: spec for spec in lowprox["regularization"]}
    assert lowprox_regs["first_layer_temporal_smoothness"] == primary_regs[
        "first_layer_temporal_smoothness"
    ]
    assert lowprox_regs["first_layer_spatial_smoothness"] == primary_regs[
        "first_layer_spatial_smoothness"
    ]
    assert lowprox_regs["hidden_spatial_smoothness"] == primary_regs[
        "hidden_spatial_smoothness"
    ]
    assert lowprox_regs["hidden_competitive_sparsity"]["lambda"] == pytest.approx(
        3.0e-5
    )
    assert lowprox_regs["readout_competitive_sparsity"]["lambda"] == pytest.approx(
        3.0e-3
    )

    model = build_model(lowprox, [{"cids": [0, 1, 2]}])
    assert model.convnet.get_output_channels() == 288
    assert model.recurrent.__class__ is torch.nn.Identity

    noprox = yaml.safe_load(
        (
            ROOT
            / "experiments/model_configs/dekel_multisession_mlp_behavior_noprox.yaml"
        ).read_text()
    )
    assert noprox["convnet"] == primary["convnet"]
    assert noprox["modulator"] == primary["modulator"]
    noprox_regs = {spec["name"]: spec for spec in noprox["regularization"]}
    for name in (
        "first_layer_temporal_smoothness",
        "first_layer_spatial_smoothness",
        "hidden_spatial_smoothness",
    ):
        assert noprox_regs[name] == primary_regs[name]
    assert not any(
        spec["type"] == "proximal_sparsity_dekel"
        for spec in noprox["regularization"]
    )


def test_capacity_scaled_dekel_priors_preserve_historical_total_strength():
    config = yaml.safe_load(
        (
            ROOT
            / "experiments/model_configs/dekel_capacitymatched_freqmasked_readoutfloor0p5_scaleddekelreg_mlp_behavior.yaml"
        ).read_text()
    )
    regs = {spec["name"]: spec for spec in config["regularization"]}

    assert regs["first_layer_spatial_smoothness"]["lambda"] == pytest.approx(
        1.0e-3 * 4 / 14, rel=1.0e-4
    )
    assert regs["hidden_spatial_smoothness"]["lambda"] == pytest.approx(
        1.0e-3 * (24 / 84) ** 2, rel=1.0e-4
    )
    assert regs["gaussian_width_floor"]["lambda"] == pytest.approx(0.5)
    for name in (
        "first_layer_spatial_smoothness",
        "first_layer_temporal_smoothness",
        "hidden_spatial_smoothness",
        "hidden_competitive_sparsity",
        "readout_competitive_sparsity",
    ):
        assert regs[name]["schedule"] == {
            "kind": "ramp_then_constant",
            "start_epoch": 4,
            "end_epoch": 20,
        }


def test_capacity_regularization_ablations_change_only_intended_priors():
    config_dir = ROOT / "experiments/model_configs"
    full = yaml.safe_load(
        (
            config_dir
            / "dekel_capacitymatched_freqmasked_readoutfloor0p5_scaleddekelreg_mlp_behavior.yaml"
        ).read_text()
    )
    half = yaml.safe_load(
        (
            config_dir
            / "dekel_capacitymatched_freqmasked_readoutfloor0p5_halfscaleddekelreg_mlp_behavior.yaml"
        ).read_text()
    )
    smooth = yaml.safe_load(
        (
            config_dir
            / "dekel_capacitymatched_freqmasked_readoutfloor0p5_smoothonly_mlp_behavior.yaml"
        ).read_text()
    )
    half_smooth = yaml.safe_load(
        (
            config_dir
            / "dekel_capacitymatched_freqmasked_readoutfloor0p5_halfsmoothonly_mlp_behavior.yaml"
        ).read_text()
    )
    asymmetric = yaml.safe_load(
        (
            config_dir
            / "dekel_capacitymatched_freqmasked_readoutfloor0p5_fulltemporal_halfspatial_smoothonly_mlp_behavior.yaml"
        ).read_text()
    )

    assert half["convnet"] == full["convnet"] == smooth["convnet"] == half_smooth["convnet"]
    assert half["modulator"] == full["modulator"] == smooth["modulator"] == half_smooth["modulator"]
    assert half["readout"] == full["readout"] == smooth["readout"] == half_smooth["readout"]
    assert asymmetric["convnet"] == smooth["convnet"]
    assert asymmetric["modulator"] == smooth["modulator"]
    assert asymmetric["readout"] == smooth["readout"]

    full_regs = {spec["name"]: spec for spec in full["regularization"]}
    half_regs = {spec["name"]: spec for spec in half["regularization"]}
    smooth_regs = {spec["name"]: spec for spec in smooth["regularization"]}
    half_smooth_regs = {
        spec["name"]: spec for spec in half_smooth["regularization"]
    }
    asymmetric_regs = {
        spec["name"]: spec for spec in asymmetric["regularization"]
    }
    structural = (
        "first_layer_spatial_smoothness",
        "first_layer_temporal_smoothness",
        "hidden_spatial_smoothness",
        "hidden_competitive_sparsity",
        "readout_competitive_sparsity",
    )
    for name in structural:
        assert half_regs[name]["lambda"] == pytest.approx(
            0.5 * full_regs[name]["lambda"], rel=1.0e-4
        )
        assert half_regs[name]["schedule"] == full_regs[name]["schedule"]

    for name in (
        "first_layer_spatial_smoothness",
        "first_layer_temporal_smoothness",
        "hidden_spatial_smoothness",
    ):
        assert smooth_regs[name] == full_regs[name]
        assert half_smooth_regs[name]["lambda"] == pytest.approx(
            0.5 * smooth_regs[name]["lambda"], rel=1.0e-4
        )
        assert half_smooth_regs[name]["schedule"] == smooth_regs[name]["schedule"]
    for config in (smooth, half_smooth):
        assert not any(
            spec["type"] == "proximal_sparsity_dekel"
            for spec in config["regularization"]
        )
    assert asymmetric_regs["first_layer_temporal_smoothness"] == smooth_regs[
        "first_layer_temporal_smoothness"
    ]
    for name in (
        "first_layer_spatial_smoothness",
        "hidden_spatial_smoothness",
    ):
        assert asymmetric_regs[name] == half_smooth_regs[name]
    assert not any(
        spec["type"] == "proximal_sparsity_dekel"
        for spec in asymmetric["regularization"]
    )


def test_rebinned_evaluator_only_pairs_complete_bins_within_trial():
    raw_indices = torch.tensor([2, 3, 4, 5, 6, 8, 9])
    dataset = SimpleNamespace(
        inds=torch.stack((torch.zeros_like(raw_indices), raw_indices), dim=1),
        dsets=[
            {
                "trial_inds": torch.tensor(
                    [0, 0, 1, 1, 1, 2, 2, 3, 4, 4], dtype=torch.long
                )
            }
        ],
    )
    pairs = adjacent_pair_positions(dataset, 0)
    assert torch.equal(pairs, torch.tensor([[0, 1], [5, 6]]))


def test_mixed_rate_supervision_bins_counts_without_decimating_stimulus_axis():
    value = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    trials = torch.tensor([0, 0, 0, 1, 1, 1])
    counts = causal_supervision_bin(value, 2, "sum", trials)
    averages = causal_supervision_bin(value, 2, "mean", trials)
    assert torch.equal(counts[:, 0], torch.tensor([0.0, 1.0, 3.0, 0.0, 7.0, 9.0]))
    assert torch.equal(
        averages[:, 0], torch.tensor([0.0, 0.5, 1.5, 0.0, 3.5, 4.5])
    )

    raw = DictDataset(
        {
            "stim": torch.arange(6, dtype=torch.float32).reshape(6, 1),
            "robs": counts,
            "dfs": torch.ones_like(counts),
        },
        metadata={"name": "synthetic"},
    )
    embedded = CombinedEmbeddedDataset(
        raw,
        torch.arange(6),
        {"stim": [0], "robs": 0, "dfs": 0},
    )
    coarse_targets = subsample_embedded_supervision(embedded, factor=2, phase=1)
    assert torch.equal(coarse_targets.inds[:, 1], torch.tensor([1, 3, 5]))
    # The embedded input is still selected from native indices, not a
    # decimated stimulus tensor.
    assert torch.equal(coarse_targets[torch.arange(3)]["stim"][:, 0, 0], torch.tensor([1.0, 3.0, 5.0]))


def test_figure3_width1_config_still_builds_unchanged():
    path = ROOT / "paper/model_selection/configs/width1.yaml"
    config = yaml.safe_load(path.read_text())
    model = build_model(config, [{"cids": [0, 1]}])
    assert model.convnet.__class__.__name__ == "ResNet"
    assert model.recurrent.__class__.__name__ == "ConvGRU"
