"""Tests for the identity-preserving smooth spatial residual readout."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.modules.models import MultiDatasetV1Model
from models.modules.readout import (
    DynamicGaussianReadout,
    ResidualGaussianReadout,
)
from scripts.spatial_info import compute_rate_map, get_spatial_readout
from paper.model_selection.diagnose_dekel import summarize_residual_readouts


def test_residual_gaussian_is_exact_zero_and_starts_on_base_geometry():
    base = DynamicGaussianReadout(
        in_channels=4,
        n_units=3,
        initial_std=0.75,
        initial_mean_scale=0.0,
    )
    with torch.no_grad():
        base.mean.copy_(
            torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 1.5]])
        )
        base.theta.copy_(torch.tensor([0.1, -0.2, 0.3]))
    residual = ResidualGaussianReadout(
        in_channels=4,
        n_units=3,
        initial_std_scale=2.0,
    )
    stimulus_features = torch.randn(5, 4, 9, 9)

    assert torch.equal(
        residual(stimulus_features, base),
        torch.zeros(5, 3),
    )
    mean, std, theta = residual.effective_geometry(base)
    assert torch.equal(mean, base.mean)
    assert torch.allclose(std, 2.0 * base.std)
    assert torch.equal(theta, base.theta)


def test_residual_gaussian_learns_bounded_second_spatial_component():
    base = DynamicGaussianReadout(in_channels=2, n_units=3)
    residual = ResidualGaussianReadout(
        in_channels=2,
        n_units=3,
        max_mean_delta=3.0,
    )
    with torch.no_grad():
        residual.features.weight.fill_(0.2)
        residual.mean_delta.fill_(100.0)
    features = torch.randn(4, 2, 7, 7, requires_grad=True)

    output = residual(features, base)
    mean, _, _ = residual.effective_geometry(base)
    output.sum().backward()

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
    assert torch.all(mean - base.mean <= 3.0)
    assert torch.all(mean - base.mean >= -3.0)
    assert features.grad is not None


def test_base_scaled_residual_reuses_frozen_feature_selectivity():
    base = DynamicGaussianReadout(
        # The production base also has appended behavior channels.  The
        # matched surround must reuse only its leading visual columns.
        in_channels=5,
        n_units=3,
        initial_std=0.75,
    )
    residual = ResidualGaussianReadout(
        in_channels=2,
        n_units=3,
        feature_mode="base_scaled",
        max_base_scale=0.8,
    )
    features = torch.randn(4, 2, 7, 7, requires_grad=True)

    assert residual.features is None
    assert torch.equal(residual(features, base), torch.zeros(4, 3))

    with torch.no_grad():
        residual.base_scale.copy_(torch.tensor([0.5, -0.5, 100.0]))
    output = residual(features, base)
    output.sum().backward()

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
    assert features.grad is not None
    assert residual.base_scale.grad is not None
    effective_scale = residual.max_base_scale * torch.tanh(
        residual.base_scale.detach()
    )
    assert torch.all(effective_scale.abs() <= 0.8)
    # The base projection is a fixed reference, not an accidentally trainable
    # path through the residual head.
    assert base.features.weight.grad is None


def test_population_lowrank_residual_is_identity_and_bounded():
    base = DynamicGaussianReadout(in_channels=6, n_units=5)
    residual = ResidualGaussianReadout(
        in_channels=4,
        n_units=5,
        feature_mode="base_population_lowrank",
        population_rank=2,
        max_population_mix=0.3,
    )
    features = torch.randn(3, 4, 7, 7, requires_grad=True)

    assert torch.equal(residual(features, base), torch.zeros(3, 5))
    with torch.no_grad():
        residual.population_left.fill_(0.25)
    output = residual(features, base)
    output.sum().backward()

    raw_mix = (
        residual.population_left.detach()
        @ residual.population_right.detach().T
    ) / residual.population_rank ** 0.5
    effective_mix = residual.max_population_mix * torch.tanh(raw_mix)
    assert output.shape == (3, 5)
    assert torch.isfinite(output).all()
    assert torch.all(effective_mix.abs() <= 0.3)
    assert residual.population_left.grad is not None
    assert residual.population_right.grad is not None
    assert base.features.weight.grad is None
    assert features.grad is not None


def _tiny_gaussian_model_config(with_residual, feature_mode="independent"):
    config = {
        "model_type": "v1multi",
        "sampling_rate": 240,
        "initial_input_channels": 1,
        "adapter": {"type": "none", "params": {}},
        "frontend": {"type": "none", "params": {}},
        "convnet": {"type": "none", "params": {}},
        "modulator": {"type": "none", "params": {}},
        "recurrent": {"type": "none", "params": {}},
        "readout": {
            "type": "gaussian",
            "params": {
                "bias": True,
                "initial_std": 0.5,
                "initial_mean_scale": 0.1,
            },
        },
        "output_activation": "softplus",
    }
    if with_residual:
        config["residual_readout"] = {
            "type": "residual_gaussian",
            "params": {
                "input_scope": "visual",
                "max_mean_delta": 2.0,
                "initial_std_scale": 2.0,
                "feature_mode": feature_mode,
            },
        }
    return config


def test_multidataset_residual_readout_preserves_old_model_exactly():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(
        _tiny_gaussian_model_config(False), datasets
    )
    new = MultiDatasetV1Model(
        _tiny_gaussian_model_config(True), datasets
    )
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(name.startswith("residual_readouts.") for name in missing)

    stimulus = torch.randn(4, 1, 2, 5, 5)
    expected = old(stimulus, dataset_idx=0)
    actual = new(stimulus, dataset_idx=0)

    assert torch.equal(actual, expected)
    with torch.no_grad():
        new.residual_readouts[0].features.weight.fill_(0.1)
    assert not torch.equal(new(stimulus, dataset_idx=0), expected)


def test_multidataset_base_scaled_residual_is_identity_then_changes_output():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(
        _tiny_gaussian_model_config(False), datasets
    )
    new = MultiDatasetV1Model(
        _tiny_gaussian_model_config(True, "base_scaled"), datasets
    )
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(name.startswith("residual_readouts.") for name in missing)

    stimulus = torch.randn(4, 1, 2, 5, 5)
    expected = old(stimulus, dataset_idx=0)
    assert torch.equal(new(stimulus, dataset_idx=0), expected)

    with torch.no_grad():
        new.residual_readouts[0].base_scale.fill_(-0.2)
    assert not torch.equal(new(stimulus, dataset_idx=0), expected)


def test_residual_readout_diagnostic_separates_active_geometry():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    model = MultiDatasetV1Model(
        _tiny_gaussian_model_config(True), datasets
    )
    summary = summarize_residual_readouts(SimpleNamespace(model=model))
    assert summary["n_active_units"] == 0
    assert summary["active_center_displacement_scaffold_pixels"] is None

    with torch.no_grad():
        residual = model.residual_readouts[0]
        residual.features.weight[1].fill_(0.1)
        residual.mean_delta[1].fill_(0.25)
    summary = summarize_residual_readouts(SimpleNamespace(model=model))
    assert summary["n_active_units"] == 1
    assert summary["fraction_active"] == pytest.approx(1 / 3)
    assert summary["active_center_displacement_scaffold_pixels"]["minimum"] > 0


def _tiny_dekel_boost_config(
    with_auxiliary,
    with_auxiliary_residual=False,
    with_residual_visual=False,
):
    core = {
        "temporal_support": 2,
        "temporal_channels": 2,
        "spatial_channels": [2, 2, 2],
        "spatial_kernels": [3, 3, 3],
        "norm_groups": [2, 2, 2, 2],
        "input_size": [5, 5],
        "scaffold_size": 2,
        "strict_input_size": True,
        "frequency_mask": {
            "temporal": True,
            "stem_spatial": True,
            "hidden_spatial": True,
        },
    }
    config = {
        "model_type": "v1multi",
        "sampling_rate": 240,
        "initial_input_channels": 1,
        "adapter": {"type": "none", "params": {}},
        "frontend": {"type": "none", "params": {}},
        "convnet": {"type": "dekel", "params": dict(core)},
        "modulator": {"type": "none", "params": {}},
        "recurrent": {"type": "none", "params": {}},
        "readout": {
            "type": "gaussian",
            "params": {
                "bias": True,
                "initial_std": 0.5,
                "initial_mean_scale": 0.1,
            },
        },
        "output_activation": "softplus",
    }
    if with_auxiliary:
        config["auxiliary_visual"] = {
            "type": "dekel",
            "params": {
                "core": dict(core),
                "readout": {
                    "max_mean_delta": 2.0,
                    "initial_std_scale": 1.0,
                    "min_std_scale": 0.5,
                    "max_std_scale": 2.0,
                },
            },
        }
    if with_auxiliary_residual:
        config["auxiliary_residual_readout"] = {
            "type": "residual_gaussian",
            "params": {
                "input_scope": "auxiliary_visual",
                "max_mean_delta": 2.0,
                "initial_std_scale": 1.5,
                "min_std_scale": 0.75,
                "max_std_scale": 3.0,
                "feature_mode": "independent",
            },
        }
    if with_residual_visual:
        config["residual_visual"] = {
            "type": "dekel",
            "params": {
                "core": dict(core),
                "readout": {
                    "feature_mode": "independent",
                    "max_mean_delta": 2.0,
                    "initial_std_scale": 1.0,
                    "min_std_scale": 0.5,
                    "max_std_scale": 2.0,
                },
            },
        }
    return config


def test_auxiliary_visual_branch_is_exact_identity_then_learns_residual():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(
        _tiny_dekel_boost_config(False), datasets
    )
    new = MultiDatasetV1Model(
        _tiny_dekel_boost_config(True), datasets
    )
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("auxiliary_convnet.", "auxiliary_readouts."))
        for name in missing
    )

    stimulus = torch.randn(4, 1, 2, 5, 5)
    expected = old(stimulus, dataset_idx=0)
    actual = new(stimulus, dataset_idx=0)
    assert torch.equal(actual, expected)

    with torch.no_grad():
        new.auxiliary_readouts[0].features.weight.fill_(0.05)
    changed = new(stimulus, dataset_idx=0)
    assert not torch.equal(changed, expected)

    changed.sum().backward()
    assert new.auxiliary_readouts[0].features.weight.grad is not None
    assert new.auxiliary_convnet.temporal_conv.conv.weight.grad is not None

    ordinary_features = new.auxiliary_convnet(stimulus)
    spatial_features = new.auxiliary_visual_forward_spatial_map(stimulus)
    assert torch.equal(spatial_features, ordinary_features)

    wrapper = SimpleNamespace(
        model=new,
        names=["test"],
        cfgs=datasets,
        device=torch.device("cpu"),
    )
    outputs = [
        {
            "sess": "test",
            "cids_used": np.asarray([10, 20, 30]),
            "ccnorm": {"ccnorm": np.ones(3)},
        }
    ]
    population_readout = get_spatial_readout(wrapper, outputs)
    spatial_rates = compute_rate_map(
        wrapper,
        population_readout,
        stimulus,
    )
    assert torch.allclose(
        spatial_rates[:, :, 0, 0],
        changed,
        rtol=1e-6,
        atol=1e-6,
    )


def test_second_auxiliary_component_is_identity_then_adds_localized_output():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    parent = MultiDatasetV1Model(
        _tiny_dekel_boost_config(True), datasets
    )
    child = MultiDatasetV1Model(
        _tiny_dekel_boost_config(True, with_auxiliary_residual=True),
        datasets,
    )
    missing, unexpected = child.load_state_dict(
        parent.state_dict(), strict=False
    )
    assert not unexpected
    assert missing
    assert all(
        name.startswith("auxiliary_residual_readouts.")
        for name in missing
    )

    stimulus = torch.randn(4, 1, 2, 5, 5)
    expected = parent(stimulus, dataset_idx=0)
    assert torch.equal(child(stimulus, dataset_idx=0), expected)

    with torch.no_grad():
        child.auxiliary_residual_readouts[0].features.weight.fill_(0.05)
    changed = child(stimulus, dataset_idx=0)
    assert not torch.equal(changed, expected)
    changed.sum().backward()
    assert (
        child.auxiliary_residual_readouts[0].features.weight.grad
        is not None
    )
    assert child.auxiliary_convnet.temporal_conv.conv.weight.grad is not None


def test_auxiliary_component_requires_auxiliary_visual_core():
    config = _tiny_dekel_boost_config(False)
    config["auxiliary_residual_readout"] = {
        "type": "residual_gaussian",
        "params": {},
    }
    with pytest.raises(ValueError, match="requires auxiliary_visual"):
        MultiDatasetV1Model(
            config,
            [{"session": "test", "cids": [10]}],
        )


def test_smooth_residual_core_is_exact_identity_then_learns_new_features():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    parent = MultiDatasetV1Model(
        _tiny_dekel_boost_config(True), datasets
    )
    child = MultiDatasetV1Model(
        _tiny_dekel_boost_config(True, with_residual_visual=True),
        datasets,
    )
    missing, unexpected = child.load_state_dict(
        parent.state_dict(), strict=False
    )
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("residual_convnet.", "residual_visual_readouts."))
        for name in missing
    )

    stimulus = torch.randn(4, 1, 2, 5, 5)
    expected = parent(stimulus, dataset_idx=0)
    assert torch.equal(child(stimulus, dataset_idx=0), expected)

    with torch.no_grad():
        child.residual_visual_readouts[0].features.weight.fill_(0.05)
    changed = child(stimulus, dataset_idx=0)
    assert not torch.equal(changed, expected)
    changed.sum().backward()
    assert child.residual_convnet.temporal_conv.conv.weight.grad is not None
    assert (
        child.residual_visual_readouts[0].features.weight.grad is not None
    )


def test_dual_aperture_preserves_center_model_and_exposes_surround_to_auxiliary():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(
        _tiny_dekel_boost_config(False), datasets
    )
    dual_config = _tiny_dekel_boost_config(True)
    dual_config["base_input_crop"] = 5
    dual_config["auxiliary_visual"]["params"]["core"]["input_size"] = [7, 7]
    new = MultiDatasetV1Model(dual_config, datasets)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("auxiliary_convnet.", "auxiliary_readouts."))
        for name in missing
    )

    stimulus = torch.randn(4, 1, 2, 7, 7)
    center = stimulus[..., 1:6, 1:6]
    expected = old(center, dataset_idx=0)
    actual = new(stimulus, dataset_idx=0)
    assert torch.equal(actual, expected)
    assert torch.equal(
        new.core_forward(stimulus),
        old.core_forward(center),
    )
    assert torch.equal(
        new.auxiliary_visual_forward_spatial_map(stimulus),
        new.auxiliary_convnet(stimulus),
    )

    with torch.no_grad():
        new.auxiliary_readouts[0].features.weight.fill_(0.05)
    changed = new(stimulus, dataset_idx=0)
    assert not torch.equal(changed, expected)

    wrapper = SimpleNamespace(
        model=new,
        names=["test"],
        cfgs=datasets,
        device=torch.device("cpu"),
    )
    outputs = [
        {
            "sess": "test",
            "cids_used": np.asarray([10, 20, 30]),
            "ccnorm": {"ccnorm": np.ones(3)},
        }
    ]
    population_readout = get_spatial_readout(wrapper, outputs)
    spatial_rates = compute_rate_map(
        wrapper,
        population_readout,
        stimulus,
    )
    assert spatial_rates.shape[-2:] == (1, 1)
    assert torch.allclose(
        spatial_rates[:, :, 0, 0],
        changed,
        rtol=1e-6,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="base_input_crop exceeds"):
        new(torch.randn(1, 1, 2, 3, 3), dataset_idx=0)


def test_longer_history_auxiliary_preserves_shorter_mature_core():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(
        _tiny_dekel_boost_config(False), datasets
    )
    longer_config = _tiny_dekel_boost_config(True)
    longer_config["base_input_temporal_support"] = 2
    longer_config["auxiliary_visual"]["params"]["core"][
        "temporal_support"
    ] = 4
    new = MultiDatasetV1Model(longer_config, datasets)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("auxiliary_convnet.", "auxiliary_readouts."))
        for name in missing
    )

    stimulus = torch.randn(4, 1, 4, 5, 5)
    expected = old(stimulus[:, :, :2], dataset_idx=0)
    actual = new(stimulus, dataset_idx=0)
    assert torch.equal(actual, expected)
    assert torch.equal(
        new.core_forward(stimulus),
        old.core_forward(stimulus[:, :, :2]),
    )

    with torch.no_grad():
        new.auxiliary_readouts[0].features.weight.fill_(0.05)
    changed = new(stimulus, dataset_idx=0)
    assert not torch.equal(changed, expected)

    with pytest.raises(
        ValueError, match="base_input_temporal_support exceeds"
    ):
        new(torch.randn(1, 1, 1, 5, 5), dataset_idx=0)
