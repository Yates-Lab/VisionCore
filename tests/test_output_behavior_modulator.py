"""Tests for the identity-preserving M16 behavior-residual path."""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from models.modules.models import MultiDatasetV1Model
from models.modules.modulator import (
    MLPBehaviorModulator,
    MultiDatasetBehaviorOutputModulator,
)
from paper.model_selection.evaluate_output_behavior_residual import (
    output_residual_conditions,
)
from training.pl_modules.multidataset_model import (
    MultiDatasetModel,
    _adamw_param_groups_named,
)


def _output_config(use_gain=True):
    return {
        "behavior_dim": 4,
        "hidden_dims": [7],
        "bottleneck_dim": 3,
        "activation": "gelu",
        "dropout": 0.0,
        "use_gain": use_gain,
        "max_gain": 0.5,
    }


@pytest.mark.parametrize("use_gain", [False, True])
def test_zero_initialized_behavior_residual_is_exact_identity(use_gain):
    module = MultiDatasetBehaviorOutputModulator(
        _output_config(use_gain), [3, 5]
    )
    logits = torch.randn(6, 5)
    behavior = torch.randn(6, 4)

    actual = module(logits, behavior, dataset_idx=1)

    assert torch.equal(actual, logits)


def test_gain_is_bounded_and_does_not_mix_neuron_jacobians():
    module = MultiDatasetBehaviorOutputModulator(_output_config(True), [3])
    with torch.no_grad():
        module.gain_layers[0].weight.fill_(4.0)
        module.gain_layers[0].bias.fill_(4.0)
        module.offset_layers[0].weight.fill_(0.2)

    logits = torch.randn(2, 3, requires_grad=True)
    behavior = torch.randn(2, 4)
    output = module(logits, behavior, dataset_idx=0)
    jacobian = torch.autograd.grad(output[0, 1], logits, retain_graph=True)[0]

    # A fixed behavior context can only rescale the matching unit's visual
    # derivative; it cannot create a spatial/temporal or cross-unit pathway.
    assert torch.count_nonzero(jacobian[0]).item() == 1
    assert jacobian[0, 0] == 0
    assert jacobian[0, 2] == 0
    assert 0.5 <= jacobian[0, 1].item() <= 1.5


def test_mlp_behavior_supports_biasless_pure_film_without_additive_channels():
    module = MLPBehaviorModulator(
        {
            "behavior_dim": 4,
            "feature_dim": 6,
            "hidden_dims": [7],
            "encoded_dim": 5,
            "additive_dim": 0,
            "use_film": True,
            "film_bias": False,
            "film_max_gain": 1.0,
        }
    )
    features = torch.randn(3, 6, 1, 5, 5)
    behavior = torch.randn(3, 4)

    assert module.out_dim == 0
    assert module.scale_layer.bias is None
    assert torch.equal(module(features, behavior), features)

    with torch.no_grad():
        module.scale_layer.weight.fill_(0.1)
    changed = module(features, behavior)
    assert changed.shape == features.shape
    assert not torch.equal(changed, features)


def test_public_gain_offset_matches_forward_transform():
    module = MultiDatasetBehaviorOutputModulator(_output_config(True), [3])
    with torch.no_grad():
        module.offset_layers[0].weight.fill_(0.2)
        module.gain_layers[0].weight.fill_(0.1)
    logits = torch.randn(4, 3)
    behavior = torch.randn(4, 4)

    gain, offset = module.gain_offset(behavior, 0)

    assert torch.allclose(
        module(logits, behavior, 0),
        logits * (1.0 + gain) + offset,
    )


def test_dataset_behavior_adapter_is_identity_initialized_and_session_specific():
    config = _output_config(True)
    config.update(
        dataset_adapter_dim=2,
        dataset_adapter_max_scale=0.5,
    )
    module = MultiDatasetBehaviorOutputModulator(config, [3, 3])
    with torch.no_grad():
        for layer in module.offset_layers:
            layer.weight.fill_(0.2)
        for layer in module.gain_layers:
            layer.weight.fill_(0.1)
    reference = MultiDatasetBehaviorOutputModulator(
        _output_config(True), [3, 3]
    )
    reference.load_state_dict(
        {
            name: value
            for name, value in module.state_dict().items()
            if not name.startswith("dataset_adapters.")
        },
        strict=True,
    )
    logits = torch.randn(6, 3)
    behavior = torch.randn(6, 4)

    # Adding adapters to a mature checkpoint must not perturb either session
    # until the new residual projection is trained.
    for dataset_idx in (0, 1):
        assert torch.equal(
            module(logits, behavior, dataset_idx),
            reference(logits, behavior, dataset_idx),
        )

    with torch.no_grad():
        module.dataset_adapters[0][-1].weight.fill_(0.25)
    session_zero = module(logits, behavior, 0)
    session_one = module(logits, behavior, 1)

    assert not torch.equal(session_zero, reference(logits, behavior, 0))
    assert torch.equal(session_one, reference(logits, behavior, 1))


def test_dataset_behavior_adapter_preserves_diagonal_visual_jacobian():
    config = _output_config(True)
    config["dataset_adapter_dim"] = 2
    module = MultiDatasetBehaviorOutputModulator(config, [3])
    with torch.no_grad():
        module.dataset_adapters[0][-1].weight.fill_(0.2)
        module.gain_layers[0].weight.fill_(0.1)
        module.offset_layers[0].weight.fill_(0.1)

    logits = torch.randn(2, 3, requires_grad=True)
    behavior = torch.randn(2, 4)
    output = module(logits, behavior, dataset_idx=0)
    jacobian = torch.autograd.grad(output[0, 1], logits)[0]

    assert torch.count_nonzero(jacobian[0]).item() == 1
    assert jacobian[0, 0] == 0
    assert jacobian[0, 2] == 0


def test_output_behavior_residual_validates_inputs():
    module = MultiDatasetBehaviorOutputModulator(_output_config(True), [3])
    with pytest.raises(ValueError, match="behavior shape"):
        module(torch.randn(2, 3), torch.randn(2, 5), 0)
    with pytest.raises(ValueError, match="unit mismatch"):
        module(torch.randn(2, 4), torch.randn(2, 4), 0)
    with pytest.raises(IndexError, match="out of range"):
        module(torch.randn(2, 3), torch.randn(2, 4), 2)


def _tiny_model_config(with_output):
    config = {
        "model_type": "v1multi",
        "sampling_rate": 240,
        "initial_input_channels": 1,
        "adapter": {"type": "none", "params": {}},
        "frontend": {"type": "none", "params": {}},
        "convnet": {"type": "none", "params": {}},
        "modulator": {"type": "none", "params": {}},
        "recurrent": {"type": "none", "params": {}},
        "readout": {"type": "linear", "params": {"bias": True}},
        "output_activation": "softplus",
    }
    if with_output:
        config["output_modulator"] = {
            "type": "mlp_behavior_residual",
            "params": _output_config(True),
        }
    return config


def test_multidataset_model_adds_head_without_changing_old_predictions():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    old = MultiDatasetV1Model(_tiny_model_config(False), datasets)
    new = MultiDatasetV1Model(_tiny_model_config(True), datasets)
    new.load_state_dict(old.state_dict(), strict=False)

    stimulus = torch.randn(4, 1, 2, 5, 5)
    behavior = torch.randn(4, 4)
    expected = old(stimulus, dataset_idx=0)
    actual = new(stimulus, dataset_idx=0, behavior=behavior)

    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="behavior-conditioned"):
        new(stimulus, dataset_idx=0, behavior=None)


def test_distilled_second_head_is_identity_initialized_and_sequential():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    base_config = _tiny_model_config(True)
    config = deepcopy(base_config)
    config["distilled_output_modulator"] = {
        "type": "mlp_behavior_residual",
        "params": _output_config(True),
    }
    base = MultiDatasetV1Model(base_config, datasets)
    model = MultiDatasetV1Model(config, datasets)
    model.load_state_dict(base.state_dict(), strict=False)
    stimulus = torch.randn(5, 1, 2, 5, 5)
    behavior = torch.randn(5, 4)

    expected = base(stimulus, dataset_idx=0, behavior=behavior)
    actual = model(stimulus, dataset_idx=0, behavior=behavior)
    assert torch.equal(actual, expected)

    with torch.no_grad():
        model.distilled_output_modulator.offset_layers[0].bias.fill_(0.2)
    changed = model(stimulus, dataset_idx=0, behavior=behavior)
    assert not torch.equal(changed, expected)


def test_paired_evaluator_matches_normal_forward_and_inherited_model():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    inherited = MultiDatasetV1Model(_tiny_model_config(False), datasets)
    successor = MultiDatasetV1Model(_tiny_model_config(True), datasets)
    successor.load_state_dict(inherited.state_dict(), strict=False)
    with torch.no_grad():
        successor.output_modulator.offset_layers[0].weight.fill_(0.1)
        successor.output_modulator.gain_layers[0].weight.fill_(0.05)

    stimulus = torch.randn(5, 1, 2, 5, 5)
    behavior = torch.randn(5, 4)
    conditions, gain, offset = output_residual_conditions(
        successor, stimulus, behavior, 0
    )

    assert torch.allclose(
        conditions["intact"],
        successor(stimulus, dataset_idx=0, behavior=behavior),
    )
    assert torch.allclose(
        conditions["inherited"],
        inherited(stimulus, dataset_idx=0),
    )
    assert gain.shape == offset.shape == (5, 3)


def test_separate_output_behavior_leaves_inherited_behavior_path_unchanged():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    model = MultiDatasetV1Model(_tiny_model_config(True), datasets)
    with torch.no_grad():
        model.output_modulator.offset_layers[0].weight.fill_(0.2)
        model.output_modulator.gain_layers[0].weight.fill_(0.1)

    stimulus = torch.randn(5, 1, 2, 5, 5)
    inherited_behavior = torch.randn(5, 4)
    output_behavior = torch.randn(5, 4)
    expected, _, _ = output_residual_conditions(
        model,
        stimulus,
        inherited_behavior,
        0,
        output_behavior=output_behavior,
    )
    actual = model(
        stimulus,
        dataset_idx=0,
        behavior=inherited_behavior,
        output_behavior=output_behavior,
    )
    assert torch.allclose(actual, expected["intact"])


def test_output_residual_can_concatenate_native_and_lower_rate_behavior():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    config = _tiny_model_config(True)
    config["output_modulator"]["input_mode"] = (
        "concatenate_feature_and_output"
    )
    config["output_modulator"]["params"]["behavior_dim"] = 8
    model = MultiDatasetV1Model(config, datasets)
    with torch.no_grad():
        model.output_modulator.offset_layers[0].weight.fill_(0.2)

    stimulus = torch.randn(5, 1, 2, 5, 5)
    native = torch.randn(5, 4)
    lower_rate = torch.randn(5, 4)
    actual = model(
        stimulus,
        dataset_idx=0,
        behavior=native,
        output_behavior=lower_rate,
    )
    conditions, _, _ = output_residual_conditions(
        model,
        stimulus,
        native,
        0,
        output_behavior=lower_rate,
    )
    assert torch.allclose(actual, conditions["intact"])
    assert torch.equal(
        model.resolve_output_behavior(native, lower_rate),
        torch.cat([native, lower_rate], dim=-1),
    )
    with pytest.raises(ValueError, match="requires both"):
        model(stimulus, dataset_idx=0, behavior=native)


def test_feature_modulator_can_use_separate_lower_rate_behavior_contract():
    datasets = [{"session": "test", "cids": [10, 20, 30]}]
    config = _tiny_model_config(True)
    config["feature_behavior_source"] = "output_behavior"
    # Use a behavior-conditioned feature path so routing is observable.
    config["modulator"] = {
        "type": "mlp_behavior",
        "params": {
            "behavior_dim": 4,
            "hidden_dims": [5],
            "encoded_dim": 3,
            "additive_dim": 0,
            "use_film": True,
            "film_bias": False,
        },
    }
    model = MultiDatasetV1Model(config, datasets)
    native = torch.randn(5, 4)
    lower_rate = torch.randn(5, 4)
    assert torch.equal(
        model.resolve_feature_behavior(native, lower_rate), lower_rate
    )
    with pytest.raises(ValueError, match="requires an output_behavior"):
        model.resolve_feature_behavior(native, None)


class _TinyWarmStartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.frontend = nn.Linear(2, 2)
        self.convnet = nn.Linear(2, 2)
        self.modulator = nn.Linear(2, 2)
        self.recurrent = nn.Linear(2, 2)
        self.readouts = nn.ModuleList([nn.Linear(2, 3)])
        self.output_modulator = nn.Linear(3, 3)


def _bare_lightning_model():
    wrapper = MultiDatasetModel.__new__(MultiDatasetModel)
    nn.Module.__init__(wrapper)
    wrapper.model = _TinyWarmStartModel()
    wrapper.names = ["session_a"]
    wrapper.cfgs = [{"cids": [7, 8, 9]}]
    wrapper._pretrained_parameter_names = set()
    wrapper._pretrained_readout_indices = set()
    return wrapper


class _TinyGaussianLikeReadout(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Conv2d(in_channels, 3, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.mean = nn.Parameter(torch.zeros(3, 2))
        self.std = nn.Parameter(torch.ones(3, 2))
        self.theta = nn.Parameter(torch.zeros(3))


class _TinyAdaptWarmStartModel(nn.Module):
    def __init__(self, readout_channels, scaffold_size=9):
        super().__init__()
        self.model_config = {
            "convnet": {"params": {"scaffold_size": scaffold_size}}
        }
        self.adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.frontend = nn.Linear(2, 2)
        self.convnet = nn.Linear(2, 2)
        self.modulator = nn.Linear(2, 2)
        self.recurrent = nn.Linear(2, 2)
        self.readouts = nn.ModuleList(
            [_TinyGaussianLikeReadout(readout_channels)]
        )
        self.output_modulator = nn.Linear(3, 3)


def _bare_adapt_lightning_model(readout_channels, scaffold_size=9):
    wrapper = MultiDatasetModel.__new__(MultiDatasetModel)
    nn.Module.__init__(wrapper)
    wrapper.model = _TinyAdaptWarmStartModel(readout_channels, scaffold_size)
    wrapper.model_config = wrapper.model.model_config
    wrapper.names = ["session_a"]
    wrapper.cfgs = [{"cids": [7, 8, 9]}]
    wrapper._pretrained_parameter_names = set()
    wrapper._pretrained_readout_indices = set()
    return wrapper


def test_distilled_output_artifact_loads_with_exact_dataset_and_cid_contract(tmp_path):
    config = _output_config(True)
    source = MultiDatasetBehaviorOutputModulator(config, [3])
    with torch.no_grad():
        source.offset_layers[0].weight.normal_()
        source.gain_layers[0].weight.normal_()
    artifact = tmp_path / "distilled.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "config": config,
            "dataset_names": ["session_a"],
            "cids_by_session": {"session_a": [7, 8, 9]},
        },
        artifact,
    )

    target = _bare_lightning_model()
    target.model.distilled_output_modulator = (
        MultiDatasetBehaviorOutputModulator(config, [3])
    )
    target.model_config = {
        "distilled_output_modulator": {
            "type": "mlp_behavior_residual",
            "params": config,
        }
    }
    loaded = target.load_distilled_output_modulator(str(artifact))
    assert loaded == len(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(
            target.model.distilled_output_modulator.state_dict()[key], value
        )

    wrong = torch.load(artifact, weights_only=False)
    wrong["cids_by_session"] = {"session_a": [9, 8, 7]}
    wrong_path = tmp_path / "wrong.pt"
    torch.save(wrong, wrong_path)
    with pytest.raises(ValueError, match="neuron identities"):
        target.load_distilled_output_modulator(str(wrong_path))


def test_compatible_warm_start_loads_and_freezes_every_old_tensor(tmp_path):
    source = _TinyWarmStartModel()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index / 10.0)
    source_state = {
        "model." + name: value.detach().clone()
        for name, value in source.state_dict().items()
        if not name.startswith("output_modulator")
    }
    checkpoint = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": source_state,
            "hyper_parameters": {"dataset_cids": {"session_a": [7, 8, 9]}},
        },
        checkpoint,
    )

    target = _bare_lightning_model()
    output_before = deepcopy(target.model.output_modulator.state_dict())
    loaded = target._load_pretrained_components(
        str(checkpoint),
        pretrained_scope="compatible",
        freeze_pretrained=True,
    )

    assert loaded == len(source_state)
    for name, value in target.model.state_dict().items():
        if name.startswith("output_modulator"):
            assert torch.equal(value, output_before[name.split('.', 1)[1]])
        else:
            assert torch.equal(value, source.state_dict()[name])
    assert target._pretrained_readout_indices == {0}
    assert all(
        not parameter.requires_grad
        for name, parameter in target.model.named_parameters()
        if not name.startswith("output_modulator")
    )
    assert all(
        parameter.requires_grad
        for parameter in target.model.output_modulator.parameters()
    )


def test_compatible_warm_start_refuses_changed_neuron_order(tmp_path):
    source = _TinyWarmStartModel()
    checkpoint = tmp_path / "wrong-cids.ckpt"
    torch.save(
        {
            "state_dict": {
                "model." + name: value for name, value in source.state_dict().items()
                if not name.startswith("output_modulator")
            },
            "hyper_parameters": {"dataset_cids": {"session_a": [9, 8, 7]}},
        },
        checkpoint,
    )

    target = _bare_lightning_model()
    with pytest.raises(ValueError, match="neuron identities/order"):
        target._load_pretrained_components(
            str(checkpoint), pretrained_scope="compatible"
        )


def test_complete_warm_start_includes_an_existing_output_head(tmp_path):
    source = _TinyWarmStartModel()
    with torch.no_grad():
        source.output_modulator.weight.fill_(0.75)
        source.output_modulator.bias.fill_(-0.25)
    checkpoint = tmp_path / "complete.ckpt"
    torch.save(
        {
            "state_dict": {
                "model." + name: value for name, value in source.state_dict().items()
            },
            "hyper_parameters": {"dataset_cids": {"session_a": [7, 8, 9]}},
        },
        checkpoint,
    )

    target = _bare_lightning_model()
    loaded = target._load_pretrained_components(
        str(checkpoint), pretrained_scope="complete"
    )

    assert loaded == len(source.state_dict())
    assert torch.equal(
        target.model.output_modulator.weight,
        source.output_modulator.weight,
    )


def test_complete_warm_start_can_add_identity_session_behavior_adapters(tmp_path):
    source = _TinyWarmStartModel()
    source.output_modulator = MultiDatasetBehaviorOutputModulator(
        _output_config(True), [3]
    )
    with torch.no_grad():
        source.output_modulator.offset_layers[0].weight.normal_()
        source.output_modulator.gain_layers[0].weight.normal_()
    checkpoint = tmp_path / "source-with-shared-behavior-head.ckpt"
    torch.save(
        {
            "state_dict": {
                "model." + name: value.detach().clone()
                for name, value in source.state_dict().items()
            },
            "hyper_parameters": {
                "dataset_cids": {"session_a": [7, 8, 9]}
            },
        },
        checkpoint,
    )

    target = _bare_lightning_model()
    adapter_config = _output_config(True)
    adapter_config["dataset_adapter_dim"] = 2
    target.model.output_modulator = MultiDatasetBehaviorOutputModulator(
        adapter_config, [3]
    )
    target._load_pretrained_components(
        str(checkpoint),
        pretrained_scope="complete",
        freeze_pretrained=True,
        exclude_prefixes="output_modulator.dataset_adapters",
    )

    logits = torch.randn(5, 3)
    behavior = torch.randn(5, 4)
    assert torch.equal(
        target.model.output_modulator(logits, behavior, 0),
        source.output_modulator(logits, behavior, 0),
    )
    assert all(
        parameter.requires_grad == name.startswith(
            "output_modulator.dataset_adapters"
        )
        for name, parameter in target.model.named_parameters()
    )


def test_explicit_warm_start_can_reset_modulator_and_trim_readout_channels(tmp_path):
    source = _TinyAdaptWarmStartModel(readout_channels=6)
    with torch.no_grad():
        source.readouts[0].features.weight.copy_(
            torch.arange(18, dtype=torch.float32).reshape(3, 6, 1, 1)
        )
        source.modulator.weight.fill_(4.0)
    checkpoint = tmp_path / "source-wide-readout.ckpt"
    torch.save(
        {
            "state_dict": {
                "model." + name: value.detach().clone()
                for name, value in source.state_dict().items()
            },
            "hyper_parameters": {"dataset_cids": {"session_a": [7, 8, 9]}},
        },
        checkpoint,
    )

    target = _bare_adapt_lightning_model(readout_channels=4)
    modulator_before = deepcopy(target.model.modulator.state_dict())
    target._load_pretrained_components(
        str(checkpoint),
        pretrained_scope="complete",
        exclude_prefixes="modulator",
        shape_adaptation="trim_readout_features",
    )

    assert torch.equal(
        target.model.readouts[0].features.weight,
        source.readouts[0].features.weight[:, :4],
    )
    assert all(
        torch.equal(value, modulator_before[name])
        for name, value in target.model.modulator.state_dict().items()
    )
    assert not any(
        name.startswith("modulator")
        for name in target._pretrained_parameter_names
    )

    unadapted = _bare_adapt_lightning_model(readout_channels=4)
    with pytest.raises(RuntimeError, match="shape mismatches"):
        unadapted._load_pretrained_components(
            str(checkpoint), pretrained_scope="complete",
            exclude_prefixes="modulator",
        )


def test_surround_warm_start_preserves_gaussian_physical_coordinates(tmp_path):
    source = _TinyAdaptWarmStartModel(readout_channels=4, scaffold_size=9)
    with torch.no_grad():
        source.readouts[0].mean.copy_(
            torch.tensor([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.0]])
        )
        source.readouts[0].std.copy_(
            torch.tensor([[0.5, 1.0], [1.5, 2.0], [0.75, 0.25]])
        )
    checkpoint = tmp_path / "source-scaffold9.ckpt"
    torch.save(
        {
            "state_dict": {
                "model." + name: value.detach().clone()
                for name, value in source.state_dict().items()
            },
            "hyper_parameters": {
                "dataset_cids": {"session_a": [7, 8, 9]},
                "model_config_dict": source.model_config,
            },
        },
        checkpoint,
    )

    target = _bare_adapt_lightning_model(
        readout_channels=4, scaffold_size=13
    )
    target._load_pretrained_components(
        str(checkpoint),
        pretrained_scope="complete",
        shape_adaptation="scale_gaussian_readout",
    )

    assert torch.allclose(target.model.readouts[0].mean, source.readouts[0].mean * 1.5)
    assert torch.allclose(target.model.readouts[0].std, source.readouts[0].std * 1.5)
    assert torch.equal(target.model.readouts[0].theta, source.readouts[0].theta)


def test_selective_unfreeze_matches_only_explicit_parameter_families():
    target = _bare_lightning_model()
    names = target._set_trainable_parameter_patterns(
        "readouts,output_modulator"
    )

    assert names
    assert all(
        parameter.requires_grad
        == ("readouts" in name or "output_modulator" in name)
        for name, parameter in target.model.named_parameters()
    )


def test_output_modulator_uses_head_lr_not_reduced_core_lr():
    parameters = [
        ("model.convnet.weight", nn.Parameter(torch.ones(2, 2))),
        ("model.output_modulator.encoder.weight", nn.Parameter(torch.ones(2, 2))),
    ]
    groups = _adamw_param_groups_named(
        parameters,
        wd=1e-5,
        excluded_names=set(),
        core_lr=1e-4,
        head_lr=1e-3,
    )
    lr_by_parameter = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }

    assert lr_by_parameter[id(parameters[0][1])] == pytest.approx(1e-4)
    assert lr_by_parameter[id(parameters[1][1])] == pytest.approx(1e-3)
