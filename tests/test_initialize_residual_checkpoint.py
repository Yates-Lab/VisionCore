import pytest
import torch

from paper.model_selection.initialize_residual_readout_checkpoint import (
    missing_residual_components,
    validate_upgrade_state,
)


def test_validate_upgrade_state_requires_exact_parent_and_zero_control():
    parent = {"model.readouts.0.bias": torch.tensor([1.0, 2.0])}
    child = {
        **parent,
        "model.residual_readouts.0.features.weight": torch.zeros(2, 3, 1, 1),
        "model.residual_readouts.0.mean_delta": torch.zeros(2, 2),
    }
    report = validate_upgrade_state(parent, child)
    assert report["inherited_tensors_exact"] == 1
    assert report["new_residual_tensors"] == 2
    assert report["zero_identity_controls"] == 1
    assert report["residual_datasets"] == 1


def test_validate_upgrade_state_rejects_parent_drift_and_nonzero_residual():
    parent = {"model.readouts.0.bias": torch.tensor([1.0])}
    drifted = {
        "model.readouts.0.bias": torch.tensor([2.0]),
        "model.residual_readouts.0.base_scale": torch.zeros(1),
    }
    with pytest.raises(RuntimeError, match="changed inherited"):
        validate_upgrade_state(parent, drifted)

    nonzero = {
        **parent,
        "model.residual_readouts.0.base_scale": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="not initialized to zero"):
        validate_upgrade_state(parent, nonzero)


def test_validate_upgrade_state_accepts_auxiliary_residual_component():
    parent = {"model.readouts.0.bias": torch.tensor([1.0, 2.0])}
    child = {
        **parent,
        "model.auxiliary_residual_readouts.0.features.weight": torch.zeros(
            2, 3, 1, 1
        ),
        "model.auxiliary_residual_readouts.0.mean_delta": torch.zeros(2, 2),
    }
    report = validate_upgrade_state(parent, child)
    assert report["new_residual_tensors"] == 2
    assert report["zero_identity_controls"] == 1


def test_validate_upgrade_state_accepts_identity_gated_residual_core():
    parent = {"model.readouts.0.bias": torch.tensor([1.0, 2.0])}
    child = {
        **parent,
        "model.residual_convnet.stage.weight": torch.randn(3, 3),
        "model.residual_visual_readouts.0.features.weight": torch.zeros(
            2, 3, 1, 1
        ),
        "model.residual_visual_readouts.0.mean_delta": torch.zeros(2, 2),
    }
    report = validate_upgrade_state(parent, child)
    assert report["new_residual_tensors"] == 3
    assert report["zero_identity_controls"] == 1


def test_chained_upgrade_excludes_only_residual_components_missing_from_parent():
    parent = {
        "model.convnet.stage.weight": torch.ones(1),
        "model.residual_readouts.0.features.weight": torch.zeros(2, 3, 1, 1),
        "model.auxiliary_residual_readouts.0.features.weight": torch.zeros(
            2, 3, 1, 1
        ),
    }
    assert missing_residual_components(parent) == (
        "residual_convnet",
        "residual_visual_readouts",
    )
