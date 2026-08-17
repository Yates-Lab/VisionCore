import pytest
import torch

from paper.model_selection.compose_residual_readout_checkpoint import (
    compose_readout_states,
)


def _states(shared_value=1.0):
    recipient = {
        "model.convnet.weight": torch.tensor([shared_value]),
        "model.residual_readouts.0.features.weight": torch.tensor([0.2]),
        "model.auxiliary_residual_readouts.0.features.weight": torch.tensor([0.3]),
        "model.residual_convnet.weight": torch.zeros(2),
        "model.residual_visual_readouts.0.features.weight": torch.zeros(2),
    }
    donor = {
        "model.convnet.weight": torch.tensor([shared_value]),
        "model.residual_readouts.0.features.weight": torch.tensor([2.0]),
        "model.auxiliary_residual_readouts.0.features.weight": torch.tensor([3.0]),
        # A different residual-visual width is valid and intentionally ignored.
        "model.residual_convnet.weight": torch.ones(3),
        "model.residual_visual_readouts.0.features.weight": torch.ones(3),
    }
    return recipient, donor


def test_compose_readout_states_copies_only_dual_source_readouts():
    recipient, donor = _states()
    composed, report = compose_readout_states(recipient, donor)
    assert torch.equal(
        composed["model.residual_readouts.0.features.weight"], torch.tensor([2.0])
    )
    assert torch.equal(
        composed["model.auxiliary_residual_readouts.0.features.weight"],
        torch.tensor([3.0]),
    )
    assert torch.equal(composed["model.residual_convnet.weight"], torch.zeros(2))
    assert report["shared_parent_tensors_exact"] == 1
    assert report["donor_readout_tensors_copied"] == 2


def test_compose_readout_states_rejects_shared_parent_drift():
    recipient, donor = _states()
    donor["model.convnet.weight"] = torch.tensor([1.1])
    with pytest.raises(RuntimeError, match="different shared parents"):
        compose_readout_states(recipient, donor)


def test_compose_readout_states_rejects_missing_recipient_readout():
    recipient, donor = _states()
    recipient.pop("model.residual_readouts.0.features.weight")
    with pytest.raises(RuntimeError, match="lacks donor readout tensors"):
        compose_readout_states(recipient, donor)
