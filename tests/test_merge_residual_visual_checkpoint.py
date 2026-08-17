import pytest
import torch

from paper.model_selection.merge_residual_visual_checkpoint import compose_states


def _states(shared_value=1.0):
    recipient = {
        "model.convnet.weight": torch.tensor([shared_value]),
        "model.residual_readouts.0.features.weight": torch.tensor([0.2]),
        "model.residual_convnet.weight": torch.tensor([0.0]),
        "model.residual_visual_readouts.0.features.weight": torch.tensor([0.0]),
    }
    donor = {
        "model.convnet.weight": torch.tensor([shared_value]),
        "model.residual_convnet.weight": torch.tensor([2.0]),
        "model.residual_visual_readouts.0.features.weight": torch.tensor([3.0]),
    }
    return recipient, donor


def test_compose_states_copies_only_residual_visual_branch():
    recipient, donor = _states()
    composed, report = compose_states(recipient, donor)
    assert torch.equal(
        composed["model.residual_readouts.0.features.weight"],
        recipient["model.residual_readouts.0.features.weight"],
    )
    assert torch.equal(composed["model.residual_convnet.weight"], torch.tensor([2.0]))
    assert torch.equal(
        composed["model.residual_visual_readouts.0.features.weight"],
        torch.tensor([3.0]),
    )
    assert report["shared_parent_tensors_exact"] == 1
    assert report["donor_branch_tensors_copied"] == 2


def test_compose_states_rejects_shared_parent_drift():
    recipient, donor = _states()
    donor["model.convnet.weight"] = torch.tensor([1.1])
    with pytest.raises(RuntimeError, match="different shared parents"):
        compose_states(recipient, donor)


def test_compose_states_rejects_incompatible_recipient():
    recipient, donor = _states()
    recipient.pop("model.residual_convnet.weight")
    with pytest.raises(RuntimeError, match="lacks donor tensors"):
        compose_states(recipient, donor)
