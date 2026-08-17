import pytest
import torch

from paper.model_selection.average_checkpoints import (
    average_state_dicts,
    build_soup,
)


def _checkpoint(epoch, head_value, core_value=2.0):
    return {
        "epoch": epoch,
        "global_step": 10 * epoch,
        "state_dict": {
            "model.convnet.weight": torch.tensor([core_value]),
            "model.readouts.0.weight": torch.tensor(
                [head_value], dtype=torch.bfloat16
            ),
            "integer_buffer": torch.tensor([3], dtype=torch.int64),
        },
        "hyper_parameters": {
            "model_config_dict": {"model_type": "tiny"},
            "dataset_cids": {"session": [1]},
            "cfg_dir": "config.yaml",
            "max_ds": 1,
        },
        "optimizer_states": [{"state": "discard"}],
        "loops": {"discard": True},
    }


def test_average_state_dicts_preserves_frozen_core_and_dtype():
    left = _checkpoint(1, 1.0)["state_dict"]
    right = _checkpoint(2, 3.0)["state_dict"]
    averaged = average_state_dicts(
        [left, right], equal_prefixes=("model.convnet",)
    )

    assert torch.equal(averaged["model.convnet.weight"], torch.tensor([2.0]))
    assert averaged["model.readouts.0.weight"].dtype == torch.bfloat16
    assert float(averaged["model.readouts.0.weight"].item()) == 2.0
    assert torch.equal(averaged["integer_buffer"], torch.tensor([3]))


def test_average_state_dicts_rejects_changed_required_equal_prefix():
    left = _checkpoint(1, 1.0)["state_dict"]
    right = _checkpoint(2, 3.0, core_value=2.1)["state_dict"]
    with pytest.raises(ValueError, match="Required-equal state changed"):
        average_state_dicts(
            [left, right], equal_prefixes=("model.convnet",)
        )


def test_build_soup_records_inputs_and_removes_training_state(tmp_path):
    paths = [tmp_path / "a.ckpt", tmp_path / "b.ckpt"]
    torch.save(_checkpoint(1, 1.0), paths[0])
    torch.save(_checkpoint(2, 3.0), paths[1])

    soup = build_soup(paths, equal_prefixes=("model.convnet",))

    assert soup["epoch"] == -1
    assert soup["global_step"] == -1
    assert "optimizer_states" not in soup
    assert "loops" not in soup
    assert soup["checkpoint_soup"]["input_epochs"] == [1, 2]
    assert soup["checkpoint_soup"]["equal_prefixes"] == ["model.convnet"]
