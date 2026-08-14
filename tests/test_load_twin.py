"""Shared twin loading: size readouts from the checkpoint, fail loudly on drift.

TWIN_IMPROVEMENTS item 1. Readout sizes are not stored in older checkpoints;
`MultiDatasetModel.__init__` rebuilds each head to `len(dataset_config['cids'])`
read from mutable per-session YAMLs. Editing those YAMLs after training makes a
checkpoint stop loading with a wall of size-mismatch errors that names no
session. These tests pin the diagnosis path.
"""
import pytest
import torch

from eval.load_twin import (
    CidDriftError,
    readout_sizes_from_state_dict,
    resolve_checkpoint_cids,
    validate_readout_sizes,
)


def _state_dict(sizes, feature_dim=8):
    sd = {"model.convnet.block0.weight": torch.zeros(4, 4, 3, 3)}
    for i, n in enumerate(sizes):
        sd[f"model.readouts.{i}.bias"] = torch.zeros(n)
        sd[f"model.readouts.{i}.mean"] = torch.zeros(n, 2)
        sd[f"model.readouts.{i}.std"] = torch.zeros(n, 2)
        sd[f"model.readouts.{i}.features.weight"] = torch.zeros(n, feature_dim, 1, 1)
    return sd


# ---------------------------------------------------------------------------
# Sizing from the checkpoint itself
# ---------------------------------------------------------------------------
def test_readout_sizes_are_read_from_the_state_dict():
    assert readout_sizes_from_state_dict(_state_dict([116, 77, 149])) == [116, 77, 149]


def test_readout_sizes_are_ordered_numerically_not_lexically():
    """readouts.10 must not sort between readouts.1 and readouts.2."""
    sizes = list(range(20, 32))
    assert readout_sizes_from_state_dict(_state_dict(sizes)) == sizes


def test_readout_sizes_fall_back_to_bias_when_mean_is_absent():
    sd = _state_dict([5, 6])
    del sd["model.readouts.0.mean"]
    del sd["model.readouts.1.mean"]
    assert readout_sizes_from_state_dict(sd) == [5, 6]


def test_a_checkpoint_with_no_readouts_is_rejected():
    with pytest.raises(ValueError, match="no readout"):
        readout_sizes_from_state_dict({"model.convnet.w": torch.zeros(2)})


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def test_matching_sizes_validate_silently():
    validate_readout_sizes(
        {"Allen_A": list(range(116)), "Logan_B": list(range(77))},
        [116, 77], checkpoint_path="ckpt")


def test_drifting_session_is_named_with_both_counts():
    """The whole point: say which session drifted, not 200 lines of tensors."""
    with pytest.raises(CidDriftError) as exc:
        validate_readout_sizes(
            {"Allen_2022-02-16": list(range(116)), "Logan_B": list(range(77))},
            [120, 77], checkpoint_path="ckpt")

    msg = str(exc.value)
    assert "Allen_2022-02-16" in msg
    assert "116" in msg and "120" in msg
    assert "Logan_B" not in msg


def test_every_drifting_session_is_reported_not_just_the_first():
    with pytest.raises(CidDriftError) as exc:
        validate_readout_sizes(
            {"A": list(range(10)), "B": list(range(20)), "C": list(range(30))},
            [10, 21, 31], checkpoint_path="ckpt")

    msg = str(exc.value)
    assert "B" in msg and "C" in msg


def test_session_count_mismatch_is_its_own_error():
    with pytest.raises(CidDriftError, match="30 readouts.*2 sessions|2 sessions.*30 readouts"):
        validate_readout_sizes(
            {"A": list(range(10)), "B": list(range(20))},
            [10] * 30, checkpoint_path="ckpt")


# ---------------------------------------------------------------------------
# cid resolution: checkpoint snapshot preferred over mutable YAML
# ---------------------------------------------------------------------------
def test_snapshotted_cids_are_preferred_over_the_yaml():
    ckpt = {"hyper_parameters": {
        "cfg_dir": "/nonexistent/parent.yaml",
        "dataset_cids": {"Allen_A": [1, 2, 3]},
    }}
    cids, source = resolve_checkpoint_cids(ckpt)

    assert source == "checkpoint"
    assert cids == {"Allen_A": [1, 2, 3]}


def test_missing_snapshot_falls_back_to_the_yaml_path(tmp_path, monkeypatch):
    import eval.load_twin as lt

    monkeypatch.setattr(lt, "_load_cids_from_config",
                        lambda p: {"Allen_A": [1, 2], "Logan_B": [3]})
    cids, source = resolve_checkpoint_cids(
        {"hyper_parameters": {"cfg_dir": str(tmp_path / "parent.yaml")}})

    assert source == "yaml"
    assert set(cids) == {"Allen_A", "Logan_B"}


def test_a_checkpoint_without_any_config_reference_fails_clearly():
    with pytest.raises(ValueError, match="cfg_dir"):
        resolve_checkpoint_cids({"hyper_parameters": {}})
