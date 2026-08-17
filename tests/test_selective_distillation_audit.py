from pathlib import Path

import pytest
import torch

from paper.model_selection.audit_selective_distillation_checkpoint import (
    audit_checkpoint,
)


def _checkpoint(path: Path, state, metadata=None):
    artifact = {"state_dict": state}
    if metadata is not None:
        artifact["selective_ryan_distillation"] = metadata
    torch.save(artifact, path)


def test_audit_rejects_changes_outside_recorded_trainable_tensors(tmp_path):
    parent = tmp_path / "parent.ckpt"
    child = tmp_path / "child.ckpt"
    _checkpoint(
        parent,
        {
            "model.readouts.0.bias": torch.zeros(2),
            "model.convnet.weight": torch.zeros(2),
        },
    )
    _checkpoint(
        child,
        {
            "model.readouts.0.bias": torch.ones(2),
            "model.convnet.weight": torch.ones(2),
        },
        {
            "student_checkpoint": str(parent),
            "trainable_names": ["model.readouts.0.bias"],
            "unit_isolation_report": {},
        },
    )

    with pytest.raises(RuntimeError, match="Frozen tensors changed"):
        audit_checkpoint(child)


def test_audit_reports_exactly_preserved_nontrainable_tensors(tmp_path):
    parent = tmp_path / "parent.ckpt"
    child = tmp_path / "child.ckpt"
    _checkpoint(
        parent,
        {
            "model.readouts.0.bias": torch.zeros(2),
            "model.convnet.weight": torch.zeros(2),
        },
    )
    _checkpoint(
        child,
        {
            "model.readouts.0.bias": torch.ones(2),
            "model.convnet.weight": torch.zeros(2),
        },
        {
            "student_checkpoint": str(parent),
            "trainable_names": ["model.readouts.0.bias"],
            "unit_isolation_report": {},
        },
    )

    report = audit_checkpoint(child)

    assert report["changed_tensors"] == 1
    assert report["exactly_preserved_tensors"] == 1
    assert report["unexpected_changed_tensors"] == 0
