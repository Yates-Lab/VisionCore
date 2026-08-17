import numpy as np
import pytest
import torch

from paper.model_selection.cache_ryan_teacher_rates import (
    teacher_to_student_positions,
)
from paper.model_selection.train_selective_ryan_distillation import (
    TeacherRateView,
    install_unit_isolation,
    relative_parameter_anchor_loss,
    spatial_laplacian_loss,
    unit_teacher_weights,
    weighted_teacher_loss,
    weighted_teacher_shape_loss,
)


class _Raw:
    def __init__(self, name):
        self.metadata = {"name": name}


class _Embedded:
    def __init__(self, names, inds):
        self.dsets = [_Raw(name) for name in names]
        self.n_dsets = len(self.dsets)
        self.inds = torch.as_tensor(inds, dtype=torch.long)

    def __len__(self):
        return len(self.inds)


def test_teacher_positions_map_to_causal_odd_student_endpoints():
    teacher = _Embedded(
        ["a", "b"],
        [[0, 2], [0, 3], [1, 4], [1, 5]],
    )
    # Deliberately permute split order and omit teacher endpoint 2*5+1.
    student = _Embedded(
        ["a", "b"],
        [[1, 9], [0, 7], [0, 5], [1, 3]],
    )
    mapping = teacher_to_student_positions(
        teacher, student, factor=2, phase=1
    )
    assert mapping.tolist() == [2, 1, 0, -1]


def test_same_grid_parent_positions_map_identically():
    parent = _Embedded(["a"], [[0, 7], [0, 5], [0, 9]])
    student = _Embedded(["a"], [[0, 9], [0, 7], [0, 5]])
    mapping = teacher_to_student_positions(
        parent, student, factor=1, phase=0
    )
    assert mapping.tolist() == [1, 2, 0]


def test_cached_rate_views_can_attach_teacher_and_parent_targets():
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"index": torch.tensor(index)}

    teacher = torch.tensor([[1.0], [2.0]])
    parent = torch.tensor([[3.0], [4.0]])
    dataset = TeacherRateView(Dataset(), teacher, "teacher_rate")
    dataset = TeacherRateView(dataset, parent, "parent_rate")
    item = dataset[1]
    torch.testing.assert_close(item["teacher_rate"], torch.tensor([2.0]))
    torch.testing.assert_close(item["parent_rate"], torch.tensor([4.0]))


def test_unit_isolation_masks_gradients_and_restores_unselected_rows():
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    original = parameter.detach().clone()
    restore, report = install_unit_isolation(
        [("model.readouts.0.features.weight", parameter)],
        {0: torch.tensor([0.0, 0.5, 0.0])},
    )
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
    parameter.sum().backward()
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
    )
    optimizer.step()
    restore()
    torch.testing.assert_close(parameter[[0, 2]], original[[0, 2]])
    assert not torch.equal(parameter[1], original[1])
    assert report["model.readouts.0.features.weight"]["n_trainable_rows"] == 1


def test_unit_isolation_accepts_residual_readout_rows():
    parameter = torch.nn.Parameter(torch.ones(3, 4, 1, 1))
    original = parameter.detach().clone()
    restore, report = install_unit_isolation(
        [("model.residual_readouts.0.features.weight", parameter)],
        {0: torch.tensor([1.0, 0.0, 1.0])},
    )
    parameter.sum().backward()
    assert parameter.grad[1].count_nonzero() == 0
    with torch.no_grad():
        parameter.add_(1.0)
    restore()
    torch.testing.assert_close(parameter[1], original[1])
    assert report[
        "model.residual_readouts.0.features.weight"
    ]["n_trainable_rows"] == 2


def test_unit_isolation_rejects_shared_trainable_parameters():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    try:
        install_unit_isolation(
            [("model.output_modulator.encoder.weight", parameter)],
            {0: torch.ones(2)},
        )
    except RuntimeError as error:
        assert "unexpected tensors" in str(error)
    else:
        raise AssertionError("Shared parameter should have been rejected")


def test_unit_isolation_allows_explicit_new_shared_residual_core():
    shared = torch.nn.Parameter(torch.ones(2, 2))
    rows = torch.nn.Parameter(torch.ones(3, 2))
    restore, report = install_unit_isolation(
        [
            ("model.residual_convnet.stage.weight", shared),
            ("model.residual_visual_readouts.0.features.weight", rows),
        ],
        {0: torch.tensor([1.0, 0.0, 1.0])},
        allowed_shared_patterns=("model.residual_convnet",),
    )
    (shared.sum() + rows.sum()).backward()
    torch.testing.assert_close(shared.grad, torch.ones_like(shared))
    torch.testing.assert_close(
        rows.grad,
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
    )
    with torch.no_grad():
        shared.add_(2.0)
        rows.add_(2.0)
    restore()
    torch.testing.assert_close(shared, torch.full_like(shared, 3.0))
    torch.testing.assert_close(rows[1], torch.ones_like(rows[1]))
    assert list(report) == [
        "model.residual_visual_readouts.0.features.weight"
    ]


def test_unit_isolation_rejects_unmatched_shared_allowlist():
    rows = torch.nn.Parameter(torch.ones(3, 2))
    with pytest.raises(RuntimeError, match="matched no trainable"):
        install_unit_isolation(
            [("model.residual_visual_readouts.0.features.weight", rows)],
            {0: torch.ones(3)},
            allowed_shared_patterns=("model.residual_convnet",),
        )


def test_weighted_teacher_loss_ignores_unmatched_and_unselected_values():
    prediction = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
    target = torch.tensor([[1.5, float("nan")], [1.0, 4.0]])
    dfs = torch.ones_like(prediction)
    loss = weighted_teacher_loss(
        prediction, target, dfs, torch.tensor([1.0, 0.0])
    )
    expected = torch.nn.functional.poisson_nll_loss(
        prediction[:, 0], target[:, 0], log_input=False, reduction="mean"
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad[:, 1].abs().sum() == 0


def test_teacher_shape_loss_is_affine_invariant_and_masks_unselected_units():
    teacher = torch.tensor(
        [[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [8.0, 2.0]]
    )
    prediction = torch.stack(
        [3.0 * teacher[:, 0] + 7.0, teacher[:, 1].flip(0)], dim=1
    ).requires_grad_()
    loss = weighted_teacher_shape_loss(
        prediction,
        teacher,
        torch.ones_like(teacher),
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    loss.backward()
    assert prediction.grad[:, 1].abs().sum() == 0


def test_teacher_shape_loss_ignores_nan_and_zero_variance_units():
    prediction = torch.tensor(
        [[1.0, 2.0], [2.0, 2.0], [4.0, 2.0]], requires_grad=True
    )
    teacher = torch.tensor([[2.0, 1.0], [4.0, float("nan")], [8.0, 1.0]])
    loss = weighted_teacher_shape_loss(
        prediction, teacher, torch.ones_like(prediction), torch.ones(2)
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def _write_report(tmp_path, stem, values):
    archive = tmp_path / f"{stem}.npz"
    np.savez_compressed(
        archive,
        session_names=np.asarray(["session"]),
        cids_0=np.asarray([3, 7, 9]),
        bps_0=np.asarray(values, dtype=np.float32),
    )
    report = tmp_path / f"{stem}.json"
    report.write_text('{"per_unit_bps_npz": "' + str(archive) + '"}')
    return report


def test_unit_teacher_weights_select_only_advantages_above_margin(tmp_path):
    teacher = _write_report(tmp_path, "teacher", [0.2, 0.3, 0.5])
    student = _write_report(tmp_path, "student", [0.3, 0.28, 0.4])
    weights, report = unit_teacher_weights(
        teacher,
        student,
        ["session"],
        {"session": [3, 7, 9]},
        margin=0.01,
        scale=0.04,
    )
    torch.testing.assert_close(
        weights[0], torch.tensor([0.0, 0.25, 1.0])
    )
    assert report["session"]["n_teacher_weighted"] == 2


def test_relative_parameter_anchor_is_zero_at_parent_and_scale_normalized():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    reference = {"weight": parameter.detach().clone()}
    loss = relative_parameter_anchor_loss([("weight", parameter)], reference)
    assert loss == 0

    with torch.no_grad():
        parameter.mul_(2)
    loss = relative_parameter_anchor_loss([("weight", parameter)], reference)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert parameter.grad is not None


def test_spatial_laplacian_ignores_vectors_and_penalizes_kernel_curvature():
    smooth = torch.nn.Parameter(torch.ones(1, 1, 5, 5))
    vector = torch.nn.Parameter(torch.ones(5))
    loss = spatial_laplacian_loss([("kernel", smooth), ("norm", vector)])
    assert loss > 0  # constant padding retains the Dekel boundary penalty
    loss.backward()
    assert smooth.grad is not None
    assert vector.grad is None
