from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.readout import SparseGaussianLowRankReadout, SparseGaussianReadout
from paper.fig4.upstream.real_trace_matrix.model import load_spatial_readout


class _Core:
    @staticmethod
    def get_phase_spatial_stride() -> int:
        return 2


def _fake_model():
    torch.manual_seed(12)
    deep = nn.ModuleList(
        [
            SparseGaussianReadout(4, 2, spatial_shape=(3, 3)),
            SparseGaussianReadout(4, 1, spatial_shape=(3, 3)),
        ]
    )
    phase = nn.ModuleList(
        [
            SparseGaussianLowRankReadout(
                3, 2, rank=2, bias=False, spatial_shape=(7, 7), output_scale=7.0
            ),
            SparseGaussianLowRankReadout(
                3, 1, rank=2, bias=False, spatial_shape=(7, 7), output_scale=7.0
            ),
        ]
    )
    inner = SimpleNamespace(
        dataset_configs=[{"cids": [10, 43]}, {"cids": [5]}],
        readouts=deep,
        phase_readouts=phase,
        convnet=_Core(),
        baseline_enabled=False,
    )
    return SimpleNamespace(names=["session-a", "session-b"], model=inner)


def _outputs():
    return [
        {
            "sess": "session-a",
            "cids_used": np.asarray([43, 77, 10]),
            "ccnorm": {"ccnorm": np.asarray([0.8, 0.9, 0.7])},
        },
        {
            "sess": "session-b",
            "cids_used": np.asarray([5]),
            "ccnorm": {"ccnorm": np.asarray([0.75])},
        },
    ]


def _translated_native(native, feature: torch.Tensor, row: int, *, stride: int) -> torch.Tensor:
    rank = int(getattr(native, "rank", 1))
    weight = native.features.weight.reshape(native.n_units, rank, native.in_channels, 1, 1)[row]
    projected = F.conv2d(feature, weight.reshape(rank, native.in_channels, 1, 1))
    height, width = native.spatial_shape
    space = native.effective_spatial_weights(height, width, feature.device)[row]
    if rank == 1 and space.ndim == 2:
        space = space[None]
    value = F.conv2d(projected, space[:, None], stride=stride, groups=rank).sum(dim=1)
    return value * float(getattr(native, "output_scale", 1.0))


def test_exact_cid_spatial_readout_uses_cids_not_historical_positions_and_keeps_phase():
    model = _fake_model()
    readout, rows = load_spatial_readout(model, _outputs(), device="cpu")

    assert [(row["source_cid"], row["model_readout_row"], row["available"]) for row in rows] == [
        (43, 1, True),
        (77, None, False),
        (10, 0, True),
        (5, 0, True),
    ]
    assert readout.has_phase_branch
    assert readout.phase_rank == 2
    assert readout.scalar_equivalence_audit["passed"]
    assert readout.scalar_equivalence_audit["n_available_logits_checked"] == 3
    assert readout.scalar_equivalence_audit["n_unavailable_inert_placeholders"] == 1

    deep = torch.randn(2, 4, 5, 5)
    phase = torch.randn(2, 3, 11, 11)
    actual = readout(deep, phase)
    expected_43 = (
        _translated_native(model.model.readouts[0], deep, 1, stride=1)
        + _translated_native(model.model.phase_readouts[0], phase, 1, stride=2)
        + model.model.readouts[0].bias[1]
    )
    torch.testing.assert_close(actual[:, 0], expected_43, atol=3e-5, rtol=3e-5)
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))


def test_exact_cid_spatial_readout_rejects_duplicate_configured_cids():
    model = _fake_model()
    model.model.dataset_configs[0]["cids"] = [43, 43]
    try:
        load_spatial_readout(model, _outputs(), device="cpu")
    except ValueError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("duplicate checkpoint CIDs were accepted")
