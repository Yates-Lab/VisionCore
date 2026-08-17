"""Identity/alignment regressions for the production Figure 4 twin adapter."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from paper.fig4.upstream.real_trace_matrix.model import adapt_population_view_to_available
from ryan.population_information.rr_population.redundancy_resolved_v1_population import (
    PopulationView,
)
from scripts.spatial_info import PopulationReadout, get_spatial_readout


class _DummyDatasetReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight[0].fill_(20.0)
            self.features.weight[1].fill_(10.0)
        self.bias = nn.Parameter(torch.tensor([2.0, 1.0]))

    def compute_gaussian_mask(self, height: int, width: int, device: str) -> torch.Tensor:
        assert (height, width) == (3, 3)
        return torch.stack(
            (
                torch.full((height, width), 2.0),
                torch.full((height, width), 1.0),
            )
        ).to(device)


class _DummyAuxiliaryReadout(nn.Module):
    def __init__(self, scaffold_size: int = 3) -> None:
        super().__init__()
        self.scaffold_size = scaffold_size
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight[0].fill_(200.0)
            self.features.weight[1].fill_(100.0)

    def compute_gaussian_mask(
        self,
        height: int,
        width: int,
        device: str,
        dtype: torch.dtype,
        base_readout: nn.Module,
    ) -> torch.Tensor:
        assert isinstance(base_readout, _DummyDatasetReadout)
        assert (height, width) == (self.scaffold_size, self.scaffold_size)
        return torch.stack(
            (
                torch.full((height, width), 2.0),
                torch.full((height, width), 1.0),
            )
        ).to(device=device, dtype=dtype)


def test_spatial_readout_maps_mcfarland_cids_to_checkpoint_rows() -> None:
    dataset_readout = _DummyDatasetReadout()
    wrapper = SimpleNamespace(
        names=["session_a"],
        cfgs=[{"cids": [20, 10]}],
        device="cpu",
        model=SimpleNamespace(
            convnet=SimpleNamespace(scaffold_size=3),
            readouts=nn.ModuleList([dataset_readout]),
            output_modulator=None,
        ),
    )
    outputs = [
        {
            "sess": "session_a",
            "cids_used": np.asarray([10, 30, 20]),
            "ccnorm": {"ccnorm": np.asarray([0.7, 0.8, 0.9])},
        }
    ]

    readout, rows = get_spatial_readout(wrapper, outputs, return_unit_rows=True)

    assert [row["source_cid"] for row in rows] == [10, 30, 20]
    assert [row["model_readout_row"] for row in rows] == [1, None, 0]
    assert [row["available"] for row in rows] == [True, False, True]
    assert torch.all(readout.features.weight[0] == 10.0)
    assert torch.all(readout.features.weight[1] == 0.0)
    assert torch.all(readout.features.weight[2] == 20.0)
    assert readout.bias.tolist() == [1.0, -50.0, 2.0]
    assert torch.all(readout.space_weights[1] == 0.0)


def test_spatial_readout_replays_auxiliary_visual_branch_in_cid_order() -> None:
    dataset_readout = _DummyDatasetReadout()
    auxiliary_readout = _DummyAuxiliaryReadout()
    wrapper = SimpleNamespace(
        names=["session_a"],
        cfgs=[{"cids": [20, 10]}],
        device="cpu",
        model=SimpleNamespace(
            convnet=SimpleNamespace(scaffold_size=3),
            readouts=nn.ModuleList([dataset_readout]),
            output_modulator=None,
            auxiliary_convnet=SimpleNamespace(scaffold_size=3),
            auxiliary_readouts=nn.ModuleList([auxiliary_readout]),
        ),
    )
    outputs = [
        {
            "sess": "session_a",
            "cids_used": np.asarray([10, 30, 20]),
            "ccnorm": {"ccnorm": np.asarray([0.7, 0.8, 0.9])},
        }
    ]

    readout, rows = get_spatial_readout(
        wrapper,
        outputs,
        return_unit_rows=True,
    )

    assert [row["source_cid"] for row in rows] == [10, 30, 20]
    assert torch.all(readout.auxiliary_features.weight[0] == 100.0)
    assert torch.all(readout.auxiliary_features.weight[1] == 0.0)
    assert torch.all(readout.auxiliary_features.weight[2] == 200.0)
    assert torch.all(readout.auxiliary_space_weights[1] == 0.0)

    base_features = torch.ones(1, 2, 3, 3)
    auxiliary_features = torch.ones(1, 3, 3, 3)
    output = readout(base_features, auxiliary_features).flatten()
    assert torch.equal(output, torch.tensor([2881.0, -50.0, 11522.0]))


def test_spatial_readout_replays_both_new_residual_components() -> None:
    dataset_readout = _DummyDatasetReadout()
    main_residual = _DummyAuxiliaryReadout()
    main_residual.features = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    auxiliary_readout = _DummyAuxiliaryReadout()
    auxiliary_residual = _DummyAuxiliaryReadout()
    with torch.no_grad():
        main_residual.features.weight[0].fill_(4.0)
        main_residual.features.weight[1].fill_(3.0)
        auxiliary_residual.features.weight[0].fill_(40.0)
        auxiliary_residual.features.weight[1].fill_(30.0)
    wrapper = SimpleNamespace(
        names=["session_a"],
        cfgs=[{"cids": [20, 10]}],
        device="cpu",
        model=SimpleNamespace(
            convnet=SimpleNamespace(scaffold_size=3),
            readouts=nn.ModuleList([dataset_readout]),
            output_modulator=None,
            auxiliary_convnet=SimpleNamespace(scaffold_size=3),
            auxiliary_readouts=nn.ModuleList([auxiliary_readout]),
            residual_readouts=nn.ModuleList([main_residual]),
            auxiliary_residual_readouts=nn.ModuleList([
                auxiliary_residual
            ]),
        ),
    )
    outputs = [{
        "sess": "session_a",
        "cids_used": np.asarray([10, 30, 20]),
        "ccnorm": {"ccnorm": np.asarray([0.7, 0.8, 0.9])},
    }]

    readout = get_spatial_readout(wrapper, outputs)
    assert torch.all(readout.residual_features.weight[0] == 3.0)
    assert torch.all(readout.residual_features.weight[1] == 0.0)
    assert torch.all(readout.auxiliary_residual_features.weight[0] == 30.0)
    assert torch.all(readout.auxiliary_residual_features.weight[1] == 0.0)

    base_features = torch.ones(1, 2, 3, 3)
    auxiliary_features = torch.ones(1, 3, 3, 3)
    output = readout(base_features, auxiliary_features).flatten()
    assert torch.equal(output, torch.tensor([3745.0, -50.0, 13826.0]))


def test_population_residual_consumes_only_leading_visual_channels() -> None:
    readout = PopulationReadout(
        feat_weights=torch.ones(1, 3, 1, 1),
        biases=torch.zeros(1),
        space_weights=torch.ones(1, 1, 1),
        residual_feat_weights=torch.full((1, 2, 1, 1), 2.0),
        residual_space_weights=torch.ones(1, 1, 1),
    )
    features = torch.tensor([[[[1.0]], [[2.0]], [[100.0]]]])
    # Mature term: 1 + 2 + 100. Residual term: 2 * (1 + 2).
    assert torch.equal(readout(features).flatten(), torch.tensor([109.0]))


def test_spatial_readout_replays_identity_gated_residual_core() -> None:
    dataset_readout = _DummyDatasetReadout()
    residual_visual_readout = _DummyAuxiliaryReadout()
    wrapper = SimpleNamespace(
        names=["session_a"],
        cfgs=[{"cids": [20, 10]}],
        device="cpu",
        model=SimpleNamespace(
            convnet=SimpleNamespace(scaffold_size=3),
            readouts=nn.ModuleList([dataset_readout]),
            output_modulator=None,
            residual_convnet=SimpleNamespace(scaffold_size=3),
            residual_visual_readouts=nn.ModuleList([
                residual_visual_readout
            ]),
        ),
    )
    outputs = [{
        "sess": "session_a",
        "cids_used": np.asarray([10, 30, 20]),
        "ccnorm": {"ccnorm": np.asarray([0.7, 0.8, 0.9])},
    }]

    readout = get_spatial_readout(wrapper, outputs)
    assert torch.all(readout.residual_visual_features.weight[0] == 100.0)
    assert torch.all(readout.residual_visual_features.weight[1] == 0.0)
    assert torch.all(readout.residual_visual_features.weight[2] == 200.0)

    base_features = torch.ones(1, 2, 3, 3)
    residual_features = torch.ones(1, 3, 3, 3)
    output = readout(
        base_features,
        residual_visual_x=residual_features,
    ).flatten()
    assert torch.equal(output, torch.tensor([2881.0, -50.0, 11522.0]))


def test_spatial_readout_aligns_larger_auxiliary_scaffold_by_center() -> None:
    dataset_readout = _DummyDatasetReadout()
    auxiliary_readout = _DummyAuxiliaryReadout(scaffold_size=5)
    wrapper = SimpleNamespace(
        names=["session_a"],
        cfgs=[{"cids": [20, 10]}],
        device="cpu",
        model=SimpleNamespace(
            convnet=SimpleNamespace(scaffold_size=3),
            readouts=nn.ModuleList([dataset_readout]),
            output_modulator=None,
            auxiliary_convnet=SimpleNamespace(scaffold_size=5),
            auxiliary_readouts=nn.ModuleList([auxiliary_readout]),
        ),
    )
    outputs = [
        {
            "sess": "session_a",
            "cids_used": np.asarray([10, 30, 20]),
            "ccnorm": {"ccnorm": np.asarray([0.7, 0.8, 0.9])},
        }
    ]

    readout = get_spatial_readout(wrapper, outputs)
    assert readout.space_weights.shape[-2:] == (3, 3)
    assert readout.auxiliary_space_weights.shape[-2:] == (5, 5)

    base_features = torch.ones(1, 2, 7, 7)
    auxiliary_features = torch.ones(1, 3, 7, 7)
    output = readout(base_features, auxiliary_features)

    assert output.shape == (1, 3, 3, 3)
    expected = torch.tensor([7681.0, -50.0, 30722.0])
    assert torch.equal(output[0, :, 0, 0], expected)
    assert torch.all(output == expected[None, :, None, None])


def test_rr_medoid_substitution_stays_within_saved_cluster() -> None:
    membership = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cluster_membership = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    view = PopulationView(
        name="test_rr",
        membership=membership,
        input_channels=4,
        n_units=3,
        meta={},
        labels=np.asarray([0, 0, -1, -1]),
        cluster_membership=cluster_membership,
    )
    rows = [
        {"available": False, "ccnorm": 0.9},
        {"available": True, "ccnorm": 0.7},
        {"available": False, "ccnorm": 0.8},
        {"available": True, "ccnorm": 0.6},
    ]

    adapted, report = adapt_population_view_to_available(view, rows)

    assert np.array_equal(adapted.membership[0], np.asarray([0.0, 1.0, 0.0, 0.0]))
    assert np.array_equal(adapted.membership[1], np.zeros(4, dtype=np.float32))
    assert np.array_equal(adapted.membership[2], membership[2])
    assert report["available_channels"] == 2
    assert report["missing_channels"] == 2
    assert report["inactive_units"] == [1]
    assert report["active_units"] == 2
    assert report["substitutions"][0]["selected_channels"] == [1]
