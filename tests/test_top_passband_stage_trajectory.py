import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from models import build_model
from models.config_loader import load_config
from paper.fig4.upstream.real_trace_matrix.model import ExactCIDSpatialReadout

from paper.fig4.spatiotemporal_tuning.analyze_top_passband_stage_trajectory import (
    select_top_traces,
    unit_top_bin_effects,
    unit_top_bin_normalized_effects,
    cumulative_rate_maps,
    stage_names_for_readout,
)


@pytest.mark.parametrize("rank", [1, 2])
def test_no_phase_cumulative_output_equals_full_model(rank):
    torch.manual_seed(8)
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / f"experiments/model_configs/dekel_native240_no_phase_rank{rank}_stage3.yaml")
    model = build_model(config, [{"session": "test", "cids": [0, 1, 2]}]).eval()
    head = model.readouts[0]
    spatial = head.get_spatial_weights()
    if spatial.ndim == 3:
        spatial = spatial[:, None]
    readout = ExactCIDSpatialReadout(
        deep_features=head.features.weight.reshape(3, rank, -1, 1, 1),
        deep_space=spatial, bias=head.bias,
        available=torch.ones(3, dtype=torch.bool), deep_output_scale=head.output_scale,
    )
    zeros = lambda n, dtype: torch.zeros(n, 42, dtype=dtype)
    scorer = SimpleNamespace(
        model=SimpleNamespace(model=model), readout=readout,
        _zero_behavior=zeros, population_view=None,
        apply_population_view=lambda value, view: value,
        _compute_rate_map=lambda stimulus: model.activation(readout(
            model.core_forward_spatial_map(stimulus, zeros(len(stimulus), stimulus.dtype))
        )),
    )
    with torch.inference_mode():
        rates, checks = cumulative_rate_maps(scorer, torch.randn(2, 1, 60, 39, 39), check_identity=True)
    assert rates.shape[:3] == (3, 2, 3)
    assert checks["ordinary_output_max_abs"] < 2e-5
    assert stage_names_for_readout(readout) == ("S1", "+ S2", "+ S3 / output")


def test_select_top_traces_stratifies_full_top_bin_and_covers_units():
    percentile = np.tile(np.linspace(0.25, 99.75, 200)[:, None], (1, 5))
    rows, membership = select_top_traces(
        percentile,
        n_traces=10,
        threshold=80.0,
    )
    selected = percentile[rows, 0]
    assert len(np.unique(rows)) == 10
    assert selected.min() < 84.0
    assert selected.max() > 97.0
    assert membership[rows].all(axis=0).all()


def test_unit_top_bin_effects_pool_images_before_percent_change():
    rate = np.ones((2, 3, 2, 3, 4), dtype=float)
    expected = np.ones_like(rate)
    ssi = np.ones_like(rate)
    rate[:, :, 1] *= 1.10
    ssi[:, :, 1] *= 1.20
    membership = np.ones((3, 4), dtype=bool)
    unit_rate, unit_ssi = unit_top_bin_effects(
        rate,
        expected,
        ssi,
        membership,
    )
    np.testing.assert_allclose(unit_rate, 10.0)
    np.testing.assert_allclose(unit_ssi, 20.0)


def test_normalized_effects_report_modulation_points_and_absolute_ssi():
    temporal = np.zeros((2, 3, 2, 3, 4), dtype=float)
    temporal[:, :, 0] = 0.05
    temporal[:, :, 1] = 0.15
    expected = np.ones_like(temporal)
    ssi = np.full_like(temporal, 0.25)
    ssi[:, :, 1] = 0.30
    membership = np.ones((3, 4), dtype=bool)
    unit_temporal, unit_ssi = unit_top_bin_normalized_effects(
        temporal,
        expected,
        ssi,
        membership,
    )
    np.testing.assert_allclose(unit_temporal, 10.0)
    np.testing.assert_allclose(unit_ssi, 0.05)
