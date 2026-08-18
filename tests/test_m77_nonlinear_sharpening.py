from __future__ import annotations

import numpy as np
import torch

from paper.fig4.spatiotemporal_tuning.audit_m77_nonlinear_sharpening import (
    activation_metrics,
    crossed_bootstrap_median,
    evenly_spaced_rows,
)


def test_evenly_spaced_rows_are_unique_and_cover_endpoints() -> None:
    rows = evenly_spaced_rows(100, 20)
    assert len(rows) == 20
    assert rows[0] == 0
    assert rows[-1] == 99
    assert np.all(np.diff(rows) > 0)


def test_activation_information_distinguishes_uniform_and_sharp_maps() -> None:
    uniform = torch.ones(5, 2, 4, 4)
    sharp = torch.zeros_like(uniform)
    sharp[..., 0, 0] = 16.0
    uniform_metric = activation_metrics(uniform, signed=False)
    sharp_metric = activation_metrics(sharp, signed=False)
    assert uniform_metric["spatial_information"] == 0.0
    assert sharp_metric["spatial_information"] > 3.9
    assert sharp_metric["spatial_modulation"] > uniform_metric["spatial_modulation"]


def test_signed_stage_uses_energy_without_clipping_negative_drive() -> None:
    value = torch.tensor([[[[-2.0, 2.0], [-1.0, 1.0]]]])
    result = activation_metrics(value, signed=True)
    assert result["mean_activation"] == 2.5
    assert result["negative_fraction_before_clip"] == 0.0


def test_crossed_bootstrap_constant_matrix_is_exact() -> None:
    center, low, high = crossed_bootstrap_median(
        np.full((5, 6), 1.25), n_bootstrap=30, rng=np.random.default_rng(3)
    )
    assert center == low == high == 1.25
