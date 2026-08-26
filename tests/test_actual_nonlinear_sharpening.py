from __future__ import annotations

import numpy as np
import torch

from paper.fig4.spatiotemporal_tuning.audit_actual_nonlinear_sharpening import (
    crossed_bootstrap_pooled_information_percent,
    finalize_components,
    information_components,
    positive_representation,
)


def test_signed_metric_matches_materialized_split_relu() -> None:
    signed = torch.tensor(
        [[[[2.0, -1.0], [0.5, -3.0]], [[-2.0, 4.0], [1.0, -0.5]]]]
    )
    split = positive_representation(signed, signed=True)
    signed_parts = information_components(signed, signed=True)
    split_parts = information_components(split, signed=False)
    for left, right in zip(signed_parts[:3], split_parts[:3]):
        torch.testing.assert_close(left, right)
    assert signed_parts[3] == split_parts[3]


def test_information_is_pooled_by_activation_mass() -> None:
    uniform = torch.ones(1, 1, 2, 2)
    sharp = torch.zeros(1, 1, 2, 2)
    sharp[..., 0, 0] = 40.0
    first = information_components(uniform, signed=False)
    second = information_components(sharp, signed=False)
    numerator = np.asarray([float(first[0] + second[0])])
    mass = np.asarray([float(first[1] + second[1])])
    total = np.asarray([float(first[2] + second[2])])
    count = np.asarray([int(first[3] + second[3])])
    result = finalize_components(numerator, mass, total, count)
    expected = float((first[0] + second[0]) / (first[1] + second[1]))
    assert np.isclose(result[0, 1], expected)
    assert result[0, 1] > 1.8


def test_pooled_percent_is_not_median_of_pairwise_percentages() -> None:
    numerator = np.asarray([[[[1.0], [1.1]]], [[[9.0], [18.0]]]])
    mass = np.ones_like(numerator)
    center, low, high = crossed_bootstrap_pooled_information_percent(
        numerator,
        mass,
        n_bootstrap=50,
        seed=4,
    )
    # Pooled baseline is 5 and pooled motion information is 9.55: +91%.
    # The median of the two pairwise changes would instead be +55%.
    np.testing.assert_allclose(center, 91.0)
    assert low.shape == high.shape == center.shape == (1,)
