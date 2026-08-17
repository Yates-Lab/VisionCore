"""Unit tests for the exact Figure 4 SSI mechanism decomposition."""

import math

import numpy as np
import torch

from paper.fig4.mechanism_audit_v1.ssi_mechanism_v2.run_exact_decomposition import (
    hybrid,
    rate_components,
)


def test_uniform_rate_map_has_zero_ssi_and_spatial_variance() -> None:
    rates = torch.full((2, 3, 7, 9), 4.25)
    components = rate_components(rates)
    torch.testing.assert_close(components["ssi"], torch.zeros((2, 3), dtype=torch.float64))
    torch.testing.assert_close(components["cv2"], torch.zeros((2, 3), dtype=torch.float64))


def test_ssi_is_invariant_to_uniform_rate_scaling() -> None:
    generator = torch.Generator().manual_seed(7)
    rates = torch.rand((2, 4, 8, 8), generator=generator) + 0.2
    original = rate_components(rates)["ssi"]
    scaled = rate_components(17.0 * rates)["ssi"]
    torch.testing.assert_close(original, scaled, atol=5e-8, rtol=5e-7)


def test_small_spatial_contrast_has_quadratic_information_limit() -> None:
    pattern = torch.tensor([-1.0, 1.0] * 50).reshape(1, 1, 10, 10)
    rates = 3.0 * (1.0 + 0.02 * pattern)
    components = rate_components(rates)
    exact = float(components["ssi"])
    predicted = float(components["cv2"] / (2.0 * math.log(2.0)))
    assert np.isclose(exact, predicted, rtol=5e-4)


def test_channel_hybrid_replaces_only_selected_channels() -> None:
    anchor = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    donor = -anchor - 1
    observed = hybrid(anchor, donor, np.asarray([1, 4]))
    torch.testing.assert_close(observed[:, [0, 2, 3]], anchor[:, [0, 2, 3]])
    torch.testing.assert_close(observed[:, [1, 4]], donor[:, [1, 4]])
    # The operation must not mutate either source tensor.
    torch.testing.assert_close(anchor, torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3))
