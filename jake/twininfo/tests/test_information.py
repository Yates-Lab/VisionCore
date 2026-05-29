from __future__ import annotations

import numpy as np

from jake.twininfo.information import (
    event_code_information,
    event_code_information_pattern_only,
    fisher_scalars,
    poisson_fisher_count_pattern_decomposition,
    poisson_fisher_from_counts,
    single_spike_info_event_code,
    single_spike_info_pattern_only,
)


def test_flat_counts_have_zero_fisher_and_information():
    mu = np.ones((5, 4)) * 0.1
    j = np.zeros((5, 4, 2))
    f = poisson_fisher_from_counts(mu, j)
    assert np.allclose(f, 0.0)
    total, count, pattern, info = poisson_fisher_count_pattern_decomposition(mu, j)
    assert np.allclose(total, 0.0)
    assert np.allclose(count, 0.0)
    assert np.allclose(pattern, 0.0)
    assert info["n_valid"] == mu.size
    by_shift = np.stack([mu, mu], axis=0)
    assert single_spike_info_event_code(by_shift)["bits_per_spike"] == 0.0
    assert single_spike_info_pattern_only(by_shift)["bits_per_spike_pattern"] == 0.0


def test_global_gain_modulation_is_count_information():
    base = np.array([[0.2, 0.3, 0.4], [0.1, 0.25, 0.35]])
    d_gain = base[..., None] * np.array([[[1.0, -0.5]]])
    total, count, pattern, _ = poisson_fisher_count_pattern_decomposition(base, d_gain)
    assert np.allclose(total, count + pattern, atol=1e-10)
    assert np.linalg.norm(count) > 0
    assert np.linalg.norm(pattern) < 1e-10


def test_pattern_only_modulation_has_no_count_information():
    mu = np.array([0.25, 0.25, 0.25, 0.25])
    j = np.array([[1.0], [-1.0], [0.5], [-0.5]]) * 0.01
    total, count, pattern, _ = poisson_fisher_count_pattern_decomposition(mu, j)
    assert np.allclose(count, 0.0, atol=1e-12)
    assert np.allclose(total, pattern, atol=1e-12)


def test_fisher_scales_with_expected_counts():
    mu = np.array([0.2, 0.4, 0.8])
    j = np.array([[0.1, 0.0], [0.0, 0.2], [0.3, 0.1]])
    f = poisson_fisher_from_counts(mu, j)
    scale = 3.5
    f_scaled = poisson_fisher_from_counts(scale * mu, scale * j)
    assert np.allclose(f_scaled, scale * f)


def test_single_spike_gain_and_pattern_controls():
    base = np.array([0.1, 0.2, 0.3])
    gain = np.stack([base, 2.0 * base, 0.5 * base], axis=0)
    event = single_spike_info_event_code(gain)
    pattern = single_spike_info_pattern_only(gain)
    assert event["bits_per_spike"] > 0.0
    assert abs(pattern["bits_per_spike_pattern"]) < 1e-12

    pattern_mu = np.array([
        [0.4, 0.1, 0.1],
        [0.1, 0.4, 0.1],
        [0.1, 0.1, 0.4],
    ])
    pattern_info = single_spike_info_pattern_only(pattern_mu)
    assert pattern_info["bits_per_spike_pattern"] > 0.0
    assert np.isclose(
        single_spike_info_pattern_only(5.0 * pattern_mu)["bits_per_spike_pattern"],
        pattern_info["bits_per_spike_pattern"],
    )


def test_event_info_zero_for_identical_states():
    mu = np.ones((3, 4, 5)) * 0.2
    total = event_code_information(mu)
    pattern = event_code_information_pattern_only(mu)
    assert abs(total["bits_per_spike_total"]) < 1e-12
    assert abs(pattern["bits_per_spike_pattern"]) < 1e-12
    assert total["n_states"] == 3
    assert total["n_events"] == 20


def test_event_info_pattern_zero_for_gain_only_states():
    base = np.array([[0.2, 0.4], [0.1, 0.3]])
    mu = np.stack([base, 2.0 * base, 0.5 * base], axis=0)
    total = event_code_information(mu)
    pattern = event_code_information_pattern_only(mu)
    assert total["bits_per_spike_total"] > 0.0
    assert abs(pattern["bits_per_spike_pattern"]) < 1e-12


def test_event_info_pattern_positive_for_pattern_states():
    mu = np.array([
        [[0.5, 0.1], [0.1, 0.1]],
        [[0.1, 0.5], [0.1, 0.1]],
        [[0.1, 0.1], [0.5, 0.1]],
    ])
    total = event_code_information(mu)
    pattern = event_code_information_pattern_only(mu)
    assert total["bits_per_spike_total"] > 0.0
    assert pattern["bits_per_spike_pattern"] > 0.0
    assert np.isclose(total["bits_per_spike_total"], pattern["bits_per_spike_pattern"])


def test_event_info_bits_per_spike_invariant_to_global_rate_scaling():
    mu = np.array([
        [[0.5, 0.1], [0.1, 0.1]],
        [[0.1, 0.5], [0.1, 0.1]],
    ])
    total = event_code_information(mu)
    scaled = event_code_information(7.0 * mu)
    assert np.isclose(total["bits_per_spike_total"], scaled["bits_per_spike_total"])
    assert np.isclose(7.0 * total["bits_per_window_total"], scaled["bits_per_window_total"])


def test_fisher_scalars_are_finite_for_psd_matrix():
    out = fisher_scalars(np.array([[3.0, 0.5], [0.5, 2.0]]))
    assert out["trace"] == 5.0
    assert np.isfinite(out["logdet"])
    assert out["ellipse_area"] > 0.0
