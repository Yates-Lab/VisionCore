from __future__ import annotations

import numpy as np

from jake.twininfo.lagcube_information import (
    approximate_unit_fisher_scores,
    block_endpoint_lag_cubes,
    block_current_samples,
    cross_shift_grid,
    cumulative_pattern_fisher,
    cumulative_spatial_ssi,
    cumulative_shift_grid_ssi,
    finite_difference_shift_set,
    square_shift_grid,
    unique_shifts,
)
from jake.twininfo.run_lagcube_information import (
    MAIN_CONDITIONS,
    PHASE_COMPARISON_CONDITIONS,
    SF_COMPARISON_CONDITIONS,
)


def test_main_lagcube_conditions_use_pyramid_phase_and_sf_bands():
    assert PHASE_COMPARISON_CONDITIONS == ("real", "stabilized", "pyramid_phase_scrambled")
    assert SF_COMPARISON_CONDITIONS == ("real", "sf_low", "sf_mid_low", "sf_mid_high", "sf_high")
    assert MAIN_CONDITIONS == (
        "real",
        "stabilized",
        "pyramid_phase_scrambled",
        "sf_low",
        "sf_mid_low",
        "sf_mid_high",
        "sf_high",
    )


def test_lag_cubes_use_all_overlapping_current_samples():
    rng = np.random.default_rng(3)
    cubes = np.zeros((8, 4, 5, 6), dtype=np.float32)
    for t in range(cubes.shape[0]):
        for lag in range(cubes.shape[1]):
            cubes[t, lag] = rng.normal(loc=t - lag, scale=0.5, size=(5, 6))

    blocks, current = block_endpoint_lag_cubes(cubes, n_lags=4)

    assert np.all(current == block_current_samples(8, 4))
    assert np.all(current == np.arange(8))
    assert blocks.shape == (8, 4, 5, 6)
    assert np.allclose(blocks, cubes)


def test_shift_sets_are_ordered_unique():
    square = square_shift_grid(1.0, 1.0)
    cross = cross_shift_grid(1.0, 1.0)
    fisher = finite_difference_shift_set(0.5)
    merged = unique_shifts(square, cross, fisher)

    assert square.shape == (9, 2)
    assert cross.shape == (5, 2)
    assert fisher.shape == (5, 2)
    assert merged.shape[1] == 2
    assert len({tuple(row) for row in merged}) == merged.shape[0]


def test_cumulative_metrics_increase_for_shift_sensitive_rates():
    t = 6
    n = 4
    h = 0.5 / 60.0
    base = np.ones((t, n), dtype=np.float32) * 5.0
    ramp = np.linspace(0.1, 0.6, t, dtype=np.float32)[:, None]
    pattern = np.array([[1.0, -0.5, 0.25, -0.75]], dtype=np.float32)
    rates = {
        (0.0, 0.0): base,
        (round(h, 8), 0.0): base + ramp * pattern,
        (round(-h, 8), 0.0): base - ramp * pattern,
        (0.0, round(h, 8)): base + 0.5 * ramp * pattern[:, ::-1],
        (0.0, round(-h, 8)): base - 0.5 * ramp * pattern[:, ::-1],
    }

    fisher = cumulative_pattern_fisher(rates, fisher_step_arcmin=0.5)

    assert fisher["cumulative_fisher_pattern"].shape == (t,)
    assert fisher["cumulative_fisher_pattern"][-1] > fisher["cumulative_fisher_pattern"][0]
    assert np.all(np.diff(fisher["cumulative_expected_spikes"]) > 0)


def test_cumulative_shift_grid_ssi_detects_pattern_difference():
    t = 5
    n = 3
    shifts = cross_shift_grid(1.0, 1.0)
    rates = {}
    for i, (dx, dy) in enumerate(shifts):
        arr = np.ones((t, n), dtype=np.float32) * 5.0
        arr[:, i % n] += 2.0
        rates[(round(float(dx), 8), round(float(dy), 8))] = arr

    ssi = cumulative_shift_grid_ssi(rates, shifts)

    assert ssi["prefix_ssi_pattern_bits_per_spike"].shape == (t,)
    assert ssi["prefix_ssi_pattern_bits_per_spike"][-1] > 0.0
    assert ssi["prefix_ssi_total_bits_per_spike"][-1] > 0.0
    assert ssi["cumulative_ssi_total_bits_per_window"][-1] > ssi["cumulative_ssi_total_bits_per_window"][0]


def test_cumulative_spatial_ssi_uses_spatial_rate_map():
    t = 5
    n = 3
    h = 4
    w = 4
    uniform = np.ones((t, n, h, w), dtype=np.float32) * 5.0
    assert np.allclose(cumulative_spatial_ssi(uniform)["spatial_ssi_bits_per_spike"], 0.0, atol=1e-7)

    structured = uniform.copy()
    structured[:, 0, :2, :2] = 12.0
    structured[:, 1, 2:, 2:] = 9.0
    spatial = cumulative_spatial_ssi(structured)
    scaled = cumulative_spatial_ssi(4.0 * structured)

    assert spatial["spatial_ssi_bits_per_spike"].shape == (t,)
    assert spatial["cumulative_spatial_ssi_bits_per_spike"][-1] > 0.0
    assert np.all(np.diff(spatial["cumulative_spatial_ssi_bits"]) >= -1e-7)
    assert np.allclose(
        scaled["cumulative_spatial_ssi_bits"],
        4.0 * spatial["cumulative_spatial_ssi_bits"],
        rtol=1e-6,
    )
    assert np.allclose(
        spatial["cumulative_spatial_ssi_bits_per_spike"],
        scaled["cumulative_spatial_ssi_bits_per_spike"],
    )


def test_approximate_unit_fisher_scores_rank_sensitive_unit():
    mu = np.ones((4, 3), dtype=np.float32)
    dmu = np.zeros((4, 3, 2), dtype=np.float32)
    dmu[:, 1, 0] = 3.0

    scores = approximate_unit_fisher_scores(mu, dmu)

    assert int(np.argmax(scores)) == 1
    assert scores[1] > scores[0]


def test_approximate_unit_fisher_scores_aggregate_spatial_maps_by_unit():
    mu = np.ones((4, 3, 2, 2), dtype=np.float32)
    dmu = np.zeros((4, 3, 2, 2, 2), dtype=np.float32)
    dmu[:, 2, :, :, 1] = 2.0

    scores = approximate_unit_fisher_scores(mu, dmu)

    assert scores.shape == (3,)
    assert int(np.argmax(scores)) == 2
