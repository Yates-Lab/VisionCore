"""Tests for the figure3e_extended single-trial Poisson metrics.

The module under test lives beside the figure (`ryan/figure3e_extended/`) rather
than in an installed package, so it is imported off an explicit sys.path entry.
It deliberately imports nothing from `_ext_data`/`_ext_stim`, which pull in the
data packages and a GPU checkpoint.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "ryan" / "figure3e_extended"
for _p in (str(REPO_ROOT), str(FIG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- closed-form likelihood gain ---

def test_likelihood_gain_matches_manual_sum():
    """Both gains equal the term-by-term sum of the report's expressions."""
    from _ext_metrics import EPS, poisson_likelihood_gain

    q = np.array([[0.10], [0.40], [0.02]])
    y = np.array([[0.0], [2.0], [1.0]])
    q0 = np.array([[0.15]])
    valid = np.ones_like(q, dtype=bool)

    g_obs, g_self = poisson_likelihood_gain(q, y, q0, valid)

    exp_obs = exp_self = 0.0
    for i in range(3):
        a = math.log(q[i, 0] + EPS) - math.log(q0[0, 0] + EPS)
        exp_obs += y[i, 0] * a - q[i, 0] + q0[0, 0]
        exp_self += q[i, 0] * a - q[i, 0] + q0[0, 0]

    assert_allclose(g_obs, [exp_obs], rtol=1e-12)
    assert_allclose(g_self, [exp_self], rtol=1e-12)


def test_self_gain_is_sum_of_poisson_kl_divergences():
    """G_self is sum_i KL[Pois(q_i) || Pois(q0_i)] (up to the shared log floor)."""
    from _ext_metrics import poisson_likelihood_gain

    q = np.array([[0.05, 1.0], [0.30, 0.5], [0.90, 0.2]])
    q0 = np.array([[0.20, 0.4]])
    valid = np.ones_like(q, dtype=bool)

    _, g_self = poisson_likelihood_gain(q, np.zeros_like(q), q0, valid)
    kl = (q * np.log(q / q0) + q0 - q).sum(axis=0)

    assert_allclose(g_self, kl, rtol=1e-6)
    assert np.all(g_self > 0)


# --- the self-consistency fraction ---

def test_fraction_is_one_when_counts_equal_predicted_rates():
    """y == q makes the observed gain identical to its own Poisson reference."""
    from _ext_metrics import poisson_self_consistency

    rng = np.random.default_rng(0)
    q = rng.uniform(0.01, 0.5, size=(200, 3))
    dfs = np.ones_like(q)

    S, g_obs, g_self, n_floored = poisson_self_consistency(q, q, dfs)

    assert_allclose(S, np.ones(3), rtol=1e-10)
    assert_allclose(g_obs, g_self, rtol=1e-10)
    assert np.all(g_self > 0)
    assert n_floored == 0


def test_masked_bins_are_excluded():
    """Bins with dfs == 0 or non-finite entries cannot influence the fraction."""
    from _ext_metrics import poisson_self_consistency

    rng = np.random.default_rng(1)
    q = rng.uniform(0.01, 0.5, size=(50, 2))
    y = rng.poisson(q).astype(float)
    dfs = np.ones_like(q)

    S_ref, g_obs_ref, g_self_ref, _ = poisson_self_consistency(q, y, dfs)

    # Pad with bins that are either masked out or NaN, holding wild values.
    q_pad = np.concatenate([q, np.full((10, 2), 1e6), np.full((5, 2), np.nan)])
    y_pad = np.concatenate([y, np.full((10, 2), 500.0), np.full((5, 2), 7.0)])
    dfs_pad = np.concatenate([dfs, np.zeros((10, 2)), np.ones((5, 2))])

    S, g_obs, g_self, _ = poisson_self_consistency(q_pad, y_pad, dfs_pad)

    assert_allclose(S, S_ref, rtol=1e-12)
    assert_allclose(g_obs, g_obs_ref, rtol=1e-12)
    assert_allclose(g_self, g_self_ref, rtol=1e-12)


def test_null_equivalent_prediction_is_undefined():
    """A prediction equal to the mean-rate null has a zero denominator -> NaN."""
    from _ext_metrics import poisson_self_consistency

    rng = np.random.default_rng(2)
    y = rng.poisson(0.2, size=(300, 2)).astype(float)
    dfs = np.ones_like(y)
    q = np.broadcast_to(y.mean(axis=0), y.shape).copy()  # == the mean-rate null

    S, _, g_self, _ = poisson_self_consistency(q, y, dfs)

    assert_allclose(g_self, np.zeros(2), atol=1e-12)
    assert np.all(np.isnan(S))


def test_fraction_is_negative_when_null_beats_the_prediction():
    """A prediction anti-correlated with the counts scores below zero."""
    from _ext_metrics import poisson_self_consistency

    y = np.tile(np.array([[0.0], [0.0], [4.0], [4.0]]), (25, 1))
    q = np.tile(np.array([[1.0], [1.0], [0.01], [0.01]]), (25, 1))
    dfs = np.ones_like(y)

    S, g_obs, g_self, _ = poisson_self_consistency(q, y, dfs)

    assert g_self[0] > 0
    assert g_obs[0] < 0
    assert S[0] < 0


# --- the expectation the metric rests on ---

def test_monte_carlo_average_gain_matches_closed_form():
    """Averaging G(q; Y) over Poisson(q) draws converges to the closed form.

    The null is held fixed across draws, as the derivation requires. Tolerance
    is 4 standard errors of the Monte Carlo mean, using the analytic
    Var[G] = sum_i q_i [log(q_i / q0_i)]^2.
    """
    from _ext_metrics import EPS, poisson_likelihood_gain

    rng = np.random.default_rng(20240724)
    n_bins, n_draws = 400, 20000
    q = rng.uniform(0.01, 0.6, size=(n_bins, 1))
    q0 = np.full((1, 1), 0.2)
    valid = np.ones_like(q, dtype=bool)

    _, g_self = poisson_likelihood_gain(q, np.zeros_like(q), q0, valid)

    draws = rng.poisson(q[:, 0], size=(n_draws, n_bins)).astype(float)
    a = (np.log(q + EPS) - np.log(q0 + EPS))[:, 0]
    base = float((-q + q0).sum())
    g_obs_draws = draws @ a + base

    se = math.sqrt(float((q[:, 0] * a ** 2).sum()) / n_draws)
    assert abs(g_obs_draws.mean() - g_self[0]) < 4 * se


# --- agreement with the existing bits/spike row ---

def test_observed_gain_reproduces_bits_per_spike():
    """G_obs / (N log 2) is exactly the existing bits/spike number."""
    from _ext_metrics import bits_per_spike, poisson_self_consistency

    rng = np.random.default_rng(3)
    q = rng.uniform(0.005, 0.8, size=(400, 4))
    y = rng.poisson(q).astype(float)
    dfs = (rng.uniform(size=q.shape) > 0.1).astype(float)

    bps, _ = bits_per_spike(q, y, dfs)
    _, g_obs, _, _ = poisson_self_consistency(q, y, dfs)

    valid = dfs > 0
    n_spikes = np.maximum((valid * y).sum(axis=0), 1.0)

    assert_allclose(g_obs / (n_spikes * math.log(2)), bps, rtol=1e-8, atol=1e-10)


def test_prediction_floor_is_applied():
    """Non-positive predictions are raised to the floor and counted."""
    from _ext_metrics import BPS_FLOOR, poisson_self_consistency

    q = np.array([[0.3], [-0.2], [0.0], [0.5]])
    y = np.array([[1.0], [0.0], [0.0], [2.0]])
    dfs = np.ones_like(y)

    _, _, _, n_floored = poisson_self_consistency(q, y, dfs)
    assert n_floored == 2

    q_floored = np.maximum(q, BPS_FLOOR)
    S_a, _, _, _ = poisson_self_consistency(q, y, dfs)
    S_b, _, _, _ = poisson_self_consistency(q_floored, y, dfs)
    assert_allclose(S_a, S_b, rtol=1e-12)


def test_accepts_trial_time_neuron_arrays():
    """The wrapper flattens leading axes, like the bits/spike wrapper does."""
    from _ext_metrics import poisson_self_consistency

    rng = np.random.default_rng(4)
    q = rng.uniform(0.01, 0.4, size=(12, 30, 3))
    y = rng.poisson(q).astype(float)
    dfs = np.ones_like(q)

    S, g_obs, g_self, _ = poisson_self_consistency(q, y, dfs)
    S_flat, _, _, _ = poisson_self_consistency(
        q.reshape(-1, 3), y.reshape(-1, 3), dfs.reshape(-1, 3))

    assert S.shape == (3,)
    assert_allclose(S, S_flat, rtol=1e-12)
