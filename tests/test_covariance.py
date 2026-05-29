"""Tests for VisionCore.covariance — LOTC decomposition primitives."""
import numpy as np
import torch
import pytest
from numpy.testing import assert_allclose


# --- cov_to_corr ---

def test_cov_to_corr_identity():
    """Identity covariance -> zero correlations (diagonal set to 0)."""
    from VisionCore.covariance import cov_to_corr
    C = np.eye(3)
    R = cov_to_corr(C)
    assert_allclose(R, np.zeros((3, 3)), atol=1e-10)


def test_cov_to_corr_known():
    """Known covariance -> correct off-diagonal correlations."""
    from VisionCore.covariance import cov_to_corr
    C = np.array([[4.0, 2.0], [2.0, 4.0]])
    R = cov_to_corr(C)
    assert_allclose(R[0, 1], 0.5, atol=1e-5)
    assert_allclose(R[1, 0], 0.5, atol=1e-5)
    # diagonal is 0 by convention
    assert R[0, 0] == 0.0
    assert R[1, 1] == 0.0


def test_cov_to_corr_low_variance_neuron():
    """Neuron with variance below min_var gets NaN correlations."""
    from VisionCore.covariance import cov_to_corr
    C = np.array([[4.0, 1.0, 0.5],
                  [1.0, 0.0001, 0.0],  # below min_var=1e-3
                  [0.5, 0.0, 4.0]])
    R = cov_to_corr(C, min_var=1e-3)
    assert np.isnan(R[0, 1])
    assert np.isnan(R[1, 0])
    assert np.isnan(R[1, 2])
    # valid pair should be fine
    assert np.isfinite(R[0, 2])


# --- pava_nonincreasing ---

def test_pava_already_nonincreasing():
    """Already monotone sequence is unchanged."""
    from VisionCore.covariance import pava_nonincreasing
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    w = np.ones(5)
    yhat = pava_nonincreasing(y, w)
    assert_allclose(yhat, y)


def test_pava_single_violation():
    """Single violation gets pooled."""
    from VisionCore.covariance import pava_nonincreasing
    y = np.array([5.0, 3.0, 4.0, 2.0])  # violation at index 1->2
    w = np.ones(4)
    yhat = pava_nonincreasing(y, w)
    # y[1] and y[2] should be pooled to their mean (3.5)
    assert_allclose(yhat, [5.0, 3.5, 3.5, 2.0])
    # result must be non-increasing
    assert np.all(np.diff(yhat) <= 1e-10)


def test_pava_all_equal():
    """All equal values stay equal."""
    from VisionCore.covariance import pava_nonincreasing
    y = np.array([3.0, 3.0, 3.0])
    w = np.ones(3)
    yhat = pava_nonincreasing(y, w)
    assert_allclose(yhat, [3.0, 3.0, 3.0])


# --- project_to_psd (covariance version) ---

def test_project_to_psd_psd_unchanged():
    """PSD matrix is unchanged."""
    from VisionCore.covariance import project_to_psd
    C = np.eye(3) * 2.0
    C_psd = project_to_psd(C)
    assert_allclose(C_psd, C, atol=1e-10)


def test_project_to_psd_negative_eigenvalue():
    """Negative eigenvalues clamped to 0."""
    from VisionCore.covariance import project_to_psd
    C = np.diag([3.0, 1.0, -0.5])
    C_psd = project_to_psd(C, eps=0.0)
    evals = np.linalg.eigvalsh(C_psd)
    assert np.all(evals >= -1e-12)


# --- get_upper_triangle ---

def test_get_upper_triangle():
    """Extract upper triangle values."""
    from VisionCore.covariance import get_upper_triangle
    C = np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
    vals = get_upper_triangle(C)
    assert_allclose(sorted(vals), sorted([0.5, 0.3, 0.2]))


# --- extract_valid_segments ---

def test_extract_valid_segments_basic():
    """Finds contiguous valid segments above min length."""
    from VisionCore.covariance import extract_valid_segments
    # 2 trials, 10 time bins
    mask = np.zeros((2, 10), dtype=bool)
    mask[0, 2:8] = True   # 6-bin segment
    mask[1, 0:4] = True   # 4-bin segment
    mask[1, 6:10] = True  # 4-bin segment
    segs = extract_valid_segments(mask, min_len_bins=5)
    assert len(segs) == 1  # only the 6-bin segment
    assert segs[0] == (0, 2, 8)


def test_extract_valid_segments_none():
    """No segments above threshold."""
    from VisionCore.covariance import extract_valid_segments
    mask = np.zeros((2, 10), dtype=bool)
    mask[0, 0:3] = True
    segs = extract_valid_segments(mask, min_len_bins=5)
    assert len(segs) == 0


# --- rate_variance_components (analytic ANOVA decomposition of model rates) ---

def test_rate_variance_components_recovers_known_variances():
    """Unbiased: averaged over sims, components recover true within/between var.

    Generate a deterministic-rate matrix r[i,t] = mu(t) + eps(i,t) where the
    phase means mu(t) carry the between (PSTH) variance and eps carries the
    within (FEM) variance. The random-effects estimator must be unbiased for
    both, so the Monte-Carlo mean of the estimates equals the truth.
    """
    from VisionCore.covariance import rate_variance_components
    rng = np.random.default_rng(0)
    sigma_b, sigma_w = 1.5, 0.8  # std devs
    T, n, n_sims = 60, 40, 300
    sw, sb = [], []
    for _ in range(n_sims):
        mu = rng.normal(0.0, sigma_b, size=T)
        r = mu[None, :] + rng.normal(0.0, sigma_w, size=(n, T))
        out = rate_variance_components(r)
        sw.append(out["sigma2_within"])
        sb.append(out["sigma2_between"])
    assert_allclose(np.mean(sw), sigma_w ** 2, rtol=0.03)
    assert_allclose(np.mean(sb), sigma_b ** 2, rtol=0.06)


def test_rate_variance_components_affine_invariant():
    """1-alpha is exactly invariant to an affine rescale a*r + b of the rates."""
    from VisionCore.covariance import rate_variance_components
    rng = np.random.default_rng(1)
    mu = rng.normal(0.0, 3.0, size=50)
    r = mu[None, :] + rng.normal(0.0, 1.0, size=(30, 50))
    base = rate_variance_components(r)["one_minus_alpha"]
    scaled = rate_variance_components(7.3 * r - 2.1)["one_minus_alpha"]
    assert_allclose(scaled, base, rtol=1e-10)


def test_rate_variance_components_pure_psth_gives_zero():
    """No within-phase variability -> all variance is PSTH -> 1-alpha = 0."""
    from VisionCore.covariance import rate_variance_components
    mu = np.linspace(0.0, 10.0, 40)
    r = np.tile(mu, (25, 1))  # every trial identical within a phase
    out = rate_variance_components(r)
    assert out["sigma2_within"] == pytest.approx(0.0, abs=1e-12)
    assert out["one_minus_alpha"] == pytest.approx(0.0, abs=1e-9)


def test_rate_variance_components_pure_fem_debiased():
    """Equal phase means + large within-variance -> 1-alpha = 1 even at small n.

    With only n=10 trials/phase a naive Var_t(mean) would inflate the PSTH
    component by ~sigma_w^2/n and pull 1-alpha well below 1; the debiased
    random-effects estimator must still return ~1.
    """
    from VisionCore.covariance import rate_variance_components
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 5.0, size=(10, 200))  # equal means across phases
    out = rate_variance_components(r)
    assert out["one_minus_alpha"] > 0.97


def test_rate_variance_components_unequal_groups_exact_between():
    """Unequal n_t with zero within-variance: between component is exact.

    Pins the unbalanced n0 correction n0 = (N - sum n_t^2 / N) / (T - 1).
    """
    from VisionCore.covariance import rate_variance_components
    mu = np.array([0.0, 2.0, 5.0, 9.0])
    counts = [10, 7, 4, 3]
    r = np.full((10, 4), np.nan)
    for t, c in enumerate(counts):
        r[:c, t] = mu[t]
    out = rate_variance_components(r, min_trials_per_phase=2)

    n_t = np.array(counts, dtype=float)
    N, T = n_t.sum(), len(counts)
    grand = (n_t * mu).sum() / N
    ms_between = (n_t * (mu - grand) ** 2).sum() / (T - 1)
    n0 = (N - (n_t ** 2).sum() / N) / (T - 1)
    expected_between = ms_between / n0  # MS_within = 0 here

    assert out["sigma2_within"] == pytest.approx(0.0, abs=1e-12)
    assert out["sigma2_between"] == pytest.approx(expected_between, rel=1e-9)
    assert out["one_minus_alpha"] == pytest.approx(0.0, abs=1e-9)


def test_psth_variance_splithalf_matches_analytic():
    """Split-half PSTH variance agrees with the analytic between component."""
    from VisionCore.covariance import (
        rate_variance_components, psth_variance_splithalf,
    )
    rng = np.random.default_rng(3)
    sigma_b, sigma_w = 1.2, 1.0
    T, n = 80, 60
    mu = rng.normal(0.0, sigma_b, size=T)
    r = mu[None, :] + rng.normal(0.0, sigma_w, size=(n, T))
    analytic = rate_variance_components(r)["sigma2_between"]
    sh = psth_variance_splithalf(r, n_boot=100, seed=0)
    assert_allclose(sh, analytic, rtol=0.15)
