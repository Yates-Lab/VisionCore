"""Tests for VisionCore.stats — general-purpose statistical utilities."""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_geomean_positive_values():
    from VisionCore.stats import geomean
    x = np.array([1.0, 2.0, 4.0])
    result = geomean(x)
    assert_allclose(result, 2.0, rtol=1e-10)


def test_geomean_with_nonpositive():
    from VisionCore.stats import geomean
    x = np.array([1.0, -1.0, 4.0])
    result = geomean(x)
    # only positive values used: geomean([1, 4]) = 2
    assert_allclose(result, 2.0, rtol=1e-10)


def test_iqr_25_75():
    from VisionCore.stats import iqr_25_75
    x = np.arange(100, dtype=float)
    q25, q75 = iqr_25_75(x)
    assert_allclose(q25, 24.75, rtol=1e-2)
    assert_allclose(q75, 74.25, rtol=1e-2)


def test_bootstrap_mean_ci_returns_mean():
    from VisionCore.stats import bootstrap_mean_ci
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, size=1000)
    mean, (lo, hi) = bootstrap_mean_ci(x, nboot=2000, seed=0)
    assert_allclose(mean, x.mean(), rtol=1e-10)
    assert lo < mean < hi


def test_bootstrap_mean_ci_handles_nans():
    from VisionCore.stats import bootstrap_mean_ci
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    mean, (lo, hi) = bootstrap_mean_ci(x)
    assert_allclose(mean, 3.0, rtol=1e-10)


def test_bootstrap_paired_diff_ci():
    from VisionCore.stats import bootstrap_paired_diff_ci
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 200)
    b = rng.normal(8, 1, 200)
    diff, (lo, hi) = bootstrap_paired_diff_ci(a, b, nboot=5000, seed=0)
    assert_allclose(diff, np.mean(a - b), rtol=1e-10)
    assert lo > 0  # a > b reliably


def test_fisher_z_inverse():
    from VisionCore.stats import fisher_z
    rho = np.array([0.0, 0.5, -0.5, 0.9])
    z = fisher_z(rho)
    recovered = np.tanh(z)
    assert_allclose(recovered, rho, atol=1e-5)


def test_fisher_z_mean():
    from VisionCore.stats import fisher_z_mean
    rho = np.array([0.5, 0.5, 0.5])
    result = fisher_z_mean(rho)
    assert_allclose(result, np.arctanh(0.5), rtol=1e-5)


def test_emp_p_one_sided_greater():
    from VisionCore.stats import emp_p_one_sided
    null = np.arange(100, dtype=float)
    p = emp_p_one_sided(null, 50.0, direction="greater")
    # ~50/102 values >= 50 (with +1 smoothing)
    assert 0.4 < p < 0.6


def test_emp_p_one_sided_less():
    from VisionCore.stats import emp_p_one_sided
    null = np.arange(100, dtype=float)
    p = emp_p_one_sided(null, 50.0, direction="less")
    assert 0.4 < p < 0.6


def test_wilcoxon_signed_rank():
    from VisionCore.stats import wilcoxon_signed_rank
    rng = np.random.default_rng(0)
    a = rng.normal(5, 1, 100)
    b = rng.normal(3, 1, 100)
    stat, p = wilcoxon_signed_rank(a, b, alternative="greater")
    assert p < 0.001


def test_fdr_correct():
    from VisionCore.stats import fdr_correct
    pvals = np.array([0.001, 0.01, 0.04, 0.5])
    p_adj, sig = fdr_correct(pvals, q=0.05)
    assert sig[0] and sig[1]  # first two should survive
    assert not sig[3]  # 0.5 should not survive


def test_paired_valid_filters_nans():
    from VisionCore.stats import paired_valid
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([2.0, 3.0, np.nan, 5.0])
    av, bv, mask = paired_valid(a, b, positive=False)
    assert len(av) == 2
    assert_allclose(av, [1.0, 4.0])
    assert_allclose(bv, [2.0, 5.0])


def test_paired_valid_positive_filter():
    from VisionCore.stats import paired_valid
    a = np.array([1.0, -1.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    av, bv, mask = paired_valid(a, b, positive=True)
    assert len(av) == 2
    assert_allclose(av, [1.0, 3.0])
