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
