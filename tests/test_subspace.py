"""Tests for VisionCore.subspace — subspace alignment and dimensionality."""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_participation_ratio_identity():
    """Identity matrix: PR = N (all eigenvalues equal)."""
    from VisionCore.subspace import participation_ratio
    C = np.eye(5)
    pr = participation_ratio(C)
    assert_allclose(pr, 5.0, rtol=1e-10)


def test_participation_ratio_rank1():
    """Rank-1 matrix: PR = 1."""
    from VisionCore.subspace import participation_ratio
    v = np.array([1.0, 0, 0, 0, 0])
    C = np.outer(v, v)
    pr = participation_ratio(C)
    assert_allclose(pr, 1.0, rtol=1e-10)


def test_symmetric_subspace_overlap_identical():
    """Identical subspaces: overlap = 1."""
    from VisionCore.subspace import symmetric_subspace_overlap
    U = np.eye(5, 2)  # first 2 columns of identity
    overlap = symmetric_subspace_overlap(U, U)
    assert_allclose(overlap, 1.0, rtol=1e-10)


def test_symmetric_subspace_overlap_orthogonal():
    """Orthogonal subspaces: overlap = 0."""
    from VisionCore.subspace import symmetric_subspace_overlap
    Ua = np.eye(5, 2)          # cols 0, 1
    Ub = np.zeros((5, 2))
    Ub[2, 0] = 1.0
    Ub[3, 1] = 1.0
    overlap = symmetric_subspace_overlap(Ua, Ub)
    assert_allclose(overlap, 0.0, atol=1e-10)


def test_directional_variance_capture_aligned():
    """Source basis = eigenvectors of target: capture = k/N for identity."""
    from VisionCore.subspace import directional_variance_capture
    C = np.eye(5)
    U = np.eye(5, 2)  # top 2 eigenvectors
    capture = directional_variance_capture(C, U)
    assert_allclose(capture, 2.0 / 5.0, rtol=1e-10)


def test_directional_variance_capture_full_basis():
    """Full basis captures all variance."""
    from VisionCore.subspace import directional_variance_capture
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    C = A @ A.T  # PSD
    U = np.linalg.eigh(C)[1]  # full eigenbasis
    capture = directional_variance_capture(C, U)
    assert_allclose(capture, 1.0, rtol=1e-10)


def test_project_to_psd_already_psd():
    """PSD matrix unchanged by projection."""
    from VisionCore.subspace import project_to_psd
    C = np.eye(3) * 2.0
    C_psd = project_to_psd(C)
    assert_allclose(C_psd, C, atol=1e-10)


def test_project_to_psd_clamps_negatives():
    """Negative eigenvalues are clamped to eps."""
    from VisionCore.subspace import project_to_psd
    # Construct matrix with a known negative eigenvalue
    C = np.diag([3.0, 1.0, -0.5])
    C_psd = project_to_psd(C, eps=0.0)
    evals = np.linalg.eigvalsh(C_psd)
    assert np.all(evals >= -1e-12)
