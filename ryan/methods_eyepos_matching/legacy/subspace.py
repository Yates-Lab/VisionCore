"""Subspace alignment and dimensionality analysis for covariance matrices."""
import numpy as np


def project_to_psd(C, eps=0.0, max_reg_attempts=5):
    """
    Project a symmetric matrix onto the positive semi-definite cone.

    Eigenvalues below eps are clamped to eps. This handles the small
    negative eigenvalues that arise from numerical noise in covariance
    estimation (e.g., split-half cross-covariance).

    Non-finite values (NaN/Inf) are replaced with zeros before
    decomposition. This is safe when NaN entries span entire rows and
    columns (as produced by the bad_mask in estimate_rate_covariance),
    because a zero row/column forms a null eigenspace that does not
    interact with the valid subblock: the PSD projection of the valid
    entries is identical to projecting the valid submatrix alone. Downstream,
    cov_to_corr assigns NaN correlations to zero-variance neurons (below
    min_var), so they are excluded from summary statistics.

    If eigh fails to converge (e.g., from near-singular structure),
    increasing diagonal regularization is applied.

    Parameters
    ----------
    C : ndarray (N, N)
        Symmetric matrix.
    eps : float
        Floor for eigenvalues. Default 0.0 clamps negatives to zero.
    max_reg_attempts : int
        Number of regularization attempts if eigh fails. Default 5.

    Returns
    -------
    C_psd : ndarray (N, N)
        Nearest PSD matrix (in Frobenius norm).
    """
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)  # symmetrize

    if not np.isfinite(C).all():
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    reg = 0.0
    for attempt in range(max_reg_attempts):
        try:
            C_reg = C + reg * np.eye(C.shape[0])
            w, V = np.linalg.eigh(C_reg)
            w_clamped = np.maximum(w, eps)
            return (V * w_clamped) @ V.T
        except np.linalg.LinAlgError:
            reg = 10 ** (-10 + 2 * attempt)

    return C + 1e-2 * np.eye(C.shape[0])


def participation_ratio(C_psd, eps=1e-12):
    """
    Participation ratio: effective dimensionality of a PSD covariance matrix.

    PR = (tr C)^2 / tr(C^2)

    Ranges from 1 (rank-1) to N (identity / equal eigenvalues).

    Parameters
    ----------
    C_psd : ndarray (N, N)
        Positive semi-definite matrix.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    pr : float
    """
    w = np.linalg.eigvalsh(C_psd)
    w = np.maximum(w, 0.0)
    tr = w.sum()
    tr2 = (w ** 2).sum()
    return float(tr ** 2 / (tr2 + eps))


def symmetric_subspace_overlap(Ua, Ub):
    """
    Symmetric subspace overlap between two orthonormal bases.

    overlap = ||Ua^T Ub||_F^2 / k

    Ranges from 0 (orthogonal subspaces) to 1 (identical subspaces).

    Parameters
    ----------
    Ua : ndarray (N, k)
        Orthonormal columns spanning subspace A.
    Ub : ndarray (N, k)
        Orthonormal columns spanning subspace B (same k).

    Returns
    -------
    overlap : float in [0, 1]
    """
    M = Ua.T @ Ub  # (k, k)
    k = M.shape[0]
    return float(np.sum(M ** 2) / k)


def directional_variance_capture(C_target_psd, U_source, eps=1e-12):
    """
    Fraction of target variance captured by source subspace basis.

    capture = tr(U^T C U) / tr(C)

    Measures how much of C_target's variance lies in the subspace
    spanned by U_source columns.

    Parameters
    ----------
    C_target_psd : ndarray (N, N)
        PSD covariance matrix whose variance we want to capture.
    U_source : ndarray (N, k)
        Orthonormal columns defining the source subspace.
    eps : float
        Avoids division by zero if tr(C) is tiny.

    Returns
    -------
    capture : float in [0, 1]
    """
    projected = U_source.T @ C_target_psd @ U_source  # (k, k)
    return float(np.trace(projected) / (np.trace(C_target_psd) + eps))
