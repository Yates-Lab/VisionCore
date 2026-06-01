r"""Synthetic ground-truth generator for the eye-position-matching methods study.

A single unified generator. Every result in this folder uses it; the assumption
violations (A1) and (A2) are parameter switches on top of one architecture.

Generative model
----------------
For neuron c, analysis time bin t (one frozen-stimulus draw of the field; see
the caveat in writeup §2.3), and absolute eye
position e in R^2,

    r_c(t, e) = mu_0 + M_c(e) * alpha(t) * s_t(e)        (deterministic rate)

  * s_t(.):  per-time-bin iid draw of a STATIONARY 2-D zero-mean Gaussian random
             field with covariance K(delta) = tau^2 * exp(-||delta||^2 / (2*ell^2)).
             This is the rate map at time bin t. Different bins are independent
             draws of the field. The field is (A2)-respecting by construction:
             at any fixed e, s_t(e) is N(0, tau^2) across time bins, so the
             across-time-bin distribution does not depend on e.

  * alpha(t): per-time-bin amplitude envelope. Default 1. Used in the Extension-1
              demo, where a decreasing alpha(t) co-varying with n_t (high
              amplitude in early, high-n_t bins) mirrors fixRSVP onset
              transients and drives the variable-n_t time-bin-weighting bias.

  * M_c(e):  per-cell deterministic SPATIAL MASK in [0, 1]. The (A2) switch.
             Physically: the fraction of the stimulus the cell sees at eye
             position e (e.g., when the RF leaves the windowed stimulus,
             M -> 0 and the rate collapses to baseline).
                 flat       M(e) = 1                              (A2 holds)
                 central    M(e) = exp(-||e||^2 / (2 ell_M^2))    (A2 broken; peak at center)
                 eccentric  M(e) = 1 - exp(-||e||^2 / (2 ell_M^2))(A2 broken; center suppressed)
                 linear     M(e) = 0.5*(1 + tanh(x / ell_M))      (A2 broken; smooth x-gradient)

  * mu_0:    baseline rate (counts/window), keeping r positive.

Spikes add observation noise that is INDEPENDENT across distinct trials (the
property the close-pair estimator exploits to cancel it):

    Y_c(trial, t) ~ Poisson( max( r_c(t, e_{trial,t}) + u_c(trial,t), 0 ) )

with an optional shared latent u ~ N(0, Sigma_u) per (trial, time-bin) giving a KNOWN
stimulus-independent ("noise") covariance across cells. Default Sigma_u = 0
(pure Poisson, true noise correlation 0, true Fano 1).

Why this works for both assumptions
-----------------------------------
(A2) violation. M(e) const => E_t[r^k(t,e)] independent of e. Any non-constant
mask makes the SECOND moment E_t[r^2(t,e)] = mu_0^2 + M(e)^2 * E_t[alpha^2]*tau^2
depend on e (the first moment stays mu_0). This matches McFarland's specific
statement that his estimator's correctness hinges on E[r^2(e,t)] being the
same under any distribution over e (their text around Eqs. M7-M10).

(A1) violation. n_trials_per_time_bin = staircase => different number of trials
per time bin. Combined with a non-constant alpha(t) envelope, this makes the
pair-count-weighted vs uniform-weighted average over time bins differ, biasing
the unmatched estimator.

Closed-form decomposition under any (eye distribution D, time-bin weighting w_t)
----------------------------------------------------------------------------
    Var_total^{D,w} = E_w[alpha^2] * tau^2 * E_D[M^2]                       (1)
    Var_PSTH^{D,w}  = E_w[alpha^2] * \iint M(e1) M(e2) K(e1-e2) D(e1) D(e2)  (2)
    Var_FEM^{D,w}   = Var_total - Var_PSTH
    1-alpha^{D,w}   = 1 - (2)/(E_w[alpha^2] * tau^2 * E_D[M^2])              (3)

Equation (3) is independent of E_w[alpha^2] -- the envelope cancels in the
ratio. So the GROUND TRUTH 1-alpha is invariant under time-bin weighting. The
Extension-1 bias is therefore a property of the ESTIMATOR (finite-sample
inconsistency under w-mismatch), not of the ground truth.

For Gaussian D = N(0, sigma^2 I) and Gaussian K, the integral in (2) closes
analytically for the `flat` mask:
    1-alpha^p   = 2*sigma^2 / (ell^2 + 2*sigma^2)
    1-alpha^p^2 = sigma^2   / (ell^2 + sigma^2)
For `central` (Gaussian) mask there is also a closed form (Appendix A.5 of the
writeup). For `eccentric` / `linear` we use Monte Carlo (4M-sample default;
sub-1e-3 sampling noise, well below test tolerances).
"""
from __future__ import annotations

import numpy as np

# Large MC sample size for closed-form-quality ground truth.
_GT_N = 4_000_000


# ---------------------------------------------------------------------------
# Spatial mask  M_c(e) -- the (A2) switch
# ---------------------------------------------------------------------------

PROFILE_KINDS = ("flat", "central", "eccentric", "linear")


def _default_ell_M(sigma_eye):
    """Default mask length scale; 0.6 * sigma_eye keeps the central/eccentric
    profile narrow enough that the close-pair (p^2) and full-p moments differ
    meaningfully -- which is the regime the (A2) demo lives in. Tunable via
    the make_session(ell_M=...) argument when needed."""
    return 0.6 * float(sigma_eye)


def profile_M(e, kind, sigma_eye, ell_M=None):
    """Per-cell spatial mask M(e) in [0, 1].

    Parameters
    ----------
    e : ndarray (..., 2)
        Eye positions in degrees.
    kind : str
        One of PROFILE_KINDS.
    sigma_eye : float
        Fixational spread (deg); sets the default mask length scale.
    ell_M : float or None
        Mask length scale (deg). If None, falls back to ``_default_ell_M``.

    Returns
    -------
    ndarray (...) in [0, 1].
    """
    if ell_M is None:
        ell_M = _default_ell_M(sigma_eye)
    x, y = e[..., 0], e[..., 1]
    r2 = x ** 2 + y ** 2
    if kind == "flat":
        return np.ones_like(x)
    if kind == "central":
        return np.exp(-r2 / (2.0 * ell_M ** 2))
    if kind == "eccentric":
        return 1.0 - np.exp(-r2 / (2.0 * ell_M ** 2))
    if kind == "linear":
        return 0.5 * (1.0 + np.tanh(x / ell_M))
    raise ValueError(f"unknown profile kind: {kind!r}")


# ---------------------------------------------------------------------------
# Closed-form / Monte-Carlo ground-truth decomposition
# ---------------------------------------------------------------------------

def _flat_one_minus_alpha_closed_form(sigma_eye, ell, distribution):
    """Closed form for the `flat` mask under Gaussian D and Gaussian K.

    For D = p:    1 - alpha = 2 sigma^2 / (ell^2 + 2 sigma^2)
    For D = p^2:  1 - alpha = sigma^2 / (ell^2 + sigma^2)
    """
    s2 = float(sigma_eye) ** 2
    L2 = float(ell) ** 2
    if distribution == "p":
        return 2.0 * s2 / (L2 + 2.0 * s2)
    if distribution == "p2":
        return s2 / (L2 + s2)
    raise ValueError(f"unknown distribution: {distribution!r}")


def _time_bin_amp_sq_mean(psth_envelope, time_bin_weights):
    """Mean of alpha(t)^2 under the given time-bin weighting (default uniform)."""
    if psth_envelope is None:
        return 1.0
    a = np.asarray(psth_envelope, dtype=float)
    if time_bin_weights is None:
        return float((a ** 2).mean())
    w = np.asarray(time_bin_weights, dtype=float)
    if w.shape != a.shape:
        raise ValueError(
            f"time_bin_weights shape {w.shape} != psth_envelope shape {a.shape}")
    w = w / w.sum()
    return float((w * a ** 2).sum())


def ground_truth(kind, sigma_eye, ell, tau=1.0, ell_M=None,
                 psth_envelope=None, time_bin_weights=None,
                 n=_GT_N, seed=12345):
    """Closed-form / MC ground truth for the LOTC decomposition of one cell.

    Returns a dict keyed by distribution 'p' (full fixational) and 'p^2'
    (close-pair / central), each with:
        var_psth, var_fem, var_total, one_minus_alpha.

    The ``one_minus_alpha`` ratio is invariant under (time_bin_weights, psth_envelope)
    -- those only scale var_psth, var_fem, var_total by E_w[alpha^2]. Pass them
    when the absolute variances matter (e.g. Extension-1 bias diagnosis on
    var_psth alone).
    """
    a2_mean = _time_bin_amp_sq_mean(psth_envelope, time_bin_weights)
    rng = np.random.default_rng(seed)
    tau2 = float(tau) ** 2

    out = {}
    for dist in ("p", "p2"):
        # Closed-form shortcut for the only kind that has a clean one.
        if kind == "flat":
            oma = _flat_one_minus_alpha_closed_form(sigma_eye, ell, dist)
            var_total = a2_mean * tau2  # E_D[M^2] = 1
            var_psth = (1.0 - oma) * var_total
            var_fem = var_total - var_psth
        else:
            sd = float(sigma_eye) if dist == "p" \
                else float(sigma_eye) / np.sqrt(2.0)
            e1 = rng.normal(0.0, sd, size=(n, 2))
            e2 = rng.normal(0.0, sd, size=(n, 2))
            M1 = profile_M(e1, kind, sigma_eye, ell_M)
            M2 = profile_M(e2, kind, sigma_eye, ell_M)
            E_M2 = float((M1 ** 2).mean())
            d2 = ((e1 - e2) ** 2).sum(-1)
            K12 = tau2 * np.exp(-d2 / (2.0 * float(ell) ** 2))
            I = float((M1 * M2 * K12).mean())  # the M-K-D integral
            var_total = a2_mean * tau2 * E_M2
            var_psth = a2_mean * I
            var_fem = var_total - var_psth
            oma = var_fem / var_total if var_total > 0 else float("nan")
        out[dist] = {
            "var_psth": var_psth,
            "var_fem": var_fem,
            "var_total": var_total,
            "one_minus_alpha": oma,
        }
    return out


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

def _resolve_n_t(n_trials_per_time_bin, n_trials, n_time_bins):
    """Build the per-time-bin trial-count array n_t of length n_time_bins."""
    if n_trials_per_time_bin is None:
        return np.full(n_time_bins, int(n_trials), dtype=int)
    if callable(n_trials_per_time_bin):
        return np.array([int(n_trials_per_time_bin(i)) for i in range(n_time_bins)],
                        dtype=int)
    if np.isscalar(n_trials_per_time_bin):
        return np.full(n_time_bins, int(n_trials_per_time_bin), dtype=int)
    arr = np.asarray(n_trials_per_time_bin, dtype=int).ravel()
    if arr.size != n_time_bins:
        raise ValueError(
            f"n_trials_per_time_bin length {arr.size} != n_time_bins {n_time_bins}")
    return arr


def _draw_field_at(eyes, ell, tau, rng, n_cells, jitter=1e-8):
    """Sample one time bin's stationary GP field at `n` observed eye positions for
    each of `n_cells` cells.

    Per-time-bin covariance Sigma[i,j] = tau^2 * exp(-||e_i - e_j||^2 / (2 ell^2));
    sample s = L @ z with z iid N(0, I_n) per cell, retry Cholesky with
    growing jitter on the (rare) near-singular case.
    """
    n = eyes.shape[0]
    if n == 0:
        return np.zeros((0, n_cells))
    diff = eyes[:, None, :] - eyes[None, :, :]
    Sigma = (float(tau) ** 2) * np.exp(-(diff ** 2).sum(-1)
                                       / (2.0 * float(ell) ** 2))
    eye_n = np.eye(n)
    j = jitter
    for _ in range(6):
        try:
            L = np.linalg.cholesky(Sigma + j * eye_n)
            break
        except np.linalg.LinAlgError:
            j *= 100.0
    else:
        raise np.linalg.LinAlgError(
            f"Cholesky failed up to jitter {j}; bad covariance configuration")
    z = rng.standard_normal((n, n_cells))
    return L @ z  # (n, n_cells)


def make_session(kinds, n_trials=600, n_time_bins=100, sigma_eye=0.15,
                 ell=None, tau=1.0, mu_0=6.0, ell_M=None,
                 noise_cov=None, seed=0,
                 return_rate=True, n_trials_per_time_bin=None,
                 psth_envelope=None):
    """Generate a unified multi-cell synthetic session.

    Parameters
    ----------
    kinds : sequence of str
        Per-cell mask kind (see PROFILE_KINDS).
    n_trials, n_time_bins : int
        Maximum trials per time bin and number of analysis time bins.
    sigma_eye : float
        Fixational spread (deg); eyes ~ N(0, sigma_eye^2 I), iid per (trial, time-bin).
    ell : float or None
        Field length scale (deg). None defaults to sigma_eye (puts 1-alpha^p
        ~ 2/3 -- non-trivial but not at the extremes of (0,1)).
    tau : float
        Field amplitude (sd); K(0) = tau^2.
    mu_0 : float
        Baseline rate (counts/window).
    ell_M : float or None
        Mask length scale (deg); see ``profile_M``.
    noise_cov : ndarray (n_cells, n_cells), optional
        Per-(trial, time-bin) shared latent covariance Sigma_u; None -> pure Poisson.
    seed : int
    return_rate : bool
        If True include the deterministic rate field (trials, time bins, cells).
    n_trials_per_time_bin : None, int, length-n_time_bins array, or callable(i)->int
        Per-time-bin trial count. Variable -> breaks (A1). Invalid (trial >= n_t)
        entries in spikes / rate / eye are filled with NaN; ``valid`` mask is
        True for the real trials.
    psth_envelope : None or length-n_time_bins array
        Per-time-bin amplitude envelope alpha(t). Default 1 (no envelope).

    Returns
    -------
    dict with:
        rate    : (n_rows, n_time_bins, n_cells) deterministic r(t,e)  [if return_rate]
        spikes  : (n_rows, n_time_bins, n_cells) Poisson observations
        eye     : (n_rows, n_time_bins, 2) eye positions (deg)
        valid   : (n_rows, n_time_bins) bool
        n_t     : (n_time_bins,) per-time-bin trial count
        truth   : list[dict] per-cell ground_truth(...) over 'p' and 'p2'
        noise_cov : Sigma_u used (or None)
        kinds   : the profile list
        sigma_eye, ell, tau, mu_0, ell_M, alpha
    """
    if ell is None:
        ell = float(sigma_eye)
    if ell_M is None:
        ell_M = _default_ell_M(sigma_eye)

    rng = np.random.default_rng(seed)
    n_cells = len(kinds)
    n_t = _resolve_n_t(n_trials_per_time_bin, n_trials, n_time_bins)
    n_rows = max(int(n_t.max()), 1)
    valid = (np.arange(n_rows)[:, None] < n_t[None, :])  # (n_rows, n_time_bins)

    if psth_envelope is None:
        alpha = np.ones(n_time_bins)
    else:
        alpha = np.asarray(psth_envelope, dtype=float).reshape(n_time_bins)

    eye = rng.normal(0.0, float(sigma_eye), size=(n_rows, n_time_bins, 2))

    # Field s_t(e_i) evaluated at observed eyes, independent across time bins,
    # shared spatial covariance across cells via the same Cholesky factor (each
    # cell gets its own N(0,I) draw).
    s = np.zeros((n_rows, n_time_bins, n_cells))
    for t in range(n_time_bins):
        nt = int(n_t[t])
        if nt < 1:
            continue
        s[:nt, t, :] = _draw_field_at(eye[:nt, t, :], ell, tau, rng, n_cells)

    # Per-cell mask M(e); then r = mu_0 + M(e) * alpha(t) * s_t(e).
    rate = np.empty((n_rows, n_time_bins, n_cells))
    for c, kind in enumerate(kinds):
        M = profile_M(eye, kind, sigma_eye, ell_M)              # (n_rows, n_time_bins)
        rate[:, :, c] = mu_0 + M * alpha[None, :] * s[:, :, c]
    rate = np.clip(rate, 1e-6, None)

    lam = rate
    if noise_cov is not None:
        u = rng.multivariate_normal(np.zeros(n_cells), noise_cov,
                                    size=(n_rows, n_time_bins))
        lam = np.clip(rate + u, 1e-6, None)
    spikes = rng.poisson(lam).astype(np.float64)

    if not valid.all():
        inv = ~valid
        spikes[inv] = np.nan
        rate[inv] = np.nan
        eye[inv] = np.nan

    truth = [ground_truth(kind, sigma_eye, ell=ell, tau=tau, ell_M=ell_M,
                          psth_envelope=psth_envelope)
             for kind in kinds]

    out = {
        "spikes": spikes,
        "eye": eye,
        "valid": valid,
        "n_t": n_t,
        "truth": truth,
        "noise_cov": noise_cov,
        "kinds": list(kinds),
        "sigma_eye": float(sigma_eye),
        "ell": float(ell),
        "tau": float(tau),
        "mu_0": float(mu_0),
        "ell_M": float(ell_M),
        "alpha": alpha,
    }
    if return_rate:
        out["rate"] = rate
    return out


# ---------------------------------------------------------------------------
# Trajectory-mode generator (§4.6: multi-bin eye trajectories)
# ---------------------------------------------------------------------------

def make_trajectory_session(kinds, n_samples_per_time_bin=30, n_time_bins=40,
                            t_window=5, sigma_eye=0.15, sigma_drift=0.0,
                            ell=None, tau=1.0, mu_0=6.0, ell_M=None,
                            psth_envelope=None, seed=0, return_rate=True):
    """Generate a trajectory-mode synthetic session for §4.6 validation.

    Each sample is one window of ``t_window`` contiguous bins of eye trajectory,
    assigned to one analysis time bin ``T_idx``. The trajectory is parameterised
    as a centroid plus i.i.d. per-bin drift:

        c_i           ~ N(0, sigma_eye^2 I)          (fixational position)
        xi_{i,t}      ~ N(0, sigma_drift^2 I)         (within-window drift, t = 0..t_window-1)
        e_{i,t}       = c_i + xi_{i,t}                (per-bin eye position)

    All ``n_samples_per_time_bin`` samples assigned to a given T_idx see the
    SAME field draw s_{T_idx, t}(.) at each within-window offset t (different
    samples sample the same stimulus at the same absolute time). Field draws
    across (T_idx, t) pairs are independent — t_window successive offsets each
    give a fresh field, matching the McFarland "phase" abstraction extended to
    a windowed analysis (writeup §2.3 caveat).

    The deterministic per-bin rate is

        r_c(T_idx, t, e_{i,t}) = mu_0 + M_c(e_{i,t}) * alpha(T_idx)
                                 * s_{T_idx, t}(e_{i,t})

    and the window count per sample is the rate summed across the t_window
    within-window offsets (matching the production ``extract_windows`` /
    ``SpikeCounts = S_raw.sum(dim=1)`` pattern in ``VisionCore/covariance.py``).

    In the flat-trajectory limit (``sigma_drift = 0``) per-bin positions
    coincide with the centroid, and the LOTC truth reduces to §4.4's
    centroid-distribution decomposition exactly: the t_window factor cancels
    in 1-alpha (it scales both Crate and Cpsth by the same t_window because
    fields at different t's are independent). For ``sigma_drift > 0`` the
    per-bin marginal density broadens to N(0, (sigma_eye^2 + sigma_drift^2) I);
    the user's pooled-per-bin KDE estimator targets that broadened marginal by
    construction, so the truth to compare against is

        truth_traj = ground_truth(kind, sqrt(sigma_eye^2 + sigma_drift^2),
                                  ell, ell_M=ell_M, psth_envelope=...)

    (returned as ``traj_truth`` in the output). The standard
    ``ground_truth(..., sigma_eye=sigma_eye, ...)`` is also returned as
    ``centroid_truth`` for the centroid-distribution reference.

    Parameters
    ----------
    kinds : sequence of str
        Per-cell mask kind (see PROFILE_KINDS).
    n_samples_per_time_bin : int
        Number of trajectory samples per analysis time bin T_idx.
    n_time_bins : int
        Number of analysis time bins (T_idx range).
    t_window : int
        Number of within-window offsets per sample (trajectory length).
    sigma_eye, sigma_drift : float
        Centroid spread and within-window drift (deg).
    ell, tau, mu_0, ell_M : as in make_session.
    psth_envelope : None or length-n_time_bins array
        Per-T_idx amplitude envelope alpha(T_idx). Default 1.
    seed : int
    return_rate : bool
        If True, include the per-sample per-cell window-summed rate
        (deterministic, no Poisson).

    Returns
    -------
    dict with:
        trajectories : (N, t_window, 2) per-sample per-bin eye positions
        centroids    : (N, 2) per-sample trajectory centroid
        counts       : (N, n_cells) per-sample window-summed deterministic rate
        T_idx        : (N,) analysis time bin index
        kinds, sigma_eye, sigma_drift, ell, tau, mu_0, ell_M, alpha, t_window
        centroid_truth : per-cell ground_truth(kind, sigma_eye, ...) — flat-limit reference
        traj_truth     : per-cell ground_truth(kind, sqrt(sigma_eye^2+sigma_drift^2), ...)
                         — per-bin marginal target (what the pooled-per-bin KDE estimator
                         targets; coincides with centroid_truth at sigma_drift = 0)
    """
    if ell is None:
        ell = float(sigma_eye)
    if ell_M is None:
        ell_M = _default_ell_M(sigma_eye)
    rng = np.random.default_rng(seed)
    n_cells = len(kinds)
    n_per = int(n_samples_per_time_bin)
    N = n_per * int(n_time_bins)

    if psth_envelope is None:
        alpha = np.ones(int(n_time_bins))
    else:
        alpha = np.asarray(psth_envelope, dtype=float).reshape(int(n_time_bins))

    centroids = rng.normal(0.0, float(sigma_eye), size=(N, 2))
    if float(sigma_drift) > 0.0:
        drift = rng.normal(0.0, float(sigma_drift), size=(N, int(t_window), 2))
    else:
        drift = np.zeros((N, int(t_window), 2))
    trajectories = centroids[:, None, :] + drift            # (N, t_window, 2)
    T_idx = np.repeat(np.arange(int(n_time_bins)), n_per)   # (N,)

    # Per-bin rate. For each (T_idx, t) draw ONE field at the eye positions of
    # the n_per samples that share that (T_idx, t); same field across the n_per
    # samples, independent across (T_idx, t) pairs (and across cells via
    # _draw_field_at's per-cell Cholesky right-side).
    rate_traj = np.empty((N, int(t_window), n_cells))
    for T in range(int(n_time_bins)):
        ix = np.where(T_idx == T)[0]
        if len(ix) == 0:
            continue
        for t in range(int(t_window)):
            eyes_t = trajectories[ix, t, :]                                  # (n_T, 2)
            s_field = _draw_field_at(eyes_t, ell, tau, rng, n_cells)         # (n_T, n_cells)
            for c, kind in enumerate(kinds):
                M_t = profile_M(eyes_t, kind, sigma_eye, ell_M)              # (n_T,)
                rate_traj[ix, t, c] = mu_0 + M_t * alpha[T] * s_field[:, c]
    rate_traj = np.clip(rate_traj, 1e-6, None)
    counts = rate_traj.sum(axis=1)                          # (N, n_cells)

    centroid_truth = [ground_truth(kind, sigma_eye, ell=ell, tau=tau,
                                   ell_M=ell_M, psth_envelope=psth_envelope)
                      for kind in kinds]
    sigma_traj = float(np.sqrt(float(sigma_eye) ** 2 + float(sigma_drift) ** 2))
    traj_truth = [ground_truth(kind, sigma_traj, ell=ell, tau=tau,
                               ell_M=ell_M, psth_envelope=psth_envelope)
                  for kind in kinds]

    out = {
        "trajectories": trajectories,
        "centroids": centroids,
        "counts": counts,
        "T_idx": T_idx,
        "kinds": list(kinds),
        "sigma_eye": float(sigma_eye),
        "sigma_drift": float(sigma_drift),
        "ell": float(ell),
        "tau": float(tau),
        "mu_0": float(mu_0),
        "ell_M": float(ell_M),
        "alpha": alpha,
        "t_window": int(t_window),
        "centroid_truth": centroid_truth,
        "traj_truth": traj_truth,
    }
    if return_rate:
        out["rate_traj"] = rate_traj
    return out


def _self_check():
    """Sanity print: closed-form vs MC vs direct sampling under (A2)."""
    sig = 0.15
    for kind in PROFILE_KINDS:
        gt = ground_truth(kind, sig, ell=sig, tau=1.0)
        print(f"{kind:10s}  1-alpha^p = {gt['p']['one_minus_alpha']:.4f}   "
              f"1-alpha^p2 = {gt['p2']['one_minus_alpha']:.4f}")
    # Trajectory-mode smoke check: a few samples, flat-limit truth match.
    sess = make_trajectory_session(["flat", "central"], n_samples_per_time_bin=20,
                                   n_time_bins=10, t_window=4,
                                   sigma_eye=sig, sigma_drift=0.0, seed=0)
    assert sess["counts"].shape == (200, 2)
    assert sess["trajectories"].shape == (200, 4, 2)
    # centroid_truth == traj_truth at sigma_drift = 0
    for c, k in enumerate(sess["kinds"]):
        a = sess["centroid_truth"][c]["p"]["one_minus_alpha"]
        b = sess["traj_truth"][c]["p"]["one_minus_alpha"]
        assert abs(a - b) < 1e-9, f"{k}: centroid {a} != traj {b} at sigma_drift=0"
    print("trajectory-mode smoke check: ok")


if __name__ == "__main__":
    _self_check()
