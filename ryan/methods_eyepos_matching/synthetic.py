r"""Synthetic ground-truth generator for the eye-position-matching methods study.

A single unified generator. Every result in this folder uses it; the assumption
violations (A1) and (A2) are parameter switches on top of one architecture.

Generative model
----------------
For neuron c, stimulus phase t (a frozen-stimulus time bin), and absolute eye
position e in R^2,

    r_c(t, e) = mu_0 + M_c(e) * alpha(t) * s_t(e)        (deterministic rate)

  * s_t(.):  per-phase iid draw of a STATIONARY 2-D zero-mean Gaussian random
             field with covariance K(delta) = tau^2 * exp(-||delta||^2 / (2*ell^2)).
             This is the rate map at phase t. Different phases are independent
             draws of the field. The field is (A2)-respecting by construction:
             at any fixed e, s_t(e) is N(0, tau^2) across phases, so the
             across-phase distribution does not depend on e.

  * alpha(t): per-phase amplitude envelope. Default 1. Used in the Extension-1
              demo, where a decreasing alpha(t) co-varying with n_t (high
              amplitude in early, high-n_t phases) mirrors fixRSVP onset
              transients and drives the variable-n_t phase-weighting bias.

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

with an optional shared latent u ~ N(0, Sigma_u) per (trial, phase) giving a KNOWN
stimulus-independent ("noise") covariance across cells. Default Sigma_u = 0
(pure Poisson, true noise correlation 0, true Fano 1).

Why this works for both assumptions
-----------------------------------
(A2) violation. M(e) const => E_t[r^k(t,e)] independent of e. Any non-constant
mask makes the SECOND moment E_t[r^2(t,e)] = mu_0^2 + M(e)^2 * E_t[alpha^2]*tau^2
depend on e (the first moment stays mu_0). This matches McFarland's specific
statement that his estimator's correctness hinges on E[r^2(e,t)] being the
same under any distribution over e (their text around Eqs. M7-M10).

(A1) violation. n_trials_per_phase = staircase => different number of trials
per phase. Combined with a non-constant alpha(t) envelope, this makes the
pair-count-weighted vs uniform-weighted average over phases differ, biasing
the unmatched estimator.

Closed-form decomposition under any (eye distribution D, phase weighting w_t)
----------------------------------------------------------------------------
    Var_total^{D,w} = E_w[alpha^2] * tau^2 * E_D[M^2]                       (1)
    Var_PSTH^{D,w}  = E_w[alpha^2] * \iint M(e1) M(e2) K(e1-e2) D(e1) D(e2)  (2)
    Var_FEM^{D,w}   = Var_total - Var_PSTH
    1-alpha^{D,w}   = 1 - (2)/(E_w[alpha^2] * tau^2 * E_D[M^2])              (3)

Equation (3) is independent of E_w[alpha^2] -- the envelope cancels in the
ratio. So the GROUND TRUTH 1-alpha is invariant under phase weighting. The
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


def _phase_amp_sq_mean(psth_envelope, phase_weights):
    """Mean of alpha(t)^2 under the given phase weighting (default uniform)."""
    if psth_envelope is None:
        return 1.0
    a = np.asarray(psth_envelope, dtype=float)
    if phase_weights is None:
        return float((a ** 2).mean())
    w = np.asarray(phase_weights, dtype=float)
    if w.shape != a.shape:
        raise ValueError(
            f"phase_weights shape {w.shape} != psth_envelope shape {a.shape}")
    w = w / w.sum()
    return float((w * a ** 2).sum())


def ground_truth(kind, sigma_eye, ell, tau=1.0, ell_M=None,
                 psth_envelope=None, phase_weights=None,
                 n=_GT_N, seed=12345):
    """Closed-form / MC ground truth for the LOTC decomposition of one cell.

    Returns a dict keyed by distribution 'p' (full fixational) and 'p^2'
    (close-pair / central), each with:
        var_psth, var_fem, var_total, one_minus_alpha.

    The ``one_minus_alpha`` ratio is invariant under (phase_weights, psth_envelope)
    -- those only scale var_psth, var_fem, var_total by E_w[alpha^2]. Pass them
    when the absolute variances matter (e.g. Extension-1 bias diagnosis on
    var_psth alone).
    """
    a2_mean = _phase_amp_sq_mean(psth_envelope, phase_weights)
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

def _resolve_n_t(n_trials_per_phase, n_trials, n_phases):
    """Build the per-phase trial-count array n_t of length n_phases."""
    if n_trials_per_phase is None:
        return np.full(n_phases, int(n_trials), dtype=int)
    if callable(n_trials_per_phase):
        return np.array([int(n_trials_per_phase(i)) for i in range(n_phases)],
                        dtype=int)
    if np.isscalar(n_trials_per_phase):
        return np.full(n_phases, int(n_trials_per_phase), dtype=int)
    arr = np.asarray(n_trials_per_phase, dtype=int).ravel()
    if arr.size != n_phases:
        raise ValueError(
            f"n_trials_per_phase length {arr.size} != n_phases {n_phases}")
    return arr


def _draw_field_at(eyes, ell, tau, rng, n_cells, jitter=1e-8):
    """Sample one phase's stationary GP field at `n` observed eye positions for
    each of `n_cells` cells.

    Per-phase covariance Sigma[i,j] = tau^2 * exp(-||e_i - e_j||^2 / (2 ell^2));
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


def make_session(kinds, n_trials=600, n_phases=100, sigma_eye=0.15,
                 ell=None, tau=1.0, mu_0=6.0, ell_M=None,
                 noise_cov=None, seed=0,
                 return_rate=True, n_trials_per_phase=None,
                 psth_envelope=None):
    """Generate a unified multi-cell synthetic session.

    Parameters
    ----------
    kinds : sequence of str
        Per-cell mask kind (see PROFILE_KINDS).
    n_trials, n_phases : int
        Maximum trials per phase and number of frozen-stimulus phases.
    sigma_eye : float
        Fixational spread (deg); eyes ~ N(0, sigma_eye^2 I), iid per (trial, phase).
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
        Per-(trial, phase) shared latent covariance Sigma_u; None -> pure Poisson.
    seed : int
    return_rate : bool
        If True include the deterministic rate field (trials, phases, cells).
    n_trials_per_phase : None, int, length-n_phases array, or callable(i)->int
        Per-phase trial count. Variable -> breaks (A1). Invalid (trial >= n_t)
        entries in spikes / rate / eye are filled with NaN; ``valid`` mask is
        True for the real trials.
    psth_envelope : None or length-n_phases array
        Per-phase amplitude envelope alpha(t). Default 1 (no envelope).

    Returns
    -------
    dict with:
        rate    : (n_rows, n_phases, n_cells) deterministic r(t,e)  [if return_rate]
        spikes  : (n_rows, n_phases, n_cells) Poisson observations
        eye     : (n_rows, n_phases, 2) eye positions (deg)
        valid   : (n_rows, n_phases) bool
        n_t     : (n_phases,) per-phase trial count
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
    n_t = _resolve_n_t(n_trials_per_phase, n_trials, n_phases)
    n_rows = max(int(n_t.max()), 1)
    valid = (np.arange(n_rows)[:, None] < n_t[None, :])  # (n_rows, n_phases)

    if psth_envelope is None:
        alpha = np.ones(n_phases)
    else:
        alpha = np.asarray(psth_envelope, dtype=float).reshape(n_phases)

    eye = rng.normal(0.0, float(sigma_eye), size=(n_rows, n_phases, 2))

    # Field s_t(e_i) evaluated at observed eyes, independent across phases,
    # shared spatial covariance across cells via the same Cholesky factor (each
    # cell gets its own N(0,I) draw).
    s = np.zeros((n_rows, n_phases, n_cells))
    for t in range(n_phases):
        nt = int(n_t[t])
        if nt < 1:
            continue
        s[:nt, t, :] = _draw_field_at(eye[:nt, t, :], ell, tau, rng, n_cells)

    # Per-cell mask M(e); then r = mu_0 + M(e) * alpha(t) * s_t(e).
    rate = np.empty((n_rows, n_phases, n_cells))
    for c, kind in enumerate(kinds):
        M = profile_M(eye, kind, sigma_eye, ell_M)              # (n_rows, n_phases)
        rate[:, :, c] = mu_0 + M * alpha[None, :] * s[:, :, c]
    rate = np.clip(rate, 1e-6, None)

    lam = rate
    if noise_cov is not None:
        u = rng.multivariate_normal(np.zeros(n_cells), noise_cov,
                                    size=(n_rows, n_phases))
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


def _self_check():
    """Sanity print: closed-form vs MC vs direct sampling under (A2)."""
    sig = 0.15
    for kind in PROFILE_KINDS:
        gt = ground_truth(kind, sig, ell=sig, tau=1.0)
        print(f"{kind:10s}  1-alpha^p = {gt['p']['one_minus_alpha']:.4f}   "
              f"1-alpha^p2 = {gt['p2']['one_minus_alpha']:.4f}")


if __name__ == "__main__":
    _self_check()
