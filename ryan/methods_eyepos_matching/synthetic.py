r"""Synthetic ground-truth generator for the eye-position-matching methods study.

Everything in this folder validates against the rates and decompositions defined
here. The generative model is deliberately simple so the law-of-total-(co)variance
decomposition has a *closed form* we can sample to arbitrary precision, which lets
us assert that an estimator recovers known truth rather than merely "looks right".

Generative model
----------------
For neuron c, stimulus phase t (a frozen-stimulus time bin), and absolute eye
position e in R^2,

    r_c(t, e) = base_c + a_c * Pbar_c(t) + b_c * F_c(e)            (deterministic rate)

  * Pbar_c(t):  zero-mean phase-locked ("PSTH") drive -> the stimulus-locked part.
  * F_c(e):     the eye-position-sensitivity profile -> the FEM part. Its SPATIAL
                shape is the whole story: where in eye-space the rate is sensitive.

Spikes add observation noise that is INDEPENDENT across distinct trials (the
property the close-pair estimator exploits to cancel it):

    Y_c(trial, t) ~ Poisson( max(r_c(t, e_trial,t) + u_c(trial,t), 0) )

with an optional shared latent u ~ N(0, Sigma_u) per (trial, phase) giving a KNOWN
stimulus-independent ("noise") covariance across cells. Default Sigma_u = 0 (pure
Poisson, true noise correlation 0, true Fano 1) -- the cleanest regime in which any
spurious noise covariance / Fano inflation is *purely* the distribution mismatch.

Why the eye distribution matters (the crux)
-------------------------------------------
Eye positions are drawn e ~ p = N(0, sigma^2 I). Distinct-trial pairs with eye
distance below threshold (the close pairs the rate estimator conditions on)
concentrate where p is large: their density goes as p(e)^2. For a 2-D isotropic
Gaussian, p(e)^2 ∝ N(0, (sigma^2/2) I) EXACTLY -- a tighter, more central Gaussian.
So we can sample the "full" distribution p and the "close-pair" distribution p2
directly and exactly, and read off the closed-form decomposition under each.

Because F_c does not depend on t in this model, the law of total variance gives,
under any eye distribution D over e,

    Var_psth_c   = a_c^2 * Var_t[Pbar_c(t)]            (independent of D!)
    Var_fem_c(D) = b_c^2 * Var_{e~D}[F_c(e)]           (depends on D)
    one_minus_alpha_c(D) = Var_fem_c(D) / (Var_psth_c + Var_fem_c(D)).

The stimulus is "homogeneous" (McFarland's assumption) iff F_c is constant in e:
then Var_fem = 0 under every D and the choice of distribution is irrelevant. Any
non-constant F_c breaks that, and Var_fem(p) != Var_fem(p2) in general -- which is
exactly the bias the matched estimator must remove.
"""
from __future__ import annotations

import numpy as np

# A large, fixed Monte-Carlo sample size for closed-form-quality ground truth.
_GT_N = 4_000_000


# ---------------------------------------------------------------------------
# Eye-position sensitivity profiles  F_c(e)
# ---------------------------------------------------------------------------

def profile_F(e, kind, sigma_eye):
    """Eye-position sensitivity F(e) for a named spatial profile.

    Parameters
    ----------
    e : ndarray (..., 2)
        Eye positions in degrees.
    kind : str
        'flat'      -> homogeneous (no eye dependence); correction is a no-op.
        'central'   -> sensitivity peaked at the fixation center (small Gaussian).
        'eccentric' -> sensitivity grows with eccentricity (|e|^2).
        'linear'    -> sensitivity linear in x (spatially flat gradient).
    sigma_eye : float
        Fixational spread (deg); sets the natural length scale of 'central'.

    Returns
    -------
    ndarray (...)
    """
    x, y = e[..., 0], e[..., 1]
    r2 = x ** 2 + y ** 2
    if kind == "flat":
        return np.zeros_like(x)
    if kind == "central":
        return 2.2 * np.exp(-r2 / (2 * (0.6 * sigma_eye) ** 2))
    if kind == "eccentric":
        return 9.0 * r2
    if kind == "linear":
        return 3.2 * x
    raise ValueError(f"unknown profile kind: {kind!r}")


PROFILE_KINDS = ("flat", "central", "eccentric", "linear")


# ---------------------------------------------------------------------------
# Ground-truth decomposition (closed form via direct sampling of p and p^2)
# ---------------------------------------------------------------------------

def _fem_variance(kind, sigma_eye, distribution, n=_GT_N, seed=12345):
    """Var_{e~D}[F(e)] under D in {'p', 'p2'} for one profile.

    For e ~ N(0, sigma^2 I), the close-pair density p(e)^2 is exactly
    N(0, (sigma^2/2) I), so 'p2' is sampled from a Gaussian with sd sigma/sqrt(2).
    """
    rng = np.random.default_rng(seed)
    sd = sigma_eye if distribution == "p" else sigma_eye / np.sqrt(2.0)
    e = rng.normal(0.0, sd, size=(n, 2))
    F = profile_F(e, kind, sigma_eye)
    return float(F.var())


def ground_truth(kind, sigma_eye, psth_profile, a=1.0, b=1.0,
                 phase_weights=None):
    """Closed-form LOTC decomposition for one cell under both distributions.

    Returns a dict keyed by distribution 'p' (full fixational) and 'p2'
    (close-pair / central), each with var_psth, var_fem, var_total,
    one_minus_alpha. var_psth is shared (F independent of t).

    ``phase_weights``: optional length-n_phases array. When given, ``var_psth`` is
    the phase-weighted variance of ``psth_profile`` (used by the variable-n_t
    tests, which compare against the pair-count phase weighting that the matched
    estimator targets). Default = uniform across phases.
    """
    if phase_weights is None:
        var_psth = (a ** 2) * float(np.var(psth_profile))
    else:
        w = np.asarray(phase_weights, dtype=float)
        w = w / w.sum()
        mu = float((w * psth_profile).sum())
        var_psth = (a ** 2) * float((w * (psth_profile - mu) ** 2).sum())
    out = {}
    for dist in ("p", "p2"):
        var_fem = (b ** 2) * _fem_variance(kind, sigma_eye, dist)
        var_total = var_psth + var_fem
        oma = var_fem / var_total if var_total > 0 else np.nan
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
    """Build the per-phase trial-count array `n_t` of length n_phases."""
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


def make_session(kinds, n_trials=200, n_phases=100, sigma_eye=0.15,
                 a=1.0, b=1.0, base=6.0, noise_cov=None, seed=0,
                 return_rate=True, n_trials_per_phase=None,
                 psth_envelope=None):
    """Generate a synthetic multi-cell session with Poisson (optionally
    co-fluctuating) spikes and the known per-cell ground-truth decomposition.

    Parameters
    ----------
    kinds : sequence of str
        One profile name per cell (see PROFILE_KINDS).
    n_trials, n_phases : int
        Maximum repeats per phase and number of frozen-stimulus phases (time bins).
        With ``n_trials_per_phase=None`` every phase has exactly n_trials trials
        (constant n_t; the default and McFarland's assumed regime).
    sigma_eye : float
        Fixational spread (deg); eyes ~ N(0, sigma_eye^2 I), iid per (trial, phase).
    a, b : float
        Global PSTH and FEM gains applied to every cell (kept simple/shared).
    base : float
        Baseline rate (counts/window) keeping rates positive.
    noise_cov : ndarray (n_cells, n_cells), optional
        Stimulus-independent latent covariance Sigma_u shared across cells within
        each (trial, phase). None -> pure Poisson (no noise correlation).
    seed : int
    return_rate : bool
        If True include the deterministic rate field (trials, phases, cells).
    n_trials_per_phase : None, int, length-n_phases array, or callable(i)->int
        Variable per-phase trial count, used to break McFarland's uniform-trial
        assumption (Extension 1). Invalid (trial >= n_t[phase]) entries in
        ``spikes``, ``rate``, and ``eye`` are filled with NaN; the returned
        ``valid`` mask is True for the real trials. The storage shape stays
        (n_trials, n_phases, ...) with n_trials >= max(n_t). Default None ->
        constant n_t, all entries valid.
    psth_envelope : None or length-n_phases array
        Per-phase scaling of the PSTH amplitude. The default iid-Gaussian P_bar
        is statistically symmetric across phases, so pair-count and uniform
        phase weights both give Var[P_bar] ~ 1 in expectation -- the Extension-1
        bias is only detectable when the data has a structural correlation
        between PSTH amplitude and trial count, as in fixRSVP (onset transients
        in early, high-n_t phases). Pass an envelope to give the synthetic
        that structure.

    Returns
    -------
    dict with keys:
        rate   : (n_trials, n_phases, n_cells) deterministic r(t,e)  [if return_rate]
        spikes : (n_trials, n_phases, n_cells) Poisson observations
        eye    : (n_trials, n_phases, 2) eye positions (deg)
        valid  : (n_trials, n_phases) bool, True at real (trial, phase) entries
        n_t    : (n_phases,) per-phase trial count
        truth  : list[dict] per-cell ground_truth(...) over 'p' and 'p2'
        noise_cov : the Sigma_u used (or None)
        kinds  : the profile list
    """
    rng = np.random.default_rng(seed)
    n_cells = len(kinds)
    n_t = _resolve_n_t(n_trials_per_phase, n_trials, n_phases)
    n_rows = max(int(n_t.max()), 1)
    valid = (np.arange(n_rows)[:, None] < n_t[None, :])  # (n_rows, n_phases)

    # Zero-mean phase-locked PSTH drive, one profile per cell. An optional
    # per-phase envelope scales amplitude before the second mean-removal so
    # late phases stay near-zero (mimicking onset transients).
    psth = rng.normal(0.0, 1.0, size=(n_phases, n_cells))
    if psth_envelope is not None:
        env = np.asarray(psth_envelope, dtype=float).reshape(n_phases, 1)
        if env.shape[0] != n_phases:
            raise ValueError(
                f"psth_envelope length {env.shape[0]} != n_phases {n_phases}")
        psth = psth * env
    psth -= psth.mean(axis=0, keepdims=True)

    eye = rng.normal(0.0, sigma_eye, size=(n_rows, n_phases, 2))

    # Deterministic rate r_c(t, e) = base + a*Pbar_c(t) + b*F_c(e).
    rate = np.empty((n_rows, n_phases, n_cells))
    for c, kind in enumerate(kinds):
        F = profile_F(eye, kind, sigma_eye)             # (trials, phases)
        rate[:, :, c] = base + a * psth[None, :, c] + b * F
    rate = np.clip(rate, 1e-6, None)

    # Observation noise: optional shared latent + Poisson.
    lam = rate
    if noise_cov is not None:
        u = rng.multivariate_normal(np.zeros(n_cells), noise_cov,
                                    size=(n_rows, n_phases))
        lam = np.clip(rate + u, 1e-6, None)
    spikes = rng.poisson(lam).astype(np.float64)

    # NaN-out invalid entries so downstream finite-mask logic drops them.
    if not valid.all():
        inv = ~valid
        spikes[inv] = np.nan
        rate[inv] = np.nan
        eye[inv] = np.nan

    truth = [ground_truth(kind, sigma_eye, psth[:, c], a=a, b=b)
             for c, kind in enumerate(kinds)]

    out = {
        "spikes": spikes,
        "eye": eye,
        "valid": valid,
        "n_t": n_t,
        "truth": truth,
        "noise_cov": noise_cov,
        "kinds": list(kinds),
        "psth": psth,
        "sigma_eye": sigma_eye,
    }
    if return_rate:
        out["rate"] = rate
    return out


def _self_check():
    """Print the closed-form decomposition and confirm the trusted all-samples
    ANOVA recovers it: A on the native (p) eyes -> GT(p); A on p2-drawn eyes ->
    GT(p2). Run: ``uv run python synthetic.py`` from this folder.
    """
    from VisionCore.covariance import rate_variance_components
    sig = 0.15
    kinds = list(PROFILE_KINDS)
    Ap, Ap2 = [], []
    for s in range(6):
        sess = make_session(kinds, n_trials=200, n_phases=100, sigma_eye=sig, seed=s)
        psth, rate = sess["psth"], sess["rate"]
        Ap.append([rate_variance_components(rate[:, :, c], min_trials_per_phase=10)
                   ["one_minus_alpha"] for c in range(len(kinds))])
        rng = np.random.default_rng(1000 + s)
        eye2 = rng.normal(0.0, sig / np.sqrt(2), size=(200, 100, 2))
        r2 = np.stack([6.0 + psth[:, c][None, :] + profile_F(eye2, k, sig)
                       for c, k in enumerate(kinds)], axis=-1)
        Ap2.append([rate_variance_components(r2[:, :, c], min_trials_per_phase=10)
                    ["one_minus_alpha"] for c in range(len(kinds))])
    Ap, Ap2 = np.nanmean(Ap, 0), np.nanmean(Ap2, 0)
    gt = sess["truth"]
    print(f"{'profile':10s} {'GT(p)':>7s} {'A(p)':>7s}  {'GT(p2)':>7s} {'A(p2)':>7s}")
    for c, k in enumerate(kinds):
        print(f"{k:10s} {gt[c]['p']['one_minus_alpha']:7.3f} {Ap[c]:7.3f}  "
              f"{gt[c]['p2']['one_minus_alpha']:7.3f} {Ap2[c]:7.3f}")


if __name__ == "__main__":
    _self_check()
