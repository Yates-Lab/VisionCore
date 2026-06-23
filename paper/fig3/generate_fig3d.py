r"""Figure 3 panel D: does the digital twin recapitulate each cell's 1-alpha?

Usage:
    uv run declan/fig3/generate_fig3d.py

Scientific question
-------------------
1-alpha is the fraction of a cell's total *rate* variance that is driven by
fixational eye movements (FEM), as opposed to the stimulus-locked (PSTH)
fraction alpha. It is measured empirically from the neural data in Figure 2.
This panel asks whether the digital twin reproduces each cell's 1-alpha *in its
own predictions* -- i.e. whether the model gets the proportion of
eye-movement-driven variance right, not merely the mean response. The twin has
no extraretinal channel for eye movements (the bottom-row ablation, panels
H-J, shows zeroing the behavior input changes nothing), so any FEM-driven
variance it produces arrives through the *retinal* consequence of drift: the
stimulus shifting on the retina. Matching 1-alpha is therefore a strong test of
that retinal mechanism.

The decomposition (Law of Total Variance)
-----------------------------------------
For one neuron, let the rate depend on stimulus phase t (time bin within the
frozen RSVP sequence, shared across repeats) and the within-trial eye
trajectory e. Conditioning the rate on phase t splits its variance into a
stimulus-locked part and an eye-movement part:

    Var_{t,e}(r) =   Var_t( E_e[r | t] )   +   E_t( Var_e[r | t] )
                   \_______ PSTH ________/     \_______ FEM ______/
                          (= Cpsth)                  (= Cfem)

    alpha     = Cpsth / Crate            (stimulus-locked fraction)
    1 - alpha = Cfem  / Crate            (FEM fraction)        Crate = Cpsth + Cfem

At a fixed stimulus phase the only thing varying across repeats is the eye, so
the within-phase term is the FEM-driven variance.

Why the model needs none of the empirical machinery
----------------------------------------------------
For real neurons, spike counts S are *noisy* observations of r(t, e). Every
heavy step in VisionCore.covariance exists to strip Poisson observation noise
out of S so the latent rate variance can be recovered:
  * the eye-distance binning + intercept fit extrapolate distinct-trial
    cross-products to delta_e -> 0, giving the Poisson-free rate variance Crate
    (fig2 uses intercept_mode='below_threshold', 0.05 deg);
  * bagged_split_half_psth_covariance debiases Cpsth against the same noise.

The digital twin is *deterministic*: given its inputs it emits the rate
rhat[i, t] directly, with no observation noise, evaluated at each trial's actual
eye trajectory. There is no Poisson term to remove, the eye-distance apparatus
is unnecessary, and the decomposition collapses to a textbook one-way
random-effects ANOVA of rhat grouped by stimulus phase t.

Estimator (analytic random-effects ANOVA -- primary)
----------------------------------------------------
Group rhat by phase t; phase t has n_t valid trials, T kept phases,
N = sum_t n_t. With phase mean rhat_bar(t) and grand mean rhat_bar:

    SS_within  = sum_t sum_i (rhat[i,t] - rhat_bar(t))^2 ,   df = N - T
    SS_between = sum_t n_t (rhat_bar(t) - rhat_bar)^2     ,   df = T - 1
    MS_within  = SS_within / (N - T) ,   MS_between = SS_between / (T - 1)

The mean squares have exact expectations (method of moments, no Gaussian
assumption) E[MS_within] = sigma2_W, E[MS_between] = sigma2_W + n0 * sigma2_B
with the unbalanced effective group size n0 = (N - sum_t n_t^2 / N) / (T - 1)
(= n when balanced). Hence the unbiased components

    sigma2_within (FEM)  = MS_within
    sigma2_between (PSTH) = max((MS_between - MS_within) / n0, 0)
    1 - alpha            = clip(sigma2_within / (sigma2_within + sigma2_between), 0, 1)

Even with exact rates, MS_between must be debiased: with finite n_t the naive
between-phase variance Var_t(rhat_bar(t)) is inflated by E_t[Var_e(r|t)/n_t]
(each phase mean averages only n_t eye draws). The subtraction removes exactly
that term; skipping it would deflate 1-alpha by an alpha-dependent amount --
worst precisely for the FEM-dominated cells this panel is about.

Why analytic over split-half, and the cross-check
--------------------------------------------------
The empirical PSTH variance uses bagged split-half. Both estimators target the
same sigma2_between, but:
  * Analytic ANOVA  -- closed-form, deterministic (seed-free), uses all data
    once (minimum-variance under balance/Gaussianity). Cleanest "proper" choice
    now that we hold the true rates. Caveat: its exact unbiasedness assumes
    balanced/homoscedastic groups; under unbalanced n_t with phase-varying
    within-variance the MS_between - MS_within subtraction leaves a small
    residual bias.
  * Split-half     -- Monte-Carlo (seed-dependent), less efficient (each split
    discards half the trials, recovered by bagging), but cancels the per-phase
    sampling noise across *disjoint* halves and so is robust to that
    heteroscedasticity. It is what the empirical side literally uses.
We take analytic as primary and report split-half agreement (psth_variance_splithalf)
as an assumption-light cross-check; large divergence would flag real imbalance.

Conventions and invariances
---------------------------
  * Convention vs empirical: the empirical side estimates Crate directly and
    takes FEM as the residual Crate - Cpsth; here both components are estimated
    directly and Crate is their sum. Both target the same population ratio.
  * 1-alpha is exactly invariant to the upstream affine rescale rhat -> a*rhat+b
    (rescale_rhat): a shift leaves variances unchanged, a scale multiplies both
    components by a^2 and cancels in the ratio. So rescaled vs raw rhat agree.
  * Window/units: empirical alpha is mats[0] (1-bin counting window, 1/120 s);
    rhat_used is per-bin at the same dt, so windows match, and the dimensionless
    ratio is unaffected by rate-vs-count units.

Matching conditions
-------------------
Good cells (data['good'], ccmax > 0.85) with finite empirical alpha; per-neuron
valid mask = (dfs_used != 0); phases grouped on the rhat_used time axis (the
psth_inds alignment, same notion as the empirical T_idx); min_trials_per_phase
= 10 to mirror the empirical min_trials_per_time; 1-alpha clipped to [0, 1] as
on the empirical side. The empirical T_idx drops a short per-segment history
prefix and uses fixation *segments* rather than psth-phase, so the two samplings
are near-identical but not bit-identical -- the per-cell ratio absorbs this.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress

from VisionCore.covariance import rate_variance_components, psth_variance_splithalf
from _fig3_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig3_data,
)

MIN_TRIALS_PER_PHASE = 10   # mirror empirical min_trials_per_time
SPLITHALF_N_BOOT = 50       # bags for the split-half cross-check
SPLITHALF_SEED = 0


def compute_model_one_minus_alpha(data, min_trials_per_phase=MIN_TRIALS_PER_PHASE,
                                  n_boot=SPLITHALF_N_BOOT, seed=SPLITHALF_SEED):
    """Per-cell model 1-alpha (analytic) vs empirical 1-alpha, aligned by neuron.

    Iterates over good cells with finite empirical alpha, decomposes each cell's
    cached per-trial model rate rhat_used via the random-effects ANOVA, and
    pairs the result with the empirical 1-alpha = 1 - data['alpha']. Also returns
    the split-half model 1-alpha (reusing the analytic within-component) as a
    cross-check. Returns a dict of equal-length arrays.
    """
    session_results = data["session_results"]
    valid_indices = data["valid_indices"]
    ats = data["all_trace_neuron_session"]
    alpha = data["alpha"]
    good = data["good"]
    subjects = data["subjects"]

    emp, mod, mod_sh, subj_out = [], [], [], []
    for k in range(len(alpha)):
        if not good[k] or not np.isfinite(alpha[k]):
            continue
        si, ni = ats[valid_indices[k]]
        sr = session_results[si]
        rhat = sr["rhat_used"][:, :, ni]      # (trials, time) for this neuron
        valid = sr["dfs_used"][:, :, ni] != 0

        out = rate_variance_components(
            rhat, valid=valid, min_trials_per_phase=min_trials_per_phase
        )
        m = out["one_minus_alpha"]
        if not np.isfinite(m):
            continue

        sh_between = psth_variance_splithalf(
            rhat, valid=valid, min_trials_per_phase=min_trials_per_phase,
            n_boot=n_boot, seed=seed,
        )
        sw = out["sigma2_within"]
        tot_sh = sw + sh_between
        m_sh = float(np.clip(sw / tot_sh, 0.0, 1.0)) if tot_sh > 0 else np.nan

        emp.append(1.0 - alpha[k])
        mod.append(m)
        mod_sh.append(m_sh)
        subj_out.append(subjects[k])

    return {
        "emp": np.asarray(emp),
        "model": np.asarray(mod),
        "model_splithalf": np.asarray(mod_sh),
        "subjects": np.asarray(subj_out),
    }


def plot_panel_d(ax=None, data=None, legend_fontsize=8, print_stats=True):
    """Scatter model 1-alpha vs empirical 1-alpha per cell. Returns (fig, ax)."""
    if data is None:
        data = load_fig3_data()
    comp = compute_model_one_minus_alpha(data)
    emp, mod, mod_sh, subjects = (
        comp["emp"], comp["model"], comp["model_splithalf"], comp["subjects"]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    for subj in SUBJECTS:
        mask = subjects == subj
        if not mask.any():
            continue
        x, y = emp[mask], mod[mask]
        rho = spearmanr(x, y).correlation
        ax.scatter(x, y, s=5, alpha=0.5, color=SUBJECT_COLORS[subj],
                   label=f"{subj}: ρ={rho:.2f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Empirical $1-\alpha$")
    ax.set_ylabel(r"Model $1-\alpha$")
    ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if print_stats:
        for subj in SUBJECTS + ["All"]:
            mask = np.ones(len(emp), dtype=bool) if subj == "All" else subjects == subj
            x, y = emp[mask], mod[mask]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 3:
                print(f"Panel D — {subj}: N={len(x)} (too few)")
                continue
            rho = spearmanr(x, y).correlation
            r = pearsonr(x, y)[0]
            slope = linregress(x, y).slope
            mse = float(np.median(y - x))  # signed: model - empirical
            print(f"Panel D — {subj} (N={len(x)}): "
                  f"Spearman ρ={rho:.3f}, Pearson r={r:.3f}, "
                  f"slope={slope:.3f}, median(model-emp)={mse:+.3f}, "
                  f"median emp={np.median(x):.3f}, median model={np.median(y):.3f}")
        # split-half cross-check on the analytic estimator
        ok = np.isfinite(mod) & np.isfinite(mod_sh)
        if ok.sum() >= 3:
            d = mod[ok] - mod_sh[ok]
            r_sh = pearsonr(mod[ok], mod_sh[ok])[0]
            print(f"Panel D — analytic vs split-half (N={ok.sum()}): "
                  f"Pearson r={r_sh:.4f}, median|diff|={np.median(np.abs(d)):.4f}, "
                  f"max|diff|={np.max(np.abs(d)):.4f}")

    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_d()
    fig.tight_layout()
    out = FIG_DIR / "panel_d_one_minus_alpha.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
