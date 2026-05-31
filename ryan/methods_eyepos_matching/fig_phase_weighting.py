r"""Figure: Extension 1 -- consistent phase weighting under variable n_t.

McFarland's cross-trial decomposition assumes a uniform trial/phase structure: the
same number of trials in every phase. In fixRSVP and similar paradigms with
variable fixation durations, the number of trials per phase n_t drops across
phases. The close-pair rate estimator is intrinsically pair-count weighted across
phases (~n_t^2). If Cpsth uses a different across-phase weighting -- as in the
pre-fix pipeline, where Cpsth was uniform (1/T) -- the decomposition does not
hold term-by-term and the estimator deviates from the closed-form 1-alpha^p,
even on a homogeneous (flat) mask under (A2) where the closed-form truth is
analytic.

This figure validates Extension 1 against the unified-architecture ground truth
(stationary GP rate field + multiplicative mask; the (A1) violation is variable
n_t, the (A2) violation is a non-flat mask):

  A  the variable-n_t staircase across phases (lo=15, hi=360) and the matched
     pair-count phase weights ~ n_t(n_t-1)/2.
  B  on a homogeneous (flat-mask) synthetic with the staircase and an
     onset-transient envelope alpha(t), the unmatched (uniform) phase weighting
     biases 1-alpha away from the closed-form 1-alpha^p = 2 sigma^2 / (ell^2 +
     2 sigma^2); the matched (pair_count) weighting recovers it.
  C  on a non-homogeneous-mask synthetic (central, eccentric, linear), the
     matched weighting tracks the ground-truth 1-alpha^p on the identity line;
     the unmatched is systematically off.

Run from this folder:  uv run python fig_phase_weighting.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session, ground_truth
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

NTR, NPH, SIG = 600, 100, 0.15
NT_LO, NT_HI = 15, 360
PROFILE_COLOR = {"central": "#8e44ad", "eccentric": "#e67e22", "linear": "#16a085"}


def staircase_nt():
    return np.linspace(NT_HI, NT_LO, NPH).round().astype(int)


def run_homogeneous(seeds=range(8)):
    """Flat-mask + variable n_t + onset-transient envelope -> truth = analytic
    1-alpha^p (closed form for the flat mask). Deterministic rates isolate the
    weighting effect from Poisson noise."""
    nt = staircase_nt()
    env = np.linspace(1.0, 0.05, NPH)
    out = {"uniform": [], "pair_count": []}
    for s in seeds:
        sess = make_session(["flat"] * 4, n_trials=NTR, n_phases=NPH, sigma_eye=SIG,
                            seed=s, n_trials_per_phase=nt, psth_envelope=env)
        for pw in out:
            d = decompose(sess["rate"], sess["eye"], target="full",
                          density="gaussian", phase_weighting=pw)
            out[pw].append(d["one_minus_alpha"])
    for k in out:
        out[k] = np.concatenate(out[k])
    return out


def run_recovery(seeds=range(6)):
    """Non-homogeneous masks + variable n_t. The closed-form 1-alpha^p is
    invariant under phase weighting in the unified model (the envelope
    cancels in the ratio); the bias is entirely in the ESTIMATOR's choice
    of Cpsth phase weighting against Crate's intrinsic pair-count weighting."""
    nt = staircase_nt()
    env = np.linspace(1.0, 0.05, NPH)
    kinds = ["central", "eccentric", "linear"] * 4
    out = {"uniform": [], "pair_count": [], "truth": [], "kind": []}
    for s in seeds:
        sess = make_session(kinds, n_trials=NTR, n_phases=NPH, sigma_eye=SIG,
                            seed=s, n_trials_per_phase=nt, psth_envelope=env)
        truth = np.array([sess["truth"][c]["p"]["one_minus_alpha"]
                          for c in range(len(kinds))])
        out["truth"].append(truth)
        out["kind"].append(np.array(kinds))
        for pw in ("uniform", "pair_count"):
            d = decompose(sess["rate"], sess["eye"], target="full",
                          density="gaussian", phase_weighting=pw)
            out[pw].append(d["one_minus_alpha"])
    for k in out:
        out[k] = np.concatenate(out[k])
    return out


def main():
    configure()
    nt = staircase_nt()
    pair_w = nt * (nt - 1) / 2.0
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # --- A: staircase n_t and pair-count weights ---
    ax = axes[0]
    t = np.arange(NPH)
    ax.bar(t, nt, color="0.75", width=1.0, label=r"$n_t$ (trials/phase)")
    ax.set_xlabel("phase index t")
    ax.set_ylabel(r"$n_t$", color="0.3")
    ax.tick_params(axis="y", labelcolor="0.3")
    ax2 = ax.twinx()
    ax2.plot(t, pair_w / pair_w.sum(), color=C_FULL, lw=1.4,
             label=r"pair-count weight $w_t \propto n_t(n_t{-}1)/2$")
    ax2.plot(t, np.full(NPH, 1.0 / NPH), color=C_CLOSE, lw=1.4, ls="--",
             label=r"uniform weight $w_t = 1/T$")
    ax2.set_ylabel("phase weight (normalized)")
    ax2.spines["top"].set_visible(False)
    ax.set_title(rf"A  variable $n_t$  (range {NT_LO}–{NT_HI})")
    lns_a, lbl_a = ax.get_legend_handles_labels()
    lns_b, lbl_b = ax2.get_legend_handles_labels()
    ax2.legend(lns_a + lns_b, lbl_a + lbl_b, loc="upper right", fontsize=7)

    # --- B: homogeneous mask -- closed-form truth; unmatched biased away ---
    ax = axes[1]
    out = run_homogeneous()
    truth_flat = ground_truth("flat", SIG, ell=SIG)["p"]["one_minus_alpha"]
    bins = np.linspace(0.2, 1.0, 41)
    ax.hist(out["uniform"], bins=bins, color=C_CLOSE, alpha=0.55,
            label=f"uniform (med={np.nanmedian(out['uniform']):.2f})")
    ax.hist(out["pair_count"], bins=bins, color=C_FULL, alpha=0.55,
            label=f"pair_count (med={np.nanmedian(out['pair_count']):.2f})")
    ax.axvline(truth_flat, color=C_TRUTH, lw=1.0, ls="--",
               label=f"truth = {truth_flat:.2f}")
    ax.set_xlabel(r"$1-\alpha$  (flat mask, truth = closed form)")
    ax.set_ylabel("cell count")
    ax.set_title("B  homogeneous mask: uniform biased away from truth")
    ax.legend(fontsize=7, loc="upper right")

    # --- C: non-homogeneous masks -- matched on identity, unmatched off ---
    ax = axes[2]
    out = run_recovery()
    ax.plot([0, 1], [0, 1], color=C_TRUTH, lw=0.8, ls="--", zorder=0)
    for kind in ("central", "eccentric", "linear"):
        m = out["kind"] == kind
        ax.scatter(out["truth"][m], out["uniform"][m], s=20,
                   color=PROFILE_COLOR[kind], marker="x", alpha=0.7,
                   label=f"{kind} (uniform)")
        ax.scatter(out["truth"][m], out["pair_count"][m], s=18,
                   facecolors="none", edgecolors=PROFILE_COLOR[kind], alpha=0.9)
    ax.set_xlabel(r"true $1-\alpha^p$  (closed form)")
    ax.set_ylabel(r"estimated $1-\alpha$")
    ax.set_title("C  non-homogeneous masks: matched (○) on truth, uniform (×) off")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    save(fig, "fig_phase_weighting.png")


if __name__ == "__main__":
    main()
