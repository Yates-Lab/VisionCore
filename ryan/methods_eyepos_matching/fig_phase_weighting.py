r"""Figure: Extension 1 -- consistent phase weighting under variable n_t.

McFarland's cross-trial decomposition assumes a uniform trial/phase structure: the
same number of trials in every phase. In fixRSVP and similar paradigms with
variable fixation durations, the number of trials per phase n_t drops across
phases. The close-pair rate estimator is intrinsically pair-count weighted across
phases (~n_t^2). If Cpsth uses a different across-phase weighting -- as in the
pre-fix pipeline, where Cpsth was uniform (1/T) -- the decomposition does not
hold term-by-term and 1-alpha is biased even on a homogeneous (flat) stimulus
where the truth is exactly 0.

This figure validates Extension 1 against synthetic ground truth using the
extended synthetic generator (n_trials_per_phase + psth_envelope):

  A  the variable-n_t staircase across phases (lo=15, hi=360) and the matched
     pair-count phase weights ~ n_t(n_t-1)/2.
  B  on a homogeneous (flat) synthetic with the same staircase and an
     onset-transient PSTH envelope, the unmatched (uniform) phase weighting
     biases 1-alpha well above 0 (truth); the matched (pair_count) weighting
     recovers 0.
  C  on a structured-profile synthetic (central, eccentric, linear), the matched
     weighting tracks the pair-count-weighted ground truth on the identity line;
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
    """Flat profile + variable n_t + onset-transient envelope -> truth 1-alpha = 0.
    Deterministic rates isolate the weighting effect from Poisson noise."""
    nt = staircase_nt()
    env = np.linspace(1.0, 0.05, NPH)
    out = {"uniform": [], "pair_count": []}
    for s in seeds:
        sess = make_session(["flat"] * 4, n_trials=NTR, n_phases=NPH, sigma_eye=SIG,
                            seed=s, n_trials_per_phase=nt, psth_envelope=env)
        for pw in out:
            d = decompose(sess["rate"], sess["eye"], target="naive",
                          density="gaussian", phase_weighting=pw)
            out[pw].append(d["one_minus_alpha"])
    for k in out:
        out[k] = np.concatenate(out[k])
    return out


def run_recovery(seeds=range(6)):
    """Mixed profiles + variable n_t -> compare estimate vs pair-count-weighted
    ground truth on the actual viewing distribution p (target='full')."""
    nt = staircase_nt()
    pair_w = nt * (nt - 1) / 2.0
    env = np.linspace(1.0, 0.05, NPH)
    kinds = ["central", "eccentric", "linear"] * 4
    out = {"uniform": [], "pair_count": [], "truth": [], "kind": []}
    for s in seeds:
        sess = make_session(kinds, n_trials=NTR, n_phases=NPH, sigma_eye=SIG,
                            seed=s, n_trials_per_phase=nt, psth_envelope=env)
        truth = np.array([
            ground_truth(k, SIG, sess["psth"][:, c], phase_weights=pair_w)
            ["p"]["one_minus_alpha"] for c, k in enumerate(kinds)])
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

    # --- B: homogeneous stimulus -- unmatched biased above 0, matched on 0 ---
    ax = axes[1]
    out = run_homogeneous()
    bins = np.linspace(-0.15, 0.7, 41)
    ax.hist(out["uniform"], bins=bins, color=C_CLOSE, alpha=0.55,
            label=f"uniform (med={np.nanmedian(out['uniform']):.2f})")
    ax.hist(out["pair_count"], bins=bins, color=C_FULL, alpha=0.55,
            label=f"pair_count (med={np.nanmedian(out['pair_count']):.2f})")
    ax.axvline(0, color=C_TRUTH, lw=1.0, ls="--", label="truth = 0")
    ax.set_xlabel(r"$1-\alpha$  (flat profile, truth = 0)")
    ax.set_ylabel("cell count")
    ax.set_title("B  homogeneous stimulus: uniform biased away from 0")
    ax.legend(fontsize=7, loc="upper right")

    # --- C: structured profiles -- matched on identity, unmatched off ---
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
    ax.set_xlabel(r"true $1-\alpha$  (pair-count-weighted, target $p$)")
    ax.set_ylabel(r"estimated $1-\alpha$")
    ax.set_title("C  structured profiles: matched (○) on truth, uniform (×) off")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    save(fig, "fig_phase_weighting.png")


if __name__ == "__main__":
    main()
