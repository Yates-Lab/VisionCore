r"""Figure: Extension 1 -- two consistent w_t directions under variable n_t.

McFarland's cross-trial decomposition has implicit per-estimator across-bin
weightings that coincide only under constant n_t (writeup §1.5 table). To
restore term-by-term LOTC consistency under variable n_t, all three estimators
(Ctotal, Cpsth, Crate) must be pinned to a single w_t. Two directions are
consistent:

  uniform 1/T            -- McFarland's literal nested-bracket reading; each
                            fixation-aligned bin contributes equally.
  pair-count             -- pool-then-average over all close pairs; bin t
    n_t(n_t-1)/2            weighted by its close-pair count, inverse-variance
                            optimal on the close-pair pool.

The truth 1-alpha^p is invariant under w_t in the unified rate field (the
envelope cancels in the ratio), so both directions are unbiased; they differ
in finite-sample efficiency.

  A  variable n_t staircase + the two w_t weight curves.
  B  histograms of 1-alpha for both directions on flat-mask synthetic with
     staircase + envelope at ell=sigma. Both unbiased around the closed-form
     truth; pair-count is tighter than uniform.
  C  across-seed SD of 1-alpha vs ell/sigma sweep for both directions. The
     efficiency gap of uniform vs pair-count persists across spatial scales.

Run from this folder:  uv run python fig_time_bin_weighting.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session, ground_truth
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

NTR, NPH, SIG = 600, 100, 0.15
NT_LO, NT_HI = 15, 360

# Panel B: histograms at ell=sigma.
N_SEEDS_B = 25
N_CELLS_B = 4  # 4 'flat' cells per seed -> 100 datapoints per direction

# Panel C: continuous ell/sigma sweep.
ELL_RATIOS = np.array([0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.8, 4.0])
N_SEEDS_C = 12


def staircase_nt():
    return np.linspace(NT_HI, NT_LO, NPH).round().astype(int)


def run_histograms(seeds=range(N_SEEDS_B)):
    """Flat-mask + variable n_t + onset-transient envelope, ell=sigma.
    Returns dict[direction] -> flattened 1-alpha across (seed, cell)."""
    nt = staircase_nt()
    env = np.linspace(1.0, 0.05, NPH)
    out = {"uniform": [], "pair_count": []}
    for s in seeds:
        sess = make_session(["flat"] * N_CELLS_B, n_trials=NTR, n_time_bins=NPH,
                            sigma_eye=SIG, seed=s, n_trials_per_time_bin=nt,
                            psth_envelope=env)
        for pw in out:
            d = decompose(sess["rate"], sess["eye"], target="full",
                          density="gaussian", time_bin_weighting=pw)
            out[pw].append(d["one_minus_alpha"])
    for k in out:
        out[k] = np.concatenate(out[k])
    return out


def run_ell_sweep(ratios=ELL_RATIOS, seeds=range(N_SEEDS_C)):
    """Flat-mask + variable n_t + envelope across an ell/sigma sweep.
    Returns dict with ratio array and per-ratio mean/SD for each direction."""
    nt = staircase_nt()
    env = np.linspace(1.0, 0.05, NPH)
    means = {"uniform": [], "pair_count": []}
    sds = {"uniform": [], "pair_count": []}
    truths = []
    for r in ratios:
        ell = float(r) * SIG
        per_pw = {"uniform": [], "pair_count": []}
        for s in seeds:
            sess = make_session(["flat"], n_trials=NTR, n_time_bins=NPH,
                                sigma_eye=SIG, ell=ell, seed=s,
                                n_trials_per_time_bin=nt, psth_envelope=env)
            for pw in per_pw:
                d = decompose(sess["rate"], sess["eye"], target="full",
                              density="gaussian", time_bin_weighting=pw)
                per_pw[pw].append(float(d["one_minus_alpha"][0]))
        truths.append(ground_truth("flat", SIG, ell=ell)["p"]["one_minus_alpha"])
        for pw in per_pw:
            arr = np.array(per_pw[pw])
            means[pw].append(float(np.nanmean(arr)))
            sds[pw].append(float(np.nanstd(arr, ddof=1)))
    return {
        "ratios": np.asarray(ratios, float),
        "truths": np.asarray(truths, float),
        "mean": {k: np.asarray(v, float) for k, v in means.items()},
        "sd": {k: np.asarray(v, float) for k, v in sds.items()},
    }


def main():
    configure()
    nt = staircase_nt()
    pair_w = nt * (nt - 1) / 2.0
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # --- A: staircase n_t and the two w_t weight curves ---
    ax = axes[0]
    t = np.arange(NPH)
    ax.bar(t, nt, color="0.75", width=1.0, label=r"$n_t$ (trials/bin)")
    ax.set_xlabel("time-bin index t")
    ax.set_ylabel(r"$n_t$", color="0.3")
    ax.tick_params(axis="y", labelcolor="0.3")
    ax2 = ax.twinx()
    ax2.plot(t, pair_w / pair_w.sum(), color=C_FULL, lw=1.4,
             label=r"pair-count $w_t \propto n_t(n_t{-}1)/2$")
    ax2.plot(t, np.full(NPH, 1.0 / NPH), color=C_CLOSE, lw=1.4, ls="--",
             label=r"uniform $w_t = 1/T$")
    ax2.set_ylabel("time-bin weight (normalized)")
    ax2.spines["top"].set_visible(False)
    ax.set_title(rf"A  variable $n_t$  (range {NT_LO}–{NT_HI})")
    lns_a, lbl_a = ax.get_legend_handles_labels()
    lns_b, lbl_b = ax2.get_legend_handles_labels()
    ax2.legend(lns_a + lns_b, lbl_a + lbl_b, loc="upper right", fontsize=7)

    # --- B: histograms around closed-form truth on flat mask ---
    ax = axes[1]
    out = run_histograms()
    truth_flat = ground_truth("flat", SIG, ell=SIG)["p"]["one_minus_alpha"]
    bins = np.linspace(0.4, 0.95, 36)
    ax.hist(out["uniform"], bins=bins, color=C_CLOSE, alpha=0.55,
            label=f"uniform (sd={np.nanstd(out['uniform'], ddof=1):.3f})")
    ax.hist(out["pair_count"], bins=bins, color=C_FULL, alpha=0.55,
            label=f"pair-count (sd={np.nanstd(out['pair_count'], ddof=1):.3f})")
    ax.axvline(truth_flat, color=C_TRUTH, lw=1.0, ls="--",
               label=f"closed-form truth = {truth_flat:.3f}")
    ax.set_xlabel(r"$1-\hat\alpha$  (flat mask, $\ell=\sigma$)")
    ax.set_ylabel("count")
    ax.set_title("B  both directions unbiased; pair-count tighter")
    ax.legend(fontsize=7, loc="upper right")

    # --- C: across-seed SD vs ell/sigma sweep ---
    ax = axes[2]
    sweep = run_ell_sweep()
    rs = sweep["ratios"]
    sd_uni = sweep["sd"]["uniform"]
    sd_pair = sweep["sd"]["pair_count"]
    # Closed-form across-bin SD floor (§A.6, derived under constant n_t; included
    # as a reference scale, not a perfect predictor for variable n_t).
    truths = sweep["truths"]                          # 1 - alpha^p
    alpha_star = 1.0 - truths                         # alpha^p
    floor = alpha_star * np.sqrt(2.0 / (NPH - 1))
    ax.plot(rs, sd_uni, "o-", color=C_CLOSE, lw=1.4, ms=4,
            label="uniform")
    ax.plot(rs, sd_pair, "o-", color=C_FULL, lw=1.4, ms=4,
            label="pair-count")
    ax.plot(rs, floor, color="0.5", lw=1.0, ls=":",
            label=r"const-$n_t$ floor $\alpha^*\sqrt{2/(T-1)}$")
    ax.set_xlabel(r"$\ell / \sigma$")
    ax.set_ylabel(r"across-seed SD of $1-\hat\alpha$")
    ax.set_title(r"C  efficiency across spatial scale")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    save(fig, "fig_time_bin_weighting.png")


if __name__ == "__main__":
    main()
