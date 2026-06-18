r"""Figure: the distribution-matched estimator recovers ground truth, and the
Direction-1 vs Direction-2 tradeoff.

  A  recovery: target='full' recovers the p decomposition and target='central'
     recovers the p^2 decomposition (points on the identity line), across profiles.
  B  stability tradeoff: for an eccentric-modulated cell the variance lives in the
     periphery, where Direction 1's unbounded 1/p weights are noisy; the across-seed
     SD of 1-alpha grows as the close-pair threshold shrinks for 'full' but stays
     low for 'central' (bounded weights).
  C  the gap |1-alpha(full) - 1-alpha(central)| measures whether the cell's rate
     has spatial structure on the fixation scale (ell ~ sigma). Even a homogeneous
     (flat) cell under (A2) has a non-zero gap from the random field's spatial
     correlation length; non-homogeneous masks (central, eccentric) add to
     it. The closed-form (A2) baseline is shown as a reference line.

Run from this folder:  uv run python fig_correction.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

SIG = 0.15
PCOLOR = {"central": "#8e44ad", "eccentric": "#e67e22", "flat": "#7f8c8d"}


def recovery(seeds=range(6), N=400):
    kinds = ["central", "eccentric"] * 4
    full_gt, full_es, cent_gt, cent_es, col = [], [], [], [], []
    for s in seeds:
        sess = make_session(kinds, n_trials=N, n_time_bins=100, sigma_eye=SIG, seed=s)
        df = decompose(sess["rate"], sess["eye"], target="full", density="gaussian")
        dc = decompose(sess["rate"], sess["eye"], target="central", density="gaussian")
        for c, k in enumerate(kinds):
            full_gt.append(sess["truth"][c]["p"]["one_minus_alpha"])
            full_es.append(df["one_minus_alpha"][c])
            cent_gt.append(sess["truth"][c]["p2"]["one_minus_alpha"])
            cent_es.append(dc["one_minus_alpha"][c])
            col.append(k)
    return map(np.array, (full_gt, full_es, cent_gt, cent_es, np.array(col)))


def stability(thresholds=(0.08, 0.05, 0.03, 0.02, 0.012), seeds=range(8), N=500):
    sd_full, sd_cent = [], []
    for thr in thresholds:
        f, c = [], []
        for s in seeds:
            sess = make_session(["eccentric"], n_trials=N, n_time_bins=100,
                                sigma_eye=SIG, seed=s)
            f.append(decompose(sess["rate"], sess["eye"], target="full",
                               density="gaussian", threshold=thr)["one_minus_alpha"][0])
            c.append(decompose(sess["rate"], sess["eye"], target="central",
                               density="gaussian", threshold=thr)["one_minus_alpha"][0])
        sd_full.append(np.nanstd(f)); sd_cent.append(np.nanstd(c))
    return np.array(thresholds), np.array(sd_full), np.array(sd_cent)


def gap(seeds=range(6), N=400):
    kinds = ["flat", "eccentric", "central"]
    g = {k: [] for k in kinds}
    for s in seeds:
        sess = make_session(kinds, n_trials=N, n_time_bins=100, sigma_eye=SIG, seed=s)
        df = decompose(sess["rate"], sess["eye"], target="full", density="gaussian")
        dc = decompose(sess["rate"], sess["eye"], target="central", density="gaussian")
        for c, k in enumerate(kinds):
            g[k].append(abs(df["one_minus_alpha"][c] - dc["one_minus_alpha"][c]))
    return {k: np.nanmean(v) for k, v in g.items()}, {k: np.nanstd(v) for k, v in g.items()}


def main():
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # --- A: recovery ---
    fg, fe, cg, ce, col = recovery()
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C_TRUTH, lw=0.8, ls="--", zorder=0)
    ax.scatter(fg, fe, s=16, color=C_FULL, alpha=0.7,
               label=r"'full' vs GT$(p)$")
    ax.scatter(cg, ce, s=16, color=C_CLOSE, alpha=0.7, marker="s",
               label=r"'central' vs GT$(p^2)$")
    ax.set_xlabel(r"true $1-\alpha$"); ax.set_ylabel(r"estimated $1-\alpha$")
    ax.set_title("A  each target recovers its decomposition")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")

    # --- B: stability vs threshold ---
    thr, sf, sc = stability()
    ax = axes[1]
    ax.plot(thr, sf, "o-", color=C_FULL, label="Direction 1 ('full', $1/p$)")
    ax.plot(thr, sc, "s-", color=C_CLOSE, label="Direction 2 ('central', $\\propto p$)")
    ax.set_xlabel("close-pair threshold (deg)")
    ax.set_ylabel(r"across-seed SD of $1-\alpha$")
    ax.set_title("B  eccentric cell: Direction 2 is stabler")
    ax.invert_xaxis()
    ax.legend(fontsize=7)

    # --- C: fixation-scale spatial-structure gap ---
    gm, gs = gap()
    ax = axes[2]
    order = ["flat", "eccentric", "central"]
    xs = np.arange(len(order))
    ax.bar(xs, [gm[k] for k in order], yerr=[gs[k] for k in order],
           color=[PCOLOR[k] for k in order], alpha=0.8, capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_ylabel(r"$|\,(1-\alpha)_{\rm full} - (1-\alpha)_{\rm central}\,|$")
    ax.set_title("C  gap = fixation-scale spatial-structure measure")
    # Analytical (A2) baseline for `flat` at ell = sigma:
    # gap = sigma^2 ell^2 / [(ell^2 + 2 sigma^2)(ell^2 + sigma^2)] = 1/6.
    sig = SIG
    ell = sig
    gap_a2 = sig**2 * ell**2 / ((ell**2 + 2 * sig**2) * (ell**2 + sig**2))
    ax.axhline(gap_a2, color=C_TRUTH, lw=0.8, ls="--",
               label=rf"(A2) baseline (flat, $\ell=\sigma_e$): {gap_a2:.2f}")
    ax.axhline(0, color=C_TRUTH, lw=0.6)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    save(fig, "fig_correction.png")


if __name__ == "__main__":
    main()
