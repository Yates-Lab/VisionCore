r"""Figure: the matched estimators recover the analytical 1-alpha under each
target distribution, on the non-homogeneous (central / eccentric) masks.

This is fig_sanity_check panel A generalised to the non-homogeneous regime,
where p and p^2 give DIFFERENT analytical truths. For each mask we sweep
ell/sigma_e and draw the two closed-form curves 1-alpha^p (solid) and
1-alpha^{p^2} (dashed); overlaid are the empirical estimators evaluated at the
same ell/sigma_e locations (mean +- sd across seeds):

  * Direction 1 ('full',    p-weighted)  -> tracks 1-alpha^p
  * Direction 2 ('central', p^2-weighted)-> tracks 1-alpha^{p^2}
  * naive (close-pair p^2 in the denominator, p elsewhere) -> biased off both,
    upward for central, downward (clipping toward 0) for eccentric.

Run from this folder:  uv run python fig_recovery.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session, ground_truth
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE

SIG = 0.15
SIGMA_M = 1.0 * SIG          # mask comparable to fixation scale: near the
                             # maximum p-vs-p^2 separation (writeup Fig. 0a) and
                             # the most representative regime
NTR = 800                    # close pairs ~ N^2: tames the naive Crate variance,
                             # which is the dominant seed-to-seed noise here
NPH = 100
THR = 0.05
C_NAIVE = "#8e44ad"          # distinct from the p / p^2 distribution colors
RATIOS = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _analytical(kind, ratios, sigma_M=SIGMA_M):
    """Closed-form 1-alpha^p, 1-alpha^{p^2} across an ell/sigma_e sweep."""
    p, p2 = [], []
    for r in ratios:
        gt = ground_truth(kind, sigma_eye=SIG, ell=r * SIG, sigma_M=sigma_M)
        p.append(gt["p"]["one_minus_alpha"])
        p2.append(gt["p2"]["one_minus_alpha"])
    return np.array(p), np.array(p2)


def _empirical(kind, ratios, seeds=range(6), sigma_M=SIGMA_M):
    """Empirical 1-alpha for the three targets at each ell/sigma_e."""
    out = {t: ([], []) for t in ("full", "central", "naive")}
    for r in ratios:
        ell = r * SIG
        vals = {t: [] for t in out}
        for s in seeds:
            sess = make_session([kind], n_trials=NTR, n_time_bins=NPH,
                                sigma_eye=SIG, ell=ell, sigma_M=sigma_M, seed=s)
            for t in out:
                # synthetic Gaussian-eye closed-form recovery validates the p^2
                # construction (p_pair = p^2 holds here); pin to 'squared'.
                d = decompose(sess["rate"], sess["eye"], target=t,
                              density="gaussian", threshold=THR,
                              closepair_density="squared")
                vals[t].append(float(d["one_minus_alpha"][0]))
        for t in out:
            out[t][0].append(np.nanmean(vals[t]))
            out[t][1].append(np.nanstd(vals[t]))
    return {t: (np.array(m), np.array(sd)) for t, (m, sd) in out.items()}


def panel(ax, kind, title):
    r_dense = np.linspace(RATIOS.min() * 0.85, RATIOS.max() * 1.1, 200)
    p_d, p2_d = _analytical(kind, r_dense)
    ax.plot(r_dense, p_d, color=C_FULL, lw=1.6,
            label=r"analytical $1-\alpha^{p}$")
    ax.plot(r_dense, p2_d, color=C_CLOSE, lw=1.6, ls="--",
            label=r"analytical $1-\alpha^{p^2}$")

    emp = _empirical(kind, RATIOS)
    ax.errorbar(RATIOS, *emp["full"], fmt="o", color=C_FULL, capsize=2.5,
                ms=4, lw=1, label=r"Direction 1 ('full', $p$)")
    ax.errorbar(RATIOS, *emp["central"], fmt="s", color=C_CLOSE, capsize=2.5,
                ms=4, lw=1, label=r"Direction 2 ('central', $p^2$)")
    ax.errorbar(RATIOS, *emp["naive"], fmt="x", color=C_NAIVE, capsize=2.5,
                ms=5, lw=1, label="naive")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell\,/\,\sigma_e$")
    ax.set_ylabel(r"$1-\alpha$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=6.5, loc="upper right")


def main():
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    panel(axes[0], "central", r"A  central mask  ($\sigma_M=\sigma_e$)")
    panel(axes[1], "eccentric", r"B  eccentric mask  ($\sigma_M=\sigma_e$)")
    fig.tight_layout()
    save(fig, "fig_recovery.png")


if __name__ == "__main__":
    main()
