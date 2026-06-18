r"""Figure: McFarland's estimator under (A1)+(A2) recovers the analytical
1-alpha^p on the unified random-field synthetic.

The additive synthetic could not sanity-check McFarland in his native regime
because the only (A2)-respecting profile (flat F) forced 1-alpha = 0. The
unified random-field synthetic has a stationary GP rate map whose (A2) is
satisfied at every length scale ell, with a non-trivial closed-form

    1-alpha^p   = 2 sigma^2 / (ell^2 + 2 sigma^2)
    1-alpha^p2  = sigma^2   / (ell^2 + sigma^2)

So we can sweep ell/sigma to cover (0, 1) and verify that:

  A  the empirical decompose(target='naive') with constant n_t matches
     1-alpha^p (= 1 - V_p / tau^2) across the sweep -- mean +- seed std on
     6 seeds overlaid on the analytical curve;
  B  the estimate is robust to the close-pair threshold over a useful range.

The closed-form 1-alpha^p / 1-alpha^p2 / gap panel that used to live here has
moved to fig_distribution_truth.py (writeup Fig. 0a), where it belongs at the
end of section 2.3 -- before any estimator is introduced.

Consistency (how SEM depends on n_trials AND n_time_bins (T), the T-floor at
sqrt(2 alpha^2 / (T-1)), and the boundary-clipping bias at high SEM) is
explored separately in writeup Appendix A.6, with its own figure.

Run from this folder:  uv run python fig_sanity_check.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

SIG = 0.15
NPH = 100
NTR_DEF = 600
THR_DEF = 0.05


def _closed_form(ell, sigma=SIG):
    """Analytical 1-alpha^p, 1-alpha^p2, and gap at given ell/sigma."""
    s2 = sigma ** 2
    L2 = ell ** 2
    oma_p = 2.0 * s2 / (L2 + 2.0 * s2)
    oma_p2 = s2 / (L2 + s2)
    gap = s2 * L2 / ((L2 + 2.0 * s2) * (L2 + s2))
    return oma_p, oma_p2, gap


def panel_A(ax, ratios=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            seeds=range(6)):
    """Empirical decompose(target='naive') vs analytical 1-alpha^p, over a
    sweep of ell/sigma."""
    r = np.array(ratios, dtype=float)
    ells = r * SIG
    ana_p, _, _ = _closed_form(ells)

    # smooth analytical curve
    r_dense = np.linspace(min(r) * 0.8, max(r) * 1.1, 200)
    ana_dense_p, _, _ = _closed_form(r_dense * SIG)
    ax.plot(r_dense, ana_dense_p, color=C_FULL, lw=1.5,
            label=r"analytical  $1-\alpha^p$")

    means, sds = [], []
    for ell in ells:
        vals = []
        for s in seeds:
            sess = make_session(["flat"], n_trials=NTR_DEF, n_time_bins=NPH,
                                sigma_eye=SIG, ell=ell, seed=s)
            d = decompose(sess["rate"], sess["eye"], target="naive",
                          density="gaussian", threshold=THR_DEF)
            vals.append(float(d["one_minus_alpha"][0]))
        means.append(np.nanmean(vals))
        sds.append(np.nanstd(vals))
    means = np.array(means); sds = np.array(sds)
    ax.errorbar(r, means, yerr=sds, fmt="o", color=C_CLOSE, capsize=3,
                lw=1.2, label="McFarland empirical (mean$\\pm$sd, 6 seeds)")
    ax.set_xlabel(r"$\ell\,/\,\sigma_e$")
    ax.set_ylabel(r"$1-\alpha^p$")
    ax.set_title("A  empirical recovers analytical across $\\ell/\\sigma_e$")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)


def panel_B(ax, thresholds=(0.02, 0.035, 0.05, 0.08, 0.12), seeds=range(8),
            ell_over_sigma=1.0):
    """Robustness vs close-pair threshold at fixed (ell, sigma)."""
    ell = ell_over_sigma * SIG
    oma_p, _, _ = _closed_form(ell)
    means, sds = [], []
    for thr in thresholds:
        vals = []
        for s in seeds:
            sess = make_session(["flat"], n_trials=NTR_DEF, n_time_bins=NPH,
                                sigma_eye=SIG, ell=ell, seed=s)
            d = decompose(sess["rate"], sess["eye"], target="naive",
                          density="gaussian", threshold=thr)
            vals.append(float(d["one_minus_alpha"][0]))
        means.append(np.nanmean(vals))
        sds.append(np.nanstd(vals))
    means = np.array(means); sds = np.array(sds)
    ax.errorbar(thresholds, means, yerr=sds, fmt="o-", color=C_CLOSE,
                capsize=3, lw=1.2, label="empirical (8 seeds)")
    ax.axhline(oma_p, color=C_FULL, lw=1.2, ls="--",
               label=rf"analytical $1-\alpha^p$ = {oma_p:.3f}")
    ax.set_xlabel(r"close-pair threshold $\varepsilon$ (deg)")
    ax.set_ylabel(r"$1-\alpha^p$")
    ax.set_title(r"B  threshold robustness  ($\ell=\sigma_e$)")
    ax.set_xscale("log")
    ax.legend(loc="lower left", fontsize=7)


def main():
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panel_A(axes[0])
    panel_B(axes[1])
    fig.tight_layout()
    save(fig, "fig_sanity_check.png")


if __name__ == "__main__":
    main()
