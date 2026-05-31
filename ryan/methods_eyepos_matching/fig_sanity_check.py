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
  B  the estimate is robust to the close-pair threshold over a useful range;
  C  closed-form 1-alpha^p, 1-alpha^p2, and gap vs ell/sigma -- the (A2)-
     under-the-field gap (sigma^2 ell^2)/[(ell^2 + 2 sigma^2)(ell^2 + sigma^2)]
     is non-zero for finite ell, vanishing only as ell -> 0 (all FEM) or
     ell -> infty (no FEM). This is the load-bearing reframe of section 4.5.

Consistency (how SEM depends on n_trials AND n_phases, the T-floor at
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
            sess = make_session(["flat"], n_trials=NTR_DEF, n_phases=NPH,
                                sigma_eye=SIG, ell=ell, seed=s)
            d = decompose(sess["rate"], sess["eye"], target="naive",
                          density="gaussian", threshold=THR_DEF)
            vals.append(float(d["one_minus_alpha"][0]))
        means.append(np.nanmean(vals))
        sds.append(np.nanstd(vals))
    means = np.array(means); sds = np.array(sds)
    ax.errorbar(r, means, yerr=sds, fmt="o", color=C_CLOSE, capsize=3,
                lw=1.2, label="McFarland empirical (mean$\\pm$sd, 6 seeds)")
    ax.set_xlabel(r"$\ell\,/\,\sigma$")
    ax.set_ylabel(r"$1-\alpha^p$")
    ax.set_title("A  empirical recovers analytical across $\\ell/\\sigma$")
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
            sess = make_session(["flat"], n_trials=NTR_DEF, n_phases=NPH,
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
    ax.set_title(r"B  threshold robustness  ($\ell=\sigma$)")
    ax.set_xscale("log")
    ax.legend(loc="lower left", fontsize=7)


def panel_C(ax):
    """Closed-form 1-alpha^p, 1-alpha^p2 and gap vs ell/sigma.
    The gap is non-zero under (A2) -- it measures fixation-scale spatial
    structure, not (A2) violation."""
    r = np.geomspace(0.1, 20, 400)
    ells = r * SIG
    oma_p, oma_p2, gap = _closed_form(ells)
    ax.plot(r, oma_p, color=C_FULL, lw=1.6, label=r"$1-\alpha^p$")
    ax.plot(r, oma_p2, color=C_CLOSE, lw=1.6, label=r"$1-\alpha^{p^2}$")
    ax.plot(r, gap, color="#2e8b57", lw=1.6, ls="--",
            label=r"gap = $1{-}\alpha^p - 1{-}\alpha^{p^2}$")
    # Annotate the closed-form gap maximum.
    i_max = int(np.argmax(gap))
    ax.scatter([r[i_max]], [gap[i_max]], s=24, color="#2e8b57", zorder=5)
    ax.annotate(rf"max gap $\approx${gap[i_max]:.2f} at $\ell/\sigma\approx${r[i_max]:.2f}",
                (r[i_max], gap[i_max]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=7, color="#2e8b57")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell\,/\,\sigma$")
    ax.set_ylabel(r"closed-form")
    ax.set_title(r"C  (A2)-respecting gap is non-zero on fixation scale")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(-0.02, 1.02)


def main():
    configure()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    panel_A(axes[0])
    panel_B(axes[1])
    panel_C(axes[2])
    fig.tight_layout()
    save(fig, "fig_sanity_check.png")


if __name__ == "__main__":
    main()
