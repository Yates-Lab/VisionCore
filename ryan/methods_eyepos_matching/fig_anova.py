r"""Appendix figure: one-way ANOVA on known rates recovers 1-alpha^p.

When the rate r(t, e) is observed without Poisson noise (the digital-twin
setting in `VisionCore/ryan/fig4/generate_fig4d.py`), the Law-of-Total-
Covariance decomposition reduces to a one-way random-effects ANOVA over time
bin t. The ANOVA targets `1-alpha^p` (Direction 1) because trial-level eye
positions are drawn from the marginal viewing distribution p -- there is no
close-pair filter and no importance reweighting.

This figure validates `VisionCore.covariance.rate_variance_components`
against two references on the same noise-free synthetic data:

  - the analytical closed-form `1-alpha^p` (Direction-1 truth);
  - the Direction-1 close-pair estimator `decompose(target='full')` from
    this folder, which targets the same population quantity by a different
    route (importance-reweighted close-pair second moment).

Across all four mask types (flat, central, eccentric, linear) the two
estimators fall on the analytical line; the ANOVA error bars are visibly
tighter, because the ANOVA uses every (trial, time-bin) sample once and
pays no close-pair variance tax (no Delta-e threshold, no unbounded 1/p
importance weights).

Run from this folder:  uv run python fig_anova.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session, ground_truth, PROFILE_KINDS
from estimators import decompose
from VisionCore.covariance import rate_variance_components

from _style import configure, save, C_FULL, C_TRUTH, C_OK


SIG = 0.15
N_TIME_BINS = 80
N_TRIALS = 300
THR = 0.05

RATIOS = np.array([0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0])  # ell / sigma
SEEDS = range(8)

MASK_TITLES = {
    "flat":      "flat  (A2 holds)",
    "central":   "central",
    "eccentric": "eccentric",
    "linear":    "linear",
}


def _truth_curve(kind, n_dense=200):
    """Smooth analytical 1-alpha^p curve over a dense ell/sigma grid."""
    r = np.geomspace(RATIOS.min() * 0.85, RATIOS.max() * 1.15, n_dense)
    oma = np.array([ground_truth(kind, SIG, ell=ratio * SIG)["p"]["one_minus_alpha"]
                    for ratio in r])
    return r, oma


def _anova_estimate(sess, cell_idx=0):
    rate = sess["rate"][:, :, cell_idx]
    valid = sess["valid"]
    out = rate_variance_components(rate, valid=valid, min_trials_per_phase=2)
    return out["one_minus_alpha"]


def _direction1_estimate(sess, cell_idx=0):
    out = decompose(sess["rate"], sess["eye"], target="full",
                    density="gaussian", threshold=THR,
                    time_bin_weighting="pair_count",
                    cpsth_method="mcfarland")
    return float(out["one_minus_alpha"][cell_idx])


def _sweep(kind, ratios, seeds):
    """Run both estimators across the ell/sigma sweep on noise-free rates."""
    anova_mean = np.empty(len(ratios))
    anova_sem  = np.empty(len(ratios))
    cp_mean    = np.empty(len(ratios))
    cp_sem     = np.empty(len(ratios))
    for i, ratio in enumerate(ratios):
        ell = ratio * SIG
        a_vals, c_vals = [], []
        for s in seeds:
            sess = make_session([kind], n_trials=N_TRIALS,
                                n_time_bins=N_TIME_BINS,
                                sigma_eye=SIG, ell=ell, seed=s)
            a_vals.append(_anova_estimate(sess))
            c_vals.append(_direction1_estimate(sess))
        a_vals = np.asarray(a_vals)
        c_vals = np.asarray(c_vals)
        # SEM across seeds (finite-only)
        a_ok = a_vals[np.isfinite(a_vals)]
        c_ok = c_vals[np.isfinite(c_vals)]
        anova_mean[i] = a_ok.mean() if a_ok.size else np.nan
        anova_sem[i]  = a_ok.std(ddof=1) / np.sqrt(a_ok.size) if a_ok.size > 1 else 0.0
        cp_mean[i]    = c_ok.mean() if c_ok.size else np.nan
        cp_sem[i]     = c_ok.std(ddof=1) / np.sqrt(c_ok.size) if c_ok.size > 1 else 0.0
    return anova_mean, anova_sem, cp_mean, cp_sem


def _panel(ax, kind, ratios, seeds, show_y_label=True, show_legend=False):
    r_dense, oma_dense = _truth_curve(kind)
    am, asem, cm, csem = _sweep(kind, ratios, seeds)

    # Print a one-line per-panel diagnostic (handy for verifying tightness).
    bias_anova = np.nanmean(am - np.interp(ratios, r_dense, oma_dense))
    bias_cp    = np.nanmean(cm - np.interp(ratios, r_dense, oma_dense))
    print(f"  {kind:9s}  mean(ANOVA - truth) = {bias_anova:+.4f}    "
          f"mean(D1 close-pair - truth) = {bias_cp:+.4f}")

    ax.plot(r_dense, oma_dense, color=C_TRUTH, lw=1.6,
            label=r"analytical  $1-\alpha^p$")

    # offset the two marker sets slightly on the log x-axis so error bars
    # don't overlap visually
    off = 1.05
    ax.errorbar(np.asarray(ratios) / off, am, yerr=asem, fmt="o",
                color=C_FULL, capsize=2.5, ms=5, lw=1.0,
                label="ANOVA  (rate_variance_components)")
    ax.errorbar(np.asarray(ratios) * off, cm, yerr=csem, fmt="s",
                color=C_OK, capsize=2.5, ms=5, lw=1.0, mfc="white",
                label=r"Direction-1 close-pair  (target='full')")

    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel(r"$\ell\,/\,\sigma$")
    if show_y_label:
        ax.set_ylabel(r"$1-\alpha^p$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(MASK_TITLES.get(kind, kind), pad=6)
    if show_legend:
        ax.legend(loc="lower left", fontsize=7, handlelength=2.2)


def main():
    configure()
    print("fig_anova.py  --  sweep over ell/sigma per mask")
    print(f"  N_TRIALS={N_TRIALS}  N_TIME_BINS={N_TIME_BINS}  "
          f"SEEDS={len(list(SEEDS))}  threshold={THR}")
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.2), sharey=True)
    for ax, kind in zip(axes, PROFILE_KINDS):
        _panel(ax, kind, RATIOS, SEEDS,
               show_y_label=(kind == PROFILE_KINDS[0]),
               show_legend=(kind == PROFILE_KINDS[0]))
    fig.suptitle(
        r"With known rates, ANOVA and Direction-1 close-pair both recover "
        r"$1-\alpha^p$; ANOVA is tighter.",
        fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_anova.png")


if __name__ == "__main__":
    main()
