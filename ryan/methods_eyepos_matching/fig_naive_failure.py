r"""Figure: how the naive (distribution-unmatched) estimator fails.

On synthetic Poisson data with KNOWN ground truth (pure Poisson => true noise
correlation 0, true Fano 1; rate decomposition known in closed form), the naive
close-pair estimator -- 2nd moment over the central p(e)^2 but mean/total over the
full p(e) -- is biased on all three reported quantities, with a sign set by where
each cell's eye-sensitivity lives:

  A  1-alpha : naive over-states FEM for central cells, under-states (-> NaN) for
               eccentric cells; the matched 'full' estimator lands on truth.
  B  noise correlation : the rate-variance distribution mismatch leaks into the
               stimulus-independent covariance Ctotal - Crate -> spurious noise
               correlations where the truth is exactly 0.
  C  Fano factor : the same leak biases diag(Ctotal - Crate)/rate away from 1.

Run from this folder:  uv run python fig_naive_failure.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import make_session
from estimators import decompose
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH
from VisionCore.covariance import get_upper_triangle

PROFILES = ["central"] * 6 + ["eccentric"] * 6
PCOLOR = {"central": "#8e44ad", "eccentric": "#e67e22"}
N, NPH, SIG, THR = 400, 100, 0.15, 0.05


def run(seeds=range(4)):
    oma = {"naive": [], "full": [], "gt": [], "kind": []}
    ncorr = {"naive": [], "full": []}
    fano = {"naive": [], "full": []}
    for s in seeds:
        sess = make_session(PROFILES, n_trials=N, n_time_bins=NPH, sigma_eye=SIG,
                            seed=s)  # pure Poisson: true ncorr 0, Fano 1
        dn = decompose(sess["spikes"], sess["eye"], target="naive",
                       density="gaussian", threshold=THR)
        df = decompose(sess["spikes"], sess["eye"], target="full",
                       density="gaussian", threshold=THR)
        oma["naive"].append(dn["one_minus_alpha"])
        oma["full"].append(df["one_minus_alpha"])
        oma["gt"].append([sess["truth"][c]["p"]["one_minus_alpha"]
                          for c in range(len(PROFILES))])
        oma["kind"].append(PROFILES)
        ncorr["naive"].append(get_upper_triangle(dn["noise_corr"]))
        ncorr["full"].append(get_upper_triangle(df["noise_corr"]))
        fano["naive"].append(dn["fano"])
        fano["full"].append(df["fano"])
    for d in (oma, ncorr, fano):
        for k in d:
            d[k] = np.concatenate([np.ravel(x) for x in d[k]]) if k != "kind" \
                else np.concatenate(d[k])
    return oma, ncorr, fano


def main():
    configure()
    oma, ncorr, fano = run()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # --- A: 1-alpha, naive vs matched, colored by profile ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C_TRUTH, lw=0.8, ls="--", zorder=0)
    for kind in ("central", "eccentric"):
        m = oma["kind"] == kind
        ax.scatter(oma["gt"][m], np.nan_to_num(oma["naive"][m], nan=-0.03)[:],
                   s=18, color=PCOLOR[kind], marker="x", alpha=0.8,
                   label=f"{kind} (naive)")
        ax.scatter(oma["gt"][m], oma["full"][m], s=16, facecolors="none",
                   edgecolors=PCOLOR[kind], alpha=0.9)
    ax.set_xlabel(r"true $1-\alpha$  (over $p$)")
    ax.set_ylabel(r"estimated $1-\alpha$")
    ax.set_title("A  1−α: naive biased (× ), matched (○) on truth")
    ax.set_xlim(-0.05, 1); ax.set_ylim(-0.06, 1)
    ax.legend(fontsize=6, loc="upper left", ncol=1)
    ax.annotate("NaN", (0.0, -0.03), fontsize=6, color="0.4")

    # --- B: noise correlation, true = 0 ---
    ax = axes[1]
    bins = np.linspace(-0.5, 0.5, 51)
    ax.hist(ncorr["naive"], bins=bins, color=C_CLOSE, alpha=0.55,
            label=f"naive (med |r|={np.nanmedian(np.abs(ncorr['naive'])):.3f})")
    ax.hist(ncorr["full"], bins=bins, color=C_FULL, alpha=0.55,
            label=f"matched (med |r|={np.nanmedian(np.abs(ncorr['full'])):.3f})")
    ax.axvline(0, color=C_TRUTH, lw=1.0, ls="--", label="truth = 0")
    ax.set_xlabel("noise correlation (off-diag)"); ax.set_ylabel("pair count")
    ax.set_title("B  spurious noise correlation")
    ax.legend(fontsize=7)

    # --- C: Fano, true = 1 ---
    ax = axes[2]
    bins = np.linspace(0.6, 1.6, 41)
    ax.hist(fano["naive"], bins=bins, color=C_CLOSE, alpha=0.55,
            label=f"naive (med={np.nanmedian(fano['naive']):.2f})")
    ax.hist(fano["full"], bins=bins, color=C_FULL, alpha=0.55,
            label=f"matched (med={np.nanmedian(fano['full']):.2f})")
    ax.axvline(1, color=C_TRUTH, lw=1.0, ls="--", label="truth = 1")
    ax.set_xlabel("Fano factor"); ax.set_ylabel("cell count")
    ax.set_title("C  biased Fano factor")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "fig_naive_failure.png")


if __name__ == "__main__":
    main()
