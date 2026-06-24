r"""Supplemental: the digital twin replicates Figure 2's FEM modulation.

Figure 2 measures, per neuron, the fraction of rate variance driven by fixational
eye movements (1-alpha) with the eye-position-distribution-matched close-pair
estimator (target='full'). This supplemental runs the SAME estimator B on the
deterministic digital-twin rates and shows the twin reproduces the empirical
per-cell 1-alpha -- i.e. the model has learned the FEM-driven rate structure, not
just the stimulus-locked PSTH.

Both axes use estimator B (the matched close-pair pipeline, target='full'):
  x : B_obs   -- observed spikes through the pipeline (the Fig. 2 quantity,
                 recomputed in the fig3 frame).
  y : B_model -- twin rates through the identical pipeline.

Data come from ``covariance_decomposition.load_model_data`` (cache
``covdecomp_model.pkl``), which consumes the fig3 digital-twin inference cache.

Usage:
    uv run python paper/fig3/generate_figure3_supp_twin_replication.py [--refresh]
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from VisionCore.paths import VISIONCORE_ROOT
sys.path.insert(0, str(VISIONCORE_ROOT / "paper"))
from covariance_decomposition import load_model_data  # noqa: E402

from _fig3_data import (  # noqa: E402
    FIG_DIR, SUBJECTS, SUBJECT_COLORS, CCMAX_THRESHOLD, configure_matplotlib,
)


def make_figure(refresh=False):
    mb = load_model_data(refresh=refresh)
    good = mb["ccmax"] > CCMAX_THRESHOLD
    x = mb["B_obs"]            # neuron 1-alpha (estimator B, full)
    y = mb["B_model"]          # twin   1-alpha (estimator B, full)
    subj = mb["subj"]

    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(7.4, 3.4))

    # ---- per-cell scatter: twin vs neuron 1-alpha ----
    ax_s.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.6, zorder=0)
    for s in SUBJECTS:
        m = good & (subj == s) & np.isfinite(x) & np.isfinite(y)
        ax_s.scatter(x[m], y[m], s=7, alpha=0.5, color=SUBJECT_COLORS[s],
                     label=f"{s} (n={int(m.sum())})", linewidths=0)
    ok = good & np.isfinite(x) & np.isfinite(y)
    rho = spearmanr(x[ok], y[ok]).correlation
    r = pearsonr(x[ok], y[ok])[0]
    med_d = float(np.median(y[ok] - x[ok]))
    ax_s.text(0.04, 0.96,
              rf"$\rho$={rho:.2f}  r={r:.2f}" "\n"
              rf"med($\Delta$)={med_d:+.3f}  n={int(ok.sum())}",
              transform=ax_s.transAxes, ha="left", va="top", fontsize=8)
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_aspect("equal", "box")
    ax_s.set_xlabel(r"Neuron  $1-\alpha$  (estimator B, full)")
    ax_s.set_ylabel(r"Twin  $1-\alpha$  (estimator B, full)")
    ax_s.set_title("A", loc="left", fontweight="bold")
    ax_s.legend(frameon=False, fontsize=7, loc="lower right")
    ax_s.spines["top"].set_visible(False); ax_s.spines["right"].set_visible(False)

    # ---- matched marginal distributions ----
    bins = np.linspace(0, 1, 26)
    ax_h.hist(x[ok], bins=bins, color="0.45", alpha=0.55, label="Neuron (Fig. 2)",
              density=True, edgecolor="white", linewidth=0.3)
    ax_h.hist(y[ok], bins=bins, histtype="step", color="tab:red", lw=1.8,
              label="Twin", density=True)
    ax_h.axvline(np.median(x[ok]), color="0.3", ls="--", lw=1.0)
    ax_h.axvline(np.median(y[ok]), color="tab:red", ls="--", lw=1.0)
    ax_h.text(0.04, 0.96,
              f"median: neuron {np.median(x[ok]):.3f}\n"
              f"        twin   {np.median(y[ok]):.3f}",
              transform=ax_h.transAxes, ha="left", va="top", fontsize=8)
    ax_h.set_xlabel(r"$1-\alpha$  (FEM modulation)")
    ax_h.set_ylabel("density")
    ax_h.set_title("B", loc="left", fontweight="bold")
    ax_h.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.84))
    ax_h.spines["top"].set_visible(False); ax_h.spines["right"].set_visible(False)

    fig.suptitle("Digital twin replicates the empirical FEM modulation (Fig. 2)",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"figure3_supp_twin_replication.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved {out}")
    print(f"Twin replication: rho={rho:.3f}, r={r:.3f}, median(twin-neuron)={med_d:+.3f}, "
          f"n={int(ok.sum())} good cells")


if __name__ == "__main__":
    configure_matplotlib()
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute the model decomposition bundle.")
    args = ap.parse_args()
    make_figure(refresh=args.refresh)
