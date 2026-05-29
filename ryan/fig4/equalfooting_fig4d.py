r"""Panel-D equal-footing test: is the model-vs-empirical 1-alpha cloud a real
signal difference or an estimator artifact?

Panel D plots estimator A (all-samples ANOVA, rate_variance_components) on the
MODEL rates against estimator B (the fig2 close-pair below_threshold pipeline) on
the NEURONS. The two axes differ in *two* ways at once -- signal (twin vs neuron)
AND estimator (clean ANOVA over all eye samples vs close-pair intercept over
Delta_e<0.05deg). The reviewer risk ("the model is more eye-sensitive than the
neurons") is a claim about the *signal*, but it is confounded by the *estimator*.

This script disentangles them, fully cache-only (no GPU model inference; the
covariance ops run on GPU if present). For every good cell it computes 1-alpha
four ways, all in the SAME fig4 frame:

  A_panel  : rate_variance_components(rhat, valid = dfs!=0)   -- exact panel-D y-axis.
  A_shared : rate_variance_components(rhat, valid = eye-mask) -- estimator A, B's mask.
  B_model  : pipeline_one_minus_alpha(rhat, eye, eye-mask)    -- estimator B on model.
  B_obs    : pipeline_one_minus_alpha(robs, eye, eye-mask)    -- estimator B on neurons.
  fig2     : 1 - sr['alpha']                                  -- panel-D x-axis.

Decisive comparisons (good cells):
  * A_panel  vs fig2   -- reproduces panel D (plumbing sanity).
  * A_shared vs B_model-- ESTIMATOR effect on IDENTICAL model rates. A>B here means
                          the close-pair pipeline reports systematically lower
                          1-alpha than the ANOVA on the very same rates, so the
                          panel-D cloud is (at least partly) an estimator artifact.
  * B_model  vs B_obs  -- SIGNAL effect under the IDENTICAL estimator and frame.
                          If these agree, the twin is NOT more eye-sensitive than
                          the neurons; the apparent over-sensitivity was the
                          A-on-model / B-on-neuron mismatch.
  * B_obs    vs fig2   -- fig4-frame pipeline vs fig2-frame pipeline (alignment).
  * A_shared vs A_panel-- does the valid-mask choice (eye-finite vs dfs!=0) matter?

Usage:
    uv run python equalfooting_fig4d.py [--recompute]
"""
import argparse
import numpy as np
import dill
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from VisionCore.covariance import rate_variance_components, pipeline_one_minus_alpha
from _fig4_data import (
    FIG_DIR, CACHE_DIR, SUBJECTS, SUBJECT_COLORS, CCMAX_THRESHOLD,
    configure_matplotlib, load_fig4_data,
)

RESULT_CACHE = CACHE_DIR / "fig4d_equalfooting.pkl"
THRESHOLD = 0.05
MIN_TPP = 10


def compute(data, device="cuda"):
    cols = ("subj", "good", "A_panel", "A_shared", "B_model", "B_obs", "fig2")
    rows = {k: [] for k in cols}
    for si, sr in enumerate(data["session_results"]):
        rhat = sr["rhat_used"]                 # (trials, time, neurons), rescaled
        robs = sr["robs_used"]
        eye = sr["eyepos_used"]                # (trials, time, 2)
        vmask = sr["valid_mask"]               # (trials, time) eye-finite
        dfs = sr["dfs_used"]
        alpha = np.asarray(sr["alpha"], float)
        ccmax = np.asarray(sr["ccmax"], float)
        nN = rhat.shape[2]
        print(f"[{si+1}/{len(data['session_results'])}] {sr['session']} "
              f"({sr['subject']}): {rhat.shape[0]} trials, {nN} neurons")

        # estimator B (multi-cell) on model rates and on observed spikes, SAME mask
        B_model = pipeline_one_minus_alpha(rhat, eye, valid=vmask, threshold=THRESHOLD,
                                           min_trials_per_phase=MIN_TPP, device=device)
        B_obs = pipeline_one_minus_alpha(robs, eye, valid=vmask, threshold=THRESHOLD,
                                         min_trials_per_phase=MIN_TPP, device=device)

        for ni in range(nN):
            a_panel = rate_variance_components(
                rhat[:, :, ni], valid=dfs[:, :, ni] != 0,
                min_trials_per_phase=MIN_TPP)["one_minus_alpha"]
            a_shared = rate_variance_components(
                rhat[:, :, ni], valid=vmask,
                min_trials_per_phase=MIN_TPP)["one_minus_alpha"]
            rows["subj"].append(sr["subject"])
            rows["good"].append(bool(ccmax[ni] > CCMAX_THRESHOLD
                                     and np.isfinite(alpha[ni])))
            rows["A_panel"].append(a_panel)
            rows["A_shared"].append(a_shared)
            rows["B_model"].append(float(B_model["one_minus_alpha"][ni]))
            rows["B_obs"].append(float(B_obs["one_minus_alpha"][ni]))
            rows["fig2"].append(1.0 - alpha[ni])

    out = {k: (np.asarray(v, dtype=object).astype(str) if k == "subj"
               else np.asarray(v, dtype=bool) if k == "good"
               else np.asarray(v, dtype=float))
           for k, v in rows.items()}
    return out


def _pair(name, x, y, mask):
    """Report y-vs-x relationship over finite, masked cells (x first = baseline)."""
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        print(f"    {name}: N={ok.sum()} (too few)")
        return
    a, b = x[ok], y[ok]
    rho = spearmanr(a, b).correlation
    r = pearsonr(a, b)[0]
    print(f"    {name}: N={ok.sum()}, ρ={rho:.3f}, r={r:.3f}, "
          f"median(y-x)={np.median(b - a):+.3f}, "
          f"med x={np.median(a):.3f}, med y={np.median(b):.3f}")


def report(res):
    for subj in SUBJECTS + ["All"]:
        g = res["good"] if subj == "All" else (res["good"] & (res["subj"] == subj))
        print(f"\n=== {subj}: N_good={g.sum()} ===")
        print("  [panel-D plumbing]  y=A_panel  x=fig2")
        _pair("A_panel  vs fig2   ", res["fig2"], res["A_panel"], g)
        print("  [ESTIMATOR effect, identical model rates]  y=B_model  x=A_shared")
        _pair("B_model  vs A_shared", res["A_shared"], res["B_model"], g)
        print("  [SIGNAL effect, identical estimator+frame]  y=B_model  x=B_obs")
        _pair("B_model  vs B_obs   ", res["B_obs"], res["B_model"], g)
        print("  [alignment]  y=B_obs  x=fig2")
        _pair("B_obs    vs fig2   ", res["fig2"], res["B_obs"], g)
        print("  [mask choice]  y=A_shared  x=A_panel")
        _pair("A_shared vs A_panel ", res["A_panel"], res["A_shared"], g)


def make_figure(res):
    g = res["good"]
    pairs = [
        ("fig2", "A_panel", "fig2 (empirical)", "model ANOVA (panel D)"),
        ("A_shared", "B_model", "model ANOVA (A)", "model pipeline (B)"),
        ("B_obs", "B_model", "neuron pipeline (B)", "model pipeline (B)"),
        ("fig2", "B_obs", "fig2 (empirical)", "neuron pipeline (B, fig4 frame)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, (xk, yk, xl, yl) in zip(axes.ravel(), pairs):
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
        for subj in SUBJECTS:
            m = g & (res["subj"] == subj) & np.isfinite(res[xk]) & np.isfinite(res[yk])
            ax.scatter(res[xk][m], res[yk][m], s=6, alpha=0.5,
                       color=SUBJECT_COLORS[subj], label=subj)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(rf"{xl}  $1-\alpha$")
        ax.set_ylabel(rf"{yl}  $1-\alpha$")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "panel_d_equalfooting.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    configure_matplotlib()
    data = load_fig4_data()
    if RESULT_CACHE.exists() and not args.recompute:
        print(f"Loading equal-footing results from {RESULT_CACHE}")
        with open(RESULT_CACHE, "rb") as f:
            res = dill.load(f)
    else:
        res = compute(data)
        with open(RESULT_CACHE, "wb") as f:
            dill.dump(res, f)
        print(f"Cached equal-footing results to {RESULT_CACHE}")

    report(res)
    make_figure(res)


if __name__ == "__main__":
    main()
