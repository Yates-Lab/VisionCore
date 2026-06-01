r"""Figure 4 panel D: does the digital twin recapitulate the population's 1-alpha?

Usage:
    uv run ryan/fig4/generate_fig4d.py [--recompute]

Scientific question
-------------------
1-alpha is the fraction of a cell's total *rate* variance driven by fixational
eye movements (FEM), as opposed to the stimulus-locked (PSTH) fraction alpha.
This panel asks whether the digital twin reproduces each cell's 1-alpha in its
own predictions -- i.e. whether the model gets the proportion of
eye-movement-driven variance right, not merely the mean response. The twin has
no extraretinal channel for eye movements (the bottom-row ablation, panels
H-J, shows zeroing the behavior input changes nothing), so any FEM-driven
variance it produces arrives through the *retinal* consequence of drift: the
stimulus shifting on the retina. Matching 1-alpha is therefore a strong test of
that retinal mechanism.

Equal-footing estimator (close-pair on both axes)
-------------------------------------------------
Both axes use the SAME estimator -- the empirical close-pair decomposition that
figure 2 runs on real spikes (``pipeline_one_minus_alpha``, "estimator B"):

  * x-axis (empirical 1-alpha): close-pair estimator on the observed spike
    counts ``robs_used``.
  * y-axis (model 1-alpha):     close-pair estimator on the model rates
    ``rhat_used``, evaluated at each trial's actual eye trajectory.

Both run in the same fig4 trial frame, with the same per-(trial, bin) eye
trajectory, validity mask, close-pair threshold (Delta_e < 0.05 deg) and
min_trials_per_phase = 10. Putting model and neurons on one estimator isolates
the *signal* difference (twin vs neuron) from the *estimator* difference.

Why not the analytic ANOVA on model rates
------------------------------------------
The previous version of this panel plotted estimator A (all-samples one-way
ANOVA, ``rate_variance_components``) on the model rates against fig2's published
close-pair alpha on the neurons. That mixes two estimators across the axes: A
integrates FEM over the *full* fixational eye distribution while the close-pair
estimator weights it by the squared eye density p(e)^2 (close pairs concentrate
where the eye dwells). On identical model rates the two estimators correlate
only weakly and A compresses 1-alpha, so the ANOVA-vs-close-pair mismatch -- not
a real model property -- manufactured the apparent "twin is more eye-sensitive
than the neurons." On equal footing (close-pair on both axes) the twin
reproduces the cells (see VisionCore/ryan/methods_eyepos_matching writeup A.8).

Population line: total least squares
------------------------------------
Both axes are noisy estimates of the same latent 1-alpha produced by the same
estimator, so their error variances are comparable. Ordinary least squares would
attenuate the slope toward 0 (regression dilution from x-noise), biasing us
*against* the "model tracks the population" claim. We therefore fit the
population line by equal-variance total least squares (orthogonal regression);
OLS is retained in the exported JSON for reference.

Matching conditions
-------------------
Good cells (ccmax > CCMAX_THRESHOLD) with finite 1-alpha on both axes. The
close-pair threshold (0.05 deg) and min_trials_per_phase (10) mirror the
empirical fig2 settings. 1-alpha is clipped to [0, 1] on both sides.
"""
import argparse
import json

import numpy as np
import dill
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress, wilcoxon

from VisionCore.covariance import pipeline_one_minus_alpha
from _fig4_data import (
    FIG_DIR, STAT_DIR, CACHE_DIR, SUBJECTS, SUBJECT_COLORS, CCMAX_THRESHOLD,
    configure_matplotlib, load_fig4_data, annotate_corr,
)

THRESHOLD = 0.05            # close-pair eye-distance threshold (deg), mirrors fig2
MIN_TRIALS_PER_PHASE = 10   # mirror empirical min_trials_per_time
RESULT_CACHE = CACHE_DIR / "fig4d_panel.pkl"
STATS_PATH = STAT_DIR / "panel_d_one_minus_alpha.json"


def compute_panel_d(data, threshold=THRESHOLD,
                    min_trials_per_phase=MIN_TRIALS_PER_PHASE, device="cuda"):
    """Per-cell close-pair 1-alpha on model rates and observed spikes, by neuron.

    For each session runs ``pipeline_one_minus_alpha`` once on the model rates
    ``rhat_used`` (-> model 1-alpha) and once on the observed spike counts
    ``robs_used`` (-> empirical 1-alpha), sharing the same eye trajectory,
    validity mask, threshold and min_trials_per_phase. Returns a dict of
    equal-length per-cell arrays (all cells; goodness is applied at plot time).
    """
    session_results = data["session_results"]
    emp, mod, subj_out, ccmax_out = [], [], [], []
    for si, sr in enumerate(session_results):
        rhat = sr["rhat_used"]                 # (trials, time, neurons), rescaled
        robs = sr["robs_used"]                 # (trials, time, neurons)
        eye = sr["eyepos_used"]                # (trials, time, 2)
        vmask = sr["valid_mask"]               # (trials, time) eye-finite
        ccmax = np.asarray(sr["ccmax"], float)
        print(f"[{si + 1}/{len(session_results)}] {sr['session']} "
              f"({sr['subject']}): {rhat.shape[0]} trials, {rhat.shape[2]} neurons")

        B_model = pipeline_one_minus_alpha(
            rhat, eye, valid=vmask, threshold=threshold,
            min_trials_per_phase=min_trials_per_phase, device=device)
        B_obs = pipeline_one_minus_alpha(
            robs, eye, valid=vmask, threshold=threshold,
            min_trials_per_phase=min_trials_per_phase, device=device)

        for ni in range(rhat.shape[2]):
            emp.append(float(B_obs["one_minus_alpha"][ni]))
            mod.append(float(B_model["one_minus_alpha"][ni]))
            subj_out.append(sr["subject"])
            ccmax_out.append(float(ccmax[ni]))

    return {
        "emp": np.asarray(emp),
        "model": np.asarray(mod),
        "subjects": np.asarray(subj_out),
        "ccmax": np.asarray(ccmax_out),
    }


def load_panel_d_results(data=None, recompute=False, device="cuda"):
    """Return cached per-cell panel-D results, computing + caching if needed."""
    if RESULT_CACHE.exists() and not recompute:
        print(f"Loading panel-D results from {RESULT_CACHE}")
        with open(RESULT_CACHE, "rb") as f:
            return dill.load(f)
    if data is None:
        data = load_fig4_data()
    comp = compute_panel_d(data, device=device)
    with open(RESULT_CACHE, "wb") as f:
        dill.dump(comp, f)
    print(f"Cached panel-D results to {RESULT_CACHE}")
    return comp


def _good_mask(results):
    return (results["ccmax"] > CCMAX_THRESHOLD) & \
        np.isfinite(results["emp"]) & np.isfinite(results["model"])


def _tls_fit(x, y):
    """Equal-variance total least squares (orthogonal) line y = slope*x + intercept."""
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    Sxx = float(np.dot(dx, dx))
    Syy = float(np.dot(dy, dy))
    Sxy = float(np.dot(dx, dy))
    slope = (Syy - Sxx + np.sqrt((Syy - Sxx) ** 2 + 4 * Sxy ** 2)) / (2 * Sxy)
    intercept = my - slope * mx
    return float(slope), float(intercept)


def _tls_band(x, y, n_boot=1000, seed=0, xs=None):
    """Bootstrap 95% confidence band for the TLS line (resample cells, refit)."""
    if xs is None:
        xs = np.linspace(0.0, 1.0, 101)
    rng = np.random.default_rng(seed)
    n = len(x)
    preds = np.empty((n_boot, len(xs)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s, i = _tls_fit(x[idx], y[idx])
        preds[b] = s * xs + i
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    return xs, lo, hi


def _stats_for(x, y):
    """Population statistics for the model(y)-vs-empirical(x) relationship."""
    r_p, p_p = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)
    slope_tls, intercept_tls = _tls_fit(x, y)
    ols = linregress(x, y)
    diff = y - x
    try:
        w_stat, w_p = wilcoxon(diff)
    except ValueError:           # all-zero differences
        w_stat, w_p = float("nan"), float("nan")
    return {
        "N": int(len(x)),
        "tls_slope": slope_tls,
        "tls_intercept": intercept_tls,
        "ols_slope": float(ols.slope),
        "ols_intercept": float(ols.intercept),
        "pearson_r": float(r_p),
        "pearson_r2": float(r_p ** 2),
        "pearson_p": float(p_p),
        "spearman_rho": float(rho),
        "spearman_p": float(p_s),
        "median_emp": float(np.median(x)),
        "median_model": float(np.median(y)),
        "median_diff_model_minus_emp": float(np.median(diff)),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
    }


def export_panel_d_stats(results, out_path=STATS_PATH, verbose=True):
    """Compute per-subject and pooled statistics and write them to JSON."""
    good = _good_mask(results)
    emp, mod, subjects = results["emp"], results["model"], results["subjects"]
    stats = {}
    for subj in SUBJECTS + ["All"]:
        mask = good if subj == "All" else good & (subjects == subj)
        if mask.sum() < 3:
            stats[subj] = {"N": int(mask.sum())}
            continue
        stats[subj] = _stats_for(emp[mask], mod[mask])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        for subj in SUBJECTS + ["All"]:
            s = stats[subj]
            if s.get("N", 0) < 3:
                print(f"Panel D — {subj}: N={s.get('N', 0)} (too few)")
                continue
            print(f"Panel D — {subj} (N={s['N']}): "
                  f"TLS slope={s['tls_slope']:.3f}, intercept={s['tls_intercept']:+.3f}, "
                  f"Pearson r={s['pearson_r']:.3f} (p={s['pearson_p']:.2g}), "
                  f"Spearman ρ={s['spearman_rho']:.3f}, "
                  f"median(model-emp)={s['median_diff_model_minus_emp']:+.3f} "
                  f"(Wilcoxon p={s['wilcoxon_p']:.2g})")
        print(f"Saved panel-D stats to {out_path}")
    return stats


def plot_panel_d(ax=None, data=None, results=None, legend_fontsize=8,
                 print_stats=True, recompute=False, device="cuda"):
    """Scatter model 1-alpha vs empirical 1-alpha per cell. Returns (fig, ax)."""
    if results is None:
        results = load_panel_d_results(data=data, recompute=recompute, device=device)
    emp, mod, subjects = results["emp"], results["model"], results["subjects"]
    good = _good_mask(results)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    for subj in SUBJECTS:
        mask = good & (subjects == subj)
        if not mask.any():
            continue
        ax.scatter(emp[mask], mod[mask], s=5, alpha=0.5,
                   color=SUBJECT_COLORS[subj])

    # Population total-least-squares line + bootstrap 95% band over good cells.
    x_all, y_all = emp[good], mod[good]
    slope, intercept = _tls_fit(x_all, y_all)
    xs, lo, hi = _tls_band(x_all, y_all)
    ax.fill_between(xs, lo, hi, color="red", alpha=0.2, linewidth=0)
    ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.0)

    rho_all, p_all = spearmanr(x_all, y_all)
    annotate_corr(ax, rho_all, p_all, loc="upper left")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Empirical $1-\alpha$")
    ax.set_ylabel(r"Model $1-\alpha$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if print_stats:
        export_panel_d_stats(results)

    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 4 panel D.")
    p.add_argument("--recompute", action="store_true",
                   help="Force recomputation of the close-pair decomposition.")
    args = p.parse_args()

    configure_matplotlib()
    data = load_fig4_data()
    results = load_panel_d_results(data=data, recompute=args.recompute)
    fig, ax = plot_panel_d(data=data, results=results)
    fig.tight_layout()
    out = FIG_DIR / "panel_d_one_minus_alpha.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
