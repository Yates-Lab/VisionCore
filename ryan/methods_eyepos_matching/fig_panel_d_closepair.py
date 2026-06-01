r"""Appendix §A.8 path-1 experiment: apply the Direction-1 close-pair
estimator to BOTH the empirical spike counts (`robs_used`) and the twin's
deterministic rates (`rhat_used`). Same estimator, same time-bin weighting,
both axes — answers whether the §A.8.6 gap (twin ANOVA ~0.04--0.09 above
cell D1) is an ANOVA-vs-close-pair method difference or a real twin
property.

For each fig4 panel-D good cell, with cell-specific validity `dfs!=0`,
compute four numbers:

  - emp_pair  : `_percell_close_pair_session(robs_used, ...,
                  time_bin_weighting='pair_count')`
  - emp_trial : `_percell_close_pair_session(robs_used, ...,
                  time_bin_weighting='trial_count')`
  - twin_pair : `_percell_close_pair_session(rhat_used,  ...,
                  time_bin_weighting='pair_count')`
  - twin_trial: `_percell_close_pair_session(rhat_used,  ...,
                  time_bin_weighting='trial_count')`

and the twin ANOVA from `rate_variance_components(rhat_used, dfs!=0)` for
reference.

Plot 1x4:

  A. cell D1 trial_count  vs  twin ANOVA            (the §A.8.6 gap; reference)
  B. twin D1 trial_count  vs  twin ANOVA            (same data on twin,
                                                    two estimators -- isolates
                                                    where the gap lives)
  C. cell D1 pair_count   vs  twin D1 pair_count    (matched estimator)
  D. cell D1 trial_count  vs  twin D1 trial_count   (matched estimator)

If the gap collapses in C/D, the §A.8.6 finding is an estimator-method
difference. Panel B is the diagnostic: ANOVA and close-pair Direction-1
target the same population truth (§A.8.2) and agree on synthetic (§A.8.5),
so the on-real-rhat gap in B is the puzzling residual.

Run from this folder:  uv run python fig_panel_d_closepair.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fig4"))
from _fig4_data import (                                            # noqa: E402
    load_fig4_data, SUBJECTS, SUBJECT_COLORS,
)

from VisionCore.covariance import rate_variance_components          # noqa: E402
from _style import configure, save, C_TRUTH                         # noqa: E402

from fig_panel_d_anova import (                                     # noqa: E402
    _percell_close_pair_session, MIN_TRIALS_PER_PHASE,
)


def compute():
    data = load_fig4_data()
    ats = data["all_trace_neuron_session"]
    valid_indices = data["valid_indices"]
    alpha_emp = data["alpha"]
    good = data["good"]
    subjects = data["subjects"]

    emp_pair, emp_trial = [], []
    twin_pair, twin_trial = [], []
    mod_anova = []
    subj_out = []
    emp_pair_cache, emp_trial_cache = {}, {}
    twin_pair_cache, twin_trial_cache = {}, {}

    for k in range(len(alpha_emp)):
        if not good[k] or not np.isfinite(alpha_emp[k]):
            continue
        si, ni = ats[valid_indices[k]]
        sr = data["session_results"][si]

        if si not in emp_pair_cache:
            print(f"  session {si:2d}  {sr['session']:22s}  "
                  f"({sr['n_neurons']} neurons)  emp pair_count ...")
            emp_pair_cache[si] = _percell_close_pair_session(
                sr["robs_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="pair_count")
        if si not in emp_trial_cache:
            print(f"  session {si:2d}  {sr['session']:22s}  "
                  f"({sr['n_neurons']} neurons)  emp trial_count ...")
            emp_trial_cache[si] = _percell_close_pair_session(
                sr["robs_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="trial_count")
        if si not in twin_pair_cache:
            print(f"  session {si:2d}  {sr['session']:22s}  "
                  f"({sr['n_neurons']} neurons)  twin pair_count ...")
            twin_pair_cache[si] = _percell_close_pair_session(
                sr["rhat_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="pair_count")
        if si not in twin_trial_cache:
            print(f"  session {si:2d}  {sr['session']:22s}  "
                  f"({sr['n_neurons']} neurons)  twin trial_count ...")
            twin_trial_cache[si] = _percell_close_pair_session(
                sr["rhat_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="trial_count")

        ep = emp_pair_cache[si][ni]
        et = emp_trial_cache[si][ni]
        tp = twin_pair_cache[si][ni]
        tt = twin_trial_cache[si][ni]
        if not (np.isfinite(ep) and np.isfinite(et)
                and np.isfinite(tp) and np.isfinite(tt)):
            continue

        rhat = sr["rhat_used"][:, :, ni]
        vmask = sr["dfs_used"][:, :, ni] != 0
        anova = rate_variance_components(
            rhat, valid=vmask, min_trials_per_phase=MIN_TRIALS_PER_PHASE)
        m = anova["one_minus_alpha"]
        if not np.isfinite(m):
            continue

        emp_pair.append(ep)
        emp_trial.append(et)
        twin_pair.append(tp)
        twin_trial.append(tt)
        mod_anova.append(m)
        subj_out.append(subjects[k])

    return {
        "emp_pair":   np.asarray(emp_pair),
        "emp_trial":  np.asarray(emp_trial),
        "twin_pair":  np.asarray(twin_pair),
        "twin_trial": np.asarray(twin_trial),
        "mod_anova":  np.asarray(mod_anova),
        "subjects":   np.asarray(subj_out),
    }


def _scatter_panel(ax, x, y, subjects, x_label, y_label, title):
    ax.plot([0, 1], [0, 1], color=C_TRUTH, lw=0.7, ls="--", alpha=0.6)
    for subj in SUBJECTS:
        mask = subjects == subj
        if not mask.any():
            continue
        rho = spearmanr(x[mask], y[mask]).correlation
        ax.scatter(x[mask], y[mask], s=10, alpha=0.55,
                   color=SUBJECT_COLORS[subj],
                   label=f"{subj}: $\\rho$ = {rho:.2f}  (N={mask.sum()})")
    rho_all = spearmanr(x, y).correlation
    med_dy = float(np.median(y - x))
    ax.text(0.04, 0.96,
            f"all: $\\rho$ = {rho_all:.2f}\nmed(y-x) = {med_dy:+.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=6)
    ax.legend(loc="lower right", fontsize=7, handlelength=0.6)


def _report(name, x, y):
    rho = spearmanr(x, y).correlation
    r = pearsonr(x, y)[0]
    med_x, med_y = float(np.median(x)), float(np.median(y))
    print(f"  {name:42s}  N={len(x):3d}  "
          f"spearman={rho:.3f}  pearson={r:.3f}  "
          f"median x={med_x:.3f}  median y={med_y:.3f}  "
          f"median(x-y)={med_x-med_y:+.3f}")


def main():
    configure()
    print("fig_panel_d_closepair.py  --  path-1: D1 close-pair on both axes")
    res = compute()
    n = len(res["mod_anova"])
    print(f"  {n} cells matched")

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))
    _scatter_panel(
        axes[0], res["emp_trial"], res["mod_anova"], res["subjects"],
        x_label=r"Empirical D1, trial_count   $1-\alpha^p$",
        y_label=r"Twin (ANOVA)   $1-\alpha^p$",
        title="A  reference: §A.8.6 gap")
    _scatter_panel(
        axes[1], res["twin_trial"], res["mod_anova"], res["subjects"],
        x_label=r"Twin D1, trial_count   $1-\alpha^p$",
        y_label=r"Twin (ANOVA)   $1-\alpha^p$",
        title="B  same rhat, two estimators")
    _scatter_panel(
        axes[2], res["emp_pair"], res["twin_pair"], res["subjects"],
        x_label=r"Empirical D1, pair_count   $1-\alpha^p$",
        y_label=r"Twin D1, pair_count   $1-\alpha^p$",
        title="C  matched estimator, pair_count")
    _scatter_panel(
        axes[3], res["emp_trial"], res["twin_trial"], res["subjects"],
        x_label=r"Empirical D1, trial_count   $1-\alpha^p$",
        y_label=r"Twin D1, trial_count   $1-\alpha^p$",
        title="D  matched estimator, trial_count")
    fig.tight_layout()
    save(fig, "fig_panel_d_closepair.png")

    print()
    print("axis label is (x, y); median(x-y) > 0 means y lower than x")
    _report("A  emp D1 trial   vs  twin ANOVA",
            res["emp_trial"], res["mod_anova"])
    _report("B  twin D1 trial  vs  twin ANOVA",
            res["twin_trial"], res["mod_anova"])
    _report("C  emp D1 pair    vs  twin D1 pair",
            res["emp_pair"], res["twin_pair"])
    _report("D  emp D1 trial   vs  twin D1 trial",
            res["emp_trial"], res["twin_trial"])
    print()
    print("for reference, twin D1 pair vs twin ANOVA:")
    _report("    twin D1 pair  vs  twin ANOVA",
            res["twin_pair"], res["mod_anova"])


if __name__ == "__main__":
    main()
