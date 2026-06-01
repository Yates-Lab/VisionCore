r"""Appendix §A.8 real-data figure: panel D, naive → matched (pair_count) →
matched (trial_count).

For each fig4 panel-D good cell (`data['good'] & isfinite(data['alpha'])`):

  A. **Naive baseline.** Empirical $1-\alpha$ from `data['alpha']` (production
     pipeline, the naive close-pair / all-pair mix of §4.3) plotted against
     the twin ANOVA from `rate_variance_components(rhat, dfs!=0)`. This is
     fig4 panel D as published — the reference scatter.

  B. **Matched target only.** Same twin axis; empirical axis swapped to
     `decompose(target='full', time_bin_weighting='pair_count')` per cell
     with cell-specific validity (`dfs!=0`). Both axes now target
     $1-\alpha^p$ (Direction 1 of §4.4) — but the empirical close-pair
     estimator still uses the default pair_count bin weighting
     ($w_t \propto n_t(n_t-1)/2$), while the twin ANOVA's effective bin
     weighting is $w_t \propto n_t$. The mismatch leaves cells correlated
     between $\alpha(t)$ and $n_t$ (the fixRSVP transient) with residual
     bias.

  C. **Matched target + matched time-bin weighting.** Empirical axis is
     `decompose(target='full', time_bin_weighting='trial_count')`. Now both
     axes target $1-\alpha^p$ AND share the $w_t \propto n_t$ effective bin
     weighting (§A.8.4). The residual (A1)-style bias from $\alpha$–$n_t$
     covariance is removed.

To keep the panel-by-panel comparison apples-to-apples, both B and C use
the same close-pair / split-half estimator family with per-cell validity
(`_percell_close_pair_session` below). The function enumerates close pairs
ONCE per session and then applies the per-cell loop with the chosen
time-bin weighting, so the only difference between B and C is the
weighting flag.

Run from this folder:  uv run python fig_panel_d_anova.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr, pearsonr

# fig4 data loader lives in the sibling fig4/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fig4"))
from _fig4_data import (                                            # noqa: E402
    load_fig4_data, CCMAX_THRESHOLD, SUBJECTS, SUBJECT_COLORS,
)

from VisionCore.covariance import rate_variance_components          # noqa: E402
from _style import configure, save, C_TRUTH                         # noqa: E402


THR = 0.05
MIN_TPP = 10
MIN_TRIALS_PER_PHASE = 10
WEIGHT_CLIP = 1e6
N_BOOT_SPLIT = 20
SEED_SPLIT = 0


# ---------------------------------------------------------------------------
# Session-cached per-cell Direction-1 close-pair estimator with configurable
# time-bin weighting. Mirrors generate_realdata._percell_session but factored
# so pair_count and trial_count share the close-pair enumeration.
# ---------------------------------------------------------------------------

def _percell_close_pair_session(robs, eye, vm, dfs,
                                time_bin_weighting='trial_count',
                                threshold=THR, weight_clip=WEIGHT_CLIP):
    """Per-cell Direction-1 (`target='full'`) $1-\\alpha$ for one session.

    Close pairs are enumerated once on the session-level eye-finite mask
    `vm`. For each cell c, the per-cell validity is `Df[:, c] = dfs!=0`,
    intersected with the close-pair endpoint validity for the Crate
    second moment.

    Time-bin weighting (applied to Crate close-pair and Cpsth split-half
    consistently):

      'pair_count' : Crate per-pair w=1 (bin ~|P_t|), Cpsth wph_t=n_t(n_t-1)/2
      'uniform'    : Crate per-pair w=1/|P_t| (bin 1),  Cpsth wph_t=1
      'trial_count': Crate per-pair w=n_t/|P_t| (bin n_t), Cpsth wph_t=n_t

    Erate (the subtractor `Crate = mm - er²`) uses per-sample time-bin
    weighting matching the chosen scheme so the subtraction is consistent
    (the analogue of `decompose`'s `sw = tw * time_bin_w`).
    """
    n_tr, n_ph, C = robs.shape
    S = np.nan_to_num(robs, nan=0.0)
    Dfs = (dfs != 0)
    samp = np.where(vm)
    si_tr, si_ph = samp
    Sf = S[si_tr, si_ph, :]                  # (N, C)
    Ef = eye[si_tr, si_ph, :]                # (N, 2)
    Df = Dfs[si_tr, si_ph, :]                # (N, C) per-cell validity
    Tf = si_ph

    kde = gaussian_kde(Ef.T)

    # Close pairs, session-level, once
    pi_acc, pj_acc, tp_acc = [], [], []
    for t in np.unique(Tf):
        ix = np.where(Tf == t)[0]
        if len(ix) < 2:
            continue
        a, b = np.triu_indices(len(ix), k=1)
        d = np.linalg.norm(Ef[ix[a]] - Ef[ix[b]], axis=1)
        cmask = d < threshold
        if not cmask.any():
            continue
        pi_acc.append(ix[a[cmask]])
        pj_acc.append(ix[b[cmask]])
        tp_acc.append(np.full(int(cmask.sum()), t))
    if not pi_acc:
        return np.full(C, np.nan)
    pi = np.concatenate(pi_acc)
    pj = np.concatenate(pj_acc)
    tpair = np.concatenate(tp_acc)
    mid = 0.5 * (Ef[pi] + Ef[pj])

    pm = np.clip(kde(mid.T), 1e-12, None)
    pw_full = np.clip(1.0 / pm, None,
                      weight_clip * np.median(1.0 / pm))

    SiSj = Sf[pi] * Sf[pj]                   # (P, C)
    Vp = Df[pi] & Df[pj]                     # (P, C) per-cell pair valid

    one_minus_alpha = np.full(C, np.nan)
    for c in range(C):
        vc = Df[:, c]
        vpc = Vp[:, c]
        if vc.sum() < 2 * MIN_TPP or vpc.sum() < MIN_TPP:
            continue
        sc = Sf[vc, c]
        Tf_c = Tf[vc]
        Tpair_c = tpair[vpc]
        nt_by_bin = {int(t): int((Tf_c == t).sum()) for t in np.unique(Tf_c)}
        # Drop cell-valid pairs whose bin is below MIN_TPP for the cell.
        keep_pair = np.array(
            [nt_by_bin.get(int(t), 0) >= MIN_TPP for t in Tpair_c],
            dtype=bool,
        )
        if keep_pair.sum() < MIN_TPP:
            continue

        # Per-pair time-bin weight
        unique_tp, inv = np.unique(Tpair_c[keep_pair], return_inverse=True)
        m_t_by_bin = np.bincount(inv).astype(float)
        if time_bin_weighting == "pair_count":
            pw_t = np.ones(int(keep_pair.sum()))
        elif time_bin_weighting == "uniform":
            pw_t = 1.0 / m_t_by_bin[inv]
        elif time_bin_weighting == "trial_count":
            n_t_per_pair = np.array(
                [nt_by_bin[int(t)] for t in Tpair_c[keep_pair]], dtype=float)
            pw_t = n_t_per_pair / m_t_by_bin[inv]
        else:
            raise ValueError(
                f"unknown time_bin_weighting: {time_bin_weighting!r}")

        # Per-sample time-bin weight for Erate (matching scheme)
        if time_bin_weighting == "pair_count":
            n_per_samp = np.array([nt_by_bin[int(t)] for t in Tf_c], dtype=float)
            tb_sw = (n_per_samp - 1) / 2.0
        elif time_bin_weighting == "uniform":
            n_per_samp = np.array([nt_by_bin[int(t)] for t in Tf_c], dtype=float)
            tb_sw = 1.0 / np.clip(n_per_samp, 1.0, None)
        else:                                          # 'trial_count'
            tb_sw = np.ones(int(vc.sum()))

        sc_w = tb_sw / tb_sw.sum()
        er = float(np.sum(sc_w * sc))

        # Crate via close-pair second moment with target × time-bin weights
        pw_pair = pw_full[vpc][keep_pair] * pw_t
        pw_pair_norm = pw_pair / pw_pair.sum()
        mm = float(np.sum(pw_pair_norm * SiSj[vpc, c][keep_pair]))
        crate = mm - er ** 2

        # Cpsth via split-half with matching wph_t
        cpsth = _split_half_psth_one_cell(
            sc, Tf_c, time_bin_weighting=time_bin_weighting,
            n_boot=N_BOOT_SPLIT, seed=SEED_SPLIT)

        if crate > 0:
            one_minus_alpha[c] = float(
                1.0 - np.clip(cpsth / crate, 0.0, 1.0))

    return one_minus_alpha


def _split_half_psth_one_cell(s, t_arr, time_bin_weighting='trial_count',
                              n_boot=N_BOOT_SPLIT, seed=SEED_SPLIT):
    """Split-half PSTH variance for one cell with the chosen time-bin
    weighting. Matches `estimators._split_half_psth_cov`'s wph_t scheme.
    """
    rng = np.random.default_rng(seed)
    time_bins = [u for u in np.unique(t_arr) if (t_arr == u).sum() >= MIN_TPP]
    if len(time_bins) < 2:
        return np.nan
    idx = {u: np.where(t_arr == u)[0] for u in time_bins}
    nt = np.array([len(idx[u]) for u in time_bins], dtype=float)
    if time_bin_weighting == "pair_count":
        wph = nt * (nt - 1) / 2.0
    elif time_bin_weighting == "uniform":
        wph = np.ones_like(nt)
    elif time_bin_weighting == "trial_count":
        wph = nt
    else:
        raise ValueError(
            f"unknown time_bin_weighting: {time_bin_weighting!r}")
    wph = wph / wph.sum()
    acc = 0.0
    for _ in range(n_boot):
        A, B = [], []
        for u in time_bins:
            ix = rng.permutation(idx[u])
            m = len(ix) // 2
            A.append(float(s[ix[:m]].mean()))
            B.append(float(s[ix[m:]].mean()))
        A = np.asarray(A); B = np.asarray(B)
        A_bar = float(np.sum(wph * A))
        B_bar = float(np.sum(wph * B))
        acc += float(np.sum(wph * (A - A_bar) * (B - B_bar)))
    return acc / n_boot


# ---------------------------------------------------------------------------
# Panel-D pairing: per-cell (production-naive, D1 pair-count, D1 trial-count,
# twin ANOVA), subject-aligned.
# ---------------------------------------------------------------------------

def compute():
    data = load_fig4_data()
    ats = data["all_trace_neuron_session"]
    valid_indices = data["valid_indices"]
    alpha_emp = data["alpha"]
    good = data["good"]
    subjects = data["subjects"]

    emp_prod, emp_pair, emp_trial, mod_anova, subj_out = [], [], [], [], []
    pair_cache, trial_cache = {}, {}

    for k in range(len(alpha_emp)):
        if not good[k] or not np.isfinite(alpha_emp[k]):
            continue
        si, ni = ats[valid_indices[k]]
        sr = data["session_results"][si]

        if si not in pair_cache:
            print(f"  session {si:2d}  {sr['session']:22s}"
                  f"  ({sr['n_neurons']} neurons)  pair_count ...")
            pair_cache[si] = _percell_close_pair_session(
                sr["robs_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="pair_count")
        if si not in trial_cache:
            print(f"  session {si:2d}  {sr['session']:22s}"
                  f"  ({sr['n_neurons']} neurons)  trial_count ...")
            trial_cache[si] = _percell_close_pair_session(
                sr["robs_used"], sr["eyepos_used"],
                sr["valid_mask"], sr["dfs_used"],
                time_bin_weighting="trial_count")

        d1_pair = pair_cache[si][ni]
        d1_trial = trial_cache[si][ni]
        if not (np.isfinite(d1_pair) and np.isfinite(d1_trial)):
            continue

        rhat = sr["rhat_used"][:, :, ni]
        vmask = sr["dfs_used"][:, :, ni] != 0
        anova = rate_variance_components(
            rhat, valid=vmask, min_trials_per_phase=MIN_TRIALS_PER_PHASE)
        m = anova["one_minus_alpha"]
        if not np.isfinite(m):
            continue

        emp_prod.append(1.0 - alpha_emp[k])
        emp_pair.append(d1_pair)
        emp_trial.append(d1_trial)
        mod_anova.append(m)
        subj_out.append(subjects[k])

    return {
        "emp_prod":  np.asarray(emp_prod),
        "emp_pair":  np.asarray(emp_pair),
        "emp_trial": np.asarray(emp_trial),
        "mod_anova": np.asarray(mod_anova),
        "subjects":  np.asarray(subj_out),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_panel(ax, x, y, subjects, x_label, title, show_y_label=True):
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
    ax.text(0.04, 0.96, f"all: $\\rho$ = {rho_all:.2f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    if show_y_label:
        ax.set_ylabel(r"Twin model (ANOVA)   $1-\alpha$")
    ax.set_title(title, pad=6)
    ax.legend(loc="lower right", fontsize=7, handlelength=0.6)


def _report(name, x, y):
    rho = spearmanr(x, y).correlation
    r   = pearsonr(x, y)[0]
    med_x, med_y = float(np.median(x)), float(np.median(y))
    print(f"  {name:32s}  N={len(x):3d}  "
          f"spearman={rho:.3f}  pearson={r:.3f}  "
          f"median emp={med_x:.3f}  median model={med_y:.3f}  "
          f"median(emp-model)={med_x-med_y:+.3f}")


def main():
    configure()
    print("fig_panel_d_anova.py  --  panel D with the matched empirical estimator")
    res = compute()
    n = len(res["mod_anova"])
    print(f"  {n} cells matched")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    _scatter_panel(
        axes[0], res["emp_prod"], res["mod_anova"], res["subjects"],
        x_label=r"Empirical (production, naive)   $1-\alpha$",
        title="A  fig4 panel D (reference)")
    _scatter_panel(
        axes[1], res["emp_pair"], res["mod_anova"], res["subjects"],
        x_label=r"Empirical D1, pair_count   $1-\alpha^p$",
        title="B  matched target, pair_count $w_t$",
        show_y_label=False)
    _scatter_panel(
        axes[2], res["emp_trial"], res["mod_anova"], res["subjects"],
        x_label=r"Empirical D1, trial_count   $1-\alpha^p$",
        title="C  matched target + ANOVA-matched $w_t$",
        show_y_label=False)
    fig.tight_layout()
    save(fig, "fig_panel_d_anova.png")

    print()
    _report("naive (production)",    res["emp_prod"],  res["mod_anova"])
    _report("D1 pair_count",         res["emp_pair"],  res["mod_anova"])
    _report("D1 trial_count",        res["emp_trial"], res["mod_anova"])


if __name__ == "__main__":
    main()
