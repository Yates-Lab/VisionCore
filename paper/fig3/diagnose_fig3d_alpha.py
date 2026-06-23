r"""Diagnostic for panel D: why does model 1-alpha not reach low values, and
why is the Logan correlation weak?

Read-only; uses the existing fig3 cache (no GPU/model load). Computes, per good
cell, several quantities on the affine-matched scale (rhat_used is rescaled to
robs_used) to discriminate:

  H1 PSTH under-prediction  -> Delta correlates with low ccnorm; model PSTH var
                               < data PSTH var.
  H2 irreducible FEM floor  -> floor on model 1-alpha persists even for well-fit
                               (high-ccnorm) cells.
  H3 empirical bias/scale   -> a Poisson-corrected data 1-alpha on the matched
                               scale also exceeds fig2's emp 1-alpha.
  H4 Logan range/N          -> spread of emp 1-alpha and trials/phase by subject.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr

from VisionCore.covariance import rate_variance_components, psth_variance_splithalf
from _fig3_data import SUBJECTS, load_fig3_data

MIN_TPP = 10


def pooled_stats(x, valid):
    """Pooled mean and ddof=1 variance over valid (trial, phase) samples."""
    v = valid & np.isfinite(x)
    vals = x[v]
    if vals.size < 2:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.var(ddof=1))


def main():
    data = load_fig3_data()
    sr_all = data["session_results"]
    vidx = data["valid_indices"]
    ats = data["all_trace_neuron_session"]
    alpha = data["alpha"]
    good = data["good"]
    subjects = data["subjects"]
    ccnorm = data["ccnorm"]
    rhos = data["rhos"]

    rows = []
    for k in range(len(alpha)):
        if not good[k] or not np.isfinite(alpha[k]):
            continue
        si, ni = ats[vidx[k]]
        sr = sr_all[si]
        rhat = sr["rhat_used"][:, :, ni]
        robs = sr["robs_used"][:, :, ni]
        valid = sr["dfs_used"][:, :, ni] != 0

        m = rate_variance_components(rhat, valid=valid, min_trials_per_phase=MIN_TPP)
        if not np.isfinite(m["one_minus_alpha"]):
            continue

        # data side, same estimator + scale (Poisson-subtracted total)
        d_psth = psth_variance_splithalf(robs, valid=valid,
                                         min_trials_per_phase=MIN_TPP,
                                         n_boot=50, seed=0)
        mean_c, var_pool = pooled_stats(robs, valid)
        d_crate = var_pool - mean_c               # Poisson: obs var = signal var + rate
        d_cfem = d_crate - d_psth
        d_1ma = (np.clip(d_cfem / d_crate, 0, 1)
                 if np.isfinite(d_crate) and d_crate > 0 else np.nan)

        m_psth = psth_variance_splithalf(rhat, valid=valid,
                                         min_trials_per_phase=MIN_TPP,
                                         n_boot=50, seed=0)

        # trials/phase (median over kept phases)
        n_t = valid.sum(axis=0)
        tpp = float(np.median(n_t[n_t >= MIN_TPP])) if (n_t >= MIN_TPP).any() else np.nan

        rows.append(dict(
            subj=subjects[k],
            emp=1.0 - alpha[k],
            model=m["one_minus_alpha"],
            s2w=m["sigma2_within"], s2b=m["sigma2_between"],
            m_psthvar=m_psth, d_psthvar=d_psth,
            d_1ma=d_1ma,
            ccnorm=ccnorm[k], rho=rhos[k], tpp=tpp,
        ))

    R = {key: np.array([r[key] for r in rows], dtype=object if key == "subj" else float)
         for key in rows[0]}
    Delta = R["model"] - R["emp"]
    subj = R["subj"].astype(str)

    print(f"\n=== Panel D diagnostic: N={len(rows)} good cells ===")

    # ---- H2: floor on model 1-alpha ----
    print("\n[Floor] model 1-alpha distribution:")
    for p in (1, 5, 10, 25, 50):
        print(f"   p{p:02d} = {np.nanpercentile(R['model'], p):.3f}")
    print(f"   min  = {np.nanmin(R['model']):.3f}")

    # ---- H1 vs H2: does Delta survive for well-fit cells? ----
    print("\n[H1 vs H2] Delta = model - emp:")
    print(f"   corr(Delta, ccnorm): r={pearsonr(Delta, R['ccnorm'])[0]:+.3f}, "
          f"rho={spearmanr(Delta, R['ccnorm']).correlation:+.3f}")
    print(f"   corr(Delta, rho)   : r={pearsonr(Delta, R['rho'])[0]:+.3f}, "
          f"rho={spearmanr(Delta, R['rho']).correlation:+.3f}")
    low = R["emp"] < 0.5
    print(f"\n   Low-empirical cells (emp 1-alpha < 0.5): N={low.sum()}")
    for lab, m in [("high ccnorm (>0.95)", low & (R["ccnorm"] > 0.95)),
                   ("low  ccnorm (<0.95)", low & (R["ccnorm"] <= 0.95))]:
        if m.sum() >= 3:
            print(f"     {lab}: N={m.sum()}, median Delta={np.nanmedian(Delta[m]):+.3f}, "
                  f"median model 1-a={np.nanmedian(R['model'][m]):.3f}, "
                  f"median emp 1-a={np.nanmedian(R['emp'][m]):.3f}")

    # ---- H1: PSTH variance ratio (model / data), matched scale ----
    ratio = R["m_psthvar"] / R["d_psthvar"]
    ok = np.isfinite(ratio) & (R["d_psthvar"] > 0)
    print("\n[H1] model PSTH var / data PSTH var (matched scale, split-half):")
    print(f"   all good : median={np.nanmedian(ratio[ok]):.3f} (N={ok.sum()})")
    print(f"   low emp  : median={np.nanmedian(ratio[ok & low]):.3f} (N={(ok&low).sum()})")
    print(f"   high emp : median={np.nanmedian(ratio[ok & ~low]):.3f} (N={(ok&~low).sum()})")

    # ---- H3: Poisson-corrected data 1-alpha on matched scale vs fig2 emp ----
    okd = np.isfinite(R["d_1ma"])
    print("\n[H3] data 1-alpha recomputed on matched scale (Poisson-subtracted) "
          "vs fig2 emp 1-alpha:")
    print(f"   median matched-data 1-a={np.nanmedian(R['d_1ma'][okd]):.3f}, "
          f"median fig2 emp 1-a={np.nanmedian(R['emp'][okd]):.3f}")
    print(f"   corr(matched-data, fig2 emp): "
          f"rho={spearmanr(R['d_1ma'][okd], R['emp'][okd]).correlation:.3f}")
    print(f"   corr(model, matched-data)   : "
          f"rho={spearmanr(R['model'][okd], R['d_1ma'][okd]).correlation:.3f}")

    # ---- H4: Logan range / N / trials-per-phase ----
    print("\n[H4] per-subject spread and sampling:")
    for s in SUBJECTS:
        ms = subj == s
        if ms.sum() == 0:
            continue
        print(f"   {s}: N={ms.sum()}, emp 1-a std={np.nanstd(R['emp'][ms]):.3f} "
              f"[range {np.nanmin(R['emp'][ms]):.2f}-{np.nanmax(R['emp'][ms]):.2f}], "
              f"median trials/phase={np.nanmedian(R['tpp'][ms]):.0f}, "
              f"median ccnorm={np.nanmedian(R['ccnorm'][ms]):.2f}")


if __name__ == "__main__":
    main()
