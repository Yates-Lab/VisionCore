"""Stage-2 derived metrics for the methods-folder pipeline.

Per-cell 1-alpha distributions, slope-through-origin Fano (with session-
clustered bootstrap CIs), and Fisher-z noise correlation means (with shuffle
nulls). Lifted from ``legacy.compute_fig2_data._compute_{metrics,alpha_stats,
fano_stats,nc_stats}`` and ``_clustered_slope_bootstrap`` with two changes:

  1. Loops over targets (legacy was implicitly naive). Each top-level stats
     dict is keyed by (target, window_ms).
  2. Consumes the methods ``windows[w]['targets'][target]`` layout instead of
     legacy ``mats[w]``. Inclusion filters and Fisher-z / bootstrap formulae
     are byte-identical -- this file's only job is to feed the same numbers
     to both pipelines.

Deferred (matches plan §scope): subspace overlaps, eigenspectra, ANOVA panel
D.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from legacy.covariance import cov_to_corr, get_upper_triangle             # noqa: E402
from legacy.subspace import project_to_psd                                # noqa: E402
from legacy.stats import (                                                # noqa: E402
    geomean, iqr_25_75, bootstrap_mean_ci, fisher_z_mean, emp_p_one_sided,
    wilcoxon_signed_rank, paired_valid,
)

# Inclusion constants -- match legacy compute_fig2_data
MIN_RATE_HZ = 2.0
MIN_PSTH_R2 = 0.05
MIN_VAR = 0
EPS_RHO = 1e-3
SUBJECTS = ["Allen", "Logan", "Luke"]


# ---------------------------------------------------------------------------
# Per-(session, window, target) metric extraction
# ---------------------------------------------------------------------------

def _metrics_one_methods(sr, w_idx, tgt):
    """Per-(window, session, target) metric block, methods layout."""
    if w_idx >= len(sr["windows"]):
        return None
    w = sr["windows"][w_idx]
    if tgt not in w["targets"]:
        return None
    Ctotal = w["Ctotal"]
    block = w["targets"][tgt]
    return _metrics_one_common(
        sr, Ctotal, block["Crate"], block["Cpsth"], block["Erate"],
        block.get("Shuffled_Crates", []), session=sr["session"],
        subject=sr["subject"], rate_hz=sr["rate_hz"], psth_r2=sr["psth_r2"],
    )


def _metrics_one_legacy(sr, w_idx):
    """Legacy layout: results[w]['Erates'], mats[w]['Total/PSTH/Intercept']."""
    if w_idx >= len(sr["results"]):
        return None
    res = sr["results"][w_idx]
    mats = sr["mats"][w_idx]
    return _metrics_one_common(
        sr, mats["Total"], mats["Intercept"], mats["PSTH"], res["Erates"],
        mats.get("Shuffled_Intercepts", []), session=sr["session"],
        subject=sr["subject"], rate_hz=sr["rate_hz"], psth_r2=sr["psth_r2"],
    )


def _metrics_one_common(sr, Ctotal, Crate, Cpsth, erate, shuffled_crates,
                        session, subject, rate_hz, psth_r2):
    """Shared inclusion-filter + Fisher-z work, identical to legacy
    ``_metrics_one``. Kept as a single function so the two layouts feed in
    via thin wrappers above."""
    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

    valid = (
        np.isfinite(erate)
        & np.isfinite(rate_hz)
        & (rate_hz > MIN_RATE_HZ)
        & np.isfinite(psth_r2)
        & (psth_r2 > MIN_PSTH_R2)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < 3:
        return None

    diag_psth = np.diag(Cpsth)[valid]
    diag_rate = np.diag(Crate)[valid]
    alpha = diag_psth / diag_rate

    ff_u = np.diag(CnoiseU)[valid] / erate[valid]
    ff_c = np.diag(CnoiseC)[valid] / erate[valid]

    NoiseCorrU = cov_to_corr(
        project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    NoiseCorrC = cov_to_corr(
        project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    rho_u_full = get_upper_triangle(NoiseCorrU)
    rho_c_full = get_upper_triangle(NoiseCorrC)
    pair_ok = np.isfinite(rho_u_full) & np.isfinite(rho_c_full)
    rho_u = rho_u_full[pair_ok]
    rho_c = rho_c_full[pair_ok]

    if len(rho_u) > 0:
        rho_u_meanz = fisher_z_mean(rho_u, eps=EPS_RHO)
        rho_c_meanz = fisher_z_mean(rho_c, eps=EPS_RHO)
        rho_delta_meanz = rho_c_meanz - rho_u_meanz
    else:
        rho_u_meanz = rho_c_meanz = rho_delta_meanz = np.nan

    n_valid = int(valid.sum())
    shuff_alphas = []
    ds_shuff_var_c = []
    shuff_rho_c_meanz_list = []
    shuff_rho_delta_meanz_list = []
    shuff_rho_subject_list = []
    diag_rate_dim = diag_rate.size
    for Crate_shuf in shuffled_crates:
        diag_rate_shuf = np.diag(Crate_shuf)[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_shuf = diag_psth / diag_rate_shuf
        shuff_alphas.append(1 - alpha_shuf)

        CnoiseC_shuf = Ctotal - Crate_shuf
        CnoiseC_shuf = 0.5 * (CnoiseC_shuf + CnoiseC_shuf.T)
        ds_shuff_var_c.append(np.diag(CnoiseC_shuf)[valid])
        NC_shuf = cov_to_corr(
            project_to_psd(CnoiseC_shuf[np.ix_(valid, valid)]),
            min_var=MIN_VAR,
        )
        rho_c_shuf = get_upper_triangle(NC_shuf)
        ok = np.isfinite(rho_c_shuf) & pair_ok
        if ok.sum() > 0:
            shuff_rho_c_meanz_list.append(
                fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
            )
            shuff_rho_delta_meanz_list.append(
                fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                - fisher_z_mean(rho_u_full[ok[:len(rho_u_full)]], eps=EPS_RHO)
            )
            shuff_rho_subject_list.append(subject)

    _ = diag_rate_dim  # silence unused

    return dict(
        subject=subject, session=session, n_valid=n_valid,
        alpha=alpha, ff_uncorr=ff_u, ff_corr=ff_c, erate=erate[valid],
        rho_uncorr=rho_u, rho_corr=rho_c,
        rho_u_meanz=rho_u_meanz, rho_c_meanz=rho_c_meanz,
        rho_delta_meanz=rho_delta_meanz,
        Ctotal=Ctotal[np.ix_(valid, valid)],
        Cpsth=Cpsth[np.ix_(valid, valid)],
        Crate=Crate[np.ix_(valid, valid)],
        CnoiseU=CnoiseU[np.ix_(valid, valid)],
        CnoiseC=CnoiseC[np.ix_(valid, valid)],
        Cfem=(Crate - Cpsth)[np.ix_(valid, valid)],
        shuff_alphas=shuff_alphas,
        ds_shuff_var_c=np.asarray(ds_shuff_var_c) if ds_shuff_var_c else None,
        shuff_rho_c_meanz=shuff_rho_c_meanz_list,
        shuff_rho_delta_meanz=shuff_rho_delta_meanz_list,
        shuff_rho_subject=shuff_rho_subject_list,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Per-target metric aggregation (Stage 2 -- mirrors legacy._compute_metrics)
# ---------------------------------------------------------------------------

def _aggregate_window_target(per_session_blocks):
    """Take a list of _metrics_one_common dicts (one per session) and stack
    them into the legacy per-window dict shape, plus the shuffle-pool matrix
    needed for the Fano slope null.
    """
    (
        all_alpha, all_ff_uncorr, all_ff_corr, all_erate,
        all_rho_uncorr, all_rho_corr,
        rho_u_meanz_by_ds, rho_c_meanz_by_ds, rho_delta_meanz_by_ds,
        all_Ctotal, all_Cpsth, all_Crate, all_CnoiseU, all_CnoiseC, all_Cfem,
        shuff_alphas, shuff_rho_delta_meanz, shuff_rho_c_meanz,
        shuff_rho_subject,
        subject_by_ds, subject_per_neuron, subject_per_pair, session_per_neuron,
        shuff_var_c_blocks, shuff_var_c_nvalid,
    ) = ([] for _ in range(25))

    for r in per_session_blocks:
        if r is None:
            continue
        subject_by_ds.append(r["subject"])
        all_alpha.append(r["alpha"])
        subject_per_neuron.extend([r["subject"]] * r["n_valid"])
        session_per_neuron.extend([r["session"]] * r["n_valid"])
        all_ff_uncorr.append(r["ff_uncorr"])
        all_ff_corr.append(r["ff_corr"])
        all_erate.append(r["erate"])
        all_rho_uncorr.append(r["rho_uncorr"])
        all_rho_corr.append(r["rho_corr"])
        subject_per_pair.extend([r["subject"]] * len(r["rho_uncorr"]))

        if len(r["rho_uncorr"]) > 0:
            rho_u_meanz_by_ds.append(r["rho_u_meanz"])
            rho_c_meanz_by_ds.append(r["rho_c_meanz"])
            rho_delta_meanz_by_ds.append(r["rho_delta_meanz"])

        all_Ctotal.append(r["Ctotal"])
        all_Cpsth.append(r["Cpsth"])
        all_Crate.append(r["Crate"])
        all_CnoiseU.append(r["CnoiseU"])
        all_CnoiseC.append(r["CnoiseC"])
        all_Cfem.append(r["Cfem"])

        shuff_alphas.extend(r["shuff_alphas"])
        shuff_rho_c_meanz.extend(r["shuff_rho_c_meanz"])
        shuff_rho_delta_meanz.extend(r["shuff_rho_delta_meanz"])
        shuff_rho_subject.extend(r["shuff_rho_subject"])

        shuff_var_c_blocks.append(r["ds_shuff_var_c"])
        shuff_var_c_nvalid.append(r["n_valid"])

    present_S = [b.shape[0] for b in shuff_var_c_blocks if b is not None]
    if present_S:
        S_min = min(present_S)
        rows = []
        for blk, nv in zip(shuff_var_c_blocks, shuff_var_c_nvalid):
            if blk is None:
                rows.append(np.full((nv, S_min), np.nan))
            else:
                rows.append(blk[:S_min].T)
        shuff_var_c = np.concatenate(rows, axis=0)
    else:
        shuff_var_c = np.empty((0, 0))

    return {
        "alpha": np.concatenate(all_alpha) if all_alpha else np.array([]),
        "uncorr": np.concatenate(all_ff_uncorr) if all_ff_uncorr else np.array([]),
        "corr": np.concatenate(all_ff_corr) if all_ff_corr else np.array([]),
        "erate": np.concatenate(all_erate) if all_erate else np.array([]),
        "rho_uncorr": (
            np.concatenate(all_rho_uncorr) if all_rho_uncorr else np.array([])
        ),
        "rho_corr": (
            np.concatenate(all_rho_corr) if all_rho_corr else np.array([])
        ),
        "rho_u_meanz_by_ds": np.array(rho_u_meanz_by_ds),
        "rho_c_meanz_by_ds": np.array(rho_c_meanz_by_ds),
        "rho_delta_meanz_by_ds": np.array(rho_delta_meanz_by_ds),
        "subject_by_ds": subject_by_ds,
        "subject_per_neuron": np.array(subject_per_neuron),
        "session_per_neuron": np.array(session_per_neuron),
        "subject_per_pair": np.array(subject_per_pair),
        "shuff_var_c": shuff_var_c,
        "Ctotal": all_Ctotal, "Cpsth": all_Cpsth, "Crate": all_Crate,
        "CnoiseU": all_CnoiseU, "CnoiseC": all_CnoiseC, "Cfem": all_Cfem,
        "shuff_alphas": shuff_alphas,
        "shuff_rho_delta_meanz": np.array(shuff_rho_delta_meanz),
        "shuff_rho_c_meanz": np.array(shuff_rho_c_meanz),
        "shuff_rho_subject": np.array(shuff_rho_subject),
    }


# ---------------------------------------------------------------------------
# Clustered slope-through-origin Fano bootstrap (verbatim from legacy)
# ---------------------------------------------------------------------------

def _slope_through_origin(erate, var):
    ok = np.isfinite(erate) & np.isfinite(var) & (erate > 0) & (var >= 0)
    if ok.sum() < 3:
        return np.nan
    e, v = erate[ok], var[ok]
    return float(np.sum(e * v) / np.sum(e ** 2))


def _clustered_slope_bootstrap(erate, var_u, var_c, sessions, nboot=5000, seed=0):
    """Verbatim from legacy.compute_fig2_data."""
    erate = np.asarray(erate, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    var_c = np.asarray(var_c, dtype=float)
    sessions = np.asarray(sessions)
    uniq = np.unique(sessions)
    rng = np.random.default_rng(seed)

    if uniq.size >= 2:
        K = uniq.size
        sum_ee = np.zeros(K); sum_eu = np.zeros(K); sum_ec = np.zeros(K)
        for i, s in enumerate(uniq):
            m = sessions == s
            e = erate[m]; vu = var_u[m]; vc = var_c[m]
            ok = np.isfinite(e) & np.isfinite(vu) & np.isfinite(vc)
            e, vu, vc = e[ok], vu[ok], vc[ok]
            sum_ee[i] = np.sum(e * e)
            sum_eu[i] = np.sum(e * vu)
            sum_ec[i] = np.sum(e * vc)

        draws = rng.integers(0, K, size=(nboot, K))
        D_ee = sum_ee[draws].sum(axis=1)
        D_eu = sum_eu[draws].sum(axis=1)
        D_ec = sum_ec[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
            sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)
    else:
        ok = np.isfinite(erate) & np.isfinite(var_u) & np.isfinite(var_c)
        e_v = erate[ok]; vu_v = var_u[ok]; vc_v = var_c[ok]
        if e_v.size == 0:
            su = np.full(nboot, np.nan); sc = np.full(nboot, np.nan)
        else:
            idx = rng.integers(0, e_v.size, size=(nboot, e_v.size))
            E = e_v[idx]
            D_ee = np.sum(E * E, axis=1)
            D_eu = np.sum(E * vu_v[idx], axis=1)
            D_ec = np.sum(E * vc_v[idx], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
                sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)

    diff = su - sc

    def _ci(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    diff_f = diff[np.isfinite(diff)]
    p = (float((np.sum(diff_f <= 0) + 1) / (diff_f.size + 1))
         if diff_f.size else np.nan)
    return {"unc_ci": _ci(su), "cor_ci": _ci(sc), "diff_ci": _ci(diff),
            "p": p, "n_sessions": int(uniq.size)}


# ---------------------------------------------------------------------------
# Public stats helpers (one block per target × window)
# ---------------------------------------------------------------------------

def _compute_alpha_stats_one(m):
    alpha = m["alpha"]
    m_raw = 1 - alpha
    in_range = np.isfinite(m_raw) & (m_raw >= 0.0) & (m_raw <= 1.0)
    n_total = int(np.isfinite(m_raw).sum())
    n_dropped = int(n_total - in_range.sum())
    mv = m_raw[in_range]

    mean_m, (ci_lo, ci_hi) = bootstrap_mean_ci(mv, nboot=5000, seed=0)
    med_m = float(np.nanmedian(mv))
    q25, q75 = iqr_25_75(mv)

    shuff_m = [
        s[np.isfinite(s) & (s >= 0.0) & (s <= 1.0)]
        for s in m["shuff_alphas"]
    ]
    shuff_m = [s for s in shuff_m if s.size > 0]
    if shuff_m:
        null_means = np.array([np.nanmean(s) for s in shuff_m])
        null_mean_ci = (float(np.percentile(null_means, 2.5)),
                        float(np.percentile(null_means, 97.5)))
        p_emp = emp_p_one_sided(null_means, mean_m, direction="less")
    else:
        null_mean_ci = (np.nan, np.nan)
        p_emp = np.nan

    return {
        "n": len(mv), "mean": mean_m, "ci": (ci_lo, ci_hi),
        "median": med_m, "iqr": (q25, q75),
        "null_ci": null_mean_ci, "p_emp": p_emp,
        "n_dropped": n_dropped, "n_total": n_total,
        "m": mv, "subject_per_neuron": m["subject_per_neuron"][in_range],
    }


def _compute_fano_stats_one(m):
    ff_u, ff_c, erate = m["uncorr"], m["corr"], m["erate"]
    ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
    erate_v = erate[mask]
    subject_labels_v = m["subject_per_neuron"][mask]
    session_labels_v = m["session_per_neuron"][mask]

    g_unc = geomean(ff_u_v); g_cor = geomean(ff_c_v)
    ratio = g_cor / g_unc; pct_red = (1 - ratio) * 100
    _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

    var_u = ff_u_v * erate_v; var_c = ff_c_v * erate_v
    slope_unc = _slope_through_origin(erate_v, var_u)
    slope_cor = _slope_through_origin(erate_v, var_c)
    boot = _clustered_slope_bootstrap(
        erate_v, var_u, var_c, session_labels_v, nboot=5000, seed=0
    )
    slope_diff = slope_unc - slope_cor

    shuff_var_c = m.get("shuff_var_c", np.empty((0, 0)))
    if shuff_var_c.size and shuff_var_c.shape[0] == mask.shape[0]:
        svc = shuff_var_c[mask]
        null_slope_cor = np.array([
            _slope_through_origin(erate_v, svc[:, b])
            for b in range(svc.shape[1])
        ])
        null_slope_cor = null_slope_cor[np.isfinite(null_slope_cor)]
    else:
        null_slope_cor = np.array([])

    if null_slope_cor.size:
        null_reduction = slope_unc - null_slope_cor
        obs_reduction = slope_unc - slope_cor
        slope_cor_null_ci = (float(np.percentile(null_slope_cor, 2.5)),
                             float(np.percentile(null_slope_cor, 97.5)))
        p_emp_slope = emp_p_one_sided(null_reduction, obs_reduction,
                                      direction="greater")
    else:
        slope_cor_null_ci = (np.nan, np.nan); p_emp_slope = np.nan

    return {
        "n": len(ff_u_v), "g_unc": g_unc, "g_cor": g_cor,
        "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
        "slope_unc": slope_unc, "slope_cor": slope_cor,
        "slope_unc_ci": boot["unc_ci"], "slope_cor_ci": boot["cor_ci"],
        "slope_diff": slope_diff, "slope_diff_ci": boot["diff_ci"],
        "p_slope": boot["p"], "n_sessions": boot["n_sessions"],
        "slope_cor_null_ci": slope_cor_null_ci, "p_emp_slope": p_emp_slope,
        "erate": erate_v, "var_u": var_u, "var_c": var_c,
        "subject_per_neuron": subject_labels_v,
        "session_per_neuron": session_labels_v,
    }


def _compute_nc_stats_one(m):
    rho_u, rho_c = m["rho_uncorr"], m["rho_corr"]
    z_u_ds = m["rho_u_meanz_by_ds"]
    z_c_ds = m["rho_c_meanz_by_ds"]
    dz_ds = m["rho_delta_meanz_by_ds"]
    n_ds = len(z_u_ds)

    z_u_mean, z_u_ci = bootstrap_mean_ci(z_u_ds, nboot=5000, seed=0)
    z_c_mean, z_c_ci = bootstrap_mean_ci(z_c_ds, nboot=5000, seed=0)
    dz_mean, dz_ci = bootstrap_mean_ci(dz_ds, nboot=5000, seed=0)

    if n_ds >= 5:
        _, p_wil = wilcoxon_signed_rank(z_c_ds, z_u_ds, alternative="less")
    else:
        p_wil = np.nan

    shuff_dz = m["shuff_rho_delta_meanz"]
    if len(shuff_dz) > 0:
        null_dz_ci = (float(np.percentile(shuff_dz, 2.5)),
                      float(np.percentile(shuff_dz, 97.5)))
        p_emp_dz = emp_p_one_sided(shuff_dz, dz_mean, direction="less")
    else:
        null_dz_ci = (np.nan, np.nan); p_emp_dz = np.nan

    return {
        "n_pairs": len(rho_u), "n_ds": n_ds,
        "z_u_mean": z_u_mean, "z_u_ci": z_u_ci,
        "z_c_mean": z_c_mean, "z_c_ci": z_c_ci,
        "dz_mean": dz_mean, "dz_ci": dz_ci,
        "p_wil": p_wil, "null_dz_ci": null_dz_ci, "p_emp_dz": p_emp_dz,
        "rho_u": rho_u, "rho_c": rho_c,
    }


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def derive_methods(session_results, windows_ms, windows_bins,
                   targets=("naive", "full", "central")):
    """Stage 2 over the methods per-session pipeline output.

    Returns:
        bundle dict with keys:
          windows_ms, windows_bins, targets,
          metrics[target][w]    -- per-(target, window) aggregate dict,
          alpha_stats[target]   -- {window_ms: alpha_stats_one}
          fano_stats[target]    -- {window_ms: fano_stats_one}
          nc_stats[target]      -- {window_ms: nc_stats_one}
          session_names, subjects
    """
    out = {"windows_ms": list(windows_ms),
           "windows_bins": list(windows_bins),
           "targets": list(targets),
           "metrics": {}, "alpha_stats": {}, "fano_stats": {}, "nc_stats": {},
           "session_names": [sr["session"] for sr in session_results],
           "subjects": [sr["subject"] for sr in session_results]}

    for tgt in targets:
        out["metrics"][tgt] = []
        out["alpha_stats"][tgt] = {}
        out["fano_stats"][tgt] = {}
        out["nc_stats"][tgt] = {}
        for w_idx, w_ms in enumerate(windows_ms):
            blocks = [_metrics_one_methods(sr, w_idx, tgt) for sr in session_results]
            agg = _aggregate_window_target(blocks)
            agg["window_ms"] = w_ms
            agg["window_bins"] = windows_bins[w_idx]
            out["metrics"][tgt].append(agg)
            out["alpha_stats"][tgt][w_ms] = _compute_alpha_stats_one(agg)
            out["fano_stats"][tgt][w_ms] = _compute_fano_stats_one(agg)
            out["nc_stats"][tgt][w_ms] = _compute_nc_stats_one(agg)
    return out


def derive_legacy(session_results, windows_ms, windows_bins):
    """Stage 2 over the legacy per-session pipeline output (single-target =
    'naive' by construction). Output keyed by 'naive' for symmetry with the
    methods bundle so figures can iterate identically.
    """
    out = {"windows_ms": list(windows_ms),
           "windows_bins": list(windows_bins),
           "targets": ["naive"],
           "metrics": {"naive": []},
           "alpha_stats": {"naive": {}}, "fano_stats": {"naive": {}},
           "nc_stats": {"naive": {}},
           "session_names": [sr["session"] for sr in session_results],
           "subjects": [sr["subject"] for sr in session_results]}

    for w_idx, w_ms in enumerate(windows_ms):
        blocks = [_metrics_one_legacy(sr, w_idx) for sr in session_results]
        agg = _aggregate_window_target(blocks)
        agg["window_ms"] = w_ms
        agg["window_bins"] = windows_bins[w_idx]
        out["metrics"]["naive"].append(agg)
        out["alpha_stats"]["naive"][w_ms] = _compute_alpha_stats_one(agg)
        out["fano_stats"]["naive"][w_ms] = _compute_fano_stats_one(agg)
        out["nc_stats"]["naive"][w_ms] = _compute_nc_stats_one(agg)
    return out
