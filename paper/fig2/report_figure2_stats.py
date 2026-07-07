"""Single source of truth for every Figure 2 number quoted in the manuscript.

This dumps the exact pooled (and per-subject) statistics that the Figure 2
panels are built from, so the prose in ``main_v2.tex`` and the figure caption
can be kept in lockstep with ``generate_figure2.py``. Anything reported in the
text should be traceable to a line printed here.

Panels:
    C  1 - alpha          FEM fraction of single-neuron rate variance
    E  Fano               single-neuron geometric-mean ratio + population slope
    F  noise correlation  Fisher-z mean, uncorrected vs FEM-corrected
    G  participation ratio corrected residual vs stimulus vs FEM components
    I  subspace alignment  variance captured (both directions) + overlap_k1

Pooled values match the default figure (subjects pooled, Luke omitted);
per-subject values match ``--split-subjects`` and the per-subject caption lines.

Usage:
    uv run paper/fig2/report_figure2_stats.py
    uv run paper/fig2/report_figure2_stats.py --refresh
"""
import argparse

import numpy as np
from scipy.stats import binomtest, wilcoxon

from compute_fig2_data import (
    load_fig2_data, _compute_alpha_stats, compute_alignment_aggregate,
)
from generate_figure2 import _filter_subjects, OMIT_SUBJECTS

SUBJECT_DISPLAY = {"Allen": "Monkey A", "Logan": "Monkey L", "Pooled": "pooled"}


def _fmt(x, p=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def _ci(t, p=3):
    return f"[{_fmt(t[0], p)}, {_fmt(t[1], p)}]"


def _median_iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def _mean_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if x.size > 1 else np.nan), x.size


def _wilcox(a, b, alt="two-sided"):
    try:
        return wilcoxon(a, b, alternative=alt).pvalue
    except ValueError:
        return np.nan


def _signtest(a, b, alt="two-sided"):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return np.nan, 0, 0
    n_pos = int(np.sum(d > 0))
    return binomtest(n_pos, d.size, 0.5, alternative=alt).pvalue, n_pos, d.size


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Panel C: 1 - alpha (FEM fraction of single-neuron rate variance)
# --------------------------------------------------------------------------- #
def report_alpha(bundles):
    section("PANEL C  -  1 - alpha  (FEM fraction of single-neuron rate variance)")
    windows = bundles["pooled"]["WINDOWS_MS"]
    for label, b in bundles.items():
        _, _, astats = _compute_alpha_stats(b["metrics"], windows)
        print(f"\n  [{SUBJECT_DISPLAY.get(label, label)}]")
        for w in windows:
            s = astats[w]
            line = (f"    {w:5.1f} ms: n={s['n']:4d}  "
                    f"median={_fmt(s['median'])} IQR=[{_fmt(s['iqr'][0])}, "
                    f"{_fmt(s['iqr'][1])}]  mean={_fmt(s['mean'])} "
                    f"95%CI={_ci(s['ci'])}")
            if label == "pooled":
                line += (f"  shuffle-null median 95%={_ci(s['null_median_ci'])} "
                         f"p_emp(median)={_fmt(s['p_emp_median'], 4)}")
            print(line)


# --------------------------------------------------------------------------- #
# Panel E: Fano factor (single-neuron ratio + population slope)
# --------------------------------------------------------------------------- #
def report_fano(pooled):
    section("PANEL E  -  Fano factor  (single-neuron geomean ratio + population slope)")
    windows = pooled["WINDOWS_MS"]
    fs = pooled["fano_stats"]
    print("\n  Pooled across subjects:")
    for w in windows:
        s = fs[w]
        print(f"    {w:5.1f} ms: n={s['n']:4d}  single-neuron geomean "
              f"{_fmt(s['g_unc'])} -> {_fmt(s['g_cor'])} "
              f"(ratio {_fmt(s['ratio'])}, {_fmt(s['pct_red'], 1)}% reduction; "
              f"shuffle-null geomean 95%={_ci(s['g_cor_null_ci'])} "
              f"p_emp={_fmt(s['p_emp_geomean'], 4)})")
        print(f"             population slope {_fmt(s['slope_unc'])} -> "
              f"{_fmt(s['slope_cor'])}  Delta={_fmt(s['slope_diff'])} "
              f"95%CI={_ci(s['slope_diff_ci'])}  p_slope={_fmt(s['p_slope'], 4)}  "
              f"shuffle-null p_emp={_fmt(s['p_emp_slope'], 4)} "
              f"(n_sessions={s['n_sessions']})")
    for subj in ("Allen", "Logan"):
        print(f"\n  [{SUBJECT_DISPLAY[subj]}] population slope:")
        for w in windows:
            ps = fs[w]["per_subject"].get(subj)
            if ps is None:
                continue
            print(f"    {w:5.1f} ms: slope {_fmt(ps['slope_unc'])} -> "
                  f"{_fmt(ps['slope_cor'])}  Delta={_fmt(ps['slope_diff'])} "
                  f"95%CI={_ci(ps['slope_diff_ci'])}  p={_fmt(ps['p_slope'], 4)} "
                  f"(n_sessions={ps['n_sessions']}, n={ps['n']})")


# --------------------------------------------------------------------------- #
# Panel F: noise correlations (Fisher-z)
# --------------------------------------------------------------------------- #
def report_noisecorr(pooled):
    section("PANEL F  -  noise correlations  (Fisher-z, uncorrected vs FEM-corrected)")
    windows = pooled["WINDOWS_MS"]
    ns = pooled["nc_stats"]
    for w in windows:
        s = ns[w]
        rho_u = np.tanh(s["z_u_mean"])
        rho_c = np.tanh(s["z_c_mean"])
        print(f"\n    {w:5.1f} ms: n_pairs={s['n_pairs']}  n_datasets={s['n_ds']}")
        print(f"      mean rho (from z): {_fmt(rho_u, 4)} -> {_fmt(rho_c, 4)}")
        print(f"      z_uncorr={_fmt(s['z_u_mean'], 4)} {_ci(s['z_u_ci'], 4)}  "
              f"z_corr={_fmt(s['z_c_mean'], 4)} {_ci(s['z_c_ci'], 4)}")
        print(f"      delta_z={_fmt(s['dz_mean'], 4)} {_ci(s['dz_ci'], 4)}  "
              f"Wilcoxon p={_fmt(s['p_wil'], 5)}")
        print(f"      shuffle-null delta_z 95%CI={_ci(s['null_dz_ci'], 4)}  "
              f"p_emp={_fmt(s['p_emp_dz'], 5)} (n_shuff={s['n_shuff_dz']})")


# --------------------------------------------------------------------------- #
# Panel G: participation ratio (corrected residual vs stimulus vs FEM)
# --------------------------------------------------------------------------- #
def report_pr(bundles):
    section("PANEL G  -  participation ratio  "
            f"(subspace window = {bundles['pooled']['SUBSPACE_WINDOW_IDX']}, "
            f"{bundles['pooled']['WINDOWS_MS'][bundles['pooled']['SUBSPACE_WINDOW_IDX']]:.1f} ms)")
    for label, b in bundles.items():
        resid = np.asarray(b.get("pr_resid_list", []), float)
        psth = np.asarray(b.get("pr_psth_list", []), float)
        fem = np.asarray(b.get("pr_fem_list", []), float)
        ok = np.isfinite(resid) & np.isfinite(psth) & np.isfinite(fem)
        resid, psth, fem = resid[ok], psth[ok], fem[ok]
        if resid.size == 0:
            continue
        rm = _median_iqr(resid); pm = _median_iqr(psth); fm = _median_iqr(fem)
        ra = _mean_sd(resid); pa = _mean_sd(psth); fa = _mean_sd(fem)
        print(f"\n  [{SUBJECT_DISPLAY.get(label, label)}]  n_sessions={resid.size}")
        print(f"    corrected residual: mean={_fmt(ra[0])}+-{_fmt(ra[1])}  "
              f"median={_fmt(rm[0])} IQR=[{_fmt(rm[1])}, {_fmt(rm[2])}]")
        print(f"    stimulus covariance: mean={_fmt(pa[0])}+-{_fmt(pa[1])}  "
              f"median={_fmt(pm[0])} IQR=[{_fmt(pm[1])}, {_fmt(pm[2])}]")
        print(f"    FEM component:       mean={_fmt(fa[0])}+-{_fmt(fa[1])}  "
              f"median={_fmt(fm[0])} IQR=[{_fmt(fm[1])}, {_fmt(fm[2])}]")
        p_rp = _wilcox(resid, psth); p_rf = _wilcox(resid, fem)
        p_sf, n_pos, n_used = _signtest(psth, fem)
        print(f"    Wilcoxon resid>stimulus p={_fmt(p_rp, 4)}  "
              f"resid>FEM p={_fmt(p_rf, 4)}")
        print(f"    sign test stimulus>FEM: {n_pos}/{n_used} sessions, "
              f"p={_fmt(p_sf, 4)}")


# --------------------------------------------------------------------------- #
# Panel I: subspace alignment (variance captured) + overlap_k1
# --------------------------------------------------------------------------- #
def report_alignment(bundles):
    section("PANEL I  -  subspace alignment  (variance captured) + leading-dim overlap")
    print("  x = stimulus variance in the FEM subspace (var_p_given_f)")
    print("  y = FEM variance in the stimulus subspace (var_f_given_p)")
    for label, b in bundles.items():
        agg = compute_alignment_aggregate(b)
        ok1 = np.asarray(b.get("overlap_k1_list", []), float)
        ok1 = ok1[np.isfinite(ok1)]
        nok1 = np.asarray(b.get("null_overlap_k1", []), float)
        nok1 = nok1[np.isfinite(nok1)]
        print(f"\n  [{SUBJECT_DISPLAY.get(label, label)}]  "
              f"n_sessions={agg['n_sessions']}")
        for t in ("x", "y"):
            s = agg[t]
            print(f"    {t}: mean={_fmt(s['mean'])}+-{_fmt(s['sd'])}  "
                  f"null_mean95={_ci(np.percentile(s['null_mean'], [2.5, 97.5]))}  "
                  f"p_emp={_fmt(s['p'], 4)} (n_shuff={s['n_shuff']})  "
                  f"sessions sig p<0.05: {s['n_sig05']}, p<0.01: {s['n_sig01']}")
        print(f"    joint-significant sessions (both x,y): "
              f"p<0.05 {agg['n_sig05_joint']}, p<0.01 {agg['n_sig01_joint']}")
        if ok1.size:
            print(f"    leading-dim overlap (k=1): mean={_fmt(ok1.mean())}+-"
                  f"{_fmt(ok1.std(ddof=1) if ok1.size > 1 else np.nan)}  "
                  f"null mean={_fmt(nok1.mean()) if nok1.size else 'nan'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--refresh", action="store_true")
    args, _ = ap.parse_known_args()

    full = load_fig2_data(refresh=args.refresh)
    pooled = _filter_subjects(full)  # real subject labels, Luke omitted
    allen = _filter_subjects(full, omit=OMIT_SUBJECTS | {"Logan"})
    logan = _filter_subjects(full, omit=OMIT_SUBJECTS | {"Allen"})

    # Ordered: pooled first, then each subject. Pooled fano/nc already carry a
    # per-subject breakdown, so those panels only need the pooled bundle.
    bundles = {"pooled": pooled, "Allen": allen, "Logan": logan}

    section("FIGURE 2 CONFIG")
    print(f"  windows (ms): {[round(w, 1) for w in pooled['WINDOWS_MS']]}")
    print(f"  subspace window idx: {pooled['SUBSPACE_WINDOW_IDX']} "
          f"({pooled['WINDOWS_MS'][pooled['SUBSPACE_WINDOW_IDX']]:.1f} ms), "
          f"k = {pooled['SUBSPACE_K']}")
    print(f"  sessions: pooled {len(pooled['sub_subjects'])}  "
          f"Allen {len(allen['sub_subjects'])}  Logan {len(logan['sub_subjects'])}")
    for subj in ("Allen", "Logan"):
        n = int(np.sum(np.asarray(pooled["metrics"][0]["subject_per_neuron"]) == subj))
        print(f"  {SUBJECT_DISPLAY[subj]} neurons (window 0): {n}")

    report_alpha(bundles)
    report_fano(pooled)
    report_noisecorr(pooled)
    report_pr(bundles)
    report_alignment(bundles)
    print()


if __name__ == "__main__":
    main()
