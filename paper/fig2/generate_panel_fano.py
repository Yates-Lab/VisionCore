"""
Figure 2 Fano factor panels.

    plot_fano_population  per-session population (slope-through-origin) Fano at
                          one window: one faint line per session joining its
                          uncorrected and corrected value, with the
                          across-session mean +/- SD overlaid. Main-figure
                          Panel E.
    plot_panel_e          population Fano vs counting window, per subject.
                          Window-robustness supplemental.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save, pstars, nearest_window, fmt_emp_p
from compute_fig2_data import load_fig2_data, _slope_through_origin


def _per_session_slopes(s):
    """Per-session slope-through-origin Fano (uncorrected, corrected) from the
    per-neuron arrays in a fano_stats[window] entry. Drops sessions where
    either slope is undefined."""
    erate = np.asarray(s["erate"], dtype=float)
    var_u = np.asarray(s["var_u"], dtype=float)
    var_c = np.asarray(s["var_c"], dtype=float)
    sess = np.asarray(s["session_per_neuron"])
    su, sc = [], []
    for session in np.unique(sess):
        m = sess == session
        a = _slope_through_origin(erate[m], var_u[m])
        b = _slope_through_origin(erate[m], var_c[m])
        if np.isfinite(a) and np.isfinite(b):
            su.append(a); sc.append(b)
    return np.asarray(su), np.asarray(sc)


def plot_fano_population(ax=None, refresh=False, data=None, window_ms=25.0):
    """Single-window population (slope-through-origin) Fano factor: one faint
    line per session joining its uncorrected and corrected value, with the
    across-session mean +/- SD overlaid (open uncorrected -> filled corrected,
    joined by a line) and the Poisson (Fano = 1) reference dotted. A grey
    shuffle-null band at the corrected marker shows where the corrected mean
    would fall if the FEM correction were random, and the bracket carries stars
    + the explicit shuffle-null p-value (session-matched to the across-session
    mean statistic)."""
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 3.0))
    else:
        fig = ax.figure

    w = nearest_window(data["WINDOWS_MS"], window_ms)
    s = data["fano_stats"][w]
    su = np.asarray(s.get("sess_slope_unc", []), dtype=float)
    sc = np.asarray(s.get("sess_slope_cor", []), dtype=float)
    if su.size == 0 or sc.size == 0:
        su, sc = _per_session_slopes(s)
    x = [0.0, 1.0]

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.8, zorder=0)
    # Sit just above the Fano = 1 line, left-justified a touch right of the
    # FEM-corrected tick (x = 1.0).
    ax.text(1.06, 1.0, "Poisson", color="gray", fontsize=7, va="bottom",
            ha="left")

    # one faint line per session
    for a, b in zip(su, sc):
        ax.plot(x, [a, b], color="0.75", lw=0.7, alpha=0.7, zorder=1)

    # Grey shuffle-null band (2.5-97.5%) at the corrected marker: where the
    # corrected across-session mean would fall if the FEM correction were random.
    null_lo, null_hi = s.get("mean_sess_cor_null_ci", (np.nan, np.nan))
    if np.isfinite(null_lo) and np.isfinite(null_hi):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (x[1] - 0.2, null_lo), 0.4, null_hi - null_lo,
            facecolor="0.6", alpha=0.30, edgecolor="0.35", lw=0.7,
            linestyle="--", zorder=2))
        ax.text(x[1] + 0.22, 0.5 * (null_lo + null_hi), "shuffle\nnull",
                color="0.4", fontsize=6.5, va="center", ha="left")

    # across-session mean +/- SD overlay
    mu = (float(su.mean()), float(sc.mean()))
    sd = (float(su.std(ddof=1)), float(sc.std(ddof=1)))
    ax.plot(x, mu, "-", color="tab:blue", lw=2.0, zorder=4)
    ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="tab:blue", elinewidth=1.5,
                capsize=0, zorder=5)
    ax.plot(x[0], mu[0], "o", mfc="white", mec="tab:blue", ms=7, mew=1.6,
            zorder=6)
    ax.plot(x[1], mu[1], "o", mfc="tab:blue", mec="tab:blue", ms=7, mew=1.6,
            zorder=6)

    p = s.get("p_emp_mean_sess", np.nan)
    n_shuff = s.get("n_shuff_sess")

    lo = np.nanmin([float(su.min()), float(sc.min()), 1.0, mu[1] - sd[1],
                    null_lo])
    hi = np.nanmax([float(su.max()), float(sc.max()), 1.0, mu[0] + sd[0],
                    null_hi])
    span = hi - lo if hi > lo else 1.0
    ybr = hi + 0.06 * span
    h = 0.03 * span
    ax.plot([x[0], x[0], x[1], x[1]], [ybr, ybr + h, ybr + h, ybr],
            color="k", lw=1.0, clip_on=False)
    ax.text(0.5, ybr + h, pstars(p), ha="center", va="bottom", color="k",
            fontsize=9, clip_on=False)
    ax.text(0.5, ybr + h + 0.085 * span, fmt_emp_p(p, n_shuff), ha="center",
            va="bottom", color="k", fontsize=7, clip_on=False)
    ax.set_ylim(lo - 0.06 * span, ybr + h + 0.085 * span + 0.12 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(["uncorrected", "FEM-\ncorrected"])
    ax.set_ylabel("Variability (Fano factor)")
    ax.set_xlim(-0.5, 1.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _yerr(slope, ci):
    """Asymmetric yerr columns from a (lo, hi) CI; NaN where slope is missing."""
    lo = np.array([c[0] for c in ci], dtype=float)
    hi = np.array([c[1] for c in ci], dtype=float)
    return np.vstack([slope - lo, hi - slope])


def plot_panel_e(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = np.asarray(data["WINDOWS_MS"], dtype=float)
    fano_stats = data["fano_stats"]

    series = {}  # per-subject slopes/CIs/p for plotting + star placement
    for subj in SUBJECTS:
        s_unc, s_cor, ci_unc, ci_cor, p_list = [], [], [], [], []
        for w in WINDOWS_MS:
            ps = fano_stats[w]["per_subject"].get(subj)
            if ps is None:
                s_unc.append(np.nan); s_cor.append(np.nan)
                ci_unc.append((np.nan, np.nan)); ci_cor.append((np.nan, np.nan))
                p_list.append(np.nan)
                continue
            s_unc.append(ps["slope_unc"]); s_cor.append(ps["slope_cor"])
            ci_unc.append(ps["slope_unc_ci"]); ci_cor.append(ps["slope_cor_ci"])
            p_list.append(ps["p_slope"])
        s_unc = np.array(s_unc); s_cor = np.array(s_cor)
        if not np.any(np.isfinite(s_unc)):
            continue
        color = SUBJECT_COLORS[subj]
        # Raw de-emphasized: dashed, open markers. Corrected is the hero: solid,
        # filled markers. Linestyle convention matches panel D.
        ax.errorbar(WINDOWS_MS, s_unc, yerr=_yerr(s_unc, ci_unc), fmt="o--",
                    color=color, lw=1.5, capsize=0, markerfacecolor="white")
        ax.errorbar(WINDOWS_MS, s_cor, yerr=_yerr(s_cor, ci_cor), fmt="o-",
                    color=color, lw=1.5, capsize=0)
        series[subj] = dict(ci_cor=ci_cor, p=p_list)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)

    # Per-window stars sit just below the lower corrected point (emphasis on the
    # corrected estimate), stacked downward per subject in subject color.
    present = list(series)
    span = max(c[1] for s in present for c in series[s]["ci_cor"]
               if np.isfinite(c[1]))
    gap = 0.035 * span
    overall_bot = np.inf
    for wi, w in enumerate(WINDOWS_MS):
        base = np.nanmin([series[s]["ci_cor"][wi][0] for s in present])
        for row, subj in enumerate(present):
            y = base - gap - row * gap
            ax.text(w, y, pstars(series[subj]["p"][wi]), ha="center",
                    va="top", color=SUBJECT_COLORS[subj], fontsize=9)
        overall_bot = min(overall_bot, base - gap - len(present) * gap)

    ax.set_ylim(bottom=overall_bot - 0.04 * span)
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Variability (Fano factor)")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_e()
    fig.tight_layout()
    standalone_save(fig, "panel_e_fano_vs_window")
