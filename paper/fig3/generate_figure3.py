"""Figure 3: a retinal-input digital twin captures FEM-linked V1 variability.

Renders the digital-twin mechanism figure:

  A  Training and test stimuli (schematic provenance row)
  B  Digital twin schematic (architecture render)
  C  Held-out (trial-averaged) ccnorm: full twin vs retinal-only (extraretinal
     input zeroed) vs stabilized-retina (extraretinal input retained)
  D  Captured count variance over fig. 2's explainable rate variance:
     leave-one-out PSTH vs full twin vs retinal-only vs stabilized-retina
  E  FEM modulation fraction (f_FEM = 1-alpha): the neuron distribution vs
     each within-model twin condition, with a paired TOST equivalence test — Full
     and Ablated reproduce the empirical FEM modulation, Stabilized does not.

Panels C/D draw on the unified analysis-row cache
(`fig3_ablation_inference.pkl`) on the fig2 inclusion population (rate > 2 Hz &
PSTH R^2 > 0.10). Panel E is computed by `_fig3_femfraction` on the same fig2
frame + intersection population and is filtered to the same >=10-analyzed-unit
session floor (via `_restrict_femdata_to_floor`), so C/D/E describe nearly the
same cells fig2 reports (residual gap is only the twin's per-neuron spike/overlap
requirements, not sessions).

Usage:
    uv run python paper/fig3/generate_figure3.py [--recompute]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import wilcoxon

from VisionCore.paths import VISIONCORE_ROOT
from VisionCore.figure_typography import apply_font_floor

from _fig3_data import (
    CHECKPOINT_PATH,
    DATASET_CONFIGS_PATH,
    FIG_DIR,
    configure_matplotlib,
    _load_fig2_included_sessions,
)
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from _fig3_femfraction import compute_femfraction_data, CONDITIONS as FEM_CONDITIONS
from _fig3a_data import load_panel_a_assets
from generate_fig3a import plot_panel_a


# Condition colors: intact/full = blue, extraretinal-ablated (retinal only) = red,
# stabilized-retina (extraretinal retained) = purple, PSTH = grey.
INTACT_COLOR = "#1f77b4"
ABLATED_COLOR = "#d62728"
STABILIZED_COLOR = "#b97800"
PSTH_COLOR = "0.55"
SCATTER_COLOR = "0.35"
ACCENT = "#c0392b"
PANEL_LETTER_SIZE = 10   # match fig2's panel-letter size
PANEL_TITLE_SIZE = 8.0

# Whisker quantiles shared by the panel C and D box summaries. Panel D's score
# is a ratio with a heavy right tail (a few units carry a small positive rate
# variance in the denominator), so 2.5/97.5 whiskers reached twice the fig. 2
# reference and set the axis scale for a few percent of the units. The 10th to
# 90th percentiles keep the boxes legible; the tail is reported in the manifest.
WHISKER_PERCENTILES = (10, 90)


def _clear_panel_heading(ax):
    """Remove source-panel headings so the figure can place them uniformly."""
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    for txt in list(ax.texts):
        if txt.get_transform() == ax.transAxes:
            x, y = txt.get_position()
            if y >= 0.98 and x <= 0.28:
                txt.remove()


def _standard_panel_heading(ax, letter: str, title: str):
    """Place a consistent panel letter/title just above the axes. Titles may
    contain a newline to wrap; the block is top-anchored so the bold letter
    aligns with the first title line. Title is regular weight, matching fig2's
    declarative panel titles (only the letter is bold)."""
    _clear_panel_heading(ax)
    y_top = 1.14
    ax.text(
        -0.035, y_top, letter,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=PANEL_LETTER_SIZE, fontweight="bold", color="#202124",
        clip_on=False,
    )
    ax.text(
        0.085, y_top, title,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=PANEL_TITLE_SIZE, color="#202124",
        linespacing=1.05, clip_on=False,
    )


# ---------------------------------------------------------------------------
# Shared box-and-whisker / significance helpers
# ---------------------------------------------------------------------------
def _stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _fmt_p(p):
    """Explicit p-value string in fig2's format (matches _panel_common.fmt_emp_p
    for the analytic-p case)."""
    if not np.isfinite(p):
        return "p = n/a"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3g}"


def _lighten(color, frac=0.5):
    """Light tint of `color`: mix `frac` of the colour with (1-frac) white, so the
    box interiors read as a faint condition wash rather than a saturated fill."""
    import matplotlib.colors as mcolors
    c = np.asarray(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - c) * frac)


def _box_whisker(ax, groups, positions, colors, *, width=0.55):
    """Box-and-whisker per condition: a faint condition-tinted box (25-75 IQR)
    with a uniform black outline (edge, whiskers, caps) and black median line,
    whiskers at `WHISKER_PERCENTILES` (interpretable fixed quantiles for this
    large-N summary, rather than Tukey 1.5*IQR). Fliers hidden. The fill tint
    signifies the condition. Returns the boxplot dict."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    bp = ax.boxplot(groups, positions=positions, widths=width,
                    patch_artist=True, showfliers=False, whis=WHISKER_PERCENTILES,
                    medianprops=dict(lw=2.0, solid_capstyle="round"),
                    boxprops=dict(lw=1.1), whiskerprops=dict(lw=1.0),
                    capprops=dict(lw=1.0), zorder=3)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(_lighten(color))
        box.set_edgecolor("black")
        box.set_zorder(3)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_zorder(4)
    for part in bp["whiskers"] + bp["caps"]:
        part.set_color("black")
    return bp


def _sig_bracket(ax, x1, x2, y, p, *, h, color="k", fontsize=7.5):
    """Draw significance stars; effect sizes and tests belong in the caption."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=color, lw=0.9, clip_on=False)
    xc = (x1 + x2) / 2
    yc = y + h
    ax.text(xc, yc, _stars(p), ha="center", va="bottom", linespacing=1.05,
            fontsize=fontsize, color=color, clip_on=False)
    return yc


def _finite_mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


# ---------------------------------------------------------------------------
# Panel C — trial-averaged held-out prediction (intact vs ablated ccnorm)
# ---------------------------------------------------------------------------
def _plot_ccnorm_violins(ax, abl):
    pop = np.asarray(abl["cd_population"], dtype=bool)
    intact = np.asarray(abl["ccnorm"]["intact"], dtype=float)
    ablated = np.asarray(abl["ccnorm"]["zeroed"], dtype=float)
    stab = np.asarray(abl["ccnorm"]["stabilized"], dtype=float)
    m = pop & _finite_mask(intact, ablated, stab)
    gi, ga, gs = intact[m], ablated[m], stab[m]

    _box_whisker(ax, [gi, ga, gs], [0, 1, 2],
                 [INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR])

    intact_med = float(np.median(gi))
    p_z = wilcoxon(gi, ga).pvalue
    p_s = wilcoxon(gi, gs).pvalue
    d_z = float(np.median(ga - gi))   # extraretinal ablation cost (retinal only)
    d_s = float(np.median(gs - gi))   # reafferent ablation cost (stabilized retina)
    pct_z = 100.0 * abs(d_z) / intact_med if intact_med != 0 else np.nan
    pct_s = 100.0 * abs(d_s) / intact_med if intact_med != 0 else np.nan
    # Zoom to the displayed whiskers and reserve only a narrow bracket band.
    ax.set_ylim(0.20, 0.97)
    ax.set_yticks(np.arange(0.2, 0.901, 0.1))
    _sig_bracket(ax, 0, 1, 0.875, p_z, h=0.009)
    _sig_bracket(ax, 0, 2, 0.93, p_s, h=0.009)

    ax.set_xlim(-0.6, 2.9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Full", "Retinal", "Stabilized"], rotation=25, ha="right",
                       fontsize=5.3)
    ax.set_ylabel("Held-out prediction\n(ccnorm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"Panel C — ccnorm (N={m.sum()}): intact med={intact_med:.3f}, "
          f"zeroed med={np.median(ga):.3f} (Δ={d_z:+.3f}, p={p_z:.2e}), "
          f"stabilized med={np.median(gs):.3f} (Δ={d_s:+.3f}, p={p_s:.2e})")
    return {
        "n_units": int(m.sum()),
        "n_sessions": int(len(np.unique(np.asarray(abl["sessions"])[m]))),
        "medians": {
            "intact": intact_med,
            "zeroed": float(np.median(ga)),
            "stabilized": float(np.median(gs)),
        },
        "contrasts": {
            "zeroed_vs_intact": {
                "median_difference": d_z,
                "percent_of_intact_median": float(pct_z),
                "wilcoxon_p": float(p_z),
            },
            "stabilized_vs_intact": {
                "median_difference": d_s,
                "percent_of_intact_median": float(pct_s),
                "wilcoxon_p": float(p_s),
            },
        },
    }


# ---------------------------------------------------------------------------
# Panel D — explainable rate variance over fig. 2's diag(Crate)
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42


def _session_cluster_bootstrap(first, second, sessions, mask,
                               n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Session-clustered bootstrap of the median paired difference (second-first).

    The session-median Wilcoxon collapses ~1000 units into 19 numbers and then
    uses only their signs and ranks, which leaves it hinging on whichever session
    happens to sit nearest zero: on this data one borderline session moves it
    between p = 0.019 and p = 0.073. This resamples SESSIONS (the unit of
    independence) with replacement, pools the drawn sessions' units, and
    recomputes the population median difference, giving an effect size with a
    confidence interval rather than a fragile dichotomous p.

    Sessions are resampled rather than units because units within a session share
    a stimulus sequence and an eye trace. Pooling weights a session by its unit
    count; `session_equal_weight` repeats the estimate weighting sessions
    equally, as a sensitivity check.

    Returns a dict with the observed median, the percentile CI, a two-sided
    bootstrap p, and the equal-weight companion.
    """
    d = np.asarray(second, float)[mask] - np.asarray(first, float)[mask]
    sess = np.asarray(sessions)[mask]
    uniq = np.unique(sess)
    idx = [np.where(sess == s)[0] for s in uniq]
    if len(uniq) < 2 or d.size == 0:
        nan = float("nan")
        return {"median": nan, "ci_low": nan, "ci_high": nan, "p_boot": nan,
                "n_sessions": len(uniq), "n_units": int(d.size),
                "session_equal_weight": {"median": nan, "ci_low": nan,
                                         "ci_high": nan}}

    rng = np.random.default_rng(seed)
    session_medians = np.array([np.median(d[i]) for i in idx])
    pooled = np.empty(n_boot)
    equal = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        pooled[b] = np.median(d[np.concatenate([idx[j] for j in draw])])
        equal[b] = np.median(session_medians[draw])

    obs = float(np.median(d))
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    tail = np.mean(pooled <= 0) if obs > 0 else np.mean(pooled >= 0)
    elo, ehi = np.percentile(equal, [2.5, 97.5])
    return {
        "median": obs,
        "ci_low": float(lo), "ci_high": float(hi),
        "p_boot": float(min(1.0, 2 * max(tail, 1.0 / n_boot))),
        "n_sessions": int(len(uniq)), "n_units": int(d.size),
        "n_boot": int(n_boot),
        "session_equal_weight": {
            "median": float(np.median(session_medians)),
            "ci_low": float(elo), "ci_high": float(ehi),
        },
    }


def _session_sign_test(first, second, sessions, mask):
    """Sign test on per-session median differences.

    Reported alongside the bootstrap because it is invariant to the explainable
    fraction's denominator: `diag(Crate)` is shared by both conditions within a
    unit, so it cannot change the sign of any per-unit difference, only its
    magnitude. A test that uses only signs therefore carries none of the
    denominator's estimation noise.
    """
    from scipy.stats import binomtest
    d = np.asarray(second, float)[mask] - np.asarray(first, float)[mask]
    sess = np.asarray(sessions)[mask]
    sm = np.array([np.median(d[sess == s]) for s in np.unique(sess)])
    if sm.size < 2:
        return float("nan"), 0, int(sm.size)
    n_pos = int((sm > 0).sum())
    return float(binomtest(n_pos, sm.size, 0.5).pvalue), n_pos, int(sm.size)


def _session_paired_test(first, second, sessions, mask):
    """Wilcoxon test across per-session median paired differences (second-first)."""
    session_differences = []
    for session in np.unique(sessions[mask]):
        sm = mask & (sessions == session)
        session_differences.append(float(np.median(second[sm] - first[sm])))
    session_differences = np.asarray(session_differences)
    if session_differences.size < 2:
        p = np.nan
    elif np.allclose(session_differences, 0):
        p = 1.0
    else:
        p = float(wilcoxon(session_differences).pvalue)
    return p, session_differences


def _plot_explainable_variance_boxes(ax, abl):
    pop = np.asarray(abl["cd_population"], dtype=bool)
    sessions = np.asarray(abl["sessions"])
    score = abl["explainable_fraction"]
    keys = ["psth", "intact", "zeroed", "stabilized"]
    vals = {key: np.asarray(score[key], dtype=float) for key in keys}
    m = pop & _finite_mask(*(vals[key] for key in keys))
    groups = [vals[key][m] for key in keys]
    colors = [PSTH_COLOR, INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR]
    positions = np.arange(4)

    _box_whisker(ax, groups, positions, colors)

    # Include all displayed whiskers, with compact headroom for the brackets.
    ax.set_ylim(-0.25, 0.84)
    ax.set_yticks(np.arange(-0.2, 0.801, 0.2))
    # Zero is the constant-prediction reference: no captured rate variance.
    # Solid, so it reads as the panel's zero-variance-explained baseline (the
    # dashed grey line above it is the trial-average median).
    ax.axhline(0, color="0.35", lw=1.0, zorder=0)

    # The trial-average reference: every twin condition except the stabilized
    # one sits above it.
    psth_median = float(np.median(vals["psth"][m]))
    ax.axhline(psth_median, color="0.55", lw=0.8, ls="--", alpha=0.8, zorder=0)
    ax.annotate("PSTH median", xy=(3.60, psth_median), xytext=(3.60, .47),
                ha="right", va="bottom", fontsize=5.3, color="0.4",
                arrowprops={"arrowstyle": "-", "lw": .6, "color": ".55"})

    # Nothing is clipped or folded onto one. The above-reference tail is left off
    # the panel to keep it readable and is disclosed in the caption, the manifest,
    # and the console diagnostics instead.
    above_one = {key: int(np.sum(group > 1)) for key, group in zip(keys, groups)}

    # Stars sit above the whiskers; effect sizes and their reference medians
    # remain in the numerical report and caption.
    ref_median = {key: float(np.median(vals[key][m])) for key in keys}
    contrasts = [
        # name, x1, x2, first, second, y, reference key
        ("ablated_vs_full", 1, 2, "intact", "zeroed", 0.68, None),
        ("full_vs_psth", 0, 1, "psth", "intact", 0.735, "psth"),
        ("stabilized_vs_full", 1, 3, "intact", "stabilized", 0.79, "intact"),
    ]
    contrast_stats = {}
    for (name, x1, x2, first_key, second_key, y_bracket, ref_key) in contrasts:
        p_wilcoxon, session_differences = _session_paired_test(
            vals[first_key], vals[second_key], sessions, m
        )
        boot = _session_cluster_bootstrap(
            vals[first_key], vals[second_key], sessions, m
        )
        p_sign, n_pos, n_sess = _session_sign_test(
            vals[first_key], vals[second_key], sessions, m
        )
        # The bracket reports the bootstrap: it is the primary test (see
        # `_session_cluster_bootstrap`); Wilcoxon and the sign test are kept in
        # the manifest so the reported claim can be checked against both.
        p = boot["p_boot"]
        delta = float(np.median(vals[second_key][m] - vals[first_key][m]))
        pct = np.nan
        if ref_key is not None:
            denom = ref_median[ref_key]
            pct = 100.0 * delta / denom if denom != 0 else np.nan
        _sig_bracket(ax, x1, x2, y_bracket, p, h=0.014)
        contrast_stats[name] = {
            "median_unit_difference": delta,
            "percent_reference": ref_key,
            "percent_of_reference_median": float(pct),
            "session_median_differences": session_differences.tolist(),
            "bootstrap": boot,
            "session_wilcoxon_p": p_wilcoxon,
            "session_sign_test": {"p": p_sign, "n_positive": n_pos,
                                  "n_sessions": n_sess},
        }

    ax.set_xlim(-0.6, 3.75)
    ax.set_xticks(positions)
    ax.set_xticklabels([
        "PSTH",
        "Full",
        "Retinal",
        "Stabilized",
    ], fontsize=4.9, rotation=25, ha="right")
    ax.set_ylabel("Fraction of conditional rate\nvariance explained")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    condition_stats = {}
    print(f"Panel D — explainable rate variance over fig. 2 diag(Crate) "
          f"(N={int(m.sum())} units, {len(np.unique(sessions[m]))} sessions):")
    # Sensitivity: the same numerators over the abandoned matched denominator.
    matched_score = abl["explainable_fraction_matched"]
    for key, group in zip(keys, groups):
        matched = np.asarray(matched_score[key], dtype=float)
        matched_group = matched[m & np.isfinite(matched)]
        condition_stats[key] = {
            "n": int(group.size),
            "median": float(np.median(group)),
            "q025": float(np.percentile(group, 2.5)),
            "q975": float(np.percentile(group, 97.5)),
            "n_above_one": above_one[key],
            "minimum": float(np.min(group)),
            "maximum": float(np.max(group)),
            "matched_denominator_sensitivity": {
                "n": int(matched_group.size),
                "median": float(np.median(matched_group))
                if matched_group.size else float("nan"),
                "n_above_one": int(np.sum(matched_group > 1)),
            },
        }
        s = condition_stats[key]
        print(f"  {key:<10} median={s['median']:+.4f}, "
              f"2.5-97.5%=[{s['q025']:+.4f}, {s['q975']:+.4f}], "
              f">1={s['n_above_one']}, range=[{s['minimum']:+.3f}, "
              f"{s['maximum']:+.3f}], matched-denominator "
              f"median={s['matched_denominator_sensitivity']['median']:+.4f} "
              f"(n={s['matched_denominator_sensitivity']['n']})")
    for name, stats in contrast_stats.items():
        b = stats["bootstrap"]
        st = stats["session_sign_test"]
        print(f"  {name}: Δ={stats['median_unit_difference']:+.4f} "
              f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] "
              f"boot p={b['p_boot']:.3g}  |  equal-session Δ="
              f"{b['session_equal_weight']['median']:+.4f} "
              f"[{b['session_equal_weight']['ci_low']:+.4f}, "
              f"{b['session_equal_weight']['ci_high']:+.4f}]  |  sign "
              f"{st['n_positive']}/{st['n_sessions']} p={st['p']:.3g}  |  "
              f"Wilcoxon p={stats['session_wilcoxon_p']:.3g}")
    with np.errstate(divide="ignore", invalid="ignore"):
        fig2_r2max = np.asarray(abl["fig2_c_rate"], float) / np.asarray(
            abl["fig2_c_total"], float
        )
        matched_r2max = np.asarray(abl["matched_c_rate"], float) / np.asarray(
            abl["matched_c_total"], float
        )

    def summary(values, *, positive_only=True):
        keep = pop & np.isfinite(values)
        if positive_only:
            keep &= values > 0
        if not np.any(keep):
            return {"n": 0, "percentiles": {}}
        q = np.percentile(values[keep], [0, 1, 2.5, 25, 50, 75, 97.5, 99, 100])
        return {
            "n": int(keep.sum()),
            "percentiles": dict(zip(
                ["0", "1", "2.5", "25", "50", "75", "97.5", "99", "100"],
                q.tolist(),
            )),
        }

    # The numerator drops every bin the twin cannot predict, so its Var(y) is
    # not fig. 2's diag(Ctotal); the ratio is reported rather than asserted.
    var_ratio = np.asarray(abl["total_variance_ratio"], float)
    scored_windows = np.asarray(abl["matched_n_windows"], float)
    print(f"  sampling: matched Var(y)/Ctotal_fig2 median="
          f"{np.nanmedian(var_ratio[pop]):.3f}, median "
          f"{int(np.nanmedian(scored_windows[pop]))} scored windows/unit; "
          f"fig. 2 denominator positive for "
          f"{int((pop & (np.asarray(abl['fig2_c_rate'], float) > 0)).sum())}"
          f"/{int(pop.sum())} population units, matched denominator for "
          f"{int((pop & (np.asarray(abl['matched_c_rate'], float) > 0)).sum())}")

    all_sessions = set(np.unique(sessions[pop]))
    scored_sessions = set(np.unique(sessions[m]))
    return {
        "n_units": int(m.sum()),
        "n_units_fig2_population": int(pop.sum()),
        "n_units_excluded": int(pop.sum() - m.sum()),
        "n_sessions": int(len(scored_sessions)),
        "n_sessions_fig2_population": int(len(all_sessions)),
        "sessions_excluded": sorted(all_sessions - scored_sessions),
        "fig2_denominator": summary(fig2_r2max),
        "matched_denominator_sensitivity": summary(matched_r2max),
        "total_variance_ratio": summary(var_ratio, positive_only=False),
        "scored_windows_per_unit": summary(scored_windows, positive_only=False),
        "conditions": condition_stats,
        "contrasts": contrast_stats,
    }


# ---------------------------------------------------------------------------
# Panel E — FEM modulation fraction: neurons vs each within-model twin condition
# ---------------------------------------------------------------------------
TOST_MARGIN = 0.10   # equivalence margin on 1-alpha (robust for any Δ >= 0.05)


def _in01(v):
    """fig2's 1-alpha inclusion: finite and within [0, 1] (unclipped values)."""
    v = np.asarray(v, float)
    return np.isfinite(v) & (v >= 0.0) & (v <= 1.0)


def _paired_tost(emp, mod, margin=TOST_MARGIN):
    """Paired two-one-sided-t equivalence test on the matched-cell differences
    d = emp - mod (cells with both 1-alpha in [0, 1]). Equivalence to the neuron
    distribution is established at level a when the returned p_tost < a.

    Returns (p_tost, median_d, n)."""
    from scipy.stats import t as tdist
    both = _in01(emp) & _in01(mod)
    d = np.asarray(emp, float)[both] - np.asarray(mod, float)[both]
    n = int(d.size)
    md = float(np.median(d)) if n else np.nan
    if n < 3:
        return np.nan, md, n
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / np.sqrt(n)
    df = n - 1
    if se == 0:
        return (0.0 if abs(mean) < margin else 1.0), md, n
    p_lower = float(tdist.sf((mean + margin) / se, df))   # H1: mean > -margin
    p_upper = float(tdist.cdf((mean - margin) / se, df))  # H1: mean < +margin
    return max(p_lower, p_upper), md, n


def _nice_step(x):
    """A '1/2/2.5/5 x 10^k' step near x (for count-axis ticks)."""
    if x <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= x:
            return m * mag
    return 10 * mag


def _plot_femfraction(ax, femdata, *, margin=TOST_MARGIN):
    """FEM modulation fraction (f_FEM = 1 - alpha) in per-unit counts: the
    empirical neuron distribution (grey) vs each within-model condition, on
    the fig2 frame + intersection population (the same quantity fig2 panel C
    reports). Median triangles mark each distribution. The histograms fill only
    the lower ~60% of the axis; the headroom above holds the median markers,
    legend, and equivalence annotation. A paired TOST (margin ±margin) tests
    every condition's equivalence to the neurons: Full and Ablated reproduce the
    empirical FEM modulation, Stabilized does not — so reafference carries the
    fig2 effect."""
    from matplotlib.lines import Line2D

    bins = np.linspace(0, 1, 26)
    emp = np.asarray(femdata["intact"]["B_obs_uncl"], dtype=float)
    e = emp[_in01(emp)]
    conds = [("intact", "Model (full)", INTACT_COLOR),
             ("zeroed", "Model (ablated)", ABLATED_COLOR),
             ("stabilized", "Model (stabilized)", STABILIZED_COLOR)]
    model_v = {k: np.asarray(femdata[k]["B_model_uncl"], dtype=float)[
        _in01(femdata[k]["B_model_uncl"])] for k, _, _ in conds}

    # Count ceiling: the first round tick above the tallest bar of any series.
    # The axis then extends to ceiling / 0.60 so the histograms fill ~60% of the
    # vertical span and the top ~40% is free for markers/legend/stats.
    counts = [np.histogram(e, bins=bins)[0].max()]
    counts += [np.histogram(model_v[k], bins=bins)[0].max() for k, _, _ in conds]
    maxcount = int(max(counts))
    step = _nice_step(maxcount / 3.0)
    ceiling = float(np.ceil(maxcount / step) * step)
    if ceiling <= maxcount:
        ceiling += step
    ylim_top = ceiling / 0.60

    # Histograms (per-unit counts). Empirical filled grey; models as step lines.
    ax.hist(e, bins=bins, color="0.6", alpha=0.5, edgecolor="white",
            linewidth=0.3, zorder=1)
    for key, _label, color in conds:
        ax.hist(model_v[key], bins=bins, histtype="step", color=color, lw=1.6,
                zorder=3)

    head = ylim_top - ceiling
    y_tri = ceiling + 0.12 * head
    med = {"emp": float(np.median(e))}
    tost = {}
    for key, _label, _color in conds:
        med[key] = float(np.median(model_v[key]))
        p_tost, md, n = _paired_tost(emp, femdata[key]["B_model_uncl"], margin)
        tost[key] = (p_tost, np.isfinite(p_tost) and p_tost < 0.05)
        print(f"Panel E — {_label}: model med={med[key]:.3f} (n={n}); "
              f"median(neuron-model)={md:+.3f}; TOST(Δ={margin}) p={p_tost:.2e} -> "
              f"{'EQUIVALENT' if tost[key][1] else 'not equivalent'} to neurons")

    # Equivalence display: a TOST equivalence zone = empirical median ± margin,
    # shaded in the headroom. A distribution whose median triangle lands inside
    # the zone is statistically equivalent to the neurons (paired TOST, p<0.05);
    # one outside is not. Full and Ablated fall inside, Stabilized far outside.
    from matplotlib.patches import Rectangle
    band_lo, band_hi = med["emp"] - margin, med["emp"] + margin
    yb0, yb1 = ceiling + 0.02 * head, ceiling + 0.22 * head
    ax.add_patch(Rectangle((band_lo, yb0), band_hi - band_lo, yb1 - yb0,
                           facecolor="0.78", alpha=0.5, edgecolor="none", zorder=2))
    for xb in (band_lo, band_hi):
        ax.plot([xb, xb], [yb0, yb1], color="0.55", lw=0.8, zorder=3)
    ax.text(band_hi + 0.015, 0.5 * (yb0 + yb1),
            f"±{margin:g} from\nemp. median",
            ha="left", va="center", fontsize=4.8, color="0.4", linespacing=1.1)

    # Median triangles (replacing the empirical dashed line): one per
    # distribution, in a single headroom row inside/against the zone. Whether a
    # triangle lands inside the shaded zone reads the equivalence verdict on its
    # own, so no per-marker text is needed.
    ax.plot(med["emp"], y_tri, marker="v", ms=9.5, color="0.3", mec="white",
            mew=0.7, clip_on=False, zorder=6)
    for key, _label, color in conds:
        ax.plot(med[key], y_tri, marker="v", ms=9.0, color=color, mec="white",
                mew=0.7, clip_on=False, zorder=6)

    # Difference test to complement the equivalence test: a paired Wilcoxon
    # signed-rank on the same matched cells, bracketing the neurons against the
    # stabilized twin — the one condition the TOST rejects. Drawn in the
    # headroom above the triangle row, clear of the TOST band.
    both = _in01(emp) & _in01(femdata["stabilized"]["B_model_uncl"])
    d_stab = emp[both] - np.asarray(
        femdata["stabilized"]["B_model_uncl"], float)[both]
    p_stab = float(wilcoxon(d_stab).pvalue) if d_stab.size >= 3 else np.nan
    y_sig = ceiling + 0.26 * head
    _sig_bracket(ax, med["stabilized"], med["emp"], y_sig, p_stab,
                 h=0.035 * head, fontsize=7.5)
    print(f"Panel E — empirical vs Model (stabilized): paired Wilcoxon "
          f"p={p_stab:.2e} (n={int(d_stab.size)}, "
          f"median difference={float(np.median(d_stab)):+.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(np.arange(0, ceiling + 0.5 * step, step))
    ax.set_xlabel("FEM fraction of rate modulation")
    ax.set_ylabel("Units")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Line2D([0], [0], color="0.55", lw=5, alpha=0.5),
               Line2D([0], [0], color=INTACT_COLOR, lw=1.6),
               Line2D([0], [0], color=ABLATED_COLOR, lw=1.6),
               Line2D([0], [0], color=STABILIZED_COLOR, lw=1.6)]
    labels = ["Empirical", "Full", "Retinal", "Stabilized"]
    leg = ax.legend(handles, labels, frameon=False, fontsize=7.2, loc="upper left",
                    handlelength=1.3, handletextpad=0.5, labelspacing=0.35,
                    borderaxespad=0.2, ncol=2, columnspacing=0.7)
    leg.set_zorder(7)
    return {
        "n_empirical_in_unit_interval": int(e.size),
        "medians": med,
        "equivalence_margin": float(margin),
        "tost": {
            key: {"p": float(tost[key][0]), "equivalent": bool(tost[key][1])}
            for key, _label, _color in conds
        },
        "stabilized_vs_empirical": {
            "n": int(d_stab.size),
            "median_empirical_minus_model": float(np.median(d_stab)),
            "wilcoxon_p": float(p_stab),
        },
    }


def _restrict_femdata_to_floor(femdata, included):
    """Mask a femfraction dict's per-cell arrays to cells from floored sessions
    (fig2's >=10-analyzed-unit floor). Panel E's caches are computed over every
    session with >=3 included cells, but panels C/D go through the floor via
    `_load_fig2_included_sessions`; applying the same floor here (fresh, not
    baked into the cache — symmetric with `load_ablation_data`) keeps C/D/E on
    the same session population fig2 reports. Per-cell entries are arrays of
    equal length and are filtered by one mask; scalar provenance entries (e.g.
    `count_bins`, the counting window the estimate was built on) are passed
    through untouched."""
    sess = np.asarray(femdata["session"])
    keep = np.isin(sess, list(included))
    out = {}
    for k, v in femdata.items():
        a = np.asarray(v)
        out[k] = a[keep] if a.shape[:1] == keep.shape else v
    return out


def _plot_missing_cache(ax):
    ax.set_axis_off()
    ax.text(0.5, 0.58, "ablation cache not found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(0.5, 0.42, f"Missing: {ABLATION_CACHE_PATH.name}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="0.45")


def _load_ablation_cache():
    """Load the unified analysis-row cache without triggering a heavy run."""
    if not ABLATION_CACHE_PATH.exists():
        return None
    return load_ablation_data(recompute=False)


def _caption_p(value):
    value = float(value)
    if not np.isfinite(value):
        return "p=n/a"
    if value < 0.001:
        return f"p={value:.1e}"
    return f"p={value:.3f}"


def _build_caption(manifest: dict) -> str:
    model = manifest.get("model", {})
    family = model.get("family", "unknown")
    if family == "dekel":
        architecture = "nonrecurrent, anti-aliased Dekel convolutional twin"
    else:
        architecture = "convolutional-recurrent twin"
    parts = [
        "Figure 3. A retinal-input digital twin captures FEM-linked V1 response variability.",
        "(A) The twin was trained on gaze-contingent gratings, Gabors, and natural "
        "images and evaluated on the held-out fixated flashed-image dataset. "
        f"(B) The {architecture} receives a moving retinal stimulus and separate "
        "extraretinal eye-position and eye-velocity inputs. The retinal-only "
        "condition zeroes the extraretinal inputs; the stabilized-retina condition "
        "retains them but freezes the retinal input at one session-global gaze centroid.",
    ]
    c = manifest.get("panel_c_stats")
    if c:
        med = c["medians"]
        dz = c["contrasts"]["zeroed_vs_intact"]
        ds = c["contrasts"]["stabilized_vs_intact"]
        parts.append(
            f"(C) Held-out trial-averaged prediction across {c['n_units']} cells "
            f"from {c['n_sessions']} sessions. Median ccnorm was {med['intact']:.3f} "
            f"for the full twin, {med['zeroed']:.3f} retinal-only "
            f"(median Δ={dz['median_difference']:+.3f}; "
            f"{_caption_p(dz['wilcoxon_p'])}), and {med['stabilized']:.3f} after "
            f"retinal stabilization (Δ={ds['median_difference']:+.3f}; "
            f"{_caption_p(ds['wilcoxon_p'])})."
        )
    d = manifest.get("panel_d_stats")
    if d:
        med = {key: value["median"] for key, value in d["conditions"].items()}
        parts.append(
            "(D) Single-trial prediction was measured as captured count variance, "
            "Var(Y)-Var(Y-Yhat), on Figure 2-matched model-valid bins, divided by "
            "each cell's diag(Sigma_rate). "
            f"Across {d['n_units']} cells, median fractions for the trial average, "
            f"full, retinal-only, and stabilized predictions were {med['psth']:.3f}, "
            f"{med['intact']:.3f}, {med['zeroed']:.3f}, and "
            f"{med['stabilized']:.3f}, respectively. Values above one were retained "
            "because the denominator is estimated rather than a hard bound."
        )
        for key, label in (("full_vs_psth", "Full minus PSTH"),
                           ("ablated_vs_full", "Retinal minus full"),
                           ("stabilized_vs_full", "Stabilized minus full")):
            contrast = d["contrasts"][key]
            boot = contrast["bootstrap"]
            parts.append(
                f"{label}: median paired Δ={contrast['median_unit_difference']:+.3f}, "
                f"95% session-cluster bootstrap CI [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}], "
                f"{_caption_p(boot['p_boot'])}."
            )
        parts.append("Brackets: * p<0.05, ** p<0.01, *** p<0.001; paired Wilcoxon in C and session-cluster bootstrap in D.")
    e = manifest.get("panel_e_stats")
    if e:
        med = e["medians"]
        condition_labels = {
            "intact": "full",
            "zeroed": "retinal-only",
            "stabilized": "stabilized",
        }
        equivalent = [
            condition_labels.get(key, key)
            for key, result in e["tost"].items()
            if result["equivalent"]
        ]
        not_equivalent = [
            condition_labels.get(key, key)
            for key, result in e["tost"].items()
            if not result["equivalent"]
        ]

        def condition_phrase(labels):
            if not labels:
                return "no model prediction"
            joined = (
                labels[0]
                if len(labels) == 1
                else " and ".join(labels)
                if len(labels) == 2
                else f"{', '.join(labels[:-1])}, and {labels[-1]}"
            )
            noun = "prediction" if len(labels) == 1 else "predictions"
            return f"the {joined} {noun}"

        parts.append(
            "(E) FEM modulation fraction, f_FEM (=1-alpha), had empirical, full, "
            f"retinal-only, and stabilized medians of {med['emp']:.3f}, "
            f"{med['intact']:.3f}, {med['zeroed']:.3f}, and "
            f"{med['stabilized']:.3f}. Paired TOST with a "
            f"±{e['equivalence_margin']:.2g} margin classified "
            f"{condition_phrase(equivalent)} as equivalent to the empirical "
            f"distribution and {condition_phrase(not_equivalent)} as not equivalent. "
            "Boxes in C and D show the interquartile range with 10th–90th "
            "percentile whiskers; triangles in E mark medians."
        )
    return "\n\n".join(parts) + "\n"


def _write_sidecars(out_dir, manifest: dict):
    caption = _build_caption(manifest)
    (out_dir / "figure3_caption.md").write_text(caption, encoding="utf-8")

    readme = """# Figure 3

Generated by `paper/fig3/generate_figure3.py`.

The digital-twin mechanism figure: a retinal-input twin whose single-trial
prediction survives zeroing the extraretinal eye-state pathway. Panel D reports
captured count variance on Figure 2-matched, model-valid windows relative to
Figure 2's own diag(Crate) at the one-bin window, including the leave-one-out
PSTH as a predictor. The FEM modulation fraction, f_FEM (= 1-alpha), reproduces
the empirical distribution under the full and extraretinal-ablated conditions but not when the
retinal image is stabilized. Panels C/D use `fig3_ablation_inference.pkl`; panel E uses the
per-condition f_FEM caches (`fig3_femfraction_{condition}.pkl`), both on the
fig2 inclusion population.

## Outputs
- `figure3.png`
- `figure3.pdf`
- `figure3.svg`
- `figure3_caption.md`
- `figure3_manifest.json`
"""
    (out_dir / "figure3_README.md").write_text(readme, encoding="utf-8")

    with open(out_dir / "figure3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def compose(*, recompute: bool = False, out_dir=FIG_DIR, dpi: int = 300, layout: str = "production", schematic_no_phase: bool = False):
    configure_matplotlib()
    # Font sizes tuned for the final 8.5-inch-wide (page-width) render. Applied
    # after configure_matplotlib() so only figure 3's main composite is
    # affected, not other scripts that share the style helper.
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.0,
        # Keep separate input cubes separate in vector exports, so transparent
        # space between them remains measurable by the manuscript overlap audit.
        "image.composite_image": False,
    })
    out_dir.mkdir(parents=True, exist_ok=True)

    abl = load_ablation_data(recompute=recompute) if recompute else _load_ablation_cache()
    assets = load_panel_a_assets(recompute=recompute)

    # Panel A is a two-row schematic (aspect ≈ 1.1), so it needs a taller top
    # slot to render wide enough for its architecture labels to breathe.
    fig = plt.figure(figsize=(8.5, 9.6 if layout == "manuscript" else 9.7), constrained_layout=False)
    gs = GridSpec(
        2, 1,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.090,
        top=0.990,
        height_ratios=[2.15 if layout == "manuscript" else 2.4, 1.0],
        hspace=0.095 if layout == "manuscript" else 0.11,
    )

    # Row 1. Native schematic (stimulus + architecture), fitted into the slot.
    # The A/B panel letters and the grey divider are drawn inside the schematic
    # (see generate_fig3a._draw_all), so no composite letter is placed here.
    ax_a = fig.add_subplot(gs[0, 0])
    if layout == "manuscript":
        from manuscript_schematic import plot_panel_ab
        plot_panel_ab(ax_a, assets, no_phase_preview=schematic_no_phase)
    else:
        plot_panel_a(ax=ax_a, assets=assets)

    # Row 2. Three analysis panels: C/D box-and-whisker, E the FEM-fraction
    # distribution overlay (no marginal axis).
    gs_mid = gs[1, 0].subgridspec(
        1, 5,
        width_ratios=[0.01, 1.0, 1.28, 1.25, 0.01],
        wspace=0.65 if layout == "manuscript" else 0.5,
    )

    ax_c = fig.add_subplot(gs_mid[0, 1])
    ax_d = fig.add_subplot(gs_mid[0, 2])
    ax_e = fig.add_subplot(gs_mid[0, 3])

    panel_c_stats = None
    panel_d_stats = None
    panel_e_stats = None
    if abl is not None:
        panel_c_stats = _plot_ccnorm_violins(ax_c, abl)
        panel_d_stats = _plot_explainable_variance_boxes(ax_d, abl)
        # Panel E on the same fig2 session floor as C/D (>=10 analyzed units).
        included = _load_fig2_included_sessions()
        femdata = {c: _restrict_femdata_to_floor(
                       compute_femfraction_data(condition=c, refresh=recompute),
                       included)
                   for c in FEM_CONDITIONS}
        panel_e_stats = _plot_femfraction(ax_e, femdata)
    else:
        for a in (ax_c, ax_d, ax_e):
            _plot_missing_cache(a)

    _standard_panel_heading(ax_c, "C", "Ablations modestly reduce\ntrial-averaged predictions")
    _standard_panel_heading(ax_d, "D", "Retinal input predicts\nsingle-trial responses")
    _standard_panel_heading(ax_e, "E", "Retinal input reproduces\nempirical FEM modulation")

    # No bbox_inches="tight": keep the canvas at exactly the intended
    # page-width figsize (8.5 in) rather than cropping to the ink bounds.
    apply_font_floor(fig)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"figure3.{ext}", dpi=dpi)

    manifest = {
        "figure": "figure3",
        "layout": layout,
        "schematic_no_phase_preview": schematic_no_phase,
        "analysis_row_cache": str(ABLATION_CACHE_PATH),
        "analysis_row_cache_present": abl is not None,
        "source_script": str(__file__),
        "panel_mapping": {
            "A": "training and test stimuli (schematic provenance row)",
            "B": "digital-twin architecture and fixed-weight input interventions",
            "C": "trial-averaged held-out ccnorm: full vs retinal-only (zeroed) "
                 "vs stabilized-retina (extraretinal retained)",
            "D": "captured variance over fig. 2 diag(Crate): leave-one-out "
                 "PSTH vs full vs retinal-only vs stabilized-retina",
            "E": "FEM modulation fraction (f_FEM = 1-alpha): neuron distribution "
                 "vs each within-model twin condition, paired TOST equivalence test",
        },
        "model": {
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_sha256": hashlib.sha256(
                Path(CHECKPOINT_PATH).read_bytes()
            ).hexdigest(),
            "dataset_configs_path": str(DATASET_CONFIGS_PATH),
            "family": assets.arch.get("model_family", "unknown"),
            "sampling_rate_hz": assets.arch.get("sampling_rate"),
            "history_frames": assets.arch.get("frontend_k"),
        },
        "panel_c_stats": panel_c_stats,
        "panel_d_stats": panel_d_stats,
        "panel_e_stats": panel_e_stats,
    }
    _write_sidecars(out_dir, manifest)
    return fig, manifest


def parse_args():
    p = argparse.ArgumentParser(description="Generate digital-twin mechanism Figure 3.")
    p.add_argument("--recompute", action="store_true",
                   help="Force digital-twin recomputation instead of cached results.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory for figure outputs (default: canonical fig3 dir).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--layout", choices=("production", "manuscript"), default="production")
    p.add_argument("--schematic-no-phase", action="store_true",
                   help="Preview the revised no-branch schematic while retaining the selected model's results.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = FIG_DIR if args.out_dir is None else Path(args.out_dir)
    if args.schematic_no_phase and args.layout != "manuscript":
        raise ValueError("--schematic-no-phase requires --layout manuscript")
    fig, _manifest = compose(recompute=args.recompute, out_dir=out_dir, dpi=args.dpi, layout=args.layout,
                            schematic_no_phase=args.schematic_no_phase)
    plt.close(fig)
    print(f"Saved Figure 3 to: {out_dir}")


if __name__ == "__main__":
    main()
