"""Figure 3: a retinal-input digital twin captures FEM-linked V1 variability.

Renders the digital-twin mechanism figure:

  A  Training and test stimuli (schematic provenance row)
  B  Digital twin schematic (architecture render)
  C  Held-out (trial-averaged) ccnorm: full twin vs retinal-only (behavior
     zeroed) vs extraretinal-only (retina stabilized)
  D  Single-trial r^2 (vs the PSTH-median reference line): full twin vs
     retinal-only vs extraretinal-only
  E  FEM modulation fraction (1-alpha): the neuron distribution vs each
     within-model twin condition, with a paired TOST equivalence test — Full
     and Ablated reproduce the empirical FEM modulation, Stabilized does not.

Panels C/D draw on the unified analysis-row cache
(`fig3_bottomrow_ablation.pkl`) on the fig2 inclusion population (rate > 2 Hz &
PSTH R^2 > 0.05). Panel E is computed by `_fig3_femfraction` on the same fig2
frame + intersection population, so C/D/E describe (almost) the same cells fig2
reports.

Usage:
    uv run python paper/fig3/generate_figure3.py [--recompute]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import wilcoxon

from VisionCore.paths import VISIONCORE_ROOT

from _fig3_data import FIG_DIR, configure_matplotlib
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from _fig3_femfraction import compute_femfraction_data, CONDITIONS as FEM_CONDITIONS
from _fig3a_data import load_panel_a_assets
from generate_fig3a import plot_panel_a


# Condition colors: intact/full = blue, behavior-ablated (retinal only) = red,
# stabilized (extraretinal only) = purple, PSTH = grey.
INTACT_COLOR = "#1f77b4"
ABLATED_COLOR = "#d62728"
STABILIZED_COLOR = "#9467bd"
PSTH_COLOR = "0.55"
SCATTER_COLOR = "0.35"
ACCENT = "#c0392b"
PANEL_LETTER_SIZE = 10   # match fig2's panel-letter size
PANEL_TITLE_SIZE = 8.0


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
    whiskers at the 2.5th and 97.5th percentiles (interpretable fixed quantiles
    for this large-N summary, rather than Tukey 1.5*IQR). Fliers hidden. The fill
    tint signifies the condition. Returns the boxplot dict."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    bp = ax.boxplot(groups, positions=positions, widths=width,
                    patch_artist=True, showfliers=False, whis=(2.5, 97.5),
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


def _sig_bracket(ax, x1, x2, y, p, *, h, gap, color="k", fontsize=7.5,
                 delta=None):
    """Significance bracket at height y. Reading upward from the bracket line:
    optional `delta` (effect-size string; may contain a newline to add a second
    line below it), then the significance stars. The p-value is intentionally
    omitted — it is redundant with the stars. `delta` carries the same weight as
    the stars (matched font/colour). `gap` is the per-line vertical step; the
    stars clear a multi-line delta. Returns the top y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=color, lw=0.9, clip_on=False)
    xc = (x1 + x2) / 2
    yc = y + h
    if delta is not None:
        ax.text(xc, yc, delta, ha="center", va="bottom",
                fontsize=fontsize - 0.5, color=color, clip_on=False)
        yc += gap * (delta.count("\n") + 1)
    ax.text(xc, yc, _stars(p), ha="center", va="bottom",
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

    # Horizontal dashed guide at each condition's median, spanning the panel so
    # the three medians can be read off against one another at a glance.
    for grp, color in zip((gi, ga, gs),
                          (INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR)):
        ax.axhline(float(np.median(grp)), color=color, lw=0.9,
                   ls=(0, (4, 3)), alpha=0.55, zorder=1)

    _box_whisker(ax, [gi, ga, gs], [0, 1, 2],
                 [INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR])

    intact_med = float(np.median(gi))
    p_z = wilcoxon(gi, ga).pvalue
    p_s = wilcoxon(gi, gs).pvalue
    d_z = float(np.median(ga - gi))   # extraretinal ablation cost (retinal only)
    d_s = float(np.median(gs - gi))   # reafferent ablation cost (extraretinal only)
    pct_z = 100.0 * abs(d_z) / intact_med if intact_med != 0 else np.nan
    pct_s = 100.0 * abs(d_s) / intact_med if intact_med != 0 else np.nan
    # ccnorm is bounded at 1: keep 1.0 as the top tick but extend the axis so the
    # two stacked significance connectors sit clear above the distributions.
    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.arange(0, 1.001, 0.2))
    _sig_bracket(ax, 0, 1, 1.02, p_z, h=0.014, gap=0.058,
                 delta=f"Δ={d_z:+.3f}\n({pct_z:.0f}% of total)")
    _sig_bracket(ax, 0, 2, 1.24, p_s, h=0.014, gap=0.058,
                 delta=f"Δ={d_s:+.3f}\n({pct_s:.0f}% of total)")

    ax.set_xlim(-0.6, 2.9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Retinal +\nbehavioral\n(full)", "Retinal\nonly\n(ablated)",
                        "Extraretinal\nonly\n(stabilized)"], fontsize=5.3)
    ax.set_ylabel("Held-out prediction\n(ccnorm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"Panel C — ccnorm (N={m.sum()}): intact med={intact_med:.3f}, "
          f"zeroed med={np.median(ga):.3f} (Δ={d_z:+.3f}, p={p_z:.2e}), "
          f"stabilized med={np.median(gs):.3f} (Δ={d_s:+.3f}, p={p_s:.2e})")


# ---------------------------------------------------------------------------
# Panel D — single-trial r^2 (PSTH baseline vs full vs ablated)
# ---------------------------------------------------------------------------
def _plot_singletrial_r2_violins(ax, abl):
    pop = np.asarray(abl["cd_population"], dtype=bool)
    psth = np.asarray(abl["ve_psth"], dtype=float)
    full = np.asarray(abl["ve"]["intact"], dtype=float)
    ablated = np.asarray(abl["ve"]["zeroed"], dtype=float)
    stab = np.asarray(abl["ve"]["stabilized"], dtype=float)
    m = pop & _finite_mask(psth, full, ablated, stab)
    gp, gf, ga, gs = psth[m], full[m], ablated[m], stab[m]

    # PSTH is demoted from a box to just its median reference line (kept as the
    # "trial-average" baseline); the three model conditions are the boxes.
    _box_whisker(ax, [gf, ga, gs], [0, 1, 2],
                 [INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR])

    psth_med = float(np.median(gp))
    conc = np.concatenate([gf, ga, gs])
    lo = float(min(np.nanpercentile(conc, 1), psth_med))
    hi = float(np.nanpercentile(conc, 99))
    rng = hi - lo
    # Bottom of the axis keys to the lowest whisker (2.5th percentile), which
    # dips below 0 for the stabilized condition — so the box whiskers/caps are
    # fully visible rather than clipped. Still include 0 (the reference line).
    whis_lo = min(s["whislo"] for s in boxplot_stats([gf, ga, gs],
                                                     whis=(2.5, 97.5)))
    y_bottom = min(0.0, whis_lo - 0.04 * rng)
    ax.set_ylim(y_bottom, hi + 0.62 * rng)

    # Baseline: the leave-one-out PSTH-ceiling median (kept as a reference line;
    # the per-condition PSTH contrasts are reported in the printout/text).
    ax.axhline(psth_med, color="0.55", lw=0.8, ls="--", alpha=0.8, zorder=0)
    # Label sits above the reference line, just right of the stabilized whiskers
    # (kept inside the panel via the extended x-limit) so it no longer collides
    # with panel E's y-axis label.
    ax.text(2.28, psth_med + 0.006, "Trial avg.\nmedian", color="0.5",
            fontsize=5.4, va="bottom", ha="left", clip_on=False)

    # Two ablation contrasts vs the full twin, staggered so the brackets never
    # overlap: the small extraretinal (zeroed) cost sits lower, the large
    # reafferent (stabilized) cost above it. Each carries stars + its median Δ,
    # contextualised as a fraction of the full twin's single-trial r^2 lost.
    p_fz = wilcoxon(gf, ga).pvalue
    p_fs = wilcoxon(gf, gs).pvalue
    full_med = float(np.median(gf))
    d_fz = float(np.median(ga - gf))   # zeroed - full: extraretinal ablation cost
    d_fs = float(np.median(gs - gf))   # stabilized - full: reafferent ablation cost
    pct_fz = 100.0 * abs(d_fz) / full_med if full_med != 0 else np.nan
    pct_fs = 100.0 * abs(d_fs) / full_med if full_med != 0 else np.nan
    y0 = hi + 0.06 * rng
    step = 0.24 * rng
    h = 0.016 * rng
    gap = 0.072 * rng
    _sig_bracket(ax, 0, 1, y0, p_fz, h=h, gap=gap,
                 delta=f"Δ={d_fz:+.3f}\n({pct_fz:.0f}% of total)")
    _sig_bracket(ax, 0, 2, y0 + step, p_fs, h=h, gap=gap,
                 delta=f"Δ={d_fs:+.3f}\n({pct_fs:.0f}% of total)")

    ax.axhline(0, color="0.7", lw=0.6, ls=":")
    ax.set_xlim(-0.6, 2.9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Retinal +\nbehavioral\n(full)", "Retinal\nonly\n(ablated)",
                        "Extraretinal\nonly\n(stabilized)"], fontsize=5.3)
    ax.set_ylabel("Single-trial $r^2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # PSTH contrasts (line, not violins) reported here for the text.
    p_pf = wilcoxon(gf, gp).pvalue
    p_ps = wilcoxon(gs, gp).pvalue
    print(f"Panel D — single-trial r² (N={m.sum()}): PSTH med={psth_med:.4f}, "
          f"full med={full_med:.4f}, zeroed med={np.median(ga):.4f}, "
          f"stabilized med={np.median(gs):.4f}; "
          f"full-vs-zeroed Δ={d_fz:+.4f} p={p_fz:.2e}, "
          f"full-vs-stabilized Δ={d_fs:+.4f} p={p_fs:.2e}; "
          f"full-vs-PSTH p={p_pf:.2e}, stabilized-vs-PSTH p={p_ps:.2e}")


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
    """FEM modulation fraction (1 - alpha) in per-unit counts: the empirical
    neuron distribution (grey) vs each within-model twin condition (step), on
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
            "TOST equiv.\n" rf"zone ($\pm${margin:g})",
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(np.arange(0, ceiling + 0.5 * step, step))
    ax.set_xlabel("Fraction of rate modulation\ndue to FEM (1-α)")
    ax.set_ylabel("Units")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Line2D([0], [0], color="0.55", lw=5, alpha=0.5),
               Line2D([0], [0], color=INTACT_COLOR, lw=1.6),
               Line2D([0], [0], color=ABLATED_COLOR, lw=1.6),
               Line2D([0], [0], color=STABILIZED_COLOR, lw=1.6)]
    labels = ["Empirical", "Model (full)", "Model (ablated)", "Model (stabilized)"]
    leg = ax.legend(handles, labels, frameon=False, fontsize=7.2, loc="upper left",
                    handlelength=1.3, handletextpad=0.5, labelspacing=0.35,
                    borderaxespad=0.2)
    leg.set_zorder(7)


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


def _write_sidecars(out_dir, manifest: dict):
    caption = """Figure 3. A retinal-input digital twin captures FEM-linked V1 response variability.

(A) Training objective and held-out test. The twin is trained on gratings, gabors, and natural images to predict simultaneously recorded V1 spikes continuously: at each timepoint its input is a space × space × time crop of the gaze-contingent stimulus history (the natural-image "model input" cube) combined with the extraretinal behavior covariates, and its target is that timepoint's population spike counts (the units × time raster, with the single predicted bin highlighted). The fixated-flashed-image test stimulus (right) runs through the same pipeline but was held out during training. (B) Gaze-contingent digital twin architecture. The model receives the retinal stimulus history (a moving, reafferent space × space × time crop) and an optional extraretinal behavior input, then predicts simultaneously recorded V1 responses. The schematic depicts both within-model ablation routes quantified in C–E: the behavior input can be zeroed (the Full/Ablated switch), and the retinal input can be stabilized — frozen so it no longer moves with the eye (the second, temporally constant cube). (C, D) Two symmetric within-model ablations isolate the twin's two FEM information routes, pooled across reliable Allen and Logan cells (matching the fig. 2 session population, >=10 analyzed units/session): retinal-only zeroes the separate extraretinal behavior input, and extraretinal-only stabilizes the retinal input by freezing it at one common (session-global centroid) gaze so the image no longer moves with the eye (behavior intact). (C) Held-out, trial-averaged prediction (normalized correlation, ccnorm). Removing the extraretinal pathway lowers the trial-averaged prediction only slightly, whereas stabilizing the retinal input lowers it more, though much of the mean response survives. (D) Single-trial prediction (r^2) against the leave-one-out PSTH median (dashed reference line). The twin predicts single trials well above the PSTH ceiling with the extraretinal pathway zeroed, but stabilizing the retinal input collapses single-trial prediction to at or below the PSTH baseline — so the twin's trial-to-trial predictive power is carried by the moving retinal image (reafference), not by extraretinal modulation. (E) FEM modulation fraction (\\(1-\\alpha\\), the fraction of rate modulation due to FEM — the same quantity as fig. 2), in per-unit counts. The grey filled distribution is the neurons; each within-model twin condition overlays as a step histogram, all on the fig. 2 fixation frame and intersection population. Downward triangles mark each distribution's median. The shaded band is the empirical median \\(\\pm 0.1\\), a paired two-one-sided-t (TOST) equivalence zone (margin \\(\\Delta=0.1\\); the verdict is robust for any \\(\\Delta\\ge0.05\\)): a condition whose median falls inside is statistically equivalent to the neurons. The full twin and the behavior-ablated (retinal-only) twin are both equivalent to the empirical FEM modulation (\\(\\equiv\\) neurons; median offset ~0.03, TOST \\(p<10^{-20}\\)), whereas stabilizing the retinal input abolishes it (\\(\\neq\\) neurons; median 0.20 vs 0.67). Reafference alone reproduces the FEM-driven rate modulation that drives the fig. 2 population structure.
"""
    (out_dir / "figure3_caption.md").write_text(caption, encoding="utf-8")

    readme = """# Figure 3

Generated by `paper/fig3/generate_figure3.py`.

The digital-twin mechanism figure: a retinal-input twin whose single-trial
prediction survives zeroing the extraretinal eye-state pathway, and whose FEM
modulation fraction (1-alpha) reproduces the empirical distribution under the
full and behavior-ablated conditions but not when the retinal image is
stabilized. Panels C/D use `fig3_bottomrow_ablation.pkl`; panel E uses the
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


def compose(*, recompute: bool = False, out_dir=FIG_DIR, dpi: int = 300):
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
    })
    out_dir.mkdir(parents=True, exist_ok=True)

    abl = load_ablation_data(recompute=recompute) if recompute else _load_ablation_cache()
    assets = load_panel_a_assets(recompute=recompute)

    # Panel A is a two-row schematic (aspect ≈ 1.1), so it needs a taller top
    # slot to render wide enough for its architecture labels to breathe.
    fig = plt.figure(figsize=(8.5, 9.7), constrained_layout=False)
    gs = GridSpec(
        2, 1,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.052,
        top=0.955,
        height_ratios=[2.4, 1.0],
        hspace=0.11,
    )

    # Row 1. Native schematic (stimulus + architecture), fitted into the slot.
    # The A/B panel letters and the grey divider are drawn inside the schematic
    # (see generate_fig3a._draw_all), so no composite letter is placed here.
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax=ax_a, assets=assets)

    # Row 2. Three analysis panels: C/D box-and-whisker, E the FEM-fraction
    # distribution overlay (no marginal axis).
    gs_mid = gs[1, 0].subgridspec(
        1, 5,
        width_ratios=[0.12, 1.0, 1.0, 1.25, 0.12],
        wspace=0.5,
    )

    ax_c = fig.add_subplot(gs_mid[0, 1])
    ax_d = fig.add_subplot(gs_mid[0, 2])
    ax_e = fig.add_subplot(gs_mid[0, 3])

    if abl is not None:
        _plot_ccnorm_violins(ax_c, abl)
        _plot_singletrial_r2_violins(ax_d, abl)
        femdata = {c: compute_femfraction_data(condition=c) for c in FEM_CONDITIONS}
        _plot_femfraction(ax_e, femdata)
    else:
        for a in (ax_c, ax_d, ax_e):
            _plot_missing_cache(a)

    _standard_panel_heading(ax_c, "C", "Ablations modestly reduce\ntrial-averaged predictions")
    _standard_panel_heading(ax_d, "D", "Single-trial prediction needs\nthe moving retinal image")
    _standard_panel_heading(ax_e, "E", "Reafference reproduces the\nempirical FEM modulation")

    # No bbox_inches="tight": keep the canvas at exactly the intended
    # page-width figsize (8.5 in) rather than cropping to the ink bounds.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"figure3.{ext}", dpi=dpi)

    manifest = {
        "figure": "figure3",
        "analysis_row_cache": str(ABLATION_CACHE_PATH),
        "analysis_row_cache_present": abl is not None,
        "source_script": str(__file__),
        "panel_mapping": {
            "A": "training and test stimuli (schematic provenance row)",
            "B": "digital-twin architecture schematic",
            "C": "trial-averaged held-out ccnorm: full vs retinal-only (zeroed) "
                 "vs extraretinal-only (stabilized)",
            "D": "single-trial r2 vs PSTH-median line: full vs retinal-only "
                 "vs extraretinal-only (stabilized)",
            "E": "FEM modulation fraction (1-alpha): neuron distribution vs each "
                 "within-model twin condition, paired TOST equivalence test",
        },
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
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = FIG_DIR if args.out_dir is None else Path(args.out_dir)
    fig, _manifest = compose(recompute=args.recompute, out_dir=out_dir, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved Figure 3 to: {out_dir}")


if __name__ == "__main__":
    main()
