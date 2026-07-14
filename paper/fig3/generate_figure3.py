"""Figure 3: a retinal-input digital twin captures FEM-linked V1 variability.

Renders the digital-twin mechanism figure:

  A  Training and test stimuli (schematic provenance row)
  B  Digital twin schematic (architecture render)
  C  Held-out (trial-averaged) prediction: intact vs behavior-ablated ccnorm
  D  Single-trial r^2: PSTH baseline vs full twin vs behavior-ablated twin
  E  Empirical FEM modulation (1-alpha) vs the ablated twin's single-trial
     r^2 gain over the PSTH baseline, with a marginal gain distribution

Panels C/D/E all draw on the single unified analysis-row cache
(`fig3_bottomrow_ablation.pkl`), so they share sessions, neurons, and the
`good` reliability mask.

Usage:
    uv run python paper/fig3/generate_figure3.py [--recompute]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from VisionCore.paths import VISIONCORE_ROOT

from _fig3_data import FIG_DIR, configure_matplotlib
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from _fig3a_data import load_panel_a_assets
from generate_fig3a import plot_panel_a


# Condition colors: intact/full = blue, behavior-ablated = red, PSTH = grey.
INTACT_COLOR = "#1f77b4"
ABLATED_COLOR = "#d62728"
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
# Shared violin / significance helpers
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


def _violin_box(ax, groups, positions, colors, *, width=0.72, box_width=0.16):
    """Grey violin bodies (fig2's neutral grammar) with a per-condition cross
    overlay: a vertical stroke spanning the IQR (25-75) and a horizontal median
    tick, coloured by condition. Robust median/IQR analog of fig2's
    marker+errorbar violin grammar. Returns the violin parts."""
    parts = ax.violinplot(groups, positions=positions, widths=width,
                          showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("0.82")
        body.set_edgecolor("none")
        body.set_alpha(0.9)
        body.set_zorder(1)
    for grp, pos, color in zip(groups, positions, colors):
        arr = np.asarray(grp, dtype=float)
        arr = arr[np.isfinite(arr)]
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        # Vertical stroke = interquartile range (25-75).
        ax.plot([pos, pos], [q1, q3], color=color, lw=1.4,
                solid_capstyle="round", zorder=3)
        # Horizontal stroke = median.
        ax.plot([pos - box_width, pos + box_width], [med, med], color=color,
                lw=2.0, solid_capstyle="round", zorder=4)
    return parts


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
    good = np.asarray(abl["good"], dtype=bool)
    intact = np.asarray(abl["ccnorm"]["intact"], dtype=float)
    ablated = np.asarray(abl["ccnorm"]["zeroed"], dtype=float)
    m = good & _finite_mask(intact, ablated)
    gi, ga = intact[m], ablated[m]

    _violin_box(ax, [gi, ga], [0, 1], [INTACT_COLOR, ABLATED_COLOR])

    p = wilcoxon(gi, ga).pvalue
    d = float(np.median(ga - gi))   # Ablated - Full: a decrease under ablation
    # ccnorm is bounded at 1 (normalized correlation): keep 1.0 as the top tick,
    # but extend the axis above it so the significance connector sits clear above
    # the distributions rather than crowding them.
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.001, 0.2))
    _sig_bracket(ax, 0, 1, 1.02, p, h=0.014, gap=0.045, delta=f"Δ={d:+.3f}")

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full", "Ablated"])
    ax.set_ylabel("Held-out prediction\n(ccnorm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"Panel C — ccnorm (N={m.sum()}): intact med={np.median(gi):.3f}, "
          f"ablated med={np.median(ga):.3f}, Δ={d:+.3f}, Wilcoxon p={p:.2e}")


# ---------------------------------------------------------------------------
# Panel D — single-trial r^2 (PSTH baseline vs full vs ablated)
# ---------------------------------------------------------------------------
def _plot_singletrial_r2_violins(ax, abl):
    good = np.asarray(abl["good"], dtype=bool)
    psth = np.asarray(abl["ve_psth"], dtype=float)
    full = np.asarray(abl["ve"]["intact"], dtype=float)
    ablated = np.asarray(abl["ve"]["zeroed"], dtype=float)
    m = good & _finite_mask(psth, full, ablated)
    gp, gf, ga = psth[m], full[m], ablated[m]

    _violin_box(ax, [gp, gf, ga], [0, 1, 2],
                [PSTH_COLOR, INTACT_COLOR, ABLATED_COLOR])

    lo = float(np.nanpercentile(np.concatenate([gp, gf, ga]), 1))
    hi = float(np.nanpercentile(np.concatenate([gp, gf, ga]), 99))
    rng = hi - lo
    ax.set_ylim(min(0.0, lo), hi + 0.66 * rng)

    # Baseline: the PSTH-ceiling median that the twin conditions beat.
    psth_med = float(np.median(gp))
    ax.axhline(psth_med, color="0.55", lw=0.8, ls="--", alpha=0.8, zorder=0)
    ax.text(2.58, psth_med, "PSTH\nmedian", color="0.5", fontsize=5.6,
            va="center", ha="left", clip_on=False)

    # Three paired comparisons, staggered so the brackets never overlap: the
    # adjacent Full-vs-Ablated bracket sits lowest, then the two PSTH contrasts.
    # Each carries stars + its median Δ (p-values omitted as redundant with the
    # stars). The Full->Ablated Δ is shown as a decrease (the ablation cost) and
    # contextualised as a fraction of the full twin's single-trial r^2 that is
    # lost by ablating — a small effect whose raw star would otherwise over-read
    # a negligible difference.
    p_fa = wilcoxon(gf, ga).pvalue
    p_pf = wilcoxon(gf, gp).pvalue
    p_pa = wilcoxon(ga, gp).pvalue
    full_med = float(np.median(gf))
    d_fa = float(np.median(ga - gf))   # Ablated - Full: a decrease under ablation
    d_pf = float(np.median(gf - gp))
    d_pa = float(np.median(ga - gp))
    pct_fa = 100.0 * abs(d_fa) / full_med if full_med != 0 else np.nan
    y0 = hi + 0.08 * rng
    step = 0.20 * rng
    h = 0.016 * rng
    gap = 0.072 * rng   # clears the taller (larger-font) multi-line Δ labels
    _sig_bracket(ax, 1, 2, y0, p_fa, h=h, gap=gap,
                 delta=f"Δ={d_fa:+.3f}\n({pct_fa:.0f}% of total)")
    _sig_bracket(ax, 0, 1, y0 + step, p_pf, h=h, gap=gap,
                 delta=f"Δ={d_pf:+.3f}")
    _sig_bracket(ax, 0, 2, y0 + 2 * step, p_pa, h=h, gap=gap,
                 delta=f"Δ={d_pa:+.3f}")

    ax.axhline(0, color="0.7", lw=0.6, ls=":")
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["PSTH", "Full", "Ablated"])
    ax.set_ylabel("Single-trial $r^2$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"Panel D — single-trial r² (N={m.sum()}): PSTH med={np.median(gp):.4f}, "
          f"full med={np.median(gf):.4f}, ablated med={np.median(ga):.4f}; "
          f"full-vs-PSTH Δ={d_pf:+.4f} p={p_pf:.2e}, "
          f"ablated-vs-PSTH Δ={d_pa:+.4f} p={p_pa:.2e}, "
          f"full-vs-ablated Δ={d_fa:+.4f} p={p_fa:.2e}")


# ---------------------------------------------------------------------------
# Panel E — FEM modulation vs ablated single-trial gain over PSTH
# ---------------------------------------------------------------------------
def _plot_payoff_with_marginal(ax, ax_marg, abl):
    good = np.asarray(abl["good"], dtype=bool)
    alpha = np.asarray(abl["alpha"], dtype=float)
    ve_abl = np.asarray(abl["ve"]["zeroed"], dtype=float)
    ve_psth = np.asarray(abl["ve_psth"], dtype=float)
    m = good & _finite_mask(alpha, ve_abl, ve_psth) & (ve_psth > 0)
    x = 1.0 - alpha[m]
    y = ve_abl[m] / ve_psth[m]

    ymax = float(np.nanpercentile(y, 98))
    ymax = max(2.0, ymax)

    ax.scatter(x, y, s=6, alpha=0.5, color=SCATTER_COLOR, linewidths=0)
    ax.axhline(1, color="k", ls="--", lw=0.5, alpha=0.5)

    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, b * xs + a, color=ACCENT, lw=1.6, zorder=5)

    sr = spearmanr(x, y)
    rho, pval = sr.correlation, sr.pvalue
    # Stars sit above rho, but must stay below y=0.98: _clear_panel_heading()
    # culls any transAxes text with y >= 0.98 near the left edge.
    ax.text(0.045, 0.91, _stars(pval), transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="0.12")
    ax.text(0.04, 0.85, f"ρ = {rho:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="0.12")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Fraction of rate modulation\ndue to FEM")
    ax.set_ylabel("Single-trial $r^2$ gain\n(Ablated / PSTH)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right marginal: distribution of the r^2 gain, sharing the scatter's y.
    med = float(np.median(y))
    bins = np.linspace(0, ymax, 31)
    counts, _, _ = ax_marg.hist(np.clip(y, 0, ymax), bins=bins,
                                orientation="horizontal",
                                color=SCATTER_COLOR, alpha=0.55, edgecolor="none")
    cmax = float(counts.max())
    ax_marg.axhline(1, color="k", ls="--", lw=0.5, alpha=0.5)
    # Median: dark-grey dashed line ending exactly at a left-pointing triangle
    # (so the line never pokes past the marker), with the value in black to the
    # marker's right (the triangle is identified as the median in the caption).
    tri_x = cmax * 1.18
    ax_marg.plot([0, tri_x], [med, med], color="0.3", ls="--", lw=1.2, zorder=5)
    ax_marg.plot(tri_x, med, marker="<", ms=6, color="0.3", mec="0.3",
                 clip_on=False, zorder=6)
    ax_marg.text(tri_x + cmax * 0.26, med, f"{med:.2f}", color="black",
                 fontsize=6.5, va="center", ha="left", clip_on=False)
    ax_marg.set_ylim(0, ymax)
    ax_marg.set_xlim(0, cmax * 1.55)
    # Count axis (shared y with the scatter, ticks kept but unlabelled to show
    # the shared axis; x labelled 'Units', re-centred under the bars). Drop the
    # 0 tick (it collides with the scatter's 1.0 x-label); keep a single rounded
    # tick near the max count.
    ax_marg.set_xticks([int(round(cmax / 10.0) * 10)])
    ax_marg.set_xlabel("Units")
    ax_marg.xaxis.set_label_coords(0.38, -0.16)
    ax_marg.tick_params(axis="y", labelleft=False, left=True)
    for side in ("top", "right"):
        ax_marg.spines[side].set_visible(False)
    ax_marg.spines["left"].set_visible(True)
    ax_marg.spines["bottom"].set_visible(True)

    print(f"Panel E — payoff (N={m.sum()}): Spearman ρ={rho:.3f} (p={pval:.2e}); "
          f"gain median={med:.2f}, frac>1={np.mean(y > 1):.2f}, "
          f"OLS slope={b:.2f}")


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

(A) Training and test stimuli. The twin is trained on gratings, gabors, and natural images, and evaluated on fixated flashed images; the retinal model input is a space × space × time crop of the gaze-contingent stimulus history. (B) Gaze-contingent digital twin architecture. The model receives the retinal stimulus history and an optional extraretinal behavior input, then predicts simultaneously recorded V1 responses. (C) Held-out, trial-averaged response prediction (normalized correlation, ccnorm) for the full twin and the same twin with the separate extraretinal behavior input ablated (zeroed), pooled across reliable Allen and Logan cells (matching the fig. 2 session population, >=10 analyzed units/session). Ablating the extraretinal pathway costs only a small, though significant, amount of trial-averaged prediction, confirming the twin is competitive at the PSTH level and that extraretinal signals contribute little on average. (D) Single-trial prediction (r^2) for the leave-one-out PSTH baseline, the full twin, and the behavior-ablated twin. The twin predicts single trials better than the PSTH ceiling even with the extraretinal pathway zeroed, so its trial-to-trial predictive power comes from the moving retinal image. (E) The behavior-ablated twin's single-trial r^2 gain over the PSTH baseline grows with a cell's empirical FEM modulation \\(1-\\alpha\\), the fraction of rate modulation due to FEM (OLS fit; Spearman rho with p-value inset; right marginal shows the per-unit gain distribution, with the left-pointing triangle marking the median). Retinal-input prediction alone tracks the empirically measured increase in apparent rate variance under fixational eye movements.
"""
    (out_dir / "figure3_caption.md").write_text(caption, encoding="utf-8")

    readme = """# Figure 3

Generated by `paper/fig3/generate_figure3.py`.

The digital-twin mechanism figure: a retinal-input twin whose single-trial
prediction survives zeroing the extraretinal eye-state pathway, with the largest
single-trial gains over a PSTH baseline for cells with stronger empirical FEM
modulation. Analysis panels C/D/E share one cache
(`fig3_bottomrow_ablation.pkl`).

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
    fig = plt.figure(figsize=(8.5, 8.9), constrained_layout=False)
    gs = GridSpec(
        2, 1,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.052,
        top=0.955,
        height_ratios=[2.7, 1.0],
        hspace=0.11,
    )

    # Row 1. Native schematic (stimulus + architecture), fitted into the slot.
    # The A/B panel letters and the grey divider are drawn inside the schematic
    # (see generate_fig3a._draw_all), so no composite letter is placed here.
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax=ax_a, assets=assets)

    # Row 2. Three analysis panels. Panel E is wider to host its right marginal.
    gs_mid = gs[1, 0].subgridspec(
        1, 5,
        width_ratios=[0.12, 1.0, 1.15, 1.65, 0.12],
        wspace=0.5,
    )

    ax_c = fig.add_subplot(gs_mid[0, 1])
    ax_d = fig.add_subplot(gs_mid[0, 2])
    gs_e = gs_mid[0, 3].subgridspec(1, 2, width_ratios=[1.0, 0.26], wspace=0.06)
    ax_e = fig.add_subplot(gs_e[0, 0])
    ax_e_marg = fig.add_subplot(gs_e[0, 1], sharey=ax_e)

    if abl is not None:
        _plot_ccnorm_violins(ax_c, abl)
        _plot_singletrial_r2_violins(ax_d, abl)
        _plot_payoff_with_marginal(ax_e, ax_e_marg, abl)
    else:
        for a in (ax_c, ax_d, ax_e):
            _plot_missing_cache(a)
        ax_e_marg.set_axis_off()

    _standard_panel_heading(ax_c, "C", "Ablation minimally affects\ntrial-averaged predictions")
    _standard_panel_heading(ax_d, "D", "Ablated model beats\nthe PSTH ceiling")
    _standard_panel_heading(ax_e, "E", "Reafference alone recovers\nFEM-linked variability")

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
            "C": "trial-averaged held-out ccnorm, intact vs behavior-ablated",
            "D": "single-trial r2: PSTH baseline vs full twin vs ablated twin",
            "E": "empirical 1-alpha vs ablated single-trial r2 gain over PSTH, "
                 "with marginal gain distribution",
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
