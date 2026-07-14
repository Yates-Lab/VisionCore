"""Figure 3: a retinal-input digital twin captures FEM-linked V1 variability.

Renders the digital-twin mechanism figure:

  A  Digital twin schematic (native stimulus + architecture render)
  B  Example neuron PSTH: observed vs intact and behavior-zeroed twin
  C  Held-out response validation (intact vs zeroed ccnorm)
  D  Extraretinal-pathway zeroing control (single-trial r^2)
  E  FEM-linked model gain over a PSTH baseline

Usage:
    uv run python paper/fig3/generate_figure3.py [--recompute]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import spearmanr

from VisionCore.paths import VISIONCORE_ROOT

from _fig3_data import FIG_DIR, configure_matplotlib, load_fig3_data
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from _fig3_helpers import select_example_neuron
from _fig3a_data import load_panel_a_assets
from generate_fig3a import plot_panel_a
from generate_fig3b import plot_panel_b as plot_example_psth


POOLED_COLOR = "0.25"
POOLED_FILL = "0.55"
BEHAVIOR_COLOR = "#d62728"
ZEROED_COLOR = "#1f77b4"
ACCENT = "#c0392b"
WITHIN_MODEL_CACHE = VISIONCORE_ROOT / "outputs" / "cache" / "behavior_vs_vision_within_model.pkl"
PANEL_LETTER_SIZE = 9
PANEL_TITLE_SIZE = 8.0
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}


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
    """Place a consistent panel letter/title just above the axes."""
    _clear_panel_heading(ax)
    ax.text(
        -0.035,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color="#202124",
        clip_on=False,
    )
    ax.text(
        0.08,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color="#202124",
        linespacing=0.9,
        clip_on=False,
    )


def _plot_example_psth_intact_vs_zeroed(ax, abl_data, *, fallback_data, fallback_example):
    """Example PSTH with behavior-input and zeroed-behavior predictions."""
    ex = None if abl_data is None else abl_data.get("example")
    if ex is None:
        plot_example_psth(
            ax=ax,
            data=fallback_data,
            example=fallback_example,
            legend_fontsize=6.0,
            show_ccnorm_title=False,
        )
        if len(ax.lines) >= 2:
            ax.lines[1].set_label("Intact")
            ax.lines[1].set_color(BEHAVIOR_COLOR)
            ax.legend(frameon=False, fontsize=6.0)
        return

    obs = np.nanmean(ex["obs_rate"], axis=0)
    intact = np.nanmean(ex["rate"]["intact"], axis=0)
    zeroed = np.nanmean(ex["rate"]["zeroed"], axis=0)
    t = np.linspace(0, float(ex["window_s"]), obs.size, endpoint=False)
    ax.plot(t, obs, color="k", lw=1.0, label="Observed")
    ax.plot(t, intact, color=BEHAVIOR_COLOR, lw=1.0, label="Intact")
    ax.plot(t, zeroed, color=ZEROED_COLOR, lw=1.0, label="Zeroed")
    ax.set_xlim(0, float(ex["window_s"]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (sp/s)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_within_model_ccnorm(data):
    """Load matched intact-vs-zeroed ccnorm arrays from the within-model cache."""
    if not WITHIN_MODEL_CACHE.exists():
        return None
    with open(WITHIN_MODEL_CACHE, "rb") as f:
        rows = dill.load(f)

    intact = np.concatenate([np.asarray(r["cc_norm"]["beh_intact"], dtype=float) for r in rows])
    zeroed = np.concatenate([np.asarray(r["cc_norm"]["beh_zeroed"], dtype=float) for r in rows])
    ccmax = np.concatenate([np.asarray(r["cc_max"]["beh_intact"], dtype=float) for r in rows])
    good = ccmax > 0.85
    return {"intact": intact, "zeroed": zeroed, "good": good}


def _plot_ccnorm_hist_intact_vs_zeroed(ax, data, *, letter: str = "C"):
    """Overlaid normalized-correlation histograms for intact and zeroed inputs."""
    matched = _load_within_model_ccnorm(data)
    if matched is None:
        intact = data["ccnorm"]
        zeroed = None
        good = np.asarray(data["good"], dtype=bool)
    else:
        intact = matched["intact"]
        zeroed = matched["zeroed"]
        good = matched["good"]

    m_intact = good & np.isfinite(intact)
    bins = np.linspace(0, 1, 21)
    ax.hist(intact[m_intact], bins=bins, color=BEHAVIOR_COLOR, alpha=0.32,
            edgecolor="none", label="Intact")
    ax.axvline(float(np.nanmedian(intact[m_intact])), color=BEHAVIOR_COLOR, lw=1.4)

    if zeroed is not None and len(zeroed) == len(intact):
        both = good & np.isfinite(intact) & np.isfinite(zeroed)
        ax.hist(zeroed[both], bins=bins, color=ZEROED_COLOR, alpha=0.32,
                edgecolor="none", label="Zeroed")
        ax.axvline(float(np.nanmedian(zeroed[both])), color=ZEROED_COLOR, lw=1.4)
        note = f"median Δ={np.nanmedian(zeroed[both] - intact[both]):+.3f}"
        print(f"Panel {letter} — intact/zeroed ccnorm (N={both.sum()}): {note}")
    else:
        note = "zeroed ccnorm\nnot cached"
        ax.text(0.97, 0.92, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.0, color=ZEROED_COLOR)
        print(f"Panel {letter} — intact ccnorm only; zeroed ccnorm not cached")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.text(0.97, 0.08,
            f"intact median {np.nanmedian(intact[m_intact]):.2f}"
            if zeroed is None else note,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.0, color="0.25")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_ablation_r2_pooled(ax, data, *, cond: str = "zeroed", letter: str = "D"):
    """Pooled intact-vs-zeroed single-trial r2 scatter."""
    x = data["ve"]["intact"]
    y = data["ve"][cond]
    good = data["good"]
    m = good & np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=5, alpha=0.38, color=POOLED_COLOR)
    lims = [0, 0.35]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Single-trial $r^2$ (intact)")
    ax.set_ylabel("Single-trial $r^2$ (zeroed)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    d = y[m] - x[m]
    med_delta = float(np.nanmedian(d))
    pct = 100.0 * med_delta / float(np.nanmedian(x[m]))
    ax.text(0.97, 0.08,
            f"{pct:+.0f}% of intact median\nmedian Δ$r^2$={med_delta:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.25")
    print(f"Panel {letter} — pooled (N={m.sum()}): median Δr²={med_delta:+.4f}")


def _plot_ablation_placeholder(ax):
    ax.set_axis_off()
    ax.text(0.5, 0.58, "ablation cache not found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(0.5, 0.42, f"Missing: {ABLATION_CACHE_PATH.name}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="0.45")


def _plot_improvement_vs_fem_modulation(ax, data, *, legend_fontsize: float = 5.8):
    """Model/PSTH single-trial r2 improvement vs empirical FEM modulation."""
    ve_model = np.asarray(data["ve_model"], dtype=float)
    ve_psth = np.asarray(data["ve_psth"], dtype=float)
    alpha = np.asarray(data["alpha"], dtype=float)
    subjects = np.asarray(data["subjects"])
    good = np.asarray(data["good"], dtype=bool)
    has_alpha = good & np.isfinite(alpha) & np.isfinite(ve_model) & np.isfinite(ve_psth) & (ve_psth > 0)

    plotted = False
    for subj, color in SUBJECT_COLORS.items():
        mask = has_alpha & (subjects == subj)
        if not mask.any():
            continue
        fem_mod = 1.0 - alpha[mask]
        improvement = ve_model[mask] / ve_psth[mask]
        ax.scatter(
            fem_mod,
            improvement,
            s=5,
            alpha=0.5,
            color=color,
            linewidths=0,
            label=subj,
        )
        plotted = True

    ax.axhline(1, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"FEM modulation ($1-\alpha$)")
    ax.set_ylabel("$r^2$ improvement\n(Model / PSTH)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    if plotted:
        ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fem_all = 1.0 - alpha[has_alpha]
    improvement_all = ve_model[has_alpha] / ve_psth[has_alpha]
    ok = np.isfinite(fem_all) & np.isfinite(improvement_all)
    rho = spearmanr(fem_all[ok], improvement_all[ok]).correlation if ok.sum() >= 3 else np.nan
    ax.text(0.97, 0.92, f"ρ={rho:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color="0.25")
    print(
        f"Panel E — improvement vs FEM modulation "
        f"(N={ok.sum()}): Spearman ρ={rho:.3f}"
    )


def _load_ablation_cache():
    """Load ablation data without triggering a heavy inference run."""
    if not ABLATION_CACHE_PATH.exists():
        return None
    return load_ablation_data(recompute=False)


def _write_sidecars(out_dir, manifest: dict):
    caption = """Figure 3. A retinal-input digital twin captures FEM-linked V1 response variability.

(A) Gaze-contingent digital twin architecture. The model receives the retinal stimulus history and an optional extraretinal behavior input, then predicts simultaneously recorded V1 responses. (B) Observed PSTH for an example reliable neuron, overlaid with predictions from the intact behavior-input twin and the same twin with the separate behavior input zeroed. (C) Held-out stimulus-locked response prediction across pooled Allen and Logan cells, shown as normalized-correlation (ccnorm) distributions for intact and behavior-zeroed predictions from the same twin. (D) Single-trial prediction is nearly unchanged when the separate extraretinal eye-state pathway is zeroed, pooled across Allen and Logan, supporting a retinal-input route for FEM-linked variability. (E) The twin's single-trial improvement over a PSTH baseline is largest for cells with stronger empirical FEM modulation, measured as \\(1-\\alpha\\).
"""
    (out_dir / "figure3_caption.md").write_text(caption, encoding="utf-8")

    readme = """# Figure 3

Generated by `paper/fig3/generate_figure3.py`.

The digital-twin mechanism figure: a retinal-input twin whose held-out response
prediction survives zeroing the extraretinal eye-state pathway, with the largest
prediction gains over a PSTH baseline for cells with stronger empirical FEM
modulation.

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

    data = load_fig3_data(recompute=recompute)
    example = select_example_neuron(data)
    abl = _load_ablation_cache()
    assets = load_panel_a_assets(recompute=recompute)

    # Panel A is now a two-row schematic (aspect ≈ 1.1), so it needs a taller
    # top slot to render wide enough for its architecture labels to breathe.
    fig = plt.figure(figsize=(8.5, 8.1), constrained_layout=False)
    gs = GridSpec(
        2, 1,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.052,
        top=0.955,
        height_ratios=[2.45, 1.0],
        hspace=0.11,
    )

    # Row 1. Native schematic (stimulus + architecture), fitted into the slot.
    ax_a = fig.add_subplot(gs[0, 0])
    rect = ax_a.get_position()
    plot_panel_a(ax=ax_a, assets=assets)
    fig.text(rect.x0, rect.y1, "A", fontweight="bold", ha="left", va="top",
             fontsize=PANEL_LETTER_SIZE, color="#202124")

    # Row 2. Digital-twin example, validation, ablation control, FEM-linked gain.
    gs_mid = gs[1, 0].subgridspec(
        1,
        6,
        width_ratios=[0.15, 1.0, 1.0, 1.0, 1.0, 0.15],
        wspace=0.36,
    )
    ax_b = fig.add_subplot(gs_mid[0, 1])
    _plot_example_psth_intact_vs_zeroed(
        ax_b,
        abl,
        fallback_data=data,
        fallback_example=example,
    )
    _standard_panel_heading(ax_b, "B", "Example PSTH")

    ax_c = fig.add_subplot(gs_mid[0, 2])
    _plot_ccnorm_hist_intact_vs_zeroed(ax_c, data, letter="C")
    _standard_panel_heading(ax_c, "C", "Held-out responses")

    ax_d = fig.add_subplot(gs_mid[0, 3])
    if abl is not None:
        _plot_ablation_r2_pooled(ax_d, abl, cond="zeroed", letter="D")
    else:
        _plot_ablation_placeholder(ax_d)
    _standard_panel_heading(ax_d, "D", "Eye-state zeroing")

    ax_e = fig.add_subplot(gs_mid[0, 4])
    _plot_improvement_vs_fem_modulation(ax_e, data)
    _standard_panel_heading(ax_e, "E", "FEM-linked model gain")

    # No bbox_inches="tight": keep the canvas at exactly the intended
    # page-width figsize (8.5 in) rather than cropping to the ink bounds.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"figure3.{ext}", dpi=dpi)

    manifest = {
        "figure": "figure3",
        "digital_twin_cache": str(VISIONCORE_ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"),
        "ablation_cache": str(ABLATION_CACHE_PATH),
        "ablation_cache_present": abl is not None,
        "within_model_cache": str(WITHIN_MODEL_CACHE),
        "source_script": str(__file__),
        "panel_mapping": {
            "A": "native digital-twin schematic (stimulus + architecture)",
            "B": "example reliable-neuron PSTH with intact and zeroed-behavior predictions",
            "C": "intact and behavior-zeroed ccnorm histograms from behavior_vs_vision_within_model cache",
            "D": "zeroed extraretinal input vs intact single-trial r2",
            "E": "model/PSTH single-trial r2 improvement vs empirical 1-alpha",
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
