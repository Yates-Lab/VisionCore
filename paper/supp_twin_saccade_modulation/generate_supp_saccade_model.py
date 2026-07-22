r"""Figure: a simple saccade modulator vs the twin's learned extraretinal one.

Companion to generate_supp_twin_saccade_modulation.py (same folder). Fits a
saccade-locked gain+offset modulator to the ablation gap (full - ablated) and
asks (row 1) whether it is additive or multiplicative, and (row 2) how much of
the gap it recovers on held-out trials and how stereotyped it is.

Row 1 - what is the modulation:
  A shared canonical waveforms: multiplicative gain g(τ) and additive offset a(τ).
  B example unit: actual gap STA vs model fit, split into mult + additive parts.
  C additive-only vs mult-only vs both: per-neuron recovered-gap.
Row 2 - does a simple modulator capture the complicated one:
  D recovered-gap ladder: per-neuron (ceiling) / pooled / pooled+δ.
  E pooled+δ vs per-neuron recovered (per unit).
  F example held-out trial: full / ablated / pooled prediction.

Usage:
    uv run python paper/supp_twin_saccade_modulation/generate_supp_saccade_model.py
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig3"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _fig3_data import configure_matplotlib  # noqa: E402
from _supp_saccade_model import compute_model_bundle  # noqa: E402

FIG_DIR = FIGURES_DIR / "supp_twin_saccade_modulation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GAIN_COLOR = "#0f9e8c"      # multiplicative / gain
OFFSET_COLOR = "#e07b1a"    # additive / offset
CEIL_COLOR = "#333333"
POOL_COLOR = "#d62728"


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def _violin(ax, data_list, colors, labels, ylabel, medians=True):
    xs = np.arange(len(data_list))
    parts = ax.violinplot([d[np.isfinite(d)] for d in data_list], positions=xs,
                          showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.35); pc.set_edgecolor(c)
    for x, d, c in zip(xs, data_list, colors):
        m = np.nanmedian(d)
        ax.hlines(m, x - 0.35, x + 0.35, color=c, lw=2, zorder=3)
        if medians:
            ax.text(x, m, f" {m:.2f}", va="center", ha="left", fontsize=6.5,
                    color=c)
    ax.axhline(0, lw=0.5, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    _clean(ax)


def draw_waveforms(ax, b):
    t = b["lags_ms"]
    gain = b["med_wg"] * b["waveform_g0"]        # gain deviation (1+g = gain)
    offset = b["med_wa"] * b["waveform_a0"]       # sp/s
    ax.axhline(0, lw=0.5, color="gray")
    ax.axvline(0, ls=":", lw=0.8, color="gray")
    ax.plot(t, gain, color=GAIN_COLOR, lw=1.8, label="gain  g(τ)  (×drive)")
    ax.set_ylabel("gain deviation", color=GAIN_COLOR, fontsize=8)
    ax.tick_params(axis="y", labelcolor=GAIN_COLOR)
    axt = ax.twinx()
    axt.plot(t, offset, color=OFFSET_COLOR, lw=1.8, label="offset a(τ)")
    axt.set_ylabel("offset (sp/s)", color=OFFSET_COLOR, fontsize=8)
    axt.tick_params(axis="y", labelcolor=OFFSET_COLOR, labelsize=7)
    ax.set_xlabel("time from saccade (ms)", fontsize=8)
    ax.set_title("Shared modulation waveforms", fontsize=8)
    _clean(ax)
    axt.spines["top"].set_visible(False)


def draw_example_fit(ax, b):
    ex = b["example"]
    if ex is None:
        ax.set_axis_off(); ax.text(0.5, 0.5, "no example", ha="center"); return
    t = ex["lags_ms"]
    ax.axhline(0, lw=0.5, color="gray"); ax.axvline(0, ls=":", lw=0.8, color="gray")
    ax.plot(t, ex["sta_gap"], "o", ms=2.5, color="k", label="actual gap STA")
    ax.plot(t, ex["fit_full"], "-", color=POOL_COLOR, lw=1.6, label="model fit")
    ax.plot(t, ex["fit_mult"], "--", color=GAIN_COLOR, lw=1.2, label="mult part")
    ax.plot(t, ex["fit_add"], ":", color=OFFSET_COLOR, lw=1.4, label="add part")
    ax.set_xlabel("time from saccade (ms)", fontsize=8)
    ax.set_ylabel("gap  full−ablated (sp/s)", fontsize=8)
    ax.set_title(f"Example unit fit ({ex['session']} n{ex['neuron_id']})", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    _clean(ax)


def draw_addmult(ax, b):
    g = b["good"]
    _violin(ax, [b["rec_add"][g], b["rec_mult"][g], b["rec_both"][g]],
            [OFFSET_COLOR, GAIN_COLOR, CEIL_COLOR],
            ["add\nonly", "mult\nonly", "both"],
            "recovered gap (CV)")
    ax.set_title("Additive vs multiplicative", fontsize=8)
    ax.set_ylim(-0.2, 1.0)


def draw_ladder(ax, b):
    g = b["good"]
    _violin(ax, [b["rec_both"][g], b["rec_pooled"][g], b["rec_pooled_delta"][g]],
            [CEIL_COLOR, POOL_COLOR, POOL_COLOR],
            ["per-neuron\n(ceiling)", "pooled", "pooled+δ"],
            "recovered gap (CV)")
    ax.set_title("Simple vs flexible modulator", fontsize=8)
    ax.set_ylim(-0.2, 1.0)
    ratio = np.nanmedian(b["rec_pooled_delta"][g]) / np.nanmedian(b["rec_both"][g])
    ax.text(0.5, 0.02, f"pooled+δ = {ratio:.0%} of ceiling",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)


def draw_scatter(ax, b):
    g = b["good"]
    x = b["rec_both"][g]; y = b["rec_pooled_delta"][g]
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=6, alpha=0.35, color=POOL_COLOR, edgecolor="none")
    lo, hi = -0.2, 1.0
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("per-neuron recovered", fontsize=8)
    ax.set_ylabel("pooled+δ recovered", fontsize=8)
    ax.set_title("Stereotypy (per unit)", fontsize=8)
    ax.set_aspect("equal")
    _clean(ax)


def draw_trial(ax, b):
    ex = b["example"]
    tp = ex["trial"] if ex else None
    if tp is None:
        ax.set_axis_off(); ax.text(0.5, 0.5, "no trial", ha="center"); return
    t = tp["t_ms"]
    ax.plot(t, tp["full"], color="k", lw=1.4, label="twin (full)")
    ax.plot(t, tp["abl"], color="0.6", lw=1.2, label="twin (ablated)")
    ax.plot(t, tp["pred"], color=POOL_COLOR, lw=1.4, ls="--",
            label="ablated + saccade model")
    ymax = np.nanmax(tp["full"])
    for sm in tp["sacc_ms"]:
        ax.axvline(sm, color=GAIN_COLOR, lw=0.8, alpha=0.5)
    ax.set_xlabel("time in trial (ms)", fontsize=8)
    ax.set_ylabel("rate (sp/s)", fontsize=8)
    ax.set_title(f"Held-out trial (n{ex['neuron_id']}, trial {tp['trial']})",
                 fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    _clean(ax)


def main():
    configure_matplotlib()
    b = compute_model_bundle()

    fig = plt.figure(figsize=(11, 6.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.42)
    axes = {(i, j): fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)}

    draw_waveforms(axes[(0, 0)], b)
    draw_example_fit(axes[(0, 1)], b)
    draw_addmult(axes[(0, 2)], b)
    draw_ladder(axes[(1, 0)], b)
    draw_scatter(axes[(1, 1)], b)
    draw_trial(axes[(1, 2)], b)

    for (i, j), lab in zip([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                           "ABCDEF"):
        axes[(i, j)].text(-0.16, 1.08, lab, transform=axes[(i, j)].transAxes,
                          fontsize=12, fontweight="bold", va="bottom", ha="right")

    out_pdf = FIG_DIR / "supp_saccade_model.pdf"
    out_png = FIG_DIR / "supp_saccade_model.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
