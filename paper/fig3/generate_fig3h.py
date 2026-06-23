"""Figure 3 panel H: example-neuron rasters, Twin | Twin(ablated) | residual.

The first two strips share the rate colormap; the residual (intact - ablated)
uses a diverging map on the SAME magnitude scale as the rates, so a near-empty
residual panel honestly conveys how little the twin's prediction changes when
the extraretinal behavior input is ablated.

Usage:
    uv run declan/fig3/generate_fig3h.py [--cond zeroed|permuted]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from _fig3_data import FIG_DIR, configure_matplotlib
from _fig3_ablation_data import load_ablation_data, COND_LABEL, PANEL_B_SESSION, PANEL_B_NEURON_ID


def plot_panel_h(ax=None, data=None, cond="zeroed", label_fontsize=8):
    """Draw the raster triplet on `ax`. Returns (fig, ax, im_rate, im_resid)."""
    if data is None:
        data = load_ablation_data()
    example = data["example"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 2.5))
    else:
        fig = ax.figure

    if example is None:
        ax.text(0.5, 0.5,
                f"no example neuron\n({PANEL_B_SESSION} n{PANEL_B_NEURON_ID})",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, None, None

    w = example["window_s"]
    intact = example["rate"]["intact"]
    abl = example["rate"][cond]
    resid = intact - abl
    n = intact.shape[0]
    vmax = np.nanpercentile(np.concatenate([intact, abl], axis=1), 97)

    im_rate = ax.imshow(
        np.concatenate([intact, abl], axis=1), aspect="auto", origin="upper",
        extent=[0, 2 * w, n, 0], vmin=0, vmax=vmax,
        cmap="binary", interpolation="none",
    )
    im_resid = ax.imshow(
        resid, aspect="auto", origin="upper",
        extent=[2 * w, 3 * w, n, 0], vmin=-vmax, vmax=vmax,
        cmap="RdBu_r", interpolation="none",
    )
    ax.set_xlim(0, 3 * w)
    ax.set_ylim(n, 0)
    ax.axvline(w, color="k", lw=0.8)
    ax.axvline(2 * w, color="k", lw=0.8)
    for frac, lab in [(1 / 6, "Twin"), (3 / 6, f"Twin\n({COND_LABEL[cond]})"),
                      (5 / 6, "residual")]:
        ax.text(frac, 1.02, lab, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=label_fontsize)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    return fig, ax, im_rate, im_resid


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 3 panel H.")
    p.add_argument("--cond", default="zeroed", choices=["zeroed", "permuted"])
    args = p.parse_args()

    configure_matplotlib()
    fig, ax, im_rate, im_resid = plot_panel_h(cond=args.cond)
    if im_rate is not None:
        fig.colorbar(im_rate, ax=ax, shrink=0.7, pad=0.02, label="rate (sp/s)")
        fig.colorbar(im_resid, ax=ax, shrink=0.7, pad=0.08, label="Δ rate")
    fig.tight_layout()
    out = FIG_DIR / f"panel_h_rasters_{args.cond}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
