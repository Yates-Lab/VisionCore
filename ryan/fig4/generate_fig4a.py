"""Figure 4 panel A: stimulus example + encoding-model schematic.

Thin composer that places two subscripts side-by-side:
  * `_fig4a_stimulus.plot_panel_a_stimulus` — training/test screens, lag cube
  * `_fig4a_architecture.plot_panel_a_architecture` — Adapter → … → Readout

Both subscripts consume the same `PanelAAssets` produced by
`_fig4a_data.load_panel_a_assets`. The stimulus subscript renders every
element in a single 3D world with a shared cabinet projection so the
panel reads as one consistent viewpoint.

Usage:
    uv run ryan/fig4/generate_fig4a.py [--recompute]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib
from _fig4a_data import load_panel_a_assets
from _fig4a_stimulus import plot_panel_a_stimulus
from _fig4a_architecture import plot_panel_a_architecture


# Width ratio of stimulus : architecture halves.
WIDTH_RATIOS = (1.0, 1.25)


def plot_panel_a(*, ax=None, subplotspec=None, fig=None,
                 assets=None, recompute=False):
    """Render panel A.

    Pass one of:
      * `subplotspec=` — embed inside an existing figure's GridSpec slot
        (preferred for the composite figure).
      * `ax=` — backwards-compatible: subdivide the given axes into two.
        The supplied axes is hidden; two new axes are created from its
        SubplotSpec.
      * neither — create a fresh standalone figure.
    """
    if assets is None:
        assets = load_panel_a_assets(recompute=recompute)

    if subplotspec is not None and fig is None:
        fig = subplotspec.get_gridspec().figure
    elif ax is not None:
        fig = ax.figure
        subplotspec = ax.get_subplotspec()
        ax.set_visible(False)
    elif subplotspec is None:
        fig = plt.figure(figsize=(14.0, 6.0))
        subplotspec = fig.add_gridspec(1, 1)[0, 0]

    gs = subplotspec.subgridspec(1, 2, width_ratios=WIDTH_RATIOS, wspace=0.02)
    ax_stim = fig.add_subplot(gs[0, 0])
    ax_arch = fig.add_subplot(gs[0, 1])

    plot_panel_a_stimulus(ax_stim, assets)
    plot_panel_a_architecture(ax_arch, assets)

    return fig, (ax_stim, ax_arch)


def main(recompute=False):
    configure_matplotlib()
    fig, _ = plot_panel_a(recompute=recompute)
    out_pdf = FIG_DIR / "panel_a_schematic.pdf"
    out_png = FIG_DIR / "panel_a_schematic.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render figure 4 panel A.")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    main(recompute=args.recompute)
