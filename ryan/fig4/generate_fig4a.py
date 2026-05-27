"""Figure 4 panel A: stimulus example + encoding-model schematic.

Thin composer that places two subscripts side-by-side:
  * `_fig4a_stimulus.plot_panel_a_stimulus` — training/test screens, lag cube
  * `_fig4a_architecture.plot_panel_a_architecture` — Adapter → … → Readout

Both subscripts consume the same `PanelAAssets` produced by
`_fig4a_data.load_panel_a_assets`. The stimulus subscript renders every
element in a single 3D world with a shared cabinet projection so the
panel reads as one consistent viewpoint.

`main()` renders each half into its own tight figure, then stitches the
two PNGs at matched pixel height. This guarantees vertical alignment of
the two halves and eliminates the dead space that gridspec-based
embedding produces when each axes uses `set_aspect("equal")` over
different data aspect ratios.

`plot_panel_a()` remains for embedding inside a larger composite figure
(generate_figure4.py). It tries to match each half's drawn-content
aspect ratio when sizing the gridspec slots.

Usage:
    uv run ryan/fig4/generate_fig4a.py [--recompute]
"""
from __future__ import annotations

import argparse
import io

import matplotlib.pyplot as plt
from PIL import Image

from _fig4_data import FIG_DIR, configure_matplotlib
from _fig4a_data import load_panel_a_assets
from _fig4a_stimulus import plot_panel_a_stimulus
from _fig4a_architecture import plot_panel_a_architecture


# Width ratio of stimulus : architecture halves (used in the embedded path).
WIDTH_RATIOS = (1.0, 1.25)

# Inter-panel gutter as a fraction of stitched panel height.
STITCH_GUTTER_FRAC = 0.05


def plot_panel_a(*, ax=None, subplotspec=None, fig=None,
                 assets=None, recompute=False):
    """Render panel A inline into an existing figure / subplotspec.

    Pass one of:
      * `subplotspec=` — embed inside an existing figure's GridSpec slot
        (preferred for the composite figure).
      * `ax=` — backwards-compatible: subdivide the given axes into two.
      * neither — create a fresh standalone figure.

    Note: the standalone `main()` below produces a tighter result by
    stitching independently-rendered PNGs. Use that for the published
    panel.
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

    gs = subplotspec.subgridspec(1, 2, width_ratios=WIDTH_RATIOS, wspace=0.04)
    ax_stim = fig.add_subplot(gs[0, 0])
    ax_arch = fig.add_subplot(gs[0, 1])

    plot_panel_a_stimulus(ax_stim, assets)
    plot_panel_a_architecture(ax_arch, assets)

    return fig, (ax_stim, ax_arch)


def _render_half_png(plot_fn, assets, figsize, *, dpi=300, pad_inches=0.02):
    """Render a single half into a tight PNG and return a PIL.Image."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, assets)
    fig.tight_layout(pad=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                pad_inches=pad_inches)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _stitch_halves(stim_img, arch_img, *, gutter_frac=STITCH_GUTTER_FRAC):
    """Scale both halves to a common pixel height, then concatenate.

    Height is matched to the taller of the two so the smaller half is
    upsampled (cheaper visual cost than downsampling the denser
    architecture). The gutter is sized as a fraction of the matched
    height so it scales with the figure.
    """
    target_h = max(stim_img.height, arch_img.height)

    def _scale(im, h):
        if im.height == h:
            return im
        new_w = int(round(im.width * h / im.height))
        return im.resize((new_w, h), Image.LANCZOS)

    stim_r = _scale(stim_img, target_h)
    arch_r = _scale(arch_img, target_h)
    gutter = int(round(gutter_frac * target_h))
    total_w = stim_r.width + gutter + arch_r.width
    composite = Image.new("RGBA", (total_w, target_h), (255, 255, 255, 255))
    composite.paste(stim_r, (0, 0), stim_r)
    composite.paste(arch_r, (stim_r.width + gutter, 0), arch_r)
    return composite


def main(recompute=False):
    configure_matplotlib()
    assets = load_panel_a_assets(recompute=recompute)

    stim_img = _render_half_png(plot_panel_a_stimulus, assets, (10, 4))
    arch_img = _render_half_png(plot_panel_a_architecture, assets, (12, 4.5))

    # Also persist the standalone halves so the per-subscript browsers
    # stay in sync with whatever the composite is showing.
    stim_img.convert("RGB").save(FIG_DIR / "panel_a_stimulus.png",
                                  dpi=(300, 300))
    arch_img.convert("RGB").save(FIG_DIR / "panel_a_architecture.png",
                                  dpi=(300, 300))

    composite = _stitch_halves(stim_img, arch_img)
    out_png = FIG_DIR / "panel_a_schematic.png"
    out_pdf = FIG_DIR / "panel_a_schematic.pdf"
    composite.convert("RGB").save(out_png, dpi=(300, 300))
    composite.convert("RGB").save(out_pdf, resolution=300.0)
    print(f"Saved {out_png}  ({composite.width}×{composite.height} px)")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render figure 4 panel A.")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    main(recompute=args.recompute)
