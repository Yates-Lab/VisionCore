"""Figure 4 panel A: stimulus example + encoding-model schematic.

Thin composer that places two subscripts side-by-side:
  * `_fig4a_stimulus.plot_panel_a_stimulus` — training/test screens, lag cube
  * `_fig4a_architecture.plot_panel_a_architecture` — Adapter → … → Readout

Both subscripts consume the same `PanelAAssets` produced by
`_fig4a_data.load_panel_a_assets`. Each renders every element into a single
axes with a shared `set_aspect("equal")` projection.

Both halves are drawn natively into one matplotlib figure as two
manually-positioned axes. Each axes box is sized to its half's data aspect
ratio (read back after plotting), so `set_aspect("equal")` fills the box with
no dead space and the two halves share a common drawn height. Because the
output stays vector and text stays in points, the panel is editable and its
font sizes match the other panels — unlike the old approach, which stitched
two rasterized PNGs at matched pixel height (scaling the text with the
bitmap).

`render_panel_a()` builds the standalone figure (used by `main()` and by the
composite's standalone-PDF pass). `plot_panel_a()` fits the same two
matched-aspect axes inside an existing figure's slot for the composite.

Usage:
    uv run ryan/fig4/generate_fig4a.py [--recompute]
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib
from _fig4a_data import load_panel_a_assets
from _fig4a_stimulus import plot_panel_a_stimulus
from _fig4a_architecture import plot_panel_a_architecture


# Common drawn height of the standalone panel (inches). Each half's width is
# derived from this and its data aspect ratio.
PANEL_HEIGHT_IN = 4.0

# Inter-half gutter as a fraction of the panel's drawn height.
GUTTER_FRAC = 0.05

# Per-half size multipliers applied to the shared base height. Shrinking the
# stimulus and growing the architecture rebalances emphasis between the two
# halves; because each half's width is its own height times its data aspect,
# these scale both dimensions uniformly (aspect stays equal).
STIM_SCALE = 0.8
ARCH_SCALE = 1.2


def _data_aspect(ax):
    """Width:height of the axes' data limits (set by the plot function)."""
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    return (x_hi - x_lo) / (y_hi - y_lo)


def _assert_aspect_equal(ax, where):
    """Warn loudly if a plot function left the axes with non-equal aspect.
    The usual culprit is an `ax.imshow(..., aspect="auto")` call that
    silently overwrites `ax.set_aspect("equal")` — circle/marker glyphs
    then render as ellipses. Catches regressions when new imshows are
    added without the protective wrapper."""
    asp = ax.get_aspect()
    if asp != "equal" and asp != 1.0:
        warnings.warn(
            f"[{where}] axes aspect is {asp!r} after plotting — circles "
            f"will render as ellipses. Likely cause: an imshow call with "
            f"aspect='auto' (or a numeric aspect). Either drop the aspect "
            f"arg (extent= already pins the image) or re-call "
            f"ax.set_aspect('equal') after.",
            stacklevel=2,
        )


def _plot_halves(fig, ax_stim, ax_arch, assets):
    """Draw both halves and return their (stimulus, architecture) aspects."""
    plot_panel_a_stimulus(ax_stim, assets)
    _assert_aspect_equal(ax_stim, "plot_panel_a_stimulus")
    plot_panel_a_architecture(ax_arch, assets)
    _assert_aspect_equal(ax_arch, "plot_panel_a_architecture")
    return _data_aspect(ax_stim), _data_aspect(ax_arch)


def render_panel_a(assets=None, *, recompute=False,
                   height_in=PANEL_HEIGHT_IN, gutter_frac=GUTTER_FRAC,
                   stim_scale=STIM_SCALE, arch_scale=ARCH_SCALE):
    """Build a standalone panel-A figure with both halves drawn directly.

    Returns `(fig, (ax_stim, ax_arch))`. Each half's axes box exactly matches
    its data aspect ratio at a per-half height (`height_in` times its scale),
    so the two halves can carry different emphasis. The figure height is the
    taller half and each half is vertically centered — no dead space, no
    rasterization, fonts in true points.
    """
    if assets is None:
        assets = load_panel_a_assets(recompute=recompute)

    # Placeholder size; rescaled once the data aspects are known.
    fig = plt.figure(figsize=(2 * height_in, height_in))
    ax_stim = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax_arch = fig.add_axes([0.5, 0.0, 0.5, 1.0])

    a_stim, a_arch = _plot_halves(fig, ax_stim, ax_arch, assets)

    h_stim = height_in * stim_scale
    h_arch = height_in * arch_scale
    fig_h = max(h_stim, h_arch)
    gutter_in = gutter_frac * height_in
    w_stim = h_stim * a_stim
    w_arch = h_arch * a_arch
    total_w = w_stim + gutter_in + w_arch
    fig.set_size_inches(total_w, fig_h)

    y_stim = (fig_h - h_stim) / 2.0 / fig_h
    y_arch = (fig_h - h_arch) / 2.0 / fig_h
    ax_stim.set_position([0.0, y_stim, w_stim / total_w, h_stim / fig_h])
    ax_arch.set_position([(w_stim + gutter_in) / total_w, y_arch,
                          w_arch / total_w, h_arch / fig_h])
    return fig, (ax_stim, ax_arch)


def _fit_two_axes_in_rect(fig, ax_stim, ax_arch, rect, a_stim, a_arch,
                          *, gutter_frac=GUTTER_FRAC,
                          stim_scale=STIM_SCALE, arch_scale=ARCH_SCALE):
    """Position two equal-aspect axes side-by-side, centered inside `rect`.

    `rect` is the target slot as a figure-fraction Bbox. Each axes is given a
    height of a shared base times its own scale (so the halves carry different
    emphasis) and a width proportional to its data aspect, then the block is
    scaled to fit within the slot (height-limited by the taller half if the
    slot is wide enough, else width-limited) and each half is vertically
    centered. The axes are removed from the layout engine so
    constrained_layout won't override these positions.
    """
    fw, fh = fig.get_size_inches()
    slot_w_in = rect.width * fw
    slot_h_in = rect.height * fh

    # Effective per-base-height width (incl. gutter) and tallest-half scale.
    max_scale = max(stim_scale, arch_scale)
    width_per_h = stim_scale * a_stim + arch_scale * a_arch + gutter_frac * max_scale

    # Fill the slot height with the taller half; shrink to width if that overflows.
    h_in = slot_h_in / max_scale
    if h_in * width_per_h > slot_w_in:
        h_in = slot_w_in / width_per_h

    h_stim = h_in * stim_scale
    h_arch = h_in * arch_scale
    w_stim = h_stim * a_stim
    w_arch = h_arch * a_arch
    gutter_in = gutter_frac * (h_in * max_scale)
    used_w_in = w_stim + gutter_in + w_arch

    # Center the used block horizontally; center each half vertically.
    x0 = rect.x0 + (rect.width - used_w_in / fw) / 2.0
    y_stim = rect.y0 + (rect.height - h_stim / fh) / 2.0
    y_arch = rect.y0 + (rect.height - h_arch / fh) / 2.0

    ax_stim.set_position([x0, y_stim, w_stim / fw, h_stim / fh])
    ax_arch.set_position([x0 + (w_stim + gutter_in) / fw, y_arch,
                          w_arch / fw, h_arch / fh])
    for ax in (ax_stim, ax_arch):
        ax.set_in_layout(False)


def plot_panel_a(*, ax=None, subplotspec=None, fig=None,
                 assets=None, recompute=False, gutter_frac=GUTTER_FRAC,
                 stim_scale=STIM_SCALE, arch_scale=ARCH_SCALE):
    """Render panel A, fitting both halves at matched data aspect.

    Pass one of:
      * `subplotspec=` — embed inside an existing figure's GridSpec slot
        (preferred for the composite figure).
      * `ax=` — embed into the slot occupied by `ax` (the axes is hidden).
      * neither — build a fresh standalone figure via `render_panel_a`.
    """
    if assets is None:
        assets = load_panel_a_assets(recompute=recompute)

    if subplotspec is not None:
        if fig is None:
            fig = subplotspec.get_gridspec().figure
        rect = subplotspec.get_position(fig)
    elif ax is not None:
        fig = ax.figure
        ss = ax.get_subplotspec()
        rect = ss.get_position(fig) if ss is not None else ax.get_position()
        ax.set_visible(False)
        ax.set_in_layout(False)
    else:
        return render_panel_a(assets, gutter_frac=gutter_frac,
                              stim_scale=stim_scale, arch_scale=arch_scale)

    ax_stim = fig.add_axes([rect.x0, rect.y0, rect.width / 2, rect.height])
    ax_arch = fig.add_axes([rect.x0 + rect.width / 2, rect.y0,
                            rect.width / 2, rect.height])

    a_stim, a_arch = _plot_halves(fig, ax_stim, ax_arch, assets)
    _fit_two_axes_in_rect(fig, ax_stim, ax_arch, rect, a_stim, a_arch,
                          gutter_frac=gutter_frac,
                          stim_scale=stim_scale, arch_scale=arch_scale)
    return fig, (ax_stim, ax_arch)


def main(recompute=False):
    configure_matplotlib()
    assets = load_panel_a_assets(recompute=recompute)

    fig, _ = render_panel_a(assets)
    out_png = FIG_DIR / "panel_a_schematic.png"
    out_pdf = FIG_DIR / "panel_a_schematic.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render figure 4 panel A.")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    main(recompute=args.recompute)
