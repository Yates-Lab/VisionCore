"""
Compose figure 1 into a single SVG, then export PDF and PNG via cairosvg.

Layout:
    Row 1 (3 in tall):  A | B | C
    Rows 2-4:           D-E-F block  |  G-H-I block

Only panel A is an external SVG (Illustrator schematic); the remaining panels
(B, C, D-F, G-I) are rendered together inside one matplotlib figure with
nested subfigures so spacing and labels stay coherent.

Usage:
    uv run ryan/fig1/generate_fig1.py [-r] [--recalc-c] [--recalc-d] [--recalc-f]
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import svgutils.transform as sg
import cairosvg

from VisionCore.paths import FIGURES_DIR
from generate_fig1b import plot_panel_b
from generate_fig1c import plot_panel_c
from generate_fig1d import plot_panel_d, _add_block_label
from generate_fig1f import plot_panel_f

HERE = Path(__file__).resolve().parent
FIG_DIR = FIGURES_DIR / "fig1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Layout in inches.
ROW_HEIGHT_IN = 3.0
PANEL_C_W_IN = 2.0
PANEL_B_W_IN = 2.5
PANEL_A_W_IN = 1.5 * PANEL_C_W_IN
PAD_IN = 0.25

# Second + third row block (D-F and G-I, each rendered as its own subfigure).
BLOCK_HEIGHT_IN = 6.0

# Total figure size.
TOTAL_W_IN = PANEL_A_W_IN + PANEL_B_W_IN + PANEL_C_W_IN + 2 * PAD_IN
TOTAL_H_IN = ROW_HEIGHT_IN + BLOCK_HEIGHT_IN

# Matplotlib region spans the full width; the top-left cell is left empty
# so panel A (SVG schematic) can be composited over it.
A_RESERVE_W_IN = PANEL_A_W_IN + PAD_IN

# 1 inch = 96 SVG user units.
PPI = 96.0

PANEL_LABEL_FONTSIZE_PT = 16
# svgutils sizes in SVG user units (px). matplotlib renders 16pt @ 96 DPI as
# 16 * 96/72 ≈ 21.33 px, so match that for the A label.
PANEL_LABEL_FONTSIZE_PX = PANEL_LABEL_FONTSIZE_PT * 96.0 / 72.0


def _render_main_svg(out_path, recalc_c=False, recalc_d=False, recalc_f=False):
    """Render B, C, D-F, G-I together as a single full-width matplotlib
    figure. The top-left cell is left empty for panel A (composited later)."""
    fig = plt.figure(
        figsize=(TOTAL_W_IN, TOTAL_H_IN),
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0,
    )

    top, bottom = fig.subfigures(
        2, 1, height_ratios=[ROW_HEIGHT_IN, BLOCK_HEIGHT_IN], hspace=0.0,
    )
    _sub_a_blank, sub_b, sub_c = top.subfigures(
        1, 3,
        width_ratios=[A_RESERVE_W_IN, PANEL_B_W_IN, PANEL_C_W_IN],
        wspace=0.0,
    )
    sub_d, sub_g = bottom.subfigures(1, 2, wspace=0.0)

    ax_b = sub_b.add_subplot(1, 1, 1)
    plot_panel_b(ax=ax_b)
    _add_block_label(ax_b, "B")

    ax_c = sub_c.add_subplot(1, 1, 1)
    plot_panel_c(ax=ax_c, refresh=recalc_c)
    _add_block_label(ax_c, "C")

    plot_panel_d(fig=sub_d, refresh=recalc_d, panel_letters=("D", "E", "F"))
    plot_panel_f(fig=sub_g, refresh=recalc_f, panel_letters=("G", "H", "I"))

    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def compose(recalc_c=False, recalc_d=False, recalc_f=False):
    main_svg = FIG_DIR / "_fig1_main.svg"
    _render_main_svg(main_svg, recalc_c=recalc_c, recalc_d=recalc_d, recalc_f=recalc_f)

    panel_a_path = HERE / "fig1a.svg"

    fig = sg.SVGFigure(f"{TOTAL_W_IN}in", f"{TOTAL_H_IN}in")
    fig.root.set("viewBox", f"0 0 {TOTAL_W_IN * PPI} {TOTAL_H_IN * PPI}")

    def _load_and_place(path, x_in, y_in, target_w_in, target_h_in):
        f = sg.fromfile(str(path))
        root = f.getroot()
        vb_w, vb_h = _viewbox_size(f.root)
        sx = (target_w_in * PPI) / vb_w
        sy = (target_h_in * PPI) / vb_h
        scale = min(sx, sy)
        root.moveto(x_in * PPI, y_in * PPI, scale_x=scale)
        return root

    main = _load_and_place(main_svg, 0.0, 0.0, TOTAL_W_IN, TOTAL_H_IN)
    panel_a = _load_and_place(panel_a_path, 0.0, 0.0,
                              PANEL_A_W_IN, ROW_HEIGHT_IN)

    label_a = sg.TextElement(
        0.05 * PPI, 0.25 * PPI, "A",
        size=PANEL_LABEL_FONTSIZE_PX, weight="bold", font="Arial",
    )

    fig.append([main, panel_a, label_a])

    out_svg = FIG_DIR / "fig1.svg"
    fig.save(str(out_svg))

    out_pdf = FIG_DIR / "fig1.pdf"
    out_png = FIG_DIR / "fig1.png"
    cairosvg.svg2pdf(url=str(out_svg), write_to=str(out_pdf))
    cairosvg.svg2png(url=str(out_svg), write_to=str(out_png), dpi=300)

    print(f"Saved {out_svg}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


def _viewbox_size(root_element):
    vb = root_element.get("viewBox") or root_element.get("viewbox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    return (_to_user_units(root_element.get("width")),
            _to_user_units(root_element.get("height")))


def _to_user_units(value):
    if value is None:
        return 1.0
    s = str(value).strip()
    units = {"in": 96.0, "cm": 96.0 / 2.54, "mm": 96.0 / 25.4, "pt": 96.0 / 72.0,
             "pc": 96.0 / 6.0, "px": 1.0}
    for u, factor in units.items():
        if s.endswith(u):
            return float(s[: -len(u)]) * factor
    return float(s)


def _parse_args():
    p = argparse.ArgumentParser(description="Compose figure 1.")
    p.add_argument("-r", "--recalc", action="store_true",
                   help="Force recalc of all cached panels (C, D, F).")
    p.add_argument("--recalc-c", action="store_true", help="Force recalc of panel C.")
    p.add_argument("--recalc-d", action="store_true", help="Force recalc of panels D-F.")
    p.add_argument("--recalc-f", action="store_true", help="Force recalc of panels G-I.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compose(
        recalc_c=args.recalc or args.recalc_c,
        recalc_d=args.recalc or args.recalc_d,
        recalc_f=args.recalc or args.recalc_f,
    )
