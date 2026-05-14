"""
Compose figure 1 (panels A, B, C on the top row; D, F on the second row)
into a single SVG, then export PDF and PNG via cairosvg.

Layout:
    Row 1:  A | B | C   (A is 1.5x C's width)
    Row 2:  D | F       (each half the total width)

Usage:
    uv run ryan/fig1/generate_fig1.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import svgutils.transform as sg
import cairosvg

from VisionCore.paths import FIGURES_DIR
from generate_fig1b import plot_panel_b
from generate_fig1c import plot_panel_c
from generate_fig1d import plot_panel_d
from generate_fig1f import plot_panel_f

HERE = Path(__file__).resolve().parent
FIG_DIR = FIGURES_DIR / "fig1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Layout in inches. A is 2x B's width.
ROW_HEIGHT_IN = 3.0
PANEL_C_W_IN = 2.0
# Panel B needs extra horizontal room so its square data box matches panel C
# after the colorbar is added on the right.
PANEL_B_W_IN = 2.5
PANEL_A_W_IN = 1.5 * PANEL_C_W_IN
PAD_IN = 0.25
LABEL_OFFSET_IN = 0.05

# Row 2: D and F each take half the total figure width.
ROW2_HEIGHT_IN = 6.0

# 1 inch = 96 SVG user units (matplotlib's default for SVG).
PPI = 96.0


def _render_panel_svg(plot_fn, out_path, width_in, height_in):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    plot_fn(ax=ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path)
    plt.close(fig)


def _render_multiaxis_panel_svg(plot_fn, out_path, width_in, height_in):
    """For panels that build their own multi-axis figure (D, F).
    dpi controls the resolution of rasterized artists (e.g. spike rasters)
    embedded inside the otherwise-vector SVG."""
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
    plot_fn(fig=fig)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def _panel_label(text, x_in, y_in):
    return sg.TextElement(
        x_in * PPI, y_in * PPI, text,
        size=14, weight="bold", font="Arial",
    )


def compose():
    panel_b_svg = FIG_DIR / "_fig1b_panel.svg"
    panel_c_svg = FIG_DIR / "_fig1c_panel.svg"
    panel_d_svg = FIG_DIR / "_fig1d_panel.svg"
    panel_f_svg = FIG_DIR / "_fig1f_panel.svg"

    _render_panel_svg(plot_panel_b, panel_b_svg, PANEL_B_W_IN, ROW_HEIGHT_IN)
    _render_panel_svg(plot_panel_c, panel_c_svg, PANEL_C_W_IN, ROW_HEIGHT_IN)

    panel_a_path = HERE / "fig1a.svg"

    total_w_in = PANEL_A_W_IN + PANEL_B_W_IN + PANEL_C_W_IN + 2 * PAD_IN
    # Row 2: D and F split the total width with a single pad between them.
    panel_df_w_in = (total_w_in - PAD_IN) / 2.0
    _render_multiaxis_panel_svg(plot_panel_d, panel_d_svg,
                                panel_df_w_in, ROW2_HEIGHT_IN)
    _render_multiaxis_panel_svg(plot_panel_f, panel_f_svg,
                                panel_df_w_in, ROW2_HEIGHT_IN)

    total_h_in = ROW_HEIGHT_IN + PAD_IN + ROW2_HEIGHT_IN

    fig = sg.SVGFigure(f"{total_w_in}in", f"{total_h_in}in")
    fig.root.set("viewBox", f"0 0 {total_w_in * PPI} {total_h_in * PPI}")

    def _load_and_place(path, x_in, y_in, target_w_in, target_h_in):
        f = sg.fromfile(str(path))
        root = f.getroot()
        # svgutils' moveto scaling acts in the *parent's* user-unit space,
        # but the child SVG's content is laid out in its own viewBox units.
        # We want the rendered size of the child to match target_*_in.
        # rendered_size = (parent_units / child_viewBox_units) * scale
        # parent units are at PPI (96/in); child viewBox units come from the
        # source SVG's viewBox attribute.
        vb_w, vb_h = _viewbox_size(f.root)
        sx = (target_w_in * PPI) / vb_w
        sy = (target_h_in * PPI) / vb_h
        scale = min(sx, sy)
        root.moveto(x_in * PPI, y_in * PPI, scale_x=scale)
        return root

    x = 0.0
    panel_a = _load_and_place(panel_a_path, x, 0.0, PANEL_A_W_IN, ROW_HEIGHT_IN)
    x += PANEL_A_W_IN + PAD_IN
    panel_b = _load_and_place(panel_b_svg, x, 0.0, PANEL_B_W_IN, ROW_HEIGHT_IN)
    x += PANEL_B_W_IN + PAD_IN
    panel_c = _load_and_place(panel_c_svg, x, 0.0, PANEL_C_W_IN, ROW_HEIGHT_IN)

    row2_y = ROW_HEIGHT_IN + PAD_IN
    panel_d = _load_and_place(panel_d_svg, 0.0, row2_y,
                              panel_df_w_in, ROW2_HEIGHT_IN)
    panel_f = _load_and_place(panel_f_svg, panel_df_w_in + PAD_IN, row2_y,
                              panel_df_w_in, ROW2_HEIGHT_IN)

    labels = [
        _panel_label("A", LABEL_OFFSET_IN, 0.2),
        _panel_label("B", PANEL_A_W_IN + PAD_IN + LABEL_OFFSET_IN, 0.2),
        _panel_label("C", PANEL_A_W_IN + 2 * PAD_IN + PANEL_B_W_IN + LABEL_OFFSET_IN, 0.2),
        _panel_label("D", LABEL_OFFSET_IN, row2_y + 0.2),
        _panel_label("F", panel_df_w_in + PAD_IN + LABEL_OFFSET_IN, row2_y + 0.2),
    ]

    fig.append([panel_a, panel_b, panel_c, panel_d, panel_f, *labels])

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
    """Return (width, height) of an SVG root's viewBox in its own user units.
    Falls back to the width/height attributes if no viewBox is set."""
    vb = root_element.get("viewBox") or root_element.get("viewbox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    return (_to_user_units(root_element.get("width")),
            _to_user_units(root_element.get("height")))


def _to_user_units(value):
    """Parse svgutils width/height string ('4in', '300', '76.2mm') into
    SVG user units (1 in == 96 units)."""
    if value is None:
        return 1.0
    s = str(value).strip()
    units = {"in": 96.0, "cm": 96.0 / 2.54, "mm": 96.0 / 25.4, "pt": 96.0 / 72.0,
             "pc": 96.0 / 6.0, "px": 1.0}
    for u, factor in units.items():
        if s.endswith(u):
            return float(s[: -len(u)]) * factor
    return float(s)


if __name__ == "__main__":
    compose()
