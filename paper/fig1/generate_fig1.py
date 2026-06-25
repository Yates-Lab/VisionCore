"""
Compose figure 1 into a single SVG, then export PDF and PNG via cairosvg.

Layout:
    Row 1 (3 in tall):  A | B gaze distribution + C RF map
    Rows 2-4:           D-G single-unit block | H-J population block

Only panel A is an external SVG (Illustrator schematic); the remaining panels
are rendered together inside one matplotlib figure and composited with
separate SVG panel labels.

Usage:
    uv run ryan/fig1/generate_fig1.py [-r] [--recalc-c] [--recalc-d] [--recalc-f]
"""

import argparse
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import svgutils.transform as sg
import cairosvg
from PIL import Image, ImageEnhance, ImageOps

from VisionCore.paths import FIGURES_DIR
from generate_fig1b import (
    plot_panel_b, pick_representative_session, _load_all_fixrsvp_stimuli,
    EXTENT_VIEW_MARGIN_DEG,
)
from generate_fig1c import plot_panel_c
from generate_fig1d import (
    plot_panel_d_roi, plot_panel_d_gaze, plot_panel_trial_order_raster,
    plot_panel_ef, _add_block_label, SUBJECT as EXAMPLE_SUBJECT,
    DATE as EXAMPLE_DATE, DEFAULT_CELL,
)
from generate_fig1f import plot_panel_f, draw_gaze_raster_arrows

HERE = Path(__file__).resolve().parent
FIG_DIR = FIGURES_DIR / "fig1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# fixRSVP face shown behind panel B's gaze cloud in the image-backdrop variant.
PANEL_B_IMAGE_ID = 18

# Layout in inches.
ROW_HEIGHT_IN = 3.0
PANEL_C_W_IN = 2.0
PANEL_D_W_IN = 2.0
PANEL_A_W_IN = 1.5 * PANEL_C_W_IN
PANEL_A_BASE_X_IN = 0.20
PANEL_A_X_IN = PANEL_A_BASE_X_IN + 0.12
PANEL_A_Y_IN = 0.06
PAD_IN = 0.25
PANEL_B_INSET_W_IN = 1.25
PANEL_B_INSET_H_IN = 1.25
PANEL_B_INSET_X_IN = PANEL_A_X_IN + PANEL_A_W_IN - PANEL_B_INSET_W_IN + 0.00
PANEL_B_INSET_Y_IN = PANEL_A_Y_IN + ROW_HEIGHT_IN - PANEL_B_INSET_H_IN - 0.27
PANEL_A_DPIEG_W_IN = 0.82
PANEL_A_DPIEG_H_IN = 0.82
PANEL_A_DPIEG_X_IN = PANEL_B_INSET_X_IN
PANEL_A_DPIEG_Y_IN = PANEL_B_INSET_Y_IN + 0.29

# Second + third row block.
BLOCK_HEIGHT_IN = 6.0

# Total figure size.
TOTAL_W_IN = PANEL_A_W_IN + PANEL_C_W_IN + PANEL_D_W_IN + 2 * PAD_IN
TOTAL_H_IN = ROW_HEIGHT_IN + BLOCK_HEIGHT_IN
CANVAS_H_IN = 541.348 / 72.0

# Matplotlib region spans the full width; the top-left cell is left empty
# so panel A (SVG schematic) can be composited over it.
A_RESERVE_W_IN = PANEL_A_W_IN + 0.34

# 1 inch = 96 SVG user units.
PPI = 96.0
CANVAS_W_PX = TOTAL_W_IN * PPI
CANVAS_H_PX = CANVAS_H_IN * PPI

# Final panel boxes in SVG pixels: x, y, width, height, with y measured from
# the top of the exported figure. These are intentionally explicit because
# figure 1 is being matched by hand to a reference PDF.
PANEL_BOXES_PX = {
    "B": (380, 35, 146, 146),
    "C": (568, 35, 146, 146),
    "G": (60, 281, 242, 66),
    "H_RF": (93, 376, 80, 80),
    "H_GAZE": (191, 376, 80, 80),
    "I": (59, 501, 248, 66),
    "J": (59, 618, 248, 60),
    "D_AZ": (388, 226, 292, 26),
    "D_EL": (388, 268, 292, 26),
    "E": (388, 339, 292, 236),
    "F": (388, 616, 292, 58),
}

PANEL_LABELS_PX = {
    "A": (34, 24),
    "B": (332, 24),
    "C": (532, 24),
    "D": (34, 272),
    "E": (34, 370),
    "F": (34, 474),
    "G": (34, 606),
    "H": (360, 219),
    "I": (360, 320),
    "J": (360, 598),
}

# Box (in inches) the panel-A schematic SVG is scaled into, anchored at the
# top-left. Hand-tuned against the original reference crop; the live fig1a.svg
# (4:3) is fit inside preserving aspect.
PANEL_A_SVG_W_IN = 330.0 / 96.0
PANEL_A_SVG_H_IN = 250.0 / 96.0

PANEL_LABEL_FONTSIZE_PT = 16
# svgutils sizes in SVG user units (px). matplotlib renders 16pt @ 96 DPI as
# 16 * 96/72 ≈ 21.33 px, so match that for the A label.
PANEL_LABEL_FONTSIZE_PX = PANEL_LABEL_FONTSIZE_PT * 96.0 / 72.0
PANEL_A_LABEL_FONTSIZE_PX = 20.0

FIG1_CAPTION = """# Figure 1

Population recordings in the foveal representation in marmoset V1 show strong dependence on gaze during fixation.

(A) Schematic of the experimental paradigm. A head-fixed marmoset was trained to fixate a rapidly updating sequence of flashed images. Electrical activity was recorded using laminar probes from the lateral surface of V1.

(B) The distribution of gaze during an example experiment shows tight oculomotor control by the marmosets. A majority of gaze positions were within 0.5 deg during the experiment.

(C) Receptive field locations for each experiment. Blue represents RFs of units recorded from monkey A, while green represents RFs from monkey L. The bold black contour marks the example unit shown in (E).

(D) Trial rasters for the example unit shown in (E), ordered by trial number. The single-unit responses vary across repeated presentations before any sorting by eye position.

(E) Left: A grayscale example RF for a single foveal unit with clear Gabor-like structure. Right: The position of gaze measured across all trials in a 50 ms bin. Points are colored according to their projection onto the line of maximal sensitivity, which is orthogonal to the subunits under a linear model of the unit's response.

(F) Trial rasters sorted by eye position across the RF. Sorting occurs on the same bins as in (G). Clear structure emerges when sorting by eye position.

(G) The peristimulus time histogram of responses for the example unit shown in (E). The gray band and black trace depict the overall PSTH across all trials, while the blue and red lines represent the PSTH of trials with negative or positive projections onto the line of maximum sensitivity. Since eye position is not steady throughout individual trials, the projection is computed on the average gaze position in 50 ms bins.

(H) Gaze position over time for eight representative trials colored in red and blue. The eye position traces are highly self-similar within groups, but not between groups.

(I) Population rasters for the eight representative trials from (H) and (J). Colored arrows connect the blue and red gaze traces in (H) to their matching raster groups. There is clear similarity between individual trials with similar gaze traces, and large differences at the population level between trials with dissimilar gaze, even though the difference in position is just fractions of a degree.

(J) Spiking activity averaged across all recorded units for the four red trials and four blue trials over the same time course. The black trace and gray band show the mean and uncertainty across the selected trials.
"""


def _write_caption_files():
    """Write caption/legend text beside the generated figure files."""
    for name in ("fig1_caption.md", "fig1_legend.md"):
        (FIG_DIR / name).write_text(FIG1_CAPTION, encoding="utf-8")


def _box_to_fig(box_px):
    """Convert an SVG-pixel top-left box to matplotlib figure coordinates."""
    x, y, w, h = box_px
    return [
        x / CANVAS_W_PX,
        1.0 - (y + h) / CANVAS_H_PX,
        w / CANVAS_W_PX,
        h / CANVAS_H_PX,
    ]


def _set_axes_box(ax, name):
    ax.set_in_layout(False)
    ax.set_position(_box_to_fig(PANEL_BOXES_PX[name]))


def _add_panel_labels(elements):
    for letter, (x, y) in PANEL_LABELS_PX.items():
        elements.append(sg.TextElement(
            x, y, letter, size=PANEL_A_LABEL_FONTSIZE_PX,
            weight="bold", font="DejaVu Sans",
        ))


def _render_main_svg(out_path, recalc_c=False, recalc_d=False, recalc_f=False,
                     panel_b_show_image=False, panel_c_show_image=False):
    """Render B-I together as a single full-width matplotlib
    figure. The top-left cell is left empty for panel A (composited later).

    ``panel_b_show_image`` / ``panel_c_show_image`` draw a sample fixRSVP face
    (``PANEL_B_IMAGE_ID``) behind the gaze cloud (B) and the RF contours (C).
    The face is drawn at its true on-screen extent (``±faceRadius``) and the
    view is set to that extent so it fills the panel; the 3°-image circle is
    dropped since the face itself conveys the stimulus footprint."""
    fig = plt.figure(figsize=(TOTAL_W_IN, CANVAS_H_IN), layout=None)

    # Representative session + the fixRSVP image extent (faceRadius). Cheap
    # (cached); needed for both the panel-C extent circle and the image lims.
    _b_name, _ = pick_representative_session()
    _, _face_radius = _load_all_fixrsvp_stimuli(_b_name)

    ax_trial_order = fig.add_axes(_box_to_fig(PANEL_BOXES_PX["G"]))
    plot_panel_trial_order_raster(
        ax=ax_trial_order, refresh=recalc_d, panel_letter=None,
    )
    ax_trial_order.set_title(
        "Single unit responses vary across repeats", fontsize=8.5, pad=7,
    )

    ax_d = fig.add_axes(_box_to_fig(PANEL_BOXES_PX["H_RF"]))
    _, _, roi_extent = plot_panel_d_roi(ax=ax_d, refresh=recalc_d, panel_letter=None)
    ax_d_gaze = fig.add_axes(_box_to_fig(PANEL_BOXES_PX["H_GAZE"]))
    plot_panel_d_gaze(ax=ax_d_gaze, refresh=recalc_d)

    ax_b = fig.add_axes(_box_to_fig(PANEL_BOXES_PX["B"]))
    if panel_b_show_image:
        # The gaze cloud overlaid on a sample fixRSVP face at its true
        # on-screen extent; the view fills to ±faceRadius and the 3° circle is
        # dropped (the face conveys the footprint). Gaze density outside the
        # analysis window is greyed (grey_outside_analysis default) to show
        # external gaze is not considered.
        plot_panel_b(ax=ax_b, image_id=PANEL_B_IMAGE_ID, show_stimulus=True,
                     show_extent_circle=False, lim=_face_radius)
    else:
        plot_panel_b(ax=ax_b)
    ax_b.set_title("Eye position", fontsize=8.5, pad=7)

    ax_c = fig.add_axes(_box_to_fig(PANEL_BOXES_PX["C"]))
    # The bold example-cell contour and ROI box are omitted — they read as
    # distracting.
    if panel_c_show_image:
        # Match panel B: the face backdrop at true extent, view filled to
        # ±faceRadius, no 3° image circle.
        plot_panel_c(
            ax=ax_c, refresh=recalc_c, show_extent_circle=False,
            lim=_face_radius, show_stimulus=True, image_id=PANEL_B_IMAGE_ID,
            session_name=_b_name,
        )
    else:
        # No-image baseline: 3° image-extent circle with the matched margin.
        plot_panel_c(
            ax=ax_c, refresh=recalc_c,
            show_extent_circle=True, extent_radius_deg=_face_radius,
            lim=_face_radius + EXTENT_VIEW_MARGIN_DEG,
        )
    ax_c.set_title("Receptive fields", fontsize=8.5, pad=7)

    _, pop_axes = plot_panel_f(
        fig=fig, refresh=recalc_f, panel_letters=None,
        bottom_pad=0.60, raster_height=3.70, gaze_height=1.82,
        psth_height=0.62,
    )
    _, ef_axes = plot_panel_ef(
        fig=fig, refresh=recalc_d, panel_letters=None,
        vertical_pad=(0.05, 1.35), raster_height=0.75, psth_height=0.48,
    )
    ef_axes["raster"].set_title(
        "Reordering by eye position across the RF...", fontsize=8.5, pad=7,
    )
    ef_axes["psth"].set_title(
        "... reveals previously hidden structure", fontsize=8.5, pad=7,
    )
    ef_axes["psth"].set_xlabel("Time (ms)", fontsize=8)
    pop_axes["psth"].set_xlabel("Time (ms)", fontsize=8)
    pop_axes["psth"].set_ylabel("Pop. Spikes/s", fontsize=8)

    fig.canvas.draw()
    _set_axes_box(ax_trial_order, "G")
    _set_axes_box(ax_d, "H_RF")
    _set_axes_box(ax_d_gaze, "H_GAZE")
    _set_axes_box(ax_b, "B")
    _set_axes_box(ax_c, "C")
    _set_axes_box(pop_axes["gaze_h"], "D_AZ")
    _set_axes_box(pop_axes["gaze_v"], "D_EL")
    _set_axes_box(pop_axes["raster"], "E")
    _set_axes_box(pop_axes["psth"], "F")
    _set_axes_box(ef_axes["raster"], "I")
    _set_axes_box(ef_axes["psth"], "J")

    # Gaze->raster arrows are drawn in figure coords, so place them only after
    # every panel has been moved to its final box.
    draw_gaze_raster_arrows(fig)

    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def _render_panel_b_svg(out_path):
    fig, ax = plt.subplots(
        figsize=(PANEL_B_INSET_W_IN, PANEL_B_INSET_H_IN),
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0,
    )
    plot_panel_b(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def _panel_a_legend_overlay():
    """Cover the old panel A legend area while B carries the constraint legend."""
    x_shift = (PANEL_A_X_IN - PANEL_A_BASE_X_IN) * PPI
    y_shift = PANEL_A_Y_IN * PPI
    cover_x = 82 + x_shift
    cover_y = 170 + y_shift
    return sg.fromstring(f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{TOTAL_W_IN * PPI}" height="{CANVAS_H_IN * PPI}"
         viewBox="0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}">
      <g>
        <rect x="{cover_x}" y="{cover_y}" width="185" height="72" fill="#ffffff"/>
      </g>
    </svg>
    """).getroot()


def _panel_a_inset_image():
    placeholder_w = PANEL_A_DPIEG_W_IN * PPI
    placeholder_h = PANEL_A_DPIEG_H_IN * PPI
    with Image.open(HERE / "dpieg.png") as raw:
        im = ImageOps.grayscale(raw)
        im = ImageOps.autocontrast(im, cutoff=0.5)
        im = ImageEnhance.Contrast(im).enhance(1.75)
        im = ImageEnhance.Sharpness(im).enhance(1.15)
        buf = BytesIO()
        im.save(buf, format="PNG")
    buf.seek(0)
    img = sg.ImageElement(buf, placeholder_w, placeholder_h)
    img.moveto(PANEL_A_DPIEG_X_IN * PPI, PANEL_A_DPIEG_Y_IN * PPI)
    return img


def _panel_a_inset_border():
    placeholder_x = PANEL_A_DPIEG_X_IN * PPI
    placeholder_y = PANEL_A_DPIEG_Y_IN * PPI
    placeholder_w = PANEL_A_DPIEG_W_IN * PPI
    placeholder_h = PANEL_A_DPIEG_H_IN * PPI
    return sg.fromstring(f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{TOTAL_W_IN * PPI}" height="{CANVAS_H_IN * PPI}"
         viewBox="0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}">
      <rect x="{placeholder_x}" y="{placeholder_y}"
            width="{placeholder_w}" height="{placeholder_h}"
            fill="none" stroke="#201c1d" stroke-width="1"/>
    </svg>
    """).getroot()


def compose(recalc_c=False, recalc_d=False, recalc_f=False,
            panel_b_show_image=False, panel_c_show_image=False,
            out_stem="fig1"):
    main_svg = FIG_DIR / f"_{out_stem}_main.svg"
    _render_main_svg(main_svg, recalc_c=recalc_c, recalc_d=recalc_d,
                     recalc_f=recalc_f, panel_b_show_image=panel_b_show_image,
                     panel_c_show_image=panel_c_show_image)

    fig = sg.SVGFigure(f"{TOTAL_W_IN}in", f"{CANVAS_H_IN}in")
    fig.root.set("viewBox", f"0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}")

    def _load_and_place(path, x_in, y_in, target_w_in, target_h_in):
        f = sg.fromfile(str(path))
        root = f.getroot()
        vb_w, vb_h = _viewbox_size(f.root)
        sx = (target_w_in * PPI) / vb_w
        sy = (target_h_in * PPI) / vb_h
        scale = min(sx, sy)
        root.moveto(x_in * PPI, y_in * PPI, scale_x=scale)
        return root

    main = _load_and_place(main_svg, 0.0, 0.0, TOTAL_W_IN, CANVAS_H_IN)
    # Panel A is the hand-edited Illustrator/Inkscape schematic, pulled live
    # from fig1a.svg in this directory and fit into the top-left panel box.
    panel_a = _load_and_place(HERE / "fig1a.svg", 0.0, 0.0,
                              PANEL_A_SVG_W_IN, PANEL_A_SVG_H_IN)
    elements = [main, panel_a]
    _add_panel_labels(elements)
    fig.append(elements)

    out_svg = FIG_DIR / f"{out_stem}.svg"
    fig.save(str(out_svg))

    out_pdf = FIG_DIR / f"{out_stem}.pdf"
    out_png = FIG_DIR / f"{out_stem}.png"
    cairosvg.svg2pdf(url=str(out_svg), write_to=str(out_pdf))
    cairosvg.svg2png(url=str(out_svg), write_to=str(out_png), dpi=300)

    print(f"Saved {out_svg}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")

    # The caption text is identical across variants — write it once, with the
    # canonical figure.
    if out_stem == "fig1":
        _write_caption_files()
        print(f"Saved {FIG_DIR / 'fig1_caption.md'}")
        print(f"Saved {FIG_DIR / 'fig1_legend.md'}")


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
    p.add_argument("--recalc-d", action="store_true",
                   help="Force recalc of the single-unit panels.")
    p.add_argument("--recalc-f", action="store_true",
                   help="Force recalc of the population panels.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Canonical figure: plain white backgrounds behind panels B and C.
    compose(
        recalc_c=args.recalc or args.recalc_c,
        recalc_d=args.recalc or args.recalc_d,
        recalc_f=args.recalc or args.recalc_f,
        panel_b_show_image=False, panel_c_show_image=False,
        out_stem="fig1",
    )
    # Face-backdrop variant: the sample fixRSVP face behind both panel B and
    # panel C, drawn at its true on-screen extent and filling each panel.
    # Reuses the cached panels (recalc, if requested, already ran above).
    compose(
        recalc_c=False, recalc_d=False, recalc_f=False,
        panel_b_show_image=True, panel_c_show_image=True,
        out_stem="fig1_imageBC",
    )
