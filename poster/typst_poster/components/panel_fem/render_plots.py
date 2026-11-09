"""Render the poster's "gaze shifts within an RF" panel and the intro eye trace.

Reads only the frozen ``plot_data.npz`` and the two PNG assets next to it.
All geometry is in poster points (1/72 in) from the panel's top-left corner.
"""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
POSTER = HERE.parents[1]
sys.path.insert(0, str(POSTER))
from fonts import checked_font_paths  # noqa: E402

os.environ["MPLCONFIGDIR"] = str(HERE / ".mpl")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcdefaults()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

for path in checked_font_paths():
    fm.fontManager.addfont(str(path))
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": "panel-fem", "font.size": 21})

NAVY, TEXT, MUTED, COPPER = "#193654", "#202632", "#526474", "#B8651B"
RED, BLUE, RED_LIGHT, BLUE_LIGHT = "#c43c3c", "#3b6db7", "#df6556", "#6f95ee"
FRAME, MIRROR, WINDOW = "#1aa7d8", "#f2a7a3", "#e9edf2"
PANEL_W, PANEL_H = 1183.79296875, 974.5703125
OUT = POSTER / "assets"
d = np.load(HERE / "plot_data.npz")


def text_width(text, size, bold=False):
    import fitz
    name = "arialbd.ttf" if bold else "arial.ttf"
    return fitz.Font(fontfile=str(POSTER / ".fonts" / name)).text_length(text, size)


def marmoset_filled():
    """Marmoset line drawing with an opaque white interior, so lines drawn behind
    it (the zoom connectors) are hidden by the animal instead of crossing it."""
    from PIL import ImageDraw
    rgba = Image.open(HERE / "assets" / "marmoset.png").convert("RGBA")
    lines = np.asarray(rgba)[..., 3] > 40
    mask = Image.fromarray(np.where(lines, 0, 255).astype(np.uint8))
    w, h = mask.size
    ImageDraw.Draw(mask).line([(0, h - 1), (w - 1, h - 1)], fill=0, width=3)  # close the body
    ImageDraw.floodfill(mask, (0, 0), 128)                                     # exterior
    interior = np.asarray(mask) == 255
    out = np.asarray(rgba).copy()
    out[interior] = (255, 255, 255, 255)
    return out


def canvas(w, h):
    fig = plt.figure(figsize=(w / 72, h / 72))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, w), bg.set_ylim(h, 0), bg.axis("off")
    return fig, bg


def place(fig, w, h, x0, y0, x1, y1):
    return fig.add_axes([x0 / w, 1 - y1 / h, (x1 - x0) / w, (y1 - y0) / h])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", transparent=True)
    plt.close(fig)


def fem_panel():
    W, H = PANEL_W, PANEL_H
    fig, bg = canvas(W, H)
    ax = lambda *box: place(fig, W, H, *box)  # noqa: E731

    # Header, matching the other panels' badge and title geometry.
    bg.add_patch(Circle((29.14, 36.34), 22.13, fc=COPPER, ec="white", lw=2.5))
    bg.text(29.14, 46.5, "1", ha="center", fontsize=29, weight="bold", color="white")
    bg.text(59.74, 48.06, "Tiny gaze shifts reshape population responses",
            fontsize=48, weight="bold", color=NAVY)

    image, half = d["image"], float(d["image_half_deg"])
    gaze_hi, gaze_lo, rf = d["gaze_hi"], d["gaze_lo"], d["rf"]
    shift, diam = float(d["gaze_shift_deg"]), d["rf_diam_deg"]
    NOTE = "#25327a"

    def note(x, y, text, size=25):
        # Arial Italic is not in the locked font set; Liberation Sans Italic is
        # metrically identical and bundled under its open licence.
        bg.text(x, y, text, family="Liberation Sans", style="italic", fontsize=size,
                color=TEXT, va="top", linespacing=1.15)

    def arrow(xy_a, xy_b, color, ax_a=None, ax_b=None, lw=2.4):
        fig.add_artist(ConnectionPatch(xy_a, xy_b, coordsA="data", coordsB="data",
                                       axesA=ax_a or bg, axesB=ax_b or bg, color=color,
                                       lw=lw, arrowstyle="-|>", mutation_scale=20,
                                       shrinkA=4, shrinkB=3))

    # ---- Eye tracker ------------------------------------------------------------
    note(22, 78, "Dual Purkinje\nhigh-resolution\neye tracker")
    dpi = ax(262, 62, 402, 202)
    dpi.imshow(np.asarray(Image.open(HERE / "assets" / "dpi_eye.png")), cmap="gray",
               vmin=0, vmax=255)
    dpi.add_patch(Circle((270, 262), 222, fill=False, ec="#c9e8a0", lw=2))
    dpi.add_patch(Circle((382, 238), 30, fill=False, ec="#f2e27a", lw=1.5))
    for (tx, ty), (px, py), label in (((150, 170), (204, 278), "P4"),
                                      ((205, 410), (262, 330), "Pupil"),
                                      ((345, 395), (382, 268), "CR")):
        dpi.annotate(label, (px, py), (tx, ty), color="#c9e8a0", fontsize=17,
                     arrowprops=dict(arrowstyle="-|>", color="#d9483b", lw=1.6,
                                     mutation_scale=10, shrinkB=2))
    dpi.axis("off")
    bg.text(406, 202, "[2–4]", fontsize=15, color=TEXT, va="bottom")

    # ---- Rig (own coordinates, scaled into place) ---------------------------------
    rig = ax(20, 210, 380, 447)
    rig.set_xlim(20, 400), rig.set_ylim(345, 95), rig.axis("off")
    rig.add_patch(Polygon([(28, 168), (170, 132), (170, 262), (28, 296)], fc="#8c8c8c", ec="none"))
    rig.imshow(image, cmap="gray", extent=[96, 140, 236, 192], zorder=2)
    patch = (108, 204, 128, 224)
    rig.add_patch(Rectangle(patch[:2], 20, 20, fill=False, ec=FRAME, lw=2.5, zorder=3))
    eye = (262, 186)
    for corner in ((patch[2], patch[1]), (patch[2], patch[3])):
        rig.plot([corner[0], eye[0]], [corner[1], eye[1]], ls=(0, (4, 3)), color=FRAME, lw=2)
    rig.plot([214, 238], [232, 194], color=MIRROR, lw=5, solid_capstyle="butt")
    rig.text(150, 112, "hot mirror", fontsize=20, color=TEXT, ha="left", va="bottom")
    rig.plot([196, 222], [116, 196], color=MIRROR, lw=1.4)
    rig.add_patch(Rectangle((236, 110), 32, 26, color="#1a1a1a"))
    rig.add_patch(Polygon([(242, 136), (262, 136), (257, 154), (247, 154)], color="#1a1a1a"))
    rig.text(280, 112, "Eye camera", fontsize=20, color=TEXT, ha="left", va="bottom")
    rig.imshow(marmoset_filled(), extent=[232, 388, 336, 142], zorder=4)

    # ---- Zoomed stimulus: gaze dots and the RF on each group's trials -------------
    pts = np.vstack([gaze_hi, gaze_lo, rf + gaze_hi.mean(0), rf + gaze_lo.mean(0)])
    cx, cy = (pts.min(0) + pts.max(0)) / 2
    view = np.ptp(pts, axis=0).max() / 2 + 0.08
    zx0, zy0, zs = 30, 462, 380
    # Zoom connectors on the background layer, so the marmoset hides them.
    sx = lambda x: 20 + (x - 20) * 360 / 380  # noqa: E731  rig -> panel coordinates
    sy = lambda y: 210 + (y - 95) * 237 / 250  # noqa: E731
    for (xa, ya), (xb, yb) in (((patch[0], patch[3]), (zx0, zy0)),
                               ((patch[2], patch[3]), (zx0 + zs, zy0))):
        bg.plot([sx(xa), xb], [sy(ya), yb], ls=(0, (5, 3)), color=FRAME, lw=2.5)
    im = ax(zx0, zy0, zx0 + zs, zy0 + zs)
    im.imshow(image, cmap="gray", extent=[-half, half, -half, half], origin="upper",
              vmin=0, vmax=255, interpolation="bilinear")
    halo = [pe.Stroke(linewidth=8.5, foreground="white"), pe.Normal()]
    centres = {}
    for gaze, color in ((gaze_lo, BLUE), (gaze_hi, RED)):
        outline = rf + gaze.mean(0)
        im.plot(*np.vstack([outline, outline[:1]]).T, color=color, lw=5, path_effects=halo,
                zorder=3)
        centres[color] = outline.mean(0)
    im.annotate("", centres[RED], centres[BLUE], zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=3.5, mutation_scale=22,
                                shrinkA=0, shrinkB=0,
                                path_effects=[pe.Stroke(linewidth=7, foreground="#111111"),
                                              pe.Normal()]))
    for gaze, color in ((gaze_lo, BLUE), (gaze_hi, RED)):
        im.scatter(*gaze.T, s=170, color=color, edgecolor="white", linewidth=2.5, zorder=6)
    im.set_xlim(cx - view, cx + view), im.set_ylim(cy - view, cy + view)
    im.set_xticks([]), im.set_yticks([])
    for spine in im.spines.values():
        spine.set_color(FRAME), spine.set_linewidth(4)
    bar_deg = 1.0 if zs / (2 * view) <= zs else 0.5      # 1 deg unless wider than the view
    bar_pt = bar_deg * zs / (2 * view)
    bar_y = zy0 + zs + 18
    bg.plot([zx0, zx0 + bar_pt], [bar_y] * 2, color="black", lw=4, solid_capstyle="butt")
    bg.text(zx0 + bar_pt, bar_y + 10, "1 degree" if bar_deg == 1 else "0.5 degree",
            ha="right", va="top", fontsize=20)

    # ---- Rasters: red (higher-rate) group above blue, trials numbered upward ------
    t_lo, t_hi = d["raster_ms"]
    w0, w1 = d["window_ms"]
    n_per = int(d["n_per"])
    rx0, rx1, gap, ggap, top = 760, 1095, 5, 22, 72
    bh = (612 - ggap - 2 * (n_per - 1) * gap) / (2 * n_per)   # rasters span y 72-684
    raster = {}
    y = top
    for k in range(2 * n_per):
        group, number = (RED, n_per - k) if k < n_per else (BLUE, 2 * n_per - k)
        row = (number - 1) if k < n_per else n_per + number - 1
        a = ax(rx0, y, rx1, y + bh)
        a.axvspan(w0, w1, color=WINDOW, lw=0, zorder=0)
        sel = d["spk_k"] == row
        a.scatter(d["spk_t"][sel], d["spk_u"][sel], s=7, marker="|", color="black",
                  linewidths=0.9, zorder=2)
        a.set_xlim(t_lo, t_hi), a.set_ylim(int(d["n_units"]), -1), a.axis("off")
        bg.text(rx0 - 12, y + bh / 2, f"Trial {number}", ha="right", va="center",
                fontsize=21, color=group)
        raster[(group, number)] = (y, a)
        y += bh + (ggap if k == n_per - 1 else gap)
    y_last = raster[(BLUE, 1)][0]
    n_pt = bh / (int(d["n_units"]) + 1) * 100
    bg.plot([rx1 + 12] * 2, [y_last + bh, y_last + bh - n_pt], color="black", lw=2.2)
    bg.text(rx1 + 22, y_last + bh, "100 neurons", rotation=90, ha="left", va="bottom",
            fontsize=21)

    # ---- PSTH ---------------------------------------------------------------------
    py0, py1 = y + 14, y + 100
    ps = ax(rx0, py0, rx1, py1)
    ps.axvspan(w0, w1, color=WINDOW, lw=0, zorder=0)
    ps.plot(d["psth_t"], d["psth_hi"], color=RED, lw=3)
    ps.plot(d["psth_t"], d["psth_lo"], color=BLUE, lw=3)
    lo = min(d["psth_hi"].min(), d["psth_lo"].min())
    hi = max(d["psth_hi"].max(), d["psth_lo"].max())
    ps.set_xlim(t_lo, t_hi), ps.set_ylim(lo, hi), ps.axis("off")
    rate_bar = 20
    bg.plot([rx1 + 12] * 2, [py1, py1 - rate_bar / (hi - lo) * (py1 - py0)], color="black", lw=2.2)
    bg.text(rx1 + 22, py1, f"{rate_bar} sp/s", rotation=90, ha="left", va="bottom", fontsize=21)
    ms_pt = (rx1 - rx0) / (t_hi - t_lo)
    bg.plot([rx0, rx0 + 50 * ms_pt], [py1 + 14] * 2, color="black", lw=4, solid_capstyle="butt")
    bg.text(rx0, py1 + 24, "50 ms", va="top", fontsize=21)

    # ---- Call-outs: the argument, in three phrases ----------------------------------
    note(445, 556, "example trials\nfor different\nfixation positions", size=23)
    for gaze, color in ((gaze_hi, RED), (gaze_lo, BLUE)):
        yy, _ = raster[(color, 1)]
        arrow(tuple(gaze.mean(0)), (rx0 - 98, yy + bh / 2), color if color == RED else NOTE,
              ax_a=im)
    note(440, 690, f"RF shift ({shift:.2f}°)\nsmaller than\nRF size ({np.median(diam):.2f}°)")
    arrow((436, 730), tuple((centres[RED] + centres[BLUE]) / 2 + np.array([0, -0.05])),
          NOTE, ax_b=im)
    note(440, 838, "Small changes in fixation drive\nlarge changes in population response")
    arrow((440 + text_width("large changes in population response", 25) + 10, 884),
          (rx0 + 210, py1 + 2), NOTE)

    # ---- Provenance -------------------------------------------------------------------
    bg.text(20, 930, f"One session, 132 neurons, {n_per} trials per gaze group. Dots: median gaze in the "
            "shaded window. RF: population footprint from noise mapping,", fontsize=17,
            color=MUTED, va="top")
    bg.text(20, 952, "located from these trials. Dataset: 2 marmosets · 1,022 foveal V1 units · "
            "19 sessions · 20 Hz natural images.", fontsize=17, color=MUTED, va="top")
    save(fig, "panel_fem")


def intro_trace():
    """Gaze paths on every repeat of the same sequence, against a typical RF."""
    W = H = 330
    fig, bg = canvas(W, H)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    k /= k.sum()

    def smooth(xy):
        # Light Gaussian smoothing (sigma 2 samples = 8 ms) suppresses tracker noise.
        pad = np.pad(xy, ((6, 6), (0, 0)), mode="edge")
        return np.column_stack([np.convolve(pad[:, i], k, "valid") for i in (0, 1)])

    traces = [t[~np.isnan(t).any(1)] * 60 for t in d["intro_all_deg"]]      # arcmin
    traces = [smooth(t) for t in traces if len(t) > 20]
    centre = np.median(np.vstack(traces), axis=0)
    rf_r = np.median(d["rf_diam_deg"]) * 60 / 2
    lim = rf_r * 1.15
    a = place(fig, W, H, 15, 15, 315, 315)
    a.add_patch(Circle((0, 0), rf_r, fc="#eef1f5", ec=NAVY, lw=2.5, ls=(0, (6, 4))))
    medians = np.array([np.median(t, axis=0) for t in traces]) - centre
    a.scatter(*medians.T, s=40, color=COPPER, alpha=0.55, lw=0, zorder=2)
    hi = smooth(d["intro_trace_deg"] * 60) - centre
    a.plot(*hi.T, color=NAVY, lw=2.4, zorder=3)
    a.plot(*hi[0], "o", color=NAVY, ms=7, zorder=3)
    a.set_xlim(-lim, lim), a.set_ylim(-lim, lim), a.set_aspect("equal"), a.axis("off")
    a.plot([-lim * 0.9, -lim * 0.9 + 6], [-lim * 0.92] * 2, color="black", lw=3,
           solid_capstyle="butt")
    a.text(-lim * 0.9 + 3, -lim * 0.86, "6′", ha="center", va="bottom", fontsize=19)
    save(fig, "intro_trace")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fem_panel()
    intro_trace()
    print("Rendered panel_fem.pdf and intro_trace.pdf")
