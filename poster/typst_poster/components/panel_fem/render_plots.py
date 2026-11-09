"""Render the poster's "gaze shifts within an RF" panel and the intro eye trace.

Reads the frozen ``plot_data.npz`` and ``rf_all_sessions.npz``, the DPI image,
and vector marmoset artwork. Typst adds the vector marmoset and brain drawings.
All geometry is in poster points (1/72 in) from the panel's top-left corner.
"""
from pathlib import Path
import json
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
from matplotlib.patches import (Circle, ConnectionPatch, FancyBboxPatch, Polygon,  # noqa: E402
                                Rectangle)
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


MARMOSET_SVG = HERE / "assets" / "marmoset.svg"
# Marmoset placement in panel points (x0, y0, width). The vector drawing is laid
# over the Matplotlib layer by panel_fem.typ; its eye sits where the rig's
# camera lines end.
MARMOSET_BOX = (216.8, 264.8, 150.0)


def marmoset_size():
    import fitz
    with fitz.open(MARMOSET_SVG) as doc:
        r = doc[0].rect
    return r.width, r.height


def marmoset_interior(scale=8, close=6):
    """Opaque white silhouette of the marmoset, rasterised from the vector
    drawing. It sits under the vector strokes, so any rig lines behind the
    animal are hidden instead of crossing it. Gaps between the
    hair strokes are closed by dilating the strokes before flood-filling the
    exterior, then eroding back."""
    import fitz
    from PIL import ImageDraw, ImageFilter
    with fitz.open(MARMOSET_SVG) as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    ink = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    strokes = ink.point(lambda v: 255 if v < 128 else 0)
    w, h = strokes.size
    ImageDraw.Draw(strokes).line([(0, h - 1), (w - 1, h - 1)], fill=255, width=3)
    grown = strokes
    for _ in range(close):
        grown = grown.filter(ImageFilter.MaxFilter(9))
    filled = grown.copy()
    ImageDraw.floodfill(filled, (0, 0), 128)
    inside = filled.point(lambda v: 255 if v != 128 else 0)
    for _ in range(close):
        inside = inside.filter(ImageFilter.MinFilter(9))
    alpha = np.asarray(inside)
    return np.dstack([np.full(alpha.shape + (3,), 255, np.uint8), alpha])


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
    n_per, n_units = int(d["n_per"]), int(d["n_units"])
    t_lo, t_hi = d["raster_ms"]

    def note(x, y, text, size=22, **kw):
        # Arial Italic is not in the locked font set; Liberation Sans Italic is
        # metrically identical and bundled under its open licence.
        bg.text(x, y, text, family="Liberation Sans", style="italic", fontsize=size,
                color=TEXT, va="top", linespacing=1.15, **kw)

    # Groups top to bottom by mean gaze height, so the stack mirrors the eye.
    # spk_k rows: red trials 0..n-1, then blue n..2n-1 (export order).
    groups = [(RED, 0, gaze_hi, gaze_lo, d["trace_hi"]),
              (BLUE, n_per, gaze_lo, gaze_hi, d["trace_lo"])]
    if gaze_lo.mean(0)[1] > gaze_hi.mean(0)[1]:
        groups.reverse()

    # ---- Left column: eye tracker, rig, RFs from all sessions -----------------
    note(22, 74, "Laminar\nrecordings in\nfoveal V1")
    dpi = ax(22, 448, 132, 558)
    dpi.imshow(np.asarray(Image.open(HERE / "assets" / "dpi_eye.png")), cmap="gray",
               vmin=0, vmax=255)
    dpi.add_patch(Circle((270, 262), 222, fill=False, ec="#c9e8a0", lw=2))
    dpi.add_patch(Circle((382, 238), 30, fill=False, ec="#f2e27a", lw=1.5))
    for (tx, ty), (px, py), label in (((150, 170), (204, 278), "P4"),
                                      ((50, 430), (262, 330), "Pupil"),
                                      ((345, 430), (382, 268), "CR")):
        dpi.annotate(label, (px, py), (tx, ty), color="#c9e8a0", fontsize=15,
                     arrowprops=dict(arrowstyle="-|>", color="#d9483b", lw=1.4,
                                     mutation_scale=9, shrinkB=2))
    dpi.axis("off")
    note(148, 466, "Dual Purkinje\nhigh-resolution\neye tracker", size=20)
    bg.text(148 + text_width("eye tracker", 20) + 4, 520, "[2–4]", fontsize=14, color=TEXT,
            va="bottom")

    rig = ax(16, 214, 376, 451)
    rig.set_xlim(20, 400), rig.set_ylim(345, 95), rig.axis("off")
    rig.add_patch(Polygon([(28, 168), (170, 132), (170, 262), (28, 296)], fc="#8c8c8c", ec="none"))
    rig.imshow(image, cmap="gray", extent=[96, 140, 236, 192], zorder=2)
    patch = (108, 204, 128, 224)
    rig.add_patch(Rectangle(patch[:2], 20, 20, fill=False, ec=FRAME, lw=2.5, zorder=3))
    for corner in ((patch[2], patch[1]), (patch[2], patch[3])):
        rig.plot([corner[0], 262], [corner[1], 186], ls=(0, (4, 3)), color=FRAME, lw=2)
    rig.plot([214, 238], [232, 194], color=MIRROR, lw=5, solid_capstyle="butt")
    rig.text(150, 112, "hot mirror", fontsize=19, color=TEXT, ha="left", va="bottom")
    rig.plot([196, 222], [116, 196], color=MIRROR, lw=1.4)
    rig.add_patch(Rectangle((236, 110), 32, 26, color="#1a1a1a"))
    rig.add_patch(Polygon([(242, 136), (262, 136), (257, 154), (247, 154)], color="#1a1a1a"))
    rig.text(280, 112, "Eye camera", fontsize=19, color=TEXT, ha="left", va="bottom")
    mx, my, mw = MARMOSET_BOX
    mh = mw * marmoset_size()[1] / marmoset_size()[0]
    bg.imshow(marmoset_interior(), extent=[mx, mx + mw, my + mh, my], zorder=5,
              interpolation="bilinear", aspect="auto")

    # Enlarged stimulus views align with their corresponding raster groups.
    vx, vs = 400, 248
    y_top, y_bot, ggap, gap = 186, 744, 22, 4
    group_h = (y_bot - y_top - ggap) / 2
    # Foveal RFs from every mapping session (Figure 1C outlines), with this
    # session's population RF and the example gaze shift drawn at the same scale.
    allrf = np.load(HERE / "rf_all_sessions.npz")
    bg.text(22, 580, "Foveal RFs, all sessions", fontsize=22, weight="bold", color=NAVY,
            va="bottom")
    fov = ax(22, 586, 300, 864)
    th = np.linspace(0, 2 * np.pi, 200)
    fov.plot(np.cos(th), np.sin(th), color="black", lw=1.6, ls=(0, (5, 3)))
    fov.axhline(0, color="#bbbbbb", lw=0.8, ls=":"), fov.axvline(0, color="#bbbbbb", lw=0.8, ls=":")
    for subj, col in (("Allen", "#59606a"), ("Logan", "#2e8b57")):
        pts, lens = allrf[f"{subj}_pts"], allrf[f"{subj}_len"]
        for seg in np.split(pts, np.cumsum(lens)[:-1]):
            fov.plot(*np.vstack([seg, seg[:1]]).T, color=col, lw=0.5, alpha=0.06)
    fov.plot(*np.vstack([rf, rf[:1]]).T, color="black", lw=2.6,
             path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
    fov.plot([0, 1], [0, 0], color="black", lw=3, solid_capstyle="butt")
    fov.text(0.5, 0.05, "1°", ha="center", va="bottom", fontsize=20)
    # Gaze shift between the example groups, at the same scale.
    fov.plot([0, shift], [0.5, 0.5], color=NAVY, lw=5, solid_capstyle="butt")
    fov.text(shift + 0.05, 0.5, f"{shift:.2f}°\ngaze shift", va="center", fontsize=18,
             color=NAVY, linespacing=1.05)
    fov.set_xlim(-1.2, 1.2), fov.set_ylim(-1.2, 1.2), fov.set_aspect("equal")
    fov.set_xticks([]), fov.set_yticks([])
    for sp in fov.spines.values():
        sp.set_linewidth(1.2)
    lx, ly = 22, 866
    for i, (lab, col, lw, al) in enumerate((("Monkey A", "#59606a", 6, 0.6),
                                            ("Monkey L", "#2e8b57", 6, 0.6),
                                            ("this session", "black", 2.6, 1))):
        x = lx + i * 116
        bg.plot([x, x + 18], [ly + 10] * 2, color=col, lw=lw, alpha=al, solid_capstyle="butt")
        bg.text(x + 24, ly + 10, lab, fontsize=17, color=TEXT, va="center")

    # ---- Gaze traces over the window ---------------------------------------------
    rx0, rx1 = 728, 1080
    tt = d["trace_t"]
    for k, (comp, name) in enumerate(((0, "horizontal"), (1, "vertical"))):
        y0 = 72 + k * 50
        a = ax(rx0, y0, rx1, y0 + 42)
        vals = np.concatenate([d["trace_hi"][..., comp], d["trace_lo"][..., comp]])
        lo, hi = np.nanmin(vals) - 0.02, np.nanmax(vals) + 0.02
        for col, _, _, _, traces in groups:
            for row in traces:
                a.plot(tt, row[:, comp], color=col, lw=1.6, alpha=0.85)
        a.set_ylim(lo, hi), a.set_xlim(t_lo, t_hi), a.axis("off")
        bg.text(rx0 - 10, y0 + 21, f"{name} gaze", ha="right", va="center", fontsize=18,
                color=MUTED)
        if k == 0:
            bar = 0.1 / (hi - lo) * 42
            bg.plot([rx1 + 22] * 2, [y0 + 42, y0 + 42 - bar], color="black", lw=2)
            bg.text(rx1 + 30, y0 + 42, "0.1°", rotation=90, va="bottom", fontsize=18)

    # ---- One enlarged view per group, beside that group's rasters -------------------
    pts = np.vstack([gaze_hi, gaze_lo, rf + gaze_hi.mean(0), rf + gaze_lo.mean(0)])
    cx, cy = (pts.min(0) + pts.max(0)) / 2
    view = np.ptp(pts, axis=0).max() / 2 + 0.08
    bh = (group_h - (n_per - 1) * gap) / n_per
    halo = [pe.Stroke(linewidth=8.5, foreground="white"), pe.Normal()]
    ghost_fx = [pe.Stroke(linewidth=4.2, foreground="#444444"), pe.Normal()]
    for gi, (col, offset, gaze, other, _) in enumerate(groups):
        gy = y_top + gi * (group_h + ggap)
        vy = gy + (group_h - vs) / 2
        v = ax(vx, vy, vx + vs, vy + vs)
        v.imshow(image, cmap="gray", extent=[-half, half, -half, half], origin="upper",
                 vmin=0, vmax=255, interpolation="bilinear")
        ghost = rf + other.mean(0)
        v.plot(*np.vstack([ghost, ghost[:1]]).T, color="white", lw=2, ls=(0, (3, 2)),
               path_effects=ghost_fx, zorder=3)
        here = rf + gaze.mean(0)
        v.plot(*np.vstack([here, here[:1]]).T, color=col, lw=5, path_effects=halo, zorder=4)
        v.scatter(*gaze.T, s=150, color=col, edgecolor="white", linewidth=2.5, zorder=5)
        v.set_xlim(cx - view, cx + view), v.set_ylim(cy - view, cy + view)
        v.set_xticks([]), v.set_yticks([])
        for sp in v.spines.values():
            sp.set_color(col), sp.set_linewidth(4)
        bg.text(vx, vy - 6, "higher gaze" if gi == 0 else "lower gaze", fontsize=19,
                weight="bold", color=col, va="bottom")
        y = gy
        for j in range(n_per):
            number = n_per - j
            r = ax(rx0, y, rx1, y + bh)
            sel = d["spk_k"] == offset + number - 1
            r.scatter(d["spk_t"][sel], d["spk_u"][sel], s=7, marker="|", color="black",
                      linewidths=0.9)
            r.set_xlim(t_lo, t_hi), r.set_ylim(n_units, -1), r.axis("off")
            bg.text(rx0 - 10, y + bh / 2, f"Trial {number}", ha="right", va="center",
                    fontsize=19, color=col)
            y += bh + gap
        bg.add_patch(Rectangle((rx1 + 6, gy), 7, group_h, color=col, lw=0))
    # Scale: 0.5 deg bar under the lower view.
    bar = 0.5 * vs / (2 * view)
    by = y_bot + 14
    bg.plot([vx, vx + bar], [by] * 2, color="black", lw=4, solid_capstyle="butt")
    bg.text(vx + bar + 8, by, "0.5°", va="center", fontsize=19)
    n_pt = bh / (n_units + 1) * 100
    bg.plot([rx1 + 22] * 2, [y_bot, y_bot - n_pt], color="black", lw=2)
    bg.text(rx1 + 30, y_bot, "100 neurons", rotation=90, va="bottom", fontsize=18)

    # ---- Group PSTHs ---------------------------------------------------------------
    py0, py1 = 764, 846
    ps = ax(rx0, py0, rx1, py1)
    ps.plot(d["psth_t"], d["psth_hi"], color=RED, lw=3)
    ps.plot(d["psth_t"], d["psth_lo"], color=BLUE, lw=3)
    lo = min(d["psth_hi"].min(), d["psth_lo"].min())
    hi = max(d["psth_hi"].max(), d["psth_lo"].max())
    ps.set_xlim(t_lo, t_hi), ps.set_ylim(lo, hi), ps.axis("off")
    bg.plot([rx1 + 22] * 2, [py1, py1 - 20 / (hi - lo) * (py1 - py0)], color="black", lw=2)
    bg.text(rx1 + 30, py1, "20 sp/s", rotation=90, va="bottom", fontsize=18)
    ms_pt = (rx1 - rx0) / (t_hi - t_lo)
    bg.plot([rx0, rx0 + 50 * ms_pt], [py1 + 12] * 2, color="black", lw=4, solid_capstyle="butt")
    bg.text(rx0, py1 + 20, "50 ms", va="top", fontsize=19)
    bg.text(rx1, py1 + 20, f"{t_lo:.0f}–{t_hi:.0f} ms after fixation onset", va="top",
            ha="right", fontsize=17, color=MUTED)

    # ---- Message, then the dataset ------------------------------------------------------
    note(400, 786, "Small changes in\nfixation drive large\nchanges in population\nresponse")
    fig.add_artist(ConnectionPatch((640, 816), (rx0 - 6, (py0 + py1) / 2), coordsA="data",
                                   coordsB="data", axesA=bg, axesB=bg, color="#25327a", lw=2.4,
                                   arrowstyle="-|>", mutation_scale=20, shrinkA=2, shrinkB=2))
    bg.add_patch(FancyBboxPatch((20, 902), 1144, 64, boxstyle="round,pad=0,rounding_size=8",
                                fc="#f0f3f6", ec="none"))
    for i, (big, small) in enumerate((("2", "awake marmosets"), ("1,022", "foveal V1 units"),
                                      ("19", "recording sessions"),
                                      ("20 Hz", "repeated natural images"))):
        bg.text(40 + i * 286, 934, big, fontsize=28, weight="bold", color=NAVY, va="bottom")
        bg.text(40 + i * 286, 938, small, fontsize=17, color=MUTED, va="top")
    save(fig, "panel_fem_plots")   # composed with the vector marmoset by panel_fem.typ
    (HERE / "marmoset_box.json").write_text(json.dumps(
        {"x": mx, "y": my, "width": mw, "brain_x": 176, "brain_y": 62, "brain_width": 210,
         "panel_width": W, "panel_height": H}) + "\n")


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
    print("Rendered panel_fem_plots.pdf and intro_trace.pdf; Typst composes panel_fem.pdf")
