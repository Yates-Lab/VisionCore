"""Render the poster's "gaze shifts within an RF" panel and intro schematic.

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
FRAME, MIRROR, WINDOW = "#5595aa", "#f2a7a3", "#e9edf2"
PANEL_W, PANEL_H = 1183.79296875, 974.5703125
# Align the body artwork's x=22 edge to panel 2's 90 pt plotting inset.
CONTENT_DX = 68.0
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
MARMOSET_BOX = (210.0, 205.0, 155.0)


def marmoset_size():
    import fitz
    with fitz.open(MARMOSET_SVG) as doc:
        r = doc[0].rect
    return r.width, r.height


SCREEN_PX = 6.0          # texture resolution, pixels per panel point


def quad_point(q, u, v):
    """Bilinear point in quadrilateral q (TL, TR, BR, BL) at fractions u (x), v (y)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = q
    top = np.array([x0 + (x1 - x0) * u, y0 + (y1 - y0) * u])
    bot = np.array([x3 + (x2 - x3) * u, y3 + (y2 - y3) * u])
    return top + (bot - top) * v


def screen_texture(image, quad, fill_fraction=0.78):
    """Warp the stimulus, centred on its own background gray, onto a screen
    quadrilateral with a perspective transform. Returns (x0, y0, rgba)."""
    gray = int(np.median(np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])))
    qx, qy = np.array(quad)[:, 0], np.array(quad)[:, 1]
    x0, y0 = qx.min(), qy.min()
    w = int(np.ceil((qx.max() - x0) * SCREEN_PX))
    h = int(np.ceil((qy.max() - y0) * SCREEN_PX))
    tw, th = 600, 760                          # texture with the screen's aspect
    tex = Image.new("L", (tw, th), gray)
    side = int(tw * fill_fraction)
    tex.paste(Image.fromarray(image.astype(np.uint8)).resize((side, side), Image.BICUBIC),
              ((tw - side) // 2, (th - side) // 2))
    dst = [((x - x0) * SCREEN_PX, (y - y0) * SCREEN_PX) for x, y in quad]
    src = [(0, 0), (tw, 0), (tw, th), (0, th)]
    # Coefficients mapping output (dst) pixels back to texture (src) pixels.
    A, b = [], []
    for (X, Y), (u, v) in zip(dst, src):
        A += [[X, Y, 1, 0, 0, 0, -u * X, -u * Y], [0, 0, 0, X, Y, 1, -v * X, -v * Y]]
        b += [u, v]
    coef = np.linalg.solve(np.array(A, float), np.array(b, float))
    warped = tex.transform((w, h), Image.PERSPECTIVE, tuple(coef), Image.BICUBIC,
                           fillcolor=gray)
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).polygon(dst, fill=255)
    rgba = np.dstack([np.repeat(np.asarray(warped)[..., None], 3, 2), np.asarray(mask)])
    return x0, y0, rgba


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
    ax = lambda x0, y0, x1, y1: place(fig, W, H, x0 + CONTENT_DX, y0,
                                     x1 + CONTENT_DX, y1)  # noqa: E731

    # Header, matching the other panels' badge and title geometry.
    bg.add_patch(Circle((29.14, 36.34), 22.13, fc=COPPER, ec="white", lw=2.5))
    bg.text(20.81, 45.82, "1", ha="left", fontsize=29, weight="bold", color="white")
    bg.text(59.75, 48.07, "Tiny gaze shifts reshape population responses",
            fontsize=48, weight="bold", color=NAVY)

    # Body layer translates independently of the fixed title and badge.
    bg = fig.add_axes([CONTENT_DX / W, 0, 1, 1])
    bg.set_xlim(0, W), bg.set_ylim(H, 0), bg.axis("off")

    image, half = d["image"], float(d["image_half_deg"])
    gaze_hi, gaze_lo, rf = d["gaze_hi"], d["gaze_lo"], d["rf"]
    shift, diam = float(d["gaze_shift_deg"]), d["rf_diam_deg"]
    n_per, n_units = int(d["n_per"]), int(d["n_units"])
    t_lo, t_hi = d["raster_ms"]

    def note(x, y, text, size=22, **kw):
        bg.text(x, y, text, fontsize=size,
                color=TEXT, va="top", linespacing=1.15, **kw)

    # Groups top to bottom by mean gaze height, so the stack mirrors the eye.
    # spk_k rows: red trials 0..n-1, then blue n..2n-1 (export order).
    groups = [(RED, 0, gaze_hi, gaze_lo, d["trace_hi"]),
              (BLUE, n_per, gaze_lo, gaze_hi, d["trace_lo"])]
    if gaze_lo.mean(0)[1] > gaze_hi.mean(0)[1]:
        groups.reverse()

    # ---- Compact apparatus: brain above the animal, eye inset beside it ------
    note(22, 86, "Laminar\nrecordings in\nfoveal V1")
    # The stimulus is drawn on the screen surface in perspective: a texture of
    # the image centred on its own background gray is warped onto the screen
    # quadrilateral, so image and screen blend without a visible square.
    screen_quad = [(22, 172), (140, 178), (140, 313), (22, 342)]
    sx0, sy0, warped = screen_texture(image, screen_quad)
    bg.imshow(warped, extent=[sx0, sx0 + warped.shape[1] / SCREEN_PX,
                              sy0 + warped.shape[0] / SCREEN_PX, sy0],
              interpolation="bilinear", zorder=0.5)
    screen_c = quad_point(screen_quad, 0.5, 0.5)
    # Eye position and the hot mirror use separate, explicit visual keys.
    bg.plot([screen_c[0], 239], [screen_c[1], 241], color=RED, lw=2.5, ls=(0, (3, 2)),
            zorder=6)
    # Camera post stops at the mirror's lower edge, beneath its outline.
    bg.plot([165, 165], [278, 331], color="#b7b7b7", lw=8, zorder=1,
            solid_capstyle="butt")
    bg.add_patch(Polygon([(155, 219), (222, 225), (203, 278), (136, 272)],
                         fill=False, ec=FRAME, lw=3.5, zorder=2))
    bg.add_patch(Rectangle((150, 331), 36, 22, fc="#414141", ec="black", lw=1))
    bg.add_patch(Rectangle((160, 316), 16, 15, fc="#414141", ec="black", lw=1))
    bg.text(144, 377, "Eye camera", fontsize=20, color=TEXT)
    for yy, label, color, style in ((395, "Eye position", RED, (0, (3, 2))),
                                     (421, "Hot mirror", FRAME, "-")):
        bg.plot([22, 40], [yy, yy], color=color, lw=2.5, ls=style)
        bg.text(48, yy, label, fontsize=20, color=TEXT, va="center")
    mx, my, mw = MARMOSET_BOX
    mh = mw * marmoset_size()[1] / marmoset_size()[0]

    # The eye image is a separate PDF layer placed after the vector marmoset,
    # so it can tuck over the lower-right silhouette without crossed strokes.
    eye_fig, _ = canvas(110, 110)
    dpi = place(eye_fig, 110, 110, 0, 0, 110, 110)
    dpi.imshow(np.asarray(Image.open(HERE / "assets" / "dpi_eye.png")), cmap="gray",
               vmin=0, vmax=255)
    eye_accent = "#d4c487"  # One muted highlight on the grayscale eye image.
    dpi.add_patch(Circle((270, 262), 222, fill=False, ec=eye_accent, lw=2))
    dpi.add_patch(Circle((382, 238), 30, fill=False, ec=eye_accent, lw=1.5))
    for (tx, ty), (px, py), label in (((150, 170), (204, 278), "P4"),
                                      ((150, 405), (262, 330), "Pupil"),
                                      ((345, 170), (380, 212), "CR")):
        dpi.annotate(label, (px, py), (tx, ty), color="white", fontsize=15,
                     arrowprops=dict(arrowstyle="-|>", color=eye_accent, lw=1.4,
                                     mutation_scale=9, shrinkB=2))
    dpi.set_xticks([]), dpi.set_yticks([])
    for spine in dpi.spines.values():
        spine.set_linewidth(1.2)
    save(eye_fig, "panel_fem_eye")
    note(22, 438, "Dual Purkinje high-resolution\neye tracker", size=20)
    bg.text(22 + text_width("eye tracker", 20) + 4, 471, "[2–4]", fontsize=14,
            color=TEXT, va="bottom")

    # Enlarged stimulus views align with their corresponding raster groups.
    vx, vs = 400, 248
    y_top, y_bot, ggap, gap = 186, 744, 34, 12
    group_h = (y_bot - y_top - ggap) / 2
    # Foveal RFs from every mapping session (Figure 1C outlines), with this
    # session's population RF and the example gaze shift drawn at the same scale.
    allrf = np.load(HERE / "rf_all_sessions.npz")
    bg.text(22, 534, "Foveal RFs,\nall mapping sessions", fontsize=22, weight="bold", color=NAVY,
            va="bottom")
    fov = ax(22, 542, 328, 848)
    th = np.linspace(0, 2 * np.pi, 200)
    fov.plot(np.cos(th), np.sin(th), color="black", lw=1.6, ls=(0, (5, 3)))
    fov.axhline(0, color="#bbbbbb", lw=0.8, ls=":"), fov.axvline(0, color="#bbbbbb", lw=0.8, ls=":")
    for subj, col in (("Allen", "#59606a"), ("Logan", "#2e8b57")):
        pts, lens = allrf[f"{subj}_pts"], allrf[f"{subj}_len"]
        for seg in np.split(pts, np.cumsum(lens)[:-1]):
            fov.plot(*np.vstack([seg, seg[:1]]).T, color=col, lw=0.5, alpha=0.10)
    fov.plot(*np.vstack([rf, rf[:1]]).T, color="black", lw=2.6,
             path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
    fov.plot([0, 1], [0, 0], color="black", lw=3, solid_capstyle="butt")
    fov.text(0.5, 0.05, "1°", ha="center", va="bottom", fontsize=20)
    # Gaze shift between the example groups, at the same scale.
    fov.plot([-0.75, -0.75 + shift], [0.62, 0.62], color=NAVY, lw=5, solid_capstyle="butt")
    fov.text(-0.75 + shift + 0.05, 0.62, f"{shift:.2f}° gaze shift,\nsmaller than RF size",
             va="center", fontsize=17, color=NAVY, linespacing=1.05)
    fov.set_xlim(-1.2, 1.2), fov.set_ylim(-1.2, 1.2), fov.set_aspect("equal")
    fov.set_xticks([]), fov.set_yticks([])
    for sp in fov.spines.values():
        sp.set_linewidth(1.2)
    # Two legend rows fit within the RF plot width; final baseline matches
    # the takeaway and response annotations above the dataset strip.
    for i, (lab, col, lw, al) in enumerate((("Monkey A", "#59606a", 6, 0.6),
                                            ("Monkey L", "#2e8b57", 6, 0.6),
                                            ("this session", "black", 2.6, 1))):
        x, baseline = ((22, 866), (180, 866), (22, 888))[i]
        bg.plot([x, x + 18], [baseline - 6] * 2, color=col, lw=lw, alpha=al, solid_capstyle="butt")
        bg.text(x + 24, baseline, lab, fontsize=20, color=TEXT)

    # ---- Gaze traces over the window ---------------------------------------------
    # Inset the complete right-hand group to clear the panel border, retaining
    # the 332 pt plot width and the existing gaps between plots and scale bars.
    rx0, rx1 = 708, 1040
    tt = d["trace_t"]
    # One angular scale for both strips, so the single 0.1 deg bar holds for both;
    # each strip is centred on its own data.
    spans = [np.nanmax(v) - np.nanmin(v) for v in
             (np.concatenate([d["trace_hi"][..., c], d["trace_lo"][..., c]]) for c in (0, 1))]
    half_range = max(spans) / 2 + 0.02
    for k, (comp, name) in enumerate(((0, "horizontal"), (1, "vertical"))):
        y0 = 78 + k * 50
        a = ax(rx0, y0, rx1, y0 + 42)
        vals = np.concatenate([d["trace_hi"][..., comp], d["trace_lo"][..., comp]])
        mid = (np.nanmax(vals) + np.nanmin(vals)) / 2
        lo, hi = mid - half_range, mid + half_range
        for col, _, _, _, traces in groups:
            for row in traces:
                a.plot(tt, row[:, comp], color=col, lw=1.6, alpha=0.85)
        a.set_ylim(lo, hi), a.set_xlim(t_lo, t_hi), a.axis("off")
        bg.text(rx0 - 10, y0 + 21, f"{name} gaze", ha="right", va="center", fontsize=21,
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
    for gi, (col, offset, gaze, other, _) in enumerate(groups):
        gy = y_top + gi * (group_h + ggap)
        vy = gy + (group_h - vs) / 2
        v = ax(vx, vy, vx + vs, vy + vs)
        v.imshow(image, cmap="gray", extent=[-half, half, -half, half], origin="upper",
                 vmin=0, vmax=255, interpolation="bilinear")
        # Where the RF sat on the other group's trials: dashed, in that group's
        # colour, thinner and semi-transparent so it stays secondary.
        ghost = rf + other.mean(0)
        v.plot(*np.vstack([ghost, ghost[:1]]).T, color=groups[1 - gi][0], lw=2.6,
               ls=(0, (4, 2.5)), alpha=0.85, zorder=3,
               path_effects=[pe.Stroke(linewidth=5, foreground="white", alpha=0.7),
                             pe.Normal()])
        here = rf + gaze.mean(0)
        v.plot(*np.vstack([here, here[:1]]).T, color=col, lw=5, path_effects=halo, zorder=4)
        # The other group's gaze, faint, alongside its dashed RF outline.
        other_col = groups[1 - gi][0]
        v.scatter(*other.T, s=110, color=other_col, alpha=0.5, linewidth=0, zorder=4.5)
        v.scatter(*gaze.T, s=150, color=col, edgecolor="white", linewidth=2.5, zorder=5)
        v.set_xlim(cx - view, cx + view), v.set_ylim(cy - view, cy + view)
        v.set_xticks([]), v.set_yticks([])
        for sp in v.spines.values():
            sp.set_color(col), sp.set_linewidth(4)
        if gi == 0:
            # Direct labels on the first view only; the second reads by analogy.
            lab_fx = [pe.withStroke(linewidth=4, foreground="#1d1d1d")]
            lab = dict(fontsize=18, color="white", weight="bold", zorder=7,
                       path_effects=lab_fx)
            g_mid = gaze.mean(0)
            v.text(g_mid[0], gaze[:, 1].max() + 0.035, f"gaze,\n{n_per} trials",
                   ha="center", va="bottom", linespacing=1.0, **lab)
            top = here[np.argmax(here[:, 1])]
            v.text(top[0], top[1] + 0.02, "RF", ha="center", va="bottom", **lab)
            low = ghost[np.argmin(ghost[:, 1])]
            other_trials = f"{(1 - gi) * n_per + 1}–{(2 - gi) * n_per}"
            v.text(low[0], low[1] - 0.02, f"RF on trials {other_trials}", ha="center",
                   va="top", **lab)
        bg.text(vx, vy - 6, f"Trials {gi * n_per + 1}–{(gi + 1) * n_per}", fontsize=21,
                weight="bold", color=col, va="bottom")
        y = gy
        for j in range(n_per):
            # Keep the existing source-trial order, but number the displayed
            # rows continuously from top to bottom across both groups.
            number = n_per - j
            display_number = gi * n_per + j + 1
            r = ax(rx0, y, rx1, y + bh)
            sel = d["spk_k"] == offset + number - 1
            # 3 pt-tall vector ticks with 1.35 pt strokes retain their centers
            # while giving the population pattern more contrast at a distance.
            r.scatter(d["spk_t"][sel], d["spk_u"][sel], s=9, marker="|", color="black",
                      linewidths=1.35)
            r.set_xlim(t_lo, t_hi), r.set_ylim(n_units, -1), r.axis("off")
            bg.text(rx0 - 10, y + bh / 2, str(display_number), ha="right", va="center",
                    fontsize=21, color=col)
            y += bh + gap
        bg.add_patch(Rectangle((rx1 + 6, gy), 7, group_h, color=col, lw=0))
    # Scale: 0.5 deg bar under the lower view.
    bar = 0.5 * vs / (2 * view)
    by = y_bot + 14
    bg.plot([vx, vx + bar], [by] * 2, color="black", lw=4, solid_capstyle="butt")
    bg.text(vx + bar + 8, by, "0.5°", va="center", fontsize=19)
    # Rasters are separate presentations of one image; rows are the same
    # neurons in the same order on every trial.
    bg.text(rx0 - 38, (y_top + y_bot) / 2, "Repeated presentations of the same image",
            rotation=90, ha="center", va="center", fontsize=17, color=MUTED)
    bg.text(rx1 + 32, y_top + group_h / 2, "same neurons, same order", rotation=90,
            ha="center", va="center", fontsize=16, color=MUTED)
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
    bg.text(rx0, 888, "50 ms", va="baseline", fontsize=19)
    bg.text(rx1, 888, f"{t_lo:.0f}–{t_hi:.0f} ms after fixation onset", va="baseline",
            ha="right", fontsize=17, color=MUTED)

    # ---- Message, then the dataset ------------------------------------------------------
    bg.text(400, 795.77, "Small changes in\nfixation drive large\nchanges in population\nresponse", fontsize=24, weight="semibold", color=NAVY, va="top",
            linespacing=1.15)
    fig.add_artist(ConnectionPatch((640, 831.77), (rx0 - 6, (py0 + py1) / 2), coordsA="data",
                                   coordsB="data", axesA=bg, axesB=bg, color="#25327a", lw=2.4,
                                   arrowstyle="-|>", mutation_scale=20, shrinkA=2, shrinkB=2))
    bg.add_patch(FancyBboxPatch((20, 902), 1144 - CONTENT_DX, 64, boxstyle="round,pad=0,rounding_size=8",
                                fc="#f0f3f6", ec="none"))
    for i, (big, small) in enumerate((("2", "awake marmosets"), ("1,022", "foveal V1 units"),
                                      ("19", "recording sessions"),
                                      ("20 Hz", "repeated natural images"))):
        bg.text(40 + i * 286, 934, big, fontsize=28, weight="bold", color=NAVY, va="bottom")
        bg.text(40 + i * 286, 938, small, fontsize=17, color=MUTED, va="top")
    save(fig, "panel_fem_plots")   # composed with the vector marmoset by panel_fem.typ
    (HERE / "marmoset_box.json").write_text(json.dumps(
        {"x": mx + CONTENT_DX, "y": my, "width": mw,
         "brain_x": 170 + CONTENT_DX, "brain_y": 80, "brain_width": 200,
         "eye_x": 257 + CONTENT_DX, "eye_y": 318, "eye_size": 110,
         "panel_width": W, "panel_height": H}) + "\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fem_panel()
    print("Rendered panel_fem_plots.pdf; Typst composes panel_fem.pdf")
