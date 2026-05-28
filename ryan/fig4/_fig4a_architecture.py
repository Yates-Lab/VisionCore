"""Panel A — right half: digital-twin architecture (detailed).

Every output channel of every learned stage is rendered as a 3D kernel
prism in shared cabinet projection (same visual language as the stimulus
half's lag cube): time → +x (right), H → +y (up), W → +z (INTO page).

Stages, left → right — front-face centers are all anchored at the shared
ARCH_CENTER_Y so inter-stage flow arrows are perfectly horizontal:

    Frontend   — 4 depthwise temporal kernels (kt=16) stacked vertically;
                 each kernel's learned temporal weight curve is drawn on
                 its own front face.
    Stem       — 8 channels, 1×7×7 kernels, 2×4 grid.
    ResBlock 1 — 64 channels, 3×9×9 kernels, 8×8 grid. ⊔-staple residual
                 beneath; ↓2× downsample badge on the outgoing arrow.
    ResBlock 2 — 128 channels, 3×5×5 kernels, 16×8 grid (taller than wide).
                 ⊔-staple residual.
    ConvGRU    — 128 hidden channels, 3×3 kernels, 16×8 grid; tall closed
                 recurrent loop on top. A small Behavior placeholder sits
                 above the incoming arrow, with a concatenation arrow to
                 the ConvGRU input.
    Readout    — per example unit: a horizontal feature-weight block (128
                 depthwise weights as bands) + a conv-kernel-style spatial
                 prism whose front face is a localized Gaussian sampler.
                 Laid out as two units, a ⋮, then one (the rest are implied).

Output channels are tiled in (y, z); within each tile-grid we draw
back-to-front in z so front kernels correctly occlude rear kernels.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from _fig4a_glyphs import (
    TEXT_COLOR,
    draw_channel_grid,
    draw_recurrent_loop, draw_feature_weight_block, draw_spatial_readout_prism,
    draw_pool_glyph, draw_arrow_skip, draw_op_marker,
)


# ──────────────────────────────────────────────────────────────────────────
# World-unit scales — every convnet/GRU kernel uses the SAME mapping
# (taps → world units) so a 9×9 reads visibly larger than a 5×5.
# ──────────────────────────────────────────────────────────────────────────
S_PIX = 0.030           # world units per spatial tap (convnet & GRU)
S_T   = S_PIX           # world units per time tap (isotropic: 1×1×1 is a cube)
# Frontend kernels are depthwise temporal (kt=frontend_k taps, 1×1 spatial),
# rendered at their true (frontend_k : 1 : 1) aspect so each beam reads as
# frontend_k× longer than it is wide/tall. FE_SCALE is the single knob: the
# frontend's per-tap size is FE_SCALE × the per-tap size (S_PIX) every other
# kernel uses. Lower it to shrink the whole frontend — length and
# cross-section scale together, so the aspect is preserved.
FE_SCALE = 2.0
STEM_SCALE = 1.5              # per-tap multiplier on stem kt/kh/kw
GRID_GAP = 0.04         # gap between cells in a grid
FE_GAP   = 0.06         # vertical gap between stacked frontend channels

# Inter-stage horizontal gaps (front-face right of N → front-face left of N+1).
# Keyed by the transition name so the Flask tuner can override per-edge.
DEFAULT_GAPS = {
    "fe_to_stem":      0.385,  # frontend → stem (frontend has tiny depth)
    "stem_to_blk1":    0.45,
    "blk1_to_blk2":    1.21875,
    "blk2_to_gru":     0.80,   # widened to fit Behavior box below the arrow
    "gru_to_readout":  0.30,   # readouts sit close to ConvGRU
}
DEFAULT_X_START = 0.0           # world x of the front-lower-left of the frontend

# Shared front-face vertical CENTER for every stage. Choosing a center
# (rather than a baseline) means stages with different y-extents all line
# up — the inter-stage flow arrows run perfectly horizontal at this y.
ARCH_CENTER_Y = 7.0

# Per-stage label spacing (labels attach to each stage's front-face top,
# so they ride the architectural "skyline" instead of floating in a fixed
# band far above the shorter stages).
LABEL_GAP     = 0.18    # title sits this far above the stage front-top
SUB_GAP       = 0.32    # sub-label sits this far ABOVE the title
HEADER_GAP    = 0.55    # header sits this far above the highest sub-label

# Residual ⊔-skip that taps off the upstream flow arrow and merges into a
# '+' marker on the downstream flow arrow. SKIP_DEPTH is measured from
# ARCH_CENTER_Y (the flow line) and must clear the deepest block bottom
# plus a small margin so the ⊔ floor sits cleanly under the blocks.
SKIP_DEPTH = 1.8
SKIP_COLOR = "#222"     # same colour as flow arrows
SKIP_LW    = 1.0
SKIP_CORNER_R = 0.12

# Behavior placeholder. Drops below the ConvGRU block (clearing the
# pipeline body) and is left-justified to the vertical concat arrow that
# feeds the concat marker on the blk2→gru flow arrow.
BEH_HEIGHT = 0.32
BEH_WIDTH  = 0.825
BEH_Y_GAP  = 0.20       # gap from ConvGRU bottom down to top of behavior box

# Canvas (will be tightened in plot)
CANVAS_W = 22.0
CANVAS_H = 11.0


# Per-stage color palettes (front, side/left, top, edge).
PAL_FRONTEND = ("#fff2cc", "#e6c97a", "#c9a945", "#7a5e10")
PAL_STEM     = ("#d8e4f4", "#9fbdda", "#6a8db0", "#1b3a5b")
PAL_BLOCK1   = ("#cfe2f3", "#7fa4c4", "#4d7396", "#1b3a5b")
PAL_BLOCK2   = ("#b9d3eb", "#6790b8", "#3a6189", "#0e2b4d")
PAL_GRU      = ("#ead6f5", "#b685d3", "#7e3f8a", "#3e1a48")
PAL_READOUT  = ("#d9ecd9", "#8cc28c", "#3f8a3f", "#1f5e1f")


# Cabinet depth vector (mirror of _fig4a_glyphs.CAB_DEPTH_VEC).
# +z (into page) projects UP-AND-RIGHT, so back-layer kernels in a grid
# stack toward the upper-right of the front layer.
_CAB_ALPHA = np.deg2rad(45.0)
_CAB_DEPTH = 0.50
_CAB_DEPTH_VEC = np.array([
    +np.cos(_CAB_ALPHA) * _CAB_DEPTH,
    +np.sin(_CAB_ALPHA) * _CAB_DEPTH,
])


def plot_panel_a_architecture(ax, assets, *, gaps=None, x_start=None):
    """Render the model-architecture half of panel A into `ax`.

    `gaps` overrides individual entries of `DEFAULT_GAPS` (front-face right of
    one stage → front-face left of the next). `x_start` overrides the
    world x of the frontend's front-lower-left corner. Both are wired up so
    the Flask tuner (`tune_fig4a_arch.py`) can sweep placement live.
    """
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    g = dict(DEFAULT_GAPS)
    if gaps:
        g.update(gaps)

    arch = assets.arch
    arch_kernels = arch["convnet_kernels"]   # [(3,9,9), (3,5,5)]
    blk1_kt, blk1_kh, blk1_kw = arch_kernels[0]
    blk2_kt, blk2_kh, blk2_kw = arch_kernels[1]
    stem_kt, stem_kh, stem_kw = (1, 7, 7)     # from YAML
    gru_kt, gru_kh, gru_kw = (1, arch["gru_kernel"], arch["gru_kernel"])

    # Track inter-stage flow arrow endpoints in projected 2D.
    stage_records = []
    label_tops = []   # collected y of each stage's title baseline
    x_cursor = float(DEFAULT_X_START if x_start is None else x_start)

    # ── Frontend ────────────────────────────────────────────────────────
    # Vertical stack (rows=fe_n, cols=1): each channel is its own beam,
    # visually distinct. Front face = oscilloscope showing the learned
    # temporal weight curve for that channel.
    fe_unit = S_PIX * FE_SCALE
    fe_kt = arch["frontend_k"] * fe_unit
    fe_kh = fe_unit
    fe_kw = fe_unit
    fe_n  = arch["frontend_channels"]
    fe_y0 = _y0_for_center(rows=fe_n, kh=fe_kh, gap=FE_GAP,
                           cols=1, kw=fe_kw)
    fe_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=fe_y0, z0=0.0,
        n_channels=fe_n, rows=fe_n, cols=1,
        kt=fe_kt, kh=fe_kh, kw=fe_kw,
        gap=FE_GAP, palette=PAL_FRONTEND, base_zorder=2.0, edge_width=0.45,
    )

    # Overlay the learned temporal weight curve on each kernel's front
    # face. All channels share one y-scale so amplitudes are comparable.
    fe_weights = np.asarray(assets.frontend_weights)
    w_min = float(fe_weights.min())
    w_max = float(fe_weights.max())
    if w_max == w_min:
        w_max = w_min + 1.0
    pad = 0.10 * (w_max - w_min)
    w_min -= pad; w_max += pad
    K = fe_weights.shape[1]
    # Flip time L→R so present (t=0) is on the right edge of the kernel
    # face — matches the stimulus lag cube's "newest frame at front" axis.
    ts = np.linspace(1, 0, K)
    for c in range(fe_n):
        y_cell = fe_y0 + c * (fe_kh + FE_GAP)
        xs = x_cursor + ts * fe_kt
        ys = y_cell + (fe_weights[c] - w_min) / (w_max - w_min) * fe_kh
        # Faint zero baseline if range crosses zero.
        if w_min < 0 < w_max:
            y_zero = y_cell + (0 - w_min) / (w_max - w_min) * fe_kh
            ax.plot([x_cursor, x_cursor + fe_kt], [y_zero, y_zero],
                    color="#bbb", lw=0.4, zorder=3.6)
        ax.plot(xs, ys, color="#3a2406", lw=1.0, zorder=3.8,
                solid_capstyle="round")

    label_tops.append(_stage_label_top(
        ax, fe_grid, name="Frontend",
        sub=f"{fe_n} ch · k={arch['frontend_k']}"))

    stage_records.append({"name": "frontend", "grid": fe_grid})
    x_cursor = _next_x(fe_grid, g["fe_to_stem"])

    # ── Stem ────────────────────────────────────────────────────────────
    stem_kt_w = stem_kt * S_T   * STEM_SCALE
    stem_kh_w = stem_kh * S_PIX * STEM_SCALE
    stem_kw_w = stem_kw * S_PIX * STEM_SCALE
    stem_y0 = _y0_for_center(rows=2, kh=stem_kh_w, gap=GRID_GAP,
                             cols=4, kw=stem_kw_w)
    stem_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=stem_y0, z0=0.0,
        n_channels=8, rows=2, cols=4,
        kt=stem_kt_w, kh=stem_kh_w, kw=stem_kw_w,
        gap=GRID_GAP, palette=PAL_STEM, base_zorder=2.0, edge_width=0.25,
        hue_jitter=0.08,
    )
    label_tops.append(_stage_label_top(
        ax, stem_grid, name="Stem",
        sub=f"{stem_kt}×{stem_kh}×{stem_kw} · 8 ch"))
    stage_records.append({"name": "stem", "grid": stem_grid})
    x_cursor = _next_x(stem_grid, g["stem_to_blk1"])

    # ── ResBlock 1 ──────────────────────────────────────────────────────
    blk1_y0 = _y0_for_center(rows=8, kh=blk1_kh * S_PIX, gap=GRID_GAP,
                             cols=8, kw=blk1_kw * S_PIX)
    blk1_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=blk1_y0, z0=0.0,
        n_channels=64, rows=8, cols=8,
        kt=blk1_kt * S_T, kh=blk1_kh * S_PIX, kw=blk1_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_BLOCK1, base_zorder=2.0, edge_width=0.20,
        hue_jitter=0.10,
    )
    label_tops.append(_stage_label_top(
        ax, blk1_grid, name="ResBlock 1",
        sub=f"{blk1_kt}×{blk1_kh}×{blk1_kw} · 64 ch"))
    stage_records.append({"name": "block1", "grid": blk1_grid})
    x_cursor = _next_x(blk1_grid, g["blk1_to_blk2"])

    # ── ResBlock 2 ──────────────────────────────────────────────────────
    # Tall layout (rows=16, cols=8): taller than wide for a tighter footprint.
    blk2_y0 = _y0_for_center(rows=16, kh=blk2_kh * S_PIX, gap=GRID_GAP,
                             cols=8, kw=blk2_kw * S_PIX)
    blk2_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=blk2_y0, z0=0.0,
        n_channels=128, rows=16, cols=8,
        kt=blk2_kt * S_T, kh=blk2_kh * S_PIX, kw=blk2_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_BLOCK2, base_zorder=2.0, edge_width=0.18,
        hue_jitter=0.10,
    )
    label_tops.append(_stage_label_top(
        ax, blk2_grid, name="ResBlock 2",
        sub=f"{blk2_kt}×{blk2_kh}×{blk2_kw} · 128 ch"))
    stage_records.append({"name": "block2", "grid": blk2_grid})
    x_cursor = _next_x(blk2_grid, g["blk2_to_gru"])

    # ── ConvGRU ─────────────────────────────────────────────────────────
    # Tall layout to match ResBlock 2.
    gru_y0 = _y0_for_center(rows=16, kh=gru_kh * S_PIX, gap=GRID_GAP,
                            cols=8, kw=gru_kw * S_PIX)
    gru_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=gru_y0, z0=0.0,
        n_channels=128, rows=16, cols=8,
        kt=gru_kt * S_T, kh=gru_kh * S_PIX, kw=gru_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_GRU, base_zorder=2.0, edge_width=0.18,
        hue_jitter=0.10,
    )
    # Tall closed recurrent loop on top of the GRU.
    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    gru_loop_base = g_ymax - 0.10
    gru_loop_height = 0.37
    gru_loop_shift = gru_loop_height / 2.0 * 0.75   # nudge loop 0.75 radius right
    draw_recurrent_loop(ax, x0=g_xmin + gru_loop_shift,
                        x1=g_xmax + gru_loop_shift,
                        y_top=gru_loop_base,
                        arc_height=gru_loop_height,
                        color="#7e3f8a", lw=1.4,
                        label=None, gap_frac=0.28)
    # Label above the loop (loop_top + tiny pad)
    gru_loop_top = gru_loop_base + gru_loop_height
    label_tops.append(_stage_label_top(
        ax, gru_grid, name="ConvGRU",
        sub=f"{arch['gru_hidden']} ch · k={arch['gru_kernel']}",
        y_top_override=gru_loop_top))
    stage_records.append({"name": "gru", "grid": gru_grid})
    x_cursor = _next_x(gru_grid, g["gru_to_readout"])

    # ── Readouts (example units: 2 on top, ⋮, 1 on bottom) ──────────────
    # Each readout = a horizontal feature-weight block (the 128 depthwise
    # weights as colored bands, frontend-beam style) with a conv-kernel-style
    # spatial prism centered to its right (front face = a localized Gaussian
    # sampler, no per-feature coloring). The split makes the factorized form
    # explicit: a per-feature weighting × a shared spatial sampler. The ⋮
    # stands in for the remaining trained units. All units are real, pulled
    # from the checkpoint (assets.example_neurons), spanning baseline rate.
    examples = assets.example_neurons
    feat_w, feat_h = 0.62, 0.16        # feature-weight block (horizontal beam)
    prism_size = 0.50                  # spatial prism front-face side
    feat_to_prism = 0.14               # gap: feature block → spatial prism
    row_pitch = prism_size + 0.30      # vertical pitch between adjacent readouts
    ellipsis_gap = 0.60                # extra vertical room holding the ⋮ glyph

    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    gru_center_y = 0.5 * (g_ymin + g_ymax)

    # Front-face centers, top → bottom: two readouts a normal pitch apart, a
    # larger gap (with the ⋮), then one readout. Centered on the GRU midline.
    span = 2 * row_pitch + ellipsis_gap
    centers = [gru_center_y + span / 2,                  # top
               gru_center_y + span / 2 - row_pitch,      # second
               gru_center_y - span / 2]                  # bottom
    ellipsis_y = 0.5 * (centers[1] + centers[2])
    # Highest baseline on top → lowest on bottom (examples sorted low→high).
    slot_neurons = [examples[min(i, len(examples) - 1)] for i in (2, 1, 0)]

    prism_x = x_cursor + feat_w + feat_to_prism
    readout_records = []
    for cy, neuron in zip(centers, slot_neurons):
        fb = draw_feature_weight_block(
            ax, neuron["features"],
            x0=x_cursor, y0=cy - feat_h / 2, w=feat_w, h=feat_h, zorder=4.0)
        sp = draw_spatial_readout_prism(
            ax, neuron["mean"], neuron["std"],
            x0=prism_x, y0=cy - prism_size / 2, size=prism_size, zorder=4.2)
        readout_records.append({"fb": fb, "sp": sp})

    # ⋮ between the top pair and the bottom readout (three stacked dots —
    # robust across PDF/PNG backends vs. relying on a unicode glyph).
    ell_x = 0.5 * (x_cursor + prism_x + prism_size)
    for dy in (-0.11, 0.0, 0.11):
        ax.add_patch(Circle((ell_x, ellipsis_y + dy), 0.022,
                            facecolor="#666", edgecolor="none", zorder=4.6))

    # Column extent for the title, GRU→readout arrow, and axis tightening.
    col_x_left = x_cursor
    col_x_right = max(r["sp"]["x_right"] for r in readout_records)
    col_y_bottom = min(r["sp"]["y_bottom"] for r in readout_records)
    col_y_top = max(r["sp"]["y_top"] for r in readout_records)

    # Title + sub-label in the shared stage-label style (bold name + italic
    # grey sub), riding above the top readout.
    n_units = assets.arch.get("n_trained_units")
    sub = f"factorized · N={n_units}" if n_units else "factorized"
    title_x = 0.5 * (col_x_left + col_x_right)
    y_sub = col_y_top + LABEL_GAP
    y_title = y_sub + SUB_GAP
    ax.text(title_x, y_sub, sub, ha="center", va="baseline",
            fontsize=7.0, color="#555", style="italic")
    ax.text(title_x, y_title, "Readouts", ha="center", va="baseline",
            fontsize=8.5, color=TEXT_COLOR, fontweight="bold")
    label_tops.append(y_title)

    stage_records.append({"name": "readout", "ro": {
        "x_left": col_x_left, "x_right": col_x_right,
        "y_bottom": col_y_bottom, "y_top": col_y_top}})

    # ── Inter-stage flow arrows ─────────────────────────────────────────
    arrows = _connect_stages(ax, stage_records[:-1])
    # Single short GRU → readout arrow: from the GRU right anchor to the
    # left edge of the readout column (feature blocks) at the GRU center.
    gru_anchor = _stage_right_anchor(stage_records[-2])
    ax.annotate("",
                xy=(col_x_left - 0.04, gru_anchor[1]),
                xytext=(gru_anchor[0] + 0.04, gru_anchor[1]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                zorder=4.8)

    # ── ResBlock 1 residual: fork from mid(stem→blk1) into '+' at 1/5(blk1→blk2)
    a_in1  = arrows["stem→block1"]
    a_out1 = arrows["block1→block2"]
    x_fork1 = _arrow_frac(a_in1,  0.50)[0]
    x_plus1 = _arrow_frac(a_out1, 1.0 / 5.0)[0]
    draw_arrow_skip(ax, x_fork1, x_plus1, ARCH_CENTER_Y,
                    depth=SKIP_DEPTH, corner_r=SKIP_CORNER_R,
                    color=SKIP_COLOR, lw=SKIP_LW, zorder=4.7)
    draw_op_marker(ax, x_plus1, ARCH_CENTER_Y, color=SKIP_COLOR,
                   radius=0.10, lw=0.9, zorder=12.0)

    # ── ↓2× downsample badge at the midpoint of blk1→blk2 ───────────────
    mx, my = _arrow_frac(a_out1, 0.50)
    draw_pool_glyph(ax, mx, my)

    # ── ResBlock 2 residual: fork 3/4(blk1→blk2) → '+' at 1/3(blk2→gru)
    a_out2 = arrows["block2→gru"]
    x_fork2 = _arrow_frac(a_out1, 0.75)[0]
    x_plus2 = _arrow_frac(a_out2, 1.0 / 3.0)[0]
    draw_arrow_skip(ax, x_fork2, x_plus2, ARCH_CENTER_Y,
                    depth=SKIP_DEPTH, corner_r=SKIP_CORNER_R,
                    color=SKIP_COLOR, lw=SKIP_LW, zorder=4.7)
    draw_op_marker(ax, x_plus2, ARCH_CENTER_Y, color=SKIP_COLOR,
                   radius=0.10, lw=0.9, zorder=12.0)

    # ── Behavior concat at 2/3 of blk2→gru ──────────────────────────────
    # Marker sits on the flow arrow; the Behavior box drops below the
    # ConvGRU block (so it stops crowding the pipeline body) and is
    # left-justified to the vertical arrow feeding the marker.
    x_concat, y_concat = _arrow_frac(a_out2, 2.0 / 3.0)
    gru_bottom = gru_grid["bbox2d"][2]
    beh_x = x_concat
    beh_top = gru_bottom - BEH_Y_GAP
    beh_bot = beh_top - BEH_HEIGHT
    ax.add_patch(Rectangle((beh_x, beh_bot), BEH_WIDTH, BEH_HEIGHT,
                           facecolor="#ead6f5", edgecolor="#7e3f8a",
                           linewidth=0.9, zorder=12.0))
    ax.text(beh_x + BEH_WIDTH / 2, beh_bot + BEH_HEIGHT / 2,
            "Behavior", ha="center", va="center",
            fontsize=8.0, color="#3e1a48", fontweight="bold",
            zorder=12.1)
    # Vertical concat arrow from the box top up to just below the marker,
    # rising along the box's left edge (= x_concat).
    ax.add_patch(FancyArrowPatch(
        (x_concat, beh_top + 0.01),
        (x_concat, y_concat - 0.12),
        arrowstyle="-|>", lw=1.0, color="#7e3f8a",
        mutation_scale=10, zorder=11.9,
    ))
    # Concatenation marker: black circle with a '||' glyph (NOT the '+'
    # used by residual sums) so concat and addition read differently.
    # Same radius as the residual-sum markers. Drawn LAST so it sits above
    # the flow arrow and the behavior arrowhead.
    draw_op_marker(ax, x_concat, y_concat, color="#222",
                   radius=0.10, lw=1.0, zorder=12.5, symbol="||")

    # ── Header ──────────────────────────────────────────────────────────
    # Centered over the pipeline body (Stem → ConvGRU), sitting above the
    # highest stage label.
    body_x_lo = stage_records[1]["grid"]["bbox2d"][0]   # Stem left
    body_x_hi = stage_records[4]["grid"]["bbox2d"][1]   # ConvGRU right
    header_y = max(label_tops) + HEADER_GAP
    ax.text(0.5 * (body_x_lo + body_x_hi), header_y,
            "Digital twin",
            ha="center", va="baseline",
            fontsize=11, color=TEXT_COLOR, fontweight="bold")

    # ── Tighten axes ────────────────────────────────────────────────────
    # Only include actual content extents — historically this seeded the
    # bounds with (0, 0), which left ~half the figure empty below the
    # architecture once we enforced set_aspect("equal").
    all_xs = []
    all_ys = []
    for s in stage_records:
        if "grid" in s:
            xmin, xmax, ymin, ymax = s["grid"]["bbox2d"]
            all_xs.extend([xmin, xmax])
            all_ys.extend([ymin, ymax])
        elif "ro" in s:
            all_xs.extend([s["ro"]["x_left"], s["ro"]["x_right"]])
            all_ys.extend([s["ro"]["y_bottom"], s["ro"]["y_top"]])
    all_ys.extend([col_y_bottom, y_title + 0.35])
    # Header sits above all labels.
    all_ys.append(header_y + 0.35)
    # Residual ⊔-skips dip to y = ARCH_CENTER_Y - SKIP_DEPTH (the U bottoms
    # are anchored to the flow line, not to the block bottoms).
    all_ys.append(ARCH_CENTER_Y - SKIP_DEPTH - 0.20)
    pad_x, pad_y = 0.3, 0.1
    x_lo, x_hi = min(all_xs) - pad_x, max(all_xs) + pad_x
    y_lo, y_hi = min(all_ys) - pad_y, max(all_ys) + pad_y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    # Defensive: matplotlib's imshow always calls ax.set_aspect(...), so
    # any imshow with aspect="auto" silently flattens the axes — making
    # circle patches render as ellipses. Re-anchor to "equal" here so the
    # final state is correct even if a new imshow sneaks in upstream.
    ax.set_aspect("equal")


# ──────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────
def _y0_for_center(*, rows, kh, gap, cols=1, kw=None, center_y=ARCH_CENTER_Y):
    """Front-face bottom y such that the PROJECTED grid (front + depth lift)
    is vertically centered on `center_y`. With the new vertical-heavy
    cabinet projection, the depth stack lifts the visual bbox well above
    the front face; we offset y0 down by half that lift so every stage's
    visual center sits on the same horizontal flow line."""
    if kw is None:
        kw = kh
    front_h = (rows - 1) * (kh + gap) + kh
    z_extent = (cols - 1) * (kw + gap) + kw
    depth_y = abs(_CAB_DEPTH_VEC[1]) * z_extent
    total_h = front_h + depth_y
    return center_y - total_h / 2


def _next_x(grid, gap):
    """Next x-cursor = projected bbox right + gap. With depth projecting
    up-and-right, the back-row kernels of stage N extend rightward beyond
    the front-face right edge; advancing from bbox-right keeps a clean
    visual gap between N's back layer and N+1's front face. In the prior
    up-and-left orientation bbox-right collapsed to front-face right, so
    DEFAULT_GAPS values still read the same visually."""
    return grid["bbox2d"][1] + gap


def _stage_label_top(ax, grid, *, name, sub=None, name_fs=8.5, sub_fs=7.0,
                     y_top_override=None):
    """Place stage name + (optional) sub-label ABOVE the grid, centered on
    the grid's projected bbox so labels clear the back-most kernel layer.
    `y_top_override` lets callers push the label above an overlay
    (e.g. the recurrent loop on top of GRU). Returns the y of the topmost
    text baseline so the caller can position a header above all labels."""
    xmin, xmax, _, ymax = grid["bbox2d"]
    front_xmid = (xmin + xmax) / 2
    front_top = y_top_override if y_top_override is not None else ymax
    if sub:
        y_sub = front_top + LABEL_GAP
        y_title = y_sub + SUB_GAP
        ax.text(front_xmid, y_sub, sub,
                ha="center", va="baseline",
                fontsize=sub_fs, color="#555", style="italic",
                linespacing=1.05)
    else:
        y_title = front_top + LABEL_GAP
    ax.text(front_xmid, y_title, name,
            ha="center", va="baseline",
            fontsize=name_fs, color=TEXT_COLOR, fontweight="bold")
    return y_title


def _connect_stages(ax, stage_records):
    """Draw horizontal flow arrows between consecutive stages, threading
    them along the projected mid-y of each stage's front face. Returns a
    dict keyed by "stageA→stageB" mapping to the arrow's start/end x and
    its y (constant along the arrow). Callers use `_arrow_frac` to anchor
    residual forks, sum markers, and concat markers at fractional points
    along these arrows."""
    arrows = {}
    for i in range(len(stage_records) - 1):
        a = stage_records[i]
        b = stage_records[i + 1]
        ax0 = _stage_right_anchor(a)
        bx0 = _stage_left_anchor(b)
        x_start = ax0[0] + 0.04
        x_end = bx0[0] - 0.04
        y_mid = 0.5 * (ax0[1] + bx0[1])
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                    zorder=4.8)
        arrows[f"{a['name']}→{b['name']}"] = {
            "x_start": x_start, "x_end": x_end, "y": y_mid,
        }
    return arrows


def _arrow_frac(arrow, frac):
    """(x, y) at fraction `frac ∈ [0, 1]` along an arrow record."""
    return (arrow["x_start"] + frac * (arrow["x_end"] - arrow["x_start"]),
            arrow["y"])


def _stage_right_anchor(rec):
    """2D anchor on the right side of a stage for flow-arrow start. X sits
    at the bbox-right (past the back-layer overhang, since depth projects
    up-and-right) so flow arrows don't slice across this stage's own
    back-row kernels. Y sits at the visual bbox center (= ARCH_CENTER_Y
    after the depth-aware shift) so arrows trace a single horizontal line
    through every stage."""
    if "grid" in rec:
        g = rec["grid"]
        xmin, xmax, ymin, ymax = g["bbox2d"]
        return np.array([xmax, 0.5 * (ymin + ymax)])
    ro = rec["ro"]
    return np.array([ro["x_right"], (ro["y_bottom"] + ro["y_top"]) / 2])


def _stage_left_anchor(rec):
    """2D anchor on the left side of a stage for flow-arrow end. Y sits at
    the visual bbox center, mirroring `_stage_right_anchor`."""
    if "grid" in rec:
        g = rec["grid"]
        _, _, ymin, ymax = g["bbox2d"]
        return np.array([g["x_left"], 0.5 * (ymin + ymax)])
    ro = rec["ro"]
    return np.array([ro["x_left"], (ro["y_bottom"] + ro["y_top"]) / 2])
