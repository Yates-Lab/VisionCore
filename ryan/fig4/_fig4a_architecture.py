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
    Readout    — Gaussian spatial sampler (mean/std per example neuron),
                 depthwise feature-weight strip, observed/predicted traces.

Output channels are tiled in (y, z); within each tile-grid we draw
back-to-front in z so front kernels correctly occlude rear kernels.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _fig4a_glyphs import (
    TEXT_COLOR,
    draw_channel_grid, draw_skip_staple,
    draw_recurrent_loop, draw_gaussian_readout, draw_pool_glyph,
)


# ──────────────────────────────────────────────────────────────────────────
# World-unit scales — every kernel everywhere uses the SAME mapping
# (taps → world units) so a 9×9 reads visibly larger than a 5×5. The
# frontend uses the same temporal scale as the rest so its 16-tap kernels
# don't dominate; it gets a slightly larger spatial face only because each
# channel is 1×1 in space and needs *some* visible footprint.
# ──────────────────────────────────────────────────────────────────────────
S_PIX = 0.030           # world units per spatial tap (convnet & GRU)
S_T   = S_PIX           # world units per time tap (isotropic: 1×1×1 is a cube)
S_T_FE = S_T            # frontend time scale: identical to rest
S_FE_SPATIAL = 0.20     # frontend "unit" spatial face (taps=1 but drawn big)
GRID_GAP = 0.04         # gap between cells in a grid
FE_GAP   = 0.06         # vertical gap between stacked frontend channels

# Inter-stage horizontal gaps (front-face right of N → front-face left of N+1).
# Keyed by the transition name so the Flask tuner can override per-edge.
DEFAULT_GAPS = {
    "fe_to_stem":      0.55,   # frontend → stem (frontend has tiny depth)
    "stem_to_blk1":    1.00,
    "blk1_to_blk2":    0.65,
    "blk2_to_gru":     0.50,
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

# Residual staple (⊔) beneath each block.
SKIP_DEPTH = 0.55
SKIP_COLOR = "#222"     # same colour as flow arrows (was cyan)
SKIP_LW    = 1.0
SKIP_CORNER_R = 0.10

# Behavior placeholder.
BEH_HEIGHT = 0.40
BEH_WIDTH  = 1.00
BEH_Y_GAP  = 0.55       # gap above the inter-stage arrow midpoint

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
_CAB_DEPTH = 0.5
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
    fe_kt = arch["frontend_k"] * S_T_FE
    fe_kh = S_FE_SPATIAL
    fe_kw = S_FE_SPATIAL
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
    ts = np.linspace(0, 1, K)
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
    stem_y0 = _y0_for_center(rows=2, kh=stem_kh * S_PIX, gap=GRID_GAP,
                             cols=4, kw=stem_kw * S_PIX)
    stem_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=stem_y0, z0=0.0,
        n_channels=8, rows=2, cols=4,
        kt=stem_kt * S_T, kh=stem_kh * S_PIX, kw=stem_kw * S_PIX,
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
    _draw_block_skip(ax, blk1_grid, depth=SKIP_DEPTH)
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
    _draw_block_skip(ax, blk2_grid, depth=SKIP_DEPTH)
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
    gru_loop_base = g_ymax + 0.04
    gru_loop_height = 0.55
    draw_recurrent_loop(ax, x0=g_xmin, x1=g_xmax,
                        y_top=gru_loop_base,
                        arc_height=gru_loop_height,
                        color="#7e3f8a", lw=1.4, inset=0.18,
                        label=None)
    # Label above the loop (loop_top + tiny pad)
    gru_loop_top = gru_loop_base + gru_loop_height
    label_tops.append(_stage_label_top(
        ax, gru_grid, name="ConvGRU",
        sub=f"{arch['gru_hidden']} ch · k={arch['gru_kernel']}",
        y_top_override=gru_loop_top))
    stage_records.append({"name": "gru", "grid": gru_grid})
    x_cursor = _next_x(gru_grid, g["gru_to_readout"])

    # ── Readouts (three example neurons stacked) ────────────────────────
    # Three vertically stacked Gaussian readouts illustrate the per-neuron
    # spatial sampler + depthwise feature weights. Stacking ordering:
    # lowest-baseline neuron at the bottom (already sorted in assets).
    examples = assets.example_neurons
    n_examples = len(examples)
    readout_size = 0.75
    feat_width = 0.18
    row_pitch = readout_size + 0.25
    col_height = readout_size + row_pitch * (n_examples - 1)
    # Center the stack on the kernel baseline so the fan-out arrows from
    # the GRU exit horizontally and the column sits visually inline with
    # the rest of the pipeline.
    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    gru_center_y = 0.5 * (g_ymin + g_ymax)
    col_y0 = gru_center_y - col_height / 2

    readout_records = []
    for k, neuron in enumerate(examples):
        # Bottom-up positions so examples[0] (lowest baseline) sits at bottom.
        y_k = col_y0 + k * row_pitch
        ro_k = draw_gaussian_readout(
            ax, neuron["mean"], neuron["std"], neuron["features"],
            x0=x_cursor, y0=y_k, size=readout_size, feat_width=feat_width,
            zorder=4.0,
        )
        readout_records.append({"ro": ro_k})

    # Header label above the stack.
    col_x_lo = readout_records[0]["ro"]["x_left"]
    col_x_hi = max(r["ro"]["x_right"] for r in readout_records)
    col_top_y = col_y0 + col_height
    ax.text(0.5 * (col_x_lo + col_x_hi), col_top_y + 0.12,
            "Readouts", ha="center", va="bottom",
            fontsize=8.5, color=TEXT_COLOR)

    # Use the middle readout as the "stage anchor" for inter-stage arrows.
    stage_records.append({"name": "readout",
                          "ro": readout_records[len(readout_records) // 2]["ro"]})

    # ── Inter-stage flow arrows ─────────────────────────────────────────
    arrow_mids = _connect_stages(ax, stage_records[:-1])
    # Single short GRU → readout arrow: from the GRU right anchor to the
    # left edge of the readout column at the GRU center height.
    gru_anchor = _stage_right_anchor(stage_records[-2])
    col_x_left = min(r["ro"]["x_left"] for r in readout_records)
    ax.annotate("",
                xy=(col_x_left - 0.04, gru_anchor[1]),
                xytext=(gru_anchor[0] + 0.04, gru_anchor[1]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                zorder=4.8)

    # ── ↓2× downsample badge on block1→block2 arrow ─────────────────────
    if "block1→block2" in arrow_mids:
        mx, my = arrow_mids["block1→block2"]
        draw_pool_glyph(ax, mx, my)

    # ── Behavior placeholder above block2→gru arrow ─────────────────────
    if "block2→gru" in arrow_mids:
        ax_mid_x, ax_mid_y = arrow_mids["block2→gru"]
        beh_x = ax_mid_x - BEH_WIDTH / 2
        beh_y = ax_mid_y + BEH_Y_GAP
        ax.add_patch(Rectangle((beh_x, beh_y), BEH_WIDTH, BEH_HEIGHT,
                               facecolor="#ead6f5", edgecolor="#7e3f8a",
                               linewidth=0.9, zorder=12.0))
        ax.text(beh_x + BEH_WIDTH / 2, beh_y + BEH_HEIGHT / 2,
                "Behavior", ha="center", va="center",
                fontsize=8.0, color="#3e1a48", fontweight="bold",
                zorder=12.1)
        # Concatenation arrow from box bottom down to the inter-stage
        # arrow midpoint (symbolises ⊕ into the recurrent input).
        ax.add_patch(FancyArrowPatch(
            (ax_mid_x, beh_y - 0.02),
            (ax_mid_x, ax_mid_y + 0.05),
            arrowstyle="-|>", lw=1.0, color="#7e3f8a",
            mutation_scale=10, zorder=12.0,
        ))

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
    all_xs = [0]
    all_ys = [0]
    for s in stage_records:
        if "grid" in s:
            xmin, xmax, ymin, ymax = s["grid"]["bbox2d"]
            all_xs.extend([xmin, xmax])
            all_ys.extend([ymin, ymax])
        elif "ro" in s:
            all_xs.extend([s["ro"]["x_left"], s["ro"]["x_right"]])
            all_ys.extend([s["ro"]["y_bottom"], s["ro"]["y_top"]])
    all_ys.extend([col_y0, col_y0 + col_height + 0.35])
    # Header sits above all labels.
    all_ys.append(header_y + 0.35)
    # Skip staples dip below each block.
    blk_y_bot = min(stage_records[i]["grid"]["y_bottom"] for i in (2, 3))
    all_ys.append(blk_y_bot - SKIP_DEPTH - 0.20)
    pad_x, pad_y = 0.3, 0.1
    x_lo, x_hi = min(all_xs) - pad_x, max(all_xs) + pad_x
    y_lo, y_hi = min(all_ys) - pad_y, max(all_ys) + pad_y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


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


def _draw_block_skip(ax, grid, *, depth=SKIP_DEPTH):
    """Per-block residual ⊔-staple beneath a kernel grid. Drops from just
    below the block's front-bottom-left corner, runs across, climbs back
    up to the front-bottom-right corner. Rendered in flow-arrow black."""
    x0 = grid["x_left"] - 0.05
    x1 = grid["x_right"] + 0.05
    y_top = grid["y_bottom"] - 0.05
    draw_skip_staple(ax, x0, x1, y_top,
                     depth=depth, corner_r=SKIP_CORNER_R,
                     color=SKIP_COLOR, lw=SKIP_LW)


def _connect_stages(ax, stage_records):
    """Draw horizontal flow arrows between consecutive stages, threading
    them along the projected mid-y of each stage's front face."""
    arrow_mids = {}
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
        arrow_mids[f"{a['name']}→{b['name']}"] = ((x_start + x_end) / 2, y_mid)
    return arrow_mids


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
