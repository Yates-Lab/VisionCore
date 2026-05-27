"""Panel A — right half: encoding-model architecture (detailed).

Every output channel of every learned stage is rendered as a 3D kernel
prism in shared cabinet projection (same visual language as the stimulus
half's lag cube): time → +x (right), H → +y (up), W → +z (INTO page).

Stages, left → right:
    Frontend   — 4 depthwise temporal kernels (kt=16), with actual learned
                 weights drawn as small line plots above each kernel.
    Stem       — 8 output channels, 1×7×7 kernels, 2×4 tile grid.
    ResBlock 1 — 64 output channels, 3×9×9 kernels, 8×8 tile grid. Per-block
                 residual U-connector beneath. 2× max-pool annotation after.
    ResBlock 2 — 128 output channels, 3×5×5 kernels, 8×16 tile grid. Per-block
                 residual U-connector beneath.
    ConvGRU    — 128 hidden channels, 3×3 kernels, 8×16 tile grid. Curved
                 recurrent self-loop overhead. Behavior trace → GRU arrow.
    Readout    — Gaussian spatial sampler (mean/std of an example neuron)
                 followed by a depthwise feature-weight strip.

The adapter stage is intentionally omitted (likely to be removed from the
model). Output channels are tiled in (y, z); within each tile-grid we draw
back-to-front in z so front kernels correctly occlude rear kernels.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch

from _fig4a_glyphs import (
    CYAN, TEXT_COLOR, cab_project,
    draw_channel_grid, draw_temporal_weight_traces, draw_skip_U,
    draw_recurrent_loop, draw_gaussian_readout, draw_pool_glyph,
    draw_behavior_traces, draw_kernel_prism, draw_neuron_trace_panel,
)


# ──────────────────────────────────────────────────────────────────────────
# World-unit scales — every kernel everywhere uses the SAME mapping
# (taps → world units) so a 9×9 reads visibly larger than a 5×5 etc.
# Frontend gets a separate, larger pair of scales so the 16-tap temporal
# beams + their weight line-plots are readable.
# ──────────────────────────────────────────────────────────────────────────
S_PIX = 0.030           # world units per spatial tap (convnet & GRU)
S_T   = 0.13            # world units per time tap   (convnet & GRU)
S_T_FE = 0.16           # frontend time scale (kt=16 → 2.56 wide)
S_FE_SPATIAL = 0.55     # frontend "unit" spatial face (taps=1 but drawn big)
GRID_GAP = 0.04         # gap between cells in a grid

# Inter-stage horizontal gap (after cabinet z-shift accounted for).
STAGE_GAP = 1.10

# Vertical anchor for the front-bottom edge of every stage's grid.
ARCH_BASE_Y = 6.5

# Vertical zones below the baseline.
SKIP_DEPTH = 1.05         # how deep the residual U dips
LABEL_Y_OFFSET = -1.65    # stage labels go BELOW the skip-U bottom

# Canvas (will be tightened in plot)
CANVAS_W = 22.0
CANVAS_H = 13.5


# Per-stage color palettes (front, side/left, top, edge).
PAL_FRONTEND = ("#fff2cc", "#e6c97a", "#c9a945", "#7a5e10")
PAL_STEM     = ("#d8e4f4", "#9fbdda", "#6a8db0", "#1b3a5b")
PAL_BLOCK1   = ("#cfe2f3", "#7fa4c4", "#4d7396", "#1b3a5b")
PAL_BLOCK2   = ("#b9d3eb", "#6790b8", "#3a6189", "#0e2b4d")
PAL_GRU      = ("#ead6f5", "#b685d3", "#7e3f8a", "#3e1a48")
PAL_READOUT  = ("#d9ecd9", "#8cc28c", "#3f8a3f", "#1f5e1f")


# Cabinet depth vector (mirror of _fig4a_glyphs.CAB_DEPTH_VEC).
_CAB_ALPHA = np.deg2rad(30.0)
_CAB_DEPTH = 0.5
_CAB_DEPTH_VEC = np.array([
    -np.cos(_CAB_ALPHA) * _CAB_DEPTH,
    +np.sin(_CAB_ALPHA) * _CAB_DEPTH,
])


def _proj(x, y, z):
    return np.array([x, y]) + z * _CAB_DEPTH_VEC


def plot_panel_a_architecture(ax, assets):
    """Render the model-architecture half of panel A into `ax`."""
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    arch = assets.arch
    arch_kernels = arch["convnet_kernels"]   # [(3,9,9), (3,5,5)]
    blk1_kt, blk1_kh, blk1_kw = arch_kernels[0]
    blk2_kt, blk2_kh, blk2_kw = arch_kernels[1]
    stem_kt, stem_kh, stem_kw = (1, 7, 7)     # from YAML
    gru_kt, gru_kh, gru_kw = (1, arch["gru_kernel"], arch["gru_kernel"])

    # Track inter-stage flow arrow endpoints in projected 2D.
    stage_records = []
    arrow_mids = {}
    x_cursor = 0.6   # world x of the front-lower-left corner of next stage

    # ── Frontend ────────────────────────────────────────────────────────
    fe_kt = arch["frontend_k"] * S_T_FE
    fe_kh = S_FE_SPATIAL
    fe_kw = S_FE_SPATIAL
    fe_n = arch["frontend_channels"]
    fe_z_gap = 0.30   # extra depth gap between frontend channels
    fe_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=ARCH_BASE_Y, z0=0.0,
        n_channels=fe_n, rows=1, cols=fe_n,
        kt=fe_kt, kh=fe_kh, kw=fe_kw,
        gap=fe_z_gap, palette=PAL_FRONTEND, base_zorder=2.0, edge_width=0.4,
    )

    # Temporal weights — 4 mini line plots in a HORIZONTAL row above the
    # frontend block, fixed y (not projected through cabinet depth so they
    # don't overlap), each connected to its kernel by a thin hairline.
    fe_weights = np.asarray(assets.frontend_weights)
    fe_xmin, fe_xmax, fe_ymin, fe_ymax = fe_grid["bbox2d"]
    panel_h = 0.85
    panel_gap = 0.18
    panel_w = (fe_xmax - fe_xmin - panel_gap * (fe_n - 1)) / fe_n
    panel_y = fe_ymax + 0.50
    ymin = float(fe_weights.min()); ymax = float(fe_weights.max())
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.10 * (ymax - ymin)
    ymin -= pad; ymax += pad
    ts = np.linspace(0, 1, fe_weights.shape[1])
    for c in range(fe_n):
        px = fe_xmin + c * (panel_w + panel_gap)
        # zero baseline
        if ymin < 0 < ymax:
            y_zero = panel_y + (0 - ymin) / (ymax - ymin) * panel_h
            ax.plot([px, px + panel_w], [y_zero, y_zero],
                    color="#ccc", lw=0.4, zorder=3)
        # framing rectangle (very faint)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((px, panel_y), panel_w, panel_h,
                               fill=False, edgecolor="#bbb", lw=0.4,
                               zorder=2.9))
        xs = px + ts * panel_w
        ys = panel_y + (fe_weights[c] - ymin) / (ymax - ymin) * panel_h
        ax.plot(xs, ys, color="#b8860b", lw=1.2, zorder=4,
                solid_capstyle="round")
        # Hairline from the bottom-middle of the panel down to the
        # corresponding kernel's projected top-center.
        z_c = c * (fe_kw + fe_z_gap)
        kernel_top_center_2d = _proj(x_cursor + fe_kt / 2,
                                     ARCH_BASE_Y + fe_kh, z_c)
        ax.plot([px + panel_w / 2, kernel_top_center_2d[0]],
                [panel_y, kernel_top_center_2d[1]],
                color="#c9a945", lw=0.5, alpha=0.55, zorder=2.85,
                linestyle=(0, (2, 1.5)))
    # "learned temporal kernels" label above the row of plots
    ax.text((fe_xmin + fe_xmax) / 2, panel_y + panel_h + 0.12,
            "learned temporal kernels",
            ha="center", va="bottom",
            fontsize=6.8, color="#555", style="italic")

    _stage_label(ax, fe_grid, name="Frontend",
                 sub=f"depthwise temporal\n{fe_n} ch · k={arch['frontend_k']}")

    stage_records.append({"name": "frontend", "grid": fe_grid})
    x_cursor = _next_x(fe_grid, STAGE_GAP)

    # ── Stem ────────────────────────────────────────────────────────────
    stem_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=ARCH_BASE_Y, z0=0.0,
        n_channels=8, rows=2, cols=4,
        kt=stem_kt * S_T, kh=stem_kh * S_PIX, kw=stem_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_STEM, base_zorder=2.0, edge_width=0.25,
        hue_jitter=0.08,
    )
    _stage_label(ax, stem_grid, name="Stem",
                 sub=f"{stem_kt}×{stem_kh}×{stem_kw} · 8 ch")
    stage_records.append({"name": "stem", "grid": stem_grid})
    x_cursor = _next_x(stem_grid, STAGE_GAP)

    # ── ResBlock 1 ──────────────────────────────────────────────────────
    blk1_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=ARCH_BASE_Y, z0=0.0,
        n_channels=64, rows=8, cols=8,
        kt=blk1_kt * S_T, kh=blk1_kh * S_PIX, kw=blk1_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_BLOCK1, base_zorder=2.0, edge_width=0.20,
        hue_jitter=0.10,
    )
    # Skip-U beneath block1
    _draw_block_skip(ax, blk1_grid, depth=SKIP_DEPTH, label=None)
    _stage_label(ax, blk1_grid, name="ResBlock 1",
                 sub=f"{blk1_kt}×{blk1_kh}×{blk1_kw} · 64 ch")
    stage_records.append({"name": "block1", "grid": blk1_grid})
    x_cursor = _next_x(blk1_grid, STAGE_GAP)

    # ── ResBlock 2 ──────────────────────────────────────────────────────
    blk2_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=ARCH_BASE_Y, z0=0.0,
        n_channels=128, rows=8, cols=16,
        kt=blk2_kt * S_T, kh=blk2_kh * S_PIX, kw=blk2_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_BLOCK2, base_zorder=2.0, edge_width=0.18,
        hue_jitter=0.10,
    )
    _draw_block_skip(ax, blk2_grid, depth=SKIP_DEPTH, label=None)
    _stage_label(ax, blk2_grid, name="ResBlock 2",
                 sub=f"{blk2_kt}×{blk2_kh}×{blk2_kw} · 128 ch")
    stage_records.append({"name": "block2", "grid": blk2_grid})
    x_cursor = _next_x(blk2_grid, STAGE_GAP)

    # ── ConvGRU ─────────────────────────────────────────────────────────
    gru_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=ARCH_BASE_Y, z0=0.0,
        n_channels=128, rows=8, cols=16,
        kt=gru_kt * S_T, kh=gru_kh * S_PIX, kw=gru_kw * S_PIX,
        gap=GRID_GAP, palette=PAL_GRU, base_zorder=2.0, edge_width=0.18,
        hue_jitter=0.10,
    )
    # Recurrent loop arching over the GRU grid.
    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    draw_recurrent_loop(ax, x0=g_xmin + 0.1, x1=g_xmax - 0.1,
                        y_top=g_ymax + 0.05, arc_height=0.75,
                        color="#7e3f8a", lw=1.2)
    _stage_label(ax, gru_grid, name="ConvGRU",
                 sub=f"hidden={arch['gru_hidden']}, k={arch['gru_kernel']}")
    stage_records.append({"name": "gru", "grid": gru_grid})
    x_cursor = _next_x(gru_grid, STAGE_GAP)

    # ── Readouts (three example neurons stacked) ────────────────────────
    # The single illustrative readout is replaced by three vertically
    # stacked readouts, each followed by a small observed/predicted trace
    # panel. Stacking ordering: lowest-baseline neuron at the bottom, so
    # the column reads low → high upward (already sorted in assets).
    examples = assets.example_neurons
    n_examples = len(examples)
    readout_size = 0.55
    feat_width = 0.14
    trace_gap = 0.14
    trace_w = 1.45
    row_pitch = readout_size + 0.22
    col_height = readout_size + row_pitch * (n_examples - 1)
    # Center the stack on the kernel baseline so the fan-out arrows from
    # the GRU exit horizontally and the column sits visually inline with
    # the rest of the pipeline.
    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    gru_center_y = 0.5 * (g_ymin + g_ymax)
    col_y0 = gru_center_y - col_height / 2

    pred_colors = ["#1f5e1f", "#1a508a", "#7a3a1e"]
    readout_records = []
    for k, neuron in enumerate(examples):
        # Bottom-up positions so examples[0] (lowest baseline) sits at bottom.
        y_k = col_y0 + k * row_pitch
        ro_k = draw_gaussian_readout(
            ax, neuron["mean"], neuron["std"], neuron["features"],
            x0=x_cursor, y0=y_k, size=readout_size, feat_width=feat_width,
            zorder=4.0,
        )
        # Trace panel directly to the right of the feature strip.
        trace_x0 = ro_k["x_right"] + trace_gap
        trace_y0 = y_k + 0.02
        trace_h = readout_size - 0.04
        pred_color = pred_colors[k % len(pred_colors)]
        baseline_label = f"{neuron['baseline_rate']:.1f} sp/s"
        is_bottom = (k == 0)
        draw_neuron_trace_panel(
            ax, neuron["t"], neuron["robs_rate"], neuron["rhat_rate"],
            trace_x0, trace_y0, trace_w, trace_h,
            obs_color="#888", pred_color=pred_color,
            zorder=4.2,
            baseline_label=baseline_label,
            show_scale=is_bottom, scale_ms=100,
            scale_sp_s=None,
        )
        # Short arrow connecting readout → trace panel.
        arrow_y = y_k + readout_size / 2
        ax.annotate("",
                    xy=(trace_x0 - 0.02, arrow_y),
                    xytext=(ro_k["x_right"] + 0.02, arrow_y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#444"),
                    zorder=4.1)
        readout_records.append({
            "ro": ro_k,
            "trace_xR": trace_x0 + trace_w,
            "trace_y": arrow_y,
            "pred_color": pred_color,
        })

    # Header label above the stack.
    col_x_lo = examples and readout_records[0]["ro"]["x_left"] or x_cursor
    col_x_hi = max(r["trace_xR"] for r in readout_records)
    col_top_y = col_y0 + col_height
    ax.text(0.5 * (col_x_lo + col_x_hi), col_top_y + 0.22,
            "Readouts", ha="center", va="bottom",
            fontsize=8.5, color=TEXT_COLOR, fontweight="bold")
    ax.text(0.5 * (col_x_lo + col_x_hi), col_top_y + 0.04,
            "Gaussian × depthwise   →   observed / predicted rate",
            ha="center", va="bottom",
            fontsize=6.5, color="#555", style="italic")

    # Mini legend for trace colors (placed under the bottom panel).
    leg_y = col_y0 - 0.32
    leg_x = readout_records[0]["ro"]["x_right"] + trace_gap
    ax.plot([leg_x, leg_x + 0.18], [leg_y, leg_y], color="#888", lw=0.9,
            clip_on=False)
    ax.text(leg_x + 0.22, leg_y, "observed", ha="left", va="center",
            fontsize=6.2, color="#444", clip_on=False)
    ax.plot([leg_x + 0.82, leg_x + 1.00], [leg_y, leg_y],
            color="#1f5e1f", lw=1.3, clip_on=False)
    ax.text(leg_x + 1.04, leg_y, "twin", ha="left", va="center",
            fontsize=6.2, color="#1f5e1f", clip_on=False)

    # Use the bottom readout as the "stage anchor" for inter-stage arrows.
    stage_records.append({"name": "readout",
                          "ro": readout_records[len(readout_records) // 2]["ro"]})

    # ── Inter-stage flow arrows ─────────────────────────────────────────
    arrow_mids = _connect_stages(ax, stage_records[:-1])
    # Custom GRU → readout fan-out: from the GRU right anchor to each
    # readout's left edge.
    gru_anchor = _stage_right_anchor(stage_records[-2])
    for rec in readout_records:
        ro_l = rec["ro"]["x_left"]
        ro_y = (rec["ro"]["y_bottom"] + rec["ro"]["y_top"]) / 2
        ax.annotate("",
                    xy=(ro_l - 0.04, ro_y),
                    xytext=(gru_anchor[0] + 0.04, gru_anchor[1]),
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color="#555",
                                    connectionstyle="arc3,rad=0.05"),
                    zorder=4.8)

    # ── Input arrow into the frontend ───────────────────────────────────
    fe_front_left_2d = _proj(stage_records[0]["grid"]["x_left"],
                             ARCH_BASE_Y + fe_kh / 2, 0.0)
    ax.annotate("",
                xy=(fe_front_left_2d[0] - 0.05, fe_front_left_2d[1]),
                xytext=(fe_front_left_2d[0] - 0.95, fe_front_left_2d[1]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"))
    ax.text(fe_front_left_2d[0] - 0.95, fe_front_left_2d[1] + 0.18,
            "stim", ha="left", va="bottom",
            fontsize=7, color="#555", style="italic")

    # ── ↓2 pool annotation on block1→block2 arrow ───────────────────────
    if "block1→block2" in arrow_mids:
        mx, my = arrow_mids["block1→block2"]
        draw_pool_glyph(ax, mx, my + 0.22)

    # ── Behavior block + behavior → GRU arrow ───────────────────────────
    beh_h = 1.5
    beh_w = 3.6
    gru_xmin = gru_grid["bbox2d"][0]
    gru_xmax = gru_grid["bbox2d"][1]
    beh_x = (gru_xmin + gru_xmax) / 2 - beh_w / 2
    beh_y = 0.75
    draw_behavior_traces(
        ax,
        assets.behavior_t,
        assets.behavior_eyepos,
        assets.behavior_speed,
        beh_x, beh_y, beh_w, beh_h,
    )
    ax.text(beh_x + beh_w / 2, beh_y + beh_h + 0.10,
            f"Behavior  (d = {arch['behavior_dim']})",
            ha="center", va="bottom",
            fontsize=8.0, color=TEXT_COLOR, fontweight="bold")
    # Arrow from behavior block UP to the GRU stack bottom.
    gru_bot_y = ARCH_BASE_Y + LABEL_Y_OFFSET - 0.45
    ax.add_patch(FancyArrowPatch(
        (beh_x + beh_w / 2, beh_y + beh_h + 0.40),
        (gru_xmin + (gru_xmax - gru_xmin) * 0.45, gru_bot_y),
        arrowstyle="->", linewidth=1.0, color="#7e3f8a",
        connectionstyle="arc3,rad=0.10",
        zorder=4.0,
    ))
    # Track behavior block extents for the axis-tightening pass.
    _BEH_EXTENTS = (beh_x, beh_x + beh_w, beh_y - 0.40,
                    beh_y + beh_h + 0.30)

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
    # Trace panels extend further right than the readout strip itself.
    all_xs.extend([r["trace_xR"] for r in readout_records])
    all_ys.extend([col_y0, col_y0 + col_height + 0.35])
    bxmin, bxmax, bymin, bymax = _BEH_EXTENTS
    all_xs.extend([bxmin, bxmax])
    all_ys.extend([bymin, bymax])
    # Top headroom for "Encoding model" header
    all_ys.append(max(all_ys) + 0.5)
    pad_x, pad_y = 0.3, 0.6
    x_lo, x_hi = min(all_xs) - pad_x, max(all_xs) + pad_x
    y_lo, y_hi = min(all_ys) - pad_y, max(all_ys) + pad_y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Header — placed inside the final (tightened) bbox.
    ax.text(0.5 * (x_lo + x_hi), y_hi - 0.10, "Encoding model",
            ha="center", va="top",
            fontsize=11, color=TEXT_COLOR, fontweight="bold")


# ──────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────
def _next_x(grid, gap):
    """Next x-cursor = right edge of grid's projected bbox + gap."""
    xmin, xmax, ymin, ymax = grid["bbox2d"]
    return xmax + gap


def _stage_label(ax, grid, *, name, sub=None, name_fs=8.6, sub_fs=6.6):
    """Place stage name and sublabel below a grid, centered on the grid's
    projected bbox. Labels sit BELOW any skip-U dip (at LABEL_Y_OFFSET)."""
    xmin, xmax, ymin, ymax = grid["bbox2d"]
    cx = (xmin + xmax) / 2
    ax.text(cx, ARCH_BASE_Y + LABEL_Y_OFFSET, name,
            ha="center", va="top",
            fontsize=name_fs, color=TEXT_COLOR, fontweight="bold")
    if sub:
        ax.text(cx, ARCH_BASE_Y + LABEL_Y_OFFSET - 0.28, sub,
                ha="center", va="top",
                fontsize=sub_fs, color="#555", style="italic",
                linespacing=1.1)


def _draw_block_skip(ax, grid, *, depth=0.7, label=None):
    """Per-block residual U beneath a kernel grid. Goes from just below the
    block's front-bottom-left corner across to its front-bottom-right
    corner."""
    x0 = grid["x_left"] - 0.05
    x1 = grid["x_right"] + 0.05
    y_top = ARCH_BASE_Y - 0.05
    draw_skip_U(ax, x0, x1, y_top, depth=depth, color=CYAN,
                lw=1.3, label=label, label_fontsize=7.5)


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
    """2D anchor on the right side of a stage for flow-arrow start."""
    if "grid" in rec:
        g = rec["grid"]
        # Use the front-face right-mid (z=0, x=x_right, y=center).
        return _proj(g["x_right"],
                     (g["y_bottom"] + g["y_top"]) / 2,
                     0.0)
    ro = rec["ro"]
    return np.array([ro["x_right"], (ro["y_bottom"] + ro["y_top"]) / 2])


def _stage_left_anchor(rec):
    """2D anchor on the left side of a stage for flow-arrow end."""
    if "grid" in rec:
        g = rec["grid"]
        return _proj(g["x_left"],
                     (g["y_bottom"] + g["y_top"]) / 2,
                     0.0)
    ro = rec["ro"]
    return np.array([ro["x_left"], (ro["y_bottom"] + ro["y_top"]) / 2])
