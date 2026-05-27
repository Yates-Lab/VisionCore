"""Panel A — right half: encoding-model architecture.

Renders the Adapter → Frontend → ConvNet → ConvGRU → Readout prism chain,
the behavior trace mini-plots, the behavior → GRU arrow, and the final
"predicted spike rate" output arrow. Consumes a `PanelAAssets` instance
(loaded by `_fig4a_data.load_panel_a_assets`).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from _fig4a_glyphs import (
    draw_arch_prism,
    draw_behavior_traces,
    flow_arrow,
    TEXT_COLOR,
)


# ──────────────────────────────────────────────────────────────────────────
# Layout (data coordinates within this subplot)
# ──────────────────────────────────────────────────────────────────────────
CANVAS_W = 22.0
CANVAS_H = 13.0

ARCH_BASE_Y = 6.3
ARCH_X0 = 0.5
ARCH_X1 = CANVAS_W - 1.0
ARCH_W = ARCH_X1 - ARCH_X0
N_STAGES = 5
STAGE_SPACING = ARCH_W / N_STAGES

BEH_X = ARCH_X0 + 1.5
BEH_Y = 1.2
BEH_W = ARCH_W - 3.0
BEH_H = 2.6

# (name, w, h, depth_x, depth_y, front_color, side_color, top_color)
ARCH_STAGES = [
    ("Adapter",   1.7, 2.6, 0.30, 0.18,
     "#e8f0fd", "#bfd1ed", "#9eb6d3"),
    ("Frontend",  1.7, 2.6, 0.55, 0.30,
     "#fff2cc", "#e6c97a", "#c9a945"),
    ("ConvNet",   1.6, 2.4, 1.05, 0.55,
     "#cfe2f3", "#7fa4c4", "#4d7396"),
    ("ConvGRU",   1.5, 2.2, 0.95, 0.50,
     "#ead6f5", "#b685d3", "#7e3f8a"),
    ("Readout",   1.0, 1.8, 0.25, 0.15,
     "#d9ecd9", "#8cc28c", "#3f8a3f"),
]


def plot_panel_a_architecture(ax, assets):
    """Render the model-architecture half of panel A into `ax`."""
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    arch = assets.arch

    # Header for the whole zone
    header_y = ARCH_BASE_Y + max(s[2] for s in ARCH_STAGES) + 1.6
    ax.text(CANVAS_W / 2, header_y, "Encoding model",
            ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontweight="bold")

    stage_labels = [s[0] for s in ARCH_STAGES]
    stage_sublabels = [
        f"{arch['adapter_grid']}×{arch['adapter_grid']} lattice",
        f"depthwise T-conv\nc={arch['frontend_channels']}, k={arch['frontend_k']}",
        f"ResNet\nch=[{arch['convnet_channels'][0]}, {arch['convnet_channels'][1]}]",
        f"hidden={arch['gru_hidden']}, k={arch['gru_kernel']}",
        "Gaussian\n→ 1 neuron",
    ]
    max_h = max(s[2] for s in ARCH_STAGES)
    uniform_label_y = ARCH_BASE_Y - 0.45

    stage_centers_x = []
    for i, (stage_def, name, sub) in enumerate(zip(ARCH_STAGES, stage_labels,
                                                    stage_sublabels)):
        _, w, h, dx_p, dy_p, fc, sc, tc = stage_def
        x = ARCH_X0 + i * STAGE_SPACING + (STAGE_SPACING - w - dx_p) / 2
        y = ARCH_BASE_Y + (max_h - h) / 2
        draw_arch_prism(
            ax, x, y, w, h, depth_x=dx_p, depth_y=dy_p,
            front_color=fc, side_color=sc, top_color=tc,
            edge_color="#222", edge_width=0.7, zorder=3,
            label=name, sublabel=sub,
            label_above_y=uniform_label_y,
        )
        stage_centers_x.append(x + (w + dx_p) / 2)

    # Connecting arrows between prisms
    y_mid = ARCH_BASE_Y + max_h / 2
    for i in range(N_STAGES - 1):
        dx_i = ARCH_STAGES[i][3]
        x_i = ARCH_X0 + i * STAGE_SPACING + (STAGE_SPACING - ARCH_STAGES[i][1] - dx_i) / 2
        rx = x_i + ARCH_STAGES[i][1] + dx_i
        x_n = (ARCH_X0 + (i + 1) * STAGE_SPACING
               + (STAGE_SPACING - ARCH_STAGES[i + 1][1] - ARCH_STAGES[i + 1][3]) / 2)
        flow_arrow(ax, rx + 0.05, y_mid, x_n - 0.05, lw=0.9)

    # Input arrow into the leftmost prism
    flow_arrow(ax, -0.6, y_mid, ARCH_X0 + 0.05, lw=1.0)

    # Behavior traces below the chain
    draw_behavior_traces(
        ax,
        assets.behavior_t,
        assets.behavior_eyepos,
        assets.behavior_speed,
        BEH_X, BEH_Y, BEH_W, BEH_H,
    )
    ax.text(BEH_X + BEH_W / 2, BEH_Y + BEH_H + 0.15,
            f"Behavior  (d = {arch['behavior_dim']})",
            ha="center", va="bottom",
            fontsize=8.5, color=TEXT_COLOR, fontweight="bold")

    # Arrow from behavior block UP to the GRU prism
    gru_idx = 3
    gru_x = stage_centers_x[gru_idx]
    gru_h = ARCH_STAGES[gru_idx][2]
    gru_bot = ARCH_BASE_Y + (max_h - gru_h) / 2

    behavior_top = BEH_Y + BEH_H + 0.10
    ax.add_patch(FancyArrowPatch(
        (gru_x, behavior_top),
        (gru_x, gru_bot - 0.05),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="->", linewidth=1.0, color="#7e3f8a",
        zorder=4,
    ))

    # Output arrow + "predicted spike rate" label
    readout_idx = 4
    rdef = ARCH_STAGES[readout_idx]
    x_r = (ARCH_X0 + readout_idx * STAGE_SPACING
           + (STAGE_SPACING - rdef[1] - rdef[3]) / 2)
    out_x0 = x_r + rdef[1] + rdef[3] + 0.05
    out_x1 = min(out_x0 + 1.4, CANVAS_W - 0.05)
    flow_arrow(ax, out_x0, y_mid, out_x1, lw=1.2)
    ax.text(out_x1 + 0.05, y_mid, "predicted\nspike rate",
            ha="left", va="center", fontsize=7, color=TEXT_COLOR,
            linespacing=1.1)
