"""
Figure 2 panel D: mean-variance relationship for the Allen population, in
spikes/second on log-log axes. Each neuron is a point (mean rate, noise-rate
variance). The uncorrected (blue, solid) and FEM-corrected (red, dashed)
slope-through-origin Fano factors appear as parallel slope-1 lines; the grey
dashed Poisson reference (variance = mean count) is also slope-1, so the Fano
factor reads as the vertical gap from Poisson. Shaded bands are session-
clustered bootstrap CIs. The red vertical arrow marks the corrected-vs-
uncorrected drop; stars give its clustered-bootstrap significance.

Counts are converted to rates with the counting-window duration T: rate = N / T,
so mean scales by 1/T and variance by 1/T^2 (slope = Fano / T). In count units
x and y share units and Poisson is the 45-degree identity; in rate units they do
not, which is why this panel is log-log rather than equal-axis linear.

All quantitative annotation (slopes, CIs, p-values) lives in the caption.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from _panel_common import standalone_save, pstars

# White halo so the fitted lines stay legible on top of the point cloud.
_HALO = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
from compute_fig2_data import load_fig2_data

SUBJECT = "Allen"
COLOR_UNCORR = "tab:blue"
COLOR_CORR = "tab:red"


def plot_panel_d(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    WINDOWS_MS = data["WINDOWS_MS"]
    s0 = data["fano_stats"][WINDOWS_MS[0]]
    T = data["WINDOWS_BINS"][0] * data["config"]["DT"]  # window duration (s)

    labels = s0["subject_per_neuron"]
    ps = s0["per_subject"][SUBJECT]
    mask = labels == SUBJECT

    # Counts -> rates: mean / T, variance / T^2.
    x = s0["erate"][mask] / T
    yu = s0["var_u"][mask] / T ** 2
    yc = s0["var_c"][mask] / T ** 2

    # Rate-space slopes (Fano / T) and their clustered-bootstrap CIs.
    s_unc = ps["slope_unc"] / T
    s_cor = ps["slope_cor"] / T
    u_lo, u_hi = np.array(ps["slope_unc_ci"]) / T
    c_lo, c_hi = np.array(ps["slope_cor_ci"]) / T

    xpos = x[x > 0]
    # Extend the fitted lines a bit past the data so the right-side bracket has
    # room to sit clear of the point cloud.
    x_line = xpos.max() * 1.3
    xx = np.geomspace(xpos.min(), x_line, 100)

    # Poisson reference: Var(count) = E(count) -> var_rate = x_rate / T. Drawn
    # above the fitted lines so its position shows even though the uncorrected
    # slope (Fano ~ 1) nearly coincides with it.
    ax.plot(xx, xx / T, "--", color="0.4", lw=1, zorder=6)

    # Neuron cloud.
    ax.scatter(x, yu, s=6, alpha=0.3, c=COLOR_UNCORR, linewidths=0, zorder=2)
    ax.scatter(x, yc, s=6, alpha=0.3, c=COLOR_CORR, linewidths=0, zorder=2)

    # Clustered-bootstrap CI bands (parallel on log-log).
    if np.isfinite(u_lo) and np.isfinite(u_hi):
        ax.fill_between(xx, u_lo * xx, u_hi * xx, color=COLOR_UNCORR, alpha=0.18,
                        lw=0, zorder=3)
    if np.isfinite(c_lo) and np.isfinite(c_hi):
        ax.fill_between(xx, c_lo * xx, c_hi * xx, color=COLOR_CORR, alpha=0.18,
                        lw=0, zorder=3)

    # Fitted lines: blue solid = raw, red dashed = corrected (white halo so they
    # stay visible over the point cloud).
    ax.plot(xx, s_unc * xx, "-", color=COLOR_UNCORR, lw=2, zorder=5,
            path_effects=_HALO)
    ax.plot(xx, s_cor * xx, "--", color=COLOR_CORR, lw=2, zorder=5,
            path_effects=_HALO)

    # Right-side bracket joining the line ends, with stars beside it -- keeps the
    # significance annotation clear of the point cloud.
    y_top = s_unc * x_line
    y_bot = s_cor * x_line
    x_arm = x_line * 1.125  # arm base, gapped off the line ends
    x_b = x_line * 1.25     # bracket spine, separated to the right
    ax.plot([x_b, x_b], [y_bot, y_top], color="k", lw=1.4, zorder=8)
    ax.plot([x_arm, x_b], [y_top, y_top], color="k", lw=1.4, zorder=8)
    ax.plot([x_arm, x_b], [y_bot, y_bot], color="k", lw=1.4, zorder=8)
    ax.text(x_b * 1.12, np.sqrt(y_top * y_bot), pstars(ps["p_slope"]),
            ha="left", va="center", color="k", fontsize=12, zorder=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xpos.min() / 1.2, x_b * 1.9)
    # Extend the y-axis to match the x headroom (avoid lopsided whitespace).
    ymin = float(np.nanmin([yu.min(), yc.min()]))
    ax.set_ylim(ymin / 1.5, y_top * 2.0)
    ax.set_xlabel("Mean rate (spk/s)")
    ax.set_ylabel("Noise variance ((spk/s)$^2$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR_UNCORR, lw=2, label="Raw"),
        Line2D([0], [0], color=COLOR_CORR, lw=2, ls="--", label="Corrected"),
        Line2D([0], [0], color="0.4", lw=1, ls="--", label="Poisson"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_d()
    fig.tight_layout()
    standalone_save(fig, "panel_d_mean_var")
