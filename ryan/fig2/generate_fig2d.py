"""
Figure 2 panel D: mean-variance relationship for the Allen population, in
spikes/second on log-log axes. Each neuron is a point (mean rate, noise-rate
variance). The uncorrected (grey cloud, dark dashed line) and FEM-corrected
(light-blue cloud, solid blue line) slope-through-origin Fano factors appear as
parallel slope-1 lines; the grey dotted Poisson reference (variance = mean
count) is also slope-1, so the Fano factor reads as the vertical gap from
Poisson. A right-side bracket joins the line ends; stars give the corrected-vs-
uncorrected clustered-bootstrap significance. CIs are shown in panel E.

Counts are converted to rates with the counting-window duration T: rate = N / T,
so mean scales by 1/T and variance by 1/T^2 (slope = Fano / T). In count units
x and y share units and Poisson is the 45-degree identity; in rate units they do
not, which is why this panel is log-log rather than equal-axis linear.

All quantitative annotation (slopes, CIs, p-values) lives in the caption.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _panel_common import standalone_save, pstars
from compute_fig2_data import load_fig2_data

SUBJECT = "Allen"
WINDOW_MS = 25.0  # widest separation between raw and corrected slopes

# Raw is de-emphasized (grey cloud, dark dashed line); the FEM-corrected estimate
# is the hero (light-blue cloud, solid red line). Linestyle (dashed=raw,
# solid=corrected) is shared with panel E; red flags the corrected estimate and
# contrasts hard with the blue cloud so the points can stay fairly opaque.
COLOR_UNCORR_SCATTER = "0.65"
COLOR_UNCORR_LINE = "0.25"
COLOR_CORR_SCATTER = "tab:blue"
COLOR_CORR_LINE = "tab:red"


def plot_panel_d(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    WINDOWS_MS = data["WINDOWS_MS"]
    w_idx = WINDOWS_MS.index(WINDOW_MS)
    s0 = data["fano_stats"][WINDOW_MS]
    T = data["WINDOWS_BINS"][w_idx] * data["config"]["DT"]  # window duration (s)

    labels = s0["subject_per_neuron"]
    ps = s0["per_subject"][SUBJECT]
    mask = labels == SUBJECT

    # Counts -> rates: mean / T, variance / T^2.
    x = s0["erate"][mask] / T
    yu = s0["var_u"][mask] / T ** 2
    yc = s0["var_c"][mask] / T ** 2

    # Rate-space slopes (Fano / T). CIs are shown in panel E, not here.
    s_unc = ps["slope_unc"] / T
    s_cor = ps["slope_cor"] / T

    xpos = x[x > 0]
    # Extend the fitted lines a bit past the data so the right-side bracket has
    # room to sit clear of the point cloud.
    x_line = xpos.max() * 1.3
    xx = np.geomspace(xpos.min(), x_line, 100)

    # Poisson reference: Var(count) = E(count) -> var_rate = x_rate / T. Dotted
    # so it reads distinctly from the dark-dashed uncorrected line it sits near.
    ax.plot(xx, xx / T, ":", color="0.4", lw=1, zorder=6)

    # Neuron clouds: grey = raw (de-emphasized), blue = corrected. Red line
    # over blue points means the cloud can stay fairly opaque without muddying.
    ax.scatter(x, yu, s=6, alpha=0.45, c=COLOR_UNCORR_SCATTER, linewidths=0,
               zorder=2)
    ax.scatter(x, yc, s=6, alpha=0.6, c=COLOR_CORR_SCATTER, linewidths=0,
               zorder=3)

    # Fitted lines: dark dashed = raw, solid blue = corrected (the hero).
    ax.plot(xx, s_unc * xx, "--", color=COLOR_UNCORR_LINE, lw=2, zorder=5)
    ax.plot(xx, s_cor * xx, "-", color=COLOR_CORR_LINE, lw=2.2, zorder=7)

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
    x0, x1 = xpos.min() / 1.2, x_b * 1.9
    # Drop the two lowest-variance outliers when picking the y floor so the
    # frame doesn't zoom out to accommodate them; they'll fall off the plot.
    y_all = np.sort(np.concatenate([yu, yc]))
    ymin = float(y_all[2])
    y0, y1 = ymin / 1.5, y_top * 2.0
    # Equalise log-decade spans and lock aspect so all slope-1 lines (raw,
    # corrected, Poisson) read at 45 deg and the frame is square.
    lx = np.log10(x1 / x0); ly = np.log10(y1 / y0)
    span = max(lx, ly)
    pad_x = (span - lx) / 2
    pad_y = (span - ly) / 2
    ax.set_xlim(x0 * 10 ** -pad_x, x1 * 10 ** pad_x)
    ax.set_ylim(y0 * 10 ** -pad_y, y1 * 10 ** pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean rate (spk/s)")
    ax.set_ylabel("Noise variance ((spk/s)$^2$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR_UNCORR_LINE, lw=2, ls="--", label="Raw"),
        Line2D([0], [0], color=COLOR_CORR_LINE, lw=2.2, label="Corrected"),
        Line2D([0], [0], color="0.4", lw=1, ls=":", label="Poisson"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_d()
    fig.tight_layout()
    standalone_save(fig, "panel_d_mean_var")
