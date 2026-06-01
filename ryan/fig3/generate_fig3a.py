"""
Figure 3 panel A: covariance decomposition demonstration.

Two rows of matrix heatmaps for Allen_2022-02-16 (primary counting
window):

  Uncorrected:     Σ_total  =  Σ_PSTH  +  Σ_int            (Σ_int lumps FEM + internal)
  FEM-corrected:   Σ_total  =  Σ_PSTH  +  Σ_FEM  +  Σ_int

Σ_total and Σ_PSTH are identical between rows and sit in the same
columns. The uncorrected Σ_int (= Σ_total − Σ_PSTH) spans the bottom
row's Σ_FEM + Σ_int columns so the eye reads the split visually.

Composer usage:
    from generate_fig3a import plot_panel_a, make_axes
    axes = make_axes(fig, subplot_spec)
    plot_panel_a(axes=axes, data=data)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch

from VisionCore.covariance import project_to_psd
from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0

TITLES_BOTTOM = [
    r"$\Sigma_{\mathrm{total}}$",
    r"$\Sigma_{\mathrm{PSTH}}$",
    r"$\Sigma_{\mathrm{FEM}}$",
    r"$\Sigma_{\mathrm{int}}$",
]
TITLES_TOP = [
    r"$\Sigma_{\mathrm{total}}$",
    r"$\Sigma_{\mathrm{PSTH}}$",
    r"$\Sigma_{\mathrm{int}}$",
]
SEPARATORS_BOTTOM = ["=", "+", "+"]
SEPARATORS_TOP = ["=", "+"]

# vlim multipliers (× max(Ctotal)) per matrix
VLIM_BOTTOM = [0.5, 0.25, 0.5, 0.5]
VLIM_TOP = [0.5, 0.25, 0.5]


def make_axes(fig, subplot_spec=None):
    """Create axes for panel A. Returns dict with "mats_top", "seps_top",
    "mats_bot", "seps_bot"."""
    kwargs = dict(width_ratios=[1, 0.25, 1, 0.25, 1, 0.25, 1],
                  height_ratios=[1, 1], wspace=0.05, hspace=0.25)
    if subplot_spec is None:
        gs = fig.add_gridspec(2, 7, **kwargs)
    else:
        gs = GridSpecFromSubplotSpec(2, 7, subplot_spec=subplot_spec, **kwargs)

    # Top row: total at col 0, PSTH at col 2, int_uncorrected spanning cols 4:7
    mats_top = [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[0, 4:7])]
    seps_top = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]

    # Bottom row: 4 matrices in their canonical columns
    mats_bot = [fig.add_subplot(gs[1, c]) for c in (0, 2, 4, 6)]
    seps_bot = [fig.add_subplot(gs[1, c]) for c in (1, 3, 5)]

    for ax in seps_top + seps_bot:
        ax.axis("off")
    return {"mats_top": mats_top, "seps_top": seps_top,
            "mats_bot": mats_bot, "seps_bot": seps_bot}


def _style_matrix_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("k")
        ax.spines[side].set_linewidth(1.0)


def plot_panel_a(axes=None, fig=None, refresh=False, data=None, font_scale=1.0):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=(14, 7))
        axes = make_axes(fig)

    sr = next((s for s in data["session_results"]
               if s["session"] == TARGET_SESSION), None)
    if sr is None:
        avail = [s["session"] for s in data["session_results"]]
        raise ValueError(f"{TARGET_SESSION} not in fig2 cache. Available: {avail}")
    mats = sr["mats"][WINDOW_IDX]
    Crate_raw = mats["Intercept"]

    valid = (np.isfinite(np.diag(Crate_raw))
             & np.isfinite(np.diag(mats["PSTH"])))
    ix = np.ix_(valid, valid)
    Ctotal = project_to_psd(mats["Total"][ix])
    Cpsth = project_to_psd(mats["PSTH"][ix])
    Cfem = project_to_psd(Crate_raw[ix] - mats["PSTH"][ix])
    Cint = project_to_psd(mats["Total"][ix] - Crate_raw[ix])
    # Uncorrected internal lumps FEM + internal: Σ_total − Σ_PSTH
    Cint_uncorr = project_to_psd(mats["Total"][ix] - mats["PSTH"][ix])

    cmap = plt.get_cmap("seismic_r")
    cmax = float(np.nanmax(Ctotal))

    s = font_scale
    title_fs = max(8.0, 22.0 * s)
    side_fs = max(7.0, 14.0 * s)
    sep_fs = max(12.0, 36.0 * s)
    panel_label_fs = max(8.0, 14.0 * s)
    arrow_lw = max(1.0, 2.8 * s)
    arrow_scale = max(10.0, 28.0 * s)

    # --- Top row (uncorrected) ---
    top_mats = [Ctotal, Cpsth, Cint_uncorr]
    for ax, mat, title, frac in zip(axes["mats_top"], top_mats,
                                    TITLES_TOP, VLIM_TOP):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=title_fs, pad=6)
        _style_matrix_axis(ax)
    axes["mats_top"][0].set_ylabel("Uncorrected", fontsize=side_fs, labelpad=10)

    for ax, sym in zip(axes["seps_top"], SEPARATORS_TOP):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=sep_fs, transform=ax.transAxes)

    # --- Bottom row (FEM-corrected) ---
    bot_mats = [Ctotal, Cpsth, Cfem, Cint]
    for ax, mat, title, frac in zip(axes["mats_bot"], bot_mats,
                                    TITLES_BOTTOM, VLIM_BOTTOM):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=title_fs, pad=6)
        _style_matrix_axis(ax)
    axes["mats_bot"][0].set_ylabel("FEM-corrected", fontsize=side_fs, labelpad=10)

    for ax, sym in zip(axes["seps_bot"], SEPARATORS_BOTTOM):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=sep_fs, transform=ax.transAxes)

    # Decomposition arrows: top Σ_int splits into bottom Σ_FEM + Σ_int.
    N = Cint_uncorr.shape[0]
    top_int_ax = axes["mats_top"][2]
    fem_ax = axes["mats_bot"][2]
    int_ax = axes["mats_bot"][3]
    fig_obj = top_int_ax.figure
    inset_top = 0.33  # fraction of N inward from corner toward image center (top Σ_int origin)
    inset_bot = 0.18  # fraction of N inward at the bottom-row endpoints
    shrink = 8    # points pulled back from each endpoint so arrows clear the boxes

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=arrow_scale, lw=arrow_lw,
                    color="black", shrinkA=shrink, shrinkB=shrink,
                    zorder=10)
    # Left arrow: top Σ_int bottom-left  →  bottom Σ_FEM top-right
    fig_obj.add_artist(ConnectionPatch(
        xyA=(-0.5 + inset_top * N, N - 0.5), coordsA=top_int_ax.transData,
        xyB=(N - 0.5 - inset_bot * N, -0.5), coordsB=fem_ax.transData,
        **arrow_kw))
    # Right arrow: top Σ_int bottom-right  →  bottom Σ_int top-left
    fig_obj.add_artist(ConnectionPatch(
        xyA=(N - 0.5 - inset_top * N, N - 0.5), coordsA=top_int_ax.transData,
        xyB=(-0.5 + inset_bot * N, -0.5), coordsB=int_ax.transData,
        **arrow_kw))

    # "A" label above the top-left matrix
    axes["mats_top"][0].text(-0.15, 1.20, "A",
                             transform=axes["mats_top"][0].transAxes,
                             fontweight="bold", fontsize=panel_label_fs,
                             va="top", ha="left")


if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 7))
    axes = make_axes(fig)
    plot_panel_a(axes=axes)
    standalone_save(fig, "panel_a_cov_decomp")
