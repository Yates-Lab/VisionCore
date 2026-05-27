"""
Figure 3 panel A: covariance decomposition demonstration.

A row of four matrix heatmaps for Allen_2022-02-16 (primary counting
window), titled with their names, separated by large = and + signs that
spell out the equation in-line:

    Σ_total  =  Σ_PSTH  +  Σ_FEM  +  Σ_int

Each matrix is projected to PSD before display (matching
generate_fig2_supplemental.py). The PSTH matrix uses a tighter colour
limit so its structure is visible alongside the larger-variance Σ_total.

Composer usage:
    from generate_fig3a import plot_panel_a, make_axes
    axes = make_axes(fig, subplot_spec)
    plot_panel_a(axes=axes, data=data)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from VisionCore.covariance import project_to_psd
from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0

TITLES = [
    r"$\Sigma_{\mathrm{total}}$",
    r"$\Sigma_{\mathrm{PSTH}}$",
    r"$\Sigma_{\mathrm{FEM}}$",
    r"$\Sigma_{\mathrm{int}}$",
]
SEPARATORS = ["=", "+", "+"]    # between matrices 0-1, 1-2, 2-3

# vlim multipliers (× max(Ctotal)) per matrix
VLIM_FRAC = [0.5, 0.25, 0.5, 0.5]


def make_axes(fig, subplot_spec=None):
    """Create axes for panel A. Returns dict with "mats" (list of 4
    imshow axes) and "seps" (list of 3 separator text axes)."""
    kwargs = dict(width_ratios=[1, 0.25, 1, 0.25, 1, 0.25, 1], wspace=0.05)
    if subplot_spec is None:
        gs = fig.add_gridspec(1, 7, **kwargs)
    else:
        gs = GridSpecFromSubplotSpec(1, 7, subplot_spec=subplot_spec, **kwargs)

    mat_axes = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4, 6)]
    sep_axes = [fig.add_subplot(gs[0, c]) for c in (1, 3, 5)]
    for ax in sep_axes:
        ax.axis("off")
    return {"mats": mat_axes, "seps": sep_axes}


def plot_panel_a(axes=None, fig=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=(14, 4))
        axes = make_axes(fig)
    mat_axes = axes["mats"]
    sep_axes = axes["seps"]

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

    # Symmetric extent so center = 0; coolwarm_r so positive = blue
    # with a neutral grey midpoint.
    cmap = plt.get_cmap("seismic_r")
    cmax = float(np.nanmax(Ctotal))
    mats_to_show = [Ctotal, Cpsth, Cfem, Cint]

    for ax, mat, title, frac in zip(mat_axes, mats_to_show, TITLES, VLIM_FRAC):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=22, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("k")
            ax.spines[side].set_linewidth(1.0)

    for ax, sym in zip(sep_axes, SEPARATORS):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=36, transform=ax.transAxes)

    # "A" label above the first matrix
    mat_axes[0].set_title(TITLES[0], fontsize=22, pad=6)
    mat_axes[0].text(-0.05, 1.20, "A",
                     transform=mat_axes[0].transAxes,
                     fontweight="bold", fontsize=14, va="top", ha="left")


if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 4))
    axes = make_axes(fig)
    plot_panel_a(axes=axes)
    standalone_save(fig, "panel_a_cov_decomp")
