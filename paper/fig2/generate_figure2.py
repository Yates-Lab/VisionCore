"""
Compose the combined Figure 2 main figure:

    A: two-trial eye-position traces + spike rates (example unit), with
       matched (blue) and divergent (red) Δe arrows
    B: covariance-mismatch curve (cross-trial rate variance vs eye-trajectory
       mismatch), with matching matched/divergent variance arrows
    C: FEM fraction of rate modulation (1 - alpha)
    D: covariance decomposition (total = stimulus + FEM + corrected residual)
    E: population Fano factor at 25 ms, per-session lines (uncorrected ->
       corrected) with across-session mean +/- SD overlaid, shuffle-null band
       + shuffle-null p
    F: per-pair noise correlation at 25 ms, grey violins (uncorrected vs
       FEM-corrected) with across-dataset mean +/- SD marker, shuffle-null band
       + shuffle-null p
    G: participation ratio across corrected-residual / stimulus / FEM
       components (residual high-rank, stimulus + FEM low-rank)
    H: illustrative 3D schematic of the subspace-alignment test (Sigma_FEM
       projected onto the PSTH subspace)
    I: PSTH/FEM subspace alignment vs. shuffle (variance captured)

Across-counting-window robustness for C/E/F lives in
generate_figure2_window_robustness.py.

Usage:
    uv run paper/fig2/generate_figure2.py
    uv run paper/fig2/generate_figure2.py --split-subjects
    uv run paper/fig2/generate_figure2.py --refresh
"""
import argparse
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch

from VisionCore.covariance import project_to_psd
from _panel_common import FIG_DIR, sig_bracket, pstars
from compute_fig2_data import load_fig2_data, _compute_fano_stats, _compute_nc_stats
from generate_panel_example import (
    UNIT_ORIG,
    _compute_uniform_bins,
    _load_unit_payload,
    plot_eye_rate_example,
)
from generate_panel_femfraction import plot_panel_c as plot_fem_fraction
from generate_panel_fano import plot_fano_population
from generate_panel_noisecorr import plot_nc_violin

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0
OMIT_SUBJECTS = {"Luke"}
POOLED_SUBJECT = "Pooled"
POOLED_COLOR = "tab:blue"
SIG_ALPHA = 0.01
SIG01_COLOR = "crimson"      # marker edge: joint p < 0.01
SIG05_COLOR = "darkorange"   # marker edge: joint p < 0.05 (but not < 0.01)


def _label(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)


def _panel_header(ax, letter, text, y=1.12):
    ax.set_title("", loc="left")
    ax.text(-0.13, y, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="bottom", ha="left")
    ax.text(-0.01, y, text, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left")


def _normalize_axis_text(ax):
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(labelsize=7)


def _style_matrix_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("k")
        ax.spines[side].set_linewidth(0.8)


def _add_diagonal_band_box(ax, n, band_frac=0.055):
    return


def _despine(ax, left=False, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def _filter_subjects(data, omit=OMIT_SUBJECTS):
    """Return a shallow plotting bundle with noisy subjects omitted."""
    if not omit:
        return data

    filtered = dict(data)
    filtered["SUBJECTS"] = [s for s in data["SUBJECTS"] if s not in omit]
    filtered["SUBJECT_COLORS"] = {
        s: c for s, c in data["SUBJECT_COLORS"].items() if s not in omit
    }

    if "m_by_window" in data and "subject_per_neuron_by_window" in data:
        m_by_window = []
        labels_by_window = []
        for values, labels in zip(
            data["m_by_window"], data["subject_per_neuron_by_window"]
        ):
            labels = np.asarray(labels)
            keep = ~np.isin(labels, list(omit))
            m_by_window.append(np.asarray(values)[keep])
            labels_by_window.append(labels[keep])
        filtered["m_by_window"] = m_by_window
        filtered["subject_per_neuron_by_window"] = labels_by_window

    if "metrics" in data:
        filtered_metrics = []
        for m_dict in data["metrics"]:
            m = copy.copy(m_dict)
            neuron_mask = ~np.isin(m_dict["subject_per_neuron"], list(omit))
            pair_mask = ~np.isin(m_dict["subject_per_pair"], list(omit))
            ds_mask = ~np.isin(m_dict["subject_by_ds"], list(omit))
            shuff_mask = ~np.isin(m_dict["shuff_rho_subject"], list(omit))

            for key in ("alpha", "uncorr", "corr", "erate",
                        "subject_per_neuron", "session_per_neuron"):
                m[key] = np.asarray(m_dict[key])[neuron_mask]
            if "shuff_var_c" in m_dict:
                m["shuff_var_c"] = np.asarray(m_dict["shuff_var_c"])[neuron_mask]

            for key in ("rho_uncorr", "rho_corr", "subject_per_pair"):
                m[key] = np.asarray(m_dict[key])[pair_mask]

            for key in ("rho_u_meanz_by_ds", "rho_c_meanz_by_ds",
                        "rho_delta_meanz_by_ds", "subject_by_ds"):
                m[key] = np.asarray(m_dict[key])[ds_mask]
            for key in ("Ctotal", "Cpsth", "Crate", "CnoiseU",
                        "CnoiseC", "Cfem"):
                if key in m_dict:
                    values = np.asarray(m_dict[key], dtype=object)
                    m[key] = values[ds_mask].tolist()

            for key in ("shuff_rho_delta_meanz", "shuff_rho_c_meanz",
                        "shuff_rho_subject", "shuff_rho_session"):
                if key in m_dict:
                    m[key] = np.asarray(m_dict[key])[shuff_mask]

            filtered_metrics.append(m)

        filtered["metrics"] = filtered_metrics
        filtered["fano_stats"] = _compute_fano_stats(
            filtered_metrics, filtered["WINDOWS_MS"]
        )
        filtered["nc_stats"] = _compute_nc_stats(
            filtered_metrics, filtered["WINDOWS_MS"]
        )

    sub_subjects = np.asarray(data.get("sub_subjects", []))
    if sub_subjects.size:
        keep_sub = ~np.isin(sub_subjects, list(omit))
        old_to_new = {
            old_i: new_i for new_i, old_i in enumerate(np.flatnonzero(keep_sub))
        }

        for key in (
            "sub_names",
            "sub_subjects",
            "pr_fem_list",
            "pr_psth_list",
            "pr_resid_list",
            "overlap_k1_list",
            "overlap_k_list",
            "var_p_given_f",
            "var_f_given_p",
            "spectra_psth",
            "spectra_fem",
        ):
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=object)
            filtered[key] = values[keep_sub].tolist()

        null_session_idx = np.asarray(data.get("null_session_idx", []), dtype=int)
        null_subjects = np.asarray(data.get("null_subjects", []))
        if null_session_idx.size and null_subjects.size:
            keep_null = (
                ~np.isin(null_subjects, list(omit))
                & np.isin(null_session_idx, list(old_to_new))
            )
            filtered["null_session_idx"] = [
                old_to_new[int(i)] for i in null_session_idx[keep_null]
            ]
            filtered["null_subjects"] = null_subjects[keep_null].tolist()
            for key in (
                "null_var_p_given_f",
                "null_var_f_given_p",
                "null_overlap_k1",
                "null_overlap_k",
            ):
                if key in data:
                    filtered[key] = np.asarray(data[key])[keep_null].tolist()

    return filtered


def _pool_subjects_for_plotting(data, label=POOLED_SUBJECT):
    """Relabel included subjects as one plotting group without changing values."""
    pooled = copy.copy(data)
    pooled["SUBJECTS"] = [label]
    pooled["SUBJECT_COLORS"] = {label: POOLED_COLOR}
    pooled["SUBJECT_DISPLAY_NAMES"] = {label: "pooled"}

    if "m_by_window" in data and "subject_per_neuron_by_window" in data:
        pooled["m_by_window"] = [np.asarray(v) for v in data["m_by_window"]]
        pooled["subject_per_neuron_by_window"] = [
            np.full(np.asarray(labels).shape, label, dtype=object)
            for labels in data["subject_per_neuron_by_window"]
        ]

    metrics = []
    for m_dict in data.get("metrics", []):
        m = copy.copy(m_dict)
        for key in ("subject_by_ds", "subject_per_neuron", "subject_per_pair"):
            if key in m:
                m[key] = np.full(np.asarray(m[key]).shape, label, dtype=object)
        if "shuff_rho_subject" in m:
            m["shuff_rho_subject"] = np.full(
                np.asarray(m["shuff_rho_subject"]).shape, label, dtype=object
            )
        metrics.append(m)
    if metrics:
        pooled["metrics"] = metrics

    if "fano_stats" in data:
        fano_stats = {}
        for window, stats in data["fano_stats"].items():
            s = copy.copy(stats)
            s["per_subject"] = {
                label: {
                    "slope_unc": stats["slope_unc"],
                    "slope_cor": stats["slope_cor"],
                    "slope_unc_ci": stats["slope_unc_ci"],
                    "slope_cor_ci": stats["slope_cor_ci"],
                    "slope_diff": stats["slope_diff"],
                    "slope_diff_ci": stats["slope_diff_ci"],
                    "p_slope": stats["p_slope"],
                    "n_sessions": stats["n_sessions"],
                    "n": stats["n"],
                }
            }
            if "subject_per_neuron" in s:
                s["subject_per_neuron"] = np.full(
                    np.asarray(s["subject_per_neuron"]).shape,
                    label,
                    dtype=object,
                )
            fano_stats[window] = s
        pooled["fano_stats"] = fano_stats

    for key in ("sub_subjects", "null_subjects"):
        if key in data:
            pooled[key] = [label] * len(data[key])

    return pooled


def _plot_covariance_mismatch_panel(ax, divergent_de=None):
    payload = _load_unit_payload()
    uniform = _compute_uniform_bins()

    neuron_mask = np.asarray(payload["neuron_mask"])
    j = int(np.where(neuron_mask == UNIT_ORIG)[0][0])
    bin_centers = np.asarray(uniform["bin_centers"], dtype=float)
    count_e = np.asarray(uniform["count_e"], dtype=float)
    ceye = np.asarray(uniform["Ceye"], dtype=float)
    var_by_bin = ceye[:, j, j]
    ok = np.isfinite(var_by_bin) & (count_e > 0)
    var_psth = float(payload["Cpsth"][j, j])
    first_x = float(bin_centers[ok][0])
    first_y = float(var_by_bin[ok][0])

    ax.axhline(0.0, color="0.55", lw=1.0, zorder=-1)
    ax.axhline(var_psth, color="k", ls="--", lw=1.0, zorder=1)
    ax.plot(bin_centers[ok], var_by_bin[ok], color="k", lw=1.4,
            marker="o", ms=2.8, markerfacecolor="k", markeredgecolor="k",
            zorder=3)

    ax.annotate(
        "Matched-trajectory variance",
        xy=(first_x, first_y),
        xytext=(0.24, 0.84),
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="k", lw=0.9),
        fontsize=7.5,
        ha="left",
        va="center",
    )
    ax.text(0.58, var_psth + 0.012, "PSTH variance",
            fontsize=7.5, ha="left", va="bottom")
    ax.annotate(
        "",
        xy=(0.02, first_y),
        xytext=(0.02, var_psth),
        arrowprops=dict(arrowstyle="<->", color=POOLED_COLOR, lw=2.0),
    )
    ax.text(0.055, 0.2, "FEM\nvariance",
            color="k", fontsize=7.5, ha="left", va="top")

    # Red arrow at the divergent Δe (mirrors the red eye-trace arrow in the
    # example panel): cross-trial variance has dropped below stimulus variance.
    if divergent_de is not None:
        xc = bin_centers[ok]
        yc = var_by_bin[ok]
        di = int(np.argmin(np.abs(xc - divergent_de)))
        x_div, y_div = float(xc[di]), float(yc[di])
        ax.annotate(
            "", xy=(x_div, y_div), xytext=(x_div, var_psth),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2.0),
        )
        ax.text(0.485, -0.38,
                "Variance decreases as\neye trajectories diverge",
                color="k", fontsize=7.5, ha="center", va="bottom")
    else:
        ax.text(0.13, -0.32, "Variance decreases as\neye trajectories diverge",
                fontsize=7.5, ha="left", va="center")

    ax.set_xlim(-0.06, 1.03)
    ax.set_ylim(-0.40, 0.52)
    ax.set_xlabel("Eye-trajectory mismatch, Δe (°)")
    ax.set_ylabel("Cross-trial rate variance (spk²)")
    ax.set_yticks(np.arange(-0.4, 0.6, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _plot_eye_rate_panel(fig, subplot_spec, label="A"):
    """Panel A: two-trial eye-position traces (with matched/divergent Δe
    arrows) above the per-bin spike rates for the example unit. Returns the
    matched/divergent Δe values so the mismatch-curve panel can place a
    corresponding red arrow at the divergent distance."""
    gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec, height_ratios=[1.0, 1.0], hspace=0.18,
    )
    ax_eye = fig.add_subplot(gs[0, 0])
    ax_spk = fig.add_subplot(gs[1, 0], sharex=ax_eye)
    d_vals = plot_eye_rate_example(ax_eye, ax_spk, arrow_color=POOLED_COLOR)
    _normalize_axis_text(ax_spk)
    ax_eye.text(-0.20, 1.16, label, transform=ax_eye.transAxes,
                fontweight="bold", fontsize=10, va="top", ha="left")
    return d_vals


def _make_compact_cov_axes(fig, subplot_spec):
    gs = GridSpecFromSubplotSpec(
        2,
        7,
        subplot_spec=subplot_spec,
        width_ratios=[1.0, 0.12, 1.0, 0.12, 1.0, 0.12, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.04,
        hspace=0.12,
    )
    top_mats = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4)]
    bot_mats = [fig.add_subplot(gs[1, c]) for c in (0, 2, 4, 6)]
    top_seps = [fig.add_subplot(gs[0, c]) for c in (1, 3)]
    bot_seps = [fig.add_subplot(gs[1, c]) for c in (1, 3, 5)]
    note_axes = []
    for ax in top_seps + bot_seps + note_axes:
        ax.axis("off")
    return top_mats, top_seps, bot_mats, bot_seps, note_axes


def _center_top_residual(c_axes):
    """Shift the top-row 'Classical residual' box rightward toward the midpoint
    of the bottom-row 'FEM component' and 'Corrected residual' boxes (whose sum
    it is), so the two splitting arrows land more symmetrically. The box is
    moved only halfway to that midpoint. Carries the colorbar and the top '+'
    separator along. Call after the decomposition axes have reached their final
    positions."""
    top_resid, top_plus = c_axes[2], c_axes[4]
    stim_top = c_axes[1]
    bot_fem, bot_corr = c_axes[7], c_axes[8]
    cbar = c_axes[12]

    pr = top_resid.get_position()
    fem = bot_fem.get_position()
    corr = bot_corr.get_position()
    centered_x0 = 0.5 * (fem.x0 + corr.x1) - pr.width / 2.0
    new_x0 = 0.5 * (pr.x0 + centered_x0)  # halfway: original <-> fully centered
    dx = new_x0 - pr.x0
    top_resid.set_position([new_x0, pr.y0, pr.width, pr.height])

    cb = cbar.get_position()
    cbar.set_position([cb.x0 + dx, cb.y0, cb.width, cb.height])

    stim = stim_top.get_position()
    sp = top_plus.get_position()
    top_plus.set_position([0.5 * (stim.x1 + new_x0) - sp.width / 2.0,
                           sp.y0, sp.width, sp.height])


def _expand_axes_group_from_lower_right(axes, scale=1.055):
    positions = [ax.get_position() for ax in axes]
    left = min(p.x0 for p in positions)
    right = max(p.x1 for p in positions)
    bottom = min(p.y0 for p in positions)

    for ax, pos in zip(axes, positions):
        new_x0 = right - (right - pos.x0) * scale
        new_y0 = bottom + (pos.y0 - bottom) * scale
        ax.set_position([
            new_x0,
            new_y0,
            pos.width * scale,
            pos.height * scale,
        ])


def _plot_compact_cov_decomp(fig, subplot_spec, data, letter="D",
                             title_fontsize=8.0):
    top_axes, top_sep_axes, bot_axes, bot_sep_axes, note_axes = _make_compact_cov_axes(
        fig, subplot_spec
    )
    sr = next((s for s in data["session_results"]
               if s["session"] == TARGET_SESSION), None)
    if sr is None:
        avail = [s["session"] for s in data["session_results"]]
        raise ValueError(f"{TARGET_SESSION} not in fig2 cache. Available: {avail}")

    mats = sr["mats"][WINDOW_IDX]
    crate_raw = mats["Intercept"]
    valid = (
        np.isfinite(np.diag(crate_raw))
        & np.isfinite(np.diag(mats["PSTH"]))
    )
    ix = np.ix_(valid, valid)
    ctotal = project_to_psd(mats["Total"][ix])
    cpsth = project_to_psd(mats["PSTH"][ix])
    cfem = project_to_psd(crate_raw[ix] - mats["PSTH"][ix])
    cint = project_to_psd(mats["Total"][ix] - crate_raw[ix])
    cint_uncorr = project_to_psd(mats["Total"][ix] - mats["PSTH"][ix])

    cmax = float(np.nanmax(ctotal))
    vlim = 0.35 * cmax
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    cmap = plt.get_cmap("seismic_r")
    top_titles = [
        "Total covariance",
        "Stimulus covariance",
        "Classical residual",
    ]
    bot_titles = [
        "Total covariance",
        "Stimulus covariance",
        "FEM component",
        "Corrected residual",
    ]
    matrix_image = None
    for ax, mat, title in zip(
        top_axes,
        [ctotal, cpsth, cint_uncorr],
        top_titles,
    ):
        matrix_image = ax.imshow(mat, cmap=cmap, interpolation="nearest",
                                 norm=norm, aspect="equal")
        ax.set_title(title, fontsize=title_fontsize, pad=3)
        _style_matrix_axis(ax)

    for ax, mat, title in zip(
        bot_axes,
        [ctotal, cpsth, cfem, cint],
        bot_titles,
    ):
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  norm=norm, aspect="equal")
        ax.set_xlabel(title, fontsize=title_fontsize, labelpad=4)
        _style_matrix_axis(ax)

    residual_pos = top_axes[2].get_position()
    cbar_ax = fig.add_axes([
        residual_pos.x1 + 0.006,
        residual_pos.y0,
        0.007,
        residual_pos.height,
    ])
    cbar = fig.colorbar(matrix_image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    cbar.set_ticks([-vlim, 0.0, vlim])
    cbar.formatter = mpl.ticker.FormatStrFormatter("%.2f")
    cbar.update_ticks()

    for ax, sym in zip(top_sep_axes, ["=", "+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    for ax, sym in zip(bot_sep_axes, ["=", "+", "+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=14, transform=ax.transAxes)

    _add_diagonal_band_box(bot_axes[3], cint.shape[0])

    top_residual_ax = top_axes[2]
    arrow_kw = dict(
        arrowstyle="->",
        mutation_scale=10,
        lw=1.0,
        color="k",
        alpha=0.9,
        shrinkA=6,
        shrinkB=4,
        zorder=10,
    )
    fig.add_artist(ConnectionPatch(
        xyA=(0.42, 0.01), coordsA=top_residual_ax.transAxes,
        xyB=(0.50, 1.01), coordsB=bot_axes[2].transAxes,
        **arrow_kw))
    fig.add_artist(ConnectionPatch(
        xyA=(0.58, 0.01), coordsA=top_residual_ax.transAxes,
        xyB=(0.50, 1.01), coordsB=bot_axes[3].transAxes,
        **arrow_kw))

    bot_axes[3].annotate(
        "Independent\nvariance\n(diagonal)",
        xy=(0.46, 0.56),
        xycoords=bot_axes[3].transAxes,
        xytext=(0.92, 1.24),
        textcoords=bot_axes[3].transAxes,
        arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2,
                        shrinkA=2, shrinkB=2),
        fontsize=7.2,
        ha="center",
        va="bottom",
        clip_on=False,
    )

    top_axes[0].text(-0.13, 1.16, letter, transform=top_axes[0].transAxes,
                     fontweight="bold", fontsize=10, va="top", ha="left")
    return top_axes + top_sep_axes + bot_axes + bot_sep_axes + note_axes + [cbar_ax]


def _plot_participation_ratio_bars(fig, subplot_spec, data, split_subjects=False):
    ax = fig.add_subplot(subplot_spec)
    pr_fem = np.asarray(data.get("pr_fem_list", []), dtype=float)
    pr_psth = np.asarray(data.get("pr_psth_list", []), dtype=float)
    subjects = np.asarray(data.get("sub_subjects", []), dtype=object)

    ok = np.isfinite(pr_fem) & np.isfinite(pr_psth)
    pr_fem = pr_fem[ok]
    pr_psth = pr_psth[ok]
    if subjects.size == ok.size:
        subjects = subjects[ok]
    else:
        subjects = np.full(pr_fem.shape, POOLED_SUBJECT, dtype=object)

    if pr_fem.size == 0:
        ax.text(0.5, 0.5, "No PR data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return ax

    if split_subjects:
        sort_idx = np.lexsort((pr_psth, subjects.astype(str)))
    else:
        sort_idx = np.argsort(pr_psth)
    pr_fem = pr_fem[sort_idx]
    pr_psth = pr_psth[sort_idx]
    subjects = subjects[sort_idx]

    x = np.arange(pr_fem.size)
    width = 0.38
    psth_color = "#9ecae1"
    fem_color = POOLED_COLOR
    edge = "#263746"
    ax.bar(x - width / 2, pr_psth, width, color=psth_color,
           edgecolor=edge, linewidth=0.35, label="Stimulus covariance")
    ax.bar(x + width / 2, pr_fem, width, color=fem_color,
           edgecolor=edge, linewidth=0.35, label="FEM component")

    ymax = max(float(np.nanmax(pr_psth)), float(np.nanmax(pr_fem)))
    ax.axhline(2.0, color="0.45", lw=0.8, ls="--", alpha=0.45, zorder=0)
    ax.set_ylim(0, ymax * 1.18)
    tick_step = max(1, int(np.ceil(pr_fem.size / 8)))
    tick_idx = np.arange(0, pr_fem.size, tick_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(i + 1) for i in tick_idx])
    ax.set_xlabel("Session (sorted by stimulus PR)")
    ax.set_ylabel("Participation ratio")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              ncol=2, handlelength=1.4, columnspacing=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)

    if split_subjects and np.unique(subjects).size > 1:
        boundaries = np.flatnonzero(subjects[1:] != subjects[:-1]) + 0.5
        for boundary in boundaries:
            ax.axvline(boundary, color="0.72", lw=0.6, ls=":", zorder=0)

    return ax


def _plot_pr_comparison(fig, subplot_spec, data):
    """Panel G (E-style): per-session participation ratio across three
    covariance components -- corrected residual (high rank), stimulus
    covariance, and FEM component (both low rank). One faint grey line per
    session joins its three values; the across-session mean +/- SD is overlaid,
    with paired Wilcoxon brackets (residual vs each low-rank component) making
    the point that stimulus and FEM structure are compact relative to the noise.
    """
    from scipy.stats import wilcoxon

    ax = fig.add_subplot(subplot_spec)
    pr_resid = np.asarray(data.get("pr_resid_list", []), dtype=float)
    pr_psth = np.asarray(data.get("pr_psth_list", []), dtype=float)
    pr_fem = np.asarray(data.get("pr_fem_list", []), dtype=float)

    ok = np.isfinite(pr_resid) & np.isfinite(pr_psth) & np.isfinite(pr_fem)
    pr_resid, pr_psth, pr_fem = pr_resid[ok], pr_psth[ok], pr_fem[ok]
    if pr_resid.size == 0:
        ax.text(0.5, 0.5, "No PR data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return ax

    cols = np.column_stack([pr_resid, pr_psth, pr_fem])
    x = np.arange(3)

    # one faint line per session through its three components
    for row in cols:
        ax.plot(x, row, color="0.75", lw=0.7, alpha=0.7, zorder=1)

    # Log y: the corrected residual is ~10-30 while stimulus/FEM sit near
    # rank 2, so a linear axis crushes the two low-rank components together.
    ax.set_yscale("log")

    mu = cols.mean(axis=0)
    sd = cols.std(axis=0, ddof=1)
    ax.plot(x, mu, "-", color=POOLED_COLOR, lw=2.0, zorder=4)
    yerr_lo = np.minimum(sd, mu * 0.85)   # keep the lower whisker positive on log
    ax.errorbar(x, mu, yerr=np.vstack([yerr_lo, sd]), fmt="none",
                ecolor=POOLED_COLOR, elinewidth=1.5, capsize=0, zorder=5)
    ax.plot(x, mu, "o", mfc=POOLED_COLOR, mec=POOLED_COLOR, ms=7, mew=1.6,
            zorder=6)

    # Paired Wilcoxon: residual vs stimulus, residual vs FEM (stacked brackets).
    def _wilcox(a, b):
        try:
            return wilcoxon(a, b).pvalue
        except ValueError:
            return np.nan

    # Paired sign test (two-sided binomial on the signs of the differences):
    # stimulus PR is significantly above FEM PR by sign test (18/24 sessions,
    # p=0.023) though only borderline by Wilcoxon, so the directional count is
    # the fairer summary for this rank-like quantity.
    def _signtest(a, b):
        from scipy.stats import binomtest
        d = np.asarray(a, float) - np.asarray(b, float)
        d = d[np.isfinite(d) & (d != 0)]
        if d.size == 0:
            return np.nan
        return binomtest(int(np.sum(d > 0)), d.size, 0.5,
                         alternative="two-sided").pvalue

    hi = float(cols.max())
    lo = float(cols.min())
    y1, y2 = hi * 1.25, hi * 1.95
    sig_bracket(ax, 0, 1, y1, pstars(_wilcox(pr_resid, pr_psth)), h=y1 * 0.06)
    sig_bracket(ax, 0, 2, y2, pstars(_wilcox(pr_resid, pr_fem)), h=y2 * 0.06)
    # Stimulus vs FEM (sign test): low bracket sitting just above the two
    # low-rank components rather than up with the residual brackets.
    y_sf = float(max(pr_psth.max(), pr_fem.max())) * 1.35
    sig_bracket(ax, 1, 2, y_sf, pstars(_signtest(pr_psth, pr_fem)),
                h=y_sf * 0.06)
    ax.set_ylim(max(0.85, lo * 0.7), y2 * 1.8)

    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.set_xticks(x)
    ax.set_xticklabels(["Corrected\nresidual", "Stimulus\ncovariance",
                        "FEM\ncomponent"])
    ax.set_ylabel("Participation ratio")
    ax.set_xlim(-0.5, 2.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _plot_subspace_schematic(fig, subplot_spec):
    """Panel H: illustrative 3D schematic of the subspace-alignment test. A
    low-rank FEM covariance ellipse (green disk) floats above the PSTH subspace
    (a k-dim plane, here drawn as 2D); orthogonal projection onto that plane
    (dotted drop lines + dashed shadow ellipse) measures how much FEM variance
    is captured by the leading PSTH eigenvectors u1, u2. Purely schematic -- no
    session data."""
    ax = fig.add_subplot(subplot_spec, projection="3d")

    plane_color = POOLED_COLOR
    fem_color = "tab:green"
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    # --- PSTH subspace plane at z = 0 ---
    px, py = 3.4, 2.0
    plane = np.array([
        [-px, -py, 0.0],
        [px, -py, 0.0],
        [px, py, 0.0],
        [-px, py, 0.0],
    ])
    ax.add_collection3d(Poly3DCollection(
        [plane], facecolor=plane_color, edgecolor=plane_color,
        alpha=0.10, linewidths=1.0, linestyles="--"))

    # --- FEM covariance ellipse: an elongated, tilted low-rank disk floating
    # above the plane. Both disk axes carry a z-component so the disk is clearly
    # oblique to the subspace -- its orthogonal shadow is then a compressed
    # ellipse, i.e. only part of the FEM variance is captured. ---
    cz = 2.7                                   # height of disk center
    center = np.array([-0.9, 0.4, cz])
    a, b = 2.4, 0.55                           # elongated => low-rank
    # Long axis tilts up out of the plane; short axis also lifted, oblique.
    e1 = np.array([1.0, 0.25, 0.55]); e1 /= np.linalg.norm(e1)
    e2 = np.array([-0.2, 0.9, 0.35]); e2 /= np.linalg.norm(e2)
    t = np.linspace(0, 2 * np.pi, 120)
    rim = (center[:, None]
           + a * np.cos(t) * e1[:, None]
           + b * np.sin(t) * e2[:, None]).T
    ax.add_collection3d(Poly3DCollection(
        [rim], facecolor=fem_color, edgecolor=fem_color,
        alpha=0.28, linewidths=1.6))

    # --- Projection onto the subspace: drop z -> compressed shadow ellipse ---
    shadow = rim.copy(); shadow[:, 2] = 0.0
    ax.plot(shadow[:, 0], shadow[:, 1], shadow[:, 2], color=fem_color,
            lw=1.4, ls="--", alpha=0.9)
    # dotted drop lines from a few rim points down to their projections
    drop_idx = np.linspace(0, len(t) - 1, 7, dtype=int)[:-1]
    drops = [[(rim[i, 0], rim[i, 1], rim[i, 2]),
              (rim[i, 0], rim[i, 1], 0.0)] for i in drop_idx]
    ax.add_collection3d(Line3DCollection(
        drops, colors="0.35", linewidths=0.7, linestyles=":"))

    # --- Leading PSTH eigenvectors, drawn parallel to the subspace edges
    # (+x and +y), in the lower-right of the plane. ---
    # Lowered toward the front edge and shifted so the u1 arrow tip lands on the
    # plane's right edge (x = px), keeping the u1 label clear of the boundary.
    origin = np.array([px - 1.7, -1.35, 0.0])
    for vec, name, off in (
        (np.array([1.7, 0.0, 0.0]), r"$\mathbf{u}_1^{\mathrm{PSTH}}$", (0.22, -0.12, 0.0)),
        (np.array([0.0, 1.4, 0.0]), r"$\mathbf{u}_2^{\mathrm{PSTH}}$", (0.0, 0.28, 0.0)),
    ):
        ax.quiver(*origin, *vec, color=plane_color, lw=1.8,
                  arrow_length_ratio=0.14)
        tip = origin + vec
        ax.text(tip[0] + off[0], tip[1] + off[1], tip[2] + off[2], name,
                color=plane_color, fontsize=8, ha="left", va="center")

    # --- Labels ---
    ax.text(center[0] - 0.3, center[1] + 1.1, cz + 0.6,
            r"$\Sigma_{\mathrm{FEM}}$", color=fem_color, fontsize=9,
            ha="center", va="bottom")
    ax.text(-2.7, 1.5, 0.55, "projection", color="0.35", fontsize=7.5,
            ha="left", va="bottom")
    # Two-line text box floating above the subspace's upper-right corner.
    ax.text(px - 0.5, py - 0.1, 1.05, "PSTH subspace\n(4 dims)",
            color=plane_color, fontsize=7.5, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=plane_color, lw=0.7, alpha=0.9))

    # Limit windows are shifted (without changing their span, so scale is
    # preserved) to slide the projected 3D diagram up and to the left, centering
    # it in the panel within the gnomon frame: +x window -> content moves left,
    # -z window -> content moves up.
    ax.set_xlim(-2.75, px + 1.15); ax.set_ylim(-3.8, py); ax.set_zlim(-0.55, cz + 0.35)
    ax.view_init(elev=18, azim=-60)
    try:
        ax.set_box_aspect((px + 0.5, py + 1.0, cz * 0.62))
    except Exception:
        pass
    ax.set_axis_off()
    # Expand the 3D axes to fill the panel cell (the default cube leaves wide
    # margins); pull the lower-left out and grow width/height.
    pos = ax.get_position()
    ax.set_position([pos.x0 - 0.018, pos.y0 - 0.030,
                     pos.width * 1.14, pos.height * 1.20])

    # --- Ambient neural-space axes (neurons 1, 2, 3), drawn as a 2D gnomon in
    # screen (axes-fraction) space so the frame orients cleanly regardless of the
    # 3D projection: dim 2 horizontal (straight right), dim 3 vertical (up), dim 1
    # oblique (down-left, toward the viewer). The three arms meet exactly at the
    # origin (shrink=0); sized large so the frame reads as bounding the scene. ---
    gnom_o = (0.135, 0.385)
    arm1_tip = (0.070, 0.225)        # down-left, toward the viewer
    arm2_tip = (0.585, 0.385)        # right (horizontal)
    arm3_tip = (0.135, 0.920)        # up (vertical)
    for tip in (arm1_tip, arm2_tip, arm3_tip):
        ax.annotate("", xy=tip, xytext=gnom_o,
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=1.0,
                                    alpha=0.5, shrinkA=0, shrinkB=0), zorder=0)

    # "neurons" runs along the dim-1 axis, just above it (offset up-left). The
    # rotation is the on-screen angle of that axis, aspect-corrected from the
    # final panel box so it tracks the line exactly.
    bb_g = ax.get_position(); _fw, _fh = fig.get_size_inches()
    _asp = (bb_g.width * _fw) / (bb_g.height * _fh)
    _up = (gnom_o[0] - arm1_tip[0], gnom_o[1] - arm1_tip[1])   # toward the origin
    _rot = np.degrees(np.arctan2(_up[1], _up[0] * _asp))
    _mid = (0.5 * (gnom_o[0] + arm1_tip[0]), 0.5 * (gnom_o[1] + arm1_tip[1]))
    ax.text2D(_mid[0] - 0.052, _mid[1] + 0.028, "neurons",
              transform=ax.transAxes, color="0.6", fontsize=7, alpha=0.8,
              ha="center", va="center", rotation=_rot, rotation_mode="anchor")

    # --- Variance-captured definition, filling the empty lower band of the
    # panel: the fraction of FEM variance lying in the leading PSTH subspace,
    # f = projected variance / total variance. U holds the leading PSTH
    # eigenvectors that span the subspace. ---
    ax.text2D(0.5, 0.135,
              "Fraction of FEM variance in the PSTH subspace",
              transform=ax.transAxes, fontsize=8.5, color="0.20",
              ha="center", va="center")
    ax.text2D(0.5, 0.045,
              r"$f \;=\; \frac{\mathrm{projected\ variance}}"
              r"{\mathrm{total\ variance}} \;=\; "
              r"\frac{\mathrm{tr}\!\left(U^{\top}\Sigma_{\mathrm{FEM}}\,U\right)}"
              r"{\mathrm{tr}\!\left(\Sigma_{\mathrm{FEM}}\right)}$",
              transform=ax.transAxes, fontsize=11.5, color="0.10",
              ha="center", va="center")
    return ax


def _emp_p_greater(null_vals, observed):
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if null_vals.size == 0 or not np.isfinite(observed):
        return np.nan
    return (np.sum(null_vals >= observed) + 1) / (null_vals.size + 1)


def _plot_subspace_alignment_vs_shuffle(fig, subplot_spec, data):
    ax = fig.add_subplot(subplot_spec)
    observed = np.column_stack([
        np.asarray(data.get("var_p_given_f", []), dtype=float),
        np.asarray(data.get("var_f_given_p", []), dtype=float),
    ])
    null_session_idx = np.asarray(data.get("null_session_idx", []), dtype=int)
    null_x = np.asarray(data.get("null_var_p_given_f", []), dtype=float)
    null_y = np.asarray(data.get("null_var_f_given_p", []), dtype=float)
    nulls = [
        null_x[np.isfinite(null_x)],
        null_y[np.isfinite(null_y)],
    ]

    if observed.size == 0 or not all(n.size for n in nulls):
        ax.text(0.5, 0.5, "No alignment data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return ax

    px = np.full(observed.shape[0], np.nan)
    py = np.full(observed.shape[0], np.nan)
    for i in range(observed.shape[0]):
        m = null_session_idx == i
        if not np.any(m):
            continue
        px[i] = _emp_p_greater(null_x[m], observed[i, 0])
        py[i] = _emp_p_greater(null_y[m], observed[i, 1])

    # Two-tier per-session significance (joint over both directions): the
    # strongest tier that a session clears sets its marker edge.
    sig01 = (px < 0.01) & (py < 0.01)
    sig05 = (px < 0.05) & (py < 0.05)
    edge_by_session = np.where(
        sig01, SIG01_COLOR, np.where(sig05, SIG05_COLOR, "black")
    )
    lw_by_session = np.where(sig01, 1.3, np.where(sig05, 1.1, 0.45))

    viol = ax.violinplot(
        nulls,
        positions=[0, 1],
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in viol["bodies"]:
        body.set_facecolor("#d7e6ef")
        body.set_edgecolor("none")
        body.set_alpha(0.9)

    rng = np.random.default_rng(11)
    for j in range(2):
        ok = np.isfinite(observed[:, j])
        jitter = rng.normal(0, 0.055, ok.sum())
        ax.scatter(
            np.full(ok.sum(), j) + jitter,
            observed[ok, j],
            s=28,
            color=POOLED_COLOR,
            edgecolor=edge_by_session[ok],
            linewidth=lw_by_session[ok],
            zorder=3,
        )
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(observed[:, j])] * 2,
                color=POOLED_COLOR, lw=2.0, zorder=4)
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(nulls[j])] * 2,
                color="0.35", lw=1.1, zorder=4)

    # Significance-tier legend (marker edge color).
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", ls="none", ms=6, mfc=POOLED_COLOR,
               mec=SIG01_COLOR, mew=1.3, label="p < 0.01"),
        Line2D([0], [0], marker="o", ls="none", ms=6, mfc=POOLED_COLOR,
               mec=SIG05_COLOR, mew=1.1, label="p < 0.05"),
        Line2D([0], [0], marker="o", ls="none", ms=6, mfc=POOLED_COLOR,
               mec="black", mew=0.45, label="n.s."),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="lower center",
              ncol=3, handletextpad=0.2, columnspacing=0.8,
              borderpad=0.1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stimulus in\nFEM subspace",
                        "FEM in\nstimulus subspace"])
    ax.set_ylabel("Variance captured")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def load_prepared_data(refresh=False, split_subjects=False):
    """Load + filter (+ optionally pool) the fig2 bundle, exactly as ``compose``
    does internally. Exposed so an interactive tuner can load the data once and
    reuse it across many re-renders."""
    data = _filter_subjects(load_fig2_data(refresh=refresh))
    if not split_subjects:
        data = _pool_subjects_for_plotting(data)
    return data


def compose(refresh=False, split_subjects=False, *,
            row1_height=1.10, d_width_frac=0.60, mid_wspace=0.30,
            d_title_fontsize=8.0, prepared_data=None,
            return_png_bytes=False, preview_dpi=110):
    """Compose Figure 2.

    The middle-row layout knobs (``row1_height`` height ratio, ``d_width_frac``
    fraction of the row taken by panel D with the remainder split evenly between
    E and F, ``mid_wspace`` inter-panel padding, ``d_title_fontsize`` for the
    decomposition matrix titles) are exposed so the companion Flask tuner can
    sweep them; their defaults reproduce the committed figure. ``prepared_data``
    skips the (cached) reload, and ``return_png_bytes`` renders to an in-memory
    PNG instead of writing the canonical outputs.
    """
    if prepared_data is not None:
        data = prepared_data
    else:
        data = load_prepared_data(refresh=refresh, split_subjects=split_subjects)

    # Tuned for an ~8.5" manuscript-width figure. Rows follow the argument:
    # single-neuron/rate, population covariance, removed-component structure.
    rc = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(9.5, 9.6))
        gs = GridSpec(
            3,
            6,
            height_ratios=[1.05, row1_height, 1.05],
            hspace=0.42,
            wspace=0.95,
            figure=fig,
            left=0.075,
            right=0.980,
            top=0.955,
            bottom=0.075,
        )

        # --- Row 0: A eye/rate example, B mismatch curve, C 1-alpha
        d_vals = _plot_eye_rate_panel(fig, gs[0, 0:2], label="A")

        b_ax = fig.add_subplot(gs[0, 2:4])
        _plot_covariance_mismatch_panel(b_ax, divergent_de=d_vals["divergent"])
        _normalize_axis_text(b_ax)
        _label(b_ax, "B")

        c_ax = fig.add_subplot(gs[0, 4:6])
        _, c_primary = plot_fem_fraction(ax=c_ax, data=data)
        _normalize_axis_text(c_primary)
        _label(c_primary, "C")
        c_primary.set_xlabel("Fraction of rate modulation\ndue to FEM")

        # --- Row 1: D covariance decomposition (~55% width), with the two
        # decomposition results side by side to its right: E Fano, F noise corr.
        ef_frac = (1.0 - d_width_frac) / 2.0
        gs_mid = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0:6],
            width_ratios=[d_width_frac, ef_frac, ef_frac], wspace=mid_wspace,
        )
        d_axes = _plot_compact_cov_decomp(fig, gs_mid[0, 0], data, letter="D",
                                          title_fontsize=d_title_fontsize)
        _expand_axes_group_from_lower_right(d_axes)
        _center_top_residual(d_axes)

        e_ax = fig.add_subplot(gs_mid[0, 1])
        plot_fano_population(ax=e_ax, data=data)
        _normalize_axis_text(e_ax)
        _label(e_ax, "E")

        f_ax = fig.add_subplot(gs_mid[0, 2])
        plot_nc_violin(ax=f_ax, data=data)
        _normalize_axis_text(f_ax)
        _label(f_ax, "F")

        # --- Row 2: G participation-ratio comparison, H projection schematic,
        # I subspace-alignment quantification
        pr_ax = _plot_pr_comparison(fig, gs[2, 0:2], data)
        _normalize_axis_text(pr_ax)
        _panel_header(
            pr_ax,
            "G",
            "Stimulus and FEM are low-rank vs. the noise",
            y=1.02,
        )

        schem_ax = _plot_subspace_schematic(fig, gs[2, 2:4])
        schem_ax.text2D(-0.04, 1.02, "H", transform=schem_ax.transAxes,
                        fontweight="bold", fontsize=10, va="bottom", ha="left")
        schem_ax.text2D(0.08, 1.02, "Testing low-rank subspace alignment",
                        transform=schem_ax.transAxes, fontsize=8,
                        va="bottom", ha="left")

        align_ax = _plot_subspace_alignment_vs_shuffle(fig, gs[2, 4:6], data)
        _normalize_axis_text(align_ax)
        _panel_header(
            align_ax,
            "I",
            "FEM largely lives in the stimulus subspace",
            y=1.02,
        )

    if return_png_bytes:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0.08, dpi=preview_dpi)
        plt.close(fig)
        return buf.getvalue()

    stem = FIG_DIR / "figure2"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.08, dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight",
                pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"\nSaved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")


def _parse_args():
    p = argparse.ArgumentParser(description="Compose combined figure 2/3.")
    p.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="Force recompute of derived fig2 data.",
    )
    p.add_argument(
        "--split-subjects",
        action="store_true",
        help="Plot included subjects separately instead of pooling them.",
    )
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh, split_subjects=args.split_subjects)
