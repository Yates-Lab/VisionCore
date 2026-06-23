"""
Compose a combined Figure 2/3 main figure:

    A: compact eye/rate matching example
    B: FEM fraction of rate modulation
    C: Fano factor before/after FEM correction
    D: compact covariance decomposition
    E: mean pairwise Fisher-z noise correlations before/after correction
    F: PSTH/FEM participation ratio by session
    G: PSTH/FEM subspace alignment

Usage:
    uv run declan/fig2/generate_figure2_3_combined.py
    uv run declan/fig2/generate_figure2_3_combined.py --split-subjects
    uv run declan/fig2/generate_figure2_3_combined.py --refresh
"""
import argparse
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Polygon

from VisionCore.covariance import project_to_psd
from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data, _compute_fano_stats, _compute_nc_stats
from generate_fig2a import (
    UNIT_ORIG,
    _compute_uniform_bins,
    _load_unit_payload,
)
from generate_fig2c import plot_panel_c as plot_fem_fraction
from generate_fig2e import plot_panel_e as plot_fano
from generate_fig3c import plot_panel_c as plot_noise_corr

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0
OMIT_SUBJECTS = {"Luke"}
POOLED_SUBJECT = "Pooled"
POOLED_COLOR = "tab:blue"
SIG_ALPHA = 0.01


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
                        "shuff_rho_subject"):
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


def _plot_covariance_mismatch_panel(fig, subplot_spec):
    ax = fig.add_subplot(subplot_spec)
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

    ax.axvspan(0.0, 0.05, color="0.82", zorder=-2)
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
    ax.text(0.58, var_psth + 0.012, "Stimulus variance",
            fontsize=7.5, ha="left", va="bottom")
    ax.annotate(
        "",
        xy=(0.02, first_y),
        xytext=(0.02, var_psth),
        arrowprops=dict(arrowstyle="<->", color=POOLED_COLOR, lw=2.0),
    )
    ax.text(0.055, 0.50 * (first_y + var_psth), "FEM\nvariance",
            color="k", fontsize=7.5, ha="left", va="center")
    ax.text(0.13, -0.32, "Variance decreases as\neye trajectories diverge",
            fontsize=7.5, ha="left", va="center")

    ax.set_xlim(-0.06, 1.03)
    ax.set_ylim(-0.40, 0.52)
    ax.set_xlabel("Eye-trajectory mismatch, Δe (°)")
    ax.set_ylabel("Cross-trial rate variance (spk²)")
    ax.set_yticks(np.arange(-0.4, 0.6, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _label(ax, "A")
    return ax


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


def _shift_axes_down(axes, dy=0.018):
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - dy, pos.width, pos.height])


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


def _plot_compact_cov_decomp(fig, subplot_spec, data, letter="D"):
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
        ax.set_title(title, fontsize=8, pad=3)
        _style_matrix_axis(ax)

    for ax, mat, title in zip(
        bot_axes,
        [ctotal, cpsth, cfem, cint],
        bot_titles,
    ):
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  norm=norm, aspect="equal")
        ax.set_xlabel(title, fontsize=8, labelpad=4)
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
        "Independent variance\nalong diagonal",
        xy=(0.46, 0.56),
        xycoords=bot_axes[3].transAxes,
        xytext=(0.72, 1.24),
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


def _add_condition_legend(ax, loc="upper left"):
    handles = [
        Line2D([0], [0], color="0.2", lw=1.4, ls="--",
               marker="o", markerfacecolor="white", markeredgecolor="0.2",
               markersize=4, label="uncorrected"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="-",
               marker="o", markerfacecolor="0.2", markeredgecolor="0.2",
               markersize=4, label="FEM-corrected"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc=loc,
              handlelength=1.8, borderpad=0.1, labelspacing=0.25)


def _plot_noise_corr_panel(fig, subplot_spec, data):
    ax_main = fig.add_subplot(subplot_spec)
    _, primary = plot_noise_corr(ax=ax_main, data=data)
    _normalize_axis_text(primary)
    _label(primary, "E")
    primary.yaxis.set_label_coords(-0.15, 0.5)
    _add_condition_legend(primary, loc="upper left")
    primary.text(
        0.12,
        1.08,
        "FEM covariance makes up\nmost of the ‘noise’ correlations",
        transform=primary.transAxes,
        fontsize=7.2,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    return primary


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
    sig = (px < SIG_ALPHA) & (py < SIG_ALPHA)

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
        edgecolors = np.where(sig[ok], "crimson", "black")
        linewidths = np.where(sig[ok], 1.2, 0.45)
        ax.scatter(
            np.full(ok.sum(), j) + jitter,
            observed[ok, j],
            s=28,
            color=POOLED_COLOR,
            edgecolor=edgecolors,
            linewidth=linewidths,
            zorder=3,
        )
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(observed[:, j])] * 2,
                color=POOLED_COLOR, lw=2.0, zorder=4)
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(nulls[j])] * 2,
                color="0.35", lw=1.1, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stimulus in\nFEM subspace",
                        "FEM in\nstimulus subspace"])
    ax.set_ylabel("Variance captured")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def compose(refresh=False, split_subjects=False):
    data = _filter_subjects(load_fig2_data(refresh=refresh))
    if not split_subjects:
        data = _pool_subjects_for_plotting(data)

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
        fig = plt.figure(figsize=(8.5, 9.2))
        gs = GridSpec(
            3,
            6,
            height_ratios=[0.95, 1.38, 1.05],
            hspace=0.38,
            wspace=0.95,
            figure=fig,
            left=0.090,
            right=0.980,
            top=0.965,
            bottom=0.080,
        )

        _plot_covariance_mismatch_panel(fig, gs[0, 0:2])

        panel_specs = [
            ("B", plot_fem_fraction, gs[0, 2:4]),
            ("C", plot_fano, gs[0, 4:6]),
        ]
        d_axes = _plot_compact_cov_decomp(fig, gs[1, 0:4],
                                          data, letter="D")
        _expand_axes_group_from_lower_right(d_axes)
        e_ax = _plot_noise_corr_panel(fig, gs[1, 4:6], data)
        _shift_axes_down([e_ax])

        pr_ax = _plot_participation_ratio_bars(
            fig, gs[2, 0:4], data, split_subjects=split_subjects
        )
        _normalize_axis_text(pr_ax)
        _panel_header(
            pr_ax,
            "F",
            "FEM covariance is compact across sessions",
            y=1.02,
        )

        align_ax = _plot_subspace_alignment_vs_shuffle(fig, gs[2, 4:6], data)
        _normalize_axis_text(align_ax)
        _panel_header(
            align_ax,
            "G",
            "and largely lives in the stimulus subspace",
            y=1.02,
        )

        for letter, plot_fn, spec in panel_specs:
            ax = fig.add_subplot(spec)
            _, primary_ax = plot_fn(ax=ax, data=data)
            _normalize_axis_text(primary_ax)
            if letter == "B":
                _label(primary_ax, letter)
                primary_ax.set_xlabel("Fraction of rate modulation\ndue to FEM")
            elif letter == "C":
                _label(primary_ax, letter)
                _add_condition_legend(primary_ax, loc="upper left")
            else:
                _label(primary_ax, letter)

    stem = FIG_DIR / "fig2_3_combined"
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
