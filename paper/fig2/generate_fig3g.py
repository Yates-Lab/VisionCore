"""
Figure 3 panel G: subspace alignment scatter — PSTH variance captured by
the FEM subspace (X) vs FEM variance captured by the PSTH subspace (Y).

Per-subject shuffle clouds (eye-shuffled Intercepts → shuffled FEM subspace,
same PSTH subspace) are drawn as 2D hexbin density per animal. Real
sessions are overlaid as colored markers; sessions whose observed (X, Y)
are jointly significant against that session's own shuffle cloud
(one-sided p<α for both X and Y) are outlined in red.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


# Joint significance against each session's own shuffle cloud.
SIG_ALPHA = 0.01
EDGE_NS = ("black", 0.6)
EDGE_SIG = ("crimson", 1.5)


def _emp_p_greater(null_vals, observed):
    """One-sided empirical p: P(null >= observed). Adds +1 smoothing."""
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if null_vals.size == 0 or not np.isfinite(observed):
        return np.nan
    return (np.sum(null_vals >= observed) + 1) / (null_vals.size + 1)


def plot_panel_g(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    sub_subjects = np.asarray(data["sub_subjects"])
    var_p_given_f = np.asarray(data["var_p_given_f"], dtype=float)
    var_f_given_p = np.asarray(data["var_f_given_p"], dtype=float)

    null_subjects = np.asarray(data.get("null_subjects", []))
    null_session_idx = np.asarray(data.get("null_session_idx", []), dtype=int)
    null_x = np.asarray(data.get("null_var_p_given_f", []), dtype=float)
    null_y = np.asarray(data.get("null_var_f_given_p", []), dtype=float)

    # Per-session one-sided empirical p-values vs that session's own cloud.
    n_sessions = len(sub_subjects)
    px_sess = np.full(n_sessions, np.nan)
    py_sess = np.full(n_sessions, np.nan)
    if null_session_idx.size:
        for i in range(n_sessions):
            m = null_session_idx == i
            if not m.any():
                continue
            px_sess[i] = _emp_p_greater(null_x[m], var_p_given_f[i])
            py_sess[i] = _emp_p_greater(null_y[m], var_f_given_p[i])
    sig = (px_sess < SIG_ALPHA) & (py_sess < SIG_ALPHA)

    # Per-subject shuffle scatter clouds + mean ×.
    for subj in sorted(set(sub_subjects.tolist())):
        color = SUBJECT_COLORS.get(subj, "gray")
        if null_subjects.size:
            n_mask = null_subjects == subj
            xs = null_x[n_mask]
            ys = null_y[n_mask]
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size:
                ax.scatter(xs, ys, s=6, alpha=0.12, color=color,
                           edgecolors="none", zorder=1)
                ax.plot(np.mean(xs), np.mean(ys), marker="x",
                        color=color, ms=8, mew=2, zorder=3)

    # Real sessions: red outline if jointly significant, else black.
    # NS first so sig markers sit on top.
    for subj in sorted(set(sub_subjects.tolist())):
        color = SUBJECT_COLORS.get(subj, "gray")
        s_mask = sub_subjects == subj
        x = var_p_given_f[s_mask]
        y = var_f_given_p[s_mask]
        s = sig[s_mask]
        if (~s).any():
            ax.scatter(x[~s], y[~s], c=color, s=55,
                       edgecolors=EDGE_NS[0], linewidths=EDGE_NS[1],
                       zorder=5)
        if s.any():
            ax.scatter(x[s], y[s], c=color, s=55,
                       edgecolors=EDGE_SIG[0], linewidths=EDGE_SIG[1],
                       zorder=6)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, zorder=0)
    ax.set_xlabel("X: PSTH var in FEM subspace")
    ax.set_ylabel("Y: FEM var in PSTH subspace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    from matplotlib.lines import Line2D
    sig_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="lightgray",
               markeredgecolor=EDGE_SIG[0], markeredgewidth=EDGE_SIG[1],
               markersize=8, label=f"p<{SIG_ALPHA:g} (both)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="lightgray",
               markeredgecolor=EDGE_NS[0], markeredgewidth=EDGE_NS[1],
               markersize=8, label="n.s."),
    ]
    ax.legend(handles=sig_handles, frameon=False, fontsize=7,
              loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_g()
    fig.tight_layout()
    standalone_save(fig, "panel_g_subspace_alignment")
