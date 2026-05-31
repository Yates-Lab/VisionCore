"""
Figure 3 panel G: subspace alignment scatter — PSTH variance captured by
the FEM subspace (X) vs FEM variance captured by the PSTH subspace (Y).

Per-subject shuffle clouds (eye-shuffled Intercepts → shuffled FEM subspace,
same PSTH subspace) provide chance-alignment baselines. Each animal is
treated as an independent replicate of the experiment.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


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
    null_x = np.asarray(data.get("null_var_p_given_f", []), dtype=float)
    null_y = np.asarray(data.get("null_var_f_given_p", []), dtype=float)

    annotations = []

    for subj in sorted(set(sub_subjects.tolist())):
        color = SUBJECT_COLORS.get(subj, "gray")
        s_mask = sub_subjects == subj
        x_real = var_p_given_f[s_mask]
        y_real = var_f_given_p[s_mask]

        # Per-subject shuffle cloud
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

                # Empirical p-values: observed = mean of this subject's real
                # sessions, null = per-shuffle pooled across this subject's
                # sessions (treating each shuffle draw as an independent
                # chance realization for that animal).
                if np.isfinite(x_real).any():
                    mx = np.nanmean(x_real)
                    my = np.nanmean(y_real)
                    px = _emp_p_greater(xs, mx)
                    py = _emp_p_greater(ys, my)
                    annotations.append((subj, mx, my, px, py, color))

        ax.scatter(x_real, y_real, c=color, s=50,
                   edgecolors="black", linewidths=0.6,
                   label=subj, zorder=5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("X: PSTH var in FEM subspace")
    ax.set_ylabel("Y: FEM var in PSTH subspace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Per-subject p-value annotation block
    if annotations:
        lines = []
        for subj, mx, my, px, py, _ in annotations:
            lines.append(
                f"{subj}: X={mx:.2f} (p={px:.3f}), Y={my:.2f} (p={py:.3f})"
            )
        ax.text(0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.8))

    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_g()
    fig.tight_layout()
    standalone_save(fig, "panel_g_subspace_alignment")
