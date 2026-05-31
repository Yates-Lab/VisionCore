"""
Figure 3 panel E: cumulative eigenvariance for PSTH vs FEM covariances.

Each session's spectrum is normalized to its own total (so the curve is
about shape / dimensionality, not absolute amount of variance). Per-session
traces are drawn faintly; the bold line is the per-subject median. A
vertical reference at PR = 2 marks the eye's translational degrees of
freedom.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def _cumulative(spec, max_dims):
    """Cumulative fraction of own variance, padded to length max_dims."""
    s = np.asarray(spec, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0 or s.sum() <= 0:
        return np.full(max_dims, np.nan)
    csum = np.cumsum(s / s.sum())
    out = np.full(max_dims, np.nan)
    L = min(csum.size, max_dims)
    out[:L] = csum[:L]
    # Sessions with fewer than max_dims eigenvalues already saturate at 1
    # by definition; fill the tail so they don't visually dip back to NaN.
    if L < max_dims and np.isfinite(out[L - 1]):
        out[L:] = out[L - 1]
    return out


def plot_panel_e(ax=None, refresh=False, data=None, max_dims=10):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    sub_subjects = np.asarray(data["sub_subjects"])
    spectra_psth = data["spectra_psth"]
    spectra_fem = data["spectra_fem"]

    dims = np.arange(1, max_dims + 1)

    present_subjects = []
    for subj in SUBJECTS:
        s_mask = sub_subjects == subj
        if not s_mask.any():
            continue
        present_subjects.append(subj)
        color = SUBJECT_COLORS[subj]
        for spec_list, ls in [(spectra_psth, "-"), (spectra_fem, "--")]:
            specs = [s for s, m in zip(spec_list, s_mask) if m]
            if not specs:
                continue
            A = np.vstack([_cumulative(s, max_dims) for s in specs])
            for row in A:
                ax.plot(dims, row, color=color, ls=ls, alpha=0.12, lw=1)
            med = np.nanmedian(A, axis=0)
            ax.plot(dims, med, color=color, ls=ls, lw=2,
                    marker="o", markersize=4)

    ax.set_xlim(1, max_dims)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Cumulative frac. of own variance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Single legend: subjects by color, conditions by linestyle.
    handles = [Line2D([0], [0], color=SUBJECT_COLORS[s], lw=2, label=s)
               for s in present_subjects]
    handles += [
        Line2D([0], [0], color="k", lw=2, ls="-", label="PSTH"),
        Line2D([0], [0], color="k", lw=2, ls="--", label="FEM"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_e()
    fig.tight_layout()
    standalone_save(fig, "panel_e_eigenspectra_cumulative")
