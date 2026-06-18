r"""Figure for writeup §4.6: trajectory-mode estimator with pooled-per-bin KDE.

Two pieces:

  * Panels A-D: a one-session snapshot showing what the two KDEs estimate.
    Panel A draws ~30 trajectories (centroid + per-bin drift) overlaid; panels
    B and C are the pooled-per-bin KDEs of (B) all per-bin positions
    (p_marg) and (C) close-pair midpoint-trajectory per-bin positions
    (p_cp,marg). Panel D is the ratio p_cp,marg / p_marg -- the §4.1 mechanism
    visualised in the multi-bin setting (the ratio peaks at the centre where
    close pairs over-represent the high-density region).

  * Panel E: a sigma_drift sweep that validates the estimator's recovery
    against the per-bin marginal truth (the ground_truth computed at
    sigma_traj = sqrt(sigma_eye^2 + sigma_drift^2)). The flat limit
    (sigma_drift = 0) is exact; the bias grows smoothly with sigma_drift.

Run from this folder:  uv run python fig_trajectory.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from synthetic import make_trajectory_session
from estimators import decompose_trajectory
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

SIG = 0.15
GRID = np.linspace(-0.5, 0.5, 80)


def _traj_threshold(sigma_drift, base=0.05):
    return float(np.sqrt(base ** 2 + 4.0 * float(sigma_drift) ** 2))


def _kde_grid(points):
    kde = gaussian_kde(np.asarray(points).reshape(-1, 2).T)
    X, Y = np.meshgrid(GRID, GRID)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def _close_pair_midpoints(sess, threshold):
    """Reproduce the estimator's close-pair filter to extract midpoint trajectories
    (per-bin) for the p_cp,marg KDE snapshot. Mirrors _rms_traj_close_pairs."""
    Tr = sess["trajectories"]; T_idx = sess["T_idx"]
    N, t_window, _ = Tr.shape
    flat = Tr.reshape(N, -1)
    inv = 1.0 / np.sqrt(float(t_window))
    mids = []
    for t in np.unique(T_idx):
        ix = np.where(T_idx == t)[0]
        if len(ix) < 2:
            continue
        F = flat[ix]
        D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1) * inv
        i, j = np.triu_indices(len(ix), k=1)
        close = D[i, j] < threshold
        if not close.any():
            continue
        gi, gj = ix[i[close]], ix[j[close]]
        mids.append(0.5 * (Tr[gi] + Tr[gj]))
    if not mids:
        return np.zeros((0, t_window, 2))
    return np.concatenate(mids, axis=0)


def panel_AtoD(ax_A, ax_B, ax_C, ax_D, sess_demo, threshold):
    Tr = sess_demo["trajectories"]
    # A: a handful of trajectory traces (centroid scatter + per-bin lines)
    ax_A.set_title("A  example trajectories")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Tr), size=40, replace=False)
    for i in pick:
        ax_A.plot(Tr[i, :, 0], Tr[i, :, 1], color="#999", lw=0.4, alpha=0.4)
        ax_A.plot(Tr[i, 0, 0], Tr[i, 0, 1], "o", ms=1.6, color=C_FULL, alpha=0.5)
    ax_A.set_xlabel("x (deg)"); ax_A.set_ylabel("y (deg)")
    ax_A.set_xlim(-0.5, 0.5); ax_A.set_ylim(-0.5, 0.5)
    ax_A.set_aspect("equal")
    ax_A.text(0.04, 0.92, f"$\\sigma_{{\\rm drift}}={sess_demo['sigma_drift']:.2f}$",
              transform=ax_A.transAxes, fontsize=8)

    # B: p_marg from all per-bin positions
    X, Y, Zm = _kde_grid(Tr.reshape(-1, 2))
    ax_B.set_title(r"B  $\hat p_{\rm marg}(e)$ (pooled per-bin)")
    ax_B.imshow(Zm, extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]],
                origin="lower", cmap="Blues", aspect="equal")
    ax_B.set_xlabel("x (deg)"); ax_B.set_ylabel("y (deg)")

    # C: p_cp,marg from close-pair midpoint-trajectory positions
    mid = _close_pair_midpoints(sess_demo, threshold)
    if len(mid) == 0:
        ax_C.set_title("C  no close pairs"); ax_C.axis("off")
        Zcp = None
    else:
        Xc, Yc, Zcp = _kde_grid(mid.reshape(-1, 2))
        ax_C.set_title(r"C  $\hat p_{cp,marg}(e)$ (close-pair midpoints)")
        ax_C.imshow(Zcp, extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]],
                    origin="lower", cmap="Reds", aspect="equal")
        ax_C.set_xlabel("x (deg)"); ax_C.set_ylabel("y (deg)")

    # D: ratio p_cp,marg / p_marg  -- the §4.1 mechanism in 2-D
    if Zcp is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            R = Zcp / np.clip(Zm, 1e-6, None)
        # mask out where p_marg is tiny so the ratio doesn't explode visually
        R[Zm < 0.1 * Zm.max()] = np.nan
        ax_D.set_title(r"D  $\hat p_{cp,marg}/\hat p_{\rm marg}$ (centroid kernel)")
        im = ax_D.imshow(R, extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]],
                         origin="lower", cmap="magma", aspect="equal")
        plt.colorbar(im, ax=ax_D, fraction=0.046, pad=0.04, label="ratio")
        ax_D.set_xlabel("x (deg)"); ax_D.set_ylabel("y (deg)")
    else:
        ax_D.axis("off")


def panel_E(ax, kind="flat", sigma_drift_values=(0.0, 0.015, 0.03, 0.045, 0.06,
                                                 0.09, 0.12, 0.15),
            n_seeds=4, n_per=30, n_T=40, t_window=5):
    """sigma_drift sweep of 1-alpha for naive, full, central vs the per-bin
    marginal truth."""
    sd_arr = np.array(sigma_drift_values, float)
    out = {tgt: np.full((len(sd_arr), n_seeds), np.nan) for tgt in
           ("naive", "full", "central")}
    truth_p = np.zeros(len(sd_arr))
    truth_p2 = np.zeros(len(sd_arr))
    for k, sd in enumerate(sd_arr):
        thr = _traj_threshold(sd)
        for s in range(n_seeds):
            sess = make_trajectory_session(
                [kind], n_samples_per_time_bin=n_per, n_time_bins=n_T,
                t_window=t_window, sigma_eye=SIG, sigma_drift=sd, seed=s)
            for tgt in out:
                d = decompose_trajectory(sess["counts"], sess["trajectories"],
                                         sess["T_idx"], target=tgt,
                                         threshold=thr)
                out[tgt][k, s] = d["one_minus_alpha"][0]
            if s == 0:
                truth_p[k] = sess["traj_truth"][0]["p"]["one_minus_alpha"]
                truth_p2[k] = sess["traj_truth"][0]["p2"]["one_minus_alpha"]
        print(f"  sigma_drift={sd:.3f}  full={np.nanmean(out['full'][k]):.3f}  "
              f"central={np.nanmean(out['central'][k]):.3f}  "
              f"naive={np.nanmean(out['naive'][k]):.3f}  truth_p={truth_p[k]:.3f}")

    for tgt, color, marker, lbl in [
        ("naive", C_TRUTH, "x", "naive"),
        ("full", C_FULL, "o", "Direction 1 (full, $p$)"),
        ("central", C_CLOSE, "s", "Direction 2 (central, $p^2$)"),
    ]:
        m = np.nanmean(out[tgt], axis=1)
        sd_ = np.nanstd(out[tgt], axis=1)
        ax.errorbar(sd_arr / SIG, m, yerr=sd_, fmt=marker + "-", color=color,
                    ms=4, capsize=2, lw=0.9, label=lbl)
    ax.plot(sd_arr / SIG, truth_p, color=C_FULL, lw=0.6, ls=":",
            label=r"truth $1-\alpha^p$")
    ax.plot(sd_arr / SIG, truth_p2, color=C_CLOSE, lw=0.6, ls=":",
            label=r"truth $1-\alpha^{p^2}$")
    ax.set_xlabel(r"$\sigma_{\rm drift}/\sigma_e$"); ax.set_ylabel(r"$1-\alpha$")
    ax.set_title(f"E  $\\sigma_{{\\rm drift}}$ sweep ({kind} mask)")
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    ax.set_ylim(0, 1)


def main():
    configure()
    # demo session for panels A-D: moderate drift to make the broadening of
    # p_marg vs p_cp,marg visually obvious.
    sess_demo = make_trajectory_session(
        ["flat"], n_samples_per_time_bin=40, n_time_bins=40,
        t_window=5, sigma_eye=SIG, sigma_drift=SIG / 4.0, seed=0)
    thr_demo = _traj_threshold(SIG / 4.0)

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.1], hspace=0.4, wspace=0.45)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[0, 3])
    ax_E = fig.add_subplot(gs[1, :])

    panel_AtoD(ax_A, ax_B, ax_C, ax_D, sess_demo, thr_demo)
    print("running sigma_drift sweep ...")
    panel_E(ax_E)

    save(fig, "fig_trajectory.png")


if __name__ == "__main__":
    main()
