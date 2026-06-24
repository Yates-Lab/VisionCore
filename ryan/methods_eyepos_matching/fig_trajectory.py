r"""Figure for writeup §4.4: trajectory-mode estimator validated on REAL
fixational eye statistics.

The §4.4 estimator reduces each multi-bin eye trajectory to a single
representative 2-D point (geometric median) and then applies the §4.2 single-bin
construction (one KDE p_hat on the representative points; p^2 implied). This
figure grounds the flat-trajectory approximation in real data rather than a
synthetic drift sweep: real fixational eye trajectories (cache/aligned_sessions.pkl)
carry a known synthetic rate field on top, so the truth is known while the eye
statistics are real.

  * Panel A: real within-window flatness -- the distribution of within-window
    RMS deviation from the representative point, relative to the across-fixation
    spread sigma_e. Real trajectories are nearly flat (drift << sigma_e), with a
    microsaccade tail. Colour-matched arrows mark where the two example windows
    of B fall.

  * Panel B: the two example windows (a pure-drift fixation and a
    microsaccade-containing one), justifying the geometric-median reduction: the
    geometric median (star) tracks the majority of the window's per-bin points
    (the fixation cluster), while the centroid (cross) is dragged toward the
    microsaccade excursion.

  * Panel C: recovery. An ell/sigma_e sweep on real eyes (flat + central masks):
    Direction 1 (full) recovers the empirical 1-alpha^p over the real eye
    distribution; Direction 2 (central) recovers 1-alpha^{p^2} (the close-pair
    density of the real distribution). Truths (lines) are computed empirically
    over the real eye sample -- no Gaussian assumption, no density fit; markers
    are the matched-estimator means ± sd.

Run from this folder:  uv run python fig_trajectory.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import data_loading as dl
from synthetic import profile_M, _draw_field_at, _default_sigma_M
from estimators import decompose_trajectory, _geometric_median
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH, C_OK

# Field / mask defaults (tau cancels in 1-alpha; mu_0 keeps rates positive).
TAU, MU0 = 1.0, 6.0
T_WINDOW = 12            # ~100 ms at 120 Hz -- a representative analysis window
THRESHOLD = 0.05         # RMS-trajectory close-pair threshold (per-bin units)
MAX_WIN = 6000           # subsample budget for windows
ELL_OVER_SIG = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
MASKS = ("flat", "central", "eccentric")


# ---------------------------------------------------------------------------
# Real eye-trajectory windows
# ---------------------------------------------------------------------------

def build_real_windows(t_window=T_WINDOW, stride=None, max_win=MAX_WIN,
                       per_bin=40, seed=0):
    """Collect valid eye-trajectory windows across all sessions and assign them
    to synthetic analysis bins.

    Each window is t_window contiguous fully-valid analysis bins of the
    fixation-aligned eye trace. Windows are pooled across all sessions to form
    the real fixational eye-position distribution, then partitioned at random
    into analysis bins of ~``per_bin`` windows each. A random bin therefore
    samples the *full* real distribution, so windows sharing a bin (and a
    synthetic field draw, as in make_trajectory_session) define a clean
    PSTH/close-pair structure whose population target is exactly the global
    per-bin marginal -- no per-session heterogeneity confound. Returns
    (trajectories (N, t_window, 2), T_idx (N,)).
    """
    stride = t_window if stride is None else stride
    sessions = dl.load_cache()
    traj = []
    for s in sessions:
        eye = np.asarray(s["eyepos"], float)            # (n_tr, 120, 2)
        vm = np.asarray(s["valid_mask"], bool)          # (n_tr, 120)
        n_tr, n_bin = vm.shape
        for tr in range(n_tr):
            v = vm[tr]
            start = 0
            while start + t_window <= n_bin:
                sl = slice(start, start + t_window)
                if v[sl].all() and np.isfinite(eye[tr, sl]).all():
                    traj.append(eye[tr, sl, :])
                start += stride
    traj = np.asarray(traj, float)                      # (M, t_window, 2)
    rng = np.random.default_rng(seed)
    if len(traj) > max_win:
        traj = traj[rng.choice(len(traj), max_win, replace=False)]
    perm = rng.permutation(len(traj))
    traj = traj[perm]
    tidx = np.arange(len(traj)) // per_bin              # random bins of ~per_bin
    return traj, tidx


def synth_counts(traj, tidx, kind, sigma_eye, ell, sigma_M, seed=0):
    """Synthetic single-cell window-summed counts on GIVEN real trajectories.

    r(T,t,e) = mu_0 + M(e) * s_{T,t}(e); one field per (T_idx, offset) shared
    across the windows in that analysis bin (independent across bins/offsets),
    summed over offsets. Mirrors make_trajectory_session with alpha == 1.
    """
    N, t_window, _ = traj.shape
    rng = np.random.default_rng(seed)
    rate = np.empty((N, t_window))
    for T in np.unique(tidx):
        ix = np.where(tidx == T)[0]
        for t in range(t_window):
            eyes = traj[ix, t, :]
            s = _draw_field_at(eyes, ell, TAU, rng, 1)[:, 0]       # (len(ix),)
            M = profile_M(eyes, kind, sigma_eye, sigma_M)
            rate[ix, t] = MU0 + M * s
    rate = np.clip(rate, 1e-6, None)
    return rate.sum(axis=1)[:, None]                                # (N, 1)


# ---------------------------------------------------------------------------
# Empirical (non-parametric) ground truth over the real eye distribution
# ---------------------------------------------------------------------------

def empirical_one_minus_alpha(points, kind, sigma_eye, ell, sigma_M,
                              w=None, max_pts=2500, seed=0):
    """1-alpha for the synthetic field over an arbitrary eye distribution given
    as a point set (optionally weighted). Uses the field's analytic moments:

        Var_total ∝ E_D[M^2],   Var_psth ∝ ∬ M(e1)M(e2)K(e1-e2)D(e1)D(e2)

    so 1-alpha = 1 - I / (tau^2 E_D[M^2]) with D the empirical measure of
    `points`. No Gaussian assumption; tau cancels.
    """
    pts = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    if len(pts) > max_pts:
        idx = rng.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        w = None if w is None else np.asarray(w)[idx]
    M = profile_M(pts, kind, sigma_eye, sigma_M)
    w = np.ones(len(pts)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    EM2 = float((w * M ** 2).sum())
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    K = TAU ** 2 * np.exp(-d2 / (2.0 * ell ** 2))
    I = float(w @ (M[:, None] * K * M[None, :]) @ w)
    return 1.0 - I / (TAU ** 2 * EM2)


def close_pair_midpoints(rho, eps, max_n=1500, max_pairs=5000, seed=0):
    """Midpoints of representative-point pairs within eps -- a sample of the
    close-pair density p^2 of the real eye distribution (no KDE)."""
    rng = np.random.default_rng(seed)
    r = rho
    if len(r) > max_n:
        r = r[rng.choice(len(r), max_n, replace=False)]
    d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    i, j = np.triu_indices(len(r), 1)
    close = d[i, j] < eps
    mids = 0.5 * (r[i[close]] + r[j[close]])
    if len(mids) > max_pairs:
        mids = mids[rng.choice(len(mids), max_pairs, replace=False)]
    return mids


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_flatness(ax, ax_ex, traj, sigma_eye):
    """A: within-window RMS drift / sigma_e (with arrows marking the two example
    windows of B); B: the example windows themselves, justifying the
    geometric-median reduction (it tracks the majority of the window's points)."""
    rho = _geometric_median(traj)                                  # (N, 2)
    dev = np.sqrt(((traj - rho[:, None, :]) ** 2).sum(-1).mean(1))  # (N,) per window
    ratio = dev / sigma_eye
    med = float(np.median(ratio))
    counts, bins, _ = ax.hist(ratio, bins=np.linspace(0, 1.5, 40),
                              color=C_OK, alpha=0.85)
    ymax = float(counts.max())
    ax.axvline(med, color=C_TRUTH, ls="--", label=f"median {med:.2f}")

    # pick the two example windows: a MODAL window (drift at the distribution
    # mode -- a representative fixation, not the cherry-picked flattest), and a
    # mostly-fixating window with the largest geometric-median–centroid
    # divergence (a brief microsaccade), restricted to the plotted drift range.
    centers = 0.5 * (bins[:-1] + bins[1:])
    mode_x = float(centers[int(np.argmax(counts))])
    disp = np.linalg.norm(rho - traj.mean(1), axis=1)
    flat_i = int(np.argmin(np.abs(ratio - mode_x)))
    cand = np.where(ratio < 1.0)[0]
    sacc_i = int(cand[np.argmax(disp[cand])])
    examples = [(flat_i, "fixation", C_FULL), (sacc_i, "microsaccade", C_CLOSE)]
    print(f"  example drift ratios: modal-fixation={ratio[flat_i]:.2f}  "
          f"microsaccade={ratio[sacc_i]:.2f}  (mode={mode_x:.2f})")

    # short arrows in the headroom above the bars, colour-matched to B
    ax.set_ylim(0, ymax * 1.32)
    for i, lbl, col in examples:
        ax.annotate("", xy=(ratio[i], ymax * 1.02), xytext=(ratio[i], ymax * 1.22),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
    ax.set_xlabel(r"within-window RMS drift / $\sigma_e$")
    ax.set_ylabel("window count")
    ax.set_title("A  real trajectories are nearly flat")
    ax.legend(fontsize=7, loc="upper right")

    # B: the two example windows. The geometric median (star) sits in the
    # fixation cluster -- the majority of the window's per-bin points -- while the
    # centroid (cross) is dragged toward the microsaccade excursion.
    for i, lbl, col in examples:
        ax_ex.plot(traj[i, :, 0], traj[i, :, 1], "-o", ms=2, lw=0.8, color=col,
                   alpha=0.8, label=lbl)
        g = _geometric_median(traj[i])
        c = traj[i].mean(0)
        ax_ex.plot(*g, "*", ms=12, color=col, mec="k", mew=0.5)
        ax_ex.plot(*c, "x", ms=7, color=col, mew=1.6)
    ax_ex.set_title(r"B  example windows ($\bigstar$ geo-median, $\times$ centroid)")
    ax_ex.set_xlabel("x (deg)"); ax_ex.set_ylabel("y (deg)")
    ax_ex.set_aspect("equal"); ax_ex.legend(fontsize=7)


def _estimate(traj, tidx, kind, sigma_eye, ell, sigma_M, target,
              reduction="geometric_median", seeds=(0, 1, 2)):
    """Seed-averaged 1-alpha for one (mask, ell, target, reduction)."""
    vals = []
    for s in seeds:
        counts = synth_counts(traj, tidx, kind, sigma_eye, ell, sigma_M, seed=s)
        # validates the trajectory reduction against the idealized p / p^2
        # closed forms on real eye windows; pin to the 'squared' construction.
        d = decompose_trajectory(counts, traj, tidx, target=target,
                                 threshold=THRESHOLD, reduction=reduction,
                                 closepair_density="squared")
        vals.append(float(d["one_minus_alpha"][0]))
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def panel_recovery(ax, traj, tidx, sigma_eye, sigma_M, rho, mids_p2):
    """C: ell/sigma_e recovery sweep on real eyes (flat + central masks).
    Lines = non-parametric truth, markers = matched-estimator mean ± sd."""
    from matplotlib.lines import Line2D
    xs = ELL_OVER_SIG
    for kind, ls, mk in [("flat", "-", "o"), ("central", "--", "s")]:
        tp, tp2, d1, d2, d1e, d2e = ([] for _ in range(6))
        for r in xs:
            ell = r * sigma_eye
            tp.append(empirical_one_minus_alpha(rho, kind, sigma_eye, ell, sigma_M))
            tp2.append(empirical_one_minus_alpha(mids_p2, kind, sigma_eye, ell, sigma_M))
            m, e = _estimate(traj, tidx, kind, sigma_eye, ell, sigma_M, "full")
            d1.append(m); d1e.append(e)
            m, e = _estimate(traj, tidx, kind, sigma_eye, ell, sigma_M, "central")
            d2.append(m); d2e.append(e)
            print(f"  {kind:9s} ell/sig={r:.1f}  truth_p={tp[-1]:.3f} D1={d1[-1]:.3f} "
                  f"truth_p2={tp2[-1]:.3f} D2={d2[-1]:.3f}")
        ax.plot(xs, tp, color=C_FULL, lw=0.9, ls=ls, alpha=0.7)
        ax.plot(xs, tp2, color=C_CLOSE, lw=0.9, ls=ls, alpha=0.7)
        ax.errorbar(xs, d1, yerr=d1e, fmt=mk, color=C_FULL, ms=4, capsize=2, lw=0,
                    elinewidth=0.8)
        ax.errorbar(xs, d2, yerr=d2e, fmt=mk, color=C_CLOSE, ms=4, capsize=2, lw=0,
                    elinewidth=0.8)
    ax.set_xlabel(r"$\ell/\sigma_e$"); ax.set_ylabel(r"$1-\alpha$")
    ax.set_title("C  recovery on real eye statistics")
    ax.set_ylim(0, 1)
    handles = [
        Line2D([0], [0], color=C_FULL, lw=2,
               label=r"Direction 1 (full) $\to 1-\alpha^p$"),
        Line2D([0], [0], color=C_CLOSE, lw=2,
               label=r"Direction 2 (central) $\to 1-\alpha^{p^2}$"),
        Line2D([0], [0], color="0.4", ls="-", marker="o", ms=4, label="flat mask"),
        Line2D([0], [0], color="0.4", ls="--", marker="s", ms=4, label="central mask"),
    ]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")


# ---------------------------------------------------------------------------

def main():
    configure()
    print("building real eye-trajectory windows ...")
    traj, tidx = build_real_windows()
    rho = _geometric_median(traj)
    sigma_eye = float(np.sqrt(0.5 * (rho[:, 0].var() + rho[:, 1].var())))
    sigma_M = _default_sigma_M(sigma_eye)
    print(f"  {len(traj)} windows, t_window={T_WINDOW}, sigma_e(real)={sigma_eye:.3f} deg")
    mids_p2 = close_pair_midpoints(rho, eps=THRESHOLD / 2.0)
    print(f"  p^2 truth from {len(mids_p2)} close-pair midpoints (eps={THRESHOLD/2:.3f})")

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.4, wspace=0.3)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_ex = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, :])

    panel_flatness(ax_A, ax_ex, traj, sigma_eye)
    print("recovery sweep ...")
    panel_recovery(ax_C, traj, tidx, sigma_eye, sigma_M, rho, mids_p2)

    save(fig, "fig_trajectory.png")


if __name__ == "__main__":
    main()
