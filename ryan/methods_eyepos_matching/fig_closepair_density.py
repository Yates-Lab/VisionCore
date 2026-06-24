r"""Figure for note_closepair_density.md: directly-estimated close-pair density
vs the squared-marginal assumption, and its effect on the §4.5 results.

Reads ``cache/closepair_density.pkl`` (built by ``compute_closepair_density.py``)
for the per-session diagnostics and pooled per-cell 1-α, and re-derives one
example session's representative-point / close-pair-midpoint clouds for the
marginal-overlay panel.

  A  x-marginal of the eye density: p (representative points), p^2 (the squared
     marginal the 'squared' weights assume), and the directly-estimated
     close-pair density p_pair. p_pair sits inside p^2 -- real close pairs are
     more central than the Gaussian p^2 prediction.
  B  per-session close-pair variance ratio tr cov(rho_mid)/tr cov(rho) against
     the ideal Gaussian value 0.5, coloured by KL(p_pair ‖ p^2).
  C  per-cell 1-α full (p): 'squared' vs 'direct' close-pair density.
  D  per-cell 1-α central (p^2): 'squared' vs 'direct'.

Run from this folder:  uv run python fig_closepair_density.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import dill
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH      # noqa: E402
from compute_closepair_density import CACHE                        # noqa: E402

C_PAIR = "#8e44ad"        # purple -- directly-estimated p_pair


def _example_session_clouds():
    """Recompute (rho, rho_mid) for one representative session for panel A."""
    from data_loading import load_cache
    from pipeline import _extract_windows_numpy, DT as PIPE_DT
    from estimators import _geometric_median, _rms_traj_close_pairs
    from legacy.covariance import extract_valid_segments
    from compute_closepair_density import (T_HIST_MS, T_COUNT_BINS, MIN_SEG_LEN)

    with open(CACHE, "rb") as f:
        res = dill.load(f)
    vr = {d["session"]: d["var_ratio"] for d in res["diagnostics"]}
    target = float(np.median(list(vr.values())))
    pick = min(vr, key=lambda s: abs(vr[s] - target))   # nearest-median session

    for rec in load_cache():
        if rec["session"] != pick:
            continue
        robs = np.nan_to_num(np.asarray(rec["robs"], float), nan=0.0)
        eyepos = np.nan_to_num(np.asarray(rec["eyepos"], float), nan=0.0)
        segments = extract_valid_segments(np.asarray(rec["valid_mask"], bool),
                                          min_len_bins=MIN_SEG_LEN)
        t_hist = max(int(T_HIST_MS / (PIPE_DT * 1000)), T_COUNT_BINS)
        _c, traj, T_idx = _extract_windows_numpy(robs, eyepos, segments,
                                                 T_COUNT_BINS, t_hist)
        rho = _geometric_median(traj)
        gi, gj, _t, _m = _rms_traj_close_pairs(traj, T_idx, 0.05)
        rho_mid = 0.5 * (rho[gi] + rho[gj])
        return pick, vr[pick], rho, rho_mid
    raise RuntimeError("example session not found in cache")


def _scatter_oma(ax, x, y, color, title):
    lim = (0, 1)
    ax.plot(lim, lim, color=C_TRUTH, lw=0.8, ls="--", zorder=1)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=9, color=color, alpha=0.4, lw=0, zorder=2)
    med = float(np.nanmedian(y[ok] - x[ok]))
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel(r"squared ($p^2$) $1-\alpha$")
    ax.set_ylabel(r"direct ($p_{\rm pair}$) $1-\alpha$")
    ax.set_title(title)
    ax.text(0.04, 0.96, f"$\\Delta$median {med:+.3f}\n$n={int(ok.sum())}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            color=C_TRUTH)


def make_figure():
    configure()
    with open(CACHE, "rb") as f:
        res = dill.load(f)
    diag = res["diagnostics"]
    vr = np.array([d["var_ratio"] for d in diag])
    kl = np.array([d["kl_pair_vs_squared"] for d in diag])

    g = res["squared"]["good"]
    full_sq, full_dir = res["squared"]["full"][g], res["direct"]["full"][g]
    cent_sq, cent_dir = res["squared"]["central"][g], res["direct"]["central"][g]

    pick, vr_pick, rho, rho_mid = _example_session_clouds()

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))

    # --- A: x-marginal of p, p^2, p_pair on the example session -------------
    ax = axes[0, 0]
    x_rho = rho[:, 0]
    x_mid = rho_mid[:, 0]
    sd = float(np.std(x_rho))
    xs = np.linspace(-4 * sd, 4 * sd, 400)
    gauss = lambda s: np.exp(-xs ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
    bins = np.linspace(-4 * sd, 4 * sd, 51)
    ax.hist(x_rho, bins=bins, density=True, color=C_FULL, alpha=0.25,
            label=r"$p$ (rep. points)")
    ax.hist(x_mid, bins=bins, density=True, color=C_PAIR, alpha=0.30,
            label=r"$p_{\rm pair}$ (direct, close-pair mid)")
    ax.plot(xs, gauss(sd), color=C_FULL, lw=1.6)
    ax.plot(xs, gauss(sd / np.sqrt(2)), color=C_CLOSE, lw=1.6, ls="--",
            label=r"$p^2$ assumption $\mathcal{N}(0,\sigma^2/2)$")
    ax.set_xlim(-4 * sd, 4 * sd)
    ax.set_xlabel(r"eye position $x$ (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"A  close-pair density vs $p^2$ ({pick})")
    ax.legend(fontsize=7)

    # --- B: per-session variance ratio vs ideal 0.5 -------------------------
    ax = axes[0, 1]
    order = np.argsort(vr)
    sc = ax.scatter(np.arange(len(vr)), vr[order], c=kl[order], cmap="viridis",
                    s=28, zorder=3)
    ax.axhline(0.5, color=C_CLOSE, ls="--", lw=1.2,
               label=r"ideal $p^2$ ($=0.5$)")
    ax.axhline(float(np.median(vr)), color=C_TRUTH, ls=":", lw=1.0,
               label=f"median {np.median(vr):.3f}")
    ax.set_xlabel("session (sorted by variance ratio)")
    ax.set_ylabel(r"$\mathrm{tr}\,\mathrm{cov}(\rho_{\rm mid})/"
                  r"\mathrm{tr}\,\mathrm{cov}(\rho)$")
    ax.set_title("B  close-pair variance ratio")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"KL$(p_{\rm pair}\,\|\,p^2)$", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")

    # --- C, D: per-cell 1-α shift, squared vs direct ------------------------
    _scatter_oma(axes[1, 0], full_sq, full_dir, C_FULL,
                 "C  full ($p$): squared vs direct")
    _scatter_oma(axes[1, 1], cent_sq, cent_dir, C_CLOSE,
                 "D  central ($p^2$): squared vs direct")

    fig.tight_layout()
    save(fig, "fig_closepair_density.png")


if __name__ == "__main__":
    make_figure()
