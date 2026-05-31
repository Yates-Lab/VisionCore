r"""Figure: the geometric origin of the distribution mismatch.

Distinct-trial pairs whose eye positions lie within a small threshold (the close
pairs the rate-covariance estimator conditions on) are over-represented where the
eye-position density p(e) is high. Their density is the SQUARED density p(e)^2,
a tighter, more central distribution. For an isotropic Gaussian p = N(0, sigma^2 I)
the close-pair distribution is exactly N(0, (sigma^2/2) I) -- the variance halves.

That re-weighting is harmless for a homogeneous stimulus (rate independent of
absolute eye position) but biases every term of the LOTC decomposition for a
non-homogeneous stimulus, with a sign set by where the rate's eye-sensitivity lives.

Run from this folder:  uv run python fig_mechanism.py
"""
import numpy as np
import matplotlib.pyplot as plt

from synthetic import profile_M
from _style import configure, save, C_FULL, C_CLOSE

SIGMA = 0.15
THR = 0.05


def close_pair_positions(n=3000, sigma=SIGMA, thr=THR, seed=0):
    """Empirical close-pair representative positions from iid eyes ~ N(0,sigma^2 I)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=(n, 2))
    i, j = np.triu_indices(n, k=1)
    d = np.linalg.norm(e[i] - e[j], axis=1)
    close = d < thr
    mid = 0.5 * (e[i][close] + e[j][close])
    return e, mid


def main():
    configure()
    e, mid = close_pair_positions()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # --- Panel A: 2D -- all eyes (p) vs close-pair midpoints (p^2) ---
    ax = axes[0]
    ax.scatter(e[:, 0], e[:, 1], s=3, alpha=0.10, color="0.6",
               label=r"all eyes  $p(e)$", zorder=1)
    ax.scatter(mid[:, 0], mid[:, 1], s=3, alpha=0.14, color=C_CLOSE,
               label=r"close pairs  $\propto p(e)^2$", zorder=2)
    th = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2):  # k-sigma circles for each distribution
        ax.plot(k * SIGMA * np.cos(th), k * SIGMA * np.sin(th), color=C_FULL,
                lw=1.6, zorder=3, label=(r"$p$: $1,2\sigma$" if k == 1 else None))
        s2 = SIGMA / np.sqrt(2)
        ax.plot(k * s2 * np.cos(th), k * s2 * np.sin(th), color=C_CLOSE,
                lw=1.6, ls="--", zorder=3,
                label=(r"$p^2$: $1,2\sigma/\sqrt{2}$" if k == 1 else None))
    lim = 3 * SIGMA
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"eye $x$ (deg)"); ax.set_ylabel(r"eye $y$ (deg)")
    ax.set_title("A  close pairs concentrate centrally")
    lg = ax.legend(loc="upper right", markerscale=3, handletextpad=0.2, fontsize=6.5)
    for h in lg.legend_handles:
        h.set_alpha(1.0)

    # --- Panel B: 1D marginal -- variance halves, exact Gaussian overlays ---
    ax = axes[1]
    bins = np.linspace(-3 * SIGMA, 3 * SIGMA, 60)
    ax.hist(e[:, 0], bins=bins, density=True, color=C_FULL, alpha=0.35)
    ax.hist(mid[:, 0], bins=bins, density=True, color=C_CLOSE, alpha=0.35)
    xx = np.linspace(-3 * SIGMA, 3 * SIGMA, 400)
    gauss = lambda x, s: np.exp(-x**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
    ax.plot(xx, gauss(xx, SIGMA), color=C_FULL,
            label=rf"$\mathcal{{N}}(0,\sigma^2)$, $\sigma$={SIGMA}")
    ax.plot(xx, gauss(xx, SIGMA / np.sqrt(2)), color=C_CLOSE,
            label=r"$\mathcal{N}(0,\sigma^2/2)$ (= $p^2$)")
    var_ratio = mid[:, 0].var() / e[:, 0].var()
    ax.set_xlabel(r"eye $x$ (deg)"); ax.set_ylabel("density")
    ax.set_title(f"B  variance halves  (obs ratio {var_ratio:.2f})")
    ax.legend(loc="upper right")

    # --- Panel C: consequence -- E_D[M^2] differs across distributions ---
    # In the unified (multiplicative-mask) model the naive Crate bias source
    # is the ratio E_{p^2}[M^2] / E_p[M^2]: bias > 1 (central) -> over-state
    # rate variance; < 1 (eccentric) -> under-state.
    ax = axes[2]
    rng = np.random.default_rng(1)
    M = 1_000_000
    ep = rng.normal(0, SIGMA, size=(M, 2))
    ep2 = rng.normal(0, SIGMA / np.sqrt(2), size=(M, 2))
    kinds = ["central", "eccentric", "linear"]
    xpos = np.arange(len(kinds))
    EM2p = [(profile_M(ep, k, SIGMA) ** 2).mean() for k in kinds]
    EM2p2 = [(profile_M(ep2, k, SIGMA) ** 2).mean() for k in kinds]
    w = 0.38
    ax.bar(xpos - w / 2, EM2p, w, color=C_FULL, label=r"$\mathbb{E}_{p}[M^2]$")
    ax.bar(xpos + w / 2, EM2p2, w, color=C_CLOSE, label=r"$\mathbb{E}_{p^2}[M^2]$")
    ax.set_xticks(xpos); ax.set_xticklabels(kinds)
    ax.set_ylabel(r"$\mathbb{E}_D[M^2]$")
    ax.set_title(r"C  the close-pair / full ratio sets the Crate bias")
    ax.legend(loc="upper right")
    for x in xpos:
        # the close-pair (p^2) estimator over/under-states Crate relative to p
        sign = "+" if EM2p2[x] > EM2p[x] else "−"
        ax.annotate(rf"$1{{-}}\alpha$ bias {sign}", (x, max(EM2p[x], EM2p2[x])),
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, "fig_mechanism.png")


if __name__ == "__main__":
    main()
