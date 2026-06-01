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
from matplotlib.lines import Line2D

from synthetic import ground_truth
from _style import configure, save, C_FULL, C_CLOSE

SIGMA = 0.15
THR = 0.05
ELL = SIGMA   # fixation-scale rate structure for the Panel C bias bars


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

    # --- Panel A: analytical p(e) heatmap + iso-density contours of p and p^2 ---
    # No simulation: imshow the closed-form density p(e) = N(0, σ²I) (grayscale)
    # and draw iso-density contour lines at radius 1σ, 2σ for both p (solid) and
    # p² = N(0, σ²/2 I) (dashed) at each distribution's own characteristic scale.
    # For a 2D Gaussian with covariance S²I, the density at radius r = kS is
    # peak * exp(-k²/2); so contour levels are peak * {exp(-2), exp(-1/2)}.
    ax = axes[0]
    lim = 3 * SIGMA
    xx = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(xx, xx)
    r2 = X ** 2 + Y ** 2
    p_xy = np.exp(-r2 / (2 * SIGMA ** 2)) / (2 * np.pi * SIGMA ** 2)
    p2_xy = np.exp(-r2 / SIGMA ** 2) / (np.pi * SIGMA ** 2)        # N(0, σ²/2 I)
    ax.imshow(p_xy, extent=(-lim, lim, -lim, lim), origin="lower",
              cmap="Greys_r", aspect="equal")
    peak_p = 1.0 / (2 * np.pi * SIGMA ** 2)
    peak_p2 = 1.0 / (np.pi * SIGMA ** 2)
    rel_levels = np.array([np.exp(-2.0), np.exp(-0.5)])             # 2σ, 1σ
    ax.contour(X, Y, p_xy, levels=peak_p * rel_levels, colors=C_FULL, linewidths=1.6)
    ax.contour(X, Y, p2_xy, levels=peak_p2 * rel_levels, colors=C_CLOSE,
               linewidths=1.6, linestyles="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"eye $x$ (deg)"); ax.set_ylabel(r"eye $y$ (deg)")
    ax.set_title("A  close pairs concentrate centrally")
    p_proxy = Line2D([0], [0], color=C_FULL, lw=1.6,
                     label=r"$p(e)$:  $1,2\sigma$")
    p2_proxy = Line2D([0], [0], color=C_CLOSE, lw=1.6, ls="--",
                      label=r"$p(e)^2$:  $1,2\sigma/\sqrt{2}$")
    ax.legend(handles=[p_proxy, p2_proxy], loc="upper right",
              handletextpad=0.6, fontsize=7, frameon=False,
              labelcolor="white")

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

    # --- Panel C: consequence -- direction of the naive bias on 1-α ---
    # The naive estimator has C_psth on p (correct) but C_rate on p^2 (close-pair),
    # so 1-α^naive = 1 - I_{M,K,p} / (τ² E_{p²}[M^2]) -- numerator unchanged from
    # truth, denominator scaled by E_{p²}[M^2] / E_p[M^2]. Direction of the bias:
    # central > 1 → naive 1-α biased up; eccentric/linear < 1 → biased down.
    # ground_truth(...) uses closed-form 4M-sample MC of the M-K-D integral
    # I_{M,K,D} and E_D[M^2] under D ∈ {p, p²}; sampling noise ≲ 1e-3.
    ax = axes[2]
    kinds = ["central", "eccentric", "linear"]
    xpos = np.arange(len(kinds))
    oma_p, oma_naive = [], []
    for k in kinds:
        gt = ground_truth(k, sigma_eye=SIGMA, ell=ELL)
        oma_p.append(gt["p"]["one_minus_alpha"])
        # naive: var_psth over p (correct), var_total over p^2 (wrong)
        oma_naive.append(1.0 - gt["p"]["var_psth"] / gt["p2"]["var_total"])
    w = 0.38
    ax.bar(xpos - w / 2, oma_p, w, color=C_FULL,
           label=r"$1{-}\alpha^{p}$  (truth)")
    ax.bar(xpos + w / 2, oma_naive, w, color=C_CLOSE,
           label=r"$1{-}\alpha^{\mathrm{naive}}$")
    ax.set_xticks(xpos); ax.set_xticklabels(kinds)
    ax.set_ylabel(r"$1{-}\alpha$")
    ax.set_ylim(0, max(max(oma_p), max(oma_naive)) * 1.25)
    ax.set_title(rf"C  naive bias on $1{{-}}\alpha$  ($\ell{{=}}\sigma$)")
    ax.legend(loc="upper right", fontsize=7)
    for x, (a, b) in enumerate(zip(oma_p, oma_naive)):
        sign = "+" if b > a else "−"
        ax.annotate(f"bias {sign}", (x, max(a, b)),
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, "fig_mechanism.png")


if __name__ == "__main__":
    main()
