r"""Figure 0a: the closed-form LOTC decomposition depends on the eye-position
distribution and the mask width -- before any estimator is introduced.

This figure relocates the old fig_sanity_check panel C (the flat-mask gap) and
adds central- and eccentric-mask panels, to make the §4 point at the end of
§2.3: even for a homogeneous (flat) stimulus the truth 1-alpha differs between
the viewing density p and the close-pair density p^2, and for a non-homogeneous
mask the spread between the two widens as the mask narrows toward the fixation
scale. All curves are closed-form (writeup Appendix A.2-A.4), not estimator
output -- closed forms now exist for all three masks (flat, central, eccentric).

  A  flat mask: 1-alpha^p, 1-alpha^p2, and their gap vs ell/sigma_e. The gap is
     non-zero at every finite ell, peaks at ell/sigma_e ~ 1.18 (max ~ 0.17), and
     vanishes only as ell -> 0 (decorrelated) or ell -> infinity (uniform).
  B  central (Gaussian) mask: 1-alpha^p (solid) and 1-alpha^p2 (dashed) vs
     ell/sigma_e for several mask widths sigma_M. The p-vs-p2 spread is set
     jointly by ell/sigma_e and sigma_M, peaking (~0.2) when sigma_M is
     comparable to the fixation scale (sigma_M ~ 0.6 sigma_e here).
  C  eccentric (1 - Gaussian) mask: same axes, the bounded-complement case
     whose modulated variance lives in the periphery (Appendix A.4 closed form).

Run from this folder:  uv run python fig_distribution_truth.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from synthetic import (_central_one_minus_alpha_closed_form,
                       _eccentric_one_minus_alpha_closed_form)
from _style import configure, save, C_FULL, C_CLOSE, C_OK

SIG = 0.15


def _flat_closed_form(ell, sigma=SIG):
    """Analytical 1-alpha^p, 1-alpha^p2, and gap for the flat mask."""
    s2 = sigma ** 2
    L2 = ell ** 2
    oma_p = 2.0 * s2 / (L2 + 2.0 * s2)
    oma_p2 = s2 / (L2 + s2)
    return oma_p, oma_p2, oma_p - oma_p2


def panel_A(ax):
    """Flat mask: 1-alpha^p, 1-alpha^p2, gap vs ell/sigma (relocated from the
    old sanity-check panel C)."""
    r = np.geomspace(0.1, 20, 400)
    oma_p, oma_p2, gap = _flat_closed_form(r * SIG)
    ax.plot(r, oma_p, color=C_FULL, lw=1.6, label=r"$1-\alpha^p$")
    ax.plot(r, oma_p2, color=C_CLOSE, lw=1.6, label=r"$1-\alpha^{p^2}$")
    ax.plot(r, gap, color=C_OK, lw=1.6, ls="--",
            label=r"gap $= (1{-}\alpha^p) - (1{-}\alpha^{p^2})$")
    i_max = int(np.argmax(gap))
    ax.scatter([r[i_max]], [gap[i_max]], s=24, color=C_OK, zorder=5)
    ax.annotate(rf"max gap $\approx${gap[i_max]:.2f} at $\ell/\sigma_e\approx${r[i_max]:.2f}",
                (r[i_max], gap[i_max]), xytext=(8, 8),
                textcoords="offset points", fontsize=7, color=C_OK)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell\,/\,\sigma_e$")
    ax.set_ylabel(r"closed-form $1-\alpha$")
    ax.set_title(r"A  flat mask: the truth depends on the eye distribution")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=7)


def _mask_panel(ax, closed_form, title, sigma_M_ratios=(0.4, 0.6, 1.0)):
    """Non-homogeneous mask: 1-alpha^p (solid) vs 1-alpha^p2 (dashed) vs
    ell/sigma_e for several mask widths sigma_M; the solid-dashed gap is the
    p-vs-p2 spread. ``closed_form(sigma_e, ell, sigma_M, dist)`` selects the mask."""
    r = np.geomspace(0.1, 20, 400)
    ells = r * SIG
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(sigma_M_ratios)))
    for ratio, col in zip(sigma_M_ratios, colors):
        sigma_M = ratio * SIG
        oma_p = np.array([closed_form(SIG, e, sigma_M, "p") for e in ells])
        oma_p2 = np.array([closed_form(SIG, e, sigma_M, "p2") for e in ells])
        ax.plot(r, oma_p, color=col, lw=1.6, ls="-")
        ax.plot(r, oma_p2, color=col, lw=1.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell\,/\,\sigma_e$")
    ax.set_ylabel(r"closed-form $1-\alpha$")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    width_handles = [Line2D([0], [0], color=c, lw=1.6,
                            label=rf"$\sigma_M = {ratio:.1f}\,\sigma_e$")
                     for ratio, c in zip(sigma_M_ratios, colors)]
    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.6, ls="-", label=r"$1-\alpha^p$"),
        Line2D([0], [0], color="0.3", lw=1.6, ls="--", label=r"$1-\alpha^{p^2}$"),
    ]
    ax.legend(handles=width_handles + style_handles, loc="upper right",
              fontsize=7, ncol=1)


def panel_B(ax):
    """Central (Gaussian) mask: mask width sets the p-vs-p2 spread."""
    _mask_panel(ax, _central_one_minus_alpha_closed_form,
                r"B  central mask: mask width sets the $p$ vs $p^2$ spread")


def panel_C(ax):
    """Eccentric (1 - Gaussian) mask: the complement case, variance in the
    periphery; the closed form is writeup Appendix A.4."""
    _mask_panel(ax, _eccentric_one_minus_alpha_closed_form,
                r"C  eccentric mask: the bounded-complement case")


def main():
    configure()
    # Panel A spans the full top row; B and C share the row below, so all
    # three are legible at HTML size.
    fig, axd = plt.subplot_mosaic([["A", "A"], ["B", "C"]], figsize=(10, 7))
    panel_A(axd["A"])
    panel_B(axd["B"])
    panel_C(axd["C"])
    fig.tight_layout()
    save(fig, "fig_distribution_truth.png")


if __name__ == "__main__":
    main()
