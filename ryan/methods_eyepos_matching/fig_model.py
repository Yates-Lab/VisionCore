r"""Figure: visual introduction to the unified generative model.

For colleagues meeting the synthetic for the first time, this figure walks
through each ingredient of

    r_c(t, e) = mu_0 + M_c(e) * alpha(t) * s_t(e)

and shows the resulting data structure.

  A  eye distribution p(e) = N(0, sigma^2 I) -- where the eye lands.
  B  one draw of the stationary Gaussian random field s_t(e); kernel
     K(delta) = tau^2 exp(-||delta||^2 / (2 ell^2)) at a small ell to make
     the spatial structure visible.
  C  the four spatial masks M_c(e) tiled into a single 2x2 square.
  D  envelope effect: r(t, e_t) over one trial, with alpha(t) constant vs
     decaying -- the only difference between the two curves is alpha.
  E  full model rate r over (trial, time bin) with constant n_t; the
     (N_trials, T) array is a full rectangle.
  F  same with variable n_t (staircase); trial/time cells past trial end
     are NaN and rendered white -- the (A1) regime.

Run from this folder:  uv run python fig_model.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patheffects
from matplotlib.colors import Normalize

from synthetic import profile_M, _draw_field_at, make_session
from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH

SIG = 0.15        # fixational spread (deg)
TAU = 1.0
MU0 = 6.0
ELL_M = 0.6 * SIG       # default mask length scale (synthetic._default_ell_M)
ELL_VIS = 0.5 * SIG     # smaller field length scale -> visible structure

GRID_LIM = 3.0 * SIG    # heatmap extent: +- 3 sigma
GRID_N = 56             # 56 x 56 = 3136 grid points (one Cholesky)

T_D = 80                # time bins for panel D
N_TRIALS_VIS = 15       # panels E / F
N_TIME_VIS = 50
NT_VAR_LO = 2           # variable n_t staircase: from N_TRIALS_VIS down to here


def _eye_grid(n=GRID_N, lim=GRID_LIM):
    x = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(x, x)
    return x, np.stack([xx, yy], axis=-1)


def panel_A_eye(ax, rng):
    e = rng.normal(0.0, SIG, size=(1500, 2))
    ax.scatter(e[:, 0], e[:, 1], s=5, alpha=0.30, color=C_FULL, lw=0)
    th = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2):
        ax.plot(k * SIG * np.cos(th), k * SIG * np.sin(th),
                color=C_TRUTH, lw=1.0, ls="--")
    ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"eye $x$ (deg)"); ax.set_ylabel(r"eye $y$ (deg)")
    ax.set_title(rf"A  eye distribution  $p(e)=\mathcal{{N}}(0,\sigma^2 I)$,"
                 rf" $\sigma={SIG}^\circ$")
    ax.text(0.04, 0.04, r"dashed: $1\sigma, 2\sigma$",
            transform=ax.transAxes, fontsize=7, color=C_TRUTH)


def panel_B_field(ax, rng):
    _, e_grid = _eye_grid()
    s = _draw_field_at(e_grid.reshape(-1, 2), ell=ELL_VIS, tau=TAU,
                       rng=rng, n_cells=1).reshape(GRID_N, GRID_N)
    vmax = float(np.max(np.abs(s)))
    im = ax.imshow(s, extent=[-GRID_LIM, GRID_LIM, -GRID_LIM, GRID_LIM],
                   origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"eye $x$ (deg)"); ax.set_ylabel(r"eye $y$ (deg)")
    ax.set_title(r"B  field  $s_t(e)\sim\mathrm{GP}(0,K)$,  "
                 r"$K(\delta)=\tau^2 e^{-\|\delta\|^2/(2\ell^2)}$"
                 r"  ($\ell=0.5\sigma$)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$s_t(e)$")


def panel_C_masks_tiled(ax):
    """All four masks rendered into one square imshow, 2x2 quadrants."""
    n = GRID_N
    _, e_grid = _eye_grid()
    tile = np.zeros((2 * n, 2 * n))
    # origin='lower' -> rows index from bottom up:
    #   TL (top-left)   = eccentric   TR (top-right)   = linear
    #   BL (bottom-left)= flat        BR (bottom-right)= central
    layout = [
        (slice(0, n),     slice(0, n),     "flat"),
        (slice(0, n),     slice(n, 2*n),   "central"),
        (slice(n, 2*n),   slice(0, n),     "eccentric"),
        (slice(n, 2*n),   slice(n, 2*n),   "linear"),
    ]
    for rs, cs, kind in layout:
        tile[rs, cs] = profile_M(e_grid, kind, SIG, ELL_M)
    im = ax.imshow(tile, extent=[0, 2, 0, 2], origin="lower",
                   cmap="viridis", vmin=0, vmax=1)
    ax.axhline(1.0, color="w", lw=1.2)
    ax.axvline(1.0, color="w", lw=1.2)
    labels = [
        ("flat",      (0.5, 0.95)),
        ("central",   (1.5, 0.95)),
        ("eccentric", (0.5, 1.95)),
        ("linear",    (1.5, 1.95)),
    ]
    for txt, (x, y) in labels:
        ax.text(x, y, txt, color="w", ha="center", va="top",
                fontsize=9, fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=1.6,
                                                     foreground="k")])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(r"C  spatial masks  $M_c(e)$  (4 kinds, tiled)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$M(e)$")


def panel_D_envelope(ax, seed=2):
    T = T_D
    alpha_decay = np.linspace(1.0, 0.05, T)
    # Same seed in both calls => same eye trajectory and same field draws;
    # the only difference between the two rate traces is alpha(t).
    sess_const = make_session(
        ["central"], n_trials=1, n_time_bins=T,
        sigma_eye=SIG, ell=ELL_VIS, tau=TAU, mu_0=MU0, ell_M=ELL_M,
        psth_envelope=None, seed=seed)
    sess_decay = make_session(
        ["central"], n_trials=1, n_time_bins=T,
        sigma_eye=SIG, ell=ELL_VIS, tau=TAU, mu_0=MU0, ell_M=ELL_M,
        psth_envelope=alpha_decay, seed=seed)
    t = np.arange(T)
    ax.plot(t, sess_const["rate"][0, :, 0], color=C_TRUTH, lw=1.4, ls="--",
            label=r"$\alpha\equiv 1$")
    ax.plot(t, sess_decay["rate"][0, :, 0], color=C_FULL, lw=1.6,
            label=r"$\alpha(t)$: $1\to 0.05$")
    ax.axhline(MU0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel(r"time bin $t$")
    ax.set_ylabel(r"rate $r(t,e_t)$")
    ax.set_title(r"D  envelope effect (one trial, `central` cell)")
    ax.legend(loc="upper right", fontsize=7)


def _make_rate(seed, n_trials_per_time_bin=None):
    sess = make_session(
        ["central"], n_trials=N_TRIALS_VIS, n_time_bins=N_TIME_VIS,
        sigma_eye=SIG, ell=ELL_VIS, tau=TAU, mu_0=MU0, ell_M=ELL_M,
        n_trials_per_time_bin=n_trials_per_time_bin, seed=seed)
    return sess["rate"][:, :, 0]


def panels_EF_rates(ax_E, ax_F, seed=10):
    """Constant-n_t (E) and variable-n_t (F) rate heatmaps on a shared scale."""
    rate_E = _make_rate(seed=seed)
    nt_var = np.linspace(N_TRIALS_VIS, NT_VAR_LO,
                         N_TIME_VIS).round().astype(int)
    rate_F = _make_rate(seed=seed, n_trials_per_time_bin=nt_var)

    finite = np.concatenate([rate_E[np.isfinite(rate_E)],
                             rate_F[np.isfinite(rate_F)]])
    dev = float(np.max(np.abs(finite - MU0)))
    norm = Normalize(vmin=MU0 - dev, vmax=MU0 + dev)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")

    im = ax_E.imshow(rate_E, aspect="auto", origin="lower",
                     cmap=cmap, norm=norm, interpolation="nearest")
    ax_E.set_xlabel(r"time bin $t$"); ax_E.set_ylabel(r"trial $i$")
    ax_E.set_title(r"E  $r(t,e)$ over (trial, time), constant $n_t$")

    rate_F_m = np.ma.masked_invalid(rate_F)
    ax_F.imshow(rate_F_m, aspect="auto", origin="lower",
                cmap=cmap, norm=norm, interpolation="nearest")
    # Step line tracing the trial-end boundary makes the staircase explicit.
    ax_F.step(np.arange(N_TIME_VIS), nt_var - 0.5, where="mid",
              color="k", lw=1.2)
    ax_F.set_xlim(-0.5, N_TIME_VIS - 0.5)
    ax_F.set_ylim(-0.5, N_TRIALS_VIS - 0.5)
    ax_F.set_xlabel(r"time bin $t$"); ax_F.set_ylabel(r"trial $i$")
    ax_F.set_title(r"F  variable $n_t$ (NaN past trial end, white)")
    return im


def main():
    configure()
    fig = plt.figure(figsize=(13.0, 8.5))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[0.12, 1.0, 1.0],
                           hspace=0.55, wspace=0.40)
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.axis("off")
    ax_eq.text(0.5, 0.5,
               r"$r_c(t,e)\;=\;\mu_0\;+\;M_c(e)\,\alpha(t)\,s_t(e)$",
               ha="center", va="center", fontsize=16)

    ax_A = fig.add_subplot(gs[1, 0])
    ax_B = fig.add_subplot(gs[1, 1])
    ax_C = fig.add_subplot(gs[1, 2])
    panel_A_eye(ax_A, np.random.default_rng(0))
    panel_B_field(ax_B, np.random.default_rng(7))
    panel_C_masks_tiled(ax_C)

    ax_D = fig.add_subplot(gs[2, 0])
    ax_E = fig.add_subplot(gs[2, 1])
    ax_F = fig.add_subplot(gs[2, 2])
    panel_D_envelope(ax_D)
    im_EF = panels_EF_rates(ax_E, ax_F)
    cb = plt.colorbar(im_EF, ax=ax_F, shrink=0.85, pad=0.02)
    cb.set_label(r"rate (counts / window)")

    save(fig, "fig_model.png")


if __name__ == "__main__":
    main()
