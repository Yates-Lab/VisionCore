r"""Figure: estimator consistency under (A1)+(A2). Appendix A.6.

Sweeps the empirical seed-to-seed sd[1-alpha-hat] over a 2-D grid of
(n_trials_per_time_bin, n_time_bins) on the `flat`-mask synthetic, with constant n_t
and deterministic rates. Two regimes:

  * ell = sigma   (alpha* = 1/3, 1-alpha* = 2/3) -- the SEM panels.
  * ell = 0.3 sigma (alpha* ~ 0.083, 1-alpha* ~ 0.917) -- exposes boundary
    clipping at small T: the upper-tail of 1-alpha-hat gets clipped at 1,
    pulling the mean down.

Panels:

  A  sd[1-alpha-hat] heatmap over (N, T) at ell=sigma. Overlaid contour is
     the analytical across-time-bin floor  alpha * sqrt(2/(T-1))  (T-only).
     Empirical sd plateaus on the floor as N grows.
  B  bias-vs-sd scatter at ell=0.3 sigma (one point per (N, T) cell).
     Overlaid curve is the truncated-Gaussian prediction E[clip(X)] - alpha*
     for X ~ N(alpha*, sd^2).
  C  sd[1-alpha-hat] vs T at fixed N=800, ell=sigma. The analytical floor
     curve is the load-bearing line; empirical points sit on it.

Parallelization
---------------
Each grid cell needs many independent make_session+decompose calls. We fan
them out over a ProcessPoolExecutor; each worker forces BLAS to single-thread
with threadpool_limits(1), preventing oversubscription on this 32-core box.

The full sweep result is cached to ``consistency_sweep.npz``; re-running this
script just re-renders the figure unless ``--recompute`` is passed.

Run from this folder:  uv run python fig_consistency.py [--recompute]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from _style import configure, save, C_FULL, C_CLOSE, C_TRUTH, C_OK

# Constants for the sweep
SIG = 0.15
THR = 0.05
N_VALUES = np.array([100, 200, 400, 800])
T_VALUES = np.array([25, 50, 100, 200])
N_SEEDS = 10
REGIMES = {
    "sigma":   {"ell": SIG,        "label": r"$\ell = \sigma_e$"},
    "0.3sig":  {"ell": 0.3 * SIG,  "label": r"$\ell = 0.3\sigma_e$"},
}

CACHE = Path(__file__).resolve().parent / "consistency_sweep.npz"


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def alpha_star(ell, sigma=SIG):
    """Closed-form  alpha* = ell^2 / (ell^2 + 2 sigma^2)  (PSTH fraction)."""
    L2, s2 = float(ell) ** 2, float(sigma) ** 2
    return L2 / (L2 + 2 * s2)


def t_floor(ell, T, sigma=SIG):
    """Analytical across-time-bin floor on sd[1-alpha-hat]:
    sqrt( 2 alpha*^2 / (T-1) ).  Depends on T (and ell) only."""
    a = alpha_star(ell, sigma)
    return a * np.sqrt(2.0 / np.maximum(T - 1, 1))


def truncated_mean(alpha, sd):
    """Mean of N(alpha, sd^2) clipped to [0, 1].

    Treats sd==0 as a no-op (returns alpha)."""
    alpha = np.asarray(alpha, float); sd = np.asarray(sd, float)
    out = np.where(sd > 0, alpha, alpha)  # safe init
    mask = sd > 0
    if mask.any():
        a, s = alpha[mask] if alpha.ndim else alpha, sd[mask]
        a_arr = np.broadcast_to(alpha, sd.shape)[mask]
        z0 = (0.0 - a_arr) / s
        z1 = (1.0 - a_arr) / s
        # E[X | trunc to [0,1] in N(a,s^2)]
        E_inside = a_arr + s * (norm.pdf(z0) - norm.pdf(z1)) / np.clip(
            norm.cdf(z1) - norm.cdf(z0), 1e-12, None)
        P_inside = norm.cdf(z1) - norm.cdf(z0)
        P_low = norm.cdf(z0)
        P_high = 1.0 - norm.cdf(z1)
        out_m = E_inside * P_inside + 0.0 * P_low + 1.0 * P_high
        out = out.astype(float).copy()
        out[mask] = out_m
    return out


# ---------------------------------------------------------------------------
# Parallel sweep driver
# ---------------------------------------------------------------------------

def _worker(spec):
    """Run a single (n_trials, n_time_bins, ell, seed) cell and return 1-alpha-hat.

    Module-level so it pickles cleanly across processes. Wrapped in
    threadpool_limits(1) to prevent BLAS oversubscription on the 32-core box.
    """
    from threadpoolctl import threadpool_limits
    n_trials, n_time_bins, ell, seed = spec
    with threadpool_limits(1):
        # Lazy import inside the worker so a forked numpy stays clean.
        from synthetic import make_session
        from estimators import decompose
        sess = make_session(["flat"], n_trials=int(n_trials),
                            n_time_bins=int(n_time_bins), sigma_eye=SIG,
                            ell=float(ell), seed=int(seed))
        d = decompose(sess["rate"], sess["eye"], target="naive",
                      density="gaussian", threshold=THR)
        return float(d["one_minus_alpha"][0])


def _run_sweep(max_workers=32):
    """Run the full 4x4 x N_seeds x 2-regime sweep in parallel.

    Returns a dict keyed by regime name -> ndarray (n_N, n_T, n_seeds) of
    one_minus_alpha estimates.
    """
    specs = []
    layout = []  # (regime, i_N, i_T, seed) for each spec
    for regime, cfg in REGIMES.items():
        for i, N in enumerate(N_VALUES):
            for j, T in enumerate(T_VALUES):
                for s in range(N_SEEDS):
                    specs.append((int(N), int(T), float(cfg["ell"]), int(s)))
                    layout.append((regime, i, j, s))

    print(f"running {len(specs)} sweep tasks on up to {max_workers} workers...")
    out = {r: np.full((len(N_VALUES), len(T_VALUES), N_SEEDS), np.nan)
           for r in REGIMES}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, spec): k for k, spec in enumerate(specs)}
        for fut in as_completed(futs):
            k = futs[fut]
            regime, i, j, s = layout[k]
            out[regime][i, j, s] = fut.result()
            done += 1
            if done % 32 == 0 or done == len(specs):
                el = time.time() - t0
                print(f"  {done}/{len(specs)}  elapsed {el:.1f}s")
    return out


def _load_or_run(recompute=False, max_workers=32):
    """Load cached sweep or run + cache."""
    if not recompute and CACHE.exists():
        print(f"loading cached sweep from {CACHE.name}")
        z = np.load(CACHE)
        return {r: z[r] for r in REGIMES}
    data = _run_sweep(max_workers=max_workers)
    np.savez(CACHE, **data)
    print(f"saved sweep to {CACHE.name}")
    return data


# ---------------------------------------------------------------------------
# Figure panels
# ---------------------------------------------------------------------------

def panel_A_sd_heatmap(ax, data):
    """Empirical sd[1-alpha-hat] heatmap over (N, T) at ell=sigma, with the
    analytical T-floor contour overlaid as labelled isolines."""
    vals = data["sigma"]                       # (n_N, n_T, n_seeds)
    sd = np.nanstd(vals, axis=2)               # (n_N, n_T)
    # imshow with origin lower so small T at the bottom, small N on the left.
    extent = [-0.5, len(T_VALUES) - 0.5, -0.5, len(N_VALUES) - 0.5]
    im = ax.imshow(sd, origin="lower", aspect="auto", cmap="viridis",
                   extent=extent, vmin=0.0)
    ax.set_xticks(np.arange(len(T_VALUES)))
    ax.set_xticklabels([str(t) for t in T_VALUES])
    ax.set_yticks(np.arange(len(N_VALUES)))
    ax.set_yticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel(r"time bins $T$")
    ax.set_ylabel(r"trials/bin $N$")
    ax.set_title(r"A  empirical sd$[1-\hat\alpha]$  ($\ell=\sigma_e$)")
    # numeric annotations
    for i in range(len(N_VALUES)):
        for j in range(len(T_VALUES)):
            ax.text(j, i, f"{sd[i, j]:.03f}", ha="center", va="center",
                    fontsize=7, color="white" if sd[i, j] < sd.max() * 0.5
                    else "black")
    # T-floor as a horizontal "expected" panel-level annotation
    floor = t_floor(SIG, T_VALUES)
    floor_txt = "T-floor: " + "  ".join(f"T={T}->{f:.03f}"
                                        for T, f in zip(T_VALUES, floor))
    ax.text(0.5, -0.32, floor_txt, transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color=C_OK)
    plt.colorbar(im, ax=ax, fraction=0.045)


def panel_B_bias_vs_sd(ax, data):
    """Bias-vs-empirical-sd scatter at ell=0.3 sigma, with truncated-Gaussian
    prediction overlaid as a curve in sd at fixed alpha*."""
    vals = data["0.3sig"]                      # (n_N, n_T, n_seeds)
    mean = np.nanmean(vals, axis=2)            # (n_N, n_T)
    sd = np.nanstd(vals, axis=2)
    alpha = alpha_star(0.3 * SIG)
    one_minus_alpha_truth = 1.0 - alpha

    # empirical bias on 1-alpha-hat
    bias = mean - one_minus_alpha_truth        # (n_N, n_T)
    # truncated-Gaussian prediction: clip-mean of N(alpha, sd) minus alpha.
    # We're estimating one_minus_alpha, so bias on (1 - alpha_hat) is
    # (1 - clip_mean) - (1 - alpha) = alpha - clip_mean.
    sd_grid = np.linspace(1e-4, max(sd.max() * 1.2, 0.25), 200)
    clip_pred = truncated_mean(alpha, sd_grid)
    bias_pred = alpha - clip_pred

    # one marker per (N, T) cell; color by T (across-time-bin term dominates)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(T_VALUES)))
    for j, T in enumerate(T_VALUES):
        ax.scatter(sd[:, j], bias[:, j], color=colors[j], s=42,
                   edgecolor="k", lw=0.5, label=f"T={T}", zorder=3)
    ax.plot(sd_grid, bias_pred, color=C_TRUTH, lw=1.4, ls="--",
            label=r"truncated-Gaussian prediction", zorder=2)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(r"empirical sd$[1-\hat\alpha]$")
    ax.set_ylabel(r"bias  mean$[1-\hat\alpha] - (1-\alpha^*)$")
    ax.set_title(rf"B  clipping bias  ($\ell=0.3\sigma_e$,  $\alpha^*$={alpha:.3f})")
    ax.legend(fontsize=7, loc="lower left", ncol=2)


def panel_C_sd_vs_T(ax, data):
    """sd[1-alpha-hat] vs T at fixed N=800, ell=sigma, with the analytical
    floor curve as the load-bearing line."""
    vals = data["sigma"]
    i_N = int(np.argmax(N_VALUES == 800))
    sd = np.nanstd(vals[i_N], axis=1)          # (n_T,)
    T_dense = np.geomspace(15, 500, 200)
    floor_dense = t_floor(SIG, T_dense)
    ax.plot(T_dense, floor_dense, color=C_TRUTH, lw=1.4, ls="--",
            label=r"analytical floor  $\alpha^*\sqrt{2/(T-1)}$")
    ax.errorbar(T_VALUES, sd, fmt="o", color=C_CLOSE, capsize=3,
                ms=6, label=f"empirical (N={N_VALUES[i_N]}, 10 seeds)")
    # Also show the smaller-N curves to expose the within-bin term.
    for ii, N in enumerate(N_VALUES):
        if N == N_VALUES[i_N]:
            continue
        sd_n = np.nanstd(vals[ii], axis=1)
        ax.plot(T_VALUES, sd_n, "o-", color=plt.cm.Greys(0.3 + 0.15 * ii),
                lw=1.0, ms=4, label=f"N={N}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"time bins $T$")
    ax.set_ylabel(r"sd$[1-\hat\alpha]$")
    ax.set_title(r"C  sd vs $T$  ($\ell=\sigma_e$)")
    ax.legend(fontsize=7, loc="upper right")


def main(recompute=False, max_workers=32):
    configure()
    data = _load_or_run(recompute=recompute, max_workers=max_workers)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    panel_A_sd_heatmap(axes[0], data)
    panel_B_bias_vs_sd(axes[1], data)
    panel_C_sd_vs_T(axes[2], data)
    fig.tight_layout()
    save(fig, "fig_consistency.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="re-run the sweep, ignoring the cache")
    parser.add_argument("--workers", type=int, default=32,
                        help="max ProcessPoolExecutor workers")
    args = parser.parse_args()
    main(recompute=args.recompute, max_workers=args.workers)
