"""Step 2 — Content vs drift geometry in the in-silico population.

Drives the co-centered population with P natural backimage patches x R Brownian
drift repeats (drift-consistent behavior), then decomposes the population
covariance (per-timepoint, post-transient) into:
  - content : across-patch covariance of the drift-averaged mean response (base)
  - drift   : within-patch across-repeat covariance (fiber / FEM nuisance)
  - Poisson : private spiking variance
and asks the fiber/base question: is drift variance separable from the content
code (orthogonal) or does it corrupt it (aligned)? Plus patch-decodability vs
integration time (does averaging over the drift fiber recover the base).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _pop_common import (
    load_twin, select_units, build_centered_population, drift_rates,
    get_drift_params, DT, OUT_DIR,
)
from _drift import brownian_drift
from _backimage import load_backimage_patches
from _common import fit_eval_bayes_poisson  # _pop_common put digital-twin-fem on sys.path

P = 150
R = 20
T_FIX = 36
TRANSIENT = 10           # drop onset frames from the covariance window
N_TRAIN = 16             # repeats used for decoder training


def participation_ratio(cov):
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12))


def var_in_subspace(cov, Q):
    return float(np.trace(Q.T @ cov @ Q) / (np.trace(cov) + 1e-12))


def main():
    model, info, device = load_twin()
    units = select_units()
    pop = build_centered_population(model, units)
    N = pop.N
    kappa, vel_scale = get_drift_params(model)
    print(f"kappa={kappa:.4f} deg^2/s, vel_scale={vel_scale:.4f}")

    rng = np.random.default_rng(2)
    patches = load_backimage_patches(P, rng)
    n_p = len(patches)
    Tout = T_FIX + 1
    print(f"driving {n_p} patches x {R} repeats ...")

    rates = np.zeros((n_p, R, Tout, N), dtype=np.float32)
    for p in range(n_p):
        for r in range(R):
            dtrace = brownian_drift(T_FIX, kappa, rng)
            rates[p, r] = drift_rates(model, pop, patches[p], dtrace, vel_scale=vel_scale)
        if p % 25 == 0:
            print(f"  patch {p}/{n_p}")
    # Model output is a per-bin spike count (~0.1/bin ~ 12 Hz), so lambda = rate.
    spikes = rng.poisson(np.clip(rates, 0.0, None)).astype(np.int16)

    # --- covariance decomposition (per-time, averaged over post-transient) ---
    ts = np.arange(TRANSIENT, Tout)
    C_content = np.zeros((N, N))
    C_drift = np.zeros((N, N))
    for t in ts:
        Rt = rates[:, :, t, :]                 # (P, R, N)
        mu = Rt.mean(axis=1)                    # (P, N) drift-averaged
        C_content += np.cov(mu.T)
        dacc = np.zeros((N, N))
        for p in range(n_p):
            dacc += np.cov(Rt[p].T)            # (N, N) across repeats
        C_drift += dacc / n_p
    C_content /= len(ts)     # between-image: scatter of per-image drift-manifold centroids
    C_drift /= len(ts)       # within-image: mean drift-manifold spread
    poisson_diag = rates[:, :, ts, :].mean(axis=(0, 1, 2))  # (N,) mean count = Poisson var

    tot_c, tot_d, tot_p = np.trace(C_content), np.trace(C_drift), poisson_diag.sum()
    pr_c, pr_d = participation_ratio(C_content), participation_ratio(C_drift)
    print(f"between-image (centroid) PR={pr_c:.1f}  within-image (drift) PR={pr_d:.1f}")
    print(f"\nvariance  content={tot_c:.3g}  drift={tot_d:.3g}  poisson={tot_p:.3g}")
    print(f"participation ratio  content={pr_c:.1f}  drift={pr_d:.1f}  (N={N})")

    # drift variance inside the content subspace vs random baseline
    _, evecs = np.linalg.eigh(C_content)       # ascending
    ks = [2, 5, 10, 20]
    drift_in_c, content_in_c, rand_in_c = [], [], []
    for k in ks:
        Q = evecs[:, -k:]
        drift_in_c.append(var_in_subspace(C_drift, Q))
        content_in_c.append(var_in_subspace(C_content, Q))
        rb = np.mean([var_in_subspace(C_drift, np.linalg.qr(
            rng.standard_normal((N, k)))[0]) for _ in range(20)])
        rand_in_c.append(rb)
    for k, dc, cc, rb in zip(ks, drift_in_c, content_in_c, rand_in_c):
        print(f"  top-{k:2d} content subspace: content={cc:.2f}  "
              f"drift={dc:.2f}  (random drift={rb:.2f})")

    # --- decodability vs integration time (plug-in Poisson Bayes on spikes) ---
    ytr = spikes[:, :N_TRAIN].reshape(n_p * N_TRAIN, Tout, N)
    ltr = np.repeat(np.arange(n_p), N_TRAIN)
    yte = spikes[:, N_TRAIN:].reshape(n_p * (R - N_TRAIN), Tout, N)
    lte = np.repeat(np.arange(n_p), R - N_TRAIN)
    windows = [1, 2, 4, 8, 16, T_FIX]
    acc, acc_rate = [], []
    for w in windows:
        a, _ = fit_eval_bayes_poisson(ytr, ltr, yte, lte, t_window=w, n_classes=n_p)
        acc.append(a)
        # noise-free: nearest-centroid on time-averaged RATES (perfect observer)
        Rtr = rates[:, :N_TRAIN, :w, :].mean(axis=(1, 2))        # (P, N) centroids
        Rte = rates[:, N_TRAIN:, :w, :].mean(axis=2)              # (P, R-tr, N)
        d = ((Rte[:, :, None, :] - Rtr[None, None, :, :]) ** 2).sum(-1)  # (P, rte, P)
        pred = d.argmin(-1)
        ar = float((pred == np.arange(n_p)[:, None]).mean())
        acc_rate.append(ar)
        print(f"  decode @ {w:2d} frames ({w*DT*1000:3.0f} ms): "
              f"spikes={a:.3f}  noise-free-rate={ar:.3f}")

    np.savez(OUT_DIR / "content_drift_geometry.npz",
             C_content=C_content, C_drift=C_drift,
             poisson_diag=poisson_diag, pr_c=pr_c, pr_d=pr_d,
             ks=ks, drift_in_c=drift_in_c, content_in_c=content_in_c,
             rand_in_c=rand_in_c, windows=windows, acc=acc, acc_rate=acc_rate,
             kappa=kappa, n_p=n_p, R=R)

    # --- figure ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ec = np.sort(np.linalg.eigvalsh(C_content))[::-1]
    ed = np.sort(np.linalg.eigvalsh(C_drift))[::-1]
    ax[0, 0].semilogy(ec / ec.sum(), "o-", label=f"between-image (PR={pr_c:.0f})", ms=3)
    ax[0, 0].semilogy(ed / ed.sum(), "s-", label=f"within-image drift (PR={pr_d:.0f})", ms=3)
    ax[0, 0].set_xlabel("component"); ax[0, 0].set_ylabel("norm. eigenvalue")
    ax[0, 0].set_title("between- vs within-image spectra"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].bar([0, 1, 2], [tot_c, tot_d, tot_p], color=["teal", "indianred", "gray"])
    ax[0, 1].set_xticks([0, 1, 2])
    ax[0, 1].set_xticklabels(["between-\nimage", "within-\nimage", "Poisson"])
    ax[0, 1].set_ylabel("total variance"); ax[0, 1].set_title("variance budget")

    ax[1, 0].plot(ks, content_in_c, "o-", label="between-image", color="teal")
    ax[1, 0].plot(ks, drift_in_c, "s-", label="within-image drift", color="indianred")
    ax[1, 0].plot(ks, rand_in_c, "^--", label="drift (random)", color="gray")
    ax[1, 0].set_xlabel("between-image subspace dim k")
    ax[1, 0].set_ylabel("frac variance captured")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_title("manifold overlap: drift vs between-image axes")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(np.array(windows) * DT * 1000, acc_rate, "^-", label="noise-free rate")
    ax[1, 1].plot(np.array(windows) * DT * 1000, acc, "o-", label="Poisson spikes")
    ax[1, 1].axhline(1.0 / n_p, color="k", ls=":", lw=1, label="chance")
    ax[1, 1].set_xlabel("integration time (ms)"); ax[1, 1].set_ylabel("patch decode acc")
    ax[1, 1].set_ylim(0, 1); ax[1, 1].set_title(f"decodability ({n_p}-way)")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "content_drift_geometry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
