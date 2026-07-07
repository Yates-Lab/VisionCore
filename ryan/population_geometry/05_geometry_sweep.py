"""Step 2 (full) — drift-manifold geometry: kappa sweep + luminance control.

Rate-level analysis (Poisson spiking noise is averaged out by population
redundancy, so the rate is the accessible signal). For each drift amplitude
kappa we characterise the per-image drift manifolds:

  - between-image covariance : scatter of per-image manifold centroids (content)
  - within-image covariance  : mean drift-manifold spread (FEM)
  - manifold overlap         : within-image variance inside the top-k between-image
                               subspace (do drift and content share axes?)
  - decodability             : noise-free nearest-centroid image ID vs integration time

Luminance control (in-domain, no extra forwards): is the dominant mode a global
luminance gain, and does the drift/content overlap survive projecting it out?

Everything stays in-domain (all stimuli drift); drift amplitude is the knob,
replacing the invalid static control.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _pop_common import (
    load_twin, select_units, build_centered_population, drift_rates,
    get_drift_params, OUT_SIZE, DT, OUT_DIR,
)
from _drift import brownian_drift
from _backimage import load_backimage_patches

P = 250
R = 20
T_FIX = 36
TRANSIENT = 10
KAPPA_FACTORS = [0.25, 0.5, 1.0]
DEC_WINDOWS = [4, 8, 16, 36]


def participation_ratio(cov):
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12))


def var_in_subspace(cov, Q):
    return float(np.trace(Q.T @ cov @ Q) / (np.trace(cov) + 1e-12))


def central_luminance(patch, crop=OUT_SIZE[0]):
    H, W = patch.shape
    h = crop // 2
    roi = patch[H // 2 - h:H // 2 + h, W // 2 - h:W // 2 + h]
    return float(roi.mean()), float(roi.std())


def nearest_centroid_decode(Ravg, n_train):
    """Ravg: (P, R, N) per-trial time-averaged rates. Returns accuracy."""
    n_p = Ravg.shape[0]
    C = Ravg[:, :n_train].mean(axis=1)                      # (P, N) centroids
    Xte = Ravg[:, n_train:]                                 # (P, R-tr, N)
    d = ((Xte[:, :, None, :] - C[None, None, :, :]) ** 2).sum(-1)  # (P, rte, P)
    pred = d.argmin(-1)
    return float((pred == np.arange(n_p)[:, None]).mean())


def geometry_for_kappa(model, pop, patches, kappa, vel_scale, rng):
    n_p, N, Tout = len(patches), pop.N, T_FIX + 1
    ts = np.arange(TRANSIENT, Tout)
    rates = np.zeros((n_p, R, Tout, N), dtype=np.float32)
    for p in range(n_p):
        for r in range(R):
            dtrace = brownian_drift(T_FIX, kappa, rng)
            rates[p, r] = drift_rates(model, pop, patches[p], dtrace, vel_scale=vel_scale)
        if p % 50 == 0:
            print(f"    patch {p}/{n_p}")

    # drift-manifold samples per image = (repeat x post-transient time)
    X = rates[:, :, ts, :].reshape(n_p, R * len(ts), N)     # (P, M, N)
    mu = X.mean(axis=1)                                     # (P, N) centroids
    C_content = np.cov(mu.T)
    C_drift = np.mean([np.cov(X[p].T) for p in range(n_p)], axis=0)

    pr_c, pr_d = participation_ratio(C_content), participation_ratio(C_drift)
    _, evecs = np.linalg.eigh(C_content)
    overlap = {k: var_in_subspace(C_drift, evecs[:, -k:]) for k in (2, 5, 10)}

    # decode vs integration time (noise-free rate, nearest centroid)
    dec = {}
    for w in DEC_WINDOWS:
        Ravg = rates[:, :, TRANSIENT:TRANSIENT + w, :].mean(axis=2)  # (P,R,N)
        dec[w] = nearest_centroid_decode(Ravg, R // 2)

    # --- luminance control: is the top mode a global gain? ---
    v_top = evecs[:, -1]
    sign_frac = float(max((v_top > 0).mean(), (v_top < 0).mean()))  # 1.0 => uniform sign
    lum = np.array([central_luminance(p)[0] for p in patches])
    proj_top = mu @ v_top
    lum_corr = float(np.corrcoef(lum, proj_top)[0, 1])
    # project out the top global mode, recompute overlap in the residual
    Pp = np.eye(N) - np.outer(v_top, v_top)
    Cc_res = Pp @ C_content @ Pp
    Cd_res = Pp @ C_drift @ Pp
    _, ev_res = np.linalg.eigh(Cc_res)
    overlap_res = {k: var_in_subspace(Cd_res, ev_res[:, -k:]) for k in (2, 5, 10)}

    return dict(kappa=kappa, pr_c=pr_c, pr_d=pr_d, overlap=overlap,
                overlap_res=overlap_res, dec=dec, tot_c=float(np.trace(C_content)),
                tot_d=float(np.trace(C_drift)), sign_frac=sign_frac,
                lum_corr=lum_corr, lum=lum, proj_top=proj_top,
                ec=np.sort(np.linalg.eigvalsh(C_content))[::-1],
                ed=np.sort(np.linalg.eigvalsh(C_drift))[::-1])


def main():
    model, info, device = load_twin()
    pop = build_centered_population(model, select_units())
    kappa0, vel_scale = get_drift_params(model)
    print(f"measured kappa={kappa0:.4f}, vel_scale={vel_scale:.4f}, N={pop.N}")

    rng = np.random.default_rng(3)
    patches = load_backimage_patches(P, rng)
    print(f"{len(patches)} patches")

    results = []
    for f in KAPPA_FACTORS:
        k = kappa0 * f
        print(f"  kappa factor {f} (kappa={k:.4f}) ...")
        results.append(geometry_for_kappa(model, pop, patches, k, vel_scale, rng))

    for r in results:
        rms = np.sqrt(4 * r["kappa"] * T_FIX * DT)
        print(f"\nkappa={r['kappa']:.4f} (RMS {rms:.3f} deg): "
              f"content PR={r['pr_c']:.1f} drift PR={r['pr_d']:.1f}  "
              f"overlap top5={r['overlap'][5]:.2f} (resid {r['overlap_res'][5]:.2f})  "
              f"decode@300ms={r['dec'][36]:.2f}  "
              f"top-mode signfrac={r['sign_frac']:.2f} lum_corr={r['lum_corr']:.2f}")

    np.savez(OUT_DIR / "geometry_sweep.npz",
             kappas=[r["kappa"] for r in results],
             pr_c=[r["pr_c"] for r in results], pr_d=[r["pr_d"] for r in results],
             overlap5=[r["overlap"][5] for r in results],
             overlap5_res=[r["overlap_res"][5] for r in results],
             dec36=[r["dec"][36] for r in results],
             lum_corr=[r["lum_corr"] for r in results],
             sign_frac=[r["sign_frac"] for r in results])

    # --- figure ---
    ks = np.array([r["kappa"] for r in results])
    rms = np.sqrt(4 * ks * T_FIX * DT)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    ax[0, 0].plot(rms, [r["pr_c"] for r in results], "o-", label="between-image")
    ax[0, 0].plot(rms, [r["pr_d"] for r in results], "s-", label="within-image drift")
    ax[0, 0].set_xlabel("drift RMS @300ms (deg)"); ax[0, 0].set_ylabel("participation ratio")
    ax[0, 0].set_title("dimensionality vs drift"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylim(bottom=0)

    ax[0, 1].plot(rms, [r["overlap"][5] for r in results], "o-", label="raw")
    ax[0, 1].plot(rms, [r["overlap_res"][5] for r in results], "s--",
                  label="global mode removed")
    ax[0, 1].set_xlabel("drift RMS @300ms (deg)")
    ax[0, 1].set_ylabel("drift var in top-5 content subspace")
    ax[0, 1].set_ylim(0, 1); ax[0, 1].set_title("manifold overlap vs drift")
    ax[0, 1].legend(fontsize=8)

    for r in results:
        ax[0, 2].plot(np.array(DEC_WINDOWS) * DT * 1000, [r["dec"][w] for w in DEC_WINDOWS],
                      "o-", label=f"RMS {np.sqrt(4*r['kappa']*T_FIX*DT):.2f}°")
    ax[0, 2].set_xlabel("integration time (ms)"); ax[0, 2].set_ylabel("decode acc")
    ax[0, 2].set_ylim(0, 1); ax[0, 2].set_title(f"decodability ({len(patches)}-way)")
    ax[0, 2].legend(fontsize=8, title="drift")

    rlast = results[-1]
    ax[1, 0].semilogy(rlast["ec"] / rlast["ec"].sum(), "o-", ms=3, label="between-image")
    ax[1, 0].semilogy(rlast["ed"] / rlast["ed"].sum(), "s-", ms=3, label="within-image")
    ax[1, 0].set_xlabel("component"); ax[1, 0].set_ylabel("norm. eigenvalue")
    ax[1, 0].set_title(f"spectra (kappa={rlast['kappa']:.3f})"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].scatter(rlast["lum"], rlast["proj_top"], s=10, alpha=0.5)
    ax[1, 1].set_xlabel("patch central luminance")
    ax[1, 1].set_ylabel("projection on top content mode")
    ax[1, 1].set_title(f"top mode vs luminance (r={rlast['lum_corr']:.2f}, "
                       f"signfrac={rlast['sign_frac']:.2f})")

    ax[1, 2].plot(rms, [abs(r["lum_corr"]) for r in results], "o-",
                  label="|corr| top-mode vs lum")
    ax[1, 2].plot(rms, [r["sign_frac"] for r in results], "s-", label="top-mode sign fraction")
    ax[1, 2].set_xlabel("drift RMS @300ms (deg)"); ax[1, 2].set_ylim(0, 1)
    ax[1, 2].set_title("global-gain diagnostics"); ax[1, 2].legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "geometry_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
