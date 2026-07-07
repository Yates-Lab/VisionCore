"""Step 1.4-1.6 — Brownian drift + behavior reconstruction: pipeline smoke test.

  - measure the drift diffusion constant kappa from real fixRSVP traces
  - RF-size vs drift-amplitude sanity (does drift meaningfully move the RF?)
  - drive the co-centered population with Brownian drift over natural patches,
    feeding reconstructed drift-consistent behavior
  - verify behavior alignment + that behavior barely changes rates (retinal-only
    vs drift-consistent), as fig4 predicts
  - first H1 geometry check: does the drift-induced response fluctuation live in
    the 2-D gradient (orbit-tangent) subspace?
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _pop_common import (
    load_twin, select_units, build_centered_population, drift_rates,
    _behavior_for_frames, extract_fixrsvp_eye_traces, grating, OUT_DIR, DT, N_LAGS,
)
from _drift import brownian_drift, estimate_kappa

T_FIX = 36            # ~300 ms at 120 Hz
N_TRIALS = 15
N_PATCHES = 6
DX = 0.02            # deg, finite-difference step for tangent vectors


def onef_patch(rng, image_shape=(540, 540), beta=1.0):
    H, W = image_shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2); f[0, 0] = 1.0
    amp = 1.0 / (f ** beta)
    ph = rng.uniform(0, 2 * np.pi, (H, W))
    img = np.fft.ifft2(amp * np.exp(1j * ph)).real
    img = (img - img.mean()) / (img.std() + 1e-8)
    return np.clip(127.0 + 40.0 * img, 0, 255).astype(np.float32)


def var_frac_in_subspace(delta, basis):
    """Fraction of ||delta||^2 captured by the column space of basis."""
    Q, _ = np.linalg.qr(basis)
    proj = delta @ Q
    return float((proj ** 2).sum() / ((delta ** 2).sum() + 1e-12))


def main():
    model, info, device = load_twin()
    units = select_units()
    pop = build_centered_population(model, units, center=True)
    print(f"population N={pop.N}")

    # --- kappa + velocity scale from real fixRSVP drift ---
    traces, dur = extract_fixrsvp_eye_traces(model, fixation_radius_deg=0.5)
    kappa, lags, msd = estimate_kappa(traces, dt=DT, fit_lags=10)
    comps = [np.abs(np.diff(t[:int(np.isfinite(t[:, 0]).sum())], axis=0)).ravel()
             for t in traces if np.isfinite(t[:, 0]).sum() > 2]
    vel_scale = float(np.quantile(np.concatenate(comps), 0.999))
    np.save(OUT_DIR / "vel_scale.npy", np.array(vel_scale))
    print(f"measured kappa = {kappa:.5f} deg^2/s  ({kappa*3600:.1f} arcmin^2/s)")
    print(f"imputed vel_scale (q0.999) = {vel_scale:.5f} deg/frame")

    # --- RF vs drift sanity ---
    drift_rms_300 = np.sqrt(4 * kappa * (T_FIX * DT))     # radial RMS displacement
    px_per_grid = 151.0 / pop.feat_size[0]                # OUT_SIZE / feature grid
    deg_per_grid = px_per_grid / 37.50476617
    tune = None
    try:
        import pandas as pd
        tune = pd.read_csv(OUT_DIR / "population_tuning.csv")
        pref_sf = np.median(tune.pref_sf)
        rf_period_deg = 1.0 / pref_sf
    except Exception:
        rf_period_deg = np.nan
    print(f"drift RMS displacement @300ms = {drift_rms_300:.3f} deg")
    print(f"median RF period (1/pref_SF) = {rf_period_deg:.3f} deg; "
          f"1 feature-grid cell = {deg_per_grid:.3f} deg")
    print(f"drift / RF-period ratio = {drift_rms_300 / rf_period_deg:.2f}")

    # --- drift trials + geometry over patches ---
    rng = np.random.default_rng(1)
    tan_frac, rnd_frac = [], []
    beh_corr, rate_corr = [], []
    example = None
    for pi in range(N_PATCHES):
        patch = onef_patch(rng)
        # tangent vectors: retinal response derivative wrt pure x / y shift
        hold = np.ones((N_LAGS + 4, 1))
        def held(off):  # eye held at a constant offset (steady retinal shift)
            tr = np.tile(np.array(off, np.float32), (N_LAGS + 4, 1))
            return drift_rates(model, pop, patch, tr, vel_scale=None)[-1]
        tx = (held((DX, 0)) - held((-DX, 0))) / (2 * DX)
        ty = (held((0, DX)) - held((0, -DX))) / (2 * DX)
        tangent = np.stack([tx, ty], axis=1)              # (N, 2)

        # drift trials (drift-consistent behavior)
        rr = []
        for ti in range(N_TRIALS):
            dtrace = brownian_drift(T_FIX, kappa, rng)
            rr.append(drift_rates(model, pop, patch, dtrace, vel_scale=vel_scale))
        rr = np.stack(rr)                                 # (n_trials, T+1, N)
        delta = (rr - rr.mean(axis=(0, 1), keepdims=True)).reshape(-1, pop.N)
        tan_frac.append(var_frac_in_subspace(delta, tangent))
        rnd_frac.append(np.mean([
            var_frac_in_subspace(delta, rng.standard_normal((pop.N, 2)))
            for _ in range(20)]))

        # behavior sanity on the first patch: alignment + zero-vs-drift equivalence
        if pi == 0:
            dtrace = brownian_drift(T_FIX, kappa, rng)
            beh = _behavior_for_frames(dtrace, vel_scale).numpy()
            # eye_pos component = last 2 cols; compare to trace (aligned frames 1:)
            ep = beh[1:, -2:]
            beh_corr = [np.corrcoef(ep[:, k], dtrace[:, k])[0, 1] for k in range(2)]
            r_beh = drift_rates(model, pop, patch, dtrace, vel_scale=vel_scale)
            r_zero = drift_rates(model, pop, patch, dtrace, vel_scale=None)
            rate_corr = np.corrcoef(r_beh.ravel(), r_zero.ravel())[0, 1]
            example = (rr[0], tangent, patch)

    tan_frac = np.array(tan_frac); rnd_frac = np.array(rnd_frac)
    print(f"\nbehavior eye_pos alignment corr (x,y): "
          f"{beh_corr[0]:.3f}, {beh_corr[1]:.3f}")
    print(f"drift-consistent vs zero behavior rate corr: {rate_corr:.4f}")
    print(f"drift-fluctuation variance in 2-D tangent subspace: "
          f"{tan_frac.mean():.3f} (random 2-D: {rnd_frac.mean():.3f})")

    # --- figure ---
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax[0, 0].plot(lags * DT * 1000, msd, "o-")
    ax[0, 0].plot(lags * DT * 1000, 4 * kappa * lags * DT, "k--",
                  label=f"4κτ, κ={kappa:.4f}")
    ax[0, 0].set_xlabel("lag (ms)"); ax[0, 0].set_ylabel("MSD (deg²)")
    ax[0, 0].set_title("real drift MSD"); ax[0, 0].legend(fontsize=8)

    ex_rr, ex_tan, _ = example
    ax[0, 1].plot(np.arange(ex_rr.shape[0]) * DT * 1000, ex_rr[:, :12])
    ax[0, 1].set_xlabel("time (ms)"); ax[0, 1].set_ylabel("rate (spk/bin)")
    ax[0, 1].set_title("drift responses (12 units, 1 trial)")

    ax[0, 2].bar([0, 1], [tan_frac.mean(), rnd_frac.mean()],
                 yerr=[tan_frac.std(), rnd_frac.std()], color=["teal", "gray"])
    ax[0, 2].set_xticks([0, 1]); ax[0, 2].set_xticklabels(["gradient\ntangent", "random\n2-D"])
    ax[0, 2].set_ylabel("frac drift variance"); ax[0, 2].set_title("H1: tangent alignment")

    d = brownian_drift(200, kappa, rng)
    ax[1, 0].plot(d[:, 0], d[:, 1], lw=0.8)
    ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlabel("x (deg)"); ax[1, 0].set_ylabel("y (deg)")
    ax[1, 0].set_title("example Brownian drift")

    ax[1, 1].scatter(
        drift_rates(model, pop, example[2], brownian_drift(T_FIX, kappa, rng),
                    vel_scale=vel_scale).ravel(),
        drift_rates(model, pop, example[2], brownian_drift(T_FIX, kappa, rng, x0=(0, 0)),
                    vel_scale=None).ravel(), s=3, alpha=0.2)
    ax[1, 1].set_xlabel("drift-consistent beh (spk/bin)")
    ax[1, 1].set_ylabel("zero beh (spk/bin)")
    ax[1, 1].set_title(f"behavior effect (r≈{rate_corr:.3f})")

    ax[1, 2].scatter(ex_tan[:, 0], ex_tan[:, 1], s=8, alpha=0.5)
    ax[1, 2].set_xlabel("∂r/∂x"); ax[1, 2].set_ylabel("∂r/∂y")
    ax[1, 2].set_title("gradient tangent (per unit)")

    fig.tight_layout()
    out = OUT_DIR / "drift_smoke.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
