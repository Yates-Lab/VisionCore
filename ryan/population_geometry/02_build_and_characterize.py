"""Step 1.2-1.3 — Build the co-centered in-silico population and characterize it.

  1.2  Assemble the population (selected units, readouts re-centered at ROI center)
       and quantify the co-centering perturbation (native vs centered responses).
  1.3  Representativeness: orientation / spatial-frequency tuning, orientation
       selectivity, and a phase-modulation (simple/complex) proxy, via static
       full-field gratings.

Run on a GPU node. Saves a figure + a per-neuron tuning table.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _pop_common import (
    load_twin, select_units, build_centered_population,
    static_response, grating, OUT_DIR, DT,
)

# Tuning grid.
ORIS = np.arange(0, 180, 22.5)                        # 8 orientations
SFS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0])  # cyc/deg
PHASES = np.array([0, 90, 180, 270]) * np.pi / 180.0


def onef_patch(rng, image_shape=(540, 540), beta=1.0):
    """A 1/f^beta natural-like grayscale patch (0..255)."""
    H, W = image_shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0
    amp = 1.0 / (f ** beta)
    phase = rng.uniform(0, 2 * np.pi, (H, W))
    img = np.fft.ifft2(amp * np.exp(1j * phase)).real
    img = (img - img.mean()) / (img.std() + 1e-8)
    return np.clip(127.0 + 40.0 * img, 0, 255).astype(np.float32)


def main():
    model, info, device = load_twin()
    print(f"twin: epoch {info.get('epoch')}  device {device}")

    units = select_units()
    pop = build_centered_population(model, units, center=True)
    pop_native = build_centered_population(model, units, center=False)
    print(f"population N={pop.N}, feature map {pop.feat_size}")
    print(f"co-centering displacement (grid units): "
          f"median={np.median(pop.displacement):.3f}, max={pop.displacement.max():.3f}")

    # --- co-centering perturbation: native vs centered response to natural patch ---
    rng = np.random.default_rng(0)
    patch = onef_patch(rng)
    r_centered = static_response(model, pop, patch)
    r_native = static_response(model, pop_native, patch)
    alive = np.mean(r_centered > 1e-3)
    print(f"fraction of units responsive (>1e-3 Hz) to 1/f patch: {alive:.2f}")

    # --- orientation x SF x phase tuning (static gratings) ---
    n_o, n_s, n_p = len(ORIS), len(SFS), len(PHASES)
    resp = np.zeros((n_o, n_s, n_p, pop.N), dtype=np.float32)
    for io, o in enumerate(ORIS):
        for isf, sf in enumerate(SFS):
            for ip, ph in enumerate(PHASES):
                resp[io, isf, ip] = static_response(model, pop, grating(o, sf, ph))
        print(f"  tuning: orientation {o:.0f} deg done")

    mean_resp = resp.mean(axis=2)               # (n_o, n_s, N) phase-averaged (F0)
    # preferred (ori, sf) per neuron
    flat = mean_resp.reshape(n_o * n_s, pop.N)
    pref = flat.argmax(axis=0)
    pref_o = ORIS[pref // n_s]
    pref_sf = SFS[pref % n_s]
    # orientation selectivity at preferred SF
    osi = np.zeros(pop.N)
    phase_mod = np.zeros(pop.N)                  # simple/complex proxy at pref (o,sf)
    for n in range(pop.N):
        isf = pref[n] % n_s
        io = pref[n] // n_s
        tc = mean_resp[:, isf, n]
        osi[n] = (tc.max() - tc.min()) / (tc.max() + tc.min() + 1e-8)
        pc = resp[io, isf, :, n]
        phase_mod[n] = (pc.max() - pc.min()) / (pc.max() + pc.min() + 1e-8)

    # --- save tuning table ---
    import pandas as pd
    tbl = pd.DataFrame(dict(
        session=pop.session, readout_idx=pop.readout_idx,
        pref_ori=pref_o, pref_sf=pref_sf, osi=osi, phase_mod=phase_mod,
        r_natural=r_centered, displacement=pop.displacement,
    ))
    tbl.to_csv(OUT_DIR / "population_tuning.csv", index=False)

    # --- figure ---
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))

    ax[0, 0].hist(pop.displacement, bins=30, color="slategray")
    ax[0, 0].set_xlabel("native |mean| (grid units)")
    ax[0, 0].set_ylabel("units"); ax[0, 0].set_title("co-centering displacement")

    a = ax[0, 1]
    hz_n, hz_c = r_native / DT, r_centered / DT
    a.scatter(hz_n, hz_c, s=8, alpha=0.4)
    lim = max(hz_n.max(), hz_c.max()) * 1.05
    a.plot([0, lim], [0, lim], "k", lw=0.8)
    rho = np.corrcoef(r_native, r_centered)[0, 1]
    a.set_xlabel("native-position rate (Hz)"); a.set_ylabel("centered rate (Hz)")
    a.set_title(f"centering effect on responses (r={rho:.3f})")

    a = ax[0, 2]
    a.bar(ORIS, [(np.abs(pref_o - o) < 1).sum() for o in ORIS], width=18,
          color="teal")
    a.set_xlabel("preferred orientation (deg)"); a.set_ylabel("units")
    a.set_title("orientation preference")

    a = ax[1, 0]
    a.hist(np.log2(pref_sf), bins=np.log2(np.append(SFS, SFS[-1] * 2)) - 0.25,
           color="darkorange")
    a.set_xticks(np.log2(SFS)); a.set_xticklabels([f"{s:g}" for s in SFS])
    a.set_xlabel("preferred SF (cyc/deg)"); a.set_ylabel("units")
    a.set_title("spatial-frequency preference")

    ax[1, 1].hist(osi, bins=30, color="purple")
    ax[1, 1].set_xlabel("orientation selectivity index"); ax[1, 1].set_ylabel("units")
    ax[1, 1].set_title("orientation selectivity")

    ax[1, 2].hist(phase_mod, bins=30, color="firebrick")
    ax[1, 2].axvline(np.median(phase_mod), color="k", ls="--", lw=1)
    ax[1, 2].set_xlabel("phase modulation (simple<->complex)")
    ax[1, 2].set_ylabel("units")
    ax[1, 2].set_title("simple/complex proxy")

    fig.tight_layout()
    out = OUT_DIR / "population_characterization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    print(f"pref-ori spread: {np.unique(pref_o).size} bins used; "
          f"SF median={np.median(pref_sf):.1f} c/d; OSI median={np.median(osi):.2f}; "
          f"phase-mod median={np.median(phase_mod):.2f}")


if __name__ == "__main__":
    main()
