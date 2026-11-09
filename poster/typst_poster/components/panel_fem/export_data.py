"""Freeze the data behind the poster's intro and "sub-RF gaze shift" panels.

Provenance step only; the poster build reads the saved ``plot_data.npz`` and
never runs this. It reuses ``paper/fig1/generate_fem_overview.py`` (same
session, trials, gaze window, population RF and image), so the poster panel and
the standalone R01 figure stay identical in content.

Run from the repository root with the analysis environment:
    uv run python poster/typst_poster/components/panel_fem/export_data.py
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from skimage import measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "paper" / "fig1"))
import generate_fem_overview as g  # noqa: E402

# Rasters and PSTH cover exactly the grouping window (see generate_fem_overview).
INTRO_TRIAL = 72          # 1 s fixation with drift and one small microsaccade


def single_unit_rf_diameters(payload, assets):
    """Equivalent diameters (deg) of single-unit RFs passing Figure 1 criteria."""
    z = np.load(g.STE_PATH)
    stes, spikes = z["stes"], z["num_spikes"]
    sig = gaussian_filter(np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True)),
                          [0, 1, 1, 1])
    snr_lag = sig.max(axis=(2, 3)) / np.median(sig[:, 0], axis=(1, 2))[:, None]
    snr, lag = snr_lag.max(1), snr_lag.argmax(1)
    out = []
    for u in payload["cids"]:
        if snr[u] <= g.SNR_THRESH or spikes[u] <= g.SPIKE_THRESH:
            continue
        im = stes[u, lag[u]] - np.median(stes[u, lag[u]])
        if im.max() < abs(im.min()) or np.ptp(im) < 1e-8:
            continue
        c = max(measure.find_contours((im - im.min()) / np.ptp(im), 0.5), key=len)
        if len(c) < 3:
            continue
        hull = ConvexHull(c)
        h = c[hull.vertices]
        perimeter = np.linalg.norm(np.diff(np.vstack([h, h[:1]]), axis=0), axis=1).sum()
        diameter = max(np.linalg.norm(a - b) for a in h for b in h)
        if perimeter / (np.pi * diameter) < 0.9:
            continue
        out.append(2 * np.sqrt(hull.volume / np.pi) / assets["gaborium_ppd"])
    return np.asarray(out)


def main():
    payload = g.load_payload()
    assets = g.load_assets(payload)
    low, high = g.selected_trials(payload)
    s, e = g.clustering_window(payload)
    t0, t1 = s, e + g.RASTER_PAD_BINS

    spk_t, spk_u, spk_k = [], [], []
    for k, trial in enumerate(high + low):         # red trials 1..n, then blue 1..n
        times, units = g.trial_spikes(payload, trial, t0, t1)
        spk_t.append(times), spk_u.append(units), spk_k.append(np.full(times.size, k))
    psth_t, psth_hi = g.group_psth(payload, high, t0, t1)
    _, psth_lo = g.group_psth(payload, low, t0, t1)

    image_id, _ = g.displayed_image(payload, assets, low, high)
    rf, n_rf = g.population_rf(payload, assets)
    gaze_hi, gaze_lo = g.trial_gaze(payload, high, s, e), g.trial_gaze(payload, low, s, e)
    diam = single_unit_rf_diameters(payload, assets)
    trace = payload["eyepos"][INTRO_TRIAL]
    trace = trace[~np.isnan(trace).any(axis=1)]

    np.savez_compressed(
        HERE / "plot_data.npz",
        image=assets[f"im_{image_id}"], image_half_deg=float(assets[f"half_{image_id}"]),
        gaze_hi=gaze_hi, gaze_lo=gaze_lo, rf=rf, n_rf_units=n_rf,
        spk_t=np.concatenate(spk_t), spk_u=np.concatenate(spk_u),
        spk_k=np.concatenate(spk_k), n_units=payload["robs"].shape[2], n_per=len(high),
        raster_ms=np.array([t0, t1]) * g.DT * 1000, window_ms=np.array([s, e]) * g.DT * 1000,
        psth_t=psth_t, psth_hi=psth_hi, psth_lo=psth_lo,
        gaze_shift_deg=float(np.hypot(*(gaze_hi.mean(0) - gaze_lo.mean(0)))),
        rf_diam_deg=diam, intro_trace_deg=trace, intro_dt=g.DT,
        intro_all_deg=payload["eyepos"].astype(np.float32),
    )
    # assets/marmoset.png is the canonical line drawing (retired panel-1 artwork).
    (HERE / "assets").mkdir(exist_ok=True)
    Image.open(g.DPI_IMAGE).convert("L").save(HERE / "assets" / "dpi_eye.png")
    print(f"trials red={high} blue={low}; {len(diam)} single-unit RFs, median "
          f"{np.median(diam):.3f} deg; shift {np.hypot(*(gaze_hi.mean(0) - gaze_lo.mean(0))):.3f}"
          f" deg; intro trace {len(trace)} samples")


if __name__ == "__main__":
    main()
