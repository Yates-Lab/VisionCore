"""Quick utility: render every lag of the STA for the fig1d cell into a grid.

Usage:
    uv run ryan/fig1/_sta_all_lags.py
"""
import numpy as np
import matplotlib.pyplot as plt

from VisionCore.paths import FIGURES_DIR, CACHE_DIR
from generate_fig1d import (
    SUBJECT, DATE, DEFAULT_CELL, RF_CACHE_DIR,
    load_cell_payload, _load_gaborium_geometry, _gaborium_row_for_cluster,
)


def main(subject=SUBJECT, date=DATE, cell=DEFAULT_CELL):
    payload = load_cell_payload(subject, date, cell)
    session = payload["session"]
    row = _gaborium_row_for_cluster(session, cell)

    path = RF_CACHE_DIR / f"{session}_sta_ste.npz"
    z = np.load(path)
    stas = z["stas"][row]   # (n_lags, h, w)
    stes = z["stes"][row]
    n_lags, h, w = stas.shape

    ppd, roi_origin = _load_gaborium_geometry(session)
    # centroid based on per-lag peak (use lag with max STE std for centering)
    peak_lag_ste = int(stes.std(axis=(1, 2)).argmax())
    ref = stas[peak_lag_ste] - np.median(stas[peak_lag_ste])
    w_ref = np.abs(ref)
    rg, cg = np.indices(ref.shape)
    if w_ref.sum() > 0:
        cr = (rg * w_ref).sum() / w_ref.sum()
        cc = (cg * w_ref).sum() / w_ref.sum()
    else:
        cr = (h - 1) / 2.0
        cc = (w - 1) / 2.0
    extent = ((-0.5 - cc) / ppd, (w - 0.5 - cc) / ppd,
              (cr - h + 0.5) / ppd, (cr + 0.5) / ppd)

    # Symmetric vmax shared across lags for fair comparison.
    centered_all = stas - np.median(stas, axis=(1, 2), keepdims=True)
    vmax = float(np.nanmax(np.abs(centered_all)))
    if vmax == 0:
        vmax = 1.0

    ncols = 5
    nrows = int(np.ceil(n_lags / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows),
                              constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n_lags:
            ax.axis("off")
            continue
        img = centered_all[i]
        ax.imshow(img, extent=extent, origin="upper",
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        marker = " *" if i == peak_lag_ste else ""
        ax.set_title(f"lag {i}{marker}", fontsize=9)

    fig.suptitle(f"{session}  cell {cell}  STA all lags (* = STE peak)",
                 fontsize=10)
    out_dir = FIGURES_DIR / "fig1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig1d_sta_all_lags.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
