"""
Shared STA/STE computation and caching for the gaborium stimulus.

Both ``ryan/fig1/rf_contours.py`` (RF contour extraction over all sessions)
and the figure scripts ``generate_fig1c/d/f.py`` need spike-triggered
averages and energies (STA/STE) for the same set of sessions, with the
same parameters. This module centralizes that computation so every caller
hits the same cache and produces bit-identical results when ``recalc`` is
left False.

Cache layout (preserved from the original ``rf_contours.py``):

    CACHE_DIR / "fig1_rf_contours" / "<session_name>_sta_ste.npz"

Each .npz contains ``stas`` (n_units, n_lags, h, w), ``stes`` (same
shape), and ``num_spikes`` (n_units,).

Usage:

    from eval.sta_ste import compute_sta_ste, compute_snr, peak_lag_from_ste

    arrs = compute_sta_ste("Allen_2022-03-04")          # cached
    arrs = compute_sta_ste("Allen_2022-03-04", recalc=True)  # force refresh
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from VisionCore.paths import CACHE_DIR

# Default parameters — match the values used historically by rf_contours.py.
N_LAGS = 20
DT = 1.0 / 240.0
DEVICE = "cuda"
BATCH_SIZE = 10000

CACHE_SUBDIR = CACHE_DIR / "fig1_rf_contours"
CACHE_SUBDIR.mkdir(parents=True, exist_ok=True)


def _cache_path(session_name: str) -> Path:
    return CACHE_SUBDIR / f"{session_name}_sta_ste.npz"


def cache_path(session_name: str) -> Path:
    """Public alias for the npz cache path of a session."""
    return _cache_path(session_name)


def compute_sta_ste(
    session_name: str,
    n_lags: int = N_LAGS,
    recalc: bool = False,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    progress: bool = True,
) -> dict | None:
    """Return cached or freshly computed STAs and STEs for a session.

    Parameters
    ----------
    session_name
        Yates V1 session name, e.g. ``"Allen_2022-03-04"``.
    n_lags
        Number of stimulus lags.
    recalc
        If True, recompute from raw data even when a cache exists.
    device, batch_size, progress
        Passed through to ``DataYatesV1.calc_sta``.

    Returns
    -------
    dict with keys ``stas``, ``stes``, ``num_spikes`` — or ``None`` if the
    session has no gaborium dataset.
    """
    path = _cache_path(session_name)
    if path.exists() and not recalc:
        z = np.load(path)
        return {"stas": z["stas"], "stes": z["stes"],
                "num_spikes": z["num_spikes"]}

    # Imports are local so callers that only need the cache (e.g. fig1c)
    # don't pull torch/DataYatesV1 unless they have to.
    from DataYatesV1 import calc_sta
    from DataYatesV1.utils.io import YatesV1Session

    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        return None

    dset["stim"] = dset["stim"].float()
    dset["stim"] = (dset["stim"] - dset["stim"].mean()) / dset["stim"].std()

    robs = dset["robs"].numpy() if hasattr(dset["robs"], "numpy") else dset["robs"]
    dpi = dset["dpi_valid"].numpy() if hasattr(dset["dpi_valid"], "numpy") else dset["dpi_valid"]
    if dpi.ndim == 1:
        dpi = dpi[:, None]
    num_spikes = (robs * dpi).sum(axis=0)

    stas = calc_sta(
        dset["stim"], dset["robs"], n_lags, dset["dpi_valid"],
        device=device, batch_size=batch_size, progress=progress,
    ).cpu().numpy()
    stes = calc_sta(
        dset["stim"], dset["robs"], n_lags, dset["dpi_valid"],
        device=device, batch_size=batch_size,
        stim_modifier=lambda x: x ** 2, progress=progress,
    ).cpu().numpy()

    np.savez(path, stas=stas, stes=stes, num_spikes=num_spikes)
    return {"stas": stas, "stes": stes, "num_spikes": num_spikes}


def compute_sta_ste_for_sessions(
    session_names,
    n_lags: int = N_LAGS,
    recalc: bool = False,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Compute (or load cached) STA/STE for each session.

    Returns a dict ``{session_name: result_or_None}``.
    """
    out = {}
    for name in session_names:
        out[name] = compute_sta_ste(
            name, n_lags=n_lags, recalc=recalc,
            device=device, batch_size=batch_size,
        )
    return out


def load_cached_sta_ste(session_name: str) -> dict | None:
    """Load STA/STE from cache without ever computing. Returns None if absent."""
    path = _cache_path(session_name)
    if not path.exists():
        return None
    z = np.load(path)
    return {"stas": z["stas"], "stes": z["stes"], "num_spikes": z["num_spikes"]}


def compute_snr(stes: np.ndarray):
    """Per-unit SNR over the noise floor of lag 0.

    Parameters
    ----------
    stes
        Array of shape (n_units, n_lags, h, w).

    Returns
    -------
    cluster_snr : (n_units,) max SNR across lags
    cluster_lag : (n_units,) argmax lag
    snr_per_lag : (n_units, n_lags) full SNR map
    """
    signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 1, 1, 1])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    return snr_per_lag.max(axis=1), snr_per_lag.argmax(axis=1), snr_per_lag


def peak_lag_from_ste(ste_cell: np.ndarray) -> int:
    """Lag at which a single-unit STE has maximum spatial std."""
    return int(ste_cell.std(axis=(1, 2)).argmax())


def population_peak_lag(stes: np.ndarray) -> int:
    """Median across units of the per-unit peak lag (in bins)."""
    lags = [peak_lag_from_ste(stes[u]) for u in range(stes.shape[0])]
    return int(np.median(lags))


def sessions_from_yaml(yaml_path, subjects=None):
    """List ``(session_name, subject)`` tuples drawn from a dataset config YAML."""
    from models.config_loader import load_dataset_configs
    cfgs = load_dataset_configs(str(yaml_path))
    out = []
    for cfg in cfgs:
        name = cfg["session"]
        subject = name.split("_")[0]
        if subjects is None or subject in subjects:
            out.append((name, subject))
    return out
