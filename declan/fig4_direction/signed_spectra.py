"""Direction-resolved (signed-TF) movie spectra and direction-resolved passbands.

The production spectral replay (paper/fig4/spatiotemporal_tuning/spectral_power.py) folds +TF and -TF
together and maps spatial modes k and -k to one orientation, so it cannot tell motion along a unit's
preferred direction from motion the opposite way. This module keeps the sign.

Conventions (derived, then checked by validate_direction.py):
  * Assay direction theta (run_exact_cid_drifting_tuning.drifting_histories): motion along
    (cos theta, sin theta) in array coordinates, x = column (right), y = row (DOWN).
  * np.fft.fft2 over (row, col) puts a grating exp(+2 pi i SF (cos theta x + sin theta y)) at
    (fx, fy) = SF (cos theta, sin theta); frequency_grid() stores kxy = (fx, -fy).
  * Its time course exp(-2 pi i TF t) lands at temporal frequency f = -TF (np.fft sign convention).
  So a mode with kxy = (kx, ky) at f < 0 moves along assay direction atan2(-ky, kx); at f > 0 the
  opposite direction. Opposite (k, f) pairs are Hermitian mirrors and map to the same direction.

Direction passband: the production passband is separable, Yu SF x TF prediction x orientation weight,
where the orientation weight is the mean clipped dF0 of the two opposite directions sharing a bar
orientation (build_exact_cid_figure4_contract._grouped_tuning). Here each orientation's weight is split
between its two directions in proportion to their clipped dF0, so pb_dir[d] + pb_dir[d + 180] equals
twice the production value exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal.windows import dpss, tukey

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT))
from paper.fig4.spatiotemporal_tuning.audit_exact_cid_drifting_tuning import (  # noqa: E402
    preferred_direction, response_cube,
)
from paper.fig4.spatiotemporal_tuning.spectral_power import (  # noqa: E402
    EPS, log_interpolation_weights,
)

DIRECTIONS_DEG = np.arange(0.0, 360.0, 20.0)


def signed_dpss_mode_power(selected: np.ndarray, frame_rate_hz: float, *, nw: float = 1.5, n_tapers: int = 2):
    """DPSS power per mode at each positive |TF|, split by temporal-frequency sign.

    Returns (positive_hz, power_neg, power_pos), each power [mode, len(positive_hz)]. The Nyquist bin has no
    sign and is split equally, so power_neg + power_pos equals spectral_power.folded_dpss_mode_power.
    """
    value = np.asarray(selected, dtype=np.complex128)
    value = value - value.mean(axis=1, keepdims=True)
    n_time = value.shape[1]
    signed_hz = np.fft.fftfreq(n_time, d=1.0 / float(frame_rate_hz))
    positive_hz = np.fft.rfftfreq(n_time, d=1.0 / float(frame_rate_hz))[1:]
    tapers = dpss(n_time, NW=float(nw), Kmax=int(n_tapers), sym=False)
    raw = np.zeros((len(value), n_time), dtype=np.float64)
    for taper in tapers:
        raw += np.square(np.abs(np.fft.fft(value * taper[None], axis=1, norm="ortho")))
    raw /= len(tapers)
    neg = np.zeros((len(value), len(positive_hz)))
    pos = np.zeros_like(neg)
    for j, hz in enumerate(positive_hz):
        p = np.flatnonzero(np.isclose(signed_hz, hz))
        n = np.flatnonzero(np.isclose(signed_hz, -hz))
        if len(p) and len(n):
            pos[:, j], neg[:, j] = raw[:, p].sum(1), raw[:, n].sum(1)
        else:                                   # Nyquist: one unsigned bin
            both = raw[:, np.flatnonzero(np.isclose(np.abs(signed_hz), hz))].sum(1)
            pos[:, j] = neg[:, j] = 0.5 * both
    return positive_hz, neg, pos


def circular_direction_weights(angle_deg: np.ndarray, directions_deg: np.ndarray = DIRECTIONS_DEG):
    step = float(np.diff(directions_deg)[0])
    if not np.isclose(step * len(directions_deg), 360.0):
        raise ValueError("directions must tile 360 degrees uniformly")
    position = np.mod(np.asarray(angle_deg, float) - directions_deg[0], 360.0) / step
    lower = np.floor(position).astype(int) % len(directions_deg)
    frac = position - np.floor(position)
    return lower, (lower + 1) % len(directions_deg), 1.0 - frac, frac


def mode_direction_matrices(kxy: np.ndarray, spatial_cpd: np.ndarray, directions_deg: np.ndarray = DIRECTIONS_DEG):
    """Sparse [mode, sf * n_dir] distributors for negative-TF and positive-TF power."""
    radial = np.linalg.norm(kxy, axis=1)
    sf0, sf1, sw0, sw1, resolved = log_interpolation_weights(radial, spatial_cpd)
    along_neg = np.degrees(np.arctan2(-kxy[:, 1], kxy[:, 0]))          # f < 0 moves along this direction
    out = []
    for angle in (along_neg, along_neg + 180.0):
        d0, d1, dw0, dw1 = circular_direction_weights(angle, directions_deg)
        rows, cols, vals = [], [], []
        mode = np.arange(len(kxy))
        for si, sw in ((sf0, sw0), (sf1, sw1)):
            for di, dw in ((d0, dw0), (d1, dw1)):
                w = sw * dw * resolved
                keep = w > 0
                rows.append(mode[keep]); cols.append(si[keep] * len(directions_deg) + di[keep]); vals.append(w[keep])
        out.append(sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(len(kxy), len(spatial_cpd) * len(directions_deg))).tocsr())
    return out[0], out[1]


def movie_direction_cube(movie, *, flat_index, dist_neg, dist_pos, n_spatial, frame_rate_hz,
                         directions_deg: np.ndarray = DIRECTIONS_DEG):
    """SF x |TF| x direction power of a chronological [time, y, x] movie (same preprocessing as production)."""
    value = np.asarray(movie, dtype=np.float64)
    value = (value - 127.0) / 255.0
    value -= value.mean(axis=0, keepdims=True)
    w1 = tukey(value.shape[1], alpha=0.15, sym=False)
    coef = np.fft.fft2(value * np.outer(w1, w1)[None], axes=(-2, -1), norm="ortho")
    selected = coef.reshape(len(value), -1)[:, flat_index].T
    hz, p_neg, p_pos = signed_dpss_mode_power(selected, frame_rate_hz)
    flat = dist_neg.T @ p_neg + dist_pos.T @ p_pos
    cube = np.asarray(flat).reshape(n_spatial, len(directions_deg), len(hz)).transpose(0, 2, 1)
    return hz, cube


def bar_orientation_of_direction(directions_deg: np.ndarray = DIRECTIONS_DEG) -> np.ndarray:
    """Assay rule (run_exact_cid_drifting_tuning._conditions): bar = (direction + 90) mod 180."""
    return np.mod(directions_deg + 90.0, 180.0)


def fold_to_orientation(cube_dir: np.ndarray, orientations_deg: np.ndarray, directions_deg=DIRECTIONS_DEG):
    """Sum each orientation's two directions: [..., n_dir] -> [..., n_ori] in the assay's bar convention."""
    bars = bar_orientation_of_direction(directions_deg)
    out = np.zeros((*cube_dir.shape[:-1], len(orientations_deg)))
    for o, ori in enumerate(orientations_deg):
        out[..., o] = cube_dir[..., np.isclose(bars, ori)].sum(-1)
    return out


def direction_weights_from_assay(measurement_dir: Path, source_unit_index: np.ndarray):
    """Per-unit clipped dF0 across directions at the unit's peak SF x TF (the production orientation-weight rule).

    Returns weights [unit, n_dir], preferred direction index [unit], directions."""
    conditions = pd.read_csv(measurement_dir / "conditions.csv")
    with np.load(measurement_dir / "responses.npz", allow_pickle=False) as z:
        delta = z["delta_f0_expected_count"]
    sf, tf, directions, cube = response_cube(conditions, delta)
    dynamic = tf > 0
    weights = np.zeros((len(source_unit_index), len(directions)))
    pref = np.zeros(len(source_unit_index), dtype=int)
    for k, s in enumerate(source_unit_index):
        unit = cube[..., int(s)]
        di = preferred_direction(unit, dynamic)
        surface = np.maximum(unit[:, dynamic, di], 0.0).T               # [tf, sf], as in the contract
        peak_tf, peak_sf = np.unravel_index(int(np.argmax(surface)), surface.shape)
        weights[k] = np.maximum(unit[peak_sf, np.flatnonzero(dynamic)[peak_tf]], 0.0)
        pref[k] = di
    return weights, pref, directions


def direction_passband(orientation_passband: np.ndarray, orientations_deg: np.ndarray, dir_weights: np.ndarray,
                       directions_deg=DIRECTIONS_DEG) -> np.ndarray:
    """[unit, sf, tf, ori] production passband -> [unit, sf, tf, dir]; each orientation's value is split between
    its two directions in proportion to their weights (equally when both are zero). Sum over a direction pair
    equals twice the orientation value."""
    bars = bar_orientation_of_direction(directions_deg)
    out = np.zeros((*orientation_passband.shape[:3], len(directions_deg)))
    for o, ori in enumerate(orientations_deg):
        pair = np.flatnonzero(np.isclose(bars, ori))
        w = dir_weights[:, pair]
        tot = w.sum(1, keepdims=True)
        share = np.where(tot > 0, w / np.maximum(tot, EPS), 0.5)       # [unit, 2]
        for j, d in enumerate(pair):
            out[..., d] = 2.0 * orientation_passband[..., o] * share[:, j][:, None, None]
    return out
