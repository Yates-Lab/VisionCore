"""Recompute the Figure 4 spectral-replay predictors with the assay-consistent orientation convention.

Identical to paper/fig4/spatiotemporal_tuning/build_matrix_spectral_replay.py except for one change: spatial modes
are assigned orientations from kxy = (kx, +fy) instead of frequency_grid()'s (kx, -fy), so a mode's orientation uses
the same array-rows-down convention as the grating assay that labels the tuning. validate_direction.py (gates 1-2)
showed the production mapping pairs image orientation 180 - theta with tuning orientation theta.

Writes one archive with every production predictor field for all 40 images x 200 histories x 725 units, in the
production merged image order, plus the unchanged response arrays copied from the production shards. The first
movies are checked against the fold of the direction-resolved cube, which uses the validated assay convention.

Usage (needs DataYatesV1 importable):
    PYTHONPATH=/home/declan/DataYatesV1 .venv/bin/python declan/fig4_direction/corrected_orientation_replay.py
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from VisionCore.paths import STATS_DIR, VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT))
sys.path.insert(0, str(VISIONCORE_ROOT / "declan" / "fig4_direction"))

SOURCE = Path("/home/jake/repos/VisionCore/outputs/no_phase_readout_comparison_20260910/rank1/figure4")
MERGED = SOURCE / "response_matrix_40img_x_200fix/merged"
TUNING = SOURCE / "all_available_yu_tuning"
OUT = STATS_DIR / "fig4_direction" / "corrected_orientation_replay"
PREDICTORS = ("total_dynamic_power", "joint_signed_rate_drive", "joint_passband_power", "tf_marginal_power",
              "sf_orientation_marginal_power", "separable_passband_power")
FR = 240.0
_STATE: dict = {}


def _init():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from paper.fig4.spatiotemporal_tuning.spectral_power import (
        frequency_grid, interpolate_tuning_temporal, load_tuning_tensors, mode_to_grid_matrix,
    )
    import signed_spectra as ss
    tun = load_tuning_tensors(TUNING / "population_tuning_tensors_240hz.npz")
    g = frequency_grid()
    kxy = np.asarray(g["kxy"]).copy()
    kxy_fixed = kxy * np.array([1.0, -1.0])                        # undo the y flip: assay (rows-down) convention
    dist_fixed, _ = mode_to_grid_matrix(kxy_fixed, tun["spatial_cpd"], tun["orientation_deg"])
    dneg, dpos = ss.mode_direction_matrices(kxy, tun["spatial_cpd"])
    with np.load(TUNING / "population_signed_projection_controls_240hz_matrix60.npz") as z:
        controls = {k: z[k] for k in ("tf_marginal", "sf_orientation_marginal", "separable")}
        controls_hz = z["temporal_hz"]
    _STATE.update(tun=tun, flat=np.asarray(g["flat_index"], int), dist=dist_fixed, dneg=dneg, dpos=dpos,
                  controls=controls, controls_hz=controls_hz, interp=interpolate_tuning_temporal)


def _image(args):
    row_idx, image_row = args
    if not _STATE:
        _init()
    from paper.fig4.spatiotemporal_tuning.retinal_replay import render_movies
    from paper.fig4.spatiotemporal_tuning.spectral_power import movie_power_cube, spectral_predictors
    from paper.fig4.upstream.real_trace_matrix.core import extract_patch
    import signed_spectra as ss
    s = _STATE
    traces = np.load(MERGED / "trace_xy.npy").astype(np.float32)
    patch, _ = extract_patch(image_row, canvas_cache={}, patch_size_px=540)
    movies = render_movies(patch, traces, device="cpu")
    out = {k: np.zeros((len(traces), len(s["tun"]["unit_indices"])), np.float32) for k in PREDICTORS}
    check = []
    for t, movie in enumerate(movies):
        hz, cube = movie_power_cube(movie, flat_index=s["flat"], mode_to_grid=s["dist"],
                                    n_spatial=len(s["tun"]["spatial_cpd"]),
                                    n_orientation=len(s["tun"]["orientation_deg"]), frame_rate_hz=FR)
        if "signed" not in s:
            if not np.allclose(hz, s["controls_hz"]):
                raise RuntimeError("temporal grid differs from the production signed controls")
            s["signed"] = s["interp"](s["tun"]["signed_rate_sensitivity"], s["tun"]["temporal_hz"], hz, normalize=False)
            s["passband"] = s["interp"](s["tun"]["phase_rms"], s["tun"]["temporal_hz"], hz, normalize=True)
        if t < 3:   # independent check against the validated direction-cube fold
            _, dcube = ss.movie_direction_cube(movie, flat_index=s["flat"], dist_neg=s["dneg"], dist_pos=s["dpos"],
                                               n_spatial=len(s["tun"]["spatial_cpd"]), frame_rate_hz=FR)
            folded = ss.fold_to_orientation(dcube, s["tun"]["orientation_deg"])
            check.append(float(np.max(np.abs(folded - cube)) / np.max(np.abs(cube))))
        proj = spectral_predictors(cube, s["signed"], s["passband"], signed_controls=s["controls"])
        for k in PREDICTORS:
            out[k][t] = proj[k]
    return row_idx, out, max(check)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(MERGED / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    t0 = time.time()
    results = {}
    with Pool(20, initializer=_init) as pool:
        for row_idx, out, chk in pool.imap_unordered(_image, list(images.iterrows())):
            results[row_idx] = (out, chk)
            print(f"image {len(results)}/{len(images)}  fold-check rel err {chk:.1e}  {time.time() - t0:.0f}s", flush=True)
    order = sorted(results)
    fold_err = max(results[i][1] for i in order)
    if fold_err > 1e-9:
        raise RuntimeError(f"corrected orientation cube disagrees with the direction-cube fold: {fold_err:.2e}")
    corrected = {k: np.stack([results[i][0][k] for i in order]) for k in PREDICTORS}   # [img, trace, unit]
    # production arrays in the same image order, for a paired comparison
    shards = sorted((SOURCE / "matrix_spectral_replay").glob("shard_*/causal_chain_shard.npz"))
    prod = {}
    for p in shards:
        with np.load(p) as z:
            for k in ("image_indices", "mean_rate", "expected_spikes", "map_ssi", *PREDICTORS):
                prod.setdefault(k, []).append(z[k])
            for k in ("trace_indices", "motion_scales", "unit_indices", "spatial_cpd", "temporal_hz", "orientation_deg"):
                prod[k] = z[k]
    idx = np.concatenate(prod["image_indices"])
    perm = np.argsort(idx)
    if not np.array_equal(idx[perm], images.image_index.values):
        raise RuntimeError("image order mismatch against the production shards")
    arrays = {k: np.concatenate(prod[k])[perm] for k in ("mean_rate", "expected_spikes", "map_ssi")}
    production = {k: np.concatenate(prod[k])[perm][:, :, 1] for k in PREDICTORS}
    np.savez_compressed(
        OUT / "corrected_orientation_replay.npz",
        image_indices=images.image_index.values, trace_indices=prod["trace_indices"],
        motion_scales=prod["motion_scales"], unit_indices=prod["unit_indices"], spatial_cpd=prod["spatial_cpd"],
        temporal_hz=prod["temporal_hz"], orientation_deg=prod["orientation_deg"], **arrays,
        **{k: np.stack([np.zeros_like(corrected[k]), corrected[k]], axis=2) for k in PREDICTORS},
        **{f"production_{k}": production[k] for k in PREDICTORS},
    )
    summary = {"n_images": int(len(order)), "fold_check_max_rel_err": fold_err, "seconds": round(time.time() - t0),
               "change": "mode orientation from kxy=(kx,+fy) (assay convention) instead of (kx,-fy)",
               "source_shards": [str(p) for p in shards]}
    for k in PREDICTORS:
        a, b = production[k].astype(float), corrected[k].astype(float)
        same = bool(np.allclose(a, b, rtol=1e-6, atol=0))
        summary[k] = {"identical_to_production": same,
                      "median_abs_rel_change": float(np.median(np.abs(b - a) / np.maximum(np.abs(a), 1e-12)))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
