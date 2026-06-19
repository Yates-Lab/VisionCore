"""Build per-radius methods-derived caches under progressive fixation masking.

For each session we freeze the geometric median of the baseline-valid eye
positions as the fixation center, then for each radius r in {inf, 1.0, 0.75,
0.5} deg tighten the validity mask to ``valid_mask & (||eyepos-center|| <= r)``
and re-run the UNCHANGED methods pipeline (``decompose_session`` + shuffles,
then ``derive_methods``). r=inf is the baseline (no extra mask) run through the
identical code path -- it must reproduce the existing cache/methods_derived.pkl
and supplies matched per-session attrition counts.

Outputs (under cache/):
    methods_derived_rbase.pkl   (r=inf, baseline)
    methods_derived_r1.0.pkl
    methods_derived_r0.75.pkl
    methods_derived_r0.5.pkl
    masking_attrition.pkl       per (radius, session, window) sample/close-pair
                                counts + per-session fixation geometry

Run: uv run python eyepos_masking/build_masked_caches.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from data_loading import load_cache                                   # noqa: E402
from estimators import _geometric_median                             # noqa: E402
from pipeline import decompose_session, TARGETS_DEFAULT, N_SHUFFLES_DEFAULT  # noqa: E402
from metrics import derive_methods                                   # noqa: E402

CACHE_DIR = METHODS_DIR / "cache"
RADII = [np.inf, 1.0, 0.75, 0.5]
RADIUS_TAGS = {np.inf: "base", 1.0: "1.0", 0.75: "0.75", 0.5: "0.5"}


def _worker_init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _masked_copy(s, center, r):
    """Session copy with valid_mask tightened to the radius-r disk around center.

    Neuron set / rate_hz / psth_r2 / contam_rate / neuron_mask are held fixed
    (only the eye-position sample mask changes)."""
    if not np.isfinite(r):
        return s
    dist = np.linalg.norm(np.asarray(s["eyepos"], float) - center, axis=-1)
    sm = dict(s)
    sm["valid_mask"] = np.asarray(s["valid_mask"], bool) & (dist <= r)
    return sm


def _run_one(s, center, r):
    _worker_init()
    sm = _masked_copy(s, center, r)
    return decompose_session(sm, targets=TARGETS_DEFAULT,
                             n_shuffles=N_SHUFFLES_DEFAULT)


def _fixation_geometry(s, center):
    """Per-session center, offset from fixation target (0,0), and spread."""
    e = np.asarray(s["eyepos"], float)[np.asarray(s["valid_mask"], bool)]
    d = np.linalg.norm(e - center, axis=-1)
    return {
        "center": center,
        "offset": float(np.linalg.norm(center)),
        "sigma_e": float(np.sqrt(np.mean(d ** 2))),   # RMS spread about center
        "n_valid_bins_base": int(s["valid_mask"].sum()),
    }


def main():
    from joblib import Parallel, delayed

    sessions = load_cache()
    print(f"Loaded {len(sessions)} sessions.")
    centers = [_geometric_median(s["eyepos"][s["valid_mask"]]) for s in sessions]
    geom = [_fixation_geometry(s, c) for s, c in zip(sessions, centers)]

    attrition = {"radii": [], "session_names": [s["session"] for s in sessions],
                 "subjects": [s["subject"] for s in sessions],
                 "geometry": geom, "per_radius": {}}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for r in RADII:
        tag = RADIUS_TAGS[r]
        print(f"\n=== radius {tag} (r={r}) : decomposing {len(sessions)} "
              f"sessions ===")
        t0 = time.perf_counter()
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_run_one)(s, c, r) for s, c in zip(sessions, centers)
        )
        print(f"  stage1 {time.perf_counter()-t0:.1f}s")

        windows_ms = [w["window_ms"] for w in results[0]["windows"]]
        windows_bins = [w["window_bins"] for w in results[0]["windows"]]
        derived = derive_methods(results, windows_ms, windows_bins)
        out = CACHE_DIR / f"methods_derived_r{tag}.pkl"
        with open(out, "wb") as f:
            dill.dump(derived, f)
        print(f"  cached -> {out}")

        # per (session, window) sample + close-pair counts straight from stage 1
        per_session = []
        for sr in results:
            wmap = {w["window_bins"]: w for w in sr["windows"]}
            per_session.append({
                "session": sr["session"], "subject": sr["subject"],
                "windows_bins": list(wmap.keys()),
                "n_samples": {wb: wmap[wb]["n_samples"] for wb in wmap},
                "n_close_pairs": {wb: wmap[wb]["n_close_pairs"] for wb in wmap},
            })
        attrition["radii"].append(tag)
        attrition["per_radius"][tag] = {
            "windows_ms": windows_ms, "windows_bins": windows_bins,
            "per_session": per_session,
        }

    with open(CACHE_DIR / "masking_attrition.pkl", "wb") as f:
        dill.dump(attrition, f)
    print(f"\nCached -> {CACHE_DIR / 'masking_attrition.pkl'}")

    _check_baseline_equivalence()


def _check_baseline_equivalence():
    """r=inf baseline must reproduce the existing methods_derived.pkl."""
    base = CACHE_DIR / "methods_derived_rbase.pkl"
    ref = CACHE_DIR / "methods_derived.pkl"
    if not ref.exists():
        print("(no existing methods_derived.pkl to cross-check)")
        return
    with open(base, "rb") as f:
        b = dill.load(f)
    with open(ref, "rb") as f:
        r = dill.load(f)
    print("\n=== baseline (r=inf) vs existing methods_derived.pkl ===")
    worst = 0.0
    for tgt in b["targets"]:
        for w in b["windows_ms"]:
            for key, sub in (("slope_cor", "fano_stats"), ("dz_mean", "nc_stats")):
                vb = b[sub][tgt][w][key]
                vr = r[sub][tgt][w][key]
                worst = max(worst, abs(float(vb) - float(vr)))
    print(f"  max |baseline - existing| over fano slope_cor & nc dz_mean: "
          f"{worst:.2e}  ({'MATCH' if worst < 1e-6 else 'DIFFER'})")


if __name__ == "__main__":
    main()
