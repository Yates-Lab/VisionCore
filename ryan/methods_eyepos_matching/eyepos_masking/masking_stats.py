"""Attrition + fixation-geometry tables for the eye-position masking task.

Reads cache/masking_attrition.pkl (sample / close-pair counts straight from the
pipeline stage-1 outputs) and recomputes the geometry-derived quantities
(valid-bin retention, segment loss, and the whole-trajectory-vs-count-bin drop
share) from the aligned cache via the unchanged windowing helpers.

Prints a per-session x radius table and a population roll-up. Saves
cache/masking_geom.pkl for the writeup.

Run: uv run python eyepos_masking/masking_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from data_loading import load_cache                                   # noqa: E402
from estimators import _geometric_median                             # noqa: E402
from pipeline import (                                               # noqa: E402
    _extract_windows_numpy, DT, WINDOW_BINS_DEFAULT, T_HIST_MS_DEFAULT,
    MIN_SEG_LEN_DEFAULT,
)
from legacy.covariance import extract_valid_segments                 # noqa: E402

CACHE_DIR = METHODS_DIR / "cache"
RADII = [1.0, 0.75, 0.5]
T_HIST_BINS = int(T_HIST_MS_DEFAULT / (DT * 1000))


def _geom_drops(s, center, r):
    """Disk-geometry attrition for one (session, radius), isolating the disk
    effect on BASELINE-enumerated windows (segmentation held at valid_mask):

      retained_traj  : windows with the FULL trajectory inside the disk
      retained_count : windows with only the COUNT bins inside the disk
      traj_extra     : retained_count - retained_traj  (windows lost specifically
                       because a non-count history bin left the disk)

    plus segment counts under valid_mask vs the tightened mask.
    """
    robs = np.nan_to_num(np.asarray(s["robs"], float), nan=0.0)
    eyepos = np.nan_to_num(np.asarray(s["eyepos"], float), nan=0.0)
    vm = np.asarray(s["valid_mask"], bool)
    dist = np.linalg.norm(eyepos - center, axis=-1)
    vm_r = vm & (dist <= r)

    segs_base = extract_valid_segments(vm, min_len_bins=MIN_SEG_LEN_DEFAULT)
    segs_r = extract_valid_segments(vm_r, min_len_bins=MIN_SEG_LEN_DEFAULT)

    per_w = {}
    for t_count in WINDOW_BINS_DEFAULT:
        t_hist = max(T_HIST_BINS, t_count)
        _, traj, _ = _extract_windows_numpy(robs, eyepos, segs_base, t_count, t_hist)
        if traj is None:
            per_w[t_count] = dict(n_base=0, ret_traj=0, ret_count=0, traj_extra=0)
            continue
        d = np.linalg.norm(traj - center, axis=-1)            # (N, total_len)
        inside_all = (d <= r).all(axis=1)
        inside_count = (d[:, t_hist:] <= r).all(axis=1)
        per_w[t_count] = dict(
            n_base=int(traj.shape[0]),
            ret_traj=int(inside_all.sum()),
            ret_count=int(inside_count.sum()),
            traj_extra=int(inside_count.sum() - inside_all.sum()),
        )
    return {
        "n_valid_bins_base": int(vm.sum()),
        "n_valid_bins_r": int(vm_r.sum()),
        "n_segments_base": len(segs_base),
        "n_segments_r": len(segs_r),
        "per_window": per_w,
    }


def main():
    sessions = load_cache()
    A = dill.load(open(CACHE_DIR / "masking_attrition.pkl", "rb"))
    centers = [_geometric_median(s["eyepos"][s["valid_mask"]]) for s in sessions]

    geom = {r: [_geom_drops(s, c, r) for s, c in zip(sessions, centers)]
            for r in RADII}

    # ---- fixation geometry table ----
    print("\n## Fixation geometry (per session)\n")
    print(f"{'session':22s} {'subj':6s} {'offset':>7s} {'sigma_e':>8s} "
          f"{'valid':>7s} {'%@1.0':>6s} {'%@.75':>6s} {'%@.5':>6s}")
    for i, s in enumerate(sessions):
        g = A["geometry"][i]
        fr = {r: geom[r][i]["n_valid_bins_r"] / g["n_valid_bins_base"]
              for r in RADII}
        print(f"{s['session']:22s} {s['subject']:6s} {g['offset']:7.3f} "
              f"{g['sigma_e']:8.3f} {g['n_valid_bins_base']:7d} "
              f"{100*fr[1.0]:6.1f} {100*fr[0.75]:6.1f} {100*fr[0.5]:6.1f}")
    off = np.array([g["offset"] for g in A["geometry"]])
    sig = np.array([g["sigma_e"] for g in A["geometry"]])
    print(f"\npopulation offset: median {np.median(off):.3f} "
          f"(IQR {np.percentile(off,25):.3f}-{np.percentile(off,75):.3f}) deg; "
          f"sigma_e median {np.median(sig):.3f} deg")

    # ---- attrition roll-up (population, per window) ----
    WB = list(WINDOW_BINS_DEFAULT)
    ms = {1: 8.3, 2: 16.7, 3: 25.0, 6: 50.0}

    def tot(tag, field, wb):
        return sum(ps[field][wb] for ps in A["per_radius"][tag]["per_session"]
                   if wb in ps[field])

    print("\n## Sample / close-pair attrition (population, per window)\n")
    print(f"{'win':>6s} {'radius':>7s} {'samples':>9s} {'%drop':>6s} "
          f"{'clpairs':>9s} {'%drop':>6s} {'trajExtra%':>10s}")
    for wb in WB:
        base_s = tot("base", "n_samples", wb)
        base_c = tot("base", "n_close_pairs", wb)
        for tag, r in [("base", None), ("1.0", 1.0), ("0.75", 0.75), ("0.5", 0.5)]:
            s_ = tot(tag, "n_samples", wb)
            c_ = tot(tag, "n_close_pairs", wb)
            ds = 100 * (1 - s_ / base_s)
            dc = 100 * (1 - c_ / base_c)
            if r is None:
                te = float("nan")
            else:
                rt = sum(geom[r][i]["per_window"][wb]["ret_traj"]
                         for i in range(len(sessions)))
                rc = sum(geom[r][i]["per_window"][wb]["ret_count"]
                         for i in range(len(sessions)))
                te = 100 * (rc - rt) / max(rc, 1)  # share of count-inside windows
                # additionally lost because a history bin left the disk
            print(f"{ms[wb]:6.1f} {tag:>7s} {s_:9d} {ds:6.1f} "
                  f"{c_:9d} {dc:6.1f} {te:10.2f}")

    # ---- segment loss ----
    print("\n## Segment loss (segments >= min_seg_len=36 surviving the mask)\n")
    print(f"{'radius':>7s} {'segs_base':>10s} {'segs_r':>8s} {'%lost':>6s}")
    nseg_base = sum(g["n_segments_base"] for g in geom[1.0])
    for r in RADII:
        nseg_r = sum(g["n_segments_r"] for g in geom[r])
        print(f"{r:7.2f} {nseg_base:10d} {nseg_r:8d} "
              f"{100*(1-nseg_r/nseg_base):6.1f}")

    with open(CACHE_DIR / "masking_geom.pkl", "wb") as f:
        dill.dump({"geom": geom, "radii": RADII}, f)
    print(f"\nCached -> {CACHE_DIR / 'masking_geom.pkl'}")


if __name__ == "__main__":
    main()
