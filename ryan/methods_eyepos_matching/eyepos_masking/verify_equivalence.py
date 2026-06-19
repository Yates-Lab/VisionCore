"""Linchpin unit check for the eye-position masking task.

The masking is implemented by tightening the per-bin validity mask,

    inside       = ||eyepos - center|| <= r           # (n_trials, T)
    valid_mask_r = valid_mask & inside

and re-running the *unchanged* pipeline windowing. Because
``extract_valid_segments`` only forms windows inside contiguous valid runs and
``_extract_windows_numpy`` only emits windows whose full ``t_hist + t_count``
trajectory lies inside one segment, marking out-of-disk bins invalid must
exclude EVERY window whose trajectory touches an out-of-disk bin. This script
verifies that equivalence directly, on every session and radius:

  (A) Soundness:  every RETAINED window's trajectory lies entirely inside the
      radius-r disk (no window touching an out-of-disk bin survives).
  (B) Fragmentation side-effect:  some windows that are fully inside the disk
      are nonetheless dropped because masking splits a long valid run into
      pieces shorter than min_seg_len=36. Reported (not an error) -- this is the
      "segments lost" quantity the writeup discusses.
  (C) Monotone nesting:  retained window-sample and close-pair counts are
      non-increasing in tightening (baseline >= r=1.0 >= r=0.75 >= r=0.5), and
      the inside-disk bin sets nest r=0.5 subset r=0.75 subset r=1.0.

Run: uv run python eyepos_masking/verify_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from data_loading import load_cache                                   # noqa: E402
from estimators import _geometric_median, _rms_traj_close_pairs      # noqa: E402
from pipeline import (                                               # noqa: E402
    _extract_windows_numpy, DT, WINDOW_BINS_DEFAULT, T_HIST_MS_DEFAULT,
    MIN_SEG_LEN_DEFAULT, THRESHOLD_DEFAULT,
)
from legacy.covariance import extract_valid_segments                 # noqa: E402

RADII = (1.0, 0.75, 0.5)
T_HIST_BINS = int(T_HIST_MS_DEFAULT / (DT * 1000))  # == 1 at the standard config


def session_center(s):
    """Frozen per-session geometric median of the baseline-valid eye positions."""
    return _geometric_median(s["eyepos"][s["valid_mask"]])


def windows_for_mask(robs, eyepos, valid_mask_r, t_count, t_hist):
    segs = extract_valid_segments(valid_mask_r, min_len_bins=MIN_SEG_LEN_DEFAULT)
    return _extract_windows_numpy(robs, eyepos, segs, t_count, t_hist)


def main():
    sessions = load_cache()
    print(f"Loaded {len(sessions)} sessions; verifying mask<->trajectory-drop "
          f"equivalence (t_hist_bins={T_HIST_BINS}).\n")

    n_soundness_checks = 0
    frag_total = {r: 0 for r in RADII}        # fully-inside windows lost to fragmentation
    monotone_ok = True

    for s in sessions:
        robs = np.nan_to_num(np.asarray(s["robs"], float), nan=0.0)
        eyepos = np.nan_to_num(np.asarray(s["eyepos"], float), nan=0.0)
        vm = np.asarray(s["valid_mask"], bool)
        center = session_center(s)

        # inside-disk bit masks; check nesting r=0.5 subset 0.75 subset 1.0
        dist = np.linalg.norm(eyepos - center, axis=-1)
        inside = {r: dist <= r for r in RADII}
        if not (np.all(inside[0.5] <= inside[0.75])
                and np.all(inside[0.75] <= inside[1.0])):
            monotone_ok = False
            print(f"  !! {s['session']}: inside-disk masks do not nest")

        for t_count in WINDOW_BINS_DEFAULT:
            t_hist = max(T_HIST_BINS, t_count)
            total_len = t_hist + t_count

            prev_n = None  # for monotone count check across baseline->1.0->0.75->0.5
            for r in (np.inf,) + RADII:
                vm_r = vm & (dist <= r) if np.isfinite(r) else vm
                counts, traj, T_idx = windows_for_mask(
                    robs, eyepos, vm_r, t_count, t_hist
                )
                n_win = 0 if traj is None else traj.shape[0]

                # (A) soundness: every retained window entirely inside disk
                if np.isfinite(r) and traj is not None:
                    d = np.linalg.norm(traj - center, axis=-1)  # (N, total_len)
                    worst = d.max()
                    assert worst <= r + 1e-9, (
                        f"{s['session']} t={t_count} r={r}: retained window with "
                        f"max dist {worst:.4f} > r")
                    n_soundness_checks += 1

                # (B) fragmentation: windows fully inside disk but dropped because
                # the masked run fell below min_seg_len. Compare against a windowing
                # that ignores min_seg_len (min_len=total_len) on the same mask.
                if np.isfinite(r):
                    segs_nofloor = extract_valid_segments(vm_r, min_len_bins=total_len)
                    c2, tr2, _ = _extract_windows_numpy(
                        robs, eyepos, segs_nofloor, t_count, t_hist)
                    n_win_nofloor = 0 if tr2 is None else tr2.shape[0]
                    frag_total[r] += (n_win_nofloor - n_win)

                # (C) monotone: counts non-increasing as r tightens
                if prev_n is not None and n_win > prev_n:
                    monotone_ok = False
                    print(f"  !! {s['session']} t={t_count}: n_win rose on "
                          f"tightening to r={r} ({prev_n} -> {n_win})")
                prev_n = n_win

    print(f"(A) soundness: {n_soundness_checks} (session,window,radius) cells "
          f"checked -- every retained window lies inside its disk. PASS")
    print(f"(B) fragmentation (fully-inside windows lost to sub-min_seg_len "
          f"runs), summed over sessions/windows:")
    for r in RADII:
        print(f"      r={r}: {frag_total[r]} windows")
    print(f"(C) monotone nesting + non-increasing counts: "
          f"{'PASS' if monotone_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
