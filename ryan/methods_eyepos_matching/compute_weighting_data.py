r"""Extension-1 real-data driver: how MIXING time-bin weightings biases the
cross-cell covariance, on real `fixRSVP` recordings (cache-only).

Two weighting regimes, both at the actual-viewing eye target (target='naive',
so the only axis varied is the across-bin weighting -- Extension 1, orthogonal
to Extension 2):

  consistent  -- pair-count throughout: the close-pair 2nd moment, Cpsth, the
                 Ybar subtractor and Ctotal are ALL pair-count weighted (the
                 post-fix production state; LOTC holds term-by-term).
  mixed       -- the documented pre-fix production state (writeup §3.4):
                 close-pair 2nd moment at pair-count, Cpsth at uniform 1/T,
                 Ybar at trial-count, Ctotal the unweighted sample covariance.
                 Not a covariance under any single weighting.

For each session (one analysis window) we compute, off the cached aligned
spikes (no GPU, no covariance.py, no realdata_results.pkl):

  * corrected noise correlation rho_corr = corr(Ctotal - Crate), off-diagonal,
    under each regime -- the scientific consequence (mixed biased low).
  * shuffle-null Dz = fz(Ctotal - Crate_shuf) - fz(Ctotal - Cpsth), off-diagonal
    mean, under each regime. Under the eye-shuffle null Crate_shuf should match
    Cpsth, so Dz -> 0 for the consistent regime; the mixed regime is biased
    (the empirical fingerprint of the inconsistency).

Run:
  uv run python compute_weighting_data.py            # all sessions, window=2
  uv run python compute_weighting_data.py --window 2 --n-shuffles 100
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from estimators import decompose_trajectory                      # noqa: E402
from pipeline import (_extract_windows_numpy, _ctotal_unweighted,  # noqa: E402
                      _legacy_compat_crate, _enumerate_close_pairs,
                      _naive_close_pair_crate)
from legacy.covariance import extract_valid_segments             # noqa: E402
from VisionCore.covariance import cov_to_corr                    # noqa: E402

CACHE = HERE / "cache"
DT = 1 / 120
THRESHOLD = 0.05
MIN_TPP = 10


def _offdiag(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _fz_offdiag(C):
    """Fisher-z of the off-diagonal correlations of covariance matrix C."""
    r = cov_to_corr(C)
    r = _offdiag(r)
    r = np.clip(r, -0.999, 0.999)
    return np.arctanh(r)


def _filter_window(counts, trajectories, T_idx, min_tpp=MIN_TPP):
    """Keep only T_idx bins with >= min_tpp samples (matches decompose_*)."""
    keep = np.zeros(len(T_idx), bool)
    for t in np.unique(T_idx):
        ix = np.where(T_idx == t)[0]
        if len(ix) >= min_tpp:
            keep[ix] = True
    return counts[keep], trajectories[keep], T_idx[keep]


def _session_record(aligned, window, n_shuffles, seed):
    """Compute the weighting-regime quantities for one aligned session."""
    robs = np.nan_to_num(np.asarray(aligned["robs"], float))
    eyepos = np.nan_to_num(np.asarray(aligned["eyepos"], float))
    valid_mask = np.asarray(aligned["valid_mask"], bool)
    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_TPP)
    t_hist = max(int(0.1 / DT), window)
    counts, trajectories, T_idx = _extract_windows_numpy(
        robs, eyepos, segments, window, t_hist)
    if counts is None or counts.shape[0] < 100:
        return None
    counts, trajectories, T_idx = _filter_window(counts, trajectories, T_idx)
    if counts.shape[0] < 100 or len(np.unique(T_idx)) < 2:
        return None

    n_cells = counts.shape[1]

    # --- the three weighted decompositions (target='naive') ---
    d_pc = decompose_trajectory(counts, trajectories, T_idx, target="naive",
                                threshold=THRESHOLD, time_bin_weighting="pair_count",
                                cpsth_method="mcfarland", min_trials_per_time_bin=MIN_TPP)
    d_un = decompose_trajectory(counts, trajectories, T_idx, target="naive",
                                threshold=THRESHOLD, time_bin_weighting="uniform",
                                cpsth_method="mcfarland", min_trials_per_time_bin=MIN_TPP)
    Erate_pc = d_pc["Erate"]
    Erate_un = d_un["Erate"]
    Erate_trial = np.nanmean(counts, axis=0)

    # close-pair Crate (uncentred legacy form), pair-count pooling
    Crate_pc, n_pairs, _, _ = _legacy_compat_crate(
        counts, trajectories, T_idx, "naive", THRESHOLD, Erate=Erate_pc,
        time_bin_weighting="pair_count", weight_clip=1e6)
    MM_close = Crate_pc + np.outer(Erate_pc, Erate_pc)           # shared 2nd moment

    Cpsth_pc = d_pc["Cpsth"]
    Cpsth_un = d_un["Cpsth"]
    MM_psth_un = Cpsth_un + np.outer(Erate_un, Erate_un)

    # consistent regime: pair-count throughout
    Ctotal_c = d_pc["Ctotal"]
    Crate_c = Crate_pc
    Cpsth_c = Cpsth_pc

    # mixed regime (pre-fix): pair-count 2nd moment, uniform Cpsth, trial Ybar,
    # unweighted-sample Ctotal
    Ctotal_m = _ctotal_unweighted(counts)
    Crate_m = MM_close - np.outer(Erate_trial, Erate_trial)
    Cpsth_m = MM_psth_un - np.outer(Erate_trial, Erate_trial)

    rec = {
        "session": aligned["session"],
        "subject": aligned.get("subject"),
        "n_cells": int(n_cells),
        "n_samples": int(counts.shape[0]),
        "n_close_pairs": int(n_pairs),
        "nc_corrected_consistent": _offdiag(cov_to_corr(Ctotal_c - Crate_c)),
        "nc_corrected_mixed": _offdiag(cov_to_corr(Ctotal_m - Crate_m)),
        "nc_uncorrected_consistent": _offdiag(cov_to_corr(Ctotal_c - Cpsth_c)),
        "nc_uncorrected_mixed": _offdiag(cov_to_corr(Ctotal_m - Cpsth_m)),
    }

    # --- shuffle null: permute trajectories, recompute MM_close, form Dz ---
    rng = np.random.default_rng(seed)
    N = counts.shape[0]
    zeros = np.zeros(n_cells)
    dz_c, dz_m = [], []
    fz_cpsth_c = _fz_offdiag(Ctotal_c - Cpsth_c)
    fz_cpsth_m = _fz_offdiag(Ctotal_m - Cpsth_m)
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        gi, gj, tpair = _enumerate_close_pairs(trajectories[perm], T_idx, THRESHOLD)
        if len(gi) == 0:
            continue
        MM_shuf = _naive_close_pair_crate(counts, gi, gj, tpair, T_idx, zeros,
                                          "pair_count")     # Erate=0 -> raw MM
        Crate_shuf_c = MM_shuf - np.outer(Erate_pc, Erate_pc)
        Crate_shuf_m = MM_shuf - np.outer(Erate_trial, Erate_trial)
        dz_c.append(float(np.nanmean(
            _fz_offdiag(Ctotal_c - Crate_shuf_c) - fz_cpsth_c)))
        dz_m.append(float(np.nanmean(
            _fz_offdiag(Ctotal_m - Crate_shuf_m) - fz_cpsth_m)))
    rec["dz_shuffle_consistent"] = np.asarray(dz_c, float)
    rec["dz_shuffle_mixed"] = np.asarray(dz_m, float)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=2, help="analysis window (bins)")
    ap.add_argument("--n-shuffles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=str, default=str(CACHE / "weighting_realdata.pkl"))
    args = ap.parse_args()

    with open(CACHE / "aligned_sessions.pkl", "rb") as f:
        sessions = pickle.load(f)
    print(f"loaded {len(sessions)} sessions; window={args.window}, "
          f"n_shuffles={args.n_shuffles}")

    records = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_session_record)(a, args.window, args.n_shuffles, args.seed + i)
        for i, a in enumerate(sessions))
    records = [r for r in records if r is not None]

    # pooled summaries (printed for sanity; full per-session arrays saved)
    def _pooled(key):
        return np.concatenate([r[key] for r in records])

    nc_c = _pooled("nc_corrected_consistent")
    nc_m = _pooled("nc_corrected_mixed")
    dz_c = np.array([np.nanmean(r["dz_shuffle_consistent"]) for r in records])
    dz_m = np.array([np.nanmean(r["dz_shuffle_mixed"]) for r in records])

    print(f"\n{len(records)} sessions kept.")
    print(f"corrected NC consistent:     median={np.nanmedian(nc_c):+.4f}")
    print(f"corrected NC mixed:          median={np.nanmedian(nc_m):+.4f}")
    print(f"shuffle Dz consistent (per-session mean): "
          f"mean={np.nanmean(dz_c):+.5f}")
    print(f"shuffle Dz mixed      (per-session mean): "
          f"mean={np.nanmean(dz_m):+.5f}")

    out = {
        "window": args.window,
        "n_shuffles": args.n_shuffles,
        "threshold": THRESHOLD,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
