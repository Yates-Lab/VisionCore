"""Stage-1 orchestrator: run the methods pipeline (or the legacy snapshot) on
every session in the aligned cache, in parallel, and emit:

  cache/methods_decomposition.pkl      methods per-session pipeline output
  cache/methods_derived.pkl            methods stage-2 metrics
  cache/legacy_decomposition.pkl       legacy SNAPSHOT per-session output
  cache/legacy_derived.pkl             legacy stage-2 metrics

The same aligned cache (`cache/aligned_sessions.pkl`) is the input for both,
so the comparison is apples-to-apples by construction.

Usage:
  uv run python compute_methods_data.py             # methods pipeline
  uv run python compute_methods_data.py --legacy    # legacy snapshot
  uv run python compute_methods_data.py --both      # methods then legacy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loading import load_cache                                 # noqa: E402
from pipeline import (                                              # noqa: E402
    DT, WINDOW_BINS_DEFAULT, TARGETS_DEFAULT, N_SHUFFLES_DEFAULT,
    decompose_session, decompose_session_legacy,
)
from metrics import derive_methods, derive_legacy                   # noqa: E402

CACHE_DIR = THIS_DIR / "cache"
METHODS_DECOMP = CACHE_DIR / "methods_decomposition.pkl"
METHODS_DERIVED = CACHE_DIR / "methods_derived.pkl"
LEGACY_DECOMP = CACHE_DIR / "legacy_decomposition.pkl"
LEGACY_DERIVED = CACHE_DIR / "legacy_derived.pkl"


def _worker_init():
    """Pin BLAS to one thread per worker so n_jobs workers don't oversubscribe
    the box."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _run_methods(aligned, windows_bins, n_shuffles):
    _worker_init()
    return decompose_session(
        aligned, windows_bins=windows_bins, targets=TARGETS_DEFAULT,
        n_shuffles=n_shuffles,
    )


def _run_legacy(aligned, windows_bins, n_shuffles):
    _worker_init()
    return decompose_session_legacy(
        aligned, device="cpu", windows_bins=windows_bins, n_shuffles=n_shuffles,
        dt=DT,
    )


def compute_methods(sessions, n_jobs=-1, windows_bins=WINDOW_BINS_DEFAULT,
                    n_shuffles=N_SHUFFLES_DEFAULT):
    """Run methods stage 1 + stage 2, cache, return derived bundle."""
    from joblib import Parallel, delayed

    print(f"\n=== Methods stage 1: {len(sessions)} sessions, "
          f"n_jobs={n_jobs}, windows={list(windows_bins)}, "
          f"n_shuffles={n_shuffles} ===")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_methods)(s, list(windows_bins), n_shuffles)
        for s in sessions
    )
    wall = time.perf_counter() - t0
    print(f"Methods stage 1: {wall:.1f} s ({wall/len(sessions):.1f} s/session)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(METHODS_DECOMP, "wb") as f:
        dill.dump(dict(results=results, wall_s=wall,
                       windows_bins=list(windows_bins)), f)
    print(f"Cached -> {METHODS_DECOMP}")

    windows_ms = [r["window_ms"] for r in results[0]["windows"]]
    windows_bins_out = [r["window_bins"] for r in results[0]["windows"]]
    print(f"=== Methods stage 2: derive metrics ({len(results)} sessions) ===")
    t1 = time.perf_counter()
    derived = derive_methods(results, windows_ms, windows_bins_out)
    print(f"Methods stage 2: {time.perf_counter()-t1:.1f} s")
    with open(METHODS_DERIVED, "wb") as f:
        dill.dump(derived, f)
    print(f"Cached -> {METHODS_DERIVED}")
    return derived


def compute_legacy(sessions, n_jobs=-1, windows_bins=WINDOW_BINS_DEFAULT,
                   n_shuffles=N_SHUFFLES_DEFAULT):
    """Run legacy snapshot stage 1 + stage 2 from the same aligned cache."""
    from joblib import Parallel, delayed

    print(f"\n=== Legacy snapshot stage 1: {len(sessions)} sessions, "
          f"n_jobs={n_jobs}, windows={list(windows_bins)}, "
          f"n_shuffles={n_shuffles} ===")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_legacy)(s, list(windows_bins), n_shuffles)
        for s in sessions
    )
    wall = time.perf_counter() - t0
    print(f"Legacy stage 1: {wall:.1f} s ({wall/len(sessions):.1f} s/session)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEGACY_DECOMP, "wb") as f:
        dill.dump(dict(results=results, wall_s=wall,
                       windows_bins=list(windows_bins)), f)
    print(f"Cached -> {LEGACY_DECOMP}")

    windows_ms = [r["window_ms"] for r in results[0]["results"]]
    windows_bins_out = [r["window_bins"] for r in results[0]["results"]]
    print(f"=== Legacy snapshot stage 2: derive metrics ===")
    t1 = time.perf_counter()
    derived = derive_legacy(results, windows_ms, windows_bins_out)
    print(f"Legacy stage 2: {time.perf_counter()-t1:.1f} s")
    with open(LEGACY_DERIVED, "wb") as f:
        dill.dump(derived, f)
    print(f"Cached -> {LEGACY_DERIVED}")
    return derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                    help="Run the legacy snapshot pipeline (CPU).")
    ap.add_argument("--both", action="store_true",
                    help="Run methods then legacy.")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--windows-bins", type=int, nargs="+",
                    default=list(WINDOW_BINS_DEFAULT))
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES_DEFAULT)
    args = ap.parse_args()

    sessions = load_cache()
    print(f"Loaded {len(sessions)} aligned sessions.")

    if args.both:
        compute_methods(sessions, n_jobs=args.n_jobs,
                        windows_bins=args.windows_bins,
                        n_shuffles=args.n_shuffles)
        compute_legacy(sessions, n_jobs=args.n_jobs,
                       windows_bins=args.windows_bins,
                       n_shuffles=args.n_shuffles)
    elif args.legacy:
        compute_legacy(sessions, n_jobs=args.n_jobs,
                       windows_bins=args.windows_bins,
                       n_shuffles=args.n_shuffles)
    else:
        compute_methods(sessions, n_jobs=args.n_jobs,
                        windows_bins=args.windows_bins,
                        n_shuffles=args.n_shuffles)


if __name__ == "__main__":
    main()
