"""Wall-time benchmark: legacy snapshot vs methods pipeline, on the same
aligned sessions, CPU vs CPU.

Writes ``cache/timing.csv`` with columns:
    session, subject, n_cells, n_windows, t_count, pipeline, wall_s

`pipeline` is one of {'methods', 'legacy'}.

Per-session work runs sequentially here -- the goal is honest per-session
timing, not throughput. The methods pipeline's parallelism is reported via the
total wall-time of ``compute_methods_data.py``.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loading import load_cache                    # noqa: E402
from pipeline import (                                 # noqa: E402
    decompose_session, decompose_session_legacy,
    WINDOW_BINS_DEFAULT,
)

TIMING_CSV = THIS_DIR / "cache" / "timing.csv"


def _run_methods_one(aligned, t_count, n_shuffles):
    t0 = time.perf_counter()
    out = decompose_session(
        aligned, windows_bins=(t_count,),
        targets=("naive", "full", "central"),
        n_shuffles=n_shuffles,
    )
    return time.perf_counter() - t0, out


def _run_legacy_one(aligned, t_count, n_shuffles):
    t0 = time.perf_counter()
    out = decompose_session_legacy(
        aligned, device="cpu", windows_bins=(t_count,),
        n_shuffles=n_shuffles,
    )
    return time.perf_counter() - t0, out


def benchmark(sessions, windows_bins=WINDOW_BINS_DEFAULT,
              n_shuffles=20, max_sessions=None):
    """Sequential per-(session, window, pipeline) timing.

    Defaults to 20 shuffles (vs the production 100) -- a single per-window
    benchmark with 100 shuffles × 30 sessions × 4 windows × 2 pipelines on one
    box is ~hours. The shuffles scale linearly so this is enough to show the
    pipeline speed ratio without an overnight wait. Users can pass
    ``n_shuffles=100`` for the production number.
    """
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    TIMING_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sess in sessions:
        n_cells = sess["n_neurons_used"]
        for t_count in windows_bins:
            print(f"  {sess['session']:22s} t={t_count}: ", end="", flush=True)
            t_m, _ = _run_methods_one(sess, t_count, n_shuffles)
            print(f"methods={t_m:5.1f}s ", end="", flush=True)
            t_l, _ = _run_legacy_one(sess, t_count, n_shuffles)
            print(f"legacy={t_l:5.1f}s  speedup={t_l/max(t_m, 1e-6):.2f}x")

            rows.append(dict(session=sess["session"], subject=sess["subject"],
                             n_cells=n_cells, t_count=t_count, pipeline="methods",
                             wall_s=t_m))
            rows.append(dict(session=sess["session"], subject=sess["subject"],
                             n_cells=n_cells, t_count=t_count, pipeline="legacy",
                             wall_s=t_l))

    with open(TIMING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {TIMING_CSV}")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="Subset to first N sessions (faster smoke runs).")
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--windows-bins", type=int, nargs="+",
                    default=list(WINDOW_BINS_DEFAULT))
    args = ap.parse_args()

    sessions = load_cache()
    print(f"Loaded {len(sessions)} aligned sessions.")
    benchmark(sessions, windows_bins=args.windows_bins,
              n_shuffles=args.n_shuffles, max_sessions=args.max_sessions)


if __name__ == "__main__":
    main()
