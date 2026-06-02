"""cProfile a single methods-pipeline session.

Picks the first Allen session in the aligned cache, runs at window=2 with all
three targets and 10 shuffles (tractable for a profiler), saves the binary
profile to ``cache/profile.prof``, and prints the top 20 by tottime.

Usage:
  uv run python profile_pipeline.py [--session SESSION_NAME] [--window 2]
                                    [--n-shuffles 10]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_loading import load_cache               # noqa: E402
from pipeline import decompose_session            # noqa: E402

PROFILE_PATH = THIS_DIR / "cache" / "profile.prof"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, default=None,
                    help="Session name to profile (default: first Allen).")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--n-shuffles", type=int, default=10)
    args = ap.parse_args()

    sessions = load_cache()
    if args.session is None:
        candidates = [s for s in sessions if s["subject"] == "Allen"]
        target = candidates[0] if candidates else sessions[0]
    else:
        target = next(s for s in sessions if s["session"] == args.session)
    print(f"Profiling: {target['session']} ({target['subject']}), "
          f"window={args.window}, n_shuffles={args.n_shuffles}")

    pr = cProfile.Profile()
    pr.enable()
    decompose_session(target, windows_bins=(args.window,),
                      targets=("naive", "full", "central"),
                      n_shuffles=args.n_shuffles)
    pr.disable()

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(PROFILE_PATH)
    print(f"\nSaved -> {PROFILE_PATH}")

    print("\n--- Top 20 by cumulative time ---")
    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    stats.print_stats(20)

    print("\n--- Top 20 by self time ---")
    stats.sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
