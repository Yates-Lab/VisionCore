"""Phase 1 inclusion diff: per-window n_total / n_dropped / n for PROD vs methods
NAIVE/FULL, plus the base per-neuron count entering each window. Pins down why
unit/pair counts diverge at larger windows. Read-only.

Run: uv run python consistency/phase1_inclusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
FIG2_DIR = METHODS_DIR.parent / "fig2"
CACHE = METHODS_DIR / "cache"
for p in (METHODS_DIR, FIG2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

with open(CACHE / "methods_derived.pkl", "rb") as f:
    md = dill.load(f)
from compute_fig2_data import load_fig2_data
prod = load_fig2_data(refresh=False)
W = prod["WINDOWS_MS"]

print("\n=== ALPHA inclusion: n_total (finite 1-alpha) / n_dropped (out [0,1]) / n (kept) ===")
hdr = f"{'win_ms':>8} | {'source':6} | {'base':>5} {'n_total':>7} {'n_drop':>6} {'n_kept':>6}"
print(hdr); print("-" * len(hdr))
for i, w in enumerate(W):
    base_p = prod["metrics"][i]["alpha"].size
    pa = prod["alpha_stats"][w]
    print(f"{w:8.1f} | {'PROD':6} | {base_p:5d} {pa['n_total']:7d} {pa['n_dropped']:6d} {pa['n']:6d}")
    for src in ("naive", "full"):
        base_m = md["metrics"][src][i]["alpha"].size
        a = md["alpha_stats"][src][w]
        print(f"{w:8.1f} | {src.upper():6} | {base_m:5d} {a['n_total']:7d} {a['n_dropped']:6d} {a['n']:6d}")
    print()

print("\n=== base per-neuron count by window (does production shrink units per window?) ===")
for i, w in enumerate(W):
    bp = prod["metrics"][i]["alpha"].size
    bn = md["metrics"]["naive"][i]["alpha"].size
    print(f"  win {w:6.1f} ms: PROD base={bp}  METHODS base={bn}  diff={bp-bn}")

print("\n=== NC n_pairs by window ===")
for w in W:
    print(f"  win {w:6.1f} ms: PROD={prod['nc_stats'][w]['n_pairs']:6d}  "
          f"NAIVE={md['nc_stats']['naive'][w]['n_pairs']:6d}  "
          f"FULL={md['nc_stats']['full'][w]['n_pairs']:6d}")
