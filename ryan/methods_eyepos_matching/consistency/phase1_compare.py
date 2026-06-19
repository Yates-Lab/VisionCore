"""Phase 1 comparison: per-window PROD vs methods NAIVE vs methods FULL for the
headline statistics, plus the raw-vs-clipped alpha check and the
exclude-not-clip recount. Read-only. Grounds the consistency diff table.

Run: uv run python consistency/phase1_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
FIG2_DIR = METHODS_DIR.parent / "fig2"
FIG3_DIR = METHODS_DIR.parent / "fig3"
CACHE = METHODS_DIR / "cache"
for p in (METHODS_DIR, FIG2_DIR, FIG3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

with open(CACHE / "methods_derived.pkl", "rb") as f:
    md = dill.load(f)
from compute_fig2_data import load_fig2_data
prod = load_fig2_data(refresh=False)

W_MS = prod["WINDOWS_MS"]


def get_fano(stats):
    return (stats["g_unc"], stats["g_cor"], stats["ratio"],
            stats.get("slope_unc"), stats.get("slope_cor"))


print("\n=== FANO (geomean g_unc / g_cor / ratio ; slope_unc / slope_cor) ===")
hdr = f"{'win_ms':>8} | {'source':6} | {'g_unc':>7} {'g_cor':>7} {'ratio':>7} | {'sl_unc':>8} {'sl_cor':>8}"
print(hdr); print("-" * len(hdr))
for w in W_MS:
    for src, stats in (("PROD", prod["fano_stats"][w]),
                       ("NAIVE", md["fano_stats"]["naive"][w]),
                       ("FULL", md["fano_stats"]["full"][w])):
        gu, gc, r, su, sc = get_fano(stats)
        su = np.nan if su is None else su
        sc = np.nan if sc is None else sc
        print(f"{w:8.1f} | {src:6} | {gu:7.4f} {gc:7.4f} {r:7.4f} | {su:8.4f} {sc:8.4f}")
    print()

print("\n=== NC (Fisher-z: z_u_mean / z_c_mean / dz_mean) ===")
hdr = f"{'win_ms':>8} | {'source':6} | {'z_u':>9} {'z_c':>9} {'dz':>9} {'n_pairs':>8}"
print(hdr); print("-" * len(hdr))
for w in W_MS:
    for src, stats in (("PROD", prod["nc_stats"][w]),
                       ("NAIVE", md["nc_stats"]["naive"][w]),
                       ("FULL", md["nc_stats"]["full"][w])):
        print(f"{w:8.1f} | {src:6} | {stats['z_u_mean']:9.5f} {stats['z_c_mean']:9.5f} "
              f"{stats['dz_mean']:9.5f} {stats.get('n_pairs', -1):8d}")
    print()

print("\n=== ALPHA (1-alpha) stats: n / mean / median ===")
hdr = f"{'win_ms':>8} | {'source':6} | {'n':>6} {'mean':>8} {'median':>8}"
print(hdr); print("-" * len(hdr))
for i, w in enumerate(W_MS):
    pa = prod["alpha_stats"][w]
    print(f"{w:8.1f} | {'PROD':6} | {pa['n']:6d} {pa['mean']:8.4f} {pa['median']:8.4f}")
    for src in ("naive", "full"):
        a = md["alpha_stats"][src][w]
        print(f"{w:8.1f} | {src.upper():6} | {a['n']:6d} {a['mean']:8.4f} {a['median']:8.4f}")
    print()

# ---- raw-vs-clipped alpha check (window 0) ----
print("\n=== raw-vs-clipped alpha check (window index 0) ===")
pm = prod["metrics"][0]["alpha"]
mm = md["metrics"]["naive"][0]["alpha"]
print(f"prod  metrics['alpha']: n={pm.size} finite={np.isfinite(pm).sum()} "
      f"min={np.nanmin(pm):.4f} max={np.nanmax(pm):.4f} "
      f"<0:{(pm[np.isfinite(pm)]<0).sum()} >1:{(pm[np.isfinite(pm)]>1).sum()}")
print(f"methods metrics['alpha']: n={mm.size} finite={np.isfinite(mm).sum()} "
      f"min={np.nanmin(mm):.4f} max={np.nanmax(mm):.4f} "
      f"<0:{(mm[np.isfinite(mm)]<0).sum()} >1:{(mm[np.isfinite(mm)]>1).sum()}")
if pm.size == mm.size:
    both = np.isfinite(pm) & np.isfinite(mm)
    d = np.abs(pm[both] - mm[both])
    print(f"elementwise |prod-methods| over {both.sum()} shared-finite: "
          f"max={d.max():.3e} mean={d.mean():.3e}")
    # where prod is out of [0,1], is methods clipped?
    oor = both & ((pm < 0) | (pm > 1))
    print(f"prod alpha out-of-[0,1] & finite: {oor.sum()}")
    if oor.sum():
        print("  sample (prod -> methods):",
              list(zip(np.round(pm[oor][:8], 3), np.round(mm[oor][:8], 3))))

# ---- recompute raw 1-alpha from methods Crate/Cpsth diagonals (window 0) ----
print("\n=== methods raw 1-alpha from diag(Cpsth)/diag(Crate) (window 0, exclude vs clip) ===")
m0 = md["metrics"]["naive"][0]
for key in ("Crate", "Cpsth"):
    v = m0.get(key)
    print(f"  {key}: type={type(v).__name__} "
          + (f"shape={v.shape}" if isinstance(v, np.ndarray) else f"len={len(v) if hasattr(v,'__len__') else '?'}"))
