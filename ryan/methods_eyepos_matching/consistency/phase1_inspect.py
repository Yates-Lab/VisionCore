"""Phase 1 inspection: dump the ACTUAL structure of both pipelines' caches so the
consistency diff is grounded in real data, not inferred from source.

Read-only. Prints:
  - aligned_sessions.pkl  : session/subject sets (is Luke present?)
  - methods_derived.pkl   : bundle schema (targets, windows, per-window keys)
  - load_fig2_data()      : production schema (no refresh; uses cached pkls)
  - cross-check           : panel-required keys vs what each side emits

Run: uv run python consistency/phase1_inspect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent          # .../methods_eyepos_matching/consistency
METHODS_DIR = THIS_DIR.parent                        # .../methods_eyepos_matching
FIG2_DIR = METHODS_DIR.parent / "fig2"               # .../ryan/fig2
FIG3_DIR = METHODS_DIR.parent / "fig3"               # .../ryan/fig3
CACHE = METHODS_DIR / "cache"

for p in (METHODS_DIR, FIG2_DIR, FIG3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def describe(x, name="", depth=0, maxdepth=2):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{name}: dict[{len(x)}] keys={list(x.keys())[:12]}"
              + (" ..." if len(x) > 12 else ""))
        if depth < maxdepth:
            for k in list(x.keys())[:6]:
                describe(x[k], repr(k), depth + 1, maxdepth)
    elif isinstance(x, (list, tuple)):
        kind = type(x).__name__
        print(f"{pad}{name}: {kind}[{len(x)}]")
        if len(x) and depth < maxdepth:
            describe(x[0], "[0]", depth + 1, maxdepth)
    elif isinstance(x, np.ndarray):
        print(f"{pad}{name}: ndarray shape={x.shape} dtype={x.dtype}")
    else:
        print(f"{pad}{name}: {type(x).__name__} = {x!r}"[:120])


# ---------------------------------------------------------------- aligned cache
rule("aligned_sessions.pkl  (input to BOTH pipelines)")
with open(CACHE / "aligned_sessions.pkl", "rb") as f:
    aligned = dill.load(f)
print(f"type={type(aligned).__name__}")
if isinstance(aligned, dict):
    print(f"top keys: {list(aligned.keys())}")
    sess_list = aligned.get("sessions", aligned.get("results", aligned))
else:
    sess_list = aligned
if isinstance(sess_list, list):
    print(f"n_sessions = {len(sess_list)}")
    from collections import Counter
    subj_counts = Counter(s.get("subject") for s in sess_list)
    print(f"subject -> n_sessions: {dict(subj_counts)}")
    print("first session keys:", list(sess_list[0].keys()))
    for s in sess_list[:50]:
        nu = s.get("n_neurons_used")
        print(f"  {s.get('subject'):8s} {str(s.get('session'))[:40]:40s} "
              f"n_used={nu} contam={'yes' if s.get('contam_rate') is not None else 'None'}")

# ---------------------------------------------------------------- methods bundle
rule("methods_derived.pkl  (derive_methods bundle)")
with open(CACHE / "methods_derived.pkl", "rb") as f:
    md = dill.load(f)
print("top-level keys:", list(md.keys()))
for k in ("windows_ms", "windows_bins", "targets", "session_names", "subjects"):
    if k in md:
        print(f"  {k}: {md[k]}")
tgt0 = md["targets"][0]
w_ms = md["windows_ms"][0]
print(f"\n-- metrics[{tgt0!r}][0] (window {w_ms} ms) --")
describe(md["metrics"][tgt0][0], "metrics", maxdepth=1)
print(f"\n-- alpha_stats[{tgt0!r}][{w_ms}] --")
describe(md["alpha_stats"][tgt0][w_ms], "alpha_stats", maxdepth=1)
print(f"\n-- fano_stats[{tgt0!r}][{w_ms}] --")
describe(md["fano_stats"][tgt0][w_ms], "fano_stats", maxdepth=2)
print(f"\n-- nc_stats[{tgt0!r}][{w_ms}] --")
describe(md["nc_stats"][tgt0][w_ms], "nc_stats", maxdepth=2)

# ---------------------------------------------------------------- production
rule("load_fig2_data()  (PRODUCTION, no refresh)")
try:
    from compute_fig2_data import load_fig2_data
    prod = load_fig2_data(refresh=False)
    print("top-level keys:", sorted(prod.keys()))
    for k in ("WINDOWS_MS", "WINDOWS_BINS", "SUBJECTS", "session_names",
              "subjects", "n_sessions"):
        if k in prod:
            print(f"  {k}: {prod[k]}")
    print("\n-- metrics[0] keys --")
    describe(prod["metrics"][0], "metrics[0]", maxdepth=1)
    w0 = prod["WINDOWS_MS"][0]
    print(f"\n-- fano_stats[{w0}] --")
    describe(prod["fano_stats"][w0], "fano_stats", maxdepth=2)
    print(f"\n-- nc_stats[{w0}] --")
    describe(prod["nc_stats"][w0], "nc_stats", maxdepth=2)
    print("\n-- m_by_window / subject_per_neuron_by_window --")
    describe(prod["m_by_window"], "m_by_window", maxdepth=1)
    describe(prod["subject_per_neuron_by_window"], "subj_per_neuron", maxdepth=1)
    m0 = prod["m_by_window"][0]
    finite = np.isfinite(m0)
    print(f"  m0: n={m0.size} finite={finite.sum()} "
          f"min={np.nanmin(m0):.3f} max={np.nanmax(m0):.3f} "
          f"<0: {(m0[finite] < 0).sum()} >1: {(m0[finite] > 1).sum()}")
    PROD_OK = True
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n!! load_fig2_data failed: {e}")
    PROD_OK = False

# ---------------------------------------------------------------- panel-key check
rule("PANEL-REQUIRED KEYS  (do the methods metrics carry them?)")
PANEL_METRIC_KEYS = [
    "subject_per_pair",      # fig3b
    "subject_by_ds",         # fig3c, fig3d
    "rho_u_meanz_by_ds",     # fig3c
    "rho_c_meanz_by_ds",     # fig3c
    "rho_delta_meanz_by_ds", # fig3d
    "shuff_rho_subject",     # fig3d
    "shuff_rho_delta_meanz", # fig3d
]
PANEL_FANO_KEYS = ["per_subject"]                       # fig2e
PANEL_NC_KEYS = ["rho_u", "rho_c", "null_dz_ci_by_subject"]  # fig3b, fig3d

m_keys = set(md["metrics"][tgt0][0].keys())
print("methods metrics[tgt][0] keys:", sorted(m_keys))
for k in PANEL_METRIC_KEYS:
    print(f"  metric {k:24s}: {'PRESENT' if k in m_keys else 'MISSING'}")
f_keys = set(md["fano_stats"][tgt0][w_ms].keys())
for k in PANEL_FANO_KEYS:
    print(f"  fano   {k:24s}: {'PRESENT' if k in f_keys else 'MISSING'}")
n_keys = set(md["nc_stats"][tgt0][w_ms].keys())
for k in PANEL_NC_KEYS:
    print(f"  nc     {k:24s}: {'PRESENT' if k in n_keys else 'MISSING'}")

print("\nDONE.")
