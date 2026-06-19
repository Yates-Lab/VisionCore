"""Verify the adapter output against a live load_fig2_data() dict:
  - every panel-required key present with matching type/shape
  - NAIVE reproduces PROD on the panel-driving statistics within tolerance
  - per-subject sub-keys (fano per_subject, nc null_dz_ci_by_subject) reconstructed

Run: uv run python consistency/verify_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from adapter import methods_to_fig2_schema, load_methods_bundle  # noqa: E402
from compute_fig2_data import load_fig2_data  # noqa: E402

md = load_methods_bundle()
prod = load_fig2_data(refresh=False)
naive = methods_to_fig2_schema(md, "naive")
full = methods_to_fig2_schema(md, "full")
central = methods_to_fig2_schema(md, "central", null_from="naive")

ok = True


def check(cond, msg):
    global ok
    flag = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{flag}] {msg}")


print("=== panel-required keys present in adapter output ===")
W = naive["WINDOWS_MS"]
w0 = W[0]
for k in ("SUBJECTS", "SUBJECT_COLORS", "WINDOWS_MS", "WINDOWS_BINS",
          "metrics", "m_by_window", "subject_per_neuron_by_window",
          "alpha_stats", "fano_stats", "nc_stats"):
    check(k in naive, f"top-level key {k!r}")

# fig2c
check(len(naive["m_by_window"]) == len(W), "m_by_window length == n_windows")
check(np.all(np.isfinite(naive["m_by_window"][0])), "m_by_window[0] all finite (excluded)")
m0 = naive["m_by_window"][0]
check((m0 >= 0).all() and (m0 <= 1).all(), f"m_by_window[0] in [0,1] (range [{m0.min():.3f},{m0.max():.3f}], no pile-up)")

# fig2e
check("per_subject" in naive["fano_stats"][w0], "fano_stats[w0]['per_subject'] reconstructed")
ps = naive["fano_stats"][w0]["per_subject"]
for subj in ("Allen", "Logan"):
    check(subj in ps and "slope_cor" in ps[subj], f"fano per_subject[{subj}] has slope_cor")

# fig3b
check("rho_u" in naive["nc_stats"][w0] and "rho_c" in naive["nc_stats"][w0],
      "nc_stats[w0] has rho_u/rho_c")
check("subject_per_pair" in naive["metrics"][0], "metrics[0]['subject_per_pair']")

# fig3c
for k in ("subject_by_ds", "rho_u_meanz_by_ds", "rho_c_meanz_by_ds"):
    check(k in naive["metrics"][0], f"metrics[0][{k!r}]")

# fig3d
for k in ("rho_delta_meanz_by_ds", "shuff_rho_subject", "shuff_rho_delta_meanz"):
    check(k in naive["metrics"][0], f"metrics[0][{k!r}]")
check("null_dz_ci_by_subject" in naive["nc_stats"][w0],
      "nc_stats[w0]['null_dz_ci_by_subject'] reconstructed")

print("\n=== adapter NAIVE vs PROD key-set parity (panel-relevant) ===")
prod_fano_keys = set(prod["fano_stats"][w0].keys())
naive_fano_keys = set(naive["fano_stats"][w0].keys())
check(prod_fano_keys == naive_fano_keys,
      f"fano_stats[w0] keys identical (prod\\naive={prod_fano_keys-naive_fano_keys}, "
      f"naive\\prod={naive_fano_keys-prod_fano_keys})")
prod_nc_keys = set(prod["nc_stats"][w0].keys())
naive_nc_keys = set(naive["nc_stats"][w0].keys())
check(prod_nc_keys == naive_nc_keys,
      f"nc_stats[w0] keys identical (prod\\naive={prod_nc_keys-naive_nc_keys}, "
      f"naive\\prod={naive_nc_keys-prod_nc_keys})")

print("\n=== NAIVE reproduces PROD within tolerance (panel statistics) ===")
TOL_FANO = 0.02   # ~2% on slopes/geomeans
TOL_NC = 0.003
for w in W:
    pf, nf = prod["fano_stats"][w], naive["fano_stats"][w]
    d_sc = abs(pf["slope_cor"] - nf["slope_cor"])
    d_su = abs(pf["slope_unc"] - nf["slope_unc"])
    check(d_sc < TOL_FANO, f"fano slope_cor @{w:.1f}ms |Δ|={d_sc:.4f} < {TOL_FANO}")
    check(d_su < TOL_FANO, f"fano slope_unc @{w:.1f}ms |Δ|={d_su:.4f} < {TOL_FANO}")
    pn, nn = prod["nc_stats"][w], naive["nc_stats"][w]
    d_dz = abs(pn["dz_mean"] - nn["dz_mean"])
    check(d_dz < 0.012, f"nc dz_mean   @{w:.1f}ms |Δ|={d_dz:.4f} < 0.012")

print("\n=== corrected Fano slope < 0.8 @ 25 ms (brief target) ===")
check(prod["fano_stats"][25.0]["slope_cor"] < 0.8,
      f"PROD slope_cor @25ms = {prod['fano_stats'][25.0]['slope_cor']:.4f}")
check(naive["fano_stats"][25.0]["slope_cor"] < 0.8,
      f"NAIVE slope_cor @25ms = {naive['fano_stats'][25.0]['slope_cor']:.4f}")

print("\n=== target movement: NAIVE / FULL / CENTRAL ===")
print(f"{'win':>6} | {'slope_cor n/f/c':>26} | {'dz_mean n/f/c':>26} | {'z_c n/f/c':>26}")
for w in W:
    sc = (naive['fano_stats'][w]['slope_cor'], full['fano_stats'][w]['slope_cor'],
          central['fano_stats'][w]['slope_cor'])
    dz = (naive['nc_stats'][w]['dz_mean'], full['nc_stats'][w]['dz_mean'],
          central['nc_stats'][w]['dz_mean'])
    zc = (naive['nc_stats'][w]['z_c_mean'], full['nc_stats'][w]['z_c_mean'],
          central['nc_stats'][w]['z_c_mean'])
    print(f"{w:6.1f} | {sc[0]:7.4f} {sc[1]:7.4f} {sc[2]:7.4f}   | "
          f"{dz[0]:7.4f} {dz[1]:7.4f} {dz[2]:7.4f}   | "
          f"{zc[0]:7.4f} {zc[1]:7.4f} {zc[2]:7.4f}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
