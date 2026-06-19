"""Movement of the headline statistics as the fixation window tightens.

For each counting window and radius (BASELINE | r=1.0 | r=0.75 | r=0.5), report
  * 1-alpha       (mean [CI], median)        -- FEM modulation fraction
  * Fano slope_cor (corrected slope [CI])     -- corrected Fano
  * NC z_c_mean    (residual corrected NC [CI])
  * NC dz_mean     (reduction corr-uncorr [CI])
at target='full', reading the production statistics' own clustered-bootstrap CIs
(slope_cor_ci / z_c_ci / dz_ci / alpha ci) through the consistency adapter.

The headline question is whether the residual corrected NC (z_c) is STABLE as the
window tightens (effect robust), SHRINKS toward zero (periphery inflated it), or
GROWS. We answer it statistically via baseline-vs-r=0.5 CI overlap per window.

Run: uv run python eyepos_masking/movement_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
CACHE = METHODS_DIR / "cache"
sys.path.insert(0, str(METHODS_DIR))
sys.path.insert(0, str(METHODS_DIR / "consistency"))
from adapter import methods_to_fig2_schema  # noqa: E402

TAGS = ["base", "1.0", "0.75", "0.5"]
LABEL = {"base": "BASE", "1.0": "r=1.0", "0.75": "r=0.75", "0.5": "r=0.5"}


def ci(t):
    return f"[{t[0]:+.3f},{t[1]:+.3f}]"


def overlap(a, b):
    """Do two CIs overlap?"""
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    data = {}
    for tag in TAGS:
        md = dill.load(open(CACHE / f"methods_derived_r{tag}.pkl", "rb"))
        data[tag] = methods_to_fig2_schema(md, "full", null_from="naive")
    windows = data["base"]["WINDOWS_MS"]

    print("\n# Movement table (target='full'), with clustered-bootstrap 95% CIs\n")
    for w in windows:
        print(f"\n## window {w:.1f} ms")
        print(f"{'radius':>7s} {'1-a mean':>9s} {'1-a CI':>17s} {'1-a med':>8s} "
              f"{'slopeCor':>9s} {'slope CI':>17s} {'z_c':>8s} {'z_c CI':>17s} "
              f"{'dz':>8s} {'dz CI':>17s}")
        for tag in TAGS:
            a = data[tag]["alpha_stats"][w]
            f = data[tag]["fano_stats"][w]
            n = data[tag]["nc_stats"][w]
            print(f"{LABEL[tag]:>7s} {a['mean']:9.3f} {ci(a['ci']):>17s} "
                  f"{a['median']:8.3f} {f['slope_cor']:9.3f} "
                  f"{ci(f['slope_cor_ci']):>17s} {n['z_c_mean']:8.3f} "
                  f"{ci(n['z_c_ci']):>17s} {n['dz_mean']:8.3f} "
                  f"{ci(n['dz_ci']):>17s}")

    # ---- verdict: residual corrected NC (z_c) stability base -> r=0.5 ----
    print("\n# Residual corrected NC (z_c): baseline vs r=0.5 per window\n")
    print(f"{'window':>7s} {'z_c base':>9s} {'z_c r0.5':>9s} {'delta':>8s} "
          f"{'CIs overlap?':>13s}  verdict")
    for w in windows:
        zb = data["base"]["nc_stats"][w]["z_c_mean"]
        cb = data["base"]["nc_stats"][w]["z_c_ci"]
        z5 = data["0.5"]["nc_stats"][w]["z_c_mean"]
        c5 = data["0.5"]["nc_stats"][w]["z_c_ci"]
        ov = overlap(cb, c5)
        if ov:
            verdict = "STABLE (CIs overlap)"
        elif z5 < zb:
            verdict = "SHRINKS toward 0"
        else:
            verdict = "GROWS"
        print(f"{w:7.1f} {zb:9.3f} {z5:9.3f} {z5-zb:+8.3f} "
              f"{str(ov):>13s}  {verdict}")


if __name__ == "__main__":
    main()
