"""Is the twin worse on Logan, or is Logan's population just harder?

TWIN_IMPROVEMENTS item 3 proposed balanced per-subject sampling because "the
twin underperforms on Logan ... training is dominated by Allen". `data_census.py`
refutes the mechanism: Allen holds 51.8% of training samples to Logan's 48.2%,
Logan contributes *more* sessions (16 vs 14), and the loss is a masked mean per
session followed by a mean over the sessions in the batch, so unit count -- the
only lopsided quantity (71/29) -- never enters the weighting. Per unit, Logan is
already weighted ~2.5x more heavily than Allen.

The held-out gap is nonetheless real (median CC_norm 0.653 vs 0.570). This
script asks whether it survives conditioning on how hard each unit is to
predict. Logan's included cells fire at 10.5 Hz to Allen's 19.5 and have
split-half PSTH R^2 of 0.079 to Allen's 0.183, so the raw contrast compares
populations that are not exchangeable.

Method: coarsened exact matching on firing rate and PSTH R^2 (and optionally
fixrsvp trial count), strata defined on pooled quantiles, contribution weighted
by stratum size. Uncertainty from a session-clustered bootstrap, matching the
resampling unit fig3 uses for its contrasts.

    uv run python paper/model_selection/subject_gap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np

HERE = Path(__file__).resolve().parent
VISIONCORE_ROOT = HERE.parent.parent
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from VisionCore.paths import CACHE_DIR  # noqa: E402

TWIN_CACHE = CACHE_DIR / "fig3_digitaltwin.pkl"
ALIGNED_CACHE = CACHE_DIR / "covdecomp_aligned_sessions.pkl"

REFERENCE = "Allen"
FOCAL = "Logan"
N_BOOT = 2000
BOOT_SEED = 0


def build_unit_table():
    """Per-unit table joining twin performance to fig2 inclusion statistics."""
    twin = dill.load(open(TWIN_CACHE, "rb"))
    aligned = {a["session"]: a for a in dill.load(open(ALIGNED_CACHE, "rb"))}

    rows = []
    for t in twin:
        sess = t["session"]
        if sess not in aligned:
            continue
        nm = np.asarray(t["neuron_mask"])
        a = aligned[sess]
        anm = np.asarray(a["neuron_mask"])

        # Both index the session's cids, so align by original unit index.
        pos = {int(o): i for i, o in enumerate(anm)}
        take = np.array([pos[int(o)] for o in nm if int(o) in pos], dtype=int)
        keep = np.array([i for i, o in enumerate(nm) if int(o) in pos], dtype=int)
        if take.size == 0:
            continue

        rate = np.asarray(a["rate_hz"], float)[take]
        psth_r2 = np.asarray(a["psth_r2"], float)[take]
        for j, k in enumerate(keep):
            rows.append({
                "session": sess,
                "subject": t["subject"],
                "ccnorm": float(np.asarray(t["ccnorm"])[k]),
                "ccabs": float(np.asarray(t["ccabs"])[k]),
                "ccmax": float(np.asarray(t["ccmax"])[k]),
                "rate_hz": float(rate[j]),
                "psth_r2": float(psth_r2[j]),
                "n_trials": int(t["n_trials"]),
                "n_units_session": int(len(nm)),
            })
    return rows


def _finite(rows, keys):
    return [r for r in rows if all(np.isfinite(r[k]) for k in keys)]


def _strata(rows, covariates, n_bins):
    """Coarsened exact matching cells, from pooled quantile edges."""
    labels = []
    edges = {}
    for cov in covariates:
        v = np.array([r[cov] for r in rows], float)
        qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges[cov] = qs.tolist()
        labels.append(np.digitize(v, qs))
    return [tuple(int(lab[i]) for lab in labels) for i in range(len(rows))], edges


def matched_gap(rows, metric="ccnorm", covariates=("rate_hz", "psth_r2"),
                n_bins=4):
    """Stratum-weighted focal-minus-reference difference in `metric`.

    Only strata containing both subjects contribute; the weight is the number
    of focal units in the stratum, so the estimand is the gap over Logan's
    population had Allen units of the same difficulty been used as reference.
    """
    keys = [metric, *covariates]
    rows = _finite(rows, keys)
    cells, edges = _strata(rows, covariates, n_bins)

    by_cell = {}
    for r, c in zip(rows, cells):
        by_cell.setdefault(c, {REFERENCE: [], FOCAL: []})
        if r["subject"] in (REFERENCE, FOCAL):
            by_cell[c][r["subject"]].append(r[metric])

    num = den = 0.0
    used, dropped = 0, 0
    per_cell = []
    for cell, d in sorted(by_cell.items()):
        ref, foc = d[REFERENCE], d[FOCAL]
        if not ref or not foc:
            dropped += len(foc)
            continue
        diff = float(np.median(foc) - np.median(ref))
        w = float(len(foc))
        num += w * diff
        den += w
        used += len(foc)
        per_cell.append({"cell": list(cell), "n_ref": len(ref), "n_focal": len(foc),
                         "median_ref": float(np.median(ref)),
                         "median_focal": float(np.median(foc)), "diff": diff})

    return {
        "metric": metric,
        "covariates": list(covariates),
        "n_bins": n_bins,
        "gap": (num / den) if den else float("nan"),
        "n_focal_used": used,
        "n_focal_unmatched": dropped,
        "edges": edges,
        "cells": per_cell,
    }


def session_clustered_bootstrap(rows, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample sessions with replacement; return (lo, hi) 95% interval."""
    rng = np.random.default_rng(seed)
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    names = sorted(by_session)

    vals = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(names), size=len(names))
        resampled = []
        for d in draw:
            resampled.extend(by_session[names[d]])
        v = fn(resampled)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def raw_gap(rows, metric="ccnorm"):
    ref = [r[metric] for r in rows
           if r["subject"] == REFERENCE and np.isfinite(r[metric])]
    foc = [r[metric] for r in rows
           if r["subject"] == FOCAL and np.isfinite(r[metric])]
    if not ref or not foc:
        return float("nan")
    return float(np.median(foc) - np.median(ref))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="ccnorm", choices=["ccnorm", "ccabs"])
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="subject_gap.json")
    args = ap.parse_args()

    rows = build_unit_table()
    print(f"{len(rows)} units from {len({r['session'] for r in rows})} sessions")

    for subj in (REFERENCE, FOCAL):
        s = [r for r in rows if r["subject"] == subj]
        print(f"  {subj}: {len(s)} units, "
              f"median rate {np.nanmedian([r['rate_hz'] for r in s]):.1f} Hz, "
              f"PSTH R2 {np.nanmedian([r['psth_r2'] for r in s]):.3f}, "
              f"CC_max {np.nanmedian([r['ccmax'] for r in s]):.3f}, "
              f"trials {np.median([r['n_trials'] for r in s]):.0f}")

    m = args.metric
    out = {"metric": m, "n_units": len(rows)}

    raw = raw_gap(rows, m)
    raw_ci = session_clustered_bootstrap(
        rows, lambda rr: raw_gap(rr, m), n_boot=args.n_boot)
    out["raw"] = {"gap": raw, "ci95": raw_ci}
    print(f"\nRaw {FOCAL}-{REFERENCE} {m} gap: {raw:+.3f} "
          f"[{raw_ci[0]:+.3f}, {raw_ci[1]:+.3f}]")

    for covs in (("rate_hz",), ("psth_r2",), ("rate_hz", "psth_r2"),
                 ("rate_hz", "psth_r2", "n_trials")):
        res = matched_gap(rows, metric=m, covariates=covs, n_bins=args.bins)
        ci = session_clustered_bootstrap(
            rows,
            lambda rr, c=covs: matched_gap(rr, metric=m, covariates=c,
                                           n_bins=args.bins)["gap"],
            n_boot=args.n_boot)
        res["ci95"] = ci
        out[f"matched_{'+'.join(covs)}"] = res
        print(f"Matched on {'+'.join(covs):<34} {res['gap']:+.3f} "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"({res['n_focal_used']} matched, {res['n_focal_unmatched']} dropped)")

    path = HERE / args.out
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
