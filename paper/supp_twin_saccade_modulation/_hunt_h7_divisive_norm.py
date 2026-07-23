"""Hunt H7: proper divisive-normalization / gain-reduction functional form.

#6 found the gap y = full - ablated is monotonically DECREASING in stimulus
drive (median within-unit corr -0.51, 73% units negative): the behavior pathway
ADDS at low drive and SUBTRACTS at high drive -- a gain-reduction / divisive-
normalization signature. #6 modelled that with ADDITIVE log-drive proxies
(log1p(drive) quad +0.0035 alone; drive*[t,t^2] +0.0046; best cumulative with #1
speed = +0.0094).

Here we test the ACTUAL divisive functional form. Since drive ~ ablated rate,
a divisive full = ablated / (1 + kappa*P) gives

    y = full - ablated = ablated * (1/(1+kappa*P) - 1) = -ablated * kappa*P/(1+kappa*P)

so the single interpretable regressor is  -drive * P/(1+kappa*P)  for a
normalization pool P and a saturation kappa. OLS absorbs the sign and the
overall -kappa scale, so we fit column  drive * P/(1+kappa*P)  and expect a
NEGATIVE beta (gain reduction).  We sweep kappa (small hyperparameter sweep,
stated) and try pools P = (a) drive itself, (b) a causal running/exponential
drive average over recent bins (a real normalization pool integrates nearby
drive -- adaptation), (c) per-trial mean drive.

drive/speed are input-side (ablated output / eye signal); no leakage from y/full.

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h7_divisive_norm.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _hunt_h6_drive_gain import (  # noqa: E402
    speed_valid, logspeed_quad, mult_drive_logspeed_quad,
    drive_linear, logdrive_quad, drive_x_time,
)


# ---------------------------------------------------------------------------
# normalization-pool builders (per-sample, aligned to rec["tr"], rec["b"])
# ---------------------------------------------------------------------------
def pool_drive(rec, designs):
    """Pool = instantaneous drive itself (classic self-normalization)."""
    return rec["drive"].astype(np.float64)


def _running_drive(rec, tau_bins):
    """Causal exponential running average of drive within each trial, aligned to
    samples. Handles non-contiguous bins (gated dropouts) via a^(bin gap).
    Includes the current sample in the pool (P -> drive as tau -> 0)."""
    tr = rec["tr"].astype(np.int64)
    b = rec["b"].astype(np.int64)
    d = rec["drive"].astype(np.float64)
    a = np.exp(-1.0 / float(tau_bins))
    order = np.lexsort((b, tr))            # sort by trial, then bin
    out = np.empty(len(d), dtype=np.float64)
    ema = 0.0
    norm = 0.0
    prev_tr = -1
    prev_b = 0
    for idx in order:
        ti = tr[idx]
        bi = b[idx]
        if ti != prev_tr:
            ema = d[idx]
            norm = 1.0
        else:
            decay = a ** max(bi - prev_b, 0)
            ema = decay * ema + d[idx]
            norm = decay * norm + 1.0
        out[idx] = ema / norm
        prev_tr = ti
        prev_b = bi
    return out


def make_pool_running(tau_bins):
    def builder(rec, designs):
        return _running_drive(rec, tau_bins)
    return builder


def pool_trialmean(rec, designs):
    """Pool = per-trial mean drive (slow normalization set-point)."""
    tr = rec["tr"].astype(np.int64)
    d = rec["drive"].astype(np.float64)
    uniq, inv = np.unique(tr, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    means = sums / np.maximum(counts, 1)
    return means[inv]


# ---------------------------------------------------------------------------
# divisive column: drive * P / (1 + kappa*P), scaled to unit std for conditioning
# (single column -> scaling does not change the OLS fit, only conditioning).
# ---------------------------------------------------------------------------
def make_divisive(pool_fn, kappa, name=""):
    def builder(rec, designs):
        P = np.asarray(pool_fn(rec, designs), dtype=np.float64)
        P = np.where(np.isfinite(P), P, 0.0)
        d = rec["drive"].astype(np.float64)
        col = d * P / (1.0 + kappa * P)
        s = np.std(col)
        if s > 0:
            col = col / s
        return col
    builder.__name__ = f"divisive_{name}_k{kappa}"
    return builder


def make_running_interaction(tau_bins):
    """drive * running-drive-history (linear interaction) -- the 'adaptation'
    reading of the drive*t win: gain scales with recent cumulative drive."""
    def builder(rec, designs):
        P = _running_drive(rec, tau_bins)
        d = rec["drive"].astype(np.float64)
        col = d * P
        s = np.std(col)
        return col / s if s > 0 else col
    return builder


# ---------------------------------------------------------------------------
def sign_consistency(ctx, pool_fn, kappa, label):
    """Per-unit: fit y ~ [1, divcol] and report fraction with NEGATIVE beta
    (expected gain-reduction sign)."""
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    build = make_divisive(pool_fn, kappa)
    betas = []
    for rec in recs:
        col = np.asarray(build(rec, designs), dtype=np.float64)
        col = np.where(np.isfinite(col), col, 0.0)
        y = rec["y"]
        X = np.column_stack([np.ones_like(col), col])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        betas.append(beta[1])
    betas = np.array(betas)
    frac_neg = float(np.mean(betas < 0))
    print(f"  [{label}] per-unit divisive beta: median {np.median(betas):+.4f}, "
          f"frac NEGATIVE (gain-reduction) {frac_neg:.2f}  (n={len(betas)})")
    return frac_neg


if __name__ == "__main__":
    ctx = load_augment_context()
    print(f"reliable neurons: {len(ctx['reliable_recs'])}\n")

    evaluate_augmentation([], "baseline-check (no extra cols)", ctx)

    # --- reference: #6 additive proxies -------------------------------------
    print("\n=== REF: #6 additive drive proxies (alone, on saccade base) ===")
    evaluate_augmentation([drive_linear], "add: drive (tonic)", ctx)
    evaluate_augmentation([logdrive_quad], "add: log1p(drive) quad", ctx)
    evaluate_augmentation([drive_x_time], "mult: drive*[t,t^2]", ctx)

    # --- R1: divisive form, pool = drive, kappa sweep -----------------------
    print("\n=== R1: divisive  drive*drive/(1+k*drive)  [pool=drive], kappa sweep ===")
    kappas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    r1 = {}
    for k in kappas:
        r = evaluate_augmentation([make_divisive(pool_drive, k)],
                                  f"divisive pool=drive kappa={k}", ctx)
        r1[k] = r["med_delta"]
    best_k_drive = max(r1, key=r1.get)
    print(f"  -> best kappa (pool=drive) = {best_k_drive}  (Δ {r1[best_k_drive]:+.4f})")

    # --- R2: divisive form, pool = running drive average, tau sweep ---------
    print("\n=== R2: divisive with RUNNING-drive pool, tau sweep (kappa=best drive) ===")
    r2 = {}
    for tau in [2, 4, 8, 16]:
        r = evaluate_augmentation([make_divisive(make_pool_running(tau), best_k_drive)],
                                  f"divisive pool=running(tau={tau}) k={best_k_drive}", ctx)
        r2[("run", tau)] = r["med_delta"]
    print("\n=== R2b: divisive with per-trial-mean pool (kappa sweep) ===")
    for k in [0.005, 0.02, 0.05, 0.1]:
        evaluate_augmentation([make_divisive(pool_trialmean, k)],
                              f"divisive pool=trialmean k={k}", ctx)

    # --- R3: adaptation axis: drive x running-history interaction -----------
    print("\n=== R3: adaptation -- drive * running-drive-history (linear) ===")
    for tau in [4, 8, 16]:
        evaluate_augmentation([make_running_interaction(tau)],
                              f"drive*running(tau={tau})", ctx)
    print("  (compare to #6 drive*[t,t^2] above)")

    # --- R4: best cumulative model ------------------------------------------
    speed1 = [speed_valid, logspeed_quad, mult_drive_logspeed_quad]
    best_div = make_divisive(pool_drive, best_k_drive)
    print("\n=== R4: cumulative -- #1 speed + divisive (+/- drive*t) ===")
    evaluate_augmentation(speed1, "#1 speed ref", ctx)
    evaluate_augmentation(speed1 + [drive_x_time],
                          "#6 best: #1 speed + drive*[t,t^2]", ctx)
    evaluate_augmentation(speed1 + [best_div],
                          "#1 speed + divisive(drive)", ctx)
    evaluate_augmentation(speed1 + [best_div, drive_x_time],
                          "#1 speed + divisive(drive) + drive*[t,t^2]", ctx)
    evaluate_augmentation([best_div, drive_x_time],
                          "divisive(drive) + drive*[t,t^2] (no speed)", ctx)

    # --- R5: per-unit sign consistency of the divisive term -----------------
    print("\n=== R5: per-unit sign consistency (expected NEGATIVE beta) ===")
    sign_consistency(ctx, pool_drive, best_k_drive, f"pool=drive k={best_k_drive}")
