"""Hunt H9: joint (drift-speed x stimulus-drive) 2D gain surface.

#1 (drift-speed gain) and #6 (tonic drive gain) were fit as SEPARABLE channels.
Even so, the separable reference ALREADY contains a bilinear speed x drive term
(drive*logspeed, drive*logspeed^2 from #1's multiplicative channel). The question
here is strictly whether a NON-separable 2D function of (log-speed, log-drive) --
tensor-product interaction curvature -- adds held-out recovered OVER the separable
sum BEST6 = +0.0094 (0.4028 -> 0.4177).

Both `speed` and `drive` are input-side (ablated output / eyepos derivative), no
leakage from `full` or `y`.

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h9_joint_surface.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402

# reuse the exact reference builders from #1/#6
from _hunt_h6_drive_gain import (  # noqa: E402
    _speed, _logspeed, speed_valid, logspeed_quad,
    mult_drive_logspeed_quad, drive_x_time,
)


def _logdrive(rec):
    return np.log1p(rec["drive"].astype(np.float64))


# ---------------------------------------------------------------------------
# separable reference (BEST6) = #1 speed add+mult + #6 drive*[t,t^2]
# ---------------------------------------------------------------------------
BEST6 = [speed_valid, logspeed_quad, mult_drive_logspeed_quad, drive_x_time]


# ---------------------------------------------------------------------------
# 1-D raised-cosine bases (per-unit percentile knots), for a fair separable
# additive reference in the SAME basis family as the tensor product
# ---------------------------------------------------------------------------
def _rc_basis(x, knots, width):
    cols = []
    for k in knots:
        d = (x - k) / width
        b = np.where(np.abs(d) < 1.0, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
        cols.append(b)
    return np.column_stack(cols)


def _rc_1d(x, nknots):
    """RC bumps at per-sample-percentile knots of x. Returns (S, m>=2) basis."""
    qs = np.linspace(10, 90, nknots)
    knots = np.unique(np.percentile(x, qs))
    if len(knots) < 2:
        return None, None
    width = np.median(np.diff(knots))
    width = width if width > 0 else 1.0
    return _rc_basis(x, knots, width), knots


def make_rc_logspeed(nknots=3):
    def builder(rec, designs):
        x = _logspeed(rec, designs)
        B, _ = _rc_1d(x, nknots)
        return B
    return builder


def make_rc_logdrive(nknots=3):
    def builder(rec, designs):
        x = _logdrive(rec)
        B, _ = _rc_1d(x, nknots)
        return B
    return builder


def make_tensor(ns=3, nd=3, interaction_only=False):
    """Tensor-product RC basis over (log-speed, log-drive).

    interaction_only=False -> full ns*nd product columns (contains marginals).
    interaction_only=True  -> product columns with each dim mean-removed, i.e.
        the pure interaction after subtracting additive marginals (so it can be
        stacked ON TOP of the separable 1-D marginal bases as an isolated test).
    """
    def builder(rec, designs):
        xs = _logspeed(rec, designs)
        xd = _logdrive(rec)
        Bs, _ = _rc_1d(xs, ns)
        Bd, _ = _rc_1d(xd, nd)
        if Bs is None or Bd is None:
            return None
        if interaction_only:
            Bs = Bs - Bs.mean(axis=0, keepdims=True)
            Bd = Bd - Bd.mean(axis=0, keepdims=True)
        cols = []
        for i in range(Bs.shape[1]):
            for j in range(Bd.shape[1]):
                cols.append(Bs[:, i] * Bd[:, j])
        return np.column_stack(cols)
    return builder


# ---------------------------------------------------------------------------
# explicit low-DOF interaction terms to stack on BEST6
# ---------------------------------------------------------------------------
def logdrive_x_logspeed(rec, designs):
    return _logdrive(rec) * _logspeed(rec, designs)


def drive_x_speed(rec, designs):
    sp = _speed(rec, designs)
    sp = np.where(np.isfinite(sp), sp, 0.0)
    return rec["drive"].astype(np.float64) * sp


def drive_x_logspeed2(rec, designs):
    """drive*logspeed^2 already in BEST6; here logdrive-weighted speed curvature:
    (logdrive)*(logspeed, logspeed^2) -- speed gain that itself scales with drive
    beyond linear drive."""
    ls = _logspeed(rec, designs)
    ld = _logdrive(rec)
    return np.column_stack([ld * ls, ld * ls * ls])


def drive_speed_time(rec, designs):
    """3-way drive*speed*t : does the speed gain adapt over the trial, drive-scaled?"""
    B = designs[rec["session"]]["B"]
    t = rec["b"].astype(np.float64) / B
    ls = _logspeed(rec, designs)
    d = rec["drive"].astype(np.float64)
    return d * ls * t


def _rawspeed(rec, designs):
    sp = _speed(rec, designs)
    return np.where(np.isfinite(sp), sp, 0.0)


def drive_x_speed_quad(rec, designs):
    sp = _rawspeed(rec, designs)
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d * sp, d * sp * sp])


def drive_x_speed_t(rec, designs):
    B = designs[rec["session"]]["B"]
    t = rec["b"].astype(np.float64) / B
    sp = _rawspeed(rec, designs)
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d * sp, d * sp * t])


def make_drive_x_speed_cap(cap):
    def builder(rec, designs):
        sp = np.minimum(_rawspeed(rec, designs), cap)
        return rec["drive"].astype(np.float64) * sp
    return builder


# ---------------------------------------------------------------------------
# DIAGNOSTIC: 2D gap surface by (speed decile x drive decile), pooled
# ---------------------------------------------------------------------------
def diagnostic_surface(ctx, NS=5, ND=5):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    # accumulate per-unit-centered gap into (NS x ND) cells, averaged across units
    acc = np.zeros((NS, ND))
    cnt = np.zeros((NS, ND))
    for r in recs:
        sp = designs[r["session"]]["speed"][r["tr"], r["b"]]
        d = r["drive"].astype(np.float64)
        y = r["y"].astype(np.float64)
        ok = np.isfinite(sp)
        if ok.sum() < 50:
            continue
        sp, d, y = sp[ok], d[ok], y[ok]
        yc = y - y.mean()  # center per unit -> SHAPE not level
        se = np.percentile(sp, np.linspace(0, 100, NS + 1))
        de = np.percentile(d, np.linspace(0, 100, ND + 1))
        si = np.clip(np.searchsorted(se, sp, "right") - 1, 0, NS - 1)
        di = np.clip(np.searchsorted(de, d, "right") - 1, 0, ND - 1)
        for a in range(NS):
            for b in range(ND):
                m = (si == a) & (di == b)
                if m.any():
                    acc[a, b] += yc[m].mean()
                    cnt[a, b] += 1
    surf = acc / np.maximum(cnt, 1)
    # separable prediction = row-margin + col-margin - grand
    grand = surf.mean()
    rowm = surf.mean(axis=1, keepdims=True)
    colm = surf.mean(axis=0, keepdims=True)
    sep = rowm + colm - grand
    resid = surf - sep
    print(f"\n=== 2D gap surface (mean-centered, avg over units) "
          f"[rows=speed decile 0..{NS-1}, cols=drive decile 0..{ND-1}] ===")
    print("  (speed increases down, drive increases right)")
    for a in range(NS):
        print("  sp%d: " % a + " ".join(f"{surf[a,b]:+6.3f}" for b in range(ND)))
    print("\n  separable-fit residual (surface - [rowmargin+colmargin-grand]):")
    for a in range(NS):
        print("  sp%d: " % a + " ".join(f"{resid[a,b]:+6.3f}" for b in range(ND)))
    ss_tot = np.sum((surf - grand) ** 2)
    ss_res = np.sum(resid ** 2)
    print(f"\n  interaction SS / total SS = {ss_res/ss_tot:.3f}  "
          f"(fraction of surface variance NOT explained by additive margins)")
    print(f"  row (speed) margin: " + " ".join(f"{v:+.3f}" for v in rowm.ravel()))
    print(f"  col (drive) margin: " + " ".join(f"{v:+.3f}" for v in colm.ravel()))


if __name__ == "__main__":
    ctx = load_augment_context()
    print(f"reliable neurons: {len(ctx['reliable_recs'])}\n")

    # sanity: base must be 0.4028
    evaluate_augmentation([], "baseline-check (no extra cols)", ctx)

    diagnostic_surface(ctx, NS=5, ND=5)

    print("\n=== separable reference (BEST6) ===")
    evaluate_augmentation(BEST6, "BEST6 separable (#1 speed + drive*t)", ctx)

    print("\n=== basis-family check: RC separable vs RC joint tensor (ALONE, on saccade base) ===")
    evaluate_augmentation([speed_valid, make_rc_logspeed(3), make_rc_logdrive(3)],
                          "RC separable add: RC3(logsp)+RC3(logdr)", ctx)
    evaluate_augmentation([speed_valid, make_tensor(3, 3)],
                          "RC joint tensor 3x3 (full)", ctx)
    evaluate_augmentation([speed_valid, make_tensor(2, 2)],
                          "RC joint tensor 2x2 (full)", ctx)

    print("\n=== does an INTERACTION add over BEST6? (stack interaction cols on BEST6) ===")
    evaluate_augmentation(BEST6 + [logdrive_x_logspeed],
                          "BEST6 + logdrive*logspeed (bilinear)", ctx)
    evaluate_augmentation(BEST6 + [drive_x_speed],
                          "BEST6 + drive*speed (raw bilinear)", ctx)
    evaluate_augmentation(BEST6 + [drive_x_logspeed2],
                          "BEST6 + logdrive*(logsp,logsp^2)", ctx)
    evaluate_augmentation(BEST6 + [drive_speed_time],
                          "BEST6 + drive*logspeed*t (3-way)", ctx)
    evaluate_augmentation(BEST6 + [make_tensor(3, 3, interaction_only=True)],
                          "BEST6 + tensor3x3 interaction-only", ctx)
    evaluate_augmentation(BEST6 + [make_tensor(2, 2, interaction_only=True)],
                          "BEST6 + tensor2x2 interaction-only", ctx)

    print("\n=== CONFIRM: raw drive*speed interaction (the lone aug-median riser) ===")
    evaluate_augmentation([speed_valid, drive_x_speed],
                          "drive*speed (raw) ALONE on saccade base", ctx)
    evaluate_augmentation(BEST6 + [drive_x_speed],
                          "BEST6 + drive*speed (raw) [reconfirm]", ctx)
    evaluate_augmentation(BEST6 + [drive_x_speed_quad],
                          "BEST6 + drive*(speed, speed^2) raw", ctx)
    evaluate_augmentation(BEST6 + [drive_x_speed_t],
                          "BEST6 + drive*speed + drive*speed*t raw", ctx)
    for cap in (5.0, 10.0, 20.0):
        evaluate_augmentation(BEST6 + [make_drive_x_speed_cap(cap)],
                              f"BEST6 + drive*min(speed,{cap:.0f})", ctx)

    print("\n=== FINAL: cap plateau + add-vs-replace mechanism ===")
    for cap in (3.0, 8.0, 12.0):
        evaluate_augmentation(BEST6 + [make_drive_x_speed_cap(cap)],
                              f"BEST6 + drive*min(speed,{cap:.0f})", ctx)
    # replace #1's log mult channel with raw-capped: is the raw term ADDED info
    # or just a better basis for the SAME speed x drive interaction?
    NO_LOGMULT = [speed_valid, logspeed_quad, drive_x_time]  # BEST6 minus mult_drive_logspeed_quad
    for cap in (5.0, 8.0, 10.0, 15.0):
        evaluate_augmentation(NO_LOGMULT + [make_drive_x_speed_cap(cap)],
                              f"REPLACE: log-mult -> drive*min(speed,{cap:.0f})", ctx)
    # also: keep BOTH log-mult and raw-capped (does raw ADD on top of log-mult, or is
    # it purely a substitute?) -- already tested as "BEST6 + drive*min(speed,10)" above.
