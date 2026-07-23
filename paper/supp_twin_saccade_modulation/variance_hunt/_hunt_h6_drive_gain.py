"""Hunt H6: drive-dependent tonic gain (stimulus-locked, not movement-locked).

Hypothesis: the twin's behavior pathway may impose a *tonic* gain/offset on the
stimulus drive even at zero eye velocity, not tied to any saccade or drift event.
Then the gap y = full - ablated would be a static function of the DRIVE itself
(rec["drive"] = ablated stimulus-driven rate) -- invisible to saccade-triggered
averages AND to velocity analyses.

The baseline "both" design has drive*Aadd (drive x saccade kernel) but NO
standalone drive term outside saccade windows, so a tonic drive-proportional gain
is currently unmodeled.

drive is an INPUT-SIDE regressor (ablated model output, computed without behavior),
NOT derived from the target y or from full -> no leakage.

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h6_drive_gain.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "variance_hunt"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402


# ---------------------------------------------------------------------------
# per-sample signal helpers (aligned to rec["tr"], rec["b"])
# ---------------------------------------------------------------------------
def _speed(rec, designs):
    s = rec["session"]
    return designs[s]["speed"][rec["tr"], rec["b"]]  # (S,), NaN at edges/gate


def _logspeed(rec, designs):
    sp = _speed(rec, designs)
    return np.log1p(np.where(np.isfinite(sp), sp, 0.0))


def speed_valid(rec, designs):
    return np.isfinite(_speed(rec, designs)).astype(np.float64)


# ---------------------------------------------------------------------------
# DRIVE column builders (the hypothesis)
# ---------------------------------------------------------------------------
def drive_linear(rec, designs):
    """Pure tonic multiplicative gain offset: a constant fraction of drive."""
    return rec["drive"].astype(np.float64)


def drive_quad(rec, designs):
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d, d * d])


def logdrive_quad(rec, designs):
    d = np.log1p(rec["drive"].astype(np.float64))
    return np.column_stack([d, d * d])


def drive_cubic(rec, designs):
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d, d * d, d * d * d])


def _rc_basis(x, knots, width):
    cols = []
    for k in knots:
        dd = (x - k) / width
        b = np.where(np.abs(dd) < 1.0, 0.5 * (1 + np.cos(np.pi * dd)), 0.0)
        cols.append(b)
    return np.column_stack(cols)


def make_drive_rc(n_knots=5):
    """Raised-cosine bumps over this unit's own drive percentile knots.
    Piecewise-linear-ish basis capturing an arbitrary static drive nonlinearity."""
    def builder(rec, designs):
        d = rec["drive"].astype(np.float64)
        qs = np.linspace(0, 100, n_knots)
        knots = np.percentile(d, qs)
        # dedupe / guard degenerate knots
        knots = np.unique(knots)
        if len(knots) < 2:
            return None
        width = np.median(np.diff(knots)) if len(knots) > 1 else 1.0
        width = width if width > 0 else 1.0
        return _rc_basis(d, knots, width * 1.0)
    return builder


def make_pl_basis(n_knots=5):
    """Piecewise-linear tent basis over per-unit drive percentile knots."""
    def builder(rec, designs):
        d = rec["drive"].astype(np.float64)
        qs = np.linspace(0, 100, n_knots)
        knots = np.unique(np.percentile(d, qs))
        if len(knots) < 3:
            return None
        cols = []
        for i, k in enumerate(knots):
            lo = knots[i - 1] if i > 0 else knots[0] - (knots[1] - knots[0])
            hi = knots[i + 1] if i < len(knots) - 1 else knots[-1] + (knots[-1] - knots[-2])
            left = (d - lo) / (k - lo) if k > lo else np.zeros_like(d)
            right = (hi - d) / (hi - k) if hi > k else np.zeros_like(d)
            tent = np.clip(np.minimum(left, right), 0.0, 1.0)
            cols.append(tent)
        return np.column_stack(cols)
    return builder


# --- interactions -----------------------------------------------------------
def drive_x_time(rec, designs):
    """drive * [t, t^2] : does the tonic gain drift/adapt over the trial?"""
    B = designs[rec["session"]]["B"]
    t = rec["b"].astype(np.float64) / B
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d * t, d * t * t])


def drive_x_speed(rec, designs):
    """drive * speed : is the velocity gain (#1) actually a drive-scaled gain?"""
    sp = _speed(rec, designs)
    sp = np.where(np.isfinite(sp), sp, 0.0)
    return rec["drive"].astype(np.float64) * sp


def drive_x_logspeed(rec, designs):
    ls = _logspeed(rec, designs)
    return rec["drive"].astype(np.float64) * ls


# --- #1 instantaneous logspeed reference model ------------------------------
def logspeed_quad(rec, designs):
    x = _logspeed(rec, designs)
    return np.column_stack([x, x * x])


def mult_drive_logspeed_quad(rec, designs):
    x = _logspeed(rec, designs)
    d = rec["drive"].astype(np.float64)
    return np.column_stack([d * x, d * x * x])


# ---------------------------------------------------------------------------
def diagnostics(ctx):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    # global drive distribution
    alld = np.concatenate([r["drive"] for r in recs])
    print("=== drive diagnostics (reliable units) ===")
    qs = [1, 5, 25, 50, 75, 95, 99]
    for q, p in zip(qs, np.percentile(alld, qs)):
        print(f"  drive p{q:02d} = {p:8.3f}")
    print(f"  drive max = {alld.max():.2f}, min = {alld.min():.3f}, "
          f"frac==0 = {(alld == 0).mean():.3f}")

    # per-unit decile shape: rank each unit's drive into deciles, mean gap per
    # decile, then average across units. Reveals the WITHIN-unit gap-vs-drive
    # shape that per-unit OLS actually sees.
    ND = 10
    per_unit = np.full((len(recs), ND), np.nan)
    for ui, r in enumerate(recs):
        d = r["drive"]
        y = r["y"]
        edges = np.percentile(d, np.linspace(0, 100, ND + 1))
        idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, ND - 1)
        for k in range(ND):
            m = idx == k
            if m.sum() > 3:
                per_unit[ui, k] = y[m].mean()
    # center each unit by its own mean gap so we see the SHAPE, not level
    centered = per_unit - np.nanmean(per_unit, axis=1, keepdims=True)
    mean_shape = np.nanmean(centered, axis=0)
    print("\n=== within-unit gap-vs-drive-decile SHAPE (mean-centered, avg over units) ===")
    print("  decile:  " + " ".join(f"{k:6d}" for k in range(ND)))
    print("  gap-ctr: " + " ".join(f"{v:+6.3f}" for v in mean_shape))
    # also raw (uncentered) mean gap per decile pooled magnitude
    raw_shape = np.nanmean(per_unit, axis=0)
    print("  gap-raw: " + " ".join(f"{v:+6.3f}" for v in raw_shape))
    # correlation of decile index with centered gap, per unit -> monotonicity
    dvec = np.arange(ND)
    corrs = []
    for ui in range(len(recs)):
        v = centered[ui]
        ok = np.isfinite(v)
        if ok.sum() >= 5:
            corrs.append(np.corrcoef(dvec[ok], v[ok])[0, 1])
    corrs = np.array(corrs)
    print(f"\n  per-unit corr(decile, centered-gap): median {np.median(corrs):+.3f}, "
          f"frac>0 {np.mean(corrs > 0):.2f}  (n={len(corrs)})")

    # a few example units
    print("\n=== example units (centered gap per decile) ===")
    for ui in [0, 50, 100, 200, 300]:
        if ui < len(recs):
            r = recs[ui]
            print(f"  unit {ui:3d} {r['session']}#{r.get('ni','?')}: " +
                  " ".join(f"{v:+5.2f}" for v in centered[ui]))
    return corrs


if __name__ == "__main__":
    ctx = load_augment_context()
    print(f"reliable neurons: {len(ctx['reliable_recs'])}\n")

    # sanity: base must be 0.4028
    evaluate_augmentation([], "baseline-check (no extra cols)", ctx)
    print()

    diagnostics(ctx)

    print("\n=== R1: additive static drive nonlinearity ===")
    evaluate_augmentation([drive_linear], "add: drive (tonic gain)", ctx)
    evaluate_augmentation([drive_quad], "add: drive + drive^2", ctx)
    evaluate_augmentation([logdrive_quad], "add: log1p(drive) quad", ctx)
    evaluate_augmentation([drive_cubic], "add: drive cubic", ctx)
    evaluate_augmentation([make_drive_rc(5)], "add: drive RC(5) percentile knots", ctx)
    evaluate_augmentation([make_pl_basis(6)], "add: drive piecewise-linear(6)", ctx)

    print("\n=== R1: drive x time-in-trial interaction ===")
    evaluate_augmentation([drive_x_time], "mult: drive*[t,t^2]", ctx)
    evaluate_augmentation([drive_linear, drive_x_time], "add drive + drive*[t,t^2]", ctx)

    print("\n=== R1: drive x speed (does it subsume #1?) ===")
    evaluate_augmentation([speed_valid, drive_x_speed], "mult: drive*speed", ctx)
    evaluate_augmentation([speed_valid, drive_x_logspeed], "mult: drive*logspeed", ctx)

    # ---- ROUND 2: does STATIC drive gain STACK with #1 speed model? ----
    speed1 = [speed_valid, logspeed_quad, mult_drive_logspeed_quad]  # #1 add+mult
    print("\n=== R2: #1 speed ref, and static-drive stacked on top ===")
    evaluate_augmentation(speed1, "#1 ref: add+mult logspeed-quad", ctx)
    evaluate_augmentation(speed1 + [drive_linear],
                          "#1 speed + drive (static tonic)", ctx)
    evaluate_augmentation(speed1 + [logdrive_quad],
                          "#1 speed + log1p(drive) quad", ctx)
    evaluate_augmentation(speed1 + [drive_x_time],
                          "#1 speed + drive*[t,t^2]", ctx)
    evaluate_augmentation(speed1 + [drive_linear, drive_x_time],
                          "#1 speed + drive + drive*[t,t^2]", ctx)

    print("\n=== R2: pure drive combos (no speed) ===")
    evaluate_augmentation([drive_linear, drive_x_time], "drive + drive*[t,t^2]", ctx)
    evaluate_augmentation([logdrive_quad, drive_x_time], "log1p(drive)q + drive*[t,t^2]", ctx)

    print("\n=== R2: best cumulative interpretable model ===")
    evaluate_augmentation(
        [speed_valid, logspeed_quad, mult_drive_logspeed_quad, drive_linear, drive_x_time],
        "CUM: #1 speed + drive + drive*t", ctx)
    evaluate_augmentation(
        [speed_valid, logspeed_quad, mult_drive_logspeed_quad, logdrive_quad],
        "CUM: #1 speed + log1p(drive)q", ctx)
