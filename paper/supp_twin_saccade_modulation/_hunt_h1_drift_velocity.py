"""Hunt H1: continuous drift-velocity modulation of the extraretinal gap.

Standalone experiment script for the variance hunt (subagent #1).
Builds ctx ONCE, then evaluates interpretable drift-speed / eye-velocity columns
augmenting the "both" saccade-kernel baseline (base median recovered = 0.4028).

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h1_drift_velocity.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _supp_saccade_data import DT  # noqa: E402


# ---------------------------------------------------------------------------
# helpers to pull per-sample signals aligned to rec["tr"], rec["b"]
# ---------------------------------------------------------------------------
def _speed(rec, designs):
    s = rec["session"]
    return designs[s]["speed"][rec["tr"], rec["b"]]  # (S,), NaN at edges/gate


def _eyepos(rec, designs):
    s = rec["session"]
    return designs[s]["eyepos"][rec["tr"], rec["b"], :]  # (S,2)


def _velocity(rec, designs):
    """Central-diff eye velocity vector (vx, vy) deg/s aligned to samples.
    Computed per full (T,B,2) array then indexed, so edges/gate -> NaN."""
    s = rec["session"]
    ep = designs[s]["eyepos"]                     # (T,B,2)
    vel = np.full_like(ep, np.nan)
    vel[:, 1:-1, :] = (ep[:, 2:, :] - ep[:, :-2, :]) / (2 * DT)
    return vel[rec["tr"], rec["b"], :]            # (S,2)


def _lagged_speed(rec, designs, lag_bins):
    """Speed at bin b+lag_bins (lag>0 => future eye pos => speed leads gap)."""
    s = rec["session"]
    sp_full = designs[s]["speed"]                 # (T,B)
    B = sp_full.shape[1]
    b = rec["b"].astype(np.int64) + lag_bins
    tr = rec["tr"]
    out = np.full(len(tr), np.nan)
    ok = (b >= 0) & (b < B)
    out[ok] = sp_full[tr[ok], b[ok]]
    return out


# ---------------------------------------------------------------------------
# column builders
# ---------------------------------------------------------------------------
def valid_indicator(rec, designs):
    """1 where drift speed is finite, else 0. Guards zero-fill confound."""
    sp = _speed(rec, designs)
    return np.isfinite(sp).astype(np.float64)


def speed_linear(rec, designs):
    sp = _speed(rec, designs)
    return sp  # harness zero-fills NaN


def speed_log(rec, designs):
    sp = _speed(rec, designs)
    return np.log1p(sp)


def speed_quad(rec, designs):
    sp = _speed(rec, designs)
    return np.column_stack([sp, sp * sp])


def _rc_basis(x, knots, width):
    """Raised-cosine bumps at knots (log-speed space). Returns (S,len(knots))."""
    cols = []
    for k in knots:
        d = (x - k) / width
        b = np.where(np.abs(d) < 1.0, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
        cols.append(b)
    return np.column_stack(cols)


def make_speed_rc(knots_log, width):
    def builder(rec, designs):
        sp = _speed(rec, designs)
        x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
        return _rc_basis(x, knots_log, width)
    return builder


def make_speed_mult(fn):
    """Multiplicative: drive * f(speed). fn returns (S,) or (S,k)."""
    def builder(rec, designs):
        f = fn(rec, designs)
        f = np.asarray(f, dtype=np.float64)
        if f.ndim == 1:
            f = f[:, None]
        f = np.where(np.isfinite(f), f, 0.0)
        return rec["drive"][:, None] * f
    return builder


def velocity_vec(rec, designs):
    v = _velocity(rec, designs)  # (S,2)
    return v


def velocity_dir(rec, designs):
    """speed * (cos, sin) direction = velocity vector already; use magnitude x unit."""
    v = _velocity(rec, designs)
    sp = np.hypot(v[:, 0], v[:, 1])
    ang = np.arctan2(v[:, 1], v[:, 0])
    return np.column_stack([sp * np.cos(ang), sp * np.sin(ang),
                            np.cos(ang), np.sin(ang)])


def make_lagged_speed(lag_bins):
    def builder(rec, designs):
        return _lagged_speed(rec, designs, lag_bins)
    return builder


def make_lagged_speed_mult(lag_bins):
    def builder(rec, designs):
        sp = _lagged_speed(rec, designs, lag_bins)
        sp = np.where(np.isfinite(sp), sp, 0.0)
        return rec["drive"] * sp
    return builder


# ---------------------------------------------------------------------------
def diagnostics(ctx):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    allsp = []
    frac_valid = []
    for rec in recs:
        sp = _speed(rec, designs)
        frac_valid.append(np.isfinite(sp).mean())
        allsp.append(sp[np.isfinite(sp)])
    allsp = np.concatenate(allsp)
    print("=== speed diagnostics (reliable units) ===")
    print(f"samples with finite speed: mean frac/unit = {np.mean(frac_valid):.3f}")
    qs = [1, 5, 25, 50, 75, 95, 99]
    pcts = np.percentile(allsp, qs)
    for q, p in zip(qs, pcts):
        print(f"  speed p{q:02d} = {p:7.2f} deg/s   log1p={np.log1p(p):.3f}")
    print(f"  max speed = {allsp.max():.1f} deg/s")
    return allsp


# --- round 2 helpers ---
def make_speed_capped(cap):
    def builder(rec, designs):
        sp = _speed(rec, designs)
        return np.minimum(np.where(np.isfinite(sp), sp, 0.0), cap)
    return builder


def make_logspeed_capped(cap):
    """log1p speed but zeroed above cap deg/s (isolate low-speed drift)."""
    def builder(rec, designs):
        sp = _speed(rec, designs)
        spc = np.where(np.isfinite(sp) & (sp <= cap), sp, np.nan)
        return np.column_stack([np.log1p(np.where(np.isfinite(spc), spc, 0.0)),
                                (np.log1p(np.where(np.isfinite(spc), spc, 0.0))) ** 2])
    return builder


def make_lagged_logspeed_quad(lag_bins):
    def builder(rec, designs):
        sp = _lagged_speed(rec, designs, lag_bins)
        x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
        return np.column_stack([x, x * x])
    return builder


if __name__ == "__main__":
    ctx = load_augment_context()
    allsp = diagnostics(ctx)
    knots = [0.5, 1.5, 2.5, 3.5]

    def logspeed_quad(rec, designs):
        sp = _speed(rec, designs)
        x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
        return np.column_stack([x, x * x])

    print("\n=== R2: best single channels ===")
    evaluate_augmentation([valid_indicator, speed_quad], "add: speed+speed^2", ctx)
    evaluate_augmentation([valid_indicator, logspeed_quad], "add: logspeed quad", ctx)
    evaluate_augmentation([valid_indicator, make_speed_mult(make_speed_rc(knots, 1.0))],
                          "mult: drive*logspeed RC(4)", ctx)

    print("\n=== R2: combined additive + multiplicative speed ===")
    evaluate_augmentation(
        [valid_indicator, logspeed_quad, make_speed_mult(logspeed_quad)],
        "add logspeed-quad + mult logspeed-quad", ctx)
    evaluate_augmentation(
        [valid_indicator, make_speed_rc(knots, 1.0), make_speed_mult(make_speed_rc(knots, 1.0))],
        "add RC + mult RC", ctx)

    print("\n=== R2: cap high-speed tail (isolate drift, not saccades) ===")
    for cap in (2.0, 3.0, 5.0, 10.0):
        evaluate_augmentation([valid_indicator, make_logspeed_capped(cap)],
                              f"add: logspeed quad, speed<= {cap} deg/s", ctx)

    print("\n=== R2: lag sweep (speed leads/lags gap), additive logspeed quad ===")
    for lag in (-4, -2, -1, 0, 1, 2, 4, 6):
        evaluate_augmentation([valid_indicator, make_lagged_logspeed_quad(lag)],
                              f"add: logspeed quad, lag {lag:+d} bins ({lag*DT*1e3:+.0f} ms)", ctx)
