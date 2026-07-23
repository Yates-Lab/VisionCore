"""Hunt H5: eye-velocity TEMPORAL FILTER (integrated velocity drive).

Subagent #5 of the covariate hunt. Tests whether a jointly-fit velocity-history
kernel (temporal integration of drift speed, as the ConvGRU core would do)
recovers more of the extraretinal gap than #1's INSTANTANEOUS speed nonlinearity.

Base "both" saccade-kernel median recovered = 0.4028.

Runs are parametrized by argv[1] (a stage name) to keep each `uv run` to ~6-10
evals (~2 min). Reuses helpers from _hunt_h1_drift_velocity.

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h5_velocity_filter.py <stage>
  stages: diag | freelag | rcbasis | mult | vel2d | combined
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "covariate_hunt"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _supp_saccade_data import DT  # noqa: E402
from _hunt_h1_drift_velocity import valid_indicator  # noqa: E402


# ---------------------------------------------------------------------------
# lagged-signal matrices aligned to rec["tr"], rec["b"]
# ---------------------------------------------------------------------------
def _lagged_matrix(full_TB, tr, b0, lags):
    """(S, nlag) values of full_TB[tr, b0+lag]; out-of-range -> 0, NaN -> 0."""
    B = full_TB.shape[1]
    S = len(tr)
    M = np.zeros((S, len(lags)))
    for j, lag in enumerate(lags):
        b = b0 + lag
        ok = (b >= 0) & (b < B)
        vals = np.zeros(S)
        if ok.any():
            vals[ok] = full_TB[tr[ok], b[ok]]
        M[:, j] = np.where(np.isfinite(vals), vals, 0.0)
    return M


def lagged_logspeed(rec, designs, lags):
    s = rec["session"]
    M = _lagged_matrix(designs[s]["speed"], rec["tr"], rec["b"].astype(np.int64), lags)
    return np.log1p(M)


def lagged_speed_raw(rec, designs, lags):
    s = rec["session"]
    return _lagged_matrix(designs[s]["speed"], rec["tr"], rec["b"].astype(np.int64), lags)


def _velocity_TB(designs, s):
    ep = designs[s]["eyepos"]                       # (T,B,2)
    vel = np.full_like(ep, np.nan)
    vel[:, 1:-1, :] = (ep[:, 2:, :] - ep[:, :-2, :]) / (2 * DT)
    return vel


def lagged_velocity(rec, designs, lags):
    """(S, 2*nlag) lagged vx,vy."""
    s = rec["session"]
    vel = _velocity_TB(designs, s)
    tr, b0 = rec["tr"], rec["b"].astype(np.int64)
    vx = _lagged_matrix(vel[:, :, 0], tr, b0, lags)
    vy = _lagged_matrix(vel[:, :, 1], tr, b0, lags)
    return np.column_stack([vx, vy])


# ---------------------------------------------------------------------------
# raised-cosine basis over the lag axis (temporal regularization)
# ---------------------------------------------------------------------------
def rc_lag_basis(lags, n_basis):
    """(nlag, n_basis) raised-cosine bumps in linear lag space."""
    lags = np.asarray(lags, float)
    if n_basis == 1:
        return np.ones((len(lags), 1))
    centers = np.linspace(lags.min(), lags.max(), n_basis)
    width = centers[1] - centers[0]
    Bm = np.zeros((len(lags), n_basis))
    for j, c in enumerate(centers):
        d = (lags - c) / width
        Bm[:, j] = np.where(np.abs(d) < 1.0, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
    return Bm


# ---------------------------------------------------------------------------
# column builders
# ---------------------------------------------------------------------------
def make_freelag_add(lags, log=True):
    fn = lagged_logspeed if log else lagged_speed_raw
    def builder(rec, designs):
        return fn(rec, designs, lags)
    return builder


def make_rc_add(lags, n_basis, log=True):
    Bm = rc_lag_basis(lags, n_basis)
    fn = lagged_logspeed if log else lagged_speed_raw
    def builder(rec, designs):
        M = fn(rec, designs, lags)          # (S, nlag)
        return M @ Bm                        # (S, n_basis)
    return builder


def make_rc_mult(lags, n_basis, log=True):
    Bm = rc_lag_basis(lags, n_basis)
    fn = lagged_logspeed if log else lagged_speed_raw
    def builder(rec, designs):
        M = fn(rec, designs, lags)          # (S, nlag)
        F = M @ Bm                           # (S, n_basis)
        return rec["drive"][:, None] * F
    return builder


def make_freelag_mult(lags, log=True):
    fn = lagged_logspeed if log else lagged_speed_raw
    def builder(rec, designs):
        M = fn(rec, designs, lags)
        return rec["drive"][:, None] * M
    return builder


def make_vel2d_rc_add(lags, n_basis):
    Bm = rc_lag_basis(lags, n_basis)
    def builder(rec, designs):
        vv = lagged_velocity(rec, designs, lags)     # (S, 2*nlag)
        nlag = len(lags)
        vx, vy = vv[:, :nlag], vv[:, nlag:]
        return np.column_stack([vx @ Bm, vy @ Bm])
    return builder


# instantaneous #1 reference (add + mult logspeed-quad)
def inst_logspeed_quad(rec, designs):
    s = rec["session"]
    sp = designs[s]["speed"][rec["tr"], rec["b"]]
    x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
    return np.column_stack([x, x * x])


def inst_logspeed_quad_mult(rec, designs):
    f = inst_logspeed_quad(rec, designs)
    return rec["drive"][:, None] * f


# ---------------------------------------------------------------------------
LAGS = {
    "wide":   list(range(-18, 7)),    # -150 .. +50 ms  (25 lags, past-heavy)
    "mid":    list(range(-12, 5)),    # -100 .. +33 ms  (17 lags)
    "narrow": list(range(-6, 4)),     # -50  .. +25 ms  (10 lags)
    "past":   list(range(-24, 1)),    # -200 .. 0 ms    (25 lags, integration)
}


def _lag_ms(lags):
    return f"[{lags[0]*DT*1e3:+.0f}..{lags[-1]*DT*1e3:+.0f} ms, {len(lags)} lags]"


def stage_diag(ctx):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    # autocorrelation of speed to justify temporal-filter DOF
    r = recs[0]
    s = r["session"]
    sp = designs[s]["speed"]
    col = sp[:, sp.shape[1] // 2]
    col = col[np.isfinite(col)]
    print(f"speed sample col n={len(col)} mean={col.mean():.2f} std={col.std():.2f}")
    # base sanity
    evaluate_augmentation([], "baseline-check", ctx)


def stage_freelag(ctx):
    print("=== free-lag additive temporal filter (one col/lag, log-speed) ===")
    for w in ("narrow", "mid", "wide"):
        evaluate_augmentation([valid_indicator, make_freelag_add(LAGS[w], log=True)],
                              f"freelag add log {w} {_lag_ms(LAGS[w])}", ctx)
    print("=== free-lag additive, RAW speed ===")
    evaluate_augmentation([valid_indicator, make_freelag_add(LAGS["mid"], log=False)],
                          f"freelag add raw mid {_lag_ms(LAGS['mid'])}", ctx)


def stage_rcbasis(ctx):
    print("=== RC-basis additive temporal filter (log-speed) ===")
    for w in ("narrow", "mid", "wide", "past"):
        for nb in (3, 5):
            evaluate_augmentation([valid_indicator, make_rc_add(LAGS[w], nb, log=True)],
                                  f"RC add log {w} nb={nb} {_lag_ms(LAGS[w])}", ctx)


def stage_mult(ctx):
    print("=== multiplicative RC temporal filter drive*speed(t-tau) ===")
    for w in ("narrow", "mid", "wide"):
        evaluate_augmentation([valid_indicator, make_rc_mult(LAGS[w], 5, log=True)],
                              f"RC mult log {w} nb=5 {_lag_ms(LAGS[w])}", ctx)
    print("=== add+mult RC (best window) ===")
    for w in ("mid", "wide"):
        evaluate_augmentation(
            [valid_indicator, make_rc_add(LAGS[w], 5, log=True),
             make_rc_mult(LAGS[w], 5, log=True)],
            f"RC add+mult log {w} nb=5", ctx)


def stage_vel2d(ctx):
    print("=== 2D velocity RC temporal filter (vx,vy history, additive) ===")
    for w in ("mid", "wide"):
        for nb in (3, 5):
            evaluate_augmentation([valid_indicator, make_vel2d_rc_add(LAGS[w], nb)],
                                  f"vel2D RC add {w} nb={nb}", ctx)


def stage_combined(ctx):
    print("=== instantaneous #1 reference ===")
    evaluate_augmentation([valid_indicator, inst_logspeed_quad, inst_logspeed_quad_mult],
                          "inst add+mult logspeed-quad (#1 ref)", ctx)
    print("=== instantaneous + temporal filter (best RC windows) ===")
    for w in ("mid", "wide"):
        for nb in (3, 5):
            evaluate_augmentation(
                [valid_indicator, inst_logspeed_quad, inst_logspeed_quad_mult,
                 make_rc_add(LAGS[w], nb, log=True)],
                f"inst + RC-add {w} nb={nb}", ctx)
    print("=== instantaneous + add+mult temporal filter ===")
    evaluate_augmentation(
        [valid_indicator, inst_logspeed_quad, inst_logspeed_quad_mult,
         make_rc_add(LAGS["mid"], 5, log=True), make_rc_mult(LAGS["mid"], 5, log=True)],
        "inst + RC add+mult mid nb=5", ctx)


STAGES = {
    "diag": stage_diag, "freelag": stage_freelag, "rcbasis": stage_rcbasis,
    "mult": stage_mult, "vel2d": stage_vel2d, "combined": stage_combined,
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "diag"
    ctx = load_augment_context()
    STAGES[stage](ctx)
