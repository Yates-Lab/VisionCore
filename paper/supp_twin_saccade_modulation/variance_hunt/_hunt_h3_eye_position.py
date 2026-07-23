"""Hunt H3: eye-POSITION gain field on the extraretinal gap.

The behavior-zeroing ablation removes the model's eye-POSITION input (not just
velocity). If the twin uses a gaze-position gain field, the gap y = full-ablated
should carry a slow/static modulation as a function of absolute gaze (x,y) in deg
(within the 0.5 deg fixation window; NaN outside). This is DISTINCT from the
drift-SPEED effect found by subagent #1 (which is directionally isotropic, scalar).

Tests interpretable low-order 2D position fields augmenting the "both"
saccade-kernel baseline (base median recovered = 0.4028):
  - additive polynomial in (x,y): linear / quadratic / cubic  (+ pos-valid guard)
  - multiplicative gain field: drive * poly(x,y)  (classic gain-field form)
  - radial: r=hypot(x,y), r^2, additive and drive*[r,r^2]
  - per-trial-mean gaze (slow signal) vs instantaneous
  - stacked ON TOP of #1's best drift-speed model -> complementary or redundant?

Run one focused round at a time:
  uv run python paper/supp_twin_saccade_modulation/_hunt_h3_eye_position.py <round>
where <round> in {diag, 1, 2, 3}.
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "variance_hunt"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _hunt_h1_drift_velocity import _speed, _eyepos, make_speed_mult  # noqa: E402


# ---------------------------------------------------------------------------
# position helpers (aligned to rec["tr"], rec["b"])
# ---------------------------------------------------------------------------
def _xy(rec, designs):
    """Instantaneous gaze (x, y) deg; NaN outside 0.5 deg gate."""
    return _eyepos(rec, designs)  # (S, 2)


def _xy_trialmean(rec, designs):
    """Per-trial mean gaze (slow signal): mean over finite bins of the trial,
    broadcast to every sample of that trial."""
    s = rec["session"]
    ep = designs[s]["eyepos"]  # (T, B, 2)
    with np.errstate(invalid="ignore"):
        tm = np.nanmean(ep, axis=1)  # (T, 2) mean gaze per trial
    return tm[rec["tr"], :]  # (S, 2)


# ---------------------------------------------------------------------------
# column builders -- ADDITIVE position fields
# ---------------------------------------------------------------------------
def pos_valid(rec, designs):
    """1 where gaze is inside the gate (finite), else 0. Guards the zero-fill of
    NaN gaze bins. Diagnostics show gaze is finite for ~100% of valid samples, so
    this is all-ones for essentially every unit -> return None (skip) to avoid a
    redundant intercept-duplicate column diluting the fit."""
    xy = _xy(rec, designs)
    ok = np.isfinite(xy[:, 0])
    if ok.all():
        return None
    return ok.astype(np.float64)


def pos_linear(rec, designs):
    xy = _xy(rec, designs)
    return xy  # [x, y]; harness zero-fills NaN


def pos_quad(rec, designs):
    xy = _xy(rec, designs)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([x, y, x * x, y * y, x * y])


def pos_cubic(rec, designs):
    xy = _xy(rec, designs)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([
        x, y, x * x, y * y, x * y,
        x * x * x, y * y * y, x * x * y, x * y * y,
    ])


def pos_radial(rec, designs):
    xy = _xy(rec, designs)
    r = np.hypot(xy[:, 0], xy[:, 1])
    return np.column_stack([r, r * r])


def posmean_linear(rec, designs):
    return _xy_trialmean(rec, designs)  # [xbar, ybar]


def posmean_quad(rec, designs):
    xy = _xy_trialmean(rec, designs)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([x, y, x * x, y * y, x * y])


# ---------------------------------------------------------------------------
# MULTIPLICATIVE gain fields: drive * f(position)
# ---------------------------------------------------------------------------
def make_pos_mult(fn):
    return make_speed_mult(fn)  # same wrapper: drive[:,None] * f, NaN->0


# ---------------------------------------------------------------------------
# drift-speed reference model (subagent #1's robust combined winner)
# ---------------------------------------------------------------------------
def speed_valid(rec, designs):
    sp = _speed(rec, designs)
    return np.isfinite(sp).astype(np.float64)


def logspeed_quad(rec, designs):
    sp = _speed(rec, designs)
    x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
    return np.column_stack([x, x * x])


SPEED_MODEL = [speed_valid, logspeed_quad, make_speed_mult(logspeed_quad)]


# ---------------------------------------------------------------------------
def diagnostics(ctx):
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]
    fx, fy, fr, frac_valid = [], [], [], []
    for rec in recs:
        xy = _xy(rec, designs)
        ok = np.isfinite(xy[:, 0])
        frac_valid.append(ok.mean())
        fx.append(xy[ok, 0]); fy.append(xy[ok, 1])
        fr.append(np.hypot(xy[ok, 0], xy[ok, 1]))
    fx = np.concatenate(fx); fy = np.concatenate(fy); fr = np.concatenate(fr)
    print("=== eye-position diagnostics (reliable units) ===")
    print(f"samples with finite gaze: mean frac/unit = {np.mean(frac_valid):.3f}")
    qs = [1, 5, 25, 50, 75, 95, 99]
    for name, arr in (("x", fx), ("y", fy), ("r", fr)):
        pcts = np.percentile(arr, qs)
        print(f"  {name}: " + " ".join(f"p{q}={p:+.3f}" for q, p in zip(qs, pcts)))
    print(f"  |x|max={np.abs(fx).max():.3f}  |y|max={np.abs(fy).max():.3f}  rmax={fr.max():.3f}")


if __name__ == "__main__":
    round_ = sys.argv[1] if len(sys.argv) > 1 else "diag"
    ctx = load_augment_context()

    if round_ == "diag":
        diagnostics(ctx)

    elif round_ == "1":
        print("\n=== R1: additive position fields (alone, on saccade base) ===")
        evaluate_augmentation([pos_valid, pos_linear], "add: pos linear [x,y]", ctx)
        evaluate_augmentation([pos_valid, pos_quad], "add: pos quad [x,y,x2,y2,xy]", ctx)
        evaluate_augmentation([pos_valid, pos_cubic], "add: pos cubic", ctx)
        evaluate_augmentation([pos_valid, pos_radial], "add: radial [r,r2]", ctx)
        evaluate_augmentation([pos_valid, posmean_linear], "add: trial-mean gaze linear", ctx)
        evaluate_augmentation([pos_valid, posmean_quad], "add: trial-mean gaze quad", ctx)

    elif round_ == "2":
        print("\n=== R2: multiplicative gain fields drive*poly(pos) ===")
        evaluate_augmentation([pos_valid, make_pos_mult(pos_linear)],
                              "mult: drive*[x,y]", ctx)
        evaluate_augmentation([pos_valid, make_pos_mult(pos_quad)],
                              "mult: drive*quad(x,y)", ctx)
        evaluate_augmentation([pos_valid, make_pos_mult(pos_radial)],
                              "mult: drive*[r,r2]", ctx)
        print("\n=== R2b: additive + multiplicative combos ===")
        evaluate_augmentation([pos_valid, pos_quad, make_pos_mult(pos_quad)],
                              "add quad + mult quad", ctx)
        evaluate_augmentation([pos_valid, pos_radial, make_pos_mult(pos_radial)],
                              "add radial + mult radial", ctx)
        evaluate_augmentation([pos_valid, pos_linear, make_pos_mult(pos_linear)],
                              "add linear + mult linear", ctx)

    elif round_ == "3":
        print("\n=== R3: does position ADD beyond drift-speed? ===")
        # speed reference (reconfirm)
        evaluate_augmentation(SPEED_MODEL, "REF: speed model (add+mult logspeed-quad)", ctx)
        # stack best position variants on top of speed
        evaluate_augmentation(SPEED_MODEL + [pos_valid, pos_linear],
                              "speed + add pos linear", ctx)
        evaluate_augmentation(SPEED_MODEL + [pos_valid, pos_quad],
                              "speed + add pos quad", ctx)
        evaluate_augmentation(SPEED_MODEL + [pos_valid, make_pos_mult(pos_quad)],
                              "speed + mult drive*quad(pos)", ctx)
        evaluate_augmentation(SPEED_MODEL + [pos_valid, pos_radial],
                              "speed + add radial", ctx)
        evaluate_augmentation(SPEED_MODEL + [pos_valid, posmean_quad],
                              "speed + trial-mean gaze quad", ctx)
