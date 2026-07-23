"""Hunt H2: drift *direction* modulation of the extraretinal gap.

Refines subagent #1's drift-speed line. Question: does drift VELOCITY DIRECTION
add anything beyond scalar drift speed? And what is the single best interpretable
add+mult (speed [+ direction]) model?

Reuses helpers from _hunt_h1_drift_velocity. Builds ctx ONCE.
Direction is in the FIXED screen frame (vx, vy = central-diff eye velocity).

Run: uv run python paper/supp_twin_saccade_modulation/_hunt_h2_drift_direction.py
"""
from __future__ import annotations

import sys
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "covariate_hunt"))

from _supp_saccade_augment import evaluate_augmentation, load_augment_context  # noqa: E402
from _hunt_h1_drift_velocity import (  # noqa: E402
    _speed, _velocity, valid_indicator, make_speed_mult,
)


# ---------------------------------------------------------------------------
# speed reference model (reproduces #1's best combined add+mult speed)
# ---------------------------------------------------------------------------
def logspeed_quad(rec, designs):
    sp = _speed(rec, designs)
    x = np.log1p(np.where(np.isfinite(sp), sp, 0.0))
    return np.column_stack([x, x * x])


# ---------------------------------------------------------------------------
# direction column builders (fixed screen frame)
# ---------------------------------------------------------------------------
def velocity_vec(rec, designs):
    """[vx, vy] fixed-frame velocity = speed*[cos, sin]. Linear directional term."""
    v = _velocity(rec, designs)          # (S,2), NaN at edges -> harness zero-fills
    return v


def unit_dir(rec, designs):
    """[cos, sin] pure drift direction, magnitude-normalized (speed-independent)."""
    v = _velocity(rec, designs)
    sp = np.hypot(v[:, 0], v[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(sp > 0, v[:, 0] / sp, np.nan)
        s = np.where(sp > 0, v[:, 1] / sp, np.nan)
    return np.column_stack([c, s])


if __name__ == "__main__":
    ctx = load_augment_context()

    print("\n=== H2 sanity: base reproduces 0.4028 ===")
    evaluate_augmentation([], "baseline-check (no extra cols)", ctx)

    print("\n=== H2a: direction ALONE (augmenting saccade-kernel base) ===")
    evaluate_augmentation([valid_indicator, velocity_vec],
                          "add: velocity [vx,vy] (fixed frame)", ctx)
    evaluate_augmentation([valid_indicator, unit_dir],
                          "add: unit dir [cos,sin] (speed-independent)", ctx)
    evaluate_augmentation([valid_indicator, make_speed_mult(velocity_vec)],
                          "mult: drive*[vx,vy]", ctx)
    evaluate_augmentation([valid_indicator, make_speed_mult(unit_dir)],
                          "mult: drive*[cos,sin]", ctx)

    print("\n=== H2b: does direction add BEYOND scalar speed? ===")
    speed_model = [valid_indicator, logspeed_quad, make_speed_mult(logspeed_quad)]
    evaluate_augmentation(speed_model,
                          "SPEED ref: add logspeed-quad + mult logspeed-quad", ctx)
    evaluate_augmentation(speed_model + [velocity_vec],
                          "speed + add velocity [vx,vy]", ctx)
    evaluate_augmentation(speed_model + [velocity_vec, make_speed_mult(velocity_vec)],
                          "speed + add velocity + mult drive*velocity", ctx)
