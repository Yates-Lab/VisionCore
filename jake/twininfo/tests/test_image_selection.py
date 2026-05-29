from __future__ import annotations

import numpy as np

from jake.twininfo.image_selection import _eligible_energy, _trace_padding_margins_px


def test_trace_padding_margins_follow_model_crop_axis_convention():
    trace = np.array([
        [0.10, -0.20],
        [-0.05, 0.30],
    ], dtype=np.float32)

    margin_x, margin_y = _trace_padding_margins_px([trace], ppd=10.0)

    assert np.isclose(margin_x, 3.0)
    assert np.isclose(margin_y, 1.0)


def test_eligible_energy_respects_eye_trace_swept_crop_margin():
    energy = np.ones((220, 240), dtype=np.float32)

    baseline = _eligible_energy(energy)
    swept = _eligible_energy(energy, trace_margin_x_px=20.0, trace_margin_y_px=10.0)

    assert np.isfinite(baseline[76, 76])
    assert np.isfinite(baseline[-77, -77])
    assert not np.isfinite(swept[76, 76])
    assert not np.isfinite(swept[-77, -77])
    assert np.isfinite(swept[90, 100])
