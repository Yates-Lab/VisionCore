from __future__ import annotations

import numpy as np

from jake.twininfo.information import image_shift_grid
from jake.twininfo.stimuli import amplitude_spectrum_relative_error, phase_scramble_image, shift_image


def test_phase_scramble_preserves_mean_std_and_amplitude():
    rng = np.random.default_rng(1)
    image = rng.normal(loc=127.0, scale=30.0, size=(64, 64)).astype(np.float32)
    scrambled = phase_scramble_image(image, rng)
    assert scrambled.shape == image.shape
    assert abs(float(scrambled.mean() - image.mean())) < 1e-5
    assert abs(float(scrambled.std() - image.std())) < 1e-5
    assert amplitude_spectrum_relative_error(image, scrambled) < 1e-6
    assert not np.allclose(image, scrambled)


def test_shift_image_no_wraparound_by_default():
    image = np.zeros((16, 16), dtype=np.float32)
    image[8, 14] = 1.0
    shifted = shift_image(image, dx_deg=4.0, dy_deg=0.0, ppd=1.0, mode="nearest", order=0)
    assert shifted[:, 0].max() == 0.0
    assert shifted[:, -1].max() == 0.0


def test_shift_grid_has_expected_number_of_states():
    grid = image_shift_grid(max_shift_deg=3.0 / 60.0, step_deg=1.0 / 60.0)
    assert grid.shape == (49, 2)
