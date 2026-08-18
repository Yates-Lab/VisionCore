from __future__ import annotations

import numpy as np

from paper.fig4.spatiotemporal_tuning.run_m77_retinal_power_visualization import (
    spatial_frequency_grid,
)


def test_auxiliary_frequency_grid_excludes_dc_and_caps_radius() -> None:
    kxy, flat = spatial_frequency_grid(255, maximum_cpd=12.0)
    radius = np.linalg.norm(kxy, axis=1)
    assert len(kxy) == len(flat)
    assert np.all(radius > 0)
    assert np.max(radius) <= 12.0 * 1.15 + 1e-9
    assert len(np.unique(flat)) == len(flat)
