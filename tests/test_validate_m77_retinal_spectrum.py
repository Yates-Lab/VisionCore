from __future__ import annotations

import numpy as np
from scipy import sparse

from paper.fig4.spatiotemporal_tuning.validate_m77_retinal_spectrum import (
    cube_from_modes,
)


def test_cube_from_modes_preserves_distributed_power() -> None:
    power = np.arange(1, 13, dtype=float).reshape(3, 4)
    distributor = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.75],
            ]
        )
    )
    cube = cube_from_modes(power, distributor, n_spatial=2, n_orientation=2)
    assert cube.shape == (2, 4, 2)
    np.testing.assert_allclose(cube.sum(), power.sum())
