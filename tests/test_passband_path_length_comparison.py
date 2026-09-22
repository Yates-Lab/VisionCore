from __future__ import annotations

import numpy as np

from paper.fig4.spatiotemporal_tuning.compare_passband_path_length import (
    paired_unit_bootstrap,
)


def test_paired_unit_bootstrap_preserves_pairwise_difference() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    center, low, high = paired_unit_bootstrap(
        values, n_bootstrap=1000, seed=3
    )
    assert center == np.median(values)
    assert low > 0
    assert low <= center <= high
