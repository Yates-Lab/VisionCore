from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paper.model_selection.render_twin_m77_subspace_supplement import (
    cumulative_energy,
    fidelity_summary,
    paired_fidelity,
    rank_at_fraction,
    validate_matching_units,
)


def test_matching_units_requires_identical_order() -> None:
    twin = pd.DataFrame({"unit_index": [4, 9, 15]})
    m77 = pd.DataFrame({"unit_index": [4, 9, 15]})
    np.testing.assert_array_equal(
        validate_matching_units(twin, m77), np.asarray([4, 9, 15])
    )
    with pytest.raises(ValueError, match="not identical"):
        validate_matching_units(twin, m77.iloc[[0, 2, 1]].reset_index(drop=True))


def test_cumulative_energy_and_threshold_rank() -> None:
    energy = cumulative_energy(np.asarray([[3.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(energy, [[0.9, 1.0], [0.5, 1.0]])
    np.testing.assert_array_equal(rank_at_fraction(energy, 0.8), [1, 2])


def test_paired_fidelity_and_bootstrap_summary(tmp_path) -> None:
    rows = []
    for rank in (1, 2):
        for unit, value in zip((4, 9, 15), (0.1, 0.2, 0.3)):
            rows.append(
                {"unit_index": unit, "rank": rank, "test_rate_r2": value * rank}
            )
    twin = pd.DataFrame(rows)
    m77 = twin.copy()
    m77["test_rate_r2"] += 0.05
    twin_path = tmp_path / "twin.csv"
    m77_path = tmp_path / "m77.csv"
    twin.to_csv(twin_path, index=False)
    m77.to_csv(m77_path, index=False)

    combined, ranks = paired_fidelity(twin_path, m77_path)
    assert ranks == [1, 2]
    summary = fidelity_summary(combined, ranks, seed=7, bootstrap=200)
    assert set(summary.model) == {"Twin", "M77"}
    assert summary.n_units.eq(3).all()
    rank1 = summary.loc[summary["rank"].eq(1)].set_index("model")
    assert rank1.loc["M77", "median_test_rate_r2"] == pytest.approx(0.25)
    assert rank1.loc["Twin", "median_test_rate_r2"] == pytest.approx(0.20)

    shuffled = m77.iloc[[1, 0, 2, 3, 4, 5]].copy()
    shuffled.to_csv(m77_path, index=False)
    with pytest.raises(ValueError, match="not paired"):
        paired_fidelity(twin_path, m77_path)
