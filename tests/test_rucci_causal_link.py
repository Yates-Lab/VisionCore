from pathlib import Path

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning.analyze_rucci_causal_link import (
    spearman_with_resampling,
    unit_causal_effects,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    log_bin_indices,
    log_edges,
    nearest_axis_orientation,
)


def test_cycle_valid_frequency_binning_has_half_octave_boundaries():
    centers = np.asarray([1.0, 2.0, 4.0])
    edges = log_edges(centers)
    assert np.allclose(edges, np.asarray([-0.5, 0.5, 1.5, 2.5]))

    radial = 2.0 ** np.asarray([-0.6, -0.5, 0.49, 0.5, 1.5, 2.49, 2.5])
    index, keep = log_bin_indices(radial, centers)
    assert index.tolist() == [-1, 0, 0, 1, 2, 2, 3]
    assert keep.tolist() == [False, True, True, True, True, True, False]


def test_fourier_normal_maps_to_bar_axis_orientation():
    kxy = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    orientations = np.asarray([0.0, 45.0, 90.0, 135.0])
    # Horizontal Fourier normal -> vertical bar; vertical normal -> horizontal bar.
    assert nearest_axis_orientation(kxy, orientations).tolist() == [2, 0, 1]


def test_unit_causal_effects_uses_expected_spike_weighting_and_unit_identity(tmp_path: Path):
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    pd.DataFrame({"unit_index": [0, 1]}).to_csv(matrix / "unit_feature_table.csv", index=False)
    pd.DataFrame({"image_index": [0, 1]}).to_csv(matrix / "image_feature_table.csv", index=False)
    pd.DataFrame({"trace_index": [0, 1]}).to_csv(matrix / "trace_feature_table.csv", index=False)

    moving_ssi = np.asarray(
        [
            [[2.0, 1.0], [4.0, 2.0]],
            [[6.0, 3.0], [8.0, 4.0]],
        ]
    )
    moving_expected = np.ones_like(moving_ssi)
    stable_ssi = np.asarray([[2.0, 1.0], [4.0, 2.0]])
    stable_expected = np.ones_like(stable_ssi)
    np.save(matrix / "ssi_matrix.npy", moving_ssi.reshape(-1, 2))
    np.save(matrix / "expected_spikes_matrix.npy", moving_expected.reshape(-1, 2))
    np.save(matrix / "stabilized_ssi_by_image.npy", stable_ssi)
    np.save(matrix / "stabilized_expected_spikes_by_image.npy", stable_expected)

    overlap = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "unit_index": [1, 0],
            "motion_scale": [1.0, 1.0],
            "observed_passband_overlap": [20.0, 10.0],
            "weighted_center_sf_cpd": [4.0, 1.0],
            "sf_group": ["high_sf", "low_sf"],
        }
    ).to_csv(overlap, index=False)

    table = unit_causal_effects(matrix, overlap).set_index("unit_index")
    assert table.loc[0, "moving_ssi_bits_per_spike"] == 5.0
    assert table.loc[0, "stabilized_ssi_bits_per_spike"] == 3.0
    assert table.loc[1, "moving_ssi_bits_per_spike"] == 2.5
    assert table.loc[1, "stabilized_ssi_bits_per_spike"] == 1.5
    assert table.loc[1, "passband_overlap_relative_to_population_median"] > 1.0


def test_spearman_resampling_recovers_monotone_relation():
    result = spearman_with_resampling(
        np.arange(10.0),
        np.arange(10.0) ** 2,
        n_resamples=200,
        seed=4,
    )
    assert np.isclose(result["spearman_rho"], 1.0)
    assert result["bootstrap_ci95"][0] > 0.9
    assert result["two_sided_permutation_p"] < 0.05
