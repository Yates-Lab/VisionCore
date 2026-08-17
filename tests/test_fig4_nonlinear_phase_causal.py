"""Tests for the Figure 4 nonlinear phase counterfactuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from paper.fig4.nonlinear_phase_causal.common import (
    matched_switch_shuffled_route,
    splitrelu_from_route_magnitude,
    splitrelu_from_route_mask,
    unit_rate_metrics,
)
from paper.fig4.nonlinear_phase_causal.run_experiment import stabilized_tangent_preactivation
from paper.fig4.nonlinear_phase_causal.measure_center_rf import (
    lag_before_output_ms,
    representative_unit,
    rf_geometry,
)
from paper.fig4.nonlinear_phase_causal.measure_rr100_rf_atlas import (
    orientation_metrics,
    select_context_examples,
    temporal_separability,
)
from paper.fig4.nonlinear_phase_causal.analyze_rr100_rf_dimensionality import (
    crossed_variance_decomposition,
    dominant_spatial_modes,
    leave_one_image_out_same_unit_span,
    normalize_spatial_maps,
    normalize_spatiotemporal_volumes,
)
from paper.fig4.nonlinear_phase_causal.run_one_x_tangent import parse_scales as parse_one_x_scales
from paper.fig4.nonlinear_phase_causal.analyze_and_plot import (
    bootstrap_contrast,
    group_contributions,
)
from paper.fig4.nonlinear_phase_causal.response_subspace import (
    INPUT_SIZE,
    N_DCT_FEATURES,
    N_LAGS,
    PATCH_SIZE,
    LowRankLQ,
    dct_filters_to_movies,
    deterministic_unit_subset,
    extract_local_patches,
    orthonormal_dct,
    patches_to_dct,
    response_grid_offsets,
)
from paper.fig4.nonlinear_phase_causal.reduced_twin import (
    PopulationNonlinearity,
    ReducedTwin,
    ReducedTwinState,
)


def test_route_magnitude_reconstruction_is_exact_splitrelu() -> None:
    generator = torch.Generator().manual_seed(11)
    signed = torch.randn((3, 5, 2, 7, 9), generator=generator)
    gain = torch.tensor(1.37)
    expected = torch.cat((F.relu(signed) * gain.abs(), F.relu(-signed)), dim=1)
    observed = splitrelu_from_route_magnitude(signed, signed, positive_gain=gain)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_factorial_counterfactuals_change_only_requested_factor() -> None:
    stable = torch.tensor([[[[-2.0, 3.0], [4.0, -5.0]]]])
    moving = torch.tensor([[[[7.0, -11.0], [13.0, -17.0]]]])
    magnitude_only = splitrelu_from_route_magnitude(stable, moving)
    route_only = splitrelu_from_route_magnitude(moving, stable)
    # Summing the two branches recovers the requested magnitude exactly.
    torch.testing.assert_close(magnitude_only.sum(dim=1), moving.abs().squeeze(1))
    torch.testing.assert_close(route_only.sum(dim=1), stable.abs().squeeze(1))
    # Nonzero positive-branch locations recover the requested route.
    assert torch.equal(magnitude_only[:, :1] > 0, stable > 0)
    assert torch.equal(route_only[:, :1] > 0, moving > 0)


def test_shuffled_route_preserves_switch_count_per_sample_channel() -> None:
    generator = torch.Generator().manual_seed(4)
    stable = torch.randn((2, 3, 2, 5, 7), generator=generator)
    moving = torch.randn((2, 3, 2, 5, 7), generator=generator)
    shuffled_route = matched_switch_shuffled_route(stable, moving, seed=19)
    stable_route = stable > 0
    true_count = torch.logical_xor(stable_route, moving > 0).flatten(start_dim=2).sum(dim=2)
    shuffled_count = torch.logical_xor(stable_route, shuffled_route).flatten(start_dim=2).sum(dim=2)
    torch.testing.assert_close(shuffled_count, true_count)
    # A fixed seed must make the randomized control exactly reproducible.
    repeated = matched_switch_shuffled_route(stable, moving, seed=19)
    assert torch.equal(shuffled_route, repeated)


def test_route_mask_and_route_source_forms_agree() -> None:
    route = torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]]]])
    magnitude = torch.tensor([[[[5.0, -6.0], [-7.0, 8.0]]]])
    source_form = splitrelu_from_route_magnitude(route, magnitude)
    mask_form = splitrelu_from_route_mask(route > 0, magnitude)
    torch.testing.assert_close(source_form, mask_form, atol=0, rtol=0)


def test_unit_rate_metrics_uniform_map_has_zero_ssi_and_cv2() -> None:
    metrics = unit_rate_metrics(torch.full((2, 3, 9, 9), 4.0))
    torch.testing.assert_close(metrics["ssi"], torch.zeros((2, 3), dtype=torch.float64))
    torch.testing.assert_close(metrics["cv2"], torch.zeros((2, 3), dtype=torch.float64))


def test_stabilized_tangent_is_exact_for_affine_map() -> None:
    stable = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    moving = torch.tensor([[4.0, 1.0], [-1.0, 2.0]])
    weight = torch.tensor([[2.0, -3.0], [0.25, 4.0]])
    bias = torch.tensor([0.7, -1.2])
    function = lambda value: value @ weight.T + bias
    anchor, tangent = stabilized_tangent_preactivation(function, stable, moving)
    torch.testing.assert_close(anchor, function(stable))
    torch.testing.assert_close(tangent, function(moving))


def test_group_contributions_recover_expected_spike_weighted_value() -> None:
    metric = torch.tensor([[[1.0, 3.0, 9.0], [2.0, 4.0, 8.0]]]).numpy()
    expected = torch.tensor([[[2.0, 1.0, 7.0], [1.0, 3.0, 5.0]]]).numpy()
    numerator, denominator = group_contributions(metric, expected, torch.tensor([0, 1]).numpy())
    torch.testing.assert_close(torch.from_numpy(numerator), torch.tensor([[5.0, 14.0]]))
    torch.testing.assert_close(torch.from_numpy(denominator), torch.tensor([[3.0, 4.0]]))


def test_crossed_bootstrap_constant_contrast_is_exact_and_reproducible() -> None:
    denominator = torch.ones((3, 4), dtype=torch.float64).numpy()
    moving = (2.5 * torch.ones((3, 4), dtype=torch.float64)).numpy()
    stable = (1.0 * torch.ones((3, 4), dtype=torch.float64)).numpy()
    terms = [(1.0, moving, denominator), (-1.0, stable, denominator)]
    first = bootstrap_contrast(
        terms,
        draws=200,
        seed=17,
        percent_of_baseline=(stable, denominator),
    )
    second = bootstrap_contrast(
        terms,
        draws=200,
        seed=17,
        percent_of_baseline=(stable, denominator),
    )
    assert first == second == (150.0, 150.0, 150.0)


def test_center_rf_geometry_recovers_centered_point_mass() -> None:
    energy = np.zeros((11, 11), dtype=float)
    energy[5, 5] = 1.0
    result = rf_geometry(energy, ppd=10.0)
    assert result["centroid_offset_deg"] == 0.0
    assert result["radius_50_deg"] == 0.0
    assert result["radius_90_deg"] == 0.0


def test_rf_lag_axis_matches_production_current_first_convention() -> None:
    np.testing.assert_allclose(lag_before_output_ms(4), [0.0, 1000.0 / 120.0, 2000.0 / 120.0, 25.0])


def test_representative_rf_unit_is_well_fit_and_near_group_median() -> None:
    table = pd.DataFrame(
        {
            "unit_index": [3, 5, 7],
            "sf_split_metric": [0.1, 0.2, 0.3],
            "dynamic_log_gaussian_marginal_r2": [0.9, 0.9, 0.9],
            "dynamic_peak_response_amp": [0.2, 0.2, 0.2],
        }
    )
    assert representative_unit(table, np.asarray([3, 5, 7])) == 5


def test_rr100_atlas_orientation_metric_detects_one_axis() -> None:
    power = np.zeros((21, 21), dtype=float)
    power[10, 5] = 1.0
    power[10, 15] = 1.0
    selectivity, orientation = orientation_metrics(power, ppd=10.0)
    assert selectivity > 0.99
    assert min(abs(orientation), abs(orientation - 180.0)) < 1e-8


def test_rr100_atlas_temporal_separability_is_one_for_rank_one_rf() -> None:
    temporal = np.asarray([1.0, -2.0, 0.5])
    spatial = np.arange(20, dtype=float).reshape(4, 5)
    rf = temporal[:, None, None] * spatial[None]
    assert np.isclose(temporal_separability(rf), 1.0)


def test_rr100_context_examples_span_both_groups() -> None:
    table = pd.DataFrame(
        {
            "unit_index": np.arange(12),
            "rf_group": ["lower SF"] * 6 + ["higher SF"] * 6,
            "local_rf_spectral_peak_cpd": [1, 2, 3, 4, 5, 6] * 2,
        }
    )
    selected = select_context_examples(table, n_per_group=3)
    assert selected.tolist() == [0, 2, 5, 6, 8, 11]


def test_one_x_tangent_target_scales_require_anchor() -> None:
    np.testing.assert_array_equal(parse_one_x_scales("0,1"), np.asarray([0.0, 1.0], dtype=np.float32))
    try:
        parse_one_x_scales("0,0.5")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a missing-anchor ValueError")


def test_rf_dimensionality_dominant_mode_recovers_separable_filter() -> None:
    temporal = np.asarray([1.0, -2.0, 0.5])
    spatial = np.arange(20, dtype=float).reshape(4, 5) - 9.5
    crops = (temporal[:, None, None] * spatial[None])[None, None]
    mode, fraction = dominant_spatial_modes(crops)
    cosine = abs(float(np.vdot(mode[0, 0], spatial))) / (
        np.linalg.norm(mode[0, 0]) * np.linalg.norm(spatial)
    )
    assert np.isclose(cosine, 1.0)
    assert np.isclose(fraction[0, 0], 1.0)


def test_rf_dimensionality_variance_decomposition_detects_additive_data() -> None:
    image = np.asarray([0.0, 2.0, -1.0])[:, None, None]
    unit = np.asarray([-3.0, 1.0, 4.0, 2.0])[None, :, None]
    decomposition = crossed_variance_decomposition(image + unit)
    assert np.isclose(decomposition["image_by_unit_fraction"], 0.0, atol=1e-14)
    assert np.isclose(sum(decomposition.values()), 1.0)


def test_rf_dimensionality_normalization_removes_dc_and_equalizes_energy() -> None:
    value = np.arange(24, dtype=float).reshape(2, 3, 4) + np.asarray([0.0, 100.0])[:, None, None]
    normalized = normalize_spatial_maps(value)
    np.testing.assert_allclose(normalized.mean(axis=(-2, -1)), np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(normalized.reshape(2, -1), axis=1), np.ones(2))


def test_rf_dimensionality_same_unit_span_recovers_fixed_filter() -> None:
    spatial = np.arange(12, dtype=float).reshape(3, 4) - 5.5
    maps = np.stack([spatial, 2 * spatial, -3 * spatial, 0.5 * spatial])[:, None]
    maps = normalize_spatial_maps(maps)
    result = leave_one_image_out_same_unit_span(
        maps, np.asarray([10, 11, 12, 13]), ranks=(1, 3)
    )
    np.testing.assert_allclose(result.recovered_energy_fraction, 1.0, atol=1e-14)


def test_rf_dimensionality_spatiotemporal_normalization_removes_per_lag_dc() -> None:
    value = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(1, 2, 3, 4, 5)
    normalized = normalize_spatiotemporal_volumes(value)
    np.testing.assert_allclose(normalized.mean(axis=(-2, -1)), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(normalized.reshape(1, 2, -1), axis=-1), 1.0, atol=1e-14
    )


def test_response_subspace_dct_is_orthonormal_and_invertible() -> None:
    basis = orthonormal_dct(PATCH_SIZE)
    assert torch.allclose(basis @ basis.T, torch.eye(PATCH_SIZE), atol=1e-5)
    generator = torch.Generator().manual_seed(19)
    coefficients = torch.randn(2, 2, N_DCT_FEATURES, generator=generator)
    reconstructed = dct_filters_to_movies(coefficients)
    observed = patches_to_dct(reconstructed)
    assert torch.allclose(observed, coefficients, atol=2e-5)


def test_response_subspace_selection_is_balanced_and_deterministic() -> None:
    table = pd.DataFrame(
        {
            "unit_index": np.arange(20),
            "sf_split_metric": np.r_[np.linspace(0.0, 0.4, 12), np.linspace(0.5, 1.0, 8)],
            "prior_orientation_selectivity_index": np.linspace(1.0, 0.0, 20),
            "dynamic_peak_response_amp": np.tile(np.arange(10, dtype=float), 2),
        }
    )
    first = deterministic_unit_subset(table, n_per_group=4)
    second = deterministic_unit_subset(table, n_per_group=4)
    pd.testing.assert_frame_equal(first, second)
    assert first.response_subspace_group.value_counts().to_dict() == {"lower-SF": 4, "higher-SF": 4}


def test_response_subspace_retraction_preserves_predictions() -> None:
    torch.manual_seed(3)
    model = LowRankLQ(n_units=3, rank=2, n_features=11)
    with torch.no_grad():
        model.weights.mul_(torch.tensor([1.7, 0.4])[None, :, None])
        model.linear.normal_()
        model.quadratic.normal_()
    features = torch.randn(17, 11)
    before = model(features).detach()
    model.retract()
    after = model(features).detach()
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    gram = model.weights @ model.weights.transpose(1, 2)
    assert torch.allclose(gram, torch.eye(2)[None].expand(3, -1, -1), atol=1e-5)


def test_response_subspace_grid_contains_center_and_has_expected_count() -> None:
    offsets = response_grid_offsets()
    assert offsets.shape == (25, 2)
    assert np.any(np.all(offsets == 0, axis=1))


def test_reduced_twin_population_nonlinearity_is_positive_and_shaped() -> None:
    model = PopulationNonlinearity(7, 3, hidden_dims=(5,), dropout=0.0)
    observed = model(torch.randn(11, 7))
    assert observed.shape == (11, 3)
    assert torch.all(observed > 0)


def test_reduced_twin_decodes_generator_maps_with_expected_shape() -> None:
    n_features = N_DCT_FEATURES
    state = ReducedTwinState(
        rank=2,
        unit_indices=torch.tensor([4, 9, 11]),
        basis=torch.randn(2, n_features),
        feature_mean=torch.zeros(n_features),
        feature_std=torch.ones(n_features),
        generator_mean=torch.zeros(2),
        generator_std=torch.ones(2),
        hidden_dims=(4,),
        dropout=0.0,
        decoder_state=PopulationNonlinearity(2, 3, hidden_dims=(4,), dropout=0.0).state_dict(),
    )
    model = ReducedTwin(state)
    value = model.decode_generators(torch.randn(5, 7, 2))
    assert value.shape == (5, 7, 3)
    assert torch.all(value > 0)
