import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIG3_DIR = Path(__file__).resolve().parents[1] / "paper" / "fig3"
if str(FIG3_DIR) not in sys.path:
    sys.path.insert(0, str(FIG3_DIR))

from paper.fig3._fig3_ablation_data import (
    _center_crop_spatial,
    _compute_ccnorm_by_condition,
    _renormalize_ccnorm_to_intact_anchor,
)
from paper.fig3._fig3_femfraction import (
    _aggregate_ablation_femfraction,
    _base_valid_mask,
)
from paper.fig3._fig3_data import (
    align_native_trial_arrays_to_reference,
    analysis_endpoint_block_filter,
    analysis_endpoint_block_mean,
    analysis_endpoint_block_sum,
    analysis_endpoint_mask_and_psth,
    analysis_model_indices,
    analysis_reduce_model_output,
    figure3_analysis_grid,
)
from paper.fig3._fig3a_data import _crop_model_aperture
from paper.fig3.generate_figure3 import _build_caption
from paper.model_selection.render_m16_figure3_reproduction import aligned_table
from paper.model_selection.run_m16_response_subspace_pilot import (
    N_DCT_FEATURES,
    canonical_rr100_rows,
    dct_filters_to_movies,
    movies_to_dct,
    quadratic_design_columns,
    select_ridge_training_only,
)


def test_masked_dct_round_trip_preserves_coefficients():
    movie = torch.randn(2, 1, 60, 35, 35)
    coefficients = movies_to_dct(movie)
    reconstructed = dct_filters_to_movies(coefficients)[:, None]
    observed = movies_to_dct(reconstructed)
    assert coefficients.shape == (2, N_DCT_FEATURES)
    torch.testing.assert_close(observed, coefficients, atol=4e-5, rtol=1e-4)


def test_canonical_rr100_rows_reproduce_historical_position_order():
    model = SimpleNamespace(names=["session_b", "session_a", "unused"])
    outputs = [
        {
            "sess": "session_a",
            "cids": np.asarray([10, 11, 12]),
            "ccnorm": {"ccnorm": np.asarray([0.7, 0.2, 0.8])},
        },
        {
            "sess": "session_b",
            "cids": np.asarray([20, 21]),
            "ccnorm": {"ccnorm": np.asarray([0.1, 0.9])},
        },
    ]
    rows = canonical_rr100_rows(model, outputs)
    assert [(row["session"], row["source_cid"]) for row in rows] == [
        ("session_b", 21),
        ("session_a", 10),
        ("session_a", 12),
    ]


def test_canonical_rr100_rows_use_cids_used_for_score_row_identity():
    model = SimpleNamespace(names=["session_a"])
    outputs = [{
        "sess": "session_a",
        "cids": np.asarray([10, 11, 12, 13]),
        "cids_used": np.asarray([10, 12, 13]),
        "ccnorm": {"ccnorm": np.asarray([0.7, 0.2, 0.8])},
    }]
    rows = canonical_rr100_rows(model, outputs)
    assert [row["source_cid"] for row in rows] == [10, 13]


def test_quadratic_subspace_parameter_count_includes_cross_terms():
    assert quadratic_design_columns(1) == 3
    assert quadratic_design_columns(8) == 45
    assert quadratic_design_columns(16) == 153


def test_subspace_ridge_selection_uses_training_data_only():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(100, 3, generator=generator)
    design = torch.column_stack((torch.ones(100), x))
    target = 2.0 + x[:, 0] - 0.5 * x[:, 1] + 0.01 * torch.randn(100, generator=generator)
    ridge, score = select_ridge_training_only(design, target, [1e-5, 1e-1, 10.0])
    assert ridge in {1e-5, 1e-1}
    assert score > 0.95


def test_figure3_table_aligns_source_unit_metrics():
    base = {
        "session": "Allen_test",
        "neuron_mask": np.asarray([4, 9]),
        "robs_used": np.zeros((2, 3, 2)),
        "rhos": np.asarray([0.3, 0.4]),
        "ccnorm": np.asarray([0.5, 0.6]),
        "ccmax": np.asarray([0.8, 0.9]),
        "ve_model": np.asarray([0.01, 0.02]),
    }
    other = {
        **base,
        "rhos": np.asarray([0.35, 0.45]),
        "ccnorm": np.asarray([0.55, 0.65]),
        "ve_model": np.asarray([0.015, 0.025]),
    }
    table = aligned_table({"Allen_test": base}, {"Allen_test": other})
    assert table.source_unit_index.tolist() == [4, 9]
    np.testing.assert_allclose(table.m16_rho, [0.35, 0.45])


def test_figure3_stabilized_render_uses_model_aperture_center_crop():
    raw = np.arange(2 * 51 * 51).reshape(2, 51, 51)
    cropped = _center_crop_spatial(raw, (35, 35))
    assert cropped.shape == (2, 35, 35)
    np.testing.assert_array_equal(cropped, raw[:, 8:43, 8:43])
    np.testing.assert_array_equal(_crop_model_aperture(raw, (35, 35)), cropped)


def test_figure3_native_240_supervision_uses_120hz_pair_endpoints():
    config = {
        "sampling": {"source_rate": 240, "target_rate": 240},
        "supervision": {"target_rate": 120, "phase": 1},
    }
    keep, bins = analysis_endpoint_mask_and_psth(config, np.arange(8))
    assert keep.tolist() == [False, True, False, True, False, True, False, True]
    assert bins[keep].tolist() == [0, 1, 2, 3]


def test_figure3_true_native_240_uses_same_endpoints_but_sums_counts():
    config = {"sampling": {"source_rate": 240, "target_rate": 240}}
    keep, bins = analysis_endpoint_mask_and_psth(config, np.arange(8))
    assert keep.tolist() == [False, True, False, True, False, True, False, True]
    assert bins[keep].tolist() == [0, 1, 2, 3]
    assert figure3_analysis_grid(config)["sum_native_counts"]

    counts = np.arange(16, dtype=float).reshape(8, 2)
    summed = analysis_endpoint_block_sum(config, counts, keep)
    np.testing.assert_array_equal(
        summed[keep], counts.reshape(4, 2, 2).sum(axis=1)
    )

    data_filter = np.ones((8, 2), dtype=float)
    data_filter[0, 0] = 0
    data_filter[6, 1] = np.nan
    paired = analysis_endpoint_block_filter(config, data_filter, keep)
    np.testing.assert_array_equal(
        paired[keep], [[0, 1], [1, 1], [1, 1], [1, 0]]
    )

    endpoints, model_rows = analysis_model_indices(config, keep, minimum_index=2)
    np.testing.assert_array_equal(endpoints, [3, 5, 7])
    np.testing.assert_array_equal(model_rows, [2, 3, 4, 5, 6, 7])
    prediction = model_rows[:, None].astype(float)
    np.testing.assert_array_equal(
        analysis_reduce_model_output(config, prediction, len(endpoints)).ravel(),
        [5, 9, 13],
    )


def test_figure3_supervised_native_endpoint_is_not_summed_twice():
    config = {
        "sampling": {"source_rate": 240, "target_rate": 240},
        "supervision": {"target_rate": 120, "phase": 1},
    }
    keep, _ = analysis_endpoint_mask_and_psth(config, np.arange(4))
    values = np.arange(8, dtype=float).reshape(4, 2)
    np.testing.assert_array_equal(
        analysis_endpoint_block_sum(config, values, keep), values
    )
    assert not figure3_analysis_grid(config)["sum_native_counts"]


def test_figure3_ablation_ccnorm_is_affine_invariant_on_shared_data_mask():
    rng = np.random.default_rng(9)
    n_trials, n_time, n_units = 48, 30, 2
    time = np.linspace(0, 2 * np.pi, n_time, endpoint=False)
    mean = np.stack(
        [0.8 + 0.5 * np.sin(time), 1.1 + 0.4 * np.cos(2 * time)], axis=-1
    )
    robs = rng.poisson(mean[None], size=(n_trials, n_time, n_units)).astype(float)
    prediction = np.broadcast_to(mean[None], robs.shape).copy()
    dfs = np.ones_like(robs)
    dfs[0, 0, 0] = np.nan
    dfs[1, 1, 1] = 0

    ccnorm, ccmax, ccabs, unstable = _compute_ccnorm_by_condition(
        robs,
        {"base": prediction, "affine": 1.7 * prediction + 0.3},
        dfs,
        n_splits=20,
    )

    np.testing.assert_allclose(ccnorm["base"], ccnorm["affine"], atol=1e-12)
    np.testing.assert_allclose(ccabs["base"], ccabs["affine"], atol=1e-12)
    identity = ccabs["base"] / ccmax
    identity[unstable] = np.nan
    np.testing.assert_allclose(ccnorm["base"], identity, atol=1e-12)
    assert np.isfinite(ccmax).all()


def test_figure3_ablation_conditions_are_renormalized_to_intact_ceiling():
    ccabs = {
        "intact": np.asarray([0.4, 0.3, 0.2]),
        "zeroed": np.asarray([0.2, 0.15, 0.1]),
        "stabilized": np.asarray([0.1, 0.06, 0.05]),
    }
    anchor = {
        "ccabs": np.asarray([0.4, 0.3, 0.2]),
        "ccmax": np.asarray([0.8, 0.6, 0.5]),
        "ccnorm": np.asarray([0.5, np.nan, 0.4]),
        "ccnorm_unstable": np.asarray([False, True, False]),
    }
    ccnorm, ccmax, anchored_abs, unstable = (
        _renormalize_ccnorm_to_intact_anchor(ccabs, anchor)
    )
    np.testing.assert_array_equal(ccmax, anchor["ccmax"])
    np.testing.assert_array_equal(unstable, anchor["ccnorm_unstable"])
    for condition in ccabs:
        expected = anchored_abs[condition] / anchor["ccmax"]
        expected[anchor["ccnorm_unstable"]] = np.nan
        np.testing.assert_allclose(ccnorm[condition], expected, equal_nan=True)


def test_figure3_femfraction_uses_unified_ablation_cache_and_fig2_population():
    component = {
        "B_obs": np.asarray([0.1, 0.2, 0.3, 0.7]),
        "B_model": np.asarray([0.4, 0.5, 0.6, 0.8]),
        "B_obs_uncl": np.asarray([-0.1, 0.2, 1.3, 0.7]),
        "B_model_uncl": np.asarray([0.4, 0.5, 0.6, 0.8]),
    }
    payload = {
        "schema_version": 7,
        "femfraction_count_bins": 3,
        "results": [{
            "session": "Allen_test",
            "subject": "Allen",
            "neuron_mask": np.asarray([4, 9, 12, 14]),
            "femfraction": {"intact": component},
        }],
    }
    aligned = {
        "Allen_test": {
            "neuron_mask": np.asarray([4, 9, 12, 14]),
            "rate_hz": np.asarray([4.0, 1.0, 5.0, 6.0]),
            "psth_r2": np.asarray([0.2, 0.4, 0.3, 0.2]),
        }
    }
    out = _aggregate_ablation_femfraction(payload, "intact", aligned)
    np.testing.assert_array_equal(out["B_obs"], [0.1, 0.3, 0.7])
    np.testing.assert_array_equal(out["B_model_uncl"], [0.4, 0.6, 0.8])
    assert out["session"].tolist() == ["Allen_test"] * 3


def test_figure3_femfraction_derives_legacy_valid_mask_from_finite_eye_grid():
    eye = np.zeros((2, 3, 2), dtype=float)
    eye[1, 2, 0] = np.nan
    expected = np.ones((2, 3), dtype=bool)
    expected[1, 2] = False
    np.testing.assert_array_equal(_base_valid_mask({}, eye), expected)

    supplied = np.asarray([[True, False, True], [True, True, False]])
    np.testing.assert_array_equal(
        _base_valid_mask({"valid_mask": supplied}, eye), supplied
    )


def test_figure3_native_240_supervision_uses_block_start_trial_coordinates():
    config = {
        "sampling": {"source_rate": 240, "target_rate": 240},
        "supervision": {"target_rate": 120, "phase": 1},
    }
    # Trial 11 begins on the opposite global phase.  Endpoint labelling would
    # put its first valid block in bin 1; the block-start convention correctly
    # restarts it at Figure-3 bin 0 and drops the cross-trial pair at index 3.
    trial = np.asarray([10, 10, 10, 11, 11, 11, 11, 11])
    psth = np.asarray([0, 1, 2, 0, 1, 2, 3, 4])
    keep, bins = analysis_endpoint_mask_and_psth(config, psth, trial)
    assert keep.tolist() == [False, True, False, False, False, True, False, True]
    assert bins[keep].tolist() == [0, 0, 1]


def test_figure3_native_240_continuous_covariates_use_pair_means():
    config = {
        "sampling": {"source_rate": 240, "target_rate": 240},
        "supervision": {"target_rate": 120, "phase": 1},
    }
    trial = np.asarray([10, 10, 10, 11, 11, 11, 11, 11])
    psth = np.asarray([0, 1, 2, 0, 1, 2, 3, 4])
    values = np.stack([np.arange(8), 10 * np.arange(8)], axis=1).astype(float)
    keep, _ = analysis_endpoint_mask_and_psth(config, psth, trial)
    averaged = analysis_endpoint_block_mean(config, values, keep)
    np.testing.assert_allclose(
        averaged[keep],
        [[0.5, 5.0], [4.5, 45.0], [6.5, 65.0]],
    )
    # Non-endpoint values are irrelevant to Figure 3 but remain shape- and
    # value-preserving, which makes the helper safe for flat dataset arrays.
    np.testing.assert_array_equal(averaged[~keep], values[~keep])


def test_figure3_native_predictions_use_canonical_observation_support():
    robs = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    prediction = robs + 100
    neuron_mask = np.asarray([1, 3])
    reference_robs = robs[:, :, neuron_mask].copy()
    reference_robs[0, 0, 0] = np.nan
    reference_dfs = np.ones_like(reference_robs)
    reference_dfs[1, 2, 1] = 0
    reference = {
        "neuron_mask": neuron_mask,
        "robs_used": reference_robs,
        "dfs_used": reference_dfs,
    }
    aligned_robs, aligned_prediction, dfs, selected = (
        align_native_trial_arrays_to_reference(robs, prediction, reference)
    )
    np.testing.assert_array_equal(selected, neuron_mask)
    np.testing.assert_array_equal(aligned_robs, reference_robs)
    np.testing.assert_array_equal(dfs, reference_dfs)
    assert np.isnan(aligned_prediction[0, 0, 0])
    np.testing.assert_allclose(
        aligned_prediction[np.isfinite(reference_robs)],
        prediction[:, :, neuron_mask][np.isfinite(reference_robs)],
    )


def test_figure3_legacy_downsampled_data_keeps_existing_psth_frame():
    config = {"sampling": {"source_rate": 240, "target_rate": 120}}
    keep, bins = analysis_endpoint_mask_and_psth(config, np.arange(4))
    assert keep.all()
    assert bins.tolist() == [0, 1, 2, 3]


def test_figure3_caption_is_derived_from_selected_model_and_current_stats():
    caption = _build_caption(
        {
            "model": {"family": "dekel"},
            "panel_c_stats": {
                "n_units": 12,
                "n_sessions": 3,
                "medians": {"intact": 0.71, "zeroed": 0.68, "stabilized": 0.42},
                "contrasts": {
                    "zeroed_vs_intact": {
                        "median_difference": -0.03,
                        "wilcoxon_p": 0.02,
                    },
                    "stabilized_vs_intact": {
                        "median_difference": -0.29,
                        "wilcoxon_p": 1e-6,
                    },
                },
            },
        }
    )
    assert "nonrecurrent, anti-aliased Dekel" in caption
    assert "across 12 cells from 3 sessions" in caption
    assert "0.710" in caption
    assert "984 cells" not in caption
