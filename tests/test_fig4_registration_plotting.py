from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from paper.fig4.mechanism_audit_v1.registration_mechanism import plot_registration as plot


def compact_products() -> tuple[pd.DataFrame, dict, dict, plot.CompletenessContract]:
    contract = plot.CompletenessContract(
        pairs=2,
        folds=1,
        pairs_per_fold=2,
        frames=2,
        internal_registration_steps=2,
        contrasts=plot.CONTRASTS,
        scales=(0.0, 1.0),
        term_rows=24,
    )
    subspaces = ("learned_p", "learned_q", "readout_svd", "random_00", "random_01")
    rows = []
    for pair in range(2):
        for scale in contract.scales:
            for frame in range(contract.frames):
                for step in (1, 2):
                    expected_x = (-0.8 + 0.3 * frame + 0.2 * pair) * (0.2 + scale)
                    expected_y = (0.6 - 0.2 * step) * (0.2 + scale)
                    for contrast in contract.contrasts:
                        for subspace in subspaces:
                            coefficient = {
                                "learned_p": 0.92,
                                "learned_q": 0.45,
                                "readout_svd": 0.78,
                                "random_00": 0.18,
                                "random_01": 0.22,
                            }[subspace]
                            row = {name: 0.0 for name in plot.BASE_COLUMNS}
                            row.update(
                                scope="heldout",
                                contrast=contrast,
                                fold=0,
                                image_position=pair,
                                trajectory_position=pair + 4,
                                scale=scale,
                                frame_position=frame,
                                internal_step=step,
                                subspace=subspace,
                                eye_delta_x_deg=expected_x / 10,
                                eye_delta_y_deg=expected_y / 10,
                                expected_feature_shift_x_px=expected_x,
                                expected_feature_shift_y_px=expected_y,
                                raw_zero_lag_correlation=0.25,
                                raw_best_lag_correlation=0.55,
                                raw_lag_x_px=-expected_x,
                                raw_lag_y_px=-expected_y,
                                raw_at_search_boundary=False,
                                raw_peak_sharpness=0.1,
                                raw_residual_mismatch=0.45,
                                recurrent_zero_lag_correlation=0.25 + 0.2 * coefficient,
                                recurrent_best_lag_correlation=0.55 + 0.1 * coefficient,
                                recurrent_lag_x_px=(coefficient - 1.0) * expected_x,
                                recurrent_lag_y_px=(coefficient - 1.0) * expected_y,
                                recurrent_at_search_boundary=False,
                                recurrent_peak_sharpness=0.12,
                                recurrent_residual_mismatch=0.35,
                                transport_x_px=coefficient * expected_x,
                                transport_y_px=coefficient * expected_y,
                                expected_outside_search_window=False,
                                zero_lag_alignment_improvement=0.2 * coefficient,
                                best_lag_alignment_improvement=0.1 * coefficient,
                                valid=True,
                            )
                            rows.append(row)
    frame = pd.DataFrame(rows)
    consolidation = {
        "scope": "heldout",
        "registration_parts": 2,
        "term_parts": 2,
        "registration_rows": len(frame),
        "projected_term_rows": 24,
        "registration_settings": {"random_draws": 2, "max_lag_px": 4, "subpixel_factor": 4},
    }
    terms = {
        "metadata": np.zeros((24, 7), dtype=np.int16),
        "values": np.zeros((24, 4), dtype=np.float32),
        "scope": np.asarray("heldout"),
    }
    return frame, consolidation, terms, contract


def test_plot_module_has_no_model_renderer_core_or_cache_imports() -> None:
    source = inspect.getsource(plot)
    for forbidden in (
        "RealTraceMatrixScorer",
        "make_corrected_causal_stims",
        "run_exact_subset",
        "STATE_CACHE",
        "MODEL_CHECKPOINT_PATH",
        "models.modules",
    ):
        assert forbidden not in source


def test_completeness_contract_accepts_exact_products_and_rejects_partial() -> None:
    frame, consolidation, terms, contract = compact_products()
    validated = plot.validate_registration_frame(
        frame.copy(), consolidation, terms, contract=contract
    )
    assert len(validated) == len(frame)
    with pytest.raises(plot.DataUnavailable, match="row count"):
        plot.validate_registration_frame(
            frame.iloc[:-1].copy(), consolidation, terms, contract=contract
        )


def test_rank_gate_fails_closed() -> None:
    gate = {
        "status": "complete",
        "stop_downstream_mechanism_audit": False,
        "heldout_generalization": {
            key: {"generalizes": True} for key in plot.CONTRASTS
        },
    }
    plot.validate_rank_gate(gate)
    gate["heldout_generalization"]["high_1_to_3"]["generalizes"] = False
    with pytest.raises(plot.DataUnavailable, match="fails held-out"):
        plot.validate_rank_gate(gate)


def test_random_controls_are_averaged_per_observation_and_statistics_are_vectorial() -> None:
    frame, consolidation, terms, contract = compact_products()
    frame = plot.validate_registration_frame(frame, consolidation, terms, contract=contract)
    methods = plot.method_rows(frame)
    assert set(methods.method) == set(plot.METHOD_ORDER)
    random = methods.loc[methods.method.eq("random rank-8")]
    expected_observations = len(frame) // 5
    assert len(random) == expected_observations
    stats = plot.transport_statistics(methods)
    p = stats.loc[stats.method.eq("learned P")].iloc[0]
    random_stat = stats.loc[stats.method.eq("random rank-8")].iloc[0]
    assert p.vector_correlation > 0.99
    assert p.median_displacement_error_px < random_stat.median_displacement_error_px


def test_boundary_rows_are_excluded_from_transport_inference() -> None:
    frame, consolidation, terms, contract = compact_products()
    frame.loc[frame.subspace.eq("learned_p"), "raw_at_search_boundary"] = True
    frame = plot.validate_registration_frame(frame, consolidation, terms, contract=contract)
    with pytest.raises(plot.DataUnavailable, match="Too few resolved"):
        plot.transport_statistics(plot.method_rows(frame))


def test_common_example_component_and_four_panel_drawing() -> None:
    rng = np.random.default_rng(12)
    basis = np.eye(128, 8, dtype=np.float32)
    example = {
        "metadata": {"internal_step": 4},
        "learned_projector_basis": basis,
        "h_previous": rng.normal(size=(8, 128, 7, 7)).astype(np.float32),
        "candidate_current_preactivation": rng.normal(size=(8, 128, 7, 7)).astype(np.float32),
        "candidate_recurrent_preactivation": rng.normal(size=(8, 128, 7, 7)).astype(np.float32),
    }
    row = pd.Series(
        {
            "eye_delta_x_deg": 0.04, "eye_delta_y_deg": -0.03,
            "expected_feature_shift_x_px": -0.5, "expected_feature_shift_y_px": 0.25,
            "raw_zero_lag_correlation": 0.2, "raw_best_lag_correlation": 0.5,
            "raw_lag_x_px": 0.5, "raw_lag_y_px": -0.25,
            "recurrent_zero_lag_correlation": 0.4, "recurrent_best_lag_correlation": 0.6,
            "recurrent_lag_x_px": 0.0, "recurrent_lag_y_px": 0.0,
            "raw_at_search_boundary": False, "recurrent_at_search_boundary": False,
            "expected_outside_search_window": False,
        }
    )
    maps, metrics = plot.example_plot_data(example, row)
    assert maps["raw previous state"].shape == (7, 7)
    assert maps["common_component_direction"].shape == (8,)

    binned = pd.DataFrame(
        [
            {
                "method": method, "component": component, "bin": index, "n": 10,
                "expected_shift_px": x, "mean_transport_px": coefficient * x,
                "sem_transport_px": 0.02,
            }
            for method, coefficient in zip(plot.METHOD_ORDER, (0.9, 0.5, 0.75, 0.2))
            for component in ("x", "y")
            for index, x in enumerate(np.linspace(-1, 1, 5))
        ]
    )
    statistics = pd.DataFrame(
        [
            {
                "method": method, "vector_correlation": 0.8 - 0.1 * index,
                "median_displacement_error_px": 0.2 + 0.1 * index,
                "r2_x": 0.7, "r2_y": 0.6,
                "vector_variance_explained": 0.65 - 0.1 * index,
            }
            for index, method in enumerate(plot.METHOD_ORDER)
        ]
    )
    alignment = pd.DataFrame(
        [
            {
                "population": population, "scale": scale, "mean": 0.02 * scale,
                "ci95_low": 0.02 * scale - 0.01, "ci95_high": 0.02 * scale + 0.01,
            }
            for population in ("lower SF", "higher SF")
            for scale in plot.SCALES
        ]
    )
    residual = pd.DataFrame(
        [
            {
                "population": population, "scale": scale,
                "median_transport_error_px": 0.2 + 0.1 * scale,
                "ssi": 0.1 + 0.02 * scale,
            }
            for population in ("lower SF", "higher SF")
            for scale in plot.SCALES
        ]
    )
    figure = plot.draw_figure(
        maps, metrics, binned, statistics, alignment, residual,
        plot.residual_associations(residual), shared_high=True
    )
    assert len(figure.axes) == 8
    figure.canvas.draw()


def test_exact_ssi_requires_two_groups_by_five_scales() -> None:
    frame = pd.DataFrame(
        [
            {
                "figure4_sf_group": group, "condition": "normal_moving",
                "scale": scale, "ssi": 0.1 + 0.01 * scale,
            }
            for group in ("low", "high")
            for scale in plot.SCALES
        ]
    )
    assert len(plot.validate_exact_ssi(frame)) == 10
    with pytest.raises(plot.DataUnavailable):
        plot.validate_exact_ssi(frame.iloc[:-1])
