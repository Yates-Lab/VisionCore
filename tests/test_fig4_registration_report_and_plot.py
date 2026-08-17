from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.fig4.mechanism_audit_v1.registration_mechanism import report_and_plot as report


def compact_causal_products(scope: str = "pilot") -> report.CausalProducts:
    transport_rows = []
    for condition_index, condition in enumerate(report.TRANSPORT_CONDITIONS):
        attenuation = 1.0 if condition == "intact" else 0.45
        for scale in report.SCALES:
            for population in report.POPULATIONS:
                if population == "lower_sf_71":
                    # Exactly +0.010 bits from 0× to 2× for intact, with
                    # attenuated movement modulation under interventions.
                    ssi = 0.08 + attenuation * 0.005 * scale
                elif population == "higher_sf_29":
                    # Exactly +0.008 bits from 0× to 1× and -0.006 bits
                    # from 1× to 3× for intact.
                    if scale <= 1.0:
                        ssi = 0.08 + attenuation * 0.008 * scale
                    else:
                        ssi = 0.08 + attenuation * (0.008 - 0.003 * (scale - 1.0))
                else:
                    ssi = 0.08 + attenuation * 0.003 * scale
                transport_rows.append(
                    {
                        "schema_version": report.CAUSAL_SCHEMA,
                        "scope": scope,
                        "analysis": "transport_ablation_dose_curve",
                        "condition": condition,
                        "population": population,
                        "scale": scale,
                        "exact_ssi_bits": ssi,
                        "expected_spikes": 1000.0,
                        "normalized_map_recovery_r2_vs_intact": 1.0 if condition == "intact" else 0.8,
                    }
                )
        for contrast, population, effect in (
            (report.CAUSAL_CONTRAST["low_0_to_2"], "lower_sf_71", 0.010),
            (report.CAUSAL_CONTRAST["high_0_to_1"], "higher_sf_29", 0.008),
            (report.CAUSAL_CONTRAST["high_1_to_3"], "higher_sf_29", -0.006),
        ):
            scale_a, scale_b = {
                report.CAUSAL_CONTRAST["low_0_to_2"]: (0.0, 2.0),
                report.CAUSAL_CONTRAST["high_0_to_1"]: (0.0, 1.0),
                report.CAUSAL_CONTRAST["high_1_to_3"]: (1.0, 3.0),
            }[contrast]
            transport_rows.append(
                {
                    "schema_version": report.CAUSAL_SCHEMA,
                    "scope": scope,
                    "analysis": "transport_ablation_contrast",
                    "condition": condition,
                    "population": population,
                    "contrast": contrast,
                    "scale": np.nan,
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "scale_a_endpoint_condition": condition,
                    "scale_b_endpoint_condition": condition,
                    "contrast_endpoint_rule": "same intervention condition at both endpoints",
                    "exact_ssi_change_b_minus_a_bits": effect * attenuation,
                }
            )
    transport = pd.DataFrame(transport_rows)

    realign_values = {
        "no_shift_intact": (0.100, 0.080),
        "eye_correct_candidate_p": (0.100, 0.095),
        "activation_oracle_candidate_p": (0.100, 0.102),
        "eye_opposite_candidate_p": (0.100, 0.078),
        "eye_random_matched_candidate_p": (0.100, 0.079),
        "eye_correct_complementary_q": (0.100, 0.084),
        "induced_eye_misalignment_candidate_p": (0.088, 0.080),
    }
    realign_rows = []
    for condition in report.REALIGNMENT_CONDITIONS:
        at_one, at_three = realign_values[condition]
        for scale, high_ssi in ((1.0, at_one), (3.0, at_three)):
            for population in report.POPULATIONS:
                is_run = report._realignment_cell_is_run(condition, scale)
                realign_rows.append(
                    {
                        "schema_version": report.CAUSAL_SCHEMA,
                        "scope": scope,
                        "analysis": "realignment_dose_curve",
                        "condition": condition,
                        "population": population,
                        "scale": scale,
                        "exact_ssi_bits": (
                            high_ssi if population == "higher_sf_29" else high_ssi + 0.01
                        )
                        if is_run
                        else np.nan,
                        "expected_spikes": 1000.0 if is_run else 0.0,
                    }
                )
        realign_rows.append(
            {
                "schema_version": report.CAUSAL_SCHEMA,
                "scope": scope,
                "analysis": "realignment_contrast",
                "condition": condition,
                "population": "higher_sf_29",
                "contrast": report.CAUSAL_CONTRAST["high_1_to_3"],
                "scale": np.nan,
                "scale_a": 1.0,
                "scale_b": 3.0,
                "scale_a_endpoint_condition": (
                    "no_shift_intact"
                    if condition
                    not in {"no_shift_intact", "induced_eye_misalignment_candidate_p"}
                    else condition
                ),
                "scale_b_endpoint_condition": condition,
                "contrast_endpoint_rule": (
                    "same intervention condition at both endpoints"
                    if condition
                    in {"no_shift_intact", "induced_eye_misalignment_candidate_p"}
                    else f"scale_a uses no_shift_intact; scale_b uses {condition}"
                ),
                "exact_ssi_change_b_minus_a_bits": (
                    np.nan
                    if condition == "induced_eye_misalignment_candidate_p"
                    else at_three - realign_values["no_shift_intact"][0]
                ),
            }
        )
    realignment = pd.DataFrame(realign_rows)

    rng = np.random.default_rng(32)

    def archive(conditions: tuple[str, ...], scales: tuple[float, ...]) -> dict[str, np.ndarray]:
        maps = 1.0 + 0.04 * rng.normal(size=(len(conditions), len(scales), 3, 51, 51))
        if conditions == report.REALIGNMENT_CONDITIONS:
            for condition_index, condition in enumerate(conditions):
                for scale_index, scale in enumerate(scales):
                    if not report._realignment_cell_is_run(condition, scale):
                        maps[condition_index, scale_index] = np.nan
        return {
            "schema_version": np.asarray(report.CAUSAL_SCHEMA),
            "conditions": np.asarray(conditions),
            "scales": np.asarray(scales),
            "populations": np.asarray(report.POPULATIONS),
            "normalized_population_rate_maps": maps.astype(np.float32),
            "map_definition": np.asarray(
                "g(x,y)=r(x,y)/mean_xy r; expected-spike weighted across complete rows"
            ),
            "individual_conditions_rescaled": np.asarray(False),
        }

    return report.CausalProducts(
        scope=scope,
        root=Path("/tmp/not-used"),
        analysis_manifest={},
        pilot_gate=None,
        transport=transport,
        realignment=realignment,
        transport_archive=archive(report.TRANSPORT_CONDITIONS, report.SCALES),
        realignment_archive=archive(report.REALIGNMENT_CONDITIONS, (1.0, 3.0)),
        source_paths=(),
    )


def minimal_statistics(decision: str = report.DECISION_LABELS[0]) -> dict:
    quantitative = {
        "rank8_median_complete_map_recovery_r2": {key: 0.8 for key in report.CONTRASTS},
        "rank8_median_learned_minus_readout_svd_complete_map_recovery_r2": {
            key: 0.02 for key in report.CONTRASTS
        },
        "rank8_median_projector_overlap": {key: 0.9 for key in report.CONTRASTS},
        "native_effective_participating_channels": {key: 70.0 for key in report.CONTRASTS},
        "leverage_spearman_readout_strength": {key: 0.4 for key in report.CONTRASTS},
        "leverage_spearman_centered_motion_variance": {key: 0.5 for key in report.CONTRASTS},
        "p_motion_total_fraction": {key: 0.55 for key in report.CONTRASTS},
        "p_to_q_motion_energy_per_dimension": {key: 18.0 for key in report.CONTRASTS},
        "p_to_q_content_image_mean_energy_per_dimension": {key: 2.0 for key in report.CONTRASTS},
        "p_to_q_trajectory_energy_per_dimension": {key: 3.0 for key in report.CONTRASTS},
        "q_baseline_normalized_map_recovery_r2": {key: 0.70 for key in report.CONTRASTS},
        "p_movement_map_recovery_r2": {key: 0.82 for key in report.CONTRASTS},
        "q_movement_map_recovery_r2": {key: 0.05 for key in report.CONTRASTS},
        "candidate_p_is_rr100_output_relevant": True,
        "complementary_q_retains_substantial_baseline_prediction": True,
        "raw_and_recurrent_lag": {
            "n_pairs": 48,
            "median_raw_lag_feature_px": 0.8,
            "median_recurrent_lag_feature_px": 0.2,
            "median_paired_lag_reduction_feature_px": 0.6,
            "fraction_pairs_reduced": 0.85,
            "raw_lag_present": True,
            "recurrence_reduces_lag": True,
        },
        "p_zero_lag_alignment_improvement": {key: 0.10 for key in report.CONTRASTS},
        "transport_by_subspace": {
            "learned P": {
                "identity_line_variance_explained": 0.60,
                "median_vector_error_feature_px": 0.20,
            },
            "complementary Q": {
                "identity_line_variance_explained": 0.10,
                "median_vector_error_feature_px": 0.60,
            },
            "readout-SVD": {
                "identity_line_variance_explained": 0.20,
                "median_vector_error_feature_px": 0.50,
            },
            "random rank-8": {
                "identity_line_variance_explained": -0.10,
                "median_vector_error_feature_px": 0.90,
            },
        },
        "absolute_preactivation_variance_fraction": {
            "p": {key: 0.4 for key in report.CONTRASTS},
            "q": {key: 0.7 for key in report.CONTRASTS},
            "p_q_covariance": {key: -0.1 for key in report.CONTRASTS},
        },
        "boundary_resolved_transport": {
            "vector_correlation": 0.80,
            "identity_line_variance_explained": 0.60,
            "median_vector_error_feature_px": 0.20,
        },
        "high_sf_transport_error_by_scale_feature_px": {
            scale: 0.1 + 0.1 * scale for scale in report.SCALES
        },
        "learned_p_kernel_offcenter_fraction": {
            "candidate": 0.5,
            "reset_gate": 0.4,
            "update_gate": 0.3,
        },
        "transport_ssi_contrasts": {
            "intact": {key: 0.01 for key in report.CONTRASTS},
            "all_recurrent_center_only": {key: 0.004 for key in report.CONTRASTS},
            "offset_permuted_mean": {key: 0.003 for key in report.CONTRASTS},
            "qualifying_geometry_conditions": ["all_recurrent_center_only"],
            "maximum_stabilized_map_fidelity_r2": 0.8,
        },
        "realignment_high3_ssi": {
            "intact": 0.08,
            "correct_p": 0.095,
            "oracle_p": 0.102,
            "opposite_p": 0.078,
            "random_p": 0.079,
            "correct_q": 0.084,
        },
        "realignment_high3_gain_vs_intact_bits": {
            "correct_p": 0.015,
            "oracle_p_upper_bound": 0.022,
            "opposite_p": -0.002,
            "random_p": -0.001,
            "correct_q": 0.004,
        },
        "induced_failure_high1_ssi": {
            "intact": 0.10,
            "induced_p": 0.088,
            "intact_minus_induced_bits": 0.012,
        },
    }
    questions = [
        {
            "question": question,
            "result": "Yes",
            "primary_statistic": "saved held-out statistic",
            "confidence": "High",
        }
        for question in report.REPORT_QUESTIONS
    ]
    return {
        "decision": decision,
        "manuscript_recommendation": "SUPPORTED",
        "causal_scope": "full",
        "all_shifter_support_gates_pass": True,
        "pq_motion_energy_gate_waiver": {
            **report.PQ_GATE_WAIVER,
            "previous_gate_passed": False,
            "used_in_final_decision": False,
        },
        "quantitative": quantitative,
        "first_page_questions": questions,
        "exactly_three_manuscript_panels": ["one", "two", "three"],
    }


def assert_exact_report_contract(text: str, *, decision: str, manuscript: str) -> None:
    headings = re.findall(r"^## ([A-H])\.[^\n]+$", text, flags=re.MULTILINE)
    assert headings == list("ABCDEFGH")
    chosen_decisions = [
        label for label in report.DECISION_LABELS if label in text
    ]
    assert chosen_decisions == [decision]
    assert text.count(decision) == 1
    assert "## G. Decision" in text
    manuscript_section = text.split("## H. Manuscript recommendation", 1)[1]
    chosen_manuscripts = [
        label for label in report.MANUSCRIPT_LABELS if re.search(rf"^{re.escape(label)}$", manuscript_section, re.MULTILINE)
    ]
    assert chosen_manuscripts == [manuscript]
    recommendation = manuscript_section.split(
        "Propose exactly three final Figure 4 mechanism panels:", 1
    )[1]
    numbered = re.findall(r"^\d+\. .+$", recommendation, flags=re.MULTILINE)
    assert [line.split(".", 1)[0] for line in numbered] == ["1", "2", "3"]


def compact_upstream_inputs() -> report.UpstreamInputs:
    rank_rows = []
    stability_rows = []
    leverage_rows = []
    variance_rows = []
    readout_rows = []
    ratios = {"low_0_to_2": 1.8, "high_0_to_1": 0.95, "high_1_to_3": 1.3}
    for contrast in report.CONTRASTS:
        for fold in report.FOLDS:
            rank_rows.append(
                {
                    "fold": fold,
                    "contrast": contrast,
                    "method": "learned",
                    "complete_map_recovery_mean": 0.8,
                    "learned_minus_readout_complete_map_mean": 0.10,
                }
            )
            leverage_rows.append(
                {
                    "fold": fold,
                    "contrast": contrast,
                    "effective_participating_channel_count": 70.0,
                }
            )
            variance_rows.extend(
                [
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "movement_change_from_stabilization",
                        "scale": report.TARGET_SCALE[contrast],
                        "candidate_p_total_fraction": 0.50,
                        "p_to_q_per_dimension_energy_ratio": ratios[contrast],
                    },
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "stabilized_visual_content_image_means",
                        "scale": 0.0,
                        "candidate_p_total_fraction": 0.20,
                        "p_to_q_per_dimension_energy_ratio": 2.0,
                    },
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "trajectory_specific_at_fixed_image_scale_frame",
                        "scale": 1.0,
                        "candidate_p_total_fraction": 0.25,
                        "p_to_q_per_dimension_energy_ratio": 3.0,
                    },
                ]
            )
            population = report.TARGET_POPULATION[contrast]
            readout_rows.extend(
                [
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "baseline_visual_reconstruction",
                        "population": population,
                        "component": "complementary_q_content",
                        "normalized_map_recovery_vs_training_mean_r2": 0.7,
                        "normalized_map_movement_effect_recovery_r2": np.nan,
                    },
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "defining_contrast_movement_effect_decomposition",
                        "population": population,
                        "component": "candidate_p_only_contrast",
                        "normalized_map_recovery_vs_training_mean_r2": np.nan,
                        "normalized_map_movement_effect_recovery_r2": 0.8,
                    },
                    {
                        "fold": fold,
                        "contrast": contrast,
                        "analysis": "defining_contrast_movement_effect_decomposition",
                        "population": population,
                        "component": "complementary_q_only_contrast",
                        "normalized_map_recovery_vs_training_mean_r2": np.nan,
                        "normalized_map_movement_effect_recovery_r2": 0.1,
                    },
                ]
            )
        for first in report.FOLDS:
            for second in report.FOLDS:
                if first < second:
                    stability_rows.append(
                        {
                            "comparison_type": "within_contrast_across_folds",
                            "contrast_a": contrast,
                            "contrast_b": contrast,
                            "fold_a": first,
                            "fold_b": second,
                            "projector_overlap": 0.9,
                        }
                    )
    rank_gate = {
        "heldout_generalization": {
            contrast: {
                "generalizes": True,
                "learned_outperforms_readout_consistently": True,
            }
            for contrast in report.CONTRASTS
        }
    }
    return report.UpstreamInputs(
        root=Path("/tmp/not-used"),
        rank_gate=rank_gate,
        rank_results=pd.DataFrame(rank_rows),
        stability=pd.DataFrame(stability_rows),
        leverage=pd.DataFrame(leverage_rows),
        variance=pd.DataFrame(variance_rows),
        readout=pd.DataFrame(readout_rows),
        per_unit=pd.DataFrame([{"unused": 1}]),
        probes=pd.DataFrame([{"unused": 1}]),
        figure1_manifest={},
        source_paths=(),
    )


def test_module_imports_no_model_renderer_cache_or_core_replay_modules() -> None:
    tree = ast.parse(inspect.getsource(report))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("models", "causal_low_rank", "correction", "upstream", "instrument_subset")
    assert not any(any(term in imported for term in forbidden) for imported in imports)


def test_final_report_requires_heldout_gru_products_not_consensus_activation_replay() -> None:
    assert "gru_projected_terms_heldout.npz" in report.REQUIRED_CANONICAL
    assert "registration_metrics_heldout.csv" in report.REQUIRED_CANONICAL
    assert "gru_projected_terms.npz" not in report.REQUIRED_CANONICAL
    assert "registration_metrics.csv" not in report.REQUIRED_CANONICAL

    value_columns = np.asarray(
        [
            f"{term}__{suffix}"
            for term in report.GRU_PROJECTED_TERM_NAMES
            for suffix in report.GRU_PROJECTED_VALUE_SUFFIXES
        ]
    )
    archive = {
        "scope": np.asarray("heldout"),
        "metadata": np.zeros((2, 7), dtype=np.int16),
        "values": np.zeros((2, 63), dtype=np.float32),
        "value_columns": value_columns,
    }
    report._validate_heldout_projected_terms(archive, expected_rows=2)
    changed = dict(archive)
    changed["value_columns"] = value_columns[::-1]
    with pytest.raises(report.DataUnavailable, match="semantics changed"):
        report._validate_heldout_projected_terms(changed, expected_rows=2)
    nonfinite = dict(archive)
    nonfinite["values"] = archive["values"].copy()
    nonfinite["values"][0, 0] = np.nan
    with pytest.raises(report.DataUnavailable, match="nonfinite"):
        report._validate_heldout_projected_terms(nonfinite, expected_rows=2)

    report._validate_gru_reconstruction(
        {
            "h_reconstruction_max_abs": 0.0,
            "retained_plus_new_max_abs": 0.0,
            "candidate_split_max_abs": 2e-6,
        }
    )
    with pytest.raises(report.DataUnavailable, match="candidate_split"):
        report._validate_gru_reconstruction(
            {
                "h_reconstruction_max_abs": 0.0,
                "retained_plus_new_max_abs": 0.0,
                "candidate_split_max_abs": 4e-6,
            }
        )


def test_finalized_probe_representation_labels_are_accepted_exactly() -> None:
    probes = pd.DataFrame(
        [
            {
                "fold": 0,
                "contrast": "low_0_to_2",
                "probe": "movement_scale",
                "representation": representation,
                "value": 0.5,
                "causal_interpretation_allowed": False,
            }
            for representation in (
                "candidate_p_coordinates_rank8",
                "complementary_q_reconstructed_channels_rank120",
            )
        ]
    )
    report._validate_probe_schema(probes)
    changed = probes.copy()
    changed.loc[0, "representation"] = "candidate_p"
    with pytest.raises(report.DataUnavailable, match="candidate P and complementary Q"):
        report._validate_probe_schema(changed)


def test_causal_schema_and_archives_are_exact_and_fail_closed() -> None:
    causal = compact_causal_products()
    transport, realignment = report.validate_causal_tables(
        causal.transport.copy(), causal.realignment.copy(), scope="pilot"
    )
    assert len(transport) == 144
    assert len(realignment) == 49
    validated = report.validate_causal_archive(causal.realignment_archive, stage="realignment")
    assert validated["normalized_population_rate_maps"].shape == (7, 2, 3, 51, 51)
    sparse = validated["normalized_population_rate_maps"]
    assert np.isnan(sparse[1, 0]).all()
    assert np.isfinite(sparse[1, 1]).all()
    with pytest.raises(report.DataUnavailable, match="seven 1×→3×"):
        report.validate_causal_tables(
            transport, realignment.iloc[:-1].copy(), scope="pilot"
        )
    inconsistent = causal.transport.copy()
    contrast_row = inconsistent.index[
        inconsistent.analysis.eq("transport_ablation_contrast")
    ][0]
    inconsistent.loc[contrast_row, "exact_ssi_change_b_minus_a_bits"] += 0.001
    with pytest.raises(report.DataUnavailable, match="does not equal"):
        report.validate_causal_tables(
            inconsistent, causal.realignment.copy(), scope="pilot"
        )
    changed = dict(causal.realignment_archive)
    changed["individual_conditions_rescaled"] = np.asarray(True)
    with pytest.raises(report.DataUnavailable, match="independently rescales"):
        report.validate_causal_archive(changed, stage="realignment")
    fabricated = dict(causal.realignment_archive)
    fabricated["normalized_population_rate_maps"] = causal.realignment_archive[
        "normalized_population_rate_maps"
    ].copy()
    fabricated["normalized_population_rate_maps"][1, 0] = 1.0
    with pytest.raises(report.DataUnavailable, match="Structurally unrun"):
        report.validate_causal_archive(fabricated, stage="realignment")


def test_figure3_preparation_keeps_exact_seeds_and_one_common_map_scale() -> None:
    data = report.prepare_figure3_data(compact_causal_products())
    assert len(data["panel_a"]) == 10
    assert len(data["panel_c"]) == 5
    assert len(data["panel_d"]) == 2
    assert set(data["panel_b"].condition) == {
        "intact",
        "candidate_recurrent_center_only",
        "gate_recurrent_center_only",
        "all_recurrent_center_only",
        "offset_permuted_mean",
    }
    assert data["panel_c_maps"].shape == (5, 51, 51)
    assert data["panel_d_maps"].shape == (2, 51, 51)
    assert {
        "delta_ssi_from_intact_bits",
        "delta_ssi_from_intact_microbits",
    }.issubset(data["panel_c"].columns)
    assert {
        "delta_ssi_from_intact_bits",
        "delta_ssi_from_intact_microbits",
    }.issubset(data["panel_d"].columns)
    all_maps = np.concatenate([data["panel_c_maps"], data["panel_d_maps"]])
    assert np.isclose(data["common_map_vmax"] - 1, 1 - data["common_map_vmin"])
    assert np.nanmin(all_maps) >= data["common_map_vmin"] - 1e-12
    assert np.nanmax(all_maps) <= data["common_map_vmax"] + 1e-12


def test_figure3_draws_four_labeled_panels_without_oracle_map() -> None:
    data = report.prepare_figure3_data(compact_causal_products())
    figure = report.draw_figure3(data, scope="pilot")
    figure.canvas.draw()
    text = " ".join(item.get_text() for axis in figure.axes for item in axis.texts)
    assert all(letter in text for letter in "ABCD")
    assert "oracle" not in text.lower()
    # Main axes + seven complete maps + one colorbar.
    assert len(figure.axes) == 13
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)


def test_figure3_null_results_use_microbits_and_do_not_overstate_effects() -> None:
    data = report.prepare_figure3_data(compact_causal_products())
    panel_b = data["panel_b"].copy()
    intact_b = panel_b.loc[
        panel_b.condition.eq("intact"),
        ["population", "scale", "ssi_change_from_own_stabilization_bits"],
    ].rename(columns={"ssi_change_from_own_stabilization_bits": "intact_change"})
    panel_b = panel_b.merge(intact_b, on=["population", "scale"], validate="many_to_one")
    panel_b.loc[
        panel_b.condition.eq("offset_permuted_mean"),
        "ssi_change_from_own_stabilization_bits",
    ] = panel_b.loc[panel_b.condition.eq("offset_permuted_mean"), "intact_change"]
    data["panel_b"] = panel_b.drop(columns="intact_change")
    panel_c = data["panel_c"].copy()
    panel_c.loc[
        panel_c.condition.astype(str).eq("eye_correct_candidate_p"),
        "delta_ssi_from_intact_bits",
    ] = -1.5e-6
    panel_c["delta_ssi_from_intact_microbits"] = 1e6 * panel_c.delta_ssi_from_intact_bits
    data["panel_c"] = panel_c
    panel_d = data["panel_d"].copy()
    panel_d.loc[
        panel_d.condition.astype(str).eq("induced_eye_misalignment_candidate_p"),
        "delta_ssi_from_intact_bits",
    ] = -1.9e-6
    panel_d["delta_ssi_from_intact_microbits"] = 1e6 * panel_d.delta_ssi_from_intact_bits
    data["panel_d"] = panel_d
    figure = report.draw_figure3(
        data, scope="pilot", decision="NO EVIDENCE FOR RECURRENT REGISTRATION"
    )
    figure.canvas.draw()
    text = " ".join(item.get_text() for axis in figure.axes for item in axis.texts)
    assert "P realignment does not rescue higher-SF 3× SSI" in text
    assert "P misalignment does not impair higher-SF 1× SSI" in text
    assert "Center-only recurrence changes SSI; offset permutation does not" in text
    assert "do not support recurrent spatial transport" in figure._suptitle.get_text()
    assert any("µbits" in axis.get_ylabel() for axis in figure.axes)
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)


def test_optional_figure4_is_gated_and_uses_measured_values_only() -> None:
    causal = compact_causal_products(scope="full")
    data = report.prepare_figure3_data(causal)
    stats = minimal_statistics()
    measured = report.prepare_optional_figure4_data(stats, data["panel_a"])
    assert len(measured) == 10
    assert np.allclose(measured.groupby("population").ssi_normalized_within_population.max(), 1.0)
    figure = report.draw_optional_figure4(stats, measured)
    figure.canvas.draw()
    assert len(figure.axes) == 2
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)
    blocked = dict(stats)
    blocked["all_shifter_support_gates_pass"] = False
    with pytest.raises(report.DataUnavailable, match="forbidden"):
        report.prepare_optional_figure4_data(blocked, data["panel_a"])


def test_fixed_decision_hierarchy_and_pilot_boundary() -> None:
    base = dict(
        raw_lag_present=True,
        recurrence_reduces_lag=True,
        transport_tracks=True,
        p_transport_stronger=True,
        high_sf_registration_failure=True,
        spatial_recurrence_changes_ssi=True,
        specific_correct_realign_rescue=True,
        induced_failure=True,
    )
    decision, manuscript, all_support = report.select_decision(causal_scope="full", **base)
    assert decision == report.DECISION_LABELS[0]
    assert manuscript == "SUPPORTED" and all_support
    decision, manuscript, all_support = report.select_decision(causal_scope="pilot", **base)
    assert decision == "INCONCLUSIVE"
    assert manuscript == "CONSISTENT WITH" and all_support

    pilot_absent = dict(base, recurrence_reduces_lag=False)
    decision, manuscript, all_support = report.select_decision(
        causal_scope="pilot", **pilot_absent
    )
    assert decision == "NO EVIDENCE FOR RECURRENT REGISTRATION"
    assert manuscript == "NOT SUPPORTED" and not all_support

    failed = dict(base)
    failed["specific_correct_realign_rescue"] = False
    decision, manuscript, _ = report.select_decision(causal_scope="full", **failed)
    assert decision == "RECURRENT SPATIAL TRANSPORT EXISTS BUT DOES NOT EXPLAIN SSI"
    assert manuscript == "NOT SUPPORTED"

    absent = dict(base, recurrence_reduces_lag=False)
    decision, _, _ = report.select_decision(causal_scope="full", **absent)
    assert decision == "NO EVIDENCE FOR RECURRENT REGISTRATION"

    nonspecific = dict(base, p_transport_stronger=False)
    decision, _, _ = report.select_decision(causal_scope="full", **nonspecific)
    assert decision == "INCONCLUSIVE"


def test_decision_vocabulary_and_interface_exclude_waived_factorization_gate() -> None:
    assert report.DECISION_LABELS == (
        "SHIFTER/REGISTRATION MECHANISM SUPPORTED",
        "RECURRENT SPATIAL TRANSPORT EXISTS BUT DOES NOT EXPLAIN SSI",
        "NO EVIDENCE FOR RECURRENT REGISTRATION",
        "INCONCLUSIVE",
    )
    parameters = set(inspect.signature(report.select_decision).parameters)
    assert not parameters.intersection(
        {"motion_enriched", "motion_broad", "q_substantial", "output_mostly_p"}
    )
    assert report.PQ_GATE_WAIVER["authorized"] is True
    assert report.PQ_GATE_WAIVER["factorization_reopened"] is False
    assert report.PQ_GATE_WAIVER["not_negative_shifter_evidence"] is True
    assert report.PQ_GATE_WAIVER["p_energy_per_dimension_must_exceed_q"] is False


def test_high_sf_failure_requires_positive_registration_before_residual_interpretation() -> None:
    common = dict(
        alignment_improvement={contrast: 0.1 for contrast in report.CONTRASTS},
        error_at_1x=0.2,
        error_at_3x=0.8,
        ssi_at_1x=0.08,
        ssi_at_3x=0.06,
    )
    evaluable, established = report._high_sf_registration_failure_status(
        transport_tracks=False,
        recurrence_reduces_lag=False,
        **common,
    )
    assert not evaluable and not established
    evaluable, established = report._high_sf_registration_failure_status(
        transport_tracks=True,
        recurrence_reduces_lag=True,
        **common,
    )
    assert evaluable and established
    negative_alignment = dict(common)
    negative_alignment["alignment_improvement"] = {
        contrast: -0.1 for contrast in report.CONTRASTS
    }
    evaluable, established = report._high_sf_registration_failure_status(
        transport_tracks=True,
        recurrence_reduces_lag=True,
        **negative_alignment,
    )
    assert not evaluable and not established


def test_lag_and_transport_specificity_use_heldout_rows_and_identity_averaged_random() -> None:
    expected = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
    method_gain = {
        "learned_p": 0.8,
        "learned_q": 0.1,
        "readout_svd": 0.6,
        "random_00": 0.0,
        "random_01": 0.0,
    }
    rows = []
    for identity_index, (expected_x, expected_y) in enumerate(expected):
        for subspace, gain in method_gain.items():
            rows.append(
                {
                    "scope": "heldout",
                    "contrast": "low_0_to_2",
                    "fold": identity_index,
                    "image_position": identity_index,
                    "trajectory_position": identity_index,
                    "scale": 2.0,
                    "frame_position": 0,
                    "internal_step": 1,
                    "subspace": subspace,
                    "expected_feature_shift_x_px": expected_x,
                    "expected_feature_shift_y_px": expected_y,
                    "transport_x_px": gain * expected_x,
                    "transport_y_px": gain * expected_y,
                    "raw_lag_x_px": expected_x,
                    "raw_lag_y_px": expected_y,
                    "recurrent_lag_x_px": 0.2 * expected_x,
                    "recurrent_lag_y_px": 0.2 * expected_y,
                    "raw_at_search_boundary": False,
                    "recurrent_at_search_boundary": False,
                    "expected_outside_search_window": False,
                    "unresolved_boundary": False,
                    "valid": True,
                }
            )
    frame = pd.DataFrame(rows)
    lag = report._lag_reduction_statistics(frame.loc[frame.subspace.eq("learned_p")])
    assert lag["raw_lag_present"] is True
    assert lag["recurrence_reduces_lag"] is True
    assert lag["median_paired_lag_reduction_feature_px"] == pytest.approx(0.8)

    methods = report._transport_method_statistics(frame)
    assert methods["random rank-8"]["n_vectors"] == len(expected)
    assert methods["learned P"]["identity_line_variance_explained"] > methods[
        "readout-SVD"
    ]["identity_line_variance_explained"]
    assert methods["learned P"]["median_vector_error_feature_px"] < methods[
        "complementary Q"
    ]["median_vector_error_feature_px"]


def test_failed_p_motion_gate_is_preserved_but_waived_and_cannot_select_a_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = compact_upstream_inputs()
    ratios, counts, passed = report.defining_motion_enrichment(inputs.variance)
    assert not passed and counts == {key: 4 for key in report.CONTRASTS}
    assert ratios["high_0_to_1"] < 1
    with pytest.raises(report.DataUnavailable, match="stop is waived"):
        report.compute_stopped_pq_statistics(inputs)
    with pytest.raises(report.DataUnavailable, match="Complete registration and causal"):
        report.run_stopped_after_pq(inputs, figure1_manifest=Path("/tmp/not-used"))

    def forbidden_upstream(*args: object, **kwargs: object) -> None:
        pytest.fail("run() must not branch on the retired P/Q stop gate")

    def missing_downstream(*args: object, **kwargs: object) -> None:
        raise report.DataUnavailable("downstream products required")

    monkeypatch.setattr(report, "load_upstream_inputs", forbidden_upstream)
    monkeypatch.setattr(report, "load_final_inputs", missing_downstream)
    with pytest.raises(report.DataUnavailable, match="downstream products required"):
        report.run()


def test_defining_motion_gate_rejects_incomplete_nonfinite_or_duplicate_fold_cells() -> None:
    variance = compact_upstream_inputs().variance
    defining = variance.loc[
        variance.analysis.eq("movement_change_from_stabilization")
    ].copy()
    dropped = variance.drop(defining.index[0])
    with pytest.raises(report.DataUnavailable, match="exactly one row"):
        report.defining_motion_enrichment(dropped)
    duplicated = pd.concat([variance, variance.loc[[defining.index[0]]]], ignore_index=True)
    with pytest.raises(report.DataUnavailable, match="exactly one row"):
        report.defining_motion_enrichment(duplicated)
    nonfinite = variance.copy()
    nonfinite.loc[defining.index[0], "p_to_q_per_dimension_energy_ratio"] = np.nan
    with pytest.raises(report.DataUnavailable, match="nonfinite"):
        report.defining_motion_enrichment(nonfinite)


def test_report_has_exact_eight_question_rows_sections_a_through_h_and_three_panels() -> None:
    text = report.render_report(minimal_statistics())
    table_rows = [
        line
        for line in text.splitlines()
        if any(line.startswith(f"| {question} |") for question in report.REPORT_QUESTIONS)
    ]
    assert len(table_rows) == 8
    assert "| Question | Result | Primary statistic | Confidence |" in text
    assert [line.split("|", 2)[1].strip() for line in table_rows] == list(
        report.REPORT_QUESTIONS
    )
    assert_exact_report_contract(
        text,
        decision=report.DECISION_LABELS[0],
        manuscript="SUPPORTED",
    )
    assert "activation-derived oracle" in text.lower()
    assert "backward in retinal time" in text
    assert "not recurrence across the 40" in text
    assert "failed P/Q motion-energy factorization is preserved" in text
    assert "factorization was not reopened" in text
    assert "not required to carry more movement energy per dimension" in text
    assert "not negative evidence for a recurrent shifter" in text


def test_statistics_json_round_trip_remains_report_renderable() -> None:
    statistics = minimal_statistics()
    round_tripped = json.loads(json.dumps(statistics))
    text = report.render_report(round_tripped)
    assert "1×=0.200" in text
    assert "3×=0.400" in text
    assert text.count(round_tripped["decision"]) == 1


@pytest.mark.parametrize("decision", report.DECISION_LABELS)
def test_report_emits_exactly_one_member_of_fixed_decision_vocabulary(decision: str) -> None:
    stats = minimal_statistics(decision)
    stats["manuscript_recommendation"] = (
        "SUPPORTED" if decision == report.DECISION_LABELS[0] else "NOT SUPPORTED"
    )
    text = report.render_report(stats)
    assert [label for label in report.DECISION_LABELS if label in text] == [decision]
    assert text.count(decision) == 1


def test_hash_record_validation_rejects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "saved.csv"
    path.write_text("a\n1\n")
    record = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    assert report._validate_record(record, label="saved", root=tmp_path) == path.resolve()
    path.write_text("a\n2\n")
    with pytest.raises(report.DataUnavailable, match="hash"):
        report._validate_record(record, label="saved", root=tmp_path)
