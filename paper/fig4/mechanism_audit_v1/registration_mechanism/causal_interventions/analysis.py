"""Streaming exact-map summaries and preregistered pilot stopping gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import rate_map_components


SCHEMA_VERSION = "fig4-registration-causal-interventions-v1"
POPULATION_LABELS = ("all_100", "lower_sf_71", "higher_sf_29")
MAP_SIZE = 51
PILOT_MIN_CAUSAL_CHANGE_BITS = 0.002
PILOT_MIN_STABILIZED_GAIN_RECOVERY_R2 = 0.25


@dataclass(frozen=True)
class PopulationDefinition:
    label: str
    units: np.ndarray


def population_definitions(low: np.ndarray, high: np.ndarray) -> tuple[PopulationDefinition, ...]:
    low_units = np.asarray(low, dtype=np.int64)
    high_units = np.asarray(high, dtype=np.int64)
    if len(low_units) != 71 or len(high_units) != 29:
        raise ValueError("Historical SF populations must remain 71 lower / 29 higher units")
    if not np.array_equal(np.sort(np.r_[low_units, high_units]), np.arange(100)):
        raise ValueError("Historical SF populations must exhaust RR100 without overlap")
    return (
        PopulationDefinition("all_100", np.arange(100, dtype=np.int64)),
        PopulationDefinition("lower_sf_71", low_units),
        PopulationDefinition("higher_sf_29", high_units),
    )


class EndpointAccumulator:
    """Raw sufficient statistics for exact SSI, rates, maps, and fidelity."""

    ARRAY_NAMES = (
        "ssi_numerator",
        "ssi_weight",
        "mean_rate_sum",
        "mean_rate_count",
        "gain_map_numerator",
        "gain_map_weight",
        "preactivation_residual_sse",
        "preactivation_reference_sse",
        "rate_residual_sse",
        "rate_reference_sse",
        "gain_residual_sse",
        "gain_reference_sse",
    )

    def __init__(
        self,
        conditions: Sequence[str],
        scales: Sequence[float],
        populations: Sequence[PopulationDefinition],
    ) -> None:
        self.conditions = tuple(map(str, conditions))
        self.scales = tuple(float(value) for value in scales)
        self.populations = tuple(populations)
        core = (len(self.conditions), len(self.scales), len(self.populations))
        self.ssi_numerator = np.zeros(core, dtype=np.float64)
        self.ssi_weight = np.zeros(core, dtype=np.float64)
        self.mean_rate_sum = np.zeros(core, dtype=np.float64)
        self.mean_rate_count = np.zeros(core, dtype=np.int64)
        self.gain_map_numerator = np.zeros((*core, MAP_SIZE, MAP_SIZE), dtype=np.float64)
        self.gain_map_weight = np.zeros(core, dtype=np.float64)
        for name in self.ARRAY_NAMES[6:]:
            setattr(self, name, np.zeros(core, dtype=np.float64))

    @staticmethod
    def _components(
        preactivation: torch.Tensor,
        rate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components = rate_map_components(rate.double())
        return {
            "preactivation": preactivation.double(),
            "rate": rate.double(),
            "gain": components["gain"],
            "ssi": components["ssi"],
            "expected_spikes": components["expected_spikes"],
            "mean_rate": components["mean_rate"],
        }

    def add(
        self,
        condition_index: int,
        scale_index: int,
        *,
        preactivation: torch.Tensor,
        rate: torch.Tensor,
        intact_preactivation: torch.Tensor,
        intact_rate: torch.Tensor,
    ) -> None:
        observed = self._components(preactivation, rate)
        intact = self._components(intact_preactivation, intact_rate)
        if observed["gain"].shape[-2:] != (MAP_SIZE, MAP_SIZE):
            raise ValueError(f"Expected 51x51 RR100 maps, found {observed['gain'].shape[-2:]}")
        for population_index, population in enumerate(self.populations):
            units = torch.as_tensor(population.units, device=rate.device, dtype=torch.long)
            index = (int(condition_index), int(scale_index), population_index)
            ssi = observed["ssi"].index_select(1, units)
            expected = observed["expected_spikes"].index_select(1, units)
            mean_rate = observed["mean_rate"].index_select(1, units)
            gain = observed["gain"].index_select(1, units)
            self.ssi_numerator[index] += float((ssi * expected).sum().detach().cpu())
            self.ssi_weight[index] += float(expected.sum().detach().cpu())
            self.mean_rate_sum[index] += float(mean_rate.sum().detach().cpu())
            self.mean_rate_count[index] += int(mean_rate.numel())
            self.gain_map_numerator[index] += (
                (gain * expected[..., None, None]).sum(dim=(0, 1)).detach().cpu().numpy()
            )
            self.gain_map_weight[index] += float(expected.sum().detach().cpu())

            intact_expected = intact["expected_spikes"].index_select(1, units)
            paired_weight = 0.5 * (expected + intact_expected)
            for metric in ("preactivation", "rate", "gain"):
                estimate = observed[metric].index_select(1, units)
                target = intact[metric].index_select(1, units)
                if metric == "gain":
                    reference = torch.ones_like(target)
                else:
                    reference = target.mean(dim=(-2, -1), keepdim=True)
                weight = paired_weight[..., None, None]
                residual = ((estimate - target).square() * weight).sum()
                reference_sse = ((target - reference).square() * weight).sum()
                getattr(self, f"{metric}_residual_sse")[index] += float(residual.detach().cpu())
                getattr(self, f"{metric}_reference_sse")[index] += float(reference_sse.detach().cpu())

    def payload(self, **metadata: np.ndarray) -> dict[str, np.ndarray]:
        result = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "conditions": np.asarray(self.conditions, dtype="U80"),
            "scales": np.asarray(self.scales, dtype=np.float32),
            "populations": np.asarray([value.label for value in self.populations], dtype="U24"),
        }
        result.update({name: np.asarray(getattr(self, name)) for name in self.ARRAY_NAMES})
        result.update(metadata)
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, np.ndarray]) -> "EndpointAccumulator":
        conditions = tuple(np.asarray(payload["conditions"]).astype(str).tolist())
        scales = tuple(np.asarray(payload["scales"], dtype=float).tolist())
        labels = tuple(np.asarray(payload["populations"]).astype(str).tolist())
        populations = tuple(
            PopulationDefinition(label, np.asarray([], dtype=np.int64)) for label in labels
        )
        result = cls(conditions, scales, populations)
        for name in cls.ARRAY_NAMES:
            setattr(result, name, np.asarray(payload[name]).copy())
        return result

    def add_payload(self, payload: Mapping[str, np.ndarray]) -> None:
        if not (
            np.array_equal(np.asarray(payload["conditions"]).astype(str), np.asarray(self.conditions))
            and np.allclose(np.asarray(payload["scales"], dtype=float), self.scales)
            and np.array_equal(
                np.asarray(payload["populations"]).astype(str),
                np.asarray([value.label for value in self.populations]),
            )
        ):
            raise RuntimeError("Intervention part axes do not match during consolidation")
        for name in self.ARRAY_NAMES:
            value = np.asarray(payload[name])
            if value.shape != np.asarray(getattr(self, name)).shape:
                raise RuntimeError(f"Intervention part shape changed for {name}: {value.shape}")
            getattr(self, name)[:] += value


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    first = np.asarray(numerator, dtype=np.float64)
    second = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(first.shape, second.shape), np.nan, dtype=np.float64)
    np.divide(first, second, out=result, where=np.abs(second) > 1e-30)
    return result


def accumulator_rows(
    accumulator: EndpointAccumulator,
    *,
    analysis: str,
    scope: str,
) -> pd.DataFrame:
    """Return dose rows plus the three historical SSI contrast rows."""
    ssi = _safe_ratio(accumulator.ssi_numerator, accumulator.ssi_weight)
    rate = _safe_ratio(accumulator.mean_rate_sum, accumulator.mean_rate_count)
    map_recovery = {
        metric: 1.0
        - _safe_ratio(
            getattr(accumulator, f"{metric}_residual_sse"),
            getattr(accumulator, f"{metric}_reference_sse"),
        )
        for metric in ("preactivation", "rate", "gain")
    }
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(accumulator.conditions):
        for scale_index, scale in enumerate(accumulator.scales):
            for population_index, population in enumerate(accumulator.populations):
                idx = (condition_index, scale_index, population_index)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scope": scope,
                        "analysis": f"{analysis}_dose_curve",
                        "condition": condition,
                        "scale": scale,
                        "population": population.label,
                        "exact_ssi_bits": float(ssi[idx]),
                        "mean_rate_hz": float(rate[idx]),
                        "expected_spikes": float(accumulator.ssi_weight[idx]),
                        "preactivation_complete_map_recovery_r2_vs_intact": float(
                            map_recovery["preactivation"][idx]
                        ),
                        "rate_complete_map_recovery_r2_vs_intact": float(map_recovery["rate"][idx]),
                        "normalized_map_recovery_r2_vs_intact": float(map_recovery["gain"][idx]),
                        "map_recovery_weighting": "paired expected spikes; aggregate SSE then ratio",
                    }
                )
    scale_index = {float(value): index for index, value in enumerate(accumulator.scales)}
    contrast_specs = (
        ("lower_sf_sharpening_0_to_2", "lower_sf_71", 0.0, 2.0),
        ("higher_sf_sharpening_0_to_1", "higher_sf_29", 0.0, 1.0),
        ("higher_sf_reversal_1_to_3", "higher_sf_29", 1.0, 3.0),
    )
    population_index = {value.label: index for index, value in enumerate(accumulator.populations)}
    population_maps = normalized_population_maps(accumulator).astype(np.float64)
    reference_condition = {
        "transport_ablation": "intact",
        "realignment": "no_shift_intact",
    }.get(analysis)
    reference_condition_index = (
        accumulator.conditions.index(reference_condition)
        if reference_condition in accumulator.conditions
        else None
    )
    for label, population, scale_a, scale_b in contrast_specs:
        if scale_a not in scale_index or scale_b not in scale_index:
            continue
        p = population_index[population]
        for condition_index, condition in enumerate(accumulator.conditions):
            a_native = (condition_index, scale_index[scale_a], p)
            b_native = (condition_index, scale_index[scale_b], p)
            a = a_native
            b = b_native
            endpoint_rule = "same intervention condition at both endpoints"
            if (
                analysis == "realignment"
                and reference_condition_index is not None
                and accumulator.ssi_weight[a_native] <= 0
                and accumulator.ssi_weight[b_native] > 0
            ):
                # Rescue conditions are deliberately evaluated only at 3x.
                # Their scientifically meaningful reversal is therefore the
                # patched 3x endpoint relative to the common intact 1x donor,
                # not a nonexistent condition-specific 1x endpoint.
                a = (reference_condition_index, scale_index[scale_a], p)
                endpoint_rule = (
                    f"scale_a uses {reference_condition}; scale_b uses {condition}"
                )
            map_metrics: dict[str, Any] = {
                "normalized_population_map_effect_recovery_r2_vs_intact": math.nan,
                "normalized_population_map_rescue_fraction_to_intact_scale_a": math.nan,
                "normalized_population_map_effect_residual_rmse": math.nan,
                "normalized_population_map_effect_intact_rms": math.nan,
                "normalized_population_map_effect_observed_rms": math.nan,
                "normalized_population_map_effect_definition": (
                    "complete expected-spike-weighted population g_B-g_A map; no condition-wise rescaling"
                ),
            }
            if reference_condition_index is not None:
                reference_delta = (
                    population_maps[reference_condition_index, scale_index[scale_b], p]
                    - population_maps[reference_condition_index, scale_index[scale_a], p]
                )
                observed_delta = population_maps[b] - population_maps[a]
                if np.isfinite(reference_delta).all() and np.isfinite(observed_delta).all():
                    residual = observed_delta - reference_delta
                    denominator = float(np.square(reference_delta).sum())
                    rescue_sse = float(
                        np.square(
                            population_maps[b]
                            - population_maps[
                                reference_condition_index, scale_index[scale_a], p
                            ]
                        ).sum()
                    )
                    map_metrics.update(
                        {
                            "normalized_population_map_effect_recovery_r2_vs_intact": (
                                float(1.0 - np.square(residual).sum() / denominator)
                                if denominator > 1e-30
                                else math.nan
                            ),
                            "normalized_population_map_effect_residual_rmse": float(
                                np.sqrt(np.mean(np.square(residual)))
                            ),
                            "normalized_population_map_effect_intact_rms": float(
                                np.sqrt(np.mean(np.square(reference_delta)))
                            ),
                            "normalized_population_map_effect_observed_rms": float(
                                np.sqrt(np.mean(np.square(observed_delta)))
                            ),
                            "normalized_population_map_rescue_fraction_to_intact_scale_a": (
                                float(1.0 - rescue_sse / denominator)
                                if analysis == "realignment" and denominator > 1e-30
                                else math.nan
                            ),
                        }
                    )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": scope,
                    "analysis": f"{analysis}_contrast",
                    "condition": condition,
                    "scale": np.nan,
                    "population": population,
                    "contrast": label,
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "scale_a_endpoint_condition": accumulator.conditions[a[0]],
                    "scale_b_endpoint_condition": accumulator.conditions[b[0]],
                    "contrast_endpoint_rule": endpoint_rule,
                    "exact_ssi_change_b_minus_a_bits": float(ssi[b] - ssi[a]),
                    "mean_rate_change_b_minus_a_hz": float(rate[b] - rate[a]),
                    **map_metrics,
                }
            )
    return pd.DataFrame(rows)


def normalized_population_maps(accumulator: EndpointAccumulator) -> np.ndarray:
    denominator = accumulator.gain_map_weight[..., None, None]
    return _safe_ratio(accumulator.gain_map_numerator, denominator).astype(np.float32)


def _dose_value(
    rows: pd.DataFrame,
    condition: str,
    scale: float,
    population: str,
    column: str = "exact_ssi_bits",
) -> float:
    selected = rows.loc[
        rows.analysis.str.endswith("_dose_curve")
        & rows.condition.eq(condition)
        & np.isclose(rows.scale, float(scale))
        & rows.population.eq(population),
        column,
    ]
    return float(selected.iloc[0]) if len(selected) == 1 else math.nan


def _contrast_value(rows: pd.DataFrame, condition: str, contrast: str) -> float:
    selected = rows.loc[
        rows.analysis.eq("transport_ablation_contrast")
        & rows.condition.eq(condition)
        & rows.contrast.eq(contrast),
        "exact_ssi_change_b_minus_a_bits",
    ]
    return float(selected.iloc[0]) if len(selected) == 1 else math.nan


def pilot_decision_gate(
    transport_rows: pd.DataFrame,
    realignment_rows: pd.DataFrame,
    *,
    upstream_gate: Mapping[str, Any],
    identity_max_relative_error: float,
    realignment_shift_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Apply the prospective pilot-to-full and no-effect stopping rules."""
    geometry_transport = [
        "candidate_recurrent_center_only",
        "all_recurrent_center_only",
        *sorted(
            value
            for value in transport_rows.condition.dropna().unique()
            if str(value).startswith("all_recurrent_offset_permuted_seed_")
        ),
    ]
    reported_transport = [
        "candidate_recurrent_center_only",
        "gate_recurrent_center_only",
        "all_recurrent_center_only",
        *sorted(
            value
            for value in transport_rows.condition.dropna().unique()
            if str(value).startswith("all_recurrent_offset_permuted_seed_")
        ),
    ]
    contrasts = (
        "lower_sf_sharpening_0_to_2",
        "higher_sf_sharpening_0_to_1",
        "higher_sf_reversal_1_to_3",
    )
    intact_effect = {
        contrast: _contrast_value(transport_rows, "intact", contrast)
        for contrast in contrasts
    }
    disruption_changes = {
        condition: {
            contrast: _contrast_value(transport_rows, condition, contrast)
            - intact_effect[contrast]
            for contrast in contrasts
        }
        for condition in reported_transport
    }
    # SSI is in bits while the replay identity error is dimensionless.  Keep
    # those gates separate instead of constructing an invalid mixed-unit
    # threshold.  A geometry intervention must attenuate every preregistered
    # movement effect toward zero by at least the fixed SSI floor.  This makes
    # the pilot-to-full gate directional: an arbitrary large degradation is
    # not evidence for the proposed transport mechanism.
    numerical_floor = PILOT_MIN_CAUSAL_CHANGE_BITS
    attenuation = {
        condition: {
            contrast: abs(intact_effect[contrast])
            - abs(_contrast_value(transport_rows, condition, contrast))
            for contrast in contrasts
        }
        for condition in geometry_transport
    }
    directional_geometry_conditions = [
        condition
        for condition, values in attenuation.items()
        if all(
            np.isfinite(values[contrast]) and values[contrast] >= numerical_floor
            for contrast in contrasts
        )
    ]
    transport_nonzero = bool(directional_geometry_conditions)

    no_shift_3 = _dose_value(
        realignment_rows, "no_shift_intact", 3.0, "higher_sf_29"
    )
    no_shift_1 = _dose_value(
        realignment_rows, "no_shift_intact", 1.0, "higher_sf_29"
    )
    correct_3 = _dose_value(
        realignment_rows, "eye_correct_candidate_p", 3.0, "higher_sf_29"
    )
    oracle_3 = _dose_value(
        realignment_rows, "activation_oracle_candidate_p", 3.0, "higher_sf_29"
    )
    opposite_3 = _dose_value(
        realignment_rows, "eye_opposite_candidate_p", 3.0, "higher_sf_29"
    )
    random_3 = _dose_value(
        realignment_rows, "eye_random_matched_candidate_p", 3.0, "higher_sf_29"
    )
    q_3 = _dose_value(
        realignment_rows, "eye_correct_complementary_q", 3.0, "higher_sf_29"
    )
    induced_1 = _dose_value(
        realignment_rows, "induced_eye_misalignment_candidate_p", 1.0, "higher_sf_29"
    )
    realignment_values = np.asarray(
        [no_shift_3, no_shift_1, correct_3, oracle_3, opposite_3, random_3, q_3, induced_1]
    )
    realignment_finite = bool(np.isfinite(realignment_values).all())
    rescue_effect = correct_3 - no_shift_3
    oracle_rescue_effect = oracle_3 - no_shift_3
    opposite_control_effect = opposite_3 - no_shift_3
    random_control_effect = random_3 - no_shift_3
    q_control_effect = q_3 - no_shift_3
    induced_impairment = no_shift_1 - induced_1
    wrong_control_median = float(
        np.median([opposite_control_effect, random_control_effect, q_control_effect])
    )
    predicted_pattern = bool(
        realignment_finite
        and rescue_effect >= numerical_floor
        and oracle_rescue_effect >= numerical_floor
        and opposite_control_effect < numerical_floor
        and random_control_effect < numerical_floor
        and rescue_effect - q_control_effect >= numerical_floor
        and induced_impairment >= numerical_floor
    )
    realignment_nonzero = bool(
        realignment_finite
        and rescue_effect >= numerical_floor
        and induced_impairment >= numerical_floor
    )
    oracle_diagnostics_passed = False
    oracle_valid_fraction = math.nan
    oracle_boundary_fraction = math.nan
    oracle_diagnostic_rows = 0
    if realignment_shift_diagnostics is not None:
        diagnostics = realignment_shift_diagnostics.copy()
        required_columns = {
            "condition",
            "scale",
            "internal_step",
            "valid",
            "oracle_at_search_boundary",
        }
        if required_columns.issubset(diagnostics.columns):
            oracle_rows = diagnostics.loc[
                diagnostics.condition.eq("activation_oracle_candidate_p")
                & np.isclose(pd.to_numeric(diagnostics.scale, errors="coerce"), 3.0)
                & pd.to_numeric(diagnostics.internal_step, errors="coerce").gt(0)
            ]
            oracle_diagnostic_rows = int(len(oracle_rows))
            if oracle_diagnostic_rows:
                valid_flags = oracle_rows.valid.astype(str).str.lower().isin(("true", "1"))
                boundary_flags = (
                    oracle_rows.oracle_at_search_boundary.astype(str)
                    .str.lower()
                    .isin(("true", "1"))
                )
                oracle_valid_fraction = float(valid_flags.mean())
                oracle_boundary_fraction = float(boundary_flags.mean())
                oracle_diagnostics_passed = bool(
                    valid_flags.all() and not boundary_flags.any()
                )

    stabilized = transport_rows.loc[
        transport_rows.analysis.eq("transport_ablation_dose_curve")
        & np.isclose(transport_rows.scale, 0.0)
        & transport_rows.population.eq("all_100")
        & transport_rows.condition.isin(directional_geometry_conditions),
        ["condition", "normalized_map_recovery_r2_vs_intact"],
    ].copy()
    stabilized_values = pd.to_numeric(
        stabilized.normalized_map_recovery_r2_vs_intact, errors="coerce"
    ).to_numpy(float)
    stabilized_fidelity = bool(
        len(stabilized_values)
        and np.isfinite(stabilized_values).all()
        and float(np.max(stabilized_values)) >= PILOT_MIN_STABILIZED_GAIN_RECOVERY_R2
    )
    identity_passed = bool(
        np.isfinite(identity_max_relative_error) and identity_max_relative_error <= 3e-3
    )
    upstream_passed = bool(upstream_gate.get("passed", False))
    advance = bool(
        upstream_passed
        and identity_passed
        and stabilized_fidelity
        and transport_nonzero
        and realignment_nonzero
        and predicted_pattern
        and oracle_diagnostics_passed
    )
    reasons: list[str] = []
    for passed, reason in (
        (upstream_passed, "upstream rank-8/PQ/registration gate failed"),
        (identity_passed, "intact replay did not reproduce the frozen endpoint"),
        (stabilized_fidelity, "targeted transport controls destroyed stabilized maps pathologically"),
        (transport_nonzero, "spatial-transport disruption had no effect above the numerical floor"),
        (realignment_nonzero, "candidate realignment/misalignment had no effect above the numerical floor"),
        (predicted_pattern, "pilot realignment controls did not show the preregistered directional pattern"),
        (
            oracle_diagnostics_passed,
            "activation-oracle rescue contains invalid or lag-search-boundary shifts",
        ),
    ):
        if not passed:
            reasons.append(reason)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "advance_to_full" if advance else "stop_after_pilot",
        "advance_to_full": advance,
        "failure_reasons": reasons,
        "numerical_effect_floor_bits": numerical_floor,
        "identity_endpoint_passed": identity_passed,
        "identity_max_relative_error": float(identity_max_relative_error),
        "substantial_stabilized_prediction_retained": stabilized_fidelity,
        "maximum_targeted_stabilized_gain_recovery_r2": (
            float(np.max(stabilized_values)) if len(stabilized_values) else math.nan
        ),
        "transport_disruption_nonzero": transport_nonzero,
        "transport_directional_geometry_conditions": directional_geometry_conditions,
        "transport_intact_contrast_effects": intact_effect,
        "transport_changes_from_intact": disruption_changes,
        "transport_absolute_effect_attenuation_bits": attenuation,
        "realignment_nonzero": realignment_nonzero,
        "realignment_predicted_directional_pattern": predicted_pattern,
        "activation_oracle_diagnostics_passed": oracle_diagnostics_passed,
        "activation_oracle_diagnostic_rows": oracle_diagnostic_rows,
        "activation_oracle_valid_fraction": oracle_valid_fraction,
        "activation_oracle_search_boundary_fraction": oracle_boundary_fraction,
        "realignment_high3_ssi": {
            "no_shift": no_shift_3,
            "eye_correct_p": correct_3,
            "activation_oracle_p": oracle_3,
            "eye_opposite_p": opposite_3,
            "eye_random_matched_p": random_3,
            "eye_correct_q": q_3,
        },
        "realignment_high3_changes_from_no_shift_bits": {
            "eye_correct_p": rescue_effect,
            "activation_oracle_p": oracle_rescue_effect,
            "eye_opposite_p": opposite_control_effect,
            "eye_random_matched_p": random_control_effect,
            "eye_correct_q": q_control_effect,
            "wrong_control_median": wrong_control_median,
        },
        "induced_failure_high1_ssi": {
            "no_shift": no_shift_1,
            "induced_p_misalignment": induced_1,
            "impairment_bits": induced_impairment,
        },
        "upstream_gate": dict(upstream_gate),
        "stopping_rule": (
            "Do not run 8x24 unless upstream gates pass, intact replay is valid, a targeted "
            "control retains substantial stabilized-map fidelity, transport disruption and "
            "realignment exceed the numerical floor, correct P realignment beats no/wrong "
            "controls individually, opposite/random controls do not materially rescue, the oracle "
            "is positive and not clipped by the lag-search boundary, and induced P misalignment "
            "materially impairs 1x sharpening."
        ),
    }
