from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.analysis import (
    EndpointAccumulator,
    accumulator_rows,
    normalized_population_maps,
    pilot_decision_gate,
    population_definitions,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.mechanics import (
    TRANSPORT_INTERVENTIONS,
    KernelIntervention,
    fourier_shift_2d,
    matched_random_directions,
    oracle_candidate_shift_yx,
    realignment_step,
    recurrent_kernel_invariants,
    replay_realignment,
    replay_transport_intervention,
    transform_recurrent_kernel,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.provenance import (
    valid_complete_marker,
    write_complete_marker,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.selection import (
    selection_for_scope,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions import (
    run_interventions as intervention_runner,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.equations import (
    project_native,
    split_conv2d_contributions,
)


class TinyConvGRUCell(torch.nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 6) -> None:
        super().__init__()
        channels = input_size + hidden_size
        self.update_gate = torch.nn.Conv2d(channels, hidden_size, 3, padding=1)
        self.reset_gate = torch.nn.Conv2d(channels, hidden_size, 3, padding=1)
        self.out_gate = torch.nn.Conv2d(channels, hidden_size, 3, padding=1)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                x.shape[0], self.hidden_size, x.shape[-2], x.shape[-1],
                dtype=x.dtype, device=x.device,
            )
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.update_gate(xh))
        r = torch.sigmoid(self.reset_gate(xh))
        n = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


def _basis(channels: int = 6, rank: int = 2) -> torch.Tensor:
    return torch.eye(channels, rank)


def _condition(name: str) -> KernelIntervention:
    return next(value for value in TRANSPORT_INTERVENTIONS if value.name == name)


def test_registry_contains_every_preregistered_transport_condition() -> None:
    names = [value.name for value in TRANSPORT_INTERVENTIONS]
    assert names[:4] == [
        "intact",
        "candidate_recurrent_center_only",
        "gate_recurrent_center_only",
        "all_recurrent_center_only",
    ]
    assert len([value for value in names if "offset_permuted" in value]) == 3
    assert names[-1] == "no_recurrence_reference"


def test_offset_permutation_preserves_center_matrices_and_norm() -> None:
    torch.manual_seed(1)
    weight = torch.randn(7, 6, 3, 3)
    first = transform_recurrent_kernel(weight, "permute_offcenter", seed=20260913)
    second = transform_recurrent_kernel(weight, "permute_offcenter", seed=20260913)
    assert torch.equal(first, second)
    assert not torch.equal(first, weight)
    audit = recurrent_kernel_invariants(weight, first)
    assert audit["center_max_abs"] == 0.0
    assert audit["total_norm_relative_error"] <= 1e-7
    assert audit["offcenter_tap_norm_multiset_equal"] is True


def test_intact_transport_replay_is_literal_and_no_recurrence_resets_each_step() -> None:
    torch.manual_seed(2)
    cell = TinyConvGRUCell().eval()
    sequence = torch.randn(2, 3, 4, 9, 9)
    intact = replay_transport_intervention(cell, sequence, _condition("intact"))
    state = None
    literal = []
    for step in range(sequence.shape[2]):
        state = cell(sequence[:, :, step], state)
        literal.append(state)
    torch.testing.assert_close(intact, torch.stack(literal, dim=2), atol=0, rtol=0)

    no_recurrence = replay_transport_intervention(
        cell, sequence, _condition("no_recurrence_reference")
    )
    independent = torch.stack(
        [cell(sequence[:, :, step], None) for step in range(sequence.shape[2])], dim=2
    )
    torch.testing.assert_close(no_recurrence, independent, atol=0, rtol=0)


def test_center_only_is_identity_when_all_recurrent_offcenter_taps_are_zero() -> None:
    torch.manual_seed(3)
    cell = TinyConvGRUCell().eval()
    with torch.no_grad():
        for layer in (cell.update_gate, cell.reset_gate, cell.out_gate):
            hidden = layer.weight[:, -cell.hidden_size :]
            hidden[..., 0, :] = 0
            hidden[..., 2, :] = 0
            hidden[..., 1, 0] = 0
            hidden[..., 1, 2] = 0
    sequence = torch.randn(2, 3, 3, 7, 7)
    intact = replay_transport_intervention(cell, sequence, _condition("intact"))
    center = replay_transport_intervention(
        cell, sequence, _condition("all_recurrent_center_only")
    )
    torch.testing.assert_close(center, intact, atol=0, rtol=0)


def test_fourier_shift_has_declared_sign_norm_and_exact_zero_endpoint() -> None:
    value = torch.zeros(2, 3, 11, 13)
    value[0, :, 4, 5] = torch.tensor([1.0, 2.0, 3.0])
    value[1] = torch.randn(3, 11, 13)
    shifted = fourier_shift_2d(value, torch.tensor([[2.0, -3.0], [0.0, 0.0]]))
    assert torch.argmax(shifted[0, 0]).item() == (6 * 13 + 2)
    torch.testing.assert_close(
        shifted.square().sum((-2, -1)), value.square().sum((-2, -1)), atol=2e-6, rtol=2e-6
    )
    assert torch.equal(shifted[1], value[1])


def test_no_shift_and_zero_p_shift_reproduce_literal_endpoint_exactly() -> None:
    torch.manual_seed(4)
    cell = TinyConvGRUCell().eval()
    sequence = torch.randn(2, 3, 4, 8, 8)
    literal = replay_transport_intervention(cell, sequence, _condition("intact"))
    none = replay_realignment(
        cell, sequence, _basis(), shift_yx_by_step=None, target="none"
    )
    zero = replay_realignment(
        cell,
        sequence,
        _basis(),
        shift_yx_by_step=torch.zeros(2, 4, 2),
        target="candidate_p",
    )
    torch.testing.assert_close(none.sequence, literal, atol=0, rtol=0)
    torch.testing.assert_close(zero.sequence, literal, atol=0, rtol=0)
    assert torch.count_nonzero(zero.applied_shift_yx_px[:, 0]) == 0


def test_candidate_intervention_changes_only_projected_recurrent_preactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5)
    cell = TinyConvGRUCell().eval()
    x = torch.randn(2, 3, 7, 7)
    h = torch.randn(2, 6, 7, 7)
    basis = _basis()

    def double(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        del shift
        return 2.0 * value

    monkeypatch.setattr(
        "paper.fig4.mechanism_audit_v1.registration_mechanism."
        "causal_interventions.mechanics.fourier_shift_2d",
        double,
    )
    result, _, _ = realignment_step(
        cell,
        x,
        h,
        basis,
        shift_yx_px=torch.ones(2, 2),
        target="candidate_p",
    )
    _, _, z_literal = split_conv2d_contributions(cell.update_gate, x, h)
    _, _, r_literal = split_conv2d_contributions(cell.reset_gate, x, h)
    z = torch.sigmoid(z_literal)
    r = torch.sigmoid(r_literal)
    _, recurrent, candidate_literal = split_conv2d_contributions(cell.out_gate, x, r * h)
    candidate_expected = torch.tanh(candidate_literal + project_native(recurrent, basis))
    expected = (1.0 - z) * h + z * candidate_expected
    torch.testing.assert_close(result, expected)


def test_activation_oracle_returns_corrective_content_shift() -> None:
    torch.manual_seed(6)
    current = torch.randn(1, 6, 25, 25)
    recurrent = torch.roll(current, shifts=(1, -2), dims=(-2, -1))
    shift, valid = oracle_candidate_shift_yx(
        current, recurrent, torch.eye(6), max_lag_px=4
    )
    assert bool(valid[0])
    np.testing.assert_allclose(shift[0].numpy(), [-1.0, 2.0], atol=0.26)


def test_oracle_replay_bypasses_shift_at_zero_state_step() -> None:
    torch.manual_seed(61)
    cell = TinyConvGRUCell().eval()
    sequence = torch.randn(2, 3, 3, 15, 15)
    replay = replay_realignment(
        cell,
        sequence,
        _basis(),
        shift_yx_by_step=None,
        target="candidate_p",
        oracle=True,
        max_oracle_lag_px=3,
    )
    torch.testing.assert_close(replay.sequence[:, :, 0], cell(sequence[:, :, 0], None))
    assert torch.count_nonzero(replay.applied_shift_yx_px[:, 0]) == 0
    assert bool(replay.oracle_valid[:, 0].all())


def test_random_direction_control_is_deterministic_and_magnitude_matched() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    labels = ["a", "b", "c"]
    first = matched_random_directions(values, labels=labels)
    second = matched_random_directions(values, labels=labels)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.linalg.norm(values, axis=1))


def test_frozen_pilot_is_exact_4x12_and_p_jobs_are_fold_heldout() -> None:
    pilot = selection_for_scope("pilot")
    assert pilot.image_positions == (1, 5, 7, 4)
    assert pilot.trajectory_positions == (1, 3, 4, 6, 8, 11, 12, 14, 16, 18, 20, 22)
    assert len(pilot.transport_pairs) == 48
    assert {fold: sum(row[0] == fold for row in pilot.heldout_realign_pairs) for fold in range(4)} == {
        0: 4,
        1: 6,
        2: 0,
        3: 2,
    }

    from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold

    for fold, image, trajectory in pilot.heldout_realign_pairs:
        assert (image, trajectory) in set(load_fold(fold).test.pairs)


def test_endpoint_accumulator_preserves_exact_ssi_and_complete_maps() -> None:
    populations = population_definitions(np.arange(71), np.arange(71, 100))
    accumulator = EndpointAccumulator(("intact",), (0.0,), populations)
    y, x = torch.meshgrid(torch.arange(51), torch.arange(51), indexing="ij")
    spatial = (1.0 + 0.1 * torch.cos(2 * torch.pi * x / 51))[None, None]
    rate = spatial.expand(2, 100, 51, 51).clone()
    preactivation = torch.log(torch.expm1(rate))
    accumulator.add(
        0,
        0,
        preactivation=preactivation,
        rate=rate,
        intact_preactivation=preactivation,
        intact_rate=rate,
    )
    rows = accumulator_rows(accumulator, analysis="transport_ablation", scope="pilot")
    dose = rows.loc[
        rows.analysis.eq("transport_ablation_dose_curve")
        & rows.population.eq("all_100")
    ].iloc[0]
    gain = rate.double() / rate.double().mean((-2, -1), keepdim=True)
    expected_ssi = float((gain * torch.log2(gain + 1e-8)).mean((-2, -1)).mean())
    assert dose.exact_ssi_bits == pytest.approx(expected_ssi)
    assert dose.normalized_map_recovery_r2_vs_intact == pytest.approx(1.0)
    maps = normalized_population_maps(accumulator)
    assert maps.shape == (1, 1, 3, 51, 51)
    np.testing.assert_allclose(maps[0, 0, 0], gain[0, 0].numpy(), atol=1e-6)


def test_contrast_rows_report_complete_normalized_population_map_transformations() -> None:
    populations = population_definitions(np.arange(71), np.arange(71, 100))
    accumulator = EndpointAccumulator(("intact", "ablated"), (0.0, 2.0), populations)
    y, x = torch.meshgrid(torch.arange(51), torch.arange(51), indexing="ij")
    baseline = torch.ones(1, 100, 51, 51)
    modulation = (0.1 * torch.cos(2 * torch.pi * x / 51))[None, None]
    target = baseline + modulation
    halfway = baseline + 0.5 * modulation
    for condition_index, endpoints in enumerate(((baseline, target), (baseline, halfway))):
        for scale_index, rate in enumerate(endpoints):
            rate = rate.expand(1, 100, 51, 51).clone()
            accumulator.add(
                condition_index,
                scale_index,
                preactivation=torch.log(torch.expm1(rate)),
                rate=rate,
                intact_preactivation=torch.log(torch.expm1(rate)),
                intact_rate=rate,
            )
    rows = accumulator_rows(accumulator, analysis="transport_ablation", scope="pilot")
    contrast = rows.loc[
        rows.analysis.eq("transport_ablation_contrast")
        & rows.condition.eq("ablated")
        & rows.contrast.eq("lower_sf_sharpening_0_to_2")
    ].iloc[0]
    # A half-amplitude transformation leaves one quarter of the intact
    # squared-error denominator, hence R2=0.75.
    assert contrast.normalized_population_map_effect_recovery_r2_vs_intact == pytest.approx(
        0.75, abs=2e-5
    )
    assert contrast.normalized_population_map_effect_intact_rms > 0
    assert contrast.normalized_population_map_effect_observed_rms > 0


def test_realignment_contrast_uses_common_intact_one_x_endpoint_and_reports_map_rescue() -> None:
    populations = population_definitions(np.arange(71), np.arange(71, 100))
    accumulator = EndpointAccumulator(
        ("no_shift_intact", "eye_correct_candidate_p"), (1.0, 3.0), populations
    )
    y, x = torch.meshgrid(torch.arange(51), torch.arange(51), indexing="ij")
    one_x = 1.0 + (0.1 * torch.cos(2 * torch.pi * x / 51))[None, None]
    three_x = 1.0 + (0.1 * torch.sin(2 * torch.pi * y / 51))[None, None]
    for scale_index, rate in enumerate((one_x, three_x)):
        rate = rate.expand(1, 100, 51, 51).clone()
        accumulator.add(
            0,
            scale_index,
            preactivation=torch.log(torch.expm1(rate)),
            rate=rate,
            intact_preactivation=torch.log(torch.expm1(rate)),
            intact_rate=rate,
        )
    rescued = one_x.expand(1, 100, 51, 51).clone()
    intact_three = three_x.expand(1, 100, 51, 51).clone()
    accumulator.add(
        1,
        1,
        preactivation=torch.log(torch.expm1(rescued)),
        rate=rescued,
        intact_preactivation=torch.log(torch.expm1(intact_three)),
        intact_rate=intact_three,
    )
    rows = accumulator_rows(accumulator, analysis="realignment", scope="pilot")
    contrast = rows.loc[
        rows.analysis.eq("realignment_contrast")
        & rows.condition.eq("eye_correct_candidate_p")
        & rows.contrast.eq("higher_sf_reversal_1_to_3")
    ].iloc[0]
    assert contrast.scale_a_endpoint_condition == "no_shift_intact"
    assert contrast.scale_b_endpoint_condition == "eye_correct_candidate_p"
    assert contrast.exact_ssi_change_b_minus_a_bits == pytest.approx(0.0, abs=1e-8)
    assert contrast.normalized_population_map_rescue_fraction_to_intact_scale_a == pytest.approx(
        1.0, abs=1e-7
    )


def _pilot_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = {
        "lower_sf_sharpening_0_to_2": 0.012,
        "higher_sf_sharpening_0_to_1": 0.010,
        "higher_sf_reversal_1_to_3": -0.011,
    }
    transport_conditions = [
        "intact",
        "candidate_recurrent_center_only",
        "gate_recurrent_center_only",
        "all_recurrent_center_only",
        "all_recurrent_offset_permuted_seed_20260913",
    ]
    transport: list[dict[str, object]] = []
    for condition in transport_conditions:
        transport.append(
            {
                "analysis": "transport_ablation_dose_curve",
                "condition": condition,
                "scale": 0.0,
                "population": "all_100",
                "normalized_map_recovery_r2_vs_intact": 1.0 if condition == "intact" else 0.8,
            }
        )
        for contrast, effect in contrasts.items():
            value = effect
            if condition in {"candidate_recurrent_center_only", "all_recurrent_center_only"}:
                value = np.sign(effect) * max(abs(effect) - 0.004, 0.0)
            transport.append(
                {
                    "analysis": "transport_ablation_contrast",
                    "condition": condition,
                    "contrast": contrast,
                    "exact_ssi_change_b_minus_a_bits": value,
                }
            )
    realignment = pd.DataFrame(
        [
            ("no_shift_intact", 3.0, 0.020),
            ("no_shift_intact", 1.0, 0.030),
            ("eye_correct_candidate_p", 3.0, 0.025),
            ("activation_oracle_candidate_p", 3.0, 0.027),
            ("eye_opposite_candidate_p", 3.0, 0.019),
            ("eye_random_matched_candidate_p", 3.0, 0.020),
            ("eye_correct_complementary_q", 3.0, 0.021),
            ("induced_eye_misalignment_candidate_p", 1.0, 0.025),
        ],
        columns=("condition", "scale", "exact_ssi_bits"),
    )
    realignment["analysis"] = "realignment_dose_curve"
    realignment["population"] = "higher_sf_29"
    return pd.DataFrame(transport), realignment


def _oracle_diagnostics(*, at_boundary: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["activation_oracle_candidate_p"] * 3,
            "scale": [3.0] * 3,
            "internal_step": [1, 2, 3],
            "valid": [True] * 3,
            "oracle_at_search_boundary": [at_boundary] * 3,
        }
    )


def test_pilot_gate_advances_only_for_directional_nonpathological_pattern() -> None:
    transport, realignment = _pilot_tables()
    gate = pilot_decision_gate(
        transport,
        realignment,
        upstream_gate={"passed": True},
        identity_max_relative_error=1e-4,
        realignment_shift_diagnostics=_oracle_diagnostics(),
    )
    assert gate["advance_to_full"] is True
    assert "candidate_recurrent_center_only" in gate["transport_directional_geometry_conditions"]
    assert gate["numerical_effect_floor_bits"] == pytest.approx(0.002)

    failed = realignment.copy()
    failed.loc[failed.condition.eq("activation_oracle_candidate_p"), "exact_ssi_bits"] = 0.020
    stopped = pilot_decision_gate(
        transport,
        failed,
        upstream_gate={"passed": True},
        identity_max_relative_error=1e-4,
        realignment_shift_diagnostics=_oracle_diagnostics(),
    )
    assert stopped["advance_to_full"] is False
    assert stopped["status"] == "stop_after_pilot"

    wrong_direction_rescues = realignment.copy()
    wrong_direction_rescues.loc[
        wrong_direction_rescues.condition.eq("eye_opposite_candidate_p"), "exact_ssi_bits"
    ] = 0.024
    stopped = pilot_decision_gate(
        transport,
        wrong_direction_rescues,
        upstream_gate={"passed": True},
        identity_max_relative_error=1e-4,
        realignment_shift_diagnostics=_oracle_diagnostics(),
    )
    assert stopped["advance_to_full"] is False

    subthreshold_primary = realignment.copy()
    subthreshold_primary.loc[
        subthreshold_primary.condition.eq("eye_correct_candidate_p"), "exact_ssi_bits"
    ] = 0.021
    stopped = pilot_decision_gate(
        transport,
        subthreshold_primary,
        upstream_gate={"passed": True},
        identity_max_relative_error=1e-4,
        realignment_shift_diagnostics=_oracle_diagnostics(),
    )
    assert stopped["advance_to_full"] is False

    boundary_oracle = pilot_decision_gate(
        transport,
        realignment,
        upstream_gate={"passed": True},
        identity_max_relative_error=1e-4,
        realignment_shift_diagnostics=_oracle_diagnostics(at_boundary=True),
    )
    assert boundary_oracle["advance_to_full"] is False
    assert boundary_oracle["activation_oracle_diagnostics_passed"] is False


def test_upstream_execution_gate_records_but_waives_negative_pq_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intervention_runner, "_rank8_gate", lambda: {"passed": True})
    monkeypatch.setattr(
        intervention_runner,
        "_pq_motion_gate",
        lambda: {"passed": False, "median_p_to_q_per_dimension": {"high_0_to_1": 0.97}},
    )
    monkeypatch.setattr(
        intervention_runner,
        "_registration_gate",
        lambda: {
            "passed": True,
            "execution_readiness_passed": True,
            "alignment_hypothesis_supported": False,
        },
    )
    monkeypatch.setattr(intervention_runner, "_calibration_gate", lambda: {"passed": True})
    gate = intervention_runner.upstream_mechanism_gate()
    assert gate["passed"] is True
    assert gate["p_motion_enrichment"]["passed"] is False
    assert gate["p_motion_enrichment"]["execution_gate_applied"] is False
    assert gate["recurrent_alignment"]["alignment_hypothesis_supported"] is False
    assert gate["recurrent_alignment"]["positive_alignment_execution_gate_applied"] is False


def test_registration_execution_readiness_requires_complete_folds_not_positive_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "registration.csv"
    manifest_path = tmp_path / "manifest.json"
    parts_path = tmp_path / "parts"
    parts_path.mkdir()
    rows = []
    defining = {"low_0_to_2": 2.0, "high_0_to_1": 1.0, "high_1_to_3": 3.0}
    for contrast, scale in defining.items():
        for fold in range(4):
            rows.append(
                {
                    "contrast": contrast,
                    "subspace": "learned_p",
                    "scale": scale,
                    "fold": fold,
                    "valid": True,
                    "raw_at_search_boundary": False,
                    "recurrent_at_search_boundary": False,
                    "expected_outside_search_window": False,
                    "zero_lag_alignment_improvement": -0.1,
                }
            )
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    manifest_path.write_text(json.dumps({"registration_parts": 48, "term_parts": 48}))
    for row_index, (fold, image, trajectory) in enumerate(
        selection_for_scope("full").heldout_realign_pairs
    ):
        (parts_path / f"heldout__row_{row_index}__complete.json").write_text(
            json.dumps(
                {
                    "schema": intervention_runner.INSTRUMENTATION_SCHEMA,
                    "scope": "heldout",
                    "fold": fold,
                    "pair": [image, trajectory],
                }
            )
        )
    monkeypatch.setattr(intervention_runner, "REGISTRATION_METRICS", metrics_path)
    monkeypatch.setattr(intervention_runner, "REGISTRATION_MANIFEST", manifest_path)
    monkeypatch.setattr(intervention_runner, "INSTRUMENTATION_PARTS", parts_path)
    gate = intervention_runner._registration_gate()
    assert gate["execution_readiness_passed"] is True
    assert gate["passed"] is True
    assert gate["alignment_hypothesis_supported"] is False


def test_complete_marker_is_hash_and_fingerprint_fail_closed(tmp_path: Path) -> None:
    product = tmp_path / "arrays.npz"
    product.write_bytes(b"first")
    fingerprint = {"checkpoint": "abc"}
    write_complete_marker(
        tmp_path,
        schema_version="schema",
        stage="transport",
        scope="pilot",
        identity={"image_position": 1, "trajectory_position": 2},
        input_fingerprint=fingerprint,
        products=(product,),
        diagnostics={"identity": 0.0},
    )
    arguments = {
        "schema_version": "schema",
        "stage": "transport",
        "scope": "pilot",
        "identity": {"image_position": 1, "trajectory_position": 2},
        "input_fingerprint": fingerprint,
        "required_product_names": ("arrays.npz",),
    }
    assert valid_complete_marker(tmp_path, **arguments) is not None
    assert valid_complete_marker(
        tmp_path, **{**arguments, "input_fingerprint": {"checkpoint": "changed"}}
    ) is None
    product.write_bytes(b"changed")
    assert valid_complete_marker(tmp_path, **arguments) is None


def test_production_source_uses_fold_bases_shared_ledger_and_no_consensus_file() -> None:
    path = Path(
        "paper/fig4/mechanism_audit_v1/registration_mechanism/"
        "causal_interventions/run_interventions.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "fit_directory(contrast, fold)" in source
    assert "exclusive_gpu_lock()" in source
    assert "budget_deadline()" in source
    assert "record_gpu_time(" in source
    assert "consensus_projectors.npz" not in source
    assert 'REGISTRATION_METRICS = BASE_OUT / "registration_metrics_heldout.csv"' in source
    assert "derive_fig4_convgru_input_lag_supports" in source
