from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.equations import (
    complementary_native,
    instrument_convgru_step,
    project_native,
    replay_convgru_cell,
    split_conv2d_contributions,
    term_energy_summary,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.kernels import (
    pq_offset_energies,
    recurrent_kernel_half,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.pilot_selection import (
    stratified_medoids,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.registration import (
    FIG4_CONVGRU_INPUT_LAG_SUPPORTS,
    apply_shift_calibration,
    derive_fig4_convgru_input_lag_supports,
    fit_shift_calibration,
    internal_step_eye_displacements,
    normalized_multichannel_xcorr,
    registration_step_metrics,
    support_midpoint_lags,
    vector_transport_statistics,
)


def coordinate_basis(channels: int = 8, rank: int = 2) -> torch.Tensor:
    return torch.eye(channels, rank)


class TinyConvGRUCell(torch.nn.Module):
    """Dependency-light replica of the exact source convention for unit tests."""

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        channels = input_size + hidden_size
        padding = kernel_size // 2
        self.update_gate = torch.nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
        self.reset_gate = torch.nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
        self.out_gate = torch.nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
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


def test_literal_gru_equations_use_candidate_write_gate_and_reconstruct() -> None:
    torch.manual_seed(3)
    cell = TinyConvGRUCell(5, 8, 3).eval()
    x = torch.randn(2, 5, 9, 11)
    h = torch.randn(2, 8, 9, 11)
    terms = instrument_convgru_step(cell, x, h, verify=True)
    torch.testing.assert_close(
        terms.h_t,
        (1.0 - terms.update_gate) * h
        + terms.update_gate * terms.candidate_state,
    )
    torch.testing.assert_close(
        terms.candidate_preactivation,
        terms.candidate_current_preactivation
        + terms.candidate_recurrent_preactivation,
        atol=2e-6,
        rtol=2e-6,
    )
    assert terms.x_t.shape[1] == 5
    assert all(value.shape[1] == 8 for value in terms.hidden_space_terms().values())


def test_split_keeps_direct_recurrent_kernel_and_exact_literal_residual() -> None:
    torch.manual_seed(31)
    layer = torch.nn.Conv2d(13, 8, 3, padding=1).eval()
    x = torch.randn(2, 5, 17, 19)
    hidden = torch.randn(2, 8, 17, 19)
    current, recurrent, literal = split_conv2d_contributions(layer, x, hidden)
    expected_recurrent = F.conv2d(
        hidden,
        layer.weight[:, 5:],
        None,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    torch.testing.assert_close(recurrent, expected_recurrent, atol=0, rtol=0)
    torch.testing.assert_close(current + recurrent, literal, atol=2e-6, rtol=2e-6)


def test_replay_matches_literal_recurrence_and_zero_initial_state() -> None:
    torch.manual_seed(4)
    cell = TinyConvGRUCell(3, 6, 3).eval()
    sequence = torch.randn(2, 3, 4, 7, 7)
    terms = replay_convgru_cell(cell, sequence, verify=True)
    assert len(terms) == 4
    assert torch.count_nonzero(terms[0].h_previous) == 0
    torch.testing.assert_close(terms[1].h_previous, terms[0].h_t)


def test_projected_term_energy_is_exact_pythagorean_split() -> None:
    torch.manual_seed(5)
    cell = TinyConvGRUCell(3, 8, 3).eval()
    terms = instrument_convgru_step(
        cell, torch.randn(2, 3, 5, 5), torch.randn(2, 8, 5, 5)
    )
    basis = coordinate_basis()
    summary = term_energy_summary(terms, basis, names=("h_t",))
    torch.testing.assert_close(
        summary["h_t__total_energy"],
        summary["h_t__p_energy"] + summary["h_t__q_energy"],
    )
    torch.testing.assert_close(
        project_native(terms.h_t, basis) + complementary_native(terms.h_t, basis),
        terms.h_t,
    )


def test_projected_term_energy_includes_exact_named_readout_svd_view() -> None:
    torch.manual_seed(51)
    cell = TinyConvGRUCell(3, 8, 3).eval()
    terms = instrument_convgru_step(
        cell, torch.randn(2, 3, 5, 5), torch.randn(2, 8, 5, 5)
    )
    learned = coordinate_basis(channels=8, rank=2)
    readout_svd = torch.eye(8)[:, 2:5]
    summary = term_energy_summary(
        terms,
        learned,
        names=("h_t",),
        comparison_bases={"readout_svd": readout_svd},
    )
    expected = torch.einsum(
        "ck,bcyx->bkyx", readout_svd, terms.h_t
    ).square().sum(dim=(1, 2, 3))
    torch.testing.assert_close(summary["h_t__readout_svd_energy"], expected)
    torch.testing.assert_close(
        summary["h_t__readout_svd_energy_per_dimension"], expected / 3.0
    )


def _translated_map(dy: float, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    height = width = 33
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    base = (
        torch.exp(-((x - 10.3) ** 2 + (y - 15.7) ** 2) / 16.0)
        + 0.65 * torch.exp(-((x - 23.2) ** 2 + (y - 8.4) ** 2) / 8.0)
    )[None, None]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
    )
    grid = torch.stack(
        [xx - 2.0 * dx / (width - 1), yy - 2.0 * dy / (height - 1)], dim=-1
    )[None]
    shifted = F.grid_sample(base, grid, align_corners=True, padding_mode="zeros")
    return base, shifted


def test_fourier_xcorr_recovers_content_displacement_with_subpixel_sign() -> None:
    a, b = _translated_map(dy=-0.75, dx=1.5)
    peak = normalized_multichannel_xcorr(a, b, max_lag_px=4)
    assert bool(peak.valid[0])
    assert float(peak.lag_x_px[0]) == pytest.approx(1.5, abs=0.22)
    assert float(peak.lag_y_px[0]) == pytest.approx(-0.75, abs=0.22)
    assert float(peak.best_lag_correlation[0]) > float(peak.zero_lag_correlation[0])


def test_xcorr_rejects_zero_variance_maps() -> None:
    peak = normalized_multichannel_xcorr(
        torch.ones(2, 3, 9, 9), torch.ones(2, 3, 9, 9), max_lag_px=3
    )
    assert not bool(peak.valid.any())
    assert torch.isnan(peak.best_lag_correlation).all()


def test_grouped_registration_reduction_preserves_namespaced_views() -> None:
    a, b = _translated_map(dy=1.0, dx=-1.0)
    current = a.repeat(2, 8, 1, 1)
    previous = b.repeat(2, 8, 1, 1)
    recurrent = a.repeat(2, 8, 1, 1)
    basis = coordinate_basis()
    result = registration_step_metrics(
        current,
        previous,
        recurrent,
        {
            "low::learned_p": basis,
            "low::learned_q": basis,
            "low::random_00": basis,
        },
        max_lag_px=3,
    )
    assert set(result) == {"low::learned_p", "low::learned_q", "low::random_00"}
    for row in result.values():
        assert row["transport_x_px"].shape == (2,)
        assert torch.isfinite(row["recurrent_best_lag_correlation"]).all()
    # Previous content is at (-1,+1) relative to current; the corrective
    # content transport is the opposite, (+1,-1).
    assert float(result["low::learned_p"]["transport_x_px"][0]) == pytest.approx(1.0, abs=0.26)
    assert float(result["low::learned_p"]["transport_y_px"][0]) == pytest.approx(-1.0, abs=0.26)


def test_shift_calibration_recovers_matrix_and_transport_statistics() -> None:
    rng = np.random.default_rng(10)
    eye = rng.normal(size=(100, 2))
    matrix = np.asarray([[1.5, -0.2], [0.1, -1.2]])
    feature = apply_shift_calibration(eye, matrix) + np.asarray([0.03, -0.05])
    fit = fit_shift_calibration(eye, feature)
    np.testing.assert_allclose(fit["matrix_feature_px_per_eye_deg"], matrix, atol=1e-10)
    stats = vector_transport_statistics(feature - np.asarray([0.03, -0.05]), eye @ matrix.T)
    assert stats["vector_correlation"] == pytest.approx(1.0)
    assert stats["median_displacement_error_px"] == pytest.approx(0.0, abs=1e-12)
    assert stats["vector_variance_explained"] == pytest.approx(1.0)


def test_internal_time_support_is_eight_steps_newer_to_older() -> None:
    assert FIG4_CONVGRU_INPUT_LAG_SUPPORTS == (
        (0, 17), (0, 19), (0, 21), (0, 23),
        (2, 25), (4, 27), (6, 29), (8, 31),
    )
    np.testing.assert_allclose(support_midpoint_lags(), [8.5, 9.5, 10.5, 11.5, 13.5, 15.5, 17.5, 19.5])
    history = np.column_stack([np.arange(71), np.zeros(71)])
    displacement = internal_step_eye_displacements(history, scored_frame=0)
    assert np.isnan(displacement[0]).all()
    np.testing.assert_allclose(displacement[1:, 0], [-1, -1, -1, -2, -2, -2, -2])


def test_internal_supports_are_derived_from_executed_temporal_graph() -> None:
    config = {
        "frontend": {"params": {"kernel_size": 16}},
        "convnet": {
            "params": {
                "block_configs": [
                    {
                        "conv_params": {"kernel_size": [3, 9, 9]},
                        "pool_params": {"kernel_size": 2, "stride": 2},
                    },
                    {
                        "conv_params": {"kernel_size": [3, 5, 5]},
                        "pool_params": None,
                    },
                ]
            }
        },
        "recurrent": {"params": {"n_layers": 1, "hidden_dim": 128}},
    }
    assert derive_fig4_convgru_input_lag_supports(config) == FIG4_CONVGRU_INPUT_LAG_SUPPORTS


def test_kernel_pq_energy_matches_explicit_projectors() -> None:
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(8, 8))
    basis = np.eye(8, 2)
    p = basis @ basis.T
    q = np.eye(8) - p
    result = pq_offset_energies(kernel, basis)
    assert result["energy_pp"] == pytest.approx(np.square(p @ kernel @ p).sum())
    assert result["energy_pq"] == pytest.approx(np.square(p @ kernel @ q).sum())
    assert result["energy_qp"] == pytest.approx(np.square(q @ kernel @ p).sum())
    assert result["energy_qq"] == pytest.approx(np.square(q @ kernel @ q).sum())
    assert result["decomposition_residual"] == pytest.approx(0.0, abs=1e-10)


def test_hidden_kernel_split_is_last_128_serialized_channels() -> None:
    weight = np.zeros((128, 384, 3, 3), dtype=np.float32)
    weight[:, 256:] = 4.0
    recurrent = recurrent_kernel_half(weight)
    assert recurrent.shape == (128, 128, 3, 3)
    assert np.all(recurrent == 4.0)


def test_pilot_strata_are_equal_count_outcome_blind_and_deterministic() -> None:
    values = np.asarray([7, 0, 3, 2, 1, 6, 4, 5], dtype=float)
    stable_ids = np.asarray([70, 0, 30, 20, 10, 60, 40, 50])
    first = stratified_medoids(values, stable_ids, n_strata=4)
    second = stratified_medoids(values, stable_ids, n_strata=4)
    assert first == second
    assert len(first) == 4
    assert [row["stable_id"] for row in first] == [0, 20, 40, 60]
