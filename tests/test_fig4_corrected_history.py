"""Regression tests for the corrected Figure 4 causal-history convention."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    N_LAGS,
    N_PRECEDING,
    N_SCORED,
    SCALE_FACTORS,
    lag_windows_from_sequence,
    validate_monotone_causal_windows,
)
from paper.fig4.mechanism_audit_v1.correction.run_corrected_controlled_scaling import (
    scaled_histories,
)
from paper.fig4.upstream.real_trace_matrix.model import (
    _LegacyStaticStimulusContract,
    _embed_time_lags,
    _expand_trace_to_model_grid,
    _trace_on_output_grid,
    RealTraceMatrixScorer,
    infer_model_time_contract,
    make_counterfactual_stim,
)
from paper.fig4.upstream.run_selected_twin_frequency_tuning_probe import (
    _declared_time_contract,
    _output_phase_for_prefixed_movie,
)
from paper.fig4.upstream.run_selected_twin_instantaneous_unit_maps import (
    _trace_on_selected_output_grid,
)


def production_windows(sequence: np.ndarray) -> np.ndarray:
    movie = torch.from_numpy(sequence.astype(np.float32)[:, None, None])
    return _embed_time_lags(movie, n_lags=N_LAGS, torch=torch)[:, 0, :, 0, 0].numpy().astype(int)


def test_corrected_history_is_t_minus_31_through_t() -> None:
    sequence = np.arange(-N_PRECEDING, N_SCORED)
    windows = production_windows(sequence)
    assert windows.shape == (40, 32)
    assert np.array_equal(windows[0], np.arange(0, -32, -1))
    assert np.array_equal(windows[-1], np.arange(39, 7, -1))
    assert np.array_equal(windows, lag_windows_from_sequence(sequence))


def test_corrected_history_has_no_future_or_wrap() -> None:
    sequence = np.arange(-N_PRECEDING, N_SCORED)
    validation = validate_monotone_causal_windows(production_windows(sequence), np.arange(N_SCORED))
    assert validation == {
        "total_outputs": 40,
        "outputs_with_future_samples": 0,
        "outputs_with_nonmonotonic_indices": 0,
        "outputs_with_nonunit_index_steps": 0,
    }


def test_legacy_prefix_exposes_31_retained_outputs_to_future_samples() -> None:
    sequence = np.concatenate((np.arange(N_LAGS), np.arange(N_SCORED)))
    retained = production_windows(sequence)[1:]
    output_times = np.arange(N_SCORED)
    assert np.sum(np.any(retained > output_times[:, None], axis=1)) == 31
    assert np.sum(np.any(np.diff(retained[:, ::-1], axis=1) < 0, axis=1)) == 31


def test_true_history_controlled_scaling_preserves_prefix_and_scales_only_score() -> None:
    base = np.arange((N_PRECEDING + N_SCORED) * 2, dtype=np.float32).reshape(1, -1, 2) / 100.0
    scaled = scaled_histories(base, bank_index=0).reshape(1, len(SCALE_FACTORS), -1, 2)
    e0 = base[0, N_PRECEDING]
    displacement = base[0, N_PRECEDING:] - e0
    for scale_index, scale in enumerate(SCALE_FACTORS):
        assert np.array_equal(scaled[0, scale_index, :N_PRECEDING], base[0, :N_PRECEDING])
        assert np.allclose(
            scaled[0, scale_index, N_PRECEDING:],
            e0 + float(scale) * displacement,
        )


def test_held_history_controlled_scaling_has_continuous_constant_prefix() -> None:
    base = np.arange((N_PRECEDING + N_SCORED) * 2, dtype=np.float32).reshape(1, -1, 2) / 100.0
    scaled = scaled_histories(base, bank_index=1).reshape(1, len(SCALE_FACTORS), -1, 2)
    e0 = base[0, N_PRECEDING]
    assert np.all(scaled[0, :, :N_PRECEDING] == e0[None, None, :])
    assert np.all(scaled[0, :, N_PRECEDING] == e0[None, :])


def test_production_scorer_supports_history_longer_than_trace() -> None:
    n_timepoints = 40
    n_lags = 60
    stack = np.zeros((n_timepoints + n_lags + 1, 7, 7), dtype=np.float32)
    eye = torch.zeros((n_timepoints, 2), dtype=torch.float32)

    stim = make_counterfactual_stim(stack, eye, n_lags=n_lags, out_size=(5, 5))

    assert tuple(stim.shape) == (n_timepoints, 1, n_lags, 5, 5)


def test_mixed_rate_trace_uses_interpolated_native_grid_and_odd_endpoints() -> None:
    eye = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]],
        dtype=torch.float32,
    )
    native, endpoints = _expand_trace_to_model_grid(
        eye,
        temporal_factor=2,
        supervision_phase=1,
        torch=torch,
    )

    assert torch.equal(endpoints, torch.tensor([1, 3, 5]))
    assert torch.allclose(
        native,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 6.0],
                [4.0, 8.0],
            ]
        ),
    )


def test_mixed_rate_counterfactual_returns_one_output_per_scored_sample() -> None:
    n_timepoints = 40
    n_lags = 60
    temporal_factor = 2
    stack = np.zeros(
        (n_timepoints * temporal_factor + n_lags + 1, 7, 7),
        dtype=np.float32,
    )
    eye = torch.zeros((n_timepoints, 2), dtype=torch.float32)

    stim = make_counterfactual_stim(
        stack,
        eye,
        n_lags=n_lags,
        out_size=(5, 5),
        temporal_factor=temporal_factor,
        supervision_phase=1,
    )

    assert tuple(stim.shape) == (n_timepoints, 1, n_lags, 5, 5)


def test_retained_120hz_trace_expands_to_true_native_240_output_grid() -> None:
    trace = np.asarray(
        [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float32
    )

    expanded = _trace_on_output_grid(
        trace,
        source_rate_hz=120,
        output_rate_hz=240,
        torch=torch,
    )

    np.testing.assert_allclose(
        expanded,
        np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 6.0],
                [4.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )


def test_true_native_240_replay_preserves_duration_and_scores_every_output() -> None:
    source_timepoints = 40
    n_lags = 60
    trace = np.zeros((source_timepoints, 2), dtype=np.float32)
    output_trace = _trace_on_output_grid(
        trace,
        source_rate_hz=120,
        output_rate_hz=240,
        torch=torch,
    )
    stack = np.zeros(
        (output_trace.shape[0] + n_lags + 1, 7, 7), dtype=np.float32
    )

    stim = make_counterfactual_stim(
        stack,
        torch.from_numpy(output_trace),
        n_lags=n_lags,
        out_size=(5, 5),
        temporal_factor=1,
        supervision_phase=0,
    )

    assert output_trace.shape == (80, 2)
    assert tuple(stim.shape) == (80, 1, n_lags, 5, 5)
    assert source_timepoints / 120.0 == stim.shape[0] / 240.0


def test_native240_matrix_integrates_upsampled_outputs_at_240hz() -> None:
    class EvalOnly:
        def eval(self):
            return self

    scorer = RealTraceMatrixScorer(
        model=SimpleNamespace(model=EvalOnly(), names=["session"]),
        readout=EvalOnly(),
        population_view=SimpleNamespace(n_units=1),
        apply_population_view=lambda value, _view: value,
        canonical_unit_rows=[{}],
        rr_unit_rows=[{}],
        torch=torch,
        device="cpu",
        n_lags=2,
        input_rate_hz=240,
        output_rate_hz=240,
        temporal_factor=1,
        supervision_phase=0,
        out_size=(5, 5),
        provenance={},
    )
    seen_frames = []

    def constant_rate_map(stim):
        seen_frames.append(int(stim.shape[0]))
        return torch.ones((stim.shape[0], 1, 2, 2), dtype=torch.float32)

    scorer._compute_rate_map = constant_rate_map
    unit_bits, expected, mean_rate, population_bits = scorer.score_traces_for_patch(
        np.zeros((7, 7), dtype=np.float32),
        [np.zeros((3, 2), dtype=np.float32)],
        trace_batch_size=1,
        frame_batch_size=4,
        n_timepoints=3,
        bin_seconds=1 / 120,
    )

    assert sum(seen_frames) == 6
    np.testing.assert_allclose(expected, [[6 / 240]])
    np.testing.assert_allclose(mean_rate, [[1.0]])
    np.testing.assert_allclose(unit_bits, [[0.0]], atol=1e-7)
    np.testing.assert_allclose(population_bits, [0.0], atol=1e-7)


def test_recovered_static_map_adapter_extends_legacy_stack_for_mixed_rate() -> None:
    n_timepoints = 40
    n_lags = 60
    scorer = SimpleNamespace(
        n_lags=n_lags,
        out_size=(5, 5),
        temporal_factor=2,
        supervision_phase=1,
    )
    common = _LegacyStaticStimulusContract(scorer)
    # This is the exact stack length constructed by the recovered 120-Hz
    # production function.  It is too short for 40 scored samples on a 240-Hz
    # input grid, so the selected-twin adapter must extend the static image.
    stack = np.zeros((n_timepoints + n_lags + 1, 7, 7), dtype=np.float32)
    eye = torch.zeros((n_timepoints, 2), dtype=torch.float32)

    stim = common.make_counterfactual_stim(
        stack,
        eye,
        ppd=37.5,
        scale_factor=1.0,
        n_lags=n_lags,
        out_size=(5, 5),
    )

    assert tuple(stim.shape) == (n_timepoints, 1, n_lags, 5, 5)


def test_selected_map_adapter_accepts_scored_trace_shorter_than_native_history() -> None:
    fake = SimpleNamespace(
        temporal_factor=2,
        supervision_phase=1,
        n_lags=60,
        out_size=(5, 5),
        canonical_unit_rows=[{}],
        provenance={"model": {}},
        model=SimpleNamespace(names=["session"]),
        torch=torch,
        device="cpu",
    )
    from paper.fig4.upstream.real_trace_matrix.model import LegacyCanonicalTwinScorerAdapter

    adapter = LegacyCanonicalTwinScorerAdapter(fake, batch_size=8)
    adapter._compute_rate_map_batched = lambda stim: torch.zeros(
        int(stim.shape[0]), 1, 3, 3
    )

    maps = adapter.rate_map_for_trace(
        np.zeros((7, 7), dtype=np.float32),
        np.zeros((32, 2), dtype=np.float32),
    )

    assert maps.shape == (32, 1, 3, 3)


def test_frequency_probe_selects_outputs_whose_native_endpoints_match_training_phase() -> None:
    assert _output_phase_for_prefixed_movie(
        native_history_frames=60,
        temporal_factor=2,
        supervision_phase=1,
    ) == 0
    assert _output_phase_for_prefixed_movie(
        native_history_frames=59,
        temporal_factor=2,
        supervision_phase=1,
    ) == 1


def test_time_contract_uses_dataset_grid_over_legacy_constructor_default(tmp_path) -> None:
    model = SimpleNamespace(model=SimpleNamespace(sampling_rate=240))
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "sampling: {source_rate: 240, target_rate: 120}\n",
        encoding="utf-8",
    )
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        "sampling: {source_rate: 240, target_rate: 240}\n"
        "supervision: {target_rate: 120, phase: 1}\n",
        encoding="utf-8",
    )

    assert infer_model_time_contract(model, legacy) == {
        "input_rate_hz": 120,
        "output_rate_hz": 120,
        "temporal_factor": 1,
        "supervision_phase": 0,
    }
    assert infer_model_time_contract(model, mixed) == {
        "input_rate_hz": 240,
        "output_rate_hz": 120,
        "temporal_factor": 2,
        "supervision_phase": 1,
    }


def test_frequency_probe_contract_supports_mixed_and_true_native_240(tmp_path) -> None:
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        "sampling: {source_rate: 240, target_rate: 240}\n"
        "supervision: {target_rate: 120, phase: 1}\n"
        "keys_lags: {stim: [0, 1, 2, 3]}\n",
        encoding="utf-8",
    )
    native = tmp_path / "native.yaml"
    native.write_text(
        "sampling: {source_rate: 240, target_rate: 240}\n"
        "keys_lags: {stim: [0, 1, 2, 3, 4, 5]}\n",
        encoding="utf-8",
    )

    assert _declared_time_contract(mixed) == {
        "input_rate_hz": 240,
        "output_rate_hz": 120,
        "temporal_factor": 2,
        "supervision_phase": 1,
        "native_history_frames": 4,
    }
    assert _declared_time_contract(native) == {
        "input_rate_hz": 240,
        "output_rate_hz": 240,
        "temporal_factor": 1,
        "supervision_phase": 0,
        "native_history_frames": 6,
    }


def test_retained_120hz_schematic_trace_is_endpoint_interpolated_at_240hz() -> None:
    trace = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float32)

    expanded = _trace_on_selected_output_grid(
        trace,
        output_rate_hz=240,
        torch=torch,
    )

    np.testing.assert_allclose(
        expanded,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 6.0],
                [4.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )
