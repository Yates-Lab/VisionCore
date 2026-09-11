from types import SimpleNamespace

import numpy as np
import pytest
import torch

from paper.fig4.upstream.real_trace_matrix import model as replay
from paper.fig4.spatiotemporal_tuning import analyze_top_passband_stage_trajectory as stages


@pytest.mark.parametrize("source_rate_hz", [120, 240])
def test_replay_reduces_native_bin_counts_to_hz_and_movie_counts(monkeypatch, source_rate_hz):
    native_counts = torch.tensor([8.0, 12.0]) / 240
    monkeypatch.setattr(
        replay, "make_counterfactual_stim",
        lambda full_stack, eye, **kwargs: torch.zeros(len(eye), 1, 60, 2, 2),
    )
    scorer = SimpleNamespace(
        model=SimpleNamespace(model=torch.nn.Identity()), readout=torch.nn.Identity(),
        n_units=2, torch=torch, device="cpu", n_lags=60, output_rate_hz=240,
        temporal_factor=1, supervision_phase=0, out_size=(2, 2), population_view=None,
        apply_population_view=lambda value, view: value,
        _compute_rate_map=lambda value: native_counts[None, :, None, None].expand(len(value), 2, 2, 2),
    )
    samples = source_rate_hz // 4

    _, expected, rate_hz, _ = replay.RealTraceMatrixScorer.score_traces_for_patch(
        scorer, np.full((4, 4), 127, dtype=np.uint8),
        [np.zeros((samples, 2), dtype=np.float32)], trace_batch_size=1,
        frame_batch_size=8, n_timepoints=samples, bin_seconds=1 / source_rate_hz,
    )

    np.testing.assert_allclose(rate_hz, [[8.0, 12.0]], rtol=1e-6)
    np.testing.assert_allclose(expected, [[2.0, 3.0]], rtol=1e-6)
    np.testing.assert_allclose(expected, rate_hz * 0.25, rtol=1e-6)


def test_cumulative_stage_reduction_uses_the_same_physical_units(monkeypatch):
    native_counts = torch.tensor([8.0, 12.0]) / 240

    def maps(scorer, batch, **kwargs):
        return native_counts[None, None, :, None, None].expand(3, len(batch), 2, 2, 2), {}

    monkeypatch.setattr(stages, "cumulative_rate_maps", maps)
    rate_hz, expected, _, _, _ = stages.score_histories(
        SimpleNamespace(n_units=2, device="cpu"), torch.zeros(60, 1),
        batch_size=8, check_identity=False,
    )

    np.testing.assert_allclose(rate_hz, np.tile([8.0, 12.0], (3, 1)), rtol=1e-6)
    np.testing.assert_allclose(expected, np.tile([2.0, 3.0], (3, 1)), rtol=1e-6)
    np.testing.assert_allclose(expected, rate_hz * 0.25, rtol=1e-6)
