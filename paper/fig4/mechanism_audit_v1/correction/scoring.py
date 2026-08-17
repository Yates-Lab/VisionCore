"""Exact corrected-history renderer and SSI scorer, isolated from legacy code."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from paper.fig4.upstream.real_trace_matrix.model import (
    N_LAGS,
    OUT_SIZE,
    PPD,
    RealTraceMatrixScorer,
    _embed_time_lags,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)


def build_direct_rr100_readout(scorer: RealTraceMatrixScorer) -> Any:
    """Slice the canonical readout to the exact one-hot RR100 medoid channels.

    This is an algebraically exact optimization for the reconstructed one-hot
    population view.  It avoids computing 656 unused canonical rate maps.
    """
    from scripts.spatial_info import PopulationReadout

    membership = np.asarray(scorer.population_view.membership, dtype=np.float64)
    selected = np.argmax(np.abs(membership), axis=1)
    expected = np.zeros_like(membership)
    expected[np.arange(len(selected)), selected] = 1.0
    if not np.array_equal(membership, expected):
        raise ValueError("Direct RR100 readout requires an exact positive one-hot population view")
    readout = scorer.readout
    direct = PopulationReadout(
        readout.features.weight.detach()[selected].clone(),
        readout.bias.detach()[selected].clone(),
        readout.space_weights.detach()[selected, 0].clone(),
    )
    return direct.to(scorer.device).eval()


def make_corrected_causal_stims(
    image: np.ndarray,
    histories_xy: np.ndarray,
    *,
    torch: Any,
) -> Any:
    """Render full 31-prefix + 40-score histories into exactly 40 lag tensors.

    Input coordinates are canonical ``[x_deg, y_deg]``.  Unlike the legacy
    helper, this function neither swaps them nor constructs a prefix.  The
    supplied full histories already contain the intended prehistory.
    """
    histories = np.asarray(histories_xy, dtype=np.float32)
    if histories.ndim != 3 or histories.shape[1:] != (71, 2):
        raise ValueError(f"Expected (batch, 71, 2) histories, got {histories.shape}")
    batch = int(histories.shape[0])
    frames_per_history = int(histories.shape[1])
    repeated = np.broadcast_to(
        np.asarray(image, dtype=np.float32)[None],
        (batch * frames_per_history, *image.shape),
    ).copy()
    eye = torch.from_numpy(histories.reshape(-1, 2))
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    shifted = _shift_movie_with_eye(
        torch.from_numpy(repeated),
        eye_norm,
        out_size=OUT_SIZE,
        scale_factor=1.0,
        torch=torch,
    ).reshape(batch, frames_per_history, *OUT_SIZE)
    # Preserve the production embedder exactly.  Each 71-frame movie yields 40
    # windows; lag axis 0 is current and lag axis 31 is oldest.
    stims = [_embed_time_lags(shifted[index], n_lags=N_LAGS, torch=torch) for index in range(batch)]
    return torch.cat(stims, dim=0)


def score_corrected_histories_for_patch(
    scorer: RealTraceMatrixScorer,
    patch: np.ndarray,
    histories_xy: np.ndarray,
    *,
    trace_batch_size: int,
    frame_batch_size: int,
    bin_seconds: float = 1.0 / 120.0,
    direct_rr_readout: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score full causal histories using the frozen SSI definition."""
    histories = np.asarray(histories_xy, dtype=np.float32)
    n_traces = int(histories.shape[0])
    n_units = scorer.n_units
    image = _standardize_uint_like(patch)
    unit_expected = np.zeros((n_traces, n_units), dtype=np.float64)
    unit_numer = np.zeros_like(unit_expected)
    unit_rate_sum = np.zeros_like(unit_expected)
    unit_frame_count = np.zeros(n_traces, dtype=np.int64)
    scorer.model.model.eval()
    scorer.readout.eval()

    with scorer.torch.no_grad():
        for trace_start in range(0, n_traces, int(trace_batch_size)):
            trace_stop = min(trace_start + int(trace_batch_size), n_traces)
            chunk = histories[trace_start:trace_stop]
            stim_all = (make_corrected_causal_stims(image, chunk, torch=scorer.torch) - 127.0) / 255.0
            frame_to_trace = np.repeat(np.arange(trace_start, trace_stop, dtype=np.int64), 40)
            for frame_start in range(0, int(stim_all.shape[0]), int(frame_batch_size)):
                frame_stop = min(frame_start + int(frame_batch_size), int(stim_all.shape[0]))
                x = stim_all[frame_start:frame_stop].to(scorer.device)
                if direct_rr_readout is None:
                    full_map = scorer._compute_rate_map(x)
                    rr_map = scorer.apply_population_view(full_map, scorer.population_view)
                else:
                    from scripts.spatial_info import compute_rate_map

                    dtype = next(scorer.model.model.parameters()).dtype
                    behavior = scorer._zero_behavior(int(x.shape[0]), dtype)
                    full_map = None
                    rr_map = compute_rate_map(
                        scorer.model, direct_rr_readout, x, behavior=behavior
                    )
                rr_map = rr_map.clamp_min(0.0).to(scorer.torch.float64)
                flat = rr_map.reshape(rr_map.shape[0], rr_map.shape[1], -1)
                rbar = flat.mean(dim=2)
                gain = flat / (rbar[..., None] + 1e-8)
                bits = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
                rb = rbar.detach().cpu().numpy()
                ub = bits.detach().cpu().numpy()
                ids = frame_to_trace[frame_start:frame_stop]
                for trace_index in np.unique(ids):
                    mask = ids == trace_index
                    weights = rb[mask] * float(bin_seconds)
                    unit_expected[trace_index] += np.sum(weights, axis=0)
                    unit_numer[trace_index] += np.sum(ub[mask] * weights, axis=0)
                    unit_rate_sum[trace_index] += np.sum(rb[mask], axis=0)
                    unit_frame_count[trace_index] += int(np.count_nonzero(mask))
                del x, full_map, rr_map, flat, rbar, gain, bits
            del stim_all

    unit_ssi = np.divide(unit_numer, np.maximum(unit_expected, 1e-8)).astype(np.float32)
    mean_rate = np.divide(unit_rate_sum, np.maximum(unit_frame_count[:, None], 1)).astype(np.float32)
    population_ssi = np.divide(
        np.sum(unit_numer, axis=1), np.maximum(np.sum(unit_expected, axis=1), 1e-8)
    ).astype(np.float32)
    return unit_ssi, unit_expected.astype(np.float32), mean_rate, population_ssi
