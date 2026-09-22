"""Renderer-faithful retinal replay primitives shared by Figure 4 analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from paper.fig4.upstream.real_trace_matrix.model import (
    OUT_SIZE,
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
    make_counterfactual_stim,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"


def evenly_spaced_rows(length: int, requested: int) -> np.ndarray:
    if length < 1 or requested < 1:
        raise ValueError("available and requested row counts must be positive")
    rows = np.unique(
        np.round(np.linspace(0, length - 1, min(length, requested))).astype(int)
    )
    if len(rows) != min(length, requested):
        raise RuntimeError("deterministic row selection produced duplicates")
    return rows


def render_movies(
    patch: np.ndarray,
    traces: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Render ``[trace,time,y,x]`` through the scorer's exact eye geometry."""
    trace = np.asarray(traces, dtype=np.float32)
    if trace.ndim != 3 or trace.shape[-1] != 2:
        raise ValueError(f"traces must be [trace,time,2], got {trace.shape}")
    n_trace, n_time = trace.shape[:2]
    image = _standardize_uint_like(patch)
    eye = torch.from_numpy(trace.reshape(-1, 2)).to(device)
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    base = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    repeated = base.unsqueeze(0).expand(n_trace * n_time, -1, -1)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            repeated,
            eye_norm,
            out_size=OUT_SIZE,
            scale_factor=1.0,
            torch=torch,
        )
    return shifted.reshape(n_trace, n_time, *OUT_SIZE).cpu().numpy()


def causal_histories(
    patch: np.ndarray,
    trace_xy: np.ndarray,
    *,
    n_lags: int,
    out_size: tuple[int, int],
    temporal_factor: int,
    supervision_phase: int,
) -> torch.Tensor:
    """Build histories with the response matrix's held-initial-gaze policy."""
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace_xy, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None],
        (len(trace) * int(temporal_factor) + int(n_lags) + 1, *image.shape),
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    histories = make_counterfactual_stim(
        full_stack,
        eye,
        ppd=PPD,
        scale_factor=1.0,
        n_lags=int(n_lags),
        out_size=tuple(map(int, out_size)),
        temporal_factor=int(temporal_factor),
        supervision_phase=int(supervision_phase),
    )
    return (histories - 127.0) / 255.0
