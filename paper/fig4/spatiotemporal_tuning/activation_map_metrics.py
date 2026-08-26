"""Shared activation-map measurements for Figure 4 visual audits.

Keeping these small, model-agnostic helpers outside either atlas builder avoids
an import cycle and guarantees that the candidate audit and the 100-pair atlas
use identical map normalization, SSI, and temporal-anchor definitions.
"""
from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-12


def model_mean_peak_lag(core: Any) -> dict[str, object]:
    """Find the mean peak-energy lag of the model's learned temporal filters."""
    if not hasattr(core, "effective_temporal_weight"):
        raise TypeError("model core does not expose its effective temporal weights")
    weight = core.effective_temporal_weight().detach().float().cpu().numpy()
    if weight.ndim != 5 or weight.shape[1] != 1:
        raise ValueError(
            "temporal weights must have shape [channel, 1, lag, y, x]"
        )
    lag_rms = np.sqrt(np.mean(weight[:, 0] ** 2, axis=(2, 3)))
    channel_peaks = np.argmax(lag_rms, axis=1).astype(int)
    mean_lag = float(np.mean(channel_peaks))
    rounded_lag = int(np.floor(mean_lag + 0.5))
    resolved = channel_peaks[
        (channel_peaks > 0) & (channel_peaks < weight.shape[2] - 1)
    ]
    if not len(resolved):
        raise ValueError("no temporal filter has an interior peak-energy lag")
    resolved_mean_lag = float(np.mean(resolved))
    resolved_rounded_lag = int(np.floor(resolved_mean_lag + 0.5))
    if not 0 <= rounded_lag < weight.shape[2]:
        raise ValueError("model-derived mean peak lag is outside temporal support")
    return {
        "channel_peak_lag_indices": channel_peaks.tolist(),
        "mean_peak_lag_frames": mean_lag,
        "rounded_peak_lag_frames": rounded_lag,
        "boundary_censored_channel_count": int(len(channel_peaks) - len(resolved)),
        "resolved_mean_peak_lag_frames": resolved_mean_lag,
        "resolved_rounded_peak_lag_frames": resolved_rounded_lag,
        "temporal_support_frames": int(weight.shape[2]),
    }


def score_histories(
    scorer: Any,
    histories: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return nonnegative selected-population maps for newest-first histories."""
    torch = scorer.torch
    chunks = []
    scorer.model.model.eval()
    scorer.readout.eval()
    with torch.no_grad():
        for start in range(0, len(histories), int(batch_size)):
            value = torch.from_numpy(
                np.asarray(histories[start : start + int(batch_size)], dtype=np.float32)
            ).to(scorer.device)
            value = ((value - 127.0) / 255.0).flip(1).unsqueeze(1)
            full = scorer._compute_rate_map(value)
            selected = scorer.apply_population_view(
                full, scorer.population_view
            ).clamp_min(0.0)
            chunks.append(selected.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def map_statistics(maps: np.ndarray, output_rate_hz: float) -> dict[str, np.ndarray]:
    """Measure firing rate, SSI, and normalized maps with one shared definition."""
    rate = np.clip(np.asarray(maps, dtype=np.float64), 0.0, None)
    mean_native = rate.mean(axis=(-2, -1))
    gain = rate / np.maximum(mean_native[..., None, None], EPS)
    information = np.mean(
        gain * np.log2(np.maximum(gain, EPS)), axis=(-2, -1)
    )
    return {
        "rate_spikes_s": float(output_rate_hz) * mean_native,
        "ssi_bits_per_spike": information,
        "normalized_map": gain,
    }
