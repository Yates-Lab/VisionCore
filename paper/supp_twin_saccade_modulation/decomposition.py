"""Exact latent decomposition and event-alignment helpers for the FIXRSVP pilot."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def analytic_components(scaffold, behavior, modulator, readout):
    """Decompose the released no-phase model before its softplus.

    The zero-behavior reference is evaluated through the biased behavior MLP;
    it is not assumed to have unit gains or zero additive drive.
    """
    if scaffold.ndim != 5 or scaffold.shape[2] != 1:
        raise ValueError("scaffold must have shape (batch, channels, 1, height, width)")
    visual_channels = scaffold.shape[1]
    if visual_channels != modulator.feature_dim:
        raise ValueError("scaffold channel count does not match the modulator")
    if modulator.additive_dim == 0:
        raise ValueError("this pilot requires additive behavior channels")

    encoded = modulator.encoder(modulator.input_norm(behavior))
    zeros = torch.zeros_like(behavior)
    encoded0 = modulator.encoder(modulator.input_norm(zeros))
    gain = 1 + modulator.film_max_gain * torch.tanh(modulator.scale_layer(encoded))
    gain0 = 1 + modulator.film_max_gain * torch.tanh(modulator.scale_layer(encoded0))

    spatial = readout.get_spatial_weights().to(scaffold)
    weights = readout.features.weight[:, :, 0, 0]
    if weights.shape[1] != visual_channels + encoded.shape[1]:
        raise ValueError("readout channels do not match visual plus encoded channels")
    scale = float(readout.output_scale)
    visual_projection = scale * torch.einsum(
        "bchw,uhw->buc", scaffold[:, :, -1], spatial
    )
    visual_channel_drive = visual_projection * weights[:, :visual_channels].unsqueeze(0)
    spatial_sum = spatial.sum((-2, -1))
    additive_weights = weights[:, visual_channels:]
    additive = scale * (encoded @ additive_weights.T) * spatial_sum
    additive0 = scale * (encoded0 @ additive_weights.T) * spatial_sum

    visual0 = (visual_channel_drive * gain0[:, None, :]).sum(-1)
    delta_gain = (visual_channel_drive * (gain - gain0)[:, None, :]).sum(-1)
    delta_additive = additive - additive0
    bias = readout.bias.unsqueeze(0) if readout.bias is not None else 0.0
    z0 = bias + visual0 + additive0
    zfull = z0 + delta_additive + delta_gain
    r0 = F.softplus(z0)
    r_additive = F.softplus(z0 + delta_additive)
    r_gain = F.softplus(z0 + delta_gain)
    rfull = F.softplus(zfull)
    interaction = rfull - r_additive - r_gain + r0
    return {
        "z0": z0,
        "delta_additive": delta_additive,
        "delta_gain": delta_gain,
        "zfull": zfull,
        "r0": r0,
        "r_additive": r_additive,
        "r_gain": r_gain,
        "rfull": rfull,
        "interaction": interaction,
        "gain": gain,
        "gain0": gain0,
        "additive": additive,
        "additive0": additive0,
        "visual_channel_drive": visual_channel_drive,
    }


def isolated_event_indices(events: Sequence[dict], max_amplitude=1.0, isolation_s=0.15):
    """Select finite fitted detections with 0 < amplitude < max and no nearby event."""
    n = len(events)
    times = np.array([event.get("start_time", np.nan) for event in events], dtype=float)
    amplitudes = np.array([
        math.hypot(
            event.get("end_x", np.nan) - event.get("start_x", np.nan),
            event.get("end_y", np.nan) - event.get("start_y", np.nan),
        )
        for event in events
    ])
    verified = np.isfinite(times) & np.isfinite(amplitudes)
    eligible = verified & (amplitudes > 0) & (amplitudes < max_amplitude)
    order = np.argsort(times)
    sorted_times = times[order]
    nearest = np.full(n, np.inf)
    if n > 1:
        gaps = np.diff(sorted_times)
        nearest[order[:-1]] = np.minimum(nearest[order[:-1]], gaps)
        nearest[order[1:]] = np.minimum(nearest[order[1:]], gaps)
    isolated = nearest >= isolation_s
    selected = np.flatnonzero(eligible & isolated)
    return selected, {
        "raw_detections": n,
        "finite_fitted_detections": int(verified.sum()),
        "amplitude_eligible": int(eligible.sum()),
        "isolated_amplitude_eligible": int(len(selected)),
        "amplitude_definition": "hypot(end_x-start_x, end_y-start_y) degrees",
        "verification_definition": "saved detector fit with finite onset and endpoints",
        "max_amplitude_deg_exclusive": float(max_amplitude),
        "isolation_s_inclusive": float(isolation_s),
        "isolation_neighbors": "all saved detections, before amplitude filtering",
    }


def shared_condition_mask(observed, dfs, *predictions):
    """Return one support requiring finite observations, filters, and conditions."""
    observed = np.asarray(observed)
    dfs = np.asarray(dfs)
    if observed.shape != dfs.shape or any(np.asarray(value).shape != observed.shape for value in predictions):
        raise ValueError("observations, filters, and predictions must share a shape")
    support = np.isfinite(observed) & np.isfinite(dfs) & (dfs != 0)
    for value in predictions:
        support &= np.isfinite(value)
    return support


def valid_inference_indices(dfs, trials, times, history_frames, expected_dt=None):
    """Select rows with usable filters and a contiguous same-trial history."""
    dfs = np.asarray(dfs)
    trials = np.asarray(trials).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if dfs.shape[0] != len(times) or trials.shape != times.shape:
        raise ValueError("dfs, trials, and times must share their leading dimension")
    valid_filter = (np.isfinite(dfs) & (dfs != 0)).any(axis=1) if dfs.ndim > 1 else np.isfinite(dfs) & (dfs != 0)
    within_trial = np.diff(times)[np.diff(trials) == 0]
    dt = float(np.median(within_trial)) if expected_dt is None else float(expected_dt)
    kept = []
    rejected_filter = rejected_history = 0
    for index in range(len(times)):
        if not valid_filter[index]:
            rejected_filter += 1
            continue
        lo = index - int(history_frames) + 1
        if lo < 0 or np.any(trials[lo:index + 1] != trials[index]):
            rejected_history += 1
            continue
        gaps = np.diff(times[lo:index + 1])
        if not np.all(np.isclose(gaps, dt, rtol=0.01, atol=1e-9)):
            rejected_history += 1
            continue
        kept.append(index)
    return np.asarray(kept, dtype=np.int64), {
        "input": len(times),
        "history_frames": int(history_frames),
        "native_dt_s": dt,
        "rejected_nonfinite_or_zero_dfs": rejected_filter,
        "rejected_trial_or_time_history": rejected_history,
        "kept": len(kept),
    }


def aggregate_native_pairs(values, trials):
    """Sum nonoverlapping native pairs beginning at each contiguous trial start."""
    values = np.asarray(values)
    trials = np.asarray(trials).ravel()
    if values.shape[0] != len(trials):
        raise ValueError("values and trials must share their leading dimension")
    starts = np.r_[0, np.flatnonzero(np.diff(trials) != 0) + 1]
    stops = np.r_[starts[1:], len(trials)]
    endpoints = []
    for start, stop in zip(starts, stops):
        endpoints.extend(range(int(start) + 1, int(stop), 2))
    endpoints = np.asarray(endpoints, dtype=np.int64)
    return endpoints, values[endpoints - 1] + values[endpoints]


def valid_event_windows(times, trials, sample_valid, event_times, pre_bins, post_bins):
    """Map onsets to nearest native bin and require an intact contiguous trial window."""
    times = np.asarray(times, dtype=float).ravel()
    trials = np.asarray(trials).ravel()
    sample_valid = np.asarray(sample_valid, dtype=bool).ravel()
    event_times = np.asarray(event_times, dtype=float).ravel()
    if not (times.shape == trials.shape == sample_valid.shape):
        raise ValueError("times, trials, and sample_valid must have identical shapes")
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    dt = float(np.median(np.diff(times)[np.diff(trials) == 0]))
    audit = {"input": len(event_times), "outside_tolerance": 0,
             "edge_or_cross_trial": 0, "time_gap_window": 0,
             "invalid_window": 0, "kept": 0}
    windows = []
    for onset in event_times:
        if not np.isfinite(onset):
            audit["outside_tolerance"] += 1
            continue
        right = int(np.searchsorted(times, onset))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(times)]
        center = min(candidates, key=lambda index: abs(times[index] - onset))
        if abs(times[center] - onset) > dt / 2 + 1e-9:
            audit["outside_tolerance"] += 1
            continue
        lo, hi = center - int(pre_bins), center + int(post_bins) + 1
        if lo < 0 or hi > len(times) or np.any(trials[lo:hi] != trials[center]):
            audit["edge_or_cross_trial"] += 1
            continue
        if not np.all(np.isclose(np.diff(times[lo:hi]), dt, rtol=0.01, atol=1e-9)):
            audit["time_gap_window"] += 1
            continue
        if not sample_valid[lo:hi].all():
            audit["invalid_window"] += 1
            continue
        windows.append(np.arange(lo, hi, dtype=np.int64))
    audit["kept"] = len(windows)
    width = int(pre_bins) + int(post_bins) + 1
    return np.asarray(windows, dtype=np.int64).reshape(-1, width), audit
