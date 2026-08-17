#!/usr/bin/env python3
"""Compare one unit's Jacobian at identical physical FixRSVP moments.

The published 120-Hz twin and the Dekel-style candidates use different input
lattices (33 x 51 x 51 at 120 Hz versus 60 x 35 x 35 at 240 Hz).  This script
therefore aligns examples by FixRSVP trial identity and 120-Hz PSTH bin, not by
tensor row.  The Twin's spatial gradients are center-cropped to 35 x 35 only for
the comparison plot; full-resolution gradients remain in the per-model NPZ.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import dill
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONTEXT_QUANTILES = np.asarray([0.2, 0.5, 0.8])
CONTEXT_NAMES = ["low drive", "median drive", "high drive"]
MIN_FIX_DUR = 20
FIXATION_RADIUS_DEG = 1.0


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_session(cache_path: Path, session: str):
    with cache_path.open("rb") as stream:
        sessions = dill.load(stream)
    matches = [result for result in sessions if result["session"] == session]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {session} entry in {cache_path}, found {len(matches)}"
        )
    return matches[0]


def _choose_cache_contexts(cache_results, full_unit_index: int):
    """Choose three shared trial/time points from calibrated cached rates."""
    unit_positions = []
    rates = []
    for result in cache_results:
        matches = np.flatnonzero(
            np.asarray(result["neuron_mask"], dtype=np.int64) == full_unit_index
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Full unit {full_unit_index} is not unique in {result['session']} cache"
            )
        unit_positions.append(int(matches[0]))
        rates.append(np.asarray(result["rhat_used"])[..., matches[0]])

    expected_shape = rates[0].shape
    if any(rate.shape != expected_shape for rate in rates):
        raise RuntimeError("FixRSVP caches do not have identical trial/time shapes")

    valid = np.ones(expected_shape, dtype=bool)
    standardized = []
    for result, unit_position, rate in zip(cache_results, unit_positions, rates):
        valid &= np.isfinite(rate)
        valid &= np.asarray(result["dfs_used"])[..., unit_position] > 0
        log_rate = np.log(np.maximum(rate, 1e-8))
        standardized.append(log_rate)

    standardized = np.stack(standardized)
    flat_valid = valid.ravel()
    for model_index in range(len(standardized)):
        flat = standardized[model_index].ravel()
        values = flat[flat_valid]
        flat -= values.mean()
        flat /= max(float(values.std()), 1e-8)
    drive = standardized.mean(axis=0)

    candidates = np.flatnonzero(flat_valid & np.isfinite(drive.ravel()))
    targets = np.quantile(drive.ravel()[candidates], CONTEXT_QUANTILES)
    chosen = []
    for target in targets:
        available = np.asarray([index for index in candidates if index not in chosen])
        chosen.append(
            int(available[np.argmin(np.abs(drive.ravel()[available] - target))])
        )
    trial_rows, time_bins = np.unravel_index(chosen, expected_shape)
    return {
        "unit_positions": unit_positions,
        "trial_rows": np.asarray(trial_rows, dtype=np.int64),
        "time_bins": np.asarray(time_bins, dtype=np.int64),
        "drive": drive[trial_rows, time_bins],
        "targets": targets,
        "calibrated_rates": np.stack(rates)[:, trial_rows, time_bins],
    }


def _prepare_endpoint_map(checkpoint_path: Path, dataset_idx: int):
    from paper.model_selection.evaluate_dekel_fixrsvp import (
        _checkpoint_dataset_configs,
        _prepare_fixrsvp,
        _valid_endpoint_indices,
        figure3_endpoint_coordinates,
    )
    from paper.model_selection.evaluate_true240_fixrsvp import _native_pair_indices

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configs, cfg_path = _checkpoint_dataset_configs(checkpoint)
    del checkpoint
    config = configs[dataset_idx]
    dset, resolved = _prepare_fixrsvp(config)

    trial_inds = _as_numpy(dset["trial_inds"]).ravel()
    psth_inds = _as_numpy(dset["psth_inds"]).ravel()
    eyepos = _as_numpy(dset["eyepos"])
    lags = np.asarray(resolved["keys_lags"]["stim"], dtype=np.int64)
    sampling = resolved.get("sampling") or {}
    supervision = resolved.get("supervision") or {}
    true_native_240 = (
        int(sampling.get("source_rate", 0)) == 240
        and int(sampling.get("target_rate", 0)) == 240
        and not supervision
    )

    if true_native_240:
        pair_start, pair_end = _native_pair_indices(trial_inds, psth_inds)
        pair_eye = 0.5 * (eyepos[pair_start] + eyepos[pair_end])
        fixation = np.hypot(pair_eye[:, 0], pair_eye[:, 1]) < FIXATION_RADIUS_DEG
        pair_start = pair_start[fixation]
        pair_end = pair_end[fixation]
        pair_trials = trial_inds[pair_end]
        pair_psth = psth_inds[pair_start].astype(np.int64) // 2
        all_trials = np.unique(pair_trials)
        durations = np.asarray([(pair_trials == trial).sum() for trial in all_trials])
        good_trials = all_trials[durations > MIN_FIX_DUR]
        endpoint_payload = {
            "endpoint_kind": "native_pair",
            "pair_start": pair_start,
            "pair_end": pair_end,
            "endpoint_psth": pair_psth,
            "endpoint_trials": pair_trials,
            "factor": 2,
            "phase": "global native pairs",
        }
        source_rate = 240.0
    else:
        endpoints, factor, phase = _valid_endpoint_indices(dset, resolved)
        endpoints, endpoint_psth = figure3_endpoint_coordinates(
            trial_inds, psth_inds, endpoints, factor
        )
        endpoint_eye = np.stack(
            [
                eyepos[index - factor + 1 : index + 1].mean(axis=0)
                for index in endpoints
            ]
        )
        fixation = (
            np.hypot(endpoint_eye[:, 0], endpoint_eye[:, 1])
            < FIXATION_RADIUS_DEG
        )
        endpoints = endpoints[fixation]
        endpoint_psth = endpoint_psth[fixation]
        endpoint_trials = trial_inds[endpoints]
        all_trials = np.unique(endpoint_trials)
        durations = np.asarray(
            [(endpoint_trials == trial).sum() for trial in all_trials]
        )
        good_trials = all_trials[durations > MIN_FIX_DUR]
        endpoint_payload = {
            "endpoint_kind": "single",
            "endpoints": endpoints,
            "endpoint_psth": endpoint_psth,
            "endpoint_trials": endpoint_trials,
            "factor": factor,
            "phase": phase,
        }
        source_rate = float(sampling.get("target_rate", 120))
        if supervision:
            source_rate = float(sampling.get("source_rate", 240))

    return {
        "dset": dset,
        "resolved": resolved,
        "cfg_path": str(cfg_path),
        "session": resolved["session"],
        **endpoint_payload,
        "good_trials": good_trials,
        "lags": lags,
        "sampling_rate": source_rate,
    }


def _resolve_rows(endpoint_map, trial_rows, time_bins):
    good_trials = endpoint_map["good_trials"]
    if int(trial_rows.max()) >= len(good_trials):
        raise RuntimeError("Selected cache trial row is absent from prepared FixRSVP")
    physical_trials = good_trials[trial_rows]
    endpoints = []
    for trial, time_bin in zip(physical_trials, time_bins):
        matches = np.flatnonzero(
            (endpoint_map["endpoint_trials"] == trial)
            & (endpoint_map["endpoint_psth"] == time_bin)
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one endpoint for trial {trial}, bin {time_bin}; "
                f"found {len(matches)}"
            )
        if endpoint_map["endpoint_kind"] == "native_pair":
            endpoints.append(
                [
                    int(endpoint_map["pair_start"][matches[0]]),
                    int(endpoint_map["pair_end"][matches[0]]),
                ]
            )
        else:
            endpoints.append(int(endpoint_map["endpoints"][matches[0]]))
    endpoints = np.asarray(endpoints, dtype=np.int64)
    earliest = endpoints if endpoints.ndim == 1 else endpoints[:, 0]
    if np.any(earliest[:, None] - endpoint_map["lags"][None, :] < 0):
        raise RuntimeError("A selected endpoint lacks the model's full history")
    return physical_trials, endpoints


def _aligned_observation(dset, endpoints, output_unit_index: int) -> np.ndarray:
    """Return the exact count represented by each aligned model output."""
    robs = _as_numpy(dset["robs"])
    if endpoints.ndim == 1:
        return np.asarray(robs[endpoints, output_unit_index], dtype=np.float64)
    if endpoints.ndim == 2 and endpoints.shape[1] == 2:
        return np.asarray(
            robs[endpoints[:, 0], output_unit_index]
            + robs[endpoints[:, 1], output_unit_index],
            dtype=np.float64,
        )
    raise ValueError("endpoints must have shape [context] or [context, 2]")


def _exact_jacobian(
    model,
    dset,
    endpoints,
    lags,
    dataset_idx,
    unit_index,
    device,
):
    endpoints = np.asarray(endpoints, dtype=np.int64)
    lags_tensor = torch.as_tensor(lags, dtype=torch.long)
    if endpoints.ndim == 1:
        raw_indices = torch.as_tensor(endpoints, dtype=torch.long)
        lag_indices = raw_indices[:, None] - lags_tensor[None, :]
        stimulus = (
            dset["stim"][lag_indices]
            .permute(0, 2, 1, 3, 4)
            .float()
            .to(device)
            .detach()
            .requires_grad_(True)
        )
        behavior = dset["behavior"][raw_indices].float().to(device)
        output_behavior = (
            dset["output_behavior"][raw_indices].float().to(device)
            if "output_behavior" in dset
            else None
        )
        output = model(
            stimulus,
            dataset_idx,
            behavior,
            None,
            output_behavior,
        )
        if unit_index >= output.shape[1]:
            raise IndexError(f"Unit {unit_index} is outside {output.shape[1]} outputs")
        if model.log_input:
            rate = output[:, unit_index].exp()
        else:
            rate = output[:, unit_index].clamp_min(1e-8)
        gradient = torch.autograd.grad(rate.log().sum(), stimulus, create_graph=False)[0]
        return (
            gradient[:, 0].detach().float().cpu().numpy(),
            rate.detach().float().cpu().numpy(),
            np.asarray(lags, dtype=np.int64),
        )

    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("endpoints must have shape [context] or [context, 2]")
    if np.any(endpoints[:, 1] != endpoints[:, 0] + 1):
        raise RuntimeError("native endpoints must be adjacent causal pairs")

    # Differentiate the summed 8.33-ms count prediction with respect to one
    # shared native movie timeline.  The two 4.17-ms model calls overlap in 59
    # of 60 input frames; treating them as independent tensors would duplicate
    # those physical frames and would not be the Jacobian of the scored output.
    n_contexts = len(endpoints)
    max_lag = int(np.max(lags))
    min_lag = int(np.min(lags))
    timeline_start = torch.as_tensor(endpoints[:, 0] - max_lag, dtype=torch.long)
    timeline_length = max_lag - min_lag + 2
    timeline_indices = timeline_start[:, None] + torch.arange(timeline_length)[None, :]
    timeline = (
        dset["stim"][timeline_indices]
        .permute(0, 2, 1, 3, 4)
        .float()
        .to(device)
        .detach()
        .requires_grad_(True)
    )
    start_positions = max_lag - lags_tensor
    end_positions = start_positions + 1
    stimulus = torch.cat(
        [timeline[:, :, start_positions], timeline[:, :, end_positions]], dim=0
    )
    raw_indices = torch.as_tensor(
        np.concatenate([endpoints[:, 0], endpoints[:, 1]]), dtype=torch.long
    )
    behavior = dset["behavior"][raw_indices].float().to(device)
    output_behavior = (
        dset["output_behavior"][raw_indices].float().to(device)
        if "output_behavior" in dset
        else None
    )
    output = model(stimulus, dataset_idx, behavior, None, output_behavior)
    if unit_index >= output.shape[1]:
        raise IndexError(f"Unit {unit_index} is outside {output.shape[1]} outputs")
    native_rate = (
        output[:, unit_index].exp()
        if model.log_input
        else output[:, unit_index].clamp_min(1e-8)
    )
    rate = native_rate[:n_contexts] + native_rate[n_contexts:]
    gradient = torch.autograd.grad(rate.log().sum(), timeline, create_graph=False)[0]
    effective_lags = np.arange(min_lag, max_lag + 2, dtype=np.int64)
    # timeline is chronological (oldest to newest); display uses increasing
    # lag before the scored response (current to oldest).
    gradient = gradient.flip(dims=(2,))
    return (
        gradient[:, 0].detach().float().cpu().numpy(),
        rate.detach().float().cpu().numpy(),
        effective_lags,
    )


def _roughness(jacobian, sampling_rate):
    value = torch.from_numpy(jacobian).float()
    denom = value.square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    temporal = value.diff(n=2, dim=1).square().mean(dim=(1, 2, 3)) / denom

    flat = value.flatten(0, 1).unsqueeze(1)
    padded = torch.nn.functional.pad(flat, (1, 1, 1, 1))
    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    laplacian = torch.nn.functional.conv2d(padded, kernel).reshape_as(value)
    spatial = laplacian.square().mean(dim=(1, 2, 3)) / denom

    frequency = torch.fft.rfftfreq(value.shape[1], d=1.0 / sampling_rate)
    temporal_power = torch.fft.rfft(value, dim=1).abs().square().sum(dim=(2, 3))
    temporal_high = temporal_power[:, frequency >= 60].sum(dim=1)
    temporal_high /= temporal_power.sum(dim=1).clamp_min(1e-12)

    fy = torch.fft.fftfreq(value.shape[2])[:, None]
    fx = torch.fft.fftfreq(value.shape[3])[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    spatial_power = torch.fft.fft2(value, dim=(-2, -1)).abs().square().sum(dim=1)
    spatial_high = spatial_power[:, radius >= 0.25].sum(dim=1)
    spatial_high /= spatial_power.sum(dim=(1, 2)).clamp_min(1e-12)
    return {
        "rms": value.square().mean(dim=(1, 2, 3)).sqrt().tolist(),
        "temporal_second_difference_ratio": temporal.tolist(),
        "spatial_laplacian_ratio": spatial.tolist(),
        "temporal_power_at_or_above_60hz": temporal_high.tolist(),
        "spatial_power_at_or_above_0p25_cycles_per_pixel": spatial_high.tolist(),
    }


def _center_crop(jacobian, size=35):
    height, width = jacobian.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"Cannot crop {height} x {width} Jacobian to {size} x {size}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return jacobian[..., y0 : y0 + size, x0 : x0 + size], [y0, x0]


def _plot_model(
    jacobian,
    lags,
    sampling_rate,
    model_name,
    calibrated_rates,
    raw_rates,
    output_stem,
):
    import matplotlib.pyplot as plt

    display, crop_origin = _center_crop(jacobian, size=35)
    lag_ms = np.asarray(lags) * 1000.0 / sampling_rate
    temporal_energy = np.sqrt(np.mean(display**2, axis=(-2, -1)))
    colors = plt.get_cmap("tab10").colors[:3]

    figure = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    temporal_axis = figure.add_subplot(grid[0, :])
    for context_index, (context_name, color) in enumerate(
        zip(CONTEXT_NAMES, colors)
    ):
        energy = temporal_energy[context_index]
        temporal_axis.plot(
            lag_ms,
            energy / max(float(energy.max()), 1e-12),
            color=color,
            linewidth=2.2,
            label=context_name,
        )
    temporal_axis.set(
        xlabel="lag before response (ms)",
        ylabel="normalized spatial RMS",
        title=f"{model_name}: temporal Jacobian energy",
        ylim=(-0.04, 1.08),
    )
    temporal_axis.spines[["top", "right"]].set_visible(False)
    temporal_axis.legend(frameon=False, ncol=3, loc="upper right")

    for context_index, context_name in enumerate(CONTEXT_NAMES):
        axis = figure.add_subplot(grid[1, context_index])
        peak = int(np.argmax(temporal_energy[context_index]))
        spatial = display[context_index, peak]
        vmax = max(float(np.max(np.abs(spatial))), 1e-12)
        axis.imshow(
            spatial,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{context_name}\n"
            f"peak {lag_ms[peak]:.0f} ms; calibrated {calibrated_rates[context_index] * 120:.1f} sp/s\n"
            f"raw model {raw_rates[context_index] * 120:.1f} sp/s",
            fontsize=9,
        )
    figure.suptitle(
        "Exact d log(rate) / d stimulus at shared FixRSVP moments\n"
        "spatial panels use independent symmetric color scales",
        fontsize=11,
    )
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path, crop_origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--cache-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--session", default="Allen_2022-02-16")
    parser.add_argument("--unit-index", type=int, default=105)
    parser.add_argument("--dataset-idx", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    count = len(args.checkpoints)
    if len(args.model_names) != count or len(args.cache_paths) != count:
        parser.error("checkpoints, model names, and cache paths need equal lengths")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cache_results = [
        _load_session(path.resolve(), args.session) for path in args.cache_paths
    ]
    selection = _choose_cache_contexts(cache_results, args.unit_index)

    endpoint_maps = []
    physical_trials = None
    endpoints_by_model = []
    for checkpoint_path in args.checkpoints:
        endpoint_map = _prepare_endpoint_map(checkpoint_path.resolve(), args.dataset_idx)
        if endpoint_map["session"] != args.session:
            raise RuntimeError(
                f"Dataset {args.dataset_idx} is {endpoint_map['session']}, not {args.session}"
            )
        if len(endpoint_map["good_trials"]) != cache_results[0]["n_trials"]:
            raise RuntimeError(
                f"Prepared {len(endpoint_map['good_trials'])} good trials, cache has "
                f"{cache_results[0]['n_trials']}"
            )
        trials, endpoints = _resolve_rows(
            endpoint_map, selection["trial_rows"], selection["time_bins"]
        )
        if physical_trials is None:
            physical_trials = trials
        elif not np.array_equal(trials, physical_trials):
            raise RuntimeError("Model input lattices resolve different physical trials")
        cache_unit_position = int(selection["unit_positions"][len(endpoint_maps)])
        observed = _aligned_observation(
            endpoint_map["dset"], endpoints, args.unit_index
        )
        cached_observed = np.asarray(
            cache_results[len(endpoint_maps)]["robs_used"]
        )[
            selection["trial_rows"],
            selection["time_bins"],
            cache_unit_position,
        ]
        if not np.array_equal(observed, cached_observed):
            raise RuntimeError(
                "Aligned Jacobian endpoints do not reproduce the cached "
                f"FixRSVP spike counts for model {len(endpoint_maps)}: "
                f"observed={observed.tolist()}, cached={cached_observed.tolist()}"
            )
        endpoint_maps.append(endpoint_map)
        endpoints_by_model.append(endpoints)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from eval.load_twin import load_twin

    model_reports = {}
    artifacts = {}
    for model_index, (checkpoint_path, model_name, endpoint_map, endpoints) in enumerate(
        zip(args.checkpoints, args.model_names, endpoint_maps, endpoints_by_model)
    ):
        model, _ = load_twin(
            checkpoint_path.resolve(), device=str(device), verbose=False
        )
        model.eval()
        jacobian, raw_rates, effective_lags = _exact_jacobian(
            model,
            endpoint_map["dset"],
            endpoints,
            endpoint_map["lags"],
            args.dataset_idx,
            args.unit_index,
            device,
        )
        slug = _slug(model_name)
        stem = args.out_dir / slug
        npz_path = stem.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            jacobian=jacobian,
            lags=effective_lags,
            sampling_rate=endpoint_map["sampling_rate"],
            raw_rates=raw_rates,
            calibrated_rates=selection["calibrated_rates"][model_index],
            physical_trials=physical_trials,
            time_bins=selection["time_bins"],
            endpoints=endpoints,
            model_name=model_name,
            checkpoint=str(checkpoint_path.resolve()),
        )
        png_path, pdf_path, crop_origin = _plot_model(
            jacobian,
            effective_lags,
            endpoint_map["sampling_rate"],
            model_name,
            selection["calibrated_rates"][model_index],
            raw_rates,
            stem,
        )
        model_reports[model_name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "model_output_index": int(args.unit_index),
            "figure3_cache_unit_position": int(
                selection["unit_positions"][model_index]
            ),
            "sampling_rate_hz": endpoint_map["sampling_rate"],
            "input_shape": list(jacobian.shape[1:]),
            "display_center_crop_origin_yx": crop_origin,
            "raw_rate_counts_per_120hz_bin": raw_rates.tolist(),
            "calibrated_cache_rate_counts_per_120hz_bin": selection[
                "calibrated_rates"
            ][model_index].tolist(),
            "jacobian_metrics": _roughness(
                jacobian, endpoint_map["sampling_rate"]
            ),
        }
        artifacts[model_name] = {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "npz": str(npz_path.resolve()),
        }
        del model, jacobian
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "session": args.session,
        "full_unit_index": args.unit_index,
        "selection_rule": (
            "Nearest valid FixRSVP examples to the 20th, 50th, and 80th "
            "percentiles of mean standardized log calibrated rate across models"
        ),
        "context_names": CONTEXT_NAMES,
        "cache_trial_rows": selection["trial_rows"].tolist(),
        "physical_trial_ids": physical_trials.tolist(),
        "psth_time_bins_120hz": selection["time_bins"].tolist(),
        "context_drive": selection["drive"].tolist(),
        "target_drive_quantiles": selection["targets"].tolist(),
        "models": model_reports,
        "artifacts": artifacts,
    }
    report_path = args.out_dir / "aligned_fixrsvp_jacobians.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
