#!/usr/bin/env python3
"""Evaluate a Dekel twin on the out-of-training-distribution FixRSVP stimulus.

This deliberately mirrors ``paper/fig3/_fig3_data.py``.  In particular it
uses the Figure-3 trial/time window, cell population, positive per-cell affine
Poisson calibration, and normalized-correlation estimator.  The Dekel models
retain native 240-Hz stimulus histories but use causal 120-Hz count targets;
the endpoint predictions are therefore placed on the same 120-Hz PSTH grid as
the published Figure-3 cache.

FixRSVP is absent from the training configuration, so this is an expensive
external-generalization gate rather than a checkpoint-selection loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import dill
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Keep these explicit and verify them against Figure 3 at runtime.  That makes
# a future Figure-3 protocol change fail loudly instead of silently changing
# this comparison.
DT = 1 / 120
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 200
CCNORM_N_SPLITS = 500
FIXATION_RADIUS_DEG = 1.0


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reference_by_session(cache_path: Path):
    with cache_path.open("rb") as stream:
        sessions = dill.load(stream)
    return {result["session"]: result for result in sessions}


def _verify_figure3_constants():
    fig3_dir = ROOT / "paper" / "fig3"
    if str(fig3_dir) not in sys.path:
        sys.path.insert(0, str(fig3_dir))
    import _fig3_data as fig3

    expected = {
        "DT": DT,
        "VALID_TIME_BINS": VALID_TIME_BINS,
        "MIN_FIX_DUR": MIN_FIX_DUR,
        "MIN_TOTAL_SPIKES": MIN_TOTAL_SPIKES,
        "CCNORM_N_SPLITS": CCNORM_N_SPLITS,
    }
    observed = {key: getattr(fig3, key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "Figure-3 evaluation constants changed; audit this evaluator "
            f"before running. Expected {expected}, observed {observed}."
        )


def _checkpoint_dataset_configs(checkpoint: dict):
    from models.config_loader import load_dataset_configs

    hparams = checkpoint.get("hyper_parameters", {}) or {}
    cfg_path = hparams.get("cfg_dir")
    if not cfg_path:
        raise ValueError("Checkpoint does not record cfg_dir")
    cfg_path = Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    configs = load_dataset_configs(cfg_path)

    snapshot = hparams.get("dataset_cids") or {}
    for config in configs:
        name = config["session"]
        if name in snapshot:
            config["cids"] = list(snapshot[name])
    return configs, cfg_path


def _prepare_fixrsvp(config):
    """Load FixRSVP once and expose all split endpoints.

    Training/validation/test membership is irrelevant because FixRSVP was not
    a training stimulus.  We nevertheless request all three splits and take
    their union so the resulting trial population matches Figure 3's use of
    the complete FixRSVP recording.
    """
    from models.data import prepare_data

    config = copy.deepcopy(config)
    config["types"] = ["fixrsvp"]
    train, val, test, resolved = prepare_data(
        config, strict=True, return_test=True
    )
    dsets = train.dsets
    if len(dsets) != 1 or dsets[0].metadata.get("name") != "fixrsvp":
        raise RuntimeError("Expected exactly one underlying FixRSVP dataset")
    return dsets[0], resolved


def _valid_endpoint_indices(dset, config):
    sampling = config.get("sampling") or {}
    supervision = config.get("supervision") or {}
    source_rate = int(sampling.get("source_rate", 120))
    target_rate = int(supervision.get("target_rate", source_rate))
    if source_rate % target_rate:
        raise ValueError(
            f"Supervision rate {target_rate} does not divide source rate {source_rate}"
        )
    factor = source_rate // target_rate
    phase = int(supervision.get("phase", factor - 1))
    indices = np.arange(len(dset), dtype=np.int64)
    return indices[indices % factor == phase], factor, phase


def figure3_endpoint_coordinates(trial_inds, psth_inds, endpoints, factor):
    """Drop cross-trial causal blocks and return Figure-3 target coordinates."""
    trial_inds = np.asarray(trial_inds).ravel()
    psth_inds = np.asarray(psth_inds).ravel()
    endpoints = np.asarray(endpoints, dtype=np.int64)
    block_start = endpoints - factor + 1
    in_bounds = block_start >= 0
    endpoints = endpoints[in_bounds]
    block_start = block_start[in_bounds]
    same_trial = trial_inds[block_start] == trial_inds[endpoints]
    endpoints = endpoints[same_trial]
    block_start = block_start[same_trial]
    return endpoints, psth_inds[block_start].astype(np.int64) // factor


def _predict_endpoints(
    model,
    dset,
    indices,
    lags,
    dataset_idx,
    device,
    batch_size,
):
    predictions = []
    stim_source = dset["stim"]
    behavior_source = dset["behavior"]
    model.eval()
    for start in range(0, len(indices), batch_size):
        raw_indices = torch.as_tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        lag_indices = raw_indices[:, None] - lags[None, :]
        if lag_indices.min().item() < 0:
            raise RuntimeError("A FixRSVP endpoint lacks the requested stimulus history")
        stim = stim_source[lag_indices].permute(0, 2, 1, 3, 4).to(device)
        behavior = behavior_source[raw_indices].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(stim, dataset_idx, behavior, None)
            if model.log_input:
                output = output.exp()
        predictions.append(output.float().cpu())
    return torch.cat(predictions).numpy()


def _trial_arrays(model, dset, config, dataset_idx, device, batch_size):
    trial_inds = _as_numpy(dset["trial_inds"]).ravel()
    psth_inds = _as_numpy(dset["psth_inds"]).ravel().astype(np.int64)
    eyepos = _as_numpy(dset["eyepos"])
    robs = _as_numpy(dset["robs"])
    dfs = _as_numpy(dset["dfs"])

    endpoints, factor, phase = _valid_endpoint_indices(dset, config)
    lags = torch.as_tensor(config["keys_lags"]["stim"], dtype=torch.long)
    endpoints, endpoint_psth = figure3_endpoint_coordinates(
        trial_inds, psth_inds, endpoints, factor
    )
    # Figure 3 first downsamples continuous covariates by averaging each
    # non-overlapping native-rate block and then applies the fixation-radius
    # criterion.  Reconstruct that exact eye position for trial selection,
    # while the model itself still receives its native-rate behavior vector.
    endpoint_eyepos = np.stack(
        [
            eyepos[index - factor + 1 : index + 1].mean(axis=0)
            for index in endpoints
        ]
    )
    fixation = np.hypot(endpoint_eyepos[:, 0], endpoint_eyepos[:, 1]) < FIXATION_RADIUS_DEG
    endpoints = endpoints[fixation]
    endpoint_eyepos = endpoint_eyepos[fixation]

    # Figure 3 retains the first trial even when its earliest samples cannot
    # supply the complete model history; those bins are excluded by dfs.  Keep
    # them in the aligned arrays (with NaN predictions), and infer only where
    # all requested lags exist.  This preserves Figure 3's trial inclusion and
    # leave-one-out PSTH population exactly.
    inferable = endpoints >= int(lags.max())
    predictions = np.full((len(endpoints), robs.shape[1]), np.nan, np.float32)
    predictions[inferable] = _predict_endpoints(
        model,
        dset,
        endpoints[inferable],
        lags,
        dataset_idx,
        device,
        batch_size,
    )

    trials = np.unique(trial_inds[endpoints])
    # Native PSTH coordinates map to the causal 120-Hz target grid.  This is
    # the phase-1 counterpart of Figure 3's decimation then integer division.
    # Figure 3's 240->120 path labels each non-overlapping block with the
    # block-start PSTH coordinate (then integer-divides it).  The Dekel target
    # itself lives at the causal block endpoint, but must be placed at that
    # same Figure-3 coordinate for an exact trace comparison.
    target_psth = endpoint_psth[fixation]
    max_time = int(target_psth.max()) + 1
    n_units = robs.shape[1]
    robs_trial = np.full((len(trials), max_time, n_units), np.nan, np.float32)
    rhat_trial = np.full_like(robs_trial, np.nan)
    dfs_trial = np.full_like(robs_trial, np.nan)
    eyepos_trial = np.full((len(trials), max_time, 2), np.nan, np.float32)
    fix_dur = np.zeros(len(trials), dtype=np.int64)

    endpoint_trials = trial_inds[endpoints]
    for row, trial in enumerate(trials):
        select = endpoint_trials == trial
        time = target_psth[select]
        if len(np.unique(time)) != len(time):
            raise RuntimeError(f"Duplicate 120-Hz PSTH bins in trial {trial}")
        raw = endpoints[select]
        fix_dur[row] = len(time)
        robs_trial[row, time] = robs[raw]
        rhat_trial[row, time] = predictions[select]
        dfs_trial[row, time] = dfs[raw]
        eyepos_trial[row, time] = endpoint_eyepos[select]

    good = fix_dur > MIN_FIX_DUR
    if good.sum() < 10:
        raise RuntimeError(f"Only {good.sum()} FixRSVP trials exceed {MIN_FIX_DUR} bins")
    time = np.arange(min(VALID_TIME_BINS, max_time))
    return {
        "robs": robs_trial[good][:, time],
        "rhat": rhat_trial[good][:, time],
        "dfs": dfs_trial[good][:, time],
        "eyepos": eyepos_trial[good][:, time],
        "factor": factor,
        "phase": phase,
    }


def _finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "mean": None, "q25": None, "q75": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _session_metrics(arrays, neuron_mask, reference=None):
    from eval.eval_stack_utils import (
        bits_per_spike,
        rescale_rhat,
    )

    robs = arrays["robs"][:, :, neuron_mask]
    rhat = arrays["rhat"][:, :, neuron_mask]
    dfs = arrays["dfs"][:, :, neuron_mask]
    if reference is not None:
        reference_robs = np.asarray(reference["robs_used"], dtype=np.float32)
        reference_dfs = np.asarray(reference["dfs_used"], dtype=np.float32)
        if robs.shape != reference_robs.shape:
            raise RuntimeError(
                f"Extracted observations {robs.shape} do not match Figure-3 "
                f"observations {reference_robs.shape}"
            )
        overlap = np.isfinite(robs) & np.isfinite(reference_robs)
        max_abs = (
            float(np.max(np.abs(robs[overlap] - reference_robs[overlap])))
            if overlap.any()
            else float("inf")
        )
        if max_abs != 0:
            raise RuntimeError(
                "Native-rate causal count bins do not align with Figure 3; "
                f"maximum overlapping spike-count difference is {max_abs}"
            )
        # Observed responses and filters are model-independent.  Use the
        # canonical Figure-3 arrays so its early-history mask, trial population,
        # reliability, and leave-one-out PSTH are bit-for-bit identical.  Only
        # the new twin prediction changes.
        reference_finite = np.isfinite(reference_robs)
        rhat = np.where(reference_finite, rhat, np.nan)
        robs = reference_robs.copy()
        dfs = reference_dfs.copy()
    shape = robs.shape
    data_support = _data_score_mask(robs, dfs)
    missing_prediction = data_support & ~np.isfinite(rhat)
    if missing_prediction.any():
        raise RuntimeError(
            "Prediction is missing on canonical Figure-3 support: "
            f"{int(missing_prediction.sum())} samples"
        )

    robs_flat = torch.from_numpy(robs.reshape(-1, shape[-1]))
    rhat_flat = torch.from_numpy(rhat.reshape(-1, shape[-1]))
    dfs_scoring = torch.from_numpy(
        data_support.reshape(-1, shape[-1]).astype(np.float32)
    )
    calibrated, affine = rescale_rhat(
        robs_flat, rhat_flat, dfs_scoring, mode="affine"
    )
    bps_raw = bits_per_spike(rhat_flat, robs_flat, dfs_scoring).numpy()
    bps_affine = bits_per_spike(calibrated, robs_flat, dfs_scoring).numpy()
    rhat = calibrated.reshape(shape).numpy()

    cc = _ccnorm_on_fixed_support(robs, rhat, dfs)
    ccnorm = cc["ccnorm"]
    ccabs = cc["ccabs"]
    ccmax = cc["ccmax"]

    # The canonical filters contain NaNs outside the reference analysis frame.
    # Treat those as invalid explicitly: ``NaN == 0`` is false, so masking only
    # zero-valued filters would otherwise let predictions from missing trials
    # contaminate the trial-averaged PSTH correlation.
    valid_samples = data_support
    rhat_masked = np.where(valid_samples, rhat, np.nan)
    robs_masked = np.where(valid_samples, robs, np.nan)
    rhat_mean = np.nanmean(rhat_masked, axis=0)
    robs_mean = np.nanmean(robs_masked, axis=0)
    n_valid = valid_samples.sum(axis=0)
    rhos = np.asarray(
        [
            np.corrcoef(
                rhat_mean[n_valid[:, unit] > 10, unit],
                robs_mean[n_valid[:, unit] > 10, unit],
            )[0, 1]
            for unit in range(shape[-1])
        ]
    )

    leave_one_out = np.zeros_like(robs_masked)
    for trial in range(shape[0]):
        other = np.arange(shape[0]) != trial
        leave_one_out[trial] = np.nanmean(robs_masked[other], axis=0)

    with torch.no_grad():
        scale = affine.g.exp().cpu().numpy()
        offset = affine.b.exp().cpu().numpy()
    return {
        "robs_used": robs,
        "rhat_used": rhat,
        "dfs_used": dfs,
        "rhat_mean": rhat_mean,
        "robs_mean": robs_mean,
        "rhos": rhos,
        "ccnorm": ccnorm,
        "ccabs": ccabs,
        "ccmax": ccmax,
        "ccnorm_support": data_support,
        "ccnorm_unstable": cc["unstable"],
        "ccmax_seed_delta": cc["seed_delta"],
        "ve_model": _variance_explained_float64(rhat_masked, robs_masked),
        "ve_psth": _variance_explained_float64(leave_one_out, robs_masked),
        "bps_raw": bps_raw,
        "bps_affine": bps_affine,
        "affine_scale": scale,
        "affine_offset": offset,
    }


def _validate_observations(result, reference):
    expected = np.asarray(reference["robs_used"])
    observed = np.asarray(result["robs_used"])
    if expected.shape != observed.shape:
        return {
            "exact_match": False,
            "reason": f"shape {observed.shape} != Figure-3 {expected.shape}",
        }
    both = np.isfinite(expected) & np.isfinite(observed)
    same_nan = np.array_equal(np.isfinite(expected), np.isfinite(observed))
    max_abs = float(np.max(np.abs(expected[both] - observed[both]))) if both.any() else None
    return {
        "exact_match": bool(same_nan and max_abs == 0),
        "finite_mask_match": bool(same_nan),
        "max_abs_difference": max_abs,
    }


def _valid_score_mask(robs, rhat, dfs):
    """Return the exact shared sample support for model/data summaries."""
    robs = np.asarray(robs)
    rhat = np.asarray(rhat)
    dfs = np.asarray(dfs)
    if robs.shape != rhat.shape or robs.shape != dfs.shape:
        raise ValueError(
            f"Scoring arrays must have one shape; got {robs.shape}, "
            f"{rhat.shape}, and {dfs.shape}"
        )
    return (
        np.isfinite(robs)
        & np.isfinite(rhat)
        & np.isfinite(dfs)
        & (dfs != 0)
    )


def _data_score_mask(robs, dfs):
    """Return model-independent Figure-3 support.

    ``dfs`` is stored as a numeric array with NaNs outside the analysis frame.
    Passing it directly to a Boolean conversion is unsafe because NumPy treats
    ``bool(np.nan)`` as true.  Build the support explicitly once and require
    every model prediction to cover it.
    """
    robs = np.asarray(robs)
    dfs = np.asarray(dfs)
    if robs.shape != dfs.shape:
        raise ValueError(
            f"Observation/filter arrays must have one shape; got "
            f"{robs.shape} and {dfs.shape}"
        )
    return np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)


def _variance_explained_float64(prediction, observation):
    """Compute single-trial variance explained without dtype-dependent drift."""
    prediction = np.asarray(prediction, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if prediction.shape != observation.shape:
        raise ValueError(
            f"prediction/observation shape mismatch: {prediction.shape} versus "
            f"{observation.shape}"
        )
    return 1.0 - np.nanvar(prediction - observation, axis=(0, 1)) / np.nanvar(
        observation, axis=(0, 1)
    )


def _ccnorm_on_fixed_support(robs, rhat, dfs):
    """Compute CCnorm with an explicit data-only mask and fixed RNG seeds.

    The returned normalized correlation is formed from the averaged raw
    correlation and averaged data-only noise ceiling.  This enforces the exact
    per-unit identity ``CCnorm == CCabs / CCmax`` while retaining the existing
    two-seed stability check.
    """
    from eval.eval_stack_utils import ccnorm_split_half_variable_trials

    robs = np.asarray(robs, dtype=np.float64)
    rhat = np.asarray(rhat, dtype=np.float64)
    dfs = np.asarray(dfs)
    if robs.shape != rhat.shape or robs.shape != dfs.shape:
        raise ValueError(
            f"CCnorm arrays must have one shape; got {robs.shape}, "
            f"{rhat.shape}, and {dfs.shape}"
        )
    support = _data_score_mask(robs, dfs)
    missing_prediction = support & ~np.isfinite(rhat)
    if missing_prediction.any():
        raise RuntimeError(
            "Model prediction is missing on canonical Figure-3 support: "
            f"{int(missing_prediction.sum())} samples"
        )

    estimates = []
    for seed in (42, 43):
        estimates.append(
            ccnorm_split_half_variable_trials(
                robs.copy(),
                rhat.copy(),
                support.copy(),
                n_splits=CCNORM_N_SPLITS,
                return_components=True,
                rng=seed,
            )
        )
    ccnorm1, ccabs1, ccmax1, _, _ = estimates[0]
    ccnorm2, ccabs2, ccmax2, _, _ = estimates[1]
    if not np.allclose(ccabs1, ccabs2, rtol=0, atol=1e-12, equal_nan=True):
        raise AssertionError("CCabs changed across data-only split-half seeds")

    ccabs = 0.5 * (ccabs1 + ccabs2)
    ccmax = 0.5 * (ccmax1 + ccmax2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ccnorm = ccabs / ccmax
    # Stability is a property of the data-derived noise ceiling, not of the
    # model numerator.  Gating on CCnorm(seed 1) - CCnorm(seed 2) would allow
    # two models with the same observations to retain different unit sets.
    seed_delta = ccmax1 - ccmax2
    unstable = seed_delta ** 2 > 0.01
    ccnorm[unstable] = np.nan
    valid = np.isfinite(ccnorm)
    if valid.any() and not np.allclose(
        ccnorm[valid], (ccabs / ccmax)[valid], rtol=0, atol=1e-12
    ):
        raise AssertionError("CCnorm identity failed on finite units")
    return {
        "ccnorm": ccnorm,
        "ccabs": ccabs,
        "ccmax": ccmax,
        "support": support,
        "unstable": unstable,
        "seed_delta": seed_delta,
        "stability_basis": "squared difference of data-only CCmax estimates > 0.01",
    }


def _audit_shared_ccnorm(current, reference):
    """Rescore Ryan and a candidate on identical support and noise ceilings."""
    reference_cc = _ccnorm_on_fixed_support(
        reference["robs_used"], reference["rhat_used"], reference["dfs_used"]
    )
    if not np.array_equal(current["ccnorm_support"], reference_cc["support"]):
        raise AssertionError("Candidate and Ryan do not use identical CCnorm support")
    if not np.allclose(
        current["ccmax"], reference_cc["ccmax"], rtol=0, atol=1e-12, equal_nan=True
    ):
        raise AssertionError("Candidate and Ryan CCmax differ on identical data")
    if not np.array_equal(current["ccnorm_unstable"], reference_cc["unstable"]):
        raise AssertionError(
            "Candidate and Ryan CCnorm stability masks differ on identical data"
        )
    audited_reference = dict(reference)
    audited_reference.update(
        ccnorm=reference_cc["ccnorm"],
        ccabs=reference_cc["ccabs"],
        ccmax=reference_cc["ccmax"],
    )
    return audited_reference, {
        "support_exact_match": True,
        "ccmax_exact_match": True,
        "stability_mask_exact_match": True,
        "n_supported_samples": int(reference_cc["support"].sum()),
        "n_stable_units": int(np.isfinite(reference_cc["ccnorm"]).sum()),
    }


def _population_report(results, references):
    metric_names = (
        "rhos",
        "ccnorm",
        "ccabs",
        "ccmax",
        "ve_model",
        "ve_psth",
        "bps_raw",
        "bps_affine",
    )
    report = {"all_cells": {}, "reliable_cells": {}, "paired_vs_figure3": {}}
    all_values = {
        key: np.concatenate([np.asarray(result[key]) for result in results])
        for key in metric_names
    }
    reliability = all_values["ccmax"] > 0.85
    for key, values in all_values.items():
        report["all_cells"][key] = _finite_summary(values)
        report["reliable_cells"][key] = _finite_summary(values[reliability])

    reference_values = {
        key: np.concatenate([np.asarray(reference[key]) for reference in references])
        for key in ("rhos", "ccnorm", "ccabs", "ccmax", "ve_model", "ve_psth")
    }
    for key, reference in reference_values.items():
        current = all_values[key]
        valid = np.isfinite(current) & np.isfinite(reference)
        report["paired_vs_figure3"][key] = {
            "n": int(valid.sum()),
            "current": _finite_summary(current[valid]),
            "figure3": _finite_summary(reference[valid]),
            "difference": _finite_summary(current[valid] - reference[valid]),
            "fraction_current_greater": float(np.mean(current[valid] > reference[valid]))
            if valid.any()
            else None,
        }
    report["n_sessions"] = len(results)
    report["n_cells"] = int(len(all_values["rhos"]))
    report["n_reliable_cells"] = int(reliability.sum())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--figure3-cache",
        type=Path,
        default=ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-cache", type=Path, default=None)
    args = parser.parse_args()

    _verify_figure3_constants()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configs, cfg_path = _checkpoint_dataset_configs(checkpoint)
    config_by_session = {config["session"]: config for config in configs}
    references = _reference_by_session(args.figure3_cache)
    names = [name for name in config_by_session if name in references]
    if args.max_sessions is not None:
        names = names[: args.max_sessions]

    from eval.load_twin import load_twin

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, info = load_twin(checkpoint_path, device=str(device), verbose=True)
    model_name_to_idx = {name: idx for idx, name in enumerate(model.names)}

    results = []
    used_references = []
    observation_checks = []
    metric_audits = []
    for ordinal, name in enumerate(names, 1):
        print(f"\n[{ordinal}/{len(names)}] FixRSVP {name}", flush=True)
        dset, resolved = _prepare_fixrsvp(config_by_session[name])
        arrays = _trial_arrays(
            model,
            dset,
            resolved,
            model_name_to_idx[name],
            device,
            args.batch_size,
        )
        reference = references[name]
        neuron_mask = np.asarray(reference["neuron_mask"], dtype=np.int64)
        if neuron_mask.max(initial=-1) >= len(config_by_session[name]["cids"]):
            raise RuntimeError(f"Figure-3 neuron mask is out of range for {name}")
        metrics = _session_metrics(arrays, neuron_mask, reference=reference)
        check = _validate_observations(metrics, reference)
        audited_reference, metric_audit = _audit_shared_ccnorm(metrics, reference)
        check["session"] = name
        metric_audit["session"] = name
        observation_checks.append(check)
        metric_audits.append(metric_audit)
        print(
            f"  cells={len(neuron_mask)} trials={metrics['robs_used'].shape[0]} "
            f"rho={np.nanmedian(metrics['rhos']):.4f} "
            f"ccnorm={np.nanmedian(metrics['ccnorm']):.4f} "
            f"observation_match={check['exact_match']}",
            flush=True,
        )
        results.append(
            {
                "session": name,
                "subject": name.split("_")[0],
                "neuron_mask": neuron_mask,
                "n_trials": int(metrics["robs_used"].shape[0]),
                "n_time": int(metrics["robs_used"].shape[1]),
                "n_neurons": int(metrics["robs_used"].shape[2]),
                "eyepos_used": arrays["eyepos"],
                **metrics,
            }
        )
        used_references.append(audited_reference)
        del dset, arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset_config": str(cfg_path),
        "protocol": {
            "reference": "paper/fig3/_fig3_data.py",
            "stimulus": "fixrsvp (absent from training)",
            "stimulus_rate_hz": 240,
            "supervision_rate_hz": 120,
            "valid_time_bins": VALID_TIME_BINS,
            "minimum_fixation_bins": MIN_FIX_DUR,
            "minimum_total_spikes": MIN_TOTAL_SPIKES,
            "affine_calibration": "per-cell positive gain + positive offset; Poisson LBFGS",
            "ccnorm_splits_per_seed": CCNORM_N_SPLITS,
            "ccnorm_seeds": [42, 43],
            "reliable_cell_threshold": "ccmax > 0.85",
            "figure3_cache": str(args.figure3_cache.resolve()),
        },
        "model": {
            "n_units": int(info["n_units"]),
            "modulator_type": info["modulator_type"],
            "cids_source": info["cids_source"],
        },
        "observation_alignment": observation_checks,
        "metric_audits": metric_audits,
        "population": _population_report(results, used_references),
        "sessions": [
            {
                "session": result["session"],
                "n_trials": result["n_trials"],
                "n_neurons": result["n_neurons"],
                **{
                    key: _finite_summary(result[key])
                    for key in (
                        "rhos",
                        "ccnorm",
                        "ccabs",
                        "ccmax",
                        "ve_model",
                        "ve_psth",
                        "bps_raw",
                        "bps_affine",
                    )
                },
            }
            for result in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if args.trace_cache is not None:
        args.trace_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_cache.open("wb") as stream:
            dill.dump(results, stream)
    print(json.dumps(report["population"], indent=2))
    print(f"Wrote {args.out}")
    if args.trace_cache is not None:
        print(f"Wrote {args.trace_cache}")


if __name__ == "__main__":
    main()
