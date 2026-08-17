#!/usr/bin/env python3
"""Mandatory cache-only integrity gate for the Figure 4 causal interventions.

This program never loads or runs the digital twin.  It reconstructs the exact
frozen tiled RR100 readout from the saved weights and tests rank-zero,
full-rank, and a deterministic low-rank projector on actual cached ConvGRU
states from each of the three preregistered contrasts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    ANALYSIS_SEED,
    BIN_SECONDS,
    CACHE,
    CONFIG,
    CONTRASTS,
    EPS,
    MAP_CACHE,
    MAP_SIZE,
    N_CHANNELS,
    N_FRAMES,
    N_UNITS,
    OUT,
    READOUT_CACHE,
    STATE_CACHE,
    STATE_SIZE,
    haar_basis,
    intervention_preactivations,
    project_channel_delta,
    projected_delta_preactivation,
    rate_map_components,
    readout_preactivation,
    scale_index,
    sha256_file,
    softplus_rate,
    write_json,
)


TOP_LEVEL_OUTPUT = OUT / "integrity_tests.json"
CACHE_OUTPUT = CACHE / "integrity_tests.json"
CONFIG_PATH = CONFIG / "optimization_config.json"
LOW_RANK_TEST_RANK = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Validate the first N completed row-major pairs; zero validates every completed pair.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_abs(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate.double() - reference.double()).abs().max().detach().cpu())


def _rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate.double() - reference.double()).square().mean().sqrt().detach().cpu())


def _relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (estimate.double() - reference.double()).square().sum().sqrt()
    denominator = reference.double().square().sum().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).detach().cpu())


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


def _endpoint(h: torch.Tensor, readout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    preactivation = readout_preactivation(h, readout["feature"], readout["bias"], readout["space"])
    rate = softplus_rate(preactivation)
    components = rate_map_components(rate.double())
    return {
        "state": h,
        "preactivation": preactivation,
        "rate": rate,
        "gain": components["gain"],
        "ssi": components["ssi"],
        "expected_spikes": components["expected_spikes"],
    }


def _endpoint_from_preactivation(preactivation: torch.Tensor) -> dict[str, torch.Tensor]:
    rate = softplus_rate(preactivation)
    components = rate_map_components(rate.double())
    return {
        "preactivation": preactivation,
        "rate": rate,
        "gain": components["gain"],
        "ssi": components["ssi"],
        "expected_spikes": components["expected_spikes"],
    }


def _update_max(
    maxima: dict[str, dict[str, dict[str, Any]]],
    contrast: str,
    metric: str,
    value: float,
    location: dict[str, int],
) -> None:
    row = maxima.setdefault(contrast, {}).get(metric)
    if row is None or float(value) > float(row["value"]):
        maxima[contrast][metric] = {"value": float(value), "location": dict(location)}


def _compare_strict_endpoint(
    maxima: dict[str, dict[str, dict[str, Any]]],
    contrast: str,
    prefix: str,
    estimate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    location: dict[str, int],
) -> None:
    for name in ("preactivation", "rate", "gain", "ssi", "expected_spikes"):
        _update_max(maxima, contrast, f"{prefix}_{name}_max_abs", _max_abs(estimate[name], reference[name]), location)


def _compare_storage_endpoint(
    maxima: dict[str, dict[str, dict[str, Any]]],
    contrast: str,
    prefix: str,
    reconstructed: dict[str, torch.Tensor],
    stored: dict[str, torch.Tensor],
    location: dict[str, int],
) -> None:
    _update_max(
        maxima,
        contrast,
        f"storage_{prefix}_preactivation_relative_rmse",
        _relative_rmse(reconstructed["preactivation"], stored["preactivation"]),
        location,
    )
    _update_max(
        maxima,
        contrast,
        f"storage_{prefix}_rate_relative_rmse",
        _relative_rmse(reconstructed["rate"], stored["rate"]),
        location,
    )
    _update_max(
        maxima,
        contrast,
        f"storage_{prefix}_gain_rmse",
        _rmse(reconstructed["gain"], stored["gain"]),
        location,
    )
    _update_max(
        maxima,
        contrast,
        f"storage_{prefix}_ssi_max_abs",
        _max_abs(reconstructed["ssi"], stored["ssi"]),
        location,
    )
    _update_max(
        maxima,
        contrast,
        f"storage_{prefix}_expected_spikes_relative_rmse",
        _relative_rmse(reconstructed["expected_spikes"], stored["expected_spikes"]),
        location,
    )
    all_finite = all(_finite(value) for value in reconstructed.values()) and all(
        _finite(value) for value in stored.values()
    )
    _update_max(maxima, contrast, f"storage_{prefix}_nonfinite", float(not all_finite), location)


def _stored_endpoint(
    maps: h5py.File,
    image_position: int,
    trajectory_position: int,
    scale: float,
    frames: slice,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = (image_position, trajectory_position, scale_index(scale), frames)
    return {
        name: torch.as_tensor(np.asarray(maps[name][index], dtype=np.float32), device=device)
        for name in ("preactivation", "rate", "gain", "ssi", "expected_spikes")
    }


def _load_configuration() -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run prepare_analysis.py first: {CONFIG_PATH}")
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    strict = configuration["numerical_tolerances"]
    storage = strict["float16_storage_validation"]
    return configuration, strict, storage


def _load_readout(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    if not READOUT_CACHE.exists():
        raise FileNotFoundError(f"Frozen readout cache does not exist: {READOUT_CACHE}")
    with np.load(READOUT_CACHE) as archive:
        required = {
            "feature_weights",
            "bias",
            "space_weights",
            "image_ids",
            "trajectory_ids",
            "scales",
            "low_unit_indices",
            "high_unit_indices",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"Frozen readout cache is missing {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    if arrays["feature_weights"].shape != (N_UNITS, N_CHANNELS):
        raise RuntimeError(f"Unexpected feature-readout shape: {arrays['feature_weights'].shape}")
    kernel_size = STATE_SIZE - MAP_SIZE + 1
    if arrays["bias"].shape != (N_UNITS,) or arrays["space_weights"].shape != (
        N_UNITS,
        kernel_size,
        kernel_size,
    ):
        raise RuntimeError(
            f"Unexpected bias/spatial-readout shapes: {arrays['bias'].shape}, {arrays['space_weights'].shape}"
        )
    low = np.asarray(arrays["low_unit_indices"], dtype=np.int64)
    high = np.asarray(arrays["high_unit_indices"], dtype=np.int64)
    if len(low) != 71 or len(high) != 29 or not np.array_equal(
        np.sort(np.r_[low, high]), np.arange(N_UNITS)
    ):
        raise RuntimeError("Frozen readout does not preserve the historical 71/29 exhaustive unit split")
    readout = {
        "feature": torch.as_tensor(arrays["feature_weights"], dtype=torch.float32, device=device),
        "bias": torch.as_tensor(arrays["bias"], dtype=torch.float32, device=device),
        "space": torch.as_tensor(arrays["space_weights"], dtype=torch.float32, device=device),
    }
    return readout, arrays


def _make_checks(
    maxima: dict[str, dict[str, dict[str, Any]]],
    map_effect_sums: dict[str, dict[str, float]],
    strict: dict[str, Any],
    storage: dict[str, float],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(contrast: str, metric: str, observed: float, comparison: str, threshold: float, family: str) -> None:
        passed = observed <= threshold if comparison == "<=" else observed >= threshold
        checks.append(
            {
                "contrast": contrast,
                "metric": metric,
                "family": family,
                "observed": float(observed),
                "comparison": comparison,
                "threshold": float(threshold),
                "passed": bool(passed),
                "worst_location": maxima.get(contrast, {}).get(metric, {}).get("location"),
            }
        )

    for contrast in CONTRASTS:
        values = maxima[contrast.key]
        for prefix in ("rank_zero_suff", "rank_zero_nec"):
            add(
                contrast.key,
                f"{prefix}_state_max_abs",
                values[f"{prefix}_state_max_abs"]["value"],
                "<=",
                strict["rank_zero_state_max_abs"],
                "exact_algebra",
            )
        for prefix in ("identity_suff", "identity_nec"):
            add(
                contrast.key,
                f"{prefix}_state_max_abs",
                values[f"{prefix}_state_max_abs"]["value"],
                "<=",
                strict["rank_128_state_max_abs"],
                "exact_algebra",
            )
        for prefix in (
            "rank_zero_suff",
            "rank_zero_nec",
            "identity_suff",
            "identity_nec",
            "random_low_rank_suff_canonical_vs_direct",
            "random_low_rank_nec_canonical_vs_direct",
        ):
            for name, threshold in (
                ("preactivation", strict["preactivation_max_abs"]),
                ("rate", strict["rate_max_abs"]),
                ("gain", strict["normalized_map_max_abs"]),
                ("ssi", strict["ssi_max_abs_bits"]),
                ("expected_spikes", strict["rate_max_abs"] * BIN_SECONDS),
            ):
                metric = f"{prefix}_{name}_max_abs"
                add(contrast.key, metric, values[metric]["value"], "<=", threshold, "exact_algebra")

        # The configuration gives one dimensionless float16 map-RMSE limit.
        # Apply it directly to g and to RMS-normalized z/r/expected errors; SSI
        # and contrast-effect R2 retain their separately configured criteria.
        for endpoint in ("A", "B"):
            for name in ("preactivation", "rate", "expected_spikes"):
                metric = f"storage_{endpoint}_{name}_relative_rmse"
                add(
                    contrast.key,
                    metric,
                    values[metric]["value"],
                    "<=",
                    storage["normalized_map_rmse_max"],
                    "float16_storage",
                )
            metric = f"storage_{endpoint}_gain_rmse"
            add(
                contrast.key,
                metric,
                values[metric]["value"],
                "<=",
                storage["normalized_map_rmse_max"],
                "float16_storage",
            )
            metric = f"storage_{endpoint}_ssi_max_abs"
            add(
                contrast.key,
                metric,
                values[metric]["value"],
                "<=",
                storage["ssi_max_abs_bits"],
                "float16_storage",
            )
            metric = f"storage_{endpoint}_nonfinite"
            add(contrast.key, metric, values[metric]["value"], "<=", 0.0, "float16_storage")

        sums = map_effect_sums[contrast.key]
        map_r2 = 1.0 - sums["error_sse"] / max(sums["effect_sse"], EPS)
        add(
            contrast.key,
            "storage_contrast_map_effect_r2",
            map_r2,
            ">=",
            storage["map_recovery_r2_min"],
            "float16_storage",
        )
    return checks


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive")
    if args.max_pairs < 0:
        raise ValueError("--max-pairs cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    configuration, strict, storage = _load_configuration()
    if not bool(storage.get("required_before_use", False)):
        raise RuntimeError("Configuration no longer marks float16 validation as mandatory")
    readout, readout_arrays = _load_readout(device)

    maxima: dict[str, dict[str, dict[str, Any]]] = {contrast.key: {} for contrast in CONTRASTS}
    map_effect_sums = {
        contrast.key: {"error_sse": 0.0, "effect_sse": 0.0} for contrast in CONTRASTS
    }
    tested_frames = {contrast.key: 0 for contrast in CONTRASTS}
    random_basis = torch.as_tensor(
        haar_basis(LOW_RANK_TEST_RANK, ANALYSIS_SEED + 910_007), dtype=torch.float32, device=device
    )
    identity = torch.eye(N_CHANNELS, dtype=torch.float32, device=device)
    rank_zero = torch.empty((N_CHANNELS, 0), dtype=torch.float32, device=device)
    orthonormality_error = _max_abs(
        random_basis.T @ random_basis,
        torch.eye(LOW_RANK_TEST_RANK, device=device),
    )

    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        state_completed = np.asarray(state["completed_pairs"][:], dtype=bool)
        map_completed = np.asarray(maps["completed_pairs"][:], dtype=bool)
        if not np.array_equal(state_completed, map_completed):
            raise RuntimeError("State and map completion masks differ")
        completed_pairs = [tuple(map(int, value)) for value in np.argwhere(state_completed)]
        if not completed_pairs:
            raise RuntimeError("No completed image--trajectory pair is available")
        pairs = completed_pairs[: args.max_pairs] if args.max_pairs else completed_pairs
        state_complete = bool(state.attrs.get("complete", False)) and bool(state_completed.all())
        map_complete = bool(maps.attrs.get("complete", False)) and bool(map_completed.all())
        for name in ("image_ids", "trajectory_ids", "scales"):
            if not np.array_equal(state[name][:], maps[name][:]):
                raise RuntimeError(f"State and map cache metadata differ at {name}")
            if not np.array_equal(state[name][:], readout_arrays[name]):
                raise RuntimeError(f"Frozen readout metadata differs from cache at {name}")

        for contrast in CONTRASTS:
            for image_position, trajectory_position in pairs:
                for frame_start in range(0, N_FRAMES, args.frame_batch_size):
                    frame_stop = min(frame_start + args.frame_batch_size, N_FRAMES)
                    frames = slice(frame_start, frame_stop)
                    location = {
                        "image_position": image_position,
                        "trajectory_position": trajectory_position,
                        "first_frame": frame_start,
                        "last_frame_exclusive": frame_stop,
                    }
                    h_a = torch.as_tensor(
                        np.asarray(
                            state["h"][image_position, trajectory_position, scale_index(contrast.scale_a), frames],
                            dtype=np.float32,
                        ),
                        device=device,
                    )
                    h_b = torch.as_tensor(
                        np.asarray(
                            state["h"][image_position, trajectory_position, scale_index(contrast.scale_b), frames],
                            dtype=np.float32,
                        ),
                        device=device,
                    )
                    if not _finite(h_a) or not _finite(h_b):
                        raise FloatingPointError(f"Non-finite cached state at {contrast.key}, {location}")
                    delta_h = h_b - h_a
                    endpoint_a = _endpoint(h_a, readout)
                    endpoint_b = _endpoint(h_b, readout)
                    stored_a = _stored_endpoint(
                        maps,
                        image_position,
                        trajectory_position,
                        contrast.scale_a,
                        frames,
                        device,
                    )
                    stored_b = _stored_endpoint(
                        maps,
                        image_position,
                        trajectory_position,
                        contrast.scale_b,
                        frames,
                        device,
                    )
                    _compare_storage_endpoint(maxima, contrast.key, "A", endpoint_a, stored_a, location)
                    _compare_storage_endpoint(maxima, contrast.key, "B", endpoint_b, stored_b, location)
                    reconstructed_effect = endpoint_b["gain"] - endpoint_a["gain"]
                    stored_effect = stored_b["gain"].double() - stored_a["gain"].double()
                    map_effect_sums[contrast.key]["error_sse"] += float(
                        (reconstructed_effect.double() - stored_effect).square().sum().cpu()
                    )
                    map_effect_sums[contrast.key]["effect_sse"] += float(stored_effect.square().sum().cpu())

                    projected_zero = project_channel_delta(delta_h, rank_zero)
                    zero_suff_state = h_a + projected_zero
                    zero_nec_state = h_b - projected_zero
                    _update_max(
                        maxima,
                        contrast.key,
                        "rank_zero_suff_state_max_abs",
                        _max_abs(zero_suff_state, h_a),
                        location,
                    )
                    _update_max(
                        maxima,
                        contrast.key,
                        "rank_zero_nec_state_max_abs",
                        _max_abs(zero_nec_state, h_b),
                        location,
                    )
                    _compare_strict_endpoint(
                        maxima, contrast.key, "rank_zero_suff", _endpoint(zero_suff_state, readout), endpoint_a, location
                    )
                    _compare_strict_endpoint(
                        maxima, contrast.key, "rank_zero_nec", _endpoint(zero_nec_state, readout), endpoint_b, location
                    )

                    projected_identity = project_channel_delta(delta_h, identity)
                    identity_suff_state = h_a + projected_identity
                    identity_nec_state = h_b - projected_identity
                    _update_max(
                        maxima,
                        contrast.key,
                        "identity_suff_state_max_abs",
                        _max_abs(identity_suff_state, h_b),
                        location,
                    )
                    _update_max(
                        maxima,
                        contrast.key,
                        "identity_nec_state_max_abs",
                        _max_abs(identity_nec_state, h_a),
                        location,
                    )
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "identity_suff",
                        _endpoint(identity_suff_state, readout),
                        endpoint_b,
                        location,
                    )
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "identity_nec",
                        _endpoint(identity_nec_state, readout),
                        endpoint_a,
                        location,
                    )

                    projected_random = project_channel_delta(delta_h, random_basis)
                    literal_random_suff = _endpoint(h_a + projected_random, readout)
                    literal_random_nec = _endpoint(h_b - projected_random, readout)
                    canonical_z_suff, canonical_z_nec = intervention_preactivations(
                        h_a,
                        h_b,
                        random_basis,
                        readout["feature"],
                        readout["bias"],
                        readout["space"],
                    )
                    canonical_random_suff = _endpoint_from_preactivation(canonical_z_suff)
                    canonical_random_nec = _endpoint_from_preactivation(canonical_z_nec)
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "random_low_rank_suff_canonical_vs_direct",
                        canonical_random_suff,
                        literal_random_suff,
                        location,
                    )
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "random_low_rank_nec_canonical_vs_direct",
                        canonical_random_nec,
                        literal_random_nec,
                        location,
                    )

                    # Non-gating diagnostic only.  This algebraically split
                    # path has a different CUDA accumulation order from the
                    # literal state patch above and is not used by fitting or
                    # evaluation.
                    random_delta_z = projected_delta_preactivation(
                        delta_h, random_basis, readout["feature"], readout["space"]
                    )
                    fast_random_suff = _endpoint_from_preactivation(endpoint_a["preactivation"] + random_delta_z)
                    fast_random_nec = _endpoint_from_preactivation(endpoint_b["preactivation"] - random_delta_z)
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "random_low_rank_suff_fast_vs_literal",
                        fast_random_suff,
                        literal_random_suff,
                        location,
                    )
                    _compare_strict_endpoint(
                        maxima,
                        contrast.key,
                        "random_low_rank_nec_fast_vs_literal",
                        fast_random_nec,
                        literal_random_nec,
                        location,
                    )
                    tested_frames[contrast.key] += frame_stop - frame_start

    checks = _make_checks(maxima, map_effect_sums, strict, storage)
    algebra_pass = all(row["passed"] for row in checks if row["family"] == "exact_algebra")
    storage_pass = all(row["passed"] for row in checks if row["family"] == "float16_storage")
    all_completed = len(completed_pairs) == int(state_completed.size)
    full_scope = all_completed and args.max_pairs == 0 and state_complete and map_complete
    passed = bool(algebra_pass and storage_pass and orthonormality_error <= strict["rank_128_state_max_abs"])
    return {
        "analysis": "fig4_causal_low_rank_actual_data_integrity_gate",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "optimization_allowed": bool(passed and full_scope),
        "full_production_scope_validated": full_scope,
        "reason_optimization_not_allowed": (
            None
            if passed and full_scope
            else "Integrity checks failed" if not passed else "Only a partial or incomplete cache scope was validated"
        ),
        "created_utc": utc_now(),
        "cache_only": True,
        "model_loaded": False,
        "core_forward_calls": 0,
        "device": str(device),
        "selection": {
            "rule": "completed pairs in deterministic image-major, trajectory-major order",
            "completed_pairs_available": len(completed_pairs),
            "pairs_tested": len(pairs),
            "max_pairs_argument": args.max_pairs,
            "frames_per_pair_per_contrast": N_FRAMES,
            "tested_frames_by_contrast": tested_frames,
        },
        "contrasts": [
            {
                "key": contrast.key,
                "label": contrast.label,
                "scale_a": contrast.scale_a,
                "scale_b": contrast.scale_b,
            }
            for contrast in CONTRASTS
        ],
        "projectors": {
            "rank_zero": "literal U with shape 128 x 0",
            "identity": "explicit torch.eye(128) literal state patch",
            "random_low_rank": {
                "rank": LOW_RANK_TEST_RANK,
                "seed": ANALYSIS_SEED + 910_007,
                "orthonormality_max_abs": orthonormality_error,
                "canonical_implementation": (
                    "construct h_A + P(h_B-h_A) and h_B - P(h_B-h_A), then run each complete "
                    "patched state through the exact frozen tiled readout"
                ),
            },
        },
        "non_gating_split_readout_diagnostic": {
            "description": (
                "R(h_A)+R_without_bias(P delta_h) versus literal R(h_A+P delta_h), and the "
                "corresponding necessity expression. These are equal in real arithmetic but can differ "
                "on CUDA because convolution and addition accumulation orders differ. The split form is "
                "not used by optimization, evaluation, or causal reporting."
            ),
            "gating": False,
            "metrics_by_contrast": {
                contrast.key: {
                    prefix: {
                        name: maxima[contrast.key][f"{prefix}_{name}_max_abs"]
                        for name in ("preactivation", "rate", "gain", "ssi", "expected_spikes")
                    }
                    for prefix in (
                        "random_low_rank_suff_fast_vs_literal",
                        "random_low_rank_nec_fast_vs_literal",
                    )
                }
                for contrast in CONTRASTS
            },
        },
        "configured_tolerances": {
            "strict_intervention": strict,
            "float16_storage": storage,
            "float16_application_note": (
                "The configured dimensionless normalized-map RMSE threshold is applied directly to g and "
                "to RMS-normalized z, rate, and expected-spike errors; SSI and contrast-effect R2 use their "
                "separately configured thresholds."
            ),
        },
        "summary": {
            "exact_algebra_passed": algebra_pass,
            "float16_storage_passed": storage_pass,
            "checks_passed": int(sum(row["passed"] for row in checks)),
            "checks_total": len(checks),
        },
        "checks": checks,
        "worst_case_metrics": maxima,
        "map_effect_sums": map_effect_sums,
        "provenance": {
            "state_cache": str(STATE_CACHE),
            "map_cache": str(MAP_CACHE),
            "readout_cache": str(READOUT_CACHE),
            "readout_cache_sha256": sha256_file(READOUT_CACHE),
            "optimization_config": str(CONFIG_PATH),
            "optimization_config_sha256": sha256_file(CONFIG_PATH),
            "schema_version": configuration.get("schema_version"),
        },
    }


def _failure_payload(args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    return {
        "analysis": "fig4_causal_low_rank_actual_data_integrity_gate",
        "status": "fail",
        "passed": False,
        "optimization_allowed": False,
        "full_production_scope_validated": False,
        "created_utc": utc_now(),
        "cache_only": True,
        "model_loaded": False,
        "core_forward_calls": 0,
        "device": args.device,
        "error_type": type(error).__name__,
        "error": str(error),
        "arguments": vars(args),
    }


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        result = validate(args)
    except Exception as error:
        result = _failure_payload(args, error)
        write_json(TOP_LEVEL_OUTPUT, result)
        write_json(CACHE_OUTPUT, result)
        print(f"integrity gate failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    write_json(TOP_LEVEL_OUTPUT, result)
    write_json(CACHE_OUTPUT, result)
    if not result["passed"]:
        print("integrity gate failed; optimization is forbidden", file=sys.stderr)
        return 2
    if not result["optimization_allowed"]:
        print("integrity checks passed for the requested partial scope; full optimization remains forbidden")
        return 0
    print(f"integrity gate passed for all production pairs: {TOP_LEVEL_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
