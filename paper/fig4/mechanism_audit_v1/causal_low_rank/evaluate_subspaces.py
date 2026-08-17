#!/usr/bin/env python3
"""Evaluate causal ConvGRU subspaces exclusively from the saved exact caches.

The evaluator never imports or runs the core model.  It streams held-out
image--trajectory pairs, applies a spatially shared channel projection, and
passes the patched state difference through the frozen cached RR100 readout.
Screening, crossed validation, cross-scale transfer, and geometric/cross-
contrast summaries are separate resumable CLI stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    ANALYSIS_SEED,
    CONFIG,
    CONTRAST_BY_KEY,
    CONTRASTS,
    EVALUATION,
    FITS,
    N_CHANNELS,
    N_FRAMES,
    N_UNITS,
    OUT,
    READOUT_CACHE,
    SCALES,
    SOURCE_EXACT,
    STATE_CACHE,
    MAP_CACHE,
    ensure_output_dirs,
    exclusive_gpu_analysis_lock,
    haar_basis,
    intervention_preactivations,
    intervention_preactivations_from_delta,
    load_global_gpu_budget,
    normalize_rate,
    principal_angles_deg,
    projector,
    qr_basis,
    rate_map_components,
    record_global_gpu_time,
    scale_index,
    subspace_overlap,
    write_json,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold, load_readout, target_units
from paper.fig4.mechanism_audit_v1.causal_low_rank.optimize_subspaces import (
    SCREENING_RANKS,
    endpoint_denominators,
    fit_dir,
    load_or_compute_movement_pca,
)


RESULTS = EVALUATION / "results"
SCREENING_RANK_JSON = CONFIG / "stage2_ranks.json"
EPS = 1e-30
_GPU_DEADLINE_MONOTONIC = math.inf


class GlobalGPUBudgetReached(RuntimeError):
    """Raised between streaming batches when the shared GPU cap is reached."""


def _gpu_budget_checkpoint() -> None:
    if time.monotonic() >= _GPU_DEADLINE_MONOTONIC:
        raise GlobalGPUBudgetReached(
            "Global authorized causal-low-rank GPU budget reached between batches"
        )
UNIT_FEATURE_TABLE = (
    ROOT
    / "outputs/active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged/unit_feature_table.csv"
)


@dataclass(frozen=True)
class EvaluationRequest:
    stage: str
    contrast_key: str
    fold: int
    rank: int
    method: str
    scale_a: float
    scale_b: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "screening",
            "crossval",
            "shuffled-target",
            "cross-scale",
            "cross-contrast",
            "consolidate",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--contrasts", nargs="*", default=[value.key for value in CONTRASTS])
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--ranks", nargs="*", type=int)
    parser.add_argument("--random-draws", type=int, default=None)
    parser.add_argument("--shuffle-fits", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpu-budget-hours", type=float, default=4.0)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _configuration() -> dict[str, Any]:
    path = CONFIG / "optimization_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare_analysis.py first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_cache_complete() -> None:
    for path in (STATE_CACHE, MAP_CACHE, READOUT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved cache is missing: {path}")
    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        state_done = np.asarray(state["completed_pairs"][:], dtype=bool)
        map_done = np.asarray(maps["completed_pairs"][:], dtype=bool)
        if not bool(state_done.all()) or not bool(map_done.all()):
            raise RuntimeError(
                "Exact cache is incomplete: "
                f"states={int(state_done.sum())}/{state_done.size}, "
                f"maps={int(map_done.sum())}/{map_done.size}"
            )
        if not np.array_equal(state_done, map_done):
            raise RuntimeError("State and map cache completion masks differ")


def _safe_ratio(numerator: float | np.ndarray, denominator: float | np.ndarray) -> float | np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan, dtype=np.float64)
    np.divide(numerator_array, denominator_array, out=result, where=np.abs(denominator_array) > EPS)
    return float(result) if result.ndim == 0 else result


def recovery_r2(residual_sse: float | np.ndarray, effect_sse: float | np.ndarray) -> float | np.ndarray:
    """Unclipped held-out effect recovery, 1 - residual/effect."""
    ratio = _safe_ratio(residual_sse, effect_sse)
    return 1.0 - ratio


def _aggregate_weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(_safe_ratio(values, weights), dtype=np.float64)


def _read_pair_batch(
    state: h5py.File,
    maps: h5py.File,
    pair: tuple[int, int],
    scale_a: float,
    scale_b: float,
    frames: np.ndarray,
    units: np.ndarray | slice = slice(None),
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    image, trajectory = pair
    a_i, b_i = scale_index(scale_a), scale_index(scale_b)
    frame_ids = np.sort(np.asarray(frames, dtype=np.int64))
    result: dict[str, torch.Tensor] = {}
    for suffix, scale_i in (("a", a_i), ("b", b_i)):
        result[f"h_{suffix}"] = torch.as_tensor(
            np.asarray(state["h"][image, trajectory, scale_i, frame_ids], dtype=np.float32),
            device=device,
        )
        for name in ("preactivation", "gain", "ssi", "expected_spikes", "mean_rate"):
            value = np.asarray(maps[name][image, trajectory, scale_i, frame_ids], dtype=np.float32)
            value = value[:, units]
            result[f"{name}_{suffix}"] = torch.as_tensor(value, device=device)
    return result


def _unit_groups() -> np.ndarray:
    labels = np.full(N_UNITS, "low", dtype="U4")
    labels[target_units("high")] = "high"
    return labels


@lru_cache(maxsize=1)
def _unit_observational_metadata() -> dict[str, np.ndarray]:
    """Predeclared per-unit scale preference and 1x-to-3x reversal metadata."""
    with np.load(SOURCE_EXACT, allow_pickle=False) as archive:
        ssi = np.asarray(archive["ssi"], dtype=np.float64)
        expected = np.asarray(archive["expected_spikes"], dtype=np.float64)
        scales = np.asarray(archive["scales"], dtype=np.float64)
    numerator = np.sum(ssi * expected, axis=(0, 1))
    denominator = np.sum(expected, axis=(0, 1))
    curve = np.divide(numerator, np.maximum(denominator, 1e-30))
    optimum_index = np.argmax(curve, axis=0)
    index_1 = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    index_3 = int(np.flatnonzero(np.isclose(scales, 3.0))[0])
    reversal = curve[index_3] - curve[index_1]
    table = pd.read_csv(UNIT_FEATURE_TABLE).sort_values("unit_index")
    if not np.array_equal(table.unit_index.to_numpy(int), np.arange(N_UNITS)):
        raise RuntimeError("Authoritative unit-feature table no longer indexes RR100 in order")
    return {
        "sf_split_metric": pd.to_numeric(table.sf_split_metric, errors="coerce").to_numpy(float),
        "optimal_movement_scale": scales[optimum_index],
        "ssi_at_1x_bits": curve[index_1],
        "ssi_at_3x_bits": curve[index_3],
        "ssi_reversal_1_to_3_bits": reversal,
        "has_1x_to_3x_reversal": reversal < 0,
    }


def _new_accumulator(n_pairs: int) -> dict[str, Any]:
    unit_names = (
        "map_effect",
        "map_suff_residual",
        "map_nec_residual",
        "map_effect_equal",
        "map_suff_residual_equal",
        "map_nec_residual_equal",
        "z_effect",
        "z_suff_residual",
        "z_nec_residual",
    )
    value: dict[str, Any] = {name: np.zeros(N_UNITS, dtype=np.float64) for name in unit_names}
    for condition in ("a", "b", "suff", "nec"):
        value[f"ssi_numerator_{condition}"] = np.zeros(N_UNITS, dtype=np.float64)
        value[f"ssi_weight_{condition}"] = np.zeros(N_UNITS, dtype=np.float64)
        value[f"mean_rate_sum_{condition}"] = np.zeros(N_UNITS, dtype=np.float64)
    pair_names = unit_names + tuple(
        item
        for condition in ("a", "b", "suff", "nec")
        for item in (f"ssi_numerator_{condition}", f"ssi_weight_{condition}", f"mean_rate_sum_{condition}")
    )
    value["pair"] = {name: np.zeros(n_pairs, dtype=np.float64) for name in pair_names}
    value["n_records"] = 0
    return value


def _prediction(
    batch: dict[str, torch.Tensor],
    basis: torch.Tensor | None,
    readout: dict[str, torch.Tensor],
    endpoint: str | None,
) -> dict[str, torch.Tensor]:
    if endpoint == "rank_zero":
        return {
            "z_suff": batch["preactivation_a"],
            "z_nec": batch["preactivation_b"],
            "gain_suff": batch["gain_a"],
            "gain_nec": batch["gain_b"],
            "ssi_suff": batch["ssi_a"],
            "ssi_nec": batch["ssi_b"],
            "expected_suff": batch["expected_spikes_a"],
            "expected_nec": batch["expected_spikes_b"],
            "mean_rate_suff": batch["mean_rate_a"],
            "mean_rate_nec": batch["mean_rate_b"],
        }
    if endpoint == "identity":
        return {
            "z_suff": batch["preactivation_b"],
            "z_nec": batch["preactivation_a"],
            "gain_suff": batch["gain_b"],
            "gain_nec": batch["gain_a"],
            "ssi_suff": batch["ssi_b"],
            "ssi_nec": batch["ssi_a"],
            "expected_suff": batch["expected_spikes_b"],
            "expected_nec": batch["expected_spikes_a"],
            "mean_rate_suff": batch["mean_rate_b"],
            "mean_rate_nec": batch["mean_rate_a"],
        }
    if basis is None:
        raise ValueError("A learned/baseline prediction requires a basis")
    z_suff, z_nec = intervention_preactivations(
        batch["h_a"],
        batch["h_b"],
        basis,
        readout["feature"],
        readout["bias"],
        readout["space"],
    )
    suff = rate_map_components(torch.nn.functional.softplus(z_suff))
    nec = rate_map_components(torch.nn.functional.softplus(z_nec))
    return {
        "z_suff": z_suff,
        "z_nec": z_nec,
        "gain_suff": suff["gain"],
        "gain_nec": nec["gain"],
        "ssi_suff": suff["ssi"],
        "ssi_nec": nec["ssi"],
        "expected_suff": suff["expected_spikes"],
        "expected_nec": nec["expected_spikes"],
        "mean_rate_suff": suff["mean_rate"],
        "mean_rate_nec": nec["mean_rate"],
    }


def _update_accumulator(
    accumulator: dict[str, Any],
    pair_index: int,
    batch: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    target_units_array: np.ndarray,
) -> dict[str, np.ndarray]:
    weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
    map_effect_raw = (batch["gain_b"] - batch["gain_a"]).square().sum(dim=(-2, -1))
    map_suff_raw = (prediction["gain_suff"] - batch["gain_b"]).square().sum(dim=(-2, -1))
    map_nec_raw = (prediction["gain_nec"] - batch["gain_a"]).square().sum(dim=(-2, -1))
    z_effect_raw = (batch["preactivation_b"] - batch["preactivation_a"]).square().sum(dim=(-2, -1))
    z_suff_raw = (prediction["z_suff"] - batch["preactivation_b"]).square().sum(dim=(-2, -1))
    z_nec_raw = (prediction["z_nec"] - batch["preactivation_a"]).square().sum(dim=(-2, -1))
    tensors = {
        "map_effect": map_effect_raw * weight,
        "map_suff_residual": map_suff_raw * weight,
        "map_nec_residual": map_nec_raw * weight,
        "map_effect_equal": map_effect_raw,
        "map_suff_residual_equal": map_suff_raw,
        "map_nec_residual_equal": map_nec_raw,
        "z_effect": z_effect_raw * weight,
        "z_suff_residual": z_suff_raw * weight,
        "z_nec_residual": z_nec_raw * weight,
    }
    raw_record = {
        "map_effect_sse": map_effect_raw.detach().cpu().numpy().astype(np.float32),
        "map_suff_residual_sse": map_suff_raw.detach().cpu().numpy().astype(np.float32),
        "map_nec_residual_sse": map_nec_raw.detach().cpu().numpy().astype(np.float32),
        "z_effect_sse": z_effect_raw.detach().cpu().numpy().astype(np.float32),
        "z_suff_residual_sse": z_suff_raw.detach().cpu().numpy().astype(np.float32),
        "z_nec_residual_sse": z_nec_raw.detach().cpu().numpy().astype(np.float32),
    }
    for name, tensor in tensors.items():
        summed = tensor.double().sum(dim=0).detach().cpu().numpy()
        accumulator[name] += summed
        accumulator["pair"][name][pair_index] += float(summed[target_units_array].sum())

    for condition in ("a", "b", "suff", "nec"):
        if condition in ("a", "b"):
            ssi = batch[f"ssi_{condition}"]
            expected = batch[f"expected_spikes_{condition}"]
            mean_rate = batch[f"mean_rate_{condition}"]
        else:
            ssi = prediction[f"ssi_{condition}"]
            expected = prediction[f"expected_{condition}"]
            mean_rate = prediction[f"mean_rate_{condition}"]
        numerator = (ssi.double() * expected.double()).sum(dim=0).detach().cpu().numpy()
        denominator = expected.double().sum(dim=0).detach().cpu().numpy()
        rate_sum = mean_rate.double().sum(dim=0).detach().cpu().numpy()
        accumulator[f"ssi_numerator_{condition}"] += numerator
        accumulator[f"ssi_weight_{condition}"] += denominator
        accumulator[f"mean_rate_sum_{condition}"] += rate_sum
        accumulator["pair"][f"ssi_numerator_{condition}"][pair_index] += float(
            numerator[target_units_array].sum()
        )
        accumulator["pair"][f"ssi_weight_{condition}"][pair_index] += float(
            denominator[target_units_array].sum()
        )
        accumulator["pair"][f"mean_rate_sum_{condition}"][pair_index] += float(
            rate_sum[target_units_array].sum()
        )
    accumulator["n_records"] += len(batch["ssi_a"])
    return raw_record


def _population_metrics(
    accumulator: dict[str, Any],
    units: np.ndarray,
    request: EvaluationRequest,
    n_pairs: int,
) -> dict[str, Any]:
    units = np.asarray(units, dtype=np.int64)

    def total(name: str) -> float:
        return float(np.asarray(accumulator[name], dtype=np.float64)[units].sum())

    ssi: dict[str, float] = {}
    mean_rate: dict[str, float] = {}
    for condition in ("a", "b", "suff", "nec"):
        ssi[condition] = float(
            _safe_ratio(total(f"ssi_numerator_{condition}"), total(f"ssi_weight_{condition}"))
        )
        mean_rate[condition] = total(f"mean_rate_sum_{condition}") / max(
            int(accumulator["n_records"]) * len(units), 1
        )
    ssi_effect = ssi["b"] - ssi["a"]
    rate_effect = mean_rate["b"] - mean_rate["a"]
    return {
        "stage": request.stage,
        "contrast": request.contrast_key,
        "target_group": CONTRAST_BY_KEY[request.contrast_key].group,
        "fold": request.fold,
        "rank": request.rank,
        "method": request.method,
        "scale_a": request.scale_a,
        "scale_b": request.scale_b,
        "n_test_pairs": n_pairs,
        "n_test_records": int(accumulator["n_records"]),
        "n_target_units": len(units),
        "map_r2_sufficiency": float(recovery_r2(total("map_suff_residual"), total("map_effect"))),
        "map_r2_necessity": float(recovery_r2(total("map_nec_residual"), total("map_effect"))),
        "map_r2_sufficiency_equal_unit": float(
            recovery_r2(total("map_suff_residual_equal"), total("map_effect_equal"))
        ),
        "map_r2_necessity_equal_unit": float(
            recovery_r2(total("map_nec_residual_equal"), total("map_effect_equal"))
        ),
        "preactivation_r2_sufficiency": float(recovery_r2(total("z_suff_residual"), total("z_effect"))),
        "preactivation_r2_necessity": float(recovery_r2(total("z_nec_residual"), total("z_effect"))),
        "ssi_a_bits": ssi["a"],
        "ssi_b_bits": ssi["b"],
        "ssi_sufficiency_bits": ssi["suff"],
        "ssi_necessity_bits": ssi["nec"],
        "ssi_target_effect_bits": ssi_effect,
        "ssi_fraction_transferred": float(_safe_ratio(ssi["suff"] - ssi["a"], ssi_effect)),
        "ssi_fraction_removed": float(_safe_ratio(ssi["b"] - ssi["nec"], ssi_effect)),
        "mean_rate_a": mean_rate["a"],
        "mean_rate_b": mean_rate["b"],
        "mean_rate_sufficiency": mean_rate["suff"],
        "mean_rate_necessity": mean_rate["nec"],
        "mean_rate_target_effect": rate_effect,
        "mean_rate_fraction_transferred": float(
            _safe_ratio(mean_rate["suff"] - mean_rate["a"], rate_effect)
        ),
        "mean_rate_fraction_removed": float(
            _safe_ratio(mean_rate["b"] - mean_rate["nec"], rate_effect)
        ),
        "fractions_are_uncapped": True,
        "r2_is_unclipped": True,
    }


def _per_unit_metrics(
    accumulator: dict[str, Any],
    request: EvaluationRequest,
) -> list[dict[str, Any]]:
    labels = _unit_groups()
    observational = _unit_observational_metadata()
    rows: list[dict[str, Any]] = []
    n_records = max(int(accumulator["n_records"]), 1)
    for unit in range(N_UNITS):
        ssi = {}
        rate = {}
        for condition in ("a", "b", "suff", "nec"):
            ssi[condition] = float(
                _safe_ratio(
                    accumulator[f"ssi_numerator_{condition}"][unit],
                    accumulator[f"ssi_weight_{condition}"][unit],
                )
            )
            rate[condition] = float(accumulator[f"mean_rate_sum_{condition}"][unit] / n_records)
        ssi_effect = ssi["b"] - ssi["a"]
        rate_effect = rate["b"] - rate["a"]
        rows.append(
            {
                "stage": request.stage,
                "contrast": request.contrast_key,
                "fold": request.fold,
                "rank": request.rank,
                "method": request.method,
                "unit_index": unit,
                "historical_sf_group": labels[unit],
                "sf_split_metric": float(observational["sf_split_metric"][unit]),
                "is_target_population": labels[unit] == CONTRAST_BY_KEY[request.contrast_key].group,
                "observed_optimal_movement_scale": float(observational["optimal_movement_scale"][unit]),
                "observed_ssi_at_1x_bits": float(observational["ssi_at_1x_bits"][unit]),
                "observed_ssi_at_3x_bits": float(observational["ssi_at_3x_bits"][unit]),
                "observed_ssi_reversal_1_to_3_bits": float(
                    observational["ssi_reversal_1_to_3_bits"][unit]
                ),
                "observed_has_1x_to_3x_reversal": bool(
                    observational["has_1x_to_3x_reversal"][unit]
                ),
                "map_r2_sufficiency": float(
                    recovery_r2(accumulator["map_suff_residual"][unit], accumulator["map_effect"][unit])
                ),
                "map_r2_necessity": float(
                    recovery_r2(accumulator["map_nec_residual"][unit], accumulator["map_effect"][unit])
                ),
                "preactivation_r2_sufficiency": float(
                    recovery_r2(accumulator["z_suff_residual"][unit], accumulator["z_effect"][unit])
                ),
                "preactivation_r2_necessity": float(
                    recovery_r2(accumulator["z_nec_residual"][unit], accumulator["z_effect"][unit])
                ),
                "ssi_a_bits": ssi["a"],
                "ssi_b_bits": ssi["b"],
                "ssi_sufficiency_bits": ssi["suff"],
                "ssi_necessity_bits": ssi["nec"],
                "ssi_target_effect_bits": ssi_effect,
                "ssi_fraction_transferred": float(_safe_ratio(ssi["suff"] - ssi["a"], ssi_effect)),
                "ssi_fraction_removed": float(_safe_ratio(ssi["b"] - ssi["nec"], ssi_effect)),
                "mean_rate_a": rate["a"],
                "mean_rate_b": rate["b"],
                "mean_rate_sufficiency": rate["suff"],
                "mean_rate_necessity": rate["nec"],
                "mean_rate_target_effect": rate_effect,
                "mean_rate_fraction_transferred": float(
                    _safe_ratio(rate["suff"] - rate["a"], rate_effect)
                ),
                "mean_rate_fraction_removed": float(
                    _safe_ratio(rate["b"] - rate["nec"], rate_effect)
                ),
            }
        )
    return rows


def _representative_index(details: dict[str, np.ndarray], units: np.ndarray) -> tuple[int, int]:
    units = np.asarray(units, dtype=np.int64)
    paired_weight = 0.5 * (
        details["expected_a"][:, units].astype(np.float64)
        + details["expected_b"][:, units].astype(np.float64)
    )
    effect = details["map_effect_sse"][:, units].astype(np.float64) * paired_weight
    residual = details["map_suff_residual_sse"][:, units].astype(np.float64) * paired_weight
    r2 = np.asarray(recovery_r2(residual, effect), dtype=np.float64)
    valid = np.isfinite(effect) & np.isfinite(r2) & (effect > EPS)
    if not np.any(valid):
        return 0, int(units[0])
    effect_log = np.log10(np.maximum(effect, EPS))
    med_effect = np.nanmedian(np.where(valid, effect_log, np.nan))
    med_r2 = np.nanmedian(np.where(valid, r2, np.nan))
    mad_effect = np.nanmedian(np.abs(np.where(valid, effect_log, np.nan) - med_effect))
    mad_r2 = np.nanmedian(np.abs(np.where(valid, r2, np.nan) - med_r2))
    distance = (
        np.abs(effect_log - med_effect) / max(float(mad_effect), 1e-12)
        + np.abs(r2 - med_r2) / max(float(mad_r2), 1e-12)
    )
    distance[~valid] = np.inf
    record_index, local_unit = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(record_index), int(units[local_unit])


@torch.no_grad()
def _representative_maps(
    state: h5py.File,
    maps: h5py.File,
    pair: tuple[int, int],
    frame: int,
    scale_a: float,
    scale_b: float,
    unit: int,
    basis: torch.Tensor,
    readout: dict[str, torch.Tensor],
    device: str,
) -> dict[str, np.ndarray]:
    batch = _read_pair_batch(
        state,
        maps,
        pair,
        scale_a,
        scale_b,
        np.asarray([frame]),
        units=slice(None),
        device=device,
    )
    prediction = _prediction(batch, basis, readout, endpoint=None)
    gain_a = batch["gain_a"][0, unit].cpu().numpy().astype(np.float32)
    gain_b = batch["gain_b"][0, unit].cpu().numpy().astype(np.float32)
    gain_s = prediction["gain_suff"][0, unit].cpu().numpy().astype(np.float32)
    gain_n = prediction["gain_nec"][0, unit].cpu().numpy().astype(np.float32)
    z_a = batch["preactivation_a"][0, unit].cpu().numpy().astype(np.float32)
    z_b = batch["preactivation_b"][0, unit].cpu().numpy().astype(np.float32)
    z_s = prediction["z_suff"][0, unit].cpu().numpy().astype(np.float32)
    z_n = prediction["z_nec"][0, unit].cpu().numpy().astype(np.float32)
    return {
        "representative_gain_a": gain_a,
        "representative_gain_b": gain_b,
        "representative_target_minus_baseline": gain_b - gain_a,
        "representative_gain_sufficiency": gain_s,
        "representative_sufficiency_residual": gain_b - gain_s,
        "representative_gain_necessity": gain_n,
        "representative_necessity_residual": gain_n - gain_a,
        "representative_preactivation_a": z_a,
        "representative_preactivation_b": z_b,
        "representative_preactivation_sufficiency": z_s,
        "representative_preactivation_necessity": z_n,
        "representative_rate_a": torch.nn.functional.softplus(
            batch["preactivation_a"][0, unit]
        ).cpu().numpy().astype(np.float32),
        "representative_rate_b": torch.nn.functional.softplus(
            batch["preactivation_b"][0, unit]
        ).cpu().numpy().astype(np.float32),
        "representative_rate_sufficiency": torch.nn.functional.softplus(
            prediction["z_suff"][0, unit]
        ).cpu().numpy().astype(np.float32),
        "representative_rate_necessity": torch.nn.functional.softplus(
            prediction["z_nec"][0, unit]
        ).cpu().numpy().astype(np.float32),
        "representative_ssi_a_bits": np.asarray(float(batch["ssi_a"][0, unit].cpu())),
        "representative_ssi_b_bits": np.asarray(float(batch["ssi_b"][0, unit].cpu())),
        "representative_ssi_sufficiency_bits": np.asarray(
            float(prediction["ssi_suff"][0, unit].cpu())
        ),
        "representative_ssi_necessity_bits": np.asarray(
            float(prediction["ssi_nec"][0, unit].cpu())
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _result_directory(request: EvaluationRequest) -> Path:
    return (
        RESULTS
        / request.stage
        / request.contrast_key
        / f"fold_{request.fold}"
        / f"rank_{request.rank:03d}"
    )


@torch.no_grad()
def evaluate_basis(
    request: EvaluationRequest,
    pairs: list[tuple[int, int]],
    basis_array: np.ndarray | None,
    device: str,
    frame_batch: int,
    *,
    save_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    target = target_units(CONTRAST_BY_KEY[request.contrast_key].group)
    readout = load_readout(device, units=None)
    basis = None if basis_array is None else torch.as_tensor(basis_array, dtype=torch.float32, device=device)
    endpoint = request.method if request.method in ("rank_zero", "identity") else None
    accumulator = _new_accumulator(len(pairs))
    n_records = len(pairs) * N_FRAMES
    details: dict[str, np.ndarray] | None = None
    if save_predictions:
        details = {
            "image_position": np.empty(n_records, dtype=np.int16),
            "trajectory_position": np.empty(n_records, dtype=np.int16),
            "frame_position": np.empty(n_records, dtype=np.int16),
        }
        for name in (
            "map_effect_sse",
            "map_suff_residual_sse",
            "map_nec_residual_sse",
            "z_effect_sse",
            "z_suff_residual_sse",
            "z_nec_residual_sse",
            "ssi_a",
            "ssi_b",
            "ssi_suff",
            "ssi_nec",
            "expected_a",
            "expected_b",
            "expected_suff",
            "expected_nec",
            "mean_rate_a",
            "mean_rate_b",
            "mean_rate_suff",
            "mean_rate_nec",
        ):
            details[name] = np.empty((n_records, N_UNITS), dtype=np.float32)

    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        for pair_index, pair in enumerate(pairs):
            _gpu_budget_checkpoint()
            for start in range(0, N_FRAMES, frame_batch):
                stop = min(start + frame_batch, N_FRAMES)
                frames = np.arange(start, stop, dtype=np.int64)
                batch = _read_pair_batch(
                    state,
                    maps,
                    pair,
                    request.scale_a,
                    request.scale_b,
                    frames,
                    units=slice(None),
                    device=device,
                )
                prediction = _prediction(batch, basis, readout, endpoint)
                raw = _update_accumulator(accumulator, pair_index, batch, prediction, target)
                if details is not None:
                    rows = slice(pair_index * N_FRAMES + start, pair_index * N_FRAMES + stop)
                    details["image_position"][rows] = pair[0]
                    details["trajectory_position"][rows] = pair[1]
                    details["frame_position"][rows] = frames
                    for name, value in raw.items():
                        details[name][rows] = value
                    for condition in ("a", "b"):
                        details[f"ssi_{condition}"][rows] = batch[f"ssi_{condition}"].cpu().numpy()
                        details[f"expected_{condition}"][rows] = batch[f"expected_spikes_{condition}"].cpu().numpy()
                        details[f"mean_rate_{condition}"][rows] = batch[f"mean_rate_{condition}"].cpu().numpy()
                    for condition in ("suff", "nec"):
                        details[f"ssi_{condition}"][rows] = prediction[f"ssi_{condition}"].cpu().numpy()
                        details[f"expected_{condition}"][rows] = prediction[f"expected_{condition}"].cpu().numpy()
                        details[f"mean_rate_{condition}"][rows] = prediction[f"mean_rate_{condition}"].cpu().numpy()

        representative: dict[str, Any] = {}
        if details is not None and basis is not None:
            record, unit = _representative_index(details, target)
            pair = (int(details["image_position"][record]), int(details["trajectory_position"][record]))
            frame = int(details["frame_position"][record])
            representative = {
                "representative_record_index": record,
                "representative_image_position": pair[0],
                "representative_trajectory_position": pair[1],
                "representative_frame_position": frame,
                "representative_unit_index": unit,
                "representative_selection_rule": (
                    "target-population record closest in robust standardized distance to the median "
                    "paired-expected-weighted log map-effect energy and median sufficient map R2"
                ),
                **_representative_maps(
                    state,
                    maps,
                    pair,
                    frame,
                    request.scale_a,
                    request.scale_b,
                    unit,
                    basis,
                    readout,
                    device,
                ),
            }

    aggregate = _population_metrics(accumulator, target, request, len(pairs))
    per_unit = _per_unit_metrics(accumulator, request)
    pair_metrics = {
        "image_positions": [pair[0] for pair in pairs],
        "trajectory_positions": [pair[1] for pair in pairs],
        **accumulator["pair"],
    }
    payload: dict[str, Any] = {"pair_metrics": pair_metrics}
    if details is not None:
        payload["details"] = details
        payload["representative"] = representative
    return aggregate, per_unit, payload


def _save_result(
    request: EvaluationRequest,
    aggregate: dict[str, Any],
    per_unit: list[dict[str, Any]],
    payload: dict[str, Any],
    basis_path: Path | None,
) -> None:
    destination = _result_directory(request)
    destination.mkdir(parents=True, exist_ok=True)
    per_unit_path = destination / f"{request.method}_per_unit.csv"
    _write_csv(per_unit_path, per_unit)
    result = {
        "aggregate": aggregate,
        "per_unit_file": str(per_unit_path),
        "basis_path": None if basis_path is None else str(basis_path),
        "pair_metrics": payload["pair_metrics"],
    }
    write_json(destination / f"{request.method}.json", result)
    if request.method == "learned" and "details" in payload:
        fit_destination = fit_dir(request.stage, request.contrast_key, request.fold, request.rank)
        fit_destination.mkdir(parents=True, exist_ok=True)
        arrays = {
            "unit_indices": np.arange(N_UNITS, dtype=np.int64),
            **payload["details"],
            **payload.get("representative", {}),
        }
        np.savez_compressed(fit_destination / "test_predictions.npz", **arrays)


def _learned_basis(stage: str, contrast: str, fold: int, rank: int) -> tuple[np.ndarray, Path]:
    path = fit_dir(stage, contrast, fold, rank) / "U.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.shape != (N_CHANNELS, rank):
        raise ValueError(f"Basis shape mismatch at {path}: {value.shape}")
    return value, path


def _readout_svd_basis(group: str, rank: int) -> np.ndarray:
    units = target_units(group)
    with np.load(READOUT_CACHE, allow_pickle=False) as archive:
        weights = np.asarray(archive["feature_weights"], dtype=np.float64)[units]
    # full_matrices=True is required for same-rank comparisons when k exceeds
    # the number of target units (the higher-SF population has only 29 units,
    # while screening includes k=32).  The additional axes span the exact
    # readout-null space and are deterministically supplied by LAPACK.
    _, _, right = np.linalg.svd(weights, full_matrices=True)
    return right[:rank].T.astype(np.float32)


def _pca_basis(
    request: EvaluationRequest,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    rank: int,
    device: str,
) -> np.ndarray:
    with h5py.File(STATE_CACHE, "r") as state:
        full = (
            load_or_compute_movement_pca(
                state,
                pairs,
                request.contrast_key,
                request.fold,
                scale_a,
                scale_b,
                device,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    if full.shape != (N_CHANNELS, N_CHANNELS):
        raise ValueError(f"Movement-PCA cache shape mismatch: {full.shape}")
    return full[:, :rank]


def _evaluate_one(
    request: EvaluationRequest,
    pairs: list[tuple[int, int]],
    train_pairs: list[tuple[int, int]],
    device: str,
    frame_batch: int,
    overwrite: bool,
) -> bool:
    destination = _result_directory(request) / f"{request.method}.json"
    if destination.is_file() and not overwrite:
        return False
    basis: np.ndarray | None
    basis_path: Path | None = None
    if request.method in ("rank_zero", "identity"):
        basis = None
    elif request.method == "learned":
        basis, basis_path = _learned_basis(request.stage, request.contrast_key, request.fold, request.rank)
    elif request.method == "movement_pca":
        basis = _pca_basis(
            request,
            train_pairs,
            request.scale_a,
            request.scale_b,
            request.rank,
            device,
        )
    elif request.method == "readout_svd":
        basis = _readout_svd_basis(CONTRAST_BY_KEY[request.contrast_key].group, request.rank)
    else:
        raise KeyError(request.method)
    aggregate, per_unit, payload = evaluate_basis(
        request,
        pairs,
        basis,
        device,
        frame_batch,
        save_predictions=request.method == "learned",
    )
    if request.method == "learned":
        metadata_path = fit_dir(request.stage, request.contrast_key, request.fold, request.rank) / "fit_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            aggregate["best_validation_loss"] = metadata.get("best_validation_loss")
            aggregate["selected_restart"] = metadata.get("selected_restart")
            aggregate["selected_initialization"] = metadata.get("selected_initialization")
    _save_result(request, aggregate, per_unit, payload, basis_path)
    return True


@torch.no_grad()
def evaluate_random_haar(
    request: EvaluationRequest,
    pairs: list[tuple[int, int]],
    draws: int,
    device: str,
    frame_batch: int,
    *,
    basis_arrays: list[np.ndarray] | None = None,
    basis_seeds: np.ndarray | None = None,
    method_name: str = "random_haar",
) -> list[dict[str, Any]]:
    """Evaluate a basis collection in one cache stream; retain aggregate rows only."""
    contrast = CONTRAST_BY_KEY[request.contrast_key]
    units = target_units(contrast.group)
    readout = load_readout(device, units=units)
    if basis_arrays is None:
        seeds = np.asarray(
            [
                ANALYSIS_SEED
                + 500_000
                + 100_000 * list(CONTRAST_BY_KEY).index(request.contrast_key)
                + 10_000 * request.fold
                + 100 * request.rank
                + draw
                for draw in range(draws)
            ],
            dtype=np.int64,
        )
        basis_arrays = [haar_basis(request.rank, int(seed)) for seed in seeds]
    else:
        draws = len(basis_arrays)
        if basis_seeds is None:
            seeds = np.arange(draws, dtype=np.int64)
        else:
            seeds = np.asarray(basis_seeds, dtype=np.int64)
            if len(seeds) != draws:
                raise ValueError("basis_seeds and basis_arrays differ in length")
    bases = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in basis_arrays]
    names = (
        "map_effect",
        "map_suff",
        "map_nec",
        "map_effect_equal",
        "map_suff_equal",
        "map_nec_equal",
        "z_effect",
        "z_suff",
        "z_nec",
        "ssi_suff_num",
        "ssi_suff_den",
        "ssi_nec_num",
        "ssi_nec_den",
        "rate_suff_sum",
        "rate_nec_sum",
    )
    accum = {name: np.zeros(draws, dtype=np.float64) for name in names}
    endpoint = {name: 0.0 for name in ("ssi_a_num", "ssi_a_den", "ssi_b_num", "ssi_b_den", "rate_a", "rate_b")}
    n_records = 0
    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        for pair in pairs:
            _gpu_budget_checkpoint()
            for start in range(0, N_FRAMES, frame_batch):
                frames = np.arange(start, min(start + frame_batch, N_FRAMES))
                batch = _read_pair_batch(
                    state,
                    maps,
                    pair,
                    request.scale_a,
                    request.scale_b,
                    frames,
                    units=units,
                    device=device,
                )
                weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
                effect = (batch["gain_b"] - batch["gain_a"]).square().sum((-2, -1))
                z_effect = (batch["preactivation_b"] - batch["preactivation_a"]).square().sum((-2, -1))
                weighted_effect = float((effect * weight).sum().cpu())
                equal_effect = float(effect.sum().cpu())
                weighted_z_effect = float((z_effect * weight).sum().cpu())
                accum["map_effect"] += weighted_effect
                accum["map_effect_equal"] += equal_effect
                accum["z_effect"] += weighted_z_effect
                endpoint["ssi_a_num"] += float((batch["ssi_a"] * batch["expected_spikes_a"]).sum().cpu())
                endpoint["ssi_a_den"] += float(batch["expected_spikes_a"].sum().cpu())
                endpoint["ssi_b_num"] += float((batch["ssi_b"] * batch["expected_spikes_b"]).sum().cpu())
                endpoint["ssi_b_den"] += float(batch["expected_spikes_b"].sum().cpu())
                endpoint["rate_a"] += float(batch["mean_rate_a"].sum().cpu())
                endpoint["rate_b"] += float(batch["mean_rate_b"].sum().cpu())
                for draw, basis in enumerate(bases):
                    z_s, z_n = intervention_preactivations(
                        batch["h_a"],
                        batch["h_b"],
                        basis,
                        readout["feature"],
                        readout["bias"],
                        readout["space"],
                    )
                    suff = rate_map_components(torch.nn.functional.softplus(z_s))
                    nec = rate_map_components(torch.nn.functional.softplus(z_n))
                    map_s = (suff["gain"] - batch["gain_b"]).square().sum((-2, -1))
                    map_n = (nec["gain"] - batch["gain_a"]).square().sum((-2, -1))
                    z_s_res = (z_s - batch["preactivation_b"]).square().sum((-2, -1))
                    z_n_res = (z_n - batch["preactivation_a"]).square().sum((-2, -1))
                    accum["map_suff"][draw] += float((map_s * weight).sum().cpu())
                    accum["map_nec"][draw] += float((map_n * weight).sum().cpu())
                    accum["map_suff_equal"][draw] += float(map_s.sum().cpu())
                    accum["map_nec_equal"][draw] += float(map_n.sum().cpu())
                    accum["z_suff"][draw] += float((z_s_res * weight).sum().cpu())
                    accum["z_nec"][draw] += float((z_n_res * weight).sum().cpu())
                    accum["ssi_suff_num"][draw] += float((suff["ssi"] * suff["expected_spikes"]).sum().cpu())
                    accum["ssi_suff_den"][draw] += float(suff["expected_spikes"].sum().cpu())
                    accum["ssi_nec_num"][draw] += float((nec["ssi"] * nec["expected_spikes"]).sum().cpu())
                    accum["ssi_nec_den"][draw] += float(nec["expected_spikes"].sum().cpu())
                    accum["rate_suff_sum"][draw] += float(suff["mean_rate"].sum().cpu())
                    accum["rate_nec_sum"][draw] += float(nec["mean_rate"].sum().cpu())
                n_records += len(frames)

    ssi_a = float(_safe_ratio(endpoint["ssi_a_num"], endpoint["ssi_a_den"]))
    ssi_b = float(_safe_ratio(endpoint["ssi_b_num"], endpoint["ssi_b_den"]))
    rate_denominator = max(n_records * len(units), 1)
    rate_a = endpoint["rate_a"] / rate_denominator
    rate_b = endpoint["rate_b"] / rate_denominator
    rows = []
    for draw in range(draws):
        ssi_s = float(_safe_ratio(accum["ssi_suff_num"][draw], accum["ssi_suff_den"][draw]))
        ssi_n = float(_safe_ratio(accum["ssi_nec_num"][draw], accum["ssi_nec_den"][draw]))
        rate_s = accum["rate_suff_sum"][draw] / rate_denominator
        rate_n = accum["rate_nec_sum"][draw] / rate_denominator
        rows.append(
            {
                "stage": request.stage,
                "contrast": request.contrast_key,
                "target_group": contrast.group,
                "fold": request.fold,
                "rank": request.rank,
                "method": method_name,
                "draw": draw,
                "seed": int(seeds[draw]),
                "scale_a": request.scale_a,
                "scale_b": request.scale_b,
                "n_test_pairs": len(pairs),
                "map_r2_sufficiency": float(recovery_r2(accum["map_suff"][draw], accum["map_effect"][draw])),
                "map_r2_necessity": float(recovery_r2(accum["map_nec"][draw], accum["map_effect"][draw])),
                "map_r2_sufficiency_equal_unit": float(
                    recovery_r2(accum["map_suff_equal"][draw], accum["map_effect_equal"][draw])
                ),
                "map_r2_necessity_equal_unit": float(
                    recovery_r2(accum["map_nec_equal"][draw], accum["map_effect_equal"][draw])
                ),
                "preactivation_r2_sufficiency": float(
                    recovery_r2(accum["z_suff"][draw], accum["z_effect"][draw])
                ),
                "preactivation_r2_necessity": float(
                    recovery_r2(accum["z_nec"][draw], accum["z_effect"][draw])
                ),
                "ssi_a_bits": ssi_a,
                "ssi_b_bits": ssi_b,
                "ssi_sufficiency_bits": ssi_s,
                "ssi_necessity_bits": ssi_n,
                "ssi_target_effect_bits": ssi_b - ssi_a,
                "ssi_fraction_transferred": float(_safe_ratio(ssi_s - ssi_a, ssi_b - ssi_a)),
                "ssi_fraction_removed": float(_safe_ratio(ssi_b - ssi_n, ssi_b - ssi_a)),
                "mean_rate_a": rate_a,
                "mean_rate_b": rate_b,
                "mean_rate_sufficiency": rate_s,
                "mean_rate_necessity": rate_n,
                "mean_rate_target_effect": rate_b - rate_a,
                "mean_rate_fraction_transferred": float(_safe_ratio(rate_s - rate_a, rate_b - rate_a)),
                "mean_rate_fraction_removed": float(_safe_ratio(rate_b - rate_n, rate_b - rate_a)),
                "fractions_are_uncapped": True,
                "r2_is_unclipped": True,
            }
        )
    return rows


def _random_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {key: rows[0][key] for key in rows[0] if key not in ("draw", "seed")}
    metrics = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in ("fold", "rank", "draw", "seed", "n_test_pairs")
    ]
    for key in metrics:
        value = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmean(value))
        summary[f"{key}_median"] = float(np.nanmedian(value))
        summary[f"{key}_ci_low"] = float(np.nanpercentile(value, 2.5))
        summary[f"{key}_ci_high"] = float(np.nanpercentile(value, 97.5))
    summary["n_random_draws"] = len(rows)
    return summary


def _save_random(request: EvaluationRequest, rows: list[dict[str, Any]]) -> None:
    destination = _result_directory(request)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / f"{request.method}.json", {"aggregate": _random_summary(rows), "draws": rows})


def _derangement(length: int, seed: int) -> np.ndarray:
    if length < 2:
        raise ValueError("A shuffled-target split requires at least two pairs")
    rng = np.random.default_rng(int(seed))
    identity = np.arange(length, dtype=np.int64)
    for _ in range(10_000):
        candidate = rng.permutation(length)
        if np.all(candidate != identity):
            return candidate.astype(np.int64)
    # Deterministic fallback with no fixed points.
    return np.roll(identity, 1)


def _shuffled_fit_dir(contrast: str, fold: int, rank: int, draw: int) -> Path:
    return (
        FITS
        / "shuffled_target"
        / contrast
        / f"fold_{fold}"
        / f"rank_{rank:03d}"
        / f"shuffle_{draw:03d}"
    )


def _attach_state_delta_from_donor(
    batch: dict[str, torch.Tensor],
    state: h5py.File,
    donor_pair: tuple[int, int],
    scale_a: float,
    scale_b: float,
    frames: np.ndarray,
    device: str,
) -> None:
    """Attach a donor pair's delta without replacing the target state anchors."""
    image, trajectory = donor_pair
    frame_ids = np.sort(np.asarray(frames, dtype=np.int64))
    donor: dict[str, torch.Tensor] = {}
    for suffix, scale in (("a", scale_a), ("b", scale_b)):
        donor[suffix] = torch.as_tensor(
            np.asarray(
                state["h"][image, trajectory, scale_index(scale), frame_ids],
                dtype=np.float32,
            ),
            device=device,
        )
    batch["intervention_delta_h"] = donor["b"] - donor["a"]


def _shuffled_loss(
    batch: dict[str, torch.Tensor],
    basis: torch.Tensor,
    readout: dict[str, torch.Tensor],
    denominator: float,
    n_total_records: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_s, z_n = intervention_preactivations_from_delta(
        batch["h_a"],
        batch["h_b"],
        batch["intervention_delta_h"],
        basis,
        readout["feature"],
        readout["bias"],
        readout["space"],
    )
    gain_s = normalize_rate(torch.nn.functional.softplus(z_s))
    gain_n = normalize_rate(torch.nn.functional.softplus(z_n))
    weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
    scale = float(n_total_records) / max(len(batch["h_a"]), 1)
    sufficiency = scale * (
        (gain_s - batch["gain_b"]).square() * weight[..., None, None]
    ).sum() / max(denominator, EPS)
    necessity = scale * (
        (gain_n - batch["gain_a"]).square() * weight[..., None, None]
    ).sum() / max(denominator, EPS)
    return 0.5 * (sufficiency + necessity), sufficiency, necessity


@torch.no_grad()
def _evaluate_shuffled_validation(
    state: h5py.File,
    maps: h5py.File,
    target_pairs: list[tuple[int, int]],
    donor_order: np.ndarray,
    contrast,
    units: np.ndarray,
    bases: torch.Tensor,
    readout: dict[str, torch.Tensor],
    device: str,
    frame_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = endpoint_denominators(
        maps,
        target_pairs,
        contrast.scale_a,
        contrast.scale_b,
        units,
    ).sufficiency
    numerator_s = np.zeros(len(bases), dtype=np.float64)
    numerator_n = np.zeros(len(bases), dtype=np.float64)
    for target_index, target_pair in enumerate(target_pairs):
        _gpu_budget_checkpoint()
        donor_pair = target_pairs[int(donor_order[target_index])]
        for start in range(0, N_FRAMES, frame_batch):
            frames = np.arange(start, min(start + frame_batch, N_FRAMES))
            batch = _read_pair_batch(
                state,
                maps,
                target_pair,
                contrast.scale_a,
                contrast.scale_b,
                frames,
                units=units,
                device=device,
            )
            _attach_state_delta_from_donor(
                batch,
                state,
                donor_pair,
                contrast.scale_a,
                contrast.scale_b,
                frames,
                device,
            )
            weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
            for restart, basis in enumerate(bases):
                z_s, z_n = intervention_preactivations_from_delta(
                    batch["h_a"],
                    batch["h_b"],
                    batch["intervention_delta_h"],
                    basis,
                    readout["feature"],
                    readout["bias"],
                    readout["space"],
                )
                gain_s = normalize_rate(torch.nn.functional.softplus(z_s))
                gain_n = normalize_rate(torch.nn.functional.softplus(z_n))
                numerator_s[restart] += float(
                    (((gain_s - batch["gain_b"]).square() * weight[..., None, None]).sum()).cpu()
                )
                numerator_n[restart] += float(
                    (((gain_n - batch["gain_a"]).square() * weight[..., None, None]).sum()).cpu()
                )
    loss_s = numerator_s / max(float(denominator), EPS)
    loss_n = numerator_n / max(float(denominator), EPS)
    return 0.5 * (loss_s + loss_n), loss_s, loss_n


def _shuffled_initial_values(
    pca_full: torch.Tensor,
    rank: int,
    parameter_seed: int,
    device: str,
) -> tuple[list[torch.Tensor], list[int]]:
    """Create the same three preregistered starts as an ordinary causal fit."""
    if rank < 1 or rank > N_CHANNELS:
        raise ValueError(f"rank must be in [1, {N_CHANNELS}], got {rank}")
    if tuple(pca_full.shape) != (N_CHANNELS, N_CHANNELS):
        raise ValueError(
            f"movement-PCA basis must have shape {(N_CHANNELS, N_CHANNELS)}, "
            f"got {tuple(pca_full.shape)}"
        )
    parameter_seeds = [parameter_seed + 1, parameter_seed + 2, parameter_seed + 3]
    initial_values: list[torch.Tensor] = []
    for restart, restart_seed in enumerate(parameter_seeds):
        if restart == 1:
            initial_values.append(pca_full[:, :rank].clone())
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(restart_seed))
            initial_values.append(
                qr_basis(
                    torch.randn(
                        (N_CHANNELS, rank), generator=generator, device=device
                    )
                )
            )
    return initial_values, parameter_seeds


def _fit_one_shuffled_target(
    *,
    contrast,
    contrast_index: int,
    fold_index: int,
    rank: int,
    draw: int,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    destination = _shuffled_fit_dir(contrast.key, fold_index, rank, draw)
    basis_path = destination / "U.npy"
    metadata_path = destination / "fit_metadata.json"
    if basis_path.is_file() and metadata_path.is_file() and not args.overwrite:
        return np.asarray(np.load(basis_path), dtype=np.float32), json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    destination.mkdir(parents=True, exist_ok=True)
    fold = load_fold(fold_index)
    train_pairs = fold.train.pairs
    validation_pairs = fold.validation.pairs
    shuffle_seed = (
        ANALYSIS_SEED
        + 1_000_000
        + 100_000 * contrast_index
        + 10_000 * fold_index
        + 100 * rank
        + draw
    )
    train_order = _derangement(len(train_pairs), shuffle_seed)
    validation_order = _derangement(len(validation_pairs), shuffle_seed + 5_000_000)
    parameter_seed = shuffle_seed + 7_000_000
    settings = config["optimizer"]
    max_steps = int(args.max_steps or settings["maximum_steps"])
    validation_interval = int(settings["validation_every_steps"])
    patience = int(settings["early_stopping_patience_validation_evaluations"])
    minimum_improvement = float(settings["minimum_validation_improvement"])
    frame_batch = int(args.frame_batch_size or settings["frame_batch_size"])
    rng = np.random.default_rng(parameter_seed + 101)
    units = target_units(contrast.group)
    readout = load_readout(args.device, units=units)
    curve: list[dict[str, Any]] = []
    started = time.monotonic()
    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        pca_full = load_or_compute_movement_pca(
            state,
            train_pairs,
            contrast.key,
            fold_index,
            contrast.scale_a,
            contrast.scale_b,
            args.device,
        )
        initial_values, parameter_seeds = _shuffled_initial_values(
            pca_full,
            rank,
            parameter_seed,
            args.device,
        )
        parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(value) for value in initial_values]
        )
        optimizers = [
            torch.optim.AdamW(
                [parameters[restart]],
                lr=float(settings["learning_rate"]),
                weight_decay=float(settings.get("weight_decay", 0.0)),
            )
            for restart in range(3)
        ]
        best_loss = np.full(3, np.inf, dtype=np.float64)
        best_step = np.zeros(3, dtype=np.int64)
        best_basis = np.zeros((3, N_CHANNELS, rank), dtype=np.float32)
        stale = np.zeros(3, dtype=np.int64)
        converged = np.zeros(3, dtype=bool)
        last_train = [(torch.tensor(float("nan")),) * 3 for _ in range(3)]
        last_gradient = [float("nan")] * 3
        denominator_record = endpoint_denominators(
            maps,
            train_pairs,
            contrast.scale_a,
            contrast.scale_b,
            units,
        )
        denominator = float(denominator_record.sufficiency)
        for step in range(max_steps):
            _gpu_budget_checkpoint()
            target_index = int(rng.integers(len(train_pairs)))
            target_pair = train_pairs[target_index]
            donor_pair = train_pairs[int(train_order[target_index])]
            frames = np.sort(
                rng.choice(N_FRAMES, size=min(frame_batch, N_FRAMES), replace=False)
            )
            batch = _read_pair_batch(
                state,
                maps,
                target_pair,
                contrast.scale_a,
                contrast.scale_b,
                frames,
                units=units,
                device=args.device,
            )
            _attach_state_delta_from_donor(
                batch,
                state,
                donor_pair,
                contrast.scale_a,
                contrast.scale_b,
                frames,
                args.device,
            )
            train_values = list(last_train)
            gradient_norms = list(last_gradient)
            for restart in range(3):
                if converged[restart]:
                    continue
                optimizers[restart].zero_grad(set_to_none=True)
                basis = qr_basis(parameters[restart])
                loss, loss_s, loss_n = _shuffled_loss(
                    batch,
                    basis,
                    readout,
                    denominator,
                    denominator_record.n_records,
                )
                loss.backward()
                gradient_norms[restart] = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameters[restart]], float(settings["gradient_clip_norm"])
                    )
                )
                optimizers[restart].step()
                train_values[restart] = (loss.detach(), loss_s.detach(), loss_n.detach())
            last_train = train_values
            last_gradient = gradient_norms
            should_validate = (
                step == 0
                or (step + 1) % validation_interval == 0
                or step + 1 == max_steps
            )
            if should_validate:
                bases = torch.stack([qr_basis(parameter) for parameter in parameters])
                validation, validation_s, validation_n = _evaluate_shuffled_validation(
                    state,
                    maps,
                    validation_pairs,
                    validation_order,
                    contrast,
                    units,
                    bases,
                    readout,
                    args.device,
                    frame_batch,
                )
                for restart in range(3):
                    improved = validation[restart] < best_loss[restart] - minimum_improvement
                    if improved:
                        best_loss[restart] = validation[restart]
                        best_step[restart] = step + 1
                        best_basis[restart] = (
                            bases[restart].detach().cpu().numpy().astype(np.float32)
                        )
                        stale[restart] = 0
                    else:
                        stale[restart] += 1
                    converged[restart] = stale[restart] >= patience
                    curve.append(
                        {
                            "step": step + 1,
                            "restart": restart,
                            "initialization": "movement_pca" if restart == 1 else f"random_{restart}",
                            "train_total_loss_estimate": float(train_values[restart][0].cpu()),
                            "train_sufficiency_loss_estimate": float(train_values[restart][1].cpu()),
                            "train_necessity_loss_estimate": float(train_values[restart][2].cpu()),
                            "validation_total_loss": float(validation[restart]),
                            "validation_sufficiency_loss": float(validation_s[restart]),
                            "validation_necessity_loss": float(validation_n[restart]),
                            "gradient_norm_before_clip": gradient_norms[restart],
                            "best_validation_loss": float(best_loss[restart]),
                            "stale_validation_evaluations": int(stale[restart]),
                            "converged": bool(converged[restart]),
                        }
                    )
                if np.all(converged):
                    break
    if not np.all(np.isfinite(best_loss)):
        raise RuntimeError("Shuffled-target fit never completed a validation evaluation")
    selected_restart = int(np.argmin(best_loss))
    selected_basis = best_basis[selected_restart]
    np.save(basis_path, selected_basis)
    np.save(destination / "P.npy", projector(selected_basis))
    _write_csv(destination / "training_curve.csv", curve)
    metadata = {
        "stage": "shuffled_target",
        "contrast": contrast.key,
        "fold": fold_index,
        "rank": rank,
        "shuffle_draw": draw,
        "shuffle_seed": shuffle_seed,
        "parameter_seed": parameter_seed,
        "restart_seeds": parameter_seeds,
        "restarts": 3,
        "initializations": ["random_0", "movement_pca", "random_2"],
        "selected_restart": selected_restart,
        "selected_initialization": (
            "movement_pca" if selected_restart == 1 else f"random_{selected_restart}"
        ),
        "train_pair_target_positions": train_pairs,
        "train_pair_donor_positions": [train_pairs[int(value)] for value in train_order],
        "validation_pair_target_positions": validation_pairs,
        "validation_pair_donor_positions": [validation_pairs[int(value)] for value in validation_order],
        "frames_remain_together": True,
        "pair_marginals_preserved": True,
        "no_fixed_pair_matches": bool(
            np.all(train_order != np.arange(len(train_order)))
            and np.all(validation_order != np.arange(len(validation_order)))
        ),
        "selection_used_test_metrics": False,
        "selection_metric": "minimum independently shuffled validation map objective across three restarts",
        "best_validation_loss": float(best_loss[selected_restart]),
        "best_step": int(best_step[selected_restart]),
        "all_best_validation_losses": best_loss,
        "all_best_steps": best_step,
        "elapsed_seconds": time.monotonic() - started,
        "maximum_steps": max_steps,
    }
    write_json(metadata_path, metadata)
    return selected_basis, metadata


def evaluate_shuffled_target(args: argparse.Namespace) -> None:
    config = _configuration()
    selected = _selected_ranks(args)
    # The protocol explicitly requires k=4 and k=8.  Also fit every rank that
    # can be selected for the final result so that the learned-vs-shuffled
    # claim is always a same-rank comparison (including an elbow at k=1/2).
    ranks = sorted({4, 8, *selected}) if selected else []
    folds = [0, 1, 2, 3] if args.folds is None else args.folds
    required_fits = int(
        args.shuffle_fits
        if args.shuffle_fits is not None
        else config["baselines"]["shuffled_target"]["fits_per_rank"]
    )
    if required_fits < 20:
        raise ValueError("The final shuffled-target baseline requires at least 20 fits")
    stage_started = time.monotonic()
    status_rows: list[dict[str, Any]] = []
    budget_reached = False
    for contrast_index, contrast in enumerate(CONTRASTS):
        if contrast.key not in args.contrasts:
            continue
        for fold_index in folds:
            fold = load_fold(fold_index)
            for rank in ranks:
                bases_by_draw: dict[int, np.ndarray] = {}
                seeds_by_draw: dict[int, int] = {}
                # Load completed draws by their actual draw identifier before
                # resuming.  A simple list is unsafe when an interrupted run
                # leaves a hole (for example draws 0 and 2 but not draw 1).
                for draw in range(required_fits):
                    path = _shuffled_fit_dir(contrast.key, fold_index, rank, draw) / "U.npy"
                    metadata_path = path.with_name("fit_metadata.json")
                    if path.is_file() and metadata_path.is_file():
                        bases_by_draw[draw] = np.asarray(np.load(path), dtype=np.float32)
                        seeds_by_draw[draw] = int(
                            json.loads(metadata_path.read_text(encoding="utf-8"))["shuffle_seed"]
                        )
                for draw in range(required_fits):
                    if draw in bases_by_draw:
                        continue
                    if time.monotonic() >= _GPU_DEADLINE_MONOTONIC:
                        budget_reached = True
                        break
                    print(
                        f"fit shuffled-target {contrast.key} fold={fold_index} rank={rank} "
                        f"draw={draw + 1}/{required_fits}",
                        flush=True,
                    )
                    try:
                        basis, metadata = _fit_one_shuffled_target(
                            contrast=contrast,
                            contrast_index=contrast_index,
                            fold_index=fold_index,
                            rank=rank,
                            draw=draw,
                            args=args,
                            config=config,
                        )
                    except GlobalGPUBudgetReached:
                        budget_reached = True
                        break
                    bases_by_draw[draw] = basis
                    seeds_by_draw[draw] = int(metadata["shuffle_seed"])
                completed_draws = sorted(bases_by_draw)
                bases = [bases_by_draw[draw] for draw in completed_draws]
                seeds = [seeds_by_draw[draw] for draw in completed_draws]
                if bases:
                    request = EvaluationRequest(
                        "crossval",
                        contrast.key,
                        fold_index,
                        rank,
                        "shuffled_target_learned",
                        contrast.scale_a,
                        contrast.scale_b,
                    )
                    try:
                        rows = evaluate_random_haar(
                            request,
                            fold.test.pairs,
                            len(bases),
                            args.device,
                            args.frame_batch_size,
                            basis_arrays=bases,
                            basis_seeds=np.asarray(seeds, dtype=np.int64),
                            method_name="shuffled_target_learned",
                        )
                    except GlobalGPUBudgetReached:
                        budget_reached = True
                        rows = []
                    for draw, row in zip(completed_draws, rows, strict=True):
                        row["shuffle_draw"] = draw
                        row["validation_selected_only"] = True
                    if rows:
                        _save_random(request, rows)
                status_rows.append(
                    {
                        "contrast": contrast.key,
                        "fold": fold_index,
                        "rank": rank,
                        "required_fits": required_fits,
                        "completed_fits": len(bases),
                        "complete": len(bases) >= required_fits,
                    }
                )
                if budget_reached:
                    break
            if budget_reached:
                break
        if budget_reached:
            break
    status = {
        "required_fits_per_contrast_fold_rank": required_fits,
        "selected_ranks": ranks,
        "gpu_budget_hours": args.gpu_budget_hours,
        "elapsed_hours": (time.monotonic() - stage_started) / 3600.0,
        "budget_reached": budget_reached,
        "jobs": status_rows,
        "status": "budget_stop_partial" if budget_reached else "complete",
    }
    write_json(EVALUATION / "shuffled_target_status.json", status)
    write_json(OUT / "shuffled_target_status.json", status)
    consolidate_results()
    if budget_reached:
        raise GlobalGPUBudgetReached(
            "Global GPU budget reached during the shuffled-target baseline"
        )


def deterministic_validation_elbow(validation_by_rank: dict[int, float]) -> dict[str, Any]:
    ranks = np.asarray(sorted(validation_by_rank), dtype=np.int64)
    if len(ranks) < 3:
        raise ValueError("At least three learned ranks are required for an elbow")
    score = np.asarray([validation_by_rank[int(rank)] for rank in ranks], dtype=np.float64)
    x = np.log2(ranks.astype(np.float64))
    x = (x - x[0]) / max(float(x[-1] - x[0]), EPS)
    if float(np.nanmax(score) - np.nanmin(score)) <= 1e-12:
        return {"ambiguous": True, "elbow_rank": None, "adjacent_ranks": ranks.tolist(), "distances": [0.0] * len(ranks)}
    y = (score - np.nanmin(score)) / max(float(np.nanmax(score) - np.nanmin(score)), EPS)
    chord = y[0] + (y[-1] - y[0]) * x
    distance = y - chord
    candidates = np.flatnonzero(np.isclose(distance, np.nanmax(distance), atol=1e-10, rtol=1e-8))
    if len(candidates) != 1:
        return {
            "ambiguous": True,
            "elbow_rank": None,
            "adjacent_ranks": ranks.tolist(),
            "distances": distance.tolist(),
        }
    index = int(candidates[0])
    if index == 0:
        adjacent = ranks[:2]
    elif index == len(ranks) - 1:
        adjacent = ranks[-2:]
    else:
        adjacent = ranks[[index - 1, index + 1]]
    return {
        "ambiguous": False,
        "elbow_rank": int(ranks[index]),
        "adjacent_ranks": adjacent.astype(int).tolist(),
        "distances": distance.tolist(),
    }


def _screening_decision(allow_missing: bool) -> dict[str, Any]:
    config = _configuration()
    ranks = [int(value) for value in config["rank_sweep"]["learned_screening_ranks"]]
    validation_by_rank: dict[int, list[float]] = {rank: [] for rank in ranks}
    missing: list[str] = []
    for contrast in CONTRASTS:
        for rank in ranks:
            path = fit_dir("screening", contrast.key, 0, rank) / "fit_metadata.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            validation_by_rank[rank].append(1.0 - float(metadata["best_validation_loss"]))
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing {len(missing)} screening fit metadata files; first: {missing[0]}")
    complete_validation = {
        rank: float(np.mean(values)) for rank, values in validation_by_rank.items() if len(values) == len(CONTRASTS)
    }
    if len(complete_validation) < 3:
        if allow_missing:
            return {"status": "incomplete", "missing": missing, "ranks": []}
        raise RuntimeError("Insufficient complete validation ranks to select Stage 2")
    elbow = deterministic_validation_elbow(complete_validation)
    if elbow["ambiguous"]:
        selected = sorted(complete_validation)
    else:
        selected = sorted(set(elbow["adjacent_ranks"]) | {4, 8, 16})

    per_contrast_gate: dict[str, Any] = {}
    stop_rule_complete = True
    expected_gate_ranks = len([rank for rank in ranks if rank <= 16])
    for contrast in CONTRASTS:
        values = []
        for rank in ranks:
            if rank > 16:
                continue
            path = _result_directory(
                EvaluationRequest("screening", contrast.key, 0, rank, "learned", contrast.scale_a, contrast.scale_b)
            ) / "learned.json"
            if not path.is_file():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
            values.append(
                {
                    "rank": rank,
                    "sufficiency": float(row["map_r2_sufficiency"]),
                    "necessity": float(row["map_r2_necessity"]),
                }
            )
        passes = any(value["sufficiency"] >= 0.40 and value["necessity"] >= 0.40 for value in values)
        complete = len(values) == expected_gate_ranks
        stop_rule_complete &= complete
        per_contrast_gate[contrast.key] = {
            "passes_40_percent_by_rank16": passes,
            "complete": complete,
            "values": values,
        }
    stop = stop_rule_complete and bool(per_contrast_gate) and all(
        not value["passes_40_percent_by_rank16"] for value in per_contrast_gate.values()
    )
    result = {
        "status": (
            "stopped_no_compact_subspace"
            if stop
            else "stage2_predeclared" if stop_rule_complete else "stage2_predeclared_stop_rule_pending"
        ),
        "selection_used_test_metrics": False,
        "selection_source": "best validation map loss from screening fits",
        "validation_recovery_by_rank": complete_validation,
        "elbow": elbow,
        "ranks": [] if stop else selected,
        "mandatory_ranks": [4, 8, 16],
        "screening_stop_rule": per_contrast_gate,
        "screening_stop_rule_complete": stop_rule_complete,
        "stop": stop,
        "stop_statement": "No compact ConvGRU channel subspace was found." if stop else None,
        "missing": missing,
    }
    write_json(SCREENING_RANK_JSON, result)
    write_json(OUT / "stage2_ranks.json", result)
    return result


def _evaluate_stage(args: argparse.Namespace, stage: str) -> None:
    config = _configuration()
    contrasts = [CONTRAST_BY_KEY[key] for key in args.contrasts]
    if stage == "screening":
        folds = [int(config["rank_sweep"]["screening_fold"])]
        learned_ranks = list(SCREENING_RANKS) if args.ranks is None else args.ranks
    else:
        folds = [0, 1, 2, 3] if args.folds is None else args.folds
        if args.ranks is not None:
            learned_ranks = args.ranks
        else:
            if not SCREENING_RANK_JSON.is_file():
                raise FileNotFoundError("Run screening evaluation to write stage2_ranks.json")
            stage2 = json.loads(SCREENING_RANK_JSON.read_text(encoding="utf-8"))
            learned_ranks = stage2["ranks"]
    if args.folds is not None and stage == "screening":
        folds = args.folds

    random_draws = args.random_draws
    if random_draws is None:
        random_draws = int(config["baselines"]["random_haar"]["draws_per_selected_rank"])
    for contrast in contrasts:
        for fold_index in folds:
            fold = load_fold(int(fold_index))
            endpoint_requests = (
                EvaluationRequest(stage, contrast.key, fold_index, 0, "rank_zero", contrast.scale_a, contrast.scale_b),
                EvaluationRequest(stage, contrast.key, fold_index, 128, "identity", contrast.scale_a, contrast.scale_b),
            )
            for request in endpoint_requests:
                print(f"evaluate {request.stage} {request.contrast_key} fold={request.fold} {request.method}", flush=True)
                _evaluate_one(
                    request,
                    fold.test.pairs,
                    fold.train.pairs,
                    args.device,
                    args.frame_batch_size,
                    args.overwrite,
                )
            for rank in learned_ranks:
                for method in ("learned", "movement_pca", "readout_svd"):
                    request = EvaluationRequest(
                        stage, contrast.key, fold_index, int(rank), method, contrast.scale_a, contrast.scale_b
                    )
                    try:
                        print(
                            f"evaluate {stage} {contrast.key} fold={fold_index} rank={rank} {method}",
                            flush=True,
                        )
                        _evaluate_one(
                            request,
                            fold.test.pairs,
                            fold.train.pairs,
                            args.device,
                            args.frame_batch_size,
                            args.overwrite,
                        )
                    except FileNotFoundError:
                        if not args.allow_missing:
                            raise
                if stage == "crossval" and not args.skip_random:
                    request = EvaluationRequest(
                        stage,
                        contrast.key,
                        fold_index,
                        int(rank),
                        "random_haar",
                        contrast.scale_a,
                        contrast.scale_b,
                    )
                    destination = _result_directory(request) / "random_haar.json"
                    if args.overwrite or not destination.is_file():
                        print(
                            f"evaluate {stage} {contrast.key} fold={fold_index} rank={rank} "
                            f"random_haar n={random_draws}",
                            flush=True,
                        )
                        rows = evaluate_random_haar(
                            request,
                            fold.test.pairs,
                            random_draws,
                            args.device,
                            args.frame_batch_size,
                        )
                        _save_random(request, rows)
    if stage == "screening":
        _screening_decision(args.allow_missing)
    consolidate_results()


def _load_aggregate_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregate"], payload.get("draws", [])


def consolidate_results() -> None:
    rank_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    per_unit_rows: list[dict[str, Any]] = []
    representative_metadata: list[dict[str, Any]] = []
    representative_arrays: list[np.ndarray] = []
    map_names = (
        "representative_gain_a",
        "representative_gain_b",
        "representative_target_minus_baseline",
        "representative_gain_sufficiency",
        "representative_sufficiency_residual",
        "representative_gain_necessity",
        "representative_necessity_residual",
    )
    for path in sorted(RESULTS.rglob("*.json")):
        aggregate, draws = _load_aggregate_result(path)
        rank_rows.append(aggregate)
        if aggregate["method"] != "learned":
            baseline_rows.extend(draws if draws else [aggregate])
        if not draws:
            per_unit_path = path.with_name(f"{aggregate['method']}_per_unit.csv")
            if per_unit_path.is_file() and per_unit_path.stat().st_size:
                with per_unit_path.open(newline="", encoding="utf-8") as handle:
                    per_unit_rows.extend(list(csv.DictReader(handle)))
        if aggregate["method"] == "learned":
            prediction = fit_dir(
                aggregate["stage"], aggregate["contrast"], int(aggregate["fold"]), int(aggregate["rank"])
            ) / "test_predictions.npz"
            if prediction.is_file():
                with np.load(prediction, allow_pickle=False) as archive:
                    if all(name in archive for name in map_names):
                        representative_metadata.append(
                            {
                                "stage": aggregate["stage"],
                                "contrast": aggregate["contrast"],
                                "fold": aggregate["fold"],
                                "rank": aggregate["rank"],
                                "image_position": int(archive["representative_image_position"]),
                                "trajectory_position": int(archive["representative_trajectory_position"]),
                                "frame_position": int(archive["representative_frame_position"]),
                                "unit_index": int(archive["representative_unit_index"]),
                            }
                        )
                        representative_arrays.append(np.stack([archive[name] for name in map_names]))

    _write_csv(EVALUATION / "rank_summary.csv", rank_rows)
    _write_csv(OUT / "rank_summary.csv", rank_rows)
    _write_csv(EVALUATION / "baseline_results.csv", baseline_rows)
    _write_csv(OUT / "baseline_results.csv", baseline_rows)
    _write_csv(EVALUATION / "per_unit_results.csv", per_unit_rows)
    _write_csv(OUT / "per_unit_results.csv", per_unit_rows)
    if representative_arrays:
        payload = {
            "maps": np.stack(representative_arrays).astype(np.float32),
            "map_names": np.asarray(map_names),
            "metadata_json": np.asarray(json.dumps(representative_metadata)),
        }
        np.savez_compressed(EVALUATION / "objective_representative_exact_maps.npz", **payload)
        np.savez_compressed(OUT / "objective_representative_exact_maps.npz", **payload)


def _selected_ranks(args: argparse.Namespace) -> list[int]:
    if args.ranks is not None:
        return [int(value) for value in args.ranks]
    if not SCREENING_RANK_JSON.is_file():
        raise FileNotFoundError(SCREENING_RANK_JSON)
    return [int(value) for value in json.loads(SCREENING_RANK_JSON.read_text(encoding="utf-8"))["ranks"]]


def evaluate_cross_scale(args: argparse.Namespace) -> None:
    ranks = _selected_ranks(args)
    folds = [0, 1, 2, 3] if args.folds is None else args.folds
    definitions = {
        "low_0_to_2": (0.0, (0.5, 1.0, 2.0, 3.0)),
        "high_0_to_1": (0.0, (0.5, 1.0, 2.0, 3.0)),
        "high_1_to_3": (1.0, (2.0, 3.0)),
    }
    rows: list[dict[str, Any]] = []
    for key in args.contrasts:
        contrast = CONTRAST_BY_KEY[key]
        recipient, donors = definitions[key]
        for fold_index in folds:
            fold = load_fold(fold_index)
            for rank in ranks:
                try:
                    basis, _ = _learned_basis("crossval", key, fold_index, rank)
                except FileNotFoundError:
                    if args.allow_missing:
                        continue
                    raise
                for donor in donors:
                    request = EvaluationRequest(
                        "cross_scale", key, fold_index, rank, "learned", recipient, donor
                    )
                    aggregate, _, _ = evaluate_basis(
                        request,
                        fold.test.pairs,
                        basis,
                        args.device,
                        args.frame_batch_size,
                        save_predictions=False,
                    )
                    aggregate["training_scale_a"] = contrast.scale_a
                    aggregate["training_scale_b"] = contrast.scale_b
                    aggregate["is_training_contrast"] = bool(
                        np.isclose(recipient, contrast.scale_a) and np.isclose(donor, contrast.scale_b)
                    )
                    rows.append(aggregate)
    _write_csv(EVALUATION / "cross_scale_results.csv", rows)
    _write_csv(OUT / "cross_scale_results.csv", rows)


def _random_overlap_distribution(rank: int, draws: int, seed: int) -> np.ndarray:
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        first = haar_basis(rank, seed + 2 * draw)
        second = haar_basis(rank, seed + 2 * draw + 1)
        values[draw] = subspace_overlap(first, second)
    return values


def evaluate_cross_structure(args: argparse.Namespace) -> None:
    ranks = _selected_ranks(args)
    folds = [0, 1, 2, 3] if args.folds is None else args.folds
    stability_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for rank in ranks:
            available: dict[int, np.ndarray] = {}
            for fold_index in folds:
                try:
                    available[fold_index], _ = _learned_basis("crossval", contrast.key, fold_index, rank)
                except FileNotFoundError:
                    if not args.allow_missing:
                        raise
            null = _random_overlap_distribution(rank, 1000, ANALYSIS_SEED + 900_000 + rank)
            for first_fold, second_fold in combinations(sorted(available), 2):
                first, second = available[first_fold], available[second_fold]
                angles = principal_angles_deg(first, second)
                stability_rows.append(
                    {
                        "contrast": contrast.key,
                        "rank": rank,
                        "fold_a": first_fold,
                        "fold_b": second_fold,
                        "projector_overlap": subspace_overlap(first, second),
                        "principal_angle_mean_deg": float(np.mean(angles)),
                        "principal_angle_median_deg": float(np.median(angles)),
                        "principal_angle_max_deg": float(np.max(angles)),
                        "principal_angles_deg": ";".join(f"{value:.8g}" for value in angles),
                        "random_overlap_mean": float(np.mean(null)),
                        "random_overlap_ci_low": float(np.percentile(null, 2.5)),
                        "random_overlap_ci_high": float(np.percentile(null, 97.5)),
                        "analytic_random_expectation": rank / N_CHANNELS,
                    }
                )
    _write_csv(EVALUATION / "subspace_stability.csv", stability_rows)
    _write_csv(OUT / "subspace_stability.csv", stability_rows)

    cross_rows: list[dict[str, Any]] = []
    for fold_index in folds:
        fold = load_fold(fold_index)
        for rank in ranks:
            bases: dict[str, np.ndarray] = {}
            for contrast in CONTRASTS:
                try:
                    bases[contrast.key], _ = _learned_basis("crossval", contrast.key, fold_index, rank)
                except FileNotFoundError:
                    if not args.allow_missing:
                        raise
            for source_key, source_basis in bases.items():
                for target_key, target_basis in bases.items():
                    if source_key == target_key:
                        continue
                    target_contrast = CONTRAST_BY_KEY[target_key]
                    angles = principal_angles_deg(source_basis, target_basis)
                    request = EvaluationRequest(
                        "cross_contrast",
                        target_key,
                        fold_index,
                        rank,
                        f"source_{source_key}",
                        target_contrast.scale_a,
                        target_contrast.scale_b,
                    )
                    aggregate, _, _ = evaluate_basis(
                        request,
                        fold.test.pairs,
                        source_basis,
                        args.device,
                        args.frame_batch_size,
                        save_predictions=False,
                    )
                    cross_rows.append(
                        {
                            "source_contrast": source_key,
                            "target_contrast": target_key,
                            "fold": fold_index,
                            "rank": rank,
                            "source_rank": rank,
                            "geometry_target_rank": rank,
                            "geometry_same_rank_comparison": True,
                            "projector_overlap_denominator": rank,
                            "projector_overlap": subspace_overlap(source_basis, target_basis),
                            "principal_angle_mean_deg": float(np.mean(angles)),
                            "principal_angle_median_deg": float(np.median(angles)),
                            "principal_angle_max_deg": float(np.max(angles)),
                            **{
                                key: value
                                for key, value in aggregate.items()
                                if key
                                in {
                                    "map_r2_sufficiency",
                                    "map_r2_necessity",
                                    "preactivation_r2_sufficiency",
                                    "preactivation_r2_necessity",
                                    "ssi_fraction_transferred",
                                    "ssi_fraction_removed",
                                }
                            },
                        }
                    )
    _write_csv(EVALUATION / "cross_contrast_results.csv", cross_rows)
    _write_csv(OUT / "cross_contrast_results.csv", cross_rows)


def _run(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    if args.stage == "consolidate":
        consolidate_results()
        return 0
    _check_cache_complete()
    if args.stage in ("screening", "all"):
        _evaluate_stage(args, "screening")
    if args.stage in ("crossval", "all"):
        _evaluate_stage(args, "crossval")
    if args.stage in ("shuffled-target", "all"):
        evaluate_shuffled_target(args)
    if args.stage in ("cross-scale", "all"):
        evaluate_cross_scale(args)
    if args.stage in ("cross-contrast", "all"):
        evaluate_cross_structure(args)
    consolidate_results()
    return 0


def main() -> int:
    global _GPU_DEADLINE_MONOTONIC
    args = parse_args()
    ensure_output_dirs()
    if args.stage == "consolidate":
        consolidate_results()
        return 0
    if not (0.0 < float(args.gpu_budget_hours) <= 10.0):
        raise ValueError("--gpu-budget-hours must be in (0, 10]")
    configured_limit = float(_configuration()["gpu_budget"]["hard_limit_hours"])
    hard_limit = min(configured_limit, float(args.gpu_budget_hours), 10.0)
    with exclusive_gpu_analysis_lock():
        budget = load_global_gpu_budget(hard_limit)
        consumed = float(budget["total_conservative_gpu_hours"])
        if str(args.device).startswith("cuda") and consumed >= hard_limit:
            raise GlobalGPUBudgetReached(
                f"Global GPU budget already reached ({consumed:.3f}/{hard_limit:.3f} h)"
            )
        started = time.monotonic()
        completed = False
        if str(args.device).startswith("cuda"):
            _GPU_DEADLINE_MONOTONIC = started + (hard_limit - consumed) * 3600.0
        else:
            _GPU_DEADLINE_MONOTONIC = math.inf
        try:
            result = _run(args)
            completed = True
            return result
        finally:
            elapsed = time.monotonic() - started
            if str(args.device).startswith("cuda"):
                record_global_gpu_time(
                    f"evaluation:{args.stage}",
                    elapsed,
                    hard_limit_hours=hard_limit,
                    details={
                        "contrasts": list(args.contrasts),
                        "folds": args.folds,
                        "ranks": args.ranks,
                        "completed": completed,
                    },
                )
            _GPU_DEADLINE_MONOTONIC = math.inf


if __name__ == "__main__":
    raise SystemExit(main())
