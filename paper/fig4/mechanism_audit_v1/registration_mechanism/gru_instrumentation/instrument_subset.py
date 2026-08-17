#!/usr/bin/env python3
"""Replay the frozen Figure 4 core and reduce exact ConvGRU terms online.

This command never writes a dense full-subset activation cache.  Every
image--trajectory pair is reduced to compressed P/Q term energies and spatial
registration metrics.  Full recurrent terms are retained only for three
outcome-blind, predeclared visualization examples.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    ANALYSIS_SEED,
    CONTRASTS,
    N_FRAMES,
    OUT as CAUSAL_OUT,
    READOUT_CACHE,
    SCALES,
    STATE_CACHE,
    exclusive_gpu_analysis_lock,
    json_ready,
    load_global_gpu_budget,
    record_global_gpu_time,
    write_json,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold, target_units
from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import scaled_histories
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.equations import (
    GRUStepTerms,
    replay_convgru_cell,
    term_energy_summary,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.pilot_selection import (
    IMAGE_STRATIFICATION_METRIC,
    IMAGE_TABLE as PILOT_IMAGE_TABLE,
    TRAJECTORY_STRATIFICATION_METRIC,
    TRAJECTORY_TABLE as PILOT_TRAJECTORY_TABLE,
    select_pilot,
    write_pilot_selection,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.registration import (
    apply_shift_calibration,
    fit_shift_calibration,
    internal_step_eye_displacements,
    normalized_multichannel_xcorr,
    registration_step_metrics,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
PARTS = OUT / "gru_instrumentation/parts"
EXAMPLES = OUT / "gru_instrumentation/full_examples"
CONSENSUS = OUT / "consensus_projectors.npz"
IMAGE_TABLE = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/image_feature_table.csv"
)
MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"
HARD_GPU_LIMIT_HOURS = 10.0
INSTRUMENTATION_SCHEMA = "fig4-gru-registration-v4"
RANK = 8
TERM_NAMES = (
    "h_previous",
    "update_gate",
    "reset_gate",
    "candidate_current_preactivation",
    "candidate_recurrent_preactivation",
    "candidate_state",
    "retained_state_contribution",
    "new_state_contribution",
    "h_t",
)
TERM_META_COLUMNS = (
    "fold",
    "image_position",
    "trajectory_position",
    "scale_position",
    "frame_position",
    "internal_step",
    "contrast_index",
)
REGISTRATION_COLUMNS = (
    "scope",
    "contrast",
    "fold",
    "image_position",
    "trajectory_position",
    "scale",
    "frame_position",
    "internal_step",
    "subspace",
    "eye_delta_x_deg",
    "eye_delta_y_deg",
    "expected_feature_shift_x_px",
    "expected_feature_shift_y_px",
    "raw_zero_lag_correlation",
    "raw_best_lag_correlation",
    "raw_lag_x_px",
    "raw_lag_y_px",
    "raw_at_search_boundary",
    "raw_peak_sharpness",
    "raw_residual_mismatch",
    "recurrent_zero_lag_correlation",
    "recurrent_best_lag_correlation",
    "recurrent_lag_x_px",
    "recurrent_lag_y_px",
    "recurrent_at_search_boundary",
    "recurrent_peak_sharpness",
    "recurrent_residual_mismatch",
    "transport_x_px",
    "transport_y_px",
    "expected_outside_search_window",
    "zero_lag_alignment_improvement",
    "best_lag_alignment_improvement",
    "valid",
)
REGISTRATION_METRIC_NAMES = (
    "raw_zero_lag_correlation",
    "raw_best_lag_correlation",
    "raw_lag_x_px",
    "raw_lag_y_px",
    "raw_peak_sharpness",
    "raw_residual_mismatch",
    "recurrent_zero_lag_correlation",
    "recurrent_best_lag_correlation",
    "recurrent_lag_x_px",
    "recurrent_lag_y_px",
    "recurrent_peak_sharpness",
    "recurrent_residual_mismatch",
    "transport_x_px",
    "transport_y_px",
    "zero_lag_alignment_improvement",
    "best_lag_alignment_improvement",
)


class ComputeLimitReached(RuntimeError):
    """Raised only between complete streaming batches."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scope", choices=("heldout", "complete-consensus"), default="heldout")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--max-lag-px", type=int, default=4)
    parser.add_argument("--random-draws", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-pairs", type=int, default=0, help="Zero means all pending pairs.")
    parser.add_argument("--overwrite-parts", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--max-cache-state-error", type=float, default=3e-3)
    parser.add_argument("--minimum-next-batch-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-calibration-r2", type=float, default=0.80)
    parser.add_argument("--maximum-calibration-error-px", type=float, default=0.50)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return ANALYSIS_SEED + int.from_bytes(digest[:4], "little")


def _orthonormal_random(label: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(label))
    value, r = np.linalg.qr(rng.standard_normal((128, RANK)), mode="reduced")
    sign = np.where(np.diag(r) < 0, -1.0, 1.0)
    return (value * sign[None]).astype(np.float32)


def _readout_svd(group: str) -> np.ndarray:
    units = target_units(group)
    with np.load(READOUT_CACHE, allow_pickle=False) as archive:
        weights = np.asarray(archive["feature_weights"], dtype=np.float64)[units]
    _, _, right = np.linalg.svd(weights, full_matrices=True)
    return right[:RANK].T.astype(np.float32)


def _validate_basis(value: np.ndarray, label: str) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float32)
    if basis.shape != (128, RANK):
        raise ValueError(f"{label}: expected (128,8), found {basis.shape}")
    if not np.allclose(basis.T @ basis, np.eye(RANK), atol=2e-4, rtol=2e-4):
        raise ValueError(f"{label}: basis is not orthonormal")
    return basis


def _fold_basis(contrast: str, fold: int) -> np.ndarray:
    path = (
        CAUSAL_OUT
        / "fits/crossval"
        / contrast
        / f"fold_{int(fold)}"
        / "rank_008/U.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return _validate_basis(np.load(path), str(path))


def _consensus_basis(contrast: str) -> np.ndarray:
    if not CONSENSUS.is_file():
        raise FileNotFoundError(CONSENSUS)
    with np.load(CONSENSUS, allow_pickle=False) as archive:
        key = f"U__{contrast}"
        if key not in archive:
            raise RuntimeError(
                f"Consensus gate did not save {contrast}; complete-consensus replay is not authorized"
            )
        return _validate_basis(np.asarray(archive[key]), key)


def _job_plan(scope: str) -> list[dict[str, Any]]:
    if scope == "heldout":
        result = []
        for fold in range(4):
            for pair in load_fold(fold).test.pairs:
                result.append({"fold": fold, "pair": pair})
        return result
    return [
        {"fold": -1, "pair": (image, trajectory)}
        for image in range(8)
        for trajectory in range(24)
    ]


def _preflight_projectors(scope: str, jobs: list[dict[str, Any]]) -> None:
    folds = sorted({int(job["fold"]) for job in jobs})
    for contrast in CONTRASTS:
        if scope == "heldout":
            for fold in folds:
                _fold_basis(contrast.key, fold)
        else:
            _consensus_basis(contrast.key)


def _bases_for(contrast: Any, fold: int, random_draws: int) -> dict[str, torch.Tensor]:
    learned = (
        _fold_basis(contrast.key, fold)
        if fold >= 0
        else _consensus_basis(contrast.key)
    )
    values: dict[str, np.ndarray] = {
        "learned_p": learned,
        # registration_step_metrics recognizes this name and applies Q=I-P.
        "learned_q": learned,
        "readout_svd": _readout_svd(contrast.group),
    }
    for draw in range(int(random_draws)):
        label = f"random_{draw:02d}"
        values[label] = _orthonormal_random(
            f"registration:{contrast.key}:fold={fold}:draw={draw}"
        )
    return {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in values.items()
    }


def _load_model(device: str) -> tuple[RealTraceMatrixScorer, Any, Any]:
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=device,
        strict=True,
        mcfarland_outputs=MCFARLAND,
    )
    model = scorer.model.model.eval()
    if len(model.recurrent.cells) != 1:
        raise RuntimeError("Figure 4 audit is defined only for the frozen one-layer ConvGRU")
    return scorer, model, model.recurrent.cells[0]


def _core_input(model: Any, x: torch.Tensor, behavior: torch.Tensor) -> torch.Tensor:
    value = model.frontend(x)
    value = model.convnet(value)
    if model.modulator is not None:
        value = model.modulator(value, behavior)
    if value.shape[1:] != (256, 8, 64, 64):
        raise RuntimeError(f"Unexpected exact ConvGRU input shape: {tuple(value.shape)}")
    return value


def _synthetic_pattern(size: int = 540) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    pattern = (
        np.sin(2 * np.pi * (3.1 * x + 1.3 * y) + 0.2)
        + 0.8 * np.cos(2 * np.pi * (-1.7 * x + 4.2 * y) - 0.7)
        + 0.45 * np.sin(2 * np.pi * (6.1 * x - 2.4 * y) + 1.2)
    )
    pattern = (pattern - pattern.min()) / max(float(pattern.max() - pattern.min()), 1e-12)
    return (255.0 * pattern).astype(np.float32)


@torch.no_grad()
def run_synthetic_calibration(
    scorer: RealTraceMatrixScorer,
    model: Any,
    cell: Any,
    *,
    max_lag_px: int,
) -> dict[str, Any]:
    offsets = np.asarray(
        [
            (0.0, 0.0),
            (-0.05, 0.0), (0.05, 0.0), (0.0, -0.05), (0.0, 0.05),
            (-0.035, -0.035), (-0.035, 0.035), (0.035, -0.035), (0.035, 0.035),
            (-0.10, 0.0), (0.10, 0.0), (0.0, -0.10), (0.0, 0.10),
        ],
        dtype=np.float32,
    )
    histories = np.broadcast_to(offsets[:, None, :], (len(offsets), 71, 2)).copy()
    stimuli = make_corrected_causal_stims(
        _synthetic_pattern(), histories, torch=scorer.torch
    ).reshape(len(offsets), N_FRAMES, 1, 32, 151, 151)[:, 0]
    dtype = next(model.parameters()).dtype
    x = ((stimuli - 127.0) / 255.0).to(scorer.device)
    behavior = scorer._zero_behavior(len(x), dtype)
    core = _core_input(model, x, behavior)
    terms = replay_convgru_cell(cell, core, verify=True)
    # A central step avoids the special replicate-padded edge while retaining
    # broad texture support.  Current candidate input does not depend on h.
    evidence = terms[4].candidate_current_preactivation
    baseline = evidence[0:1].expand_as(evidence[1:])
    peaks = normalized_multichannel_xcorr(
        baseline, evidence[1:], max_lag_px=max_lag_px
    )
    feature_xy = torch.stack([peaks.lag_x_px, peaks.lag_y_px], dim=1).cpu().numpy()
    fit = fit_shift_calibration(offsets[1:], feature_xy)
    rows = []
    predicted = apply_shift_calibration(
        offsets[1:], np.asarray(fit["matrix_feature_px_per_eye_deg"])
    )
    for index, eye in enumerate(offsets[1:]):
        rows.append(
            {
                "eye_x_deg": float(eye[0]),
                "eye_y_deg": float(eye[1]),
                "measured_feature_x_px": float(feature_xy[index, 0]),
                "measured_feature_y_px": float(feature_xy[index, 1]),
                "predicted_feature_x_px": float(predicted[index, 0]),
                "predicted_feature_y_px": float(predicted[index, 1]),
                "best_correlation": float(peaks.best_lag_correlation[index].cpu()),
                "peak_sharpness": float(peaks.peak_sharpness[index].cpu()),
            }
        )
    return {
        "definition": (
            "Exact renderer and frozen frontend/ResNet; static synthetic texture at known eye "
            "offsets; current candidate contribution at internal step 4; lag convention is "
            "sum A(x,y)B(x+dx,y+dy)."
        ),
        "matrix_feature_px_per_eye_deg": np.asarray(
            fit["matrix_feature_px_per_eye_deg"]
        ),
        "intercept_feature_px": np.asarray(fit["intercept_feature_px"]),
        "r2_by_feature_component": np.asarray(fit["r2_by_feature_component"]),
        "median_vector_error_px": float(fit["median_vector_error_px"]),
        "rows": rows,
    }


def _part_stem(scope: str, fold: int, pair: tuple[int, int]) -> str:
    return f"{scope}__fold_{fold}__image_{pair[0]}__trajectory_{pair[1]}"


def _part_is_complete(
    *,
    scope: str,
    fold: int,
    pair: tuple[int, int],
    random_draws: int,
    max_lag_px: int,
) -> bool:
    stem = _part_stem(scope, fold, pair)
    completion = PARTS / f"{stem}__complete.json"
    registration = PARTS / f"{stem}__registration.csv.gz"
    terms = PARTS / f"{stem}__terms.npz"
    if not (completion.is_file() and registration.is_file() and terms.is_file()):
        return False
    try:
        value = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot validate instrumentation part {completion}") from error
    expected = {
        "schema": INSTRUMENTATION_SCHEMA,
        "scope": scope,
        "fold": int(fold),
        "pair": list(pair),
        "random_draws": int(random_draws),
        "max_lag_px": int(max_lag_px),
        "subpixel_factor": 4,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Existing part settings differ at {completion}; rerun with --overwrite-parts. "
            f"expected={expected}, observed={observed}"
        )
    return True


def _write_registration_part(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_term_part(
    path: Path,
    metadata: list[list[int]],
    values: list[list[float]],
    columns: tuple[str, ...],
) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(metadata, dtype=np.int16),
        values=np.asarray(values, dtype=np.float32),
        metadata_columns=np.asarray(TERM_META_COLUMNS),
        value_columns=np.asarray(columns),
        contrast_labels=np.asarray([contrast.key for contrast in CONTRASTS]),
    )
    os.replace(temporary, path)


def _consensus_example_specs(selection: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    images = selection["images"]
    trajectories = selection["trajectories"]
    return [
        {
            "contrast": "low_0_to_2",
            "image_position": int(images[0]["position"]),
            "trajectory_position": int(trajectories[8]["position"]),
            "scale": 2.0,
            "frame": 20,
            "destination": "consensus_low_0_to_2.npz",
            "scope": "complete-consensus",
        },
        {
            "contrast": "high_0_to_1",
            "image_position": int(images[2]["position"]),
            "trajectory_position": int(trajectories[5]["position"]),
            "scale": 1.0,
            "frame": 20,
            "destination": "consensus_high_0_to_1.npz",
            "scope": "complete-consensus",
        },
        {
            "contrast": "high_1_to_3",
            "image_position": int(images[3]["position"]),
            "trajectory_position": int(trajectories[11]["position"]),
            "scale": 3.0,
            "frame": 20,
            "destination": "consensus_high_1_to_3.npz",
            "scope": "complete-consensus",
        },
    ]


def _heldout_example_spec(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one outcome-blind held-out pair nearest feature-space medians."""
    images = pd.read_csv(PILOT_IMAGE_TABLE)
    trajectories = pd.read_csv(PILOT_TRAJECTORY_TABLE)
    image_metric = images[IMAGE_STRATIFICATION_METRIC].to_numpy(dtype=np.float64)
    trajectory_metric = trajectories[TRAJECTORY_STRATIFICATION_METRIC].to_numpy(
        dtype=np.float64
    )

    def standardized(value: float, population: np.ndarray) -> float:
        spread = float(np.subtract(*np.quantile(population, [0.75, 0.25])))
        return (float(value) - float(np.median(population))) / max(spread, 1e-12)

    scored: list[tuple[float, int, int, int]] = []
    for job in jobs:
        fold = int(job["fold"])
        image_position, trajectory_position = map(int, job["pair"])
        score = standardized(image_metric[image_position], image_metric) ** 2
        score += standardized(
            trajectory_metric[trajectory_position], trajectory_metric
        ) ** 2
        scored.append((score, fold, image_position, trajectory_position))
    _, fold, image_position, trajectory_position = min(scored)
    return {
        "contrast": "low_0_to_2",
        "image_position": image_position,
        "trajectory_position": trajectory_position,
        "scale": 1.0,
        "frame": 20,
        "internal_step": 4,
        "fold": fold,
        "scope": "heldout",
        "destination": "heldout_objective_example.npz",
        "selection": (
            "predeclared outcome-blind held-out pair nearest the joint medians of image "
            "oriented-high-SF power and rendered trajectory path length; fixed 1x scale, "
            "frame 20, internal step 4; squared IQR-standardized distance with stable "
            "fold/image/trajectory tie break"
        ),
    }


def _save_example(
    destination: Path,
    terms: list[GRUStepTerms],
    specification: dict[str, Any],
    projector_basis: torch.Tensor,
) -> None:
    if destination.exists():
        return
    arrays: dict[str, np.ndarray] = {}
    validation: dict[str, float] = {}
    for name in (
        "x_t",
        "h_previous",
        "update_gate",
        "reset_gate",
        "candidate_current_preactivation",
        "candidate_recurrent_preactivation",
        "candidate_state",
        "retained_state_contribution",
        "new_state_contribution",
        "h_t",
    ):
        value = torch.stack([getattr(term, name)[0] for term in terms], dim=0).float()
        quantized = value.to(torch.float16).float()
        validation[name] = float((quantized - value).abs().max().cpu())
        arrays[name] = value.to(torch.float16).cpu().numpy()
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                **specification,
                "selection": specification.get(
                    "selection", "predeclared image/trajectory strata; fixed frame 20"
                ),
                "float16_max_abs_by_term": validation,
                "equation_reconstruction_verified_before_quantization": True,
            },
            sort_keys=True,
        )
    )
    arrays["learned_projector_basis"] = (
        projector_basis.detach().to(torch.float32).cpu().numpy()
    )
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def _check_deadline(deadline: float, reserve_seconds: float) -> None:
    if time.monotonic() + max(float(reserve_seconds), 0.0) >= deadline:
        raise ComputeLimitReached(
            "Ten-hour cumulative GPU boundary reached before another complete batch"
        )


@torch.no_grad()
def instrument_pair(
    *,
    scope: str,
    fold: int,
    pair: tuple[int, int],
    histories: np.ndarray,
    movies: torch.Tensor,
    scorer: RealTraceMatrixScorer,
    model: Any,
    cell: Any,
    calibration_matrix: np.ndarray,
    args: argparse.Namespace,
    deadline: float,
    example_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    registration_rows: list[dict[str, Any]] = []
    term_metadata: list[list[int]] = []
    term_values: list[list[float]] = []
    term_columns: tuple[str, ...] | None = None
    maximum_cache_error = 0.0
    bases_cpu = {
        contrast.key: _bases_for(contrast, fold, args.random_draws)
        for contrast in CONTRASTS
    }
    bases_device = {
        contrast: {name: value.to(scorer.device) for name, value in views.items()}
        for contrast, views in bases_cpu.items()
    }
    all_registration_bases = {
        f"{contrast}::{name}": basis
        for contrast, views in bases_device.items()
        for name, basis in views.items()
    }
    dtype = next(model.parameters()).dtype
    with h5py.File(STATE_CACHE, "r") as state_cache:
        for scale_position, scale in enumerate(SCALES):
            history = histories[scale_position]
            for frame_start in range(0, N_FRAMES, int(args.frame_batch_size)):
                _check_deadline(deadline, args.minimum_next_batch_seconds)
                frame_stop = min(frame_start + int(args.frame_batch_size), N_FRAMES)
                x = movies[scale_position, frame_start:frame_stop].to(scorer.device)
                behavior = scorer._zero_behavior(len(x), dtype)
                core = _core_input(model, x, behavior)
                terms = replay_convgru_cell(cell, core, verify=True)
                cached = torch.as_tensor(
                    np.asarray(
                        state_cache["h"][
                            pair[0], pair[1], scale_position, frame_start:frame_stop
                        ],
                        dtype=np.float32,
                    ),
                    device=scorer.device,
                )
                cache_error = float((terms[-1].h_t - cached).abs().max().cpu())
                maximum_cache_error = max(maximum_cache_error, cache_error)
                if cache_error > float(args.max_cache_state_error):
                    raise RuntimeError(
                        "Instrumented final state does not reproduce the validated exact cache: "
                        f"pair={pair} scale={scale} frames={frame_start}:{frame_stop} "
                        f"max_abs={cache_error:.8g}"
                    )
                frame_numbers = list(range(frame_start, frame_stop))
                eye_deltas = [
                    internal_step_eye_displacements(history, frame)
                    for frame in frame_numbers
                ]
                expected_shifts = [
                    apply_shift_calibration(value, calibration_matrix)
                    for value in eye_deltas
                ]
                for internal_step, step_terms in enumerate(terms):
                    for contrast_index, contrast in enumerate(CONTRASTS):
                        learned = bases_device[contrast.key]["learned_p"]
                        summary = term_energy_summary(
                            step_terms,
                            learned,
                            TERM_NAMES,
                            comparison_bases={
                                "readout_svd": bases_device[contrast.key]["readout_svd"]
                            },
                        )
                        if term_columns is None:
                            term_columns = tuple(summary)
                        elif term_columns != tuple(summary):
                            raise RuntimeError("Projected-term column order changed within a run")
                        # One synchronized transfer per contrast/step, rather
                        # than one tiny GPU transfer for every scalar cell.
                        summary_cpu = torch.stack(
                            [summary[name] for name in term_columns], dim=1
                        ).detach().cpu().numpy()
                        for local_frame, frame in enumerate(frame_numbers):
                            term_metadata.append(
                                [
                                    fold,
                                    pair[0],
                                    pair[1],
                                    scale_position,
                                    frame,
                                    internal_step,
                                    contrast_index,
                                ]
                            )
                            term_values.append(summary_cpu[local_frame].tolist())
                    if internal_step > 0:
                        # All contrast/control bases are namespaced and reduced
                        # together, so P-like views share one batched FFT and
                        # Q-like views share a second.
                        metrics = registration_step_metrics(
                            step_terms.candidate_current_preactivation,
                            step_terms.h_previous,
                            step_terms.candidate_recurrent_preactivation,
                            all_registration_bases,
                            max_lag_px=args.max_lag_px,
                        )
                        metric_items = list(metrics.items())
                        metric_value_index = {
                            name: index for index, name in enumerate(REGISTRATION_METRIC_NAMES)
                        }
                        # Stack all views and scalar fields into one transfer.
                        # Per-cell ``.cpu()`` calls here would create tens of
                        # millions of GPU synchronization points in the full run.
                        metric_block = torch.stack(
                            [
                                torch.stack(
                                    [values[name] for name in REGISTRATION_METRIC_NAMES],
                                    dim=1,
                                )
                                for _, values in metric_items
                            ],
                            dim=0,
                        ).detach().cpu().numpy()
                        valid_block = torch.stack(
                            [values["valid"] for _, values in metric_items], dim=0
                        ).detach().cpu().numpy()
                        for view_index, (namespaced, values) in enumerate(metric_items):
                            contrast_key, subspace = namespaced.split("::", 1)
                            for local_frame, frame in enumerate(frame_numbers):
                                eye_delta = eye_deltas[local_frame]
                                expected_xy = expected_shifts[local_frame]

                                def scalar(name: str) -> float:
                                    return float(
                                        metric_block[
                                            view_index,
                                            local_frame,
                                            metric_value_index[name],
                                        ]
                                    )

                                registration_rows.append(
                                    {
                                        "scope": scope,
                                        "contrast": contrast_key,
                                        "fold": fold,
                                        "image_position": pair[0],
                                        "trajectory_position": pair[1],
                                        "scale": float(scale),
                                        "frame_position": frame,
                                        "internal_step": internal_step,
                                        "subspace": subspace,
                                        "eye_delta_x_deg": float(eye_delta[internal_step, 0]),
                                        "eye_delta_y_deg": float(eye_delta[internal_step, 1]),
                                        "expected_feature_shift_x_px": float(expected_xy[internal_step, 0]),
                                        "expected_feature_shift_y_px": float(expected_xy[internal_step, 1]),
                                        "raw_zero_lag_correlation": scalar("raw_zero_lag_correlation"),
                                        "raw_best_lag_correlation": scalar("raw_best_lag_correlation"),
                                        "raw_lag_x_px": scalar("raw_lag_x_px"),
                                        "raw_lag_y_px": scalar("raw_lag_y_px"),
                                        "raw_at_search_boundary": bool(
                                            max(
                                                abs(scalar("raw_lag_x_px")),
                                                abs(scalar("raw_lag_y_px")),
                                            )
                                            >= float(args.max_lag_px) - 1e-6
                                        ),
                                        "raw_peak_sharpness": scalar("raw_peak_sharpness"),
                                        "raw_residual_mismatch": scalar("raw_residual_mismatch"),
                                        "recurrent_zero_lag_correlation": scalar("recurrent_zero_lag_correlation"),
                                        "recurrent_best_lag_correlation": scalar("recurrent_best_lag_correlation"),
                                        "recurrent_lag_x_px": scalar("recurrent_lag_x_px"),
                                        "recurrent_lag_y_px": scalar("recurrent_lag_y_px"),
                                        "recurrent_at_search_boundary": bool(
                                            max(
                                                abs(scalar("recurrent_lag_x_px")),
                                                abs(scalar("recurrent_lag_y_px")),
                                            )
                                            >= float(args.max_lag_px) - 1e-6
                                        ),
                                        "recurrent_peak_sharpness": scalar("recurrent_peak_sharpness"),
                                        "recurrent_residual_mismatch": scalar("recurrent_residual_mismatch"),
                                        "transport_x_px": scalar("transport_x_px"),
                                        "transport_y_px": scalar("transport_y_px"),
                                        "expected_outside_search_window": bool(
                                            max(abs(float(expected_xy[internal_step, 0])),
                                                abs(float(expected_xy[internal_step, 1])))
                                            > float(args.max_lag_px)
                                        ),
                                        "zero_lag_alignment_improvement": scalar("zero_lag_alignment_improvement"),
                                        "best_lag_alignment_improvement": scalar("best_lag_alignment_improvement"),
                                        "valid": bool(valid_block[view_index, local_frame]),
                                    }
                                )
                for local_frame, frame in enumerate(frame_numbers):
                    for specification in example_specs:
                        if (
                            scope == specification["scope"]
                            and ("fold" not in specification or fold == specification["fold"])
                            and pair[0] == specification["image_position"]
                            and pair[1] == specification["trajectory_position"]
                            and np.isclose(scale, specification["scale"])
                            and frame == specification["frame"]
                        ):
                            selected_terms = [
                                GRUStepTerms(
                                    **{
                                        name: getattr(step_terms, name)[local_frame : local_frame + 1]
                                        for name in step_terms.__dataclass_fields__
                                    }
                                )
                                for step_terms in terms
                            ]
                            _save_example(
                                EXAMPLES / str(specification["destination"]),
                                selected_terms,
                                specification,
                                bases_cpu[specification["contrast"]]["learned_p"],
                            )
                del x, behavior, core, terms, cached
    if term_columns is None:
        raise RuntimeError("No projected terms were produced")
    stem = _part_stem(scope, fold, pair)
    _write_registration_part(PARTS / f"{stem}__registration.csv.gz", registration_rows)
    _write_term_part(
        PARTS / f"{stem}__terms.npz",
        term_metadata,
        term_values,
        term_columns,
    )
    write_json(
        PARTS / f"{stem}__complete.json",
        {
            "schema": INSTRUMENTATION_SCHEMA,
            "scope": scope,
            "fold": fold,
            "pair": list(pair),
            "random_draws": int(args.random_draws),
            "max_lag_px": int(args.max_lag_px),
            "subpixel_factor": 4,
            "registration_rows": len(registration_rows),
            "projected_term_rows": len(term_metadata),
            "maximum_final_state_cache_error": maximum_cache_error,
        },
    )
    return {
        "registration_rows": len(registration_rows),
        "projected_term_rows": len(term_metadata),
        "maximum_final_state_cache_error": maximum_cache_error,
    }


def consolidate_parts(output_dir: Path, *, scope: str) -> dict[str, Any]:
    """Consolidate one projector scope without mixing pilot and final rows.

    The held-out replay uses fold-specific projectors whereas the complete
    replay uses a consensus projector.  They are scientifically distinct and
    must never be silently concatenated.  The complete-consensus products own
    the protocol's canonical filenames; held-out products receive an explicit
    suffix.
    """
    if scope not in {"heldout", "complete-consensus"}:
        raise ValueError(f"Unknown instrumentation scope {scope!r}")
    registration_parts: list[Path] = []
    term_parts: list[Path] = []
    settings_signature: tuple[int, int, int] | None = None
    for completion in sorted(PARTS.glob(f"{scope}__*__complete.json")):
        try:
            manifest = json.loads(completion.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot validate completed part {completion}") from error
        if manifest.get("schema") != INSTRUMENTATION_SCHEMA or manifest.get("scope") != scope:
            raise RuntimeError(f"Completed-part schema/scope mismatch at {completion}")
        current_signature = (
            int(manifest.get("random_draws", -1)),
            int(manifest.get("max_lag_px", -1)),
            int(manifest.get("subpixel_factor", -1)),
        )
        if settings_signature is None:
            settings_signature = current_signature
        elif current_signature != settings_signature:
            raise RuntimeError(
                "Refusing to consolidate parts with different registration settings: "
                f"{settings_signature} versus {current_signature} at {completion}"
            )
        stem = completion.name.removesuffix("__complete.json")
        registration = PARTS / f"{stem}__registration.csv.gz"
        terms = PARTS / f"{stem}__terms.npz"
        if not registration.is_file() or not terms.is_file():
            raise RuntimeError(f"Completed part is missing payload files: {completion}")
        registration_parts.append(registration)
        term_parts.append(terms)
    if not registration_parts or not term_parts:
        raise RuntimeError("No complete instrumentation parts exist to consolidate")
    suffix = "" if scope == "complete-consensus" else "_heldout"
    registration_destination = output_dir / f"registration_metrics{suffix}.csv"
    temporary_csv = registration_destination.with_name(
        f"{registration_destination.name}.tmp.{os.getpid()}"
    )
    total_registration = 0
    with temporary_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=REGISTRATION_COLUMNS)
        writer.writeheader()
        for path in registration_parts:
            with gzip.open(path, "rt", newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle)
                if tuple(reader.fieldnames or ()) != REGISTRATION_COLUMNS:
                    raise RuntimeError(f"Registration schema mismatch at {path}")
                for row in reader:
                    writer.writerow(row)
                    total_registration += 1
    os.replace(temporary_csv, registration_destination)

    metadata_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    metadata_columns: np.ndarray | None = None
    value_columns: np.ndarray | None = None
    contrast_labels: np.ndarray | None = None
    for path in term_parts:
        with np.load(path, allow_pickle=False) as archive:
            current_meta_columns = np.asarray(archive["metadata_columns"])
            current_value_columns = np.asarray(archive["value_columns"])
            current_contrasts = np.asarray(archive["contrast_labels"])
            if metadata_columns is None:
                metadata_columns = current_meta_columns
                value_columns = current_value_columns
                contrast_labels = current_contrasts
            elif not (
                np.array_equal(metadata_columns, current_meta_columns)
                and np.array_equal(value_columns, current_value_columns)
                and np.array_equal(contrast_labels, current_contrasts)
            ):
                raise RuntimeError(f"Projected-term schema mismatch at {path}")
            metadata_chunks.append(np.asarray(archive["metadata"], dtype=np.int16))
            value_chunks.append(np.asarray(archive["values"], dtype=np.float32))
    term_destination = output_dir / f"gru_projected_terms{suffix}.npz"
    temporary_terms = term_destination.with_name(
        f"{term_destination.name}.tmp.{os.getpid()}.npz"
    )
    np.savez_compressed(
        temporary_terms,
        metadata=np.concatenate(metadata_chunks, axis=0),
        values=np.concatenate(value_chunks, axis=0),
        metadata_columns=metadata_columns,
        value_columns=value_columns,
        contrast_labels=contrast_labels,
        scope=np.asarray(scope),
        storage_definition=np.asarray(
            "Every internal step is retained as exact P/Q/total energy scalars; full maps are "
            "limited to three predeclared visualization examples to prevent another dense cache."
        ),
    )
    os.replace(temporary_terms, term_destination)
    return {
        "scope": scope,
        "registration_settings": {
            "random_draws": settings_signature[0],
            "max_lag_px": settings_signature[1],
            "subpixel_factor": settings_signature[2],
        },
        "registration_parts": len(registration_parts),
        "term_parts": len(term_parts),
        "registration_rows": total_registration,
        "projected_term_rows": int(sum(len(value) for value in value_chunks)),
        "registration_metrics": str(registration_destination),
        "gru_projected_terms": str(term_destination),
    }


def plan_payload(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    subspaces_per_contrast = 3 + int(args.random_draws)
    samples = len(jobs) * len(SCALES) * N_FRAMES
    return {
        "scope": args.scope,
        "pairs": len(jobs),
        "external_scored_samples": samples,
        "convgru_internal_steps_per_sample": 8,
        "projected_term_rows": samples * 8 * len(CONTRASTS),
        "registration_rows": samples * 7 * len(CONTRASTS) * subspaces_per_contrast,
        "subspaces_per_contrast": subspaces_per_contrast,
        "random_rank8_draws": int(args.random_draws),
        "full_term_examples": 1 if args.scope == "heldout" else 3,
        "dense_full_subset_cache_written": False,
        "projected_term_views": (
            "full energy, fold-specific learned P, exact learned Q residual, "
            "and rank-8 RR100 readout-SVD energy for every 128-channel term"
        ),
        "runtime_expectation": (
            "Core replay alone previously required about 0.10 accelerator-hours for all 192 pairs. "
            "Fourier registration dominates and must be timed on the first complete pair; the "
            "resume-safe command stops between frame batches at the shared cumulative 10-hour boundary."
        ),
        "estimated_storage": (
            "heldout: roughly 0.3-0.8 GB; complete consensus: roughly 1-3 GB, mostly the required "
            "plain registration CSV; never a 60-GB dense recurrent-term cache"
        ),
    }


def run(args: argparse.Namespace) -> int:
    global OUT, PARTS, EXAMPLES
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive")
    OUT = args.output_dir
    PARTS = OUT / "gru_instrumentation/parts"
    EXAMPLES = OUT / "gru_instrumentation/full_examples"
    PARTS.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    selection = select_pilot()
    write_pilot_selection(OUT / "pilot_4x12_selection.csv", selection)
    jobs = _job_plan(args.scope)
    examples = (
        [_heldout_example_spec(jobs)]
        if args.scope == "heldout"
        else _consensus_example_specs(selection)
    )
    write_json(
        OUT / f"gru_instrumentation/example_selection_{args.scope}.json", examples
    )
    if args.plan_only:
        payload = plan_payload(args, jobs)
        write_json(
            OUT / f"gru_instrumentation/instrumentation_plan_{args.scope}.json",
            payload,
        )
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    if args.consolidate_only:
        write_json(
            OUT / f"gru_instrumentation/consolidation_{args.scope}.json",
            consolidate_parts(OUT, scope=args.scope),
        )
        return 0

    if args.overwrite_parts:
        for path in PARTS.glob(f"{args.scope}__*"):
            path.unlink()
    pending = [
        job
        for job in jobs
        if not _part_is_complete(
            scope=args.scope,
            fold=int(job["fold"]),
            pair=tuple(job["pair"]),
            random_draws=args.random_draws,
            max_lag_px=args.max_lag_px,
        )
    ]
    pending_before_limit = len(pending)
    if args.max_pairs > 0:
        pending = pending[: int(args.max_pairs)]
    if not pending:
        write_json(
            OUT / f"gru_instrumentation/consolidation_{args.scope}.json",
            consolidate_parts(OUT, scope=args.scope),
        )
        print("All requested instrumentation pairs are already complete.")
        return 0
    if not str(args.device).startswith("cuda"):
        raise ValueError("Production ConvGRU replay must use an explicitly ledgered CUDA device")
    # A one-pair benchmark only needs the projector for that pending pair's
    # fold.  The full held-out call still preflights all remaining folds.
    _preflight_projectors(args.scope, pending)

    completed = 0
    result_rows: list[dict[str, Any]] = []
    status = "running"
    error_message: str | None = None
    with exclusive_gpu_analysis_lock():
        # Reload only after acquiring the shared run lock.  Otherwise another
        # GPU stage could finish between our preflight read and lock acquisition,
        # leaving this process with a stale (too generous) remaining budget.
        budget = load_global_gpu_budget(HARD_GPU_LIMIT_HOURS)
        consumed = float(budget["total_conservative_gpu_hours"])
        if consumed >= HARD_GPU_LIMIT_HOURS:
            raise RuntimeError("Cumulative ten-GPU-hour limit already reached")
        started = time.perf_counter()
        deadline = time.monotonic() + (HARD_GPU_LIMIT_HOURS - consumed) * 3600.0
        try:
            _check_deadline(deadline, max(120.0, args.minimum_next_batch_seconds))
            scorer, model, cell = _load_model(args.device)
            _check_deadline(deadline, max(60.0, args.minimum_next_batch_seconds))
            calibration = run_synthetic_calibration(
                scorer, model, cell, max_lag_px=args.max_lag_px
            )
            calibration_r2 = np.asarray(
                calibration["r2_by_feature_component"], dtype=np.float64
            )
            if (
                not np.isfinite(calibration_r2).all()
                or float(calibration_r2.min()) < float(args.minimum_calibration_r2)
                or float(calibration["median_vector_error_px"])
                > float(args.maximum_calibration_error_px)
            ):
                raise RuntimeError(
                    "Synthetic shift calibration did not establish a reliable sign/scale map: "
                    f"r2={calibration_r2.tolist()}, "
                    f"median_error={calibration['median_vector_error_px']} px"
                )
            write_json(OUT / "synthetic_shift_calibration.json", calibration)
            matrix = np.asarray(calibration["matrix_feature_px_per_eye_deg"], dtype=np.float64)
            images = pd.read_csv(IMAGE_TABLE).sort_values("image_index").reset_index(drop=True)
            with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
                base_histories = np.asarray(archive["true_history_xy"], dtype=np.float32)
            with h5py.File(STATE_CACHE, "r") as state_cache:
                image_ids = np.asarray(state_cache["image_ids"][:], dtype=np.int64)
                trajectory_ids = np.asarray(state_cache["trajectory_ids"][:], dtype=np.int64)
            canvas_cache: dict[Any, Any] = {}
            for job in pending:
                _check_deadline(deadline, args.minimum_next_batch_seconds)
                fold = int(job["fold"])
                pair = tuple(job["pair"])
                image_id = int(image_ids[pair[0]])
                trajectory_id = int(trajectory_ids[pair[1]])
                patch, _ = extract_patch(
                    images.iloc[image_id], canvas_cache=canvas_cache, patch_size_px=540
                )
                patch = _standardize_uint_like(patch)
                histories = scaled_histories(
                    base_histories[trajectory_id : trajectory_id + 1]
                )
                stimuli = (
                    make_corrected_causal_stims(patch, histories, torch=scorer.torch) - 127.0
                ) / 255.0
                movies = stimuli.reshape(len(SCALES), N_FRAMES, *stimuli.shape[1:])
                pair_result = instrument_pair(
                    scope=args.scope,
                    fold=fold,
                    pair=pair,
                    histories=histories,
                    movies=movies,
                    scorer=scorer,
                    model=model,
                    cell=cell,
                    calibration_matrix=matrix,
                    args=args,
                    deadline=deadline,
                    example_specs=examples,
                )
                result_rows.append({"fold": fold, "pair": pair, **pair_result})
                completed += 1
                print(
                    f"instrumented {completed}/{len(pending)} pending pairs: "
                    f"scope={args.scope} fold={fold} pair={pair}",
                    flush=True,
                )
                del histories, stimuli, movies
            status = (
                "complete"
                if len(pending) == pending_before_limit
                else "partial_requested"
            )
        except ComputeLimitReached as error:
            status = "gpu_budget_reached"
            error_message = str(error)
        except BaseException as error:
            status = "failed"
            error_message = f"{type(error).__name__}: {error}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            ledger = record_global_gpu_time(
                f"registration_instrumentation:{args.scope}",
                elapsed,
                hard_limit_hours=HARD_GPU_LIMIT_HOURS,
                details={
                    "device": args.device,
                    "completed_pairs_this_call": completed,
                    "requested_pending_pairs": len(pending),
                    "status": status,
                },
            )
            write_json(
                OUT / "gru_instrumentation/run_manifest.json",
                {
                    "status": status,
                    "error": error_message,
                    "scope": args.scope,
                    "arguments": vars(args),
                    "completed_pairs_this_call": completed,
                    "total_planned_pairs": len(jobs),
                    "results": result_rows,
                    "gpu_ledger": ledger,
                    "storage_policy": (
                        "online P/Q scalar reductions for every step; only three predeclared "
                        "full examples; no dense full-subset term cache"
                    ),
                },
            )
    if list(PARTS.glob(f"{args.scope}__*__complete.json")):
        write_json(
            OUT / f"gru_instrumentation/consolidation_{args.scope}.json",
            consolidate_parts(OUT, scope=args.scope),
        )
    return 2 if status == "gpu_budget_reached" else 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
