#!/usr/bin/env python3
"""Run Sections 10--11 on the frozen corrected Figure 4 model.

The production stages rerender only the exact frozen 8x24 subset (or its
outcome-blind 4x12 pilot), never the full natural-movie bank.  Recurrent
transport ablations are projector-free.  Candidate realignment always uses
the projector from the fold that held out both the current image and current
trajectory; consensus projectors are never loaded by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (  # noqa: E402
    MAP_CACHE,
    N_FRAMES,
    READOUT_CACHE,
    SCALES,
    STATE_CACHE,
    scale_index,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import target_units  # noqa: E402
from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR  # noqa: E402
from paper.fig4.mechanism_audit_v1.correction.scoring import (  # noqa: E402
    build_direct_rr100_readout,
    make_corrected_causal_stims,
)
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (  # noqa: E402
    scaled_histories,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.analysis import (  # noqa: E402
    EndpointAccumulator,
    SCHEMA_VERSION,
    accumulator_rows,
    normalized_population_maps,
    pilot_decision_gate,
    population_definitions,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.mechanics import (  # noqa: E402
    TRANSPORT_INTERVENTIONS,
    matched_random_directions,
    recurrent_kernel_invariants,
    replay_realignment,
    replay_transport_intervention,
    transform_recurrent_kernel,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.provenance import (  # noqa: E402
    atomic_json,
    file_record,
    json_ready,
    sha256_file,
    valid_complete_marker,
    write_complete_marker,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.causal_interventions.selection import (  # noqa: E402
    InterventionSelection,
    selection_for_scope,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.pilot_selection import (  # noqa: E402
    IMAGE_TABLE as SELECTED_IMAGE_TABLE,
    TRAJECTORY_TABLE as SELECTED_TRAJECTORY_TABLE,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.instrument_subset import (  # noqa: E402
    INSTRUMENTATION_SCHEMA,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.registration import (  # noqa: E402
    FIG4_CONVGRU_INPUT_LAG_SUPPORTS,
    apply_shift_calibration,
    derive_fig4_convgru_input_lag_supports,
    internal_step_eye_displacements,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.common import (  # noqa: E402
    budget_deadline,
    exclusive_gpu_lock,
    fit_directory,
    load_budget,
    record_gpu_time,
    validate_basis,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch  # noqa: E402
from paper.fig4.upstream.real_trace_matrix.model import (  # noqa: E402
    RealTraceMatrixScorer,
    _standardize_uint_like,
)
from paper.fig4.upstream.run_real_trace_matrix import (  # noqa: E402
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    MODEL_CHECKPOINT_SHA256,
    RR100_VERSION,
)


BASE_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
OUT = BASE_OUT / "causal_interventions"
RANK8_GATE = BASE_OUT / "rank8_validation_gate.json"
PQ_MANIFEST = BASE_OUT / "pq_semantics_manifest.json"
PQ_VARIANCE = BASE_OUT / "pq_variance_decomposition.csv"
REGISTRATION_METRICS = BASE_OUT / "registration_metrics_heldout.csv"
REGISTRATION_MANIFEST = (
    BASE_OUT / "gru_instrumentation/consolidation_heldout.json"
)
INSTRUMENTATION_PARTS = BASE_OUT / "gru_instrumentation/parts"
SHIFT_CALIBRATION = BASE_OUT / "synthetic_shift_calibration.json"
PILOT_GATE = OUT / "pilot/pilot_to_full_gate.json"
CANONICAL_MANIFEST = BASE_OUT / "causal_interventions_manifest.json"
CAUSAL_CACHE_MANIFEST = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1/cache/cache_generation_manifest.json"
)
CAUSAL_INTEGRITY = (
    ROOT / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1/integrity_tests.json"
)
FOLD_ASSIGNMENTS = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1/config/fold_assignments.json"
)
IMAGE_FEATURE_TABLE = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/image_feature_table.csv"
)
MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"

TRANSPORT_CONDITIONS = tuple(value.name for value in TRANSPORT_INTERVENTIONS)
REALIGNMENT_CONDITIONS = (
    "no_shift_intact",
    "eye_correct_candidate_p",
    "activation_oracle_candidate_p",
    "eye_opposite_candidate_p",
    "eye_random_matched_candidate_p",
    "eye_correct_complementary_q",
    "induced_eye_misalignment_candidate_p",
)
REALIGNMENT_SCALES = (1.0, 3.0)
STAGE_PRODUCTS = {
    "transport": ("summary.csv", "arrays.npz"),
    "realignment": ("summary.csv", "arrays.npz", "shift_diagnostics.csv"),
}
MAX_LITERAL_REPLAY_ABS = 3e-6
MAX_CACHE_STATE_RELATIVE_RMSE = 3e-3
MAX_CACHE_ENDPOINT_RELATIVE_RMSE = 1e-3


class ComputeBoundaryReached(RuntimeError):
    """Raised only before a new complete image--trajectory part."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("plan", "transport", "realignment", "consolidate", "all"),
        required=True,
    )
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--max-oracle-lag-px", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--overwrite-parts", action="store_true")
    parser.add_argument("--allow-incomplete-consolidation", action="store_true")
    parser.add_argument("--minimum-next-pair-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read required JSON product: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Publish a completed product without exposing a partial canonical file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (estimate.double() - reference.double()).square().sum().sqrt()
    denominator = reference.double().square().sum().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).detach().cpu())


def _parts_dir(scope: str, stage: str) -> Path:
    return OUT / scope / stage / "parts"


def _part_dir(scope: str, stage: str, identity: Mapping[str, int]) -> Path:
    if stage == "transport":
        stem = f"image_{identity['image_position']}__trajectory_{identity['trajectory_position']}"
    else:
        stem = (
            f"fold_{identity['fold']}__image_{identity['image_position']}"
            f"__trajectory_{identity['trajectory_position']}"
        )
    return _parts_dir(scope, stage) / stem


def _stage_identities(selection: InterventionSelection, stage: str) -> list[dict[str, int]]:
    if stage == "transport":
        return [
            {"image_position": int(image), "trajectory_position": int(trajectory)}
            for image, trajectory in selection.transport_pairs
        ]
    return [
        {
            "fold": int(fold),
            "image_position": int(image),
            "trajectory_position": int(trajectory),
        }
        for fold, image, trajectory in selection.heldout_realign_pairs
    ]


def _assert_cache_gate() -> None:
    for path in (STATE_CACHE, MAP_CACHE, READOUT_CACHE, CAUSAL_INTEGRITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not bool(_read_json(CAUSAL_INTEGRITY).get("optimization_allowed", False)):
        raise RuntimeError("The mandatory exact-cache P=0/P=I integrity gate is not passing")
    with h5py.File(STATE_CACHE, "r") as states, h5py.File(MAP_CACHE, "r") as maps:
        if not bool(states.attrs.get("complete", False)) or not bool(maps.attrs.get("complete", False)):
            raise RuntimeError("Exact state/map caches are not marked complete")
        if (
            states.attrs.get("checkpoint_sha256") != MODEL_CHECKPOINT_SHA256
            or maps.attrs.get("checkpoint_sha256") != MODEL_CHECKPOINT_SHA256
        ):
            raise RuntimeError("Exact cache checkpoint provenance does not match the frozen model")
        expected_shapes = {
            "h": (8, 24, 5, 40, 128, 64, 64),
            "preactivation": (8, 24, 5, 40, 100, 51, 51),
            "rate": (8, 24, 5, 40, 100, 51, 51),
            "gain": (8, 24, 5, 40, 100, 51, 51),
        }
        if tuple(states["h"].shape) != expected_shapes["h"] or any(
            tuple(maps[name].shape) != expected_shapes[name]
            for name in ("preactivation", "rate", "gain")
        ):
            raise RuntimeError("Exact cache tensor shapes no longer match the frozen 8x24 design")
        for name in ("image_ids", "trajectory_ids", "scales"):
            if not np.array_equal(np.asarray(states[name][:]), np.asarray(maps[name][:])):
                raise RuntimeError(f"State/map cache {name} axes disagree")
        if not np.allclose(np.asarray(states["scales"][:], dtype=float), SCALES):
            raise RuntimeError("Exact cache movement scales changed")
        state_done = np.asarray(states["completed_pairs"][:], dtype=bool)
        map_done = np.asarray(maps["completed_pairs"][:], dtype=bool)
        if not state_done.all() or not np.array_equal(state_done, map_done):
            raise RuntimeError("Exact state/map pair completion masks are incomplete or disagree")
        with np.load(READOUT_CACHE, allow_pickle=False) as readout:
            if not (
                np.array_equal(readout["image_ids"], states["image_ids"][:])
                and np.array_equal(readout["trajectory_ids"], states["trajectory_ids"][:])
                and np.allclose(readout["scales"], states["scales"][:])
                and readout["feature_weights"].shape == (100, 128)
                and readout["space_weights"].shape == (100, 14, 14)
                and readout["bias"].shape == (100,)
            ):
                raise RuntimeError("Exact readout-cache axes or tensor shapes changed")


def _rank8_gate() -> dict[str, Any]:
    value = _read_json(RANK8_GATE)
    passed = bool(
        value.get("status") == "complete"
        and value.get("all_three_contrasts_generalize") is True
        and value.get("stop_downstream_mechanism_audit") is False
    )
    return {
        "passed": passed,
        "path": str(RANK8_GATE),
        "status": value.get("status"),
        "all_three_contrasts_generalize": value.get("all_three_contrasts_generalize"),
        "stop_downstream_mechanism_audit": value.get("stop_downstream_mechanism_audit"),
    }


def _pq_motion_gate() -> dict[str, Any]:
    manifest = _read_json(PQ_MANIFEST)
    if not PQ_VARIANCE.is_file():
        raise FileNotFoundError(PQ_VARIANCE)
    table = pd.read_csv(PQ_VARIANCE)
    defining_scale = {"low_0_to_2": 2.0, "high_0_to_1": 1.0, "high_1_to_3": 3.0}
    medians: dict[str, float] = {}
    fold_counts: dict[str, int] = {}
    for contrast, scale in defining_scale.items():
        rows = table.loc[
            table.analysis.eq("movement_change_from_stabilization")
            & table.contrast.eq(contrast)
            & np.isclose(table.scale, scale)
        ]
        values = pd.to_numeric(rows.p_to_q_per_dimension_energy_ratio, errors="coerce").to_numpy(float)
        medians[contrast] = float(np.nanmedian(values)) if len(values) else math.nan
        fold_counts[contrast] = int(rows.fold.nunique()) if len(rows) else 0
    passed = bool(
        manifest.get("complete_all_four_folds") is True
        and all(fold_counts[key] == 4 for key in defining_scale)
        and all(np.isfinite(medians[key]) and medians[key] > 1.0 for key in defining_scale)
    )
    return {
        "passed": passed,
        "criterion": "median held-out P/Q energy-per-dimension ratio > 1 at each defining movement scale",
        "median_p_to_q_per_dimension": medians,
        "fold_counts": fold_counts,
        "manifest": str(PQ_MANIFEST),
        "table": str(PQ_VARIANCE),
    }


def _registration_gate() -> dict[str, Any]:
    if not REGISTRATION_METRICS.is_file():
        raise FileNotFoundError(REGISTRATION_METRICS)
    manifest = _read_json(REGISTRATION_MANIFEST)
    current_part_identities: set[tuple[int, int, int]] = set()
    for path in INSTRUMENTATION_PARTS.glob("heldout__*__complete.json"):
        try:
            part = _read_json(path)
        except RuntimeError:
            continue
        if not (
            part.get("schema") == INSTRUMENTATION_SCHEMA
            and part.get("scope") == "heldout"
        ):
            continue
        pair = part.get("pair")
        if not (isinstance(pair, list) and len(pair) == 2):
            continue
        current_part_identities.add(
            (int(part.get("fold", -1)), int(pair[0]), int(pair[1]))
        )
    expected_part_identities = set(
        selection_for_scope("full").heldout_realign_pairs
    )
    table = pd.read_csv(REGISTRATION_METRICS)
    rows_before_quality_gate = len(table)
    if "valid" in table:
        valid = table.valid.astype(str).str.lower().isin(("true", "1"))
    else:
        valid = pd.Series(True, index=table.index)
    # Boundary-clipped cross-correlations cannot establish transport geometry.
    # The audit explicitly forbids conclusions that depend on the bounded lag
    # search or an expected shift outside that search window.
    for column in (
        "raw_at_search_boundary",
        "recurrent_at_search_boundary",
        "expected_outside_search_window",
    ):
        if column not in table:
            valid &= False
        else:
            flagged = table[column].astype(str).str.lower().isin(("true", "1"))
            valid &= ~flagged
    table = table.loc[valid]
    defining_scale = {"low_0_to_2": 2.0, "high_0_to_1": 1.0, "high_1_to_3": 3.0}
    medians: dict[str, float] = {}
    fold_counts: dict[str, int] = {}
    for contrast, scale in defining_scale.items():
        rows = table.loc[
            table.contrast.eq(contrast)
            & table.subspace.eq("learned_p")
            & np.isclose(table.scale, scale)
        ]
        values = pd.to_numeric(rows.zero_lag_alignment_improvement, errors="coerce").to_numpy(float)
        medians[contrast] = float(np.nanmedian(values)) if len(values) else math.nan
        fold_counts[contrast] = int(rows.fold.nunique()) if len(rows) else 0
    readiness_passed = bool(
        current_part_identities == expected_part_identities
        and int(manifest.get("registration_parts", -1)) == 48
        and int(manifest.get("term_parts", -1)) == 48
        and all(fold_counts[key] == 4 for key in defining_scale)
    )
    alignment_hypothesis_supported = bool(
        readiness_passed
        and all(np.isfinite(medians[key]) and medians[key] > 0.0 for key in defining_scale)
    )
    return {
        "passed": readiness_passed,
        "execution_readiness_passed": readiness_passed,
        "execution_readiness_criterion": (
            f"all 48 {INSTRUMENTATION_SCHEMA} fold-held-out registration/term parts complete "
            "with quality-valid rows "
            "from all four folds for every defining contrast"
        ),
        "alignment_hypothesis_supported": alignment_hypothesis_supported,
        "hypothesis_criterion": (
            "median recurrent zero-lag alignment improvement in learned P > 0 in every contrast; "
            "reported as an outcome, not an execution gate for required causal analyses"
        ),
        "median_zero_lag_alignment_improvement": medians,
        "fold_counts": fold_counts,
        "quality_gate": (
            "valid rows only; raw/recurrent peaks at the lag-search boundary and expected shifts "
            "outside the search window are excluded"
        ),
        "rows_before_quality_gate": int(rows_before_quality_gate),
        "rows_after_quality_gate": int(len(table)),
        "current_schema_part_count": len(current_part_identities),
        "current_schema_part_identities_match_frozen_heldout_design": (
            current_part_identities == expected_part_identities
        ),
        "required_instrumentation_schema": INSTRUMENTATION_SCHEMA,
        "path": str(REGISTRATION_METRICS),
        "consolidation_manifest": str(REGISTRATION_MANIFEST),
    }


def _calibration_gate() -> dict[str, Any]:
    value = _read_json(SHIFT_CALIBRATION)
    r2 = np.asarray(value.get("r2_by_feature_component", []), dtype=float)
    error = float(value.get("median_vector_error_px", math.nan))
    matrix = np.asarray(value.get("matrix_feature_px_per_eye_deg", []), dtype=float)
    passed = bool(
        matrix.shape == (2, 2)
        and r2.shape == (2,)
        and np.isfinite(r2).all()
        and float(r2.min()) >= 0.80
        and np.isfinite(error)
        and error <= 0.50
    )
    return {
        "passed": passed,
        "criterion": "both calibration component R2 >= 0.80 and median vector error <= 0.50 feature pixels",
        "r2": r2.tolist(),
        "median_vector_error_px": error,
        "matrix": matrix.tolist(),
        "path": str(SHIFT_CALIBRATION),
    }


def upstream_mechanism_gate() -> dict[str, Any]:
    def checked(label: str, function: Any) -> dict[str, Any]:
        try:
            return function()
        except Exception as error:
            return {
                "passed": False,
                "gate": label,
                "error": f"{type(error).__name__}: {error}",
            }

    rank8 = checked("rank8_heldout_generalization", _rank8_gate)
    pq = checked("p_motion_enrichment", _pq_motion_gate)
    registration = checked("recurrent_alignment", _registration_gate)
    calibration = checked("eye_to_feature_calibration", _calibration_gate)
    # Analyses 3--4 are required causal tests.  The user explicitly waived the
    # P/Q motion-enrichment stop and requested that alignment *outcomes* not be
    # used to suppress those tests.  Execution remains fail-closed on rank-8
    # validity, complete/quality-valid registration products, and calibration.
    passed = all(value["passed"] for value in (rank8, registration, calibration))
    return {
        "passed": passed,
        "rank8_heldout_generalization": rank8,
        "p_motion_enrichment": {
            **pq,
            "execution_gate_applied": False,
            "waiver": (
                "explicitly waived by the user for required causal analyses 3--4; retained "
                "as descriptive evidence and for the final scientific decision"
            ),
        },
        "recurrent_alignment": {
            **registration,
            "positive_alignment_execution_gate_applied": False,
        },
        "eye_to_feature_calibration": calibration,
        "execution_gate_definition": (
            "rank-8 held-out validity + complete quality-valid registration products + "
            "passed synthetic eye-to-feature calibration"
        ),
        "failure_action": (
            None
            if passed
            else "do not launch causal inference until required input products are complete and valid"
        ),
    }


def _input_fingerprint(stage: str, *, scope: str) -> dict[str, Any]:
    if stage not in STAGE_PRODUCTS:
        raise ValueError(f"Unknown causal-intervention stage: {stage}")
    if scope not in ("pilot", "full"):
        raise ValueError(f"Unknown causal-intervention scope: {scope}")
    required = (
        MODEL_CHECKPOINT_PATH,
        READOUT_CACHE,
        FOLD_ASSIGNMENTS,
        SELECTED_IMAGE_TABLE,
        SELECTED_TRAJECTORY_TABLE,
        BANK_DIR / "corrected_history_trajectory_banks.npz",
        IMAGE_FEATURE_TABLE,
        RANK8_GATE,
        PQ_MANIFEST,
        PQ_VARIANCE,
        REGISTRATION_METRICS,
        REGISTRATION_MANIFEST,
        SHIFT_CALIBRATION,
        CAUSAL_INTEGRITY,
        CAUSAL_CACHE_MANIFEST,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = sha256_file(Path(MODEL_CHECKPOINT_PATH))
    if checkpoint_hash != MODEL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Frozen Figure 4 checkpoint hash changed: {checkpoint_hash} != {MODEL_CHECKPOINT_SHA256}"
        )
    fingerprint: dict[str, Any] = {
        "checkpoint": file_record(Path(MODEL_CHECKPOINT_PATH)),
        "readout_cache": file_record(READOUT_CACHE),
        "state_cache": file_record(STATE_CACHE, hash_file=False),
        "map_cache": file_record(MAP_CACHE, hash_file=False),
        "cache_manifest": (
            file_record(CAUSAL_CACHE_MANIFEST)
        ),
        "integrity_gate": file_record(CAUSAL_INTEGRITY),
        "fold_assignments": file_record(FOLD_ASSIGNMENTS),
        "selected_images": file_record(SELECTED_IMAGE_TABLE),
        "selected_trajectories": file_record(SELECTED_TRAJECTORY_TABLE),
        "history_bank": file_record(BANK_DIR / "corrected_history_trajectory_banks.npz"),
        "image_feature_table": file_record(IMAGE_FEATURE_TABLE),
        "rank8_gate": file_record(RANK8_GATE),
        "pq_manifest": file_record(PQ_MANIFEST),
        "pq_variance": file_record(PQ_VARIANCE),
        "registration_metrics": file_record(REGISTRATION_METRICS),
        "registration_manifest": file_record(REGISTRATION_MANIFEST),
        "shift_calibration": file_record(SHIFT_CALIBRATION),
        "mechanics_source": file_record(Path(__file__).with_name("mechanics.py")),
        "analysis_source": file_record(Path(__file__).with_name("analysis.py")),
        "selection_source": file_record(Path(__file__).with_name("selection.py")),
        "provenance_source": file_record(Path(__file__).with_name("provenance.py")),
        "gru_equations_source": file_record(
            ROOT
            / "paper/fig4/mechanism_audit_v1/registration_mechanism/"
            "gru_instrumentation/equations.py"
        ),
        "registration_source": file_record(
            ROOT
            / "paper/fig4/mechanism_audit_v1/registration_mechanism/"
            "gru_instrumentation/registration.py"
        ),
        "instrumentation_runner_source": file_record(
            ROOT
            / "paper/fig4/mechanism_audit_v1/registration_mechanism/"
            "gru_instrumentation/instrument_subset.py"
        ),
        "corrected_renderer_source": file_record(
            ROOT / "paper/fig4/mechanism_audit_v1/correction/scoring.py"
        ),
        "runner_source": file_record(Path(__file__)),
        "internal_step_supports": [list(value) for value in FIG4_CONVGRU_INPUT_LAG_SUPPORTS],
    }
    if stage == "realignment":
        fingerprint["foldwise_high_projectors"] = {
            f"{contrast}:fold_{fold}": file_record(fit_directory(contrast, fold) / "U.npy")
            for contrast in ("high_0_to_1", "high_1_to_3")
            for fold in range(4)
        }
    if scope == "full":
        # A full confirmation is scientifically downstream of the pilot.  Its
        # resume fingerprint must therefore invalidate if the pilot decision
        # or either consolidated pilot product changes.
        pilot_products = {
            "pilot_gate": PILOT_GATE,
            "pilot_analysis_manifest": OUT / "pilot/analysis_manifest.json",
            "pilot_transport_manifest": OUT / "pilot/transport/consolidation_manifest.json",
            "pilot_realignment_manifest": OUT / "pilot/realignment/consolidation_manifest.json",
            "pilot_transport_results": OUT / "pilot/transport_ablation_results.csv",
            "pilot_realignment_results": OUT / "pilot/realignment_results.csv",
        }
        for path in pilot_products.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        fingerprint["pilot_authorization"] = {
            name: file_record(path) for name, path in pilot_products.items()
        }
    return fingerprint


def _load_model(device: str) -> tuple[RealTraceMatrixScorer, Any, Any, Any]:
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
        raise RuntimeError("Frozen Figure 4 model must contain exactly one ConvGRU cell")
    if not isinstance(model.activation, torch.nn.Softplus):
        raise RuntimeError(f"Expected exact Softplus output, found {type(model.activation).__name__}")
    direct_readout = build_direct_rr100_readout(scorer).eval()
    return scorer, model, model.recurrent.cells[0], direct_readout


def _core_input(model: Any, stimulus: torch.Tensor, behavior: torch.Tensor | None) -> torch.Tensor:
    value = model.frontend(stimulus)
    value = model.convnet(value)
    if model.modulator is not None:
        if behavior is None:
            raise RuntimeError("Frozen behavior-conditioned model requires its exact zero behavior tensor")
        value = model.modulator(value, behavior)
    if value.shape[1:] != (256, 8, 64, 64):
        raise RuntimeError(f"Unexpected ConvGRU input shape: {tuple(value.shape)}")
    return value


def _endpoint(state_sequence: torch.Tensor, readout: Any, activation: Any) -> tuple[torch.Tensor, torch.Tensor]:
    state = state_sequence[:, :, -1]
    preactivation = readout(state).float()
    rate = activation(preactivation).float()
    if preactivation.shape[1:] != (100, 51, 51):
        raise RuntimeError(f"Unexpected exact RR100 map shape: {tuple(preactivation.shape)}")
    return preactivation, rate


def _cached_batch(
    handle: h5py.File,
    name: str,
    identity: Mapping[str, int],
    scale: float,
    frame_start: int,
    frame_stop: int,
    device: torch.device,
) -> torch.Tensor:
    index = (
        identity["image_position"],
        identity["trajectory_position"],
        scale_index(scale),
        slice(frame_start, frame_stop),
    )
    return torch.as_tensor(np.asarray(handle[name][index], dtype=np.float32), device=device)


def _update_identity_diagnostics(
    diagnostics: dict[str, float],
    *,
    literal_sequence: torch.Tensor,
    replay_sequence: torch.Tensor,
    preactivation: torch.Tensor,
    rate: torch.Tensor,
    cached_state: torch.Tensor,
    cached_preactivation: torch.Tensor,
    cached_rate: torch.Tensor,
) -> None:
    values = {
        "literal_replay_max_abs": float((literal_sequence - replay_sequence).abs().max().detach().cpu()),
        "state_cache_relative_rmse": _relative_rmse(replay_sequence[:, :, -1], cached_state),
        "preactivation_cache_relative_rmse": _relative_rmse(preactivation, cached_preactivation),
        "rate_cache_relative_rmse": _relative_rmse(rate, cached_rate),
    }
    for key, value in values.items():
        diagnostics[key] = max(float(diagnostics.get(key, 0.0)), value)
    limits = {
        "literal_replay_max_abs": MAX_LITERAL_REPLAY_ABS,
        "state_cache_relative_rmse": MAX_CACHE_STATE_RELATIVE_RMSE,
        "preactivation_cache_relative_rmse": MAX_CACHE_ENDPOINT_RELATIVE_RMSE,
        "rate_cache_relative_rmse": MAX_CACHE_ENDPOINT_RELATIVE_RMSE,
    }
    failures = {key: values[key] for key in values if not np.isfinite(values[key]) or values[key] > limits[key]}
    if failures:
        raise RuntimeError(f"Intact frozen endpoint identity gate failed: {failures}")


def _load_exact_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    images = pd.read_csv(IMAGE_FEATURE_TABLE).sort_values("image_index").reset_index(drop=True)
    with h5py.File(STATE_CACHE, "r") as state:
        image_ids = np.asarray(state["image_ids"][:], dtype=np.int64)
        trajectory_ids = np.asarray(state["trajectory_ids"][:], dtype=np.int64)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz", allow_pickle=False) as archive:
        base_histories = np.asarray(archive["true_history_xy"][trajectory_ids], dtype=np.float32)
    if image_ids.shape != (8,) or trajectory_ids.shape != (24,) or base_histories.shape != (24, 71, 2):
        raise RuntimeError("Frozen exact subset arrays are not 8 images x 24 trajectories x 71 samples")
    if not all(int(images.iloc[int(value)].image_index) == int(value) for value in image_ids):
        raise RuntimeError("Image feature table no longer permits stable-ID positional lookup")
    return images, image_ids, trajectory_ids, base_histories


def _render_pair(
    scorer: RealTraceMatrixScorer,
    images: pd.DataFrame,
    image_ids: np.ndarray,
    base_histories: np.ndarray,
    identity: Mapping[str, int],
    canvas_cache: dict[Any, Any],
) -> tuple[np.ndarray, torch.Tensor]:
    image_id = int(image_ids[identity["image_position"]])
    patch, _ = extract_patch(images.iloc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
    patch = _standardize_uint_like(patch)
    histories = scaled_histories(
        base_histories[identity["trajectory_position"] : identity["trajectory_position"] + 1],
        np.asarray(SCALES, dtype=np.float32),
    )
    stimuli = (make_corrected_causal_stims(patch, histories, torch=scorer.torch) - 127.0) / 255.0
    movies = stimuli.reshape(len(SCALES), N_FRAMES, *stimuli.shape[1:])
    return histories, movies


def _part_is_complete(
    scope: str,
    stage: str,
    identity: dict[str, int],
    fingerprint: dict[str, Any],
) -> bool:
    directory = _part_dir(scope, stage, identity)
    return (
        valid_complete_marker(
            directory,
            schema_version=SCHEMA_VERSION,
            stage=stage,
            scope=scope,
            identity=identity,
            input_fingerprint=fingerprint,
            required_product_names=STAGE_PRODUCTS[stage],
        )
        is not None
    )


def _validate_current_pilot_authorization() -> dict[str, Any]:
    """Re-derive the pilot gate from currently valid parts before full work.

    Merely finding an ``advance_to_full`` boolean is not sufficient: a gate
    can become stale after code, projectors, calibration, or upstream products
    change.  This check requires every pilot marker to validate against the
    current input fingerprints and deterministically re-applies the gate.
    """
    stored = _read_json(PILOT_GATE)
    if stored.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Pilot-to-full gate schema does not match the current implementation")
    selection = selection_for_scope("pilot")
    for stage in ("transport", "realignment"):
        fingerprint = _input_fingerprint(stage, scope="pilot")
        invalid = [
            identity
            for identity in _stage_identities(selection, stage)
            if not _part_is_complete("pilot", stage, identity, fingerprint)
        ]
        if invalid:
            raise RuntimeError(
                f"Pilot {stage} authorization is stale or incomplete: "
                f"{len(invalid)} invalid of {len(_stage_identities(selection, stage))} parts"
            )
        consolidation = _read_json(
            OUT / "pilot" / stage / "consolidation_manifest.json"
        )
        if not (
            consolidation.get("status") == "complete"
            and int(consolidation.get("parts", -1))
            == len(_stage_identities(selection, stage))
        ):
            raise RuntimeError(f"Pilot {stage} consolidation is not complete")

    transport = pd.read_csv(OUT / "pilot/transport_ablation_results.csv")
    realignment = pd.read_csv(OUT / "pilot/realignment_results.csv")
    transport_manifest = _read_json(
        OUT / "pilot/transport/consolidation_manifest.json"
    )
    realignment_manifest = _read_json(
        OUT / "pilot/realignment/consolidation_manifest.json"
    )
    identity_error = max(
        float(transport_manifest.get("identity_max_relative_error", math.inf)),
        float(realignment_manifest.get("identity_max_relative_error", math.inf)),
    )
    current = pilot_decision_gate(
        transport,
        realignment,
        upstream_gate=upstream_mechanism_gate(),
        identity_max_relative_error=identity_error,
        realignment_shift_diagnostics=pd.read_csv(
            OUT / "pilot/realignment_shift_diagnostics.csv"
        ),
    )
    if not bool(stored.get("advance_to_full", False)):
        raise RuntimeError("Stored pilot-to-full gate does not authorize full confirmation")
    if not bool(current.get("advance_to_full", False)):
        raise RuntimeError(
            "Pilot-to-full decision no longer passes when re-derived from current valid products: "
            f"{current.get('failure_reasons', [])}"
        )
    return {
        "stored_gate": file_record(PILOT_GATE),
        "rederived_status": current["status"],
        "all_current_pilot_parts_valid": True,
    }


def _write_part(
    *,
    scope: str,
    stage: str,
    identity: dict[str, int],
    fingerprint: dict[str, Any],
    accumulator: EndpointAccumulator,
    diagnostics: dict[str, Any],
    shift_rows: list[dict[str, Any]] | None = None,
) -> None:
    directory = _part_dir(scope, stage, identity)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "complete.json"
    if marker.exists():
        marker.unlink()
    summary = accumulator_rows(accumulator, analysis=f"{stage}_part", scope=scope)
    for key, value in identity.items():
        summary[key] = value
    summary_path = directory / "summary.csv"
    arrays_path = directory / "arrays.npz"
    _atomic_csv(summary_path, summary)
    _atomic_npz(
        arrays_path,
        accumulator.payload(
            identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
            diagnostics_json=np.asarray(json.dumps(json_ready(diagnostics), sort_keys=True)),
        ),
    )
    products = [summary_path, arrays_path]
    if stage == "realignment":
        shift_path = directory / "shift_diagnostics.csv"
        _atomic_csv(shift_path, pd.DataFrame(shift_rows or []))
        products.append(shift_path)
    write_complete_marker(
        directory,
        schema_version=SCHEMA_VERSION,
        stage=stage,
        scope=scope,
        identity=identity,
        input_fingerprint=fingerprint,
        products=products,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def _run_transport_part(
    *,
    scope: str,
    identity: dict[str, int],
    fingerprint: dict[str, Any],
    histories: np.ndarray,
    movies: torch.Tensor,
    scorer: RealTraceMatrixScorer,
    model: Any,
    cell: Any,
    readout: Any,
    state_cache: h5py.File,
    map_cache: h5py.File,
    frame_batch_size: int,
    deadline: float,
) -> dict[str, Any]:
    populations = population_definitions(target_units("low"), target_units("high"))
    accumulator = EndpointAccumulator(TRANSPORT_CONDITIONS, SCALES, populations)
    diagnostics: dict[str, float] = {}
    dtype = next(model.parameters()).dtype
    torch_device = next(model.parameters()).device
    for scale_position, scale in enumerate(SCALES):
        for frame_start in range(0, N_FRAMES, int(frame_batch_size)):
            _check_deadline(deadline, 5.0)
            frame_stop = min(frame_start + int(frame_batch_size), N_FRAMES)
            stimulus = movies[scale_position, frame_start:frame_stop].to(torch_device)
            behavior = scorer._zero_behavior(len(stimulus), dtype)
            core = _core_input(model, stimulus, behavior)
            literal_sequence = model.recurrent(core)
            intact_sequence = replay_transport_intervention(cell, core, TRANSPORT_INTERVENTIONS[0])
            intact_z, intact_rate = _endpoint(intact_sequence, readout, model.activation)
            cached_state = _cached_batch(
                state_cache, "h", identity, float(scale), frame_start, frame_stop, torch_device
            )
            cached_z = _cached_batch(
                map_cache, "preactivation", identity, float(scale), frame_start, frame_stop, torch_device
            )
            cached_rate = _cached_batch(
                map_cache, "rate", identity, float(scale), frame_start, frame_stop, torch_device
            )
            _update_identity_diagnostics(
                diagnostics,
                literal_sequence=literal_sequence,
                replay_sequence=intact_sequence,
                preactivation=intact_z,
                rate=intact_rate,
                cached_state=cached_state,
                cached_preactivation=cached_z,
                cached_rate=cached_rate,
            )
            for condition_index, intervention in enumerate(TRANSPORT_INTERVENTIONS):
                _check_deadline(deadline, 5.0)
                sequence = (
                    intact_sequence
                    if condition_index == 0
                    else replay_transport_intervention(cell, core, intervention)
                )
                preactivation, rate = (
                    (intact_z, intact_rate)
                    if condition_index == 0
                    else _endpoint(sequence, readout, model.activation)
                )
                accumulator.add(
                    condition_index,
                    scale_position,
                    preactivation=preactivation,
                    rate=rate,
                    intact_preactivation=intact_z,
                    intact_rate=intact_rate,
                )
            del stimulus, behavior, core, literal_sequence, intact_sequence, intact_z, intact_rate
    _write_part(
        scope=scope,
        stage="transport",
        identity=identity,
        fingerprint=fingerprint,
        accumulator=accumulator,
        diagnostics=diagnostics,
    )
    return diagnostics


def _fold_basis(contrast: str, fold: int, device: torch.device) -> torch.Tensor:
    path = fit_directory(contrast, fold) / "U.npy"
    basis = validate_basis(path)
    return torch.as_tensor(basis, dtype=torch.float32, device=device)


def _eye_shifts_yx(
    history: np.ndarray,
    frames: Sequence[int],
    calibration_matrix: np.ndarray,
) -> np.ndarray:
    eye_xy = np.stack([internal_step_eye_displacements(history, int(frame)) for frame in frames])
    feature_xy = apply_shift_calibration(eye_xy, calibration_matrix)
    return np.asarray(feature_xy[..., [1, 0]], dtype=np.float32)


@torch.no_grad()
def _run_realignment_part(
    *,
    scope: str,
    identity: dict[str, int],
    fingerprint: dict[str, Any],
    histories: np.ndarray,
    movies: torch.Tensor,
    scorer: RealTraceMatrixScorer,
    model: Any,
    cell: Any,
    readout: Any,
    state_cache: h5py.File,
    map_cache: h5py.File,
    frame_batch_size: int,
    max_oracle_lag_px: int,
    calibration_matrix: np.ndarray,
    deadline: float,
) -> dict[str, Any]:
    populations = population_definitions(target_units("low"), target_units("high"))
    accumulator = EndpointAccumulator(REALIGNMENT_CONDITIONS, REALIGNMENT_SCALES, populations)
    diagnostics: dict[str, float] = {}
    shift_rows: list[dict[str, Any]] = []
    dtype = next(model.parameters()).dtype
    torch_device = next(model.parameters()).device
    fold = int(identity["fold"])
    condition_index = {value: index for index, value in enumerate(REALIGNMENT_CONDITIONS)}
    for output_scale_index, scale in enumerate(REALIGNMENT_SCALES):
        source_scale_index = scale_index(scale)
        contrast = "high_0_to_1" if np.isclose(scale, 1.0) else "high_1_to_3"
        basis = _fold_basis(contrast, fold, torch_device)
        for frame_start in range(0, N_FRAMES, int(frame_batch_size)):
            _check_deadline(deadline, 5.0)
            frame_stop = min(frame_start + int(frame_batch_size), N_FRAMES)
            frames = list(range(frame_start, frame_stop))
            stimulus = movies[source_scale_index, frame_start:frame_stop].to(torch_device)
            behavior = scorer._zero_behavior(len(stimulus), dtype)
            core = _core_input(model, stimulus, behavior)
            literal_sequence = model.recurrent(core)
            no_shift = replay_realignment(
                cell,
                core,
                basis,
                shift_yx_by_step=None,
                target="none",
            )
            intact_z, intact_rate = _endpoint(no_shift.sequence, readout, model.activation)
            cached_state = _cached_batch(
                state_cache, "h", identity, scale, frame_start, frame_stop, torch_device
            )
            cached_z = _cached_batch(
                map_cache, "preactivation", identity, scale, frame_start, frame_stop, torch_device
            )
            cached_rate = _cached_batch(
                map_cache, "rate", identity, scale, frame_start, frame_stop, torch_device
            )
            _update_identity_diagnostics(
                diagnostics,
                literal_sequence=literal_sequence,
                replay_sequence=no_shift.sequence,
                preactivation=intact_z,
                rate=intact_rate,
                cached_state=cached_state,
                cached_preactivation=cached_z,
                cached_rate=cached_rate,
            )
            accumulator.add(
                condition_index["no_shift_intact"],
                output_scale_index,
                preactivation=intact_z,
                rate=intact_rate,
                intact_preactivation=intact_z,
                intact_rate=intact_rate,
            )
            expected_yx = _eye_shifts_yx(histories[source_scale_index], frames, calibration_matrix)
            expected_tensor = torch.as_tensor(expected_yx, dtype=core.dtype, device=torch_device)
            requests: list[tuple[str, str, torch.Tensor | None, bool]]
            if np.isclose(scale, 3.0):
                flat = expected_yx.reshape(-1, 2)
                labels = [
                    (
                        f"fold={fold}:image={identity['image_position']}:"
                        f"trajectory={identity['trajectory_position']}:scale=3:"
                        f"frame={frame}:step={step}"
                    )
                    for frame in frames
                    for step in range(expected_yx.shape[1])
                ]
                random_yx = matched_random_directions(flat, labels=labels).reshape(expected_yx.shape)
                requests = [
                    ("eye_correct_candidate_p", "candidate_p", expected_tensor, False),
                    ("activation_oracle_candidate_p", "candidate_p", None, True),
                    ("eye_opposite_candidate_p", "candidate_p", -expected_tensor, False),
                    (
                        "eye_random_matched_candidate_p",
                        "candidate_p",
                        torch.as_tensor(random_yx, dtype=core.dtype, device=torch_device),
                        False,
                    ),
                    ("eye_correct_complementary_q", "complementary_q", expected_tensor, False),
                ]
            else:
                requests = [
                    (
                        "induced_eye_misalignment_candidate_p",
                        "candidate_p",
                        -expected_tensor,
                        False,
                    )
                ]
            for name, target, requested, oracle in requests:
                _check_deadline(deadline, 5.0)
                replay = replay_realignment(
                    cell,
                    core,
                    basis,
                    shift_yx_by_step=requested,
                    target=target,  # type: ignore[arg-type]
                    oracle=oracle,
                    max_oracle_lag_px=max_oracle_lag_px,
                )
                preactivation, rate = _endpoint(replay.sequence, readout, model.activation)
                accumulator.add(
                    condition_index[name],
                    output_scale_index,
                    preactivation=preactivation,
                    rate=rate,
                    intact_preactivation=intact_z,
                    intact_rate=intact_rate,
                )
                applied = replay.applied_shift_yx_px.detach().cpu().numpy()
                valid = replay.oracle_valid.detach().cpu().numpy()
                for local_frame, frame in enumerate(frames):
                    for internal_step in range(applied.shape[1]):
                        shift_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "scope": scope,
                                "fold": fold,
                                "image_position": identity["image_position"],
                                "trajectory_position": identity["trajectory_position"],
                                "scale": scale,
                                "frame_position": frame,
                                "internal_step": internal_step,
                                "condition": name,
                                "target": target,
                                "source": "activation_oracle" if oracle else "eye_calibration",
                                "applied_shift_y_px": float(applied[local_frame, internal_step, 0]),
                                "applied_shift_x_px": float(applied[local_frame, internal_step, 1]),
                                "applied_shift_magnitude_px": float(
                                    np.linalg.norm(applied[local_frame, internal_step])
                                ),
                                "eye_derived_correct_shift_y_px": float(
                                    expected_yx[local_frame, internal_step, 0]
                                ),
                                "eye_derived_correct_shift_x_px": float(
                                    expected_yx[local_frame, internal_step, 1]
                                ),
                                "valid": bool(valid[local_frame, internal_step]),
                                "oracle_at_search_boundary": bool(
                                    oracle
                                    and np.max(np.abs(applied[local_frame, internal_step]))
                                    >= float(max_oracle_lag_px) - 1e-6
                                ),
                                "within_window_order": "newer_support_to_older_support",
                            }
                        )
            del stimulus, behavior, core, literal_sequence, no_shift, intact_z, intact_rate
    _write_part(
        scope=scope,
        stage="realignment",
        identity=identity,
        fingerprint=fingerprint,
        accumulator=accumulator,
        diagnostics=diagnostics,
        shift_rows=shift_rows,
    )
    return diagnostics


def _kernel_audit(cell: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in sorted(
        value.permutation_seed
        for value in TRANSPORT_INTERVENTIONS
        if value.permutation_seed is not None
    ):
        for label, layer in (
            ("update_gate", cell.update_gate),
            ("reset_gate", cell.reset_gate),
            ("candidate", cell.out_gate),
        ):
            hidden = layer.weight[:, -int(cell.hidden_size) :]
            transformed = transform_recurrent_kernel(hidden, "permute_offcenter", seed=int(seed))
            invariants = recurrent_kernel_invariants(hidden, transformed)
            if not (
                invariants["center_max_abs"] == 0.0
                and invariants["total_norm_relative_error"] <= 1e-7
                and invariants["offcenter_tap_norm_multiset_equal"] is True
            ):
                raise RuntimeError(f"Offset permutation invariant failed for {label}, seed={seed}: {invariants}")
            rows.append({"seed": int(seed), "kernel": label, **invariants})
    return {
        "schema_version": SCHEMA_VERSION,
        "definition": "one global row-major off-center permutation per seed, shared by all recurrent kernels",
        "all_channel_matrices_preserved": True,
        "center_taps_preserved": True,
        "total_norm_preserved": True,
        "rows": rows,
    }


def _check_deadline(deadline: float, reserve_seconds: float) -> None:
    if time.monotonic() + max(float(reserve_seconds), 0.0) >= deadline:
        raise ComputeBoundaryReached(
            "Cumulative ten-GPU-hour boundary reached before another complete intervention pair"
        )


def _write_selection(selection: InterventionSelection) -> None:
    destination = OUT / selection.scope / "selection.csv"
    table = pd.DataFrame(selection.rows)
    _atomic_csv(destination, table)
    atomic_json(
        OUT / selection.scope / "selection_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "scope": selection.scope,
            "n_images": len(selection.image_positions),
            "n_trajectories": len(selection.trajectory_positions),
            "n_transport_cross_product_pairs": len(selection.transport_pairs),
            "n_foldwise_heldout_realign_pairs": len(selection.heldout_realign_pairs),
            "transport_pairs": selection.transport_pairs,
            "heldout_realign_pairs": selection.heldout_realign_pairs,
            "realignment_fold_counts": {
                str(fold): sum(
                    int(row[0]) == fold for row in selection.heldout_realign_pairs
                )
                for fold in range(4)
            },
            "pilot_fold_2_note": (
                "The already frozen 4x12 pilot contains no image from fold 2. "
                "No replacement was selected after outcomes; fold 2 enters only the gated full run."
                if selection.scope == "pilot"
                else None
            ),
            "outcome_used_for_selection": False,
            "foldwise_projectors_for_inference": True,
            "consensus_projector_loaded": False,
        },
    )


def plan_payload(args: argparse.Namespace, selection: InterventionSelection) -> dict[str, Any]:
    blockers: list[str] = []
    upstream: dict[str, Any] | None = None
    try:
        _assert_cache_gate()
        upstream = upstream_mechanism_gate()
        if not upstream["passed"]:
            blockers.append("upstream execution-readiness gate is not passing")
    except Exception as error:
        blockers.append(f"{type(error).__name__}: {error}")
    if args.scope == "full":
        try:
            _validate_current_pilot_authorization()
        except Exception as error:
            blockers.append(f"pilot authorization unavailable or stale: {error}")
    transport_forward_batches = (
        len(selection.transport_pairs) * len(SCALES) * math.ceil(N_FRAMES / args.frame_batch_size)
    )
    realign_forward_batches = (
        len(selection.heldout_realign_pairs)
        * len(REALIGNMENT_SCALES)
        * math.ceil(N_FRAMES / args.frame_batch_size)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": args.scope,
        "stage": args.stage,
        "blockers": blockers,
        "production_ready": not blockers,
        "upstream_gate": upstream,
        "selection": {
            "images": len(selection.image_positions),
            "trajectories": len(selection.trajectory_positions),
            "transport_pairs": len(selection.transport_pairs),
            "foldwise_heldout_realign_pairs": len(selection.heldout_realign_pairs),
        },
        "transport_conditions": TRANSPORT_CONDITIONS,
        "realignment_conditions": REALIGNMENT_CONDITIONS,
        "movement_scales": np.asarray(SCALES).tolist(),
        "realignment_scales": REALIGNMENT_SCALES,
        "transport_core_forward_batches": transport_forward_batches,
        "realignment_core_forward_batches": realign_forward_batches,
        "identity_contract": {
            "literal_replay_max_abs": MAX_LITERAL_REPLAY_ABS,
            "cached_state_relative_rmse": MAX_CACHE_STATE_RELATIVE_RMSE,
            "cached_endpoint_relative_rmse": MAX_CACHE_ENDPOINT_RELATIVE_RMSE,
        },
        "runtime_expectation": (
            "Time one complete pilot pair first. Core rendering is shared across conditions; recurrent "
            "replays/readouts dominate. Expected pilot order is tens of minutes to a few hours, and full "
            "confirmation is roughly four times the pair count. Every call checks between frame batches, "
            "conditions, and complete pairs at the shared cumulative ten-hour boundary."
        ),
        "no_model_run_during_plan": True,
    }


def _execute_gpu(args: argparse.Namespace, selection: InterventionSelection) -> dict[str, Any]:
    if not str(args.device).startswith("cuda"):
        raise ValueError("Production causal interventions require an explicitly ledgered CUDA device")
    _assert_cache_gate()
    upstream = upstream_mechanism_gate()
    if not upstream["passed"]:
        raise RuntimeError(f"Upstream execution-readiness gate failed: {upstream}")
    pilot_authorization = None
    if args.scope == "full":
        pilot_authorization = _validate_current_pilot_authorization()
    stages = ("transport", "realignment") if args.stage == "all" else (args.stage,)
    fingerprints = {
        stage: _input_fingerprint(stage, scope=args.scope) for stage in stages
    }
    pending = {
        stage: [
            identity
            for identity in _stage_identities(selection, stage)
            if args.overwrite_parts
            or not _part_is_complete(args.scope, stage, identity, fingerprints[stage])
        ]
        for stage in stages
    }
    if args.max_pairs > 0:
        pending = {stage: values[: int(args.max_pairs)] for stage, values in pending.items()}
    if not any(pending.values()):
        value = {
            "schema_version": SCHEMA_VERSION,
            "status": "already_complete",
            "arguments": vars(args),
            "completed_parts_this_call": {stage: 0 for stage in stages},
            "pending_parts_requested": {stage: 0 for stage in stages},
            "model_loaded": False,
            "gpu_ledger_debited": False,
        }
        atomic_json(OUT / args.scope / "run_manifest.json", value)
        return {"status": "already_complete", "completed": value["completed_parts_this_call"]}
    before = load_budget()
    if float(before["remaining_gpu_hours"]) <= 0.0:
        raise RuntimeError("Cumulative ten-GPU-hour boundary already reached; write the interim report")
    completed: dict[str, int] = {stage: 0 for stage in stages}
    status = "running"
    caught: BaseException | None = None
    with exclusive_gpu_lock():
        started = time.monotonic()
        locked_fingerprints = {
            stage: _input_fingerprint(stage, scope=args.scope) for stage in stages
        }
        if locked_fingerprints != fingerprints:
            raise RuntimeError(
                "A frozen input changed between preflight and acquisition of the shared GPU lock"
            )
        if args.overwrite_parts:
            for stage in stages:
                for identity in pending[stage]:
                    target = _part_dir(args.scope, stage, identity)
                    if target.is_dir():
                        shutil.rmtree(target)
        deadline = budget_deadline()
        try:
            _check_deadline(deadline, args.minimum_next_pair_seconds)
            scorer, model, cell, readout = _load_model(args.device)
            config = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)[
                "hyper_parameters"
            ]["model_config_dict"]
            if derive_fig4_convgru_input_lag_supports(config) != FIG4_CONVGRU_INPUT_LAG_SUPPORTS:
                raise RuntimeError("Frozen within-window ConvGRU input supports changed")
            atomic_json(OUT / "kernel_intervention_audit.json", _kernel_audit(cell))
            images, image_ids, _, base_histories = _load_exact_inputs()
            calibration = _read_json(SHIFT_CALIBRATION)
            calibration_matrix = np.asarray(
                calibration["matrix_feature_px_per_eye_deg"], dtype=np.float64
            )
            canvas_cache: dict[Any, Any] = {}
            with h5py.File(STATE_CACHE, "r") as state_cache, h5py.File(MAP_CACHE, "r") as map_cache:
                for stage in stages:
                    for identity in pending[stage]:
                        _check_deadline(deadline, args.minimum_next_pair_seconds)
                        histories, movies = _render_pair(
                            scorer,
                            images,
                            image_ids,
                            base_histories,
                            identity,
                            canvas_cache,
                        )
                        if stage == "transport":
                            _run_transport_part(
                                scope=args.scope,
                                identity=identity,
                                fingerprint=fingerprints[stage],
                                histories=histories,
                                movies=movies,
                                scorer=scorer,
                                model=model,
                                cell=cell,
                                readout=readout,
                                state_cache=state_cache,
                                map_cache=map_cache,
                                frame_batch_size=args.frame_batch_size,
                                deadline=deadline,
                            )
                        else:
                            _run_realignment_part(
                                scope=args.scope,
                                identity=identity,
                                fingerprint=fingerprints[stage],
                                histories=histories,
                                movies=movies,
                                scorer=scorer,
                                model=model,
                                cell=cell,
                                readout=readout,
                                state_cache=state_cache,
                                map_cache=map_cache,
                                frame_batch_size=args.frame_batch_size,
                                max_oracle_lag_px=args.max_oracle_lag_px,
                                calibration_matrix=calibration_matrix,
                                deadline=deadline,
                            )
                        completed[stage] += 1
                        print(
                            f"completed {stage} {completed[stage]}/{len(pending[stage])}: {identity}",
                            flush=True,
                        )
                        del histories, movies
            status = "complete_requested_work"
        except ComputeBoundaryReached as error:
            status = "gpu_boundary_reached"
            caught = error
        except BaseException as error:
            status = "failed"
            caught = error
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(torch.device(args.device))
            except BaseException as error:
                if caught is None:
                    caught = error
                    status = "failed"
            elapsed = time.monotonic() - started
            ledger = record_gpu_time(
                f"registration_causal_interventions:{args.scope}:{args.stage}",
                elapsed,
                {
                    "device": args.device,
                    "scope": args.scope,
                    "stages": stages,
                    "completed_parts": completed,
                    "status": status,
                    "error": None if caught is None else repr(caught),
                    "exact_corrected_renderer": True,
                    "consensus_projector_loaded": False,
                },
            )
            atomic_json(
                OUT / args.scope / "run_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "arguments": vars(args),
                    "completed_parts_this_call": completed,
                    "pending_parts_requested": {key: len(value) for key, value in pending.items()},
                    "upstream_gate": upstream,
                    "pilot_authorization": pilot_authorization,
                    "gpu_ledger": ledger,
                },
            )
    if caught is not None:
        raise caught
    return {"status": status, "completed": completed}


def _valid_part_products(
    scope: str,
    stage: str,
    selection: InterventionSelection,
    fingerprint: dict[str, Any],
) -> tuple[list[Path], list[dict[str, int]], list[dict[str, Any]]]:
    arrays: list[Path] = []
    missing: list[dict[str, int]] = []
    markers: list[dict[str, Any]] = []
    for identity in _stage_identities(selection, stage):
        directory = _part_dir(scope, stage, identity)
        marker = valid_complete_marker(
            directory,
            schema_version=SCHEMA_VERSION,
            stage=stage,
            scope=scope,
            identity=identity,
            input_fingerprint=fingerprint,
            required_product_names=STAGE_PRODUCTS[stage],
        )
        if marker is None:
            missing.append(identity)
            continue
        arrays.append(directory / "arrays.npz")
        markers.append(marker)
    return arrays, missing, markers


def _consolidate_stage(
    scope: str,
    stage: str,
    selection: InterventionSelection,
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    fingerprint = _input_fingerprint(stage, scope=scope)
    array_paths, missing, markers = _valid_part_products(
        scope, stage, selection, fingerprint
    )
    expected = len(_stage_identities(selection, stage))
    if missing:
        value = {
            "schema_version": SCHEMA_VERSION,
            "scope": scope,
            "stage": stage,
            "status": "incomplete",
            "present_parts": len(array_paths),
            "expected_parts": expected,
            "missing": missing,
        }
        atomic_json(OUT / scope / stage / "consolidation_manifest.json", value)
        if not allow_incomplete:
            raise RuntimeError(f"{stage} is incomplete: {len(array_paths)}/{expected} valid parts")
        return value
    conditions = TRANSPORT_CONDITIONS if stage == "transport" else REALIGNMENT_CONDITIONS
    scales = tuple(float(value) for value in SCALES) if stage == "transport" else REALIGNMENT_SCALES
    populations = population_definitions(target_units("low"), target_units("high"))
    aggregate = EndpointAccumulator(conditions, scales, populations)
    shift_tables: list[pd.DataFrame] = []
    identity_error = 0.0
    for path, marker in zip(array_paths, markers):
        with np.load(path, allow_pickle=False) as archive:
            aggregate.add_payload({key: np.asarray(archive[key]) for key in archive.files})
        identity_error = max(
            identity_error,
            float(marker.get("diagnostics", {}).get("state_cache_relative_rmse", 0.0)),
            float(marker.get("diagnostics", {}).get("preactivation_cache_relative_rmse", 0.0)),
            float(marker.get("diagnostics", {}).get("rate_cache_relative_rmse", 0.0)),
        )
        if stage == "realignment":
            shift_tables.append(pd.read_csv(path.parent / "shift_diagnostics.csv"))
    analysis_name = "transport_ablation" if stage == "transport" else "realignment"
    rows = accumulator_rows(aggregate, analysis=analysis_name, scope=scope)
    directory = OUT / scope
    csv_path = directory / f"{analysis_name}_results.csv"
    arrays_path = directory / f"{analysis_name}_maps.npz"
    _atomic_csv(csv_path, rows)
    payload = aggregate.payload(
        normalized_population_rate_maps=normalized_population_maps(aggregate),
        map_definition=np.asarray(
            "g(x,y)=r(x,y)/mean_xy r; expected-spike weighted across complete frames, units, and pairs"
        ),
        individual_conditions_rescaled=np.asarray(False),
    )
    _atomic_npz(arrays_path, payload)
    products: list[Path] = [csv_path, arrays_path]
    shift_path: Path | None = None
    if stage == "realignment":
        shift_path = directory / "realignment_shift_diagnostics.csv"
        _atomic_csv(shift_path, pd.concat(shift_tables, ignore_index=True, sort=False))
        products.append(shift_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "stage": stage,
        "status": "complete",
        "parts": len(array_paths),
        "expected_parts": expected,
        "identity_max_relative_error": identity_error,
        "products": [file_record(path) for path in products],
        "part_markers": [str(path.parent / "complete.json") for path in array_paths],
        "foldwise_projectors_for_inference": stage == "realignment",
        "consensus_projector_loaded": False,
        "complete_normalized_maps_saved": True,
    }
    atomic_json(OUT / scope / stage / "consolidation_manifest.json", manifest)
    return manifest


def _publish_canonical_scope(scope: str) -> dict[str, Any]:
    """Atomically publish one fully consolidated scope, preferring full over pilot."""
    if CANONICAL_MANIFEST.is_file():
        existing = _read_json(CANONICAL_MANIFEST)
        if existing.get("scope") == "full" and scope == "pilot":
            products = existing.get("products", [])
            valid = bool(products) and all(
                Path(item.get("path", "")).is_file()
                and sha256_file(Path(item["path"])) == item.get("sha256")
                for item in products
            )
            if not valid:
                raise RuntimeError(
                    "The canonical full intervention products are corrupt; refusing to replace "
                    "them with a pilot consolidation"
                )
            return {"published": False, "reason": "validated full scope already canonical"}
    source_directory = OUT / scope
    names = (
        "transport_ablation_results.csv",
        "transport_ablation_maps.npz",
        "realignment_results.csv",
        "realignment_maps.npz",
        "realignment_shift_diagnostics.csv",
    )
    sources = [source_directory / name for name in names]
    if any(not path.is_file() for path in sources):
        raise RuntimeError("Cannot publish canonical intervention products before both stages complete")
    destinations = [BASE_OUT / name for name in names]
    for source, destination in zip(sources, destinations):
        _atomic_copy(source, destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": scope,
        "products": [file_record(path) for path in destinations],
        "source_analysis_manifest": file_record(source_directory / "analysis_manifest.json"),
        "publication_rule": "full supersedes pilot; a later pilot cannot replace validated full products",
    }
    atomic_json(CANONICAL_MANIFEST, manifest)
    return {"published": True, "scope": scope, "manifest": str(CANONICAL_MANIFEST)}


def consolidate_scope(args: argparse.Namespace, selection: InterventionSelection) -> dict[str, Any]:
    results = {
        stage: _consolidate_stage(
            args.scope,
            stage,
            selection,
            allow_incomplete=args.allow_incomplete_consolidation,
        )
        for stage in ("transport", "realignment")
    }
    if all(value.get("status") == "complete" for value in results.values()):
        transport = pd.read_csv(OUT / args.scope / "transport_ablation_results.csv")
        realignment = pd.read_csv(OUT / args.scope / "realignment_results.csv")
        identity_error = max(
            float(results["transport"]["identity_max_relative_error"]),
            float(results["realignment"]["identity_max_relative_error"]),
        )
        upstream = upstream_mechanism_gate()
        if args.scope == "pilot":
            gate = pilot_decision_gate(
                transport,
                realignment,
                upstream_gate=upstream,
                identity_max_relative_error=identity_error,
                realignment_shift_diagnostics=pd.read_csv(
                    OUT / args.scope / "realignment_shift_diagnostics.csv"
                ),
            )
            atomic_json(PILOT_GATE, gate)
        analysis_manifest = OUT / args.scope / "analysis_manifest.json"
        atomic_json(
            analysis_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "scope": args.scope,
                "transport": results["transport"],
                "realignment": results["realignment"],
                "upstream_gate": upstream,
                "pilot_gate": str(PILOT_GATE) if args.scope == "pilot" else file_record(PILOT_GATE),
                "exact_corrected_renderer": True,
                "frozen_checkpoint": file_record(Path(MODEL_CHECKPOINT_PATH)),
                "within_window_convgru_order": "8 steps, newer evidence support to older; reset for every scored output",
                "within_window_lag_supports": FIG4_CONVGRU_INPUT_LAG_SUPPORTS,
                "foldwise_heldout_projectors_for_inference": True,
                "consensus_projectors_for_visualization_only": True,
                "no_unique_recurrent_pathway_claimed": True,
            },
        )
        _publish_canonical_scope(args.scope)
    return results


def run(args: argparse.Namespace) -> int:
    if args.frame_batch_size < 1 or args.max_pairs < 0:
        raise ValueError("Frame batch size must be positive and max-pairs cannot be negative")
    selection = selection_for_scope(args.scope)
    _write_selection(selection)
    if args.stage == "plan":
        payload = plan_payload(args, selection)
        atomic_json(OUT / args.scope / "execution_plan.json", payload)
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    if args.stage == "consolidate":
        consolidate_scope(args, selection)
        return 0
    _execute_gpu(args, selection)
    # Publish only if all expected parts for both stages are now complete.
    if args.stage == "all":
        consolidate_scope(args, selection)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
