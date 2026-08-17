from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ANALYSIS_SEED,
    CACHE,
    CONFIG,
    CONTRASTS,
    MAP_CACHE,
    N_CHANNELS,
    N_FRAMES,
    N_IMAGES,
    N_SCALES,
    N_TRAJECTORIES,
    N_UNITS,
    OUT,
    READOUT_CACHE,
    ROOT,
    SCALES,
    SOURCE_EXACT,
    SOURCE_SELECTION,
    STATE_CACHE,
    ensure_output_dirs,
    sha256_file,
    write_json,
)


PHASE_OUT = SOURCE_EXACT.parents[1]
PHASE_MANIFEST = PHASE_OUT / "run_manifest.json"
SELECTED_IMAGES = SOURCE_SELECTION / "selected_images.csv"
SELECTED_TRAJECTORIES = SOURCE_SELECTION / "selected_traces.csv"
HISTORY_BANK = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1"
    / "banks/corrected_history_trajectory_banks.npz"
)
HISTORY_MANIFEST = HISTORY_BANK.parent / "bank_manifest.json"
SSI_MECHANISM_MANIFEST = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/ssi_mechanism_v2/run_manifest.json"
)
SOURCE_MATRIX = (
    ROOT
    / "outputs/active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged"
)
UNIT_TABLE = SOURCE_MATRIX / "unit_feature_table.csv"
IMAGE_TABLE = SOURCE_MATRIX / "image_feature_table.csv"

SCHEMA_VERSION = "fig4-causal-low-rank-v1"
N_TRAJECTORY_STRATA = 6
TRAJECTORIES_PER_STRATUM = 4
VALIDATION_STRATA = tuple(range(N_TRAJECTORY_STRATA))
SCREENING_FOLD = 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_selection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_rows = _read_csv(SELECTED_IMAGES)
    trajectory_rows = _read_csv(SELECTED_TRAJECTORIES)
    if len(image_rows) != N_IMAGES:
        raise ValueError(f"Expected {N_IMAGES} selected images, found {len(image_rows)}")
    if len(trajectory_rows) != N_TRAJECTORIES:
        raise ValueError(f"Expected {N_TRAJECTORIES} selected trajectories, found {len(trajectory_rows)}")

    images = [
        {
            "position": position,
            "image_id": int(row["image_index"]),
            "source_row": int(row["source_row"]),
            "session": row["session"],
            "trial_idx": int(row["trial_idx"]),
        }
        for position, row in enumerate(image_rows)
    ]
    path_key = (
        "rendered_path_length_arcmin"
        if "rendered_path_length_arcmin" in trajectory_rows[0]
        else "selection_target_path_arcmin"
    )
    trajectories = [
        {
            "position": position,
            "trajectory_id": int(row["trace_bank_index"]),
            "source_row": int(row["source_row"]),
            "session": row["session"],
            "trial_idx": int(row["trial_idx"]),
            "path_length_arcmin": float(row[path_key]),
            "selection_quantile": float(row["selection_quantile"]),
            "trace_hash": row["trace_hash"],
        }
        for position, row in enumerate(trajectory_rows)
    ]

    with np.load(SOURCE_EXACT, allow_pickle=False) as archive:
        exact_image_ids = np.asarray(archive["selected_image_index"], dtype=np.int64)
        exact_trajectory_ids = np.asarray(archive["selected_trace_index"], dtype=np.int64)
        exact_scales = np.asarray(archive["scales"], dtype=np.float64)
        low_units = np.asarray(archive["low_unit_indices"], dtype=np.int64)
        high_units = np.asarray(archive["high_unit_indices"], dtype=np.int64)
        array_metadata = {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in archive.files
        }

    image_ids = np.asarray([row["image_id"] for row in images], dtype=np.int64)
    trajectory_ids = np.asarray([row["trajectory_id"] for row in trajectories], dtype=np.int64)
    if not np.array_equal(image_ids, exact_image_ids):
        raise ValueError("Selected-image CSV order differs from exact-forward archive")
    if not np.array_equal(trajectory_ids, exact_trajectory_ids):
        raise ValueError("Selected-trajectory CSV order differs from exact-forward archive")
    if not np.allclose(exact_scales, SCALES):
        raise ValueError(f"Exact-forward scales differ: {exact_scales}")
    if len(low_units) != 71 or len(high_units) != 29:
        raise ValueError(f"Historical SF split changed: low={len(low_units)}, high={len(high_units)}")
    if sorted(np.concatenate((low_units, high_units)).tolist()) != list(range(N_UNITS)):
        raise ValueError("Historical SF populations do not partition RR100")

    exact = {
        "array_metadata": array_metadata,
        "low_unit_indices": low_units.tolist(),
        "high_unit_indices": high_units.tolist(),
    }
    return images, trajectories, exact


def _split_record(
    image_positions: list[int],
    trajectory_positions: list[int],
    images: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    image_positions = [int(value) for value in image_positions]
    trajectory_positions = [int(value) for value in trajectory_positions]
    return {
        "image_positions": image_positions,
        "trajectory_positions": trajectory_positions,
        "image_ids": [images[value]["image_id"] for value in image_positions],
        "trajectory_ids": [trajectories[value]["trajectory_id"] for value in trajectory_positions],
        "n_image_trajectory_pairs": len(image_positions) * len(trajectory_positions),
    }


def _pair_record(
    pairs: list[tuple[int, int]],
    images: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record a non-rectangular pair subset and its complete identity pool."""
    pairs = [(int(image), int(trajectory)) for image, trajectory in pairs]
    image_positions = sorted({image for image, _ in pairs}, key=lambda value: images[value]["image_id"])
    trajectory_positions = sorted(
        {trajectory for _, trajectory in pairs},
        key=lambda value: trajectories[value]["path_length_arcmin"],
    )
    return {
        "image_positions": image_positions,
        "trajectory_positions": trajectory_positions,
        "image_ids": [images[value]["image_id"] for value in image_positions],
        "trajectory_ids": [trajectories[value]["trajectory_id"] for value in trajectory_positions],
        "pair_positions": [[image, trajectory] for image, trajectory in pairs],
        "pair_ids": [
            [images[image]["image_id"], trajectories[trajectory]["trajectory_id"]]
            for image, trajectory in pairs
        ],
        "n_image_trajectory_pairs": len(pairs),
    }


def build_fold_assignments(seed: int = ANALYSIS_SEED) -> dict[str, Any]:
    """Build the fixed crossed folds without touching the model or output caches."""
    images, trajectories, _ = _load_selection()
    image_id_to_position = {row["image_id"]: row["position"] for row in images}

    sorted_image_ids = sorted(image_id_to_position)
    image_fold_positions = [
        [image_id_to_position[value] for value in sorted_image_ids[start : start + 2]]
        for start in range(0, N_IMAGES, 2)
    ]

    ordered_trajectories = sorted(
        trajectories,
        key=lambda row: (row["path_length_arcmin"], row["trajectory_id"]),
    )
    strata: list[list[int]] = []
    for stratum_index in range(N_TRAJECTORY_STRATA):
        start = stratum_index * TRAJECTORIES_PER_STRATUM
        positions = [row["position"] for row in ordered_trajectories[start : start + TRAJECTORIES_PER_STRATUM]]
        if len(positions) != TRAJECTORIES_PER_STRATUM:
            raise RuntimeError("Trajectory strata are incomplete")
        strata.append(positions)

    rng = np.random.default_rng(int(seed))
    trajectory_fold_positions: list[list[int]] = [[] for _ in range(4)]
    trajectory_fold_strata: list[list[int]] = [[] for _ in range(4)]
    for stratum_index, positions in enumerate(strata):
        # One trajectory from every adjacent path-quantile stratum goes to
        # each fold.  The sole random operation is this recorded permutation.
        permuted = np.asarray(positions)[rng.permutation(TRAJECTORIES_PER_STRATUM)]
        for fold_index, position in enumerate(permuted.tolist()):
            trajectory_fold_positions[fold_index].append(int(position))
            trajectory_fold_strata[fold_index].append(stratum_index)

    folds: list[dict[str, Any]] = []
    all_image_positions = set(range(N_IMAGES))
    all_trajectory_positions = set(range(N_TRAJECTORIES))
    for fold_index in range(4):
        test_images = sorted(image_fold_positions[fold_index], key=lambda value: images[value]["image_id"])
        test_trajectories = sorted(
            trajectory_fold_positions[fold_index],
            key=lambda value: trajectories[value]["path_length_arcmin"],
        )
        outer_images = sorted(all_image_positions - set(test_images), key=lambda value: images[value]["image_id"])
        outer_trajectories = sorted(
            all_trajectory_positions - set(test_trajectories),
            key=lambda value: trajectories[value]["path_length_arcmin"],
        )

        validation_rng = np.random.default_rng(int(seed) + 10_000 + fold_index)
        validation_trajectories: list[int] = []
        for stratum_index in VALIDATION_STRATA:
            candidates = sorted(
                set(strata[stratum_index]) & set(outer_trajectories),
                key=lambda value: trajectories[value]["path_length_arcmin"],
            )
            validation_trajectories.append(
                int(candidates[int(validation_rng.integers(len(candidates)))])
            )

        validation_images = np.asarray(outer_images)[validation_rng.permutation(len(outer_images))].tolist()
        validation_pairs = list(zip(validation_images, validation_trajectories))
        validation_pair_set = set(validation_pairs)
        outer_pairs = [(image, trajectory) for image in outer_images for trajectory in outer_trajectories]
        fit_pairs = [pair for pair in outer_pairs if pair not in validation_pair_set]

        # Early stopping uses six held-out combinations, one per training image
        # and path stratum. Every outer-training image and trajectory identity
        # remains in the gradient-fitting pool through other combinations.
        outer_train = _split_record(outer_images, outer_trajectories, images, trajectories)
        train = _pair_record(fit_pairs, images, trajectories)
        validation = _pair_record(validation_pairs, images, trajectories)
        test = _split_record(test_images, test_trajectories, images, trajectories)
        folds.append(
            {
                "index": fold_index,
                "outer_train": outer_train,
                "train": train,
                "validation": validation,
                "test": test,
                "unused_outer_train_cross_pairs": 0,
                "validation_seed": int(seed) + 10_000 + fold_index,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "random_generator": "numpy.random.Generator(PCG64)",
        "image_position_order": images,
        "trajectory_position_order": trajectories,
        "image_fold_rule": "sort stable image_index; assign adjacent pairs to folds 0..3",
        "trajectory_fold_rule": (
            "sort by rendered path length; form six adjacent strata of four; "
            "seeded permutation assigns one trajectory per stratum to each fold"
        ),
        "validation_rule": {
            "description": (
                "Within each 6-image x 18-trajectory outer-training rectangle, hold out six "
                "image-trajectory combinations: one per image and one trajectory per global "
                "path stratum. The combinations are excluded from gradient fitting, while all "
                "training identities remain represented by other combinations."
            ),
            "trajectory_strata": list(VALIDATION_STRATA),
            "n_validation_pairs": 6,
            "frames_from_each_pair_remain_together": True,
            "validation_never_used_for_reported_test_metrics": True,
        },
        "trajectory_strata": [
            {
                "stratum": stratum_index,
                "trajectory_positions": positions,
                "trajectory_ids": [trajectories[value]["trajectory_id"] for value in positions],
                "path_length_arcmin": [trajectories[value]["path_length_arcmin"] for value in positions],
            }
            for stratum_index, positions in enumerate(strata)
        ],
        "image_folds": [
            _split_record(positions, [], images, trajectories) for positions in image_fold_positions
        ],
        "trajectory_folds": [
            {
                **_split_record([], positions, images, trajectories),
                "path_strata": trajectory_fold_strata[index],
            }
            for index, positions in enumerate(trajectory_fold_positions)
        ],
        "folds": folds,
    }


def build_optimization_config() -> dict[str, Any]:
    restart_seeds: list[dict[str, Any]] = []
    learned_ranks = (1, 2, 4, 8, 16, 32)
    for contrast_index, contrast in enumerate(CONTRASTS):
        for fold_index in range(4):
            for rank in learned_ranks:
                restart_seeds.extend(
                    [
                        {
                            "contrast": contrast.key,
                            "fold": fold_index,
                            "rank": rank,
                            "initialization": "random_orthonormal_1",
                            "seed": ANALYSIS_SEED + 100_000 * contrast_index + 10_000 * fold_index + 100 * rank + 1,
                        },
                        {
                            "contrast": contrast.key,
                            "fold": fold_index,
                            "rank": rank,
                            "initialization": "movement_difference_pca",
                            "seed": ANALYSIS_SEED + 100_000 * contrast_index + 10_000 * fold_index + 100 * rank + 2,
                        },
                        {
                            "contrast": contrast.key,
                            "fold": fold_index,
                            "rank": rank,
                            "initialization": "random_orthonormal_2",
                            "seed": ANALYSIS_SEED + 100_000 * contrast_index + 10_000 * fold_index + 100 * rank + 3,
                        },
                    ]
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_seed": ANALYSIS_SEED,
        "primary_objective": {
            "quantity": "complete mean-normalized RR100 spatial rate maps g(x,y)",
            "loss": "0.5 * (normalized weighted sufficiency SSE + normalized weighted necessity SSE)",
            "normalizer": "paired baseline-to-target map-effect SSE, computed on the same split",
            "unit_weight": "paired expected-spike weight = 0.5 * (w_A + w_B)",
            "robustness_weight": "equal unit weight",
            "spatial_positions": "all 51 x 51 exact tiled-readout positions",
            "ssi_is_fitting_target": False,
            "no_r2_clipping": True,
        },
        "interventions": {
            "projection": "P = U U^T, U = reduced QR(A), A has shape 128 x k",
            "shared_over": ["all spatial positions", "all frames", "all image-trajectory pairs"],
            "sufficiency": "h_A + P(h_B-h_A)",
            "necessity": "h_B - P(h_B-h_A)",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "maximum_steps": 2000,
            "validation_every_steps": 10,
            "early_stopping_patience_validation_evaluations": 20,
            "gradient_clip_norm": 1.0,
            "frame_batch_size": 8,
            "validation_frame_batch_size": 40,
            "minimum_validation_improvement": 1e-5,
            "deviation_from_suggested_patience": (
                "20 validation evaluations at 10-step intervals equals 200 optimization steps "
                "without improvement; chosen on training/validation compute grounds before fitting"
            ),
            "best_run_selection": "minimum validation map loss only",
            "test_metrics_used_for_selection": False,
        },
        "initializations": [
            "random_orthonormal_1",
            "movement_difference_pca_training_only",
            "random_orthonormal_2",
        ],
        "restart_seeds": restart_seeds,
        "rank_sweep": {
            "screening_fold": SCREENING_FOLD,
            "screening_ranks": [0, 1, 2, 4, 8, 16, 32, 128],
            "learned_screening_ranks": list(learned_ranks),
            "restarts_per_learned_rank": 3,
            "elbow_selection_data": "validation only",
            "elbow_method": (
                "maximum distance to the endpoint chord on log2(rank) versus the mean of "
                "sufficiency and necessity validation map recovery, averaged over contrasts"
            ),
            "crossed_validation_ranks": (
                "union of the two learned ranks adjacent to the validation-selected elbow and ranks 4, 8, 16"
            ),
            "ambiguous_elbow_rule": "carry all learned screening ranks when no unique maximum exists",
            "stop_rule": (
                "After Stage 1, stop if every contrast has <0.40 held-out recovery in either "
                "sufficiency or necessity at every rank <=16; report: No compact ConvGRU channel subspace was found."
            ),
        },
        "baselines": {
            "rank_zero": True,
            "identity_rank_128": True,
            "movement_difference_pca": "training samples only; same selected ranks",
            "readout_svd": "exact frozen target-population channel weights; same selected ranks",
            "random_haar": {"draws_per_selected_rank": 100, "dimension": N_CHANNELS},
            "shuffled_target": {
                "ranks": [4, 8, "largest_selected_rank"],
                "fits_per_rank": 20,
                "shuffle_unit": "training image-trajectory pair; preserve all frames together",
            },
        },
        "bootstrap": {
            "method": "crossed image x trajectory bootstrap with Cartesian product",
            "exploratory_samples": 2000,
            "final_samples": 4000,
            "seed": ANALYSIS_SEED + 800_000,
            "confidence_interval_percent": [2.5, 97.5],
            "paired_across_interventions": True,
            "leave_one_image_out": True,
            "independent_pair_bootstrap_forbidden": True,
        },
        "cross_scale": {
            "low_0_to_2": {"recipient_scale": 0.0, "donor_scales": [0.5, 1.0, 3.0]},
            "high_0_to_1": {"recipient_scale": 0.0, "donor_scales": [0.5, 2.0, 3.0]},
            "high_1_to_3": {"recipient_scale": 1.0, "donor_scales": [2.0]},
            "evaluation_only": True,
        },
        "cross_contrast": {
            "metrics": ["subspace_overlap_tr(P1P2)/min(k1,k2)", "principal_angles_deg", "causal_cross_transfer"],
            "joint_sharpening_subspace": "optional only after overlap and cross-transfer support it",
        },
        "subspace_stability": {
            "metric": "tr(P_i P_j) / k for equal ranks",
            "comparison": "same-rank Haar-random null",
            "interpretation_requires_stability": True,
        },
        "numerical_tolerances": {
            "rank_zero_state_max_abs": 0.0,
            "rank_128_state_max_abs": 1e-6,
            "preactivation_max_abs": 2e-5,
            "rate_max_abs": 2e-5,
            "normalized_map_max_abs": 1e-4,
            "ssi_max_abs_bits": 1e-6,
            "float16_storage_validation": {
                "required_before_use": True,
                "normalized_map_rmse_max": 1e-3,
                "ssi_max_abs_bits": 1e-4,
                "map_recovery_r2_min": 0.9999,
            },
        },
        "evaluation": {
            "primary": ["heldout_sufficiency_map_R2", "heldout_necessity_map_R2"],
            "secondary": [
                "exact_SSI_transfer_fraction",
                "pre_softplus_map_recovery",
                "mean_rate_effect",
                "all_100_per_unit_results",
            ],
            "ssi_transfer_fractions_capped": False,
            "report_population_splits": ["lower SF (71)", "higher SF (29)"],
            "per_unit_strata": ["historical SF split", "optimal movement scale", "higher-SF reversal status"],
        },
        "gpu_budget": {
            "hard_limit_hours": 4.0,
            "state_generation_expected": "one existing exact-decomposition pass",
            "screening_target_minutes": 60,
            "crossed_validation_target_hours": [1, 3],
            "action_at_limit": "stop and report completed results, bottleneck, evidence, and marginal value",
        },
    }


def _storage_estimates() -> dict[str, Any]:
    n_samples = N_IMAGES * N_TRAJECTORIES * N_SCALES * N_FRAMES
    state_elements = n_samples * N_CHANNELS * 64 * 64
    map_elements = n_samples * N_UNITS * 51 * 51

    def record(elements: int) -> dict[str, Any]:
        return {
            "elements": int(elements),
            "fp32_bytes": int(elements * 4),
            "fp32_gib": elements * 4 / 2**30,
            "fp16_bytes": int(elements * 2),
            "fp16_gib": elements * 2 / 2**30,
        }

    return {
        "n_scored_scale_frames": n_samples,
        "convgru_h": record(state_elements),
        "one_rr100_map_tensor": record(map_elements),
        "three_rr100_map_tensors_z_r_g": record(3 * map_elements),
        "h_plus_z_r_g": record(state_elements + 3 * map_elements),
        "one_metric_or_weight_tensor_Nx100": record(n_samples * N_UNITS),
        "storage_plan": (
            "stream chunked losslessly compressed fp32 h; cache exact maps only if disk permits; "
            "float16 requires the predeclared downstream validation gate"
        ),
        "hdf5_state_chunk": [1, 1, 1, 4, 128, 64, 64],
        "hdf5_map_chunk": [1, 1, 1, 4, 100, 51, 51],
    }


def build_analysis_manifest(
    folds_path: Path,
    optimization_path: Path,
) -> dict[str, Any]:
    images, trajectories, exact = _load_selection()
    phase_manifest = json.loads(PHASE_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = Path(phase_manifest["checkpoint"])
    contrast_payload = []
    for contrast in CONTRASTS:
        units = exact[f"{contrast.group}_unit_indices"]
        contrast_payload.append(
            {
                "key": contrast.key,
                "label": contrast.label,
                "target_population": contrast.group,
                "n_units": len(units),
                "unit_indices": units,
                "recipient_scale_A": contrast.scale_a,
                "donor_scale_B": contrast.scale_b,
                "delta_h": f"h({contrast.scale_b}x) - h({contrast.scale_a}x)",
                "sufficiency_state": "h_A + U U^T delta_h",
                "necessity_state": "h_B - U U^T delta_h",
            }
        )

    sources = {
        "exact_subset_summary": _source_record(SOURCE_EXACT),
        "selected_images": _source_record(SELECTED_IMAGES),
        "selected_trajectories": _source_record(SELECTED_TRAJECTORIES),
        "corrected_history_bank": _source_record(HISTORY_BANK),
        "corrected_history_manifest": _source_record(HISTORY_MANIFEST),
        "phase_spatial_run_manifest": _source_record(PHASE_MANIFEST),
        "ssi_mechanism_run_manifest": _source_record(SSI_MECHANISM_MANIFEST),
        "unit_feature_table": _source_record(UNIT_TABLE),
        "image_feature_table": _source_record(IMAGE_TABLE),
        "checkpoint": _source_record(checkpoint),
        "fold_assignments": _source_record(folds_path),
        "optimization_config": _source_record(optimization_path),
        "common_code": _source_record(Path(__file__).with_name("common.py")),
        "preparation_code": _source_record(Path(__file__)),
    }
    expected_checkpoint_hash = phase_manifest.get("checkpoint_sha256")
    if expected_checkpoint_hash and sources["checkpoint"]["sha256"] != expected_checkpoint_hash:
        raise RuntimeError("Checkpoint hash differs from the validated phase-spatial run")

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "causal_low_rank_decomposition_of_convgru_state",
        "objective": (
            "Test whether a small spatially shared linear subspace of the 128-channel ConvGRU "
            "causally reproduces complete movement-dependent RR100 normalized spatial maps."
        ),
        "configuration_only_no_model_forward": True,
        "frozen_model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sources["checkpoint"]["sha256"],
            "rr100_version": json.loads(SSI_MECHANISM_MANIFEST.read_text(encoding="utf-8"))["rr100_version"],
            "trained_parameters_modified": False,
        },
        "preserved_contract": {
            "corrected_true_history": True,
            "n_preceding_samples": 31,
            "n_scored_frames": N_FRAMES,
            "exact_spatially_tiled_readout": True,
            "output_nonlinearity": "softplus",
            "ssi": "mean_xy[g * log2(g + 1e-8)]",
            "expected_spike_weight": "mean spatial rate / 120 Hz",
            "movement_scales": SCALES.tolist(),
            "unit_split": {"low": 71, "high": 29},
            "full_100_image_x_1000_trajectory_bank_forbidden": True,
        },
        "dataset": {
            "dimensions": {
                "images": N_IMAGES,
                "trajectories": N_TRAJECTORIES,
                "scales": N_SCALES,
                "frames": N_FRAMES,
                "convgru_channels": N_CHANNELS,
                "convgru_spatial": [64, 64],
                "rr100_units": N_UNITS,
                "rr100_map_spatial": [51, 51],
            },
            "images": images,
            "trajectories": trajectories,
            "exact_archive_arrays": exact["array_metadata"],
        },
        "contrasts": contrast_payload,
        "cache_products": {
            "convgru_states": str(STATE_CACHE),
            "rr100_maps": str(MAP_CACHE),
            "readout_weights": str(READOUT_CACHE),
            "one_exact_core_replay_authorized_if_state_cache_missing": True,
            "current_state_cache_exists": STATE_CACHE.is_file(),
            "current_map_cache_exists": MAP_CACHE.is_file(),
            "replay_note": (
                "One resumable production cache-generation job. The first image-trajectory pair "
                "was used for mandatory numerical pilot retries before the remaining cache was continued; "
                "the full subset was not rerun. Failed/incomplete pairs were never marked valid."
            ),
        },
        "folding": {
            "fold_assignments": str(folds_path),
            "fold_assignments_sha256": sources["fold_assignments"]["sha256"],
            "screening_fold": SCREENING_FOLD,
            "reported_scientific_performance": "held-out crossed folds only",
        },
        "optimization": {
            "configuration": str(optimization_path),
            "configuration_sha256": sources["optimization_config"]["sha256"],
        },
        "storage_estimates": _storage_estimates(),
        "sources": sources,
        "required_pipeline_separation": [
            "model-state generation",
            "subspace optimization",
            "evaluation",
            "saved-product-only plotting",
        ],
    }


def _write_canonical_and_required(name: str, payload: dict[str, Any]) -> Path:
    canonical = CONFIG / name
    write_json(canonical, payload)
    # The top-level copy satisfies the final saved-product contract while the
    # canonical config/ copy is what data.py and compute stages consume.
    write_json(OUT / name, payload)
    return canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    args = parser.parse_args()
    if int(args.seed) != ANALYSIS_SEED:
        raise ValueError(f"The predeclared analysis seed is fixed at {ANALYSIS_SEED}")

    ensure_output_dirs()
    folds = build_fold_assignments(seed=int(args.seed))
    folds_path = _write_canonical_and_required("fold_assignments.json", folds)
    optimization = build_optimization_config()
    optimization_path = _write_canonical_and_required("optimization_config.json", optimization)
    manifest = build_analysis_manifest(folds_path, optimization_path)
    manifest_path = _write_canonical_and_required("analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "analysis_manifest": str(manifest_path),
                "fold_assignments": str(folds_path),
                "optimization_config": str(optimization_path),
                "model_forward_runs": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
