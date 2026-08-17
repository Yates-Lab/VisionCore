#!/usr/bin/env python3
"""Compare native-240 and 120-Hz twins on exactly the same validation bins.

The ordinary validation loaders have slightly different support because the
native model requires 60 native frames while the historical 120-Hz model
requires 33 downsampled frames.  Comparing their independently reduced BPS
therefore mixes a model difference with a support difference.  This evaluator
maps both loaders back to the same source-data pair index, intersects their
support, verifies that the spike counts agree exactly, and only then scores the
two models with one shared per-unit validity mask.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection.evaluate_rebinned_bps import (
    adjacent_pair_positions,
    paired_native_filter as _paired_native_filter,
)


BPS_INDEPENDENT_ATOL = 1e-10


def source_pair_indices(dataset, constituent_idx: int, pair_positions: torch.Tensor) -> torch.Tensor:
    """Return the 120-Hz source-pair index for native pair positions."""
    if pair_positions.ndim != 2 or pair_positions.shape[1] != 2:
        raise ValueError("pair_positions must have shape [n_pairs, 2]")
    raw = dataset.inds[pair_positions[:, 0], 1].long()
    if torch.any(raw.remainder(2) != 0):
        raise RuntimeError("native pair geometry does not begin on even source bins")
    return raw // 2


def constituent_positions(dataset, constituent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return combined-dataset positions and their constituent raw indices."""
    positions = torch.nonzero(
        dataset.inds[:, 0] == constituent_idx, as_tuple=False
    ).flatten()
    raw_indices = dataset.inds[positions, 1].long()
    if raw_indices.numel() and torch.any(raw_indices[1:] <= raw_indices[:-1]):
        raise RuntimeError("constituent indices must be strictly increasing")
    return positions, raw_indices


def intersect_source_support(
    candidate_source_indices: np.ndarray,
    reference_source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared source indices and aligned rows in both arrays."""
    candidate = np.asarray(candidate_source_indices, dtype=np.int64)
    reference = np.asarray(reference_source_indices, dtype=np.int64)
    if candidate.ndim != 1 or reference.ndim != 1:
        raise ValueError("source indices must be one-dimensional")
    if candidate.size and np.any(np.diff(candidate) <= 0):
        raise ValueError("candidate source indices must be strictly increasing")
    if reference.size and np.any(np.diff(reference) <= 0):
        raise ValueError("reference source indices must be strictly increasing")
    shared, candidate_rows, reference_rows = np.intersect1d(
        candidate, reference, assume_unique=True, return_indices=True
    )
    return shared, candidate_rows, reference_rows


def common_support_bps(
    candidate_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    observation: np.ndarray,
    candidate_filter: np.ndarray,
    reference_filter: np.ndarray,
    *,
    return_audit: bool = False,
):
    """Score both models on one model-independent, data-only mask."""
    from eval.eval_stack_utils import bits_per_spike

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            candidate_prediction,
            reference_prediction,
            observation,
            candidate_filter,
            reference_filter,
        )
    ]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("all score arrays must share shape [sample, unit]")
    candidate_prediction, reference_prediction, observation, candidate_filter, reference_filter = arrays
    valid = (
        np.isfinite(observation)
        & np.isfinite(candidate_filter)
        & np.isfinite(reference_filter)
        & (candidate_filter > 0)
        & (reference_filter > 0)
    )
    candidate_missing = valid & ~np.isfinite(candidate_prediction)
    reference_missing = valid & ~np.isfinite(reference_prediction)
    if candidate_missing.any() or reference_missing.any():
        raise RuntimeError(
            "Prediction is missing on shared data-only validation support: "
            f"candidate={int(candidate_missing.sum())}, "
            f"reference={int(reference_missing.sum())}"
        )
    obs = torch.from_numpy(np.where(valid, observation, 0.0))
    dfs = torch.from_numpy(valid.astype(np.float64))
    candidate = bits_per_spike(
        torch.from_numpy(np.where(valid, candidate_prediction, 0.0)), obs, dfs
    ).numpy()
    reference = bits_per_spike(
        torch.from_numpy(np.where(valid, reference_prediction, 0.0)), obs, dfs
    ).numpy()
    candidate_independent = independent_bits_per_spike(
        candidate_prediction, observation, valid
    )
    reference_independent = independent_bits_per_spike(
        reference_prediction, observation, valid
    )
    candidate_error = np.abs(candidate - candidate_independent)
    reference_error = np.abs(reference - reference_independent)
    candidate_max_error = float(np.nanmax(candidate_error, initial=0.0))
    reference_max_error = float(np.nanmax(reference_error, initial=0.0))
    if not np.allclose(
        candidate,
        candidate_independent,
        rtol=0,
        atol=BPS_INDEPENDENT_ATOL,
        equal_nan=True,
    ) or not np.allclose(
        reference,
        reference_independent,
        rtol=0,
        atol=BPS_INDEPENDENT_ATOL,
        equal_nan=True,
    ):
        raise AssertionError(
            "production and independent BPS implementations disagree: "
            f"candidate max_abs={candidate_max_error:.17g}, "
            f"reference max_abs={reference_max_error:.17g}"
        )
    result = (candidate, reference, valid.sum(axis=0).astype(np.int64))
    if return_audit:
        return (*result, {
            "candidate_max_abs_error": candidate_max_error,
            "reference_max_abs_error": reference_max_error,
            "absolute_tolerance": BPS_INDEPENDENT_ATOL,
        })
    return result


def independent_bits_per_spike(prediction, observation, valid):
    """Independent float64 NumPy form of the Poisson information score."""
    prediction = np.asarray(prediction, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != observation.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, observation, and valid must share one shape")
    prediction = np.where(valid, prediction, 0.0)
    observation = np.where(valid, observation, 0.0)
    weights = valid.astype(np.float64)
    duration = np.maximum(weights.sum(axis=0), 1.0)
    spikes = np.maximum((weights * observation).sum(axis=0), 1.0)
    null_rate = spikes / duration
    model_ll = observation * np.log(prediction + 1e-8) - prediction
    null_ll = observation * np.log(null_rate[None, :] + 1e-8) - null_rate[None, :]
    return ((model_ll - null_ll) * weights).sum(axis=0) / spikes / np.log(2.0)


def score_support_digest(
    observation: np.ndarray,
    candidate_filter: np.ndarray,
    reference_filter: np.ndarray,
) -> tuple[str, str]:
    """Fingerprint exact shared support and the observations on that support."""
    observation = np.asarray(observation, dtype=np.float32)
    candidate_filter = np.asarray(candidate_filter)
    reference_filter = np.asarray(reference_filter)
    if not (
        observation.shape == candidate_filter.shape == reference_filter.shape
    ):
        raise ValueError("support digest arrays must have one shape")
    support = (
        np.isfinite(observation)
        & np.isfinite(candidate_filter)
        & np.isfinite(reference_filter)
        & (candidate_filter > 0)
        & (reference_filter > 0)
    )

    def digest(array: np.ndarray) -> str:
        value = np.ascontiguousarray(array)
        hasher = hashlib.sha256()
        hasher.update(str(value.shape).encode("ascii"))
        hasher.update(value.dtype.str.encode("ascii"))
        hasher.update(value.view(np.uint8))
        return hasher.hexdigest()

    support_hash = digest(support.astype(np.uint8))
    masked_observation = np.where(support, observation, 0.0).astype(np.float32)
    observation_hash = digest(masked_observation)
    return support_hash, observation_hash


def paired_native_filter(data_filter: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Require both native bins to be valid before scoring their summed count.

    A mean reduction would turn ``[0, 1]`` into ``0.5`` and the downstream
    positive-mask test would silently accept a pair containing one invalid
    native sample. The conservative logical-AND is the appropriate support
    for a two-bin count target.
    """
    return _paired_native_filter(data_filter, n_pairs)


def _positive_prediction(model, prediction: torch.Tensor) -> torch.Tensor:
    activation = getattr(getattr(model, "model", None), "activation", None)
    return prediction.exp() if isinstance(activation, nn.Identity) else prediction


def _predict_positions(model, dataset, dataset_idx, positions, device, batch_size):
    predictions, observations, filters = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            batch = dataset[batch_positions]
            stimulus = batch["stim"].to(device)
            behavior = batch.get("behavior")
            history = batch.get("history")
            output_behavior = batch.get("output_behavior")
            behavior = behavior.to(device) if behavior is not None else None
            history = history.to(device) if history is not None else None
            output_behavior = (
                output_behavior.to(device) if output_behavior is not None else None
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    stimulus, dataset_idx, behavior, history, output_behavior
                )
            predictions.append(_positive_prediction(model, prediction.float()).cpu())
            observations.append(batch["robs"].float().cpu())
            filters.append(batch["dfs"].float().cpu())
    return tuple(torch.cat(value).numpy() for value in (predictions, observations, filters))


def _predict_native_pairs(model, dataset, dataset_idx, pairs, device, batch_pairs):
    predictions, observations, filters = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_pairs):
            pair = pairs[start : start + batch_pairs]
            n_pairs = len(pair)
            batch = dataset[pair.flatten()]
            stimulus = batch["stim"].to(device)
            behavior = batch.get("behavior")
            history = batch.get("history")
            output_behavior = batch.get("output_behavior")
            behavior = behavior.to(device) if behavior is not None else None
            history = history.to(device) if history is not None else None
            output_behavior = (
                output_behavior.to(device) if output_behavior is not None else None
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    stimulus, dataset_idx, behavior, history, output_behavior
                )
            prediction = _positive_prediction(model, prediction.float())
            predictions.append(prediction.reshape(n_pairs, 2, -1).sum(dim=1).cpu())
            observations.append(
                batch["robs"].float().reshape(n_pairs, 2, -1).sum(dim=1).cpu()
            )
            filters.append(paired_native_filter(batch["dfs"], n_pairs).cpu())
    return tuple(torch.cat(value).numpy() for value in (predictions, observations, filters))


def _make_datamodule(cfg_dir, session, batch_size):
    from training.pl_modules import MultiDatasetDM

    dm = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=30,
        batch=batch_size,
        workers=0,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        dataset_names=[session],
    )
    dm.setup("fit")
    return dm


def _constituent_name(base_dataset, index: int) -> str:
    return str(base_dataset.dsets[index].metadata.get("name", index))


def _checkpoint_cfg(checkpoint: Path) -> tuple[dict, str]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dir = (raw.get("hyper_parameters", {}) or {}).get("cfg_dir")
    if cfg_dir is None:
        raise ValueError(f"{checkpoint} does not record cfg_dir")
    return raw, str(cfg_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_checkpoint", type=Path)
    parser.add_argument("reference_checkpoint", type=Path)
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--session", action="append")
    session_group.add_argument(
        "--all-common-sessions",
        action="store_true",
        help="Evaluate every checkpoint session shared by candidate and reference.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from eval.load_twin import load_twin
    from paper.model_selection.evaluate import overall_bps

    candidate_checkpoint = args.candidate_checkpoint.resolve()
    reference_checkpoint = args.reference_checkpoint.resolve()
    candidate_raw, candidate_cfg = _checkpoint_cfg(candidate_checkpoint)
    reference_raw, reference_cfg = _checkpoint_cfg(reference_checkpoint)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    candidate_model, candidate_info = load_twin(
        candidate_checkpoint, device=str(device), verbose=False
    )
    reference_model, reference_info = load_twin(
        reference_checkpoint, device=str(device), verbose=False
    )
    candidate_name_to_index = {
        name: index for index, name in enumerate(candidate_model.names)
    }
    reference_name_to_index = {
        name: index for index, name in enumerate(reference_model.names)
    }
    sessions = (
        [
            name
            for name in candidate_model.names
            if name in reference_name_to_index
        ]
        if args.all_common_sessions
        else list(args.session)
    )
    if not sessions:
        raise RuntimeError("Candidate and reference have no requested common sessions.")

    candidate_scores, reference_scores, valid_counts = {}, {}, {}
    shared_geometry_counts, candidate_geometry_counts, reference_geometry_counts = {}, {}, {}
    geometry_by_constituent = {}
    support_hashes, observation_hashes = {}, {}
    bps_implementation_errors = {}
    for session_ordinal, session in enumerate(sessions, start=1):
        if session not in candidate_name_to_index or session not in reference_name_to_index:
            raise RuntimeError(f"{session}: absent from one of the checkpoints")
        candidate_cids = np.asarray(candidate_info["cids_by_session"][session], dtype=np.int64)
        reference_cids = np.asarray(reference_info["cids_by_session"][session], dtype=np.int64)
        if not np.array_equal(candidate_cids, reference_cids):
            raise RuntimeError(f"{session}: candidate/reference cids differ")

        candidate_dm = _make_datamodule(candidate_cfg, session, args.batch_size)
        candidate_dataset = candidate_dm.val_dsets[session]
        candidate_base = getattr(candidate_dataset, "base", candidate_dataset)
        candidate_by_name = {}
        candidate_geometry = 0
        session_geometry_by_constituent = {}
        for constituent_idx in range(candidate_base.n_dsets):
            pairs = adjacent_pair_positions(candidate_base, constituent_idx)
            if not len(pairs):
                continue
            source_indices = source_pair_indices(
                candidate_base, constituent_idx, pairs
            ).cpu().numpy()
            prediction, observation, data_filter = _predict_native_pairs(
                candidate_model,
                candidate_dataset,
                candidate_name_to_index[session],
                pairs,
                device,
                args.batch_size,
            )
            name = _constituent_name(candidate_base, constituent_idx)
            candidate_by_name[name] = {
                "source_indices": source_indices,
                "prediction": prediction,
                "observation": observation,
                "data_filter": data_filter,
            }
            session_geometry_by_constituent[name] = {"candidate": int(len(pairs))}
            candidate_geometry += len(pairs)
        del candidate_dataset, candidate_base, candidate_dm
        gc.collect()

        reference_dm = _make_datamodule(reference_cfg, session, args.batch_size)
        reference_dataset = reference_dm.val_dsets[session]
        reference_base = getattr(reference_dataset, "base", reference_dataset)
        session_candidate, session_reference = [], []
        session_observation, session_candidate_filter, session_reference_filter = [], [], []
        shared_geometry = 0
        reference_geometry = 0
        for constituent_idx in range(reference_base.n_dsets):
            name = _constituent_name(reference_base, constituent_idx)
            if name not in candidate_by_name:
                raise RuntimeError(f"{session}: reference constituent {name!r} absent natively")
            positions, source_indices = constituent_positions(
                reference_base, constituent_idx
            )
            reference_geometry += len(positions)
            shared, candidate_rows, reference_rows = intersect_source_support(
                candidate_by_name[name]["source_indices"], source_indices.cpu().numpy()
            )
            if not len(shared):
                session_geometry_by_constituent.setdefault(name, {})[
                    "reference"
                ] = int(len(positions))
                session_geometry_by_constituent[name]["shared"] = 0
                continue
            selected_positions = positions[
                torch.as_tensor(reference_rows, dtype=torch.long)
            ]
            reference_prediction, reference_observation, reference_filter = _predict_positions(
                reference_model,
                reference_dataset,
                reference_name_to_index[session],
                selected_positions,
                device,
                args.batch_size,
            )
            candidate_entry = candidate_by_name[name]
            candidate_observation = candidate_entry["observation"][candidate_rows]
            if not np.array_equal(candidate_observation, reference_observation):
                maximum_error = float(
                    np.max(np.abs(candidate_observation - reference_observation))
                )
                raise RuntimeError(
                    f"{session}/{name}: aligned spike counts differ (max {maximum_error:g})"
                )
            session_candidate.append(candidate_entry["prediction"][candidate_rows])
            session_reference.append(reference_prediction)
            session_observation.append(reference_observation)
            session_candidate_filter.append(candidate_entry["data_filter"][candidate_rows])
            session_reference_filter.append(reference_filter)
            shared_geometry += len(shared)
            session_geometry_by_constituent.setdefault(name, {})[
                "reference"
            ] = int(len(positions))
            session_geometry_by_constituent[name]["shared"] = int(len(shared))
        del reference_dataset, reference_base, reference_dm, candidate_by_name
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        candidate_prediction = np.concatenate(session_candidate)
        reference_prediction = np.concatenate(session_reference)
        observation = np.concatenate(session_observation)
        candidate_filter = np.concatenate(session_candidate_filter)
        reference_filter = np.concatenate(session_reference_filter)
        support_hash, observation_hash = score_support_digest(
            observation, candidate_filter, reference_filter
        )
        candidate_bps, reference_bps, counts, bps_audit = common_support_bps(
            candidate_prediction,
            reference_prediction,
            observation,
            candidate_filter,
            reference_filter,
            return_audit=True,
        )
        candidate_scores[session] = candidate_bps
        reference_scores[session] = reference_bps
        valid_counts[session] = counts
        candidate_geometry_counts[session] = int(candidate_geometry)
        reference_geometry_counts[session] = int(reference_geometry)
        shared_geometry_counts[session] = int(shared_geometry)
        geometry_by_constituent[session] = session_geometry_by_constituent
        support_hashes[session] = support_hash
        observation_hashes[session] = observation_hash
        bps_implementation_errors[session] = bps_audit
        candidate_mean = float(np.nanmean(np.clip(candidate_bps, 0, None)))
        reference_mean = float(np.nanmean(np.clip(reference_bps, 0, None)))
        print(
            f"{session_ordinal:02d}/{len(sessions):02d} {session}: "
            f"candidate {candidate_mean:.4f}, reference {reference_mean:.4f}, "
            f"shared {shared_geometry:,}",
            flush=True,
        )

    candidate_overall, candidate_by_session = overall_bps(candidate_scores)
    reference_overall, reference_by_session = overall_bps(reference_scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive_path = args.out.with_name(f"{args.out.stem}_per_unit.npz")
    payload = {"session_names": np.asarray(sessions)}
    for index, session in enumerate(sessions):
        payload[f"cids_{index}"] = np.asarray(
            candidate_info["cids_by_session"][session], dtype=np.int64
        )
        payload[f"candidate_bps_{index}"] = candidate_scores[session]
        payload[f"reference_bps_{index}"] = reference_scores[session]
        payload[f"common_valid_count_{index}"] = valid_counts[session]
    np.savez_compressed(archive_path, **payload)
    report = {
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_epoch": int(candidate_raw.get("epoch", -1)),
        "reference_checkpoint": str(reference_checkpoint),
        "reference_epoch": int(reference_raw.get("epoch", -1)),
        "candidate_dataset_config": candidate_cfg,
        "reference_dataset_config": reference_cfg,
        "split": "val",
        "score_rate_hz": 120,
        "support": "exact intersection of source pair, trial split, and both per-unit data filters",
        "prediction_completeness": "required on every data-supported sample; never used to define the mask",
        "independent_bps_implementation_match": True,
        "independent_bps_absolute_tolerance": BPS_INDEPENDENT_ATOL,
        "independent_bps_max_abs_error_by_session": bps_implementation_errors,
        "observation_identity": "required exact before scoring",
        "support_sha256_by_session": support_hashes,
        "observation_on_support_sha256_by_session": observation_hashes,
        "candidate_bps_overall": candidate_overall,
        "reference_bps_overall": reference_overall,
        "candidate_bps_by_session": candidate_by_session,
        "reference_bps_by_session": reference_by_session,
        "candidate_geometry_by_session": candidate_geometry_counts,
        "reference_geometry_by_session": reference_geometry_counts,
        "shared_geometry_by_session": shared_geometry_counts,
        "shared_fraction_of_candidate_geometry_by_session": {
            session: shared_geometry_counts[session] / candidate_geometry_counts[session]
            for session in shared_geometry_counts
        },
        "shared_fraction_of_reference_geometry_by_session": {
            session: shared_geometry_counts[session] / reference_geometry_counts[session]
            for session in shared_geometry_counts
        },
        "geometry_by_session_and_constituent": geometry_by_constituent,
        "per_unit_npz": str(archive_path.resolve()),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
