#!/usr/bin/env python3
"""Score native-240 Hz checkpoints after exact adjacent-bin aggregation.

Training remains at 240 Hz so the model sees every stimulus frame and no
temporal anti-aliasing assumption is hidden in the input pipeline.  For a
like-for-like comparison with the Figure-3 120 Hz twin, this evaluator sums
two adjacent predicted and observed spike-count bins before computing BPS.
Only complete pairs from the same trial and held-out split are included.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def adjacent_pair_positions(dataset, constituent_idx: int) -> torch.Tensor:
    """Return dataset positions shaped ``[n_pairs, 2]`` for raw bins 2k,2k+1."""
    positions = torch.nonzero(
        dataset.inds[:, 0] == constituent_idx, as_tuple=False
    ).flatten()
    raw_indices = dataset.inds[positions, 1].long()
    if raw_indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)

    lookup = torch.full(
        (int(raw_indices.max()) + 2,), -1, dtype=torch.long, device=raw_indices.device
    )
    lookup[raw_indices] = positions
    starts = raw_indices[raw_indices.remainder(2) == 0]
    starts = starts[(starts + 1 < lookup.numel()) & (lookup[starts + 1] >= 0)]

    raw_dataset = dataset.dsets[constituent_idx]
    if "trial_inds" in raw_dataset:
        same_trial = raw_dataset["trial_inds"][starts] == raw_dataset["trial_inds"][
            starts + 1
        ]
        starts = starts[same_trial]
    return torch.stack((lookup[starts], lookup[starts + 1]), dim=1)


def paired_native_filter(data_filter: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Require both native bins to be valid for a summed 120-Hz target."""
    return data_filter.float().reshape(int(n_pairs), 2, -1).amin(dim=1)


def _predict_pairs(model, dataset, dataset_idx, pair_positions, device, batch_pairs):
    predictions = []
    observations = []
    filters = []
    model.eval()
    with torch.no_grad():
        for start in range(0, pair_positions.shape[0], batch_pairs):
            pair = pair_positions[start : start + batch_pairs]
            flat_positions = pair.flatten()
            batch = dataset[flat_positions]
            stimulus = batch["stim"].to(device)
            behavior = batch.get("behavior")
            if behavior is not None:
                behavior = behavior.to(device)
            history = batch.get("history")
            if history is not None:
                history = history.to(device)
            output_behavior = batch.get("output_behavior")
            if output_behavior is not None:
                output_behavior = output_behavior.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    stimulus,
                    dataset_idx,
                    behavior,
                    history,
                    output_behavior,
                )
            n_pairs = pair.shape[0]
            predictions.append(prediction.float().reshape(n_pairs, 2, -1).sum(dim=1).cpu())
            observations.append(
                batch["robs"].float().reshape(n_pairs, 2, -1).sum(dim=1).cpu()
            )
            filters.append(paired_native_filter(batch["dfs"], n_pairs).cpu())
    return (
        torch.cat(predictions),
        torch.cat(observations),
        torch.cat(filters),
    )


def score_checkpoint(
    checkpoint: Path,
    device: torch.device,
    max_datasets: int,
    split: str,
    batch_pairs: int,
    sessions: list[str] | None = None,
):
    from eval.eval_stack_utils import bits_per_spike
    from eval.load_twin import load_twin
    from paper.model_selection.evaluate import overall_bps
    from training.pl_modules import MultiDatasetDM

    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dir = (raw_checkpoint.get("hyper_parameters", {}) or {}).get("cfg_dir")
    if cfg_dir is None:
        raise ValueError("Checkpoint does not record cfg_dir")
    model, model_info = load_twin(checkpoint, device=str(device), verbose=False)
    dm = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=max_datasets,
        batch=64,
        workers=0,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        dataset_names=sessions,
    )
    dm.setup("fit")
    datasets = dm.val_dsets if split == "val" else dm.test_dsets
    model_name_to_index = {name: index for index, name in enumerate(model.names)}

    per_dataset = {}
    pair_counts = {}
    for ordinal, name in enumerate(dm.names):
        if name not in model_name_to_index:
            raise RuntimeError(f"Data session {name} is absent from the checkpoint")
        dataset_idx = model_name_to_index[name]
        dataset = datasets[name]
        # MultiDatasetDM wraps the embedded dataset in an on-demand dtype
        # view.  Pair geometry belongs to the base; sample retrieval must go
        # through the view so uint8 stimuli receive the standard float cast.
        base_dataset = getattr(dataset, "base", dataset)
        session_predictions = []
        session_observations = []
        session_filters = []
        for constituent_idx in range(base_dataset.n_dsets):
            pairs = adjacent_pair_positions(base_dataset, constituent_idx)
            if pairs.numel() == 0:
                continue
            prediction, observation, data_filter = _predict_pairs(
                model,
                dataset,
                dataset_idx,
                pairs,
                device,
                batch_pairs,
            )
            session_predictions.append(prediction)
            session_observations.append(observation)
            session_filters.append(data_filter)

        prediction = torch.cat(session_predictions)
        observation = torch.cat(session_observations)
        data_filter = torch.cat(session_filters)
        valid = (
            torch.isfinite(prediction)
            & torch.isfinite(observation)
            & torch.isfinite(data_filter)
            & (data_filter > 0)
        )
        bps = bits_per_spike(
            torch.where(valid, prediction, 0.0),
            torch.where(valid, observation, 0.0),
            valid.float(),
        ).numpy()
        per_dataset[name] = bps
        pair_counts[name] = int(prediction.shape[0])
        finite = bps[np.isfinite(bps)]
        mean = float(np.clip(finite, 0.0, None).mean()) if finite.size else float("nan")
        print(f"{ordinal + 1:02d}/{len(dm.names):02d} {name}: {mean:.4f} BPS ({prediction.shape[0]:,} pairs)")

    overall, per_dataset_mean = overall_bps(per_dataset)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(raw_checkpoint.get("epoch", -1)),
        "split": split,
        "native_rate_hz": 240,
        "score_rate_hz": 120,
        "aggregation": "sum adjacent predicted and observed count bins",
        "bps_overall": overall,
        "bps_by_session": per_dataset_mean,
        "pairs_by_session": pair_counts,
        "cids_by_session": model_info["cids_by_session"],
        "_bps_per_unit": per_dataset,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Score only this exact session name (repeatable)",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    report = score_checkpoint(
        args.checkpoint.resolve(),
        device,
        args.max_datasets,
        args.split,
        args.batch_pairs,
        args.session,
    )
    per_unit = report.pop("_bps_per_unit")
    cids_by_session = report.pop("cids_by_session")
    output = args.out or (
        ROOT
        / "outputs"
        / "dekel240_evaluation"
        / args.checkpoint.parent.name
        / f"epoch_{report['checkpoint_epoch']:03d}_{args.split}_rebinned120.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    from paper.model_selection.evaluate_dekel_split import write_per_unit_archive

    per_unit_path = output.with_name(f"{output.stem}_per_unit.npz")
    write_per_unit_archive(
        per_unit_path,
        per_unit,
        cids_by_session,
        list(report["bps_by_session"]),
    )
    report["per_unit_bps_npz"] = str(per_unit_path.resolve())
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
