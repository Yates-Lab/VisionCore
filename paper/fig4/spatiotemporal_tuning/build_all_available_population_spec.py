#!/usr/bin/env python3
"""Build a one-to-one population view for every available recorded unit.

This is an export/replay population, not a Figure-4 tuning-selected population.
It preserves the canonical ``(session, cid)`` identity and excludes only
canonical placeholder channels that have no readout row in the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-availability", type=Path, default=None)
    parser.add_argument("--unit-metrics", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--mcfarland-outputs", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_identity_view(
    availability: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required_availability = {"canonical_channel", "session", "cid", "available"}
    required_metrics = {
        "canonical_channel",
        "session",
        "cid",
        "model_readout_row",
        "available",
    }
    if missing := required_availability.difference(availability.columns):
        raise ValueError(f"canonical availability lacks columns: {sorted(missing)}")
    if missing := required_metrics.difference(metrics.columns):
        raise ValueError(f"unit metrics lack columns: {sorted(missing)}")

    availability = availability.sort_values("canonical_channel", kind="mergesort").reset_index(drop=True)
    channels = availability.canonical_channel.to_numpy(dtype=int)
    if not np.array_equal(channels, np.arange(len(availability), dtype=int)):
        raise ValueError("canonical channels must be unique, contiguous, and zero based")
    available = availability.loc[availability.available.astype(bool)].copy()
    metrics = metrics.loc[metrics.available.astype(bool)].copy()
    if metrics.canonical_channel.duplicated().any():
        raise ValueError("unit metrics contain duplicate canonical channels")
    if metrics[["session", "cid"]].duplicated().any():
        raise ValueError("unit metrics contain duplicate (session, cid) identities")

    joined = available.merge(
        metrics,
        on=["canonical_channel", "session", "cid"],
        how="left",
        validate="one_to_one",
        suffixes=("_availability", ""),
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        missing = joined.loc[joined._merge.ne("both"), ["canonical_channel", "session", "cid"]]
        raise ValueError(f"available canonical units lack metric rows:\n{missing.to_string(index=False)}")
    joined = joined.drop(columns="_merge").sort_values("canonical_channel", kind="mergesort").reset_index(drop=True)
    if joined.model_readout_row.isna().any():
        raise ValueError("an available canonical unit lacks a model readout row")
    joined.insert(0, "unit_index", np.arange(len(joined), dtype=int))
    joined.insert(1, "unit_label", [f"u{index:03d}" for index in range(len(joined))])

    n_channels = int(len(availability))
    selected_channels = joined.canonical_channel.to_numpy(dtype=int)
    membership = np.zeros((len(joined), n_channels), dtype=np.float32)
    membership[np.arange(len(joined)), selected_channels] = 1.0
    cluster_membership = membership.copy()
    labels = np.full(n_channels, -1, dtype=np.int32)
    labels[selected_channels] = np.arange(len(joined), dtype=np.int32)
    return joined, membership, cluster_membership, labels


def runtime_identity_view(
    *,
    checkpoint: Path,
    dataset_config: Path,
    mcfarland_outputs: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve availability from the exact readout construction used in replay."""
    from paper.fig4.upstream.real_trace_matrix.model import (
        load_mcfarland_outputs,
        load_pinned_multidataset_model,
        load_spatial_readout,
    )

    model, _ = load_pinned_multidataset_model(
        checkpoint_path=checkpoint,
        dataset_configs=dataset_config,
        device=str(device),
    )
    outputs, _ = load_mcfarland_outputs(mcfarland_outputs)
    _, runtime_rows = load_spatial_readout(model, outputs, device=str(device))
    canonical = pd.DataFrame(runtime_rows).rename(
        columns={"channel": "canonical_channel", "source_cid": "cid"}
    )
    required = {
        "canonical_channel",
        "session",
        "cid",
        "model_readout_row",
        "available",
    }
    if missing := required.difference(canonical.columns):
        raise ValueError(f"runtime readout rows lack columns: {sorted(missing)}")
    canonical = canonical.sort_values("canonical_channel", kind="mergesort").reset_index(drop=True)
    channels = canonical.canonical_channel.to_numpy(dtype=int)
    if not np.array_equal(channels, np.arange(len(canonical), dtype=int)):
        raise ValueError("runtime canonical channels are not contiguous and zero based")
    units = canonical.loc[canonical.available.astype(bool)].copy().reset_index(drop=True)
    if units.model_readout_row.isna().any():
        raise ValueError("runtime marked an available unit without a model readout row")
    if units[["session", "cid"]].duplicated().any():
        raise ValueError("runtime readout contains duplicate (session, cid) identities")
    units.insert(0, "unit_index", np.arange(len(units), dtype=int))
    units.insert(1, "unit_label", [f"u{index:03d}" for index in range(len(units))])
    n_channels = int(len(canonical))
    selected_channels = units.canonical_channel.to_numpy(dtype=int)
    membership = np.zeros((len(units), n_channels), dtype=np.float32)
    membership[np.arange(len(units)), selected_channels] = 1.0
    cluster_membership = membership.copy()
    labels = np.full(n_channels, -1, dtype=np.int32)
    labels[selected_channels] = np.arange(len(units), dtype=np.int32)
    return canonical, units, membership, cluster_membership, labels


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"population_spec_{args.version}"
    outputs = (
        out_dir / f"{stem}.npz",
        out_dir / f"{stem}.json",
        out_dir / "all_available_units.csv",
        out_dir / "runtime_canonical_availability.csv",
        out_dir / "summary.json",
    )
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("population-spec outputs already exist; pass --force to replace them")

    runtime_source = args.dataset_config is not None or args.mcfarland_outputs is not None
    if runtime_source:
        if args.dataset_config is None or args.mcfarland_outputs is None:
            raise ValueError("runtime resolution requires both --dataset-config and --mcfarland-outputs")
        dataset_config = args.dataset_config.resolve()
        mcfarland_outputs = args.mcfarland_outputs.resolve()
        canonical, units, membership, cluster_membership, labels = runtime_identity_view(
            checkpoint=checkpoint,
            dataset_config=dataset_config,
            mcfarland_outputs=mcfarland_outputs,
            device=str(args.device),
        )
        source_payload = {
            "availability_source": "checkpoint_runtime_spatial_readout",
            "dataset_config": str(dataset_config),
            "dataset_config_sha256": sha256_file(dataset_config),
            "mcfarland_outputs": str(mcfarland_outputs),
            "mcfarland_outputs_sha256": sha256_file(mcfarland_outputs),
        }
    else:
        if args.canonical_availability is None or args.unit_metrics is None:
            raise ValueError(
                "provide runtime --dataset-config/--mcfarland-outputs or both saved table inputs"
            )
        availability_path = args.canonical_availability.resolve()
        metrics_path = args.unit_metrics.resolve()
        availability = pd.read_csv(availability_path)
        metrics = pd.read_csv(metrics_path)
        units, membership, cluster_membership, labels = build_identity_view(availability, metrics)
        canonical = availability.copy()
        source_payload = {
            "availability_source": "saved_tables",
            "canonical_availability": str(availability_path),
            "canonical_availability_sha256": sha256_file(availability_path),
            "unit_metrics": str(metrics_path),
            "unit_metrics_sha256": sha256_file(metrics_path),
        }
    checkpoint_sha256 = sha256_file(checkpoint)
    representatives = [
        {
            "rep_idx": int(row.unit_index),
            "selected_channel": int(row.canonical_channel),
            "rep_channel": int(row.canonical_channel),
            "members": [int(row.canonical_channel)],
            "n_members": 1,
            "pooling_mode": "exact_identity",
            "session": str(row.session),
            "cid": int(row.cid),
            "model_readout_row": int(row.model_readout_row),
        }
        for row in units.itertuples(index=False)
    ]
    np.savez_compressed(
        outputs[0],
        membership=membership,
        cluster_membership=cluster_membership,
        labels=labels,
    )
    payload = {
        "version": str(args.version),
        "analysis": "one-to-one all-available-recorded-unit export population view",
        "pooling_mode": "exact_identity",
        "n_representatives": int(len(units)),
        "n_input_channels": int(membership.shape[1]),
        "n_unavailable_placeholders": int(membership.shape[1] - membership.shape[0]),
        "identity_key": "(session, cid)",
        "selection_gate": "checkpoint_available_recorded_unit; no tuning-quality gate",
        "rr_clustering": False,
        "membership_contract": (
            "one nonzero value of exactly 1.0 per row at canonical_channel; "
            "no pooling, clustering, substitution, or unavailable-channel replacement"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        **source_payload,
        "representatives": representatives,
    }
    outputs[1].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    units.to_csv(outputs[2], index=False)
    canonical.to_csv(outputs[3], index=False)
    summary = {
        **{key: value for key, value in payload.items() if key != "representatives"},
        "population_spec_npz": str(outputs[0]),
        "population_spec_npz_sha256": sha256_file(outputs[0]),
        "population_spec_json": str(outputs[1]),
        "population_spec_json_sha256": sha256_file(outputs[1]),
        "all_available_units": str(outputs[2]),
        "all_available_units_sha256": sha256_file(outputs[2]),
        "runtime_canonical_availability": str(outputs[3]),
        "runtime_canonical_availability_sha256": sha256_file(outputs[3]),
        "gates": {
            "all_available_units_included_once": bool(
                len(units) == int(canonical.available.astype(bool).sum())
            ),
            "one_hot_identity_membership": bool(
                np.all((membership != 0).sum(axis=1) == 1)
                and np.all(membership[membership != 0] == 1.0)
                and len(np.unique(np.argmax(membership, axis=1))) == len(units)
            ),
            "no_pooling_or_substitution": True,
            "checkpoint_digest_bound": True,
        },
    }
    outputs[4].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[all-unit-population] wrote {len(units)} exact-identity units "
        f"from {membership.shape[1]} canonical slots to {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
