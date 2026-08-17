#!/usr/bin/env python3
"""Reconstruct the exact RR100 medoid transform from readable audit artifacts.

The original population-spec archives in Declan's checkout are mode 0600.  The
RR100 QC output is readable and records, for every construction case, each
multi-channel representative, its selected canonical channel, and all cluster
members.  The readable pre-compression label array records the 643 retained
canonical channels.  Post-hoc compression only merged representatives, so the
retained channels absent from the final multi-channel groups are exactly the 42
singletons.  This script checks those invariants before writing a logical
equivalent of the original one-hot medoid population transform.

The generated NPZ is not expected to be byte-identical to the historical ZIP
container.  Its scientific payload is the same: RR100 unit ``uNNN`` reads the
recorded ``selected_channel`` from the canonical 756-channel rate map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_QC_CSV = Path(
    "/home/declan/VisionCore/outputs/redundancy_resolved_v1_twin/"
    f"rr100_movie_medoid_qc_{RR100_VERSION}/rr100_movie_medoid_group_quality.csv"
)
DEFAULT_BASE_LABELS = Path(
    "/home/declan/VisionCore/outputs/redundancy_resolved_v1_twin/"
    "step1_activation_fingerprints/"
    "multistim_final_labels_"
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75.npy"
)
DEFAULT_OUT_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
N_CANONICAL_CHANNELS = 756
N_RR100_UNITS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_members(value: object) -> list[int]:
    return [int(part) for part in str(value).split(",") if str(part).strip()]


def stable_group_representatives(qc: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    required = {"case", "rep_idx", "group_label", "selected_channel", "members"}
    missing = sorted(required.difference(qc.columns))
    if missing:
        raise ValueError(f"QC table is missing columns: {missing}")

    cases = sorted(qc["case"].astype(str).unique().tolist())
    if len(cases) < 2:
        raise ValueError("Expected repeated RR100 metadata from at least two QC cases.")

    reps: list[dict[str, Any]] = []
    observed_indices = sorted(pd.to_numeric(qc["rep_idx"], errors="raise").astype(int).unique().tolist())
    expected_indices = list(range(len(observed_indices)))
    if observed_indices != expected_indices:
        raise ValueError(f"Expected contiguous group representative indices, got {observed_indices}")

    for rep_idx in expected_indices:
        rows = qc[pd.to_numeric(qc["rep_idx"], errors="raise").astype(int).eq(rep_idx)].copy()
        signatures = {
            (
                int(row.group_label),
                int(row.selected_channel),
                tuple(parse_members(row.members)),
            )
            for row in rows.itertuples(index=False)
        }
        if len(signatures) != 1:
            raise ValueError(f"Representative {rep_idx} disagrees across QC cases: {sorted(signatures)}")
        group_id, selected_channel, members_tuple = next(iter(signatures))
        members = list(members_tuple)
        if selected_channel not in members:
            raise ValueError(f"Representative {rep_idx} selected channel is not a cluster member.")
        reps.append(
            {
                "rep_idx": rep_idx,
                "group_id": group_id,
                "rep_channel": selected_channel,
                "selected_channel": selected_channel,
                "pooling_mode": "medoid",
                "members": members,
            }
        )
    return reps, cases


def append_singleton_representatives(
    group_representatives: list[dict[str, Any]],
    base_labels: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(base_labels, dtype=np.int32)
    if labels.shape != (N_CANONICAL_CHANNELS,):
        raise ValueError(f"Expected {N_CANONICAL_CHANNELS} base labels, got {labels.shape}")
    retained = set(map(int, np.flatnonzero(labels != -2)))
    grouped = {
        int(channel)
        for rep in group_representatives
        for channel in rep["members"]
    }
    if not grouped.issubset(retained):
        raise ValueError(f"Final groups contain channels excluded by the base labels: {sorted(grouped - retained)}")
    singletons = sorted(retained - grouped)
    representatives = list(group_representatives)
    for channel in singletons:
        rep_idx = len(representatives)
        representatives.append(
            {
                "rep_idx": rep_idx,
                "group_id": -1,
                "rep_channel": int(channel),
                "selected_channel": int(channel),
                "pooling_mode": "singleton",
                "members": [int(channel)],
            }
        )
    if len(representatives) != N_RR100_UNITS:
        raise ValueError(
            f"Expected 58 groups + 42 singletons = {N_RR100_UNITS} representatives, "
            f"got {len(group_representatives)} groups + {len(singletons)} singletons"
        )
    return representatives


def build_arrays(representatives: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    membership = np.zeros((N_RR100_UNITS, N_CANONICAL_CHANNELS), dtype=np.float32)
    cluster_membership = np.zeros_like(membership)
    labels = np.full(N_CANONICAL_CHANNELS, -2, dtype=np.int32)
    represented = np.zeros(N_CANONICAL_CHANNELS, dtype=bool)

    for rep in representatives:
        rep_idx = int(rep["rep_idx"])
        selected = int(rep["selected_channel"])
        members = np.asarray(rep["members"], dtype=np.int64)
        if np.any(members < 0) or np.any(members >= N_CANONICAL_CHANNELS):
            raise ValueError(f"Representative {rep_idx} contains an out-of-range channel.")
        if np.any(represented[members]):
            overlap = members[represented[members]].tolist()
            raise ValueError(f"Representative {rep_idx} overlaps earlier clusters at {overlap}")
        represented[members] = True
        group_id = int(rep["group_id"])
        labels[members] = group_id if group_id >= 0 else -1
        membership[rep_idx, selected] = 1.0
        cluster_membership[rep_idx, members] = 1.0 / float(members.size)

    if not np.allclose(membership.sum(axis=1), 1.0):
        raise ValueError("Each medoid row must select exactly one canonical channel.")
    return membership, labels, cluster_membership


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-csv", type=Path, default=DEFAULT_QC_CSV)
    parser.add_argument("--base-labels", type=Path, default=DEFAULT_BASE_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qc_csv = Path(args.qc_csv)
    out_dir = Path(args.out_dir)
    stem = f"population_spec_{RR100_VERSION}"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    if (npz_path.exists() or json_path.exists()) and not bool(args.force):
        raise FileExistsError(f"Refusing to overwrite {stem}; pass --force.")

    qc = pd.read_csv(qc_csv)
    group_representatives, cases = stable_group_representatives(qc)
    base_labels = np.load(Path(args.base_labels))
    representatives = append_singleton_representatives(group_representatives, base_labels)
    membership, labels, cluster_membership = build_arrays(representatives)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        membership=membership,
        labels=labels,
        cluster_membership=cluster_membership,
    )
    payload = {
        "version": RR100_VERSION,
        "pooling_mode": "medoid",
        "n_input_channels": N_CANONICAL_CHANNELS,
        "n_representatives": N_RR100_UNITS,
        "n_represented_channels": int(np.count_nonzero(labels != -2)),
        "representatives": representatives,
        "reconstruction": {
            "logical_equivalence": (
                "membership is one-hot at the QC-recorded selected canonical channel for every RR100 unit"
            ),
            "historical_container_byte_identity": False,
            "source_qc_csv": str(qc_csv.resolve()),
            "source_qc_csv_sha256": sha256_file(qc_csv),
            "source_base_labels": str(Path(args.base_labels).resolve()),
            "source_base_labels_sha256": sha256_file(Path(args.base_labels)),
            "agreement_cases": cases,
            "n_agreement_cases": len(cases),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {npz_path} ({sha256_file(npz_path)})")
    print(f"wrote {json_path} ({sha256_file(json_path)})")
    print(
        f"verified {len(group_representatives)} groups across {len(cases)} cases and "
        f"recovered {len(representatives) - len(group_representatives)} singletons; "
        f"represented channels={np.count_nonzero(labels != -2)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
