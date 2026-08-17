#!/usr/bin/env python3
"""Merge completed corrected-history core SSI shards without touching legacy data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import CORE_DIR, sha256_file, write_json
from paper.fig4.mechanism_audit_v1.correction.run_corrected_core import BANK_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=sorted(BANK_KEYS), required=True)
    parser.add_argument("--shards", nargs="+", default=["images_000_050", "images_050_100"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bank_name, _ = BANK_KEYS[args.bank]
    bank_dir = CORE_DIR / bank_name
    shard_dirs = [bank_dir / name for name in args.shards]
    manifests = []
    for shard in shard_dirs:
        import json

        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest["completed_images"]) != int(manifest["n_images"]):
            raise RuntimeError(f"Incomplete shard: {shard}")
        manifests.append(manifest)
    ranges = [(int(m["image_start"]), int(m["image_stop"])) for m in manifests]
    if ranges != [(0, 50), (50, 100)]:
        raise ValueError(f"Expected exact 0:50 and 50:100 coverage, got {ranges}")
    merged_dir = bank_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for filename in ("ssi_matrix.npy", "expected_spikes_matrix.npy", "mean_rate_matrix.npy", "population_ssi.npy"):
        arrays = [np.load(shard / filename) for shard in shard_dirs]
        merged = np.concatenate(arrays, axis=0)
        path = merged_dir / filename
        np.save(path, merged)
        outputs[filename] = {"path": path, "shape": list(merged.shape), "sha256": sha256_file(path)}
    write_json(
        merged_dir / "manifest.json",
        {
            "analysis": "corrected_history_core_ssi_merged",
            "bank": bank_name,
            "legacy_name": "legacy_wrapped_prefix",
            "shards": shard_dirs,
            "outputs": outputs,
        },
    )
    print(f"Merged {bank_name} to {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
