#!/usr/bin/env python3
"""Verify the complete true-history primary checkpoint without the held control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR, sha256_file, write_json


PREVIEW_DIR = OUT_DIR / "preliminary_true_only"
MERGED_DIR = OUT_DIR / "core_ssi/real_trace_true_history_v1/merged"
HELD_DIR = OUT_DIR / "core_ssi/real_trace_held_initial_history_v1"


def main() -> int:
    checks: dict[str, object] = {}
    stats = json.loads((PREVIEW_DIR / "statistics.json").read_text(encoding="utf-8"))
    ids = np.asarray(stats["image_ids"], dtype=int)
    checks["complete_primary_identity"] = {
        "passed": (
            stats["status"] == "COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING"
            and stats["provisional_signature"] == "SURVIVES-LIKE"
            and np.array_equal(ids, np.arange(100))
            and int(stats["n_trajectories"]) == 1000
            and int(stats["n_units"]) == 100
            and int(stats["n_low_sf"]) == 71
            and int(stats["n_high_sf"]) == 29
        ),
        "status": stats["status"],
        "signature": stats["provisional_signature"],
        "n_images": int(len(ids)),
    }

    manifest = json.loads((MERGED_DIR / "manifest.json").read_text(encoding="utf-8"))
    expected_shapes = {
        "ssi_matrix.npy": (100, 1000, 100),
        "expected_spikes_matrix.npy": (100, 1000, 100),
        "mean_rate_matrix.npy": (100, 1000, 100),
        "population_ssi.npy": (100, 1000),
    }
    arrays = {}
    for name, expected_shape in expected_shapes.items():
        path = MERGED_DIR / name
        array = np.load(path, mmap_mode="r")
        recorded = manifest["outputs"][name]
        arrays[name] = {
            "shape": list(array.shape),
            "expected_shape": list(expected_shape),
            "all_finite": bool(np.isfinite(array).all()),
            "hash_matches_manifest": sha256_file(path) == recorded["sha256"],
        }
    checks["merged_arrays"] = {
        "passed": all(
            item["shape"] == item["expected_shape"]
            and item["all_finite"]
            and item["hash_matches_manifest"]
            for item in arrays.values()
        ),
        "arrays": arrays,
    }

    curves = pd.read_csv(PREVIEW_DIR / "preliminary_true_only_curves.csv")
    summary = pd.read_csv(PREVIEW_DIR / "preliminary_true_only_summary.csv")
    effects = pd.read_csv(PREVIEW_DIR / "preliminary_true_only_unit_effects.csv")
    checks["tidy_outputs"] = {
        "passed": len(curves) == 78 and len(summary) == 12 and len(effects) == 100,
        "curve_rows": int(len(curves)),
        "summary_rows": int(len(summary)),
        "unit_rows": int(len(effects)),
    }

    required = [
        PREVIEW_DIR / "TRUE_HISTORY_PRIMARY_REPORT.md",
        PREVIEW_DIR / "statistics.json",
        PREVIEW_DIR / "preliminary_true_only_outputs.npz",
    ] + [
        PREVIEW_DIR / f"fig_true_history_primary.{suffix}" for suffix in ("png", "pdf", "svg")
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size < 256]
    checks["required_checkpoint_artifacts"] = {"passed": not missing, "missing": missing}

    held_files = [str(path) for path in HELD_DIR.rglob("*") if path.is_file()] if HELD_DIR.exists() else []
    checks["held_control_deferred"] = {"passed": not held_files, "files": held_files}

    passed = all(bool(item["passed"]) for item in checks.values())
    payload = {"passed": passed, "checks": checks}
    write_json(PREVIEW_DIR / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
