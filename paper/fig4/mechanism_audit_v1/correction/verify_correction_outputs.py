#!/usr/bin/env python3
"""Verify every required Figure 4 correction-stage artifact without running the model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import OUT_DIR, sha256_file, write_json


CORE_BANKS = ("real_trace_true_history_v1", "real_trace_held_initial_history_v1")
FIGURE_STEMS = ("fig_history_validation", "fig_history_bug_explained", "fig_core_result_corrected")
REQUIRED_FILES = (
    "CORRECTED_HISTORY_README.md",
    "CAUSAL_INDEX_AUDIT.md",
    "CORE_CORRECTION_REPORT.md",
    "analysis_manifest.json",
    "statistics.json",
    "true_history_trajectory_table.csv",
    "held_prefix_trajectory_table.csv",
    "history_validation.csv",
    "legacy_vs_corrected_unit_effects.csv",
    "corrected_core_ssi_curves.csv",
    "corrected_core_ssi_summary.csv",
    "corrected_core_ssi_outputs.npz",
    "legacy_core_reproduction.csv",
    "plot_data/fig_core_result_corrected.csv",
    "plot_data/fig_history_bug_explained_trace.csv",
    "plot_data/fig_history_bug_explained_causal_windows.csv",
    "plot_data/fig_history_bug_explained_speed_distributions.csv",
    "plot_data/fig_history_bug_explained_example.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checks: dict[str, Any] = {}

    missing = [name for name in REQUIRED_FILES if not (OUT_DIR / name).is_file()]
    checks["required_files_present"] = {"passed": not missing, "missing": missing}

    figure_errors = []
    for stem in FIGURE_STEMS:
        for suffix in ("pdf", "svg", "png"):
            path = OUT_DIR / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size < 1024:
                figure_errors.append(str(path))
    checks["required_figure_triplets_present"] = {
        "passed": not figure_errors,
        "missing_or_too_small": figure_errors,
    }

    validation = pd.read_csv(OUT_DIR / "history_validation.csv")
    causal_columns = (
        "outputs_with_future_samples",
        "outputs_with_nonmonotonic_source_indices",
        "outputs_with_artificial_boundary_discontinuity",
        "trajectories_with_incomplete_history",
        "trajectories_excluded",
    )
    causal_errors = {column: int(pd.to_numeric(validation[column]).sum()) for column in causal_columns}
    checks["causal_validation_zero_errors"] = {
        "passed": all(value == 0 for value in causal_errors.values()),
        **causal_errors,
    }

    table_checks = {}
    for name in ("true_history_trajectory_table.csv", "held_prefix_trajectory_table.csv"):
        table = pd.read_csv(OUT_DIR / name)
        valid_history = table["valid_history"].astype(str).str.strip().str.lower().isin(("true", "1"))
        table_checks[name] = {
            "rows": int(len(table)),
            "valid_rows": int(valid_history.sum()),
            "max_reconstruction_error": float(
                pd.to_numeric(table["reconstruction_error_vs_original_40_samples"]).max()
            ),
        }
    checks["trajectory_tables"] = {
        "passed": all(
            item["rows"] == 1000
            and item["valid_rows"] == 1000
            and item["max_reconstruction_error"] <= 1e-7
            for item in table_checks.values()
        ),
        "tables": table_checks,
    }

    core_checks = {}
    for bank in CORE_BANKS:
        merged = OUT_DIR / "core_ssi" / bank / "merged"
        bank_result = {}
        for filename, shape in (
            ("ssi_matrix.npy", (100, 1000, 100)),
            ("expected_spikes_matrix.npy", (100, 1000, 100)),
            ("mean_rate_matrix.npy", (100, 1000, 100)),
            ("population_ssi.npy", (100, 1000)),
        ):
            array = np.load(merged / filename, mmap_mode="r")
            bank_result[filename] = {
                "shape": list(array.shape),
                "expected_shape": list(shape),
                "all_finite": bool(np.isfinite(array).all()),
            }
        core_checks[bank] = bank_result
    checks["corrected_core_arrays"] = {
        "passed": all(
            item["shape"] == item["expected_shape"] and item["all_finite"]
            for bank in core_checks.values()
            for item in bank.values()
        ),
        "banks": core_checks,
    }

    reproduction = pd.read_csv(OUT_DIR / "legacy_core_reproduction.csv")
    max_ssi_error = float(reproduction["absolute_ssi_error_percent_points"].max())
    max_path_error = float(reproduction["absolute_path_median_error_arcmin"].max())
    checks["frozen_figure4b_reproduction"] = {
        "passed": len(reproduction) == 26 and max_ssi_error <= 1e-8 and max_path_error <= 1e-10,
        "n_points": int(len(reproduction)),
        "max_ssi_error_percent_points": max_ssi_error,
        "max_path_error_arcmin": max_path_error,
    }

    statistics = json.loads((OUT_DIR / "statistics.json").read_text(encoding="utf-8"))
    decision = str(statistics["decision"])
    true_history = statistics["true_history"]
    overall_positive = (
        float(true_history["overall_drift"]["ci95_low_paired_image_boot"]) > 0.0
    )
    low_progressive = (
        float(true_history["low_curve_spearman_rho"]) > 0.8
        and float(true_history["low_last_minus_first_absolute_ssi"]["ci95_low"]) > 0.0
    )
    high_peak = int(true_history["high_peak_bin_index"])
    high_peak_contrast = true_history["high_peak_minus_last_absolute_ssi"]
    high_intermediate = (
        0 < high_peak < 7
        and float(high_peak_contrast["point"]) > 0.0
        and float(high_peak_contrast["ci95_low"]) > 0.0
    )
    if overall_positive and low_progressive and high_intermediate:
        expected_decision = "CORE RESULT SURVIVES"
    elif overall_positive and (low_progressive or high_intermediate):
        expected_decision = "CORE RESULT CHANGES"
    else:
        expected_decision = "CORE RESULT FAILS"
    report_text = (OUT_DIR / "CORE_CORRECTION_REPORT.md").read_text(encoding="utf-8")
    checks["decision_valid"] = {
        "passed": (
            decision in {"CORE RESULT SURVIVES", "CORE RESULT CHANGES", "CORE RESULT FAILS"}
            and decision == expected_decision
            and f"**{decision}**" in report_text
            and int(statistics["n_low_sf"]) == 71
            and int(statistics["n_high_sf"]) == 29
            and int(statistics["n_excluded"]) == 0
        ),
        "decision": decision,
        "expected_from_saved_predicates": expected_decision,
        "overall_positive": overall_positive,
        "low_progressive": low_progressive,
        "high_intermediate": high_intermediate,
        "report_contains_exact_decision": f"**{decision}**" in report_text,
    }

    controlled_required = decision == "CORE RESULT SURVIVES"
    controlled_dir = OUT_DIR / "controlled_scaling"
    controlled_missing = []
    controlled_validation: dict[str, Any] = {}
    if controlled_required:
        for name in (
            "corrected_controlled_scaling_response.npz",
            "corrected_controlled_scaling_curves.csv",
            "validation.json",
            "manifest.json",
        ):
            if not (controlled_dir / name).is_file():
                controlled_missing.append(name)
        if not controlled_missing:
            controlled_validation = json.loads(
                (controlled_dir / "validation.json").read_text(encoding="utf-8")
            )
            with np.load(controlled_dir / "corrected_controlled_scaling_response.npz") as archive:
                controlled_shape = list(archive["ssi"].shape)
            controlled_validation["ssi_shape"] = controlled_shape
        for suffix in ("pdf", "svg", "png"):
            figure_path = controlled_dir / f"fig_controlled_scaling_corrected.{suffix}"
            if not figure_path.is_file() or figure_path.stat().st_size < 1024:
                controlled_missing.append(figure_path.name)
    scale1_errors = [
        float(value)
        for key, value in controlled_validation.items()
        if key.endswith("scale1_max_abs_ssi_error_vs_core")
    ]
    checks["gate_conditional_controlled_scaling"] = {
        "passed": (not controlled_required)
        or (
            not controlled_missing
            and controlled_validation.get("ssi_shape") == [2, 8, 8, 8, 100]
            and len(scale1_errors) == 2
            and all(value <= 1e-6 for value in scale1_errors)
        ),
        "required": controlled_required,
        "missing": controlled_missing,
        "validation": controlled_validation,
    }

    manifest = json.loads((OUT_DIR / "analysis_manifest.json").read_text(encoding="utf-8"))
    hash_errors = []
    for item in manifest["outputs"]:
        path = Path(item["path"])
        if not path.is_file():
            hash_errors.append({"path": str(path), "reason": "missing"})
        else:
            observed = sha256_file(path)
            if observed != item["sha256"]:
                hash_errors.append(
                    {"path": str(path), "reason": "sha256_mismatch", "observed": observed}
                )
    checks["manifest_hashes"] = {
        "passed": not hash_errors,
        "n_hashed_outputs": int(len(manifest["outputs"])),
        "errors": hash_errors,
    }

    passed = all(bool(item["passed"]) for item in checks.values())
    payload = {"passed": passed, "checks": checks}
    if args.write:
        write_json(OUT_DIR / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
