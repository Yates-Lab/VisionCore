#!/usr/bin/env python3
"""Write correction-stage README, decision report, and complete manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    CONTROLLED_DIR,
    CORE_DIR,
    LEGACY_MATRIX_DIR,
    OUT_DIR,
    SOURCE_CSV,
    sha256_file,
    write_json,
)


def fmt(row: dict) -> str:
    return (
        f"{row['ssi_percent_vs_stabilized']:.2f}% "
        f"(95% paired-image bootstrap CI {row['ci95_low_paired_image_boot']:.2f}% to "
        f"{row['ci95_high_paired_image_boot']:.2f}%)"
    )


def main() -> int:
    stats = json.loads((OUT_DIR / "statistics.json").read_text(encoding="utf-8"))
    decision = stats["decision"]
    true = stats["true_history"]
    legacy = stats["legacy"]
    held = stats["held_initial"]
    correction = stats["correction_impact"]
    unit_summary = stats["per_unit_drift_effect_summary"]
    report = f"""# CORE FIGURE 4 HISTORY CORRECTION REPORT

## Decision

**{decision}**

The frozen legacy computation is named `legacy_wrapped_prefix` throughout this analysis and is retained only to quantify bug impact. The corrected scientific reference is `real_trace_true_history_v1`; `real_trace_held_initial_history_v1` is a synthetic no-prior-motion control.

## Causal convention and reconstruction

The exact lag embedder includes the current retinal sample. Output time `t` consumes `t-31 ... t`, so 31 preceding source samples plus the current sample are required. All 1,000 snippets had at least 31 valid preceding samples inside the same selected source fixation window. One translation—the mean of the scored 40 source samples—was subtracted from each complete 71-sample segment. This reproduced all stored scored trajectories exactly. Both corrected banks contain 40,000 outputs with zero future samples, zero nonmonotonic histories, zero artificial boundary discontinuities, and zero trajectory exclusions. The renderer matches the nine uncontaminated legacy outputs pixel-for-pixel.

## A. Does FEM motion still increase SSI relative to stabilization?

For drift-only trajectories and all 100 units, true-history motion produced {fmt(true['overall_drift'])}. The legacy estimate was {fmt(legacy['overall_drift'])}; the held-prefix control was {fmt(held['overall_drift'])}.

## B. Does low-SF SSI still increase progressively with motion?

The true-history low-SF aggregate drift benefit was {fmt(true['low_sf_drift'])}. Across the eight historical path bins, Spearman rho was {true['low_curve_spearman_rho']:.3f}. The last-minus-first absolute-SSI contrast was {true['low_last_minus_first_absolute_ssi']['point']:.5f}, CI [{true['low_last_minus_first_absolute_ssi']['ci95_low']:.5f}, {true['low_last_minus_first_absolute_ssi']['ci95_high']:.5f}].

## C. Does high-SF SSI retain an intermediate optimum?

The true-history high-SF aggregate drift benefit was {fmt(true['high_sf_drift'])}. Its maximum occurred in path bin {true['high_peak_bin_index']} (zero-based). The peak-minus-last absolute-SSI contrast was {true['high_peak_minus_last_absolute_ssi']['point']:.5f}, CI [{true['high_peak_minus_last_absolute_ssi']['ci95_low']:.5f}, {true['high_peak_minus_last_absolute_ssi']['ci95_high']:.5f}].

## D. How much of the prior effect was caused by the prefix bug?

For all units over drift-only trajectories, legacy minus true history was {correction['legacy_minus_true_drift']['all_units']['point_percent_points']:.2f} percentage points (95% paired-image CI {correction['legacy_minus_true_drift']['all_units']['ci95_low_percent_points']:.2f} to {correction['legacy_minus_true_drift']['all_units']['ci95_high_percent_points']:.2f}). Across units, the corrected true-history benefit had mean {unit_summary['real_trace_true_history_v1_benefit_percent_mean']:.2f}% and median {unit_summary['real_trace_true_history_v1_benefit_percent_median']:.2f}%; the legacy-minus-true unit effect had mean {unit_summary['legacy_minus_true_benefit_percent_points_mean']:.2f} and median {unit_summary['legacy_minus_true_benefit_percent_points_median']:.2f} percentage points. The complete curves are saved in `corrected_core_ssi_curves.csv`; paired per-unit effects are in `legacy_vs_corrected_unit_effects.csv`. The polished `fig_history_bug_explained` shows both the input artifact and its effect without treating the legacy bank as evidence.

## E. How different are true-history and held-prefix results?

The all-unit drift estimates were {fmt(true['overall_drift'])} for real prehistory and {fmt(held['overall_drift'])} after holding the prefix at `e[0]`. True minus held was {correction['true_minus_held_drift']['all_units']['point_percent_points']:.2f} percentage points (95% paired-image CI {correction['true_minus_held_drift']['all_units']['ci95_low_percent_points']:.2f} to {correction['true_minus_held_drift']['all_units']['ci95_high_percent_points']:.2f}). This contrast isolates sensitivity to naturally recorded pre-snippet history from motion within the scored interval.

## F. Does the conceptual Figure 4 result survive?

The decision rule was fixed in the analysis script: `CORE RESULT SURVIVES` requires a positive corrected overall effect, a strongly increasing low-SF curve with positive last-minus-first CI, and a statistically supported interior high-SF peak; `CORE RESULT CHANGES` requires the overall effect plus only one group-specific feature; otherwise the result fails. The resulting classification is **{decision}**.

## Statistical contract

Core curves use the historical trajectory membership and bins because no trajectories were excluded; the newly balanced equal-count bins are therefore identical. Point estimates retain the frozen spike-weighted SSI definition and stabilized baselines. Uncertainty uses the original paired image-bootstrap unit and ratio-delta convention (10,000 resamples, seed family rooted at 47). Drift-only and microsaccade-containing trajectories are reported separately, and no unit was dropped.

## Consequence for further work

{"The corrected core result passed the gate, so mechanism analyses may use true history as primary, held prefix as control, and legacy only for artifact comparison." if decision == 'CORE RESULT SURVIVES' else "Per the authorization, the mechanism audit stops here because the core result did not fully survive. The legacy mechanism must not be explained or rescued."}
"""
    (OUT_DIR / "CORE_CORRECTION_REPORT.md").write_text(report, encoding="utf-8")
    readme = f"""# Corrected Figure 4 temporal history

Primary scientific bank: `real_trace_true_history_v1`
Synthetic control: `real_trace_held_initial_history_v1`
Frozen invalid provenance bank: `legacy_wrapped_prefix`

Core decision: **{decision}**

The generation scripts are under `paper/fig4/mechanism_audit_v1/correction/`. Expensive scripts build/score histories; `analyze_core_correction.py`, `plot_core_correction.py`, and `write_correction_report.py` consume saved results without running the model. No frozen Figure 4 product is overwritten.
"""
    (OUT_DIR / "CORRECTED_HISTORY_README.md").write_text(readme, encoding="utf-8")

    files = [
        path
        for path in OUT_DIR.rglob("*")
        if path.is_file()
        and path.name != "analysis_manifest.json"
        and "logs" not in path.relative_to(OUT_DIR).parts
    ]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, check=True, capture_output=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, check=True, capture_output=True
    ).stdout.splitlines()
    manifest = {
        "analysis": "figure4_corrected_temporal_history_v1",
        "decision": decision,
        "git_commit": git_commit,
        "git_status_short_at_report_generation": git_status,
        "primary_bank": "real_trace_true_history_v1",
        "control_bank": "real_trace_held_initial_history_v1",
        "provenance_only_bank": "legacy_wrapped_prefix",
        "legacy_matrix_dir": LEGACY_MATRIX_DIR,
        "source_csv": SOURCE_CSV,
        "source_csv_sha256": sha256_file(SOURCE_CSV),
        "bank_npz": BANK_DIR / "corrected_history_trajectory_banks.npz",
        "bank_npz_sha256": sha256_file(BANK_DIR / "corrected_history_trajectory_banks.npz"),
        "core_dirs": {
            "true": CORE_DIR / "real_trace_true_history_v1/merged",
            "held": CORE_DIR / "real_trace_held_initial_history_v1/merged",
        },
        "operational_logs": {
            "directory": OUT_DIR / "logs",
            "hashing_policy": "excluded because managed-service logs may still be appended after report generation",
        },
        "controlled_scaling_dir": CONTROLLED_DIR if CONTROLLED_DIR.exists() else None,
        "outputs": [
            {"path": path, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(files)
        ],
    }
    write_json(OUT_DIR / "analysis_manifest.json", manifest)
    print(f"Wrote correction report: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
