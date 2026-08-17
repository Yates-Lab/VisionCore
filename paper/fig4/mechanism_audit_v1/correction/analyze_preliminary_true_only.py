#!/usr/bin/env python3
"""Analyze the completed true-history image subset without the held-prefix control.

This is an explicitly preliminary, writing-day checkpoint.  It preserves all
1,000 trajectories, all 100 units, the frozen bins/baselines, and the exact
paired-image bootstrap, but uses only image rows whose corrected outputs have
been atomically committed.  It must not be presented as the final correction
gate because the held-prefix bank is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.analyze_core_correction import (
    BOOTSTRAP_SEED,
    EPS,
    N_BOOTSTRAP,
    SF_SPLIT,
    baseline_contributions,
    image_contributions,
    paired_bank_difference_percent,
    paired_curve_difference_bootstrap,
    paired_image_bootstrap_percent,
    unit_effects,
)
from paper.fig4.mechanism_audit_v1.correction.common import (
    CORE_DIR,
    LEGACY_MATRIX_DIR,
    OUT_DIR,
    write_json,
)


PREVIEW_DIR = OUT_DIR / "preliminary_true_only"
TRUE_BANK = "real_trace_true_history_v1"
LEGACY_BANK = "legacy_wrapped_prefix"
SHARDS = ((0, 50), (50, 100))


def load_committed_true_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids: list[np.ndarray] = []
    ssi_rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    bank_dir = CORE_DIR / TRUE_BANK
    for start, stop in SHARDS:
        shard = bank_dir / f"images_{start:03d}_{stop:03d}"
        completed_path = shard / "completed_images.npy"
        if not completed_path.is_file():
            continue
        completed = np.asarray(np.load(completed_path), dtype=bool)
        local = np.flatnonzero(completed)
        if not len(local):
            continue
        ssi = np.load(shard / "ssi_matrix.npy", mmap_mode="r")
        expected = np.load(shard / "expected_spikes_matrix.npy", mmap_mode="r")
        ssi_rows.append(np.asarray(ssi[local], dtype=np.float64))
        expected_rows.append(np.asarray(expected[local], dtype=np.float64))
        image_ids.append(start + local)
    if not image_ids:
        raise RuntimeError("No atomically committed true-history images were found")
    ids = np.concatenate(image_ids).astype(int)
    order = np.argsort(ids, kind="stable")
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("Duplicate image IDs across true-history shards")
    true_ssi = np.concatenate(ssi_rows, axis=0)[order]
    true_expected = np.concatenate(expected_rows, axis=0)[order]
    ids = ids[order]
    if true_ssi.shape != (len(ids), 1000, 100) or true_expected.shape != true_ssi.shape:
        raise ValueError(f"Unexpected preview shapes: {true_ssi.shape}, {true_expected.shape}")
    if not np.isfinite(true_ssi).all() or not np.isfinite(true_expected).all():
        raise RuntimeError("A committed true-history row contains non-finite values")
    return ids, true_ssi, true_expected


def load_legacy_rows(image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ssi = np.load(LEGACY_MATRIX_DIR / "ssi_matrix.npy", mmap_mode="r").reshape(100, 1000, 100)
    expected = np.load(LEGACY_MATRIX_DIR / "expected_spikes_matrix.npy", mmap_mode="r").reshape(
        100, 1000, 100
    )
    return (
        np.asarray(ssi[image_ids], dtype=np.float64),
        np.asarray(expected[image_ids], dtype=np.float64),
    )


def main() -> int:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (PREVIEW_DIR / "plot_data").mkdir(parents=True, exist_ok=True)
    image_ids, true_ssi, true_expected = load_committed_true_rows()
    legacy_ssi, legacy_expected = load_legacy_rows(image_ids)
    stabilized_ssi = np.asarray(
        np.load(LEGACY_MATRIX_DIR / "stabilized_ssi_by_image.npy")[image_ids], dtype=np.float64
    )
    stabilized_expected = np.asarray(
        np.load(LEGACY_MATRIX_DIR / "stabilized_expected_spikes_by_image.npy")[image_ids],
        dtype=np.float64,
    )
    trace_table = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values(
        "trace_bank_index"
    ).reset_index(drop=True)
    unit_table = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values(
        "unit_index"
    ).reset_index(drop=True)
    path = pd.to_numeric(trace_table["rendered_path_length_arcmin"], errors="coerce").to_numpy(float)
    has_ms = (
        pd.to_numeric(trace_table["rendered_n_microsaccade_events"], errors="coerce")
        .fillna(0)
        .to_numpy(int)
        > 0
    )
    sf = pd.to_numeric(unit_table["sf_split_metric"], errors="coerce").to_numpy(float)
    groups = {
        "all_units": np.arange(100, dtype=int),
        "low_sf_lt0p5": np.flatnonzero(sf < SF_SPLIT),
        "high_sf_ge0p5": np.flatnonzero(sf >= SF_SPLIT),
    }
    contexts = {"drift_only": np.flatnonzero(~has_ms), "microsaccade": np.flatnonzero(has_ms)}
    banks = {
        LEGACY_BANK: (legacy_ssi, legacy_expected),
        TRUE_BANK: (true_ssi, true_expected),
    }

    rows: list[dict[str, Any]] = []
    contributions: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for context, context_ids in contexts.items():
        ordered = context_ids[np.argsort(path[context_ids], kind="mergesort")]
        chunks = np.array_split(ordered, 8 if context == "drift_only" else 5)
        for bin_index, trace_ids in enumerate(chunks):
            for group, unit_ids in groups.items():
                base = baseline_contributions(stabilized_ssi, stabilized_expected, unit_ids)
                for bank_index, (bank, (ssi, expected)) in enumerate(banks.items()):
                    moving = image_contributions(ssi, expected, trace_ids, unit_ids)
                    contributions[(bank, context, group, bin_index)] = moving
                    point, low, high, p = paired_image_bootstrap_percent(
                        *moving,
                        *base,
                        seed=BOOTSTRAP_SEED + bank_index * 1000 + bin_index,
                    )
                    rows.append(
                        {
                            "bank": bank,
                            "context": context,
                            "sf_group": group,
                            "bin_index": bin_index,
                            "path_min_arcmin": float(np.min(path[trace_ids])),
                            "path_median_arcmin": float(np.median(path[trace_ids])),
                            "path_max_arcmin": float(np.max(path[trace_ids])),
                            "n_images": len(image_ids),
                            "n_trajectories": len(trace_ids),
                            "n_units": len(unit_ids),
                            "ssi_percent_vs_stabilized": point,
                            "ci95_low_paired_image_boot": low,
                            "ci95_high_paired_image_boot": high,
                            "p_paired_image_boot_sign": p,
                        }
                    )
    curves = pd.DataFrame(rows)
    curves.to_csv(PREVIEW_DIR / "preliminary_true_only_curves.csv", index=False)
    curves.to_csv(PREVIEW_DIR / "plot_data/fig_preliminary_true_only.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    aggregate_base: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for context, trace_ids in contexts.items():
        for group, unit_ids in groups.items():
            base = baseline_contributions(stabilized_ssi, stabilized_expected, unit_ids)
            aggregate_base[f"{context}:{group}"] = base
            for bank_index, (bank, (ssi, expected)) in enumerate(banks.items()):
                moving = image_contributions(ssi, expected, trace_ids, unit_ids)
                aggregate[(bank, context, group)] = moving
                point, low, high, p = paired_image_bootstrap_percent(
                    *moving,
                    *base,
                    seed=BOOTSTRAP_SEED + 10000 + bank_index,
                )
                summary_rows.append(
                    {
                        "bank": bank,
                        "context": context,
                        "sf_group": group,
                        "ssi_percent_vs_stabilized": point,
                        "ci95_low_paired_image_boot": low,
                        "ci95_high_paired_image_boot": high,
                        "p_paired_image_boot_sign": p,
                        "n_images": len(image_ids),
                        "n_trajectories": len(trace_ids),
                        "n_units": len(unit_ids),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PREVIEW_DIR / "preliminary_true_only_summary.csv", index=False)

    effects = unit_effects(banks, stabilized_ssi, stabilized_expected, contexts["drift_only"])
    effects["sf_split_metric_cpd"] = sf
    effects["sf_group"] = np.where(sf < SF_SPLIT, "low_sf_lt0p5", "high_sf_ge0p5")
    effects["legacy_minus_true_benefit_percent_points"] = (
        effects[f"{LEGACY_BANK}_benefit_percent"] - effects[f"{TRUE_BANK}_benefit_percent"]
    )
    effects.to_csv(PREVIEW_DIR / "preliminary_true_only_unit_effects.csv", index=False)

    def select_curve(bank: str, group: str) -> pd.DataFrame:
        return curves[
            curves["bank"].eq(bank)
            & curves["context"].eq("drift_only")
            & curves["sf_group"].eq(group)
        ].sort_values("bin_index")

    true_low = select_curve(TRUE_BANK, "low_sf_lt0p5")
    true_high = select_curve(TRUE_BANK, "high_sf_ge0p5")
    low_rho = float(
        spearmanr(true_low["path_median_arcmin"], true_low["ssi_percent_vs_stabilized"]).statistic
    )
    high_values = true_high["ssi_percent_vs_stabilized"].to_numpy(float)
    high_peak = int(np.nanargmax(high_values))
    low_contrast = paired_curve_difference_bootstrap(
        contributions[(TRUE_BANK, "drift_only", "low_sf_lt0p5", 7)],
        contributions[(TRUE_BANK, "drift_only", "low_sf_lt0p5", 0)],
        seed=1201,
    )
    high_contrast = paired_curve_difference_bootstrap(
        contributions[(TRUE_BANK, "drift_only", "high_sf_ge0p5", high_peak)],
        contributions[(TRUE_BANK, "drift_only", "high_sf_ge0p5", 7)],
        seed=1202,
    )

    def record(bank: str, group: str, context: str = "drift_only") -> dict[str, Any]:
        return summary[
            summary["bank"].eq(bank)
            & summary["context"].eq(context)
            & summary["sf_group"].eq(group)
        ].iloc[0].to_dict()

    overall_true = record(TRUE_BANK, "all_units")
    overall_positive = float(overall_true["ci95_low_paired_image_boot"]) > 0.0
    low_progressive = low_rho > 0.8 and low_contrast[1] > 0.0
    high_intermediate = 0 < high_peak < 7 and high_contrast[0] > 0.0 and high_contrast[1] > 0.0
    if overall_positive and low_progressive and high_intermediate:
        provisional_signature = "SURVIVES-LIKE"
    elif overall_positive and (low_progressive or high_intermediate):
        provisional_signature = "CHANGES-LIKE"
    else:
        provisional_signature = "FAILS-LIKE"
    correction_impact: dict[str, dict[str, dict[str, float]]] = {}
    for context_index, context in enumerate(contexts):
        correction_impact[context] = {}
        for group_index, group in enumerate(groups):
            correction_impact[context][group] = paired_bank_difference_percent(
                aggregate[(LEGACY_BANK, context, group)],
                aggregate[(TRUE_BANK, context, group)],
                aggregate_base[f"{context}:{group}"],
                seed=2200 + context_index * 100 + group_index,
            )
    legacy_minus_true = correction_impact["drift_only"]["all_units"]
    primary_complete = len(image_ids) == 100 and np.array_equal(image_ids, np.arange(100))
    checkpoint_status = (
        "COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING"
        if primary_complete
        else "PRELIMINARY_TRUE_ONLY__NOT_FINAL_GATE"
    )
    stats = {
        "status": checkpoint_status,
        "provisional_signature": provisional_signature,
        "image_ids": image_ids.tolist(),
        "n_images": int(len(image_ids)),
        "n_trajectories": 1000,
        "n_units": 100,
        "n_low_sf": int(len(groups["low_sf_lt0p5"])),
        "n_high_sf": int(len(groups["high_sf_ge0p5"])),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed_family": BOOTSTRAP_SEED,
        "true_history": {
            "overall_drift": overall_true,
            "low_sf_drift": record(TRUE_BANK, "low_sf_lt0p5"),
            "high_sf_drift": record(TRUE_BANK, "high_sf_ge0p5"),
            "overall_microsaccade": record(TRUE_BANK, "all_units", "microsaccade"),
            "low_sf_microsaccade": record(TRUE_BANK, "low_sf_lt0p5", "microsaccade"),
            "high_sf_microsaccade": record(TRUE_BANK, "high_sf_ge0p5", "microsaccade"),
            "low_curve_spearman_rho": low_rho,
            "low_last_minus_first_absolute_ssi": {
                "point": low_contrast[0],
                "ci95_low": low_contrast[1],
                "ci95_high": low_contrast[2],
            },
            "high_peak_bin_index": high_peak,
            "high_peak_minus_last_absolute_ssi": {
                "point": high_contrast[0],
                "ci95_low": high_contrast[1],
                "ci95_high": high_contrast[2],
            },
        },
        "legacy": {
            "overall_drift": record(LEGACY_BANK, "all_units"),
            "low_sf_drift": record(LEGACY_BANK, "low_sf_lt0p5"),
            "high_sf_drift": record(LEGACY_BANK, "high_sf_ge0p5"),
            "overall_microsaccade": record(LEGACY_BANK, "all_units", "microsaccade"),
            "low_sf_microsaccade": record(LEGACY_BANK, "low_sf_lt0p5", "microsaccade"),
            "high_sf_microsaccade": record(LEGACY_BANK, "high_sf_ge0p5", "microsaccade"),
        },
        "correction_impact_legacy_minus_true": correction_impact,
        "legacy_minus_true_all_units_drift": legacy_minus_true,
        "limitations": [
            (
                "The complete 100-image true-history primary bank is included."
                if primary_complete
                else "Only atomically committed true-history image rows are included."
            ),
            "The held-prefix control has not been scored.",
            "This is not the authorized final correction gate.",
        ],
    }
    write_json(PREVIEW_DIR / "statistics.json", stats)
    np.savez_compressed(
        PREVIEW_DIR / "preliminary_true_only_outputs.npz",
        image_ids=image_ids,
        curve_records=curves.to_records(index=False),
        summary_records=summary.to_records(index=False),
        unit_effect_records=effects.to_records(index=False),
    )

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"{row['ssi_percent_vs_stabilized']:.2f}% "
            f"(95% CI {row['ci95_low_paired_image_boot']:.2f}% to "
            f"{row['ci95_high_paired_image_boot']:.2f}%)"
        )

    report_heading = (
        "COMPLETE PRIMARY TRUE-HISTORY FIGURE 4 RESULT"
        if primary_complete
        else "PRELIMINARY TRUE-HISTORY FIGURE 4 RESULT"
    )
    report = f"""# {report_heading}

**Status: {checkpoint_status.replace('_', ' ')} — NOT THE FINAL CORRECTION GATE**

This snapshot uses {len(image_ids)} atomically completed natural images, all 1,000 trajectories, and all 100 frozen representative units. It preserves the historical path bins, stabilized baselines, SSI weighting, low/high-SF split, and 10,000-draw paired-image bootstrap. It omits the held-prefix control.

## What the primary corrected bank currently shows

- Overall drift-only true-history benefit: {fmt(stats['true_history']['overall_drift'])}
- Low-SF drift-only aggregate benefit: {fmt(stats['true_history']['low_sf_drift'])}
- Low-SF path-dose Spearman rho: {low_rho:.3f}; last-minus-first absolute-SSI CI [{low_contrast[1]:.5f}, {low_contrast[2]:.5f}]
- High-SF drift-only aggregate benefit: {fmt(stats['true_history']['high_sf_drift'])}
- High-SF peak bin: {high_peak}; peak-minus-last absolute-SSI CI [{high_contrast[1]:.5f}, {high_contrast[2]:.5f}]
- Legacy minus true-history overall drift effect: {legacy_minus_true['point_percent_points']:.2f} percentage points (95% CI {legacy_minus_true['ci95_low_percent_points']:.2f} to {legacy_minus_true['ci95_high_percent_points']:.2f})

Applying the predeclared qualitative predicates to this incomplete-control snapshot gives **{provisional_signature}**. This wording is deliberately not `CORE RESULT SURVIVES/CHANGES/FAILS`: the final gate additionally requires all 100 images and the held-prefix bank.
"""
    (PREVIEW_DIR / "PRELIMINARY_TRUE_ONLY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
