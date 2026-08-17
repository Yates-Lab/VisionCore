#!/usr/bin/env python3
"""Compare frozen legacy, true-history, and held-prefix core Figure 4 SSI."""

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

from paper.fig4.mechanism_audit_v1.correction.common import (
    CORE_DIR,
    LEGACY_MATRIX_DIR,
    OUT_DIR,
    write_json,
)


EPS = 1e-8
SF_SPLIT = 0.5
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
BANKS = {
    "legacy_wrapped_prefix": LEGACY_MATRIX_DIR,
    "real_trace_true_history_v1": CORE_DIR / "real_trace_true_history_v1/merged",
    "real_trace_held_initial_history_v1": CORE_DIR / "real_trace_held_initial_history_v1/merged",
}
CACHED_FIG4B = LEGACY_MATRIX_DIR / (
    "phase1_phase2_conditioning_v1/plot_collections/"
    "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15_values.csv"
)


def load_bank(path: Path, legacy: bool) -> tuple[np.ndarray, np.ndarray]:
    ssi = np.load(path / "ssi_matrix.npy")
    expected = np.load(path / "expected_spikes_matrix.npy")
    if legacy:
        ssi = ssi.reshape(100, 1000, 100)
        expected = expected.reshape(100, 1000, 100)
    if ssi.shape != (100, 1000, 100) or expected.shape != ssi.shape:
        raise ValueError(f"Unexpected bank shapes in {path}: {ssi.shape}, {expected.shape}")
    return np.asarray(ssi, dtype=np.float64), np.asarray(expected, dtype=np.float64)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.sum(numerator) / max(float(np.sum(denominator)), EPS))


def image_contributions(
    ssi: np.ndarray,
    expected: np.ndarray,
    trace_ids: np.ndarray,
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.ix_(np.arange(ssi.shape[0]), np.asarray(trace_ids, dtype=int), np.asarray(unit_ids, dtype=int))
    return np.sum((ssi * expected)[ix], axis=(1, 2)), np.sum(expected[ix], axis=(1, 2))


def baseline_contributions(
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.sum((stabilized_ssi * stabilized_expected)[:, unit_ids], axis=1),
        np.sum(stabilized_expected[:, unit_ids], axis=1),
    )


def paired_image_bootstrap_percent(
    moving_num: np.ndarray,
    moving_den: np.ndarray,
    base_num: np.ndarray,
    base_den: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float, float]:
    point_moving = ratio(moving_num, moving_den)
    point_base = ratio(base_num, base_den)
    point = 100.0 * (point_moving - point_base) / point_base
    rng = np.random.default_rng(int(seed))
    sample = rng.integers(0, len(moving_num), size=(N_BOOTSTRAP, len(moving_num)))
    boot_moving = np.sum(moving_num[sample], axis=1) / np.maximum(np.sum(moving_den[sample], axis=1), EPS)
    boot_base = np.sum(base_num[sample], axis=1) / np.maximum(np.sum(base_den[sample], axis=1), EPS)
    # Match the frozen Figure 4 convention exactly: bootstrap the absolute
    # moving-minus-stabilized ratio delta, then express its limits relative to
    # the point-estimate stabilized SSI.  Do not divide each bootstrap draw by
    # its resampled baseline.
    boot = boot_moving - boot_base
    low_abs, high_abs = np.percentile(boot, [2.5, 97.5])
    low = 100.0 * float(low_abs) / point_base
    high = 100.0 * float(high_abs) / point_base
    below = (float(np.count_nonzero(boot <= 0.0)) + 1.0) / (float(N_BOOTSTRAP) + 1.0)
    above = (float(np.count_nonzero(boot >= 0.0)) + 1.0) / (float(N_BOOTSTRAP) + 1.0)
    p = 2.0 * min(below, above)
    return point, float(low), float(high), min(p, 1.0)


def paired_curve_difference_bootstrap(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    *,
    seed: int,
) -> tuple[float, float, float]:
    a_num, a_den = a
    b_num, b_den = b
    point = ratio(a_num, a_den) - ratio(b_num, b_den)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(a_num), size=(N_BOOTSTRAP, len(a_num)))
    av = np.sum(a_num[sample], axis=1) / np.maximum(np.sum(a_den[sample], axis=1), EPS)
    bv = np.sum(b_num[sample], axis=1) / np.maximum(np.sum(b_den[sample], axis=1), EPS)
    low, high = np.percentile(av - bv, [2.5, 97.5])
    return float(point), float(low), float(high)


def paired_bank_difference_percent(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray],
    *,
    seed: int,
) -> dict[str, float]:
    """Paired-image bank contrast in stabilized-SSI percentage points."""
    a_num, a_den = a
    b_num, b_den = b
    base_num, base_den = baseline
    point_base = ratio(base_num, base_den)
    point = 100.0 * (ratio(a_num, a_den) - ratio(b_num, b_den)) / point_base
    rng = np.random.default_rng(int(seed))
    sample = rng.integers(0, len(a_num), size=(N_BOOTSTRAP, len(a_num)))
    av = np.sum(a_num[sample], axis=1) / np.maximum(np.sum(a_den[sample], axis=1), EPS)
    bv = np.sum(b_num[sample], axis=1) / np.maximum(np.sum(b_den[sample], axis=1), EPS)
    boot = 100.0 * (av - bv) / point_base
    low, high = np.percentile(boot, [2.5, 97.5])
    below = (float(np.count_nonzero(boot <= 0.0)) + 1.0) / (float(N_BOOTSTRAP) + 1.0)
    above = (float(np.count_nonzero(boot >= 0.0)) + 1.0) / (float(N_BOOTSTRAP) + 1.0)
    return {
        "point_percent_points": float(point),
        "ci95_low_percent_points": float(low),
        "ci95_high_percent_points": float(high),
        "p_paired_image_boot_sign": float(min(1.0, 2.0 * min(below, above))),
    }


def unit_effects(
    bank_data: dict[str, tuple[np.ndarray, np.ndarray]],
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    trace_ids: np.ndarray,
) -> pd.DataFrame:
    base_num = np.sum(stabilized_ssi * stabilized_expected, axis=0)
    base_den = np.sum(stabilized_expected, axis=0)
    base = base_num / np.maximum(base_den, EPS)
    rows = {"unit_index": np.arange(100, dtype=int), "stabilized_ssi": base}
    for bank, (ssi, expected) in bank_data.items():
        moving_num = np.sum((ssi[:, trace_ids] * expected[:, trace_ids]), axis=(0, 1))
        moving_den = np.sum(expected[:, trace_ids], axis=(0, 1))
        moving = moving_num / np.maximum(moving_den, EPS)
        rows[f"{bank}_moving_ssi"] = moving
        rows[f"{bank}_benefit_percent"] = 100.0 * (moving - base) / np.maximum(base, EPS)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plot_data").mkdir(parents=True, exist_ok=True)
    bank_data = {name: load_bank(path, name == "legacy_wrapped_prefix") for name, path in BANKS.items()}
    stabilized_ssi = np.asarray(np.load(LEGACY_MATRIX_DIR / "stabilized_ssi_by_image.npy"), dtype=np.float64)
    stabilized_expected = np.asarray(
        np.load(LEGACY_MATRIX_DIR / "stabilized_expected_spikes_by_image.npy"), dtype=np.float64
    )
    trace_table = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    unit_table = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index").reset_index(drop=True)
    path = pd.to_numeric(trace_table["rendered_path_length_arcmin"], errors="coerce").to_numpy(dtype=float)
    has_ms = pd.to_numeric(trace_table["rendered_n_microsaccade_events"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0
    sf = pd.to_numeric(unit_table["sf_split_metric"], errors="coerce").to_numpy(dtype=float)
    groups = {
        "all_units": np.arange(100, dtype=int),
        "low_sf_lt0p5": np.flatnonzero(sf < SF_SPLIT),
        "high_sf_ge0p5": np.flatnonzero(sf >= SF_SPLIT),
    }
    contexts = {"drift_only": np.flatnonzero(~has_ms), "microsaccade": np.flatnonzero(has_ms)}
    curve_rows: list[dict[str, Any]] = []
    contrib: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for context, ids in contexts.items():
        ordered = ids[np.argsort(path[ids], kind="mergesort")]
        chunks = np.array_split(ordered, 8 if context == "drift_only" else 5)
        for bin_index, trace_ids in enumerate(chunks):
            for group, unit_ids in groups.items():
                base_num, base_den = baseline_contributions(stabilized_ssi, stabilized_expected, unit_ids)
                for bank_index, (bank, (ssi, expected)) in enumerate(bank_data.items()):
                    moving_num, moving_den = image_contributions(ssi, expected, trace_ids, unit_ids)
                    contrib[(bank, context, group, bin_index)] = (moving_num, moving_den)
                    point, low, high, p = paired_image_bootstrap_percent(
                        moving_num,
                        moving_den,
                        base_num,
                        base_den,
                        seed=BOOTSTRAP_SEED + bank_index * 1000 + bin_index,
                    )
                    curve_rows.append(
                        {
                            "bank": bank,
                            "context": context,
                            "sf_group": group,
                            "bin_scheme": "historical_equal_count__balanced_identical_no_exclusions",
                            "bin_index": bin_index,
                            "path_min_arcmin": float(np.min(path[trace_ids])),
                            "path_median_arcmin": float(np.median(path[trace_ids])),
                            "path_max_arcmin": float(np.max(path[trace_ids])),
                            "n_images": 100,
                            "n_trajectories": len(trace_ids),
                            "n_units": len(unit_ids),
                            "ssi_percent_vs_stabilized": point,
                            "ci95_low_paired_image_boot": low,
                            "ci95_high_paired_image_boot": high,
                            "p_paired_image_boot_sign": p,
                        }
                    )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT_DIR / "corrected_core_ssi_curves.csv", index=False)
    curves.to_csv(OUT_DIR / "plot_data/fig_core_result_corrected.csv", index=False)

    cached = pd.read_csv(CACHED_FIG4B)
    cached = cached[cached["relation"].eq("strong_contours_no_osi")].copy()
    cached["sf_group"] = cached["sf_group"].map(
        {"low_lt0p5": "low_sf_lt0p5", "high_ge0p75": "high_sf_ge0p5"}
    )
    cached["bin_index"] = cached["path_bin_order"].astype(int) - 1
    legacy_reproduction = curves[
        curves["bank"].eq("legacy_wrapped_prefix")
        & curves["sf_group"].isin(("low_sf_lt0p5", "high_sf_ge0p5"))
    ].merge(
        cached[
            [
                "context",
                "sf_group",
                "bin_index",
                "path_median_arcmin",
                "ssi_percent_vs_cell_baseline",
            ]
        ].rename(
            columns={
                "path_median_arcmin": "cached_path_median_arcmin",
                "ssi_percent_vs_cell_baseline": "cached_ssi_percent_vs_stabilized",
            }
        ),
        on=["context", "sf_group", "bin_index"],
        how="left",
        validate="one_to_one",
    )
    legacy_reproduction["absolute_path_median_error_arcmin"] = np.abs(
        legacy_reproduction["path_median_arcmin"]
        - legacy_reproduction["cached_path_median_arcmin"]
    )
    legacy_reproduction["absolute_ssi_error_percent_points"] = np.abs(
        legacy_reproduction["ssi_percent_vs_stabilized"]
        - legacy_reproduction["cached_ssi_percent_vs_stabilized"]
    )
    if legacy_reproduction[
        ["cached_path_median_arcmin", "cached_ssi_percent_vs_stabilized"]
    ].isna().any().any():
        raise RuntimeError("Frozen Figure 4B reproduction table did not match every legacy curve row")
    max_legacy_ssi_error = float(
        legacy_reproduction["absolute_ssi_error_percent_points"].max()
    )
    max_legacy_path_error = float(
        legacy_reproduction["absolute_path_median_error_arcmin"].max()
    )
    if max_legacy_ssi_error > 1e-8 or max_legacy_path_error > 1e-10:
        raise RuntimeError(
            "Frozen Figure 4B point reproduction failed: "
            f"max SSI error={max_legacy_ssi_error}, path error={max_legacy_path_error}"
        )
    legacy_reproduction.to_csv(OUT_DIR / "legacy_core_reproduction.csv", index=False)

    effects = unit_effects(bank_data, stabilized_ssi, stabilized_expected, contexts["drift_only"])
    effects["sf_split_metric_cpd"] = sf
    effects["sf_group"] = np.where(sf < SF_SPLIT, "low_sf_lt0p5", "high_sf_ge0p5")
    effects["legacy_minus_true_benefit_percent_points"] = (
        effects["legacy_wrapped_prefix_benefit_percent"]
        - effects["real_trace_true_history_v1_benefit_percent"]
    )
    effects["true_minus_held_benefit_percent_points"] = (
        effects["real_trace_true_history_v1_benefit_percent"]
        - effects["real_trace_held_initial_history_v1_benefit_percent"]
    )
    effects.to_csv(OUT_DIR / "legacy_vs_corrected_unit_effects.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    aggregate_contrib: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    aggregate_baseline: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for context, trace_ids in contexts.items():
        for group, unit_ids in groups.items():
            base_num, base_den = baseline_contributions(stabilized_ssi, stabilized_expected, unit_ids)
            aggregate_baseline[(context, group)] = (base_num, base_den)
            for bank_index, (bank, (ssi, expected)) in enumerate(bank_data.items()):
                moving_num, moving_den = image_contributions(ssi, expected, trace_ids, unit_ids)
                aggregate_contrib[(bank, context, group)] = (moving_num, moving_den)
                point, low, high, p = paired_image_bootstrap_percent(
                    moving_num,
                    moving_den,
                    base_num,
                    base_den,
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
                        "n_images": 100,
                        "n_trajectories": len(trace_ids),
                        "n_units": len(unit_ids),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "corrected_core_ssi_summary.csv", index=False)

    correction_impact: dict[str, dict[str, dict[str, float]]] = {
        "legacy_minus_true_drift": {},
        "true_minus_held_drift": {},
    }
    for group_index, group in enumerate(groups):
        correction_impact["legacy_minus_true_drift"][group] = paired_bank_difference_percent(
            aggregate_contrib[("legacy_wrapped_prefix", "drift_only", group)],
            aggregate_contrib[("real_trace_true_history_v1", "drift_only", group)],
            aggregate_baseline[("drift_only", group)],
            seed=2200 + group_index,
        )
        correction_impact["true_minus_held_drift"][group] = paired_bank_difference_percent(
            aggregate_contrib[("real_trace_true_history_v1", "drift_only", group)],
            aggregate_contrib[("real_trace_held_initial_history_v1", "drift_only", group)],
            aggregate_baseline[("drift_only", group)],
            seed=2300 + group_index,
        )

    unit_effect_summary: dict[str, float] = {}
    for column in (
        "legacy_wrapped_prefix_benefit_percent",
        "real_trace_true_history_v1_benefit_percent",
        "real_trace_held_initial_history_v1_benefit_percent",
        "legacy_minus_true_benefit_percent_points",
        "true_minus_held_benefit_percent_points",
    ):
        values = effects[column].to_numpy(dtype=float)
        unit_effect_summary[f"{column}_mean"] = float(np.nanmean(values))
        unit_effect_summary[f"{column}_median"] = float(np.nanmedian(values))

    def curve(bank: str, group: str, context: str = "drift_only") -> pd.DataFrame:
        return curves[
            curves["bank"].eq(bank) & curves["sf_group"].eq(group) & curves["context"].eq(context)
        ].sort_values("bin_index")

    true_low = curve("real_trace_true_history_v1", "low_sf_lt0p5")
    true_high = curve("real_trace_true_history_v1", "high_sf_ge0p5")
    low_rho = spearmanr(true_low["path_median_arcmin"], true_low["ssi_percent_vs_stabilized"]).statistic
    high_values = true_high["ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    high_peak = int(np.nanargmax(high_values))
    # Paired-image absolute SSI differences; sign is identical to a percent
    # difference because both bins share the same stabilized denominator.
    low_last_first = paired_curve_difference_bootstrap(
        contrib[("real_trace_true_history_v1", "drift_only", "low_sf_lt0p5", 7)],
        contrib[("real_trace_true_history_v1", "drift_only", "low_sf_lt0p5", 0)],
        seed=1201,
    )
    high_peak_last = paired_curve_difference_bootstrap(
        contrib[("real_trace_true_history_v1", "drift_only", "high_sf_ge0p5", high_peak)],
        contrib[("real_trace_true_history_v1", "drift_only", "high_sf_ge0p5", 7)],
        seed=1202,
    )
    overall_true = summary[
        summary["bank"].eq("real_trace_true_history_v1")
        & summary["context"].eq("drift_only")
        & summary["sf_group"].eq("all_units")
    ].iloc[0]
    overall_positive = float(overall_true["ci95_low_paired_image_boot"]) > 0.0
    low_progressive = bool(low_rho > 0.8 and low_last_first[1] > 0.0)
    high_intermediate = bool(0 < high_peak < 7 and high_peak_last[0] > 0.0 and high_peak_last[1] > 0.0)
    if overall_positive and low_progressive and high_intermediate:
        decision = "CORE RESULT SURVIVES"
    elif overall_positive and (low_progressive or high_intermediate):
        decision = "CORE RESULT CHANGES"
    else:
        decision = "CORE RESULT FAILS"

    def record(bank: str, group: str, context: str = "drift_only") -> dict[str, Any]:
        row = summary[
            summary["bank"].eq(bank) & summary["context"].eq(context) & summary["sf_group"].eq(group)
        ].iloc[0]
        return row.to_dict()

    stats = {
        "decision": decision,
        "decision_rule": {
            "survives": "true-history overall drift benefit CI>0, low-SF curve rho>0.8 with last-minus-first CI>0, and high-SF peak is interior with peak-minus-last CI>0",
            "changes": "overall drift benefit CI>0 and exactly one low/high qualitative feature survives",
            "fails": "otherwise",
        },
        "n_trajectories": 1000,
        "n_excluded": 0,
        "n_low_sf": len(groups["low_sf_lt0p5"]),
        "n_high_sf": len(groups["high_sf_ge0p5"]),
        "correction_impact": correction_impact,
        "per_unit_drift_effect_summary": unit_effect_summary,
        "legacy_figure4b_reproduction": {
            "cached_source": str(CACHED_FIG4B),
            "n_curve_points": int(len(legacy_reproduction)),
            "max_absolute_ssi_error_percent_points": max_legacy_ssi_error,
            "max_absolute_path_median_error_arcmin": max_legacy_path_error,
        },
        "true_history": {
            "overall_drift": record("real_trace_true_history_v1", "all_units"),
            "low_sf_drift": record("real_trace_true_history_v1", "low_sf_lt0p5"),
            "high_sf_drift": record("real_trace_true_history_v1", "high_sf_ge0p5"),
            "low_curve_spearman_rho": float(low_rho),
            "low_last_minus_first_absolute_ssi": {
                "point": low_last_first[0], "ci95_low": low_last_first[1], "ci95_high": low_last_first[2]
            },
            "high_peak_bin_index": high_peak,
            "high_peak_minus_last_absolute_ssi": {
                "point": high_peak_last[0], "ci95_low": high_peak_last[1], "ci95_high": high_peak_last[2]
            },
        },
        "legacy": {
            "overall_drift": record("legacy_wrapped_prefix", "all_units"),
            "low_sf_drift": record("legacy_wrapped_prefix", "low_sf_lt0p5"),
            "high_sf_drift": record("legacy_wrapped_prefix", "high_sf_ge0p5"),
        },
        "held_initial": {
            "overall_drift": record("real_trace_held_initial_history_v1", "all_units"),
            "low_sf_drift": record("real_trace_held_initial_history_v1", "low_sf_lt0p5"),
            "high_sf_drift": record("real_trace_held_initial_history_v1", "high_sf_ge0p5"),
        },
        "original_statistics": "paired image bootstrap of the absolute ratio delta, 10000 resamples, percent limits divided by the point stabilized SSI, seed family rooted at 47",
    }
    write_json(OUT_DIR / "statistics.json", stats)
    np.savez_compressed(
        OUT_DIR / "corrected_core_ssi_outputs.npz",
        curve_records=curves.to_records(index=False),
        summary_records=summary.to_records(index=False),
        unit_effect_records=effects.to_records(index=False),
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
