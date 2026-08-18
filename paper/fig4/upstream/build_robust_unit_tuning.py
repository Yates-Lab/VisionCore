#!/usr/bin/env python3
"""Build Figure-4 unit metadata from a cycle-valid native tuning fit.

The historical RR100 table used an SF grid containing sub-cycle gratings and
absolute thresholds tied to that grid.  A selected twin probed on a
cycle-valid grid needs a different contract: retain the measured preferred
SF/TF and censoring fields, and define relative low/middle/high populations by
stable tertile tails of the continuous, response-weighted SF center.

This script writes a *tuning-only* table.  Apply it to a merged trace bank with
``replace_unit_feature_tuning.py`` so the old checkpoint-specific columns are
removed rather than left beside the selected twin's measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRIC = "weighted_center_sf_cpd"


def production_sf_mask(
    units: pd.DataFrame,
    *,
    group: str,
    mode: str,
    metric_column: str = "sf_split_metric",
    low_max_cpd: float = 0.5,
    high_min_cpd: float = 0.75,
) -> np.ndarray:
    """Resolve historical absolute cuts or selected-twin table groups."""
    if mode == "table_tertiles":
        if "sf_group" not in units.columns:
            raise ValueError("table_tertiles mode requires unit.sf_group")
        target = {"low": "low_sf", "high": "high_sf"}.get(group)
        if target is None:
            raise ValueError(f"Unknown SF group {group!r}")
        return units["sf_group"].astype(str).eq(target).to_numpy(dtype=bool)
    if mode != "absolute_cpd":
        raise ValueError(f"Unknown SF group mode {mode!r}")
    if metric_column not in units.columns:
        raise ValueError(f"Unit table is missing {metric_column!r}")
    values = pd.to_numeric(units[metric_column], errors="coerce").to_numpy(dtype=float)
    if group == "low":
        return np.isfinite(values) & (values < float(low_max_cpd))
    if group == "high":
        return np.isfinite(values) & (values >= float(high_min_cpd))
    raise ValueError(f"Unknown SF group {group!r}")


def robust_orientation_metadata(grouped: pd.DataFrame) -> pd.DataFrame:
    """Estimate preferred orientation and vector OSI from the native probe."""
    required = {
        "unit_index",
        "probe_orientation_deg",
        "temporal_hz",
        "response_amp_rms",
    }
    if missing := sorted(required - set(grouped.columns)):
        raise ValueError(f"Grouped tuning table is missing {missing}")
    values = grouped[list(required)].copy()
    for column in required:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values[
        values["temporal_hz"].gt(0)
        & np.isfinite(values["response_amp_rms"])
        & np.isfinite(values["probe_orientation_deg"])
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for unit_index, unit in values.groupby("unit_index", sort=True):
        scores = (
            unit.groupby("probe_orientation_deg", sort=True)["response_amp_rms"]
            .apply(lambda x: float(np.sqrt(np.nanmean(np.square(x.to_numpy(dtype=float))))))
            .reset_index(name="orientation_rms_response")
        )
        weights = np.clip(scores["orientation_rms_response"].to_numpy(dtype=float), 0.0, None)
        angles = scores["probe_orientation_deg"].to_numpy(dtype=float)
        total = float(np.sum(weights))
        if total > 0:
            vector = np.sum(weights * np.exp(2j * np.deg2rad(angles))) / total
            osi = float(np.abs(vector))
            preferred = float(angles[int(np.argmax(weights))])
        else:
            osi = float("nan")
            preferred = float("nan")
        rows.append(
            {
                "unit_index": int(unit_index),
                "prior_preferred_orientation_deg": preferred,
                "prior_orientation_selectivity_index": osi,
                "orientation_tuning_contract": (
                    "preferred bar axis maximizes RMS dynamic response over the cycle-valid "
                    "SF/TF grid; OSI is the magnitude of the response-weighted doubled-angle vector"
                ),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    ).to_numpy(dtype=bool)


def build_robust_tuning_table(
    base: pd.DataFrame,
    robust: pd.DataFrame,
    *,
    metric_column: str = DEFAULT_METRIC,
    grouped: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a complete production-compatible tuning table and audit record."""
    required_base = {"unit_index", "unit_label", "rr100_active"}
    required_robust = {
        "unit_index",
        metric_column,
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "peak_censored",
        "peak_censoring",
    }
    if missing := sorted(required_base - set(base.columns)):
        raise ValueError(f"Base unit table is missing {missing}")
    if missing := sorted(required_robust - set(robust.columns)):
        raise ValueError(f"Robust tuning table is missing {missing}")
    if base["unit_index"].duplicated().any():
        raise ValueError("Base unit table contains duplicate unit_index values")
    if robust["unit_index"].duplicated().any():
        raise ValueError("Robust tuning table contains duplicate unit_index values")

    base_ids = set(base["unit_index"].astype(int))
    robust_ids = set(robust["unit_index"].astype(int))
    unknown = sorted(robust_ids - base_ids)
    if unknown:
        raise ValueError(f"Robust tuning contains units absent from the trace bank: {unknown}")
    active = _bool_values(base["rr100_active"])
    active_ids = set(base.loc[active, "unit_index"].astype(int))
    missing_active = sorted(active_ids - robust_ids)
    if missing_active:
        raise ValueError(
            "Cycle-valid tuning is missing active RR100 units; "
            f"first={missing_active[:8]}"
        )
    unexpected_inactive = sorted(robust_ids - active_ids)
    if unexpected_inactive:
        raise ValueError(
            "Cycle-valid tuning unexpectedly includes inactive RR100 units: "
            f"{unexpected_inactive}"
        )

    keep_columns = [column for column in robust.columns if column != "unit_label"]
    output = base[["unit_index", "unit_label"]].merge(
        robust[keep_columns],
        on="unit_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if grouped is not None:
        orientation = robust_orientation_metadata(grouped)
        output = output.merge(
            orientation,
            on="unit_index",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        orientation_source = "cycle_valid_native_probe"
        if "best_orientation_deg" in output.columns:
            mismatch = np.abs(
                (
                    output["prior_preferred_orientation_deg"]
                    - output["best_orientation_deg"]
                    + 90.0
                )
                % 180.0
                - 90.0
            )
            mismatch = mismatch[np.isfinite(mismatch)]
            if mismatch.size and float(np.max(mismatch)) > 1e-9:
                raise ValueError(
                    "Grouped orientation preference disagrees with robust tuning summary"
                )
    else:
        orientation_source = "base_unit_table"
        for column in (
            "prior_preferred_orientation_deg",
            "prior_orientation_selectivity_index",
        ):
            if column in base.columns:
                output[column] = base[column].to_numpy()
        output["orientation_tuning_contract"] = (
            "orientation metadata retained from the base unit table because no grouped "
            "cycle-valid tuning table was supplied"
        )
    metric = pd.to_numeric(output[metric_column], errors="coerce")
    finite = np.isfinite(metric.to_numpy(dtype=float))
    finite_ids = set(output.loc[finite, "unit_index"].astype(int))
    if finite_ids != active_ids:
        missing = sorted(active_ids - finite_ids)
        extra = sorted(finite_ids - active_ids)
        raise ValueError(
            "Finite robust SF metric must identify exactly the active population; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    ordered = output.loc[finite, ["unit_index", metric_column]].copy()
    ordered[metric_column] = pd.to_numeric(ordered[metric_column], errors="coerce")
    ordered = ordered.sort_values(
        [metric_column, "unit_index"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    n_active = int(ordered.shape[0])
    n_tail = n_active // 3
    if n_tail <= 0:
        raise ValueError(f"Need at least three active units, found {n_active}")
    ordered["sf_rank_low_to_high"] = np.arange(1, n_active + 1, dtype=int)
    ordered["sf_group"] = "middle_sf"
    ordered.loc[ordered.index < n_tail, "sf_group"] = "low_sf"
    ordered.loc[ordered.index >= n_active - n_tail, "sf_group"] = "high_sf"

    output = output.merge(
        ordered[["unit_index", "sf_rank_low_to_high", "sf_group"]],
        on="unit_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    output["sf_group"] = output["sf_group"].fillna("inactive")
    output["sf_split_metric"] = metric
    output["sf_split_metric_name"] = "cycle_valid_weighted_center_sf"
    output["sf_split_metric_column"] = metric_column

    ranges: dict[str, dict[str, float | int]] = {}
    for group in ("low_sf", "middle_sf", "high_sf"):
        values = output.loc[output["sf_group"].eq(group), "sf_split_metric"].to_numpy(
            dtype=float
        )
        ranges[group] = {
            "n": int(np.isfinite(values).sum()),
            "minimum_cpd": float(np.nanmin(values)),
            "maximum_cpd": float(np.nanmax(values)),
        }
    labels = {
        group: (
            f"{name} SF tertile ({record['minimum_cpd']:.2f}–"
            f"{record['maximum_cpd']:.2f} cpd; n={record['n']})"
        )
        for group, name, record in (
            ("low_sf", "low", ranges["low_sf"]),
            ("middle_sf", "middle", ranges["middle_sf"]),
            ("high_sf", "high", ranges["high_sf"]),
        )
    }
    labels["inactive"] = f"inactive (n={int((~finite).sum())})"
    definition = (
        "tertile tails of cycle-valid response-weighted SF center; "
        f"n_active={n_active}, n_tail={n_tail}; inactive channels excluded"
    )
    output["sf_group_label"] = output["sf_group"].map(labels)
    output["sf_group_definition"] = definition
    output["frequency_tuning_contract"] = (
        "native-rate steady periodic histories on a cycle-valid half-octave SF grid; "
        "preferred SF/TF from robust 2-D log-Gaussian fits; grouping uses the "
        "continuous response-weighted SF center and retains fit-boundary censoring"
    )

    peak_censored_active = _bool_values(output.loc[finite, "peak_censored"])
    audit = {
        "analysis": "fig4_cycle_valid_robust_unit_tuning",
        "metric_column": metric_column,
        "grouping": "stable tertile tails",
        "n_base_units": int(base.shape[0]),
        "n_active_units": n_active,
        "n_inactive_units": int((~active).sum()),
        "inactive_unit_indices": sorted(base_ids - active_ids),
        "n_tail": n_tail,
        "group_ranges_cpd": ranges,
        "group_definition": definition,
        "n_peak_censored": int(peak_censored_active.sum()),
        "fraction_peak_censored": float(peak_censored_active.mean()),
        "orientation_source": orientation_source,
    }
    return output, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-unit-table", type=Path, required=True)
    parser.add_argument("--robust-tuning-summary", type=Path, required=True)
    parser.add_argument("--grouped-tuning-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric-column", default=DEFAULT_METRIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_unit_table)
    robust = pd.read_csv(args.robust_tuning_summary)
    grouped = pd.read_csv(args.grouped_tuning_csv) if args.grouped_tuning_csv else None
    output, audit = build_robust_tuning_table(
        base,
        robust,
        metric_column=str(args.metric_column),
        grouped=grouped,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    audit.update(
        {
            "base_unit_table": str(args.base_unit_table.resolve()),
            "base_unit_table_sha256": _sha256(args.base_unit_table),
            "robust_tuning_summary": str(args.robust_tuning_summary.resolve()),
            "robust_tuning_summary_sha256": _sha256(args.robust_tuning_summary),
            "grouped_tuning_csv": (
                str(args.grouped_tuning_csv.resolve()) if args.grouped_tuning_csv else None
            ),
            "grouped_tuning_csv_sha256": (
                _sha256(args.grouped_tuning_csv) if args.grouped_tuning_csv else None
            ),
            "output": str(args.out.resolve()),
            "output_sha256": _sha256(args.out),
        }
    )
    args.out.with_name(f"{args.out.stem}_provenance.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(output)} units ({audit['n_active_units']} active) to {args.out}; "
        f"tail n={audit['n_tail']}"
    )


if __name__ == "__main__":
    main()
