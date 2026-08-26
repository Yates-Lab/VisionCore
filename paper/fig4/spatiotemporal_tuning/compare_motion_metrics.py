#!/usr/bin/env python3
"""Compare motion summaries for the Figure-4 population response panel.

The input is a completed natural-image x real-fixation response matrix.  Each
curve is a spike-weighted population estimand: firing rate is pooled over the
selected units, and population SSI is total spatial information divided by
total expected spikes.  Every moving bin is compared with the stabilized
response from the same images and units. Confidence intervals resample images,
fixation traces, and units independently.

Drift-only snippets and snippets containing a detected microsaccade are binned
separately.  This is important: pooling those regimes, taking a median of
unit-wise percentage changes, and using only total path length can erase the
high-SF plateau seen in the earlier Figure-4 analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml


EPS = 1e-12
BLUE = "#0072B2"
ORANGE = "#D55E00"
METRICS = {
    "path_length": (
        "rendered_path_length_arcmin",
        "filtered fixation path length (arcmin)",
    ),
    "mean_speed": ("rendered_speed_mean_deg_s", "mean eye speed (deg/s)"),
    "rms_displacement": ("rendered_rms_radius_arcmin", "RMS eye displacement (arcmin)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument(
        "--tuning-provenance",
        type=Path,
        default=None,
        help="Release-gated summary.json for the validated tuning table.",
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=Path("paper/model_selection/selected_manuscript_model.yaml"),
        help="Selected-model manifest used to reject response matrices from another checkpoint.",
    )
    parser.add_argument(
        "--sf-column",
        default="auto",
        help=(
            "Validated preferred-SF column. 'auto' requires "
            "validated_preferred_sf_cpd."
        ),
    )
    parser.add_argument(
        "--required-audit-category",
        default="trusted",
        help="Tuning audit category allowed to define SF groups.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-drift-bins", type=int, default=8)
    parser.add_argument("--n-microsaccade-bins", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--allow-unfiltered-diagnostic",
        action="store_true",
        help="Permit legacy matrices without continuous zero-phase trace provenance.",
    )
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def _matrix_checkpoint(summary: dict[str, object]) -> tuple[str, str]:
    """Return the unique model label/digest recorded by all matrix shards."""
    labels: set[str] = set()
    digests: set[str] = set()
    for shard in matrix_shard_summaries(summary):
        model = shard.get("model_provenance", {}).get("model", {})
        digest = str(model.get("checkpoint_sha256", ""))
        if digest:
            digests.add(digest)
        checkpoint = str(model.get("checkpoint_path", ""))
        if checkpoint:
            labels.add(Path(checkpoint).parent.name)
    if len(digests) != 1:
        raise ValueError("matrix shards must resolve to exactly one checkpoint digest")
    return (next(iter(labels)) if len(labels) == 1 else "selected model", next(iter(digests)))


def matrix_shard_summaries(summary: dict[str, object]) -> list[dict[str, object]]:
    """Normalize merged-matrix and single-shard provenance schemas."""
    shards = summary.get("shard_summaries", [])
    if isinstance(shards, list) and shards:
        return [value for value in shards if isinstance(value, dict)]
    if isinstance(summary.get("model_provenance"), dict):
        return [summary]
    return []


def selected_checkpoint_digest(model_spec: Path) -> str:
    payload = yaml.safe_load(model_spec.read_text(encoding="utf-8"))
    digest = str(payload.get("checkpoint_sha256", ""))
    if not digest:
        raise ValueError(f"model spec lacks checkpoint_sha256: {model_spec}")
    return digest


def assert_selected_matrix(
    summary: dict[str, object], *, model_spec: Path
) -> tuple[str, str]:
    label, observed = _matrix_checkpoint(summary)
    expected = selected_checkpoint_digest(model_spec)
    if observed != expected:
        raise ValueError(
            "motion-metric matrix checkpoint does not match the selected model: "
            f"observed={observed}, expected={expected}"
        )
    return label, observed


def assert_validated_tuning_provenance(
    tuning_path: Path,
    provenance_path: Path | None,
    *,
    checkpoint_sha256: str,
) -> tuple[Path, dict[str, object]]:
    path = (
        Path(provenance_path)
        if provenance_path is not None
        else Path(tuning_path).parent / "summary.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"validated tuning provenance is required: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not bool(summary.get("release_ready", False)):
        raise ValueError("tuning audit did not pass its release gates")
    observed = str(summary.get("checkpoint_sha256", ""))
    if observed != str(checkpoint_sha256):
        raise ValueError(
            "tuning audit and response matrix use different checkpoints: "
            f"tuning={observed!r}, matrix={checkpoint_sha256!r}"
        )
    contract = summary.get("trusted_tuning_contract", {})
    if str(contract.get("coordinate_assay", "")) != "exact_cid_yu_sf_tf":
        raise ValueError(
            "Figure-4 tuning coordinates must come from the exact-CID Yu SFxTF assay"
        )
    if str(contract.get("sf_column", "")) != "validated_preferred_sf_cpd":
        raise ValueError("tuning audit does not declare the validated SF contract")
    return path.resolve(), summary


def matrix_trace_filter(summary: dict[str, object]) -> dict[str, object] | None:
    contracts = []
    for shard in matrix_shard_summaries(summary):
        provenance = shard.get("trace_bank", {}).get("trace_provenance")
        if provenance is not None:
            contracts.append(provenance.get("filter", {}))
    if not contracts:
        return None
    encoded = {json.dumps(contract, sort_keys=True) for contract in contracts}
    if len(encoded) != 1:
        raise ValueError("matrix shards use different eye-trace filters")
    contract = contracts[0]
    if "zero-phase" not in str(contract.get("kind", "")).lower():
        raise ValueError("matrix eye traces were not continuously zero-phase filtered")
    return contract


def matrix_replay_selection(summary: dict[str, object]) -> dict[str, object] | None:
    contracts = []
    for shard in matrix_shard_summaries(summary):
        provenance = shard.get("trace_bank", {}).get("trace_provenance")
        if provenance is not None:
            contracts.append(
                {
                    "image_selection": provenance.get("image_selection", {}),
                    "trace_selection": provenance.get("trace_selection", {}),
                }
            )
    if not contracts:
        return None
    encoded = {json.dumps(contract, sort_keys=True) for contract in contracts}
    if len(encoded) != 1:
        raise ValueError("matrix shards use different replay selections")
    return contracts[0]


def load_matrix(
    matrix_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    summary = json.loads((matrix_dir / "summary.json").read_text(encoding="utf-8"))
    pilot = summary.get("pilot", {})
    n_images = int(summary.get("n_images", pilot.get("n_images")))
    n_traces = int(summary.get("n_traces", pilot.get("n_traces")))
    n_units = int(summary.get("n_units", pilot.get("n_units")))
    summary = dict(summary)
    summary.setdefault("n_images", n_images)
    summary.setdefault("n_traces", n_traces)
    summary.setdefault("n_units", n_units)
    moving_rate = np.load(matrix_dir / "mean_rate_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_spikes = np.load(
        matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r"
    ).reshape(n_images, n_traces, n_units)
    stable_rate = np.load(matrix_dir / "stabilized_mean_rate_by_image.npy")
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    stable_spikes = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy")
    if stable_rate.shape != (n_images, n_units) or stable_ssi.shape != (
        n_images,
        n_units,
    ) or stable_spikes.shape != (n_images, n_units):
        raise ValueError("stabilized and moving response matrices are not aligned")
    trace_table = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    if len(trace_table) != n_traces:
        raise ValueError("trace feature table does not match response matrices")
    arrays = {
        "moving_rate": np.asarray(moving_rate),
        "moving_ssi": np.asarray(moving_ssi),
        "moving_spikes": np.asarray(moving_spikes),
        "stable_rate": np.asarray(stable_rate),
        "stable_ssi": np.asarray(stable_ssi),
        "stable_spikes": np.asarray(stable_spikes),
    }
    return arrays, trace_table, summary


def high_sf_rate_tail_audit(binned: pd.DataFrame) -> dict[str, object]:
    """Describe whether the highest microsaccade bin keeps accelerating rate gain."""
    rows = binned.loc[
        binned.metric.eq("path_length")
        & binned.outcome.eq("rate")
        & binned.sf_group.eq("high SF")
        & binned.context.eq("microsaccade")
    ].sort_values("bin_index")
    if len(rows) < 4:
        raise ValueError("high-SF microsaccade rate audit needs at least four bins")
    effect_column = (
        "effect_percent" if "effect_percent" in rows else "effect_median_percent"
    )
    values = rows[effect_column].to_numpy(dtype=float)
    dynamic_range = float(np.ptp(values))
    tolerance = max(1.0, 0.15 * dynamic_range)
    last_step = float(values[-1] - values[-2])
    excess_over_prior_tail = float(values[-1] - np.max(values[-3:-1]))
    passes = bool(last_step <= tolerance and excess_over_prior_tail <= tolerance)
    return {
        "context": "detected microsaccade",
        "outcome": "spike-weighted high-SF population rate change",
        "bin_centers_arcmin": rows.x_median.to_numpy(dtype=float).tolist(),
        "bin_effect_percent": values.tolist(),
        "upper_tail_last_step_percent": last_step,
        "upper_tail_excess_over_prior_two_bins_percent": excess_over_prior_tail,
        "descriptive_plateau_tolerance_percent": tolerance,
        "highest_motion_bin_does_not_show_indefinite_acceleration": passes,
        "interpretation": (
            "descriptive shape gate only; crossed uncertainty remains on the plotted bins"
        ),
    }


def high_sf_rate_tail_contrast(
    arrays: dict[str, np.ndarray],
    groups: dict[str, np.ndarray | str | float],
    contexts: np.ndarray,
    path_length: np.ndarray,
    *,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Crossed-bootstrap the final high-SF bin against the prior two bins."""
    context_rows = np.flatnonzero(np.asarray(contexts) == "microsaccade")
    labels = quantile_bins(np.asarray(path_length)[context_rows], int(n_bins))
    trace_bins = [context_rows[labels == index] for index in range(int(n_bins))]
    if len(trace_bins) < 4 or any(len(rows) == 0 for rows in trace_bins[-3:]):
        raise ValueError("high-SF tail contrast needs three populated upper bins")
    reduced = _population_arrays(arrays, np.asarray(groups["high SF"], dtype=int))

    def contrast(
        image_rows: np.ndarray | None,
        sampled_bins: list[np.ndarray],
        unit_rows: np.ndarray | None = None,
    ) -> float:
        values = [
            population_effects(
                reduced,
                rows,
                image_indices=image_rows,
                unit_indices=unit_rows,
            )[0]
            for rows in sampled_bins[-3:]
        ]
        return float(values[-1] - np.mean(values[:-1]))

    center = contrast(None, trace_bins)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=float)
    n_images = reduced["moving_spikes"].shape[0]
    n_units = reduced["moving_spikes_by_unit"].shape[2]
    for draw in range(int(n_bootstrap)):
        image_draw = rng.integers(0, n_images, size=n_images)
        trace_draws = [
            rows[rng.integers(0, len(rows), size=len(rows))]
            for rows in trace_bins
        ]
        unit_draw = rng.integers(0, n_units, size=n_units)
        draws[draw] = contrast(image_draw, trace_draws, unit_draw)
    return {
        "upper_tail_contrast_definition": (
            "highest microsaccade path-length bin minus the mean of the prior two bins"
        ),
        "upper_tail_contrast_percent": float(center),
        "upper_tail_contrast_crossed_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "inference": "crossed image-by-trace-by-unit bootstrap",
    }


def load_effects(matrix_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Legacy unit-wise effects retained only for explicit sensitivity audits."""
    arrays, trace_table, _ = load_matrix(matrix_dir)
    rate_motion = np.mean(arrays["moving_rate"], axis=0)
    rate_stable = np.mean(arrays["stable_rate"], axis=0)[None, :]
    ssi_motion = np.mean(arrays["moving_ssi"], axis=0)
    ssi_stable = np.mean(arrays["stable_ssi"], axis=0)[None, :]
    rate_percent = 100.0 * (rate_motion - rate_stable) / np.maximum(rate_stable, EPS)
    ssi_percent = 100.0 * (ssi_motion - ssi_stable) / np.maximum(ssi_stable, EPS)
    return rate_percent, ssi_percent, trace_table


def measured_sf_groups(
    tuning_path: Path,
    n_units: int,
    *,
    sf_column: str = "auto",
    required_audit_category: str = "trusted",
    minimum_group_size: int = 8,
) -> dict[str, np.ndarray | str | float]:
    tuning = pd.read_csv(tuning_path).sort_values("unit_index")
    tuning = tuning.loc[tuning.unit_index.between(0, n_units - 1)].copy()
    if "population_active" in tuning:
        tuning = tuning.loc[tuning.population_active.astype(bool)].copy()
    elif "rr100_active" in tuning:  # historical matrix compatibility
        tuning = tuning.loc[tuning.rr100_active.astype(bool)].copy()
    if tuning.unit_index.duplicated().any() or len(tuning) < 2:
        raise ValueError("measured tuning unit indices are absent or duplicated")
    active_unit_indices = tuning.unit_index.to_numpy(dtype=int)
    if "audit_category" not in tuning or "validated_tuning" not in tuning:
        raise ValueError(
            "SF groups require an audited tuning table with audit_category and "
            "validated_tuning; unchecked tuning fallbacks are forbidden"
        )
    tuning = tuning.loc[
        tuning.audit_category.eq(str(required_audit_category))
        & tuning.validated_tuning.astype(bool)
    ].copy()
    if sf_column == "auto":
        sf_column = "validated_preferred_sf_cpd"
    if sf_column != "validated_preferred_sf_cpd":
        raise ValueError(
            "production SF groups must use validated_preferred_sf_cpd; "
            f"received {sf_column!r}"
        )
    if sf_column not in tuning:
        raise ValueError(
            f"measured tuning table lacks requested SF column {sf_column!r}"
        )
    tuning = tuning.loc[pd.to_numeric(tuning[sf_column], errors="coerce").notna()].copy()
    values = tuning[sf_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("validated SF coordinates must be finite and positive")
    unit_indices = tuning.unit_index.to_numpy(dtype=int)
    low_cut, high_cut = np.quantile(values, (1.0 / 3.0, 2.0 / 3.0))
    low_units = unit_indices[values <= low_cut]
    high_units = unit_indices[values >= high_cut]
    if min(len(low_units), len(high_units)) < int(minimum_group_size):
        raise ValueError(
            "validated SF tails are too small for the declared population analysis: "
            f"low={len(low_units)}, high={len(high_units)}, minimum={minimum_group_size}"
        )
    return {
        "all active": active_unit_indices,
        "low SF": low_units,
        "high SF": high_units,
        "low_cut_cpd": float(low_cut),
        "high_cut_cpd": float(high_cut),
        "sf_column": str(sf_column),
        "n_tuned_units": int(len(unit_indices)),
        "required_audit_category": str(required_audit_category),
    }


def quantile_bins(x: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    labels = np.empty(len(order), dtype=int)
    for index, rows in enumerate(np.array_split(order, int(count))):
        labels[rows] = int(index)
    return labels


def crossed_bootstrap(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    center = float(np.nanmedian(array))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for draw in range(int(n_bootstrap)):
        traces = rng.integers(0, array.shape[0], size=array.shape[0])
        units = rng.integers(0, array.shape[1], size=array.shape[1])
        draws[draw] = np.nanmedian(array[np.ix_(traces, units)])
    return center, float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def trace_context(trace_table: pd.DataFrame) -> np.ndarray:
    for column in (
        "rendered_n_microsaccade_events",
        "n_microsaccade_events",
        "source_n_microsaccade_events",
        "saved_microsaccade_count",
    ):
        if column in trace_table:
            count = pd.to_numeric(trace_table[column], errors="coerce").fillna(0)
            return np.where(count.to_numpy(dtype=float) > 0, "microsaccade", "drift_only")
    raise ValueError("trace table lacks an audited microsaccade-count column")


def _population_arrays(
    arrays: dict[str, np.ndarray], unit_indices: np.ndarray
) -> dict[str, np.ndarray]:
    units = np.asarray(unit_indices, dtype=int)
    moving_spikes_by_unit = arrays["moving_spikes"][:, :, units]
    moving_information_by_unit = (
        arrays["moving_spikes"][:, :, units] * arrays["moving_ssi"][:, :, units]
    )
    stable_spikes_by_unit = arrays["stable_spikes"][:, units]
    stable_information_by_unit = (
        arrays["stable_spikes"][:, units] * arrays["stable_ssi"][:, units]
    )
    # Rate and expected-spike ratios are identical because every scored sample
    # has the same duration. Keep expected spikes as the weighting primitive so
    # the population SSI numerator and denominator are exactly matched.
    return {
        "moving_spikes": moving_spikes_by_unit.sum(axis=2),
        "moving_information": moving_information_by_unit.sum(axis=2),
        "stable_spikes": stable_spikes_by_unit.sum(axis=1),
        "stable_information": stable_information_by_unit.sum(axis=1),
        "moving_spikes_by_unit": moving_spikes_by_unit,
        "moving_information_by_unit": moving_information_by_unit,
        "stable_spikes_by_unit": stable_spikes_by_unit,
        "stable_information_by_unit": stable_information_by_unit,
        "unit_indices": units,
    }


def population_effects(
    reduced: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    image_indices: np.ndarray | None = None,
    unit_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    traces = np.asarray(trace_indices, dtype=int)
    images = (
        np.arange(reduced["moving_spikes"].shape[0], dtype=int)
        if image_indices is None
        else np.asarray(image_indices, dtype=int)
    )
    if "moving_spikes_by_unit" in reduced:
        units = (
            np.arange(reduced["moving_spikes_by_unit"].shape[2], dtype=int)
            if unit_indices is None
            else np.asarray(unit_indices, dtype=int)
        )
        moving_spikes = float(
            np.mean(
                reduced["moving_spikes_by_unit"][np.ix_(images, traces, units)].sum(
                    axis=2
                )
            )
        )
        moving_information = float(
            np.mean(
                reduced["moving_information_by_unit"][
                    np.ix_(images, traces, units)
                ].sum(axis=2)
            )
        )
        stable_spikes = float(
            np.mean(
                reduced["stable_spikes_by_unit"][np.ix_(images, units)].sum(axis=1)
            )
        )
        stable_information = float(
            np.mean(
                reduced["stable_information_by_unit"][np.ix_(images, units)].sum(
                    axis=1
                )
            )
        )
    else:
        moving_spikes = float(
            np.mean(reduced["moving_spikes"][np.ix_(images, traces)])
        )
        moving_information = float(
            np.mean(reduced["moving_information"][np.ix_(images, traces)])
        )
        stable_spikes = float(np.mean(reduced["stable_spikes"][images]))
        stable_information = float(np.mean(reduced["stable_information"][images]))
    moving_ssi = moving_information / max(moving_spikes, EPS)
    stable_ssi = stable_information / max(stable_spikes, EPS)
    return (
        100.0 * (moving_spikes - stable_spikes) / max(stable_spikes, EPS),
        100.0 * (moving_ssi - stable_ssi) / max(stable_ssi, EPS),
    )


def crossed_population_bootstrap(
    reduced: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    traces = np.asarray(trace_indices, dtype=int)
    centers = population_effects(reduced, traces)
    draws = np.empty((int(n_bootstrap), 2), dtype=float)
    n_images = reduced["moving_spikes"].shape[0]
    n_units = (
        reduced["moving_spikes_by_unit"].shape[2]
        if "moving_spikes_by_unit" in reduced
        else 0
    )
    for draw in range(int(n_bootstrap)):
        image_draw = rng.integers(0, n_images, size=n_images)
        trace_draw = traces[rng.integers(0, len(traces), size=len(traces))]
        unit_draw = (
            rng.integers(0, n_units, size=n_units) if n_units else None
        )
        draws[draw] = population_effects(
            reduced,
            trace_draw,
            image_indices=image_draw,
            unit_indices=unit_draw,
        )
    low = np.quantile(draws, 0.025, axis=0)
    high = np.quantile(draws, 0.975, axis=0)
    return tuple(centers), tuple(low), tuple(high)


def summarize_metric(
    x: np.ndarray,
    arrays: dict[str, np.ndarray],
    groups: dict[str, np.ndarray | str | float],
    contexts: np.ndarray,
    *,
    n_drift_bins: int,
    n_microsaccade_bins: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, correlations = [], []
    context_specs = (
        ("drift_only", int(n_drift_bins)),
        ("microsaccade", int(n_microsaccade_bins)),
    )
    for group_index, group in enumerate(("low SF", "high SF")):
        unit_indices = np.asarray(groups[group], dtype=int)
        reduced = _population_arrays(arrays, unit_indices)
        per_trace = np.asarray(
            [population_effects(reduced, np.asarray((index,))) for index in range(len(x))]
        )
        for outcome_index, outcome in enumerate(("rate", "SSI")):
            outcome_column = 0 if outcome == "rate" else 1
            for context_index, (context, n_bins) in enumerate(context_specs):
                context_rows = np.flatnonzero(contexts == context)
                if len(context_rows) < n_bins:
                    raise ValueError(
                        f"only {len(context_rows)} {context} traces are available for {n_bins} bins"
                    )
                statistic = spearmanr(
                    x[context_rows], per_trace[context_rows, outcome_column], nan_policy="omit"
                )
                correlations.append(
                    {
                        "outcome": outcome,
                        "sf_group": group,
                        "context": context,
                        "spearman_rho": float(statistic.statistic),
                        "spearman_p": float(statistic.pvalue),
                    }
                )
                labels = quantile_bins(x[context_rows], n_bins)
                rng = np.random.default_rng(
                    int(seed)
                    + 1000 * group_index
                    + 100 * outcome_index
                    + 10 * context_index
                )
                for bin_index in range(n_bins):
                    traces = context_rows[labels == bin_index]
                    center, low, high = crossed_population_bootstrap(
                        reduced,
                        traces,
                        n_bootstrap=int(n_bootstrap),
                        rng=rng,
                    )
                    rows.append(
                        {
                            "outcome": outcome,
                            "sf_group": group,
                            "context": context,
                            "bin_index": int(bin_index),
                            "x_median": float(np.median(x[traces])),
                            "effect_percent": float(center[outcome_column]),
                            "ci_low": float(low[outcome_column]),
                            "ci_high": float(high[outcome_column]),
                            "n_traces": int(len(traces)),
                            "n_units": int(len(unit_indices)),
                            "estimator": "spike-weighted population; matched stabilized images and units; image-by-trace-by-unit bootstrap",
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def draw_candidate(
    path: Path,
    *,
    metric_name: str,
    x: np.ndarray,
    arrays: dict[str, np.ndarray],
    groups: dict[str, np.ndarray | str | float],
    contexts: np.ndarray,
    summary: pd.DataFrame,
    xlabel: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(5.7, 2.75), constrained_layout=True)
    colors = {"low SF": BLUE, "high SF": ORANGE}
    for axis, outcome, title in zip(
        axes,
        ("rate", "SSI"),
        ("firing-rate change", "single-spike information change"),
    ):
        for group in ("low SF", "high SF"):
            unit_indices = np.asarray(groups[group], dtype=int)
            reduced = _population_arrays(arrays, unit_indices)
            per_trace = np.asarray(
                [population_effects(reduced, np.asarray((index,))) for index in range(len(x))]
            )
            outcome_column = 0 if outcome == "rate" else 1
            axis.scatter(
                x,
                per_trace[:, outcome_column],
                s=5,
                color=colors[group],
                alpha=0.045,
                edgecolor="none",
                rasterized=True,
            )
            for context, marker, face, linestyle in (
                ("drift_only", "o", "white", "--"),
                ("microsaccade", "o", colors[group], "-"),
            ):
                frame = summary.loc[
                    summary.outcome.eq(outcome)
                    & summary.sf_group.eq(group)
                    & summary.context.eq(context)
                ].sort_values("bin_index")
                effect_column = (
                    "effect_percent"
                    if "effect_percent" in frame
                    else "effect_median_percent"
                )
                effect = frame[effect_column]
                axis.errorbar(
                    frame.x_median,
                    effect,
                    yerr=np.vstack(
                        (
                            effect - frame.ci_low,
                            frame.ci_high - effect,
                        )
                    ),
                    fmt=marker,
                    linestyle=linestyle,
                    color=colors[group],
                    markerfacecolor=face,
                    markeredgecolor=colors[group],
                    lw=1.55,
                    ms=3.8,
                    capsize=2.0,
                    label=(
                        f"{group} (n={int(frame.n_units.iloc[0])})"
                        if context == "microsaccade"
                        else None
                    ),
                )
        axis.axhline(0, color="0.48", lw=0.75)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("motion − stabilized (%)")
        axis.grid(axis="y", alpha=0.16)
    axes[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    axes[1].text(
        0.98,
        0.96,
        "open/dashed: drift only\nfilled/solid: detected microsaccade",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color="0.28",
    )
    figure.suptitle(
        f"Population motion dependence: {metric_name.replace('_', ' ')}",
        fontsize=9.0,
        fontweight="semibold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_comparison(path: Path, candidate_paths: list[Path]) -> None:
    images = [plt.imread(candidate) for candidate in candidate_paths]
    figure, axes = plt.subplots(3, 1, figsize=(6.1, 8.3), constrained_layout=True)
    for axis, image in zip(axes, images):
        axis.imshow(image)
        axis.axis("off")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, trace_table, matrix_summary = load_matrix(args.matrix_dir)
    model_label, checkpoint_sha256 = assert_selected_matrix(
        matrix_summary, model_spec=args.model_spec
    )
    tuning_provenance_path, tuning_provenance = assert_validated_tuning_provenance(
        args.tuning_summary,
        args.tuning_provenance,
        checkpoint_sha256=checkpoint_sha256,
    )
    trace_filter = matrix_trace_filter(matrix_summary)
    if trace_filter is None and not bool(args.allow_unfiltered_diagnostic):
        raise ValueError(
            "matrix lacks continuous zero-phase eye-trace provenance; use a rebuilt "
            "filtered replay or pass --allow-unfiltered-diagnostic for a nonproduction audit"
        )
    replay_selection = matrix_replay_selection(matrix_summary)
    path_cap = (
        replay_selection["trace_selection"].get("maximum_path_length_arcmin")
        if replay_selection is not None
        else None
    )
    selection_gate = bool(
        replay_selection is not None
        and "evenly spaced" in str(
            replay_selection["image_selection"].get("kind", "")
        ).lower()
        and "evenly spaced" in str(
            replay_selection["trace_selection"].get("kind", "")
        ).lower()
        and path_cap is not None
        and float(path_cap) <= 350.0
    )
    groups = measured_sf_groups(
        args.tuning_summary,
        arrays["moving_rate"].shape[2],
        sf_column=str(args.sf_column),
        required_audit_category=str(args.required_audit_category),
    )
    contexts = trace_context(trace_table)
    all_units = np.asarray(groups["all active"], dtype=int)
    all_population = _population_arrays(arrays, all_units)
    population_center, population_low, population_high = crossed_population_bootstrap(
        all_population,
        np.arange(len(trace_table), dtype=int),
        n_bootstrap=int(args.n_bootstrap),
        rng=np.random.default_rng(int(args.seed) + 9000),
    )
    all_bins, all_stats, candidate_paths = [], [], []
    for metric_index, (metric_name, (column, xlabel)) in enumerate(METRICS.items()):
        x = trace_table[column].to_numpy(dtype=float)
        binned, stats = summarize_metric(
            x,
            arrays,
            groups,
            contexts,
            n_drift_bins=int(args.n_drift_bins),
            n_microsaccade_bins=int(args.n_microsaccade_bins),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 1000 * metric_index,
        )
        binned.insert(0, "metric", metric_name)
        stats.insert(0, "metric", metric_name)
        all_bins.append(binned)
        all_stats.append(stats)
        candidate = args.out_dir / f"panel_b_{metric_name}.png"
        draw_candidate(
            candidate,
            metric_name=metric_name,
            x=x,
            arrays=arrays,
            groups=groups,
            contexts=contexts,
            summary=binned,
            xlabel=xlabel,
        )
        candidate_paths.append(candidate)
    binned_table = pd.concat(all_bins, ignore_index=True)
    stats_table = pd.concat(all_stats, ignore_index=True)
    binned_table.to_csv(args.out_dir / "binned_curves.csv", index=False)
    stats_table.to_csv(args.out_dir / "metric_correlations.csv", index=False)
    score = stats_table.groupby("metric").spearman_rho.apply(
        lambda value: float(np.mean(np.abs(value)))
    )
    recommended = str(score.idxmax())
    high_sf_tail = high_sf_rate_tail_audit(binned_table)
    high_sf_tail.update(
        high_sf_rate_tail_contrast(
            arrays,
            groups,
            contexts,
            trace_table[METRICS["path_length"][0]].to_numpy(dtype=float),
            n_bins=int(args.n_microsaccade_bins),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed) + 12000,
        )
    )
    draw_comparison(args.out_dir / "panel_b_metric_comparison.png", candidate_paths)
    curve_values = binned_table[["effect_percent", "ci_low", "ci_high"]].to_numpy(
        dtype=float
    )
    release_checks = {
        "selected_model_checkpoint_matches": bool(checkpoint_sha256),
        "eye_traces_are_continuously_zero_phase_filtered": bool(
            trace_filter is not None
        ),
        "image_and_trace_sampling_is_response_independent": bool(selection_gate),
        "population_tuning_audit_is_release_ready": bool(
            tuning_provenance.get("release_ready", False)
        ),
        "sf_groups_use_validated_trusted_coordinates": bool(
            groups["sf_column"] == "validated_preferred_sf_cpd"
            and groups["required_audit_category"] == "trusted"
            and tuning_provenance.get("trusted_tuning_contract", {}).get(
                "coordinate_assay"
            )
            == "exact_cid_yu_sf_tf"
        ),
        "sf_tail_groups_have_at_least_eight_units": bool(
            min(len(np.asarray(groups["low SF"])), len(np.asarray(groups["high SF"])))
            >= 8
        ),
        "matrix_has_at_least_40_images_and_200_traces": bool(
            int(matrix_summary["n_images"]) >= 40 and len(trace_table) >= 200
        ),
        "curve_estimates_and_intervals_are_finite_and_ordered": bool(
            np.isfinite(curve_values).all()
            and np.all(curve_values[:, 1] <= curve_values[:, 0])
            and np.all(curve_values[:, 0] <= curve_values[:, 2])
        ),
    }
    report = {
        "analysis": "motion-metric comparison for spike-weighted population rate and SSI by measured SF tercile and fixation dynamics",
        "estimator": (
            "spike-weighted population; moving bins use the same images and units as "
            "their stabilized baseline; crossed image-by-trace-by-unit bootstrap"
        ),
        "matrix_dir": str(args.matrix_dir.resolve()),
        "model_spec": str(args.model_spec.resolve()),
        "model_label": model_label,
        "checkpoint_sha256": checkpoint_sha256,
        "tuning_summary": str(args.tuning_summary.resolve()),
        "tuning_provenance": str(tuning_provenance_path),
        "tuning_release_ready": bool(tuning_provenance.get("release_ready", False)),
        "tuning_contract": tuning_provenance.get("trusted_tuning_contract", {}),
        "trace_filter": trace_filter,
        "production_trace_filter_gate": bool(trace_filter is not None),
        "replay_selection": replay_selection,
        "production_response_independent_sampling_gate": selection_gate,
        "n_images": int(matrix_summary["n_images"]),
        "n_traces": int(len(trace_table)),
        "n_units": int(arrays["moving_rate"].shape[2]),
        "n_tuned_units": int(groups["n_tuned_units"]),
        "n_active_units": int(len(all_units)),
        "measured_sf_column": str(groups["sf_column"]),
        "required_audit_category": str(groups["required_audit_category"]),
        "low_sf_max_cpd": float(groups["low_cut_cpd"]),
        "high_sf_min_cpd": float(groups["high_cut_cpd"]),
        "n_low_sf_units": int(len(np.asarray(groups["low SF"]))),
        "n_high_sf_units": int(len(np.asarray(groups["high SF"]))),
        "low_sf_unit_indices": np.asarray(groups["low SF"], dtype=int).tolist(),
        "high_sf_unit_indices": np.asarray(groups["high SF"], dtype=int).tolist(),
        "active_unit_indices": all_units.astype(int).tolist(),
        "n_drift_only_traces": int(np.count_nonzero(contexts == "drift_only")),
        "n_microsaccade_traces": int(np.count_nonzero(contexts == "microsaccade")),
        "n_drift_bins": int(args.n_drift_bins),
        "n_microsaccade_bins": int(args.n_microsaccade_bins),
        "population_average": {
            "scope": "all release-gated exact-CID units and the complete real-fixation distribution",
            "rate_change_percent": {
                "estimate": float(population_center[0]),
                "ci95": [float(population_low[0]), float(population_high[0])],
            },
            "ssi_change_percent": {
                "estimate": float(population_center[1]),
                "ci95": [float(population_low[1]), float(population_high[1])],
            },
            "inference": "crossed image-by-trace-by-unit bootstrap; spike-weighted population estimand",
        },
        "path_and_mean_speed_are_rank_identical": bool(
            np.array_equal(
                np.argsort(trace_table[METRICS["path_length"][0]].to_numpy()),
                np.argsort(trace_table[METRICS["mean_speed"][0]].to_numpy()),
            )
        ),
        "mean_absolute_spearman_rho": {key: float(value) for key, value in score.items()},
        "recommended_by_mean_absolute_rank_association": recommended,
        "production_recommendation": (
            "path length for continuity with the original production figure, with drift-only and "
            "microsaccade snippets shown separately; RMS displacement is a mandatory sensitivity "
            "view because total path length alone cannot establish a high-SF limit"
        ),
        "high_sf_rate_upper_tail_audit": high_sf_tail,
        "release_checks": release_checks,
        "release_ready": bool(all(release_checks.values())),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
