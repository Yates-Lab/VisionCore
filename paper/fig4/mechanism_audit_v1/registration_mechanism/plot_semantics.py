#!/usr/bin/env python3
"""Plot registration-mechanism Figure 1 from finalized saved products only.

This module has no model, optimizer, state-cache, or readout-cache imports.  It
fails closed unless rank-8 validation and all twelve fold-by-contrast P/Q
semantic cells are finalized and hash-valid.  Consensus projectors are never
used for inference; every plotted estimate is aggregated from fold-wise
crossed held-out results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
FIGURE_BASENAME = "registration_mechanism_figure1_pq_semantics"
CAPTION_NAME = "FIGURE1_PQ_SEMANTICS_CAPTION.md"
PLOT_MANIFEST_NAME = "figure1_pq_semantics_plot_manifest.json"
SCHEMA_VERSION = "fig4-registration-pq-figure1-v1"

FOLDS = (0, 1, 2, 3)
CONTRAST_ORDER = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
CONTRAST_LABEL = {
    "low_0_to_2": "Lower SF\n0×→2×",
    "high_0_to_1": "Higher SF\n0×→1×",
    "high_1_to_3": "Higher SF reversal\n1×→3×",
}
CONTRAST_DIRECT = {
    "low_0_to_2": "lower-SF 0→2×",
    "high_0_to_1": "higher-SF 0→1×",
    "high_1_to_3": "higher-SF 1→3×",
}
TARGET_POPULATION = {
    "low_0_to_2": "lower_sf_71",
    "high_0_to_1": "higher_sf_29",
    "high_1_to_3": "higher_sf_29",
}
TARGET_SCALE = {"low_0_to_2": 2.0, "high_0_to_1": 1.0, "high_1_to_3": 3.0}
BASELINE_SCALE = {"low_0_to_2": 0.0, "high_0_to_1": 0.0, "high_1_to_3": 1.0}
TARGET_UNITS = {"low_0_to_2": 71, "high_0_to_1": 29, "high_1_to_3": 29}

LOW_COLOR = "#2878B5"
HIGH_COLOR = "#D55E00"
HIGH_REVERSAL_COLOR = "#9E3F00"
Q_COLOR = "#D1D5D8"
TEXT_COLOR = "#202428"
MUTED_COLOR = "#636A70"
GRID_COLOR = "#D8DCDF"
CONTRAST_COLOR = {
    "low_0_to_2": LOW_COLOR,
    "high_0_to_1": HIGH_COLOR,
    "high_1_to_3": HIGH_REVERSAL_COLOR,
}
CONTRAST_MARKER = {"low_0_to_2": "o", "high_0_to_1": "s", "high_1_to_3": "^"}

REQUIRED_CANONICAL = (
    "native_channel_leverage.csv",
    "pq_variance_decomposition.csv",
    "pq_readout_decomposition.csv",
    "per_unit_pq_reliance.csv",
)
EXPECTED_CELLS = {(fold, contrast) for fold in FOLDS for contrast in CONTRAST_ORDER}


class DataUnavailable(RuntimeError):
    """Raised rather than plotting partial, stale, or invented data."""


@dataclass(frozen=True)
class FinalizedInputs:
    root: Path
    pq_manifest: dict[str, Any]
    rank8_gate: dict[str, Any]
    rank8_manifest: dict[str, Any]
    rank8_inventory: dict[str, Any]
    leverage: pd.DataFrame
    variance: pd.DataFrame
    readout: pd.DataFrame
    per_unit: pd.DataFrame
    source_paths: tuple[Path, ...]
    source_sha256: tuple[tuple[Path, str], ...]


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataUnavailable(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataUnavailable(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise DataUnavailable(f"{label} must be a JSON object: {path}")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise DataUnavailable(f"{label} is missing or empty: {path}")
    try:
        value = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataUnavailable(f"{label} is unreadable: {path}") from error
    if value.empty:
        raise DataUnavailable(f"{label} has no rows: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataUnavailable(f"{label} lacks required columns: {', '.join(missing)}")


def _validate_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.is_file():
        raise DataUnavailable(f"{label} is missing: {path}")
    if not expected or sha256_file(path) != str(expected):
        raise DataUnavailable(f"{label} hash does not match finalized provenance: {path}")


def _canonical_path(root: Path, name: str, record: dict[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    expected = (root / name).resolve()
    if path != expected:
        raise DataUnavailable(f"Finalized {name} points outside the requested input root: {path}")
    if not bool(record.get("exists", False)):
        raise DataUnavailable(f"Finalized manifest marks {name} unavailable")
    _validate_hash(path, record.get("sha256"), name)
    return path


def _validate_stage_markers(
    root: Path,
    basis_inventory: dict[tuple[int, str], tuple[Path, str]],
) -> list[Path]:
    paths: list[Path] = []
    fingerprints: set[str] = set()
    expected_names = {
        "variance": {
            "native_channel_leverage.csv",
            "pq_variance_decomposition.csv",
            "variance_supporting_arrays.npz",
        },
        "readout": {
            "pq_readout_decomposition.csv",
            "per_unit_pq_reliance.csv",
            "pq_readout_supporting_arrays.npz",
            "readout_numerical_diagnostics.json",
        },
        "probes": {"pq_descriptive_probes.csv", "probe_feature_cache_reference.json"},
    }
    for fold, contrast in sorted(EXPECTED_CELLS):
        for stage, required in expected_names.items():
            marker_path = root / "pq_intermediate" / f"fold_{fold}" / contrast / stage / "complete.json"
            marker = _read_json(marker_path, f"{stage} completion marker")
            if marker.get("schema_version") != "fig4-registration-pq-semantics-v1":
                raise DataUnavailable(f"Unexpected semantic marker schema: {marker_path}")
            if (
                marker.get("stage") != stage
                or marker.get("complete") is not True
                or marker.get("fold") != fold
                or str(marker.get("contrast", "")) != contrast
            ):
                raise DataUnavailable(f"Incomplete or mismatched semantic marker: {marker_path}")
            fingerprint = marker.get("input_fingerprint")
            if not isinstance(fingerprint, dict) or not fingerprint:
                raise DataUnavailable(f"Semantic marker lacks an input fingerprint: {marker_path}")
            fingerprints.add(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")))
            expected_basis_path, expected_basis_hash = basis_inventory[(fold, contrast)]
            marker_basis = Path(str(marker.get("basis_path", ""))).resolve()
            if (
                marker_basis != expected_basis_path
                or str(marker.get("basis_sha256", "")) != expected_basis_hash
            ):
                raise DataUnavailable(f"Semantic marker does not use the finalized fold basis: {marker_path}")
            products = marker.get("products")
            if not isinstance(products, list) or not products:
                raise DataUnavailable(f"Semantic marker lists no products: {marker_path}")
            names = {Path(str(item.get("path", ""))).name for item in products if isinstance(item, dict)}
            if not required.issubset(names):
                raise DataUnavailable(f"Semantic marker lacks required {stage} products: {marker_path}")
            for product in products:
                if not isinstance(product, dict):
                    raise DataUnavailable(f"Malformed product in semantic marker: {marker_path}")
                product_path = Path(str(product.get("path", ""))).resolve()
                try:
                    product_path.relative_to(root.resolve())
                except ValueError as error:
                    raise DataUnavailable(f"Marker product escapes input root: {product_path}") from error
                _validate_hash(product_path, product.get("sha256"), f"{stage} marker product")
                if product_path.name == "readout_numerical_diagnostics.json":
                    diagnostic = _read_json(product_path, "readout numerical diagnostics")
                    if diagnostic.get("passed") is not True:
                        raise DataUnavailable(f"Readout numerical diagnostics did not pass: {product_path}")
                paths.append(product_path)
            paths.append(marker_path)
    if len(fingerprints) != 1:
        raise DataUnavailable("Finalized P/Q stages do not share one input/cache/code fingerprint")
    return paths


def _validate_supporting_arrays(
    root: Path,
    expected_count: int,
    expected_sources: set[Path],
) -> list[Path]:
    manifest_path = root / "pq_supporting_arrays" / "manifest.json"
    manifest = _read_json(manifest_path, "P/Q supporting-array manifest")
    if manifest.get("schema_version") != "fig4-registration-pq-semantics-v1":
        raise DataUnavailable("Unexpected P/Q supporting-array manifest schema")
    products = manifest.get("products")
    if not isinstance(products, list) or len(products) != int(expected_count):
        raise DataUnavailable(
            f"Expected {expected_count} finalized supporting arrays, found "
            f"{len(products) if isinstance(products, list) else 'malformed'}"
        )
    paths = [manifest_path]
    observed_sources: set[Path] = set()
    for record in products:
        if not isinstance(record, dict):
            raise DataUnavailable("Malformed supporting-array record")
        path = Path(str(record.get("path", ""))).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise DataUnavailable(f"Supporting array escapes input root: {path}") from error
        _validate_hash(path, record.get("sha256"), "P/Q supporting array")
        source = Path(str(record.get("source", ""))).resolve()
        if source not in expected_sources or source in observed_sources:
            raise DataUnavailable(f"Supporting array has a stale, duplicate, or unvalidated source: {source}")
        _validate_hash(source, record.get("sha256"), "P/Q supporting-array source")
        observed_sources.add(source)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise DataUnavailable(f"Supporting array contains no arrays: {path}")
        except (OSError, ValueError) as error:
            raise DataUnavailable(f"Supporting array is unreadable without pickle: {path}") from error
        paths.append(path)
    if observed_sources != expected_sources:
        raise DataUnavailable("Supporting-array manifest does not mirror all validated fold-stage arrays")
    return paths


def _validate_rank8(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Path]]:
    gate_path = root / "rank8_validation_gate.json"
    manifest_path = root / "rank8_validation" / "analysis_manifest.json"
    inventory_path = root / "rank8_validation" / "rank8_projector_inventory.json"
    fold_results_path = root / "fold_rank8_results.csv"
    gate = _read_json(gate_path, "rank-8 validation gate")
    manifest = _read_json(manifest_path, "rank-8 validation manifest")
    inventory = _read_json(inventory_path, "rank-8 projector inventory")
    if gate.get("status") != "complete" or gate.get("rank_fixed_prospectively") != 8:
        raise DataUnavailable("Rank-8 validation is not finalized at prospectively fixed rank 8")
    if gate.get("stop_downstream_mechanism_audit") is not False:
        raise DataUnavailable("Rank-8 gate stops downstream mechanism analysis")
    heldout = gate.get("heldout_generalization")
    if not isinstance(heldout, dict) or set(heldout) != set(CONTRAST_ORDER):
        raise DataUnavailable("Rank-8 held-out generalization gates are incomplete")
    if not all(bool(heldout[key].get("complete")) and bool(heldout[key].get("generalizes")) for key in CONTRAST_ORDER):
        raise DataUnavailable("Not all rank-8 contrasts pass crossed held-out generalization")
    if manifest.get("schema_version") != "fig4-registration-rank8-validation-v1":
        raise DataUnavailable("Unexpected rank-8 validation manifest schema")
    if (
        int(manifest.get("rank", -1)) != 8
        or set(manifest.get("folds", ())) != set(FOLDS)
        or set(manifest.get("contrasts", ())) != set(CONTRAST_ORDER)
        or manifest.get("uses_saved_cache_only_for_evaluation") is not True
        or manifest.get("consensus_is_visualization_only") is not True
    ):
        raise DataUnavailable("Rank-8 validation manifest is incomplete or does not certify saved-product use")
    if inventory.get("schema_version") != "fig4-registration-rank8-projector-inventory-v1":
        raise DataUnavailable("Unexpected rank-8 inventory schema")
    if int(inventory.get("rank", -1)) != 8:
        raise DataUnavailable("Rank-8 inventory does not record rank 8")
    if inventory.get("consensus_for_visualization_only") is not True:
        raise DataUnavailable("Rank-8 inventory does not mark consensus as visualization-only")
    projectors = inventory.get("fold_projectors")
    cells = {
        (int(value.get("fold", -1)), str(value.get("contrast", "")))
        for value in projectors
        if isinstance(value, dict)
    } if isinstance(projectors, list) else set()
    if not isinstance(projectors, list) or len(projectors) != len(EXPECTED_CELLS) or cells != EXPECTED_CELLS:
        raise DataUnavailable("Rank-8 projector inventory does not contain all 12 fold-wise projectors")
    for record in projectors:
        if int(record.get("rank", -1)) != 8:
            raise DataUnavailable("Rank-8 projector inventory contains a non-rank-8 record")
        for path_key, hash_key, label in (
            ("basis_path", "basis_sha256", "fold-wise rank-8 basis"),
            ("projector_path", "projector_sha256", "fold-wise rank-8 projector"),
            (
                "heldout_predictions_path",
                "heldout_predictions_sha256",
                "fold-wise rank-8 held-out predictions",
            ),
        ):
            path = Path(str(record.get(path_key, ""))).resolve()
            digest = str(record.get(hash_key, ""))
            if (
                not str(record.get(path_key, ""))
                or path.suffix != (".npz" if path_key == "heldout_predictions_path" else ".npy")
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                raise DataUnavailable(f"{label} provenance is malformed in the rank-8 inventory")
    fold_results = _read_csv(fold_results_path, "fold-wise rank-8 results")
    _require_columns(fold_results, ("fold", "contrast", "method", "complete_map_recovery_mean"), "rank-8 results")
    observed = {
        (int(row.fold), str(row.contrast), str(row.method))
        for row in fold_results.itertuples()
    }
    expected = {
        (fold, contrast, method)
        for fold, contrast in EXPECTED_CELLS
        for method in ("learned", "readout_svd")
    }
    if len(fold_results) != len(expected) or observed != expected:
        raise DataUnavailable("Fold-wise rank-8 table is not exactly 4 folds × 3 contrasts × 2 methods")
    return gate, manifest, inventory, [
        gate_path,
        manifest_path,
        inventory_path,
        fold_results_path,
    ]


def load_finalized_inputs(root: Path) -> FinalizedInputs:
    root = Path(root).resolve()
    pq_manifest_path = root / "pq_semantics_manifest.json"
    pq_manifest = _read_json(pq_manifest_path, "P/Q semantic manifest")
    if pq_manifest.get("schema_version") != "fig4-registration-pq-semantics-v1":
        raise DataUnavailable("Unexpected P/Q semantic manifest schema")
    if pq_manifest.get("status") != "complete" or pq_manifest.get("complete_all_four_folds") is not True:
        raise DataUnavailable("P/Q semantic analysis is not complete across all four folds")
    for field in (
        "missing_basis_fold_contrasts",
        "missing_variance_fold_contrasts",
        "missing_readout_fold_contrasts",
        "missing_probe_fold_contrasts",
    ):
        if pq_manifest.get(field) not in ([], None):
            raise DataUnavailable(f"P/Q semantic manifest reports missing products in {field}")
    if pq_manifest.get("no_core_replay") is not True or pq_manifest.get("model_imported") is not False:
        raise DataUnavailable("P/Q manifest does not certify saved-product-only analysis")

    canonical = pq_manifest.get("canonical_products")
    if not isinstance(canonical, dict):
        raise DataUnavailable("P/Q manifest lacks canonical products")
    paths = {
        name: _canonical_path(root, name, canonical.get(name, {})) for name in REQUIRED_CANONICAL
    }
    leverage = _read_csv(paths["native_channel_leverage.csv"], "native-channel leverage")
    variance = _read_csv(paths["pq_variance_decomposition.csv"], "P/Q variance decomposition")
    readout = _read_csv(paths["pq_readout_decomposition.csv"], "P/Q readout decomposition")
    per_unit = _read_csv(paths["per_unit_pq_reliance.csv"], "per-unit P/Q reliance")
    for name, table in (
        ("native_channel_leverage.csv", leverage),
        ("pq_variance_decomposition.csv", variance),
        ("pq_readout_decomposition.csv", readout),
        ("per_unit_pq_reliance.csv", per_unit),
    ):
        if int(canonical[name].get("rows", -1)) != len(table):
            raise DataUnavailable(f"{name} row count differs from finalized manifest")

    gate, rank8_manifest, inventory, rank8_paths = _validate_rank8(root)
    basis_inventory = {
        (int(record["fold"]), str(record["contrast"])): (
            Path(str(record["basis_path"])).resolve(),
            str(record["basis_sha256"]),
        )
        for record in inventory["fold_projectors"]
    }
    marker_paths = _validate_stage_markers(root, basis_inventory)
    expected_arrays = int(pq_manifest.get("n_supporting_array_products", -1))
    if expected_arrays != 24:
        raise DataUnavailable(f"Expected 24 fold-wise P/Q supporting arrays, manifest reports {expected_arrays}")
    expected_array_sources = {
        path.resolve()
        for path in marker_paths
        if path.name in ("variance_supporting_arrays.npz", "pq_readout_supporting_arrays.npz")
    }
    if len(expected_array_sources) != expected_arrays:
        raise DataUnavailable("Validated semantic markers do not contain exactly 24 supporting arrays")
    array_paths = _validate_supporting_arrays(root, expected_arrays, expected_array_sources)
    source_paths = tuple(
        sorted(
            {pq_manifest_path, *paths.values(), *marker_paths, *array_paths, *rank8_paths},
            key=lambda value: str(value),
        )
    )
    source_sha256 = tuple((path, sha256_file(path)) for path in source_paths)
    return FinalizedInputs(
        root=root,
        pq_manifest=pq_manifest,
        rank8_gate=gate,
        rank8_manifest=rank8_manifest,
        rank8_inventory=inventory,
        leverage=leverage,
        variance=variance,
        readout=readout,
        per_unit=per_unit,
        source_paths=source_paths,
        source_sha256=source_sha256,
    )


def _validate_source_snapshot(inputs: FinalizedInputs) -> None:
    for path, expected in inputs.source_sha256:
        if not path.is_file() or sha256_file(path) != expected:
            raise DataUnavailable(f"A finalized source changed while plotting: {path}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].to_numpy(float)).all():
            raise DataUnavailable(f"{label} contains nonfinite {column}")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise DataUnavailable(f"{label} is not an explicit boolean: {value!r}")


def _require_exact_cells(frame: pd.DataFrame, label: str) -> None:
    try:
        observed = {
            (int(row.fold), str(row.contrast))
            for row in frame[["fold", "contrast"]].drop_duplicates().itertuples()
        }
    except (TypeError, ValueError, OverflowError) as error:
        raise DataUnavailable(f"{label} contains malformed fold/contrast identifiers") from error
    if observed != EXPECTED_CELLS:
        raise DataUnavailable(f"{label} does not cover exactly all 12 fold-by-contrast cells")


def _fold_bootstrap(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (len(FOLDS),) or not np.isfinite(data).all():
        raise DataUnavailable(f"Fold summary requires exactly four finite values, got {data}")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(data), size=(int(draws), len(data)))
    distribution = data[indices].mean(axis=1)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def prepare_panel_a(
    leverage: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (
        "fold",
        "contrast",
        "native_channel",
        "leverage_rank_descending",
        "leverage_score_p_cc",
        "cumulative_leverage_fraction_at_channel_rank",
        "effective_participating_channel_count",
    )
    _require_columns(leverage, required, "native-channel leverage")
    table = _numeric(
        leverage,
        (
            "fold",
            "native_channel",
            "leverage_rank_descending",
            "leverage_score_p_cc",
            "cumulative_leverage_fraction_at_channel_rank",
            "effective_participating_channel_count",
        ),
        "native-channel leverage",
    )
    _require_exact_cells(table, "native-channel leverage")
    raw_rows: list[dict[str, Any]] = []
    for fold, contrast in sorted(EXPECTED_CELLS):
        subset = table.loc[table.fold.eq(fold) & table.contrast.eq(contrast)].copy()
        if len(subset) != 128 or subset.native_channel.nunique() != 128:
            raise DataUnavailable(f"Leverage cell {fold}/{contrast} does not contain 128 native channels")
        subset = subset.sort_values("leverage_rank_descending")
        if not np.array_equal(subset.leverage_rank_descending.to_numpy(int), np.arange(1, 129)):
            raise DataUnavailable(f"Leverage ranks are not 1..128 in {fold}/{contrast}")
        if set(subset.native_channel.astype(int)) != set(range(128)):
            raise DataUnavailable(f"Native-channel indices are not exactly 0..127 in {fold}/{contrast}")
        leverage_values = subset.leverage_score_p_cc.to_numpy(float)
        cumulative_values = subset.cumulative_leverage_fraction_at_channel_rank.to_numpy(float)
        if (
            (leverage_values < -1e-8).any()
            or (leverage_values > 1.0 + 1e-6).any()
            or not np.all(leverage_values[:-1] >= leverage_values[1:] - 1e-12)
            or not np.isclose(leverage_values.sum(), 8.0, atol=2e-5)
            or not np.allclose(cumulative_values, np.cumsum(leverage_values) / 8.0, atol=2e-6)
            or not np.isclose(cumulative_values[-1], 1.0, atol=2e-6)
        ):
            raise DataUnavailable(f"Leverage/projector invariants fail in {fold}/{contrast}")
        neff = subset.effective_participating_channel_count.to_numpy(float)
        if not np.allclose(neff, neff[0], atol=1e-9, rtol=0):
            raise DataUnavailable(f"N_eff is not constant within {fold}/{contrast}")
        if not (8.0 - 1e-5 <= neff[0] <= 128.0 + 1e-5):
            raise DataUnavailable(f"N_eff lies outside the rank-8 projector bounds in {fold}/{contrast}")
        expected_neff = float(np.square(leverage_values.sum()) / np.square(leverage_values).sum())
        if not np.isclose(neff[0], expected_neff, atol=2e-5):
            raise DataUnavailable(f"N_eff does not match the leverage vector in {fold}/{contrast}")
        for row in subset.itertuples():
            raw_rows.append(
                {
                    "row_type": "fold",
                    "fold": fold,
                    "contrast": contrast,
                    "channel_rank": int(row.leverage_rank_descending),
                    "native_channel": int(row.native_channel),
                    "leverage_score": float(row.leverage_score_p_cc),
                    "cumulative_leverage_fraction": float(
                        row.cumulative_leverage_fraction_at_channel_rank
                    ),
                    "effective_participating_channels": float(neff[0]),
                }
            )
    raw = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, Any]] = []
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        for channel_rank in range(1, 129):
            subset = raw.loc[
                raw.contrast.eq(contrast) & raw.channel_rank.eq(channel_rank)
            ].sort_values("fold")
            if tuple(subset.fold.astype(int)) != FOLDS:
                raise DataUnavailable(f"Panel A fold aggregation is incomplete for {contrast} rank {channel_rank}")
            row: dict[str, Any] = {"contrast": contrast, "channel_rank": channel_rank}
            for metric_i, metric in enumerate(("leverage_score", "cumulative_leverage_fraction")):
                mean, low, high = _fold_bootstrap(
                    subset[metric].to_numpy(float),
                    draws=draws,
                    seed=seed + contrast_i * 1000 + channel_rank * 3 + metric_i,
                )
                row[f"{metric}_fold_mean"] = mean
                row[f"{metric}_fold_bootstrap_ci_low"] = low
                row[f"{metric}_fold_bootstrap_ci_high"] = high
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    neff_rows: list[dict[str, Any]] = []
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        values = (
            raw.loc[raw.contrast.eq(contrast), ["fold", "effective_participating_channels"]]
            .drop_duplicates()
            .sort_values("fold")
        )
        mean, low, high = _fold_bootstrap(
            values.effective_participating_channels.to_numpy(float),
            draws=draws,
            seed=seed + 9000 + contrast_i,
        )
        neff_rows.append(
            {
                "contrast": contrast,
                "effective_participating_channels_fold_mean": mean,
                "effective_participating_channels_fold_bootstrap_ci_low": low,
                "effective_participating_channels_fold_bootstrap_ci_high": high,
                "normalized_distribution_index_0_sparse_1_uniform": (mean - 8.0) / 120.0,
                "sparse_coordinate_bound": 8.0,
                "uniform_distribution_bound": 128.0,
            }
        )
    return pd.concat([raw, summary.assign(row_type="fold_summary")], ignore_index=True, sort=False), pd.DataFrame(neff_rows)


def prepare_panel_b(
    variance: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    required = (
        "fold",
        "contrast",
        "analysis",
        "scale",
        "candidate_p_total_fraction",
        "complementary_q_total_fraction",
        "candidate_p_energy_per_dimension",
        "complementary_q_energy_per_dimension",
        "p_to_q_per_dimension_energy_ratio",
    )
    _require_columns(variance, required, "P/Q variance decomposition")
    table = _numeric(
        variance,
        (
            "fold",
            "scale",
            "candidate_p_total_fraction",
            "complementary_q_total_fraction",
            "candidate_p_energy_per_dimension",
            "complementary_q_energy_per_dimension",
            "p_to_q_per_dimension_energy_ratio",
        ),
        "P/Q variance decomposition",
    )
    _require_exact_cells(table, "P/Q variance decomposition")
    rows: list[dict[str, Any]] = []
    for fold, contrast in sorted(EXPECTED_CELLS):
        cell = table.loc[table.fold.eq(fold) & table.contrast.eq(contrast)]
        definitions = (
            (
                "content",
                cell.loc[
                    cell.analysis.eq("stabilized_visual_content_image_means")
                    & np.isclose(cell.scale, 0.0)
                ],
            ),
            (
                "motion",
                cell.loc[
                    cell.analysis.eq("movement_change_from_stabilization")
                    & np.isclose(cell.scale, TARGET_SCALE[contrast])
                ],
            ),
        )
        for variation, selected in definitions:
            if len(selected) != 1:
                raise DataUnavailable(
                    f"Expected one {variation} variance row for fold {fold}/{contrast}, found {len(selected)}"
                )
            source = selected.iloc[0]
            p_fraction = float(source.candidate_p_total_fraction)
            q_fraction = float(source.complementary_q_total_fraction)
            if not np.isclose(p_fraction + q_fraction, 1.0, atol=2e-5):
                raise DataUnavailable(f"P/Q total fractions do not sum to one in {fold}/{contrast}/{variation}")
            p_per_dimension = float(source.candidate_p_energy_per_dimension)
            q_per_dimension = float(source.complementary_q_energy_per_dimension)
            ratio = float(source.p_to_q_per_dimension_energy_ratio)
            if not (-1e-8 <= p_fraction <= 1.0 + 1e-8 and -1e-8 <= q_fraction <= 1.0 + 1e-8):
                raise DataUnavailable(f"P/Q total fractions leave [0,1] in {fold}/{contrast}/{variation}")
            if p_per_dimension < 0 or q_per_dimension <= 0 or ratio <= 0:
                raise DataUnavailable(f"Nonpositive per-dimension energy ratio in {fold}/{contrast}/{variation}")
            if not np.isclose(ratio, p_per_dimension / q_per_dimension, rtol=2e-5, atol=1e-8):
                raise DataUnavailable(f"Per-dimension P/Q ratio is inconsistent in {fold}/{contrast}/{variation}")
            total_from_per_dimension = 8.0 * p_per_dimension + 120.0 * q_per_dimension
            implied_p_fraction = 8.0 * p_per_dimension / total_from_per_dimension
            if not np.isclose(p_fraction, implied_p_fraction, rtol=2e-5, atol=1e-8):
                raise DataUnavailable(
                    f"Total fraction and per-dimension energies are inconsistent in "
                    f"{fold}/{contrast}/{variation}"
                )
            rows.append(
                {
                    "row_type": "fold",
                    "fold": fold,
                    "contrast": contrast,
                    "variation": variation,
                    "scale": float(source.scale),
                    "candidate_p_total_fraction": p_fraction,
                    "complementary_q_total_fraction": q_fraction,
                    "candidate_p_energy_per_dimension": p_per_dimension,
                    "complementary_q_energy_per_dimension": q_per_dimension,
                    "p_to_q_per_dimension_energy_ratio": ratio,
                }
            )
    raw = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    metrics = (
        "candidate_p_total_fraction",
        "complementary_q_total_fraction",
        "candidate_p_energy_per_dimension",
        "complementary_q_energy_per_dimension",
        "p_to_q_per_dimension_energy_ratio",
    )
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        for variation_i, variation in enumerate(("content", "motion")):
            subset = raw.loc[
                raw.contrast.eq(contrast) & raw.variation.eq(variation)
            ].sort_values("fold")
            if tuple(subset.fold.astype(int)) != FOLDS:
                raise DataUnavailable(f"Panel B fold aggregation is incomplete for {contrast}/{variation}")
            row = {
                "row_type": "fold_summary",
                "contrast": contrast,
                "variation": variation,
                "scale": float(subset.scale.iloc[0]),
            }
            for metric_i, metric in enumerate(metrics):
                mean, low, high = _fold_bootstrap(
                    subset[metric].to_numpy(float),
                    draws=draws,
                    seed=seed + contrast_i * 100 + variation_i * 10 + metric_i,
                )
                row[f"{metric}_fold_mean"] = mean
                row[f"{metric}_fold_bootstrap_ci_low"] = low
                row[f"{metric}_fold_bootstrap_ci_high"] = high
            summary_rows.append(row)
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


def prepare_panel_c(
    readout: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    required = (
        "fold",
        "contrast",
        "analysis",
        "scale",
        "scale_a",
        "scale_b",
        "population",
        "n_units",
        "component",
        "primary_map_recovery_weighting",
        "normalized_map_recovery_vs_training_mean_r2",
        "normalized_map_movement_effect_recovery_r2",
    )
    _require_columns(readout, required, "P/Q readout decomposition")
    table = readout.copy()
    table["fold"] = pd.to_numeric(table.fold, errors="coerce")
    _require_exact_cells(table, "P/Q readout decomposition")
    rows: list[dict[str, Any]] = []
    definitions = {
        "baseline visual map": (
            "baseline_visual_reconstruction",
            {
                "candidate P": "candidate_p_content",
                "complementary Q": "complementary_q_content",
            },
            "normalized_map_recovery_vs_training_mean_r2",
        ),
        "movement map": (
            "defining_contrast_movement_effect_decomposition",
            {
                "candidate P": "candidate_p_only_contrast",
                "complementary Q": "complementary_q_only_contrast",
            },
            "normalized_map_movement_effect_recovery_r2",
        ),
    }
    for fold, contrast in sorted(EXPECTED_CELLS):
        population = TARGET_POPULATION[contrast]
        for transformation, (analysis, component_map, metric) in definitions.items():
            for component_label, component in component_map.items():
                selected = table.loc[
                    table.fold.eq(fold)
                    & table.contrast.eq(contrast)
                    & table.analysis.eq(analysis)
                    & table.population.eq(population)
                    & table.component.eq(component)
                ]
                if len(selected) != 1:
                    raise DataUnavailable(
                        f"Expected one Panel C row for {fold}/{contrast}/{transformation}/{component}, "
                        f"found {len(selected)}"
                    )
                value = pd.to_numeric(selected.iloc[0][metric], errors="coerce")
                if not np.isfinite(value):
                    raise DataUnavailable(f"Nonfinite Panel C recovery for {fold}/{contrast}/{component}")
                source = selected.iloc[0]
                n_units = pd.to_numeric(source.n_units, errors="coerce")
                if (
                    not np.isfinite(n_units)
                    or not float(n_units).is_integer()
                    or int(n_units) != TARGET_UNITS[contrast]
                ):
                    raise DataUnavailable(f"Panel C uses the wrong RR100 population size in {fold}/{contrast}")
                if str(source.primary_map_recovery_weighting) != "paired expected spikes":
                    raise DataUnavailable(f"Panel C recovery is not paired-expected-spike weighted")
                source_scale = pd.to_numeric(source.scale, errors="coerce")
                expected_source_scale = 0.0 if transformation == "baseline visual map" else TARGET_SCALE[contrast]
                if not np.isfinite(source_scale) or not np.isclose(source_scale, expected_source_scale):
                    raise DataUnavailable(f"Panel C uses the wrong endpoint scale in {fold}/{contrast}")
                if transformation == "movement map":
                    scale_a = pd.to_numeric(source.scale_a, errors="coerce")
                    scale_b = pd.to_numeric(source.scale_b, errors="coerce")
                    if (
                        not np.isfinite(scale_a)
                        or not np.isfinite(scale_b)
                        or not np.isclose(scale_a, BASELINE_SCALE[contrast])
                        or not np.isclose(scale_b, TARGET_SCALE[contrast])
                    ):
                        raise DataUnavailable(f"Panel C does not use the literal defining contrast in {fold}/{contrast}")
                rows.append(
                    {
                        "row_type": "fold",
                        "fold": fold,
                        "contrast": contrast,
                        "population": population,
                        "n_units": int(n_units),
                        "transformation": transformation,
                        "component": component_label,
                        "complete_normalized_map_recovery_r2": float(value),
                        "recovery_values_are_unclipped": True,
                        "primary_weighting": "paired expected spikes",
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        for transformation_i, transformation in enumerate(definitions):
            for component_i, component in enumerate(("candidate P", "complementary Q")):
                subset = raw.loc[
                    raw.contrast.eq(contrast)
                    & raw.transformation.eq(transformation)
                    & raw.component.eq(component)
                ].sort_values("fold")
                if tuple(subset.fold.astype(int)) != FOLDS:
                    raise DataUnavailable(f"Panel C fold aggregation is incomplete for {contrast}")
                mean, low, high = _fold_bootstrap(
                    subset.complete_normalized_map_recovery_r2.to_numpy(float),
                    draws=draws,
                    seed=seed + contrast_i * 100 + transformation_i * 10 + component_i,
                )
                summary_rows.append(
                    {
                        "row_type": "fold_summary",
                        "contrast": contrast,
                        "population": TARGET_POPULATION[contrast],
                        "n_units": TARGET_UNITS[contrast],
                        "transformation": transformation,
                        "component": component,
                        "complete_normalized_map_recovery_r2_fold_mean": mean,
                        "complete_normalized_map_recovery_r2_fold_bootstrap_ci_low": low,
                        "complete_normalized_map_recovery_r2_fold_bootstrap_ci_high": high,
                        "recovery_values_are_unclipped": True,
                        "primary_weighting": "paired expected spikes",
                    }
                )
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)


def _unit_bootstrap_correlation(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    table = frame[["unit_index", x, y]].copy()
    table[x] = pd.to_numeric(table[x], errors="coerce")
    table[y] = pd.to_numeric(table[y], errors="coerce")
    table = table.loc[np.isfinite(table[x]) & np.isfinite(table[y])]
    if len(table) < 4 or table.unit_index.nunique() != len(table):
        raise DataUnavailable(f"Per-unit association {x}→{y} lacks unique finite units")
    x_values = table[x].to_numpy(float)
    y_values = table[y].to_numpy(float)
    observed = float(spearmanr(x_values, y_values).statistic)
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        index = rng.integers(0, len(table), size=len(table))
        sampled_x = x_values[index]
        sampled_y = y_values[index]
        if np.ptp(sampled_x) <= 0 or np.ptp(sampled_y) <= 0:
            distribution[draw] = np.nan
        else:
            distribution[draw] = spearmanr(sampled_x, sampled_y).statistic
    finite = distribution[np.isfinite(distribution)]
    if len(finite) < 0.95 * int(draws):
        raise DataUnavailable(f"Too many degenerate unit-bootstrap draws for {x}→{y}")
    low, high = np.percentile(finite, [2.5, 97.5])
    return observed, float(low), float(high)


def prepare_panel_d(
    per_unit: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (
        "fold",
        "contrast",
        "unit_index",
        "historical_sf_population",
        "sf_split_metric",
        "weight_based_candidate_p_reliance",
        "activity_activity_weighted_p_reliance_excluding_covariance",
        "observed_ssi_benefit_for_projector_contrast_bits",
        "observed_ssi_1_to_3_change_bits",
        "high_motion_reversal",
        "observational_ssi_scope",
    )
    _require_columns(per_unit, required, "per-unit P/Q reliance")
    table = _numeric(
        per_unit,
        (
            "fold",
            "unit_index",
            "sf_split_metric",
            "weight_based_candidate_p_reliance",
            "activity_activity_weighted_p_reliance_excluding_covariance",
            "observed_ssi_benefit_for_projector_contrast_bits",
            "observed_ssi_1_to_3_change_bits",
        ),
        "per-unit P/Q reliance",
    )
    _require_exact_cells(table, "per-unit P/Q reliance")
    rows: list[dict[str, Any]] = []
    for fold, contrast in sorted(EXPECTED_CELLS):
        expected_group = TARGET_POPULATION[contrast]
        subset = table.loc[table.fold.eq(fold) & table.contrast.eq(contrast)].copy()
        subset = subset.loc[subset.historical_sf_population.eq(expected_group)]
        if len(subset) != TARGET_UNITS[contrast] or subset.unit_index.nunique() != len(subset):
            raise DataUnavailable(
                f"Per-unit cell {fold}/{contrast} has {len(subset)} target units, "
                f"expected {TARGET_UNITS[contrast]}"
            )
        expected_scope = f"fold_{fold}_crossed_test_block_only_2_images_x_6_trajectories_x_40_frames"
        if set(subset.observational_ssi_scope.astype(str)) != {expected_scope}:
            raise DataUnavailable(f"Per-unit SSI scope is not fold-held-out for {fold}/{contrast}")
        for row in subset.itertuples():
            weight = float(row.weight_based_candidate_p_reliance)
            activity = float(row.activity_activity_weighted_p_reliance_excluding_covariance)
            if not (0 <= weight <= 1) or not (0 <= activity <= 1):
                raise DataUnavailable(f"Per-unit P reliance is outside [0,1] in {fold}/{contrast}")
            reversal_change = float(row.observed_ssi_1_to_3_change_bits)
            reversal_flag = _strict_bool(row.high_motion_reversal, "high_motion_reversal")
            if reversal_flag != (reversal_change < 0):
                raise DataUnavailable(f"High-motion reversal flag disagrees with held-out SSI in {fold}/{contrast}")
            rows.append(
                {
                    "fold": fold,
                    "contrast": contrast,
                    "unit_index": int(row.unit_index),
                    "historical_sf_population": expected_group,
                    "sf_split_metric": float(row.sf_split_metric),
                    "weight_based_candidate_p_reliance": weight,
                    "activity_weighted_candidate_p_reliance": activity,
                    "observed_ssi_benefit_for_projector_contrast_bits": float(
                        row.observed_ssi_benefit_for_projector_contrast_bits
                    ),
                    "observed_ssi_1_to_3_change_bits": reversal_change,
                    "high_motion_reversal": reversal_flag,
                    "observational_ssi_scope": expected_scope,
                }
            )
    raw = pd.DataFrame(rows)
    for (contrast, unit_index), subset in raw.groupby(["contrast", "unit_index"]):
        if len(subset) != len(FOLDS) or not np.allclose(
            subset.sf_split_metric.to_numpy(float),
            float(subset.sf_split_metric.iloc[0]),
            atol=1e-12,
            rtol=0,
        ):
            raise DataUnavailable(
                f"Historical SF metadata is not stable across folds for {contrast}/unit {unit_index}"
            )
    # Weight reliance depends only on W and P, so it is not repeated as four
    # pseudo-observations. Activity reliance and SSI covariates are fold-heldout;
    # average the four fold estimates for each RR100 unit before association.
    unit = (
        raw.groupby(
            ["contrast", "unit_index", "historical_sf_population"], as_index=False
        )
        .agg(
            sf_split_metric=("sf_split_metric", "first"),
            weight_based_candidate_p_reliance=("weight_based_candidate_p_reliance", "mean"),
            activity_weighted_candidate_p_reliance=("activity_weighted_candidate_p_reliance", "mean"),
            observed_ssi_benefit_bits=(
                "observed_ssi_benefit_for_projector_contrast_bits",
                "mean",
            ),
            observed_high_motion_change_1_to_3_bits=("observed_ssi_1_to_3_change_bits", "mean"),
            folds_aggregated=("fold", "nunique"),
        )
    )
    if not unit.folds_aggregated.eq(4).all():
        raise DataUnavailable("Panel D unit aggregation does not contain all four folds")
    associations: list[dict[str, Any]] = []
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        subset = unit.loc[unit.contrast.eq(contrast)]
        for x_i, (association, x, label) in enumerate(
            (
                ("historical spatial frequency", "sf_split_metric", "historical SF metric"),
                ("movement benefit", "observed_ssi_benefit_bits", "held-out SSI benefit (bits)"),
                (
                    "higher-motion 1×→3× change",
                    "observed_high_motion_change_1_to_3_bits",
                    "held-out ΔSSI, 1×→3× (bits)",
                ),
            )
        ):
            rho, low, high = _unit_bootstrap_correlation(
                subset,
                x=x,
                y="activity_weighted_candidate_p_reliance",
                draws=draws,
                seed=seed + contrast_i * 100 + x_i,
            )
            associations.append(
                {
                    "contrast": contrast,
                    "historical_sf_population": TARGET_POPULATION[contrast],
                    "n_units": len(subset),
                    "association": association,
                    "x_column": x,
                    "x_axis_label": label,
                    "y_column": "activity_weighted_candidate_p_reliance",
                    "spearman_rho": rho,
                    "unit_bootstrap_ci_low": low,
                    "unit_bootstrap_ci_high": high,
                    "bootstrap_draws": int(draws),
                    "fold_rule": "first average fold-heldout values within RR100 unit, then correlate across units",
                }
            )
    return unit, pd.DataFrame(associations)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "axes.unicode_minus": True,
        }
    )


def _clean_axis(axis: plt.Axes, *, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED_COLOR)
    axis.spines["bottom"].set_color(MUTED_COLOR)
    axis.tick_params(colors=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    if grid:
        axis.grid(axis=grid, color=GRID_COLOR, lw=0.55, alpha=0.75, zorder=0)


def _direct_label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    vertical_offset_points: float = 0,
) -> None:
    axis.annotate(
        text,
        (x, y),
        xytext=(4, vertical_offset_points),
        textcoords="offset points",
        ha="left",
        va="center",
        color=color,
        fontsize=7.0,
    )


def _plot_panel_a(
    axis: plt.Axes,
    inset: plt.Axes,
    curves: pd.DataFrame,
    neff: pd.DataFrame,
) -> None:
    summary = curves.loc[curves.row_type.eq("fold_summary")]
    for contrast in CONTRAST_ORDER:
        selected = summary.loc[summary.contrast.eq(contrast)].sort_values("channel_rank")
        x = selected.channel_rank.to_numpy(float)
        y = selected.leverage_score_fold_mean.to_numpy(float)
        low = selected.leverage_score_fold_bootstrap_ci_low.to_numpy(float)
        high = selected.leverage_score_fold_bootstrap_ci_high.to_numpy(float)
        color = CONTRAST_COLOR[contrast]
        axis.fill_between(x, low, high, color=color, alpha=0.12, lw=0)
        axis.plot(x, y, color=color, lw=1.45)
        cumulative = selected.cumulative_leverage_fraction_fold_mean.to_numpy(float)
        axis.plot(x, cumulative, color=color, lw=1.0, ls="--")
        label_rank = {"low_0_to_2": 76, "high_0_to_1": 90, "high_1_to_3": 104}[contrast]
        label_row = selected.loc[selected.channel_rank.eq(label_rank)].iloc[0]
        _direct_label(
            axis,
            label_rank,
            float(label_row.cumulative_leverage_fraction_fold_mean),
            CONTRAST_DIRECT[contrast],
            color,
            vertical_offset_points={"low_0_to_2": -12, "high_0_to_1": 0, "high_1_to_3": 12}[
                contrast
            ],
        )
    axis.set_xlim(1, 128)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Native channels, ordered by leverage")
    axis.set_ylabel("Leverage  ℓc  (solid)\nCumulative fraction  (dashed)")
    axis.set_title("Native-channel participation", loc="left", fontweight="bold")
    _clean_axis(axis, grid="y")

    positions = np.arange(len(CONTRAST_ORDER))
    for position, contrast in zip(positions, CONTRAST_ORDER):
        row = neff.loc[neff.contrast.eq(contrast)].iloc[0]
        mean = float(row.effective_participating_channels_fold_mean)
        low = float(row.effective_participating_channels_fold_bootstrap_ci_low)
        high = float(row.effective_participating_channels_fold_bootstrap_ci_high)
        inset.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt=CONTRAST_MARKER[contrast],
            color=CONTRAST_COLOR[contrast],
            mfc=CONTRAST_COLOR[contrast],
            mec="white",
            mew=0.5,
            ms=5.5,
            lw=0.8,
            capsize=2,
            zorder=3,
        )
    inset.axhline(8, color=MUTED_COLOR, ls=":", lw=0.8)
    inset.axhline(128, color=MUTED_COLOR, ls=":", lw=0.8)
    inset.text(-0.42, 8, "sparse coordinate bound", va="bottom", fontsize=6.4, color=MUTED_COLOR)
    inset.text(-0.42, 128, "uniform bound", va="bottom", fontsize=6.4, color=MUTED_COLOR)
    inset.set_xticks(positions, ["low", "high\nsharp.", "high\nreversal"])
    inset.set_ylim(0, 138)
    inset.set_title("Participation  Neff", loc="left", fontsize=7.0, pad=4)
    _clean_axis(inset, grid=None)


def _plot_panel_b(
    fraction_axis: plt.Axes,
    enrichment_axis: plt.Axes,
    panel: pd.DataFrame,
) -> None:
    summary = panel.loc[panel.row_type.eq("fold_summary")]
    x = np.arange(len(CONTRAST_ORDER) * 2)
    labels: list[str] = []
    p_values: list[float] = []
    p_low: list[float] = []
    p_high: list[float] = []
    q_values: list[float] = []
    ratio_values: list[float] = []
    ratio_low: list[float] = []
    ratio_high: list[float] = []
    colors: list[str] = []
    for contrast in CONTRAST_ORDER:
        for variation in ("content", "motion"):
            selected = summary.loc[
                summary.contrast.eq(contrast) & summary.variation.eq(variation)
            ]
            if len(selected) != 1:
                raise DataUnavailable(f"Missing Panel B summary {contrast}/{variation}")
            row = selected.iloc[0]
            labels.append("content" if variation == "content" else "motion")
            p_values.append(float(row.candidate_p_total_fraction_fold_mean))
            p_low.append(float(row.candidate_p_total_fraction_fold_bootstrap_ci_low))
            p_high.append(float(row.candidate_p_total_fraction_fold_bootstrap_ci_high))
            q_values.append(float(row.complementary_q_total_fraction_fold_mean))
            ratio_values.append(float(row.p_to_q_per_dimension_energy_ratio_fold_mean))
            ratio_low.append(float(row.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_low))
            ratio_high.append(float(row.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_high))
            colors.append(CONTRAST_COLOR[contrast])
    fraction_axis.bar(x, p_values, color=colors, width=0.72, edgecolor="none", label="candidate P")
    fraction_axis.bar(
        x,
        q_values,
        bottom=p_values,
        color=Q_COLOR,
        width=0.72,
        edgecolor="none",
        label="complementary Q",
    )
    fraction_axis.errorbar(
        x,
        p_values,
        yerr=[
            np.asarray(p_values) - np.asarray(p_low),
            np.asarray(p_high) - np.asarray(p_values),
        ],
        fmt="none",
        ecolor=TEXT_COLOR,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=4,
    )
    for index, value in enumerate(p_values):
        if value > 0.075:
            fraction_axis.text(index, value / 2, "P", ha="center", va="center", fontsize=6.6)
        if q_values[index] > 0.075:
            fraction_axis.text(
                index,
                value + q_values[index] / 2,
                "Q",
                ha="center",
                va="center",
                fontsize=6.6,
                color=MUTED_COLOR,
            )
    fraction_axis.set_ylim(0, 1.03)
    fraction_axis.set_ylabel("Total-energy\nfraction")
    fraction_axis.set_xticks(x)
    fraction_axis.tick_params(axis="x", labelbottom=False)
    fraction_axis.set_title("What P and Q encode", loc="left", fontweight="bold", y=1.12)
    for center, contrast in zip((0.5, 2.5, 4.5), CONTRAST_ORDER):
        fraction_axis.text(
            center,
            1.025,
            CONTRAST_DIRECT[contrast],
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=CONTRAST_COLOR[contrast],
            clip_on=False,
        )
    _clean_axis(fraction_axis, grid="y")

    for index, (mean, low, high, color) in enumerate(zip(ratio_values, ratio_low, ratio_high, colors)):
        enrichment_axis.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o" if index % 2 == 0 else "s",
            color=color,
            mfc="white" if index % 2 == 0 else color,
            mec=color,
            ms=4.5,
            lw=0.8,
            capsize=2,
        )
    enrichment_axis.axhline(1, color=MUTED_COLOR, lw=0.8, ls=":")
    enrichment_axis.set_yscale("log")
    enrichment_axis.set_xlim(-0.55, len(x) - 0.45)
    enrichment_axis.set_xticks(x, labels, rotation=30, ha="right")
    enrichment_axis.set_ylabel("Per-dimension\nP/Q ratio")
    enrichment_axis.text(
        0.02,
        0.03,
        "(P energy / 8) ÷ (Q energy / 120)",
        transform=enrichment_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MUTED_COLOR,
    )
    enrichment_axis.text(
        len(x) - 0.55,
        1.08,
        "equal per dimension",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED_COLOR,
    )
    _clean_axis(enrichment_axis, grid="y")


def _plot_panel_c(axis: plt.Axes, panel: pd.DataFrame) -> None:
    summary = panel.loc[panel.row_type.eq("fold_summary")]
    centers = np.arange(len(CONTRAST_ORDER) * 2, dtype=float)
    width = 0.28
    labels: list[str] = []
    all_low: list[float] = []
    all_high: list[float] = []
    for contrast_i, contrast in enumerate(CONTRAST_ORDER):
        for transformation_i, transformation in enumerate(("baseline visual map", "movement map")):
            center = centers[contrast_i * 2 + transformation_i]
            labels.append("baseline" if transformation_i == 0 else "movement")
            for component_i, component in enumerate(("candidate P", "complementary Q")):
                selected = summary.loc[
                    summary.contrast.eq(contrast)
                    & summary.transformation.eq(transformation)
                    & summary.component.eq(component)
                ]
                if len(selected) != 1:
                    raise DataUnavailable(f"Missing Panel C summary {contrast}/{transformation}/{component}")
                row = selected.iloc[0]
                mean = float(row.complete_normalized_map_recovery_r2_fold_mean)
                low = float(row.complete_normalized_map_recovery_r2_fold_bootstrap_ci_low)
                high = float(row.complete_normalized_map_recovery_r2_fold_bootstrap_ci_high)
                all_low.append(low)
                all_high.append(high)
                position = center + (-width / 1.8 if component_i == 0 else width / 1.8)
                face = CONTRAST_COLOR[contrast] if component_i == 0 else Q_COLOR
                edge = CONTRAST_COLOR[contrast] if component_i == 0 else MUTED_COLOR
                axis.bar(position, mean, width=width, color=face, edgecolor=edge, lw=0.65, zorder=2)
                axis.errorbar(
                    position,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    color=TEXT_COLOR,
                    lw=0.7,
                    capsize=2,
                    zorder=3,
                )
                axis.text(
                    position,
                    mean + (0.035 if mean >= 0 else -0.035),
                    "P" if component_i == 0 else "Q",
                    ha="center",
                    va="bottom" if mean >= 0 else "top",
                    fontsize=6.4,
                    color=edge,
                )
    axis.axhline(0, color=MUTED_COLOR, lw=0.75)
    lower = min(-0.08, min(all_low) - 0.08)
    upper = max(1.05, max(all_high) + 0.08)
    axis.set_ylim(lower, upper)
    axis.set_xticks(centers, labels, rotation=30, ha="right")
    axis.set_ylabel("Complete normalized-map recovery  R²")
    axis.set_title("Exact RR100 consequences", loc="left", fontweight="bold", y=1.12)
    for center, contrast in zip((0.5, 2.5, 4.5), CONTRAST_ORDER):
        axis.text(
            center,
            upper + 0.02 * (upper - lower),
            CONTRAST_DIRECT[contrast],
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=CONTRAST_COLOR[contrast],
            clip_on=False,
        )
    axis.text(
        centers[-1] + 0.52,
        lower + 0.015 * (upper - lower),
        "negative values retained",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=MUTED_COLOR,
    )
    _clean_axis(axis, grid="y")


def _plot_panel_d(
    axes: Sequence[plt.Axes],
    unit: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    plots = (
        ("sf_split_metric", "Historical SF metric", "historical spatial frequency"),
        ("observed_ssi_benefit_bits", "Held-out SSI benefit (bits)", "movement benefit"),
        (
            "observed_high_motion_change_1_to_3_bits",
            "Held-out ΔSSI, 1×→3× (bits)",
            "higher-motion 1×→3× change",
        ),
    )
    for axis, (x_column, x_label, association) in zip(axes, plots):
        for contrast in CONTRAST_ORDER:
            subset = unit.loc[unit.contrast.eq(contrast)]
            axis.scatter(
                subset[x_column],
                subset.activity_weighted_candidate_p_reliance,
                s=13,
                alpha=0.58,
                facecolor=CONTRAST_COLOR[contrast],
                edgecolor="white",
                linewidth=0.25,
                marker=CONTRAST_MARKER[contrast],
                rasterized=False,
            )
            association_row = associations.loc[
                associations.contrast.eq(contrast)
                & associations.association.eq(association)
            ]
            if len(association_row) != 1:
                raise DataUnavailable(f"Missing Panel D association {contrast}/{association}")
            rho = float(association_row.iloc[0].spearman_rho)
            low = float(association_row.iloc[0].unit_bootstrap_ci_low)
            high = float(association_row.iloc[0].unit_bootstrap_ci_high)
            axis.text(
                0.02,
                0.97 - 0.07 * CONTRAST_ORDER.index(contrast),
                f"{CONTRAST_LABEL[contrast].replace(chr(10), ' ')}: ρ={rho:.2f} [{low:.2f}, {high:.2f}]",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.25,
                color=CONTRAST_COLOR[contrast],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
            )
        axis.set_xlabel(x_label)
        axis.set_ylim(-0.02, 1.02)
        axis.axhline(0, color=GRID_COLOR, lw=0.5)
        _clean_axis(axis, grid="y")
    axes[0].set_ylabel("Activity-weighted P reliance")
    axes[0].set_title("Per-unit reliance on candidate P", loc="left", fontweight="bold")


def draw_figure(
    panel_a_curves: pd.DataFrame,
    panel_a_neff: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d_units: pd.DataFrame,
    panel_d_associations: pd.DataFrame,
    *,
    subspace_term: str,
) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(13.8, 8.2), facecolor="white")
    outer = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.18, 1.05, 1.25),
        height_ratios=(1.0, 1.02),
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.895,
        wspace=0.34,
        hspace=0.50,
    )
    a_grid = outer[0, 0].subgridspec(1, 2, width_ratios=(1.65, 0.85), wspace=0.35)
    axis_a = figure.add_subplot(a_grid[0, 0])
    inset_a = figure.add_subplot(a_grid[0, 1])
    b_grid = outer[0, 1].subgridspec(2, 1, height_ratios=(1.15, 0.85), hspace=0.16)
    axis_b_fraction = figure.add_subplot(b_grid[0, 0])
    axis_b_ratio = figure.add_subplot(b_grid[1, 0])
    axis_c = figure.add_subplot(outer[0, 2])
    d_grid = outer[1, :].subgridspec(1, 3, wspace=0.28)
    axes_d = [figure.add_subplot(d_grid[0, index]) for index in range(3)]

    _plot_panel_a(axis_a, inset_a, panel_a_curves, panel_a_neff)
    _plot_panel_b(axis_b_fraction, axis_b_ratio, panel_b)
    _plot_panel_c(axis_c, panel_c)
    _plot_panel_d(axes_d, panel_d_units, panel_d_associations)

    panel_axes = (axis_a, axis_b_fraction, axis_c, axes_d[0])
    for label, axis in zip("ABCD", panel_axes):
        axis.text(
            -0.17,
            1.18 if label != "D" else 1.13,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            color=TEXT_COLOR,
        )
    figure.suptitle(
        f"What the {subspace_term} and its complement encode",
        x=0.055,
        y=0.978,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    return figure


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _format_interval(value: float, low: float, high: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _terminology_boundary(inputs: FinalizedInputs) -> tuple[str, dict[str, Any]]:
    gates = inputs.rank8_gate.get("heldout_generalization", {})
    consistency = {
        contrast: bool(gates[contrast].get("learned_outperforms_readout_consistently", False))
        for contrast in CONTRAST_ORDER
    }
    allowed = all(consistency.values())
    statement = (
        "The rank-8 bases are described here as candidate movement subspaces; this figure tests "
        "whether their held-out activity is motion-enriched. It does not localize a native-channel circuit."
        if allowed
        else "Because learned rank-8 does not consistently outperform rank-8 readout-SVD for every "
        "contrast, the bases are compact output-relevant subspaces. Any motion enrichment observed "
        "in this figure does not by itself make them movement-specific or establish recurrent registration."
    )
    return statement, {
        "learned_outperforms_readout_consistently": consistency,
        "all_contrasts_consistently_outperform_readout_svd": allowed,
        "allowed_term": "candidate movement subspace" if allowed else "compact output-relevant subspace",
        "forbidden_inference": "movement specificity, native-channel circuit identity, or recurrent registration",
    }


def caption_draft(
    inputs: FinalizedInputs,
    panel_a_neff: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d_associations: pd.DataFrame,
    *,
    draws: int,
) -> str:
    neff_parts = []
    for contrast in CONTRAST_ORDER:
        row = panel_a_neff.loc[panel_a_neff.contrast.eq(contrast)].iloc[0]
        neff_parts.append(
            f"{CONTRAST_DIRECT[contrast]} N_eff="
            + _format_interval(
                float(row.effective_participating_channels_fold_mean),
                float(row.effective_participating_channels_fold_bootstrap_ci_low),
                float(row.effective_participating_channels_fold_bootstrap_ci_high),
                1,
            )
        )
    b_summary = panel_b.loc[panel_b.row_type.eq("fold_summary")]
    b_parts = []
    for contrast in CONTRAST_ORDER:
        row = b_summary.loc[
            b_summary.contrast.eq(contrast) & b_summary.variation.eq("motion")
        ].iloc[0]
        b_parts.append(
            f"{CONTRAST_DIRECT[contrast]} P fraction="
            + _format_interval(
                float(row.candidate_p_total_fraction_fold_mean),
                float(row.candidate_p_total_fraction_fold_bootstrap_ci_low),
                float(row.candidate_p_total_fraction_fold_bootstrap_ci_high),
            )
            + ", per-dimension P/Q ratio="
            + _format_interval(
                float(row.p_to_q_per_dimension_energy_ratio_fold_mean),
                float(row.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_low),
                float(row.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_high),
            )
        )
    c_summary = panel_c.loc[panel_c.row_type.eq("fold_summary")]
    c_parts = []
    for contrast in CONTRAST_ORDER:
        selected = c_summary.loc[
            c_summary.contrast.eq(contrast) & c_summary.transformation.eq("movement map")
        ]
        descriptions = []
        for component in ("candidate P", "complementary Q"):
            row = selected.loc[selected.component.eq(component)].iloc[0]
            descriptions.append(
                f"{component} R²="
                + _format_interval(
                    float(row.complete_normalized_map_recovery_r2_fold_mean),
                    float(row.complete_normalized_map_recovery_r2_fold_bootstrap_ci_low),
                    float(row.complete_normalized_map_recovery_r2_fold_bootstrap_ci_high),
                )
            )
        c_parts.append(f"{CONTRAST_DIRECT[contrast]}: " + "; ".join(descriptions))
    d_parts = []
    for association in (
        "historical spatial frequency",
        "movement benefit",
        "higher-motion 1×→3× change",
    ):
        selected = panel_d_associations.loc[panel_d_associations.association.eq(association)]
        values = []
        for contrast in CONTRAST_ORDER:
            row = selected.loc[selected.contrast.eq(contrast)].iloc[0]
            values.append(
                f"{CONTRAST_DIRECT[contrast]} ρ="
                + _format_interval(
                    float(row.spearman_rho),
                    float(row.unit_bootstrap_ci_low),
                    float(row.unit_bootstrap_ci_high),
                )
            )
        d_parts.append(f"{association}: " + "; ".join(values))
    boundary, boundary_record = _terminology_boundary(inputs)
    subspace_term = str(boundary_record["allowed_term"])
    return (
        "# Figure 1 caption draft\n\n"
        f"**What the {subspace_term} and its complement encode.** "
        "All estimates use fold-specific rank-8 projectors on their crossed held-out "
        "2-image × 6-trajectory blocks; consensus projectors are visualization-only and are not used here. "
        "**A,** Native-channel leverage (solid; ℓc=Pcc) and cumulative leverage (dashed) show how each "
        "rank-8 channel subspace is distributed across the 128 native ConvGRU channels. No leverage "
        "threshold defines a circuit. "
        + "; ".join(neff_parts)
        + ". **B,** Candidate-P and complementary-Q fractions of held-out image-mean content energy and "
        "movement-induced state energy are shown separately from energy enrichment per dimension, "
        "(P energy/8)/(Q energy/120). For movement: "
        + "; ".join(b_parts)
        + ". **C,** Complete normalized RR100 map recovery from P-only and Q-only stabilized visual "
        "content, and from the literal P-only and Q-only defining movement contrasts. Recovery is "
        "paired-expected-spike weighted, unclipped, and therefore may be negative. Movement-map results: "
        + "; ".join(c_parts)
        + ". **D,** Activity-weighted per-unit reliance on P is related to historical SF, fold-held-out "
        "movement benefit, and the held-out 1×→3× higher-motion change (the reversal variable for "
        "higher-SF units). Fold-held-out values are averaged within RR100 "
        "unit before across-unit Spearman association; brackets are 95% unit-bootstrap intervals. "
        + "; ".join(d_parts)
        + f". All other brackets are 95% nonparametric fold-bootstrap intervals ({draws:,} draws; four "
        "folds). "
        + boundary
        + "\n"
    )


def _save_figure_atomic(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def run(
    *,
    input_dir: Path,
    output_dir: Path | None = None,
    figure_data_dir: Path | None = None,
    bootstrap_draws: int = 10_000,
    seed: int = 20261741,
) -> dict[str, Any]:
    if int(bootstrap_draws) < 1000:
        raise ValueError("At least 1,000 bootstrap draws are required for the production figure")
    inputs = load_finalized_inputs(input_dir)
    output = Path(output_dir or (inputs.root / "figures")).resolve()
    figure_data = Path(
        figure_data_dir or (output / f"{FIGURE_BASENAME}_plotting_data")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)

    panel_a_curves, panel_a_neff = prepare_panel_a(
        inputs.leverage, draws=bootstrap_draws, seed=seed + 1000
    )
    panel_b = prepare_panel_b(inputs.variance, draws=bootstrap_draws, seed=seed + 2000)
    panel_c = prepare_panel_c(inputs.readout, draws=bootstrap_draws, seed=seed + 3000)
    panel_d_units, panel_d_associations = prepare_panel_d(
        inputs.per_unit, draws=bootstrap_draws, seed=seed + 4000
    )
    _, boundary_record = _terminology_boundary(inputs)
    _validate_source_snapshot(inputs)

    data_paths = {
        "panel_a_leverage_curves": figure_data / "panel_a_native_channel_leverage.csv",
        "panel_a_effective_count": figure_data / "panel_a_effective_participating_channels.csv",
        "panel_b_content_motion_energy": figure_data / "panel_b_content_motion_energy.csv",
        "panel_c_rr100_recovery": figure_data / "panel_c_rr100_map_recovery.csv",
        "panel_d_per_unit": figure_data / "panel_d_per_unit_p_reliance.csv",
        "panel_d_associations": figure_data / "panel_d_association_summary.csv",
    }
    for path, frame in (
        (data_paths["panel_a_leverage_curves"], panel_a_curves),
        (data_paths["panel_a_effective_count"], panel_a_neff),
        (data_paths["panel_b_content_motion_energy"], panel_b),
        (data_paths["panel_c_rr100_recovery"], panel_c),
        (data_paths["panel_d_per_unit"], panel_d_units),
        (data_paths["panel_d_associations"], panel_d_associations),
    ):
        _atomic_csv(path, frame)
    arrays_path = figure_data / "figure1_pq_semantics_exact_arrays.npz"
    temporary_arrays = arrays_path.with_name(f"{arrays_path.name}.tmp.{os.getpid()}")
    with temporary_arrays.open("wb") as handle:
        panel_a_summary = panel_a_curves.loc[
            panel_a_curves.row_type.eq("fold_summary")
        ].reset_index(drop=True)
        panel_b_summary = panel_b.loc[panel_b.row_type.eq("fold_summary")].reset_index(
            drop=True
        )
        panel_c_summary = panel_c.loc[panel_c.row_type.eq("fold_summary")].reset_index(
            drop=True
        )
        np.savez_compressed(
            handle,
            panel_a_contrast=panel_a_summary.contrast.to_numpy(dtype="U32"),
            panel_a_channel_rank=panel_a_summary.channel_rank.to_numpy(np.int16),
            panel_a_leverage_mean=panel_a_summary.leverage_score_fold_mean.to_numpy(np.float64),
            panel_a_leverage_ci_low=panel_a_summary.leverage_score_fold_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_a_leverage_ci_high=panel_a_summary.leverage_score_fold_bootstrap_ci_high.to_numpy(
                np.float64
            ),
            panel_a_cumulative_mean=panel_a_summary.cumulative_leverage_fraction_fold_mean.to_numpy(
                np.float64
            ),
            panel_a_neff_contrast=panel_a_neff.contrast.to_numpy(dtype="U32"),
            panel_a_neff=panel_a_neff.effective_participating_channels_fold_mean.to_numpy(np.float64),
            panel_a_neff_ci_low=panel_a_neff.effective_participating_channels_fold_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_a_neff_ci_high=panel_a_neff.effective_participating_channels_fold_bootstrap_ci_high.to_numpy(
                np.float64
            ),
            panel_b_contrast=panel_b_summary.contrast.to_numpy(dtype="U32"),
            panel_b_variation=panel_b_summary.variation.to_numpy(dtype="U16"),
            panel_b_p_fraction=panel_b_summary.candidate_p_total_fraction_fold_mean.to_numpy(
                np.float64
            ),
            panel_b_p_fraction_ci_low=panel_b_summary.candidate_p_total_fraction_fold_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_b_p_fraction_ci_high=panel_b_summary.candidate_p_total_fraction_fold_bootstrap_ci_high.to_numpy(
                np.float64
            ),
            panel_b_per_dimension_ratio=panel_b_summary.p_to_q_per_dimension_energy_ratio_fold_mean.to_numpy(
                np.float64
            ),
            panel_b_per_dimension_ratio_ci_low=panel_b_summary.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_b_per_dimension_ratio_ci_high=panel_b_summary.p_to_q_per_dimension_energy_ratio_fold_bootstrap_ci_high.to_numpy(
                np.float64
            ),
            panel_c_contrast=panel_c_summary.contrast.to_numpy(dtype="U32"),
            panel_c_transformation=panel_c_summary.transformation.to_numpy(dtype="U32"),
            panel_c_component=panel_c_summary.component.to_numpy(dtype="U24"),
            panel_c_recovery=panel_c_summary.complete_normalized_map_recovery_r2_fold_mean.to_numpy(
                np.float64
            ),
            panel_c_recovery_ci_low=panel_c_summary.complete_normalized_map_recovery_r2_fold_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_c_recovery_ci_high=panel_c_summary.complete_normalized_map_recovery_r2_fold_bootstrap_ci_high.to_numpy(
                np.float64
            ),
            panel_d_unit_contrast=panel_d_units.contrast.to_numpy(dtype="U32"),
            panel_d_unit_index=panel_d_units.unit_index.to_numpy(np.int16),
            panel_d_unit_reliance=panel_d_units.activity_weighted_candidate_p_reliance.to_numpy(
                np.float64
            ),
            panel_d_unit_sf=panel_d_units.sf_split_metric.to_numpy(np.float64),
            panel_d_unit_ssi_benefit=panel_d_units.observed_ssi_benefit_bits.to_numpy(
                np.float64
            ),
            panel_d_unit_high_motion_change=panel_d_units.observed_high_motion_change_1_to_3_bits.to_numpy(
                np.float64
            ),
            panel_d_association_contrast=panel_d_associations.contrast.to_numpy(dtype="U32"),
            panel_d_association_name=panel_d_associations.association.to_numpy(dtype="U32"),
            panel_d_spearman=panel_d_associations.spearman_rho.to_numpy(np.float64),
            panel_d_spearman_ci_low=panel_d_associations.unit_bootstrap_ci_low.to_numpy(
                np.float64
            ),
            panel_d_spearman_ci_high=panel_d_associations.unit_bootstrap_ci_high.to_numpy(
                np.float64
            ),
        )
    os.replace(temporary_arrays, arrays_path)

    figure = draw_figure(
        panel_a_curves,
        panel_a_neff,
        panel_b,
        panel_c,
        panel_d_units,
        panel_d_associations,
        subspace_term=str(boundary_record["allowed_term"]),
    )
    figure_paths = {
        "svg": output / f"{FIGURE_BASENAME}.svg",
        "pdf": output / f"{FIGURE_BASENAME}.pdf",
        "png": output / f"{FIGURE_BASENAME}.png",
    }
    _save_figure_atomic(figure, figure_paths["svg"], bbox_inches="tight")
    _save_figure_atomic(figure, figure_paths["pdf"], bbox_inches="tight")
    _save_figure_atomic(figure, figure_paths["png"], bbox_inches="tight", dpi=600)
    plt.close(figure)

    caption = caption_draft(
        inputs,
        panel_a_neff,
        panel_b,
        panel_c,
        panel_d_associations,
        draws=bootstrap_draws,
    )
    caption_path = output / CAPTION_NAME
    temporary_caption = caption_path.with_name(f"{caption_path.name}.tmp.{os.getpid()}")
    temporary_caption.write_text(caption, encoding="utf-8")
    os.replace(temporary_caption, caption_path)

    boundary, boundary_record = _terminology_boundary(inputs)
    _validate_source_snapshot(inputs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "saved_products_only": True,
        "model_imported": False,
        "state_or_readout_cache_read": False,
        "rank8_basis_or_projector_arrays_read": False,
        "foldwise_heldout_inference": True,
        "consensus_projectors_used_for_inference": False,
        "consensus_role": "visualization-only metadata gate; consensus arrays are not read",
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "fold_intervals": "resample four crossed folds with replacement",
            "unit_intervals": "average held-out folds within unit, then resample RR100 units",
        },
        "terminology_boundary": {**boundary_record, "caption_statement": boundary},
        "source_products": [
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
            for path, digest in inputs.source_sha256
        ],
        "exact_plotting_data": {
            **{
                key: {"path": str(path), "sha256": sha256_file(path), "rows": len(pd.read_csv(path))}
                for key, path in data_paths.items()
            },
            "exact_arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path)},
        },
        "figure_exports": {
            key: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for key, path in figure_paths.items()
        },
        "caption": {"path": str(caption_path), "sha256": sha256_file(caption_path)},
        "scientific_guards": [
            "No leverage threshold is used to define native channels as a circuit.",
            "Total fraction and per-dimension energy are plotted on separate axes.",
            "Complete-map recovery is paired-expected-spike weighted and unclipped.",
            "Fold-heldout values are aggregated before per-unit association.",
            "Consensus projectors are not used for inference.",
            "Rank-8 U/P arrays are not read by plotting; finalized inventory hashes are cross-checked against P/Q stage markers.",
            "No movement-specific or recurrent-registration claim follows from output relevance alone.",
        ],
    }
    manifest_path = output / PLOT_MANIFEST_NAME
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-data-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261741)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        figure_data_dir=args.figure_data_dir,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
