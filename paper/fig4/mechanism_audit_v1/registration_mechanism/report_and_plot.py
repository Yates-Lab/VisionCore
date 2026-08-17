#!/usr/bin/env python3
"""Assemble the final ConvGRU registration audit from saved products only.

This module is intentionally downstream of every numerical analysis.  It does
not import the frozen model, renderer, state/readout caches, or any core replay
code.  It validates finalized provenance, derives the predeclared decision,
draws the causal Figure 3, and writes the morning report.  Missing products are
reported as unavailable; absence is never converted into negative evidence.
The historical P/Q motion-energy stop is explicitly waived: its failed
factorization remains reported, while only completed downstream registration
and causal evidence can select the final four-way decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
DEFAULT_FIGURE1 = DEFAULT_INPUT / "figures/figure1_pq_semantics_plot_manifest.json"
DEFAULT_FIGURE2 = DEFAULT_INPUT / "figure2_registration/manifest.json"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figure3_causal_shifter"

SCHEMA_VERSION = "fig4-convgru-registration-final-report-v2"
CAUSAL_SCHEMA = "fig4-registration-causal-interventions-v1"
RANK_SCHEMA = "fig4-registration-rank8-validation-v1"
PQ_SCHEMA = "fig4-registration-pq-semantics-v1"
GRU_SCHEMA = "fig4-gru-registration-v4"
GRU_PROJECTED_TERM_NAMES = (
    "h_previous",
    "update_gate",
    "reset_gate",
    "candidate_current_preactivation",
    "candidate_recurrent_preactivation",
    "candidate_state",
    "retained_state_contribution",
    "new_state_contribution",
    "h_t",
)
GRU_PROJECTED_VALUE_SUFFIXES = (
    "total_energy",
    "p_energy",
    "q_energy",
    "p_energy_per_dimension",
    "q_energy_per_dimension",
    "readout_svd_energy",
    "readout_svd_energy_per_dimension",
)

FOLDS = (0, 1, 2, 3)
CONTRASTS = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
TARGET_SCALE = {"low_0_to_2": 2.0, "high_0_to_1": 1.0, "high_1_to_3": 3.0}
TARGET_POPULATION = {
    "low_0_to_2": "lower_sf_71",
    "high_0_to_1": "higher_sf_29",
    "high_1_to_3": "higher_sf_29",
}
CAUSAL_CONTRAST = {
    "low_0_to_2": "lower_sf_sharpening_0_to_2",
    "high_0_to_1": "higher_sf_sharpening_0_to_1",
    "high_1_to_3": "higher_sf_reversal_1_to_3",
}
CONTRAST_LABEL = {
    "low_0_to_2": "lower-SF sharpening (0×→2×)",
    "high_0_to_1": "higher-SF sharpening (0×→1×)",
    "high_1_to_3": "higher-SF reversal (1×→3×)",
}
MOTION_ENERGY_LABEL = {
    "low_0_to_2": "lower-SF projector: h(2×)−h(0×)",
    "high_0_to_1": "higher-SF sharpening projector: h(1×)−h(0×)",
    "high_1_to_3": "higher-SF reversal projector: h(3×)−h(0×)",
}
SCALES = (0.0, 0.5, 1.0, 2.0, 3.0)
POPULATIONS = ("all_100", "lower_sf_71", "higher_sf_29")
PROBE_REPRESENTATION = {
    "candidate_p": "candidate_p_coordinates_rank8",
    "complementary_q": "complementary_q_reconstructed_channels_rank120",
}

TRANSPORT_CONDITIONS = (
    "intact",
    "candidate_recurrent_center_only",
    "gate_recurrent_center_only",
    "all_recurrent_center_only",
    "all_recurrent_offset_permuted_seed_20260913",
    "all_recurrent_offset_permuted_seed_20261023",
    "all_recurrent_offset_permuted_seed_20261119",
    "no_recurrence_reference",
)
REALIGNMENT_CONDITIONS = (
    "no_shift_intact",
    "eye_correct_candidate_p",
    "activation_oracle_candidate_p",
    "eye_opposite_candidate_p",
    "eye_random_matched_candidate_p",
    "eye_correct_complementary_q",
    "induced_eye_misalignment_candidate_p",
)


def _realignment_cell_is_run(condition: str, scale: float) -> bool:
    if condition == "no_shift_intact":
        return bool(scale in (1.0, 3.0))
    if condition == "induced_eye_misalignment_candidate_p":
        return bool(np.isclose(scale, 1.0))
    if condition in REALIGNMENT_CONDITIONS:
        return bool(np.isclose(scale, 3.0))
    raise DataUnavailable(f"Unknown realignment condition {condition!r}")

DECISION_LABELS = (
    "SHIFTER/REGISTRATION MECHANISM SUPPORTED",
    "RECURRENT SPATIAL TRANSPORT EXISTS BUT DOES NOT EXPLAIN SSI",
    "NO EVIDENCE FOR RECURRENT REGISTRATION",
    "INCONCLUSIVE",
)
MANUSCRIPT_LABELS = ("SUPPORTED", "CONSISTENT WITH", "NOT SUPPORTED")

REPORT_QUESTIONS = (
    "Does raw previous state lag current?",
    "Does recurrence reduce lag?",
    "Does inferred transport track retinal displacement?",
    "Is inferred transport stronger in P than Q/readout-SVD/random?",
    "Does registration fail at excessive high-SF motion?",
    "Does destroying spatial recurrence alter SSI curves?",
    "Does correct realignment rescue high-SF 3× SSI?",
    "Does induced misalignment destroy high-SF 1× sharpening?",
)

PQ_GATE_WAIVER = {
    "authorized": True,
    "previous_stop": "all-three defining-scale P-motion enrichment",
    "previous_factorization_result": "failed and preserved",
    "rejected_interpretation": "motion-only P and content-only Q",
    "allowed_interpretation": (
        "P is a compact channel subspace strongly visible to RR100; "
        "Q is its complementary control"
    ),
    "not_negative_shifter_evidence": True,
    "p_energy_per_dimension_must_exceed_q": False,
    "factorization_reopened": False,
    "downstream_registration_and_causal_evidence_required": True,
}

REQUIRED_CANONICAL = (
    "fold_rank8_results.csv",
    "projector_stability.csv",
    "consensus_projectors.npz",
    "native_channel_leverage.csv",
    "pq_variance_decomposition.csv",
    "pq_readout_decomposition.csv",
    "per_unit_pq_reliance.csv",
    "gru_equation_audit.md",
    "gru_projected_terms_heldout.npz",
    "registration_metrics_heldout.csv",
    "kernel_offset_energy.csv",
    "transport_ablation_results.csv",
    "realignment_results.csv",
)

LOW_COLOR = "#2878B5"
HIGH_COLOR = "#D55E00"
TEXT_COLOR = "#202428"
MUTED_COLOR = "#686E73"
GRID_COLOR = "#D8DCDF"
CONTROL_COLOR = "#72787D"
CORRECT_COLOR = "#5B3F99"
BAD_COLOR = "#A5A9AC"

# Prospective report-level interpretations of qualitative words in the brief.
# They are recorded in statistics.json so the labels cannot drift after seeing
# the values.
MIN_SUBSTANTIAL_Q_BASELINE_R2 = 0.25
MIN_DISTRIBUTED_NATIVE_NEFF = 16.0
MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS = 0.002
MIN_SUBSTANTIAL_P_MOVEMENT_MAP_R2 = 0.25
MIN_P_OVER_Q_MOVEMENT_MAP_R2_MARGIN = 0.05


class DataUnavailable(RuntimeError):
    """A final inference cannot be made without inventing missing data."""


@dataclass(frozen=True)
class CausalProducts:
    scope: str
    root: Path
    analysis_manifest: dict[str, Any]
    pilot_gate: dict[str, Any] | None
    transport: pd.DataFrame
    realignment: pd.DataFrame
    transport_archive: dict[str, np.ndarray]
    realignment_archive: dict[str, np.ndarray]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class FinalInputs:
    root: Path
    rank_gate: dict[str, Any]
    rank_results: pd.DataFrame
    stability: pd.DataFrame
    leverage: pd.DataFrame
    variance: pd.DataFrame
    readout: pd.DataFrame
    per_unit: pd.DataFrame
    probes: pd.DataFrame
    gru_audit: dict[str, Any]
    registration: pd.DataFrame
    kernel: pd.DataFrame
    figure1_manifest: dict[str, Any]
    figure2_manifest: dict[str, Any]
    causal: CausalProducts
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class UpstreamInputs:
    """Products available at the historical, now-waived P/Q gate."""

    root: Path
    rank_gate: dict[str, Any]
    rank_results: pd.DataFrame
    stability: pd.DataFrame
    leverage: pd.DataFrame
    variance: pd.DataFrame
    readout: pd.DataFrame
    per_unit: pd.DataFrame
    probes: pd.DataFrame
    figure1_manifest: dict[str, Any]
    source_paths: tuple[Path, ...]


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataUnavailable(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataUnavailable(f"Unreadable {label}: {path}") from error
    if not isinstance(value, dict):
        raise DataUnavailable(f"{label} must be a JSON object: {path}")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise DataUnavailable(f"Missing or empty {label}: {path}")
    try:
        value = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataUnavailable(f"Unreadable {label}: {path}") from error
    if value.empty:
        raise DataUnavailable(f"{label} contains no rows: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataUnavailable(f"{label} lacks required columns: {', '.join(missing)}")


def _validate_probe_schema(probes: pd.DataFrame) -> None:
    """Validate the finalized descriptive-probe representation vocabulary."""
    _require_columns(
        probes,
        ("fold", "contrast", "probe", "representation", "value", "causal_interpretation_allowed"),
        "descriptive P/Q probes",
    )
    if set(probes.representation.astype(str)) != set(PROBE_REPRESENTATION.values()):
        raise DataUnavailable("P/Q probes do not contain exactly candidate P and complementary Q")
    causal_allowed = probes.causal_interpretation_allowed.astype(str).str.lower()
    if not causal_allowed.isin(("false", "0")).all():
        raise DataUnavailable("A descriptive probe incorrectly permits causal interpretation")


def _path_within(path: Path, root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as error:
        raise DataUnavailable(f"{label} escapes the finalized analysis root: {resolved}") from error
    return resolved


def _validate_record(
    record: Mapping[str, Any],
    *,
    label: str,
    root: Path | None = None,
) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if root is not None:
        _path_within(path, root, label)
    if not path.is_file():
        raise DataUnavailable(f"Recorded {label} is missing: {path}")
    expected_hash = str(record.get("sha256", ""))
    if not expected_hash or sha256_file(path) != expected_hash:
        raise DataUnavailable(f"Recorded {label} hash does not match: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise DataUnavailable(f"Recorded {label} size does not match: {path}")
    return path


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _validate_rank8(root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[Path]]:
    gate_path = root / "rank8_validation_gate.json"
    manifest_path = root / "rank8_validation/analysis_manifest.json"
    inventory_path = root / "rank8_validation/rank8_projector_inventory.json"
    fold_path = root / "fold_rank8_results.csv"
    stability_path = root / "projector_stability.csv"
    consensus_path = root / "consensus_projectors.npz"
    gate = _read_json(gate_path, "rank-8 gate")
    manifest = _read_json(manifest_path, "rank-8 manifest")
    inventory = _read_json(inventory_path, "rank-8 inventory")
    if gate.get("status") != "complete" or gate.get("rank_fixed_prospectively") != 8:
        raise DataUnavailable("Rank-8 validation is not finalized at prospectively fixed rank 8")
    if gate.get("stop_downstream_mechanism_audit") is not False:
        raise DataUnavailable("The explicit rank-8 stopping gate halted downstream inference")
    heldout = gate.get("heldout_generalization")
    if not isinstance(heldout, dict) or set(heldout) != set(CONTRASTS):
        raise DataUnavailable("Rank-8 gate lacks exactly three held-out contrast decisions")
    if not all(bool(heldout[key].get("complete")) for key in CONTRASTS):
        raise DataUnavailable("At least one rank-8 held-out contrast is incomplete")
    if manifest.get("schema_version") != RANK_SCHEMA:
        raise DataUnavailable("Unexpected rank-8 manifest schema")
    if (
        manifest.get("uses_saved_cache_only_for_evaluation") is not True
        or manifest.get("consensus_is_visualization_only") is not True
        or int(manifest.get("rank", -1)) != 8
        or set(manifest.get("folds", ())) != set(FOLDS)
        or set(manifest.get("contrasts", ())) != set(CONTRASTS)
    ):
        raise DataUnavailable("Rank-8 manifest does not preserve the inference boundaries")
    outputs = manifest.get("outputs", {})
    for key, expected in (
        ("fold_results", fold_path),
        ("stability", stability_path),
        ("consensus", consensus_path),
        ("gate", gate_path),
        ("inventory", inventory_path),
    ):
        if Path(str(outputs.get(key, ""))).resolve() != expected.resolve():
            raise DataUnavailable(f"Rank-8 manifest points {key} at an unexpected product")

    fold = _read_csv(fold_path, "fold rank-8 results")
    _require_columns(
        fold,
        ("fold", "contrast", "method", "complete_map_recovery_mean"),
        "fold rank-8 results",
    )
    observed = {
        (int(row.fold), str(row.contrast), str(row.method)) for row in fold.itertuples()
    }
    expected = {
        (fold_index, contrast, method)
        for fold_index in FOLDS
        for contrast in CONTRASTS
        for method in ("learned", "readout_svd")
    }
    if observed != expected or len(fold) != 24:
        raise DataUnavailable("Fold rank-8 results are not exactly 4×3×2")

    stability = _read_csv(stability_path, "projector stability")
    _require_columns(
        stability,
        ("comparison_type", "contrast_a", "contrast_b", "fold_a", "fold_b", "projector_overlap"),
        "projector stability",
    )
    within = stability.loc[stability.comparison_type.eq("within_contrast_across_folds")]
    if len(within) != 18 or set(within.contrast_a) != set(CONTRASTS):
        raise DataUnavailable("Projector stability lacks six fold pairs for every contrast")

    if inventory.get("schema_version") != "fig4-registration-rank8-projector-inventory-v1":
        raise DataUnavailable("Unexpected rank-8 inventory schema")
    if inventory.get("consensus_for_visualization_only") is not True:
        raise DataUnavailable("Consensus projector is not explicitly visualization-only")
    projectors = inventory.get("fold_projectors")
    if not isinstance(projectors, list) or len(projectors) != 12:
        raise DataUnavailable("Rank-8 inventory does not contain 12 fold-wise projectors")
    inventory_cells: set[tuple[int, str]] = set()
    projector_products: list[Path] = []
    for record in projectors:
        if not isinstance(record, dict):
            raise DataUnavailable("Malformed fold-projector inventory record")
        inventory_cells.add((int(record.get("fold", -1)), str(record.get("contrast", ""))))
        for path_key, hash_key in (
            ("basis_path", "basis_sha256"),
            ("projector_path", "projector_sha256"),
            ("heldout_predictions_path", "heldout_predictions_sha256"),
        ):
            projector_products.append(
                _validate_record(
                    {"path": record.get(path_key), "sha256": record.get(hash_key)},
                    label=f"fold projector {path_key}",
                )
            )
    if inventory_cells != {(fold_index, contrast) for fold_index in FOLDS for contrast in CONTRASTS}:
        raise DataUnavailable("Fold-projector inventory cells are incomplete")
    if not consensus_path.is_file():
        raise DataUnavailable(f"Missing consensus-projector visualization product: {consensus_path}")
    try:
        with np.load(consensus_path, allow_pickle=False) as archive:
            for contrast in CONTRASTS:
                basis = np.asarray(archive[f"U__{contrast}"], dtype=float)
                projector = np.asarray(archive[f"P__{contrast}"], dtype=float)
                if basis.shape != (128, 8) or projector.shape != (128, 128):
                    raise DataUnavailable("Consensus projector arrays have invalid shapes")
                if not np.allclose(basis.T @ basis, np.eye(8), atol=3e-4):
                    raise DataUnavailable("Consensus basis is not orthonormal")
                if not np.allclose(projector, basis @ basis.T, atol=3e-4):
                    raise DataUnavailable("Consensus P does not equal UUᵀ")
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if metadata.get("visualization_only") is not True:
                raise DataUnavailable("Consensus archive does not mark itself visualization-only")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, DataUnavailable):
            raise
        raise DataUnavailable(f"Unreadable consensus archive: {consensus_path}") from error
    return gate, fold, stability, [
        gate_path,
        manifest_path,
        inventory_path,
        fold_path,
        stability_path,
        consensus_path,
        *projector_products,
    ]


def _validate_pq(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    manifest_path = root / "pq_semantics_manifest.json"
    manifest = _read_json(manifest_path, "P/Q semantic manifest")
    if manifest.get("schema_version") != PQ_SCHEMA:
        raise DataUnavailable("Unexpected P/Q semantic schema")
    if manifest.get("status") != "complete" or manifest.get("complete_all_four_folds") is not True:
        raise DataUnavailable("P/Q semantics are not complete across all four folds")
    if manifest.get("no_core_replay") is not True or manifest.get("model_imported") is not False:
        raise DataUnavailable("P/Q manifest does not certify a saved-product-only analysis")
    for key in (
        "missing_basis_fold_contrasts",
        "missing_variance_fold_contrasts",
        "missing_readout_fold_contrasts",
        "missing_probe_fold_contrasts",
    ):
        if manifest.get(key) not in ([], None):
            raise DataUnavailable(f"P/Q manifest reports missing cells in {key}")

    canonical = manifest.get("canonical_products")
    if not isinstance(canonical, dict):
        raise DataUnavailable("P/Q manifest lacks canonical product records")
    frames: dict[str, pd.DataFrame] = {}
    paths: list[Path] = [manifest_path]
    for name in (
        "native_channel_leverage.csv",
        "pq_variance_decomposition.csv",
        "pq_readout_decomposition.csv",
        "per_unit_pq_reliance.csv",
    ):
        record = canonical.get(name)
        if not isinstance(record, dict) or record.get("exists") is not True:
            raise DataUnavailable(f"P/Q manifest marks {name} unavailable")
        path = _validate_record(record, label=name, root=root)
        if path != (root / name).resolve():
            raise DataUnavailable(f"P/Q canonical {name} is outside its canonical location")
        frame = _read_csv(path, name)
        if int(record.get("rows", -1)) != len(frame):
            raise DataUnavailable(f"P/Q row count changed for {name}")
        frames[name] = frame
        paths.append(path)

    expected_cells = {(fold_index, contrast) for fold_index in FOLDS for contrast in CONTRASTS}
    for name, frame in frames.items():
        _require_columns(frame, ("fold", "contrast"), name)
        cells = {(int(row.fold), str(row.contrast)) for row in frame.itertuples()}
        if cells != expected_cells:
            raise DataUnavailable(f"{name} is not exactly fold-wise over all 12 cells")
    leverage = frames["native_channel_leverage.csv"]
    _require_columns(
        leverage,
        ("native_channel", "leverage_score_p_cc", "effective_participating_channel_count"),
        "native-channel leverage",
    )
    if len(leverage) != 12 * 128:
        raise DataUnavailable("Native-channel leverage is not exactly 12 projectors × 128 channels")
    counts = leverage.groupby(["fold", "contrast"]).native_channel.nunique()
    if not np.all(counts.to_numpy() == 128):
        raise DataUnavailable("A leverage projector does not contain all 128 native channels")

    probes_record = manifest.get("descriptive_probe_product")
    if not isinstance(probes_record, dict) or probes_record.get("exists") is not True:
        raise DataUnavailable("Final descriptive P/Q probe product is missing")
    probes_path = _path_within(
        Path(str(probes_record.get("path", ""))), root, "descriptive P/Q probes"
    )
    if probes_path != (root / "pq_descriptive_probes.csv").resolve() or not probes_path.is_file():
        raise DataUnavailable("Final descriptive P/Q probe product is not canonical")
    probes = _read_csv(probes_path, "descriptive P/Q probes")
    if int(probes_record.get("rows", -1)) != len(probes):
        raise DataUnavailable("Descriptive P/Q probe row count changed")
    _validate_probe_schema(probes)
    paths.append(probes_path)

    # Every fold×contrast×stage marker must hash every admitted partial.
    required_stage_names = {
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
    for fold_index, contrast in sorted(expected_cells):
        for stage, names in required_stage_names.items():
            marker_path = root / "pq_intermediate" / f"fold_{fold_index}" / contrast / stage / "complete.json"
            marker = _read_json(marker_path, f"P/Q {stage} marker")
            if (
                marker.get("schema_version") != PQ_SCHEMA
                or marker.get("stage") != stage
                or marker.get("complete") is not True
            ):
                raise DataUnavailable(f"Incomplete or mismatched P/Q marker: {marker_path}")
            basis_path = Path(str(marker.get("basis_path", ""))).resolve()
            basis_parts = set(basis_path.parts)
            if contrast not in basis_parts or f"fold_{fold_index}" not in basis_parts:
                raise DataUnavailable(
                    f"P/Q marker basis identity does not match fold {fold_index}/{contrast}: {marker_path}"
                )
            if (
                not basis_path.is_file()
                or not marker.get("basis_sha256")
                or sha256_file(basis_path) != str(marker.get("basis_sha256"))
            ):
                raise DataUnavailable(f"P/Q marker basis hash does not match: {marker_path}")
            products = marker.get("products")
            if not isinstance(products, list):
                raise DataUnavailable(f"Malformed P/Q marker products: {marker_path}")
            if not names.issubset({Path(str(item.get("path", ""))).name for item in products}):
                raise DataUnavailable(f"P/Q marker lacks required {stage} products: {marker_path}")
            for item in products:
                if not isinstance(item, dict):
                    raise DataUnavailable(f"Malformed P/Q product record: {marker_path}")
                product_path = _validate_record(item, label=f"P/Q {stage} partial", root=root)
                if product_path.parent != marker_path.parent.resolve():
                    raise DataUnavailable(f"P/Q marker admits a payload from another cell: {marker_path}")
                paths.append(product_path)
            paths.append(basis_path)
            paths.append(marker_path)
    array_manifest_path = Path(
        str(manifest.get("supporting_array_manifest", root / "pq_supporting_arrays/manifest.json"))
    ).resolve()
    if array_manifest_path != (root / "pq_supporting_arrays/manifest.json").resolve():
        raise DataUnavailable("P/Q supporting-array manifest is outside its canonical location")
    array_manifest = _read_json(array_manifest_path, "P/Q supporting-array manifest")
    if array_manifest.get("schema_version") != PQ_SCHEMA:
        raise DataUnavailable("Unexpected P/Q supporting-array schema")
    array_products = array_manifest.get("products")
    expected_array_count = int(manifest.get("n_supporting_array_products", -1))
    if (
        expected_array_count != 24
        or not isinstance(array_products, list)
        or len(array_products) != expected_array_count
    ):
        raise DataUnavailable("P/Q supporting arrays are not exactly 12 cells × 2 stages")
    for record in array_products:
        if not isinstance(record, dict):
            raise DataUnavailable("Malformed P/Q supporting-array record")
        path = _validate_record(record, label="P/Q supporting array", root=root)
        if path.parent != array_manifest_path.parent:
            raise DataUnavailable("P/Q supporting array is outside its compact product directory")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise DataUnavailable(f"P/Q supporting archive is empty: {path}")
        except (OSError, ValueError) as error:
            if isinstance(error, DataUnavailable):
                raise
            raise DataUnavailable(f"Unreadable P/Q supporting archive: {path}") from error
        paths.append(path)
    paths.append(array_manifest_path)
    return (
        leverage,
        frames["pq_variance_decomposition.csv"],
        frames["pq_readout_decomposition.csv"],
        frames["per_unit_pq_reliance.csv"],
        probes,
        paths,
    )


def _validate_instrumentation_markers(
    root: Path,
    *,
    scope: str,
    expected_parts: int,
    consolidation: Mapping[str, Any],
) -> list[Path]:
    parts = root / "gru_instrumentation/parts"
    markers = sorted(parts.glob(f"{scope}__*__complete.json"))
    if len(markers) != expected_parts:
        raise DataUnavailable(
            f"GRU {scope} marker count {len(markers)} != {expected_parts}"
        )
    registration_rows = 0
    projected_rows = 0
    identities: set[str] = set()
    for marker_path in markers:
        marker = _read_json(marker_path, f"GRU {scope} part marker")
        if marker.get("schema") != GRU_SCHEMA or marker.get("scope") != scope:
            raise DataUnavailable(f"Mismatched GRU part marker: {marker_path}")
        stem = marker_path.name.removesuffix("__complete.json")
        if stem in identities:
            raise DataUnavailable(f"Duplicate GRU part identity: {stem}")
        identities.add(stem)
        registration = parts / f"{stem}__registration.csv.gz"
        terms = parts / f"{stem}__terms.npz"
        if not registration.is_file() or not terms.is_file():
            raise DataUnavailable(f"GRU completed part lacks a saved payload: {marker_path}")
        registration_rows += int(marker.get("registration_rows", -1))
        projected_rows += int(marker.get("projected_term_rows", -1))
        if float(marker.get("maximum_final_state_cache_error", math.inf)) > 3e-3:
            raise DataUnavailable(f"GRU part exceeds saved-state replay tolerance: {marker_path}")
    if registration_rows != int(consolidation.get("registration_rows", -1)):
        raise DataUnavailable(f"GRU {scope} marker registration-row sum differs from consolidation")
    if projected_rows != int(consolidation.get("projected_term_rows", -1)):
        raise DataUnavailable(f"GRU {scope} marker projected-row sum differs from consolidation")
    return markers


def _validate_gru_reconstruction(reconstruction: Mapping[str, Any]) -> None:
    for name in (
        "h_reconstruction_max_abs",
        "retained_plus_new_max_abs",
        "candidate_split_max_abs",
    ):
        if float(reconstruction.get(name, math.inf)) > 3e-6:
            raise DataUnavailable(f"ConvGRU reconstruction exceeds tolerance: {name}")


def _validate_heldout_projected_terms(
    archive: Mapping[str, np.ndarray], *, expected_rows: int
) -> None:
    required = {"scope", "metadata", "values", "value_columns"}
    if not required.issubset(archive):
        raise DataUnavailable(
            f"Held-out projected terms lack {sorted(required - set(archive))}"
        )
    if str(np.asarray(archive["scope"]).item()) != "heldout":
        raise DataUnavailable("Inference projected terms are not fold-wise held out")
    expected_value_columns = tuple(
        f"{term}__{suffix}"
        for term in GRU_PROJECTED_TERM_NAMES
        for suffix in GRU_PROJECTED_VALUE_SUFFIXES
    )
    value_columns = tuple(np.asarray(archive["value_columns"]).astype(str).tolist())
    metadata = np.asarray(archive["metadata"])
    values = np.asarray(archive["values"])
    if value_columns != expected_value_columns:
        raise DataUnavailable("Held-out projected-term semantics changed")
    if metadata.shape != (expected_rows, 7):
        raise DataUnavailable("Held-out projected-term row count differs from consolidation")
    if values.shape != (expected_rows, len(expected_value_columns)):
        raise DataUnavailable("Held-out projected-term value shape differs from its semantics")
    if not np.isfinite(values).all():
        raise DataUnavailable("Held-out projected-term values contain nonfinite entries")


def _validate_gru(root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[Path]]:
    audit_md = root / "gru_equation_audit.md"
    audit_json_path = root / "checkpoint_gru_audit.json"
    kernel_path = root / "kernel_offset_energy.csv"
    heldout_terms_path = root / "gru_projected_terms_heldout.npz"
    heldout_registration_path = root / "registration_metrics_heldout.csv"
    heldout_consolidation_path = root / "gru_instrumentation/consolidation_heldout.json"
    for path in (
        audit_md,
        audit_json_path,
        kernel_path,
        heldout_terms_path,
        heldout_registration_path,
        heldout_consolidation_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise DataUnavailable(f"Missing finalized ConvGRU product: {path}")
    audit = _read_json(audit_json_path, "checkpoint ConvGRU audit")
    if audit.get("equations_only") is not False or int(audit.get("convgru_steps", -1)) != 8:
        raise DataUnavailable("ConvGRU equation audit is not an eight-step numerical replay")
    reconstruction = audit.get("equation_reconstruction", {})
    _validate_gru_reconstruction(reconstruction)
    if audit.get("input_lag_order") != "0=current through 31=oldest":
        raise DataUnavailable("ConvGRU audit does not preserve the corrected retinal-lag convention")
    supports = audit.get("convgru_input_lag_supports")
    if supports != [[0, 17], [0, 19], [0, 21], [0, 23], [2, 25], [4, 27], [6, 29], [8, 31]]:
        raise DataUnavailable("ConvGRU internal temporal supports changed")
    audit_text = audit_md.read_text(encoding="utf-8")
    for required in ("h_t", "newer", "older", "retinal"):
        if required.lower() not in audit_text.lower():
            raise DataUnavailable(f"ConvGRU equation audit markdown lacks {required!r}")

    heldout_consolidation = _read_json(
        heldout_consolidation_path, "fold-wise held-out GRU consolidation"
    )
    if (
        heldout_consolidation.get("scope") != "heldout"
        or int(heldout_consolidation.get("registration_parts", -1)) != 48
        or int(heldout_consolidation.get("term_parts", -1)) != 48
        or Path(str(heldout_consolidation.get("registration_metrics", ""))).resolve()
        != heldout_registration_path.resolve()
        or Path(str(heldout_consolidation.get("gru_projected_terms", ""))).resolve()
        != heldout_terms_path.resolve()
    ):
        raise DataUnavailable("Fold-wise held-out GRU consolidation counts/paths are invalid")
    heldout_markers = _validate_instrumentation_markers(
        root,
        scope="heldout",
        expected_parts=48,
        consolidation=heldout_consolidation,
    )
    registration = _read_csv(heldout_registration_path, "fold-wise held-out registration metrics")
    if int(heldout_consolidation.get("registration_rows", -1)) != len(registration):
        raise DataUnavailable("Held-out registration row count differs from its consolidation marker")
    _require_columns(
        registration,
        (
            "scope",
            "contrast",
            "fold",
            "scale",
            "internal_step",
            "subspace",
            "expected_feature_shift_x_px",
            "expected_feature_shift_y_px",
            "raw_lag_x_px",
            "raw_lag_y_px",
            "recurrent_lag_x_px",
            "recurrent_lag_y_px",
            "transport_x_px",
            "transport_y_px",
            "zero_lag_alignment_improvement",
            "valid",
            "raw_at_search_boundary",
            "recurrent_at_search_boundary",
            "expected_outside_search_window",
        ),
        "registration metrics",
    )
    if set(registration.scope.astype(str)) != {"heldout"}:
        raise DataUnavailable("Registration inference is not exclusively fold-wise held out")
    if set(registration.contrast.astype(str)) != set(CONTRASTS):
        raise DataUnavailable("Canonical registration table lacks a contrast")
    if set(pd.to_numeric(registration.scale).astype(float)) != set(SCALES):
        raise DataUnavailable("Canonical registration table lacks a movement scale")
    if set(pd.to_numeric(registration.internal_step).astype(int)) != set(range(1, 8)):
        raise DataUnavailable("Registration table does not cover internal steps 1–7")
    pair_counts = (
        registration[["fold", "image_position", "trajectory_position"]]
        .drop_duplicates()
        .groupby("fold")
        .size()
    )
    if set(pair_counts.index.astype(int)) != set(FOLDS) or not np.all(pair_counts.to_numpy() == 12):
        raise DataUnavailable("Held-out registration does not contain 12 crossed pairs per fold")
    with np.load(heldout_terms_path, allow_pickle=False) as archive:
        expected_rows = int(heldout_consolidation.get("projected_term_rows", -1))
        _validate_heldout_projected_terms(archive, expected_rows=expected_rows)
    kernel = _read_csv(kernel_path, "kernel offset energy")
    _require_columns(
        kernel,
        (
            "contrast",
            "fold",
            "projector",
            "kernel",
            "input_offset_y",
            "input_offset_x",
            "energy_pp",
            "energy_pq",
            "energy_qp",
            "energy_qq",
            "projector_set_complete",
        ),
        "kernel offset energy",
    )
    if set(kernel.kernel.astype(str)) != {"candidate", "reset_gate", "update_gate"}:
        raise DataUnavailable("Kernel audit lacks candidate/reset/update recurrent kernels")
    if not kernel.projector_set_complete.astype(str).str.lower().isin(("true", "1")).all():
        raise DataUnavailable("Kernel audit reports an incomplete projector set")
    return audit, registration, kernel, [
        audit_md,
        audit_json_path,
        heldout_terms_path,
        heldout_registration_path,
        kernel_path,
        heldout_consolidation_path,
        *heldout_markers,
    ]


def _validate_static_gru_audit(root: Path) -> list[Path]:
    """Validate equation/kernel source products without requiring activation replay."""
    audit_md = root / "gru_equation_audit.md"
    audit_json_path = root / "checkpoint_gru_audit.json"
    kernel_path = root / "kernel_offset_energy.csv"
    audit = _read_json(audit_json_path, "checkpoint ConvGRU source audit")
    if int(audit.get("convgru_steps", -1)) != 8:
        raise DataUnavailable("Static ConvGRU audit does not record eight internal steps")
    reconstruction = audit.get("equation_reconstruction", {})
    _validate_gru_reconstruction(reconstruction)
    if audit.get("input_lag_order") != "0=current through 31=oldest":
        raise DataUnavailable("Static ConvGRU audit has the wrong retinal-lag direction")
    if not audit_md.is_file() or not {"newer", "older", "h_t"}.issubset(
        set(audit_md.read_text(encoding="utf-8").lower().replace("(", " ").split())
    ):
        # Use a simpler substring fallback because equations contain punctuation.
        text = audit_md.read_text(encoding="utf-8").lower() if audit_md.is_file() else ""
        if not all(value in text for value in ("newer", "older", "h_t")):
            raise DataUnavailable("Static ConvGRU equation markdown is incomplete")
    kernel = _read_csv(kernel_path, "static recurrent-kernel audit")
    _require_columns(
        kernel,
        ("kernel", "projector_set_complete", "energy_pp", "energy_qq"),
        "static recurrent-kernel audit",
    )
    if set(kernel.kernel.astype(str)) != {"candidate", "reset_gate", "update_gate"}:
        raise DataUnavailable("Static recurrent-kernel audit lacks a kernel family")
    if not kernel.projector_set_complete.astype(str).str.lower().isin(("true", "1")).all():
        raise DataUnavailable("Static recurrent-kernel audit is incomplete")
    return [audit_md, audit_json_path, kernel_path]


def _validate_figure_manifest(path: Path, *, figure_number: int, root: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = _read_json(path, f"Figure {figure_number} manifest")
    paths: list[Path] = [path]
    if manifest.get("status") != "complete":
        raise DataUnavailable(f"Figure {figure_number} is not complete")
    if figure_number == 1:
        if (
            manifest.get("saved_products_only") is not True
            or manifest.get("foldwise_heldout_inference") is not True
            or manifest.get("consensus_projectors_used_for_inference") is not False
        ):
            raise DataUnavailable("Figure 1 violates fold-wise saved-product inference")
        for record in manifest.get("source_products", []):
            paths.append(_validate_record(record, label="Figure 1 source"))
        for record in manifest.get("exact_plotting_data", {}).values():
            if isinstance(record, dict):
                paths.append(_validate_record(record, label="Figure 1 plotting data"))
        exports = manifest.get("figure_exports", {})
        if set(exports) != {"pdf", "svg", "png"}:
            raise DataUnavailable("Figure 1 lacks PDF/SVG/PNG exports")
        for record in exports.values():
            paths.append(_validate_record(record, label="Figure 1 export"))
        paths.append(_validate_record(manifest.get("caption", {}), label="Figure 1 caption"))
    else:
        if (
            manifest.get("foldwise_inference") is not True
            or manifest.get("consensus_used_for_inference") is not False
            or manifest.get("not_recurrence_across_40_outputs") is not True
            or manifest.get("boundary_peaks_treated_as_unresolved") is not True
        ):
            raise DataUnavailable("Figure 2 violates temporal/boundary inference rules")
        for record in manifest.get("source_files", []):
            paths.append(_validate_record(record, label="Figure 2 source"))
        exports = [Path(str(value)).resolve() for value in manifest.get("outputs", [])]
        if {path.suffix for path in exports} != {".pdf", ".svg", ".png"}:
            raise DataUnavailable("Figure 2 lacks PDF/SVG/PNG exports")
        for output in exports:
            if not output.is_file() or output.parent != path.parent.resolve():
                raise DataUnavailable(f"Missing Figure 2 export: {output}")
            paths.append(output)
        plot_data = [Path(str(value)).resolve() for value in manifest.get("plot_data", [])]
        expected_names = {
            "panel_b_transport_statistics.csv",
            "panel_c_alignment_by_scale.csv",
            "panel_d_residual_vs_ssi.csv",
            "panel_d_descriptive_statistics.csv",
            "boundary_resolution_counts.csv",
        }
        if not expected_names.issubset({value.name for value in plot_data}):
            raise DataUnavailable("Figure 2 lacks required exact plotting tables")
        for output in plot_data:
            if not output.is_file() or output.parent != (path.parent / "plot_data").resolve():
                raise DataUnavailable(f"Missing Figure 2 plotting product: {output}")
            paths.append(output)
        caption = path.parent / "caption_draft.md"
        if not caption.is_file():
            raise DataUnavailable("Figure 2 caption draft is missing")
        paths.append(caption)
    return manifest, paths


def validate_causal_tables(
    transport: pd.DataFrame,
    realignment: pd.DataFrame,
    *,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = (
        "schema_version",
        "scope",
        "analysis",
        "condition",
        "population",
    )
    _require_columns(
        transport,
        (
            *common,
            "scale",
            "exact_ssi_bits",
            "contrast",
            "scale_a",
            "scale_b",
            "scale_a_endpoint_condition",
            "scale_b_endpoint_condition",
            "contrast_endpoint_rule",
            "exact_ssi_change_b_minus_a_bits",
        ),
        "transport ablation",
    )
    _require_columns(
        realignment,
        (
            *common,
            "scale",
            "exact_ssi_bits",
            "contrast",
            "scale_a",
            "scale_b",
            "scale_a_endpoint_condition",
            "scale_b_endpoint_condition",
            "contrast_endpoint_rule",
            "exact_ssi_change_b_minus_a_bits",
        ),
        "realignment",
    )
    for frame, label in ((transport, "transport"), (realignment, "realignment")):
        if set(frame.scope.astype(str)) != {scope}:
            raise DataUnavailable(f"{label} table mixes intervention scopes")
        if set(frame.schema_version.astype(str)) != {CAUSAL_SCHEMA}:
            raise DataUnavailable(f"{label} table has the wrong schema")

    transport_dose = transport.loc[transport.analysis.eq("transport_ablation_dose_curve")].copy()
    transport_contrast = transport.loc[transport.analysis.eq("transport_ablation_contrast")].copy()
    if len(transport_dose) != len(TRANSPORT_CONDITIONS) * len(SCALES) * len(POPULATIONS):
        raise DataUnavailable("Transport dose table is not exactly 8×5×3")
    if len(transport_contrast) != len(TRANSPORT_CONDITIONS) * 3:
        raise DataUnavailable("Transport contrast table is not exactly 8×3")
    if set(transport_dose.condition.astype(str)) != set(TRANSPORT_CONDITIONS):
        raise DataUnavailable("Transport intervention conditions changed")
    if set(pd.to_numeric(transport_dose.scale).astype(float)) != set(SCALES):
        raise DataUnavailable("Transport movement scales changed")
    if set(transport_dose.population.astype(str)) != set(POPULATIONS):
        raise DataUnavailable("Transport populations changed")
    if transport_dose.duplicated(["condition", "scale", "population"]).any():
        raise DataUnavailable("Transport dose table contains duplicate cells")
    if set(transport_contrast.contrast.astype(str)) != set(CAUSAL_CONTRAST.values()):
        raise DataUnavailable("Transport historical contrast labels changed")
    observed_transport_keys = {
        (str(row.condition), str(row.contrast)) for row in transport_contrast.itertuples()
    }
    expected_transport_keys = {
        (condition, contrast)
        for condition in TRANSPORT_CONDITIONS
        for contrast in CAUSAL_CONTRAST.values()
    }
    if observed_transport_keys != expected_transport_keys or transport_contrast.duplicated(
        ["condition", "contrast"]
    ).any():
        raise DataUnavailable("Transport contrast table is not an exact condition×contrast grid")
    contrast_spec = {
        CAUSAL_CONTRAST["low_0_to_2"]: ("lower_sf_71", 0.0, 2.0),
        CAUSAL_CONTRAST["high_0_to_1"]: ("higher_sf_29", 0.0, 1.0),
        CAUSAL_CONTRAST["high_1_to_3"]: ("higher_sf_29", 1.0, 3.0),
    }
    dose_lookup = transport_dose.set_index(["condition", "population", "scale"])[
        "exact_ssi_bits"
    ]
    for row in transport_contrast.itertuples():
        population, scale_a, scale_b = contrast_spec[str(row.contrast)]
        if (
            str(row.population) != population
            or not np.isclose(float(row.scale_a), scale_a)
            or not np.isclose(float(row.scale_b), scale_b)
        ):
            raise DataUnavailable("Transport contrast population/endpoints changed")
        if (
            str(row.scale_a_endpoint_condition) != str(row.condition)
            or str(row.scale_b_endpoint_condition) != str(row.condition)
            or str(row.contrast_endpoint_rule)
            != "same intervention condition at both endpoints"
        ):
            raise DataUnavailable("Transport contrast endpoint-condition identities changed")
        expected_change = float(dose_lookup.loc[(str(row.condition), population, scale_b)]) - float(
            dose_lookup.loc[(str(row.condition), population, scale_a)]
        )
        if not np.isclose(
            float(row.exact_ssi_change_b_minus_a_bits), expected_change, rtol=1e-9, atol=1e-12
        ):
            raise DataUnavailable("Transport contrast does not equal its saved exact SSI endpoints")

    realign_dose = realignment.loc[realignment.analysis.eq("realignment_dose_curve")].copy()
    realign_contrast = realignment.loc[realignment.analysis.eq("realignment_contrast")].copy()
    if len(realign_dose) != len(REALIGNMENT_CONDITIONS) * 2 * len(POPULATIONS):
        raise DataUnavailable("Realignment dose table is not exactly 7×2×3")
    if len(realign_contrast) != len(REALIGNMENT_CONDITIONS):
        raise DataUnavailable("Realignment contrast table is not exactly seven 1×→3× rows")
    if set(realign_dose.condition.astype(str)) != set(REALIGNMENT_CONDITIONS):
        raise DataUnavailable("Realignment conditions changed")
    if set(pd.to_numeric(realign_dose.scale).astype(float)) != {1.0, 3.0}:
        raise DataUnavailable("Realignment scales changed")
    if set(realign_dose.population.astype(str)) != set(POPULATIONS):
        raise DataUnavailable("Realignment populations changed")
    if realign_dose.duplicated(["condition", "scale", "population"]).any():
        raise DataUnavailable("Realignment dose table contains duplicate cells")
    for row in realign_dose.itertuples():
        is_run = _realignment_cell_is_run(str(row.condition), float(row.scale))
        finite_values = (
            np.isfinite(float(row.exact_ssi_bits)),
            np.isfinite(float(row.expected_spikes)),
        )
        if is_run and not all(finite_values):
            raise DataUnavailable("A scheduled realignment dose cell is nonfinite")
        if not is_run and (
            np.isfinite(float(row.exact_ssi_bits))
            or not np.isclose(float(row.expected_spikes), 0.0)
        ):
            raise DataUnavailable(
                "A structurally unrun realignment dose cell is not encoded as NaN SSI/zero spikes"
            )
    if set(realign_contrast.contrast.astype(str)) != {CAUSAL_CONTRAST["high_1_to_3"]}:
        raise DataUnavailable("Realignment direct 1×→3× contrast label changed")
    observed_realign_keys = {
        (str(row.condition), str(row.contrast)) for row in realign_contrast.itertuples()
    }
    expected_realign_keys = {
        (condition, CAUSAL_CONTRAST["high_1_to_3"])
        for condition in REALIGNMENT_CONDITIONS
    }
    if observed_realign_keys != expected_realign_keys or realign_contrast.duplicated(
        ["condition", "contrast"]
    ).any():
        raise DataUnavailable("Realignment contrast table is not an exact condition grid")
    realign_lookup = realign_dose.set_index(["condition", "population", "scale"])[
        "exact_ssi_bits"
    ]
    for row in realign_contrast.itertuples():
        if (
            str(row.population) != "higher_sf_29"
            or not np.isclose(float(row.scale_a), 1.0)
            or not np.isclose(float(row.scale_b), 3.0)
        ):
            raise DataUnavailable("Realignment contrast population/endpoints changed")
        condition = str(row.condition)
        expected_a_condition = (
            "no_shift_intact"
            if condition not in {"no_shift_intact", "induced_eye_misalignment_candidate_p"}
            else condition
        )
        expected_b_condition = condition
        if (
            str(row.scale_a_endpoint_condition) != expected_a_condition
            or str(row.scale_b_endpoint_condition) != expected_b_condition
        ):
            raise DataUnavailable("Realignment contrast endpoint-condition identities changed")
        expected_rule = (
            "same intervention condition at both endpoints"
            if expected_a_condition == expected_b_condition
            else f"scale_a uses {expected_a_condition}; scale_b uses {expected_b_condition}"
        )
        if str(row.contrast_endpoint_rule) != expected_rule:
            raise DataUnavailable("Realignment contrast endpoint rule changed")
        if condition == "induced_eye_misalignment_candidate_p":
            if np.isfinite(float(row.exact_ssi_change_b_minus_a_bits)):
                raise DataUnavailable(
                    "Induced-misalignment 1× test must not be recast as a 1×→3× rescue contrast"
                )
            continue
        expected_change = float(
            realign_lookup.loc[(expected_b_condition, "higher_sf_29", 3.0)]
        ) - float(
            realign_lookup.loc[(expected_a_condition, "higher_sf_29", 1.0)]
        )
        if not np.isclose(
            float(row.exact_ssi_change_b_minus_a_bits), expected_change, rtol=1e-9, atol=1e-12
        ):
            raise DataUnavailable("Realignment contrast does not equal its saved exact SSI endpoints")
    return transport, realignment


def validate_causal_archive(
    archive: Mapping[str, np.ndarray],
    *,
    stage: str,
) -> dict[str, np.ndarray]:
    conditions = TRANSPORT_CONDITIONS if stage == "transport" else REALIGNMENT_CONDITIONS
    scales = SCALES if stage == "transport" else (1.0, 3.0)
    required = {
        "schema_version",
        "conditions",
        "scales",
        "populations",
        "normalized_population_rate_maps",
        "map_definition",
        "individual_conditions_rescaled",
    }
    if not required.issubset(archive):
        raise DataUnavailable(f"{stage} archive lacks {sorted(required - set(archive))}")
    if str(np.asarray(archive["schema_version"]).item()) != CAUSAL_SCHEMA:
        raise DataUnavailable(f"{stage} archive has the wrong schema")
    if tuple(np.asarray(archive["conditions"]).astype(str).tolist()) != conditions:
        raise DataUnavailable(f"{stage} archive condition axis changed")
    if not np.allclose(np.asarray(archive["scales"], dtype=float), scales):
        raise DataUnavailable(f"{stage} archive scale axis changed")
    if tuple(np.asarray(archive["populations"]).astype(str).tolist()) != POPULATIONS:
        raise DataUnavailable(f"{stage} archive population axis changed")
    maps = np.asarray(archive["normalized_population_rate_maps"], dtype=float)
    expected_shape = (len(conditions), len(scales), len(POPULATIONS), 51, 51)
    if maps.shape != expected_shape:
        raise DataUnavailable(f"{stage} normalized-map array has wrong shape {expected_shape}")
    if stage == "transport":
        if not np.isfinite(maps).all():
            raise DataUnavailable("Transport normalized-map array contains nonfinite values")
    else:
        for condition_index, condition in enumerate(conditions):
            for scale_index, scale in enumerate(scales):
                cell = maps[condition_index, scale_index]
                if _realignment_cell_is_run(condition, float(scale)):
                    if not np.isfinite(cell).all():
                        raise DataUnavailable(
                            f"Scheduled realignment map cell is nonfinite: {condition}/{scale}"
                        )
                elif not np.isnan(cell).all():
                    raise DataUnavailable(
                        f"Structurally unrun realignment map cell is not all-NaN: {condition}/{scale}"
                    )
    rescaled = np.asarray(archive["individual_conditions_rescaled"])
    if bool(rescaled.item()):
        raise DataUnavailable(f"{stage} archive independently rescales conditions")
    definition = str(np.asarray(archive["map_definition"]).item())
    if "g(x,y)=r(x,y)/mean_xy r" not in definition:
        raise DataUnavailable(f"{stage} archive lacks the exact normalized-map definition")
    return {key: np.asarray(value) for key, value in archive.items()}


def _load_causal(root: Path) -> CausalProducts:
    causal_root = root / "causal_interventions"
    completed: list[tuple[str, Path, dict[str, Any]]] = []
    for scope in ("full", "pilot"):
        manifest_path = causal_root / scope / "analysis_manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path, f"{scope} causal manifest")
            if manifest.get("status") == "complete" and manifest.get("scope") == scope:
                completed.append((scope, manifest_path, manifest))
    if not completed:
        raise DataUnavailable("Neither a complete full nor complete preregistered-pilot causal analysis exists")
    scope, manifest_path, manifest = completed[0]  # full is deliberately preferred
    if manifest.get("schema_version") != CAUSAL_SCHEMA:
        raise DataUnavailable("Causal manifest schema changed")
    if (
        manifest.get("exact_corrected_renderer") is not True
        or manifest.get("foldwise_heldout_projectors_for_inference") is not True
        or manifest.get("consensus_projectors_for_visualization_only") is not True
        or manifest.get("no_unique_recurrent_pathway_claimed") is not True
    ):
        raise DataUnavailable("Causal manifest violates a frozen-model inference boundary")
    direction = str(manifest.get("within_window_convgru_order", ""))
    if "newer" not in direction or "older" not in direction or "reset" not in direction:
        raise DataUnavailable("Causal manifest loses the within-window backward-time convention")
    expected_parts = {"pilot": (48, 12), "full": (192, 48)}[scope]
    source_paths: list[Path] = [manifest_path]
    for stage, expected in zip(("transport", "realignment"), expected_parts):
        stage_manifest = manifest.get(stage)
        if not isinstance(stage_manifest, dict) or stage_manifest.get("status") != "complete":
            raise DataUnavailable(f"Causal {stage} consolidation is incomplete")
        if int(stage_manifest.get("parts", -1)) != expected or int(stage_manifest.get("expected_parts", -1)) != expected:
            raise DataUnavailable(f"Causal {stage} part count is not the frozen {scope} design")
        if float(stage_manifest.get("identity_max_relative_error", math.inf)) > 3e-3:
            raise DataUnavailable(f"Causal {stage} intact replay exceeds numerical tolerance")
        products = stage_manifest.get("products")
        if not isinstance(products, list) or len(products) < 2:
            raise DataUnavailable(f"Causal {stage} manifest lacks product hashes")
        for record in products:
            source_paths.append(_validate_record(record, label=f"causal {stage} product", root=causal_root))
        marker_paths = stage_manifest.get("part_markers")
        if not isinstance(marker_paths, list) or len(marker_paths) != expected:
            raise DataUnavailable(f"Causal {stage} marker count is incomplete")
        for raw_marker in marker_paths:
            marker_path = _path_within(Path(str(raw_marker)), causal_root, f"causal {stage} marker")
            marker = _read_json(marker_path, f"causal {stage} part marker")
            if (
                marker.get("complete") is not True
                or marker.get("schema_version") != CAUSAL_SCHEMA
                or marker.get("stage") != stage
                or marker.get("scope") != scope
            ):
                raise DataUnavailable(f"Malformed causal part marker: {marker_path}")
            for record in marker.get("products", []):
                _validate_record(record, label=f"causal {stage} part payload", root=causal_root)
            source_paths.append(marker_path)

    scope_root = causal_root / scope
    transport_path = scope_root / "transport_ablation_results.csv"
    realign_path = scope_root / "realignment_results.csv"
    transport_maps_path = scope_root / "transport_ablation_maps.npz"
    realign_maps_path = scope_root / "realignment_maps.npz"
    transport = _read_csv(transport_path, "transport ablation results")
    realignment = _read_csv(realign_path, "realignment results")
    validate_causal_tables(transport, realignment, scope=scope)
    try:
        with np.load(transport_maps_path, allow_pickle=False) as archive:
            transport_archive = validate_causal_archive(
                {key: np.asarray(archive[key]) for key in archive.files}, stage="transport"
            )
        with np.load(realign_maps_path, allow_pickle=False) as archive:
            realignment_archive = validate_causal_archive(
                {key: np.asarray(archive[key]) for key in archive.files}, stage="realignment"
            )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, DataUnavailable):
            raise
        raise DataUnavailable("Unreadable causal normalized-map archives") from error
    source_paths.extend((transport_path, realign_path, transport_maps_path, realign_maps_path))

    # Canonical products must be byte-for-byte copies of the preferred scope.
    for canonical, scoped in (
        (root / "transport_ablation_results.csv", transport_path),
        (root / "realignment_results.csv", realign_path),
        (root / "transport_ablation_maps.npz", transport_maps_path),
        (root / "realignment_maps.npz", realign_maps_path),
    ):
        if not canonical.is_file() or sha256_file(canonical) != sha256_file(scoped):
            raise DataUnavailable(f"Canonical causal product is not the preferred {scope} product: {canonical}")
        source_paths.append(canonical)
    pilot_gate_path = causal_root / "pilot/pilot_to_full_gate.json"
    pilot_gate = _read_json(pilot_gate_path, "pilot-to-full gate") if pilot_gate_path.is_file() else None
    if scope == "full":
        if pilot_gate is None or pilot_gate.get("advance_to_full") is not True:
            raise DataUnavailable("Full confirmation lacks an affirmative preregistered pilot gate")
        source_paths.append(pilot_gate_path)
    return CausalProducts(
        scope=scope,
        root=scope_root,
        analysis_manifest=manifest,
        pilot_gate=pilot_gate,
        transport=transport,
        realignment=realignment,
        transport_archive=transport_archive,
        realignment_archive=realignment_archive,
        source_paths=tuple(source_paths),
    )


def load_final_inputs(
    root: Path,
    *,
    figure1_manifest: Path,
    figure2_manifest: Path,
) -> FinalInputs:
    root = Path(root).resolve()
    for name in REQUIRED_CANONICAL:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise DataUnavailable(f"Missing required canonical output {name}: {path}")
    rank_gate, rank_results, stability, rank_paths = _validate_rank8(root)
    leverage, variance, readout, per_unit, probes, pq_paths = _validate_pq(root)
    gru_audit, registration, kernel, gru_paths = _validate_gru(root)
    figure1, figure1_paths = _validate_figure_manifest(
        Path(figure1_manifest).resolve(), figure_number=1, root=root
    )
    figure2, figure2_paths = _validate_figure_manifest(
        Path(figure2_manifest).resolve(), figure_number=2, root=root
    )
    causal = _load_causal(root)
    sources = tuple(
        sorted(
            {
                *rank_paths,
                *pq_paths,
                *gru_paths,
                *figure1_paths,
                *figure2_paths,
                *causal.source_paths,
            },
            key=lambda value: str(value),
        )
    )
    return FinalInputs(
        root=root,
        rank_gate=rank_gate,
        rank_results=rank_results,
        stability=stability,
        leverage=leverage,
        variance=variance,
        readout=readout,
        per_unit=per_unit,
        probes=probes,
        gru_audit=gru_audit,
        registration=registration,
        kernel=kernel,
        figure1_manifest=figure1,
        figure2_manifest=figure2,
        causal=causal,
        source_paths=sources,
    )


def load_upstream_inputs(root: Path, *, figure1_manifest: Path) -> UpstreamInputs:
    """Load the complete rank/PQ products needed to apply the stop rule."""
    root = Path(root).resolve()
    rank_gate, rank_results, stability, rank_paths = _validate_rank8(root)
    leverage, variance, readout, per_unit, probes, pq_paths = _validate_pq(root)
    figure1, figure1_paths = _validate_figure_manifest(
        Path(figure1_manifest).resolve(), figure_number=1, root=root
    )
    static_gru_paths = _validate_static_gru_audit(root)
    sources: set[Path] = {*rank_paths, *pq_paths, *figure1_paths, *static_gru_paths}
    return UpstreamInputs(
        root=root,
        rank_gate=rank_gate,
        rank_results=rank_results,
        stability=stability,
        leverage=leverage,
        variance=variance,
        readout=readout,
        per_unit=per_unit,
        probes=probes,
        figure1_manifest=figure1,
        source_paths=tuple(sorted(sources, key=lambda value: str(value))),
    )


def defining_motion_enrichment(
    variance: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int], bool]:
    """Apply the preregistered all-three defining-scale P/Q gate."""
    rows = variance.loc[variance.analysis.eq("movement_change_from_stabilization")].copy()
    medians: dict[str, float] = {}
    fold_counts: dict[str, int] = {}
    for contrast in CONTRASTS:
        selected = rows.loc[
            rows.contrast.eq(contrast)
            & np.isclose(pd.to_numeric(rows.scale, errors="coerce"), TARGET_SCALE[contrast])
        ]
        expected_folds = set(FOLDS)
        observed_folds = set(pd.to_numeric(selected.fold, errors="coerce").dropna().astype(int))
        if (
            len(selected) != len(FOLDS)
            or observed_folds != expected_folds
            or selected.duplicated(["fold"]).any()
        ):
            raise DataUnavailable(
                f"Defining P/Q motion gate lacks exactly one row for every fold: {contrast}"
            )
        values = pd.to_numeric(
            selected.p_to_q_per_dimension_energy_ratio, errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(values).all():
            raise DataUnavailable(
                f"Defining P/Q motion gate contains a nonfinite fold value: {contrast}"
            )
        medians[contrast] = float(np.median(values))
        fold_counts[contrast] = len(values)
    passed = bool(
        all(fold_counts[key] == 4 for key in CONTRASTS)
        and all(np.isfinite(medians[key]) and medians[key] > 1.0 for key in CONTRASTS)
    )
    return medians, fold_counts, passed


def _dose_value(frame: pd.DataFrame, condition: str, scale: float, population: str) -> float:
    selected = frame.loc[
        frame.condition.eq(condition)
        & np.isclose(pd.to_numeric(frame.scale, errors="coerce"), float(scale))
        & frame.population.eq(population)
        & frame.analysis.str.endswith("_dose_curve"),
        "exact_ssi_bits",
    ]
    if len(selected) != 1 or not np.isfinite(float(selected.iloc[0])):
        raise DataUnavailable(f"Missing unique exact SSI for {condition}/{scale}/{population}")
    return float(selected.iloc[0])


def _contrast_value(frame: pd.DataFrame, condition: str, contrast: str) -> float:
    contrast = CAUSAL_CONTRAST.get(contrast, contrast)
    selected = frame.loc[
        frame.condition.eq(condition)
        & frame.contrast.eq(contrast)
        & frame.analysis.str.endswith("_contrast"),
        "exact_ssi_change_b_minus_a_bits",
    ]
    if len(selected) != 1 or not np.isfinite(float(selected.iloc[0])):
        raise DataUnavailable(f"Missing unique exact SSI contrast for {condition}/{contrast}")
    return float(selected.iloc[0])


def prepare_figure3_data(causal: CausalProducts) -> dict[str, Any]:
    transport = causal.transport
    realignment = causal.realignment
    panel_a = transport.loc[
        transport.analysis.eq("transport_ablation_dose_curve")
        & transport.condition.eq("intact")
        & transport.population.isin(("lower_sf_71", "higher_sf_29")),
        ["scope", "condition", "population", "scale", "exact_ssi_bits", "expected_spikes"],
    ].copy()
    if len(panel_a) != 10:
        raise DataUnavailable("Figure 3A does not contain two intact five-scale curves")

    center_conditions = (
        "candidate_recurrent_center_only",
        "gate_recurrent_center_only",
        "all_recurrent_center_only",
    )
    offset_conditions = tuple(
        condition for condition in TRANSPORT_CONDITIONS if "offset_permuted" in condition
    )
    panel_b_source = transport.loc[
        transport.analysis.eq("transport_ablation_dose_curve")
        & transport.population.isin(("lower_sf_71", "higher_sf_29"))
        & transport.condition.isin(("intact", *center_conditions, *offset_conditions)),
        ["scope", "condition", "population", "scale", "exact_ssi_bits", "expected_spikes"],
    ].copy()
    if len(panel_b_source) != (1 + 3 + 3) * 2 * 5:
        raise DataUnavailable("Figure 3B transport-control cells are incomplete")
    baselines = panel_b_source.loc[np.isclose(panel_b_source.scale, 0.0), ["condition", "population", "exact_ssi_bits"]]
    baselines = baselines.rename(columns={"exact_ssi_bits": "stabilized_ssi_bits"})
    panel_b_source = panel_b_source.merge(
        baselines,
        on=["condition", "population"],
        validate="many_to_one",
    )
    panel_b_source["ssi_change_from_own_stabilization_bits"] = (
        panel_b_source.exact_ssi_bits - panel_b_source.stabilized_ssi_bits
    )
    offset = panel_b_source.loc[panel_b_source.condition.isin(offset_conditions)]
    offset_mean = (
        offset.groupby(["scope", "population", "scale"], as_index=False)
        .agg(
            exact_ssi_bits=("exact_ssi_bits", "mean"),
            stabilized_ssi_bits=("stabilized_ssi_bits", "mean"),
            ssi_change_from_own_stabilization_bits=("ssi_change_from_own_stabilization_bits", "mean"),
            offset_seed_sd_bits=("ssi_change_from_own_stabilization_bits", "std"),
            n_offset_permutations=("condition", "size"),
        )
    )
    offset_mean["condition"] = "offset_permuted_mean"
    panel_b = pd.concat(
        [panel_b_source.loc[~panel_b_source.condition.isin(offset_conditions)], offset_mean],
        ignore_index=True,
        sort=False,
    )

    c_conditions = (
        "no_shift_intact",
        "eye_correct_candidate_p",
        "eye_opposite_candidate_p",
        "eye_random_matched_candidate_p",
        "eye_correct_complementary_q",
    )
    panel_c = realignment.loc[
        realignment.analysis.eq("realignment_dose_curve")
        & realignment.population.eq("higher_sf_29")
        & np.isclose(realignment.scale, 3.0)
        & realignment.condition.isin(c_conditions),
        ["scope", "condition", "population", "scale", "exact_ssi_bits", "expected_spikes"],
    ].copy()
    panel_c["condition"] = pd.Categorical(panel_c.condition, categories=c_conditions, ordered=True)
    panel_c = panel_c.sort_values("condition").reset_index(drop=True)
    if len(panel_c) != 5:
        raise DataUnavailable("Figure 3C lacks a high-SF 3× realignment control")
    c_intact_ssi = float(
        panel_c.loc[panel_c.condition.astype(str).eq("no_shift_intact"), "exact_ssi_bits"].iloc[0]
    )
    panel_c["delta_ssi_from_intact_bits"] = panel_c.exact_ssi_bits - c_intact_ssi
    panel_c["delta_ssi_from_intact_microbits"] = (
        1e6 * panel_c.delta_ssi_from_intact_bits
    )

    d_conditions = ("no_shift_intact", "induced_eye_misalignment_candidate_p")
    panel_d = realignment.loc[
        realignment.analysis.eq("realignment_dose_curve")
        & realignment.population.eq("higher_sf_29")
        & np.isclose(realignment.scale, 1.0)
        & realignment.condition.isin(d_conditions),
        ["scope", "condition", "population", "scale", "exact_ssi_bits", "expected_spikes"],
    ].copy()
    panel_d["condition"] = pd.Categorical(panel_d.condition, categories=d_conditions, ordered=True)
    panel_d = panel_d.sort_values("condition").reset_index(drop=True)
    if len(panel_d) != 2:
        raise DataUnavailable("Figure 3D lacks intact and induced-P-misalignment cells")
    d_intact_ssi = float(
        panel_d.loc[panel_d.condition.astype(str).eq("no_shift_intact"), "exact_ssi_bits"].iloc[0]
    )
    panel_d["delta_ssi_from_intact_bits"] = panel_d.exact_ssi_bits - d_intact_ssi
    panel_d["delta_ssi_from_intact_microbits"] = (
        1e6 * panel_d.delta_ssi_from_intact_bits
    )

    archive = causal.realignment_archive
    condition_axis = np.asarray(archive["conditions"]).astype(str)
    scale_axis = np.asarray(archive["scales"], dtype=float)
    population_axis = np.asarray(archive["populations"]).astype(str)
    all_maps = np.asarray(archive["normalized_population_rate_maps"], dtype=float)

    def get_map(condition: str, scale: float) -> np.ndarray:
        ci = int(np.flatnonzero(condition_axis == condition)[0])
        si = int(np.flatnonzero(np.isclose(scale_axis, scale))[0])
        pi = int(np.flatnonzero(population_axis == "higher_sf_29")[0])
        return all_maps[ci, si, pi]

    panel_c_maps = np.stack([get_map(condition, 3.0) for condition in c_conditions])
    panel_d_maps = np.stack([get_map(condition, 1.0) for condition in d_conditions])
    selected_maps = np.concatenate([panel_c_maps, panel_d_maps], axis=0)
    common_deviation = float(np.nanmax(np.abs(selected_maps - 1.0)))
    if not np.isfinite(common_deviation) or common_deviation <= 0:
        raise DataUnavailable("Figure 3 maps have no finite normalized spatial contrast")
    return {
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_b_source": panel_b_source,
        "panel_c": panel_c,
        "panel_d": panel_d,
        "panel_c_conditions": c_conditions,
        "panel_d_conditions": d_conditions,
        "panel_c_maps": panel_c_maps,
        "panel_d_maps": panel_d_maps,
        "common_map_vmin": 1.0 - common_deviation,
        "common_map_vmax": 1.0 + common_deviation,
    }


def _configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", color=GRID_COLOR, lw=0.5, zorder=0)


def _panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(-0.13, 1.17, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")
    axis.text(0.0, 1.17, title, transform=axis.transAxes, fontsize=9.2, fontweight="bold", va="top")


def draw_figure3(
    data: Mapping[str, Any], *, scope: str, decision: str | None = None
) -> plt.Figure:
    _configure_plot()
    figure = plt.figure(figsize=(12.9, 9.1), facecolor="white")
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.88, 1.42),
        height_ratios=(0.92, 1.35),
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.91,
        hspace=0.48,
        wspace=0.30,
    )
    axis_a = figure.add_subplot(outer[0, 0])
    b_grid = outer[0, 1].subgridspec(1, 2, wspace=0.27)
    axes_b = [figure.add_subplot(b_grid[0, index]) for index in range(2)]
    c_grid = outer[1, 0].subgridspec(2, 5, height_ratios=(0.72, 1.0), hspace=0.20, wspace=0.08)
    axis_c = figure.add_subplot(c_grid[0, :])
    axes_c_maps = [figure.add_subplot(c_grid[1, index]) for index in range(5)]
    d_grid = outer[1, 1].subgridspec(2, 4, height_ratios=(0.72, 1.0), hspace=0.20, wspace=0.14)
    axis_d = figure.add_subplot(d_grid[0, :])
    axes_d_maps = [figure.add_subplot(d_grid[1, 1 + index]) for index in range(2)]

    # A: unmodified exact dose curves.
    panel_a = data["panel_a"]
    for population, label, color in (
        ("lower_sf_71", "lower SF (71)", LOW_COLOR),
        ("higher_sf_29", "higher SF (29)", HIGH_COLOR),
    ):
        part = panel_a.loc[panel_a.population.eq(population)].sort_values("scale")
        axis_a.plot(part.scale, part.exact_ssi_bits, color=color, marker="o", lw=1.7, ms=4.0)
        last = part.iloc[-1]
        axis_a.annotate(
            label,
            (float(last.scale), float(last.exact_ssi_bits)),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            va="center",
            fontsize=7.0,
        )
    axis_a.axvline(1.0, color=MUTED_COLOR, lw=0.7, ls=":")
    axis_a.text(1.0, axis_a.get_ylim()[1], " measured", color=MUTED_COLOR, va="top", fontsize=6.2)
    axis_a.set(xlabel="retinal-motion amplitude (× measured FEM)", ylabel="exact SSI (bits)", xticks=SCALES)
    _clean_axis(axis_a)
    _panel_label(axis_a, "A", "Intact movement-scale dependence")

    # B: effects relative to each intervention's own stabilization.
    styles = {
        "intact": ("Intact", "-", 1.9, 1.0),
        "candidate_recurrent_center_only": ("candidate center-only", "--", 1.1, 0.80),
        "gate_recurrent_center_only": ("gates center-only", ":", 1.3, 0.80),
        "all_recurrent_center_only": ("all center-only", "-.", 1.3, 0.90),
        "offset_permuted_mean": ("offset-permuted mean", (0, (3, 1, 1, 1)), 1.2, 0.67),
    }
    panel_b = data["panel_b"]
    for axis, population, title, color in (
        (axes_b[0], "lower_sf_71", "lower SF (71)", LOW_COLOR),
        (axes_b[1], "higher_sf_29", "higher SF (29)", HIGH_COLOR),
    ):
        label_positions: list[tuple[float, str, str, float]] = []
        for condition, (label, line_style, width, alpha) in styles.items():
            part = panel_b.loc[
                panel_b.population.eq(population) & panel_b.condition.eq(condition)
            ].sort_values("scale")
            if len(part) != 5:
                raise DataUnavailable(f"Figure 3B lacks {condition}/{population}")
            axis.plot(
                part.scale,
                part.ssi_change_from_own_stabilization_bits,
                color=color if condition == "intact" else CONTROL_COLOR,
                ls=line_style,
                lw=width,
                alpha=alpha,
                marker="o" if condition in ("intact", "all_recurrent_center_only") else None,
                ms=3.0,
            )
            label_positions.append(
                (
                    float(part.iloc[-1].ssi_change_from_own_stabilization_bits),
                    label,
                    color if condition == "intact" else CONTROL_COLOR,
                    alpha,
                )
            )
        axis.axhline(0, color=GRID_COLOR, lw=0.7)
        axis.set(xlabel="motion amplitude (×)", xticks=SCALES, title=title)
        _clean_axis(axis)
        # Direct labels are placed as a compact ordered list inside the axes;
        # line styles remain the visual key and no detached legend is needed.
        for index, (_, label, label_color, alpha) in enumerate(
            sorted(label_positions, key=lambda value: -value[0])
        ):
            axis.text(
                0.02,
                0.97 - 0.095 * index,
                label,
                transform=axis.transAxes,
                color=label_color,
                alpha=alpha,
                fontsize=5.9,
                va="top",
            )
    axes_b[0].set_ylabel("ΔSSI from own stabilization (bits)")
    intact_b = panel_b.loc[panel_b.condition.eq("intact"), ["population", "scale", "ssi_change_from_own_stabilization_bits"]].rename(
        columns={"ssi_change_from_own_stabilization_bits": "intact_change"}
    )
    compared_b = panel_b.merge(intact_b, on=["population", "scale"], validate="many_to_one")
    compared_b["absolute_change_from_intact"] = np.abs(
        compared_b.ssi_change_from_own_stabilization_bits - compared_b.intact_change
    )
    center_changed = bool(
        compared_b.loc[
            compared_b.condition.astype(str).str.contains("center_only"),
            "absolute_change_from_intact",
        ].max()
        >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
    )
    offset_changed = bool(
        compared_b.loc[
            compared_b.condition.eq("offset_permuted_mean"),
            "absolute_change_from_intact",
        ].max()
        >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
    )
    if center_changed and not offset_changed:
        b_title = "Center-only recurrence changes SSI; offset permutation does not"
    elif center_changed and offset_changed:
        b_title = "Center-only and offset-permuted recurrence change SSI"
    elif offset_changed:
        b_title = "Offset-permuted recurrence changes SSI; center-only does not"
    else:
        b_title = "Neither recurrence control materially changes SSI"
    _panel_label(
        axes_b[0],
        "B",
        b_title,
    )

    # C: eye-derived realignment (oracle deliberately omitted from the main panel).
    c_labels = ("intact", "correct P", "opposite P", "random P", "correct Q")
    c_colors = (HIGH_COLOR, CORRECT_COLOR, BAD_COLOR, BAD_COLOR, CONTROL_COLOR)
    panel_c = data["panel_c"]
    x_c = np.arange(len(panel_c))
    axis_c.axhline(0, color=GRID_COLOR, lw=0.8)
    for index, row in panel_c.iterrows():
        value = float(row.delta_ssi_from_intact_microbits)
        axis_c.vlines(index, 0, value, color=c_colors[index], lw=1.4, zorder=1)
        axis_c.scatter(index, value, s=34, color=c_colors[index], zorder=2)
        axis_c.annotate(
            f"{value:+.1f}",
            (index, value),
            xytext=(0, 4 if value >= 0 else -10),
            textcoords="offset points",
            fontsize=5.8,
            ha="center",
        )
    axis_c.set(
        xticks=x_c,
        xticklabels=c_labels,
        ylabel="ΔSSI from intact at 3× (µbits)",
    )
    axis_c.tick_params(axis="x", labelrotation=23)
    _clean_axis(axis_c)
    correct_p_gain = float(
        panel_c.loc[
            panel_c.condition.astype(str).eq("eye_correct_candidate_p"),
            "delta_ssi_from_intact_bits",
        ].iloc[0]
    )
    _panel_label(
        axis_c,
        "C",
        (
            "P realignment rescues higher-SF 3× SSI"
            if correct_p_gain >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
            else "P realignment does not rescue higher-SF 3× SSI"
        ),
    )
    vmin, vmax = float(data["common_map_vmin"]), float(data["common_map_vmax"])
    image_handle = None
    for axis, array, label, exact_ssi in zip(
        axes_c_maps, data["panel_c_maps"], c_labels, panel_c.exact_ssi_bits
    ):
        image_handle = axis.imshow(array, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        axis.set_title(label, fontsize=5.8, pad=2)
        axis.set_xlabel(f"SSI {float(exact_ssi):.6f}", fontsize=5.3, labelpad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color(GRID_COLOR)

    # D: induced-misalignment null result, with complete maps on the same C/D scale.
    d_labels = ("intact at 1×", "induced P misalignment")
    panel_d = data["panel_d"]
    x_d = np.arange(2)
    axis_d.axhline(0, color=GRID_COLOR, lw=0.8)
    for index, row in panel_d.iterrows():
        color = HIGH_COLOR if index == 0 else CORRECT_COLOR
        value = float(row.delta_ssi_from_intact_microbits)
        axis_d.vlines(index, 0, value, color=color, lw=1.5, zorder=1)
        axis_d.scatter(index, value, s=38, color=color, zorder=2)
        axis_d.annotate(
            f"{value:+.2f}",
            (index, value),
            xytext=(0, 4 if value >= 0 else -11),
            textcoords="offset points",
            fontsize=6.2,
            ha="center",
        )
    axis_d.set(
        xticks=x_d,
        xticklabels=d_labels,
        ylabel="ΔSSI from intact at 1× (µbits)",
    )
    _clean_axis(axis_d)
    induced_loss = -float(
        panel_d.loc[
            panel_d.condition.astype(str).eq("induced_eye_misalignment_candidate_p"),
            "delta_ssi_from_intact_bits",
        ].iloc[0]
    )
    _panel_label(
        axis_d,
        "D",
        (
            "P misalignment impairs higher-SF 1× SSI"
            if induced_loss >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
            else "P misalignment does not impair higher-SF 1× SSI"
        ),
    )
    for axis, array, label, exact_ssi in zip(
        axes_d_maps, data["panel_d_maps"], d_labels, panel_d.exact_ssi_bits
    ):
        axis.imshow(array, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        axis.set_title(label, fontsize=6.2, pad=2)
        axis.set_xlabel(f"SSI {float(exact_ssi):.6f}", fontsize=5.7, labelpad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color(GRID_COLOR)
    if image_handle is not None:
        colorbar = figure.colorbar(image_handle, ax=[*axes_c_maps, *axes_d_maps], fraction=0.018, pad=0.012)
        colorbar.set_label("mean-normalized rate  g(x,y)", fontsize=6.4)
        colorbar.ax.tick_params(labelsize=5.8)

    scope_label = "8 images × 24 trajectories" if scope == "full" else "preregistered 4 images × 12 trajectories pilot"
    conclusion = (
        "Causal interventions do not support recurrent spatial transport"
        if decision == DECISION_LABELS[2]
        else (
            "Causal interventions support recurrent spatial transport"
            if decision == DECISION_LABELS[0]
            else "Causal test of recurrent spatial transport"
        )
    )
    figure.suptitle(
        f"{conclusion} ({scope_label})",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.985,
        0.012,
        "All C/D maps share one scale; maps are never rescaled by condition. Activation-derived oracle is an upper bound and is reported outside the main panel.",
        ha="right",
        fontsize=5.8,
        color=MUTED_COLOR,
    )
    return figure


def prepare_optional_figure4_data(
    statistics: Mapping[str, Any], panel_a: pd.DataFrame
) -> pd.DataFrame:
    """Return measured values for the conditional circuit summary."""
    if statistics.get("all_shifter_support_gates_pass") is not True:
        raise DataUnavailable("Optional Figure 4 is forbidden unless every SHIFTER support gate passes")
    if statistics.get("decision") != DECISION_LABELS[0]:
        raise DataUnavailable("Optional Figure 4 requires complete 8×24 SHIFTER support")
    result = panel_a.copy()
    result["ssi_normalized_within_population"] = result.groupby("population").exact_ssi_bits.transform(
        lambda values: values / max(float(values.max()), 1e-30)
    )
    errors = statistics["quantitative"]["high_sf_transport_error_by_scale_feature_px"]
    result["higher_sf_transport_error_feature_px"] = [
        errors.get(float(scale), errors.get(str(float(scale)), math.nan))
        if population == "higher_sf_29"
        else math.nan
        for scale, population in zip(result.scale, result.population)
    ]
    return result


def draw_optional_figure4(
    statistics: Mapping[str, Any], measured: pd.DataFrame
) -> plt.Figure:
    """Draw the gated, measured circuit summary; never an evidentiary panel."""
    _configure_plot()
    figure = plt.figure(figsize=(8.2, 3.45), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.15, 0.85),
        left=0.045,
        right=0.97,
        bottom=0.18,
        top=0.83,
        wspace=0.24,
    )
    schematic = figure.add_subplot(grid[0, 0])
    signature = figure.add_subplot(grid[0, 1])
    schematic.set_axis_off()
    boxes = {
        "current": (0.03, 0.63, 0.29, 0.20, "current feature\nevidence"),
        "retained": (0.03, 0.17, 0.29, 0.20, "retained recurrent\nevidence"),
        "transport": (0.39, 0.17, 0.27, 0.20, "learned spatial\ntransport"),
        "registered": (0.72, 0.40, 0.25, 0.22, "registered state\n→ RR100 map"),
    }
    for name, (x, y, width, height, label) in boxes.items():
        color = CORRECT_COLOR if name in {"transport", "registered"} else "#E8EAEC"
        text_color = "white" if name in {"transport", "registered"} else TEXT_COLOR
        patch = plt.Rectangle(
            (x, y), width, height, transform=schematic.transAxes,
            facecolor=color, edgecolor="none", linewidth=0.0,
        )
        schematic.add_patch(patch)
        schematic.text(
            x + width / 2,
            y + height / 2,
            label,
            transform=schematic.transAxes,
            ha="center",
            va="center",
            color=text_color,
            fontsize=7.0,
        )
    arrow = dict(arrowstyle="->", color=TEXT_COLOR, lw=1.0, shrinkA=2, shrinkB=2)
    schematic.annotate("", xy=(0.72, 0.51), xytext=(0.32, 0.73), xycoords="axes fraction", arrowprops=arrow)
    schematic.annotate("", xy=(0.39, 0.27), xytext=(0.32, 0.27), xycoords="axes fraction", arrowprops=arrow)
    schematic.annotate("", xy=(0.72, 0.51), xytext=(0.66, 0.27), xycoords="axes fraction", arrowprops=arrow)
    transport = statistics["quantitative"]["boundary_resolved_transport"]
    schematic.text(
        0.52,
        0.06,
        f"within-window, newer → older evidence\nmeasured transport r={transport['vector_correlation']:.2f}; median error={transport['median_vector_error_feature_px']:.2f} px",
        transform=schematic.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=MUTED_COLOR,
    )
    schematic.text(0.0, 1.04, "Supported finite-range operation", transform=schematic.transAxes, fontweight="bold", fontsize=9.0)

    for population, label, color in (
        ("lower_sf_71", "lower SF", LOW_COLOR),
        ("higher_sf_29", "higher SF", HIGH_COLOR),
    ):
        part = measured.loc[measured.population.eq(population)].sort_values("scale")
        signature.plot(
            part.scale,
            part.ssi_normalized_within_population,
            color=color,
            lw=1.65,
            marker="o",
            ms=3.8,
        )
        optimum = part.loc[part.ssi_normalized_within_population.idxmax()]
        signature.annotate(
            f"{label} optimum {float(optimum.scale):g}×",
            (float(optimum.scale), float(optimum.ssi_normalized_within_population)),
            xytext=(4, -13 if population == "lower_sf_71" else 5),
            textcoords="offset points",
            color=color,
            fontsize=6.2,
        )
    errors = statistics["quantitative"]["high_sf_transport_error_by_scale_feature_px"]
    signature.text(
        0.98,
        0.03,
        f"higher-SF residual error\n1× {_scale_value(errors, 1.0):.2f} → 3× {_scale_value(errors, 3.0):.2f} feature px",
        transform=signature.transAxes,
        ha="right",
        va="bottom",
        color=HIGH_COLOR,
        fontsize=6.1,
    )
    signature.set(
        xlabel="retinal-motion amplitude (× measured FEM)",
        ylabel="intact SSI / population maximum",
        xticks=SCALES,
        title="Measured finite-range signature",
    )
    signature.set_ylim(0, 1.08)
    _clean_axis(signature)
    figure.suptitle(
        "Compact circuit summary (drawn only after all support gates passed)",
        x=0.045,
        y=0.96,
        ha="left",
        fontsize=10.7,
        fontweight="bold",
    )
    return figure


def _median_by_contrast(
    frame: pd.DataFrame,
    value_column: str,
    *,
    selector: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    selected = frame.copy()
    for column, value in (selector or {}).items():
        selected = selected.loc[selected[column].eq(value)]
    result: dict[str, float] = {}
    for contrast in CONTRASTS:
        values = pd.to_numeric(
            selected.loc[selected.contrast.eq(contrast), value_column], errors="coerce"
        ).to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise DataUnavailable(f"No finite {value_column} for {contrast}")
        result[contrast] = float(np.median(finite))
    return result


def _lag_reduction_statistics(learned_p: pd.DataFrame) -> dict[str, float | int | bool]:
    """Summarize paired raw/recurrent best-lag magnitudes without thresholding."""
    required = (
        "raw_lag_x_px",
        "raw_lag_y_px",
        "recurrent_lag_x_px",
        "recurrent_lag_y_px",
        "raw_at_search_boundary",
        "recurrent_at_search_boundary",
        "valid",
    )
    _require_columns(learned_p, required, "held-out learned-P lag comparison")
    rows = learned_p.loc[
        learned_p.valid
        & ~learned_p.raw_at_search_boundary
        & ~learned_p.recurrent_at_search_boundary
    ].copy()
    raw = rows[["raw_lag_x_px", "raw_lag_y_px"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    recurrent = rows[["recurrent_lag_x_px", "recurrent_lag_y_px"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(raw).all(axis=1) & np.isfinite(recurrent).all(axis=1)
    raw, recurrent = raw[finite], recurrent[finite]
    if len(raw) < 3:
        raise DataUnavailable("Too few boundary-resolved learned-P lag pairs")
    raw_magnitude = np.linalg.norm(raw, axis=1)
    recurrent_magnitude = np.linalg.norm(recurrent, axis=1)
    paired_reduction = raw_magnitude - recurrent_magnitude
    median_raw = float(np.median(raw_magnitude))
    median_recurrent = float(np.median(recurrent_magnitude))
    median_reduction = float(np.median(paired_reduction))
    return {
        "n_pairs": int(len(raw)),
        "median_raw_lag_feature_px": median_raw,
        "median_recurrent_lag_feature_px": median_recurrent,
        "median_paired_lag_reduction_feature_px": median_reduction,
        "fraction_pairs_reduced": float(np.mean(paired_reduction > 0)),
        "raw_lag_present": bool(median_raw > 0),
        "recurrence_reduces_lag": bool(median_reduction > 0),
    }


def _transport_method_statistics(registration: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Compare fold-held-out P transport with Q and rank-matched controls.

    The four random subspaces are averaged within the exact held-out identity
    before the across-identity statistic, preventing random draws from being
    counted as additional biological observations.
    """
    identity = (
        "scope",
        "contrast",
        "fold",
        "image_position",
        "trajectory_position",
        "scale",
        "frame_position",
        "internal_step",
    )
    numeric = (
        "expected_feature_shift_x_px",
        "expected_feature_shift_y_px",
        "transport_x_px",
        "transport_y_px",
    )
    _require_columns(
        registration,
        (*identity, "subspace", *numeric, "valid", "unresolved_boundary"),
        "held-out transport specificity comparison",
    )
    rows = registration.copy()
    rows["method"] = rows.subspace.map(
        {
            "learned_p": "learned P",
            "learned_q": "complementary Q",
            "readout_svd": "readout-SVD",
        }
    ).fillna("random rank-8")
    fixed = rows.loc[~rows.method.eq("random rank-8")].copy()
    random = rows.loc[rows.method.eq("random rank-8")].copy()
    if random.empty:
        raise DataUnavailable("Registration products lack random rank-8 controls")
    grouped = random.groupby([*identity, "method"], as_index=False)
    random_numeric = grouped[list(numeric)].mean()
    random_flags = grouped[["valid", "unresolved_boundary"]].agg(
        {"valid": "all", "unresolved_boundary": "any"}
    )
    random_mean = random_numeric.merge(
        random_flags, on=[*identity, "method"], validate="one_to_one"
    )
    rows = pd.concat([fixed, random_mean], ignore_index=True, sort=False)

    result: dict[str, dict[str, float | int]] = {}
    for method in ("learned P", "complementary Q", "readout-SVD", "random rank-8"):
        selected = rows.loc[
            rows.method.eq(method) & rows.valid & ~rows.unresolved_boundary
        ]
        expected = selected[
            ["expected_feature_shift_x_px", "expected_feature_shift_y_px"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        measured = selected[["transport_x_px", "transport_y_px"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(expected).all(axis=1) & np.isfinite(measured).all(axis=1)
        expected, measured = expected[finite], measured[finite]
        if len(expected) < 3:
            raise DataUnavailable(f"Too few boundary-resolved transport vectors for {method}")
        expected_flat, measured_flat = expected.ravel(), measured.ravel()
        correlation = (
            float(np.corrcoef(expected_flat, measured_flat)[0, 1])
            if np.std(expected_flat) > 0 and np.std(measured_flat) > 0
            else math.nan
        )
        denominator = float(np.square(expected - expected.mean(axis=0)).sum())
        vector_r2 = 1.0 - float(np.square(measured - expected).sum()) / max(
            denominator, 1e-30
        )
        error = np.linalg.norm(measured - expected, axis=1)
        # A null control may be spatially constant, making its correlation
        # undefined; identity-line R² and displacement error remain defined
        # and are the two prospective specificity comparisons.
        if not np.isfinite([vector_r2, np.median(error)]).all():
            raise DataUnavailable(f"Nonfinite transport specificity statistic for {method}")
        if method == "learned P" and not np.isfinite(correlation):
            raise DataUnavailable("Learned-P transport correlation is undefined")
        result[method] = {
            "n_vectors": int(len(expected)),
            "vector_correlation": correlation,
            "identity_line_variance_explained": vector_r2,
            "median_vector_error_feature_px": float(np.median(error)),
        }
    return result


def select_decision(
    *,
    causal_scope: str,
    raw_lag_present: bool,
    recurrence_reduces_lag: bool,
    transport_tracks: bool,
    p_transport_stronger: bool,
    high_sf_registration_failure: bool,
    spatial_recurrence_changes_ssi: bool,
    specific_correct_realign_rescue: bool,
    induced_failure: bool,
) -> tuple[str, str, bool]:
    """Apply the waived-factorization decision hierarchy without reading files.

    The P/Q motion-energy result is intentionally absent from this interface.
    It rejected a motion-only-P/content-only-Q factorization, not recurrent
    registration, and therefore cannot select or veto a shifter conclusion.
    """
    if causal_scope not in {"pilot", "full"}:
        raise ValueError(f"Unknown causal scope {causal_scope!r}")
    registration_exists = bool(
        raw_lag_present and recurrence_reduces_lag and transport_tracks
    )
    explains_ssi = bool(
        high_sf_registration_failure
        and spatial_recurrence_changes_ssi
        and specific_correct_realign_rescue
        and induced_failure
    )
    all_support = bool(registration_exists and p_transport_stronger and explains_ssi)

    # The fold-wise held-out registration analysis is complete independently
    # of the causal scope. Failure of its three core observations supports the
    # registration-negative label even when causal interventions remain a
    # pilot; a positive pilot can never earn definitive support.
    if not registration_exists:
        decision = DECISION_LABELS[2]
        manuscript = "NOT SUPPORTED"
    elif causal_scope != "full":
        decision = DECISION_LABELS[3]
        manuscript = "CONSISTENT WITH" if all_support else "NOT SUPPORTED"
    elif all_support:
        decision = DECISION_LABELS[0]
        manuscript = "SUPPORTED"
    elif registration_exists and not explains_ssi:
        decision = DECISION_LABELS[1]
        manuscript = "NOT SUPPORTED"
    else:
        # Registration and all SSI links are positive, but the learned-P
        # specificity comparison is not: the permitted labels do not justify
        # either a supported shifter or a claim of no recurrent registration.
        decision = DECISION_LABELS[3]
        manuscript = "NOT SUPPORTED"
    if decision not in DECISION_LABELS or manuscript not in MANUSCRIPT_LABELS:
        raise AssertionError("Decision label escaped the fixed vocabulary")
    return decision, manuscript, all_support


def _high_sf_registration_failure_status(
    *,
    transport_tracks: bool,
    recurrence_reduces_lag: bool,
    alignment_improvement: Mapping[str, float],
    error_at_1x: float,
    error_at_3x: float,
    ssi_at_1x: float,
    ssi_at_3x: float,
) -> tuple[bool, bool]:
    """Return (evaluable, established) for the finite-range failure claim.

    A larger descriptive residual at 3× cannot be called a registration
    failure unless the same inferred transport first demonstrates positive
    displacement tracking and alignment improvement.
    """
    values = (
        *alignment_improvement.values(),
        error_at_1x,
        error_at_3x,
        ssi_at_1x,
        ssi_at_3x,
    )
    if not np.isfinite(values).all():
        raise DataUnavailable("Nonfinite value in the higher-SF registration-failure test")
    evaluable = bool(
        transport_tracks
        and recurrence_reduces_lag
        and alignment_improvement
        and all(value > 0 for value in alignment_improvement.values())
    )
    established = bool(evaluable and error_at_3x > error_at_1x and ssi_at_3x < ssi_at_1x)
    return evaluable, established


def compute_statistics(inputs: FinalInputs) -> dict[str, Any]:
    # A. Crossed held-out rank-8 validation and stability.
    learned = inputs.rank_results.loc[inputs.rank_results.method.eq("learned")]
    learned_recovery = _median_by_contrast(learned, "complete_map_recovery_mean")
    within = inputs.stability.loc[
        inputs.stability.comparison_type.eq("within_contrast_across_folds")
    ]
    overlap = _median_by_contrast(within.rename(columns={"contrast_a": "contrast"}), "projector_overlap")
    rank_generalizes = bool(
        inputs.rank_gate.get("all_three_contrasts_generalize") is True
        and all(inputs.rank_gate["heldout_generalization"][contrast].get("generalizes") is True for contrast in CONTRASTS)
    )
    # B/C. Leverage and fold-wise P/Q semantics.
    neff = _median_by_contrast(inputs.leverage, "effective_participating_channel_count")
    leverage_readout_association = _median_by_contrast(
        inputs.leverage, "leverage_spearman_readout_strength"
    )
    leverage_motion_association = _median_by_contrast(
        inputs.leverage, "leverage_spearman_centered_motion_variance"
    )
    variance = inputs.variance.loc[
        inputs.variance.analysis.eq("movement_change_from_stabilization")
    ].copy()
    variance = variance.loc[
        [np.isclose(row.scale, TARGET_SCALE[str(row.contrast)]) for row in variance.itertuples()]
    ]
    motion_ratio, motion_fold_counts, previous_factorization_gate_passed = (
        defining_motion_enrichment(inputs.variance)
    )
    p_motion_fraction = _median_by_contrast(variance, "candidate_p_total_fraction")
    if motion_fold_counts != {contrast: len(FOLDS) for contrast in CONTRASTS}:
        raise DataUnavailable("The preserved P/Q factorization lacks all held-out folds")
    if previous_factorization_gate_passed:
        raise DataUnavailable(
            "The authorized waiver preserves a failed P/Q motion-energy factorization, "
            "but the finalized semantic products now imply that its old gate passed"
        )
    content = inputs.variance.loc[
        inputs.variance.analysis.eq("stabilized_visual_content_image_means")
    ]
    content_ratio = _median_by_contrast(content, "p_to_q_per_dimension_energy_ratio")
    trajectory = inputs.variance.loc[
        inputs.variance.analysis.eq("trajectory_specific_at_fixed_image_scale_frame")
    ]
    trajectory_ratio = _median_by_contrast(trajectory, "p_to_q_per_dimension_energy_ratio")

    q_baseline = inputs.readout.loc[
        inputs.readout.analysis.eq("baseline_visual_reconstruction")
        & inputs.readout.component.eq("complementary_q_content")
    ].copy()
    q_baseline = q_baseline.loc[
        [str(row.population) == TARGET_POPULATION[str(row.contrast)] for row in q_baseline.itertuples()]
    ]
    q_recovery = _median_by_contrast(q_baseline, "normalized_map_recovery_vs_training_mean_r2")
    q_substantial = bool(
        all(value >= MIN_SUBSTANTIAL_Q_BASELINE_R2 for value in q_recovery.values())
    )

    p_movement = inputs.readout.loc[
        inputs.readout.analysis.eq("defining_contrast_movement_effect_decomposition")
        & inputs.readout.component.eq("candidate_p_only_contrast")
    ].copy()
    q_movement = inputs.readout.loc[
        inputs.readout.analysis.eq("defining_contrast_movement_effect_decomposition")
        & inputs.readout.component.eq("complementary_q_only_contrast")
    ].copy()
    p_movement = p_movement.loc[
        [str(row.population) == TARGET_POPULATION[str(row.contrast)] for row in p_movement.itertuples()]
    ]
    q_movement = q_movement.loc[
        [str(row.population) == TARGET_POPULATION[str(row.contrast)] for row in q_movement.itertuples()]
    ]
    p_map_recovery = _median_by_contrast(p_movement, "normalized_map_movement_effect_recovery_r2")
    q_map_recovery = _median_by_contrast(q_movement, "normalized_map_movement_effect_recovery_r2")
    output_mostly_p = bool(
        all(
            np.isfinite(p_map_recovery[key])
            and np.isfinite(q_map_recovery[key])
            and p_map_recovery[key] >= MIN_SUBSTANTIAL_P_MOVEMENT_MAP_R2
            and p_map_recovery[key] - q_map_recovery[key]
            >= MIN_P_OVER_Q_MOVEMENT_MAP_R2_MARGIN
            for key in CONTRASTS
        )
    )

    absolute = inputs.readout.loc[
        inputs.readout.analysis.eq("absolute_preactivation_variance_across_heldout_image_means")
        & inputs.readout.component.eq("bias_once_plus_a_p_plus_a_q")
    ].copy()
    absolute = absolute.loc[
        [
            str(row.population) == TARGET_POPULATION[str(row.contrast)]
            and np.isclose(float(row.scale), TARGET_SCALE[str(row.contrast)])
            for row in absolute.itertuples()
        ]
    ]
    absolute_p_fraction = _median_by_contrast(absolute, "p_fraction_of_full_variance")
    absolute_q_fraction = _median_by_contrast(absolute, "q_fraction_of_full_variance")
    absolute_cov_fraction = _median_by_contrast(
        absolute, "covariance_fraction_of_full_variance"
    )

    reliance_column = "activity_activity_weighted_p_reliance_excluding_covariance"
    _require_columns(
        inputs.per_unit,
        (
            reliance_column,
            "sf_split_metric",
            "observed_ssi_benefit_for_projector_contrast_bits",
            "observed_ssi_1_to_3_change_bits",
            "historical_sf_population",
        ),
        "per-unit P/Q reliance",
    )
    per_unit_associations: dict[str, dict[str, float]] = {}
    for contrast in CONTRASTS:
        rows = inputs.per_unit.loc[
            inputs.per_unit.contrast.eq(contrast)
            & inputs.per_unit.historical_sf_population.eq(TARGET_POPULATION[contrast])
        ].copy()
        # Average the four crossed-fold observations within unit before the
        # descriptive across-unit rank association.
        grouped = rows.groupby("unit_index", as_index=False)[
            [
                reliance_column,
                "sf_split_metric",
                "observed_ssi_benefit_for_projector_contrast_bits",
                "observed_ssi_1_to_3_change_bits",
            ]
        ].mean()
        x = pd.to_numeric(grouped[reliance_column], errors="coerce")
        per_unit_associations[contrast] = {}
        for label, column in (
            ("sf_preference", "sf_split_metric"),
            ("movement_benefit", "observed_ssi_benefit_for_projector_contrast_bits"),
            ("high_motion_change_1_to_3", "observed_ssi_1_to_3_change_bits"),
        ):
            y = pd.to_numeric(grouped[column], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                value = math.nan
            else:
                value = float(x[valid].rank().corr(y[valid].rank()))
            per_unit_associations[contrast][label] = value

    probe_summary: dict[str, dict[str, float]] = {}
    for probe_name, rows in inputs.probes.groupby("probe"):
        probe_summary[str(probe_name)] = {
            label: float(
                np.nanmedian(
                    pd.to_numeric(
                        rows.loc[rows.representation.eq(representation), "value"],
                        errors="coerce",
                    )
                )
            )
            for label, representation in PROBE_REPRESENTATION.items()
        }

    # D. Registration statistics. Boundary/oracle distinctions are explicit.
    registration = inputs.registration.copy()
    for name in ("valid", "raw_at_search_boundary", "recurrent_at_search_boundary", "expected_outside_search_window"):
        registration[name] = registration[name].astype(str).str.lower().isin(("true", "1"))
    registration["unresolved_boundary"] = (
        registration.raw_at_search_boundary
        | registration.recurrent_at_search_boundary
        | registration.expected_outside_search_window
    )
    learned_p = registration.loc[registration.subspace.eq("learned_p") & registration.valid].copy()
    lag_reduction = _lag_reduction_statistics(learned_p)
    transport_by_method = _transport_method_statistics(registration)
    p_transport = transport_by_method["learned P"]
    comparator_transport = {
        key: transport_by_method[key]
        for key in ("complementary Q", "readout-SVD", "random rank-8")
    }
    p_transport_stronger = bool(
        all(
            float(p_transport["identity_line_variance_explained"])
            > float(values["identity_line_variance_explained"])
            and float(p_transport["median_vector_error_feature_px"])
            < float(values["median_vector_error_feature_px"])
            for values in comparator_transport.values()
        )
    )
    alignment_improvement: dict[str, float] = {}
    for contrast in CONTRASTS:
        rows = learned_p.loc[
            learned_p.contrast.eq(contrast)
            & np.isclose(learned_p.scale, TARGET_SCALE[contrast])
        ]
        values = pd.to_numeric(rows.zero_lag_alignment_improvement, errors="coerce").to_numpy(float)
        if not np.isfinite(values).any():
            raise DataUnavailable(f"No valid zero-lag alignment values for {contrast}")
        alignment_improvement[contrast] = float(np.nanmedian(values))
    resolved = learned_p.loc[~learned_p.unresolved_boundary].copy()
    expected_shift = resolved[["expected_feature_shift_x_px", "expected_feature_shift_y_px"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    inferred_shift = resolved[["transport_x_px", "transport_y_px"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(expected_shift).all(axis=1) & np.isfinite(inferred_shift).all(axis=1)
    expected_shift, inferred_shift = expected_shift[finite], inferred_shift[finite]
    if len(expected_shift) < 3:
        raise DataUnavailable("Too few boundary-resolved learned-P transport vectors")
    vector_correlation = float(p_transport["vector_correlation"])
    vector_r2 = float(p_transport["identity_line_variance_explained"])
    median_transport_error = float(p_transport["median_vector_error_feature_px"])
    transport_tracks = bool(vector_correlation > 0 and vector_r2 > 0)

    resolved["transport_error_px"] = np.hypot(
        pd.to_numeric(resolved.transport_x_px) - pd.to_numeric(resolved.expected_feature_shift_x_px),
        pd.to_numeric(resolved.transport_y_px) - pd.to_numeric(resolved.expected_feature_shift_y_px),
    )
    # Criterion 6 is specifically tied to the 1×→3× higher-SF reversal.
    # Do not pool the sharpening projector unless the explicit shared-high
    # consensus gate is allowed; fold-wise reversal rows remain the primary
    # inference space either way.
    high = resolved.loc[resolved.contrast.eq("high_1_to_3")]
    high_error_by_scale = {
        float(scale): float(np.nanmedian(high.loc[np.isclose(high.scale, scale), "transport_error_px"]))
        for scale in SCALES
    }
    intact_high_1 = _dose_value(inputs.causal.transport, "intact", 1.0, "higher_sf_29")
    intact_high_3 = _dose_value(inputs.causal.transport, "intact", 3.0, "higher_sf_29")
    registration_failure_evaluable, high_failure = _high_sf_registration_failure_status(
        transport_tracks=transport_tracks,
        recurrence_reduces_lag=bool(lag_reduction["recurrence_reduces_lag"]),
        alignment_improvement=alignment_improvement,
        error_at_1x=high_error_by_scale[1.0],
        error_at_3x=high_error_by_scale[3.0],
        ssi_at_1x=intact_high_1,
        ssi_at_3x=intact_high_3,
    )

    # Kernel organization is descriptive and never substitutes for activation transport.
    learned_kernel = inputs.kernel.loc[inputs.kernel.projector.eq("learned_p")]
    kernel_offcenter_fraction = {
        kernel: float(np.nanmedian(pd.to_numeric(part.pp_offcenter_fraction, errors="coerce")))
        for kernel, part in learned_kernel.groupby("kernel")
    }

    # E. Exact causal transport and realignment gates.
    transport = inputs.causal.transport
    intact_effect = {key: _contrast_value(transport, "intact", key) for key in CONTRASTS}
    center_effect = {
        key: _contrast_value(transport, "all_recurrent_center_only", key) for key in CONTRASTS
    }
    permuted_conditions = [condition for condition in TRANSPORT_CONDITIONS if "offset_permuted" in condition]
    permuted_effect = {
        key: float(np.mean([_contrast_value(transport, condition, key) for condition in permuted_conditions]))
        for key in CONTRASTS
    }
    expected_intact_sign = {
        "low_0_to_2": intact_effect["low_0_to_2"] > 0,
        "high_0_to_1": intact_effect["high_0_to_1"] > 0,
        "high_1_to_3": intact_effect["high_1_to_3"] < 0,
    }

    geometry_conditions = (
        "candidate_recurrent_center_only",
        "all_recurrent_center_only",
        *permuted_conditions,
    )
    geometry_effects = {
        condition: {
            key: _contrast_value(transport, condition, key) for key in CONTRASTS
        }
        for condition in geometry_conditions
    }
    attenuation_bits = {
        condition: {
            key: abs(intact_effect[key]) - abs(effects[key])
            for key in CONTRASTS
        }
        for condition, effects in geometry_effects.items()
    }
    qualifying_geometry = [
        condition
        for condition, values in attenuation_bits.items()
        if all(
            np.isfinite(values[key])
            and values[key] >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
            for key in CONTRASTS
        )
    ]
    stabilized_rows = transport.loc[
        transport.analysis.eq("transport_ablation_dose_curve")
        & np.isclose(transport.scale, 0.0)
        & transport.population.eq("all_100")
        & transport.condition.isin(qualifying_geometry)
    ]
    stabilized_values = pd.to_numeric(
        stabilized_rows.normalized_map_recovery_r2_vs_intact, errors="coerce"
    ).to_numpy(float)
    stabilized_fidelity = (
        float(np.nanmax(stabilized_values)) if np.isfinite(stabilized_values).any() else math.nan
    )
    offcenter_causal = bool(
        all(expected_intact_sign.values())
        and qualifying_geometry
        and stabilized_fidelity >= 0.25
    )

    realignment = inputs.causal.realignment
    high3 = {
        "intact": _dose_value(realignment, "no_shift_intact", 3.0, "higher_sf_29"),
        "correct_p": _dose_value(realignment, "eye_correct_candidate_p", 3.0, "higher_sf_29"),
        "oracle_p": _dose_value(realignment, "activation_oracle_candidate_p", 3.0, "higher_sf_29"),
        "opposite_p": _dose_value(realignment, "eye_opposite_candidate_p", 3.0, "higher_sf_29"),
        "random_p": _dose_value(realignment, "eye_random_matched_candidate_p", 3.0, "higher_sf_29"),
        "correct_q": _dose_value(realignment, "eye_correct_complementary_q", 3.0, "higher_sf_29"),
    }
    correct_gain = high3["correct_p"] - high3["intact"]
    control_gains = {
        key: high3[key] - high3["intact"]
        for key in ("opposite_p", "random_p", "correct_q")
    }
    oracle_gain = high3["oracle_p"] - high3["intact"]
    correct_rescue = bool(correct_gain >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS)
    oracle_positive = bool(oracle_gain >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS)
    wrong_controls_fail = bool(
        control_gains["opposite_p"] < MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
        and control_gains["random_p"] < MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
    )
    q_less_effective = bool(
        correct_gain - control_gains["correct_q"]
        >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
    )
    high1 = {
        "intact": _dose_value(realignment, "no_shift_intact", 1.0, "higher_sf_29"),
        "induced_p": _dose_value(
            realignment, "induced_eye_misalignment_candidate_p", 1.0, "higher_sf_29"
        ),
    }
    induced_loss = high1["intact"] - high1["induced_p"]
    induced_failure = bool(induced_loss >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS)

    specific_correct_rescue = bool(
        correct_rescue and oracle_positive and wrong_controls_fail and q_less_effective
    )
    support_gates = {
        "raw_previous_state_lags_current": bool(lag_reduction["raw_lag_present"]),
        "recurrence_reduces_lag": bool(lag_reduction["recurrence_reduces_lag"]),
        "inferred_transport_tracks_retinal_displacement": transport_tracks,
        "transport_stronger_in_p_than_q_readout_svd_random": p_transport_stronger,
        "registration_fails_at_excessive_higher_sf_motion": high_failure,
        "destroying_spatial_recurrence_changes_ssi_curves": offcenter_causal,
        "correct_realign_rescues_higher_sf_3x_ssi": specific_correct_rescue,
        "induced_misalignment_destroys_higher_sf_1x_sharpening": induced_failure,
    }
    learned_vs_readout_delta = {
        contrast: float(
            np.median(
                pd.to_numeric(
                    learned.loc[learned.contrast.eq(contrast), "learned_minus_readout_complete_map_mean"],
                    errors="coerce",
                )
            )
        )
        for contrast in CONTRASTS
    }
    # The previous P/Q factorization stop is explicitly waived.  None of the
    # factorization quantities above is passed to the scientific decision.
    # A positive pilot is encouraging, but cannot receive the definitive
    # SUPPORT label without complete 8×24 downstream confirmation.
    decision, manuscript, all_support = select_decision(
        causal_scope=inputs.causal.scope,
        raw_lag_present=bool(lag_reduction["raw_lag_present"]),
        recurrence_reduces_lag=bool(lag_reduction["recurrence_reduces_lag"]),
        transport_tracks=transport_tracks,
        p_transport_stronger=p_transport_stronger,
        high_sf_registration_failure=high_failure,
        spatial_recurrence_changes_ssi=offcenter_causal,
        specific_correct_realign_rescue=specific_correct_rescue,
        induced_failure=induced_failure,
    )

    questions = [
        {
            "question": REPORT_QUESTIONS[0],
            "result": "Yes" if lag_reduction["raw_lag_present"] else "No",
            "primary_statistic": (
                f"boundary-resolved held-out learned P: median raw lag="
                f"{lag_reduction['median_raw_lag_feature_px']:.3f} feature px "
                f"(n={lag_reduction['n_pairs']})"
            ),
            "confidence": "High",
        },
        {
            "question": REPORT_QUESTIONS[1],
            "result": "Yes" if lag_reduction["recurrence_reduces_lag"] else "No",
            "primary_statistic": (
                f"median paired raw−recurrent lag="
                f"{lag_reduction['median_paired_lag_reduction_feature_px']:+.3f} feature px; "
                f"raw={lag_reduction['median_raw_lag_feature_px']:.3f}, "
                f"recurrent={lag_reduction['median_recurrent_lag_feature_px']:.3f}; "
                f"{lag_reduction['fraction_pairs_reduced']:.1%} of pairs reduced"
            ),
            "confidence": "High",
        },
        {
            "question": REPORT_QUESTIONS[2],
            "result": "Yes" if transport_tracks else "No",
            "primary_statistic": (
                f"learned-P boundary-resolved vector r={vector_correlation:.3f}, "
                f"identity-line R²={vector_r2:.3f}, median error="
                f"{median_transport_error:.3f} feature px"
            ),
            "confidence": "High",
        },
        {
            "question": REPORT_QUESTIONS[3],
            "result": "Yes" if p_transport_stronger else "No",
            "primary_statistic": "; ".join(
                f"{method}: identity-line R²={values['identity_line_variance_explained']:.3f}, "
                f"median error={values['median_vector_error_feature_px']:.3f} px"
                for method, values in transport_by_method.items()
            ),
            "confidence": "High",
        },
        {
            "question": REPORT_QUESTIONS[4],
            "result": (
                "Yes"
                if high_failure
                else ("No" if registration_failure_evaluable else "Not established")
            ),
            "primary_statistic": (
                f"higher-SF reversal-space median error 1×={high_error_by_scale[1.0]:.3f} "
                f"versus 3×={high_error_by_scale[3.0]:.3f} feature px; intact SSI "
                f"1×={intact_high_1:.4f}, 3×={intact_high_3:.4f}"
                + (
                    ""
                    if registration_failure_evaluable
                    else "; descriptive only because transport/alignment evidence is not positive"
                )
            ),
            "confidence": "High" if inputs.causal.scope == "full" else "Pilot",
        },
        {
            "question": REPORT_QUESTIONS[5],
            "result": "Yes" if offcenter_causal else "No",
            "primary_statistic": (
                f"conditions attenuating all 3 defining SSI changes by ≥"
                f"{MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS:.3f} bits: "
                f"{', '.join(qualifying_geometry) or 'none'}; maximum stabilized-map "
                f"fidelity R²={stabilized_fidelity:.3f}"
            ),
            "confidence": "High" if inputs.causal.scope == "full" else "Pilot",
        },
        {
            "question": REPORT_QUESTIONS[6],
            "result": "Yes, specifically" if specific_correct_rescue else "No",
            "primary_statistic": (
                f"3× ΔSSI vs intact: correct P={1e6 * correct_gain:+.2f} µbits, "
                f"opposite={1e6 * control_gains['opposite_p']:+.2f}, "
                f"random={1e6 * control_gains['random_p']:+.2f}, "
                f"Q={1e6 * control_gains['correct_q']:+.2f}; oracle upper bound="
                f"{1e6 * oracle_gain:+.2f}; directional floor="
                f"{1e6 * MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS:.0f} µbits"
            ),
            "confidence": "High" if inputs.causal.scope == "full" else "Pilot",
        },
        {
            "question": REPORT_QUESTIONS[7],
            "result": "Yes" if induced_failure else "No",
            "primary_statistic": (
                f"higher-SF 1× SSI: intact={high1['intact']:.6f}, induced P "
                f"misalignment={high1['induced_p']:.6f}, loss="
                f"{1e6 * induced_loss:+.2f} µbits"
            ),
            "confidence": "High" if inputs.causal.scope == "full" else "Pilot",
        },
    ]
    if tuple(row["question"] for row in questions) != REPORT_QUESTIONS:
        raise AssertionError("Registration report must contain exactly the eight fixed questions")

    panel_recommendations = [
        "Output-relevant candidate P and complementary control Q, while preserving the failed motion-only-P/content-only-Q factorization as a waived non-decision result.",
        "Within-window registration: recurrent alignment and calibrated inferred transport versus retinal displacement; the higher-SF 3× residual increase remains descriptive because positive registration was not established.",
        "Causal spatial-transport test: center-only/offset-permuted recurrence plus eye-derived P realignment and induced P misalignment, with complete normalized maps.",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "manuscript_recommendation": manuscript,
        "causal_scope": inputs.causal.scope,
        "result": (
            "causal interventions do not support recurrent spatial transport"
            if decision == DECISION_LABELS[2]
            else "see fixed four-way decision in statistics.json"
        ),
        "all_shifter_support_gates_pass": all_support,
        "support_gates": support_gates,
        "pq_motion_energy_gate_waiver": {
            **PQ_GATE_WAIVER,
            "previous_gate_passed": False,
            "observed_median_p_to_q_energy_per_dimension": motion_ratio,
            "used_in_final_decision": False,
        },
        "auxiliary_support_requirements": {
            "oracle_upper_bound_positive": oracle_positive,
            "q_shift_less_effective_than_p_shift": q_less_effective,
            "wrong_direction_and_random_controls_fail": wrong_controls_fail,
            "full_8x24_confirmation": inputs.causal.scope == "full",
        },
        "decision_hierarchy": [
            "SHIFTER/REGISTRATION MECHANISM SUPPORTED requires Yes on all eight registration/causal questions and complete 8×24 confirmation.",
            "RECURRENT SPATIAL TRANSPORT EXISTS BUT DOES NOT EXPLAIN SSI requires complete evidence for raw lag, recurrence-mediated lag reduction, and displacement tracking, but incomplete SSI linkage.",
            "NO EVIDENCE FOR RECURRENT REGISTRATION applies when the completed fold-wise held-out analysis lacks the three core recurrent-registration observations; causal pilot status cannot turn that absence into support.",
            "INCONCLUSIVE covers pilots with positive core registration and the case where registration/SSI results are positive but learned-P specificity over Q/readout-SVD/random is absent.",
            "The failed P/Q motion-energy factorization is preserved but waived and never enters this hierarchy.",
        ],
        "prospective_report_thresholds": {
            "substantial_q_baseline_normalized_map_recovery_r2_min": MIN_SUBSTANTIAL_Q_BASELINE_R2,
            "distributed_native_effective_count_min": MIN_DISTRIBUTED_NATIVE_NEFF,
            "directional_causal_attenuation_bits_min_in_every_contrast": MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS,
            "preserved_waived_motion_factorization_gate": "P/Q energy per dimension > 1 in every defining contrast; reported descriptively and excluded from the final decision",
            "substantial_p_movement_map_recovery_r2_min": MIN_SUBSTANTIAL_P_MOVEMENT_MAP_R2,
            "p_over_q_movement_map_recovery_r2_margin_min": MIN_P_OVER_Q_MOVEMENT_MAP_R2_MARGIN,
            "p_transport_specificity": "learned P identity-line transport R² must exceed, and median vector error must be below, complementary Q, readout-SVD, and the within-identity mean of random rank-8 controls",
            "transport_tracking": "positive vector correlation and positive identity-line variance explained after boundary exclusions",
            "directional_realign_rescue_bits_min": MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS,
            "oracle_upper_bound_gain_bits_min": MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS,
            "p_over_q_realign_gain_difference_bits_min": MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS,
            "induced_misalignment_loss_bits_min": MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS,
            "wrong_shift_control": "opposite and matched-random SSI gains must each remain below the 0.002-bit directional floor",
        },
        "quantitative": {
            "rank8_median_complete_map_recovery_r2": learned_recovery,
            "rank8_median_learned_minus_readout_svd_complete_map_recovery_r2": learned_vs_readout_delta,
            "rank8_median_projector_overlap": overlap,
            "native_effective_participating_channels": neff,
            "leverage_spearman_readout_strength": leverage_readout_association,
            "leverage_spearman_centered_motion_variance": leverage_motion_association,
            "p_motion_total_fraction": p_motion_fraction,
            "p_to_q_motion_energy_per_dimension": motion_ratio,
            "p_to_q_content_image_mean_energy_per_dimension": content_ratio,
            "p_to_q_trajectory_energy_per_dimension": trajectory_ratio,
            "q_baseline_normalized_map_recovery_r2": q_recovery,
            "p_movement_map_recovery_r2": p_map_recovery,
            "q_movement_map_recovery_r2": q_map_recovery,
            "absolute_preactivation_variance_fraction": {
                "p": absolute_p_fraction,
                "q": absolute_q_fraction,
                "p_q_covariance": absolute_cov_fraction,
            },
            "per_unit_activity_weighted_p_reliance_spearman": per_unit_associations,
            "descriptive_probe_median_performance": probe_summary,
            "candidate_p_is_rr100_output_relevant": output_mostly_p,
            "complementary_q_retains_substantial_baseline_prediction": q_substantial,
            "raw_and_recurrent_lag": lag_reduction,
            "p_zero_lag_alignment_improvement": alignment_improvement,
            "transport_by_subspace": transport_by_method,
            "boundary_resolved_transport": {
                "n_vectors": len(expected_shift),
                "vector_correlation": vector_correlation,
                "identity_line_variance_explained": vector_r2,
                "median_vector_error_feature_px": median_transport_error,
            },
            "high_sf_transport_error_by_scale_feature_px": high_error_by_scale,
            "high_sf_registration_failure_evaluable": registration_failure_evaluable,
            "learned_p_kernel_offcenter_fraction": kernel_offcenter_fraction,
            "transport_ssi_contrasts": {
                "intact": intact_effect,
                "all_recurrent_center_only": center_effect,
                "offset_permuted_mean": permuted_effect,
                "directional_attenuation_bits_by_geometry_condition": attenuation_bits,
                "qualifying_geometry_conditions": qualifying_geometry,
                "maximum_stabilized_map_fidelity_r2": stabilized_fidelity,
            },
            "realignment_high3_ssi": high3,
            "realignment_high3_gain_vs_intact_bits": {
                "correct_p": correct_gain,
                "oracle_p_upper_bound": oracle_gain,
                **control_gains,
            },
            "induced_failure_high1_ssi": {**high1, "intact_minus_induced_bits": induced_loss},
        },
        "first_page_questions": questions,
        "exactly_three_manuscript_panels": panel_recommendations,
        "inference_boundaries": {
            "semantic_inference": "fold-wise crossed held-out projectors only",
            "pq_factorization": "failed motion-only-P/content-only-Q interpretation is preserved, waived as a stop, and excluded from the shifter decision",
            "consensus_projectors": "visualization only",
            "activation_oracle": "upper bound only; never evidence for biological availability",
            "linear_probes": "descriptive decodability only; never causal",
            "kernel_energy": "descriptive organization only; never proof of a shifter",
            "registration_boundary": "best-lag boundary peaks and calibrated shifts outside the search window are unresolved, not measured lags",
            "temporal_direction": "eight internal steps move from newer-support toward older-support within each independently scored 32-lag input window; backward in retinal time",
            "not_across_outputs": "ConvGRU recurrence is reset for every scored output and is not recurrence across the 40 movie outputs",
            "circuit_claim": "no uniquely localized recurrent pathway is claimed",
        },
    }


def compute_stopped_pq_statistics(inputs: UpstreamInputs) -> dict[str, Any]:
    """Reject the retired P/Q-only decision path.

    The historical factorization result remains in the finalized P/Q products,
    but the user-authorized waiver requires registration and causal products
    before any one of the four final decisions may be selected.
    """
    del inputs
    raise DataUnavailable(
        "The P/Q motion-energy stop is waived; finalized fold-wise registration "
        "and causal products are required before a final decision"
    )


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _scale_value(values: Mapping[Any, Any], scale: float) -> float:
    """Read a scale-keyed statistic before or after JSON stringifies keys."""
    for key in (float(scale), str(float(scale)), f"{float(scale):g}"):
        if key in values:
            value = float(values[key])
            if not math.isfinite(value):
                raise DataUnavailable(f"Nonfinite saved statistic at scale {scale:g}×")
            return value
    raise DataUnavailable(f"Missing saved statistic at scale {scale:g}×")


def render_report(statistics: Mapping[str, Any]) -> str:
    q = statistics["quantitative"]
    questions = statistics["first_page_questions"]
    if statistics.get("decision") not in DECISION_LABELS:
        raise AssertionError("Report decision escaped the four-label vocabulary")
    if len(questions) != 8 or tuple(row.get("question") for row in questions) != REPORT_QUESTIONS:
        raise AssertionError("First page must contain exactly the eight fixed questions")
    expected_question_keys = {"question", "result", "primary_statistic", "confidence"}
    if any(set(row) != expected_question_keys for row in questions):
        raise AssertionError("First-page question schema changed")
    definitive_support = (
        statistics["decision"] == DECISION_LABELS[0]
        and statistics["causal_scope"] == "full"
    )
    pilot_consistent = (
        statistics["manuscript_recommendation"] == "CONSISTENT WITH"
        and statistics["causal_scope"] == "pilot"
    )
    if definitive_support:
        biological_text = (
            "The smallest supported algorithmic implementation is a bank of oriented spatial "
            "filters with phase-sensitive temporal responses, combined with local recurrent "
            "spatial transport or phase rotation over a finite registration range and a "
            "spatially tiled output readout. Moderate retinal motion supplies complementary "
            "samples that can be aligned and accumulated; excessive displacement of fine "
            "structure can exceed the measured range."
        )
    elif pilot_consistent:
        biological_text = (
            "The pilot is consistent with, but does not establish, an implementation combining "
            "oriented phase-sensitive responses, local recurrent spatial transport or phase "
            "rotation, a finite capture range, and a tiled output readout. Complete 8×24 "
            "confirmation is required before calling this mechanism supported."
        )
    else:
        biological_text = (
            "Oriented phase-sensitive responses and a spatially tiled readout remain plausible "
            "components, but these results do not establish local recurrent transport, phase "
            "rotation, or a finite-range shifter implementation. The negative or inconclusive "
            "decision must not be retold as a supported registration algorithm."
        )
    lines = [
        "# Registration test report",
        "",
        f"**Decision: {statistics['decision']}**  ",
        f"**Manuscript recommendation: {statistics['manuscript_recommendation']}**  ",
        f"**Causal scope: {'complete 8-image × 24-trajectory confirmation' if statistics['causal_scope'] == 'full' else 'preregistered 4-image × 12-trajectory pilot only'}**  ",
        "**Authorized waiver:** the failed P/Q motion-energy factorization is preserved, not reopened, and excluded from the shifter decision; completed registration and causal evidence are required.",
        "",
        "| Question | Result | Primary statistic | Confidence |",
        "| -------- | ------ | ----------------- | ---------- |",
    ]
    for row in questions:
        clean = {key: str(value).replace("|", "\\|").replace("\n", " ") for key, value in row.items()}
        lines.append(
            f"| {clean['question']} | {clean['result']} | {clean['primary_statistic']} | {clean['confidence']} |"
        )
    lines.extend(
        [
            "",
            "## A. Validated subspace",
            "",
            "Rank 8 was fixed prospectively from the screening fold. The final estimates use each fold's projector only on its crossed held-out image–trajectory block. "
            "Consensus projectors were constructed from mean projectors (never raw bases) and are visualization-only.",
            "",
            *[
                f"- {CONTRAST_LABEL[key]}: median held-out complete-map recovery R²={_fmt(q['rank8_median_complete_map_recovery_r2'][key])}; learned−readout-SVD complete-map R²={q['rank8_median_learned_minus_readout_svd_complete_map_recovery_r2'][key]:+.3f}; median fold-projector overlap={_fmt(q['rank8_median_projector_overlap'][key])}."
                for key in CONTRASTS
            ],
            "",
            "Readout-SVD is a rank-matched output-coupled comparison. A learned projector's output relevance is not, by itself, movement specificity.",
            "",
            "## B. Native-channel participation",
            "",
            "The rank-8 object is a channel subspace, not eight native channels. Leverage ℓc=Pcc sums to eight; no leverage threshold was used to name a circuit.",
            "",
            *[
                f"- {CONTRAST_LABEL[key]}: median effective participating-channel count N_eff={q['native_effective_participating_channels'][key]:.1f}; leverage association with native readout strength ρ={q['leverage_spearman_readout_strength'][key]:.3f}; with centered motion variance ρ={q['leverage_spearman_centered_motion_variance'][key]:.3f}."
                for key in CONTRASTS
            ],
            "",
            "## C. (P/Q) semantic decomposition",
            "",
            "The original, preregistered factorization test failed and remains closed: the same fixed split did not support a motion-only P versus content-only Q interpretation across all three defining transformations. The authorized waiver changes only the former stop action. It does not change or erase that result, and the factorization was not reopened.",
            "",
            "P is a compact channel subspace strongly visible to the exact RR100 readout; Q is its 120-dimensional complementary control. Neither label asserts exclusive motion or content semantics. Total motion fraction and energy per dimension answer different descriptive questions and remain separate below, but P is not required to carry more movement energy per dimension than Q. The failed factorization is therefore not negative evidence for a recurrent shifter; registration and causal tests decide that question.",
            "",
            *[
                f"- {MOTION_ENERGY_LABEL[key]}: P carries {q['p_motion_total_fraction'][key]:.2%} of total movement-change energy; P/Q energy per dimension is {q['p_to_q_motion_energy_per_dimension'][key]:.2f} for movement, {q['p_to_q_content_image_mean_energy_per_dimension'][key]:.2f} for stabilized image-mean content, and {q['p_to_q_trajectory_energy_per_dimension'][key]:.2f} for trajectory variation. Complementary-Q baseline normalized-map recovery R²={q['q_baseline_normalized_map_recovery_r2'][key]:.3f}; P-only versus Q-only direct defining-map recovery R²={q['p_movement_map_recovery_r2'][key]:.3f} versus {q['q_movement_map_recovery_r2'][key]:.3f}. Absolute preactivation image variance fractions are P={q['absolute_preactivation_variance_fraction']['p'][key]:.3f}, Q={q['absolute_preactivation_variance_fraction']['q'][key]:.3f}, and P–Q covariance={q['absolute_preactivation_variance_fraction']['p_q_covariance'][key]:.3f}; the readout bias is added once."
                for key in CONTRASTS
            ],
            "",
            "Ridge probes of image identity, motion scale, path length, and retinal displacement are descriptive decodability analyses only; their fold-wise P/Q median scores are preserved in `statistics.json`. Per-unit activity-weighted P reliance was associated with historical SF, movement benefit, and the direct 1×3× high-motion change after averaging held-out folds within unit; those rank associations are also saved in `statistics.json`. Neither analysis localizes a biological circuit.",
            "",
            "## D. Registration computation",
            "",
            "The checkpoint implements eight ConvGRU steps inside each independently scored 32-lag input window. Lag 0 is current and lag 31 is oldest; the recurrent loop traverses feature supports from relatively newer toward relatively older evidence. This is backward in retinal time and is not recurrence across the 40 scored movie outputs.",
            "",
            f"For boundary-resolved held-out learned-P pairs, the median raw lag was {q['raw_and_recurrent_lag']['median_raw_lag_feature_px']:.3f} feature pixels and the median recurrent lag was {q['raw_and_recurrent_lag']['median_recurrent_lag_feature_px']:.3f}; the median paired reduction was {q['raw_and_recurrent_lag']['median_paired_lag_reduction_feature_px']:+.3f} feature pixels.",
            "",
            f"Across boundary-resolved learned-P rows, inferred transport versus synthetically calibrated retinal displacement had vector r={q['boundary_resolved_transport']['vector_correlation']:.3f}, identity-line variance explained={q['boundary_resolved_transport']['identity_line_variance_explained']:.3f}, and median vector error={q['boundary_resolved_transport']['median_vector_error_feature_px']:.3f} feature pixels. Boundary peaks and expected shifts outside the search window were treated as unresolved, never as measured lags.",
            "",
            "Rank-matched specificity comparison (identity-line R² / median error in feature pixels): "
            + "; ".join(
                f"{method}={q['transport_by_subspace'][method]['identity_line_variance_explained']:.3f}/{q['transport_by_subspace'][method]['median_vector_error_feature_px']:.3f}"
                for method in (
                    "learned P",
                    "complementary Q",
                    "readout-SVD",
                    "random rank-8",
                )
            )
            + ". Random subspaces were averaged within each exact held-out identity before this comparison.",
            "",
            *[
                f"- {CONTRAST_LABEL[key]}: recurrent transformation minus raw-state zero-lag alignment={q['p_zero_lag_alignment_improvement'][key]:+.3f}."
                for key in CONTRASTS
            ],
            f"- Higher-SF reversal-projector median transport error: 1×={_scale_value(q['high_sf_transport_error_by_scale_feature_px'], 1.0):.3f}, 3×={_scale_value(q['high_sf_transport_error_by_scale_feature_px'], 3.0):.3f} feature px.",
            "- That 1×→3× residual increase is descriptive only: because displacement tracking and recurrent alignment were not positive, it cannot establish failure of a registration mechanism.",
            *[
                f"- Learned-P {kernel} off-center recurrent energy fraction={value:.3f}."
                for kernel, value in sorted(q['learned_p_kernel_offcenter_fraction'].items())
            ],
            "",
            "Recurrent candidate/reset/update off-center kernel energy is descriptive. It supports a circuit interpretation only when it agrees with activation-level transport and frozen-weight interventions.",
            "",
            "## E. Causal interventions",
            "",
            "Center-only interventions retain center-tap channel mixing; offset permutations retain every channel matrix, center tap, parameter count, and norm while destroying learned offset geometry. The no-recurrence condition is a reference, not the primary control.",
            "",
            *[
                f"- {CONTRAST_LABEL[key]} exact ΔSSI (bits): intact={q['transport_ssi_contrasts']['intact'][key]:+.4f}, all-recurrent center-only={q['transport_ssi_contrasts']['all_recurrent_center_only'][key]:+.4f}, offset-permuted mean={q['transport_ssi_contrasts']['offset_permuted_mean'][key]:+.4f}."
                for key in CONTRASTS
            ],
            "- Geometry conditions meeting the fixed ≥0.002-bit attenuation floor in all three defining contrasts: "
            + (", ".join(q['transport_ssi_contrasts']['qualifying_geometry_conditions']) or "none")
            + f"; maximum stabilized-map fidelity R²={q['transport_ssi_contrasts']['maximum_stabilized_map_fidelity_r2']:.3f}.",
            "",
            f"High-SF 3× exact SSI was {q['realignment_high3_ssi']['intact']:.6f} intact and {q['realignment_high3_ssi']['correct_p']:.6f} after eye-derived P realignment, a change of {1e6 * q['realignment_high3_gain_vs_intact_bits']['correct_p']:+.2f} µbits. Opposite, matched-random, and Q shifts changed SSI by {1e6 * q['realignment_high3_gain_vs_intact_bits']['opposite_p']:+.2f}, {1e6 * q['realignment_high3_gain_vs_intact_bits']['random_p']:+.2f}, and {1e6 * q['realignment_high3_gain_vs_intact_bits']['correct_q']:+.2f} µbits. Activation-oracle P realignment changed it by {1e6 * q['realignment_high3_gain_vs_intact_bits']['oracle_p_upper_bound']:+.2f} µbits and is only an upper bound; none rescued the loss by the 2000-µbit criterion.",
            "",
            f"At useful high-SF 1× motion, exact SSI changed from {q['induced_failure_high1_ssi']['intact']:.6f} intact to {q['induced_failure_high1_ssi']['induced_p']:.6f} under induced P misalignment, a loss of only {1e6 * q['induced_failure_high1_ssi']['intact_minus_induced_bits']:.2f} µbits. This does not meet the 2000-µbit impairment criterion. Complete mean-normalized 51×51 population maps accompany these summaries on one shared scale.",
            "",
            "## F. Biological interpretation",
            "",
            biological_text,
            "",
            "This is an algorithmic interpretation of the frozen digital twin. It is not a claim that biological V1 implements the identical trained ConvGRU or a uniquely localized recurrent pathway.",
            "",
            "## G. Decision",
            "",
            "The sole decision is printed on the first page. It follows the fixed downstream registration/causal hierarchy in `statistics.json`; the waived P/Q energy gate was not used as either support or negative shifter evidence. Missing data were never interpreted as negative evidence. An activation-derived oracle is an upper bound, probe results are descriptive, and best-lag boundary cases remain unresolved.",
            "",
            "## H. Manuscript recommendation",
            "",
            statistics["manuscript_recommendation"],
            "",
            "Propose exactly three final Figure 4 mechanism panels:",
            "",
            *[
                f"{index}. {text}" for index, text in enumerate(statistics["exactly_three_manuscript_panels"], start=1)
            ],
            "",
        ]
    )
    if len(statistics["exactly_three_manuscript_panels"]) != 3:
        raise AssertionError("Report must recommend exactly three manuscript panels")
    text = "\n".join(lines)
    selected = [label for label in DECISION_LABELS if label in text]
    if selected != [statistics["decision"]] or text.count(statistics["decision"]) != 1:
        raise AssertionError("Report must contain exactly one allowed decision exactly once")
    return text


def caption_draft(statistics: Mapping[str, Any]) -> str:
    q = statistics["quantitative"]
    scope = (
        "complete 8-image × 24-trajectory confirmation"
        if statistics["causal_scope"] == "full"
        else "preregistered 4-image × 12-trajectory pilot"
    )
    correct_gain_microbits = 1e6 * float(
        q["realignment_high3_gain_vs_intact_bits"]["correct_p"]
    )
    induced_loss_microbits = 1e6 * float(
        q["induced_failure_high1_ssi"]["intact_minus_induced_bits"]
    )
    decision = str(statistics["decision"])
    title = (
        "Causal interventions do not support recurrent spatial transport"
        if decision == DECISION_LABELS[2]
        else "Causal test of recurrent spatial transport"
    )
    qualifying = q["transport_ssi_contrasts"]["qualifying_geometry_conditions"]
    center_qualifies = any("center_only" in condition for condition in qualifying)
    offset_qualifies = any("offset_permuted" in condition for condition in qualifying)
    geometry_result = (
        "Center-only manipulations changed the SSI curves, whereas offset permutation remained close to intact; this dissociation does not support a specific learned spatial-transport geometry."
        if center_qualifies and not offset_qualifies
        else "The fixed center-only and offset-permutation criteria are reported without assigning them a spatial-transport interpretation in isolation."
    )
    correct_result = (
        "rescued the loss"
        if float(q["realignment_high3_gain_vs_intact_bits"]["correct_p"])
        >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
        else "did not rescue the loss"
    )
    induced_result = (
        "impaired sharpening"
        if float(q["induced_failure_high1_ssi"]["intact_minus_induced_bits"])
        >= MIN_DIRECTIONAL_CAUSAL_ATTENUATION_BITS
        else "did not impair sharpening"
    )
    tracking_result = (
        "tracks"
        if statistics["support_gates"]["inferred_transport_tracks_retinal_displacement"]
        else "does not track"
    )
    alignment_result = (
        "improves"
        if statistics["support_gates"]["recurrence_reduces_lag"]
        else "reduces rather than improves"
    )
    return f"""# Figure 3 caption draft

**Figure 3 | {title} ({scope}).** **A,** Exact intact RR100 SSI dose curves for the historical lower-SF (71 units) and higher-SF (29 units) populations. **B,** Movement-dependent SSI relative to each condition's own stabilized value after retaining only center taps in the candidate recurrent kernel, recurrent gate kernels, or all recurrent kernels, and after three fixed norm-preserving permutations of all off-center offsets (mean shown; exact seeds are retained in the plotting data). {geometry_result} Stabilized complete-map fidelity is reported separately so general disruption is not mistaken for selective loss of movement dependence. **C,** At higher-SF 3× motion, changes from intact are plotted in microbits without an offset axis, with the exact SSI printed beneath each complete expected-spike-weighted mean-normalized 51×51 rate map. Eye-derived P realignment changed SSI by {correct_gain_microbits:+.2f} µbits and {correct_result}; opposite, matched-random, and complementary-Q controls are shown alongside it. **D,** At useful higher-SF 1× motion, induced P misalignment reduced SSI by {induced_loss_microbits:.2f} µbits and {induced_result} by the fixed 0.002-bit criterion. All C/D maps share one scale around g(x,y)=1 and no condition is independently rescaled.

The eye-derived shift is primary. Activation-derived oracle realignment ({q['realignment_high3_ssi']['oracle_p']:.6f} SSI versus {q['realignment_high3_ssi']['intact']:.6f} intact) is omitted from the main panel and retained only as an upper bound, not a biologically available computation. Gates, current-input evidence, channel amplitudes, downstream readout, unit definitions, spatial tiling, and softplus are otherwise unchanged. These interventions act within the eight-step ConvGRU replay from newer-support toward older-support inside each corrected 32-lag input window; they are not recurrence across the 40 scored outputs. Fold-specific projectors are used for inference, while consensus projectors remain visualization-only. Together with the completed held-out finding that inferred transport {tracking_result} calibrated retinal displacement and recurrence {alignment_result} alignment, the sole audit decision is **{statistics['decision']}**.
"""


def _save_figure_atomic(figure: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    figure.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def run_stopped_after_pq(
    inputs: UpstreamInputs,
    *,
    figure1_manifest: Path,
) -> dict[str, Any]:
    """Reject the retired stop path without publishing a partial decision."""
    del inputs, figure1_manifest
    raise DataUnavailable(
        "The P/Q motion-energy stop is waived; a P/Q-only report cannot select a "
        "final decision. Complete registration and causal saved products are required."
    )

def run(
    *,
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    figure1_manifest: Path = DEFAULT_FIGURE1,
    figure2_manifest: Path = DEFAULT_FIGURE2,
) -> dict[str, Any]:
    inputs = load_final_inputs(
        input_dir,
        figure1_manifest=figure1_manifest,
        figure2_manifest=figure2_manifest,
    )
    legacy_report = inputs.root / "CONVGRU_REGISTRATION_MECHANISM_REPORT.md"
    legacy_archive = inputs.root / "CONVGRU_REGISTRATION_MECHANISM_REPORT.superseded.md"
    if legacy_report.is_file():
        legacy_text = legacy_report.read_text(encoding="utf-8")
        notice = (
            "# SUPERSEDED — DO NOT SHARE AS THE CURRENT RESULT\n\n"
            "This historical stopped-after-P/Q report predates the authorized waiver. "
            "The canonical report is `REGISTRATION_TEST_REPORT.md`; the old P/Q failure "
            "is preserved there but is not treated as negative shifter evidence.\n\n"
            "---\n\n"
        )
        _atomic_text(legacy_archive, notice + legacy_text)
        legacy_report.unlink()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_data_dir = output / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    figure_data = prepare_figure3_data(inputs.causal)
    data_paths = {
        "panel_a": plot_data_dir / "panel_a_intact_ssi_dose.csv",
        "panel_b": plot_data_dir / "panel_b_transport_controls.csv",
        "panel_b_source": plot_data_dir / "panel_b_exact_offset_seed_rows.csv",
        "panel_c": plot_data_dir / "panel_c_high3_realignment.csv",
        "panel_d": plot_data_dir / "panel_d_high1_induced_misalignment.csv",
    }
    for key, path in data_paths.items():
        _atomic_csv(path, figure_data[key])
    maps_path = plot_data_dir / "figure3_complete_normalized_maps.npz"
    _atomic_npz(
        maps_path,
        panel_c_conditions=np.asarray(figure_data["panel_c_conditions"], dtype="U64"),
        panel_c_scale=np.asarray(3.0),
        panel_c_high_sf_maps=np.asarray(figure_data["panel_c_maps"], dtype=np.float32),
        panel_d_conditions=np.asarray(figure_data["panel_d_conditions"], dtype="U64"),
        panel_d_scale=np.asarray(1.0),
        panel_d_high_sf_maps=np.asarray(figure_data["panel_d_maps"], dtype=np.float32),
        common_vmin=np.asarray(figure_data["common_map_vmin"]),
        common_vmax=np.asarray(figure_data["common_map_vmax"]),
        map_definition=np.asarray("g(x,y)=r(x,y)/mean_xy r; common scale; no independent condition rescaling"),
    )

    statistics = compute_statistics(inputs)
    statistics_path = inputs.root / "statistics.json"
    _atomic_json(statistics_path, statistics)
    report_path = inputs.root / "REGISTRATION_TEST_REPORT.md"
    _atomic_text(report_path, render_report(statistics))
    caption_path = output / "FIGURE3_CAUSAL_SHIFTER_CAPTION.md"
    _atomic_text(caption_path, caption_draft(statistics))

    figure = draw_figure3(
        figure_data, scope=inputs.causal.scope, decision=statistics["decision"]
    )
    stem = output / "registration_mechanism_figure3_causal_shifter"
    figure_paths = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
    }
    _save_figure_atomic(figure, figure_paths["pdf"], bbox_inches="tight")
    _save_figure_atomic(figure, figure_paths["svg"], bbox_inches="tight")
    _save_figure_atomic(figure, figure_paths["png"], dpi=600, bbox_inches="tight")
    plt.close(figure)

    figure_manifest = {
        "schema_version": "fig4-registration-causal-figure3-v2",
        "status": "complete",
        "saved_products_only": True,
        "model_imported": False,
        "state_or_readout_cache_read": False,
        "causal_scope": inputs.causal.scope,
        "result": (
            "causal interventions do not support recurrent spatial transport"
            if statistics["decision"] == DECISION_LABELS[2]
            else "see fixed four-way decision in statistics.json"
        ),
        "foldwise_projectors_for_inference": True,
        "consensus_projectors_used_for_inference": False,
        "consensus_role": "visualization only",
        "within_window_direction": "newer-support to older-support; backward in retinal time",
        "not_recurrence_across_40_outputs": True,
        "oracle_role": "activation-derived upper bound only; excluded from the main panel",
        "source_products": [_file_record(path) for path in inputs.causal.source_paths],
        "exact_plotting_data": {
            **{key: _file_record(path, rows=len(figure_data[key])) for key, path in data_paths.items()},
            "complete_normalized_maps": _file_record(maps_path),
        },
        "figure_exports": {key: _file_record(path) for key, path in figure_paths.items()},
        "caption": _file_record(caption_path),
        "map_scale": {
            "common_across_panels_c_and_d": True,
            "vmin": figure_data["common_map_vmin"],
            "vmax": figure_data["common_map_vmax"],
            "individual_condition_rescaling": False,
        },
        "ssi_display": {
            "panels_c_and_d": "delta SSI from intact in microbits; exact SSI printed below each map",
            "scientific_offset_axis_used": False,
        },
    }
    figure_manifest_path = output / "figure3_causal_shifter_plot_manifest.json"
    _atomic_json(figure_manifest_path, figure_manifest)

    optional_figure4: dict[str, Any] = {
        "status": "not_generated",
        "reason": "all SHIFTER support gates and complete 8×24 confirmation are required",
    }
    optional_paths: list[Path] = []
    if statistics["decision"] == DECISION_LABELS[0]:
        optional_data = prepare_optional_figure4_data(statistics, figure_data["panel_a"])
        optional_data_path = output / "plot_data/optional_figure4_measured_values.csv"
        _atomic_csv(optional_data_path, optional_data)
        optional_figure = draw_optional_figure4(statistics, optional_data)
        optional_stem = output / "registration_mechanism_figure4_supported_circuit_summary"
        optional_exports = {
            "pdf": optional_stem.with_suffix(".pdf"),
            "svg": optional_stem.with_suffix(".svg"),
            "png": optional_stem.with_suffix(".png"),
        }
        _save_figure_atomic(optional_figure, optional_exports["pdf"], bbox_inches="tight")
        _save_figure_atomic(optional_figure, optional_exports["svg"], bbox_inches="tight")
        _save_figure_atomic(optional_figure, optional_exports["png"], dpi=600, bbox_inches="tight")
        plt.close(optional_figure)
        optional_caption_path = output / "FIGURE4_SUPPORTED_CIRCUIT_SUMMARY_CAPTION.md"
        _atomic_text(
            optional_caption_path,
            "# Figure 4 caption draft\n\n"
            "**Compact supported circuit summary.** This schematic was generated only after every "
            "predeclared SHIFTER/registration gate and the complete 8-image × 24-trajectory "
            "confirmation passed. Current evidence is combined with spatially transported recurrent "
            "evidence within each independently scored corrected-history window. The adjacent curves "
            "are the measured intact lower-/higher-SF SSI values normalized within population; their "
            "optima and the printed higher-SF residual errors are calculated from saved products, not "
            "drawn assumptions. The summary is algorithmic and does not claim an identical biological "
            "ConvGRU or a uniquely localized pathway.\n",
        )
        optional_manifest_path = output / "figure4_supported_circuit_summary_manifest.json"
        optional_figure4 = {
            "status": "complete",
            "generated_only_after_all_support_gates": True,
            "derived_from_measured_values": True,
            "evidentiary_role": "summary only; not independent evidence",
            "plot_data": _file_record(optional_data_path, rows=len(optional_data)),
            "exports": {key: _file_record(path) for key, path in optional_exports.items()},
            "caption": _file_record(optional_caption_path),
        }
        _atomic_json(optional_manifest_path, optional_figure4)
        optional_figure4["manifest"] = _file_record(optional_manifest_path)
        optional_paths = [
            optional_data_path,
            optional_caption_path,
            optional_manifest_path,
            *optional_exports.values(),
        ]

    required_outputs = [
        *(inputs.root / name for name in REQUIRED_CANONICAL),
        statistics_path,
        report_path,
        figure_manifest_path,
        caption_path,
        maps_path,
        *data_paths.values(),
        *figure_paths.values(),
        *optional_paths,
    ]
    if any(not path.is_file() for path in required_outputs):
        raise DataUnavailable("A required final output vanished before manifest publication")
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "saved_products_only": True,
        "model_imported": False,
        "model_or_renderer_executed": False,
        "state_or_readout_cache_read": False,
        "decision": statistics["decision"],
        "manuscript_recommendation": statistics["manuscript_recommendation"],
        "causal_scope": inputs.causal.scope,
        "foldwise_semantic_inference": True,
        "foldwise_causal_projectors": True,
        "consensus_projectors_for_visualization_only": True,
        "within_window_backward_time_convention": True,
        "not_recurrence_across_40_outputs": True,
        "boundary_cases_inferred_as_negative": False,
        "missing_products_inferred_as_negative": False,
        "oracle_is_upper_bound_only": True,
        "descriptive_probes_called_causal": False,
        "unique_recurrent_pathway_claimed": False,
        "pq_motion_energy_gate_waiver": statistics["pq_motion_energy_gate_waiver"],
        "pq_factorization_reopened": False,
        "pq_factorization_used_as_negative_shifter_evidence": False,
        "downstream_registration_and_causal_evidence_used_for_decision": True,
        "legacy_stopped_report": (
            {
                **_file_record(legacy_archive),
                "status": "superseded_and_archived",
                "canonical": False,
            }
            if legacy_archive.is_file()
            else {"status": "absent", "canonical": False}
        ),
        "sources": [_file_record(path) for path in inputs.source_paths],
        "required_canonical_products": {
            name: _file_record(inputs.root / name, rows=len(pd.read_csv(inputs.root / name)) if name.endswith(".csv") else None)
            for name in REQUIRED_CANONICAL
        },
        "figure_manifests": {
            "figure1": _file_record(Path(figure1_manifest).resolve()),
            "figure2": _file_record(Path(figure2_manifest).resolve()),
            "figure3": _file_record(figure_manifest_path),
            "figure4_optional_supported_summary": optional_figure4,
        },
        "report": _file_record(report_path),
        "statistics": _file_record(statistics_path),
        "self_hash": "omitted because a cryptographic hash of a file cannot be embedded recursively in itself",
    }
    root_manifest_path = inputs.root / "analysis_manifest.json"
    _atomic_json(root_manifest_path, root_manifest)
    return {
        "decision": statistics["decision"],
        "manuscript_recommendation": statistics["manuscript_recommendation"],
        "causal_scope": inputs.causal.scope,
        "report": str(report_path),
        "statistics": str(statistics_path),
        "analysis_manifest": str(root_manifest_path),
        "figure3_manifest": str(figure_manifest_path),
        "figure3_exports": {key: str(value) for key, value in figure_paths.items()},
        "figure4_optional_supported_summary": optional_figure4,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure1-manifest", type=Path, default=DEFAULT_FIGURE1)
    parser.add_argument("--figure2-manifest", type=Path, default=DEFAULT_FIGURE2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            figure1_manifest=args.figure1_manifest,
            figure2_manifest=args.figure2_manifest,
        )
    except DataUnavailable as error:
        raise SystemExit(f"Final registration report unavailable: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
