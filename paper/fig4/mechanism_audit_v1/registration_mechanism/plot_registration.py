#!/usr/bin/env python3
"""Build Figure 2 from finalized, fold-wise registration products only.

This module deliberately imports no model, renderer, cache, or core-scoring
code.  It fails closed unless the rank-8 marker, all 48 held-out parts, the
projected-term archive, the objective held-out example, and the exact saved SSI
table are complete and mutually consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
OUTPUT = SOURCE / "figure2_registration"
SSI_SOURCE = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/ssi_mechanism_v2/plot_data/"
    "exact_ssi_decomposition_curves.csv"
)
SCALES = (0.0, 0.5, 1.0, 2.0, 3.0)
CONTRASTS = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
BASE_COLUMNS = (
    "scope", "contrast", "fold", "image_position", "trajectory_position",
    "scale", "frame_position", "internal_step", "subspace",
    "eye_delta_x_deg", "eye_delta_y_deg",
    "expected_feature_shift_x_px", "expected_feature_shift_y_px",
    "raw_zero_lag_correlation", "raw_best_lag_correlation",
    "raw_lag_x_px", "raw_lag_y_px", "raw_at_search_boundary",
    "raw_peak_sharpness", "raw_residual_mismatch",
    "recurrent_zero_lag_correlation", "recurrent_best_lag_correlation",
    "recurrent_lag_x_px", "recurrent_lag_y_px", "recurrent_at_search_boundary",
    "recurrent_peak_sharpness", "recurrent_residual_mismatch",
    "transport_x_px", "transport_y_px", "expected_outside_search_window",
    "zero_lag_alignment_improvement", "best_lag_alignment_improvement", "valid",
)
IDENTITY_COLUMNS = (
    "scope", "contrast", "fold", "image_position", "trajectory_position",
    "scale", "frame_position", "internal_step",
)
METHOD_ORDER = ("learned P", "complementary Q", "readout-SVD", "random rank-8")
METHOD_STYLE = {
    "learned P": ("#60469C", "o"),
    "complementary Q": ("#6F777D", "s"),
    "readout-SVD": ("#2E7D9A", "D"),
    "random rank-8": ("#B7A9A0", "^"),
}
GROUP_STYLE = {
    "lower SF": ("#2878B5", "o", "-"),
    "higher SF": ("#D6651A", "o", "-"),
    "higher SF—sharpening space": ("#D6651A", "o", "-"),
    "higher SF—reversal space": ("#D6651A", "s", "--"),
}


class DataUnavailable(RuntimeError):
    """A production figure cannot be made without inventing missing data."""


@dataclass(frozen=True)
class CompletenessContract:
    pairs: int = 48
    folds: int = 4
    pairs_per_fold: int = 12
    frames: int = 40
    internal_registration_steps: int = 7
    contrasts: tuple[str, ...] = CONTRASTS
    scales: tuple[float, ...] = SCALES
    term_rows: int = 230_400


PRODUCTION = CompletenessContract()


@dataclass(frozen=True)
class Inputs:
    registration: pd.DataFrame
    exact_ssi: pd.DataFrame
    example: dict[str, Any]
    gate: dict[str, Any]
    consolidation: dict[str, Any]
    source_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--exact-ssi", type=Path, default=SSI_SOURCE)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataUnavailable(f"Missing required finalized marker: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataUnavailable(f"Unreadable finalized marker: {path}") from error
    if not isinstance(value, dict):
        raise DataUnavailable(f"Marker is not a JSON object: {path}")
    return value


def _as_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lowered = series.astype(str).str.lower()
    if not lowered.isin(("true", "false")).all():
        raise DataUnavailable(f"{label} contains values other than true/false")
    return lowered.eq("true")


def validate_rank_gate(gate: dict[str, Any]) -> None:
    if gate.get("status") != "complete":
        raise DataUnavailable("Rank-8 validation marker is not complete")
    if bool(gate.get("stop_downstream_mechanism_audit", True)):
        raise DataUnavailable(
            "Predeclared held-out rank-8 generalization gate stopped the mechanism audit"
        )
    heldout = gate.get("heldout_generalization")
    if not isinstance(heldout, dict) or set(heldout) != set(CONTRASTS):
        raise DataUnavailable("Rank-8 marker lacks all three held-out contrast gates")
    if not all(bool(heldout[key].get("generalizes", False)) for key in CONTRASTS):
        raise DataUnavailable("At least one contrast fails held-out generalization")


def validate_registration_frame(
    frame: pd.DataFrame,
    consolidation: dict[str, Any],
    term_archive: dict[str, np.ndarray],
    *,
    contract: CompletenessContract = PRODUCTION,
) -> pd.DataFrame:
    missing = sorted(set(BASE_COLUMNS) - set(frame.columns))
    if missing:
        raise DataUnavailable(f"Registration CSV lacks columns: {', '.join(missing)}")
    settings = consolidation.get("registration_settings", {})
    random_draws = int(settings.get("random_draws", -1))
    expected_subspaces = {"learned_p", "learned_q", "readout_svd"} | {
        f"random_{draw:02d}" for draw in range(random_draws)
    }
    expected_rows = (
        contract.pairs
        * len(contract.scales)
        * contract.frames
        * contract.internal_registration_steps
        * len(contract.contrasts)
        * len(expected_subspaces)
    )
    required_marker = {
        "scope": "heldout",
        "registration_parts": contract.pairs,
        "term_parts": contract.pairs,
        "registration_rows": expected_rows,
        "projected_term_rows": contract.term_rows,
    }
    for key, expected in required_marker.items():
        if consolidation.get(key) != expected:
            raise DataUnavailable(
                f"Held-out consolidation marker {key}={consolidation.get(key)!r}; "
                f"expected {expected!r}"
            )
    if len(frame) != expected_rows:
        raise DataUnavailable(f"Registration row count {len(frame)} != {expected_rows}")
    if set(frame.scope.astype(str)) != {"heldout"}:
        raise DataUnavailable("Registration inference is not exclusively fold-wise held-out")
    if set(frame.contrast.astype(str)) != set(contract.contrasts):
        raise DataUnavailable("Registration CSV does not contain exactly three contrasts")
    if set(frame.subspace.astype(str)) != expected_subspaces:
        raise DataUnavailable("Registration CSV subspaces differ from its finalized marker")
    if set(pd.to_numeric(frame.scale).astype(float)) != set(contract.scales):
        raise DataUnavailable("Registration CSV movement scales are incomplete")
    if set(pd.to_numeric(frame.internal_step).astype(int)) != set(
        range(1, contract.internal_registration_steps + 1)
    ):
        raise DataUnavailable("Registration CSV internal steps are incomplete")
    if set(pd.to_numeric(frame.frame_position).astype(int)) != set(range(contract.frames)):
        raise DataUnavailable("Registration CSV scored frames are incomplete")
    pair_counts = (
        frame[["fold", "image_position", "trajectory_position"]]
        .drop_duplicates()
        .groupby("fold")
        .size()
    )
    if len(pair_counts) != contract.folds or not np.all(
        pair_counts.to_numpy() == contract.pairs_per_fold
    ):
        raise DataUnavailable("Held-out folds do not contain the expected disjoint pair count")
    key = list(IDENTITY_COLUMNS) + ["subspace"]
    if frame.duplicated(key).any():
        raise DataUnavailable("Registration CSV has duplicate observation/subspace rows")
    for name in ("valid", "raw_at_search_boundary", "recurrent_at_search_boundary", "expected_outside_search_window"):
        frame[name] = _as_bool(frame[name], name)
    valid = frame.valid.to_numpy(bool)
    numeric = [
        name for name in BASE_COLUMNS
        if name not in set(IDENTITY_COLUMNS) | {
            "subspace", "valid", "raw_at_search_boundary",
            "recurrent_at_search_boundary", "expected_outside_search_window",
        }
    ]
    converted = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(converted.loc[valid].to_numpy(dtype=float)).all():
        raise DataUnavailable("Valid registration rows contain nonfinite metrics")
    frame[numeric] = converted
    metadata = np.asarray(term_archive.get("metadata"))
    values = np.asarray(term_archive.get("values"))
    if metadata.shape[0] != contract.term_rows or values.shape[0] != contract.term_rows:
        raise DataUnavailable("Projected-term archive is not the complete held-out replay")
    scope = str(np.asarray(term_archive.get("scope", "")).item())
    if scope != "heldout":
        raise DataUnavailable("Projected-term archive is not fold-wise held-out")
    return frame


def validate_exact_ssi(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"figure4_sf_group", "condition", "scale", "ssi"}
    if not required.issubset(frame.columns):
        raise DataUnavailable(f"Exact SSI table lacks {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame.condition.eq("normal_moving"), list(required)].copy()
    frame["scale"] = pd.to_numeric(frame.scale, errors="coerce")
    frame["ssi"] = pd.to_numeric(frame.ssi, errors="coerce")
    if len(frame) != 10 or set(frame.figure4_sf_group) != {"low", "high"}:
        raise DataUnavailable("Exact normal-movie SSI table is not exactly 2 groups × 5 scales")
    if set(frame.scale.astype(float)) != set(SCALES) or not np.isfinite(frame.ssi).all():
        raise DataUnavailable("Exact normal-movie SSI scales/values are incomplete")
    if frame.duplicated(["figure4_sf_group", "scale"]).any():
        raise DataUnavailable("Exact SSI table has duplicate group/scale rows")
    return frame.sort_values(["figure4_sf_group", "scale"]).reset_index(drop=True)


def load_example(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataUnavailable(f"Missing objective held-out full example: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "metadata_json", "learned_projector_basis", "h_previous",
            "candidate_current_preactivation", "candidate_recurrent_preactivation",
        }
        if not required.issubset(archive.files):
            raise DataUnavailable(f"Held-out example lacks {sorted(required - set(archive.files))}")
        value = {name: np.asarray(archive[name]) for name in required if name != "metadata_json"}
        value["metadata"] = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    meta = value["metadata"]
    if meta.get("scope") != "heldout" or not bool(
        meta.get("equation_reconstruction_verified_before_quantization", False)
    ):
        raise DataUnavailable("Objective example is not verified fold-wise held-out data")
    basis = value["learned_projector_basis"]
    if basis.shape != (128, 8) or not np.allclose(basis.T @ basis, np.eye(8), atol=3e-4):
        raise DataUnavailable("Objective example projector is not a valid 128×8 basis")
    for name in ("h_previous", "candidate_current_preactivation", "candidate_recurrent_preactivation"):
        if value[name].shape != (8, 128, 64, 64) or not np.isfinite(value[name]).all():
            raise DataUnavailable(f"Objective example has invalid {name} shape/values")
    return value


def load_inputs(input_dir: Path, exact_ssi_path: Path) -> Inputs:
    gate_path = input_dir / "rank8_validation_gate.json"
    consolidation_path = input_dir / "gru_instrumentation/consolidation_heldout.json"
    registration_path = input_dir / "registration_metrics_heldout.csv"
    terms_path = input_dir / "gru_projected_terms_heldout.npz"
    example_path = input_dir / "gru_instrumentation/full_examples/heldout_objective_example.npz"
    gate = _load_json(gate_path)
    validate_rank_gate(gate)
    consolidation = _load_json(consolidation_path)
    for path in (registration_path, terms_path, exact_ssi_path):
        if not path.is_file():
            raise DataUnavailable(f"Missing required saved product: {path}")
    registration = pd.read_csv(registration_path, low_memory=False)
    with np.load(terms_path, allow_pickle=False) as archive:
        term_archive = {name: np.asarray(archive[name]) for name in archive.files}
    registration = validate_registration_frame(registration, consolidation, term_archive)
    exact_ssi = validate_exact_ssi(pd.read_csv(exact_ssi_path))
    example = load_example(example_path)
    meta = example["metadata"]
    selected = registration.loc[
        registration.fold.eq(int(meta["fold"]))
        & registration.image_position.eq(int(meta["image_position"]))
        & registration.trajectory_position.eq(int(meta["trajectory_position"]))
        & registration.contrast.eq(str(meta["contrast"]))
        & np.isclose(registration.scale, float(meta["scale"]))
        & registration.frame_position.eq(int(meta["frame"]))
        & registration.internal_step.eq(int(meta["internal_step"]))
        & registration.subspace.eq("learned_p")
    ]
    if len(selected) != 1:
        raise DataUnavailable("Objective example does not map to exactly one held-out registration row")
    return Inputs(
        registration=registration,
        exact_ssi=exact_ssi,
        example=example,
        gate=gate,
        consolidation=consolidation,
        source_paths=(gate_path, consolidation_path, registration_path, terms_path, example_path, exact_ssi_path),
    )


def method_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = result.subspace.map(
        {"learned_p": "learned P", "learned_q": "complementary Q", "readout_svd": "readout-SVD"}
    ).fillna("random rank-8")
    random = result.method.eq("random rank-8")
    fixed = result.loc[~random].copy()
    if not random.any():
        raise DataUnavailable("Registration products lack random rank-8 controls")
    numeric = [
        "eye_delta_x_deg", "eye_delta_y_deg",
        "expected_feature_shift_x_px", "expected_feature_shift_y_px",
        "raw_zero_lag_correlation", "raw_best_lag_correlation", "raw_lag_x_px", "raw_lag_y_px",
        "raw_peak_sharpness", "raw_residual_mismatch",
        "recurrent_zero_lag_correlation", "recurrent_best_lag_correlation",
        "recurrent_lag_x_px", "recurrent_lag_y_px", "recurrent_peak_sharpness",
        "recurrent_residual_mismatch", "transport_x_px", "transport_y_px",
        "zero_lag_alignment_improvement", "best_lag_alignment_improvement",
    ]
    grouped = result.loc[random].groupby(list(IDENTITY_COLUMNS) + ["method"], as_index=False)
    averaged = grouped[numeric].mean()
    flags = grouped[["valid", "raw_at_search_boundary", "recurrent_at_search_boundary", "expected_outside_search_window"]].agg(
        {
            "valid": "all", "raw_at_search_boundary": "any",
            "recurrent_at_search_boundary": "any", "expected_outside_search_window": "any",
        }
    )
    averaged = averaged.merge(flags, on=list(IDENTITY_COLUMNS) + ["method"], validate="one_to_one")
    averaged["subspace"] = "mean_of_random_draws"
    output = pd.concat([fixed, averaged], ignore_index=True, sort=False)
    output["unresolved_boundary"] = (
        output.raw_at_search_boundary
        | output.recurrent_at_search_boundary
        | output.expected_outside_search_window
    )
    return output


def transport_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = frame.loc[
            frame.method.eq(method) & frame.valid & ~frame.unresolved_boundary
        ]
        expected = selected[["expected_feature_shift_x_px", "expected_feature_shift_y_px"]].to_numpy(float)
        measured = selected[["transport_x_px", "transport_y_px"]].to_numpy(float)
        if len(measured) < 3:
            raise DataUnavailable(f"Too few resolved transport rows for {method}")

        def regress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
            design = np.column_stack([x, np.ones(len(x))])
            coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            predicted = design @ coef
            sse = float(np.square(y - predicted).sum())
            sst = float(np.square(y - y.mean()).sum())
            return float(coef[0]), float(coef[1]), 1.0 - sse / max(sst, 1e-30)

        sx, ix, r2x = regress(expected[:, 0], measured[:, 0])
        sy, iy, r2y = regress(expected[:, 1], measured[:, 1])
        corr = float(np.corrcoef(expected.ravel(), measured.ravel())[0, 1])
        error = np.linalg.norm(measured - expected, axis=1)
        vector_r2 = 1.0 - float(np.square(measured - expected).sum()) / max(
            float(np.square(expected - expected.mean(axis=0)).sum()), 1e-30
        )
        rows.append(
            {
                "method": method, "n_resolved_vectors": len(measured),
                "slope_x": sx, "intercept_x": ix, "r2_x": r2x,
                "slope_y": sy, "intercept_y": iy, "r2_y": r2y,
                "vector_correlation": corr,
                "median_displacement_error_px": float(np.median(error)),
                "vector_variance_explained": vector_r2,
            }
        )
    return pd.DataFrame(rows)


def binned_transport(frame: pd.DataFrame, bins: int = 13) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = frame.loc[frame.method.eq(method) & frame.valid & ~frame.unresolved_boundary]
        for component in ("x", "y"):
            x = selected[f"expected_feature_shift_{component}_px"].to_numpy(float)
            y = selected[f"transport_{component}_px"].to_numpy(float)
            order = np.argsort(x, kind="stable")
            for index, positions in enumerate(np.array_split(order, bins)):
                if len(positions) == 0:
                    continue
                rows.append(
                    {
                        "method": method, "component": component, "bin": index,
                        "n": len(positions), "expected_shift_px": float(np.mean(x[positions])),
                        "mean_transport_px": float(np.mean(y[positions])),
                        "sem_transport_px": float(np.std(y[positions], ddof=1) / np.sqrt(len(positions)))
                        if len(positions) > 1 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _high_shared(gate: dict[str, Any]) -> bool:
    value = gate.get("shared_higher_sf_consensus", {})
    return bool(isinstance(value, dict) and value.get("allowed", False))


def population_assignments(frame: pd.DataFrame, shared_high: bool) -> pd.DataFrame:
    selected = frame.loc[frame.method.eq("learned P")].copy()
    selected["population"] = selected.contrast.map({"low_0_to_2": "lower SF"})
    if shared_high:
        selected.loc[selected.contrast.isin(("high_0_to_1", "high_1_to_3")), "population"] = "higher SF"
    else:
        selected.loc[selected.contrast.eq("high_0_to_1"), "population"] = "higher SF—sharpening space"
        selected.loc[selected.contrast.eq("high_1_to_3"), "population"] = "higher SF—reversal space"
    return selected


def alignment_by_scale(frame: pd.DataFrame, shared_high: bool) -> pd.DataFrame:
    selected = population_assignments(frame, shared_high)
    selected = selected.loc[selected.valid]
    per_pair = (
        selected.groupby(["population", "fold", "image_position", "trajectory_position", "scale"], as_index=False)
        .zero_lag_alignment_improvement.mean()
    )
    summary = per_pair.groupby(["population", "scale"]).zero_lag_alignment_improvement.agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["sem"]
    return summary


def residual_vs_ssi(
    frame: pd.DataFrame, exact_ssi: pd.DataFrame, shared_high: bool
) -> pd.DataFrame:
    selected = population_assignments(frame, shared_high)
    selected = selected.loc[selected.valid & ~selected.unresolved_boundary].copy()
    selected["transport_error_px"] = np.hypot(
        selected.transport_x_px - selected.expected_feature_shift_x_px,
        selected.transport_y_px - selected.expected_feature_shift_y_px,
    )
    per_pair = (
        selected.groupby(["population", "fold", "image_position", "trajectory_position", "scale"], as_index=False)
        .transport_error_px.median()
    )
    summary = per_pair.groupby(["population", "scale"]).transport_error_px.agg(
        ["median", "count"]
    ).reset_index().rename(columns={"median": "median_transport_error_px"})
    summary["ssi_group"] = np.where(summary.population.eq("lower SF"), "low", "high")
    joined = summary.merge(
        exact_ssi[["figure4_sf_group", "scale", "ssi"]],
        left_on=["ssi_group", "scale"], right_on=["figure4_sf_group", "scale"],
        validate="many_to_one",
    )
    if len(joined) != len(summary):
        raise DataUnavailable("Residual-alignment rows do not join exactly to saved final SSI")
    return joined.drop(columns="figure4_sf_group")


def residual_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive five-scale associations, never promoted to independent n."""
    rows: list[dict[str, Any]] = []
    for population, selected in frame.groupby("population"):
        x = selected.median_transport_error_px.to_numpy(float)
        y = selected.ssi.to_numpy(float)
        if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
            pearson = spearman = np.nan
        else:
            pearson = float(np.corrcoef(x, y)[0, 1])
            spearman = float(
                np.corrcoef(
                    pd.Series(x).rank(method="average"),
                    pd.Series(y).rank(method="average"),
                )[0, 1]
            )
        rows.append(
            {
                "population": population,
                "n_scales": len(x),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "descriptive_only_five_scale_points": True,
            }
        )
    return pd.DataFrame(rows)


def example_plot_data(example: dict[str, Any], row: pd.Series) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    step = int(example["metadata"]["internal_step"])
    basis = np.asarray(example["learned_projector_basis"], dtype=np.float64)
    names = {
        "raw previous state": "h_previous",
        "current evidence": "candidate_current_preactivation",
        "recurrently transformed evidence": "candidate_recurrent_preactivation",
    }
    coordinates: dict[str, np.ndarray] = {}
    for label, key in names.items():
        value = np.asarray(example[key][step], dtype=np.float64)
        projected = np.einsum("ck,cyx->kyx", basis, value)
        coordinates[label] = projected - projected.mean(axis=(1, 2), keepdims=True)
    # One common, sign-canonical channel direction prevents independent map
    # rescaling or arbitrary per-map component choices.
    samples = np.concatenate([value.reshape(8, -1) for value in coordinates.values()], axis=1)
    u, _, _ = np.linalg.svd(samples, full_matrices=False)
    direction = u[:, 0]
    largest = int(np.argmax(np.abs(direction)))
    if direction[largest] < 0:
        direction *= -1
    maps = {label: np.einsum("k,kyx->yx", direction, value) for label, value in coordinates.items()}
    metrics = pd.DataFrame(
        [
            {
                "eye_delta_x_deg": row.eye_delta_x_deg,
                "eye_delta_y_deg": row.eye_delta_y_deg,
                "retinal_image_delta_x_deg": -row.eye_delta_x_deg,
                "retinal_image_delta_y_deg": -row.eye_delta_y_deg,
                "calibrated_feature_shift_x_px": row.expected_feature_shift_x_px,
                "calibrated_feature_shift_y_px": row.expected_feature_shift_y_px,
                "raw_zero_lag_correlation": row.raw_zero_lag_correlation,
                "raw_best_lag_correlation": row.raw_best_lag_correlation,
                "raw_best_lag_x_px": row.raw_lag_x_px,
                "raw_best_lag_y_px": row.raw_lag_y_px,
                "recurrent_zero_lag_correlation": row.recurrent_zero_lag_correlation,
                "recurrent_best_lag_correlation": row.recurrent_best_lag_correlation,
                "recurrent_best_lag_x_px": row.recurrent_lag_x_px,
                "recurrent_best_lag_y_px": row.recurrent_lag_y_px,
                "raw_boundary_unresolved": bool(row.raw_at_search_boundary),
                "recurrent_boundary_unresolved": bool(row.recurrent_at_search_boundary),
                "expected_outside_search_window": bool(row.expected_outside_search_window),
            }
        ]
    )
    maps["common_component_direction"] = direction
    return maps, metrics


def _panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.08, 1.12, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.text(0.0, 1.12, title, transform=ax.transAxes, fontsize=8.4, fontweight="bold", va="top")


def draw_figure(
    maps: dict[str, np.ndarray], example_metrics: pd.DataFrame,
    binned: pd.DataFrame, statistics: pd.DataFrame,
    alignment: pd.DataFrame, residual: pd.DataFrame,
    residual_stats: pd.DataFrame,
    *, shared_high: bool,
) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.linewidth": 0.65, "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.6, 7.15), constrained_layout=False)
    fig.suptitle(
        "Recurrence does not register retained evidence to retinal displacement",
        x=0.055, y=0.985, ha="left", va="top", fontsize=10.5, fontweight="bold",
    )
    outer = fig.add_gridspec(
        3, 2, left=0.105, right=0.96, bottom=0.085, top=0.86,
        height_ratios=[1.06, 1.0, 1.0], hspace=0.64, wspace=0.43,
    )
    top = outer[0, :].subgridspec(
        1, 5, width_ratios=[0.90, 1, 1, 1, 0.94], wspace=0.16
    )
    axes_a = [fig.add_subplot(top[0, index]) for index in range(5)]
    axes_a[0].text(
        -0.08, 1.25, "A", transform=axes_a[0].transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    axes_a[0].text(
        0.04, 1.25, "Predeclared held-out example",
        transform=axes_a[0].transAxes, fontsize=8.4, fontweight="bold", va="top",
    )
    metric = example_metrics.iloc[0]
    ax = axes_a[0]
    ax.axhline(0, color="#DDDDDD", lw=0.5); ax.axvline(0, color="#DDDDDD", lw=0.5)
    for dx, dy, color, label in (
        (metric.eye_delta_x_deg, metric.eye_delta_y_deg, "#32363A", "eye"),
        (metric.retinal_image_delta_x_deg, metric.retinal_image_delta_y_deg, "#D6651A", "retinal image"),
    ):
        ax.arrow(0, 0, dx, dy, color=color, width=0.001, head_width=0.012, length_includes_head=True)
        ax.annotate(
            label, xy=(dx, dy),
            xytext=((-2, 5) if label == "eye" else (3, -7)),
            textcoords="offset points", color=color, fontsize=5.8,
            ha=("right" if label == "eye" else "left"), va="center",
        )
    limit = max(abs(metric.eye_delta_x_deg), abs(metric.eye_delta_y_deg), 0.025) * 1.65
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="horizontal displacement (deg)", ylabel="vertical (deg)")
    ax.set_aspect("equal"); ax.set_title("eye and retinal shifts", fontsize=6.8, pad=4)
    ax.text(
        0.02, 0.02,
        f"calibrated feature shift\n({metric.calibrated_feature_shift_x_px:.2f}, "
        f"{metric.calibrated_feature_shift_y_px:.2f}) px",
        transform=ax.transAxes, fontsize=5.3, va="bottom", color="#60469C",
    )

    map_names = ("raw previous state", "current evidence", "recurrently transformed evidence")
    map_titles = {
        "raw previous state": "previous state",
        "current evidence": "current candidate\nevidence",
        "recurrently transformed evidence": "recurrent candidate\ncontribution",
    }
    values = np.concatenate([maps[name].ravel() for name in map_names])
    bound = float(np.quantile(np.abs(values), 0.995))
    for map_ax, name in zip(axes_a[1:4], map_names):
        map_ax.imshow(maps[name], cmap="RdBu_r", vmin=-bound, vmax=bound, origin="lower", interpolation="nearest", rasterized=True)
        map_ax.set_title(map_titles[name], fontsize=6.8, pad=3); map_ax.set_xticks([]); map_ax.set_yticks([])
    ax = axes_a[4]
    xpos = np.asarray([0.0, 1.0])
    for offset, prefix, color in ((-0.10, "raw", "#6F777D"), (0.10, "recurrent", "#60469C")):
        y = [metric[f"{prefix}_zero_lag_correlation"], metric[f"{prefix}_best_lag_correlation"]]
        ax.plot(xpos + offset, y, "o-", color=color, lw=1.1, ms=3.2)
        ax.text(0.96, y[-1], prefix, color=color, fontsize=6.0, va="center", ha="right")
    ax.set_xticks(xpos, ["zero lag", "best lag"], rotation=25, ha="right")
    ax.set_ylabel("normalized correlation")
    unresolved = bool(metric.raw_boundary_unresolved or metric.recurrent_boundary_unresolved or metric.expected_outside_search_window)
    ax.set_title("alignment decreases" + ("\n(boundary unresolved)" if unresolved else ""), fontsize=6.8, pad=3)
    ax.text(
        0.02, 0.02,
        f"best lag raw ({metric.raw_best_lag_x_px:.2f},{metric.raw_best_lag_y_px:.2f}) px\n"
        f"recurrent ({metric.recurrent_best_lag_x_px:.2f},{metric.recurrent_best_lag_y_px:.2f}) px",
        transform=ax.transAxes, fontsize=4.9, va="bottom", color="#4E5356",
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax_b = fig.add_subplot(outer[1, 0]); _panel_label(ax_b, "B", "Inferred transport does not track calibrated shift")
    finite = np.concatenate([binned.expected_shift_px.to_numpy(float), binned.mean_transport_px.to_numpy(float)])
    lo, hi = np.quantile(finite, [0.01, 0.99]); pad = 0.08 * max(hi - lo, 1e-3)
    ax_b.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#BFC3C5", lw=0.7, ls="--")
    component_style = {"x": "-", "y": ":"}
    for method_index, method in enumerate(METHOD_ORDER):
        color, marker = METHOD_STYLE[method]
        for component in ("x", "y"):
            part = binned.loc[(binned.method == method) & (binned.component == component)].sort_values("expected_shift_px")
            ax_b.plot(part.expected_shift_px, part.mean_transport_px, color=color, lw=1.1, ls=component_style[component], marker=marker, ms=2.1, markevery=3)
        stat = statistics.loc[statistics.method.eq(method)].iloc[0]
        ax_b.text(
            0.98, 0.96 - 0.085 * method_index,
            f"{method}: r={stat.vector_correlation:.2f}; "
            f"vector VE={stat.vector_variance_explained:.2f}",
            transform=ax_b.transAxes, color=color, fontsize=4.9,
            va="top", ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.4},
        )
    ax_b.text(
        0.03, 0.03, "solid x-component · dotted y-component",
        transform=ax_b.transAxes, va="bottom", fontsize=5.8, color="#555555",
    )
    ax_b.axhline(0, color="#D7DADD", lw=0.5, zorder=0)
    ax_b.set(xlabel="calibrated retinal feature shift (px)", ylabel="inferred recurrent transport (px)", xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad))
    ax_b.spines[["top", "right"]].set_visible(False)

    ax_c = fig.add_subplot(outer[1, 1]); _panel_label(ax_c, "C", "Recurrence reduces zero-lag alignment")
    for population in alignment.population.unique():
        part = alignment.loc[alignment.population.eq(population)].sort_values("scale")
        color, marker, style = GROUP_STYLE[population]
        ax_c.fill_between(part.scale, part.ci95_low, part.ci95_high, color=color, alpha=0.13, lw=0)
        ax_c.plot(part.scale, part["mean"], color=color, marker=marker, ls=style, lw=1.35, ms=3)
        ax_c.text(float(part.scale.iloc[-1])+0.06, float(part["mean"].iloc[-1]), population, color=color, fontsize=6.2, va="center")
    ax_c.axhline(0, color="#BFC3C5", lw=0.65)
    ax_c.axvline(1, color="#8C8C8C", lw=0.6, ls=":")
    ax_c.text(1.02, -0.015, "measured FEM", fontsize=5.8, color="#666666", va="top")
    ax_c.set(xlabel="trajectory amplitude (× measured FEM)", ylabel="recurrent − raw zero-lag correlation", xticks=SCALES)
    ax_c.spines[["top", "right"]].set_visible(False)

    ax_d = fig.add_subplot(outer[2, :]); _panel_label(ax_d, "D", "Residual-error association with SSI is descriptive only")
    for population in residual.population.unique():
        part = residual.loc[residual.population.eq(population)].sort_values("scale")
        color, marker, style = GROUP_STYLE[population]
        ax_d.plot(part.median_transport_error_px, part.ssi, color=color, marker=marker, ls=style, lw=1.25, ms=3.6)
        for item in part.itertuples():
            ax_d.text(item.median_transport_error_px, item.ssi, f" {item.scale:g}×", color=color, fontsize=5.6, va="bottom")
        last = part.iloc[-1]
        stat = residual_stats.loc[residual_stats.population.eq(population)].iloc[0]
        ax_d.text(
            float(last.median_transport_error_px), float(last.ssi),
            f"\n{population}; r={stat.pearson_r:.2f}",
            color=color, fontsize=6.2, va="top",
        )
    ax_d.set(xlabel="median |inferred transport − calibrated shift| (feature px; boundary-resolved only)", ylabel="final exact SSI (bits)")
    ax_d.text(
        0.02, 0.97,
        "five movement scales per population; transport itself did not track retinal displacement",
        transform=ax_d.transAxes, fontsize=5.8, color="#555555", va="top",
    )
    ax_d.spines[["top", "right"]].set_visible(False)
    if not shared_high:
        fig.text(0.995, 0.006, "High-SF spaces remain separate: shared-high gate did not pass.", ha="right", fontsize=5.8, color="#666666")
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_caption(
    path: Path,
    stats: pd.DataFrame,
    unresolved: pd.DataFrame,
    shared_high: bool,
    example_metrics: pd.DataFrame,
) -> None:
    p = stats.loc[stats.method.eq("learned P")].iloc[0]
    example = example_metrics.iloc[0]
    caption = f"""# Figure 2 caption draft

**Figure 2 | Recurrence does not register retained evidence to retinal displacement.** **A,** One outcome-blind held-out image–trajectory pair (chosen before inspecting registration outcomes as the pair nearest the joint medians of image high-spatial-frequency power and trajectory path length) shows eye displacement, the equal-and-opposite retinal-image displacement, and a common-component view of the fold-specific learned-P projection of previous state, current candidate evidence, and recurrent candidate contribution. All three maps use one component direction and one color scale. In this predeclared example, zero-lag correlation fell from {example.raw_zero_lag_correlation:.3f} for the previous state to {example.recurrent_zero_lag_correlation:.3f} for the recurrent contribution; best-lag correlation likewise fell from {example.raw_best_lag_correlation:.3f} to {example.recurrent_best_lag_correlation:.3f}. **B,** Fold-specific inferred transport is compared with the exact-renderer calibration of retinal displacement in feature pixels. Learned P, its native complementary Q, readout-SVD rank 8, and the mean of four deterministic rank-matched random controls are evaluated after excluding unresolved boundary peaks. None tracked retinal displacement: for learned P, vector correlation was {p.vector_correlation:.3f}, vector variance explained relative to the calibrated identity prediction was {p.vector_variance_explained:.3f}, and median vector error was {p.median_displacement_error_px:.3f} feature pixels. **C,** Recurrent-minus-previous-state zero-lag correlation was negative at every movement scale for both populations, with every 95% interval below zero. Thus recurrent transformation reduced rather than improved alignment under this preregistered estimator. **D,** Boundary-resolved transport error is shown against independently saved final exact SSI at the five movement scales for descriptive completeness. Because inferred transport did not track calibrated displacement, these five-point associations do not validate a registration mechanism. {"The two higher-SF contrast spaces passed the predeclared shared-high gate and are pooled only after that gate." if shared_high else "The predeclared shared-high gate did not pass, so higher-SF sharpening and reversal spaces are shown separately rather than being post hoc pooled."}

**Temporal direction and interpretation.** Every point is **within one independently scored 32-lag input window**: the cell traverses broad feature supports from relatively newer evidence toward relatively older evidence (backward in retinal time). It is **not recurrence across the 40 scored movie outputs**. Structural support midpoints supply nominal eye-displacement anchors, while the synthetic renderer establishes sign and feature-pixel scale. Best-lag peaks on the ±4-pixel boundary, or calibrated shifts outside that window, are labeled unresolved and excluded from transport regressions and panel D rather than treated as measured lags; retained and excluded counts accompany the plotting data. These results provide no evidence for the proposed ConvGRU spatial-registration operation under the preregistered estimator. They do not imply that recurrence has no other function.
"""
    path.write_text(caption, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    inputs = load_inputs(args.input_dir, args.exact_ssi)
    output = args.output_dir
    data_dir = output / "plot_data"
    output.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    methods = method_rows(inputs.registration)
    stats = transport_statistics(methods)
    binned = binned_transport(methods)
    shared = _high_shared(inputs.gate)
    alignment = alignment_by_scale(methods, shared)
    residual = residual_vs_ssi(methods, inputs.exact_ssi, shared)
    residual_stats = residual_associations(residual)
    meta = inputs.example["metadata"]
    row = inputs.registration.loc[
        inputs.registration.fold.eq(int(meta["fold"]))
        & inputs.registration.image_position.eq(int(meta["image_position"]))
        & inputs.registration.trajectory_position.eq(int(meta["trajectory_position"]))
        & inputs.registration.contrast.eq(str(meta["contrast"]))
        & np.isclose(inputs.registration.scale, float(meta["scale"]))
        & inputs.registration.frame_position.eq(int(meta["frame"]))
        & inputs.registration.internal_step.eq(int(meta["internal_step"]))
        & inputs.registration.subspace.eq("learned_p")
    ].iloc[0]
    maps, example_metrics = example_plot_data(inputs.example, row)
    unresolved = (
        methods.groupby("method", as_index=False)
        .agg(total_rows=("method", "size"), valid_rows=("valid", "sum"), unresolved_boundary_rows=("unresolved_boundary", "sum"))
    )
    stats.to_csv(data_dir / "panel_b_transport_statistics.csv", index=False)
    binned.to_csv(data_dir / "panel_b_transport_binned.csv", index=False)
    alignment.to_csv(data_dir / "panel_c_alignment_by_scale.csv", index=False)
    residual.to_csv(data_dir / "panel_d_residual_vs_ssi.csv", index=False)
    residual_stats.to_csv(data_dir / "panel_d_descriptive_statistics.csv", index=False)
    example_metrics.to_csv(data_dir / "panel_a_example_metrics.csv", index=False)
    unresolved.to_csv(data_dir / "boundary_resolution_counts.csv", index=False)
    np.savez_compressed(data_dir / "panel_a_example_maps.npz", **{key.replace(" ", "_"): value for key, value in maps.items()})
    figure = draw_figure(
        maps, example_metrics, binned, stats, alignment, residual,
        residual_stats, shared_high=shared,
    )
    stem = output / "figure2_registration"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    write_caption(
        output / "caption_draft.md", stats, unresolved, shared, example_metrics
    )
    manifest = {
        "analysis": "fig4-registration-figure2-saved-products-only-v1",
        "status": "complete",
        "foldwise_inference": True,
        "consensus_used_for_inference": False,
        "consensus_examples_visualization_only": True,
        "within_window_direction": "newer-support to older-support; backward in retinal time",
        "not_recurrence_across_40_outputs": True,
        "boundary_peaks_treated_as_unresolved": True,
        "result": (
            "inferred recurrent transport does not track calibrated retinal displacement, "
            "and recurrent transformation reduces zero-lag alignment"
        ),
        "shared_higher_sf_gate_allowed_pooling": shared,
        "source_files": [{"path": str(path), "sha256": _sha256(path)} for path in inputs.source_paths],
        "outputs": [str(stem.with_suffix(ext)) for ext in (".pdf", ".svg", ".png")],
        "plot_data": sorted(str(path) for path in data_dir.iterdir()),
        "definitions": {
            "panel_b_transport": "recurrent best-lag residual minus raw previous-state best lag",
            "panel_c_alignment_success": "recurrent minus raw zero-lag normalized correlation",
            "panel_d_error": "Euclidean inferred-transport minus calibrated-shift error; boundary-resolved rows only",
            "panel_d_ssi": "saved exact normal-moving final SSI in bits",
        },
    }
    temporary = output / f"manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except DataUnavailable as error:
        raise SystemExit(f"Figure 2 unavailable: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
