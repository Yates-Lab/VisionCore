#!/usr/bin/env python3
"""Test whether local BackImage structure predicts fixation-window FEM statistics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .image_features import local_backimage_features
from .io_utils import parse_csv_list, write_csv, write_json


DEFAULT_INPUT = Path("outputs") / "fixation_statistics_by_stimulus_all_sessions_after_review" / "window_features.csv"
DEFAULT_OUT_DIR = Path("outputs") / "fixation_statistics_by_stimulus_all_sessions_after_review" / "backimage_image_structure"
DEFAULT_PHASES = ("mid_fixation", "late_fixation")
IMAGE_FEATURES = (
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_high_freq_power_fraction",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_edge_density",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
    "image_power_slope_0p5_16_cpd",
    "image_amplitude_slope_0p5_16_cpd",
    "image_power_slope_deviation_from_1f",
    "image_amplitude_slope_deviation_from_1f",
    "image_abs_power_slope_deviation_from_1f",
    "image_abs_amplitude_slope_deviation_from_1f",
)
TARGETS = (
    "rms_radius_deg",
    "diffusion_constant_deg2_s",
    "speed_mean_deg_s",
    "anisotropy",
    "path_length_deg_s",
    "return_to_center_strength",
    "position_high_freq_power_fraction_15_60hz",
)
PATCH_AUDIT_CONTROLS = (
    "image_patch_fraction_inside_image",
    "image_patch_fraction_background",
    "image_patch_distance_to_image_border_px",
)
CONTROLS = ("epoch_duration_s", "samples_since_event", "abs_mean_radius_deg") + PATCH_AUDIT_CONTROLS


@dataclass(frozen=True)
class ImageStructureConfig:
    input_window_features: str
    out_dir: str
    phases: list[str]
    patch_radius_deg: float
    max_windows: int
    n_splits: int
    n_shuffles: int
    min_patch_fraction_inside_image: float
    max_patch_fraction_background: float
    seed: int


def drift_orientation_deg(df: pd.DataFrame) -> np.ndarray:
    return np.degrees(0.5 * np.arctan2(2.0 * df["cov_xy_deg2"].to_numpy(), (df["cov_xx_deg2"] - df["cov_yy_deg2"]).to_numpy()))


def circular_axis_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a_deg - b_deg))))


def cos2_delta(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return np.cos(2.0 * np.radians(a_deg - b_deg))


def _safe_zscore_by_session(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce").astype(float)
        means = values.groupby(out["session"]).transform("mean")
        stds = values.groupby(out["session"]).transform("std").replace(0.0, np.nan)
        out[col] = (values - means) / stds
    return out


def augment_backimage_windows(df: pd.DataFrame, *, patch_radius_deg: float, max_windows: int, seed: int) -> pd.DataFrame:
    df = df[df["stimulus"].astype(str) == "backimage"].copy()
    if max_windows > 0 and len(df) > max_windows:
        df = df.sample(n=int(max_windows), replace=False, random_state=int(seed)).sort_index().copy()
    rows: list[dict[str, Any]] = []
    for row in tqdm(df.to_dict("records"), desc="image patches"):
        gaze = np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])], dtype=np.float64)
        feats = local_backimage_features(
            session_name=str(row["session"]),
            trial_idx=int(row["trial_idx"]),
            gaze_xy_deg=gaze,
            patch_radius_deg=float(patch_radius_deg),
        )
        merged = dict(row)
        merged.update(feats)
        rows.append(merged)
    out = pd.DataFrame(rows)
    if "image_feature_ok" not in out.columns:
        out["image_feature_ok"] = True
    if "image_feature_error" not in out.columns:
        out["image_feature_error"] = ""
    out["drift_orientation_deg"] = drift_orientation_deg(out)
    out["drift_gradient_delta_deg"] = circular_axis_delta_deg(
        out["drift_orientation_deg"].to_numpy(),
        out["image_gradient_axis_deg"].to_numpy(),
    )
    out["drift_edge_delta_deg"] = circular_axis_delta_deg(
        out["drift_orientation_deg"].to_numpy(),
        out["image_edge_axis_deg"].to_numpy(),
    )
    out["drift_gradient_cos2"] = cos2_delta(
        out["drift_orientation_deg"].to_numpy(),
        out["image_gradient_axis_deg"].to_numpy(),
    )
    out["drift_edge_cos2"] = cos2_delta(
        out["drift_orientation_deg"].to_numpy(),
        out["image_edge_axis_deg"].to_numpy(),
    )
    return out


def _design(df: pd.DataFrame, predictors: list[str], controls: list[str]) -> tuple[np.ndarray, list[str]]:
    phase = pd.get_dummies(df["phase"].astype(str), prefix="phase", drop_first=True, dtype=float)
    xdf = pd.concat([df[predictors + controls].astype(float).reset_index(drop=True), phase.reset_index(drop=True)], axis=1)
    cols = list(xdf.columns)
    x = xdf.to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x, cols


def _cv_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, n_splits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[finite]
    y = y[finite]
    groups = groups[finite]
    if x.shape[0] < 20 or np.unique(groups).size < 2 or np.nanstd(y) <= 0:
        return y, np.full(y.shape, np.nan, dtype=np.float64), groups
    n_splits = min(int(n_splits), int(np.unique(groups).size))
    preds = np.full(y.shape, np.nan, dtype=np.float64)
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        model = Ridge(alpha=1.0)
        model.fit(x_train, y[train])
        preds[test] = model.predict(x_test)
    return y, preds, groups


def _pooled_r2(y: np.ndarray, preds: np.ndarray) -> float:
    finite_pred = np.isfinite(preds)
    return float(r2_score(y[finite_pred], preds[finite_pred])) if np.count_nonzero(finite_pred) > 2 else float("nan")


def _session_r2_delta_summary(y: np.ndarray, full_preds: np.ndarray, ctrl_preds: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    deltas: list[float] = []
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        ok = np.isfinite(y[idx]) & np.isfinite(full_preds[idx]) & np.isfinite(ctrl_preds[idx])
        if np.count_nonzero(ok) <= 2 or np.nanstd(y[idx][ok]) <= 0:
            continue
        full = float(r2_score(y[idx][ok], full_preds[idx][ok]))
        ctrl = float(r2_score(y[idx][ok], ctrl_preds[idx][ok]))
        if np.isfinite(full) and np.isfinite(ctrl):
            deltas.append(full - ctrl)
    arr = np.asarray(deltas, dtype=np.float64)
    n_pos = int(np.count_nonzero(arr > 0))
    sign_p = float(stats.binomtest(n_pos, n=arr.size, p=0.5).pvalue) if arr.size else np.nan
    return {
        "session_delta_r2_mean": float(np.mean(arr)) if arr.size else np.nan,
        "session_delta_r2_median": float(np.median(arr)) if arr.size else np.nan,
        "session_delta_r2_sem": float(stats.sem(arr)) if arr.size > 1 else np.nan,
        "session_delta_r2_n_positive": n_pos,
        "session_delta_r2_n": int(arr.size),
        "session_delta_r2_sign_p_two_sided": sign_p,
    }


def regression_summaries(df: pd.DataFrame, *, n_splits: int, n_shuffles: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    image_cols = list(range(len(IMAGE_FEATURES)))
    for target in TARGETS:
        cols = list(IMAGE_FEATURES + (target,) + CONTROLS)
        work = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
        work = _safe_zscore_by_session(work, cols)
        groups = work["session"].to_numpy()
        phase_groups = work["session"].astype(str).to_numpy() + "||" + work["phase"].astype(str).to_numpy()
        group_indices = {group: np.where(phase_groups == group)[0] for group in sorted(np.unique(phase_groups))}
        x_full, full_cols = _design(work, list(IMAGE_FEATURES), list(CONTROLS))
        x_ctrl, ctrl_cols = _design(work, [], list(CONTROLS))
        y = work[target].to_numpy(dtype=np.float64)
        y_full, full_preds, full_groups = _cv_predictions(x_full, y, groups, n_splits=n_splits)
        y_ctrl, ctrl_preds, ctrl_groups = _cv_predictions(x_ctrl, y, groups, n_splits=n_splits)
        full_r2 = _pooled_r2(y_full, full_preds)
        ctrl_r2 = _pooled_r2(y_ctrl, ctrl_preds)
        delta = full_r2 - ctrl_r2 if np.isfinite(full_r2) and np.isfinite(ctrl_r2) else np.nan
        session_summary = _session_r2_delta_summary(y_full, full_preds, ctrl_preds, full_groups)
        shuffle_delta: list[float] = []
        for _ in range(int(n_shuffles)):
            x_shuff = x_full.copy()
            for idx in group_indices.values():
                perm = rng.permutation(idx)
                x_shuff[np.ix_(idx, image_cols)] = x_full[np.ix_(perm, image_cols)]
            _, shuff_preds, _ = _cv_predictions(x_shuff, y, groups, n_splits=n_splits)
            sr2 = _pooled_r2(y_full, shuff_preds)
            if np.isfinite(sr2) and np.isfinite(ctrl_r2):
                shuffle_delta.append(sr2 - ctrl_r2)
        sh = np.asarray(shuffle_delta, dtype=np.float64)
        p = float((1.0 + np.count_nonzero(sh >= delta)) / (1.0 + sh.size)) if sh.size and np.isfinite(delta) else np.nan
        rows.append({
            "target": target,
            "n_windows": int(work.shape[0]),
            "n_sessions": int(work["session"].nunique()),
            "cv_r2_full": full_r2,
            "cv_r2_controls": ctrl_r2,
            "cv_delta_r2_image": delta,
            **session_summary,
            "shuffle_delta_r2_mean": float(np.mean(sh)) if sh.size else np.nan,
            "shuffle_delta_r2_p95": float(np.quantile(sh, 0.95)) if sh.size else np.nan,
            "shuffle_p_ge_observed": p,
            "image_predictors": ",".join(IMAGE_FEATURES),
            "control_predictors": ",".join(ctrl_cols),
            "full_predictors": ",".join(full_cols),
        })
    return rows


def orientation_alignment_summary(df: pd.DataFrame, *, n_shuffles: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    work = df.dropna(subset=["drift_orientation_deg", "image_gradient_axis_deg", "image_edge_axis_deg"]).copy()
    rows: list[dict[str, Any]] = []
    for image_col, label in [
        ("image_gradient_axis_deg", "gradient_axis"),
        ("image_edge_axis_deg", "edge_axis"),
        ("image_spectrum_orientation_deg", "spectrum_axis"),
    ]:
        if image_col not in work.columns:
            continue
        base = work.dropna(subset=[image_col]).copy()
        subsets = [("all_windows", base)]
        reliable = base[
            (base["image_orientation_coherence"].astype(float) >= 0.20)
            & (base["anisotropy"].astype(float) >= 0.20)
        ].copy()
        subsets.append(("reliable_axes_coh_ge_0p20_aniso_ge_0p20", reliable))
        for subset_label, sub in subsets:
            if sub.empty:
                continue
            drift = sub["drift_orientation_deg"].to_numpy(dtype=np.float64)
            image = sub[image_col].to_numpy(dtype=np.float64)
            vec = np.exp(2j * np.radians(drift - image))
            mean_cos = float(np.mean(vec.real))
            mean_sin = float(np.mean(vec.imag))
            resultant = float(np.abs(np.mean(vec)))
            weights = (sub["image_orientation_coherence"].astype(float) * sub["anisotropy"].astype(float)).to_numpy(dtype=np.float64).copy()
            weights[~np.isfinite(weights) | (weights < 0)] = 0.0
            weighted_cos = float(np.average(vec.real, weights=weights)) if np.sum(weights) > 0 else np.nan
            sh_cos: list[float] = []
            sh_r: list[float] = []
            phase_groups = sub["session"].astype(str).to_numpy() + "||" + sub["phase"].astype(str).to_numpy()
            group_indices = {group: np.where(phase_groups == group)[0] for group in sorted(np.unique(phase_groups))}
            for _ in range(int(n_shuffles)):
                shuffled = image.copy()
                for idx in group_indices.values():
                    if idx.size > 1:
                        shuffled[idx] = shuffled[rng.permutation(idx)]
                sh_vec = np.exp(2j * np.radians(drift - shuffled))
                sh_cos.append(float(np.mean(sh_vec.real)))
                sh_r.append(float(np.abs(np.mean(sh_vec))))
            sh_cos_arr = np.asarray(sh_cos, dtype=np.float64)
            sh_r_arr = np.asarray(sh_r, dtype=np.float64)
            rows.append({
                "alignment_reference": label,
                "analysis_subset": subset_label,
                "n_windows": int(sub.shape[0]),
                "n_sessions": int(sub["session"].nunique()),
                "mean_cos2_delta": mean_cos,
                "weighted_mean_cos2_delta": weighted_cos,
                "mean_sin2_delta": mean_sin,
                "resultant_length": resultant,
                "mean_axis_delta_deg": float(0.5 * np.degrees(np.angle(np.mean(vec)))),
                "shuffle_mean_cos2": float(np.mean(sh_cos_arr)) if sh_cos_arr.size else np.nan,
                "shuffle_p_abs_cos_ge_observed": float((1 + np.count_nonzero(np.abs(sh_cos_arr) >= abs(mean_cos))) / (1 + sh_cos_arr.size)) if sh_cos_arr.size else np.nan,
                "shuffle_mean_resultant": float(np.mean(sh_r_arr)) if sh_r_arr.size else np.nan,
                "shuffle_p_resultant_ge_observed": float((1 + np.count_nonzero(sh_r_arr >= resultant)) / (1 + sh_r_arr.size)) if sh_r_arr.size else np.nan,
            })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-window-features", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--patch-radius-deg", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--min-patch-fraction-inside-image", type=float, default=0.98)
    parser.add_argument("--max-patch-fraction-background", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = parse_csv_list(args.phases)
    df = pd.read_csv(args.input_window_features)
    df = df[(df["stimulus"].astype(str) == "backimage") & (df["phase"].astype(str).isin(phases))].copy()
    augmented = augment_backimage_windows(
        df,
        patch_radius_deg=float(args.patch_radius_deg),
        max_windows=int(args.max_windows),
        seed=int(args.seed),
    )
    raw_augmented = augmented.copy()
    ok = augmented["image_feature_ok"].fillna(False).astype(bool)
    inside = augmented.get("image_patch_fraction_inside_image", pd.Series(np.nan, index=augmented.index)).astype(float)
    background = augmented.get("image_patch_fraction_background", pd.Series(np.nan, index=augmented.index)).astype(float)
    contamination_keep = (inside >= float(args.min_patch_fraction_inside_image)) & (background <= float(args.max_patch_fraction_background))
    augmented = augmented[ok & contamination_keep].copy()
    augmented = augmented.dropna(subset=list(IMAGE_FEATURES) + ["drift_orientation_deg"]).copy()

    regression_rows = regression_summaries(
        augmented,
        n_splits=int(args.n_splits),
        n_shuffles=int(args.n_shuffles),
        seed=int(args.seed),
    )
    alignment_rows = orientation_alignment_summary(
        augmented,
        n_shuffles=int(args.n_shuffles),
        seed=int(args.seed) + 17,
    )

    raw_augmented.to_csv(out_dir / "backimage_image_fem_windows_raw.csv", index=False)
    augmented.to_csv(out_dir / "backimage_image_fem_windows.csv", index=False)
    failure_summary = (
        raw_augmented.assign(
            image_feature_error=raw_augmented["image_feature_error"].fillna(""),
            image_feature_ok=raw_augmented["image_feature_ok"].fillna(False).astype(bool),
        )
        .groupby(["session", "trial_idx", "image_feature_ok", "image_feature_error"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
    )
    failure_summary.to_csv(out_dir / "image_feature_extraction_summary.csv", index=False)
    write_csv(out_dir / "image_feature_regression_summary.csv", regression_rows)
    write_csv(out_dir / "orientation_alignment_summary.csv", alignment_rows)
    cfg = ImageStructureConfig(
        input_window_features=str(args.input_window_features),
        out_dir=str(out_dir),
        phases=phases,
        patch_radius_deg=float(args.patch_radius_deg),
        max_windows=int(args.max_windows),
        n_splits=int(args.n_splits),
        n_shuffles=int(args.n_shuffles),
        min_patch_fraction_inside_image=float(args.min_patch_fraction_inside_image),
        max_patch_fraction_background=float(args.max_patch_fraction_background),
        seed=int(args.seed),
    )
    write_json(out_dir / "run_metadata.json", {
        "config": asdict(cfg),
        "n_raw_augmented_windows": int(raw_augmented.shape[0]),
        "n_windows": int(augmented.shape[0]),
        "n_failed_image_feature_windows": int((~raw_augmented["image_feature_ok"].fillna(False).astype(bool)).sum()),
        "n_excluded_patch_contamination_windows": int((ok & ~contamination_keep).sum()),
        "image_features": list(IMAGE_FEATURES),
        "targets": list(TARGETS),
        "controls": list(CONTROLS),
        "orientation_note": "image_gradient_axis is the cross-edge luminance-gradient axis; image_edge_axis is orthogonal/parallel-to-edge.",
        "orientation_coordinate_note": "Image-derived axes are reported in gaze coordinates: +x right, +y up. Array-coordinate audit fields retain +row-down angles.",
        "cv_note": "Predictors and targets are z-scored within session before grouped CV; interpret pooled R2 as within-session normalized association, not deployable cross-session prediction.",
        "shuffle_note": "Image predictors/orientations are shuffled within session x phase.",
    })
    print(f"Wrote image-structure analysis for {augmented.shape[0]} windows to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
