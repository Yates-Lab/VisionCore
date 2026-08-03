"""BackImage drift-axis optimization analysis with image-proxy or V1-twin scores.

This is the first-pass discrete-grid analysis from the natural-image drift
geometry plan.  It scores candidate motion axes for each reliable fixation
window, predicts the best/worst axis under pose-aware, pose-blind, and Pareto
objectives, and compares those axes with the observed drift/fixation-cloud axis.

The default ``conditional_proxy`` scorer is a fast first-order local-stability
path for the conditional fixation objective.  ``image_proxy`` scores shifted
retinal patch modulation.  Use ``--score-mode twin`` for the V1 digital-twin
forward pass; that mode reuses Ryan's figure-4 digital-twin helpers.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift as scipy_shift
from tqdm import tqdm

from .backimage_canvas import _backimage_canvas, _clip_patch
from .image_features import gaze_deg_to_screen_px


DEFAULT_INPUT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered/backimage_image_fem_windows.csv")
DEFAULT_OUT_DIR = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_twin_drift_geometry")
OBJECTIVES = ("pose_aware", "pose_blind", "pareto")


@dataclass(frozen=True)
class RunConfig:
    input: str
    out_dir: str
    score_mode: str
    max_windows: int
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    min_duration_s: float
    patch_size_px: int
    min_patch_image_margin_px: float
    n_timepoints: int
    axes_deg: list[float]
    scales_deg: list[float]
    anisotropy_ratios: list[float]
    lambdas: list[float]
    motor_gamma: float
    n_axis_nulls: int
    n_shuffle_nulls: int
    n_session_bootstrap: int
    twin_population_n: int
    twin_batch_size: int
    device: str
    seed: int


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def _cos2(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return np.cos(2.0 * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))


def _standardize_uint_like(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return image.astype(np.float32)


def _candidate_trace(axis_deg: float, scale_deg: float, ratio: float, n_timepoints: int) -> np.ndarray:
    theta = np.radians(float(axis_deg))
    major = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    minor = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, int(n_timepoints), endpoint=False)
    if float(ratio) <= 1.0:
        trace = float(scale_deg) * np.sin(t)[:, None] * major[None, :]
    else:
        trace = (
            float(scale_deg) * np.sin(t)[:, None] * major[None, :]
            + float(scale_deg) / float(ratio) * np.cos(t)[:, None] * minor[None, :]
        )
    trace -= trace.mean(axis=0, keepdims=True)
    return trace.astype(np.float32)


def _motor_cost(trace: np.ndarray) -> float:
    if trace.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))


def _path_extent_along_axis(trace: np.ndarray, axis_deg: float) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] == 0:
        return float("nan")
    theta = np.radians(float(axis_deg))
    u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    projected = trace @ u
    return float(np.nanmax(projected) - np.nanmin(projected))


def _pixel_isophote_cost(patch: np.ndarray, axis_deg: float, scale_deg: float, ppd: float, sample_size_px: int = 151) -> float:
    """Squared first-order pixel change for motion along a candidate axis.

    Low values correspond to isophote/edge-parallel directions; high values
    correspond to crossing local image gradients.  The scale term keeps this
    comparable to candidate-motion amplitude rather than being axis-only.
    """
    image = _standardize_uint_like(patch)
    half = int(sample_size_px) // 2
    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    crop = image[center_y - half : center_y + half + 1, center_x - half : center_x + half + 1]
    if crop.size == 0:
        return float("nan")
    gy, gx = np.gradient(crop.astype(np.float64))
    theta = np.radians(float(axis_deg))
    directional_gradient = gx * np.cos(theta) - gy * np.sin(theta)
    displacement_px = float(scale_deg) * float(ppd)
    return float(np.nanmean(directional_gradient**2) * displacement_px**2)


def _image_proxy_response(patch: np.ndarray, trace: np.ndarray, ppd: float, sample_size_px: int = 151) -> np.ndarray:
    """Return flattened retinal crops under the candidate trace.

    Positive gaze x means the retinal image shifts opposite in pixel space.
    For axis selection, the sign convention is less important than consistency
    across candidates; we follow the counterfactual-stimulus intuition here.
    """
    patch = _standardize_uint_like(patch)
    half = int(sample_size_px) // 2
    center_y = patch.shape[0] // 2
    center_x = patch.shape[1] // 2
    rows = []
    for xy in trace:
        shifted = scipy_shift(
            patch,
            shift=(-float(xy[1]) * float(ppd), -float(xy[0]) * float(ppd)),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        crop = shifted[center_y - half : center_y + half + 1, center_x - half : center_x + half + 1]
        rows.append(crop.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def _trace_xy_to_twin_helper_order(trace_xy: np.ndarray) -> np.ndarray:
    """Pre-flip [x, y] traces because Ryan's helper flips columns internally."""
    trace = np.asarray(trace_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {trace.shape}")
    return trace[:, [1, 0]].copy()


def _score_response_series(response: np.ndarray) -> tuple[float, float]:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 2 or response.shape[0] < 2:
        return float("nan"), float("nan")
    centered = response - np.mean(response, axis=0, keepdims=True)
    pose_variance = float(np.sum(centered * centered))
    response_path = float(np.sum(np.linalg.norm(np.diff(response, axis=0), axis=1) ** 2))
    return response_path, pose_variance


def _load_twin_common():
    root = Path(__file__).resolve().parents[2]
    path = root / "ryan" / "digital-twin-fem" / "_common.py"
    spec = importlib.util.spec_from_file_location("digital_twin_fem_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import twin common helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


class TwinScorer:
    def __init__(self, *, device: str, population_n: int, batch_size: int, seed: int):
        import torch

        common = _load_twin_common()
        self.common = common
        dev = None if str(device) == "auto" else str(device)
        self.model, _info, self.device = common.load_digital_twin(device=dev)
        self.population = common.build_population(
            self.model,
            N=int(population_n),
            rng=np.random.default_rng(int(seed)),
            ccmax_threshold=0.80,
        )
        self.readout = self.population.readout.to(self.device)
        self.batch_size = int(batch_size)
        self.torch = torch

    def response(self, patch: np.ndarray, trace: np.ndarray) -> np.ndarray:
        common = self.common
        image = _standardize_uint_like(patch)
        full_stack = np.broadcast_to(
            image[None, :, :],
            (trace.shape[0] + common.N_LAGS + 1, *image.shape),
        ).copy()
        eye = self.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
        stim = common.make_counterfactual_stim(
            full_stack,
            eye,
            ppd=common.PPD,
            scale_factor=1.0,
            n_lags=common.N_LAGS,
            out_size=common.OUT_SIZE,
        )
        stim = (stim - 127.0) / 255.0
        rate_map = common.compute_rate_map_batched(self.model, self.readout, stim, batch_size=self.batch_size)
        unit_ids = self.population.unit_ids
        return rate_map[:, unit_ids[:, 0], unit_ids[:, 1], unit_ids[:, 2]].detach().cpu().numpy()


def _candidate_grid(axes: list[float], scales: list[float], ratios: list[float], n_timepoints: int) -> list[dict[str, Any]]:
    out = []
    for axis in axes:
        for scale in scales:
            for ratio in ratios:
                trace = _candidate_trace(axis, scale, ratio, n_timepoints)
                out.append(
                    {
                        "candidate_axis_deg": float(axis),
                        "candidate_scale_deg": float(scale),
                        "candidate_anisotropy_ratio": float(ratio),
                        "trace": trace,
                        "motor_cost": _motor_cost(trace),
                    }
                )
    return out


def _normalize_scores(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        finite = vals[np.isfinite(vals)]
        scale = max(1.0, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
        if not np.isfinite(sd) or sd <= 1e-6 * scale:
            for row, val in zip(rows, vals, strict=True):
                row[f"{key}_z"] = 0.0 if np.isfinite(val) else float("nan")
            continue
        for row, val in zip(rows, vals, strict=True):
            row[f"{key}_z"] = (float(val) - mean) / sd


def _choose_predictions(rows: list[dict[str, Any]], lambdas: list[float], motor_gamma: float) -> list[dict[str, Any]]:
    _normalize_scores(rows, ("pose_aware_score", "pose_blind_cost", "pixel_instability_cost", "refresh_benefit", "motor_cost"))
    out = []
    by_best = {
        "optimized_PA": max(rows, key=lambda r: float(r["pose_aware_score_z"])),
        "adversarial_PA": min(rows, key=lambda r: float(r["pose_aware_score_z"])),
        "optimized_PB": min(rows, key=lambda r: float(r["pose_blind_cost_z"])),
        "adversarial_PB": max(rows, key=lambda r: float(r["pose_blind_cost_z"])),
        "optimized_pixel_isophote": min(rows, key=lambda r: float(r["pixel_instability_cost_z"])),
        "adversarial_pixel_isophote": max(rows, key=lambda r: float(r["pixel_instability_cost_z"])),
        "optimized_response_stability": min(rows, key=lambda r: float(r["pose_blind_cost_z"])),
        "adversarial_response_stability": max(rows, key=lambda r: float(r["pose_blind_cost_z"])),
        "optimized_refresh_only": max(rows, key=lambda r: float(r["refresh_benefit_z"] - float(motor_gamma) * float(r["motor_cost_z"]))),
    }
    for label, row in by_best.items():
        out.append({"objective": label, **{k: row[k] for k in row if k != "trace"}})
    for lam in lambdas:
        for row in rows:
            row[f"pareto_lambda_{lam:g}_score"] = (
                (1.0 - float(lam)) * float(row["pose_aware_score_z"])
                - float(lam) * float(row["pose_blind_cost_z"])
                - float(motor_gamma) * float(row["motor_cost_z"])
            )
            row[f"pixel_refresh_lambda_{lam:g}_score"] = (
                (1.0 - float(lam)) * float(row["refresh_benefit_z"])
                - float(lam) * float(row["pixel_instability_cost_z"])
                - float(motor_gamma) * float(row["motor_cost_z"])
            )
            row[f"response_refresh_lambda_{lam:g}_score"] = (
                (1.0 - float(lam)) * float(row["refresh_benefit_z"])
                - float(lam) * float(row["pose_blind_cost_z"])
                - float(motor_gamma) * float(row["motor_cost_z"])
            )
        score_key = f"pareto_lambda_{lam:g}_score"
        best = max(rows, key=lambda r: float(r[score_key]))
        worst = min(rows, key=lambda r: float(r[score_key]))
        out.append({"objective": f"optimized_Pareto_lambda_{lam:g}", **{k: best[k] for k in best if k != "trace"}})
        out.append({"objective": f"adversarial_Pareto_lambda_{lam:g}", **{k: worst[k] for k in worst if k != "trace"}})
        for prefix in ("pixel_refresh", "response_refresh"):
            score_key = f"{prefix}_lambda_{lam:g}_score"
            best = max(rows, key=lambda r: float(r[score_key]))
            worst = min(rows, key=lambda r: float(r[score_key]))
            out.append({"objective": f"optimized_{prefix}_lambda_{lam:g}", **{k: best[k] for k in best if k != "trace"}})
            out.append({"objective": f"adversarial_{prefix}_lambda_{lam:g}", **{k: worst[k] for k in worst if k != "trace"}})
    return out


def _session_mean_stat(df: pd.DataFrame, value_col: str = "cos2_alignment") -> float:
    if df.empty:
        return float("nan")
    return float(df.groupby("session")[value_col].mean().mean())


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    means = np.mean(draws, axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _session_summary(alignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(alignment_rows)
    if df.empty:
        return []
    out = []
    for objective, block in df.groupby("objective"):
        session_means = block.groupby("session")["cos2_alignment"].mean()
        weighted = block.groupby("session").apply(
            lambda g: np.average(g["cos2_alignment"], weights=np.maximum(g["alignment_weight"], 1e-12))
        )
        out.append(
            {
                "objective": objective,
                "n_windows": int(block.shape[0]),
                "n_sessions": int(session_means.shape[0]),
                "mean_cos2_window": float(block["cos2_alignment"].mean()),
                "mean_cos2_session_mean": float(session_means.mean()),
                "median_cos2_session_mean": float(session_means.median()),
                "weighted_cos2_session_mean": float(weighted.mean()),
                "n_sessions_positive": int(np.count_nonzero(session_means.to_numpy() > 0)),
            }
        )
    return out


def _paired_delta_summary(alignment_rows: list[dict[str, Any]], *, baseline: str, rng: np.random.Generator, n_bootstrap: int) -> list[dict[str, Any]]:
    df = pd.DataFrame(alignment_rows)
    if df.empty or baseline not in set(df["objective"]):
        return []
    session_obj = df.groupby(["session", "objective"])["cos2_alignment"].mean().unstack()
    if baseline not in session_obj.columns:
        return []
    out = []
    for objective in sorted(c for c in session_obj.columns if c != baseline):
        delta = (session_obj[objective] - session_obj[baseline]).dropna().to_numpy(dtype=np.float64)
        if delta.size == 0:
            continue
        ci_lo, ci_hi = _bootstrap_ci(delta, rng, n_bootstrap)
        out.append(
            {
                "objective": objective,
                "baseline_objective": baseline,
                "n_sessions": int(delta.size),
                "mean_delta_cos2_session": float(np.mean(delta)),
                "median_delta_cos2_session": float(np.median(delta)),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "n_sessions_delta_positive": int(np.count_nonzero(delta > 0)),
            }
        )
    return out


def _axis_null_summary(
    alignment_rows: list[dict[str, Any]],
    *,
    axes_deg: list[float],
    rng: np.random.Generator,
    n_nulls: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(alignment_rows)
    if df.empty or int(n_nulls) <= 0:
        return []
    axes = np.asarray(axes_deg, dtype=np.float64)
    if axes.size == 0:
        return []
    out = []
    for objective, block in df.groupby("objective"):
        observed = _session_mean_stat(block)
        null_stats = np.empty(int(n_nulls), dtype=np.float64)
        real = block["real_drift_axis_deg"].to_numpy(dtype=np.float64)
        sessions = block["session"].to_numpy()
        for j in range(int(n_nulls)):
            random_axis = rng.choice(axes, size=real.size, replace=True)
            tmp = pd.DataFrame({
                "session": sessions,
                "cos2_alignment": _cos2(real, random_axis),
            })
            null_stats[j] = _session_mean_stat(tmp)
        p_greater = (1.0 + float(np.count_nonzero(null_stats >= observed))) / (float(n_nulls) + 1.0)
        p_less = (1.0 + float(np.count_nonzero(null_stats <= observed))) / (float(n_nulls) + 1.0)
        out.append(
            {
                "objective": objective,
                "null_type": "random_axis_candidate_grid",
                "n_nulls": int(n_nulls),
                "observed_session_mean_cos2": observed,
                "null_mean": float(np.mean(null_stats)),
                "null_ci95_low": float(np.quantile(null_stats, 0.025)),
                "null_ci95_high": float(np.quantile(null_stats, 0.975)),
                "p_greater_equal": p_greater,
                "p_less_equal": p_less,
            }
        )
    return out


def _predicted_axis_shuffle_null_summary(
    alignment_rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_nulls: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(alignment_rows)
    if df.empty or int(n_nulls) <= 0:
        return []
    out = []
    for objective, block in df.groupby("objective"):
        observed = _session_mean_stat(block)
        real = block["real_drift_axis_deg"].to_numpy(dtype=np.float64)
        predicted = block["predicted_axis_deg"].to_numpy(dtype=np.float64)
        sessions = block["session"].to_numpy()
        for null_type in ("within_session_predicted_axis_shuffle", "across_session_predicted_axis_shuffle"):
            null_stats = np.empty(int(n_nulls), dtype=np.float64)
            for j in range(int(n_nulls)):
                shuffled = predicted.copy()
                if null_type == "within_session_predicted_axis_shuffle":
                    for session in np.unique(sessions):
                        idx = np.flatnonzero(sessions == session)
                        if idx.size > 1:
                            shuffled[idx] = rng.permutation(shuffled[idx])
                else:
                    shuffled = rng.permutation(shuffled)
                tmp = pd.DataFrame({
                    "session": sessions,
                    "cos2_alignment": _cos2(real, shuffled),
                })
                null_stats[j] = _session_mean_stat(tmp)
            p_greater = (1.0 + float(np.count_nonzero(null_stats >= observed))) / (float(n_nulls) + 1.0)
            p_less = (1.0 + float(np.count_nonzero(null_stats <= observed))) / (float(n_nulls) + 1.0)
            out.append(
                {
                    "objective": objective,
                    "null_type": null_type,
                    "n_nulls": int(n_nulls),
                    "observed_session_mean_cos2": observed,
                    "null_mean": float(np.mean(null_stats)),
                    "null_ci95_low": float(np.quantile(null_stats, 0.025)),
                    "null_ci95_high": float(np.quantile(null_stats, 0.975)),
                    "p_greater_equal": p_greater,
                    "p_less_equal": p_less,
                }
            )
    return out


def _stratified_summary(alignment_rows: list[dict[str, Any]], *, n_bins: int = 2) -> list[dict[str, Any]]:
    df = pd.DataFrame(alignment_rows)
    if df.empty:
        return []
    out = []
    for col in ("image_orientation_coherence", "drift_anisotropy", "alignment_weight"):
        if col not in df.columns:
            continue
        valid = df[np.isfinite(df[col].astype(float))].copy()
        if valid.empty:
            continue
        try:
            valid["stratum"] = pd.qcut(valid[col].astype(float), q=int(n_bins), labels=False, duplicates="drop")
        except ValueError:
            continue
        valid = valid.dropna(subset=["stratum"]).copy()
        if valid.empty:
            continue
        for (objective, stratum), block in valid.groupby(["objective", "stratum"]):
            out.append(
                {
                    "objective": objective,
                    "stratify_by": col,
                    "stratum": int(stratum),
                    "min_value": float(block[col].min()),
                    "max_value": float(block[col].max()),
                    "n_windows": int(block.shape[0]),
                    "n_sessions": int(block["session"].nunique()),
                    "mean_cos2_window": float(block["cos2_alignment"].mean()),
                    "mean_cos2_session_mean": _session_mean_stat(block),
                    "n_sessions_positive": int(np.count_nonzero(block.groupby("session")["cos2_alignment"].mean().to_numpy() > 0)),
                }
            )
    return out


def _raw_axis_rows(row: pd.Series) -> list[dict[str, Any]]:
    return [
        {"objective": "raw_gradient_axis", "candidate_axis_deg": float(row["image_gradient_axis_deg"])},
        {"objective": "raw_edge_axis", "candidate_axis_deg": float(row["image_edge_axis_deg"])},
        {"objective": "raw_spectrum_axis", "candidate_axis_deg": float(row["image_spectrum_orientation_deg"])},
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-mode", choices=["conditional_proxy", "image_proxy", "twin"], default="conditional_proxy")
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument(
        "--min-patch-image-margin-px",
        type=float,
        default=None,
        help="Require the fixation center to be this many pixels from the BackImage border; default is patch_size_px / 2.",
    )
    parser.add_argument("--n-timepoints", type=int, default=48)
    parser.add_argument("--axes-deg", default="0,15,30,45,60,75,90,105,120,135,150,165")
    parser.add_argument("--scales-deg", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--anisotropy-ratios", default="1,2,4")
    parser.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--motor-gamma", type=float, default=0.05)
    parser.add_argument("--n-axis-nulls", type=int, default=1000)
    parser.add_argument("--n-shuffle-nulls", type=int, default=1000)
    parser.add_argument("--n-session-bootstrap", type=int, default=1000)
    parser.add_argument("--twin-population-n", type=int, default=256)
    parser.add_argument("--twin-batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = _parse_float_list(args.axes_deg)
    scales = _parse_float_list(args.scales_deg)
    ratios = _parse_float_list(args.anisotropy_ratios)
    lambdas = _parse_float_list(args.lambdas)
    min_patch_image_margin_px = (
        float(args.min_patch_image_margin_px)
        if args.min_patch_image_margin_px is not None
        else float(args.patch_size_px) / 2.0
    )
    cfg = RunConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        score_mode=str(args.score_mode),
        max_windows=int(args.max_windows),
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
        patch_size_px=int(args.patch_size_px),
        min_patch_image_margin_px=min_patch_image_margin_px,
        n_timepoints=int(args.n_timepoints),
        axes_deg=axes,
        scales_deg=scales,
        anisotropy_ratios=ratios,
        lambdas=lambdas,
        motor_gamma=float(args.motor_gamma),
        n_axis_nulls=int(args.n_axis_nulls),
        n_shuffle_nulls=int(args.n_shuffle_nulls),
        n_session_bootstrap=int(args.n_session_bootstrap),
        twin_population_n=int(args.twin_population_n),
        twin_batch_size=int(args.twin_batch_size),
        device=str(args.device),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed))
    df = pd.read_csv(args.input)
    required = [
        "session",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "drift_orientation_deg",
        "anisotropy",
        "image_orientation_coherence",
        "image_gradient_axis_deg",
        "image_edge_axis_deg",
        "image_spectrum_orientation_deg",
        "image_patch_distance_to_image_border_px",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "duration_s" not in df.columns:
        df["duration_s"] = df.get("epoch_duration_s", np.nan)
    keep = (
        np.isfinite(df["drift_orientation_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(args.reliable_drift_anisotropy_min))
        & (df["image_orientation_coherence"].astype(float) >= float(args.reliable_image_coherence_min))
        & (df["duration_s"].astype(float) >= float(args.min_duration_s))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= min_patch_image_margin_px)
    )
    work = df.loc[keep].copy()
    work["window_id"] = np.arange(work.shape[0], dtype=int)
    if int(args.max_windows) > 0 and work.shape[0] > int(args.max_windows):
        work = work.sample(n=int(args.max_windows), replace=False, random_state=int(args.seed)).sort_values(["session", "trial_idx", "window_id"])
    work = work.reset_index(drop=True)

    scorer = None
    if args.score_mode == "twin":
        if int(args.n_timepoints) < 32:
            raise ValueError("Twin mode requires --n-timepoints >= 32 to populate the model's lag history.")
        scorer = TwinScorer(
            device=str(args.device),
            population_n=int(args.twin_population_n),
            batch_size=int(args.twin_batch_size),
            seed=int(args.seed),
        )

    candidate_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    example_payloads: list[tuple[np.ndarray, list[dict[str, Any]], pd.Series]] = []
    candidate_templates = _candidate_grid(axes, scales, ratios, int(args.n_timepoints))
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    for i, row in enumerate(tqdm(list(work.iterrows()), desc=f"{args.score_mode} drift geometry")):
        _, row = row
        try:
            canvas_key = (str(row["session"]), int(row["trial_idx"]))
            if canvas_key not in canvas_cache:
                canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
            canvas, ppd, screen_shape = canvas_cache[canvas_key]
            center_px = gaze_deg_to_screen_px(
                np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
                ppd=ppd,
                screen_shape=screen_shape,
            )
            patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
        except Exception as exc:
            predicted_rows.append(
                {
                    "window_row": int(i),
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    "status": "patch_failed",
                    "error": str(exc),
                }
            )
            continue

        candidates = candidate_templates
        scored: list[dict[str, Any]] = []
        if args.score_mode == "conditional_proxy":
            scorer_name = "first-order pixel isophote stability plus saturating path refresh"
        elif args.score_mode == "image_proxy":
            scorer_name = "shifted retinal patch modulation under candidate traces"
        else:
            scorer_name = "V1 digital twin response modulation under candidate traces"
        for cand in candidates:
            trace = cand["trace"]
            extent_deg = _path_extent_along_axis(trace, float(cand["candidate_axis_deg"]))
            refresh_benefit = float(np.log1p(max(extent_deg, 0.0) * 60.0))
            pixel_instability_cost = _pixel_isophote_cost(
                patch,
                float(cand["candidate_axis_deg"]),
                float(cand["candidate_scale_deg"]),
                ppd=float(ppd),
            )
            if args.score_mode == "conditional_proxy":
                pa = refresh_benefit
                pb = pixel_instability_cost
            elif args.score_mode == "image_proxy":
                response = _image_proxy_response(patch, trace, ppd=ppd)
                pa, pb = _score_response_series(response)
            else:
                assert scorer is not None
                response = scorer.response(patch, trace)
                pa, pb = _score_response_series(response)
            out = {
                "window_row": int(i),
                "window_id": int(row["window_id"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "real_drift_axis_deg": float(row["drift_orientation_deg"]),
                "image_gradient_axis_deg": float(row["image_gradient_axis_deg"]),
                "image_edge_axis_deg": float(row["image_edge_axis_deg"]),
                "image_spectrum_axis_deg": float(row["image_spectrum_orientation_deg"]),
                "candidate_axis_deg": float(cand["candidate_axis_deg"]),
                "candidate_scale_deg": float(cand["candidate_scale_deg"]),
                "candidate_anisotropy_ratio": float(cand["candidate_anisotropy_ratio"]),
                "motor_cost": float(cand["motor_cost"]),
                "path_extent_deg": float(extent_deg),
                "refresh_benefit": refresh_benefit,
                "pixel_instability_cost": pixel_instability_cost,
                "pose_aware_score": pa,
                "pose_blind_cost": pb,
                "response_stability_cost": pb,
                "score_mode": str(args.score_mode),
                "scorer_name": scorer_name,
            }
            scored.append({**out, "trace": trace})
            candidate_rows.append(out)

        predictions = _choose_predictions(scored, lambdas=lambdas, motor_gamma=float(args.motor_gamma))
        predictions.extend(_raw_axis_rows(row))
        if len(example_payloads) < 6:
            example_payloads.append((patch, scored, row))
        for pred in predictions:
            axis = float(pred["candidate_axis_deg"])
            base = {
                "window_row": int(i),
                "window_id": int(row["window_id"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "objective": str(pred["objective"]),
                "status": "ok",
                "predicted_axis_deg": axis,
                "real_drift_axis_deg": float(row["drift_orientation_deg"]),
                "axis_delta_deg": float(_axis_delta_deg(float(row["drift_orientation_deg"]), axis)),
                "cos2_alignment": float(_cos2(float(row["drift_orientation_deg"]), axis)),
                "alignment_weight": float(row["anisotropy"]) * float(row["image_orientation_coherence"]),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "score_mode": str(args.score_mode),
            }
            for key in (
                "candidate_scale_deg",
                "candidate_anisotropy_ratio",
                "pose_aware_score",
                "pose_blind_cost",
                "response_stability_cost",
                "pixel_instability_cost",
                "refresh_benefit",
                "path_extent_deg",
                "motor_cost",
                "pose_aware_score_z",
                "pose_blind_cost_z",
                "pixel_instability_cost_z",
                "refresh_benefit_z",
                "motor_cost_z",
            ):
                if key in pred:
                    base[key] = pred[key]
            predicted_rows.append(base)
            alignment_rows.append(base)

    _write_csv(out_dir / "candidate_trajectory_scores.csv", candidate_rows)
    _write_csv(out_dir / "predicted_axis_by_objective.csv", predicted_rows)
    _write_csv(out_dir / "real_vs_predicted_axis_alignment.csv", alignment_rows)
    summary_rows = _session_summary(alignment_rows)
    _write_csv(out_dir / "alignment_by_objective_summary.csv", summary_rows)
    paired_rows = _paired_delta_summary(
        alignment_rows,
        baseline="raw_edge_axis",
        rng=rng,
        n_bootstrap=int(args.n_session_bootstrap),
    )
    _write_csv(out_dir / "paired_session_deltas_vs_raw_edge.csv", paired_rows)
    null_rows = _axis_null_summary(
        alignment_rows,
        axes_deg=axes,
        rng=rng,
        n_nulls=int(args.n_axis_nulls),
    )
    null_rows.extend(_predicted_axis_shuffle_null_summary(
        alignment_rows,
        rng=rng,
        n_nulls=int(args.n_shuffle_nulls),
    ))
    _write_csv(out_dir / "alignment_null_summary.csv", null_rows)
    _write_csv(out_dir / "alignment_axis_null_summary.csv", null_rows)
    stratified_rows = _stratified_summary(alignment_rows)
    _write_csv(out_dir / "stratified_alignment_summary.csv", stratified_rows)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "config": asdict(cfg),
            "n_input_rows": int(df.shape[0]),
            "n_reliable_rows": int(work.shape[0]),
            "n_candidate_rows": int(len(candidate_rows)),
            "n_alignment_rows": int(len(alignment_rows)),
            "n_paired_delta_rows": int(len(paired_rows)),
            "n_axis_null_rows": int(len(null_rows)),
            "n_stratified_rows": int(len(stratified_rows)),
            "n_canvas_cache_entries": int(len(canvas_cache)),
            "score_note": (
                "conditional_proxy scores first-order pixel isophote stability plus path refresh; "
                "image_proxy scores retinal patch modulation under the same candidate grid; "
                "twin mode runs the V1 digital twin through Ryan's figure-4 helpers."
            ),
        },
    )
    _plot_summary(out_dir, summary_rows, example_payloads, args.score_mode)
    print(f"Wrote BackImage drift-geometry {args.score_mode} analysis to {out_dir}")
    return out_dir


def _plot_summary(out_dir: Path, summary_rows: list[dict[str, Any]], examples: list[tuple[np.ndarray, list[dict[str, Any]], pd.Series]], score_mode: str) -> None:
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(df)), 3.8), dpi=150)
        order = df.sort_values("mean_cos2_session_mean", ascending=False)
        ax.bar(np.arange(order.shape[0]), order["mean_cos2_session_mean"], color="#4c78a8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(order.shape[0]))
        ax.set_xticklabels(order["objective"], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("mean session cos2(real - predicted)")
        ax.set_title(f"BackImage drift-axis alignment ({score_mode})", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_alignment_by_objective.png", dpi=150)
        plt.close(fig)

    if examples:
        fig, axes = plt.subplots(len(examples), 2, figsize=(7, 2.2 * len(examples)), dpi=140)
        axes = np.atleast_2d(axes)
        for ax_pair, (patch, scored, row) in zip(axes, examples, strict=False):
            ax_pair[0].imshow(patch, cmap="gray", vmin=0, vmax=255, origin="upper")
            ax_pair[0].set_axis_off()
            ax_pair[0].set_title(f"{row['session']} trial {int(row['trial_idx'])}", fontsize=7)
            block = pd.DataFrame([{k: v for k, v in s.items() if k != "trace"} for s in scored])
            best = block.groupby("candidate_axis_deg")["pose_aware_score"].mean()
            ax_pair[1].plot(best.index, best.values, marker="o", linewidth=1)
            ax_pair[1].axvline(float(row["drift_orientation_deg"]) % 180, color="#d62728", linewidth=1, label="real")
            ax_pair[1].axvline(float(row["image_edge_axis_deg"]) % 180, color="#2ca02c", linewidth=1, label="edge")
            ax_pair[1].set_xlim(0, 180)
            ax_pair[1].set_xlabel("candidate axis (deg)")
            ax_pair[1].set_ylabel("PA score")
            ax_pair[1].tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_objective_axis_tuning_examples.png", dpi=140)
        plt.close(fig)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
