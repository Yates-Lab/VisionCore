#!/usr/bin/env python3
"""Observed-vs-random-rotation predictions across behavior coherence bins."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
import _fig4_paths as _paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import _fig4_bridge as bridge
import _fig4_bridge_rotation_null as rotation_null


OUT_DIR = _paths.FIG_DIR / "behavior_model_bridge"
OUT_STEM = "behavior_model_bridge_random_rotation_prediction_by_coherence"
DEFAULT_N_ROTATIONS = 256
DEFAULT_N_BOOTSTRAP = 10_000
SEED = 197

PRIMARY_METRICS = ("component_rms", "component_range")
PLOT_POPULATION_ORDER = (
    "high_sf_aligned",
    "high_sf_oblique",
    "high_sf_orthogonal",
    "high_sf_all",
    "low_sf_all",
)

POPULATION_COLORS = {
    "high_sf_aligned": "#b4492d",
    "high_sf_oblique": "#2f6f9f",
    "high_sf_orthogonal": "#7251a5",
    "high_sf_all": "#58606a",
    "low_sf_all": "#1b7f5c",
}

POPULATION_MARKERS = {
    "high_sf_aligned": "o",
    "high_sf_oblique": "s",
    "high_sf_orthogonal": "^",
    "high_sf_all": "D",
    "low_sf_all": "v",
}

POPULATION_LABELS = {
    "high_sf_aligned": "Aligned high-SF",
    "high_sf_oblique": "Oblique high-SF",
    "high_sf_orthogonal": "Orthogonal high-SF",
    "high_sf_all": "All high-SF",
    "low_sf_all": "All low-SF",
}

METRIC_TITLES = {
    "component_path": "Component path",
    "component_rms": "RMS excursion",
    "component_range": "Projected range",
    "path_per_range": "Path/range",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flat_bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Single-stage bootstrap treating every fixation window as an
    independent, exchangeable observation, with no session clustering --
    the CI is meant to convey uncertainty from the finite sample of
    fixations/local image content, not from the finite sample of recording
    sessions. Session effects are a separate question, checked elsewhere,
    not something this error bar marginalizes over."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(vals))
    if vals.size <= 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    draw = rng.integers(0, vals.size, size=(int(n_bootstrap), vals.size))
    boot = vals[draw].mean(axis=1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _permutation_p(obs_global: float, null_by_draw: np.ndarray) -> tuple[float, float, int]:
    """Two-sided and one-sided permutation p from a null distribution built
    one value per null draw (one rotation angle, or one scramble partner),
    pooled across sessions/windows for that draw. A separate diagnostic from
    the plotted CI -- always session-pooled, needs the per-draw axis the
    window-level nested bootstrap collapses away."""
    null_vals = null_by_draw[np.isfinite(null_by_draw)]
    if not (math.isfinite(obs_global) and null_vals.size):
        return float("nan"), float("nan"), int(null_vals.size)
    p_observed_greater = float((np.count_nonzero(null_vals >= obs_global) + 1) / (null_vals.size + 1))
    null_center = float(np.mean(null_vals))
    p_two_sided = float(
        (np.count_nonzero(np.abs(null_vals - null_center) >= abs(obs_global - null_center)) + 1) / (null_vals.size + 1)
    )
    return p_observed_greater, p_two_sided, int(null_vals.size)


def _prediction_summary(
    *,
    obs_window: np.ndarray,
    rot_window: np.ndarray,
    window_session_idx: np.ndarray,
    obs_session_for_p: np.ndarray,
    rot_session_for_p: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int]:
    obs_w = np.asarray(obs_window, dtype=float)
    rot_w = np.asarray(rot_window, dtype=float)
    paired = np.isfinite(obs_w) & np.isfinite(rot_w)

    obs_mean, obs_lo, obs_hi = _flat_bootstrap_ci(obs_w, rng=rng, n_bootstrap=n_bootstrap)
    rot_mean, rot_lo, rot_hi = _flat_bootstrap_ci(rot_w, rng=rng, n_bootstrap=n_bootstrap)
    diff_window = np.where(paired, obs_w - rot_w, np.nan)
    diff_mean, diff_lo, diff_hi = _flat_bootstrap_ci(diff_window, rng=rng, n_bootstrap=n_bootstrap)

    obs_s = np.asarray(obs_session_for_p, dtype=float)
    rot_sr = np.asarray(rot_session_for_p, dtype=float)
    obs_global = float(np.nanmean(obs_s)) if np.isfinite(obs_s).any() else float("nan")
    rot_global_by_rotation = rotation_null._nanmean(rot_sr, axis=0)
    p_observed_greater, p_two_sided, n_rotation_valid = _permutation_p(obs_global, rot_global_by_rotation)

    return {
        "observed_prediction": obs_mean,
        "observed_ci95_low": obs_lo,
        "observed_ci95_high": obs_hi,
        "random_rotated_prediction": rot_mean,
        "random_rotated_ci95_low": rot_lo,
        "random_rotated_ci95_high": rot_hi,
        "observed_minus_rotated": diff_mean,
        "observed_minus_rotated_ci95_low": diff_lo,
        "observed_minus_rotated_ci95_high": diff_hi,
        "p_rotation_observed_greater": p_observed_greater,
        "p_rotation_two_sided": p_two_sided,
        "n_windows_paired": int(np.count_nonzero(paired)),
        "n_sessions_contributing": int(np.unique(window_session_idx[paired]).size) if np.any(paired) else 0,
        "n_rotation_replicates_valid": n_rotation_valid,
    }


def _scramble_summary(
    *,
    obs_window: np.ndarray,
    scr_window: np.ndarray,
    window_session_idx: np.ndarray,
    obs_session_for_p: np.ndarray,
    scr_session_for_p: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int]:
    """Same structure as _prediction_summary, but against the trajectory-image
    scramble null instead of the axis-rotation null -- observed_prediction
    isn't repeated here since it's identical to _prediction_summary's (same
    real data), just paired against a different null."""
    obs_w = np.asarray(obs_window, dtype=float)
    scr_w = np.asarray(scr_window, dtype=float)
    paired = np.isfinite(obs_w) & np.isfinite(scr_w)

    scr_mean, scr_lo, scr_hi = _flat_bootstrap_ci(scr_w, rng=rng, n_bootstrap=n_bootstrap)
    diff_window = np.where(paired, obs_w - scr_w, np.nan)
    diff_mean, diff_lo, diff_hi = _flat_bootstrap_ci(diff_window, rng=rng, n_bootstrap=n_bootstrap)

    obs_s = np.asarray(obs_session_for_p, dtype=float)
    scr_sr = np.asarray(scr_session_for_p, dtype=float)
    obs_global = float(np.nanmean(obs_s)) if np.isfinite(obs_s).any() else float("nan")
    scr_global_by_draw = rotation_null._nanmean(scr_sr, axis=0)
    p_observed_greater, p_two_sided, n_scramble_valid = _permutation_p(obs_global, scr_global_by_draw)

    return {
        "scrambled_prediction": scr_mean,
        "scrambled_ci95_low": scr_lo,
        "scrambled_ci95_high": scr_hi,
        "observed_minus_scrambled": diff_mean,
        "observed_minus_scrambled_ci95_low": diff_lo,
        "observed_minus_scrambled_ci95_high": diff_hi,
        "p_scramble_observed_greater": p_observed_greater,
        "p_scramble_two_sided": p_two_sided,
        "n_scramble_replicates_valid": n_scramble_valid,
    }


def run_by_coherence(
    *,
    n_rotations: int = DEFAULT_N_ROTATIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(bridge.BEHAVIOR_WINDOWS_CSV)
    model_values = pd.read_csv(bridge.MODEL_VALUES_CSV)
    populations = pd.read_csv(bridge.MODEL_POPULATIONS_CSV)

    population_keys = populations["population_key"].astype(str).tolist()
    population_titles = dict(zip(populations["population_key"].astype(str), populations["population_title"].astype(str), strict=True))
    metric_keys = [str(family["key"]) for family in bridge.METRIC_FAMILIES]
    metric_titles = {str(family["key"]): str(family["title"]) for family in bridge.METRIC_FAMILIES}
    component_keys = [str(component[0]) for component in bridge.COMPONENTS]
    session_names = sorted(windows["session"].astype(str).unique())
    session_lookup = {session: idx for idx, session in enumerate(session_names)}
    coherence_lookup = {coherence: idx for idx, coherence in enumerate(bridge.COHERENCE_ORDER)}

    # Each fixation window has its own local image content (edge coherence,
    # edge axis at that gaze location), so windows -- not trials -- are the
    # right atomic unit of image-content variability: pooling same-trial
    # windows into one value first (as an earlier version of this script did)
    # wrongly treated them as repeats of "the same" measurement. Sessions are
    # kept as an outer cluster, since day-to-day behavioral state is a real,
    # separate correlation source windows don't share with each other across
    # sessions.
    n_windows_total = int(len(windows))
    window_session_idx = windows["session"].astype(str).map(session_lookup).to_numpy(dtype=np.int64)

    b = len(bridge.COHERENCE_ORDER)
    p = len(population_keys)
    f = len(metric_keys)
    c = len(component_keys)
    s = len(session_names)
    r = int(n_rotations)

    observed_sum = np.zeros((b, p, f, c, s), dtype=float)
    observed_count = np.zeros((b, p, f, c, s), dtype=np.int64)
    rotated_sum = np.zeros((b, p, f, c, s, r), dtype=float)
    rotated_count = np.zeros((b, p, f, c, s, r), dtype=np.int64)
    # A second, independent null: instead of rotating a window's real
    # trajectory away from its own true local edge axis (rotated_* above),
    # keep the trajectory's own axis-relative shape and re-evaluate it
    # against a *different* window's true edge axis, scrambling the
    # trajectory-image pairing directly. This is a distinct null hypothesis
    # from the rotation one -- it asks whether the specific pairing matters,
    # not just whether the true axis matters -- so it needs its own
    # accumulators, not a reuse of the rotation ones.
    scrambled_sum = np.zeros((b, p, f, c, s, r), dtype=float)
    scrambled_count = np.zeros((b, p, f, c, s, r), dtype=np.int64)
    window_count = np.zeros((b, s), dtype=np.int64)
    # Window-level twins used for the nested session-then-window bootstrap,
    # instead of the diagnostic per-session table. Rotations/scrambles are
    # collapsed to a per-window mean immediately (nan-safe), since the
    # bootstrap treats each window's mean prediction as one atomic
    # observation -- no need to carry all r draws per window past this point.
    observed_val_window = np.full((b, p, f, c, n_windows_total), np.nan, dtype=np.float64)
    rotated_val_window = np.full((b, p, f, c, n_windows_total), np.nan, dtype=np.float64)
    scrambled_val_window = np.full((b, p, f, c, n_windows_total), np.nan, dtype=np.float64)

    curves = {
        (population_key, metric_key, component_key): bridge._curve_for(
            model_values,
            population_key=population_key,
            metric_family=metric_key,
            component=component_key,
        )
        for population_key in population_keys
        for metric_key in metric_keys
        for component_key in component_keys
    }

    rng = np.random.default_rng(seed)
    random_angles = rng.uniform(0.0, np.pi, size=(len(windows), r))
    observed_angles = np.zeros(1, dtype=float)

    # For the scramble null: for every window, draw r partner windows (with
    # replacement, from every window with a finite true edge axis) and use
    # each partner's axis instead of the window's own. Axes are 180-periodic
    # (an edge orientation, not a directed vector), so the rotation applied
    # is the signed difference mod 180 -- same domain convention as
    # random_angles above.
    all_axis_deg = pd.to_numeric(windows["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64)
    valid_axis_pool_idx = np.flatnonzero(np.isfinite(all_axis_deg))
    partner_pool_positions = rng.integers(0, valid_axis_pool_idx.size, size=(len(windows), r))
    partner_axis_deg = all_axis_deg[valid_axis_pool_idx[partner_pool_positions]]
    scramble_angles = np.radians((partner_axis_deg - all_axis_deg[:, None]) % 180.0)

    for row_idx, row_tuple in enumerate(windows.itertuples(index=False), start=0):
        row = pd.Series(row_tuple._asdict())
        coherence_bin = str(row["coherence_bin"])
        if coherence_bin not in coherence_lookup:
            continue
        coords = rotation_null._axis_coordinates(row)
        if coords is None:
            continue
        bin_idx = coherence_lookup[coherence_bin]
        session_idx = session_lookup[str(row["session"])]
        window_count[bin_idx, session_idx] += 1

        pos_along, pos_across, step_along, step_across = coords
        observed_metrics = rotation_null._component_metrics_for_angles(
            pos_along,
            pos_across,
            step_along,
            step_across,
            observed_angles,
        )
        rotated_metrics = rotation_null._component_metrics_for_angles(
            pos_along,
            pos_across,
            step_along,
            step_across,
            random_angles[row_idx],
        )
        scrambled_metrics = rotation_null._component_metrics_for_angles(
            pos_along,
            pos_across,
            step_along,
            step_across,
            scramble_angles[row_idx],
        )

        for metric_idx, metric_key in enumerate(metric_keys):
            for component_idx, component_key in enumerate(component_keys):
                x_obs = observed_metrics[(metric_key, component_key)]
                x_rot = rotated_metrics[(metric_key, component_key)]
                x_scr = scrambled_metrics[(metric_key, component_key)]
                for population_idx, population_key in enumerate(population_keys):
                    curve = curves[(population_key, metric_key, component_key)]
                    pred_obs, _outside_obs = bridge._interpolate_curve(x_obs, curve)
                    pred_rot, _outside_rot = bridge._interpolate_curve(x_rot, curve)
                    pred_scr, _outside_scr = bridge._interpolate_curve(x_scr, curve)
                    pred_obs_scalar = float(pred_obs[0]) if pred_obs.size else float("nan")
                    if np.isfinite(pred_obs_scalar):
                        observed_sum[bin_idx, population_idx, metric_idx, component_idx, session_idx] += pred_obs_scalar
                        observed_count[bin_idx, population_idx, metric_idx, component_idx, session_idx] += 1
                        observed_val_window[bin_idx, population_idx, metric_idx, component_idx, row_idx] = pred_obs_scalar
                    rot_ok = np.isfinite(pred_rot)
                    rotated_sum[bin_idx, population_idx, metric_idx, component_idx, session_idx, rot_ok] += pred_rot[rot_ok]
                    rotated_count[bin_idx, population_idx, metric_idx, component_idx, session_idx, rot_ok] += 1
                    if np.any(rot_ok):
                        rotated_val_window[bin_idx, population_idx, metric_idx, component_idx, row_idx] = float(
                            np.mean(pred_rot[rot_ok])
                        )
                    scr_ok = np.isfinite(pred_scr)
                    scrambled_sum[bin_idx, population_idx, metric_idx, component_idx, session_idx, scr_ok] += pred_scr[scr_ok]
                    scrambled_count[bin_idx, population_idx, metric_idx, component_idx, session_idx, scr_ok] += 1
                    if np.any(scr_ok):
                        scrambled_val_window[bin_idx, population_idx, metric_idx, component_idx, row_idx] = float(
                            np.mean(pred_scr[scr_ok])
                        )

        if (row_idx + 1) % 1000 == 0:
            print(f"processed random-rotation coherence bridge windows {row_idx + 1}/{len(windows)}", flush=True)

    observed_session = rotation_null._safe_divide(observed_sum, observed_count)
    rotated_session = rotation_null._safe_divide(rotated_sum, rotated_count)
    scrambled_session = rotation_null._safe_divide(scrambled_sum, scrambled_count)

    rows: list[dict[str, Any]] = []
    rng_summary = np.random.default_rng(seed + 10_000)
    for bin_idx, coherence_bin in enumerate(bridge.COHERENCE_ORDER):
        n_windows = int(np.sum(window_count[bin_idx]))
        for population_idx, population_key in enumerate(population_keys):
            for metric_idx, metric_key in enumerate(metric_keys):
                for component_idx, component_key in enumerate(component_keys):
                    score = _prediction_summary(
                        obs_window=observed_val_window[bin_idx, population_idx, metric_idx, component_idx],
                        rot_window=rotated_val_window[bin_idx, population_idx, metric_idx, component_idx],
                        window_session_idx=window_session_idx,
                        obs_session_for_p=observed_session[bin_idx, population_idx, metric_idx, component_idx],
                        rot_session_for_p=rotated_session[bin_idx, population_idx, metric_idx, component_idx],
                        rng=rng_summary,
                        n_bootstrap=n_bootstrap,
                    )
                    scramble_score = _scramble_summary(
                        obs_window=observed_val_window[bin_idx, population_idx, metric_idx, component_idx],
                        scr_window=scrambled_val_window[bin_idx, population_idx, metric_idx, component_idx],
                        window_session_idx=window_session_idx,
                        obs_session_for_p=observed_session[bin_idx, population_idx, metric_idx, component_idx],
                        scr_session_for_p=scrambled_session[bin_idx, population_idx, metric_idx, component_idx],
                        rng=rng_summary,
                        n_bootstrap=n_bootstrap,
                    )
                    valid_obs = int(np.sum(observed_count[bin_idx, population_idx, metric_idx, component_idx]))
                    valid_rot = int(np.sum(rotated_count[bin_idx, population_idx, metric_idx, component_idx]))
                    rows.append(
                        {
                            "coherence_bin": coherence_bin,
                            "coherence_bin_order": bin_idx,
                            "score_type": "component",
                            "population_key": population_key,
                            "population_title": population_titles[population_key],
                            "metric_family": metric_key,
                            "metric_title": metric_titles[metric_key],
                            "component": component_key,
                            "component_label": rotation_null.COMPONENT_LABELS[component_key],
                            "n_windows": n_windows,
                            "n_rotations": r,
                            "observed_valid_fraction": valid_obs / n_windows if n_windows else float("nan"),
                            "rotated_valid_fraction": valid_rot / (n_windows * r) if n_windows else float("nan"),
                            **score,
                            **scramble_score,
                        }
                    )

                obs_both_window = rotation_null._nanmean(observed_val_window[bin_idx, population_idx, metric_idx], axis=0)
                rot_both_window = rotation_null._nanmean(rotated_val_window[bin_idx, population_idx, metric_idx], axis=0)
                scr_both_window = rotation_null._nanmean(scrambled_val_window[bin_idx, population_idx, metric_idx], axis=0)
                obs_both_session = rotation_null._nanmean(observed_session[bin_idx, population_idx, metric_idx], axis=0)
                rot_both_session = rotation_null._nanmean(rotated_session[bin_idx, population_idx, metric_idx], axis=0)
                scr_both_session = rotation_null._nanmean(scrambled_session[bin_idx, population_idx, metric_idx], axis=0)
                both_score = _prediction_summary(
                    obs_window=obs_both_window,
                    rot_window=rot_both_window,
                    window_session_idx=window_session_idx,
                    obs_session_for_p=obs_both_session,
                    rot_session_for_p=rot_both_session,
                    rng=rng_summary,
                    n_bootstrap=n_bootstrap,
                )
                both_scramble_score = _scramble_summary(
                    obs_window=obs_both_window,
                    scr_window=scr_both_window,
                    window_session_idx=window_session_idx,
                    obs_session_for_p=obs_both_session,
                    scr_session_for_p=scr_both_session,
                    rng=rng_summary,
                    n_bootstrap=n_bootstrap,
                )
                component_valid_obs = [
                    int(np.sum(observed_count[bin_idx, population_idx, metric_idx, component_idx]))
                    for component_idx in range(c)
                ]
                component_valid_rot = [
                    int(np.sum(rotated_count[bin_idx, population_idx, metric_idx, component_idx]))
                    for component_idx in range(c)
                ]
                rows.append(
                    {
                        "coherence_bin": coherence_bin,
                        "coherence_bin_order": bin_idx,
                        "score_type": "component_mean_marginal",
                        "population_key": population_key,
                        "population_title": population_titles[population_key],
                        "metric_family": metric_key,
                        "metric_title": metric_titles[metric_key],
                        "component": "both",
                        "component_label": rotation_null.COMPONENT_LABELS["both"],
                        "n_windows": n_windows,
                        "n_rotations": r,
                        "observed_valid_fraction": float(np.mean(component_valid_obs) / n_windows)
                        if n_windows
                        else float("nan"),
                        "rotated_valid_fraction": float(np.mean(component_valid_rot) / (n_windows * r))
                        if n_windows
                        else float("nan"),
                        **both_score,
                        **both_scramble_score,
                    }
                )

    summary = pd.DataFrame(rows)

    session_rows: list[dict[str, Any]] = []
    for bin_idx, coherence_bin in enumerate(bridge.COHERENCE_ORDER):
        for population_idx, population_key in enumerate(population_keys):
            for metric_idx, metric_key in enumerate(metric_keys):
                obs_both = rotation_null._nanmean(observed_session[bin_idx, population_idx, metric_idx], axis=0)
                rot_both = rotation_null._nanmean(rotation_null._nanmean(rotated_session[bin_idx, population_idx, metric_idx], axis=0), axis=1)
                for session_idx, session in enumerate(session_names):
                    session_rows.append(
                        {
                            "coherence_bin": coherence_bin,
                            "coherence_bin_order": bin_idx,
                            "score_type": "component_mean_marginal",
                            "population_key": population_key,
                            "metric_family": metric_key,
                            "component": "both",
                            "session": session,
                            "observed_prediction": obs_both[session_idx],
                            "random_rotated_prediction": rot_both[session_idx],
                            "observed_minus_rotated": obs_both[session_idx] - rot_both[session_idx],
                            "n_windows": int(window_count[bin_idx, session_idx]),
                        }
                    )
                for component_idx, component_key in enumerate(component_keys):
                    obs_values = observed_session[bin_idx, population_idx, metric_idx, component_idx]
                    rot_values = rotation_null._nanmean(rotated_session[bin_idx, population_idx, metric_idx, component_idx], axis=1)
                    for session_idx, session in enumerate(session_names):
                        session_rows.append(
                            {
                                "coherence_bin": coherence_bin,
                                "coherence_bin_order": bin_idx,
                                "score_type": "component",
                                "population_key": population_key,
                                "metric_family": metric_key,
                                "component": component_key,
                                "session": session,
                                "observed_prediction": obs_values[session_idx],
                                "random_rotated_prediction": rot_values[session_idx],
                                "observed_minus_rotated": obs_values[session_idx] - rot_values[session_idx],
                                "n_windows": int(window_count[bin_idx, session_idx]),
                            }
                        )

    return summary, pd.DataFrame(session_rows)


def _ordered_component_mean(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[summary["score_type"].astype(str).eq("component_mean_marginal")].copy()
    frame["coherence_bin"] = pd.Categorical(frame["coherence_bin"], categories=bridge.COHERENCE_ORDER, ordered=True)
    return frame.sort_values(["population_key", "metric_family", "coherence_bin"])


def plot_prediction_curves(summary: pd.DataFrame, *, out_dir: Path) -> dict[str, Path]:
    bridge.configure_matplotlib()
    pdf = out_dir / f"{OUT_STEM}_component_mean_predictions.pdf"
    png_paths: dict[str, Path] = {}
    frame = _ordered_component_mean(summary)
    population_order = frame["population_key"].drop_duplicates().astype(str).tolist()
    metric_order = [str(family["key"]) for family in bridge.METRIC_FAMILIES]
    x = np.arange(len(bridge.COHERENCE_ORDER), dtype=float)

    with PdfPages(pdf) as pages:
        for population_key in population_order:
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharey=True, constrained_layout=True)
            population_frame = frame[frame["population_key"].astype(str).eq(population_key)].copy()
            title = str(population_frame["population_title"].dropna().iloc[0])
            for ax, metric_key in zip(axes.ravel(), metric_order, strict=True):
                sub = population_frame[population_frame["metric_family"].astype(str).eq(metric_key)].copy()
                sub = sub.sort_values("coherence_bin")
                y_obs = sub["observed_prediction"].to_numpy(dtype=float)
                obs_lo = sub["observed_ci95_low"].to_numpy(dtype=float)
                obs_hi = sub["observed_ci95_high"].to_numpy(dtype=float)
                y_rot = sub["random_rotated_prediction"].to_numpy(dtype=float)
                rot_lo = sub["random_rotated_ci95_low"].to_numpy(dtype=float)
                rot_hi = sub["random_rotated_ci95_high"].to_numpy(dtype=float)
                ax.errorbar(
                    x - 0.03,
                    y_obs,
                    yerr=np.vstack([y_obs - obs_lo, obs_hi - y_obs]),
                    color="#111111",
                    marker="o",
                    markerfacecolor="white",
                    lw=1.8,
                    capsize=2.4,
                    label="observed matching",
                )
                ax.errorbar(
                    x + 0.03,
                    y_rot,
                    yerr=np.vstack([y_rot - rot_lo, rot_hi - y_rot]),
                    color="#7d8793",
                    marker="s",
                    markerfacecolor="white",
                    lw=1.6,
                    capsize=2.4,
                    ls=(0, (4, 2)),
                    label="random rotations",
                )
                ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
                ax.set_xticks(x)
                ax.set_xticklabels(bridge.COHERENCE_ORDER)
                ax.set_title(METRIC_TITLES.get(metric_key, metric_key), loc="left", fontweight="bold")
                ax.set_xlabel("local edge coherence")
                ax.grid(axis="y", color=bridge.GRID, lw=0.75)
                bridge._clean_axis(ax)
            axes[0, 0].set_ylabel("predicted SSI residual (%)")
            axes[1, 0].set_ylabel("predicted SSI residual (%)")
            axes[0, 0].legend(frameon=False, fontsize=7.5, loc="best")
            fig.suptitle(
                f"Observed vs random-rotated trace-contour matching: {title}\ncomponent-mean marginal predictions",
                fontsize=12.0,
                fontweight="bold",
            )
            png = out_dir / f"{OUT_STEM}_component_mean_predictions_{population_key}.png"
            fig.savefig(png, dpi=230)
            pages.savefig(fig)
            png_paths[population_key] = png
            plt.close(fig)

    return {"pdf": pdf, **{f"png_{key}": path for key, path in png_paths.items()}}


def plot_advantage_main(summary: pd.DataFrame, *, out_dir: Path) -> dict[str, Path]:
    bridge.configure_matplotlib()
    frame = _ordered_component_mean(summary)
    frame = frame[
        frame["population_key"].astype(str).isin(PLOT_POPULATION_ORDER)
        & frame["metric_family"].astype(str).isin(PRIMARY_METRICS)
    ].copy()
    x = np.arange(len(bridge.COHERENCE_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True, constrained_layout=True)

    for ax, metric_key in zip(axes, PRIMARY_METRICS, strict=True):
        sub_metric = frame[frame["metric_family"].astype(str).eq(metric_key)].copy()
        ax.axhspan(0.0, 0.25, color="#6fb58b", alpha=0.09, lw=0)
        ax.axhspan(-0.25, 0.0, color="#b9c1ca", alpha=0.10, lw=0)
        ax.axhline(0.0, color="0.30", lw=1.0, ls=":")
        for population_key in PLOT_POPULATION_ORDER:
            sub = sub_metric[sub_metric["population_key"].astype(str).eq(population_key)].copy()
            sub = sub.sort_values("coherence_bin")
            y = sub["observed_minus_rotated"].to_numpy(dtype=float)
            lo = sub["observed_minus_rotated_ci95_low"].to_numpy(dtype=float)
            hi = sub["observed_minus_rotated_ci95_high"].to_numpy(dtype=float)
            lw = 2.2 if population_key == "high_sf_aligned" else 1.45
            alpha = 1.0 if population_key in ("high_sf_aligned", "low_sf_all") else 0.78
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=POPULATION_COLORS[population_key],
                marker=POPULATION_MARKERS[population_key],
                markerfacecolor="white",
                markeredgewidth=1.3,
                lw=lw,
                capsize=2.4,
                alpha=alpha,
                label=POPULATION_LABELS[population_key],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(bridge.COHERENCE_ORDER)
        ax.set_xlabel("local edge coherence")
        ax.set_title(METRIC_TITLES[metric_key], loc="left", fontweight="bold")
        ax.grid(axis="y", color=bridge.GRID, lw=0.75)
        bridge._clean_axis(ax)

    axes[0].set_ylabel("observed - random rotated prediction\n(pp SSI)")
    axes[1].legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[0].text(
        0.02,
        0.96,
        "observed matching better",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="#42734f",
        fontsize=7.4,
    )
    axes[0].text(
        0.02,
        0.06,
        "random rotation better",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        color="#6B6F75",
        fontsize=7.4,
    )

    fig.suptitle(
        "Trace-contour matching advantage is high-SF selective across coherence",
        fontsize=12.4,
        fontweight="bold",
    )
    png = out_dir / f"{OUT_STEM}_match_advantage_main.png"
    pdf = out_dir / f"{OUT_STEM}_match_advantage_main.pdf"
    svg = out_dir / f"{OUT_STEM}_match_advantage_main.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    return {"png": png, "pdf": pdf, "svg": svg}


def build(
    out_dir: Path = OUT_DIR,
    *,
    n_rotations: int = DEFAULT_N_ROTATIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{OUT_STEM}_summary.csv"
    session_values_csv = out_dir / f"{OUT_STEM}_session_values.csv"
    provenance_json = out_dir / f"{OUT_STEM}_provenance.json"

    summary, session_values = run_by_coherence(
        n_rotations=n_rotations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary.to_csv(summary_csv, index=False)
    session_values.to_csv(session_values_csv, index=False)
    prediction_paths = plot_prediction_curves(summary, out_dir=out_dir)
    advantage_paths = plot_advantage_main(summary, out_dir=out_dir)

    outputs = {
        "summary_csv": summary_csv,
        "session_values_csv": session_values_csv,
        "component_mean_predictions_pdf": prediction_paths["pdf"],
        "match_advantage_main_png": advantage_paths["png"],
        "match_advantage_main_pdf": advantage_paths["pdf"],
        "match_advantage_main_svg": advantage_paths["svg"],
        "provenance_json": provenance_json,
    }
    _write_json(
        provenance_json,
        {
            "analysis": OUT_STEM,
            "description": "Behavior-weighted observed-vs-random-rotated model predictions split by original local edge coherence bins.",
            "inputs": {
                "behavior_windows_csv": bridge.BEHAVIOR_WINDOWS_CSV,
                "model_values_csv": bridge.MODEL_VALUES_CSV,
                "model_populations_csv": bridge.MODEL_POPULATIONS_CSV,
            },
            "null": {
                "n_rotations": int(n_rotations),
                "rotation_distribution": "independent uniform angle in [0, pi) for each behavior window and null replicate",
                "snippet_n_samples": bridge.PANEL_G_SNIPPET_N_SAMPLES,
                "snippet_duration_s": (bridge.PANEL_G_SNIPPET_N_SAMPLES - 1) * bridge.DT,
                "seed": int(seed),
            },
            "summary": {
                "coherence_bins": bridge.COHERENCE_ORDER,
                "effect": "observed_minus_rotated; positive means observed trace-contour matching predicts higher SSI than random rotations",
                "prediction_units": "percent SSI residual vs model cell baseline; differences are percentage points",
                "component_mean_marginal_warning": "Average of contour-normal and contour-parallel one-dimensional marginal predictions; not a true 2D SSI surface.",
                "n_bootstrap": int(n_bootstrap),
            },
            "outputs": outputs,
        },
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-rotations", type=int, default=DEFAULT_N_ROTATIONS)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    paths = build(
        out_dir=args.out_dir,
        n_rotations=int(args.n_rotations),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
