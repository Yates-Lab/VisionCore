#!/usr/bin/env python3
"""Random-rotation null for behavior/model contour matching."""

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
from fixation_stats import plot_backimage_contour_motion_components as contour_motion


OUT_DIR = _paths.FIG_DIR / "behavior_model_bridge"
OUT_STEM = "behavior_model_bridge_random_rotation_match_null"
DEFAULT_N_ROTATIONS = 256
DEFAULT_N_BOOTSTRAP = 10_000
SEED = 173
EPS = 1e-12

SUBSETS = (
    ("all_windows", "All reviewed BackImage windows", None),
    ("coh_ge_0p2", "Contour coherence >=0.2", 0.2),
    ("coh_ge_0p5", "Contour coherence >=0.5", 0.5),
    ("coh_ge_0p8", "Contour coherence >=0.8", 0.8),
)

POPULATION_LABELS = {
    "high_sf_all": "High SF\nall",
    "high_sf_aligned": "High SF\naligned",
    "high_sf_oblique": "High SF\noblique",
    "high_sf_orthogonal": "High SF\northogonal",
    "low_sf_all": "Low SF\nall",
}

COMPONENT_LABELS = {
    "across": "contour-normal",
    "along": "contour-parallel",
    "both": "component-mean",
}

COLORS = {
    "across": "#7a3b9a",
    "along": "#1b7f5c",
    "both": "#222222",
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


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(numer), np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=np.asarray(denom) > 0)
    return out


def _nanmean(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    summed = np.nansum(arr, axis=axis)
    count = np.sum(finite, axis=axis)
    return _safe_divide(summed, count)


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    if arr.size <= 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    sample = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    boots = np.mean(arr[sample], axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _axis_coordinates(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    trace = contour_motion._window_trace(row)
    snippet = bridge._central_snippet(trace)
    if snippet.shape[0] < bridge.PANEL_G_SNIPPET_N_SAMPLES or not np.isfinite(snippet).all():
        return None
    along_vec, across_vec = contour_motion._axis_vectors(np.asarray([float(row["image_edge_axis_deg"])], dtype=float))
    along = along_vec[0]
    across = across_vec[0]
    centered = snippet - np.mean(snippet, axis=0, keepdims=True)
    steps = np.diff(snippet, axis=0)
    return centered @ along, centered @ across, steps @ along, steps @ across


def _component_metrics_for_angles(
    pos_along: np.ndarray,
    pos_across: np.ndarray,
    step_along: np.ndarray,
    step_across: np.ndarray,
    angles: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    phi = np.asarray(angles, dtype=float)
    c = np.cos(phi)[:, None]
    s = np.sin(phi)[:, None]

    along_pos = c * pos_along[None, :] - s * pos_across[None, :]
    across_pos = s * pos_along[None, :] + c * pos_across[None, :]
    along_step = c * step_along[None, :] - s * step_across[None, :]
    across_step = s * step_along[None, :] + c * step_across[None, :]

    along_path = np.sum(np.abs(along_step), axis=1) * 60.0
    across_path = np.sum(np.abs(across_step), axis=1) * 60.0
    along_rms = np.sqrt(np.mean(along_pos * along_pos, axis=1)) * 60.0
    across_rms = np.sqrt(np.mean(across_pos * across_pos, axis=1)) * 60.0
    along_range = (np.max(along_pos, axis=1) - np.min(along_pos, axis=1)) * 60.0
    across_range = (np.max(across_pos, axis=1) - np.min(across_pos, axis=1)) * 60.0

    return {
        ("component_path", "along"): along_path,
        ("component_path", "across"): across_path,
        ("component_rms", "along"): along_rms,
        ("component_rms", "across"): across_rms,
        ("component_range", "along"): along_range,
        ("component_range", "across"): across_range,
        ("path_per_range", "along"): np.divide(
            along_path,
            along_range,
            out=np.full_like(along_path, np.nan),
            where=along_range > EPS,
        ),
        ("path_per_range", "across"): np.divide(
            across_path,
            across_range,
            out=np.full_like(across_path, np.nan),
            where=across_range > EPS,
        ),
    }


def _subset_memberships(row: pd.Series) -> list[int]:
    coherence = pd.to_numeric(pd.Series([row["image_orientation_coherence"]]), errors="coerce").iloc[0]
    memberships: list[int] = []
    for idx, (_key, _label, threshold) in enumerate(SUBSETS):
        if threshold is None:
            memberships.append(idx)
        elif np.isfinite(coherence) and float(coherence) >= float(threshold):
            memberships.append(idx)
    return memberships


def _summarize_score(
    *,
    obs_session: np.ndarray,
    rot_session: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int]:
    obs_s = np.asarray(obs_session, dtype=float)
    rot_sr = np.asarray(rot_session, dtype=float)
    rot_mean_s = _nanmean(rot_sr, axis=1)
    paired = np.isfinite(obs_s) & np.isfinite(rot_mean_s)
    diff_s = obs_s[paired] - rot_mean_s[paired]
    diff_mean, diff_lo, diff_hi = _bootstrap_ci(diff_s, rng=rng, n_bootstrap=n_bootstrap)

    obs_global = float(np.nanmean(obs_s)) if np.isfinite(obs_s).any() else float("nan")
    rot_session_mean_global = float(np.nanmean(rot_mean_s)) if np.isfinite(rot_mean_s).any() else float("nan")
    rot_global_by_rotation = _nanmean(rot_sr, axis=0)
    rot_global_by_rotation = rot_global_by_rotation[np.isfinite(rot_global_by_rotation)]
    if rot_global_by_rotation.size and np.isfinite(obs_global):
        p_observed_greater = float((np.count_nonzero(rot_global_by_rotation >= obs_global) + 1) / (rot_global_by_rotation.size + 1))
        null_center = float(np.mean(rot_global_by_rotation))
        p_two_sided = float(
            (np.count_nonzero(np.abs(rot_global_by_rotation - null_center) >= abs(obs_global - null_center)) + 1)
            / (rot_global_by_rotation.size + 1)
        )
        rot_lo, rot_hi = np.nanpercentile(rot_global_by_rotation, [2.5, 97.5])
    else:
        p_observed_greater = float("nan")
        p_two_sided = float("nan")
        rot_lo = float("nan")
        rot_hi = float("nan")

    return {
        "observed_session_mean": obs_global,
        "rotated_session_mean": rot_session_mean_global,
        "observed_minus_rotated_session_mean": diff_mean,
        "observed_minus_rotated_ci95_low": diff_lo,
        "observed_minus_rotated_ci95_high": diff_hi,
        "rotation_null_ci95_low": float(rot_lo),
        "rotation_null_ci95_high": float(rot_hi),
        "p_rotation_observed_greater": p_observed_greater,
        "p_rotation_two_sided": p_two_sided,
        "n_sessions_paired": int(np.count_nonzero(paired)),
        "n_rotation_replicates_valid": int(rot_global_by_rotation.size),
    }


def run_null(
    *,
    n_rotations: int = DEFAULT_N_ROTATIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    q = len(SUBSETS)
    p = len(population_keys)
    f = len(metric_keys)
    c = len(component_keys)
    s = len(session_names)
    r = int(n_rotations)

    observed_sum = np.zeros((q, p, f, c, s), dtype=float)
    observed_count = np.zeros((q, p, f, c, s), dtype=np.int64)
    rotated_sum = np.zeros((q, p, f, c, s, r), dtype=float)
    rotated_count = np.zeros((q, p, f, c, s, r), dtype=np.int64)
    window_count = np.zeros((q, s), dtype=np.int64)

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

    for row_idx, row_tuple in enumerate(windows.itertuples(index=False), start=0):
        row = pd.Series(row_tuple._asdict())
        coords = _axis_coordinates(row)
        if coords is None:
            continue
        subset_ids = _subset_memberships(row)
        if not subset_ids:
            continue
        session_idx = session_lookup[str(row["session"])]
        for subset_idx in subset_ids:
            window_count[subset_idx, session_idx] += 1

        pos_along, pos_across, step_along, step_across = coords
        observed_metrics = _component_metrics_for_angles(pos_along, pos_across, step_along, step_across, observed_angles)
        rotated_metrics = _component_metrics_for_angles(pos_along, pos_across, step_along, step_across, random_angles[row_idx])

        for metric_idx, metric_key in enumerate(metric_keys):
            for component_idx, component_key in enumerate(component_keys):
                x_obs = observed_metrics[(metric_key, component_key)]
                x_rot = rotated_metrics[(metric_key, component_key)]
                for population_idx, population_key in enumerate(population_keys):
                    curve = curves[(population_key, metric_key, component_key)]
                    pred_obs, _outside_obs = bridge._interpolate_curve(x_obs, curve)
                    pred_rot, _outside_rot = bridge._interpolate_curve(x_rot, curve)
                    pred_obs_scalar = float(pred_obs[0]) if pred_obs.size else float("nan")
                    obs_ok = np.isfinite(pred_obs_scalar)
                    rot_ok = np.isfinite(pred_rot)
                    for subset_idx in subset_ids:
                        if obs_ok:
                            observed_sum[subset_idx, population_idx, metric_idx, component_idx, session_idx] += pred_obs_scalar
                            observed_count[subset_idx, population_idx, metric_idx, component_idx, session_idx] += 1
                        rotated_sum[subset_idx, population_idx, metric_idx, component_idx, session_idx, rot_ok] += pred_rot[rot_ok]
                        rotated_count[subset_idx, population_idx, metric_idx, component_idx, session_idx, rot_ok] += 1

        if (row_idx + 1) % 1000 == 0:
            print(f"processed random-rotation null windows {row_idx + 1}/{len(windows)}", flush=True)

    observed_session = _safe_divide(observed_sum, observed_count)
    rotated_session = _safe_divide(rotated_sum, rotated_count)

    rows: list[dict[str, Any]] = []
    rng_summary = np.random.default_rng(seed + 10_000)
    for subset_idx, (subset_key, subset_label, threshold) in enumerate(SUBSETS):
        subset_windows = int(np.sum(window_count[subset_idx]))
        for population_idx, population_key in enumerate(population_keys):
            for metric_idx, metric_key in enumerate(metric_keys):
                for component_idx, component_key in enumerate(component_keys):
                    score = _summarize_score(
                        obs_session=observed_session[subset_idx, population_idx, metric_idx, component_idx],
                        rot_session=rotated_session[subset_idx, population_idx, metric_idx, component_idx],
                        rng=rng_summary,
                        n_bootstrap=n_bootstrap,
                    )
                    valid_obs = int(np.sum(observed_count[subset_idx, population_idx, metric_idx, component_idx]))
                    valid_rot = int(np.sum(rotated_count[subset_idx, population_idx, metric_idx, component_idx]))
                    rows.append(
                        {
                            "subset_key": subset_key,
                            "subset_label": subset_label,
                            "subset_min_coherence": threshold,
                            "score_type": "component",
                            "population_key": population_key,
                            "population_title": population_titles[population_key],
                            "metric_family": metric_key,
                            "metric_title": metric_titles[metric_key],
                            "component": component_key,
                            "component_label": COMPONENT_LABELS[component_key],
                            "n_windows": subset_windows,
                            "n_rotations": r,
                            "observed_valid_fraction": valid_obs / subset_windows if subset_windows else float("nan"),
                            "rotated_valid_fraction": valid_rot / (subset_windows * r) if subset_windows else float("nan"),
                            **score,
                        }
                    )

                obs_both = _nanmean(observed_session[subset_idx, population_idx, metric_idx], axis=0)
                rot_both = _nanmean(rotated_session[subset_idx, population_idx, metric_idx], axis=0)
                both_score = _summarize_score(
                    obs_session=obs_both,
                    rot_session=rot_both,
                    rng=rng_summary,
                    n_bootstrap=n_bootstrap,
                )
                component_valid_obs = [
                    int(np.sum(observed_count[subset_idx, population_idx, metric_idx, component_idx]))
                    for component_idx in range(c)
                ]
                component_valid_rot = [
                    int(np.sum(rotated_count[subset_idx, population_idx, metric_idx, component_idx]))
                    for component_idx in range(c)
                ]
                rows.append(
                    {
                        "subset_key": subset_key,
                        "subset_label": subset_label,
                        "subset_min_coherence": threshold,
                        "score_type": "component_mean_marginal",
                        "population_key": population_key,
                        "population_title": population_titles[population_key],
                        "metric_family": metric_key,
                        "metric_title": metric_titles[metric_key],
                        "component": "both",
                        "component_label": COMPONENT_LABELS["both"],
                        "n_windows": subset_windows,
                        "n_rotations": r,
                        "observed_valid_fraction": float(np.mean(component_valid_obs) / subset_windows)
                        if subset_windows
                        else float("nan"),
                        "rotated_valid_fraction": float(np.mean(component_valid_rot) / (subset_windows * r))
                        if subset_windows
                        else float("nan"),
                        **both_score,
                    }
                )

    summary = pd.DataFrame(rows)
    session_rows: list[dict[str, Any]] = []
    for subset_idx, (subset_key, subset_label, _threshold) in enumerate(SUBSETS):
        for population_idx, population_key in enumerate(population_keys):
            for metric_idx, metric_key in enumerate(metric_keys):
                for component_idx, component_key in enumerate(component_keys):
                    obs_values = observed_session[subset_idx, population_idx, metric_idx, component_idx]
                    rot_values = _nanmean(rotated_session[subset_idx, population_idx, metric_idx, component_idx], axis=1)
                    for session_idx, session in enumerate(session_names):
                        session_rows.append(
                            {
                                "subset_key": subset_key,
                                "subset_label": subset_label,
                                "score_type": "component",
                                "population_key": population_key,
                                "metric_family": metric_key,
                                "component": component_key,
                                "session": session,
                                "observed_prediction": obs_values[session_idx],
                                "rotated_mean_prediction": rot_values[session_idx],
                                "observed_minus_rotated": obs_values[session_idx] - rot_values[session_idx],
                                "n_windows": int(window_count[subset_idx, session_idx]),
                            }
                        )
                obs_both = _nanmean(observed_session[subset_idx, population_idx, metric_idx], axis=0)
                rot_both = _nanmean(_nanmean(rotated_session[subset_idx, population_idx, metric_idx], axis=0), axis=1)
                for session_idx, session in enumerate(session_names):
                    session_rows.append(
                        {
                            "subset_key": subset_key,
                            "subset_label": subset_label,
                            "score_type": "component_mean_marginal",
                            "population_key": population_key,
                            "metric_family": metric_key,
                            "component": "both",
                            "session": session,
                            "observed_prediction": obs_both[session_idx],
                            "rotated_mean_prediction": rot_both[session_idx],
                            "observed_minus_rotated": obs_both[session_idx] - rot_both[session_idx],
                            "n_windows": int(window_count[subset_idx, session_idx]),
                        }
                    )

    session_values = pd.DataFrame(session_rows)

    rotation_rows: list[dict[str, Any]] = []
    for subset_idx, (subset_key, subset_label, _threshold) in enumerate(SUBSETS):
        for population_idx, population_key in enumerate(population_keys):
            for metric_idx, metric_key in enumerate(metric_keys):
                for component_idx, component_key in enumerate(component_keys):
                    rot_global = _nanmean(rotated_session[subset_idx, population_idx, metric_idx, component_idx], axis=0)
                    for rot_idx, value in enumerate(rot_global):
                        rotation_rows.append(
                            {
                                "subset_key": subset_key,
                                "subset_label": subset_label,
                                "score_type": "component",
                                "population_key": population_key,
                                "metric_family": metric_key,
                                "component": component_key,
                                "rotation_idx": rot_idx,
                                "rotated_prediction": value,
                            }
                        )
                rot_both = _nanmean(rotated_session[subset_idx, population_idx, metric_idx], axis=0)
                rot_both_global = _nanmean(rot_both, axis=0)
                for rot_idx, value in enumerate(rot_both_global):
                    rotation_rows.append(
                        {
                            "subset_key": subset_key,
                            "subset_label": subset_label,
                            "score_type": "component_mean_marginal",
                            "population_key": population_key,
                            "metric_family": metric_key,
                            "component": "both",
                            "rotation_idx": rot_idx,
                            "rotated_prediction": value,
                        }
                    )

    rotation_values = pd.DataFrame(rotation_rows)
    return summary, session_values, rotation_values


def _plot_summary(
    summary: pd.DataFrame,
    *,
    out_dir: Path,
    score_type: str,
    filename_suffix: str,
    primary_subset: str = "coh_ge_0p2",
) -> dict[str, Path]:
    bridge.configure_matplotlib()
    pdf = out_dir / f"{OUT_STEM}_{filename_suffix}.pdf"
    png = out_dir / f"{OUT_STEM}_{filename_suffix}_{primary_subset}.png"
    metric_order = [str(family["key"]) for family in bridge.METRIC_FAMILIES]
    population_order = summary["population_key"].drop_duplicates().astype(str).tolist()
    x = np.arange(len(population_order), dtype=float)
    subset_order = [subset[0] for subset in SUBSETS]

    with PdfPages(pdf) as pages:
        for subset_key in subset_order:
            subset_label = str(summary.loc[summary["subset_key"].astype(str).eq(subset_key), "subset_label"].iloc[0])
            frame = summary[
                summary["score_type"].astype(str).eq(score_type)
                & summary["subset_key"].astype(str).eq(subset_key)
            ].copy()
            fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1), sharey=True, constrained_layout=True)
            for ax, metric_key in zip(axes.ravel(), metric_order, strict=True):
                metric_frame = frame[frame["metric_family"].astype(str).eq(metric_key)].copy()
                if score_type == "component":
                    offsets = {"across": -0.10, "along": 0.10}
                    for component in ["across", "along"]:
                        sub = metric_frame[metric_frame["component"].astype(str).eq(component)].set_index("population_key")
                        y = np.asarray(
                            [sub.loc[pop, "observed_minus_rotated_session_mean"] if pop in sub.index else np.nan for pop in population_order],
                            dtype=float,
                        )
                        lo = np.asarray(
                            [sub.loc[pop, "observed_minus_rotated_ci95_low"] if pop in sub.index else np.nan for pop in population_order],
                            dtype=float,
                        )
                        hi = np.asarray(
                            [sub.loc[pop, "observed_minus_rotated_ci95_high"] if pop in sub.index else np.nan for pop in population_order],
                            dtype=float,
                        )
                        ax.errorbar(
                            x + offsets[component],
                            y,
                            yerr=np.vstack([y - lo, hi - y]),
                            fmt="o",
                            ms=4,
                            lw=1.2,
                            capsize=2.5,
                            color=COLORS[component],
                            markerfacecolor="white",
                            label=COMPONENT_LABELS[component],
                        )
                else:
                    sub = metric_frame[metric_frame["component"].astype(str).eq("both")].set_index("population_key")
                    y = np.asarray(
                        [sub.loc[pop, "observed_minus_rotated_session_mean"] if pop in sub.index else np.nan for pop in population_order],
                        dtype=float,
                    )
                    lo = np.asarray(
                        [sub.loc[pop, "observed_minus_rotated_ci95_low"] if pop in sub.index else np.nan for pop in population_order],
                        dtype=float,
                    )
                    hi = np.asarray(
                        [sub.loc[pop, "observed_minus_rotated_ci95_high"] if pop in sub.index else np.nan for pop in population_order],
                        dtype=float,
                    )
                    ax.errorbar(
                        x,
                        y,
                        yerr=np.vstack([y - lo, hi - y]),
                        fmt="o",
                        ms=4.5,
                        lw=1.25,
                        capsize=2.8,
                        color=COLORS["both"],
                        markerfacecolor="white",
                    )
                    finite = np.isfinite(y) & np.isfinite(lo)
                    for xpos, ypos, low in zip(x[finite], y[finite], lo[finite], strict=True):
                        if low > 0:
                            ax.text(xpos, ypos, "*", ha="center", va="bottom", fontsize=11, color=COLORS["both"])

                ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
                ax.set_xticks(x)
                ax.set_xticklabels([POPULATION_LABELS.get(pop, pop) for pop in population_order])
                ax.set_title(metric_key.replace("_", " "), loc="left", fontweight="bold")
                ax.grid(axis="y", color=bridge.GRID, lw=0.75)
                bridge._clean_axis(ax)
            axes[0, 0].set_ylabel("observed - random rotated prediction (pp SSI)")
            axes[1, 0].set_ylabel("observed - random rotated prediction (pp SSI)")
            if score_type == "component":
                axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
            fig.suptitle(
                f"Random-rotation match null: {subset_label}\npositive means observed trace-contour matching predicts higher SSI",
                fontsize=12.0,
                fontweight="bold",
            )
            if subset_key == primary_subset:
                fig.savefig(png, dpi=230)
            pages.savefig(fig)
            plt.close(fig)

    return {"pdf": pdf, "primary_png": png}


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
    rotation_values_csv = out_dir / f"{OUT_STEM}_rotation_values.csv"
    provenance_json = out_dir / f"{OUT_STEM}_provenance.json"

    summary, session_values, rotation_values = run_null(
        n_rotations=n_rotations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary.to_csv(summary_csv, index=False)
    session_values.to_csv(session_values_csv, index=False)
    rotation_values.to_csv(rotation_values_csv, index=False)

    component_mean_paths = _plot_summary(
        summary,
        out_dir=out_dir,
        score_type="component_mean_marginal",
        filename_suffix="component_mean_summary",
    )
    component_paths = _plot_summary(
        summary,
        out_dir=out_dir,
        score_type="component",
        filename_suffix="component_specific_summary",
    )

    outputs = {
        "summary_csv": summary_csv,
        "session_values_csv": session_values_csv,
        "rotation_values_csv": rotation_values_csv,
        "component_mean_summary_pdf": component_mean_paths["pdf"],
        "component_mean_summary_primary_png": component_mean_paths["primary_png"],
        "component_specific_summary_pdf": component_paths["pdf"],
        "component_specific_summary_primary_png": component_paths["primary_png"],
        "provenance_json": provenance_json,
    }
    _write_json(
        provenance_json,
        {
            "analysis": OUT_STEM,
            "description": "Randomly rotate each behavior eye trajectory relative to the fixed image contour axis and compare behavior-weighted model predictions with observed trace-contour matching.",
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
                "effect": "observed_minus_rotated_session_mean; positive means observed trace-contour matching predicts higher SSI than random rotations",
                "uncertainty": "session bootstrap over observed-minus-mean-rotated paired session values",
                "rotation_p": "p_rotation_observed_greater is the fraction of random-rotation global means at least as large as observed, plus-one corrected",
                "component_mean_marginal_warning": "Average of the contour-normal and contour-parallel one-dimensional marginal predictions; not a true 2D SSI surface.",
                "n_bootstrap": int(n_bootstrap),
                "subsets": SUBSETS,
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
