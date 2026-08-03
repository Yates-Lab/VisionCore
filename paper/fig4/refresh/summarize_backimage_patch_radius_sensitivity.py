#!/usr/bin/env python3
"""Summarize BackImage local-feature screens across image patch radii."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from tqdm import tqdm


DEFAULT_ROOT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_patch_radius_sensitivity_v1"
)

BASE_OUTPUT_ROOT = Path("outputs") / "fixation_statistics_by_stimulus_all_sessions_after_review"

SUMMARY_RADII = (
    ("r0p25", 0.25),
    ("r0p5", 0.5),
    ("r1p0", 1.0),
)

ALIGNMENT_RADII = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
ALIGNMENT_CMAP = "cividis"
SLOPE_COHERENCE_MIN = 0.3
ALIGNMENT_SWEEP_CACHE = "patch_radius_alignment_sweep_windows.csv"

WINDOW_COUNT_TABLES = {
    0.25: BASE_OUTPUT_ROOT / "backimage_image_structure_patch_radius_0p25_v1" / "run_metadata.json",
    0.5: BASE_OUTPUT_ROOT / "backimage_image_structure_patch_radius_0p5_v1" / "run_metadata.json",
    1.0: BASE_OUTPUT_ROOT / "backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1" / "run_metadata.json",
}

LOCAL_KEY_ROWS = (
    ("orientation_coherence", "drift_edge_cos2"),
    ("orientation_coherence", "rms_across_arcmin"),
    ("orientation_coherence", "rms_delta_along_minus_across_arcmin"),
    ("spectrum_anisotropy", "drift_edge_cos2"),
    ("spectrum_anisotropy", "rms_across_arcmin"),
    ("spectrum_anisotropy", "rms_delta_along_minus_across_arcmin"),
    ("oriented_8plus_cpd_power", "drift_edge_cos2"),
    ("oriented_8plus_cpd_power", "rms_across_arcmin"),
    ("oriented_8plus_cpd_power", "rms_radius_arcmin"),
)

SF_KEY_FEATURES = ("abs_power_4_8_cpd", "abs_power_8plus_cpd")
SF_KEY_METRICS = (
    "rms_radius_arcmin",
    "rms_across_arcmin",
    "rms_along_arcmin",
    "rms_delta_along_minus_across_arcmin",
)

@dataclass(frozen=True)
class PatchRadiusSummaryConfig:
    root: str
    alignment_radii_deg: tuple[float, ...]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_window_counts() -> pd.DataFrame:
    rows = []
    for radius, path in WINDOW_COUNT_TABLES.items():
        meta = _read_json(path)
        rows.append(
            {
                "patch_radius_deg": radius,
                "patch_full_width_deg": 2.0 * radius,
                "n_raw_augmented_windows": int(meta.get("n_raw_augmented_windows", np.nan)),
                "n_windows": int(meta.get("n_windows", np.nan)),
                "n_failed_image_feature_windows": int(meta.get("n_failed_image_feature_windows", np.nan)),
                "n_excluded_patch_contamination_windows": int(
                    meta.get("n_excluded_patch_contamination_windows", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("patch_radius_deg")


def _load_local(root: Path) -> pd.DataFrame:
    frames = []
    for label, radius in SUMMARY_RADII:
        path = root / f"local_feature_poles_{label}" / "pole_eye_metric_high_low_contrasts.csv"
        df = pd.read_csv(path)
        df.insert(0, "patch_radius_deg", radius)
        df.insert(1, "patch_full_width_deg", 2.0 * radius)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_sf(root: Path) -> pd.DataFrame:
    frames = []
    for label, radius in SUMMARY_RADII:
        path = root / f"sf_scaling_{label}" / "sf_controlled_slope_summary.csv"
        df = pd.read_csv(path)
        df.insert(0, "patch_radius_deg", radius)
        df.insert(1, "patch_full_width_deg", 2.0 * radius)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _finite_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")


def _rect_sum(integral: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _circular_axis_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a_deg - b_deg)))))


def _drift_orientation_deg(row: pd.Series) -> float:
    cxx = float(row["cov_xx_deg2"])
    cxy = float(row["cov_xy_deg2"])
    cyy = float(row["cov_yy_deg2"])
    return float(0.5 * np.degrees(np.arctan2(2.0 * cxy, cxx - cyy)))


def _valid_alignment_cache(df: pd.DataFrame) -> bool:
    if "patch_radius_deg" not in df.columns:
        return False
    present = {round(float(x), 4) for x in pd.unique(df["patch_radius_deg"])}
    expected = {round(float(x), 4) for x in ALIGNMENT_RADII}
    return expected.issubset(present)


def _compute_alignment_sweep_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    from .image_features import (
        _backimage_canvas,
        backimage_trial_geometry,
        gaze_deg_to_screen_px,
        image_axis_rad_to_gaze_axis_rad,
    )

    base = pd.read_csv(BASE_OUTPUT_ROOT / "window_features.csv")
    base = base[
        (base["stimulus"].astype(str) == "backimage")
        & (base["phase"].astype(str).isin(["mid_fixation", "late_fixation"]))
    ].copy()

    rows: list[dict[str, float | int | str]] = []
    count_rows: list[dict[str, float | int]] = []
    counts = {
        radius: {
            "patch_radius_deg": float(radius),
            "patch_full_width_deg": float(2.0 * radius),
            "n_raw_augmented_windows": 0,
            "n_windows": 0,
            "n_failed_image_feature_windows": 0,
            "n_excluded_patch_contamination_windows": 0,
        }
        for radius in ALIGNMENT_RADII
    }

    grouped = base.groupby(["session", "trial_idx"], sort=False)
    for (session, trial_idx), block in tqdm(grouped, total=len(grouped), desc="alignment sweep trials"):
        try:
            canvas, ppd, (height, width) = _backimage_canvas(str(session), int(trial_idx))
            geometry = backimage_trial_geometry(str(session), int(trial_idx))
        except Exception:
            for radius in ALIGNMENT_RADII:
                counts[radius]["n_raw_augmented_windows"] += int(block.shape[0])
                counts[radius]["n_failed_image_feature_windows"] += int(block.shape[0])
            continue

        arr = np.asarray(canvas, dtype=np.float64)
        gx = ndimage.sobel(arr, axis=1, mode="nearest")
        gy = ndimage.sobel(arr, axis=0, mode="nearest")
        jxx_int = _integral_image(gx * gx)
        jyy_int = _integral_image(gy * gy)
        jxy_int = _integral_image(gx * gy)

        dest_x0, dest_y0, dest_x1, dest_y1 = geometry["dest_rect"]
        yy, xx = np.indices(arr.shape)
        inside = (
            (xx >= dest_x0)
            & (xx < dest_x1)
            & (yy >= dest_y0)
            & (yy < dest_y1)
        ).astype(np.float64)
        background = np.isclose(arr, float(geometry["background"]), atol=1e-6).astype(np.float64)
        inside_int = _integral_image(inside)
        background_int = _integral_image(background)

        for idx, row in block.iterrows():
            gaze = np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])], dtype=np.float64)
            cx, cy = gaze_deg_to_screen_px(gaze, ppd=ppd, screen_shape=(height, width))
            drift_orientation = _drift_orientation_deg(row)
            distance_to_image_border_px = float(
                min(cx - dest_x0, dest_x1 - cx, cy - dest_y0, dest_y1 - cy)
            )

            for radius in ALIGNMENT_RADII:
                counts[radius]["n_raw_augmented_windows"] += 1
                rad = max(2, int(round(float(radius) * ppd)))
                x0 = max(0, int(round(cx)) - rad)
                x1 = min(width, int(round(cx)) + rad + 1)
                y0 = max(0, int(round(cy)) - rad)
                y1 = min(height, int(round(cy)) + rad + 1)
                area = int((x1 - x0) * (y1 - y0))
                if area < 16:
                    counts[radius]["n_failed_image_feature_windows"] += 1
                    continue

                patch_fraction_inside_image = _rect_sum(inside_int, x0, x1, y0, y1) / area
                patch_fraction_background = _rect_sum(background_int, x0, x1, y0, y1) / area
                if patch_fraction_inside_image < 0.98 or patch_fraction_background > 0.05:
                    counts[radius]["n_excluded_patch_contamination_windows"] += 1
                    continue

                jxx = _rect_sum(jxx_int, x0, x1, y0, y1) / area
                jyy = _rect_sum(jyy_int, x0, x1, y0, y1) / area
                jxy = _rect_sum(jxy_int, x0, x1, y0, y1) / area
                coherence_den = jxx + jyy
                if coherence_den <= 0.0:
                    counts[radius]["n_failed_image_feature_windows"] += 1
                    continue

                coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / coherence_den
                gradient_orientation_image = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
                edge_orientation = image_axis_rad_to_gaze_axis_rad(gradient_orientation_image + np.pi / 2.0)
                edge_axis_deg = float(np.degrees(edge_orientation))
                delta_deg = _circular_axis_delta_deg(drift_orientation, edge_axis_deg)
                counts[radius]["n_windows"] += 1
                rows.append(
                    {
                        "patch_radius_deg": float(radius),
                        "patch_full_width_deg": float(2.0 * radius),
                        "source_window_index": int(idx),
                        "session": str(row["session"]),
                        "trial_idx": int(row["trial_idx"]),
                        "phase": str(row["phase"]),
                        "image_orientation_coherence": float(coherence),
                        "image_edge_axis_deg": edge_axis_deg,
                        "drift_orientation_deg": drift_orientation,
                        "drift_edge_delta_deg": delta_deg,
                        "drift_edge_cos2": float(np.cos(2.0 * np.radians(delta_deg))),
                        "image_patch_fraction_inside_image": float(patch_fraction_inside_image),
                        "image_patch_fraction_background": float(patch_fraction_background),
                        "image_patch_distance_to_image_border_px": distance_to_image_border_px,
                    }
                )

    for radius in ALIGNMENT_RADII:
        count_rows.append(counts[radius])
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def _load_alignment_sweep_windows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = root / ALIGNMENT_SWEEP_CACHE
    counts_path = root / "patch_radius_alignment_sweep_window_counts.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if _valid_alignment_cache(cached):
            counts = pd.read_csv(counts_path) if counts_path.exists() else pd.DataFrame()
            return cached, counts

    sweep, counts = _compute_alignment_sweep_windows()
    sweep.to_csv(cache_path, index=False)
    counts.to_csv(counts_path, index=False)
    return sweep, counts


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    sem95 = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size) if values.size > 1 else 0.0
    return mean, float(mean - sem95), float(mean + sem95)


def _load_alignment_summaries(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sweep, counts = _load_alignment_sweep_windows(root)
    bins = np.linspace(0.0, 1.0, 11)
    bin_rows: list[dict[str, float | int]] = []
    slope_rows: list[dict[str, float | int]] = []
    for radius, work in sweep.groupby("patch_radius_deg"):

        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            if hi == bins[-1]:
                block = work[
                    (work["image_orientation_coherence"] >= lo)
                    & (work["image_orientation_coherence"] <= hi)
                ]
            else:
                block = work[
                    (work["image_orientation_coherence"] >= lo)
                    & (work["image_orientation_coherence"] < hi)
                ]
            if block.empty:
                continue
            values = block["drift_edge_cos2"].to_numpy(dtype=float)
            mean, ci95_low, ci95_high = _mean_ci95(values)
            bin_rows.append(
                {
                    "patch_radius_deg": float(radius),
                    "patch_full_width_deg": float(2.0 * radius),
                    "coherence_bin_low": float(lo),
                    "coherence_bin_high": float(hi),
                    "coherence_bin_center": float((lo + hi) / 2.0),
                    "mean_edge_alignment_index": mean,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "n_windows": int(len(block)),
                }
            )

        fit = work[work["image_orientation_coherence"] > SLOPE_COHERENCE_MIN].copy()
        x = fit["image_orientation_coherence"].to_numpy(dtype=float)
        y = fit["drift_edge_cos2"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size >= 3:
            fit_result = stats.linregress(x, y)
            dof = int(x.size - 2)
            tcrit = float(stats.t.ppf(0.975, dof)) if dof > 0 else float("nan")
            ci_half_width = tcrit * float(fit_result.stderr) if np.isfinite(tcrit) else float("nan")
            slope_rows.append(
                {
                    "patch_radius_deg": float(radius),
                    "patch_full_width_deg": float(2.0 * radius),
                    "coherence_min": float(SLOPE_COHERENCE_MIN),
                    "slope_alignment_per_coherence": float(fit_result.slope),
                    "intercept": float(fit_result.intercept),
                    "r_value": float(fit_result.rvalue),
                    "r_squared": float(fit_result.rvalue ** 2),
                    "p_value": float(fit_result.pvalue),
                    "stderr": float(fit_result.stderr),
                    "ci95_low": float(fit_result.slope - ci_half_width),
                    "ci95_high": float(fit_result.slope + ci_half_width),
                    "n_windows": int(x.size),
                }
            )
    return pd.DataFrame(bin_rows), pd.DataFrame(slope_rows), counts


def _plot_ci_series(ax: plt.Axes, df: pd.DataFrame, *, y: str, label: str | None = None, color: str = "C0") -> None:
    work = df.sort_values("patch_radius_deg")
    x = work["patch_full_width_deg"].to_numpy(dtype=float)
    vals = work[y].to_numpy(dtype=float)
    lo = work["ci95_low"].to_numpy(dtype=float)
    hi = work["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([vals - lo, hi - vals])
    ax.errorbar(x, vals, yerr=yerr, marker="o", lw=1.7, capsize=3, label=label, color=color)


def plot_local_key_effects(local_key: pd.DataFrame, out_path: Path) -> None:
    pairs = list(LOCAL_KEY_ROWS)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), sharex=True)
    for ax, (feature, metric) in zip(axes.ravel(), pairs, strict=True):
        sub = local_key[(local_key["feature"] == feature) & (local_key["eye_metric"] == metric)]
        if sub.empty:
            ax.set_visible(False)
            continue
        row0 = sub.iloc[0]
        _plot_ci_series(ax, sub, y="median_delta", color="#315f72")
        ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
        ax.set_title(f"{row0['feature_label']}\n{row0['eye_metric_label']}", fontsize=9)
        ax.grid(axis="y", color="0.88", lw=0.8)
        ax.set_xlim(0.35, 2.1)
    for ax in axes[-1, :]:
        ax.set_xlabel("image patch full width (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("high - low pole delta")
    fig.suptitle("Patch-radius sensitivity of local-feature pole effects", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sf_key_slopes(sf_key: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    colors = {
        "abs_power_4_8_cpd": "#7a5195",
        "abs_power_8plus_cpd": "#2f8f63",
    }
    for ax, metric in zip(axes.ravel(), SF_KEY_METRICS, strict=True):
        sub_metric = sf_key[sf_key["eye_metric"] == metric]
        for feature in SF_KEY_FEATURES:
            sub = sub_metric[sub_metric["feature"] == feature]
            if sub.empty:
                continue
            label = str(sub.iloc[0]["feature_label"])
            _plot_ci_series(
                ax,
                sub,
                y="controlled_beta_z_median",
                label=label,
                color=colors.get(feature, "C0"),
            )
        label = str(sub_metric.iloc[0]["eye_metric_label"]) if not sub_metric.empty else metric
        ax.set_title(label, fontsize=10)
        ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
        ax.grid(axis="y", color="0.88", lw=0.8)
        ax.set_xlim(0.35, 2.1)
    for ax in axes[-1, :]:
        ax.set_xlabel("image patch full width (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("controlled beta")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Patch-radius sensitivity of controlled spatial-frequency slopes", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_alignment_by_coherence_radius(
    alignment_bins: pd.DataFrame,
    alignment_slopes: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, (ax_curve, ax_effect) = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.8),
        gridspec_kw={"width_ratios": [2.25, 1.0]},
    )
    radii = np.asarray(sorted(alignment_bins["patch_radius_deg"].unique()), dtype=float)
    norm = colors.Normalize(vmin=float(radii.min()), vmax=float(radii.max()))
    cmap = plt.get_cmap(ALIGNMENT_CMAP)
    for radius, sub in alignment_bins.groupby("patch_radius_deg"):
        work = sub.sort_values("coherence_bin_center")
        x = work["coherence_bin_center"].to_numpy(dtype=float)
        y = work["mean_edge_alignment_index"].to_numpy(dtype=float)
        lo = work["ci95_low"].to_numpy(dtype=float)
        hi = work["ci95_high"].to_numpy(dtype=float)
        ax_curve.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            marker="o",
            markersize=3.2,
            lw=1.45,
            capsize=1.6,
            alpha=0.95,
            color=cmap(norm(float(radius))),
        )
    ax_curve.axhline(0.0, color="0.25", lw=0.8, alpha=0.7)
    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_xlabel("local edge coherence")
    ax_curve.set_ylabel("edge-following alignment")
    ax_curve.set_title("Alignment by coherence and patch radius", fontsize=11)
    ax_curve.grid(axis="y", color="0.88", lw=0.8)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_curve, pad=0.012)
    cbar.set_label("patch radius (deg)")

    slopes = alignment_slopes.sort_values("patch_radius_deg")
    x = slopes["patch_radius_deg"].to_numpy(dtype=float)
    y = slopes["slope_alignment_per_coherence"].to_numpy(dtype=float)
    lo = slopes["ci95_low"].to_numpy(dtype=float)
    hi = slopes["ci95_high"].to_numpy(dtype=float)
    ax_effect.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        color="0.35",
        lw=1.2,
        capsize=2.0,
        zorder=2,
    )
    ax_effect.scatter(x, y, c=x, cmap=cmap, norm=norm, s=28, edgecolor="white", linewidth=0.5, zorder=3)
    ax_effect.axhline(0.0, color="0.25", lw=0.8, alpha=0.7)
    ax_effect.set_xlim(0.15, 3.1)
    ax_effect.set_xlabel("patch radius (deg)")
    ax_effect.set_ylabel("alignment/coherence slope")
    ax_effect.set_title(f"Slope for coherence > {SLOPE_COHERENCE_MIN:g}", fontsize=11)
    ax_effect.grid(axis="y", color="0.88", lw=0.8)

    fig.suptitle("Patch-radius sensitivity of edge-following alignment", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    window_counts: pd.DataFrame,
    local_key: pd.DataFrame,
    sf_key: pd.DataFrame,
    alignment_slopes: pd.DataFrame,
    alignment_counts: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# BackImage patch-radius sensitivity",
        "",
        "Patch radius is the half-width passed to the image-feature extractor. Full patch width is `2 * radius`.",
        "",
        "## Full-feature window counts",
        "",
        "These are the full image-feature/regression tables used by the local-feature and SF summaries.",
        "",
        "| radius deg | full width deg | valid windows | contamination excluded | feature failures |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in window_counts.to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['patch_full_width_deg']:.2f} | "
            f"{int(row['n_windows'])} | {int(row['n_excluded_patch_contamination_windows'])} | "
            f"{int(row['n_failed_image_feature_windows'])} |"
        )

    if not alignment_counts.empty:
        lines.extend(
            [
                "",
                "## Alignment sweep window counts",
                "",
                "These are the lightweight Sobel/structure-tensor windows used by the expanded radius plot.",
                "",
                "| radius deg | full width deg | valid windows | contamination excluded | feature failures |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in alignment_counts.sort_values("patch_radius_deg").to_dict("records"):
            lines.append(
                f"| {row['patch_radius_deg']:.2f} | {row['patch_full_width_deg']:.2f} | "
                f"{int(row['n_windows'])} | {int(row['n_excluded_patch_contamination_windows'])} | "
                f"{int(row['n_failed_image_feature_windows'])} |"
            )

    lines.extend(
        [
            "",
            "## Alignment slope by radius",
            "",
            f"Slope is the window-level OLS fit of alignment against coherence using coherence > {SLOPE_COHERENCE_MIN:g}.",
            "",
            "| radius deg | full width deg | slope | CI | R2 | n |",
            "|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in alignment_slopes.sort_values("patch_radius_deg").to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['patch_full_width_deg']:.2f} | "
            f"{row['slope_alignment_per_coherence']:.4g} | "
            f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] | "
            f"{row['r_squared']:.4g} | {int(row['n_windows'])} |"
        )

    lines.extend(
        [
            "",
            "## Key local-feature pole effects",
            "",
            "| radius | feature | eye metric | delta | CI |",
            "|---:|---|---|---:|---|",
        ]
    )
    focus_local = local_key[
        local_key["eye_metric"].isin(
            ["drift_edge_cos2", "rms_across_arcmin", "rms_delta_along_minus_across_arcmin"]
        )
    ].copy()
    for row in focus_local.sort_values(["feature", "eye_metric", "patch_radius_deg"]).to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['feature_label']} | {row['eye_metric_label']} | "
            f"{row['median_delta']:.4g} | [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] |"
        )

    lines.extend(
        [
            "",
            "## Controlled SF slopes",
            "",
            "| radius | band | eye metric | beta | CI |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in sf_key.sort_values(["feature", "eye_metric", "patch_radius_deg"]).to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['feature_label']} | {row['eye_metric_label']} | "
            f"{row['controlled_beta_z_median']:.4g} | [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    window_counts = _load_window_counts()
    local = _load_local(root)
    sf = _load_sf(root)
    alignment_bins, alignment_slopes, alignment_counts = _load_alignment_summaries(root)

    local_key = local[
        local.apply(lambda r: (r["feature"], r["eye_metric"]) in LOCAL_KEY_ROWS, axis=1)
    ].copy()
    sf_key = sf[sf["feature"].isin(SF_KEY_FEATURES) & sf["eye_metric"].isin(SF_KEY_METRICS)].copy()

    window_counts.to_csv(root / "patch_radius_window_counts.csv", index=False)
    local_key.to_csv(root / "patch_radius_key_local_feature_effects.csv", index=False)
    sf_key.to_csv(root / "patch_radius_key_sf_controlled_slopes.csv", index=False)
    alignment_bins.to_csv(root / "patch_radius_alignment_by_coherence_bins.csv", index=False)
    alignment_slopes.to_csv(root / "patch_radius_alignment_slope_coherence_gt0p3.csv", index=False)
    if not alignment_counts.empty:
        alignment_counts.to_csv(root / "patch_radius_alignment_sweep_window_counts.csv", index=False)

    plot_local_key_effects(local_key, root / "patch_radius_key_local_feature_effects")
    plot_sf_key_slopes(sf_key, root / "patch_radius_key_sf_controlled_slopes")
    plot_alignment_by_coherence_radius(
        alignment_bins,
        alignment_slopes,
        root / "patch_radius_alignment_by_coherence",
    )
    write_report(window_counts, local_key, sf_key, alignment_slopes, alignment_counts, root / "summary_report.md")
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": asdict(PatchRadiusSummaryConfig(root=str(root), alignment_radii_deg=ALIGNMENT_RADII)),
                "alignment_sweep_note": (
                    "Expanded radius curves use the same Sobel/structure-tensor orientation definition, "
                    "batched by computing full-image gradient fields once per trial and averaging gradient "
                    "products over each gaze-centered patch."
                ),
                "alignment_slope_note": (
                    f"Patch-radius summary slopes are window-level OLS fits of drift_edge_cos2 versus "
                    f"image_orientation_coherence after filtering to coherence > {SLOPE_COHERENCE_MIN:g}."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote BackImage patch-radius sensitivity summary to {root}")
    return root


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
