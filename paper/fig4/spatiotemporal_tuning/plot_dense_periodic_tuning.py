#!/usr/bin/env python3
"""Render clean RR100 SF–TF tuning fields from a dense periodic probe.

The input measurements remain visible in the saved table.  The publication
surface applies only a small Gaussian blur in the already-log-spaced grid; no
global Gaussian model is used to invent a passband or preferred frequency.
Peak annotations come from the local quadratic estimator around the raw 2-D
maximum.  A multipage PDF covers every unit, while a small PNG/PDF preview can
show explicitly requested units during figure development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    fit_local_quadratic_peak,
)


EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grouped_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("mean_rate", "signed_rate", "response_amp_rms", "f1_amplitude"),
        default="mean_rate",
    )
    parser.add_argument(
        "--display-sigma-bins",
        type=float,
        default=0.65,
        help="Gaussian display smoothing in dense log-grid bins; statistics use raw values.",
    )
    parser.add_argument("--units-per-page", type=int, default=6)
    parser.add_argument(
        "--preview-units",
        type=int,
        nargs="*",
        default=(25, 36, 63, 88),
    )
    return parser.parse_args()


def _dynamic_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "unit_index",
        "probe_orientation_deg",
        "spatial_cpd",
        "temporal_hz",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"tuning table is missing columns: {missing}")
    out = frame.loc[pd.to_numeric(frame.temporal_hz, errors="coerce") > 0].copy()
    if out.empty:
        raise ValueError("tuning table contains no dynamic temporal frequencies")
    return out


def preferred_orientation(unit: pd.DataFrame, metric: str) -> float:
    """Choose the orientation with the largest robust upper-tail response."""
    summary = (
        unit.groupby("probe_orientation_deg", sort=True)[metric]
        .quantile(0.95)
        .sort_values(ascending=False)
    )
    if summary.empty or not np.isfinite(summary.iloc[0]):
        raise ValueError("unit contains no finite orientation responses")
    return float(summary.index[0])


def unit_surface(
    frame: pd.DataFrame,
    unit_index: int,
    metric: str,
) -> dict:
    unit = frame.loc[frame.unit_index.astype(int) == int(unit_index)].copy()
    if unit.empty:
        raise KeyError(f"unit {unit_index} is absent from the tuning table")
    orientation = preferred_orientation(unit, metric)
    selected = unit.loc[np.isclose(unit.probe_orientation_deg, orientation)]
    pivot = selected.pivot_table(
        index="temporal_hz",
        columns="spatial_cpd",
        values=metric,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    spatial = pivot.columns.to_numpy(dtype=float)
    temporal = pivot.index.to_numpy(dtype=float)
    raw = pivot.to_numpy(dtype=float)
    if raw.shape != (len(temporal), len(spatial)) or not np.isfinite(raw).all():
        raise ValueError(
            f"unit {unit_index} orientation {orientation:g} has an incomplete dense grid"
        )
    low, high = float(np.min(raw)), float(np.max(raw))
    signed_metric = metric == "signed_rate"
    if signed_metric:
        normalized = raw / max(float(np.max(np.abs(raw))), EPS)
    else:
        normalized = (raw - low) / max(high - low, EPS)
    peak = fit_local_quadratic_peak(spatial, temporal, raw)
    return {
        "unit_index": int(unit_index),
        "unit_label": str(unit.unit_label.iloc[0]) if "unit_label" in unit else f"u{unit_index:03d}",
        "orientation_deg": orientation,
        "spatial_cpd": spatial,
        "temporal_hz": temporal,
        "raw": raw,
        "normalized": normalized,
        "signed_metric": signed_metric,
        "peak": peak,
    }


def draw_surface(
    axis: plt.Axes,
    payload: dict,
    *,
    sigma: float,
):
    spatial = payload["spatial_cpd"]
    temporal = payload["temporal_hz"]
    display = gaussian_filter(
        payload["normalized"], sigma=max(float(sigma), 0.0), mode="nearest"
    )
    signed_metric = bool(payload.get("signed_metric", False))
    contour = axis.contourf(
        spatial,
        temporal,
        display,
        levels=np.linspace(-1.0, 1.0, 15) if signed_metric else np.linspace(0.0, 1.0, 13),
        cmap="RdBu_r" if signed_metric else "viridis",
        vmin=-1.0 if signed_metric else 0.0,
        vmax=1.0,
        extend="both" if signed_metric else "neither",
    )
    if float(np.min(display)) <= 0.5 <= float(np.max(display)):
        axis.contour(
            spatial,
            temporal,
            display,
            levels=[0.5],
            colors="white",
            linewidths=1.15,
        )
    peak = payload["peak"]
    if peak.get("peak_status") == "ok":
        peak_sf = float(peak["preferred_sf_cpd"])
        peak_tf = float(peak["preferred_tf_hz"])
        axis.scatter(
            peak_sf,
            peak_tf,
            marker="*",
            s=85,
            facecolor="#F28E2B",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        peak_text = f"{peak_sf:.2g} c/deg, {peak_tf:.2g} Hz"
    else:
        peak_text = f"peak {peak.get('peak_status', 'unavailable')}"
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlim(float(spatial[0]), float(spatial[-1]))
    axis.set_ylim(float(temporal[0]), float(temporal[-1]))
    axis.set_xticks([value for value in (0.25, 0.5, 1, 2, 4, 8, 16) if spatial[0] <= value <= spatial[-1]])
    axis.set_yticks([value for value in (1, 2, 4, 8, 16, 32, 64, 96) if temporal[0] <= value <= temporal[-1]])
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.grid(color="white", alpha=0.16, linewidth=0.6)
    axis.set_title(
        f"{payload['unit_label']} · {payload['orientation_deg']:g}°\n{peak_text}",
        fontsize=9.2,
        fontweight="semibold",
    )
    return contour


def render_page(
    payloads: list[dict],
    *,
    metric: str,
    sigma: float,
    title: str,
) -> plt.Figure:
    n_columns = 3 if len(payloads) > 4 else 2
    n_rows = int(np.ceil(len(payloads) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.45 * n_columns, 3.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    contour = None
    for index, axis in enumerate(axes.flat):
        if index >= len(payloads):
            axis.axis("off")
            continue
        contour = draw_surface(axis, payloads[index], sigma=sigma)
        if index % n_columns == 0:
            axis.set_ylabel("temporal frequency (Hz)")
        if index // n_columns == n_rows - 1:
            axis.set_xlabel("spatial frequency (cycles/degree)")
    if contour is not None:
        colorbar = figure.colorbar(contour, ax=list(axes.flat), shrink=0.82, pad=0.018)
        colorbar.set_label(f"normalized {metric.replace('_', ' ')}")
        colorbar.set_ticks([-1.0, 0.0, 1.0] if metric == "signed_rate" else [0.0, 0.5, 1.0])
    figure.suptitle(title, fontsize=13, fontweight="semibold")
    return figure


def main() -> int:
    args = parse_args()
    if args.units_per_page < 1 or args.units_per_page > 12:
        raise ValueError("units-per-page must lie in [1, 12]")
    complete = pd.read_csv(args.grouped_csv)
    if args.metric == "signed_rate":
        static = complete.loc[np.isclose(complete.temporal_hz, 0.0), [
            "unit_index", "spatial_cpd", "probe_orientation_deg", "mean_rate"
        ]].rename(columns={"mean_rate": "static_mean_rate"})
        complete = complete.merge(
            static,
            on=["unit_index", "spatial_cpd", "probe_orientation_deg"],
            how="left",
            validate="many_to_one",
        )
        complete["signed_rate"] = complete["mean_rate"] - complete["static_mean_rate"]
    frame = _dynamic_table(complete)
    if args.metric not in frame:
        raise ValueError(f"metric {args.metric!r} is absent from the tuning table")
    units = np.sort(frame.unit_index.astype(int).unique())
    payloads = [unit_surface(frame, int(unit), args.metric) for unit in units]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    for payload in payloads:
        peak = payload["peak"]
        audit_rows.append(
            {
                "unit_index": int(payload["unit_index"]),
                "unit_label": str(payload["unit_label"]),
                "preferred_orientation_deg": float(payload["orientation_deg"]),
                "peak_status": str(peak.get("peak_status", "unavailable")),
                "peak_censored": bool(peak.get("peak_censored", True)),
                "preferred_sf_cpd": peak.get("preferred_sf_cpd", np.nan),
                "preferred_tf_hz": peak.get("preferred_tf_hz", np.nan),
                "discrete_peak_sf_cpd": peak.get("discrete_peak_sf_cpd", np.nan),
                "discrete_peak_tf_hz": peak.get("discrete_peak_tf_hz", np.nan),
                "local_fit_r2": peak.get("local_fit_r2", np.nan),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit_path = args.out_dir / f"rr100_dense_{args.metric}_peak_audit.csv"
    audit.to_csv(audit_path, index=False)

    pdf_path = args.out_dir / f"rr100_dense_{args.metric}_tuning.pdf"
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(payloads), int(args.units_per_page)):
            figure = render_page(
                payloads[start : start + int(args.units_per_page)],
                metric=args.metric,
                sigma=args.display_sigma_bins,
                title="M77 RR100 dense spatiotemporal tuning",
            )
            pdf.savefig(figure, bbox_inches="tight", facecolor="white")
            plt.close(figure)

    requested = [int(unit) for unit in args.preview_units if int(unit) in set(units)]
    if requested:
        by_unit = {int(item["unit_index"]): item for item in payloads}
        preview = render_page(
            [by_unit[unit] for unit in requested],
            metric=args.metric,
            sigma=args.display_sigma_bins,
            title="M77 RR100 dense spatiotemporal tuning · preview",
        )
        preview_path = args.out_dir / f"rr100_dense_{args.metric}_preview.png"
        preview.savefig(preview_path, dpi=240, bbox_inches="tight", facecolor="white")
        preview.savefig(preview_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(preview)

    summary = {
        "analysis": "dense RR100 SF-TF tuning visualization",
        "source": str(args.grouped_csv.resolve()),
        "metric": args.metric,
        "display_smoothing_sigma_log_grid_bins": float(args.display_sigma_bins),
        "statistics_use_display_smoothing": False,
        "peak_estimator": "local quadratic around raw dense-grid maximum in log2 SF/TF",
        "n_units": int(len(units)),
        "peak_status_counts": {
            str(key): int(value)
            for key, value in audit.peak_status.value_counts().items()
        },
        "n_uncensored_local_peaks": int(np.count_nonzero(~audit.peak_censored)),
        "peak_audit_csv": str(audit_path.resolve()),
        "multipage_pdf": str(pdf_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
