#!/usr/bin/env python3
"""Build an interpretable multi-unit audit of M77 tuning and retinal power."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_image_specific_joint_engagement import (
    interpolate_tuning_to_fft_bins,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    EPS,
    observed_tuning_tensor,
)
from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    _quadratic_prediction,
    fit_local_quadratic_peak,
)


BASE = ROOT / "outputs/dekel240_paper/m77_epoch279"
DEFAULT_GROUPED = BASE / "periodic_tuning_respaced/frequency_tuning_grouped.csv"
DEFAULT_ROBUST = BASE / "periodic_tuning_respaced/robust/robust_tuning_summary.csv"
DEFAULT_AUDIT = BASE / "periodic_tuning_respaced/fit_audit/m77_tuning_fit_audit.csv"
DEFAULT_ENGAGEMENT = (
    BASE
    / "figure4_real_trace_pilot_corrected/direct_rendered_joint_engagement"
    / "direct_rendered_joint_engagement.npz"
)
DEFAULT_OUT = (
    BASE
    / "figure4_real_trace_pilot_corrected/direct_rendered_joint_engagement"
    / "unit_gallery"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--robust-summary", type=Path, default=DEFAULT_ROBUST)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--engagement-npz", type=Path, default=DEFAULT_ENGAGEMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-units", type=int, default=6)
    return parser.parse_args()


def select_units(audit: pd.DataFrame, n_units: int) -> np.ndarray:
    trusted = audit.loc[audit.audit_category.eq("trusted")].sort_values(
        ["rms_preferred_tf_hz", "unit_index"]
    )
    if len(trusted) < n_units:
        raise ValueError(f"Only {len(trusted)} trusted units; requested {n_units}")
    positions = np.linspace(0, len(trusted) - 1, n_units).round().astype(int)
    return trusted.iloc[positions].unit_index.to_numpy(int)


def interpolate_spatial_slice(
    surface_tf_sf: np.ndarray,
    spatial_cpd: np.ndarray,
    target_sf: float,
) -> np.ndarray:
    value = np.asarray(surface_tf_sf, dtype=np.float64)
    return np.asarray(
        [
            np.interp(np.log2(target_sf), np.log2(spatial_cpd), row)
            for row in value
        ]
    )


def fitted_tf_slice(row: pd.Series, temporal_hz: np.ndarray) -> np.ndarray:
    """Evaluate the saved robust 2-D log-Gaussian at its preferred SF."""
    dy = (
        np.log2(temporal_hz) - np.log2(float(row.preferred_tf_hz))
    ) / float(row.tf_bandwidth_sigma_octaves)
    rho = float(row.sf_tf_log_correlation)
    exponent = -0.5 * dy * dy / max(1.0 - rho * rho, 1e-4)
    return float(row.baseline) + float(row.amplitude) * np.exp(exponent)


def local_quadratic_tf_slice(
    peak: dict,
    temporal_hz: np.ndarray,
) -> np.ndarray:
    """Evaluate the accepted local quadratic at its fitted preferred SF."""
    coefficients = np.asarray(peak["local_quadratic_coefficients"], dtype=float)
    x = np.full_like(
        temporal_hz, float(peak["peak_offset_sf_octaves"]), dtype=float
    )
    y = np.log2(
        np.asarray(temporal_hz, dtype=float)
        / float(peak["discrete_peak_tf_hz"])
    )
    return _quadratic_prediction(coefficients, x, y)


def normalize(values: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    return value / max(float(np.nanmax(value)), EPS)


def render_peak_selection_audit(audit: pd.DataFrame, path: Path) -> dict[str, int]:
    uncensored = audit.loc[
        ~audit.rms_low_tf_censored.astype(bool)
        & ~audit.rms_high_tf_censored.astype(bool)
        & np.isfinite(audit.rms_preferred_tf_hz)
    ].copy()
    trusted = uncensored.loc[uncensored.audit_category.eq("trusted")].copy()
    all_tf = uncensored.rms_preferred_tf_hz.to_numpy(dtype=float)
    trusted_tf = trusted.rms_preferred_tf_hz.to_numpy(dtype=float)
    log_low = math.floor(np.log2(all_tf.min()) * 2.0) / 2.0
    log_high = math.ceil(np.log2(all_tf.max()) * 2.0) / 2.0
    edges = 2.0 ** np.arange(log_low, log_high + 0.51, 0.5)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.72]},
        constrained_layout=True,
    )
    axes[0].hist(all_tf, bins=edges, color="0.72", edgecolor="white", label="all uncensored")
    axes[0].hist(
        trusted_tf,
        bins=edges,
        histtype="step",
        linewidth=2.0,
        color="#2E7D49",
        label="audit-trusted",
    )
    axes[0].set(ylabel="units per half-octave", title="A  Continuous TF peaks before and after the stability audit")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.16)

    rng = np.random.default_rng(29)
    axes[1].scatter(
        all_tf,
        rng.uniform(-0.13, 0.13, len(all_tf)),
        s=15,
        color="0.60",
        alpha=0.42,
        edgecolor="none",
    )
    axes[1].scatter(
        trusted_tf,
        1.0 + rng.uniform(-0.13, 0.13, len(trusted_tf)),
        s=30,
        color="#2E7D49",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].axvspan(4.0, 6.0, color="#E69F00", alpha=0.13, linewidth=0)
    n_all_mid = int(np.count_nonzero((all_tf >= 4.0) & (all_tf < 6.0)))
    n_trusted_mid = int(
        np.count_nonzero((trusted_tf >= 4.0) & (trusted_tf < 6.0))
    )
    axes[1].text(
        math.sqrt(24.0),
        0.5,
        f"4–6 Hz: {n_all_mid}/{len(all_tf)} uncensored, "
        f"{n_trusted_mid}/{len(trusted_tf)} trusted",
        ha="center",
        va="center",
        fontsize=7,
    )
    axes[1].set(
        xscale="log",
        yticks=[0, 1],
        yticklabels=["all", "trusted"],
        xlabel="interpolated RMS peak temporal frequency (Hz)",
        title="B  The apparent low/high split is created by the strict audit subset",
    )
    axes[1].set_xticks([1, 2, 4, 6, 8, 16, 32], ["1", "2", "4", "6", "8", "16", "32"])
    axes[1].grid(axis="x", alpha=0.16)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "n_uncensored": int(len(all_tf)),
        "n_trusted": int(len(trusted_tf)),
        "n_uncensored_4_to_6_hz": n_all_mid,
        "n_trusted_4_to_6_hz": n_trusted_mid,
    }


def render(args: argparse.Namespace) -> tuple[Path, pd.DataFrame]:
    grouped = pd.read_csv(args.grouped_csv)
    units, spatial, probe_tf, orientation, rms_raw, f1_raw = observed_tuning_tensor(
        grouped
    )
    robust = pd.read_csv(args.robust_summary).set_index("unit_index")
    audit = pd.read_csv(args.audit_csv).set_index("unit_index")
    selected = select_units(audit.reset_index(), args.n_units)
    with np.load(args.engagement_npz, allow_pickle=False) as archive:
        method = str(archive["spectrum_method_version"].item())
        if not method.startswith("direct_rendered_movie"):
            raise RuntimeError(f"Expected direct rendered spectrum, found {method}")
        archive_units = np.asarray(archive["unit_indices"], dtype=int)
        archive_spatial = np.asarray(archive["spatial_cpd"], dtype=float)
        fft_tf = np.asarray(archive["temporal_hz"], dtype=float)
        archive_orientation = np.asarray(archive["orientation_deg"], dtype=float)
        image_power = np.asarray(archive["image_rendered_movie_power"], dtype=float)
        joint = np.asarray(archive["joint_alignment_fraction"], dtype=float)
        total_power = np.asarray(archive["total_dynamic_power"], dtype=float)
        moving = np.asarray(archive["moving_rate"], dtype=float)
        stable = np.asarray(archive["stabilized_rate"], dtype=float)
        trace_rows = np.asarray(archive["trace_rows"], dtype=int)
    if not np.array_equal(units, archive_units):
        raise ValueError("Tuning and engagement unit coordinates differ")
    np.testing.assert_allclose(spatial, archive_spatial)
    np.testing.assert_allclose(orientation, archive_orientation)
    rms_interpolated = interpolate_tuning_to_fft_bins(rms_raw, probe_tf, fft_tf)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(14.8, 2.32 * len(selected)),
        gridspec_kw={"width_ratios": [1.0, 0.92, 1.0, 0.92]},
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = axes[None]
    rows: list[dict[str, float | int | str]] = []

    for row_index, unit in enumerate(selected):
        unit_row = int(np.flatnonzero(units == unit)[0])
        global_fit = robust.loc[unit]
        check = audit.loc[unit]
        ori_index = int(
            np.argmin(np.abs(orientation - float(check.rms_best_orientation_deg)))
        )
        rms_surface = normalize(rms_raw[unit_row, :, :, ori_index].T)
        f1_surface = normalize(f1_raw[unit_row, :, :, ori_index].T)
        rms_peak = fit_local_quadratic_peak(
            spatial,
            probe_tf,
            rms_raw[unit_row, :, :, ori_index].T,
        )
        f1_peak = fit_local_quadratic_peak(
            spatial,
            probe_tf,
            f1_raw[unit_row, :, :, ori_index].T,
        )
        if rms_peak.get("peak_status") != "ok" or f1_peak.get("peak_status") != "ok":
            raise RuntimeError(f"Selected trusted unit u{unit:03d} has a rejected local peak")
        best_image = int(np.argmax(joint[:, unit_row]))
        retinal = np.asarray(image_power[best_image, :, :, ori_index], dtype=float).T
        tuning_fine = normalize(rms_interpolated[unit_row, :, :, ori_index]).T
        rate_delta = moving[:, unit_row] - stable[:, unit_row]
        alignment_result = spearmanr(joint[:, unit_row], rate_delta)
        total_result = spearmanr(total_power, rate_delta)

        axis = axes[row_index, 0]
        axis.contourf(
            spatial,
            probe_tf,
            rms_surface,
            levels=np.linspace(0, 1, 11),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        sample_sf, sample_tf = np.meshgrid(spatial, probe_tf)
        axis.scatter(
            sample_sf,
            sample_tf,
            s=5,
            facecolors="none",
            edgecolors="white",
            linewidths=0.3,
        )
        axis.scatter(
            float(check.rms_preferred_sf_cpd),
            float(check.rms_preferred_tf_hz),
            marker="*",
            s=58,
            color="#E56717",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="local peak",
        )
        axis.scatter(
            float(global_fit.preferred_sf_cpd),
            float(global_fit.preferred_tf_hz),
            marker="x",
            s=24,
            color="white",
            linewidth=0.8,
            zorder=4,
            label="old global center",
        )
        axis.set_title(
            f"u{unit:03d}  measured RMS tuning ({orientation[ori_index]:g}°)\n"
            f"local peak {float(check.rms_preferred_sf_cpd):.2f} c/deg, "
            f"{float(check.rms_preferred_tf_hz):.2f} Hz",
            loc="left",
        )
        if row_index == 0:
            axis.legend(frameon=False, fontsize=5.8, loc="upper right")

        axis = axes[row_index, 1]
        rms_slice = normalize(
            interpolate_spatial_slice(
                rms_raw[unit_row, :, :, ori_index].T,
                spatial,
                float(check.rms_preferred_sf_cpd),
            )
        )
        f1_slice = normalize(
            interpolate_spatial_slice(
                f1_raw[unit_row, :, :, ori_index].T,
                spatial,
                float(check.f1_preferred_sf_cpd),
            )
        )
        dense_tf = np.geomspace(probe_tf.min(), probe_tf.max(), 300)
        rms_guide = np.clip(
            PchipInterpolator(np.log2(probe_tf), rms_slice)(np.log2(dense_tf)),
            0.0,
            None,
        )
        local_tf_index = int(rms_peak["discrete_peak_tf_index"])
        local_dense_tf = np.geomspace(
            probe_tf[local_tf_index - 1], probe_tf[local_tf_index + 1], 100
        )
        local_curve = local_quadratic_tf_slice(rms_peak, local_dense_tf)
        local_curve = np.clip(local_curve, 0.0, None)
        local_curve /= max(float(np.max(local_curve)), EPS)
        axis.plot(
            dense_tf,
            rms_guide,
            color="#2166AC",
            lw=0.9,
            alpha=0.55,
            label="shape-preserving guide",
        )
        axis.plot(
            local_dense_tf,
            local_curve,
            color="#2166AC",
            lw=1.8,
            label="local joint quadratic",
        )
        axis.plot(probe_tf, rms_slice, "o", ms=3.0, color="#2166AC", label="RMS samples")
        axis.plot(probe_tf, f1_slice, "^", ms=2.8, color="#D95F02", label="F1 samples")
        axis.axvspan(
            float(check.rms_jackknife_tf_p10_hz),
            float(check.rms_jackknife_tf_p90_hz),
            color="#2166AC",
            alpha=0.13,
            linewidth=0,
        )
        axis.axvline(float(check.rms_preferred_tf_hz), color="#2166AC", lw=1.0)
        axis.axvline(
            float(check.rms_global_center_tf_hz),
            color="0.55",
            lw=0.8,
            linestyle=":",
        )
        axis.axvline(
            float(check.f1_preferred_tf_hz),
            color="#D95F02",
            lw=1.0,
            linestyle="--",
        )
        axis.set(
            xscale="log",
            xlim=(probe_tf.min(), probe_tf.max()),
            ylim=(-0.03, 1.06),
            xlabel="temporal frequency (Hz)",
            ylabel="normalized response",
            title=(
                f"local peak audit: $R^2$={float(check.rms_local_peak_r2):.2f}\n"
                f"RMS {float(check.rms_preferred_tf_hz):.2f}; "
                f"F1 {float(check.f1_preferred_tf_hz):.2f} Hz"
            ),
        )
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        if row_index == 0:
            axis.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper right")
        axis.grid(alpha=0.15)

        axis = axes[row_index, 2]
        positive = retinal[retinal > 0]
        floor = max(float(np.quantile(positive, 0.02)), float(positive.max()) * 1e-5)
        axis.contourf(
            spatial,
            fft_tf,
            np.maximum(retinal, floor),
            levels=np.geomspace(floor, float(retinal.max()), 12),
            norm=LogNorm(vmin=floor, vmax=float(retinal.max())),
            cmap="magma",
        )
        if np.nanmax(tuning_fine) >= 0.5:
            axis.contour(
                spatial,
                fft_tf,
                tuning_fine,
                levels=[0.5],
                colors="white",
                linewidths=1.0,
            )
        axis.axhspan(1.0, fft_tf.min(), color="0.78", alpha=0.55, zorder=4)
        axis.set_title(
            f"image {best_image}: exact rendered-movie power\n"
            "white = unit half-max tuning",
            loc="left",
        )

        axis = axes[row_index, 3]
        axis.axhline(0, color="0.5", lw=0.7)
        axis.scatter(joint[:, unit_row], rate_delta, s=20, color="#0072B2", alpha=0.78)
        for image_index, (x_value, y_value) in enumerate(
            zip(joint[:, unit_row], rate_delta)
        ):
            axis.annotate(
                str(image_index),
                (x_value, y_value),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5.4,
            )
        axis.set(
            title=(
                f"causal rate test: alignment $r_s$={float(alignment_result.statistic):.2f}\n"
                f"total power $r_s$={float(total_result.statistic):.2f}"
            ),
            xlabel="fraction dynamic power in joint tuning",
            ylabel="moving − stabilized mean rate",
        )
        axis.grid(alpha=0.15)

        for map_axis in (axes[row_index, 0], axes[row_index, 2]):
            map_axis.set_xscale("log", base=2)
            map_axis.set_yscale("log", base=2)
            map_axis.set_xlim(spatial.min(), spatial.max())
            map_axis.set_ylim(1.0, fft_tf.max())
            map_axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            map_axis.set_yticks(
                [1, 2, 3, 6, 12, 24, 48, 96],
                ["1", "2", "3", "6", "12", "24", "48", "96"],
            )
            map_axis.set_xlabel("spatial frequency (cycles/deg)")
            map_axis.set_ylabel("temporal frequency (Hz)")

        rows.append(
            {
                "unit_index": int(unit),
                "best_orientation_deg": float(orientation[ori_index]),
                "interpolated_peak_sf_cpd": float(check.rms_preferred_sf_cpd),
                "interpolated_peak_tf_hz": float(check.rms_preferred_tf_hz),
                "old_global_center_tf_hz": float(check.rms_global_center_tf_hz),
                "f1_interpolated_peak_tf_hz": float(check.f1_preferred_tf_hz),
                "jackknife_tf_p10_hz": float(check.rms_jackknife_tf_p10_hz),
                "jackknife_tf_p90_hz": float(check.rms_jackknife_tf_p90_hz),
                "rms_local_peak_r2": float(check.rms_local_peak_r2),
                "global_shape_heldout_r2": float(check.rms_heldout_r2),
                "best_alignment_image": best_image,
                "alignment_rate_delta_spearman": float(alignment_result.statistic),
                "total_power_rate_delta_spearman": float(total_result.statistic),
                "n_retinal_power_traces": int(len(trace_rows)),
                "spectrum_method_version": method,
            }
        )

    figure.suptitle(
        "M77 audit: interpolated tuning peaks, exact retinal-movie power, and causal response",
        fontsize=13,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.005,
        "Units span the audit-trusted TF range. Stars and blue lines are local joint peaks; "
        "gray crosses/lines show the rejected old global centers. The blue band is the "
        "delete-one-local-point 10–90% interval. "
        "Power comes from the rendered movie, not |k·v|.",
        ha="center",
        va="top",
        fontsize=7,
        color="0.35",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "m77_direct_rendered_unit_gallery.png"
    figure.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "m77_direct_rendered_unit_gallery.csv", index=False)
    peak_audit_path = args.out_dir / "m77_tf_peak_selection_audit.png"
    peak_selection = render_peak_selection_audit(audit.reset_index(), peak_audit_path)
    summary = {
        "analysis": "multi-unit direct-rendered retinal-power and interpolated-peak audit",
        "selection": "evenly spaced ranks of local joint TF among audit_category=trusted units",
        "selected_units": selected.tolist(),
        "selected_tf_hz": table.interpolated_peak_tf_hz.tolist(),
        "n_retinal_power_traces": int(len(trace_rows)),
        "peak_selection_audit": peak_selection,
        "peak_selection_figure": str(peak_audit_path.resolve()),
        "spectrum_method_version": method,
        "claim_boundary": (
            "All selected peaks are concave local maxima inside the 3x3 neighborhood and "
            "passed RMS/F1, local delete-one, smoothing, orientation, and boundary checks. "
            "Global-surface held-out R2 is descriptive only. The local peak is an annotation only; "
            "alignment uses the raw periodic response tensor interpolated to resolved FFT bins."
        ),
        "figure": str(output.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output, table


def main() -> int:
    args = parse_args()
    output, table = render(args)
    print(output)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
