#!/usr/bin/env python3
"""Show why local joint SF/TF peaks replace global Gaussian centers for M77."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.audit_output_unit_tuning_fits import (
    complete_surface,
    orientation_score,
)
from paper.fig4.spatiotemporal_tuning.build_direct_rendered_unit_gallery import (
    fitted_tf_slice,
    interpolate_spatial_slice,
    local_quadratic_tf_slice,
)
from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    EPS,
    fit_local_quadratic_peak,
    fit_log_gaussian_surface,
)


BASE = ROOT / "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grouped-csv", type=Path, default=BASE / "frequency_tuning_grouped.csv"
    )
    parser.add_argument(
        "--audit-csv", type=Path, default=BASE / "fit_audit/m77_tuning_fit_audit.csv"
    )
    parser.add_argument("--units", type=int, nargs="+", default=[88, 25])
    parser.add_argument("--out-dir", type=Path, default=BASE / "fit_audit")
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    selected = np.clip(np.asarray(values, dtype=float), 0.0, None)
    return selected / max(float(np.nanmax(selected)), EPS)


def render(args: argparse.Namespace) -> tuple[Path, list[dict]]:
    grouped = pd.read_csv(args.grouped_csv)
    audit = pd.read_csv(args.audit_csv).set_index("unit_index")
    figure, axes = plt.subplots(
        len(args.units),
        2,
        figsize=(9.8, 3.0 * len(args.units)),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
        constrained_layout=True,
    )
    if len(args.units) == 1:
        axes = axes[None]
    records: list[dict] = []

    for row, unit_index in enumerate(args.units):
        unit = grouped.loc[
            grouped.unit_index.eq(unit_index) & grouped.temporal_hz.gt(0)
        ]
        scores = orientation_score(unit, "response_amp_rms")
        orientation = float(scores.idxmax())
        selected = unit.loc[np.isclose(unit.probe_orientation_deg, orientation)]
        sf, tf, response = complete_surface(selected, "response_amp_rms")
        local = fit_local_quadratic_peak(sf, tf, response)
        global_fit = fit_log_gaussian_surface(sf, tf, response)
        check = audit.loc[unit_index]
        if local.get("peak_status") != "ok":
            raise RuntimeError(f"u{unit_index:03d}: local peak is {local.get('peak_status')}")

        axis = axes[row, 0]
        axis.contourf(
            sf,
            tf,
            normalize(response),
            levels=np.linspace(0, 1, 11),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        sample_sf, sample_tf = np.meshgrid(sf, tf)
        axis.scatter(
            sample_sf,
            sample_tf,
            s=7,
            facecolors="none",
            edgecolors="white",
            linewidths=0.35,
        )
        axis.scatter(
            local["preferred_sf_cpd"],
            local["preferred_tf_hz"],
            marker="*",
            s=90,
            color="#F28E2B",
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
            label="local joint peak",
        )
        axis.scatter(
            global_fit["preferred_sf_cpd"],
            global_fit["preferred_tf_hz"],
            marker="x",
            s=35,
            color="white",
            linewidth=1.0,
            zorder=4,
            label="old global center",
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlim=(sf.min(), sf.max()),
            ylim=(tf.min(), tf.max()),
            xlabel="spatial frequency (cycles/deg)",
            ylabel="temporal frequency (Hz)",
            title=(
                f"u{unit_index:03d}: measured RMS surface ({orientation:g}°)\n"
                f"local peak {local['preferred_sf_cpd']:.2f} c/deg, "
                f"{local['preferred_tf_hz']:.2f} Hz"
            ),
        )
        axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        axis.set_yticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
        axis.legend(frameon=False, fontsize=7, loc="upper right")

        axis = axes[row, 1]
        local_slice = normalize(
            interpolate_spatial_slice(response, sf, float(local["preferred_sf_cpd"]))
        )
        dense_tf = np.geomspace(tf.min(), tf.max(), 500)
        guide = np.clip(
            PchipInterpolator(np.log2(tf), local_slice)(np.log2(dense_tf)),
            0.0,
            None,
        )
        old_curve = normalize(fitted_tf_slice(pd.Series(global_fit), dense_tf))
        peak_row = int(local["discrete_peak_tf_index"])
        local_tf = np.geomspace(tf[peak_row - 1], tf[peak_row + 1], 160)
        local_curve = normalize(local_quadratic_tf_slice(local, local_tf))
        axis.plot(
            dense_tf,
            old_curve,
            color="0.55",
            linestyle="--",
            lw=1.2,
            label="old global Gaussian",
        )
        axis.plot(
            dense_tf,
            guide,
            color="#4C78A8",
            lw=0.9,
            alpha=0.55,
            label="shape-preserving sample guide",
        )
        axis.plot(
            local_tf,
            local_curve,
            color="#2166AC",
            lw=2.0,
            label="local quadratic used for peak",
        )
        axis.plot(tf, local_slice, "o", ms=4.0, color="#2166AC", label="measured samples")
        axis.axvspan(
            float(check.rms_jackknife_tf_p10_hz),
            float(check.rms_jackknife_tf_p90_hz),
            color="#2166AC",
            alpha=0.13,
            linewidth=0,
        )
        axis.axvline(float(global_fit["preferred_tf_hz"]), color="0.5", linestyle="--", lw=1.0)
        axis.axvline(float(local["preferred_tf_hz"]), color="#2166AC", lw=1.1)
        axis.axvline(
            float(local["discrete_peak_tf_hz"]),
            color="#F28E2B",
            linestyle=":",
            lw=1.1,
        )
        axis.set(
            xscale="log",
            xlim=(tf.min(), tf.max()),
            ylim=(-0.03, 1.07),
            xlabel="temporal frequency (Hz)",
            ylabel="normalized response at local peak SF",
            title=(
                f"old global center {global_fit['preferred_tf_hz']:.2f} Hz  →  "
                f"local peak {local['preferred_tf_hz']:.2f} Hz\n"
                f"sampled maximum {local['discrete_peak_tf_hz']:.2f} Hz; "
                f"local delete-one 10–90% "
                f"[{check.rms_jackknife_tf_p10_hz:.2f}, {check.rms_jackknife_tf_p90_hz:.2f}]"
            ),
        )
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
        axis.grid(alpha=0.16)
        if row == 0:
            axis.legend(frameon=False, fontsize=6.6, loc="upper left")

        records.append(
            {
                "unit_index": int(unit_index),
                "orientation_deg": orientation,
                "old_global_center_sf_cpd": float(global_fit["preferred_sf_cpd"]),
                "old_global_center_tf_hz": float(global_fit["preferred_tf_hz"]),
                "local_peak_sf_cpd": float(local["preferred_sf_cpd"]),
                "local_peak_tf_hz": float(local["preferred_tf_hz"]),
                "discrete_peak_sf_cpd": float(local["discrete_peak_sf_cpd"]),
                "discrete_peak_tf_hz": float(local["discrete_peak_tf_hz"]),
                "local_fit_r2": float(local["local_fit_r2"]),
                "delete_one_tf_p10_hz": float(check.rms_jackknife_tf_p10_hz),
                "delete_one_tf_p90_hz": float(check.rms_jackknife_tf_p90_hz),
            }
        )

    figure.suptitle(
        "The global surface fit underestimates M77's observed TF peaks",
        fontsize=13,
        fontweight="semibold",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "m77_local_peak_correction_u088_u025.png"
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    pd.DataFrame(records).to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "analysis": "local joint quadratic peak correction",
                "estimator": (
                    "full quadratic in log2(SF),log2(TF) over the 3x3 neighborhood "
                    "of the sampled joint maximum; accept only a concave interior vertex"
                ),
                "global_fit_role": "descriptive broad surface shape only",
                "units": records,
            },
            indent=2,
        )
        + "\n"
    )
    return output, records


def main() -> int:
    output, records = render(parse_args())
    print(output)
    print(pd.DataFrame(records).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
