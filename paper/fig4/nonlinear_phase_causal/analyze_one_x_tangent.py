#!/usr/bin/env python3
"""Compare the 0x-forward and 1x-backward tangent predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.nonlinear_phase_causal.common import OUT, json_ready
from paper.fig4.nonlinear_phase_causal.analyze_and_plot import group_contributions
from paper.fig4.mechanism_audit_v1.correction.common import write_json


EXACT = OUT / "exact_arrays/exact_nonlinear_phase_metrics.npz"
ONE_X = OUT / "one_x_tangent/one_x_endpoint_tangent_metrics.npz"
RESULT_OUT = OUT / "one_x_tangent"
EPS = 1e-12
GROUPS = (("lower SF", "low"), ("higher SF", "high"))
METHODS = (
    ("actual secant", "#202020"),
    ("0× tangent, forward", "#0077b6"),
    ("1× tangent, backward", "#7b3294"),
)


def read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metric_samples(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    image_draw: np.ndarray,
    trace_draw: np.ndarray,
) -> tuple[float, np.ndarray]:
    point = float(numerator.sum() / max(float(denominator.sum()), EPS))
    num = numerator[image_draw[:, :, None], trace_draw[:, None, :]].sum(axis=(1, 2))
    den = denominator[image_draw[:, :, None], trace_draw[:, None, :]].sum(axis=(1, 2))
    return point, num / np.maximum(den, EPS)


def contrast_samples(
    terms: list[tuple[float, np.ndarray, np.ndarray]],
    *,
    image_draw: np.ndarray,
    trace_draw: np.ndarray,
) -> tuple[float, np.ndarray]:
    point = 0.0
    samples = np.zeros(len(image_draw), dtype=np.float64)
    for coefficient, numerator, denominator in terms:
        term_point, term_samples = metric_samples(
            numerator,
            denominator,
            image_draw=image_draw,
            trace_draw=trace_draw,
        )
        point += coefficient * term_point
        samples += coefficient * term_samples
    return float(point), samples


def interval(point: float, samples: np.ndarray) -> tuple[float, float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(point), float(low), float(high)


def analyze(exact: dict[str, np.ndarray], one: dict[str, np.ndarray], draws: int = 5000) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not np.array_equal(exact["selected_image_index"], one["selected_image_index"]):
        raise ValueError("Image selections differ")
    if not np.array_equal(exact["selected_trace_index"], one["selected_trace_index"]):
        raise ValueError("Trajectory selections differ")
    exact_scales = exact["scales"].astype(float)
    one_scales = one["scales"].astype(float)
    exact_conditions = exact["conditions"].astype(str).tolist()
    s0 = int(np.flatnonzero(np.isclose(exact_scales, 0.0))[0])
    s1 = int(np.flatnonzero(np.isclose(exact_scales, 1.0))[0])
    o0 = int(np.flatnonzero(np.isclose(one_scales, 0.0))[0])
    o1 = int(np.flatnonzero(np.isclose(one_scales, 1.0))[0])
    stable_condition = exact_conditions.index("stable")
    full_condition = exact_conditions.index("full")
    tangent0_condition = exact_conditions.index("tangent")

    rng = np.random.default_rng(20260813)
    image_draw = rng.integers(0, len(exact["selected_image_index"]), size=(draws, len(exact["selected_image_index"])))
    trace_draw = rng.integers(0, len(exact["selected_trace_index"]), size=(draws, len(exact["selected_trace_index"])))
    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {
        "one_x_anchor_ssi_max_abs_difference_from_full": float(
            np.max(
                np.abs(
                    one["ssi"][:, :, o1].astype(float)
                    - exact["ssi"][:, :, s1, full_condition].astype(float)
                )
            )
        ),
        "one_x_anchor_expected_spikes_max_abs_difference_from_full": float(
            np.max(
                np.abs(
                    one["expected_spikes"][:, :, o1].astype(float)
                    - exact["expected_spikes"][:, :, s1, full_condition].astype(float)
                )
            )
        ),
        "anchor_replicate_max_abs_error": float(np.max(one["anchor_replicate_max_abs_error"])),
        "zero_direction_max_abs_error": float(np.max(one["zero_direction_max_abs_error"])),
    }

    for group_ordinal, (group, key) in enumerate(GROUPS):
        units = exact[f"{key}_unit_indices"].astype(int)
        exact_num, exact_den = group_contributions(
            exact["ssi"].astype(float), exact["expected_spikes"].astype(float), units
        )
        one_num, one_den = group_contributions(
            one["ssi"].astype(float), one["expected_spikes"].astype(float), units
        )
        stable = (exact_num[:, :, s0, stable_condition], exact_den[:, :, s0, stable_condition])
        full1 = (exact_num[:, :, s1, full_condition], exact_den[:, :, s1, full_condition])
        tangent0_at1 = (
            exact_num[:, :, s1, tangent0_condition],
            exact_den[:, :, s1, tangent0_condition],
        )
        tangent1_at0 = (one_num[:, :, o0], one_den[:, :, o0])
        method_terms = {
            "actual secant": [(1.0, *full1), (-1.0, *stable)],
            "0× tangent, forward": [(1.0, *tangent0_at1), (-1.0, *stable)],
            "1× tangent, backward": [(1.0, *full1), (-1.0, *tangent1_at0)],
        }
        baseline_point, baseline_samples = metric_samples(
            *stable, image_draw=image_draw, trace_draw=trace_draw
        )
        actual_point, actual_samples = contrast_samples(
            method_terms["actual secant"], image_draw=image_draw, trace_draw=trace_draw
        )
        for method, _ in METHODS:
            point, samples = contrast_samples(
                method_terms[method], image_draw=image_draw, trace_draw=trace_draw
            )
            percent_point = 100.0 * point / max(abs(baseline_point), EPS)
            percent_samples = 100.0 * samples / np.maximum(np.abs(baseline_samples), EPS)
            ratio_point = point / (actual_point if abs(actual_point) > EPS else EPS)
            ratio_samples = samples / np.where(
                np.abs(actual_samples) > EPS,
                actual_samples,
                np.where(actual_samples >= 0, EPS, -EPS),
            )
            bits = interval(point, samples)
            percent = interval(percent_point, percent_samples)
            ratio = interval(ratio_point, ratio_samples)
            rows.append(
                {
                    "sf_group": group,
                    "method": method,
                    "delta_ssi_bits": bits[0],
                    "bits_ci95_low": bits[1],
                    "bits_ci95_high": bits[2],
                    "percent_of_stable_ssi": percent[0],
                    "percent_ci95_low": percent[1],
                    "percent_ci95_high": percent[2],
                    "endpoint_prediction_ratio_to_actual_secant": ratio[0],
                    "ratio_ci95_low": ratio[1],
                    "ratio_ci95_high": ratio[2],
                }
            )
    validation["pass"] = bool(
        validation["one_x_anchor_ssi_max_abs_difference_from_full"] < 2e-3
        and validation["one_x_anchor_expected_spikes_max_abs_difference_from_full"] < 2e-3
        and validation["anchor_replicate_max_abs_error"] < 2e-3
        and validation["zero_direction_max_abs_error"] < 2e-3
    )
    return pd.DataFrame(rows), validation


def render(result: pd.DataFrame, output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), constrained_layout=True)
    lookup = result.set_index(["sf_group", "method"])
    for group_index, (group, _) in enumerate(GROUPS):
        ax = axes[group_index]
        for method, color in METHODS:
            row = lookup.loc[(group, method)]
            if method == "actual secant":
                xs = [0.0, 1.0]
                ys = [0.0, row.percent_of_stable_ssi]
            elif method == "0× tangent, forward":
                xs = [0.0, 1.0]
                ys = [0.0, row.percent_of_stable_ssi]
            else:
                # Backward tangent is anchored to the intact value at 1x.
                actual = lookup.loc[(group, "actual secant")].percent_of_stable_ssi
                xs = [0.0, 1.0]
                ys = [actual - row.percent_of_stable_ssi, actual]
            ax.plot(xs, ys, marker="o", lw=1.7, ms=4, color=color, label=method)
        ax.axhline(0, color="0.55", lw=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 1], ["stabilized\n0×", "measured FEM\n1×"])
        ax.set_ylabel("SSI change (% stabilized)")
        ax.set_title(group)
        if group_index == 0:
            ax.legend(frameon=False, fontsize=6.1, loc="upper left")

    ax = axes[2]
    x = np.arange(len(GROUPS), dtype=float)
    width = 0.22
    for method_index, (method, color) in enumerate(METHODS):
        values = []
        errors = [[], []]
        for group, _ in GROUPS:
            row = lookup.loc[(group, method)]
            values.append(row.endpoint_prediction_ratio_to_actual_secant)
            errors[0].append(row.endpoint_prediction_ratio_to_actual_secant - row.ratio_ci95_low)
            errors[1].append(row.ratio_ci95_high - row.endpoint_prediction_ratio_to_actual_secant)
        ax.bar(x + (method_index - 1) * width, values, width, color=color, label=method)
        ax.errorbar(
            x + (method_index - 1) * width,
            values,
            yerr=np.asarray(errors),
            fmt="none",
            color="0.15",
            lw=0.8,
            capsize=2,
        )
    ax.axhline(1, color="0.45", lw=0.8, ls=":")
    ax.set_xticks(x, ["lower SF", "higher SF"])
    ax.set_ylabel("predicted / actual 0→1× SSI change")
    ax.set_title("Tangent endpoint prediction vs. secant")
    fig.suptitle(
        "Symmetric local-linear test along the measured FEM direction",
        fontsize=10,
        fontweight="bold",
    )
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(output_dir / f"figureS_symmetric_zero_one_tangents.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(result: pd.DataFrame, validation: dict[str, Any], output_dir: Path) -> None:
    lookup = result.set_index(["sf_group", "method"])
    lines = [
        "# Symmetric 0×/1× tangent test",
        "",
        "The stabilized tangent uses the Jacobian at 0× FEM and predicts forward to 1×. The 1× tangent uses the Jacobian at the measured-motion movie and predicts backward to 0×. The latter is exactly equal to the intact twin at 1× by construction, so only its off-anchor prediction is informative.",
        "",
        "For each population, the actual 0→1× SSI change is compared with the off-anchor prediction of each endpoint tangent model. A ratio of one matches the secant; values above one overpredict the intact change, and negative values predict the opposite sign. Because the common softplus and nonlinear SSI calculation follow the preactivation tangent, these ratios are finite endpoint predictions, not literal derivatives of SSI.",
        "",
        "## Results",
        "",
    ]
    for group, _ in GROUPS:
        actual = lookup.loc[(group, "actual secant")]
        forward = lookup.loc[(group, "0× tangent, forward")]
        backward = lookup.loc[(group, "1× tangent, backward")]
        lines.extend(
            [
                f"- **{group}:** actual change {actual.percent_of_stable_ssi:+.2f}%; 0× forward tangent {forward.percent_of_stable_ssi:+.2f}% (ratio {forward.endpoint_prediction_ratio_to_actual_secant:.2f}); 1× backward tangent {backward.percent_of_stable_ssi:+.2f}% (ratio {backward.endpoint_prediction_ratio_to_actual_secant:.2f}).",
            ]
        )
    lines.extend(
        [
            "",
            "The two endpoint tangents should not be averaged into a new model. Their disagreement measures curvature: it shows how much the fitted twin's local gain changes between stabilization and measured FEM motion.",
            "",
            f"Numerical anchor validation passed: **{validation['pass']}**.",
        ]
    )
    (output_dir / "SYMMETRIC_TANGENT_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    result, validation = analyze(read(EXACT), read(ONE_X))
    result.to_csv(RESULT_OUT / "symmetric_tangent_endpoint_statistics.csv", index=False)
    write_json(RESULT_OUT / "symmetric_tangent_validation.json", json_ready(validation))
    render(result, RESULT_OUT)
    write_report(result, validation, RESULT_OUT)
    print(json.dumps({"results": result.to_dict(orient="records"), "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
