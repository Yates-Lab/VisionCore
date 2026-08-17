#!/usr/bin/env python3
"""Build the M66 Figure-4 retinal-motion SSI panel from the full trace bank.

The comparison changes only the retinal trajectory: measured eye motion versus
the image-matched zero-motion replay.  Behavior is zero in both conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "outputs/dekel240_paper/m66_final_snapshot/fig4_trace_bank_merged"
AUDIT = ROOT / "outputs/dekel240_paper/m66_ssi_retinal_motion_audit"
OUT = ROOT / "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending"

ALL_COLOR = "#4D4D4D"
LOW_COLOR = "#0072B2"
HIGH_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--model-label", default="M66")
    parser.add_argument("--output-stem", default="figure4_panel_d_m66_eye_motion_ssi")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def population_ssi(ssi: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.sum(expected, dtype=np.float64))
    return float(np.sum(ssi * expected, dtype=np.float64) / denominator)


def load_image_values(matrix: Path) -> pd.DataFrame:
    units = pd.read_csv(matrix / "unit_feature_table.csv")
    images = pd.read_csv(matrix / "image_feature_table.csv")
    traces = pd.read_csv(matrix / "trace_feature_table.csv")
    n_images, n_traces, n_units = len(images), len(traces), len(units)
    moving_ssi = np.load(matrix / "ssi_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_expected = np.load(
        matrix / "expected_spikes_matrix.npy", mmap_mode="r"
    ).reshape(n_images, n_traces, n_units)
    stable_ssi = np.load(matrix / "stabilized_ssi_by_image.npy", mmap_mode="r")
    stable_expected = np.load(
        matrix / "stabilized_expected_spikes_by_image.npy", mmap_mode="r"
    )

    sf = pd.to_numeric(units["sf_split_metric"], errors="coerce").to_numpy(float)
    masks = {
        "all": np.ones(n_units, dtype=bool),
        "lower_sf": np.isfinite(sf) & (sf < 0.5),
        "higher_sf": np.isfinite(sf) & (sf >= 0.5),
    }
    rows: list[dict[str, float | int | str]] = []
    for group, mask in masks.items():
        for image_index in range(n_images):
            moving = population_ssi(
                moving_ssi[image_index, :, mask], moving_expected[image_index, :, mask]
            )
            stable = population_ssi(
                stable_ssi[image_index, mask], stable_expected[image_index, mask]
            )
            rows.append(
                {
                    "group": group,
                    "image_index": image_index,
                    "image_session": str(images.iloc[image_index]["session"]),
                    "moving_ssi_bits_per_spike": moving,
                    "stabilized_ssi_bits_per_spike": stable,
                    "delta_bits_per_spike": moving - stable,
                    "ssi_percent_vs_stabilized": 100.0 * (moving - stable) / stable,
                }
            )
    return pd.DataFrame(rows)


def build_panel(
    values: pd.DataFrame,
    summary: dict[str, object],
    *,
    out_dir: Path,
    model_label: str,
    output_stem: str,
) -> None:
    configure()
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(6.7, 2.75),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.42},
    )

    all_images = values.loc[values.group.eq("all")]
    x = all_images.stabilized_ssi_bits_per_spike.to_numpy(float)
    y = all_images.moving_ssi_bits_per_spike.to_numpy(float)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    margin = 0.05 * (hi - lo)
    limits = (lo - margin, hi + margin)
    left.plot(limits, limits, color="#888888", linewidth=0.8, zorder=0)
    left.scatter(
        x,
        y,
        s=13,
        color=ALL_COLOR,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    all_summary = summary["groups"]["all"]
    left.scatter(
        [all_summary["stabilized_ssi_bits_per_spike"]],
        [all_summary["moving_ssi_bits_per_spike"]],
        marker="D",
        s=42,
        color="#CC3311",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    left.set(xlim=limits, ylim=limits)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("stabilized SSI (bits/spike)")
    left.set_ylabel("measured-motion SSI (bits/spike)")
    left.set_title("Paired natural images", loc="left", pad=5)
    left.text(
        0.04,
        0.95,
        (
            f"{int((all_images.ssi_percent_vs_stabilized > 0).sum())}/"
            f"{len(all_images)} images increased\n"
            f"pooled: {all_summary['percent_vs_stabilized']:+.1f}%"
        ),
        transform=left.transAxes,
        ha="left",
        va="top",
    )

    selected_twin_groups = str(
        summary.get("group_definition", {}).get("lower_sf", "")
    ).startswith("selected-twin")
    groups = (
        ("all", "all units", ALL_COLOR),
        (
            "lower_sf",
            "low-SF\ncensored" if selected_twin_groups else "historical\nlower SF",
            LOW_COLOR,
        ),
        (
            "higher_sf",
            "resolved\nSF peak" if selected_twin_groups else "historical\nhigher SF",
            HIGH_COLOR,
        ),
    )
    rng = np.random.default_rng(20260816)
    for index, (group, label, color) in enumerate(groups):
        frame = values.loc[values.group.eq(group)]
        effect = frame.ssi_percent_vs_stabilized.to_numpy(float)
        jitter = rng.uniform(-0.16, 0.16, len(effect))
        right.scatter(
            np.full(len(effect), index) + jitter,
            effect,
            s=9,
            color=color,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        group_summary = summary["groups"][group]
        center = float(group_summary["percent_vs_stabilized"])
        ci_low, ci_high = group_summary["percent_ci95_image_bootstrap"]
        right.errorbar(
            index,
            center,
            yerr=[[center - ci_low], [ci_high - center]],
            fmt="o",
            markersize=5.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.4,
            capsize=2.5,
            zorder=3,
        )
        right.text(
            index,
            ci_high + 1.25,
            f"+{center:.1f}%",
            ha="center",
            va="bottom",
            color="#333333",
        )
    right.axhline(0.0, color="#888888", linewidth=0.8, zorder=0)
    right.set_xticks(range(len(groups)), [entry[1] for entry in groups])
    right.set_ylabel("SSI change from stabilization (%)")
    right.set_title("Population effect", loc="left", pad=5)
    right.text(
        1.0,
        0.01,
        "points: images; bars: 95% image bootstrap",
        transform=right.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )

    fig.text(0.01, 0.98, "D", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.suptitle(
        f"Measured retinal eye motion changes {model_label} spatial selectivity",
        x=0.085,
        y=0.985,
        ha="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.11, right=0.98)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            out_dir / f"{output_stem}.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    matrix = args.matrix_dir.resolve()
    audit = args.audit_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    values = load_image_values(matrix)
    with (audit / "summary.json").open() as stream:
        summary = json.load(stream)
    values.to_csv(out / f"{args.output_stem}_image_values.csv", index=False)
    build_panel(
        values,
        summary,
        out_dir=out,
        model_label=str(args.model_label),
        output_stem=str(args.output_stem),
    )
    provenance = {
        "analysis": "M66 Figure 4 measured retinal motion versus matched stabilization",
        "causal_contract": summary["causal_contract"],
        "estimand": "expected-spike-weighted population SSI",
        "uncertainty": summary["bootstrap_unit"],
        "group_summary": summary["groups"],
        "population_labels": (
            "primary all-unit result is checkpoint-native; lower/higher-SF subgroup labels "
            "are the historical RR100 labels retained in the trace bank"
        ),
        "sources": {
            "audit_summary": {
                "path": str(audit / "summary.json"),
                "sha256": sha256(audit / "summary.json"),
            },
            "trace_bank_baseline": {
                "path": str(matrix / "stabilized_baseline_summary.json"),
                "sha256": sha256(matrix / "stabilized_baseline_summary.json"),
            },
        },
    }
    (out / f"{args.output_stem}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
