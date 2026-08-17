#!/usr/bin/env python3
"""Plot an exact paired output-behavior residual evaluation.

The left panel reports exhaustive validation BPS for each residual ablation.
The right panel shows the cell-level paired change from the inherited model,
using the lossless per-unit archives written by
``evaluate_output_behavior_residual.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "inherited": "Inherited\nM16",
    "intact": "Gain +\nadditive",
    "additive_only": "Additive\nonly",
    "gain_only": "Gain\nonly",
    "residual_behavior_zero": "Zeroed\nresidual",
    "residual_behavior_shifted": "Shifted\nresidual",
}

COLORS = {
    "inherited": "#777777",
    "intact": "#1677b8",
    "additive_only": "#e28a22",
    "gain_only": "#4e9a51",
    "residual_behavior_zero": "#a176b2",
    "residual_behavior_shifted": "#c65c5c",
}


def _load_per_unit(path: Path) -> dict[tuple[str, int], float]:
    archive = np.load(path, allow_pickle=False)
    names = archive["session_names"].astype(str)
    result: dict[tuple[str, int], float] = {}
    for idx, name in enumerate(names):
        bps = np.asarray(archive[f"bps_{idx}"], dtype=float)
        cids = np.asarray(archive[f"cids_{idx}"], dtype=int)
        for cid, value in zip(cids, bps):
            result[(str(name), int(cid))] = float(value)
    return result


def _paired_delta(reference, candidate):
    keys = sorted(set(reference).intersection(candidate))
    delta = np.asarray([candidate[key] - reference[key] for key in keys])
    return delta[np.isfinite(delta)]


def render(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text())
    conditions = [name for name in DISPLAY if name in report["conditions"]]
    inherited_bps = float(report["conditions"]["inherited"]["bps_overall"])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    scores = [float(report["conditions"][name]["bps_overall"]) for name in conditions]
    bars = ax.bar(
        np.arange(len(conditions)),
        scores,
        color=[COLORS[name] for name in conditions],
        width=0.72,
    )
    ax.axhline(inherited_bps, color="0.35", lw=1, ls=(0, (3, 2)))
    spread = max(scores) - min(scores)
    pad = max(0.001, spread * 0.5)
    ax.set_ylim(min(scores) - pad, max(scores) + pad * 1.6)
    ax.set_xticks(np.arange(len(conditions)), [DISPLAY[name] for name in conditions], fontsize=8)
    ax.set_ylabel("Exhaustive validation BPS")
    ax.set_title("Same examples, exact ablations", loc="left", fontweight="bold")
    for bar, score in zip(bars, scores):
        delta = score - inherited_bps
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + pad * 0.12,
            f"{score:.4f}\n({delta:+.4f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1]
    archive_paths = report["per_unit_bps_npz"]
    inherited = _load_per_unit(Path(archive_paths["inherited"]))
    shown = [name for name in ("intact", "additive_only", "gain_only") if name in archive_paths]
    for name in shown:
        delta = np.sort(_paired_delta(inherited, _load_per_unit(Path(archive_paths[name]))))
        y = np.arange(1, len(delta) + 1) / len(delta)
        improved = np.mean(delta > 0)
        median = np.median(delta)
        ax.plot(
            delta,
            y,
            lw=2,
            color=COLORS[name],
            label=f"{DISPLAY[name].replace(chr(10), ' ')}: median {median:+.4f}, {improved:.0%} improved",
        )
    ax.axvline(0, color="0.25", lw=1, ls=(0, (3, 2)))
    ax.set_xlabel("Per-cell BPS change from inherited M16")
    ax.set_ylabel("Cumulative fraction of cells")
    ax.set_title("Paired cell-level effect", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", lw=0.7, zorder=0)

    stats = report.get("residual_stats", {})
    fig.suptitle(
        "M20 output behavior residual"
        f"  |  gain SD {stats.get('gain_std', float('nan')):.3f}, "
        f"offset SD {stats.get('offset_std', float('nan')):.3f}",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report_path = args.report.resolve()
    output = args.output or report_path.with_name(f"{report_path.stem}_comparison.png")
    render(report_path, output.resolve())
    print(output.resolve())


if __name__ == "__main__":
    main()
