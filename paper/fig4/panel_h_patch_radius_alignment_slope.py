#!/usr/bin/env python3
"""Panel K: patch-radius sensitivity of edge-following alignment.

Standalone rendering of the right panel from
``fixation_statistics_by_stimulus/summarize_backimage_patch_radius_sensitivity.py``:
the OLS slope of edge-following alignment versus local edge coherence, fit
only over coherence > 0.3, as a function of patch radius.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _fig4_panel_header

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

FIG4_DIR = ROOT / "paper" / "fig4"
OUT_DIR = _paths.PANELS_DIR
SLOPE_CSV = _paths.PATCH_RADIUS_ALIGNMENT_SLOPE_CSV
SOURCE_FIGURE = _paths.PATCH_RADIUS_ALIGNMENT_BY_COHERENCE_PDF
SOURCE_SCRIPT = FIG4_DIR / "refresh" / "summarize_backimage_patch_radius_sensitivity.py"

SLOPE_COHERENCE_MIN = 0.3
GRAY = "#6B6F75"
INK = "#111111"
GRID = "#E3E3E3"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_values(csv_path: Path = SLOPE_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    values = pd.read_csv(csv_path)
    required = {
        "patch_radius_deg",
        "coherence_min",
        "slope_alignment_per_coherence",
        "ci95_low",
        "ci95_high",
        "n_windows",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return values.sort_values("patch_radius_deg").reset_index(drop=True)


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "K",
    title: str = "Edge following saturates\nnear foveal scale",
    values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = load_values() if values is None else values.copy()
    values = values.sort_values("patch_radius_deg")

    x = values["patch_radius_deg"].to_numpy(dtype=float)
    y = values["slope_alignment_per_coherence"].to_numpy(dtype=float)
    lo = values["ci95_low"].to_numpy(dtype=float)
    hi = values["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="-",
        color="0.36",
        lw=1.15,
        elinewidth=0.95,
        capsize=0,
        zorder=2,
    )
    ax.scatter(x, y, s=24.0, facecolor="0.42", edgecolor="white", linewidth=0.45, zorder=3)
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":", alpha=0.75)

    ax.set_xlim(0.15, 3.1)
    ax.set_ylim(-0.03, max(0.72, float(np.nanmax(hi)) + 0.03))
    ax.set_xlabel("patch radius (deg)")
    ax.set_ylabel("alignment/coherence slope", labelpad=2.0)
    _fig4_panel_header.draw_bottom_row_header(ax, label, title, title_linespacing=_fig4_panel_header.PANEL_TITLE_LINESPACING, color=INK)
    ax.text(
        0.03,
        0.98,
        f"coherence > {SLOPE_COHERENCE_MIN:g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=GRAY,
    )
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    _fig4_panel_header.align_bottom_row_xlabel(ax)
    ax.spines[["top", "right"]].set_visible(False)
    return values


def load_provenance() -> dict:
    return {
        "panels": ["K"],
        "source_script": _relative(SOURCE_SCRIPT),
        "source_two_panel_figure": _relative(SOURCE_FIGURE),
        "source_slope_csv": _relative(SLOPE_CSV),
        "slope_definition": f"OLS slope of edge-following alignment versus coherence for coherence > {SLOPE_COHERENCE_MIN:g}.",
    }


def build_panel(
    out_dir: Path = OUT_DIR,
    *,
    figsize: tuple[float, float] = (2.55, 3.03),
    label: str = "K",
    title: str = "Edge following saturates\nnear foveal scale",
) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_values()
    values.to_csv(out_dir / "panel_h_patch_radius_alignment_slope_values.csv", index=False)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = _fig4_panel_header.add_bottom_row_axes(fig)
    draw_panel(ax, label=label, title=title, values=values)
    paths = {
        "png": out_dir / "panel_h_patch_radius_alignment_slope.png",
        "pdf": out_dir / "panel_h_patch_radius_alignment_slope.pdf",
        "svg": out_dir / "panel_h_patch_radius_alignment_slope.svg",
    }
    fig.savefig(paths["png"], dpi=220, transparent=True)
    fig.savefig(paths["pdf"], transparent=True)
    fig.savefig(paths["svg"], transparent=True)
    plt.close(fig)
    (out_dir / "panel_h_patch_radius_alignment_slope_provenance.json").write_text(
        json.dumps(load_provenance(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    paths = build_panel()
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
