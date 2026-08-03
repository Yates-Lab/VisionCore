#!/usr/bin/env python3
"""Panel H (displayed letter; module/file keep the "g" name for history):
aligned high-SF units on the RMS-excursion dose axis.

Originally Panel G; the whole figure's G/H/I shifted to H/I/J once a new
panel G (contour-normal/parallel decomposition, reserved/placeholder) was
inserted between F and this panel -- see _fig4_ssi_common.py's
draw_contour_components_panel and EF_INSET_* constants.

Replaces the original unsigned-component-path axis (still available,
unmodified, in _fig4_matched_bins_bracket.py) after the behavior-model
bridge work showed path is the model axis with the least support from real
behavior: the random-rotation null found no trace-contour matching benefit
on path (-0.002 pp, p=0.86) but a robust, coherence-scaling one on RMS
excursion (+0.090 pp, p=0.004 at local edge coherence >= 0.2). See
the behaviour/model bridge notes (upstream; not in this repo) and
_fig4_option_sheet.py (the four-way comparison this choice came out of).

The dose curve reuses the same aligned high-SF population, the same
across/along decomposition, and the same broken-log-axis/bracket visual
grammar as the original path-based panel -- only the x-axis metric changes,
plus two adjustments that turned out to matter once the metric changed:

- The extreme long-tail bin (n~54 image pairs, wide CI) is dropped from the
  view. It stretches the y-axis so far that the informative low-dose bins,
  where nearly all the data actually is, compress to a visually flat line.
  Its own final-bin bracket is dropped with it (see _fig4_option_sheet.py
  for why: the precomputed contrast is tied to that specific bin).
- The zero-anchor/break gap is tightened (zero_gap=0.5 instead of the
  default 1.0). That gap is a fixed fraction of the mapped axis regardless
  of the metric; RMS's real bins cluster far more tightly than path's, so
  the default gap ate a disproportionate share of the panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_option_sheet as options
import _fig4_panel_header

OUT_DIR = _paths.PANELS_DIR
METRIC_FAMILY = "component_rms"
EXCLUDE_LAST_BINS = 1
AXIS_OVERRIDE = {"min_pos": 0.9, "max_pos": 3.3, "ticks": [0, 1, 2, 3], "zero_gap": 0.5}
YLIM_PAD_LOW = 0.08
YLIM_PAD_HIGH = 0.20
FINAL_BRACKET_X_OFFSET = 0.32
DEFAULT_PANEL_LABEL = "H"
DEFAULT_PANEL_TITLE = "High-SF aligned units' information\ndepends on trajectory shape"
# The real ssi_figure_v2 gs[2, 0] cell (MAIN_GRID_KWARGS at FIGURE_SIZE_IN =
# (8.5, 11.0)), not the earlier 2.45in x 2.35in standalone-preview approximation.
FIGSIZE = (2.563, 2.432)


def configure_matplotlib() -> None:
    options.configure_matplotlib()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(options.VALUES_CSV)
    last_bin_contrasts = pd.read_csv(options.LAST_BIN_CONTRASTS_CSV)
    reference = pd.read_csv(options.TRACE_BANK_REFERENCE_CSV)
    populations = pd.read_csv(options.POPULATIONS_CSV)
    return values, last_bin_contrasts, reference, populations


def _panel_ylim(values: pd.DataFrame) -> tuple[float, float]:
    frame = values[values["metric_family"].eq(METRIC_FAMILY) & values["population_key"].eq(options.POPULATION_KEY)]
    if EXCLUDE_LAST_BINS > 0 and not frame.empty:
        keep_through = int(frame["component_bin_order"].max()) - EXCLUDE_LAST_BINS
        frame = frame[frame["component_bin_order"] <= keep_through]
    vals = [0.0]
    for col in [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    return (lo - YLIM_PAD_LOW * span, hi + YLIM_PAD_HIGH * span)


def draw_panel(
    ax: plt.Axes,
    *,
    panel_label: str = DEFAULT_PANEL_LABEL,
    panel_title: str = DEFAULT_PANEL_TITLE,
    **_kwargs: object,
) -> None:
    """Draw Panel G (RMS excursion) onto an existing axes.

    Accepts and ignores extra keyword arguments so it is a drop-in
    replacement for _fig4_matched_bins_bracket.draw_panel(ax) in
    _fig4_ssi_common.py's draw_panel_g_or_fallback.
    """
    values, last_bin_contrasts, reference, populations = _load()
    ylim = _panel_ylim(values)
    dropped_note = options._draw_dose_panel(
        ax,
        metric_family=METRIC_FAMILY,
        values=values,
        reference=reference,
        populations=populations,
        last_bin_contrasts=last_bin_contrasts,
        ylim=ylim,
        exclude_last_bins=EXCLUDE_LAST_BINS,
        axis_override=AXIS_OVERRIDE,
        final_bracket_x_offset=FINAL_BRACKET_X_OFFSET,
    )
    _fig4_panel_header.draw_middle_row_header(
        ax,
        panel_label,
        panel_title,
        title_linespacing=_fig4_panel_header.MIDDLE_ROW_TITLE_LINESPACING,
        color=options.INK,
    )
    ax.set_ylabel("SSI change (%)", labelpad=2.0)
    _fig4_panel_header.align_middle_row_ylabel(ax)
    if dropped_note is not None:
        ax.set_xlabel(f"{ax.get_xlabel()}\n{dropped_note}")
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    _fig4_panel_header.align_middle_row_xlabel(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in item else "along" for item in labels]
        ax.legend(
            handles,
            short_labels,
            frameon=False,
            fontsize=5.9,
            loc="lower left",
            handlelength=1.8,
            borderaxespad=0.2,
        )
    ax.spines[["top", "right"]].set_visible(False)


def load_provenance() -> dict:
    return {
        "panels": [DEFAULT_PANEL_LABEL],
        "metric_family": METRIC_FAMILY,
        "population_key": options.POPULATION_KEY,
        "source_script": _relative(Path(__file__)),
        "source_values_csv": _relative(options.VALUES_CSV),
        "source_last_bin_contrasts_csv": _relative(options.LAST_BIN_CONTRASTS_CSV),
        "source_trace_bank_reference_csv": _relative(options.TRACE_BANK_REFERENCE_CSV),
        "source_populations_csv": _relative(options.POPULATIONS_CSV),
        "source_random_rotation_match_null_csv": _relative(options.MATCH_NULL_SUMMARY_CSV),
        "exclude_last_bins": EXCLUDE_LAST_BINS,
        "axis_override": AXIS_OVERRIDE,
        "ylim_pad_low": YLIM_PAD_LOW,
        "ylim_pad_high": YLIM_PAD_HIGH,
        "final_bracket_x_offset": FINAL_BRACKET_X_OFFSET,
        "rationale": (
            "Switched from unsigned component path to RMS excursion because the random-rotation "
            "null (behavior_model_bridge_random_rotation_match_null_summary.csv) found no "
            "trace-contour matching benefit on path (-0.002 pp, p=0.86) but a significant, "
            "coherence-scaling benefit on RMS excursion (+0.090 pp, p=0.004 at coherence >= 0.2). "
            "See panels/_fig4_option_sheet.py for the full comparison across candidate axes."
        ),
    }


def build_panel(
    out_dir: Path = OUT_DIR,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    panel_label: str = DEFAULT_PANEL_LABEL,
    panel_title: str = DEFAULT_PANEL_TITLE,
) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = _fig4_panel_header.add_middle_row_axes(fig)
    draw_panel(ax, panel_label=panel_label, panel_title=panel_title)
    paths = {
        "png": out_dir / "panel_e_rms_excursion.png",
        "pdf": out_dir / "panel_e_rms_excursion.pdf",
        "svg": out_dir / "panel_e_rms_excursion.svg",
    }
    fig.savefig(paths["png"], dpi=220, transparent=True)
    fig.savefig(paths["pdf"], transparent=True)
    fig.savefig(paths["svg"], transparent=True)
    plt.close(fig)
    provenance = load_provenance()
    provenance["displayed_panel_label"] = panel_label
    provenance["displayed_panel_title"] = panel_title
    (out_dir / "panel_e_rms_excursion_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    paths = build_panel()
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
