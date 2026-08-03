#!/usr/bin/env python3
"""Panel B: path length separates low- and high-SF benefit.

Draws source pair (B, C) -- low-SF and high-SF units on strong contours, with
no orientation-selectivity gate -- on one shared axis in the figure's top row.
This is the panel that carries the microsaccade legend for the whole figure.

The drawing machinery is shared with panel D; see `_fig4_path_bins.py`. What
lives here is only what makes this panel this panel: which source groups it
draws, and where it sits on the page.

Usage:
    uv run python paper/fig4/panel_b_path_bins.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _fig4_path_bins as path_bins
from _fig4_path_bins import OUT_DIR

# Source letters from the measured Illustrator reference: low-SF and high-SF
# units, both on strong contours.
SOURCE_LABELS = ("B", "C")

# Top-row layout. The header is drawn on its own full-figure axis rather than
# inside the plotting axes (separate_header), which is why this panel needs its
# own letter/header placement constants.
AXES_BOX = path_bins.TOP_ROW_PAIR_AXES_BOX
YLIM_PAD_LOW = 0.12
YLIM_PAD_HIGH = 0.14
SEPARATE_HEADER = True

# Panel B is the only path-bin panel showing the microsaccade legend; repeating
# it on panel D would spend space restating the same key.
SHOW_MICROSACCADE_LEGEND = True


def build_panel(
    *,
    figsize: tuple[float, float],
    out_dir: Path = OUT_DIR,
    panel_label: str | None = None,
    panel_title: str | None = None,
    panel_subtitle: str | None = None,
    xlabel: str | None = None,
) -> Path:
    return path_bins.build_pair_panel(
        SOURCE_LABELS,
        figsize=figsize,
        out_dir=out_dir,
        panel_label=panel_label,
        panel_title=panel_title,
        panel_subtitle=panel_subtitle,
        xlabel=xlabel,
        axes_box=AXES_BOX,
        ylim_pad_low=YLIM_PAD_LOW,
        ylim_pad_high=YLIM_PAD_HIGH,
        separate_header=SEPARATE_HEADER,
        show_microsaccade_legend=SHOW_MICROSACCADE_LEGEND,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width", type=float, default=2.5)
    parser.add_argument("--height", type=float, default=3.9939)
    args = parser.parse_args()
    print(build_panel(figsize=(args.width, args.height), out_dir=args.out_dir, panel_label="B"))


if __name__ == "__main__":
    main()
