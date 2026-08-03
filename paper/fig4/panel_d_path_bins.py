#!/usr/bin/env python3
"""Panel D: contour alignment exposes a high-SF limit.

Draws source pair (E, F) -- low-SF and high-SF units restricted to those whose
preferred orientation matches the local contour within 15 degrees -- on one
shared axis in the figure's middle row. The contrast with panel B is the
alignment gate: same path-length axis, contour-matched population.

The drawing machinery is shared with panel B; see `_fig4_path_bins.py`. What
lives here is only what makes this panel this panel: which source groups it
draws, and where it sits on the page.

Usage:
    uv run python paper/fig4/panel_d_path_bins.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _fig4_panel_header
import _fig4_path_bins as path_bins
from _fig4_path_bins import OUT_DIR

# Source letters from the measured Illustrator reference: low-SF and high-SF
# units, both gated on contour alignment.
SOURCE_LABELS = ("E", "F")

# Middle-row layout. The header sits inside the plotting axes here (unlike
# panel B), so this panel shares the middle row's axes box and y-label
# placement with its neighbours rather than defining its own.
AXES_BOX = _fig4_panel_header.MIDDLE_ROW_AXES_BOX
YLABEL_X = _fig4_panel_header.MIDDLE_ROW_YLABEL_X

# Tighter than panel B's: the contour-matched series span a narrower range, and
# panel B's padding left conspicuous dead space above and below the data.
YLIM_PAD_LOW = 0.055
YLIM_PAD_HIGH = 0.055
SEPARATE_HEADER = False

# The microsaccade legend appears once, on panel B.
SHOW_MICROSACCADE_LEGEND = False


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
        ylabel_x=YLABEL_X,
        axes_box=AXES_BOX,
        ylim_pad_low=YLIM_PAD_LOW,
        ylim_pad_high=YLIM_PAD_HIGH,
        separate_header=SEPARATE_HEADER,
        show_microsaccade_legend=SHOW_MICROSACCADE_LEGEND,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width", type=float, default=2.6)
    parser.add_argument("--height", type=float, default=3.88)
    args = parser.parse_args()
    print(build_panel(figsize=(args.width, args.height), out_dir=args.out_dir, panel_label="D"))


if __name__ == "__main__":
    main()
