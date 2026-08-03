"""Shared style tokens and presentation constants for figure 4.

Every panel and every refresh script styled itself with a byte-identical copy
of `configure_matplotlib()` -- nine of them -- plus its own re-declaration of
the same hex colors. Nothing here is figure-4-specific logic; it is the one
place a change to the figure's typography or palette has to be made.

Two constants deliberately do NOT live here, because their values genuinely
differ between modules and merging them would be a silent restyle:

* `GRID` -- three distinct values in the tree (`#d8dde3`, `#E3E3E3`,
  `#E7E7E7`). `PALE_GRID` below is only the `#E7E7E7` one.
* `_fig4_contour_schematic`'s `BLUE` (`#1e4ed8`) and `GRAY` (`#5f6368`) -- the
  schematic has its own diagram palette, distinct from the series colors here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


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


# Series palette. The figure-wide convention is BLUE = low-SF, ORANGE =
# high-SF; MUTED_ORANGE is the pooled/looser high-SF variant.
BLUE = "#0072B2"
ORANGE = "#D55E00"
MUTED_ORANGE = "#A9714B"
GRAY = "#6B6F75"
INK = "#111111"
PALE_GRID = "#E7E7E7"

# All high-SF populations are shades of ORANGE (full strength for the aligned
# headline result, lighter/muted for looser or pooled high-SF groups); low-SF
# stays BLUE. Same color-means-SF convention as the other panels, applied to
# five populations instead of two.
POPULATION_COLORS = {
    "high_sf_aligned": ORANGE,
    "high_sf_oblique": "#E8956B",
    "high_sf_orthogonal": "#F2C6A0",
    "high_sf_all": MUTED_ORANGE,
    "low_sf_all": BLUE,
}
POPULATION_MARKERS = {
    "high_sf_aligned": "o",
    "high_sf_oblique": "s",
    "high_sf_orthogonal": "^",
    "high_sf_all": "D",
    "low_sf_all": "v",
}

# Local edge-coherence bins, in plotting order. Used both as x tick labels and
# as the `categories=` ordering for the `coherence_bin` column, so it is
# presentation rather than analysis: the bin edges themselves are fixed
# upstream, in whatever produced the summary CSVs.
COHERENCE_ORDER = ("0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1")
