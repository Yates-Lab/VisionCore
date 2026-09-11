"""Optional publication font floor, applied before exporting a figure.

VISIONCORE_MIN_FIGURE_FONT_PT is in the source figure's PDF points. Callers
must account for the scale at which that PDF is placed in the manuscript.
The final compiled PDF still needs an independent font and layout audit.
"""

from __future__ import annotations

import os

from matplotlib.mathtext import MathTextParser
from matplotlib.text import Text


def apply_font_floor(figure) -> None:
    minimum = float(os.environ.get("VISIONCORE_MIN_FIGURE_FONT_PT", "0"))
    if minimum <= 0:
        return
    figure.canvas.draw()
    parser = MathTextParser("path")
    for artist in figure.findobj(match=lambda obj: isinstance(obj, Text)):
        text = artist.get_text()
        if not artist.get_visible() or not text.strip():
            continue
        size = max(float(artist.get_fontsize()), minimum)
        artist.set_fontsize(size)
        if "$" in text and artist.get_parse_math():
            # MathText shrinks subscripts and superscripts independently of
            # the label font. Enforce the floor on its smallest actual glyph.
            glyphs = [glyph for line in text.splitlines()
                      for glyph in parser.parse(line, dpi=72, prop=artist.get_fontproperties()).glyphs]
            if glyphs:
                smallest = min(float(glyph[1]) for glyph in glyphs)
                if smallest < minimum:
                    artist.set_fontsize(size * minimum / smallest)
