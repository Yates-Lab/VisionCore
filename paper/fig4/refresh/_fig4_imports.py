"""Put the figure-4 directory on `sys.path`. Imported for its side effect only.

Scripts in here are run directly (`uv run python paper/fig4/refresh/foo.py`),
which puts *this* directory on `sys.path` but not its parent -- so the shared
figure modules (`_fig4_paths`, `_fig4_style`, `fixation_stats`, ...) would not
resolve. This was already broken before the refresh modules moved down here:
`panel_g_alternative_x_axes_diagnostic.py` imported `_fig4_cell_baseline_errorbars`
from the parent directory and only worked if the caller had arranged `PYTHONPATH`
themselves.

Import this before any `_fig4_*` or `fixation_stats` import:

    import _fig4_imports  # noqa: F401  (sys.path side effect)
    import _fig4_paths as _paths

Anchored on `VISIONCORE_ROOT` rather than `__file__`, per the repo convention.
"""

from __future__ import annotations

import sys

from VisionCore.paths import VISIONCORE_ROOT

_FIG4_DIR = VISIONCORE_ROOT / "paper" / "fig4"
if str(_FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(_FIG4_DIR))
