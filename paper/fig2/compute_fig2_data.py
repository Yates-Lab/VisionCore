"""Compatibility re-export shim.

The figure-2 covariance-decomposition boilerplate moved to the shared package
``paper/covariance_decomposition`` (and the cache was renamed off the "fig2"
brand to ``covdecomp_*.pkl``). This module re-exports the public API under the
old names so existing fig2 panel scripts keep importing
``from compute_fig2_data import load_fig2_data, ...`` unchanged.

New code should import from the package directly:

    from covariance_decomposition import load_empirical_data
"""
import sys as _sys

from VisionCore.paths import VISIONCORE_ROOT as _VR

_PKG = str(_VR / "paper" / "covariance_decomposition")
if _PKG not in _sys.path:
    _sys.path.insert(0, _PKG)

from derive import (  # noqa: E402,F401
    load_empirical_data as load_fig2_data,
    compute_alignment_aggregate,
    _compute_alpha_stats,
    _compute_fano_stats,
    _compute_nc_stats,
    _slope_through_origin,
    _clustered_slope_bootstrap,
    SUBJECTS,
    SUBJECT_COLORS,
    SUBSPACE_WINDOW_IDX,
    SUBSPACE_K,
)

# Close-pair threshold constant (unchanged), kept for panels that annotate it.
INTERCEPT_THRESHOLD = 0.05
