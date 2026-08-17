"""Exact ConvGRU instrumentation for the Figure 4 registration audit.

This package is deliberately separate from rank-8 fitting and from the cached
P/Q semantic decomposition.  Its production entry points replay the frozen
core only after a projector has been fixed, and reduce recurrent terms online
instead of writing another full hidden-state cache.
"""

from .equations import (
    GRUStepTerms,
    instrument_convgru_step,
    replay_convgru_cell,
    split_conv2d_contributions,
)

__all__ = [
    "GRUStepTerms",
    "instrument_convgru_step",
    "replay_convgru_cell",
    "split_conv2d_contributions",
]
