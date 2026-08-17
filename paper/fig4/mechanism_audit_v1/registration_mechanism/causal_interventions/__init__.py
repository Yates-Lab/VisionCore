"""Frozen-weight causal tests of ConvGRU spatial transport.

The package is intentionally isolated from rank fitting, P/Q semantics, and
GRU instrumentation.  Production entry points import the audited equations
but never mutate the frozen checkpoint in place.
"""

from .mechanics import (
    KernelIntervention,
    RealignmentReplay,
    TRANSPORT_INTERVENTIONS,
    fourier_shift_2d,
    replay_realignment,
    replay_transport_intervention,
)

__all__ = [
    "KernelIntervention",
    "RealignmentReplay",
    "TRANSPORT_INTERVENTIONS",
    "fourier_shift_2d",
    "replay_realignment",
    "replay_transport_intervention",
]
