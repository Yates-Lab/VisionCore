"""Compatibility entry point for the cleaned twininfo production pipeline."""
from __future__ import annotations

from .pipeline import (
    MAIN_CONDITIONS,
    PHASE_CONDITIONS,
    SF_CONDITIONS,
    PipelineConfig,
    main,
    run_pipeline,
)

PHASE_COMPARISON_CONDITIONS = PHASE_CONDITIONS
SF_COMPARISON_CONDITIONS = SF_CONDITIONS

__all__ = [
    "MAIN_CONDITIONS",
    "PHASE_COMPARISON_CONDITIONS",
    "PipelineConfig",
    "SF_COMPARISON_CONDITIONS",
    "main",
    "run_pipeline",
]


if __name__ == "__main__":
    main()
