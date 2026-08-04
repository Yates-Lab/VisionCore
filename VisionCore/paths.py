"""
Anchored path resolution for VisionCore outputs.

By default, all paths resolve relative to VisionCore/ repo root (the parent of
this package directory), so they work identically from IPython interactive
sessions, uv run, or any working directory. The output roots can be overridden
with VISIONCORE_CACHE_DIR, VISIONCORE_FIGURES_DIR, and VISIONCORE_STATS_DIR for
isolated smoke tests or bundle verification.

Directories are created at import time.
"""
import os
from pathlib import Path

VISIONCORE_ROOT = Path(__file__).resolve().parents[1]


def _output_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


CACHE_DIR = _output_path("VISIONCORE_CACHE_DIR", VISIONCORE_ROOT / "outputs" / "cache")
FIGURES_DIR = _output_path("VISIONCORE_FIGURES_DIR", VISIONCORE_ROOT / "outputs" / "figures")
STATS_DIR = _output_path("VISIONCORE_STATS_DIR", VISIONCORE_ROOT / "outputs" / "stats")

for _d in (CACHE_DIR, FIGURES_DIR, STATS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
