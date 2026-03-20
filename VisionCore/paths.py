"""
Anchored path resolution for VisionCore outputs.

All paths resolve relative to VisionCore/ repo root (the parent of this
package directory), so they work identically from IPython interactive
sessions, uv run, or any working directory.

Directories are created at import time.
"""
from pathlib import Path

VISIONCORE_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = VISIONCORE_ROOT / "outputs" / "cache"
FIGURES_DIR = VISIONCORE_ROOT / "outputs" / "figures"
STATS_DIR = VISIONCORE_ROOT / "outputs" / "stats"

for _d in (CACHE_DIR, FIGURES_DIR, STATS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
