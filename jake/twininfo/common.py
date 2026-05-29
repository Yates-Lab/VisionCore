"""Shared constants, paths, and Ryan digital-twin imports."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT


WORKSPACE_ROOT = VISIONCORE_ROOT.parent
RYAN_COMMON_DIR = VISIONCORE_ROOT / "ryan" / "digital-twin-fem"
if str(RYAN_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(RYAN_COMMON_DIR))

for _workspace_pkg in ("DataYatesV1", "DataRowleyV1V2"):
    _pkg_root = WORKSPACE_ROOT / _workspace_pkg
    if _pkg_root.exists() and str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))

from _common import (  # noqa: E402
    CHECKPOINT_PATH,
    DT,
    N_LAGS,
    OUT_SIZE,
    PPD,
    build_population,
    compute_rate_map_batched,
    extract_fixrsvp_eye_traces,
    load_digital_twin,
    make_counterfactual_stim,
)


OUTPUT_DIR = VISIONCORE_ROOT / "outputs" / "twininfo"
MPLCONFIG_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "twininfo_matplotlib"
for _path in (OUTPUT_DIR, MPLCONFIG_DIR):
    _path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


DEFAULT_SEED = 0
DEFAULT_CCMAX_THRESHOLD = 0.80
DEFAULT_BANDS_HZ = {
    "low": (0.5, 4.0),
    "mid": (4.0, 15.0),
    "high": (15.0, 60.0),
}


@dataclass(frozen=True)
class StimulusSpec:
    """Minimal natural-image descriptor used in output metadata."""

    family: str = "natural"
    image_index: int | None = None
    seed: int = DEFAULT_SEED


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write human-readable metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


__all__ = [
    "CHECKPOINT_PATH",
    "DEFAULT_BANDS_HZ",
    "DEFAULT_CCMAX_THRESHOLD",
    "DEFAULT_SEED",
    "DT",
    "MPLCONFIG_DIR",
    "N_LAGS",
    "OUTPUT_DIR",
    "OUT_SIZE",
    "PPD",
    "RYAN_COMMON_DIR",
    "StimulusSpec",
    "build_population",
    "compute_rate_map_batched",
    "extract_fixrsvp_eye_traces",
    "load_digital_twin",
    "make_counterfactual_stim",
    "to_jsonable",
    "write_json",
]
