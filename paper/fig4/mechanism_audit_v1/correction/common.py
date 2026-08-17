"""Shared constants and exact causal-history helpers for the correction stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
LEGACY_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
SOURCE_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
OUT_DIR = ROOT / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1"
PREVIOUS_ANALYSIS_DIR = ROOT / "outputs/figures/fig4/spatiotemporal_tuning"
BANK_DIR = OUT_DIR / "banks"
CORE_DIR = OUT_DIR / "core_ssi"
CONTROLLED_DIR = OUT_DIR / "controlled_scaling"
PLOT_DATA_DIR = OUT_DIR / "plot_data"

FRAME_RATE_HZ = 120.0
DT_S = 1.0 / FRAME_RATE_HZ
N_LAGS = 32
N_PRECEDING = N_LAGS - 1
N_SCORED = 40
FULL_HISTORY_SAMPLES = N_PRECEDING + N_SCORED
SCALE_FACTORS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
CORRECTION_SEED = 20260810


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def corrected_source_indices(scored_start: int) -> np.ndarray:
    """Return true source indices for 31 preceding plus 40 scored samples."""
    return np.arange(int(scored_start) - N_PRECEDING, int(scored_start) + N_SCORED, dtype=np.int64)


def held_time_indices() -> np.ndarray:
    """Monotone relative time indices for a synthetic held prefix and real score."""
    return np.arange(-N_PRECEDING, N_SCORED, dtype=np.int64)


def lag_windows_from_sequence(sequence: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """Exact source-index analogue of `_embed_time_lags`.

    The model lag axis is current-to-oldest: lag 0 is the current retinal
    sample and lag 31 is the oldest.  A 71-frame causal sequence therefore
    yields exactly 40 outputs.
    """
    values = np.asarray(sequence)
    out_frames = len(values) - int(n_lags) + 1
    if out_frames <= 0:
        raise ValueError("Sequence is shorter than the model lag dimension")
    out = np.empty((out_frames, int(n_lags)), dtype=values.dtype)
    for lag in range(int(n_lags)):
        out[:, lag] = values[int(n_lags) - 1 - lag : len(values) - lag]
    return out


def validate_monotone_causal_windows(windows: np.ndarray, output_times: np.ndarray) -> dict[str, int]:
    values = np.asarray(windows)
    output = np.asarray(output_times)
    chronological = values[:, ::-1]
    return {
        "total_outputs": int(values.shape[0]),
        "outputs_with_future_samples": int(np.sum(np.any(values > output[:, None], axis=1))),
        "outputs_with_nonmonotonic_indices": int(np.sum(np.any(np.diff(chronological, axis=1) < 0, axis=1))),
        "outputs_with_nonunit_index_steps": int(np.sum(np.any(np.diff(chronological, axis=1) != 1, axis=1))),
    }
