from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad

from .features import event_feature_rows, fixation_window_features
from .image_features import local_backimage_features


STIMULUS_REGIME = {
    "fixrsvp": "fixation_task",
    "backimage": "free_viewing_natural_image",
    "gaborium": "forage_gabor",
    "gratings": "forage_grating",
}


@dataclass(frozen=True)
class ExtractionConfig:
    dt: float = 1.0 / 120.0
    window_samples: int = 128
    stride_samples: int = 16
    min_epoch_samples: int = 24
    min_valid_fraction: float = 0.0
    fixation_radius_deg: float = 1.0
    max_abs_eye_deg: float = 12.0
    speed_z: float = 6.0
    event_pad_samples: int = 1
    early_s: float = 0.10
    mid_s: float = 0.30
    max_windows_per_stimulus: int = 1000
    seed: int = 0
    include_image_features: bool = False
    image_patch_radius_deg: float = 1.0


def _load_dict_dataset(path: Path) -> Any:
    try:
        from models.data.datasets import DictDataset

        return DictDataset.load(str(path))
    except Exception:
        from DataYatesV1 import DictDataset

        return DictDataset.load(path)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _contiguous_true_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < mask.size and mask[i]:
            i += 1
        blocks.append((start, i))
    return blocks


def _phase_label(samples_since_event: float | None, cfg: ExtractionConfig) -> str:
    if samples_since_event is None or not np.isfinite(samples_since_event):
        return "no_recent_event"
    age_s = float(samples_since_event) * cfg.dt
    if age_s < cfg.early_s:
        return "early_post_event"
    if age_s < cfg.mid_s:
        return "mid_fixation"
    return "late_fixation"


def _samples_since_previous_event(event_mask: np.ndarray) -> np.ndarray:
    out = np.full(event_mask.size, np.nan, dtype=np.float64)
    last_event = np.nan
    for i, is_event in enumerate(np.asarray(event_mask, dtype=bool)):
        if is_event:
            last_event = float(i)
            out[i] = 0.0
        elif np.isfinite(last_event):
            out[i] = float(i) - last_event
    return out


def _event_rate_denominator(valid_mask: np.ndarray, dt: float) -> float:
    return float(np.count_nonzero(valid_mask) * dt)


def _speed_threshold_mad_valid_pairs(trace: np.ndarray, valid: np.ndarray, *, dt: float, z: float) -> float:
    """Robust speed threshold using only adjacent valid sample pairs."""
    x = np.asarray(trace, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if x.shape[0] < 2 or v.shape[0] != x.shape[0]:
        return speed_threshold_mad(x, dt=dt, z=z)
    finite = np.isfinite(x).all(axis=1)
    pair_valid = v[1:] & v[:-1] & finite[1:] & finite[:-1]
    if np.count_nonzero(pair_valid) < 3:
        return speed_threshold_mad(x[finite] if np.count_nonzero(finite) >= 3 else x, dt=dt, z=z)
    inc = np.diff(x, axis=0)[pair_valid]
    speed = np.linalg.norm(inc, axis=1) / float(dt)
    speed = speed[np.isfinite(speed)]
    if speed.size < 3:
        return speed_threshold_mad(x[finite] if np.count_nonzero(finite) >= 3 else x, dt=dt, z=z)
    med = np.median(speed)
    mad = np.median(np.abs(speed - med))
    return float(med + float(z) * 1.4826 * mad)


def extract_session_stimulus(
    session: Any,
    stimulus: str,
    cfg: ExtractionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract matched fixation-window features and event rows from one stimulus dataset."""
    stimulus = str(stimulus).lower()
    dset_path = Path(session.sess_dir) / "datasets" / f"{stimulus}.dset"
    inventory = {
        "session": session.name,
        "stimulus": stimulus,
        "regime": STIMULUS_REGIME.get(stimulus, stimulus),
        "dataset_path": str(dset_path),
        "status": "missing",
        "n_samples": 0,
        "n_trials": 0,
        "n_windows": 0,
        "n_events": 0,
    }
    if not dset_path.exists():
        return [], [], inventory

    dset = _load_dict_dataset(dset_path)
    eyepos = _as_numpy(dset["eyepos"]).astype(np.float64)
    trial_inds = _as_numpy(dset.covariates["trial_inds"]).reshape(-1).astype(int)
    if "dpi_valid" in dset.covariates:
        dpi_valid = _as_numpy(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
    elif "dfs" in dset.covariates:
        dfs = _as_numpy(dset.covariates["dfs"])
        dpi_valid = np.asarray(dfs).reshape(dfs.shape[0], -1).any(axis=1)
    else:
        dpi_valid = np.ones(eyepos.shape[0], dtype=bool)

    finite = np.isfinite(eyepos).all(axis=1)
    in_bounds = (np.abs(eyepos[:, 0]) <= cfg.max_abs_eye_deg) & (np.abs(eyepos[:, 1]) <= cfg.max_abs_eye_deg)
    valid_base = dpi_valid & finite & in_bounds
    if stimulus == "fixrsvp":
        valid_base &= np.linalg.norm(eyepos, axis=1) <= cfg.fixation_radius_deg

    seed_bytes = f"{session.name}:{stimulus}:{cfg.seed}".encode("utf-8")
    seed_offset = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=4).digest(), "little")
    rng = np.random.default_rng((int(cfg.seed) + seed_offset) % (2**32 - 1))
    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    unique_trials = np.unique(trial_inds)

    for trial_idx in unique_trials:
        global_idx = np.where(trial_inds == int(trial_idx))[0]
        if global_idx.size < max(cfg.min_epoch_samples, 3):
            continue
        trace = eyepos[global_idx]
        valid = valid_base[global_idx]
        valid_fraction = float(np.mean(valid))
        if cfg.min_valid_fraction > 0 and valid_fraction < cfg.min_valid_fraction:
            continue

        threshold = _speed_threshold_mad_valid_pairs(trace, valid, dt=cfg.dt, z=cfg.speed_z)
        events, event_mask, _ = detect_microsaccade_events(
            trace,
            dt=cfg.dt,
            threshold_deg_s=threshold,
            min_samples=1,
            pad_samples=cfg.event_pad_samples,
        )
        valid_events = [
            event for event in events
            if int(event["onset"]) >= 0
            and int(event["offset"]) < valid.size
            and np.all(valid[int(event["onset"]): int(event["offset"]) + 1])
        ]
        for erow in event_feature_rows(trace, valid_events, dt=cfg.dt):
            erow.update({
                "session": session.name,
                "stimulus": stimulus,
                "regime": STIMULUS_REGIME.get(stimulus, stimulus),
                "trial_idx": int(trial_idx),
                "event_threshold_deg_s": float(threshold),
                "trial_valid_duration_s": _event_rate_denominator(valid, cfg.dt),
            })
            event_rows.append(erow)

        clean = valid & ~event_mask
        since_event = _samples_since_previous_event(event_mask)
        for block_start, block_stop in _contiguous_true_blocks(clean):
            if block_stop - block_start < cfg.min_epoch_samples:
                continue
            last_start = block_stop - int(cfg.window_samples)
            if last_start < block_start:
                continue
            starts = np.arange(block_start, last_start + 1, max(1, int(cfg.stride_samples)), dtype=int)
            if starts.size == 0:
                continue
            for local_start in starts:
                local_stop = local_start + int(cfg.window_samples)
                window = trace[local_start:local_stop]
                features = fixation_window_features(window, dt=cfg.dt)
                phase = _phase_label(since_event[local_start], cfg)
                row: dict[str, Any] = {
                    "session": session.name,
                    "stimulus": stimulus,
                    "regime": STIMULUS_REGIME.get(stimulus, stimulus),
                    "trial_idx": int(trial_idx),
                    "global_start": int(global_idx[local_start]),
                    "global_stop": int(global_idx[local_stop - 1] + 1),
                    "local_start": int(local_start),
                    "local_stop": int(local_stop),
                    "epoch_start_local": int(block_start),
                    "epoch_stop_local": int(block_stop),
                    "epoch_duration_s": float((block_stop - block_start) * cfg.dt),
                    "phase": phase,
                    "samples_since_event": float(since_event[local_start]) if np.isfinite(since_event[local_start]) else np.nan,
                    "event_threshold_deg_s": float(threshold),
                "events_in_trial": int(len(valid_events)),
                    "valid_fraction_trial": valid_fraction,
                }
                row.update(features)
                if cfg.include_image_features and stimulus == "backimage":
                    gaze = np.asarray([row["mean_x_deg"], row["mean_y_deg"]], dtype=np.float64)
                    row.update(local_backimage_features(
                        session_name=session.name,
                        trial_idx=int(trial_idx),
                        gaze_xy_deg=gaze,
                        patch_radius_deg=cfg.image_patch_radius_deg,
                    ))
                window_rows.append(row)

    if cfg.max_windows_per_stimulus > 0 and len(window_rows) > cfg.max_windows_per_stimulus:
        keep = np.sort(rng.choice(len(window_rows), size=int(cfg.max_windows_per_stimulus), replace=False))
        window_rows = [window_rows[int(i)] for i in keep]

    inventory.update({
        "status": "ok",
        "n_samples": int(eyepos.shape[0]),
        "n_trials": int(unique_trials.size),
        "valid_duration_s": float(np.count_nonzero(valid_base) * cfg.dt),
        "valid_fraction": float(np.mean(valid_base)) if valid_base.size else np.nan,
        "n_windows": int(len(window_rows)),
        "n_events": int(len(event_rows)),
    })
    return window_rows, event_rows, inventory
