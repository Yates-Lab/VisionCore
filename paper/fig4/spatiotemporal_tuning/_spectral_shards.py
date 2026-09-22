"""Strict loading for image-sharded Figure-4 spectral replay artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


PREDICTOR_KEYS = (
    "total_dynamic_power",
    "tf_marginal_power",
    "sf_orientation_marginal_power",
    "separable_passband_power",
    "joint_passband_power",
    "joint_signed_rate_drive",
)


def _archive_path(path: Path) -> Path:
    return path / "causal_chain_shard.npz" if path.is_dir() else path


def load_and_merge_shards(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Merge disjoint image shards while failing on any fixed-axis mismatch."""
    archives: list[dict[str, np.ndarray]] = []
    for raw_path in paths:
        path = _archive_path(Path(raw_path))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as handle:
            archives.append({key: handle[key] for key in handle.files})
    if not archives:
        raise ValueError("at least one spectral replay shard is required")
    archives.sort(key=lambda item: int(np.min(item["image_indices"])))
    reference = archives[0]
    fixed = (
        "trace_indices",
        "motion_scales",
        "unit_indices",
        "spatial_cpd",
        "temporal_hz",
        "orientation_deg",
    )
    for archive in archives[1:]:
        for key in fixed:
            if not np.array_equal(reference[key], archive[key]):
                raise ValueError(f"spectral replay shards disagree on {key}")
    image_indices = np.concatenate([item["image_indices"] for item in archives])
    if len(np.unique(image_indices)) != len(image_indices):
        raise ValueError("spectral replay shards contain duplicate image rows")
    order = np.argsort(image_indices)
    result = {key: reference[key] for key in fixed}
    result["image_indices"] = image_indices[order]
    for key in ("mean_rate", "expected_spikes", "map_ssi", *PREDICTOR_KEYS):
        result[key] = np.concatenate([item[key] for item in archives], axis=0)[order]
    weights = np.asarray([len(item["image_indices"]) for item in archives], dtype=float)
    result["average_power"] = np.average(
        np.stack([item["average_power"] for item in archives]), axis=0, weights=weights
    )
    for key in (
        "example_power",
        "example_rate_maps",
        "example_movie_frames",
        "example_trace_xy",
    ):
        result[key] = reference.get(key, np.empty((0,)))
    return result
