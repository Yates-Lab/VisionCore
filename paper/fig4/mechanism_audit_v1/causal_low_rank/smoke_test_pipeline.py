#!/usr/bin/env python3
"""Create a tiny synthetic exact-cache clone for end-to-end pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .common import normalize_rate, rate_map_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_image, n_trajectory, n_scale, n_frame = 4, 4, 5, 4
    n_channel, hidden, n_unit, kernel = 8, 8, 6, 3
    map_size = hidden - kernel + 1
    feature = rng.normal(size=(n_unit, n_channel)).astype(np.float32) / np.sqrt(n_channel)
    space = rng.normal(size=(n_unit, kernel, kernel)).astype(np.float32) / kernel
    bias = rng.normal(size=n_unit).astype(np.float32)
    base = rng.normal(size=(n_image, n_trajectory, n_frame, n_channel, hidden, hidden)).astype(np.float32)
    direction = rng.normal(size=base.shape).astype(np.float32) * 0.15
    h = np.stack([base + float(scale) * direction for scale in (0, 0.5, 1, 2, 3)], axis=2)
    maps = {name: [] for name in ("preactivation", "rate", "gain", "ssi", "expected_spikes", "mean_rate")}
    with torch.no_grad():
        flat = torch.from_numpy(h.reshape(-1, n_channel, hidden, hidden))
        z = F.conv2d(F.conv2d(flat, torch.from_numpy(feature)[:, :, None, None]), torch.from_numpy(space)[:, None], groups=n_unit)
        z += torch.from_numpy(bias)[None, :, None, None]
        rate = F.softplus(z)
        comp = rate_map_components(rate)
    shape_map = (n_image, n_trajectory, n_scale, n_frame, n_unit, map_size, map_size)
    shape_metric = shape_map[:-2]
    with h5py.File(args.output / "convgru_states.h5", "w") as handle:
        handle.create_dataset("h", data=h.astype(np.float16))
        handle.create_dataset("completed_pairs", data=np.ones((n_image, n_trajectory), dtype=bool))
        handle.attrs["complete"] = True
    with h5py.File(args.output / "rr100_maps.h5", "w") as handle:
        handle.create_dataset("preactivation", data=z.numpy().reshape(shape_map).astype(np.float16))
        handle.create_dataset("rate", data=rate.numpy().reshape(shape_map).astype(np.float16))
        handle.create_dataset("gain", data=comp["gain"].numpy().reshape(shape_map).astype(np.float16))
        for name in ("ssi", "expected_spikes", "mean_rate"):
            handle.create_dataset(name, data=comp[name].numpy().reshape(shape_metric).astype(np.float32))
        handle.create_dataset("completed_pairs", data=np.ones((n_image, n_trajectory), dtype=bool))
        handle.attrs["complete"] = True
    np.savez_compressed(
        args.output / "readout_weights.npz",
        feature_weights=feature,
        bias=bias,
        space_weights=space,
        low_unit_indices=np.arange(4),
        high_unit_indices=np.arange(4, 6),
    )
    folds = {
        "folds": [
            {
                "index": index,
                "train": {"image_positions": [i for i in range(4) if i != index], "trajectory_positions": [j for j in range(4) if j != index]},
                "validation": {"image_positions": [index], "trajectory_positions": [(index + 1) % 4]},
                "test": {"image_positions": [index], "trajectory_positions": [index]},
            }
            for index in range(4)
        ]
    }
    (args.output / "fold_assignments.json").write_text(json.dumps(folds, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
