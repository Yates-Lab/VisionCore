from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch

from .common import (
    CONFIG,
    CONTRAST_BY_KEY,
    MAP_CACHE,
    READOUT_CACHE,
    STATE_CACHE,
    iter_pair_indices,
    scale_index,
)


@dataclass(frozen=True)
class Split:
    image_positions: tuple[int, ...]
    trajectory_positions: tuple[int, ...]
    explicit_pairs: tuple[tuple[int, int], ...] | None = None

    @property
    def pairs(self) -> list[tuple[int, int]]:
        if self.explicit_pairs is not None:
            return [tuple(value) for value in self.explicit_pairs]
        return iter_pair_indices(self.image_positions, self.trajectory_positions)


@dataclass(frozen=True)
class Fold:
    index: int
    train: Split
    validation: Split
    test: Split


def load_fold(index: int, path: Path | None = None) -> Fold:
    source = path or (CONFIG / "fold_assignments.json")
    payload = json.loads(source.read_text())
    row = payload["folds"][int(index)]

    def make_split(key: str) -> Split:
        value = row[key]
        explicit = value.get("pair_positions")
        return Split(
            tuple(value["image_positions"]),
            tuple(value["trajectory_positions"]),
            None if explicit is None else tuple((int(pair[0]), int(pair[1])) for pair in explicit),
        )

    return Fold(index=int(index), train=make_split("train"), validation=make_split("validation"), test=make_split("test"))


def target_units(group: str) -> np.ndarray:
    with np.load(READOUT_CACHE) as archive:
        return np.asarray(archive[f"{group}_unit_indices"], dtype=np.int64)


def load_readout(device: str | torch.device = "cpu", units: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    with np.load(READOUT_CACHE) as archive:
        index = slice(None) if units is None else np.asarray(units, dtype=np.int64)
        return {
            "feature": torch.as_tensor(np.asarray(archive["feature_weights"])[index], dtype=torch.float32, device=device),
            "bias": torch.as_tensor(np.asarray(archive["bias"])[index], dtype=torch.float32, device=device),
            "space": torch.as_tensor(np.asarray(archive["space_weights"])[index], dtype=torch.float32, device=device),
        }


class ExactCache:
    """Read-only, fork-safe access to the exact state and endpoint-map caches."""

    def __init__(self, state_path: Path = STATE_CACHE, map_path: Path = MAP_CACHE):
        self.state_path = Path(state_path)
        self.map_path = Path(map_path)
        self._state: h5py.File | None = None
        self._maps: h5py.File | None = None

    def __enter__(self) -> "ExactCache":
        self._state = h5py.File(self.state_path, "r")
        self._maps = h5py.File(self.map_path, "r")
        if not bool(self._state.attrs.get("complete", False)) or not bool(self._maps.attrs.get("complete", False)):
            raise RuntimeError("Exact cache is incomplete")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._state is not None:
            self._state.close()
        if self._maps is not None:
            self._maps.close()
        self._state = None
        self._maps = None

    @property
    def state(self) -> h5py.File:
        if self._state is None:
            raise RuntimeError("ExactCache must be used as a context manager")
        return self._state

    @property
    def maps(self) -> h5py.File:
        if self._maps is None:
            raise RuntimeError("ExactCache must be used as a context manager")
        return self._maps

    def state_batch(
        self,
        pairs: list[tuple[int, int]],
        scale: float,
        frames: slice | np.ndarray | list[int] = slice(None),
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        scale_i = scale_index(scale)
        chunks = [np.asarray(self.state["h"][i, j, scale_i, frames], dtype=np.float32) for i, j in pairs]
        return torch.as_tensor(np.concatenate(chunks, axis=0), device=device)

    def map_batch(
        self,
        name: str,
        pairs: list[tuple[int, int]],
        scale: float,
        units: np.ndarray,
        frames: slice | np.ndarray | list[int] = slice(None),
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        scale_i = scale_index(scale)
        chunks = []
        for i, j in pairs:
            raw = np.asarray(self.maps[name][i, j, scale_i, frames], dtype=np.float32)
            chunks.append(raw[:, units])
        return torch.as_tensor(np.concatenate(chunks, axis=0), device=device)

    def metric_batch(
        self,
        name: str,
        pairs: list[tuple[int, int]],
        scale: float,
        units: np.ndarray,
        frames: slice | np.ndarray | list[int] = slice(None),
    ) -> np.ndarray:
        scale_i = scale_index(scale)
        chunks = []
        for i, j in pairs:
            raw = np.asarray(self.maps[name][i, j, scale_i, frames], dtype=np.float32)
            chunks.append(raw[:, units])
        return np.concatenate(chunks, axis=0)


def iter_pair_minibatches(pairs: list[tuple[int, int]], batch_pairs: int, seed: int, shuffle: bool) -> Iterator[list[tuple[int, int]]]:
    order = np.arange(len(pairs))
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(order)
    for start in range(0, len(order), int(batch_pairs)):
        yield [pairs[int(value)] for value in order[start : start + int(batch_pairs)]]


def contrast_scales(key: str) -> tuple[float, float]:
    contrast = CONTRAST_BY_KEY[key]
    return contrast.scale_a, contrast.scale_b
