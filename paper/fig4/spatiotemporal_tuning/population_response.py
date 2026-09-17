"""Population estimands and provenance gates shared by Figure 4B."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


EPS = np.finfo(np.float64).eps


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def matrix_shard_summaries(summary: dict[str, object]) -> list[dict[str, object]]:
    """Normalize merged-matrix and single-shard provenance schemas."""
    shards = summary.get("shard_summaries", [])
    if isinstance(shards, list) and shards:
        return [value for value in shards if isinstance(value, dict)]
    if isinstance(summary.get("model_provenance"), dict):
        return [summary]
    return []


def _matrix_checkpoint(summary: dict[str, object]) -> tuple[str, str]:
    labels: set[str] = set()
    digests: set[str] = set()
    for shard in matrix_shard_summaries(summary):
        model = shard.get("model_provenance", {}).get("model", {})
        digest = str(model.get("checkpoint_sha256", ""))
        if digest:
            digests.add(digest)
        checkpoint = str(model.get("checkpoint_path", ""))
        if checkpoint:
            labels.add(Path(checkpoint).parent.name)
    if len(digests) != 1:
        raise ValueError("matrix shards must resolve to exactly one checkpoint digest")
    return (next(iter(labels)) if len(labels) == 1 else "selected model", next(iter(digests)))


def selected_checkpoint_digest(model_spec: Path) -> str:
    payload = yaml.safe_load(model_spec.read_text(encoding="utf-8")) or {}
    digest = str((payload.get("checkpoint") or {}).get("sha256", ""))
    if not digest:
        # Kept only for reading an already-created pre-contract test fixture.
        digest = str(payload.get("checkpoint_sha256", ""))
    if not digest:
        raise ValueError(f"model spec lacks checkpoint.sha256: {model_spec}")
    return digest


def assert_selected_matrix(
    summary: dict[str, object], *, model_spec: Path
) -> tuple[str, str]:
    label, observed = _matrix_checkpoint(summary)
    expected = selected_checkpoint_digest(model_spec)
    if observed != expected:
        raise ValueError(
            "response matrix checkpoint does not match the production model: "
            f"observed={observed}, expected={expected}"
        )
    return label, observed


def matrix_trace_filter(summary: dict[str, object]) -> dict[str, object] | None:
    contracts = []
    for shard in matrix_shard_summaries(summary):
        provenance = shard.get("trace_bank", {}).get("trace_provenance")
        if provenance is not None:
            contracts.append(provenance.get("filter", {}))
    if not contracts:
        return None
    encoded = {json.dumps(contract, sort_keys=True) for contract in contracts}
    if len(encoded) != 1:
        raise ValueError("matrix shards use different eye-trace filters")
    contract = contracts[0]
    if "zero-phase" not in str(contract.get("kind", "")).lower():
        raise ValueError("matrix eye traces were not continuously zero-phase filtered")
    return contract


def load_matrix(
    matrix_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    summary = json.loads((matrix_dir / "summary.json").read_text(encoding="utf-8"))
    pilot = summary.get("pilot", {})
    n_images = int(summary.get("n_images", pilot.get("n_images")))
    n_traces = int(summary.get("n_traces", pilot.get("n_traces")))
    n_units = int(summary.get("n_units", pilot.get("n_units")))
    summary = dict(summary)
    summary.update(n_images=n_images, n_traces=n_traces, n_units=n_units)
    moving_rate = np.load(matrix_dir / "mean_rate_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_spikes = np.load(
        matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r"
    ).reshape(n_images, n_traces, n_units)
    stable_rate = np.load(matrix_dir / "stabilized_mean_rate_by_image.npy")
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    stable_spikes = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy")
    expected_stable = (n_images, n_units)
    if any(value.shape != expected_stable for value in (stable_rate, stable_ssi, stable_spikes)):
        raise ValueError("stabilized and moving response matrices are not aligned")
    trace_table = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    if len(trace_table) != n_traces:
        raise ValueError("trace feature table does not match response matrices")
    arrays = {
        "moving_rate": np.asarray(moving_rate),
        "moving_ssi": np.asarray(moving_ssi),
        "moving_spikes": np.asarray(moving_spikes),
        "stable_rate": np.asarray(stable_rate),
        "stable_ssi": np.asarray(stable_ssi),
        "stable_spikes": np.asarray(stable_spikes),
    }
    return arrays, trace_table, summary


def quantile_bins(x: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    labels = np.empty(len(order), dtype=int)
    for index, rows in enumerate(np.array_split(order, int(count))):
        labels[rows] = int(index)
    return labels


def _population_arrays(
    arrays: dict[str, np.ndarray], unit_indices: np.ndarray
) -> dict[str, np.ndarray]:
    units = np.asarray(unit_indices, dtype=int)
    moving_spikes_by_unit = arrays["moving_spikes"][:, :, units]
    moving_information_by_unit = (
        arrays["moving_spikes"][:, :, units] * arrays["moving_ssi"][:, :, units]
    )
    stable_spikes_by_unit = arrays["stable_spikes"][:, units]
    stable_information_by_unit = (
        arrays["stable_spikes"][:, units] * arrays["stable_ssi"][:, units]
    )
    return {
        "moving_spikes": moving_spikes_by_unit.sum(axis=2),
        "moving_information": moving_information_by_unit.sum(axis=2),
        "stable_spikes": stable_spikes_by_unit.sum(axis=1),
        "stable_information": stable_information_by_unit.sum(axis=1),
        "moving_spikes_by_unit": moving_spikes_by_unit,
        "moving_information_by_unit": moving_information_by_unit,
        "stable_spikes_by_unit": stable_spikes_by_unit,
        "stable_information_by_unit": stable_information_by_unit,
        "unit_indices": units,
    }


def population_effects(
    reduced: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    image_indices: np.ndarray | None = None,
    unit_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    traces = np.asarray(trace_indices, dtype=int)
    images = (
        np.arange(reduced["moving_spikes"].shape[0], dtype=int)
        if image_indices is None
        else np.asarray(image_indices, dtype=int)
    )
    units = (
        np.arange(reduced["moving_spikes_by_unit"].shape[2], dtype=int)
        if unit_indices is None
        else np.asarray(unit_indices, dtype=int)
    )
    moving_spikes = float(
        np.mean(reduced["moving_spikes_by_unit"][np.ix_(images, traces, units)].sum(axis=2))
    )
    moving_information = float(
        np.mean(
            reduced["moving_information_by_unit"][np.ix_(images, traces, units)].sum(axis=2)
        )
    )
    stable_spikes = float(
        np.mean(reduced["stable_spikes_by_unit"][np.ix_(images, units)].sum(axis=1))
    )
    stable_information = float(
        np.mean(reduced["stable_information_by_unit"][np.ix_(images, units)].sum(axis=1))
    )
    moving_ssi = moving_information / max(moving_spikes, EPS)
    stable_ssi = stable_information / max(stable_spikes, EPS)
    return (
        100.0 * (moving_spikes - stable_spikes) / max(stable_spikes, EPS),
        100.0 * (moving_ssi - stable_ssi) / max(stable_ssi, EPS),
    )


def crossed_population_bootstrap(
    reduced: dict[str, np.ndarray],
    trace_indices: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    return_draws: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    traces = np.asarray(trace_indices, dtype=int)
    centers = population_effects(reduced, traces)
    draws = np.empty((int(n_bootstrap), 2), dtype=float)
    n_images = reduced["moving_spikes_by_unit"].shape[0]
    n_units = reduced["moving_spikes_by_unit"].shape[2]
    for draw in range(int(n_bootstrap)):
        image_draw = rng.integers(0, n_images, size=n_images)
        trace_draw = traces[rng.integers(0, len(traces), size=len(traces))]
        unit_draw = rng.integers(0, n_units, size=n_units)
        draws[draw] = population_effects(
            reduced,
            trace_draw,
            image_indices=image_draw,
            unit_indices=unit_draw,
        )
    low = np.quantile(draws, 0.025, axis=0)
    high = np.quantile(draws, 0.975, axis=0)
    result = (tuple(centers), tuple(low), tuple(high))
    return (*result, draws) if return_draws else result
