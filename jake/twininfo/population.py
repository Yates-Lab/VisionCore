"""Population selection helpers for the twininfo analysis.

Ryan's shared ``build_population`` samples reliable unit/grid-position pairs at
random.  For the production analysis here we usually want a more interpretable
choice: rank biological readout units by digital-twin performance, then assign
each selected unit one retinotopic grid position.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dill
import numpy as np
import torch

from VisionCore.paths import CACHE_DIR, VISIONCORE_ROOT

from .common import (
    DEFAULT_CCMAX_THRESHOLD,
    N_LAGS,
    OUT_SIZE,
    build_population as build_random_reliable_population,
)

from _common import PopulationReadout, SimulatedPopulation, _zero_behavior  # noqa: E402


BPS_STIM_TYPES = ("fixrsvp", "gaborium", "backimage", "gratings")
BPS_METRICS = tuple(f"{stim}_bps" for stim in BPS_STIM_TYPES)
PERFORMANCE_METRICS = ("ccnorm", "rhos", "ccabs", "ve_model", "ve_psth", "ccmax") + BPS_METRICS
POPULATION_SELECTION_MODES = ("top_performance", "random_reliable")
GRID_POSITION_MODES = ("random", "center", "full_grid")


def load_fig3_session_results(cache_path: Path | None = None) -> list[dict[str, Any]]:
    """Load Ryan's per-session digital-twin performance cache."""
    path = CACHE_DIR / "fig3_digitaltwin.pkl" if cache_path is None else Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Figure 3 digital-twin cache not found at {path}. "
            "Run Ryan's figure-3 cache generation first."
        )
    with path.open("rb") as f:
        return dill.load(f)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def load_bps_lookup(cache_path: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load optional per-neuron bits/spike metrics from Ryan's mcfarland outputs."""
    path = VISIONCORE_ROOT / "scripts" / "mcfarland_outputs_mono.pkl" if cache_path is None else Path(cache_path)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        outputs = dill.load(f)
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for out in outputs:
        session = str(out.get("sess", out.get("session", "")))
        if not session:
            continue
        bps_results = out.get("bps_results", {})
        session_rows: dict[str, np.ndarray] = {}
        for stim in BPS_STIM_TYPES:
            if stim in bps_results and "bps" in bps_results[stim]:
                session_rows[f"{stim}_bps"] = _as_numpy(bps_results[stim]["bps"]).astype(np.float64)
        if session_rows:
            lookup[session] = session_rows
    return lookup


def ranked_units_from_session_results(
    session_results: Iterable[dict[str, Any]],
    *,
    performance_metric: str = "ccnorm",
    min_performance_score: float | None = None,
    ccmax_threshold: float = DEFAULT_CCMAX_THRESHOLD,
    model_session_names: Iterable[str] | None = None,
    bps_lookup: dict[str, dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """Return model-compatible units sorted by descending performance."""
    if performance_metric not in PERFORMANCE_METRICS:
        raise ValueError(f"Unknown performance metric {performance_metric!r}. Use one of {PERFORMANCE_METRICS}.")
    if bps_lookup is None and performance_metric in BPS_METRICS:
        bps_lookup = load_bps_lookup()
    bps_lookup = {} if bps_lookup is None else bps_lookup
    allowed_sessions = None if model_session_names is None else set(str(name) for name in model_session_names)
    rows: list[dict[str, Any]] = []
    for sr in session_results:
        session = str(sr["session"])
        if allowed_sessions is not None and session not in allowed_sessions:
            continue
        neuron_mask = np.asarray(sr["neuron_mask"], dtype=np.int64)
        ccmax = np.asarray(sr["ccmax"], dtype=np.float64)
        if performance_metric in BPS_METRICS:
            bps = bps_lookup.get(session, {}).get(performance_metric)
            if bps is None:
                continue
            score = np.full(neuron_mask.shape, np.nan, dtype=np.float64)
            in_range = neuron_mask < bps.shape[0]
            score[in_range] = bps[neuron_mask[in_range]]
        else:
            score = np.asarray(sr[performance_metric], dtype=np.float64)
        valid = np.isfinite(score) & np.isfinite(ccmax) & (ccmax > float(ccmax_threshold))
        if min_performance_score is not None:
            valid &= score >= float(min_performance_score)
        for local_idx in np.flatnonzero(valid):
            row = {
                "session_name": session,
                "local_cache_index": int(local_idx),
                "original_neuron_id": int(neuron_mask[local_idx]),
                "performance_metric": performance_metric,
                "performance_score": float(score[local_idx]),
            }
            for metric in PERFORMANCE_METRICS:
                if metric in BPS_METRICS:
                    bps = bps_lookup.get(session, {}).get(metric)
                    if bps is not None and int(neuron_mask[local_idx]) < bps.shape[0]:
                        row[metric] = float(bps[int(neuron_mask[local_idx])])
                elif metric in sr:
                    values = np.asarray(sr[metric], dtype=np.float64)
                    row[metric] = float(values[local_idx]) if local_idx < values.shape[0] else np.nan
            rows.append(row)
    rows.sort(key=lambda row: (float(row["performance_score"]), float(row.get("ccmax", np.nan))), reverse=True)
    return rows


def _readout_grid_shape(model: Any, readout: torch.nn.Module) -> tuple[int, int]:
    """Probe the spatial grid produced by core + population readout."""
    device = model.device
    dtype = next(model.model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 1, N_LAGS, OUT_SIZE[0], OUT_SIZE[1], device=device, dtype=dtype)
        beh = _zero_behavior(model, dummy.shape[0], device, dtype)
        core_out = model.model.core_forward(dummy, beh)
        y = readout.to(device)(core_out[:, :, -1])
    grid_shape = (int(y.shape[-2]), int(y.shape[-1]))
    readout.cpu()
    return grid_shape


def _readout_from_ranked_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    feature_grid: tuple[int, int],
) -> tuple[PopulationReadout, list[str], list[dict[str, Any]]]:
    """Build a population readout for ranked biological twin rows."""
    session_cache: dict[str, dict[str, torch.Tensor | int]] = {}
    feat_weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    space_weights: list[torch.Tensor] = []
    session_names: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for global_unit_idx, row in enumerate(rows):
        session = str(row["session_name"])
        if session not in session_cache:
            ridx = model.names.index(session)
            ro = model.model.readouts[ridx]
            session_cache[session] = {
                "readout_index": int(ridx),
                "features": ro.features.weight.detach().cpu(),
                "bias": ro.bias.detach().cpu(),
                "space": ro.compute_gaussian_mask(
                    feature_grid[0],
                    feature_grid[1],
                    model.device,
                ).detach().cpu(),
            }
        cache = session_cache[session]
        cid = int(row["original_neuron_id"])
        feat_weights.append(cache["features"][cid : cid + 1])  # type: ignore[index]
        biases.append(cache["bias"][cid : cid + 1])  # type: ignore[index]
        space_weights.append(cache["space"][cid : cid + 1])  # type: ignore[index]
        session_names.append(session)
        metadata_row = dict(row)
        metadata_row.update({
            "global_unit_idx": int(global_unit_idx),
            "source_readout_index": int(cache["readout_index"]),
            "biological_twin_rank": int(global_unit_idx),
        })
        metadata_rows.append(metadata_row)

    return (
        PopulationReadout(
            torch.cat(feat_weights, dim=0),
            torch.cat(biases, dim=0),
            torch.cat(space_weights, dim=0),
        ),
        session_names,
        metadata_rows,
    )


def _stimulus_battery(n_frames: int, *, seed: int) -> torch.Tensor:
    """Small shared normalized stimulus battery used for response de-duplication."""
    rng = np.random.default_rng(seed)
    h, w = OUT_SIZE
    yy, xx = np.mgrid[:h, :w]
    frames = np.zeros((int(n_frames), 1, N_LAGS, h, w), dtype=np.float32)
    orientations = np.linspace(0.0, np.pi, 8, endpoint=False)
    freqs = np.asarray([3.0, 6.0, 12.0, 24.0], dtype=np.float32)
    x0 = (xx - (w - 1) / 2.0) / max(w, 1)
    y0 = (yy - (h - 1) / 2.0) / max(h, 1)
    for t in range(int(n_frames)):
        theta = float(orientations[t % len(orientations)])
        freq = float(freqs[(t // len(orientations)) % len(freqs)])
        phase = float(rng.uniform(0, 2 * np.pi))
        carrier = np.cos(2.0 * np.pi * freq * (np.cos(theta) * x0 + np.sin(theta) * y0) + phase)
        noise = rng.normal(scale=0.35, size=(h, w))
        frame = 0.18 * carrier + 0.12 * noise
        for lag in range(N_LAGS):
            frames[t, 0, lag] = frame + 0.02 * rng.normal(size=(h, w))
    return torch.from_numpy(frames)


def _center_responses_for_readout(
    model: Any,
    readout: PopulationReadout,
    *,
    n_frames: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Run the de-duplication battery and return center-grid responses (time x unit)."""
    device = model.device
    dtype = next(model.model.parameters()).dtype
    stim = _stimulus_battery(n_frames, seed=seed)
    readout = readout.to(device)
    readout.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(n_frames), int(batch_size)):
            x = stim[start : start + int(batch_size)].to(device=device, dtype=dtype)
            beh = _zero_behavior(model, x.shape[0], device, dtype)
            core_out = model.model.core_forward(x, beh)
            y = model.model.activation(readout(core_out[:, :, -1]))
            center = y[:, :, y.shape[-2] // 2, y.shape[-1] // 2]
            chunks.append(center.detach().cpu().numpy().astype(np.float32))
            del y, core_out, x
            if device.type == "cuda":
                torch.cuda.empty_cache()
    readout.cpu()
    return np.concatenate(chunks, axis=0)


def deduplicate_ranked_units(
    model: Any,
    ranked: list[dict[str, Any]],
    *,
    target_n: int,
    correlation_threshold: float,
    candidate_multiplier: float,
    battery_frames: int,
    battery_seed: int,
    batch_size: int,
    feature_grid: tuple[int, int],
) -> list[dict[str, Any]]:
    """Greedily keep top-ranked units whose battery responses are not redundant."""
    if correlation_threshold >= 1.0:
        return [dict(row, dedupe_max_corr_to_kept=np.nan, dedupe_kept=True) for row in ranked[:target_n]]
    candidate_n = min(len(ranked), max(int(target_n), int(np.ceil(float(candidate_multiplier) * int(target_n)))))
    candidates = ranked[:candidate_n]
    readout, _session_names, _metadata = _readout_from_ranked_rows(
        model,
        candidates,
        feature_grid=feature_grid,
    )
    responses = _center_responses_for_readout(
        model,
        readout,
        n_frames=int(battery_frames),
        seed=int(battery_seed),
        batch_size=int(batch_size),
    )
    z = responses - np.nanmean(responses, axis=0, keepdims=True)
    z /= np.maximum(np.nanstd(z, axis=0, keepdims=True), 1e-8)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    denom = max(z.shape[0] - 1, 1)

    kept: list[int] = []
    selected: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        if kept:
            corr = (z[:, kept].T @ z[:, idx]) / denom
            max_corr = float(np.max(corr))
            nearest = int(kept[int(np.argmax(corr))])
        else:
            max_corr = float("nan")
            nearest = -1
        if not kept or max_corr <= float(correlation_threshold):
            out = dict(row)
            out.update({
                "dedupe_kept": True,
                "dedupe_max_corr_to_kept": max_corr,
                "dedupe_nearest_candidate_index": nearest,
                "dedupe_candidate_pool_size": int(candidate_n),
            })
            kept.append(idx)
            selected.append(out)
            if len(selected) == int(target_n):
                return selected
    raise ValueError(
        f"Only found {len(selected)} unique units below correlation threshold "
        f"{correlation_threshold} from {candidate_n} candidates. Increase candidate multiplier, "
        "lower the threshold, or request fewer units."
    )


def _selected_grid_positions(
    grid_shape: tuple[int, int],
    *,
    mode: str,
    stride: int,
    rng: np.random.Generator,
    n_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_grid, w_grid = int(grid_shape[0]), int(grid_shape[1])
    if mode == "center":
        return (
            np.full((int(n_units),), h_grid // 2, dtype=np.int64),
            np.full((int(n_units),), w_grid // 2, dtype=np.int64),
        )
    if mode == "random":
        return (
            rng.integers(0, h_grid, size=int(n_units), dtype=np.int64),
            rng.integers(0, w_grid, size=int(n_units), dtype=np.int64),
        )
    if mode == "full_grid":
        rows = np.arange(0, h_grid, max(1, int(stride)), dtype=np.int64)
        cols = np.arange(0, w_grid, max(1, int(stride)), dtype=np.int64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        return rr.ravel(), cc.ravel()
    raise ValueError(f"Unknown grid position mode {mode!r}. Use one of {GRID_POSITION_MODES}.")


def build_top_performance_population(
    model: Any,
    *,
    N: int,
    rng: np.random.Generator,
    performance_metric: str = "ccnorm",
    min_performance_score: float | None = None,
    ccmax_threshold: float = DEFAULT_CCMAX_THRESHOLD,
    grid_position_mode: str = "full_grid",
    grid_stride: int = 1,
    deduplicate_units: bool = True,
    dedupe_correlation_threshold: float = 0.95,
    dedupe_candidate_multiplier: float = 5.0,
    dedupe_battery_frames: int = 96,
    dedupe_battery_seed: int = 0,
    dedupe_batch_size: int = 64,
    feature_grid: tuple[int, int] = (14, 14),
) -> tuple[SimulatedPopulation, list[dict[str, Any]]]:
    """Build a population from the top-performing reliable model units.

    The biological readout units are selected by ``performance_metric`` from
    Ryan's Fig. 3 cache.  By default every selected unit is evaluated at every
    output grid position, so ``N`` is the number of biological twins per
    retinotopic pixel and the information calculation stays convolutional.
    """
    if grid_position_mode not in GRID_POSITION_MODES:
        raise ValueError(f"Unknown grid position mode {grid_position_mode!r}. Use one of {GRID_POSITION_MODES}.")
    ranked = ranked_units_from_session_results(
        load_fig3_session_results(),
        performance_metric=performance_metric,
        min_performance_score=min_performance_score,
        ccmax_threshold=ccmax_threshold,
        model_session_names=model.names,
    )
    if int(N) > len(ranked):
        raise ValueError(
            f"Requested N={N}, but only {len(ranked)} units pass ccmax>{ccmax_threshold} "
            f"with finite {performance_metric}."
        )
    selected = deduplicate_ranked_units(
        model,
        ranked,
        target_n=int(N),
        correlation_threshold=float(dedupe_correlation_threshold),
        candidate_multiplier=float(dedupe_candidate_multiplier),
        battery_frames=int(dedupe_battery_frames),
        battery_seed=int(dedupe_battery_seed),
        batch_size=int(dedupe_batch_size),
        feature_grid=feature_grid,
    ) if deduplicate_units else ranked[: int(N)]

    readout, session_names, biological_rows = _readout_from_ranked_rows(
        model,
        selected,
        feature_grid=feature_grid,
    )
    grid_shape = _readout_grid_shape(model, readout)
    base_rows, base_cols = _selected_grid_positions(
        grid_shape,
        mode=grid_position_mode,
        stride=int(grid_stride),
        rng=rng,
        n_units=len(selected),
    )
    if grid_position_mode == "full_grid":
        n_positions = int(base_rows.shape[0])
        unit_ids = np.column_stack([
            np.repeat(np.arange(len(selected), dtype=np.int64), n_positions),
            np.tile(base_rows, len(selected)),
            np.tile(base_cols, len(selected)),
        ])
    else:
        unit_ids = np.column_stack([np.arange(len(selected), dtype=np.int64), base_rows, base_cols])

    metadata_rows: list[dict[str, Any]] = []
    for simulated_idx, (unit_idx, grid_row, grid_col) in enumerate(np.asarray(unit_ids, dtype=np.int64)):
        row = dict(biological_rows[int(unit_idx)])
        row.update({
            "simulated_unit_idx": int(simulated_idx),
            "global_unit_idx": int(unit_idx),
            "grid_row": int(grid_row),
            "grid_col": int(grid_col),
            "grid_shape_h": int(grid_shape[0]),
            "grid_shape_w": int(grid_shape[1]),
            "grid_stride": int(grid_stride),
            "n_biological_twins": int(len(selected)),
            "n_simulated_neurons": int(unit_ids.shape[0]),
            "population_selection": "top_performance",
            "grid_position_mode": grid_position_mode,
            "min_performance_score": min_performance_score,
            "deduplicate_units": bool(deduplicate_units),
            "dedupe_correlation_threshold": float(dedupe_correlation_threshold),
        })
        metadata_rows.append(row)

    population = SimulatedPopulation(
        readout=readout,
        unit_ids=unit_ids,
        session_names=session_names,
        grid_shape=grid_shape,
        N=int(unit_ids.shape[0]),
    )
    return population, metadata_rows


def build_analysis_population(
    model: Any,
    *,
    N: int,
    rng: np.random.Generator,
    selection: str = "top_performance",
    performance_metric: str = "ccnorm",
    min_performance_score: float | None = None,
    ccmax_threshold: float = DEFAULT_CCMAX_THRESHOLD,
    grid_position_mode: str = "full_grid",
    grid_stride: int = 1,
    deduplicate_units: bool = True,
    dedupe_correlation_threshold: float = 0.95,
    dedupe_candidate_multiplier: float = 5.0,
    dedupe_battery_frames: int = 96,
    dedupe_battery_seed: int = 0,
    dedupe_batch_size: int = 64,
) -> tuple[SimulatedPopulation, list[dict[str, Any]]]:
    """Build the population used by the twininfo pipeline and metadata rows."""
    if selection == "top_performance":
        return build_top_performance_population(
            model,
            N=N,
            rng=rng,
            performance_metric=performance_metric,
            min_performance_score=min_performance_score,
            ccmax_threshold=ccmax_threshold,
            grid_position_mode=grid_position_mode,
            grid_stride=grid_stride,
            deduplicate_units=deduplicate_units,
            dedupe_correlation_threshold=dedupe_correlation_threshold,
            dedupe_candidate_multiplier=dedupe_candidate_multiplier,
            dedupe_battery_frames=dedupe_battery_frames,
            dedupe_battery_seed=dedupe_battery_seed,
            dedupe_batch_size=dedupe_batch_size,
        )
    if selection == "random_reliable":
        population = build_random_reliable_population(
            model,
            N=N,
            rng=rng,
            ccmax_threshold=ccmax_threshold,
        )
        rows = [
            {
                "global_unit_idx": int(unit_idx),
                "grid_row": int(grid_row),
                "grid_col": int(grid_col),
                "session_name": str(population.session_names[int(unit_idx)]),
                "population_selection": "random_reliable",
                "performance_metric": "",
                "performance_score": np.nan,
                "min_performance_score": min_performance_score,
                "grid_position_mode": "random",
                "grid_shape_h": int(population.grid_shape[0]),
                "grid_shape_w": int(population.grid_shape[1]),
            }
            for unit_idx, grid_row, grid_col in np.asarray(population.unit_ids, dtype=np.int64)
        ]
        return population, rows
    raise ValueError(f"Unknown population selection {selection!r}. Use one of {POPULATION_SELECTION_MODES}.")
