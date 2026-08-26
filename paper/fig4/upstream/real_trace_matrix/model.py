from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .core import ROOT, sha256_file


SCRIPTS_DIR = ROOT / "scripts"
RR_POPULATION_DIR = ROOT / "ryan" / "population_information" / "rr_population"
for _path in (ROOT, SCRIPTS_DIR, RR_POPULATION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PPD = 37.50476617
N_LAGS = 32
OUT_SIZE = (151, 151)
McfarlandOutput = dict[str, Any]


def _trace_xy_to_twin_helper_order(trace_xy: np.ndarray) -> np.ndarray:
    """Pre-flip [x, y] traces because the twin stimulus helper flips internally."""
    trace = np.asarray(trace_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {trace.shape}")
    return trace[:, [1, 0]].copy()


def _standardize_uint_like(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return image.astype(np.float32)


def _eye_deg_to_norm(eye_deg: Any, *, ppd: float, img_size: tuple[int, int], torch: Any) -> Any:
    height, width = img_size
    eye_deg = eye_deg.to(dtype=torch.float32)
    x_pix = eye_deg[:, 0] * float(ppd)
    y_pix = eye_deg[:, 1] * float(ppd)
    x_norm = 2.0 * x_pix / float(width - 1)
    y_norm = -2.0 * y_pix / float(height - 1)
    return torch.stack((x_norm, y_norm), dim=-1)


def _shift_movie_with_eye(
    movie: Any,
    eye_xy: Any,
    *,
    out_size: tuple[int, int],
    scale_factor: float,
    torch: Any,
) -> Any:
    import torch.nn.functional as functional

    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
        squeeze_channel = True
    elif movie.dim() == 4:
        squeeze_channel = False
    else:
        raise ValueError(f"movie must have shape (T, H, W) or (T, C, H, W), got {tuple(movie.shape)}")

    timepoints, _, height, width = movie.shape
    device, dtype = movie.device, movie.dtype
    eye_xy = eye_xy.to(device=device, dtype=dtype)
    out_height, out_width = out_size
    x_extent = (out_width / width) * float(scale_factor)
    y_extent = (out_height / height) * float(scale_factor)
    ys = torch.linspace(-y_extent, y_extent, out_height, device=device, dtype=dtype)
    xs = torch.linspace(-x_extent, x_extent, out_width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    grid = base_grid - eye_xy.view(timepoints, 1, 1, 2)
    shifted = functional.grid_sample(movie, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return shifted[:, 0] if squeeze_channel else shifted


def _embed_time_lags(movie: Any, *, n_lags: int, torch: Any) -> Any:
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
    timepoints, channels, height, width = movie.shape
    out_frames = timepoints - int(n_lags) + 1
    lagged = torch.zeros(
        out_frames,
        channels,
        int(n_lags),
        height,
        width,
        dtype=movie.dtype,
        device=movie.device,
    )
    for lag in range(int(n_lags)):
        lagged[:, :, lag] = movie[int(n_lags) - 1 - lag : timepoints - lag]
    return lagged


def _expand_trace_to_model_grid(
    eyepos: Any,
    *,
    temporal_factor: int,
    supervision_phase: int,
    torch: Any,
) -> tuple[Any, Any]:
    """Interpolate a scored trace onto the model's native temporal grid.

    Figure 4 stores one eye-position sample per scored 120-Hz response bin.
    The Dekel model instead consumes native 240-Hz retinal movies and scores
    the odd member of each two-frame supervision pair.  Treat each stored
    sample as the eye position at its scored endpoint, linearly interpolate
    the intervening native samples, and hold the first/last position outside
    the observed interval.  The returned endpoint indices select exactly one
    native model output for every input trace sample.

    ``temporal_factor=1`` is bit-for-bit the legacy 120-Hz path.
    """
    factor = int(temporal_factor)
    phase = int(supervision_phase)
    if factor < 1:
        raise ValueError(f"temporal_factor must be >= 1, got {factor}.")
    if not 0 <= phase < factor:
        raise ValueError(
            f"supervision_phase must be in [0, {factor}), got {phase}."
        )
    if eyepos.ndim != 2 or int(eyepos.shape[1]) != 2:
        raise ValueError(f"Expected eyepos shape (T, 2), got {tuple(eyepos.shape)}.")
    n_scored = int(eyepos.shape[0])
    if n_scored < 1:
        raise ValueError("eyepos must contain at least one scored sample.")
    if factor == 1:
        endpoints = torch.arange(n_scored, device=eyepos.device, dtype=torch.long)
        return eyepos, endpoints

    # Stored samples are anchors at native indices k*factor + phase.  Linear
    # interpolation is expressed directly in index space to avoid depending on
    # align_corners conventions from a generic resampler.
    native_length = n_scored * factor
    native_index = torch.arange(
        native_length,
        device=eyepos.device,
        dtype=eyepos.dtype,
    )
    anchor_position = (native_index - float(phase)) / float(factor)
    lo = torch.floor(anchor_position).to(dtype=torch.long)
    hi = lo + 1
    alpha = (anchor_position - lo.to(dtype=anchor_position.dtype)).unsqueeze(1)
    lo_clamped = lo.clamp(0, n_scored - 1)
    hi_clamped = hi.clamp(0, n_scored - 1)
    expanded = eyepos[lo_clamped] * (1.0 - alpha) + eyepos[hi_clamped] * alpha
    endpoints = (
        torch.arange(n_scored, device=eyepos.device, dtype=torch.long) * factor
        + phase
    )
    return expanded, endpoints


def _trace_on_output_grid(
    trace_xy: np.ndarray,
    *,
    source_rate_hz: int,
    output_rate_hz: int,
    torch: Any,
) -> np.ndarray:
    """Endpoint-interpolate a retained trace onto the model output grid.

    The recovered BackImage trace bank is sampled at 120 Hz.  A true native
    240-Hz twin therefore needs two output-grid eye positions per retained
    sample, whereas both the historical 120-Hz twin and a 240-input/120-output
    twin consume the retained trace unchanged at their output boundary.
    """
    source_rate = int(source_rate_hz)
    output_rate = int(output_rate_hz)
    if source_rate < 1 or output_rate < 1:
        raise ValueError(
            f"Trace/output rates must be positive, got {source_rate} and {output_rate}."
        )
    if output_rate < source_rate or output_rate % source_rate:
        raise ValueError(
            "Figure 4 replay requires the model output rate to be an integer "
            f"multiple of the retained trace rate; got {source_rate} -> "
            f"{output_rate} Hz."
        )
    trace = np.asarray(trace_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {trace.shape}.")
    factor = output_rate // source_rate
    if factor == 1:
        return trace
    expanded, _ = _expand_trace_to_model_grid(
        torch.from_numpy(trace),
        temporal_factor=factor,
        supervision_phase=factor - 1,
        torch=torch,
    )
    return expanded.cpu().numpy().astype(np.float32, copy=False)


def make_counterfactual_stim(
    full_stack: np.ndarray,
    eyepos: Any,
    *,
    ppd: float = PPD,
    scale_factor: float = 1.0,
    n_lags: int = N_LAGS,
    out_size: tuple[int, int] = OUT_SIZE,
    temporal_factor: int = 1,
    supervision_phase: int = 0,
) -> Any:
    """Reconstruct a causal gaze-contingent lag tensor for a trace snippet.

    The replay matrix stores only the scored snippet, not its preceding gaze
    history.  Hold the initial gaze position for the unavailable history.  This
    yields exactly one model output per supplied trace sample for any history
    length, including the 60-frame Dekel core.  It also avoids the legacy
    helper's use of early *future* samples as the prefix.
    """
    import torch

    native_eyepos, scored_endpoints = _expand_trace_to_model_grid(
        eyepos,
        temporal_factor=int(temporal_factor),
        supervision_phase=int(supervision_phase),
        torch=torch,
    )
    eye_norm = _eye_deg_to_norm(
        torch.fliplr(native_eyepos),
        ppd=float(ppd),
        img_size=full_stack.shape[1:3],
        torch=torch,
    )
    history_frames = max(0, int(n_lags) - 1)
    prefix = eye_norm[:1].repeat(history_frames, 1)
    padded_eye = torch.cat((prefix, eye_norm), dim=0)
    if int(full_stack.shape[0]) < int(padded_eye.shape[0]):
        raise ValueError(
            f"full_stack has {full_stack.shape[0]} frames but causal embedding "
            f"requires {padded_eye.shape[0]}."
        )
    eye_movie = _shift_movie_with_eye(
        torch.from_numpy(full_stack[: padded_eye.shape[0]]).float(),
        padded_eye,
        out_size=out_size,
        scale_factor=float(scale_factor),
        torch=torch,
    )
    native_lagged = _embed_time_lags(eye_movie, n_lags=int(n_lags), torch=torch)
    return native_lagged.index_select(0, scored_endpoints)


def load_mcfarland_outputs(path: Path | None = None) -> tuple[list[McfarlandOutput], Path]:
    candidates = [Path(path)] if path is not None else [
        SCRIPTS_DIR / "mcfarland_outputs_mono.pkl",
        SCRIPTS_DIR / "mcfarland_outputs.pkl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("rb") as handle:
            try:
                import dill

                obj = dill.load(handle)
            except Exception:
                handle.seek(0)
                obj = pickle.load(handle)
        if not isinstance(obj, list):
            raise TypeError(f"Expected a list of McFarland outputs in {candidate}, got {type(obj).__name__}.")
        return obj, candidate
    raise FileNotFoundError("McFarland outputs not found. Tried: " + ", ".join(str(path) for path in candidates))


def _checkpoint_hparams(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if not isinstance(hparams, dict):
        raise TypeError(f"Checkpoint hyper_parameters must be a dict, got {type(hparams).__name__}.")
    return dict(hparams)


def load_pinned_multidataset_model(
    *,
    checkpoint_path: Path,
    dataset_configs: Path,
    device: str,
    strict: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from training.pl_modules.multidataset_model import MultiDatasetModel

    checkpoint_path = Path(checkpoint_path)
    dataset_configs = Path(dataset_configs)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but torch.cuda.is_available() is false.")
    hparams = _checkpoint_hparams(checkpoint_path)
    model = MultiDatasetModel.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        strict=bool(strict),
        model_cfg=hparams.get("model_cfg"),
        cfg_dir=str(dataset_configs),
        lr=float(hparams.get("lr", 1e-3)),
        wd=float(hparams.get("wd", 0.0)),
        max_ds=int(hparams.get("max_ds", 30)),
        # A trained checkpoint is self-contained.  Replaying its historical
        # warm-start here is redundant and makes evaluation depend on a parent
        # checkpoint still existing at the original path.
        pretrained_checkpoint=None,
        freeze_vision=bool(hparams.get("freeze_vision", False)),
        compile_model=False,
        model_config_dict=hparams.get("model_config_dict"),
    )
    model = model.to(device).eval()
    convnet = getattr(model.model, "convnet", None)
    if convnet is not None and hasattr(convnet, "use_checkpointing"):
        convnet.use_checkpointing = False
    model_info = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_configs": dataset_configs,
        "dataset_configs_sha256": sha256_file(dataset_configs),
        "device": str(device),
        "strict": bool(strict),
        "hparams": {
            "model_cfg": hparams.get("model_cfg"),
            "lr": hparams.get("lr"),
            "wd": hparams.get("wd"),
            "max_ds": hparams.get("max_ds"),
            "pretrained_checkpoint": hparams.get("pretrained_checkpoint"),
            "freeze_vision": hparams.get("freeze_vision"),
            "compile_model": hparams.get("compile_model"),
            "has_model_config_dict": hparams.get("model_config_dict") is not None,
        },
        "model_names": [str(name) for name in getattr(model, "names", [])],
    }
    return model, model_info


def load_spatial_readout(model: Any, outputs: list[McfarlandOutput], *, device: str) -> tuple[Any, list[dict[str, Any]]]:
    from scripts.spatial_info import get_spatial_readout

    readout, unit_rows = get_spatial_readout(model, outputs, return_unit_rows=True)
    readout = readout.to(device).eval()
    return readout, list(unit_rows)


def infer_model_history_frames(model: Any, dataset_configs: Path) -> int:
    """Infer the checkpoint's stimulus history without changing legacy pins."""
    convnet = getattr(model.model, "convnet", None)
    temporal_support = getattr(convnet, "temporal_support", None)
    if temporal_support is not None:
        return int(temporal_support)
    # The recovered Figure-4 ConvGRU replay is pinned to 32 frames.  Its
    # historical dataset YAML lists 0..32 because one extra aligned endpoint
    # was carried by the loader; treating that list length as model history
    # introduces an off-by-one and breaks the verified replay.
    return int(N_LAGS)


def infer_model_time_contract(model: Any, dataset_configs: Path) -> dict[str, int]:
    """Return native/scored rates and endpoint phase for a loaded checkpoint."""
    config = yaml.safe_load(Path(dataset_configs).read_text()) or {}
    sampling = config.get("sampling", {}) or {}
    supervision = config.get("supervision", {}) or {}
    model_rate = getattr(getattr(model, "model", None), "sampling_rate", None)
    # The prepared dataset grid is authoritative.  Older model YAMLs inherited
    # a 240-Hz constructor default even when their dataset was downsampled to
    # 120 Hz, so preferring ``model.sampling_rate`` would silently alter the
    # legacy Figure-4 replay.
    input_rate = int(sampling.get("target_rate", model_rate or 120))
    output_rate = int(supervision.get("target_rate", sampling.get("target_rate", input_rate)))
    if input_rate < output_rate or input_rate % output_rate != 0:
        raise ValueError(
            "Figure 4 replay requires an integer native-to-scored rate ratio; "
            f"got input_rate={input_rate}, output_rate={output_rate}."
        )
    factor = input_rate // output_rate
    phase = int(supervision.get("phase", 0)) if factor > 1 else 0
    if not 0 <= phase < factor:
        raise ValueError(
            f"Invalid supervision phase {phase} for temporal factor {factor}."
        )
    return {
        "input_rate_hz": input_rate,
        "output_rate_hz": output_rate,
        "temporal_factor": factor,
        "supervision_phase": phase,
    }


def load_population_view(*, spec_dir: Path, version_name: str) -> tuple[Any, Any, Path | None, Path | None]:
    from redundancy_resolved_v1_population import apply_population_view, load_population_view, resolve_population_spec_paths

    spec_npz, spec_json = resolve_population_spec_paths(spec_dir, version_name=version_name)
    view = load_population_view(spec_dir, version_name=version_name)
    return view, apply_population_view, spec_npz, spec_json


def adapt_population_view_to_available(
    population_view: Any,
    canonical_unit_rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Adapt a pinned RR view when a checkpoint lacks canonical readout cells.

    The canonical 756-channel coordinate system is preserved by inactive
    placeholders in the spatial readout.  For a missing medoid, select the
    highest-ccnorm available cell from the same saved redundancy cluster.  A
    cluster with no modeled member remains an all-zero (inactive) output row,
    which contributes neither expected spikes nor SSI to population metrics.
    """
    membership = getattr(population_view, "membership", None)
    if membership is None:
        report = {
            "adapted": False,
            "canonical_channels": int(len(canonical_unit_rows)),
            "available_channels": int(len(canonical_unit_rows)),
            "missing_channels": 0,
            "substitutions": [],
            "inactive_units": [],
        }
        return population_view, report

    membership = np.asarray(membership, dtype=np.float32)
    if membership.ndim != 2:
        raise ValueError(f"Population membership must be 2-D, got {membership.shape}.")
    if membership.shape[1] != len(canonical_unit_rows):
        raise ValueError(
            f"Population view expects {membership.shape[1]} channels, "
            f"but the canonical readout has {len(canonical_unit_rows)}."
        )

    available = np.asarray(
        [bool(row.get("available", True)) for row in canonical_unit_rows],
        dtype=bool,
    )
    ccnorm = np.asarray(
        [float(row.get("ccnorm", float("nan"))) for row in canonical_unit_rows],
        dtype=np.float64,
    )
    cluster_membership = getattr(population_view, "cluster_membership", None)
    cluster_membership = (
        None
        if cluster_membership is None
        else np.asarray(cluster_membership, dtype=np.float32)
    )
    if cluster_membership is not None and cluster_membership.shape != membership.shape:
        raise ValueError(
            "Population cluster_membership shape does not match membership: "
            f"{cluster_membership.shape} vs {membership.shape}."
        )

    adapted = membership.copy()
    substitutions: list[dict[str, Any]] = []
    inactive_units: list[int] = []
    for unit_index in range(adapted.shape[0]):
        original_channels = np.flatnonzero(np.abs(membership[unit_index]) > 1e-7)
        if original_channels.size and np.all(available[original_channels]):
            continue

        available_original = original_channels[available[original_channels]]
        if available_original.size:
            # Mean-pooled views can simply discard unavailable members and
            # renormalize the surviving saved weights.
            adapted[unit_index] = 0.0
            weights = membership[unit_index, available_original].astype(np.float64)
            denom = float(weights.sum())
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                weights = np.full(available_original.size, 1.0 / available_original.size)
            else:
                weights = weights / denom
            adapted[unit_index, available_original] = weights.astype(np.float32)
            substitutions.append(
                {
                    "unit_index": int(unit_index),
                    "reason": "drop_unavailable_pool_members",
                    "original_channels": [int(ch) for ch in original_channels],
                    "selected_channels": [int(ch) for ch in available_original],
                }
            )
            continue

        candidates = np.zeros((0,), dtype=np.int64)
        if cluster_membership is not None:
            candidates = np.flatnonzero(
                (np.abs(cluster_membership[unit_index]) > 1e-7) & available
            )
        if candidates.size:
            scores = np.where(np.isfinite(ccnorm[candidates]), ccnorm[candidates], -np.inf)
            replacement_channel = int(candidates[int(np.argmax(scores))])
            adapted[unit_index] = 0.0
            adapted[unit_index, replacement_channel] = 1.0
            substitutions.append(
                {
                    "unit_index": int(unit_index),
                    "reason": "missing_representative",
                    "original_channels": [int(ch) for ch in original_channels],
                    "selected_channels": [replacement_channel],
                    "replacement_ccnorm": float(ccnorm[replacement_channel]),
                }
            )
        else:
            adapted[unit_index] = 0.0
            inactive_units.append(int(unit_index))
            substitutions.append(
                {
                    "unit_index": int(unit_index),
                    "reason": "no_available_cluster_member",
                    "original_channels": [int(ch) for ch in original_channels],
                    "selected_channels": [],
                }
            )

    report = {
        "adapted": bool(substitutions),
        "canonical_channels": int(available.size),
        "available_channels": int(available.sum()),
        "missing_channels": int((~available).sum()),
        "substitutions": substitutions,
        "inactive_units": inactive_units,
        "active_units": int(adapted.shape[0] - len(inactive_units)),
    }
    meta = dict(getattr(population_view, "meta", {}) or {})
    meta["checkpoint_availability_adaptation"] = report
    return replace(population_view, membership=adapted, meta=meta), report


def population_unit_rows(population_view: Any, canonical_unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    membership = np.asarray(population_view.membership, dtype=np.float32)
    cluster_membership = getattr(population_view, "cluster_membership", None)
    cluster_membership = None if cluster_membership is None else np.asarray(cluster_membership, dtype=np.float32)
    representatives = {
        int(row["rep_idx"]): row
        for row in population_view.meta.get("representatives", [])
        if isinstance(row, dict) and "rep_idx" in row
    }
    rows: list[dict[str, Any]] = []
    for unit_index in range(int(population_view.n_units)):
        weights = membership[unit_index] if membership.ndim == 2 else np.zeros((0,), dtype=np.float32)
        selected = np.flatnonzero(np.abs(weights) > 1e-7)
        input_channel = int(selected[np.argmax(np.abs(weights[selected]))]) if selected.size else None
        member_channels: list[int] = []
        if cluster_membership is not None and cluster_membership.ndim == 2 and unit_index < cluster_membership.shape[0]:
            member_channels = [int(ch) for ch in np.flatnonzero(cluster_membership[unit_index] > 0)]
        rep_meta = representatives.get(unit_index, {})
        if not member_channels and isinstance(rep_meta.get("members"), list):
            member_channels = [int(ch) for ch in rep_meta["members"]]

        row: dict[str, Any] = {
            "unit_index": int(unit_index),
            "unit_label": f"u{unit_index:03d}",
            "population_version": str(population_view.name),
            "population_input_channel": input_channel,
            "population_member_count": int(len(member_channels)),
            "population_member_channels": ",".join(str(ch) for ch in member_channels),
            "population_active": bool(input_channel is not None),
        }
        for key in ("group_id", "rep_channel", "rep_idx", "pooling_mode"):
            if key in rep_meta:
                row[f"population_{key}"] = rep_meta[key]
        if input_channel is not None and 0 <= input_channel < len(canonical_unit_rows):
            for key, value in canonical_unit_rows[input_channel].items():
                row[f"canonical_{key}"] = value
        rows.append(row)
    return rows


@dataclass
class RealTraceMatrixScorer:
    model: Any
    readout: Any
    population_view: Any
    apply_population_view: Any
    canonical_unit_rows: list[dict[str, Any]]
    unit_rows: list[dict[str, Any]]
    torch: Any
    device: str
    n_lags: int
    input_rate_hz: int
    output_rate_hz: int
    temporal_factor: int
    supervision_phase: int
    out_size: tuple[int, int]
    provenance: dict[str, Any]

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        dataset_configs: Path,
        population_spec_dir: Path,
        device: str,
        population_version: str,
        strict: bool = True,
        mcfarland_outputs: Path | None = None,
    ) -> "RealTraceMatrixScorer":
        import torch

        if not str(population_version).strip():
            raise ValueError("population_version is required")
        resolved_population_version = str(population_version)

        model, model_info = load_pinned_multidataset_model(
            checkpoint_path=Path(checkpoint_path),
            dataset_configs=Path(dataset_configs),
            device=str(device),
            strict=bool(strict),
        )
        outputs, outputs_path = load_mcfarland_outputs(mcfarland_outputs)
        readout, canonical_unit_rows = load_spatial_readout(model, outputs, device=str(device))
        n_lags = infer_model_history_frames(model, Path(dataset_configs))
        time_contract = infer_model_time_contract(model, Path(dataset_configs))
        out_size = tuple(int(value) for value in OUT_SIZE)
        population_view, apply_population_view, spec_npz, spec_json = load_population_view(
            spec_dir=Path(population_spec_dir),
            version_name=resolved_population_version,
        )
        if int(population_view.input_channels) != len(canonical_unit_rows):
            raise ValueError(
                f"Population view expects {population_view.input_channels} channels, "
                f"but the canonical readout has {len(canonical_unit_rows)}."
            )
        population_view, availability_report = adapt_population_view_to_available(
            population_view,
            canonical_unit_rows,
        )
        population_meta = dict(getattr(population_view, "meta", {}) or {})
        if population_meta.get("pooling_mode") == "exact_identity":
            membership = np.asarray(population_view.membership, dtype=np.float32)
            nonzero = np.abs(membership) > 1e-7
            exact_rows = bool(
                membership.ndim == 2
                and np.all(nonzero.sum(axis=1) == 1)
                and np.allclose(membership[nonzero], 1.0, atol=0.0, rtol=0.0)
                and len(np.unique(np.argmax(nonzero, axis=1)))
                == membership.shape[0]
            )
            if bool(availability_report.get("adapted", False)) or not exact_rows:
                raise RuntimeError(
                    "exact_identity population contract was altered or is not a "
                    "one-to-one channel selection; substitutions and pooling are forbidden"
                )
        unit_rows = population_unit_rows(population_view, canonical_unit_rows)
        provenance = {
            "model": model_info,
            "mcfarland_outputs_path": outputs_path,
            "mcfarland_outputs_sha256": sha256_file(outputs_path),
            "canonical_readout_n_units": int(len(canonical_unit_rows)),
            "population_readout": {
                "includes_phase_branch": bool(
                    getattr(readout, "has_phase_branch", False)
                ),
                "deep_rank": int(getattr(readout, "rank", 1)),
                "phase_rank": (
                    int(getattr(readout, "phase_rank"))
                    if getattr(readout, "has_phase_branch", False)
                    else None
                ),
                "phase_spatial_stride": (
                    int(getattr(readout, "phase_stride"))
                    if getattr(readout, "has_phase_branch", False)
                    else None
                ),
            },
            "population_version": str(population_view.name),
            "population_n_units": int(population_view.n_units),
            "population_checkpoint_availability": availability_report,
            "population_spec_npz": spec_npz,
            "population_spec_npz_sha256": sha256_file(spec_npz) if spec_npz is not None else None,
            "population_spec_json": spec_json,
            "population_spec_json_sha256": (
                sha256_file(spec_json) if spec_json is not None and spec_json.exists() else None
            ),
            "population_contract": {
                "pooling_mode": population_meta.get("pooling_mode"),
                "rr_clustering": population_meta.get("rr_clustering"),
                "identity_key": population_meta.get("identity_key"),
                "selection_gate": population_meta.get("selection_gate"),
            },
            "stimulus": {
                "ppd": PPD,
                "model_history_frames": int(n_lags),
                "model_input_rate_hz": int(time_contract["input_rate_hz"]),
                "model_output_rate_hz": int(time_contract["output_rate_hz"]),
                "native_frames_per_scored_sample": int(time_contract["temporal_factor"]),
                "supervision_phase": int(time_contract["supervision_phase"]),
                "model_history_seconds": (
                    float(n_lags) / float(time_contract["input_rate_hz"])
                ),
                "out_size": list(out_size),
                "readout_mask_size": int(readout.space_weights.shape[-1]),
                "trace_xy_convention": "input trace is [x_deg, y_deg]; scorer pre-flips for Ryan's helper convention",
                "history_prefix_policy": "hold_initial_gaze_for_n_lags_minus_1",
                "history_is_causal": True,
            },
        }
        return cls(
            model=model,
            readout=readout,
            population_view=population_view,
            apply_population_view=apply_population_view,
            canonical_unit_rows=canonical_unit_rows,
            unit_rows=unit_rows,
            torch=torch,
            device=str(device),
            n_lags=int(n_lags),
            input_rate_hz=int(time_contract["input_rate_hz"]),
            output_rate_hz=int(time_contract["output_rate_hz"]),
            temporal_factor=int(time_contract["temporal_factor"]),
            supervision_phase=int(time_contract["supervision_phase"]),
            out_size=out_size,
            provenance=provenance,
        )

    @property
    def n_units(self) -> int:
        return int(self.population_view.n_units)

    @property
    def rr_unit_rows(self) -> list[dict[str, Any]]:
        """Deprecated compatibility alias for pre-exact-CID callers."""
        return self.unit_rows

    def _zero_behavior(self, batch_size: int, dtype: Any) -> Any | None:
        modulator = getattr(self.model.model, "modulator", None)
        behavior_dim = getattr(modulator, "behavior_dim", None) if modulator is not None else None
        if behavior_dim is None:
            return None
        return self.torch.zeros(int(batch_size), int(behavior_dim), device=self.device, dtype=dtype)

    def _compute_rate_map(self, stim: Any) -> Any:
        from scripts.spatial_info import compute_rate_map

        dtype = next(self.model.model.parameters()).dtype
        behavior = self._zero_behavior(int(stim.shape[0]), dtype)
        return compute_rate_map(self.model, self.readout, stim, behavior=behavior)

    def score_traces_for_patch(
        self,
        patch: np.ndarray,
        traces: list[np.ndarray],
        *,
        trace_batch_size: int,
        frame_batch_size: int,
        n_timepoints: int,
        bin_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not traces:
            return (
                np.zeros((0, self.n_units), dtype=np.float32),
                np.zeros((0, self.n_units), dtype=np.float32),
                np.zeros((0, self.n_units), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        self.model.model.eval()
        self.readout.eval()
        image = _standardize_uint_like(patch)
        n_traces = len(traces)
        n_units = self.n_units
        unit_expected = np.zeros((n_traces, n_units), dtype=np.float64)
        unit_numer = np.zeros((n_traces, n_units), dtype=np.float64)
        unit_rate_sum = np.zeros((n_traces, n_units), dtype=np.float64)
        unit_frame_count = np.zeros((n_traces,), dtype=np.int64)
        trace_batch_size = max(1, int(trace_batch_size))
        frame_batch_size = max(1, int(frame_batch_size))
        n_timepoints = int(n_timepoints)
        if not np.isfinite(float(bin_seconds)) or float(bin_seconds) <= 0.0:
            raise ValueError(f"Trace bin_seconds must be positive, got {bin_seconds}.")
        source_rate_hz = int(round(1.0 / float(bin_seconds)))
        if not math.isclose(
            float(bin_seconds), 1.0 / float(source_rate_hz), rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"Trace bin_seconds={bin_seconds} does not specify an integer "
                "source sampling rate."
            )
        if self.output_rate_hz < source_rate_hz or self.output_rate_hz % source_rate_hz:
            raise ValueError(
                "Model output rate must be an integer multiple of the retained "
                f"trace rate; got {source_rate_hz} -> {self.output_rate_hz} Hz."
            )
        scored_per_source = self.output_rate_hz // source_rate_hz
        scored_timepoints = n_timepoints * scored_per_source
        output_bin_seconds = 1.0 / float(self.output_rate_hz)

        with self.torch.no_grad():
            for trace_start in range(0, n_traces, trace_batch_size):
                trace_chunk = traces[trace_start : trace_start + trace_batch_size]
                stims = []
                frame_to_trace: list[int] = []
                for local_idx, trace in enumerate(trace_chunk):
                    arr = np.asarray(trace, dtype=np.float32)
                    if arr.shape != (n_timepoints, 2):
                        raise ValueError(
                            f"Trace has shape {arr.shape}; expected ({n_timepoints}, 2) "
                            "on the retained source grid."
                        )
                    output_grid_trace = _trace_on_output_grid(
                        arr,
                        source_rate_hz=source_rate_hz,
                        output_rate_hz=self.output_rate_hz,
                        torch=self.torch,
                    )
                    native_timepoints = int(output_grid_trace.shape[0]) * int(self.temporal_factor)
                    full_stack = np.broadcast_to(
                        image[None, :, :],
                        (native_timepoints + self.n_lags + 1, *image.shape),
                    ).copy()
                    eye = self.torch.from_numpy(
                        _trace_xy_to_twin_helper_order(output_grid_trace)
                    )
                    stim = make_counterfactual_stim(
                        full_stack,
                        eye,
                        ppd=PPD,
                        scale_factor=1.0,
                        n_lags=self.n_lags,
                        out_size=self.out_size,
                        temporal_factor=self.temporal_factor,
                        supervision_phase=self.supervision_phase,
                    )
                    length = int(stim.shape[0])
                    if length == scored_timepoints:
                        trace_ids = [trace_start + local_idx] * length
                    elif length == scored_timepoints + 1:
                        trace_ids = [-1] + [trace_start + local_idx] * scored_timepoints
                    else:
                        raise ValueError(
                            f"Twin response has {length} frames for a "
                            f"{n_timepoints}-sample trace at {source_rate_hz} Hz; "
                            f"expected {scored_timepoints} or {scored_timepoints + 1} "
                            f"outputs at {self.output_rate_hz} Hz."
                        )
                    frame_to_trace.extend(trace_ids)
                    stims.append((stim - 127.0) / 255.0)

                stim_all = self.torch.cat(stims, dim=0)
                frame_to_trace_arr = np.asarray(frame_to_trace, dtype=np.int64)
                for t_start in range(0, int(stim_all.shape[0]), frame_batch_size):
                    t_end = min(t_start + frame_batch_size, int(stim_all.shape[0]))
                    x = stim_all[t_start:t_end].to(self.device)
                    full_map = self._compute_rate_map(x)
                    rr_map = self.apply_population_view(full_map, self.population_view)
                    rr_map = rr_map.clamp_min(0.0).to(self.torch.float64)
                    flat = rr_map.reshape(rr_map.shape[0], rr_map.shape[1], -1)
                    rbar = flat.mean(dim=2)
                    gain = flat / (rbar[..., None] + 1e-8)
                    unit_bits_t = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
                    rbar_cpu = rbar.detach().cpu().numpy()
                    bits_cpu = unit_bits_t.detach().cpu().numpy()
                    ids = frame_to_trace_arr[t_start:t_end]
                    for trace_idx in np.unique(ids[ids >= 0]):
                        mask = ids == int(trace_idx)
                        rb = rbar_cpu[mask]
                        ub = bits_cpu[mask]
                        weights = rb * output_bin_seconds
                        unit_expected[int(trace_idx)] += np.sum(weights, axis=0)
                        unit_numer[int(trace_idx)] += np.sum(ub * weights, axis=0)
                        unit_rate_sum[int(trace_idx)] += np.sum(rb, axis=0)
                        unit_frame_count[int(trace_idx)] += int(np.count_nonzero(mask))
                    del x, full_map, rr_map, flat, rbar, gain, unit_bits_t
                del stims, stim_all
                # Keep the CUDA caching allocator warm across frame batches.
                # Emptying it in the inner loop turns a large factorial replay
                # into allocator-bound work without reducing the live tensor
                # footprint. A single release at the trace-chunk boundary is
                # sufficient for coexistence with other GPU jobs.
                if str(self.device).startswith("cuda"):
                    self.torch.cuda.empty_cache()

        unit_bits = np.divide(unit_numer, np.maximum(unit_expected, 1e-8)).astype(np.float32)
        unit_mean_rate = np.divide(unit_rate_sum, np.maximum(unit_frame_count[:, None], 1)).astype(np.float32)
        population_numer = np.sum(unit_numer, axis=1)
        population_denom = np.sum(unit_expected, axis=1)
        population_bits = np.divide(population_numer, np.maximum(population_denom, 1e-8)).astype(np.float32)
        return unit_bits, unit_expected.astype(np.float32), unit_mean_rate, population_bits


class _LegacyStaticStimulusContract:
    """Expose the stimulus-helper protocol used by recovered Figure 4 code.

    The production instantaneous-map script constructs a static movie with the
    historical 120-Hz length before handing it to ``common.make_counterfactual_stim``.
    A mixed-rate Dekel twin needs twice as many native movie frames.  Because
    this path renders a *static patch*, extending the stack is exact; the gaze
    trace, not the source image, supplies all temporal variation.
    """

    def __init__(self, scorer: RealTraceMatrixScorer):
        self._scorer = scorer
        self.N_LAGS = int(scorer.n_lags)
        self.PPD = float(PPD)
        self.OUT_SIZE = tuple(int(value) for value in scorer.out_size)

    def make_counterfactual_stim(
        self,
        full_stack: np.ndarray,
        eyepos: Any,
        *,
        ppd: float,
        scale_factor: float,
        n_lags: int,
        out_size: tuple[int, int],
    ) -> Any:
        stack = np.asarray(full_stack)
        required_frames = (
            int(eyepos.shape[0]) * int(self._scorer.temporal_factor)
            + int(n_lags)
            - 1
        )
        if int(stack.shape[0]) < required_frames:
            stack = np.broadcast_to(
                stack[:1],
                (required_frames, *stack.shape[1:]),
            ).copy()
        return make_counterfactual_stim(
            stack,
            eyepos,
            ppd=float(ppd),
            scale_factor=float(scale_factor),
            n_lags=int(n_lags),
            out_size=tuple(int(value) for value in out_size),
            temporal_factor=int(self._scorer.temporal_factor),
            supervision_phase=int(self._scorer.supervision_phase),
        )


class LegacyCanonicalTwinScorerAdapter:
    """Run the recovered production unit-map analysis with a selected model.

    This intentionally implements the narrow ``CanonicalTwinScorer`` protocol
    consumed by ``rate_map_for_trace``.  The surrounding recovered analysis --
    movie selection, trace manipulation, exact-unit projection, SSI calculation,
    and plotting -- remains unchanged.
    """

    def __init__(
        self,
        scorer: RealTraceMatrixScorer,
        *,
        batch_size: int,
        empty_cache_every_batch: bool = False,
    ) -> None:
        self._scorer = scorer
        self.common = _LegacyStaticStimulusContract(scorer)
        self.batch_size = max(1, int(batch_size))
        self.empty_cache_every_batch = bool(empty_cache_every_batch)
        self.torch = scorer.torch
        self.device = str(scorer.device)
        self.n_units = int(len(scorer.canonical_unit_rows))
        self.population_source = "selected_model_canonical_shared_population_readout"
        self.model_family = str(
            scorer.provenance.get("model", {}).get("model_family", "selected_model")
        )
        self.model_names = [str(name) for name in scorer.model.names]
        self.provenance = scorer.provenance

    def rate_map_for_trace(self, patch: np.ndarray, trace: np.ndarray) -> np.ndarray:
        """Render canonical spatial maps without the legacy ``T >= n_lags`` guard.

        The missing pre-trace history is filled causally by holding the first
        gaze position, so a 32- or 40-sample Figure 4 trace is valid even for a
        60-frame native-history model.
        """
        image = _standardize_uint_like(patch)
        trace_arr = np.asarray(trace, dtype=np.float32)
        required_frames = (
            int(trace_arr.shape[0]) * int(self._scorer.temporal_factor)
            + int(self._scorer.n_lags)
            - 1
        )
        full_stack = np.broadcast_to(
            image[None, :, :],
            (required_frames, *image.shape),
        ).copy()
        eye = self.torch.from_numpy(_trace_xy_to_twin_helper_order(trace_arr))
        stim = make_counterfactual_stim(
            full_stack,
            eye,
            ppd=PPD,
            scale_factor=1.0,
            n_lags=int(self._scorer.n_lags),
            out_size=tuple(int(value) for value in self._scorer.out_size),
            temporal_factor=int(self._scorer.temporal_factor),
            supervision_phase=int(self._scorer.supervision_phase),
        )
        rate_map = self._compute_rate_map_batched((stim - 127.0) / 255.0)
        out = rate_map.detach().cpu().numpy().astype(np.float32, copy=False)
        del stim, rate_map
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return out

    def _compute_rate_map_batched(self, stim: Any) -> Any:
        chunks = []
        self._scorer.model.model.eval()
        self._scorer.readout.eval()
        with self.torch.no_grad():
            for start in range(0, int(stim.shape[0]), self.batch_size):
                stop = min(start + self.batch_size, int(stim.shape[0]))
                x = stim[start:stop].to(self.device)
                rate_map = self._scorer._compute_rate_map(x)
                chunks.append(rate_map.detach().cpu())
                del x, rate_map
                if self.empty_cache_every_batch and self.device.startswith("cuda"):
                    self.torch.cuda.empty_cache()
        return self.torch.cat(chunks, dim=0)
