from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def _scored_trace_ids(response_length: int, *, n_timepoints: int, trace_index: int) -> list[int]:
    """Map lagged outputs to a trace, excluding the one pre-score burn-in output."""
    if int(response_length) == int(n_timepoints):
        return [int(trace_index)] * int(n_timepoints)
    if int(response_length) == int(n_timepoints) + 1:
        return [-1] + [int(trace_index)] * int(n_timepoints)
    raise ValueError(
        f"Twin response has {int(response_length)} frames for a {int(n_timepoints)}-sample trace; "
        "expected T or T+1."
    )


def make_counterfactual_stim(
    full_stack: np.ndarray,
    eyepos: Any,
    *,
    ppd: float = PPD,
    scale_factor: float = 1.0,
    n_lags: int = N_LAGS,
    out_size: tuple[int, int] = OUT_SIZE,
) -> Any:
    """Reconstruct the same gaze-contingent lag tensor as Ryan's Fig. 4 helper."""
    import torch

    eye_norm = _eye_deg_to_norm(torch.fliplr(eyepos), ppd=float(ppd), img_size=full_stack.shape[1:3], torch=torch)
    eye_movie = _shift_movie_with_eye(
        torch.from_numpy(full_stack[: eyepos.shape[0] + int(n_lags)]).float(),
        torch.cat([eye_norm[: int(n_lags)], eye_norm], dim=0),
        out_size=out_size,
        scale_factor=float(scale_factor),
        torch=torch,
    )
    return _embed_time_lags(eye_movie, n_lags=int(n_lags), torch=torch)


def make_counterfactual_stim_explicit_history(
    full_stack: np.ndarray,
    eyepos: Any,
    *,
    ppd: float = PPD,
    scale_factor: float = 1.0,
    n_lags: int = N_LAGS,
    out_size: tuple[int, int] = OUT_SIZE,
) -> Any:
    """Embed a trace that already contains its complete causal model history."""
    import torch

    if int(eyepos.shape[0]) < int(n_lags):
        raise ValueError(f"Explicit-history trace needs at least {int(n_lags)} frames.")
    eye_norm = _eye_deg_to_norm(torch.fliplr(eyepos), ppd=float(ppd), img_size=full_stack.shape[1:3], torch=torch)
    eye_movie = _shift_movie_with_eye(
        torch.from_numpy(full_stack[: eyepos.shape[0]]).float(),
        eye_norm,
        out_size=out_size,
        scale_factor=float(scale_factor),
        torch=torch,
    )
    return _embed_time_lags(eye_movie, n_lags=int(n_lags), torch=torch)


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
        pretrained_checkpoint=hparams.get("pretrained_checkpoint"),
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


def load_population_view(*, spec_dir: Path, version_name: str) -> tuple[Any, Any, Path | None, Path | None]:
    from redundancy_resolved_v1_population import apply_population_view, load_population_view, resolve_population_spec_paths

    spec_npz, spec_json = resolve_population_spec_paths(spec_dir, version_name=version_name)
    view = load_population_view(spec_dir, version_name=version_name)
    return view, apply_population_view, spec_npz, spec_json


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
            "rr100_version": str(population_view.name),
            "rr100_input_channel": input_channel,
            "rr100_member_count": int(len(member_channels)),
            "rr100_member_channels": ",".join(str(ch) for ch in member_channels),
        }
        for key in ("group_id", "rep_channel", "rep_idx", "pooling_mode"):
            if key in rep_meta:
                row[f"rr100_{key}"] = rep_meta[key]
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
    rr_unit_rows: list[dict[str, Any]]
    torch: Any
    device: str
    provenance: dict[str, Any]

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        dataset_configs: Path,
        population_spec_dir: Path,
        rr100_version: str,
        device: str,
        strict: bool = True,
        mcfarland_outputs: Path | None = None,
    ) -> "RealTraceMatrixScorer":
        import torch

        model, model_info = load_pinned_multidataset_model(
            checkpoint_path=Path(checkpoint_path),
            dataset_configs=Path(dataset_configs),
            device=str(device),
            strict=bool(strict),
        )
        outputs, outputs_path = load_mcfarland_outputs(mcfarland_outputs)
        readout, canonical_unit_rows = load_spatial_readout(model, outputs, device=str(device))
        population_view, apply_population_view, spec_npz, spec_json = load_population_view(
            spec_dir=Path(population_spec_dir),
            version_name=str(rr100_version),
        )
        if int(population_view.input_channels) != len(canonical_unit_rows):
            raise ValueError(
                f"Population view expects {population_view.input_channels} channels, "
                f"but the canonical readout has {len(canonical_unit_rows)}."
            )
        rr_unit_rows = population_unit_rows(population_view, canonical_unit_rows)
        provenance = {
            "model": model_info,
            "mcfarland_outputs_path": outputs_path,
            "mcfarland_outputs_sha256": sha256_file(outputs_path),
            "canonical_readout_n_units": int(len(canonical_unit_rows)),
            "rr100_version": str(population_view.name),
            "rr100_n_units": int(population_view.n_units),
            "rr100_population_spec_npz": spec_npz,
            "rr100_population_spec_npz_sha256": sha256_file(spec_npz) if spec_npz is not None else None,
            "rr100_population_spec_json": spec_json,
            "rr100_population_spec_json_sha256": (
                sha256_file(spec_json) if spec_json is not None and spec_json.exists() else None
            ),
            "stimulus": {
                "ppd": PPD,
                "model_history_frames": N_LAGS,
                "model_history_includes_current_frame": True,
                "lag_history_policy": "explicit_preceding_history_for_rerun_with_legacy_prefix_replay_support",
                "lag_history_note": (
                    "Rerun traces contain 32 burn-in frames followed by 40 scored frames and use "
                    "make_counterfactual_stim_explicit_history. Its first lagged output (current "
                    "frame 31) is discarded; outputs with current frames 32..71 are scored. The "
                    "prefix-seeded helper remains only for replaying historical 40-frame caches."
                ),
                "out_size": list(OUT_SIZE),
                "trace_xy_convention": "input trace is [x_deg, y_deg]; scorer pre-flips for Ryan's helper convention",
            },
        }
        return cls(
            model=model,
            readout=readout,
            population_view=population_view,
            apply_population_view=apply_population_view,
            canonical_unit_rows=canonical_unit_rows,
            rr_unit_rows=rr_unit_rows,
            torch=torch,
            device=str(device),
            provenance=provenance,
        )

    @property
    def n_units(self) -> int:
        return int(self.population_view.n_units)

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

        with self.torch.no_grad():
            for trace_start in range(0, n_traces, trace_batch_size):
                trace_chunk = traces[trace_start : trace_start + trace_batch_size]
                stims = []
                frame_to_trace: list[int] = []
                for local_idx, trace in enumerate(trace_chunk):
                    arr = np.asarray(trace, dtype=np.float32)
                    explicit_history = arr.shape == (n_timepoints + N_LAGS, 2)
                    legacy_replay = arr.shape == (n_timepoints, 2)
                    if not explicit_history and not legacy_replay:
                        raise ValueError(
                            f"Trace has shape {arr.shape}; expected ({n_timepoints}, 2) for legacy replay or "
                            f"({n_timepoints + N_LAGS}, 2) for explicit history."
                        )
                    stack_frames = arr.shape[0] if explicit_history else arr.shape[0] + N_LAGS
                    full_stack = np.broadcast_to(
                        image[None, :, :],
                        (stack_frames, *image.shape),
                    ).copy()
                    eye = self.torch.from_numpy(_trace_xy_to_twin_helper_order(arr))
                    if explicit_history:
                        stim = make_counterfactual_stim_explicit_history(
                            full_stack,
                            eye,
                            ppd=PPD,
                            scale_factor=1.0,
                            n_lags=N_LAGS,
                            out_size=OUT_SIZE,
                        )
                    else:
                        stim = make_counterfactual_stim(
                            full_stack,
                            eye,
                            ppd=PPD,
                            scale_factor=1.0,
                            n_lags=N_LAGS,
                            out_size=OUT_SIZE,
                        )
                    length = int(stim.shape[0])
                    trace_ids = _scored_trace_ids(
                        length,
                        n_timepoints=n_timepoints,
                        trace_index=trace_start + local_idx,
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
                        weights = rb * float(bin_seconds)
                        unit_expected[int(trace_idx)] += np.sum(weights, axis=0)
                        unit_numer[int(trace_idx)] += np.sum(ub * weights, axis=0)
                        unit_rate_sum[int(trace_idx)] += np.sum(rb, axis=0)
                        unit_frame_count[int(trace_idx)] += int(np.count_nonzero(mask))
                    del x, full_map, rr_map, flat, rbar, gain, unit_bits_t
                    if str(self.device).startswith("cuda"):
                        self.torch.cuda.empty_cache()
                del stims, stim_all

        unit_bits = np.divide(unit_numer, np.maximum(unit_expected, 1e-8)).astype(np.float32)
        unit_mean_rate = np.divide(unit_rate_sum, np.maximum(unit_frame_count[:, None], 1)).astype(np.float32)
        population_numer = np.sum(unit_numer, axis=1)
        population_denom = np.sum(unit_expected, axis=1)
        population_bits = np.divide(population_numer, np.maximum(population_denom, 1e-8)).astype(np.float32)
        return unit_bits, unit_expected.astype(np.float32), unit_mean_rate, population_bits
