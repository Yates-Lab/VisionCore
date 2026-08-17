#!/usr/bin/env python3
"""Dense, resumable layerwise SF x TF grating probe for the frozen Figure 4 model.

The probe extends the existing RR100 grating assay through the model.  For
every channel at each stage it stores signed F0, F1, F2, and temporal AC RMS
from the center-aligned, causally latest activation.  Conditions are written
directly to NumPy memory maps so interruption never discards completed work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import sha256_file, write_json
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    DirectPopulationReadout,
    build_direct_readout,
)
from paper.fig4.spatiotemporal_tuning.run_grating_probe import (
    CONTRAST,
    DISCARD_FRAMES,
    FRAME_RATE_HZ,
    IMAGE_SIZE,
    N_LAGS,
    PPD,
    WINDOW_SIGMA_FRAC,
    embed_time_lags,
    make_grating_movie,
)
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
RAW = OUT / "exact_arrays"

# This is the cycle-valid half-octave grid used by the independently developed
# dense RR100 assay.  Sub-cycle gratings are deliberately excluded from the
# mechanistic surface because they do not contain one full carrier cycle in
# the 101-pixel aperture.
SPATIAL_CPDS = np.asarray(
    [0.4, 0.565685, 0.8, 1.131371, 1.6, 2.262742, 3.2, 4.525483, 6.4, 9.050967, 12.8, 16.0],
    dtype=np.float64,
)
TEMPORAL_HZ = np.asarray(
    [0.4, 0.565685, 0.8, 1.131371, 1.6, 2.262742, 3.2, 4.525483, 6.4, 9.050967, 12.8, 18.101934, 25.6, 36.203867, 51.2],
    dtype=np.float64,
)
ORIENTATIONS_DEG = np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float64)
PHASES_RAD = np.asarray([0.0, math.pi], dtype=np.float64)
DURATION_S = 3.0
METRICS = ("f0_signed_mean", "f1_amplitude", "f2_amplitude", "temporal_ac_rms")

STAGE_ORDER = (
    "retinal_input",
    "temporal_frontend",
    "stem_preactivation",
    "stem_normalized",
    "post_stem_splitrelu",
    "resblock1_preactivation",
    "resblock1_normalized",
    "resblock1_splitrelu",
    "resblock1_main_pooled",
    "resblock1_shortcut",
    "resblock1_output",
    "resblock2_preactivation",
    "resblock2_normalized",
    "resblock2_splitrelu",
    "resblock2_shortcut",
    "resblock2_output",
    "convgru",
    "rr100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def module_map(model: Any) -> dict[str, torch.nn.Module]:
    rb1 = model.convnet.layers[0]
    rb2 = model.convnet.layers[1]
    return {
        "temporal_frontend": model.frontend,
        "stem_preactivation": model.convnet.stem.components["conv"],
        "stem_normalized": model.convnet.stem.components["norm"],
        "post_stem_splitrelu": model.convnet.stem.components["act"],
        "resblock1_preactivation": rb1.main_block.components["conv"],
        "resblock1_normalized": rb1.main_block.components["norm"],
        "resblock1_splitrelu": rb1.main_block.components["act"],
        "resblock1_main_pooled": rb1.main_block.components["pool"],
        "resblock1_shortcut": rb1.shortcut,
        "resblock1_output": rb1,
        "resblock2_preactivation": rb2.main_block.components["conv"],
        "resblock2_normalized": rb2.main_block.components["norm"],
        "resblock2_splitrelu": rb2.main_block.components["act"],
        "resblock2_shortcut": rb2.shortcut,
        "resblock2_output": rb2,
        "convgru": model.recurrent,
    }


def center_latest(value: torch.Tensor) -> np.ndarray:
    """Return B x C at the native spatial center and latest causal time."""
    if value.ndim == 5:
        value = value[:, :, -1]
    if value.ndim != 4:
        raise ValueError(tuple(value.shape))
    return value[:, :, value.shape[-2] // 2, value.shape[-1] // 2].detach().float().cpu().numpy()


class TraceCollector:
    def __init__(self, model: Any):
        self.buffers: dict[str, list[np.ndarray]] = {name: [] for name in module_map(model)}
        self.handles = []
        for name, module in module_map(model).items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
            self.buffers[name].append(center_latest(output))

        return hook

    def reset(self) -> None:
        for values in self.buffers.values():
            values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def harmonic_amplitude(values: np.ndarray, frequency_hz: float, time_s: np.ndarray) -> np.ndarray:
    omega = 2.0 * math.pi * float(frequency_hz) * time_s
    design = np.column_stack((np.sin(omega), np.cos(omega), np.ones_like(omega)))
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.hypot(coefficient[0], coefficient[1])


def temporal_metrics(traces: np.ndarray, temporal_hz: float) -> dict[str, np.ndarray]:
    values = np.asarray(traces, dtype=np.float64)[DISCARD_FRAMES:]
    if values.ndim != 2 or values.shape[0] < 8:
        raise ValueError(values.shape)
    time_s = (np.arange(values.shape[0], dtype=np.float64) + DISCARD_FRAMES) / FRAME_RATE_HZ
    mean = values.mean(axis=0)
    centered = values - mean[None]
    f1 = harmonic_amplitude(values, temporal_hz, time_s)
    if 2.0 * float(temporal_hz) < FRAME_RATE_HZ / 2.0 - 1e-8:
        f2 = harmonic_amplitude(values, 2.0 * float(temporal_hz), time_s)
    else:
        f2 = np.full(values.shape[1], np.nan, dtype=np.float64)
    return {
        "f0_signed_mean": mean.astype(np.float32),
        "f1_amplitude": f1.astype(np.float32),
        "f2_amplitude": f2.astype(np.float32),
        "temporal_ac_rms": np.sqrt(np.mean(centered**2, axis=0)).astype(np.float32),
    }


def condition_movie(sf: float, tf: float, orientation: float, phase: float) -> np.ndarray:
    return make_grating_movie(
        orientation_deg=float(orientation),
        spatial_cpd=float(sf),
        temporal_hz=float(tf),
        phase_rad=float(phase),
        duration_s=DURATION_S,
    )


def collect_traces(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    collector: TraceCollector,
    movie: np.ndarray,
    frame_batch_size: int,
) -> dict[str, np.ndarray]:
    model = scorer.model.model
    normalized = (np.asarray(movie, dtype=np.float32) - 127.0) / 255.0
    stim = embed_time_lags(torch.from_numpy(normalized), torch)
    retinal: list[np.ndarray] = []
    rr100: list[np.ndarray] = []
    collector.reset()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for start in range(0, len(stim), frame_batch_size):
            x = stim[start : start + frame_batch_size].to(scorer.device)
            retinal.append(
                x[:, :, 0, x.shape[-2] // 2, x.shape[-1] // 2].detach().float().cpu().numpy()
            )
            core = model.core_forward(x, scorer._zero_behavior(len(x), dtype))
            rate_map = model.activation(readout(core[:, :, -1]))
            rr100.append(center_latest(rate_map))
    traces = {name: np.concatenate(values, axis=0) for name, values in collector.buffers.items()}
    traces["retinal_input"] = np.concatenate(retinal, axis=0)
    traces["rr100"] = np.concatenate(rr100, axis=0)
    expected_time = max(int(round(DURATION_S * FRAME_RATE_HZ)), N_LAGS + 8)
    for name, value in traces.items():
        if value.shape[0] != expected_time:
            raise ValueError((name, value.shape, expected_time))
    return traces


def stage_layout(traces: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = np.asarray([traces[name].shape[1] for name in STAGE_ORDER], dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(channels)[:-1])).astype(np.int32)
    stops = (starts + channels).astype(np.int32)
    return channels, starts, stops


def open_or_initialize_arrays(
    total_channels: int,
    *,
    overwrite: bool,
) -> tuple[dict[str, np.memmap], np.memmap]:
    shape = (
        int(total_channels),
        len(SPATIAL_CPDS),
        len(TEMPORAL_HZ),
        len(ORIENTATIONS_DEG),
        len(PHASES_RAD),
    )
    arrays: dict[str, np.memmap] = {}
    for metric in METRICS:
        path = RAW / f"{metric}.npy"
        if overwrite and path.exists():
            path.unlink()
        if path.exists():
            value = np.lib.format.open_memmap(path, mode="r+")
            if value.shape != shape:
                raise ValueError((path, value.shape, shape))
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            value[:] = np.nan
            value.flush()
        arrays[metric] = value
    completed_path = RAW / "completed.npy"
    complete_shape = (len(SPATIAL_CPDS), len(TEMPORAL_HZ), len(ORIENTATIONS_DEG), len(PHASES_RAD))
    if overwrite and completed_path.exists():
        completed_path.unlink()
    if completed_path.exists():
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if completed.shape != complete_shape:
            raise ValueError((completed.shape, complete_shape))
    else:
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=bool, shape=complete_shape)
        completed[:] = False
        completed.flush()
    return arrays, completed


def validate_against_existing_dense(
    arrays: dict[str, np.memmap], starts: np.ndarray, stops: np.ndarray
) -> dict[str, Any]:
    existing_path = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/grating_probe/dense_tf_response_tensor.npz"
    if not existing_path.exists():
        return {"status": "existing_dense_tensor_missing"}
    with np.load(existing_path) as archive:
        old_amp = np.asarray(archive["response_amplitude"], dtype=float)
        old_mean = np.asarray(archive["mean_rate"], dtype=float)
        old_sf = np.asarray(archive["spatial_cpd"], dtype=float)
        old_tf = np.asarray(archive["temporal_hz"], dtype=float)
        old_ori = np.asarray(archive["orientation_deg"], dtype=float)
    rr_index = STAGE_ORDER.index("rr100")
    new_f1 = np.asarray(arrays["f1_amplitude"][starts[rr_index] : stops[rr_index]], dtype=float)
    new_f0 = np.asarray(arrays["f0_signed_mean"][starts[rr_index] : stops[rr_index]], dtype=float)
    amp_error: list[np.ndarray] = []
    mean_error: list[np.ndarray] = []
    matched = 0
    for old_sf_i, sf in enumerate(old_sf):
        new_sf = np.flatnonzero(np.isclose(SPATIAL_CPDS, sf))
        if not len(new_sf):
            continue
        for old_tf_i, tf in enumerate(old_tf):
            new_tf = np.flatnonzero(np.isclose(TEMPORAL_HZ, tf))
            if not len(new_tf):
                continue
            for old_ori_i, orientation in enumerate(old_ori):
                new_ori = np.flatnonzero(np.isclose(ORIENTATIONS_DEG, orientation))
                if not len(new_ori):
                    continue
                for phase in range(2):
                    observed = new_f1[:, new_sf[0], new_tf[0], new_ori[0], phase]
                    observed_mean = new_f0[:, new_sf[0], new_tf[0], new_ori[0], phase]
                    if np.isfinite(observed).all():
                        amp_error.append(observed - old_amp[:, old_sf_i, old_tf_i, old_ori_i, phase])
                        mean_error.append(observed_mean - old_mean[:, old_sf_i, old_tf_i, old_ori_i, phase])
                        matched += 1
    if not amp_error:
        return {"status": "no_completed_overlapping_conditions"}
    amp = np.concatenate(amp_error)
    mean = np.concatenate(mean_error)
    return {
        "status": "validated",
        "n_matched_conditions": matched,
        "f1_max_abs_error": float(np.max(np.abs(amp))),
        "f1_median_abs_error": float(np.median(np.abs(amp))),
        "f0_max_abs_error": float(np.max(np.abs(mean))),
        "f0_median_abs_error": float(np.median(np.abs(mean))),
        "existing_tensor": existing_path,
    }


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=str(args.device),
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    scorer.model.model.eval()
    readout = build_direct_readout(scorer).eval()
    collector = TraceCollector(scorer.model.model)

    # One deterministic condition establishes the exact channel layout before
    # the resumable arrays are opened.  It is also reused if that condition is
    # pending, so the extra work is negligible.
    first_movie = condition_movie(SPATIAL_CPDS[0], TEMPORAL_HZ[0], ORIENTATIONS_DEG[0], PHASES_RAD[0])
    first_traces = collect_traces(scorer, readout, collector, first_movie, int(args.frame_batch_size))
    channels, starts, stops = stage_layout(first_traces)
    total_channels = int(channels.sum())
    np.savez_compressed(
        RAW / "layout_and_grid.npz",
        stages=np.asarray(STAGE_ORDER),
        stage_channels=channels,
        stage_starts=starts,
        stage_stops=stops,
        spatial_cpd=SPATIAL_CPDS,
        temporal_hz=TEMPORAL_HZ,
        orientation_deg=ORIENTATIONS_DEG,
        phase_rad=PHASES_RAD,
        metrics=np.asarray(METRICS),
    )
    arrays, completed = open_or_initialize_arrays(total_channels, overwrite=bool(args.overwrite))
    pending = [
        (sf_i, tf_i, ori_i, phase_i)
        for sf_i in range(len(SPATIAL_CPDS))
        for tf_i in range(len(TEMPORAL_HZ))
        for ori_i in range(len(ORIENTATIONS_DEG))
        for phase_i in range(len(PHASES_RAD))
        if not bool(completed[sf_i, tf_i, ori_i, phase_i])
    ]
    if int(args.max_conditions) > 0:
        pending = pending[: int(args.max_conditions)]
    initial_complete = int(np.count_nonzero(completed))
    total_conditions = int(completed.size)
    for run_index, (sf_i, tf_i, ori_i, phase_i) in enumerate(pending, start=1):
        if (sf_i, tf_i, ori_i, phase_i) == (0, 0, 0, 0):
            traces = first_traces
        else:
            movie = condition_movie(
                SPATIAL_CPDS[sf_i], TEMPORAL_HZ[tf_i], ORIENTATIONS_DEG[ori_i], PHASES_RAD[phase_i]
            )
            traces = collect_traces(scorer, readout, collector, movie, int(args.frame_batch_size))
        for stage_i, stage in enumerate(STAGE_ORDER):
            summary = temporal_metrics(traces[stage], TEMPORAL_HZ[tf_i])
            channel_slice = slice(int(starts[stage_i]), int(stops[stage_i]))
            for metric in METRICS:
                arrays[metric][channel_slice, sf_i, tf_i, ori_i, phase_i] = summary[metric]
        for value in arrays.values():
            value.flush()
        completed[sf_i, tf_i, ori_i, phase_i] = True
        completed.flush()
        print(
            f"layerwise SFxTF [{initial_complete + run_index}/{total_conditions}] "
            f"sf={SPATIAL_CPDS[sf_i]:g} tf={TEMPORAL_HZ[tf_i]:g} "
            f"ori={ORIENTATIONS_DEG[ori_i]:g} phase={phase_i}",
            flush=True,
        )
    collector.close()
    validation = validate_against_existing_dense(arrays, starts, stops)
    write_json(OUT / "rr100_replay_validation.json", validation)
    write_json(
        OUT / "run_manifest.json",
        {
            "analysis": "dense_layerwise_sf_tf_f0_f1_f2_probe_v1",
            "checkpoint": MODEL_CHECKPOINT_PATH,
            "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "rr100_version": RR100_VERSION,
            "stage_order": STAGE_ORDER,
            "stage_channels": channels,
            "total_channels": total_channels,
            "spatial_cpd": SPATIAL_CPDS,
            "temporal_hz": TEMPORAL_HZ,
            "orientation_deg": ORIENTATIONS_DEG,
            "phase_rad": PHASES_RAD,
            "duration_s": DURATION_S,
            "discard_external_output_frames": DISCARD_FRAMES,
            "frame_rate_hz": FRAME_RATE_HZ,
            "image_size_px": IMAGE_SIZE,
            "ppd": PPD,
            "contrast": CONTRAST,
            "window_sigma_fraction": WINDOW_SIGMA_FRAC,
            "center_alignment": "native spatial center and latest causal internal time at every hooked stage",
            "f0_definition": "signed arithmetic mean after discarding the first 32 external outputs",
            "f1_f2_definition": "least-squares sinusoid amplitude at TF or 2*TF with an intercept, evaluated separately; F2 omitted at/above Nyquist",
            "recurrent_contract": "each 32-lag external output is evaluated independently; ConvGRU state runs only within that causal lag window",
            "n_completed": int(np.count_nonzero(completed)),
            "n_total": total_conditions,
            "elapsed_minutes_this_invocation": (time.time() - start_time) / 60.0,
            "validation": validation,
        },
    )
    print(
        json.dumps(
            {"completed": int(np.count_nonzero(completed)), "total": total_conditions, "validation": validation},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
