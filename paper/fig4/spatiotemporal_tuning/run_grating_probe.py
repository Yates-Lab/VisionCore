#!/usr/bin/env python3
"""Recover the RR100 SF x TF x orientation x phase response tensors.

Two protocols are deliberately kept separate. ``historical`` exactly replays
the grating grid and timing used for the Figure 4 SF labels. ``dense_tf`` keeps
the same spatial geometry but lengthens each trial and samples TF at roughly
half-octave intervals so a temporal optimum can be resolved. Conditions are
checkpointed one at a time, making the GPU run safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visioncore-mpl-cache")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.upstream.real_trace_matrix.model import N_LAGS, RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


DEFAULT_OUT_DIR = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/grating_probe"
ORIENTATIONS_DEG = np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float64)
SPATIAL_CPDS = np.asarray([0.0125, 0.05, 0.2, 0.8, 3.2, 12.8], dtype=np.float64)
HISTORICAL_TF_HZ = np.asarray([0.0, 0.2, 0.8, 3.2, 12.8, 47.2], dtype=np.float64)
DENSE_TF_HZ = np.asarray(
    [
        0.2,
        0.4,
        0.565685,
        0.8,
        1.131371,
        1.6,
        2.262742,
        3.2,
        4.525483,
        6.4,
        9.050967,
        12.8,
        18.101934,
        25.6,
        36.203867,
        47.2,
        51.2,
    ],
    dtype=np.float64,
)
FRAME_RATE_HZ = 120.0
PPD = 37.50476617
IMAGE_SIZE = 101
CONTRAST = 0.8
WINDOW_SIGMA_FRAC = 0.28
PHASE_SEED = 17
N_DYNAMIC_PHASES = 2
N_STATIC_PHASES = 4
DISCARD_FRAMES = 32


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_grating_movie(
    *,
    orientation_deg: float,
    spatial_cpd: float,
    temporal_hz: float,
    phase_rad: float,
    duration_s: float,
) -> np.ndarray:
    n_valid_frames = max(int(round(duration_s * FRAME_RATE_HZ)), N_LAGS + 8)
    total_frames = n_valid_frames + N_LAGS - 1
    frame_idx = np.arange(total_frames, dtype=np.float64) - (N_LAGS - 1)
    time_s = frame_idx / FRAME_RATE_HZ
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE].astype(np.float64)
    x_deg = (xx - 0.5 * (IMAGE_SIZE - 1)) / PPD
    y_deg = (yy - 0.5 * (IMAGE_SIZE - 1)) / PPD
    theta = math.radians(float(orientation_deg))
    # The probe angle is the bar/contour axis.  Its SF-gradient normal is
    # n=(-sin(theta), cos(theta)); this convention is reused in the FEM analysis.
    normal_coord_deg = -math.sin(theta) * x_deg + math.cos(theta) * y_deg
    sigma_px = WINDOW_SIGMA_FRAC * IMAGE_SIZE
    radius_sq = (xx - 0.5 * (IMAGE_SIZE - 1)) ** 2 + (yy - 0.5 * (IMAGE_SIZE - 1)) ** 2
    window = np.exp(-0.5 * radius_sq / (sigma_px * sigma_px))
    carrier = np.sin(
        2.0 * math.pi * spatial_cpd * normal_coord_deg[None]
        - 2.0 * math.pi * temporal_hz * time_s[:, None, None]
        + phase_rad
    )
    movie = 127.5 + 127.5 * CONTRAST * carrier * window[None]
    return np.clip(movie, 0.0, 255.0).astype(np.float32)


def embed_time_lags(movie: Any, torch: Any) -> Any:
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
    total, channels, height, width = movie.shape
    out_frames = int(total) - N_LAGS + 1
    lagged = torch.empty(out_frames, channels, N_LAGS, height, width, dtype=movie.dtype)
    for lag in range(N_LAGS):
        lagged[:, :, lag] = movie[N_LAGS - 1 - lag : int(total) - lag]
    return lagged


def score_scalar_traces(
    scorer: RealTraceMatrixScorer,
    movie_uint: np.ndarray,
    *,
    frame_batch_size: int,
) -> np.ndarray:
    torch = scorer.torch
    movie = (np.asarray(movie_uint, dtype=np.float32) - 127.0) / 255.0
    stim = embed_time_lags(torch.from_numpy(movie), torch)
    chunks: list[np.ndarray] = []
    scorer.model.model.eval()
    scorer.readout.eval()
    with torch.no_grad():
        for start in range(0, int(stim.shape[0]), int(frame_batch_size)):
            x = stim[start : start + int(frame_batch_size)].to(scorer.device)
            full_map = scorer._compute_rate_map(x)
            rr_map = scorer.apply_population_view(full_map, scorer.population_view)
            center_y = int(rr_map.shape[-2] // 2)
            center_x = int(rr_map.shape[-1] // 2)
            chunks.append(rr_map[:, :, center_y, center_x].detach().cpu().numpy().astype(np.float32))
            del x, full_map, rr_map
    del stim
    if str(scorer.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def sinusoid_amplitude(traces: np.ndarray, temporal_hz: float) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float64)[DISCARD_FRAMES:]
    if temporal_hz <= 0.0 or values.shape[0] < 5:
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    time_s = (np.arange(values.shape[0], dtype=np.float64) + DISCARD_FRAMES) / FRAME_RATE_HZ
    omega = 2.0 * math.pi * temporal_hz * time_s
    design = np.column_stack((np.sin(omega), np.cos(omega), np.ones_like(omega)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.hypot(coefficients[0], coefficients[1]).astype(np.float32)


def phase_schedule(tf_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(PHASE_SEED)
    static = rng.uniform(0.0, 2.0 * math.pi, size=N_STATIC_PHASES)
    dynamic = np.arange(N_DYNAMIC_PHASES, dtype=np.float64) * 2.0 * math.pi / N_DYNAMIC_PHASES
    phases = np.full((len(tf_hz), N_STATIC_PHASES), np.nan, dtype=np.float64)
    valid = np.zeros_like(phases, dtype=bool)
    for index, tf in enumerate(tf_hz):
        chosen = static if np.isclose(tf, 0.0) else dynamic
        phases[index, : len(chosen)] = chosen
        valid[index, : len(chosen)] = True
    return phases, valid


def protocol_definition(name: str) -> tuple[np.ndarray, float]:
    if name == "historical":
        return HISTORICAL_TF_HZ.copy(), 1.5
    if name == "dense_tf":
        return DENSE_TF_HZ.copy(), 3.0
    raise ValueError(name)


def initialize_protocol(path: Path, tf_hz: np.ndarray, phases: np.ndarray, duration_s: float) -> None:
    shape = (100, len(SPATIAL_CPDS), len(tf_hz), len(ORIENTATIONS_DEG), N_STATIC_PHASES)
    np.savez_compressed(
        path,
        response_amplitude=np.full(shape, np.nan, dtype=np.float32),
        mean_rate=np.full(shape, np.nan, dtype=np.float32),
        rate_std=np.full(shape, np.nan, dtype=np.float32),
        completed=np.zeros(shape[1:], dtype=bool),
        spatial_cpd=SPATIAL_CPDS.astype(np.float32),
        temporal_hz=tf_hz.astype(np.float32),
        orientation_deg=ORIENTATIONS_DEG.astype(np.float32),
        phase_rad=phases.astype(np.float32),
        duration_s=np.asarray(duration_s, dtype=np.float64),
        frame_rate_hz=np.asarray(FRAME_RATE_HZ, dtype=np.float64),
        discard_frames=np.asarray(DISCARD_FRAMES, dtype=np.int32),
    )


def load_protocol(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def save_protocol(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def run_protocol(
    scorer: RealTraceMatrixScorer,
    *,
    name: str,
    out_dir: Path,
    frame_batch_size: int,
    max_conditions: int,
) -> Path:
    tf_hz, duration_s = protocol_definition(name)
    phases, valid_phases = phase_schedule(tf_hz)
    path = out_dir / f"{name}_response_tensor.npz"
    if not path.exists():
        initialize_protocol(path, tf_hz, phases, duration_s)
    arrays = load_protocol(path)
    expected_shape = (100, len(SPATIAL_CPDS), len(tf_hz), len(ORIENTATIONS_DEG), N_STATIC_PHASES)
    if arrays["response_amplitude"].shape != expected_shape:
        raise ValueError(f"Cached {name} tensor has unexpected shape {arrays['response_amplitude'].shape}")
    pending = [
        (sf_idx, tf_idx, ori_idx, phase_idx)
        for sf_idx in range(len(SPATIAL_CPDS))
        for tf_idx in range(len(tf_hz))
        for ori_idx in range(len(ORIENTATIONS_DEG))
        for phase_idx in range(N_STATIC_PHASES)
        if valid_phases[tf_idx, phase_idx] and not bool(arrays["completed"][sf_idx, tf_idx, ori_idx, phase_idx])
    ]
    if max_conditions > 0:
        pending = pending[:max_conditions]
    total_valid = int(np.count_nonzero(valid_phases)) * len(SPATIAL_CPDS) * len(ORIENTATIONS_DEG)
    already = total_valid - len([
        1
        for sf_idx in range(len(SPATIAL_CPDS))
        for tf_idx in range(len(tf_hz))
        for ori_idx in range(len(ORIENTATIONS_DEG))
        for phase_idx in range(N_STATIC_PHASES)
        if valid_phases[tf_idx, phase_idx] and not bool(arrays["completed"][sf_idx, tf_idx, ori_idx, phase_idx])
    ])
    for run_index, (sf_idx, tf_idx, ori_idx, phase_idx) in enumerate(pending, start=1):
        movie = make_grating_movie(
            orientation_deg=float(ORIENTATIONS_DEG[ori_idx]),
            spatial_cpd=float(SPATIAL_CPDS[sf_idx]),
            temporal_hz=float(tf_hz[tf_idx]),
            phase_rad=float(phases[tf_idx, phase_idx]),
            duration_s=duration_s,
        )
        traces = score_scalar_traces(scorer, movie, frame_batch_size=frame_batch_size)
        analysis = traces[DISCARD_FRAMES:]
        arrays["mean_rate"][:, sf_idx, tf_idx, ori_idx, phase_idx] = np.mean(analysis, axis=0)
        arrays["rate_std"][:, sf_idx, tf_idx, ori_idx, phase_idx] = np.std(analysis, axis=0)
        arrays["response_amplitude"][:, sf_idx, tf_idx, ori_idx, phase_idx] = sinusoid_amplitude(
            traces, float(tf_hz[tf_idx])
        )
        arrays["completed"][sf_idx, tf_idx, ori_idx, phase_idx] = True
        save_protocol(path, arrays)
        print(
            f"{name} [{already + run_index}/{total_valid}] sf={SPATIAL_CPDS[sf_idx]:g} "
            f"tf={tf_hz[tf_idx]:g} ori={ORIENTATIONS_DEG[ori_idx]:g} phase={phase_idx}",
            flush=True,
        )
    return path


def historical_validation(tensor_path: Path, existing_units_csv: Path) -> pd.DataFrame:
    arrays = load_protocol(tensor_path)
    amp = arrays["response_amplitude"]
    amp_rms = np.sqrt(np.nanmean(np.square(amp), axis=-1))
    table = pd.read_csv(existing_units_csv).sort_values("unit_index").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for unit in range(100):
        flat_index = int(np.nanargmax(amp_rms[unit]))
        sf_idx, tf_idx, ori_idx = np.unravel_index(flat_index, amp_rms[unit].shape)
        row = table.iloc[unit]
        rows.append(
            {
                "unit_index": unit,
                "observed_peak_spatial_cpd": float(arrays["spatial_cpd"][sf_idx]),
                "cached_peak_spatial_cpd": float(row["dynamic_peak_spatial_cpd_by_amp"]),
                "observed_peak_temporal_hz": float(arrays["temporal_hz"][tf_idx]),
                "cached_peak_temporal_hz": float(row["dynamic_peak_temporal_hz_by_amp"]),
                "observed_peak_orientation_deg": float(arrays["orientation_deg"][ori_idx]),
                "cached_peak_orientation_deg": float(row["dynamic_peak_orientation_deg_by_amp"]),
            }
        )
    result = pd.DataFrame(rows)
    for axis in ("spatial_cpd", "temporal_hz", "orientation_deg"):
        result[f"matches_{axis}"] = np.isclose(
            result[f"observed_peak_{axis}"], result[f"cached_peak_{axis}"]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocols", default="historical,dense_tf")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--rr100-version", default=RR100_VERSION)
    parser.add_argument("--mcfarland-outputs", type=Path, default=ROOT / "scripts/mcfarland_outputs_mono.pkl")
    parser.add_argument("--existing-units-csv", type=Path, default=ROOT / (
        "outputs/active_sensing_movie_information/"
        "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
        "merged/unit_feature_table.csv"
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--max-conditions", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=Path(args.checkpoint),
        dataset_configs=Path(args.dataset_configs),
        population_spec_dir=Path(args.population_spec_dir),
        rr100_version=str(args.rr100_version),
        device=str(args.device),
        strict=True,
        mcfarland_outputs=Path(args.mcfarland_outputs),
    )
    if scorer.n_units != 100:
        raise ValueError(f"Expected RR100, loaded {scorer.n_units} units")
    pd.DataFrame(scorer.rr_unit_rows).to_csv(out_dir / "rr100_unit_identity.csv", index=False)
    protocols = [item.strip() for item in str(args.protocols).split(",") if item.strip()]
    outputs: dict[str, Any] = {}
    for protocol in protocols:
        tensor_path = run_protocol(
            scorer,
            name=protocol,
            out_dir=out_dir,
            frame_batch_size=int(args.frame_batch_size),
            max_conditions=int(args.max_conditions),
        )
        outputs[protocol] = {
            "tensor": tensor_path,
            "sha256": sha256_file(tensor_path),
        }
        arrays = load_protocol(tensor_path)
        outputs[protocol]["completed_conditions"] = int(np.count_nonzero(arrays["completed"]))
    historical_path = out_dir / "historical_response_tensor.npz"
    if historical_path.exists() and bool(np.all(load_protocol(historical_path)["completed"] | np.isnan(load_protocol(historical_path)["phase_rad"])[:, None, :])):
        validation = historical_validation(historical_path, Path(args.existing_units_csv))
        validation.to_csv(out_dir / "historical_peak_validation.csv", index=False)
        outputs["historical_validation"] = {
            column: int(validation[column].sum())
            for column in validation.columns
            if column.startswith("matches_")
        }
    manifest = {
        "analysis": "fig4_joint_spatiotemporal_grating_probe",
        "protocols": protocols,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "population_version": str(args.rr100_version),
        "population_transform": "one-hot medoid transform reconstructed from readable final QC plus retained-channel labels",
        "scorer_provenance": scorer.provenance,
        "protocol_definitions": {
            "historical": {
                "spatial_cpd": SPATIAL_CPDS,
                "temporal_hz": HISTORICAL_TF_HZ,
                "orientation_deg": ORIENTATIONS_DEG,
                "duration_s": 1.5,
            },
            "dense_tf": {
                "spatial_cpd": SPATIAL_CPDS,
                "temporal_hz": DENSE_TF_HZ,
                "orientation_deg": ORIENTATIONS_DEG,
                "duration_s": 3.0,
            },
        },
        "stimulus_parameters": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "ppd": PPD,
            "image_size_px": IMAGE_SIZE,
            "contrast": CONTRAST,
            "window_sigma_fraction": WINDOW_SIGMA_FRAC,
            "n_lags": N_LAGS,
            "discard_frames": DISCARD_FRAMES,
            "n_dynamic_phases": N_DYNAMIC_PHASES,
            "n_static_phases": N_STATIC_PHASES,
            "static_phase_seed": PHASE_SEED,
        },
        "orientation_convention": "orientation is bar/contour axis; SF normal n=(-sin(theta), cos(theta))",
        "stimulus_normalization": "(clipped_0_255_movie - 127.0) / 255.0",
        "recurrent_state_contract": "each condition is independently lag-embedded and evaluated; no state persists between conditions",
        "outputs": outputs,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote grating probe to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
