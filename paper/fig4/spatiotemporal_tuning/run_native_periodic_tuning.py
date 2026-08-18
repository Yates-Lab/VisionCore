#!/usr/bin/env python3
"""Measure steady-state RR100 SF/TF tuning for a native-240 Dekel twin.

Unlike the recovered finite-movie probe, this evaluates analytically periodic
60-frame histories at uniformly spaced endpoint phases.  Time therefore does
not need to be estimated from a short fixation or a coarse DFT: each condition
directly samples the model's periodic steady-state response.  The spatial grid
is derived from the checkpoint's actual 35-pixel aperture, so sub-cycle SF
conditions are never mislabeled as tuning measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    recommended_native_grid,
)
from paper.fig4.upstream.real_trace_matrix.model import (
    adapt_population_view_to_available,
    load_mcfarland_outputs,
    load_pinned_multidataset_model,
    load_population_view,
    sha256_file,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_POPULATION_SPEC_DIR,
    RR100_VERSION,
)
from paper.model_selection._m77_response_subspace_impl import (
    M77EncodingModel,
    build_m77_rr100_readout,
    canonical_rr100_rows,
)


DEFAULT_UNIT_TABLE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)
DEFAULT_MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNIT_TABLE)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--contrast", type=float, default=0.25)
    parser.add_argument("--n-phases", type=int, default=32)
    parser.add_argument("--max-units", type=int, default=None, help="Explicit smoke-test limit")
    parser.add_argument("--max-conditions", type=int, default=None, help="Explicit smoke-test limit")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def dataset_contract(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sampling = config.get("sampling") or {}
    supervision = config.get("supervision") or {}
    input_rate = int(sampling.get("target_rate", sampling.get("source_rate", 120)))
    output_rate = int(supervision.get("target_rate", input_rate))
    if input_rate != 240 or output_rate != 240:
        raise ValueError(
            f"Native periodic tuning requires 240->240 input/output, got {input_rate}->{output_rate}"
        )
    lags = np.asarray((config.get("keys_lags") or {}).get("stim"), dtype=int)
    if lags.ndim != 1 or not np.array_equal(lags, np.arange(len(lags))):
        raise ValueError("stimulus lags must be consecutive and begin at zero")
    transforms = config.get("transforms") or {}
    operations = ((transforms.get("stim") or {}).get("ops") or [])
    crop_sizes = [
        int(item["center_crop"]["size"])
        for item in operations
        if isinstance(item, dict) and "center_crop" in item
    ]
    if len(crop_sizes) != 1:
        raise ValueError("dataset config must declare one stimulus center_crop")
    return {
        "input_rate_hz": input_rate,
        "output_rate_hz": output_rate,
        "n_lags": int(len(lags)),
        "input_size": crop_sizes[0],
    }


def periodic_histories(
    *,
    spatial_cpd: float,
    temporal_hz: float,
    bar_orientation_deg: float,
    phases_rad: np.ndarray,
    n_lags: int,
    input_size: int,
    ppd: float,
    frame_rate_hz: float,
    contrast: float,
    device: str,
) -> torch.Tensor:
    """Return [phase, channel, lag, y, x] periodic grating histories."""
    if spatial_cpd <= 0 or temporal_hz < 0:
        raise ValueError("spatial frequency must be positive and TF nonnegative")
    coordinate = (
        np.arange(input_size, dtype=np.float64) - 0.5 * (input_size - 1)
    ) / float(ppd)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    # Report bar orientation, while the Fourier wavevector is normal to bars.
    normal = np.deg2rad(float(bar_orientation_deg) + 90.0)
    spatial_phase = 2.0 * np.pi * float(spatial_cpd) * (
        xx * np.cos(normal) + yy * np.sin(normal)
    )
    lag = np.arange(n_lags, dtype=np.float64)
    past_phase = -2.0 * np.pi * float(temporal_hz) * lag / float(frame_rate_hz)
    value = float(contrast) * np.cos(
        phases_rad[:, None, None, None]
        + past_phase[None, :, None, None]
        + spatial_phase[None, None]
    )
    return torch.from_numpy(value.astype(np.float32)).unsqueeze(1).to(device)


def periodic_response_metrics(phases_rad: np.ndarray, rates: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean, total phase RMS, and first-harmonic amplitude per unit."""
    phase = np.asarray(phases_rad, dtype=np.float64)
    value = np.asarray(rates, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != len(phase):
        raise ValueError("rates must have shape [phase, unit]")
    design = np.column_stack((np.ones(len(phase)), np.cos(phase), np.sin(phase)))
    coefficients = np.linalg.lstsq(design, value, rcond=None)[0]
    mean = value.mean(axis=0)
    return {
        "mean_rate": mean,
        "response_amp_rms": np.sqrt(np.mean(np.square(value - mean), axis=0)),
        "f1_amplitude": np.hypot(coefficients[1], coefficients[2]),
        "minimum_rate": value.min(axis=0),
        "maximum_rate": value.max(axis=0),
    }


@torch.no_grad()
def score_histories(
    encoding_model: M77EncodingModel,
    readout,
    histories: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    values = []
    for start in range(0, len(histories), int(batch_size)):
        movie = histories[start : start + int(batch_size)]
        behavior = encoding_model.zero_behavior(len(movie), movie.dtype)
        core = encoding_model.model.model.core_forward(movie, behavior)
        preactivation = readout(core[:, :, -1])[:, :, 0, 0]
        activation = getattr(encoding_model.model.model, "activation", None)
        rate = activation(preactivation) if activation is not None else F.softplus(preactivation)
        values.append(rate.float().cpu().numpy())
    return np.concatenate(values)


def load_rr100(args: argparse.Namespace, units: pd.DataFrame):
    model, model_info = load_pinned_multidataset_model(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_config.resolve(),
        device=args.device,
        strict=True,
    )
    outputs, resolved_mcfarland = load_mcfarland_outputs(args.mcfarland_outputs)
    population_view, _, _, resolved_population = load_population_view(
        spec_dir=args.population_spec_dir.resolve(),
        version_name=RR100_VERSION,
    )
    canonical_rows = canonical_rr100_rows(model, outputs)
    for row in canonical_rows:
        configured_cids = set(
            map(
                int,
                model.model.dataset_configs[int(row["model_readout_index"])].get(
                    "cids", []
                ),
            )
        )
        row["available"] = int(row["source_cid"]) in configured_cids
    population_view, adaptation = adapt_population_view_to_available(
        population_view, canonical_rows
    )
    units = units.loc[
        ~units.unit_index.isin(adaptation["inactive_units"])
    ].reset_index(drop=True)
    if units.empty:
        raise RuntimeError("Population adaptation left no active RR100 units")
    encoding_model = M77EncodingModel(model=model, device=args.device)
    readout = build_m77_rr100_readout(
        encoding_model,
        population_view,
        canonical_rows,
        units.unit_index.to_numpy(dtype=int),
    )
    return (
        encoding_model,
        readout,
        model_info,
        resolved_mcfarland,
        resolved_population,
        adaptation,
        units,
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.contrast <= 0.5:
        raise ValueError("contrast must lie in (0, 0.5]")
    if args.n_phases < 12:
        raise ValueError("n_phases must be at least 12")
    contract = dataset_contract(args.dataset_config.resolve())
    units = pd.read_csv(args.unit_table).sort_values("unit_index").reset_index(drop=True)
    if not np.array_equal(units.unit_index.to_numpy(int), np.arange(len(units))):
        raise ValueError("unit table must contain consecutive RR100 unit indices")
    if args.max_units is not None:
        units = units.iloc[: int(args.max_units)].copy()
    grid = recommended_native_grid(
        image_size=contract["input_size"],
        frame_rate_hz=contract["input_rate_hz"],
    )
    phases = np.linspace(0.0, 2.0 * np.pi, int(args.n_phases), endpoint=False)
    (
        encoding_model,
        readout,
        model_info,
        resolved_mcfarland,
        resolved_population,
        population_adaptation,
        units,
    ) = load_rr100(args, units)
    rows = []
    conditions = [
        (float(sf), float(tf), float(orientation))
        for sf in grid["spatial_cpd"]
        for tf in grid["temporal_hz"]
        for orientation in grid["orientation_deg"]
    ]
    total_grid_conditions = len(conditions)
    if args.max_conditions is not None:
        conditions = conditions[: int(args.max_conditions)]
    for condition_index, (sf, tf, orientation) in enumerate(conditions, start=1):
        histories = periodic_histories(
            spatial_cpd=sf,
            temporal_hz=tf,
            bar_orientation_deg=orientation,
            phases_rad=phases,
            n_lags=contract["n_lags"],
            input_size=contract["input_size"],
            ppd=grid["ppd"],
            frame_rate_hz=contract["input_rate_hz"],
            contrast=args.contrast,
            device=args.device,
        )
        rates = score_histories(encoding_model, readout, histories, args.batch_size)
        metrics = periodic_response_metrics(phases, rates)
        for unit_row, unit in units.iterrows():
            rows.append(
                {
                    "unit_index": int(unit.unit_index),
                    "unit_label": str(unit.get("unit_label", f"u{int(unit.unit_index):03d}")),
                    "probe_orientation_deg": orientation,
                    "spatial_cpd": sf,
                    "temporal_hz": tf,
                    **{key: float(value[unit_row]) for key, value in metrics.items()},
                }
            )
        if condition_index == 1 or condition_index % 32 == 0 or condition_index == len(conditions):
            print(f"periodic tuning {condition_index}/{len(conditions)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = args.out_dir / "frequency_tuning_grouped.csv"
    pd.DataFrame(rows).to_csv(grouped_path, index=False)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256_file(args.dataset_config.resolve()),
        "unit_table": str(args.unit_table.resolve()),
        "unit_table_sha256": sha256_file(args.unit_table.resolve()),
        "mcfarland_outputs": str(resolved_mcfarland),
        "population_spec": str(resolved_population),
        "population_adaptation": population_adaptation,
        "model_info": model_info,
        "contract": contract,
        "grid": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in grid.items()
        },
        "contrast": float(args.contrast),
        "n_endpoint_phases": int(args.n_phases),
        "n_rr100_units_scored": int(len(units)),
        "inactive_rr100_units": population_adaptation["inactive_units"],
        "n_conditions_scored": int(len(conditions)),
        "n_conditions_in_complete_grid": int(total_grid_conditions),
        "complete_grid": bool(
            len(units) == 100 and len(conditions) == total_grid_conditions
        ),
        "response_definition": (
            "steady periodic histories sampled at uniform endpoint phases; "
            "response_amp_rms is RMS phase modulation and f1_amplitude is the "
            "least-squares first harmonic"
        ),
        "spatial_orientation_definition": "reported orientation is bar axis; Fourier wavevector is +90 degrees",
        "grouped_csv": str(grouped_path.resolve()),
    }
    (args.out_dir / "periodic_tuning_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(grouped_path)


if __name__ == "__main__":
    main()
