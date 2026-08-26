#!/usr/bin/env python3
"""Measure controlled drifting-grating tuning for exact model readouts.

This assay is intentionally separate from the recorded ``gratings.dset``
analysis.  The recorded sessions contain static forage gratings but no
independently varied temporal-frequency condition. Here, controlled periodic
movies are replayed through every available Figure-3 biological readout using
its exact ``(session, cid)`` identity.  No RR100 clustering is involved.

The primary response is the phase-averaged expected count (F0) above an
explicit gray blank, matching the usual drifting-grating mean-response
measurement. Phase-locked and all-harmonic quantities are saved as diagnostics,
not substituted for F0.
Motion directions span the full 360 degrees; an axial 0--180 orientation grid
is insufficient for direction-selective units.
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
    DEFAULT_PPD,
    recommended_native_grid,
)
from paper.fig4.upstream.real_trace_matrix.model import (
    load_mcfarland_outputs,
    load_pinned_multidataset_model,
    sha256_file,
)
from paper.model_selection.native_twin import (
    TwinEncodingModel,
    canonical_population_rows,
)


DEFAULT_MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"
DEFAULT_MODEL_SPEC = ROOT / "paper/model_selection/production_model.yaml"
METRIC_NAMES = (
    "f0_expected_count",
    "delta_f0_expected_count",
    "evoked_rms_expected_count",
    "phase_modulation_rms_expected_count",
    "f1_expected_count_amplitude",
    "f2_expected_count_amplitude",
    "minimum_expected_count",
    "maximum_expected_count",
    "half_phase_f0_expected_count",
    "half_phase_evoked_rms_expected_count",
)


def dataset_contract(path: Path) -> dict[str, int]:
    """Resolve and validate the native 240-Hz stimulus contract."""
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sampling = config.get("sampling") or {}
    supervision = config.get("supervision") or {}
    input_rate = int(sampling.get("target_rate", sampling.get("source_rate", 120)))
    output_rate = int(supervision.get("target_rate", input_rate))
    if input_rate != 240 or output_rate != 240:
        raise ValueError(
            "Exact-CID tuning requires 240->240 input/output, got "
            f"{input_rate}->{output_rate}"
        )
    lags = np.asarray((config.get("keys_lags") or {}).get("stim"), dtype=int)
    if lags.ndim != 1 or not np.array_equal(lags, np.arange(len(lags))):
        raise ValueError("stimulus lags must be consecutive and begin at zero")
    operations = (((config.get("transforms") or {}).get("stim") or {}).get("ops") or [])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.25,
        help="Input amplitude around gray; 0.25 matches the 50%% forage gratings.",
    )
    parser.add_argument("--n-phases", type=int, default=24)
    parser.add_argument("--n-directions", type=int, default=18)
    parser.add_argument(
        "--spatial-cpd",
        default="",
        help="Optional comma-separated SF grid; default is the cycle-valid native grid.",
    )
    parser.add_argument(
        "--temporal-hz",
        default="",
        help="Optional comma-separated TF grid; default is 0 plus the native grid.",
    )
    parser.add_argument("--ppd", type=float, default=DEFAULT_PPD)
    parser.add_argument("--max-conditions", type=int, default=None)
    return parser.parse_args()


def model_contract(spec_path: Path) -> tuple[dict, Path, Path]:
    """Resolve checkpoint and dataset only through a digest-bound model spec."""
    spec_path = spec_path.expanduser().resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    checkpoint_record = spec.get("checkpoint", {})
    dataset_record = (
        spec.get("training", {}).get("datasets", {}).get("descriptive_all_gratings", {})
    )
    checkpoint = Path(checkpoint_record.get("path", "")).expanduser().resolve()
    dataset = Path(dataset_record.get("path", ""))
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    dataset = dataset.resolve()
    if not checkpoint.is_file() or not dataset.is_file():
        raise FileNotFoundError("model spec checkpoint or dataset config is missing")
    if sha256_file(checkpoint) != str(checkpoint_record.get("sha256", "")):
        raise ValueError("checkpoint digest does not match the model spec")
    if sha256_file(dataset) != str(dataset_record.get("sha256", "")):
        raise ValueError("dataset digest does not match the model spec")
    return spec, checkpoint, dataset


def _parse_grid(value: str, default: np.ndarray) -> np.ndarray:
    if not value.strip():
        result = np.asarray(default, dtype=np.float64)
    else:
        result = np.asarray(
            [float(part) for part in value.split(",") if part.strip()],
            dtype=np.float64,
        )
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("frequency grids must be finite, nonempty vectors")
    if len(np.unique(result)) != len(result) or not np.all(np.diff(result) > 0):
        raise ValueError("frequency grids must be unique and strictly increasing")
    return result


def drifting_histories(
    *,
    spatial_cpd: float,
    temporal_hz: float,
    motion_direction_deg: float,
    endpoint_phases_rad: np.ndarray,
    n_lags: int,
    input_size: int,
    ppd: float,
    frame_rate_hz: float,
    contrast: float,
    device: str,
) -> torch.Tensor:
    """Return ``[phase, 1, lag, y, x]`` histories moving along ``direction``.

    The stimulus is ``cos(2*pi*SF*(d dot x - speed*t) + phase)``.  Lag zero is
    the current frame, so earlier frames acquire ``+2*pi*TF*lag/frame_rate``.
    """
    if spatial_cpd <= 0 or temporal_hz < 0:
        raise ValueError("SF must be positive and TF must be nonnegative")
    coordinate = (
        np.arange(int(input_size), dtype=np.float64)
        - 0.5 * (int(input_size) - 1)
    ) / float(ppd)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    direction = np.deg2rad(float(motion_direction_deg))
    spatial_phase = 2.0 * np.pi * float(spatial_cpd) * (
        xx * np.cos(direction) + yy * np.sin(direction)
    )
    lag = np.arange(int(n_lags), dtype=np.float64)
    past_phase = 2.0 * np.pi * float(temporal_hz) * lag / float(frame_rate_hz)
    values = float(contrast) * np.cos(
        np.asarray(endpoint_phases_rad)[:, None, None, None]
        + past_phase[None, :, None, None]
        + spatial_phase[None, None]
    )
    return torch.from_numpy(values.astype(np.float32)).unsqueeze(1).to(device)


def response_metrics(
    endpoint_phases_rad: np.ndarray,
    expected_counts: np.ndarray,
    blank_expected_count: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize expected count/bin without conflating F0 and phase locking."""
    phase = np.asarray(endpoint_phases_rad, dtype=np.float64)
    value = np.asarray(expected_counts, dtype=np.float64)
    blank = np.asarray(blank_expected_count, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != len(phase):
        raise ValueError("expected_counts must have shape [phase, unit]")
    if blank.shape != (value.shape[1],):
        raise ValueError("blank_expected_count must have shape [unit]")
    if len(phase) < 8 or len(phase) % 2:
        raise ValueError("phase grid must contain an even number of at least eight samples")

    mean = value.mean(axis=0)
    centered = value - mean[None]
    design = np.column_stack(
        (
            np.ones(len(phase)),
            np.cos(phase),
            np.sin(phase),
            np.cos(2.0 * phase),
            np.sin(2.0 * phase),
        )
    )
    coefficient = np.linalg.lstsq(design, value, rcond=None)[0]
    half = value[::2]
    return {
        "f0_expected_count": mean,
        "delta_f0_expected_count": mean - blank,
        "evoked_rms_expected_count": np.sqrt(
            np.mean(np.square(value - blank[None]), axis=0)
        ),
        "phase_modulation_rms_expected_count": np.sqrt(
            np.mean(np.square(centered), axis=0)
        ),
        "f1_expected_count_amplitude": np.hypot(coefficient[1], coefficient[2]),
        "f2_expected_count_amplitude": np.hypot(coefficient[3], coefficient[4]),
        "minimum_expected_count": value.min(axis=0),
        "maximum_expected_count": value.max(axis=0),
        "half_phase_f0_expected_count": half.mean(axis=0),
        "half_phase_evoked_rms_expected_count": np.sqrt(
            np.mean(np.square(half - blank[None]), axis=0)
        ),
    }


def exact_unit_rows(model, canonical_rows: list[dict]) -> list[dict]:
    """Attach the native model readout row to every canonical biological CID."""
    rows: list[dict] = []
    for channel, source in enumerate(canonical_rows):
        dataset_index = int(source["model_readout_index"])
        configured = np.asarray(
            model.model.dataset_configs[dataset_index].get("cids", []), dtype=int
        )
        if configured.ndim != 1 or len(np.unique(configured)) != len(configured):
            raise ValueError(
                f"configured CIDs for {source['session']!r} are not unique and one-dimensional"
            )
        matches = np.flatnonzero(configured == int(source["source_cid"]))
        row = dict(source)
        row["channel"] = int(channel)
        row["available"] = bool(len(matches) == 1)
        row["model_readout_row"] = int(matches[0]) if len(matches) == 1 else None
        rows.append(row)
    return rows


def audit_native_cid_mapping(encoding_model: TwinEncodingModel, unit_rows: list[dict]) -> dict:
    """Reject duplicate identities, bad session indices, and invalid native rows."""
    identities = [(str(row["session"]), int(row["source_cid"])) for row in unit_rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("canonical biological (session, cid) identities are not unique")
    checked = 0
    for row in unit_rows:
        dataset_index = int(row["model_readout_index"])
        if str(encoding_model.model.names[dataset_index]) != str(row["session"]):
            raise RuntimeError("canonical session does not match the native model readout")
        if not bool(row["available"]):
            continue
        native = encoding_model.model.model.readouts[dataset_index]
        model_row = int(row["model_readout_row"])
        if model_row < 0 or model_row >= int(native.n_units):
            raise RuntimeError("canonical CID maps outside the native readout")
        configured_cid = int(
            encoding_model.model.model.dataset_configs[dataset_index]["cids"][model_row]
        )
        if configured_cid != int(row["source_cid"]):
            raise RuntimeError("native readout row does not map back to the requested CID")
        checked += 1
    if checked == 0:
        raise RuntimeError("no canonical biological CIDs are available in this checkpoint")
    return {
        "n_canonical_identities": int(len(unit_rows)),
        "n_available_native_readouts_checked": int(checked),
        "n_unavailable": int(len(unit_rows) - checked),
        "identity_key": "(session, cid)",
        "readout_path": "checkpoint-native per-session readout; no reconstructed population head",
        "passed": True,
    }


def _native_scalar_output(readout, core: torch.Tensor) -> torch.Tensor:
    value = readout(core)
    value = value.reshape(value.shape[0], value.shape[1], -1)
    if value.shape[-1] != 1:
        raise RuntimeError("native readout did not return one scalar per biological unit")
    return value[:, :, 0]


@torch.no_grad()
def score_exact_histories(
    encoding_model: TwinEncodingModel,
    unit_rows: list[dict],
    histories: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    available = [row for row in unit_rows if bool(row["available"])]
    by_dataset: dict[int, list[tuple[int, int]]] = {}
    for output_index, row in enumerate(available):
        by_dataset.setdefault(int(row["model_readout_index"]), []).append(
            (int(output_index), int(row["model_readout_row"]))
        )
    chunks: list[np.ndarray] = []
    for start in range(0, len(histories), int(batch_size)):
        movie = histories[start : start + int(batch_size)]
        behavior = encoding_model.zero_behavior(len(movie), movie.dtype)
        phase_readouts = getattr(encoding_model.model.model, "phase_readouts", None)
        if phase_readouts is not None:
            core, phase = encoding_model.model.model.core_forward_with_phase(
                movie, behavior
            )
        else:
            core = encoding_model.model.model.core_forward(movie, behavior)
            phase = None
        preactivation = torch.empty(
            len(movie), len(available), device=movie.device, dtype=core.dtype
        )
        for dataset_index, mappings in by_dataset.items():
            native = _native_scalar_output(
                encoding_model.model.model.readouts[dataset_index], core
            )
            if phase_readouts is not None:
                native = native + _native_scalar_output(
                    phase_readouts[dataset_index], phase
                )
            output_index = torch.as_tensor(
                [item[0] for item in mappings], dtype=torch.long, device=movie.device
            )
            model_row = torch.as_tensor(
                [item[1] for item in mappings], dtype=torch.long, device=movie.device
            )
            preactivation[:, output_index] = native[:, model_row]
        activation = getattr(encoding_model.model.model, "activation", None)
        expected_count = (
            activation(preactivation)
            if activation is not None
            else F.softplus(preactivation)
        )
        chunks.append(expected_count.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _conditions(spatial: np.ndarray, temporal: np.ndarray, directions: np.ndarray) -> pd.DataFrame:
    rows = []
    for sf in spatial:
        for tf in temporal:
            for direction in directions:
                rows.append(
                    {
                        "condition_index": len(rows),
                        "spatial_cpd": float(sf),
                        "temporal_hz": float(tf),
                        "motion_direction_deg": float(direction),
                        "bar_orientation_deg": float((direction + 90.0) % 180.0),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0 < float(args.contrast) <= 0.5:
        raise ValueError("contrast amplitude must lie in (0, 0.5]")
    if int(args.n_phases) < 8 or int(args.n_phases) % 4:
        raise ValueError("n-phases must be a multiple of four and at least eight")
    if int(args.n_directions) < 8:
        raise ValueError("full motion tuning requires at least eight directions")

    model_spec, checkpoint, dataset_config = model_contract(args.model_spec)
    contract = dataset_contract(dataset_config)
    native = recommended_native_grid(
        ppd=float(args.ppd),
        image_size=int(contract["input_size"]),
        frame_rate_hz=float(contract["input_rate_hz"]),
    )
    spatial = _parse_grid(args.spatial_cpd, native["spatial_cpd"])
    temporal = _parse_grid(args.temporal_hz, native["temporal_hz"])
    if spatial[0] * native["fov_deg"] < 1.0 - 1e-8:
        raise ValueError("lowest SF contains less than one cycle across the model aperture")
    if spatial[-1] >= 0.9 * native["spatial_nyquist_cpd"]:
        raise ValueError("highest SF is too close to spatial Nyquist")
    if temporal[0] < 0 or temporal[-1] > 0.8 * native["temporal_nyquist_hz"]:
        raise ValueError("TF grid must remain in [0, 0.8 * temporal Nyquist]")
    directions = np.linspace(0.0, 360.0, int(args.n_directions), endpoint=False)
    phases = np.linspace(0.0, 2.0 * np.pi, int(args.n_phases), endpoint=False)

    model, model_info = load_pinned_multidataset_model(
        checkpoint_path=checkpoint,
        dataset_configs=dataset_config,
        device=args.device,
        strict=True,
    )
    outputs, resolved_mcfarland = load_mcfarland_outputs(args.mcfarland_outputs)
    canonical_rows = canonical_population_rows(model, outputs)
    unit_rows = exact_unit_rows(model, canonical_rows)
    encoding_model = TwinEncodingModel(model=model, device=args.device)
    readout_audit = audit_native_cid_mapping(encoding_model, unit_rows)

    available_channels = np.asarray(
        [int(row["channel"]) for row in unit_rows if bool(row["available"])], dtype=int
    )
    units = pd.DataFrame(
        [
            {
                "unit_index": index,
                "canonical_channel": int(row["channel"]),
                "session": str(row["session"]),
                "cid": int(row["source_cid"]),
                "model_readout_row": int(row["model_readout_row"]),
                "figure3_ccnorm": float(row["ccnorm"]),
            }
            for index, row in enumerate(row for row in unit_rows if bool(row["available"]))
        ]
    )
    if len(units) != len(available_channels):
        raise RuntimeError("available-unit table and channel index disagree")

    blank = torch.zeros(
        1,
        1,
        int(contract["n_lags"]),
        int(contract["input_size"]),
        int(contract["input_size"]),
        device=args.device,
    )
    blank_expected_count = score_exact_histories(
        encoding_model, unit_rows, blank, int(args.batch_size)
    )[0]

    conditions = _conditions(spatial, temporal, directions)
    complete_condition_count = len(conditions)
    if args.max_conditions is not None:
        conditions = conditions.iloc[: int(args.max_conditions)].copy()
    metric_arrays = {
        name: np.empty((len(conditions), len(units)), dtype=np.float32)
        for name in METRIC_NAMES
    }
    for output_row, condition in enumerate(conditions.itertuples(index=False), start=0):
        histories = drifting_histories(
            spatial_cpd=float(condition.spatial_cpd),
            temporal_hz=float(condition.temporal_hz),
            motion_direction_deg=float(condition.motion_direction_deg),
            endpoint_phases_rad=phases,
            n_lags=int(contract["n_lags"]),
            input_size=int(contract["input_size"]),
            ppd=float(args.ppd),
            frame_rate_hz=float(contract["input_rate_hz"]),
            contrast=float(args.contrast),
            device=args.device,
        )
        expected_counts = score_exact_histories(
            encoding_model, unit_rows, histories, int(args.batch_size)
        )
        metrics = response_metrics(phases, expected_counts, blank_expected_count)
        for name in METRIC_NAMES:
            metric_arrays[name][output_row] = metrics[name].astype(np.float32)
        count = output_row + 1
        if count == 1 or count % 25 == 0 or count == len(conditions):
            print(f"exact-CID drifting tuning {count}/{len(conditions)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.out_dir / "units.csv"
    conditions_path = args.out_dir / "conditions.csv"
    responses_path = args.out_dir / "responses.npz"
    units.to_csv(units_path, index=False)
    conditions.to_csv(conditions_path, index=False)
    np.savez_compressed(
        responses_path,
        blank_expected_count=blank_expected_count.astype(np.float32),
        **metric_arrays,
    )
    complete = bool(len(conditions) == complete_condition_count)
    provenance = {
        "analysis": "exact-CID controlled drifting-grating tuning",
        "model_label": str(args.model_label or model_spec.get("label", checkpoint.stem)),
        "model_spec": str(args.model_spec.expanduser().resolve()),
        "model_spec_sha256": sha256_file(args.model_spec.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_config": str(dataset_config),
        "dataset_config_sha256": sha256_file(dataset_config),
        "mcfarland_outputs": str(resolved_mcfarland),
        "mcfarland_outputs_sha256": sha256_file(Path(resolved_mcfarland)),
        "model_info": model_info,
        "contract": contract,
        "ppd": float(args.ppd),
        "contrast_amplitude": float(args.contrast),
        "n_endpoint_phases": int(args.n_phases),
        "spatial_cpd": spatial.tolist(),
        "temporal_hz": temporal.tolist(),
        "motion_direction_deg": directions.tolist(),
        "n_exact_units": int(len(units)),
        "n_canonical_units": int(len(unit_rows)),
        "n_unavailable_units": int(len(unit_rows) - len(units)),
        "readout_identity_audit": readout_audit,
        "n_conditions": int(len(conditions)),
        "n_complete_conditions": int(complete_condition_count),
        "complete_grid": complete,
        "primary_response": (
            "phase-averaged expected spikes per 1/240-s bin minus the explicit "
            "gray-blank expected count; multiply by 240 for spikes/s"
        ),
        "response_units": "expected spikes per 1/240-s model-output bin",
        "source_kind": (
            "controlled synthetic drifting gratings replayed through exact matched "
            "model readouts; not a recorded biological temporal-frequency experiment"
        ),
        "motion_definition": (
            "full 360-degree translation direction; bars are perpendicular to motion"
        ),
        "phase_convergence": (
            "all-phase and every-other-phase estimates are saved from the same replay"
        ),
        "files": {
            "units": str(units_path.resolve()),
            "conditions": str(conditions_path.resolve()),
            "responses": str(responses_path.resolve()),
        },
    }
    provenance_path = args.out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(provenance_path)


if __name__ == "__main__":
    main()
