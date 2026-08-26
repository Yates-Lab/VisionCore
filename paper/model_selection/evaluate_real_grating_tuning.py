#!/usr/bin/env python3
"""Replay recorded real grating trials through one pinned encoding model.

The assay can use either all genuine physical trials (descriptive capture) or
the same final 15% for every model (generalization).  Only the canonical
Figure-3 neural population (CCnorm > .5 in the McFarland bundle) is retained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_stack_utils import evaluate_dataset
from eval.real_grating_tuning import analyze_tuning, condition_lag_tensors, stable_trial_mask
from models.data import prepare_data
from paper.fig4.upstream.real_trace_matrix.model import (
    load_mcfarland_outputs,
    load_pinned_multidataset_model,
)
DEFAULT_OUT = ROOT / "outputs/dekel240_paper/real_neuron_grating_tuning"
DEFAULT_MODEL_SPEC = ROOT / "paper/model_selection/production_model.yaml"


def _output_cids_used(output: dict, n_output_rows: int) -> np.ndarray:
    """Return the biological CID associated with every evaluation row."""
    for key in ("cids_used", "cids"):
        values = np.asarray(output.get(key, []))
        if values.ndim == 1 and values.size == int(n_output_rows):
            return values.astype(np.int64, copy=False)
    raise ValueError(
        f"McFarland output {output.get('sess', '<unknown>')!r} has "
        f"{n_output_rows} rows but no equally sized cids_used/cids array"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_MODEL_SPEC)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Explicit candidate checkpoint; overrides the named model checkpoint",
    )
    parser.add_argument(
        "--dataset-config", type=Path, default=None,
        help="Dataset config for an explicit checkpoint",
    )
    parser.add_argument(
        "--model-label", default=None,
        help="Output label for an explicit checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--data-scope",
        choices=("heldout", "all"),
        default="all",
        help=(
            "Physical grating repeats to use. The model-capture diagnostic defaults "
            "to every valid repeat; heldout remains available only as an optional "
            "generalization analysis."
        ),
    )
    parser.add_argument("--sessions", default=None, help="Comma-separated session names")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-lag-ms", type=float, default=125.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_population() -> pd.DataFrame:
    outputs, source = load_mcfarland_outputs(ROOT / "scripts/mcfarland_outputs_mono.pkl")
    rows = []
    channel = 0
    for output in outputs:
        ccnorm = np.asarray(output["ccnorm"]["ccnorm"], dtype=np.float64)
        cids = _output_cids_used(output, len(ccnorm))
        for source_row in np.flatnonzero(ccnorm > 0.5):
            rows.append({
                "canonical_channel": channel,
                "session": str(output["sess"]),
                "cid": int(cids[source_row]),
                "figure3_ccnorm": float(ccnorm[source_row]),
                "source_output_row": int(source_row),
                "mcfarland_source": str(source),
            })
            channel += 1
    table = pd.DataFrame(rows)
    if len(table) != 756:
        raise RuntimeError(f"Expected 756 canonical channels, found {len(table)}")
    if table.duplicated(["session", "cid"]).any():
        raise RuntimeError("Canonical population contains duplicate session/cid keys")
    return table


def _model_configs(model) -> list[dict]:
    for attr in ("cfgs", "dataset_configs"):
        value = getattr(model, attr, None)
        if value is not None:
            return list(value)
    raise AttributeError("Loaded model exposes neither cfgs nor dataset_configs")


def prepare_gratings(model, dataset_idx: int, data_scope: str = "all"):
    config = copy.deepcopy(_model_configs(model)[dataset_idx])
    config["types"] = ["gratings"]
    # A single physical test set shared by the historical 80/20 and current
    # 70/15/15 protocols: the final 15% under the frozen trial permutation.
    config["train_val_split"] = 0.70
    config["test_split"] = 0.15
    train, val, test, resolved = prepare_data(config, strict=True, return_test=True)
    del train, val
    dset_idx = test.get_dataset_index("gratings")
    dset = test.dsets[dset_idx]
    # Select the physical trial IDs explicitly.  The generic 120-Hz
    # downsampler averages continuous covariates, including trial_inds, and a
    # pair straddling a trial boundary can otherwise create one fractional
    # pseudo-trial.  Excluding those boundary pairs before the split keeps the
    # native-240 and legacy-120 assays on exactly the same trials.
    trial_values = dset["trial_inds"].double()
    rounded_trials = torch.round(trial_values).long()
    # Trial IDs were incorrectly treated as continuous by the generic 120-Hz
    # converter.  Long constant runs identify real trials and remove both
    # fractional and accidentally integer-valued boundary averages.
    stable_trial = torch.from_numpy(stable_trial_mask(trial_values.cpu().numpy())).to(
        device=trial_values.device
    )
    unique_trials = torch.unique(rounded_trials[stable_trial])
    if data_scope == "all":
        selected_trials = unique_trials
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 1002)))
        permutation = torch.randperm(len(unique_trials), generator=generator)
        test_start = int(len(unique_trials) * 0.85)
        selected_trials = unique_trials[permutation[test_start:]]
    valid = dset["dfs"].any(dim=1)
    raw_rows = torch.nonzero(
        valid & stable_trial & torch.isin(rounded_trials, selected_trials),
        as_tuple=True,
    )[0]
    indices = torch.stack((torch.full_like(raw_rows, dset_idx), raw_rows), dim=1)
    if indices.numel() == 0:
        raise RuntimeError(f"No selected grating samples for {resolved['session']}")
    return test, indices, resolved


def evaluate_session(
    *,
    model,
    model_label: str,
    session: str,
    population: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
    max_lag_ms: float,
    data_scope: str,
) -> pd.DataFrame:
    dataset_idx = list(model.names).index(session)
    grating_data, indices, config = prepare_gratings(
        model, dataset_idx, data_scope=data_scope
    )
    result = evaluate_dataset(
        model, grating_data, indices, dataset_idx, batch_size=batch_size,
        desc=f"{model_label} {session} {data_scope} gratings",
    )
    dset_idx = int(indices[:, 0].unique().item())
    raw_indices = indices[:, 1].cpu().numpy().astype(np.int64)
    dset = grating_data.dsets[dset_idx]
    cids = np.asarray(config["cids"], dtype=np.int64)
    canonical = population.loc[population.session.eq(session)].copy()
    row_by_cid = {int(cid): row for row, cid in enumerate(cids)}
    canonical["model_readout_row"] = canonical.cid.map(row_by_cid)
    canonical["available"] = canonical.model_readout_row.notna()
    available = canonical.loc[canonical.available].copy()
    readout_rows = available.model_readout_row.astype(int).to_numpy()
    if not len(readout_rows):
        raise RuntimeError(f"No canonical units available in {model_label} for {session}")

    robs = result["robs"].numpy()[:, readout_rows]
    rhat = result["rhat"].float().numpy()[:, readout_rows]
    dfs = result["dfs"].numpy()
    if dfs.ndim == 2 and dfs.shape[1] > 1:
        dfs = dfs[:, readout_rows]
    elif dfs.ndim == 1 or dfs.shape[1] == 1:
        dfs = np.repeat(np.asarray(dfs).reshape(-1, 1), len(readout_rows), axis=1)
    sf = dset["sf"][raw_indices].cpu().numpy().squeeze()
    ori = dset["ori"][raw_indices].cpu().numpy().squeeze()
    phase_map = dset["stim_phase"][raw_indices].cpu().numpy()
    phase = phase_map[:, phase_map.shape[1] // 2, phase_map.shape[2] // 2]
    trials = dset["trial_inds"][raw_indices].cpu().numpy().squeeze()
    dt = 1.0 / float(config["sampling"]["target_rate"])

    lag_tensors = condition_lag_tensors(
        robs=robs,
        rhat=rhat,
        dfs=dfs,
        sf=sf,
        ori=ori,
        trials=trials,
        raw_indices=raw_indices,
        dt=dt,
        max_lag_ms=max_lag_ms,
    )
    curves, metric_rows = analyze_tuning(
        lag_tensors=lag_tensors,
        robs=robs,
        rhat=rhat,
        dfs=dfs,
        sf=sf,
        ori=ori,
        phase=phase,
        trials=trials,
        raw_indices=raw_indices,
        dt=dt,
    )
    metrics = pd.DataFrame(metric_rows)
    metrics = pd.concat(
        [available.reset_index(drop=True), metrics.drop(columns="unit")], axis=1
    )
    metrics.insert(0, "model", model_label)
    bps = np.asarray(result["bps"], dtype=np.float64)
    # ``grating_test_bps`` is retained for compatibility with the existing
    # population renderers.  ``grating_bps`` is the scope-neutral name: under
    # --data-scope all it is explicitly computed from every valid repeat.
    metrics["grating_bps"] = bps[readout_rows]
    metrics["data_scope"] = data_scope
    metrics["grating_test_bps"] = bps[readout_rows]
    metrics["sample_rate_hz"] = int(round(1.0 / dt))
    metrics["n_samples"] = len(raw_indices)
    metrics["n_trials"] = len(np.unique(trials))
    # Compatibility aliases for previously rendered population summaries.
    metrics["n_test_samples"] = metrics["n_samples"]
    metrics["n_test_trials"] = metrics["n_trials"]
    metrics["test_spikes"] = metrics["n_spikes"]

    session_dir = output_dir / model_label / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        session_dir / f"{session}_curves.npz",
        cids=available.cid.to_numpy(dtype=np.int64),
        canonical_channels=available.canonical_channel.to_numpy(dtype=np.int64),
        **curves,
    )
    metrics.to_csv(session_dir / f"{session}_metrics.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(args.model_spec.read_text(encoding="utf-8")) or {}
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        dataset_config = args.dataset_config or (
            ROOT / "paper/model_selection/configs/multi_240_long_split3_dekel35.yaml"
        )
        model_label = args.model_label or checkpoint.parent.name
        default_batch_size = 256
    else:
        checkpoint = Path(spec["checkpoint"]["path"])
        dataset_config = args.dataset_config or Path(
            spec["training"]["datasets"]["descriptive_all_gratings"]["path"]
        )
        if not dataset_config.is_absolute():
            dataset_config = ROOT / dataset_config
        model_label = args.model_label or str(spec["label"])
        default_batch_size = 256
    for path in (checkpoint, dataset_config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.checkpoint is None:
        expected_checkpoint = str(spec["checkpoint"]["sha256"])
        expected_dataset = str(
            spec["training"]["datasets"]["descriptive_all_gratings"]["sha256"]
        )
        if sha256(checkpoint) != expected_checkpoint:
            raise ValueError("checkpoint digest does not match the production model spec")
        if sha256(dataset_config) != expected_dataset:
            raise ValueError("dataset-config digest does not match the production model spec")
    population = canonical_population()
    sessions = list(dict.fromkeys(population.session.tolist()))
    if args.sessions:
        requested = [s.strip() for s in args.sessions.split(",") if s.strip()]
        missing = sorted(set(requested).difference(sessions))
        if missing:
            raise ValueError(f"Sessions are outside canonical population: {missing}")
        sessions = requested
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]

    model, model_info = load_pinned_multidataset_model(
        checkpoint_path=checkpoint,
        dataset_configs=dataset_config,
        device=args.device,
        strict=True,
    )
    model.model.eval()
    convnet = getattr(model.model, "convnet", None)
    if convnet is not None and hasattr(convnet, "use_checkpointing"):
        convnet.use_checkpointing = False
    batch_size = int(args.batch_size or default_batch_size)
    output_dir = args.output_dir or (
        DEFAULT_OUT if args.data_scope == "heldout" else DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_all_data")
    )
    model_dir = output_dir / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for session in sessions:
        cache = model_dir / "sessions" / f"{session}_metrics.csv"
        curve_cache = model_dir / "sessions" / f"{session}_curves.npz"
        if cache.exists() and curve_cache.exists() and not args.force:
            print(f"{model_label} {session}: using cache")
            all_metrics.append(pd.read_csv(cache))
            continue
        all_metrics.append(evaluate_session(
            model=model,
            model_label=model_label,
            session=session,
            population=population,
            output_dir=output_dir,
            batch_size=batch_size,
            max_lag_ms=args.max_lag_ms,
            data_scope=args.data_scope,
        ))
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(model_dir / "unit_metrics.csv", index=False)
    availability = population.merge(
        combined[["session", "cid"]].drop_duplicates().assign(available=True),
        on=["session", "cid"], how="left",
    )
    availability["available"] = availability.available.eq(True)
    availability.to_csv(model_dir / "canonical_availability.csv", index=False)
    availability.loc[~availability.available].to_csv(
        model_dir / "missing_canonical_units.csv", index=False
    )
    provenance = {
        "model": model_label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_config": str(dataset_config),
        "dataset_config_sha256": sha256(dataset_config),
        "device": args.device,
        "batch_size": batch_size,
        "sessions": sessions,
        "canonical_population_n": int(len(population)),
        "available_population_n": int(len(combined)),
        "missing_population_n": int((~availability.available).sum()),
        "data_scope": args.data_scope,
        "trial_selection": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "seed": 1002,
            "same_physical_trials_for_all_models": True,
            "selected_trials": "all genuine trials" if args.data_scope == "all" else "final 15%",
            "trial_identity_guard": (
                "integer-valued constant runs of >=4 samples; rejects continuous-downsampling "
                "boundary averages, including accidentally integer-valued artifacts"
            ),
        },
        "assay": {
            "spatial_frequencies_cpd": (
                "read from each session's measured conditions; the Allen and Logan banks differ"
            ),
            "orientations_deg": [11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75],
            "phase": "retinal phase at crop center; phase zero retained; negative values invalid",
            "temporal_measure": "response-lag profile, not temporal-frequency tuning",
            "max_lag_ms": args.max_lag_ms,
            "preference_interpolation": "three-point local quadratic; boundary peaks censored",
            "modulation_index": "standard F1/F0 = fitted first-harmonic amplitude / offset",
        },
        "model_info": {k: str(v) if isinstance(v, Path) else v for k, v in model_info.items()},
    }
    (model_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"Saved {len(combined)} unit rows to {model_dir}")


if __name__ == "__main__":
    main()
