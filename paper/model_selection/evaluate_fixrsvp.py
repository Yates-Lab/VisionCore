#!/usr/bin/env python3
"""Evaluate a native-240-Hz model on FixRSVP and on Figure 3's 120-Hz frame.

The model predicts one 4.17-ms spike-count bin at a time.  For the
Figure-3 comparison, adjacent native predictions and observations are summed
into the same non-overlapping 8.33-ms bins used by the canonical 120-Hz analysis.
The rebinned observations must match the canonical Figure-3 cache exactly;
otherwise the evaluation aborts rather than reporting misaligned metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dill
import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _fixrsvp_protocol import (  # noqa: E402
    CCNORM_N_SPLITS,
    FIXATION_RADIUS_DEG,
    MIN_FIX_DUR,
    VALID_TIME_BINS,
    _as_numpy,
    _audit_shared_ccnorm,
    _checkpoint_dataset_configs,
    _data_score_mask,
    _finite_summary,
    _population_report,
    _predict_endpoints,
    _prepare_fixrsvp,
    _reference_by_session,
    _session_metrics,
    _validate_observations,
)
from paper.model_selection.audit_production_model import (  # noqa: E402
    resolve,
    sha256,
    verify_file,
)


DEFAULT_SPEC = ROOT / "paper/model_selection/production_model.yaml"


def _native_pair_indices(trial_inds, psth_inds):
    """Return native bins matching the canonical global 240-to-120 rebin.

    The production 120-Hz data loader reshapes the complete source tensor into
    consecutive blocks beginning at source row zero; the correct pairs are
    therefore global rows ``[0, 1], [2, 3], ...``, not bins chosen from each
    trial's local PSTH parity.  Cross-trial blocks are discarded here: they
    become one-sample fractional trial labels in the legacy continuous
    downsampling path and never survive Figure 3's minimum-duration gate.
    """
    trial_inds = np.asarray(trial_inds).ravel()
    psth_inds = np.asarray(psth_inds).ravel().astype(np.int64)
    if trial_inds.shape != psth_inds.shape:
        raise ValueError("trial_inds and psth_inds must have the same length")
    starts = np.arange(0, len(psth_inds) - 1, 2, dtype=np.int64)
    ends = starts + 1
    keep = (
        (trial_inds[starts] == trial_inds[ends])
        & (psth_inds[ends] == psth_inds[starts] + 1)
    )
    return starts[keep], ends[keep]


def _true240_trial_arrays(model, dset, config, dataset_idx, device, batch_size):
    sampling = config.get("sampling") or {}
    if (
        int(sampling.get("source_rate", 0)) != 240
        or int(sampling.get("target_rate", 0)) != 240
        or config.get("supervision")
    ):
        raise ValueError("This evaluator requires true 240-Hz, factor-one supervision")

    trial_inds = _as_numpy(dset["trial_inds"]).ravel()
    psth_inds = _as_numpy(dset["psth_inds"]).ravel().astype(np.int64)
    eyepos = _as_numpy(dset["eyepos"])
    robs = _as_numpy(dset["robs"])
    dfs = _as_numpy(dset["dfs"])
    lags = torch.as_tensor(config["keys_lags"]["stim"], dtype=torch.long)

    # Match the established 240->120 count binning within each trial:
    # PSTH coordinates [0,1], [2,3], ... .  Do not infer phase from row parity.
    pair_start, pair_end = _native_pair_indices(trial_inds, psth_inds)

    pair_eye = 0.5 * (eyepos[pair_start] + eyepos[pair_end])
    fixation = np.hypot(pair_eye[:, 0], pair_eye[:, 1]) < FIXATION_RADIUS_DEG
    pair_start = pair_start[fixation]
    pair_end = pair_end[fixation]
    pair_eye = pair_eye[fixation]

    raw_indices = np.unique(np.concatenate([pair_start, pair_end]))
    inferable = raw_indices >= int(lags.max())
    predictions = np.full((len(dset), robs.shape[1]), np.nan, np.float32)
    predictions[raw_indices[inferable]] = _predict_endpoints(
        model,
        dset,
        raw_indices[inferable],
        lags,
        dataset_idx,
        device,
        batch_size,
    )

    pair_trial = trial_inds[pair_end]
    pair_time = psth_inds[pair_start] // 2
    trials = np.unique(pair_trial)
    n_units = robs.shape[1]
    max_time = int(pair_time.max()) + 1
    native_time = max_time * 2

    paired_robs = np.full((len(trials), max_time, n_units), np.nan, np.float32)
    paired_rhat = np.full_like(paired_robs, np.nan)
    paired_eye = np.full((len(trials), max_time, 2), np.nan, np.float32)
    native_robs = np.full((len(trials), native_time, n_units), np.nan, np.float32)
    native_rhat = np.full_like(native_robs, np.nan)
    native_dfs = np.full_like(native_robs, np.nan)
    fix_dur = np.zeros(len(trials), dtype=np.int64)

    for row, trial in enumerate(trials):
        select = pair_trial == trial
        time120 = pair_time[select]
        if len(np.unique(time120)) != len(time120):
            raise RuntimeError(f"Duplicate rebinned PSTH coordinates in trial {trial}")
        starts = pair_start[select]
        ends = pair_end[select]
        fix_dur[row] = len(time120)
        paired_robs[row, time120] = robs[starts] + robs[ends]
        paired_rhat[row, time120] = predictions[starts] + predictions[ends]
        paired_eye[row, time120] = pair_eye[select]
        native_robs[row, 2 * time120] = robs[starts]
        native_robs[row, 2 * time120 + 1] = robs[ends]
        native_rhat[row, 2 * time120] = predictions[starts]
        native_rhat[row, 2 * time120 + 1] = predictions[ends]
        native_dfs[row, 2 * time120] = dfs[starts]
        native_dfs[row, 2 * time120 + 1] = dfs[ends]

    good = fix_dur > MIN_FIX_DUR
    if good.sum() < 10:
        raise RuntimeError(f"Only {good.sum()} FixRSVP trials exceed {MIN_FIX_DUR} pairs")
    time120 = np.arange(min(VALID_TIME_BINS, max_time))
    time240 = np.arange(min(2 * VALID_TIME_BINS, native_time))
    return {
        "paired120": {
            "robs": paired_robs[good][:, time120],
            "rhat": paired_rhat[good][:, time120],
            # The canonical Figure-3 filters replace this placeholder in scoring.
            "dfs": np.isfinite(paired_robs[good][:, time120]).astype(np.float32),
            "eyepos": paired_eye[good][:, time120],
        },
        "native240": {
            "robs": native_robs[good][:, time240],
            "rhat": native_rhat[good][:, time240],
            "dfs": native_dfs[good][:, time240],
        },
    }


def _native_data_support(robs, native_dfs, reference_dfs):
    """Return data-only support for native-240 likelihood metrics.

    The canonical Figure-3 mask fixes the trial and 120-Hz analysis frame.
    Repeating it selects both constituent native bins; intersecting that mask
    with the native filter prevents an invalid 4.17-ms bin from being scored
    merely because its paired 120-Hz bin is valid.
    """
    robs = np.asarray(robs)
    native_dfs = np.asarray(native_dfs)
    reference_dfs = np.asarray(reference_dfs)
    if robs.shape != native_dfs.shape:
        raise ValueError(
            f"Native observations and filters differ: {robs.shape} versus "
            f"{native_dfs.shape}"
        )
    expected = (
        reference_dfs.shape[0],
        2 * reference_dfs.shape[1],
        reference_dfs.shape[2],
    )
    if robs.shape != expected:
        raise RuntimeError(
            f"Native trace shape {robs.shape} is not twice Figure 3's "
            f"{reference_dfs.shape}"
        )
    expanded_reference = np.repeat(reference_dfs, 2, axis=1)
    return (
        np.isfinite(robs)
        & np.isfinite(native_dfs)
        & (native_dfs > 0)
        & np.isfinite(expanded_reference)
        & (expanded_reference > 0)
    )


def _native_metrics(native, neuron_mask, reference):
    from eval.eval_stack_utils import bits_per_spike, rescale_rhat

    robs = np.asarray(native["robs"][:, :, neuron_mask], np.float32)
    rhat_raw = np.asarray(native["rhat"][:, :, neuron_mask], np.float32)
    native_dfs = np.asarray(native["dfs"][:, :, neuron_mask], np.float32)
    data_support = _native_data_support(
        robs,
        native_dfs,
        np.asarray(reference["dfs_used"], np.float32),
    )
    missing_prediction = data_support & ~np.isfinite(rhat_raw)
    if missing_prediction.any():
        raise RuntimeError(
            "Native-240 prediction is missing on expanded Figure-3 support: "
            f"{int(missing_prediction.sum())} samples"
        )
    dfs = data_support.astype(np.float32)

    shape = robs.shape
    robs_flat = torch.from_numpy(robs.reshape(-1, shape[-1]))
    rhat_flat = torch.from_numpy(rhat_raw.reshape(-1, shape[-1]))
    dfs_flat = torch.from_numpy(dfs.reshape(-1, shape[-1]))
    calibrated, affine = rescale_rhat(robs_flat, rhat_flat, dfs_flat, mode="affine")
    rhat = calibrated.reshape(shape).numpy()
    bps_raw = bits_per_spike(rhat_flat, robs_flat, dfs_flat).numpy()
    bps_affine = bits_per_spike(calibrated, robs_flat, dfs_flat).numpy()

    valid = data_support
    robs_mean = np.nanmean(np.where(valid, robs, np.nan), axis=0)
    rhat_mean = np.nanmean(np.where(valid, rhat, np.nan), axis=0)
    n_valid = valid.sum(axis=0)
    rhos = np.asarray([
        np.corrcoef(
            robs_mean[n_valid[:, unit] > 10, unit],
            rhat_mean[n_valid[:, unit] > 10, unit],
        )[0, 1]
        for unit in range(shape[-1])
    ])
    with torch.no_grad():
        scale = affine.g.exp().cpu().numpy()
        offset = affine.b.exp().cpu().numpy()
    return {
        "robs_used": robs,
        "rhat_used": rhat,
        "dfs_used": dfs,
        "robs_mean": robs_mean,
        "rhat_mean": rhat_mean,
        "rhos": rhos,
        "bps_raw": bps_raw,
        "bps_affine": bps_affine,
        "affine_scale": scale,
        "affine_offset": offset,
        "n_supported_samples": int(data_support.sum()),
    }


def _representative_examples(results, references, n_examples=4):
    """Choose typical and diagnostic units without top-unit selection bias."""
    pool = []
    for result, reference in zip(results, references):
        candidate = np.asarray(result["ccabs"], dtype=float)
        reference_model = np.asarray(reference["ccabs"], dtype=float)
        stable = np.isfinite(result["ccnorm"]) & np.isfinite(reference["ccnorm"])
        for local in np.flatnonzero(stable & np.isfinite(candidate) & np.isfinite(reference_model)):
            pool.append(
                {
                    "result": result,
                    "reference": reference,
                    "unit": int(local),
                    "candidate": float(candidate[local]),
                    "delta": float(candidate[local] - reference_model[local]),
                }
            )
    if not pool:
        raise RuntimeError("no stable common units are available for PSTH examples")

    candidate_values = np.asarray([item["candidate"] for item in pool])
    delta_values = np.asarray([item["delta"] for item in pool])
    specifications = [
        ("strong candidate", "candidate", float(np.quantile(candidate_values, 0.85))),
        ("typical", "candidate", float(np.quantile(candidate_values, 0.50))),
        ("candidate advantage", "delta", float(np.quantile(delta_values, 0.90))),
        ("reference-model advantage", "delta", float(np.quantile(delta_values, 0.10))),
    ]
    chosen = []
    used = set()
    for label, field, target in specifications[: int(n_examples)]:
        order = np.argsort([abs(item[field] - target) for item in pool])
        for index in order:
            item = pool[int(index)]
            key = (str(item["result"]["session"]), int(item["unit"]))
            if key not in used:
                chosen.append((label, item["result"], item["reference"], item["unit"]))
                used.add(key)
                break
    return chosen


def _plot_examples(
    results,
    references,
    output_path,
    *,
    candidate_label="the model",
    n_examples=4,
):
    import matplotlib.pyplot as plt

    chosen = _representative_examples(results, references, n_examples=n_examples)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 2.55 * len(chosen)), squeeze=False)
    colors = {
        "data": "#202020",
        "candidate": "#2c7fb8",
        "reference_model": "#e67e22",
    }
    for row, (selection_label, result, reference, unit) in enumerate(chosen):
        native = result["native240"]
        t240 = np.arange(native["robs_mean"].shape[0]) / 240.0 * 1000.0
        ax = axes[row, 0]
        ax.plot(t240, native["robs_mean"][:, unit] * 240, color=colors["data"], lw=1.25, label="data")
        ax.plot(
            t240,
            native["rhat_mean"][:, unit] * 240,
            color=colors["candidate"],
            lw=1.35,
            label=candidate_label,
        )
        unit_label = int(result["neuron_mask"][unit])
        ax.set_title(
            f"{selection_label} · {result['session']}, unit {unit_label} — native 240 Hz",
            fontsize=10,
        )
        ax.text(
            0.985, 0.78, f"PSTH r = {native['rhos'][unit]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
        )
        ax.set_ylabel("spikes/s")
        ax.set_xlim(0, 1000)
        ax.grid(alpha=0.18)

        valid = _data_score_mask(result["robs_used"], result["dfs_used"])
        valid &= np.isfinite(result["rhat_used"])
        m76_data = np.nanmean(np.where(valid, result["robs_used"], np.nan), axis=0)
        m76_pred = np.nanmean(np.where(valid, result["rhat_used"], np.nan), axis=0)
        reference_model_valid = _data_score_mask(reference["robs_used"], reference["dfs_used"])
        reference_model_valid &= np.isfinite(reference["rhat_used"])
        reference_model_pred = np.nanmean(np.where(reference_model_valid, reference["rhat_used"], np.nan), axis=0)
        t120 = np.arange(m76_data.shape[0]) / 120.0 * 1000.0
        ax = axes[row, 1]
        ax.plot(t120, m76_data[:, unit] * 120, color=colors["data"], lw=1.35, label="data")
        ax.plot(t120, reference_model_pred[:, unit] * 120, color=colors["reference_model"], lw=1.35, label="reference model")
        ax.plot(
            t120,
            m76_pred[:, unit] * 120,
            color=colors["candidate"],
            lw=1.35,
            label=candidate_label,
        )
        ax.set_title("exact paired Figure-3 view — 120 Hz", fontsize=10)
        ax.text(
            0.985, 0.06,
            f"CCnorm: {candidate_label} {result['ccnorm'][unit]:.2f}, "
            f"reference model {reference['ccnorm'][unit]:.2f}\n"
            f"single-trial $R^2$: {result['ve_model'][unit]:.3f} vs "
            f"{reference['ve_model'][unit]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        )
        ax.set_xlim(0, 1000)
        ax.grid(alpha=0.18)
        if row == 0:
            axes[row, 0].legend(frameon=False, ncol=2)
            axes[row, 1].legend(frameon=False, ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel("time in FixRSVP trial (ms)")
    fig.suptitle(
        f"{candidate_label}: native output and exact paired Figure-3 comparison",
        y=0.998,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Evaluate only this exact session name (repeatable)",
    )
    parser.add_argument(
        "--canonical-observation-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--examples-out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec_path = args.model_spec.expanduser().resolve()
    spec = yaml.safe_load(spec_path.read_text())
    checkpoint_path = verify_file(spec["checkpoint"], label="production checkpoint")
    configured_dataset = verify_file(
        spec["training"]["datasets"]["descriptive_all_gratings"],
        label="fixed-RSVP dataset config",
    )
    reference_cache = args.canonical_observation_cache.expanduser().resolve()
    if not reference_cache.is_file():
        raise FileNotFoundError(reference_cache)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "model_spec": str(spec_path),
                    "model_spec_sha256": sha256(spec_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "dataset_config": str(configured_dataset),
                    "dataset_config_sha256": sha256(configured_dataset),
                    "canonical_observation_cache": str(reference_cache),
                    "canonical_observation_cache_sha256": sha256(reference_cache),
                },
                indent=2,
            )
        )
        return
    if os.environ.get("CONDA_DEFAULT_ENV") != "yatesfv":
        raise RuntimeError(
            "Fixed-RSVP production evaluation must run in conda environment "
            "'yatesfv'."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configs, cfg_path = _checkpoint_dataset_configs(checkpoint)
    if resolve(cfg_path) != configured_dataset:
        raise ValueError(
            "Checkpoint dataset config does not match the production spec: "
            f"{resolve(cfg_path)} != {configured_dataset}"
        )
    config_by_session = {config["session"]: config for config in configs}
    reference_by = _reference_by_session(reference_cache)
    names = [name for name in config_by_session if name in reference_by]
    if args.session:
        requested = list(dict.fromkeys(args.session))
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(
                "Requested sessions are unavailable in both the checkpoint "
                f"config and Figure-3 cache: {missing}"
            )
        names = requested
    if args.max_sessions is not None:
        names = names[: args.max_sessions]

    from eval.load_twin import load_twin

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, info = load_twin(checkpoint_path, device=str(device), verbose=True)
    model_name_to_idx = {name: idx for idx, name in enumerate(model.names)}

    results = []
    references = []
    checks = []
    metric_audits = []
    for ordinal, name in enumerate(names, 1):
        print(f"\n[{ordinal}/{len(names)}] true-240 FixRSVP {name}", flush=True)
        dset, resolved = _prepare_fixrsvp(config_by_session[name])
        arrays = _true240_trial_arrays(
            model, dset, resolved, model_name_to_idx[name], device, args.batch_size
        )
        reference = reference_by[name]
        neuron_mask = np.asarray(reference["neuron_mask"], dtype=np.int64)
        metrics120 = _session_metrics(
            arrays["paired120"], neuron_mask, reference=reference
        )
        check = _validate_observations(metrics120, reference)
        if not check["exact_match"]:
            raise RuntimeError(f"{name}: paired 240-Hz observations do not exactly match Figure 3: {check}")
        audited_reference, metric_audit = _audit_shared_ccnorm(metrics120, reference)
        native = _native_metrics(arrays["native240"], neuron_mask, reference)
        result = {
            "session": name,
            "subject": name.split("_")[0],
            "neuron_mask": neuron_mask,
            "n_trials": int(metrics120["robs_used"].shape[0]),
            "n_time": int(metrics120["robs_used"].shape[1]),
            "n_neurons": int(metrics120["robs_used"].shape[2]),
            "eyepos_used": arrays["paired120"]["eyepos"],
            "native240": native,
            **metrics120,
        }
        print(
            f"  exact_pair_match=True cells={result['n_neurons']} "
            f"rho240={np.nanmedian(native['rhos']):.4f} "
            f"rho120={np.nanmedian(result['rhos']):.4f} "
            f"ccnorm120={np.nanmedian(result['ccnorm']):.4f}",
            flush=True,
        )
        check["session"] = name
        metric_audit["session"] = name
        checks.append(check)
        metric_audits.append(metric_audit)
        results.append(result)
        references.append(audited_reference)
        del dset, arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()

    population120 = _population_report(results, references)
    native_values = {
        key: np.concatenate([np.asarray(result["native240"][key]) for result in results])
        for key in ("rhos", "bps_raw", "bps_affine")
    }
    report = {
        "schema_version": 1,
        "model_spec": str(spec_path),
        "model_spec_sha256": sha256(spec_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_label": str(spec["label"]),
        "dataset_config": str(cfg_path),
        "dataset_config_sha256": sha256(configured_dataset),
        "canonical_observation_cache": str(reference_cache),
        "canonical_observation_cache_sha256": sha256(reference_cache),
        "protocol": {
            "native_rate_hz": 240,
            "figure3_rate_hz": 120,
            "rebinning": "sum adjacent native prediction/count pairs [0,1], [2,3], ...",
            "alignment_gate": "rebinned observations exactly equal canonical Figure-3 observations",
            "affine_calibration": "independent positive per-cell affine Poisson fits at 240 and 120 Hz",
            "ccnorm_splits_per_seed": CCNORM_N_SPLITS,
            "canonical_observation_cache": str(reference_cache),
        },
        "model": info,
        "observation_alignment": checks,
        "metric_audits": metric_audits,
        "paired120_population": population120,
        "native240_population": {
            key: _finite_summary(value) for key, value in native_values.items()
        },
        "sessions": [
            {
                "session": result["session"],
                "n_trials": result["n_trials"],
                "n_neurons": result["n_neurons"],
                "rho240": _finite_summary(result["native240"]["rhos"]),
                "rho120": _finite_summary(result["rhos"]),
                "ccnorm120": _finite_summary(result["ccnorm"]),
                "ve120": _finite_summary(result["ve_model"]),
            }
            for result in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    args.trace_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_cache.open("wb") as stream:
        dill.dump(results, stream)
    _plot_examples(
        results,
        references,
        args.examples_out,
        candidate_label="the model",
    )
    print(json.dumps({
        "paired120": report["paired120_population"],
        "native240": report["native240_population"],
    }, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.trace_cache}")
    print(f"Wrote {args.examples_out}")


if __name__ == "__main__":
    main()
