"""Score a Stage 0 run on the metrics `protocol.py` declares.

Training computes only BPS, and only on validation. Two gaps follow from that,
and this module closes both:

1. **The test split was built and never scored.** `test_split` is wired through
   `prepare_data` to `MultiDatasetDM.test_dataloader()`, but nothing in the
   repo called it. Selection stays on validation by design; the selected model
   has to be reported on trials that selection never saw.
2. **CC_norm and single-trial r^2 were declared and never computed.**
   `protocol.METRICS` names all three. Single-trial r^2 is what Figures 3 and 4
   actually rest on, and it is measured on the held-out `fixrsvp` condition,
   which is withheld from fitting entirely -- so it is out-of-domain
   generalisation, not a within-distribution test score.

The two passes are independent and answer different questions. The test-split
pass asks "how well does this run fit the conditions it was trained on, on
trials it never saw". The fixrsvp pass asks "does it transfer to the condition
every figure reports on". Selection uses neither; both describe the winner.

    uv run python paper/model_selection/evaluate.py E1a --gpu 0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VISIONCORE_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

import protocol  # noqa: E402
from protocol import PROTOCOL_HASH, assert_same_protocol  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def single_trial_r2(rhat, robs, dfs):
    """Per-unit variance explained on single trials, over model-valid bins.

    This is the quantity Figures 3 and 4 rest on, so it uses the same form as
    `paper/fig3`: ``1 - Var(rhat - robs) / Var(robs)``, per unit, over the bins
    the data filter admits. Predicting each trial exactly scores 1; predicting
    the unit's mean on every trial scores 0; a prediction worse than that mean
    scores below 0, which is meaningful and is not clipped.

    Parameters
    ----------
    rhat, robs, dfs : array_like, (trial, time, unit)
        Prediction, observed counts, and the data filter. Bins where ``dfs``
        is non-positive or non-finite are excluded from both variances.

    Returns
    -------
    ndarray, (unit,)
        NaN for a unit with no valid bins or no observed variance.
    """
    rhat = np.asarray(rhat, dtype=float)
    robs = np.asarray(robs, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    if rhat.shape != robs.shape or dfs.shape != robs.shape:
        raise ValueError("rhat, robs and dfs must all be (trial, time, unit)")

    valid = np.isfinite(dfs) & (dfs > 0) & np.isfinite(robs) & np.isfinite(rhat)
    robs = np.where(valid, robs, np.nan)
    rhat = np.where(valid, rhat, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        var_obs = np.nanvar(robs, axis=(0, 1))
        var_res = np.nanvar(rhat - robs, axis=(0, 1))
        out = 1.0 - var_res / var_obs
    return np.where(var_obs > 0, out, np.nan)


def bps_per_unit(rhat, robs, dfs):
    """Per-unit bits-per-spike over model-valid bins of assembled trials.

    `bits_per_spike` sanitises NaN in the rates but not in `dfs`, and the
    fixrsvp pass assembles ragged trials into a full (trial, time, unit) array
    whose unreached slots are NaN. Left alone that makes `T = dfs.sum(0)` NaN
    and every unit scores NaN -- an absent metric rather than an error, which is
    the kind of silence that reaches a table unnoticed. Treat a non-finite
    filter as "not valid data", which is what it means.
    """
    import torch
    from eval.eval_stack_utils import bits_per_spike

    rhat = np.asarray(rhat, dtype=float)
    robs = np.asarray(robs, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    n_units = robs.shape[-1]

    valid = np.isfinite(dfs) & (dfs > 0) & np.isfinite(robs) & np.isfinite(rhat)
    out = bits_per_spike(
        torch.from_numpy(np.where(valid, rhat, 0.0).reshape(-1, n_units)),
        torch.from_numpy(np.where(valid, robs, 0.0).reshape(-1, n_units)),
        torch.from_numpy(valid.astype(float).reshape(-1, n_units)),
    )
    return np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out,
                      dtype=float)


def nanmedian_or_none(values):
    """Median over finite entries, or None when there are none.

    A whole session can legitimately score no cells -- the two-seed CC_norm
    convention drops every cell whose split-half estimates disagree, which a
    low-trial session can trigger across the board. That is an absent number,
    not a NaN to propagate or a warning to emit.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def overall_bps(per_dataset):
    """Reduce per-unit BPS to one number, the way training does.

    Mirrors `MultiDatasetModel.on_validation_epoch_end`: drop NaN units (cells
    that saw no samples), clamp a negative per-unit BPS to zero, average within
    a dataset, then average over the datasets that produced anything. A dataset
    with nothing scored is skipped rather than counted as zero, which would
    penalise a run for a session that contributed no valid samples.

    This must agree with the training reduction or a run's test BPS is not
    comparable to the validation BPS it was selected on. It deliberately does
    no distributed reduction: evaluation runs on a single device.

    Parameters
    ----------
    per_dataset : mapping of str to array_like
        Dataset name -> per-unit BPS.

    Returns
    -------
    overall : float
        NaN if no dataset scored anything.
    per_dataset_mean : dict
        Dataset name -> its mean, for datasets that scored at least one unit.
    """
    per_dataset_mean = {}
    for name, bps in per_dataset.items():
        bps = np.asarray(bps, dtype=float)
        valid = bps[np.isfinite(bps)]
        if valid.size == 0:
            continue
        per_dataset_mean[name] = float(np.clip(valid, 0.0, None).mean())

    if not per_dataset_mean:
        return float("nan"), per_dataset_mean
    return float(np.mean(list(per_dataset_mean.values()))), per_dataset_mean


# ---------------------------------------------------------------------------
# Inclusion
# ---------------------------------------------------------------------------
def apply_protocol_inclusion(sessions):
    """Apply `protocol.py` unit inclusion, then the session floor.

    Order is load-bearing: the floor counts units that already passed unit
    inclusion, so a session can be dropped by the cells it lost rather than by
    the cells it started with. Matches the figure 2 / figure 3 population, so
    model-selection numbers describe the neurons the paper reports on.

    Parameters
    ----------
    sessions : iterable of dict
        Each needs `session`, `total_spikes` (unit,) and `psth_r2` (unit,).

    Returns
    -------
    list of dict
        The surviving sessions, each with an added boolean `include` mask.
        Input dicts are not mutated.
    """
    kept = []
    for entry in sessions:
        total_spikes = np.asarray(entry["total_spikes"], dtype=float)
        psth_r2 = np.asarray(entry["psth_r2"], dtype=float)

        include = (
            np.isfinite(total_spikes)
            & (total_spikes > protocol.MIN_TOTAL_SPIKES)
            & np.isfinite(psth_r2)
            & (psth_r2 > protocol.MIN_PSTH_R2)
        )
        if int(include.sum()) < protocol.MIN_SESSION_UNITS:
            continue

        out = dict(entry)
        out["include"] = include
        kept.append(out)
    return kept


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pass 1: the three-way test split, on the conditions the twin was fit to
# ---------------------------------------------------------------------------
def score_test_split(model, datamodule, device):
    """Per-unit BPS over `MultiDatasetDM.test_dataloader()`.

    Deliberately not `trainer.test()`. That would need a `test_step` on
    `MultiDatasetModel`, which is shared training code in the path Figures 1-4
    depend on; this reproduces the same aggregation with the same
    `PoissonBPSAggregator` and touches nothing. `test_dataloader()` raises
    rather than falling back to validation when no `test_split` is configured,
    so a two-way run cannot silently report its selection split here.

    Returns
    -------
    dict
        Dataset name -> per-unit BPS array, NaN where a unit saw no samples.
    """
    import torch
    from torch import nn
    from models.losses import PoissonBPSAggregator

    loader = datamodule.test_dataloader()
    aggs = {name: PoissonBPSAggregator() for name in model.names}
    identity_activation = isinstance(model.model.activation, nn.Identity)

    # The datasets are stored bfloat16 (`--dset_dtype bfloat16`) while the
    # weights are float32. Training only survives that because it runs under
    # `precision="bf16-mixed"` and Lightning autocasts for it; a hand-rolled
    # loop has to do the same or `F.conv2d` raises on the dtype mismatch.
    use_autocast = str(device).startswith("cuda")

    model.eval()
    with torch.no_grad():
        for batch in loader:
            for b in ([batch] if isinstance(batch, dict) else batch):
                b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
                ds_idx = int(b["dataset_idx"][0])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_autocast):
                    rhat = model(
                        b["stim"],
                        b["dataset_idx"][0],
                        b.get("behavior"),
                        b.get("history"),
                    )
                rhat = rhat.float()
                # Matches `_step`: an identity activation means the model was
                # trained with log_input=True and emits log-rates.
                if identity_activation:
                    rhat = torch.exp(rhat)
                aggs[model.names[ds_idx]]({
                    "rhat": rhat,
                    "robs": b["robs"].float(),
                    "dfs": b["dfs"].float(),
                })

    out = {}
    for name, agg in aggs.items():
        if len(agg.robs) == 0:
            continue
        bps = agg.closure()
        if bps is not None:
            out[name] = bps.detach().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Pass 2: the held-out fixrsvp condition
# ---------------------------------------------------------------------------
def score_fixrsvp(model, device, session_filter=None, n_splits=None):
    """CC_norm, single-trial r^2 and BPS on the withheld `fixrsvp` condition.

    `fixrsvp` is never fit (`protocol.HELD_OUT_TYPES`), so this is out-of-domain
    generalisation rather than a within-distribution test score -- a stronger
    claim, and the condition every figure reports on. The trial assembly,
    affine rescaling and split-half CC_norm mirror `paper/fig3/_fig3_data.py`
    exactly, so these numbers are commensurable with the figures.

    Returns
    -------
    list of dict
        One entry per scored session, carrying `total_spikes` and the per-unit
        metrics. Pass the list to `apply_protocol_inclusion` to reduce it to the
        reported population.
    """
    import torch
    from tqdm import tqdm
    from eval.eval_stack_utils import (
        load_single_dataset,
        run_model,
        rescale_rhat,
        ccnorm_split_half_variable_trials,
    )

    n_splits = protocol.CCNORM_N_SPLITS if n_splits is None else n_splits
    results = []

    for dataset_idx, session_name in enumerate(model.names):
        if session_filter is not None and session_name not in session_filter:
            continue

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds("fixrsvp"),
                val_data.get_dataset_inds("fixrsvp"),
            ], dim=0)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {session_name}: skipping ({exc})")
            continue

        dset = train_data.dsets[fixrsvp_inds[:, 0].unique().item()]
        trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
        psth_inds_flat = np.asarray(dset.covariates["psth_inds"]).ravel()
        robs_flat = np.asarray(dset["robs"])
        eyepos_flat = np.asarray(dset["eyepos"])
        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

        trials = np.unique(trial_inds)
        n_trials, n_units = len(trials), robs_flat.shape[1]
        n_time = int(psth_inds_flat.max()) + 1
        stim_lags = np.array(dataset_config["keys_lags"]["stim"])

        robs = np.full((n_trials, n_time, n_units), np.nan)
        rhat = np.full((n_trials, n_time, n_units), np.nan)
        dfs = np.full((n_trials, n_time, n_units), np.nan)
        fix_dur = np.full(n_trials, np.nan)

        for itrial in tqdm(range(n_trials), desc=f"  {session_name}", leave=False):
            ix = (trial_inds == trials[itrial]) & fixation
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset["stim"][stim_lag_indices].permute(0, 2, 1, 3, 4)
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            dfs[itrial, t_inds] = np.asarray(dset["dfs"][ix])
            out = run_model(model, {"stim": stim, "behavior": dset["behavior"][ix]},
                            dataset_idx=dataset_idx)
            rhat[itrial, t_inds] = out["rhat"].detach().cpu().numpy()

        good = fix_dur > protocol.MIN_FIX_DUR
        if good.sum() < 10:
            print(f"  {session_name}: skipping (only {int(good.sum())} good trials)")
            continue

        keep_bins = np.arange(min(protocol.VALID_TIME_BINS, n_time))
        robs = robs[good][:, keep_bins]
        rhat = rhat[good][:, keep_bins]
        dfs = dfs[good][:, keep_bins]
        n_kept_trials, n_kept_time = robs.shape[:2]

        # One finite, data-only support for affine fitting and every metric.
        # Numeric NaN filters must never be passed through as truthy weights.
        support = np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)
        missing_prediction = support & ~np.isfinite(rhat)
        if missing_prediction.any():
            raise RuntimeError(
                f"{session_name}: {int(missing_prediction.sum())} predictions "
                "missing on Figure-3 support"
            )

        # Affine-rescale predictions to the observed counts, as the figures do.
        rescaled, _ = rescale_rhat(
            torch.from_numpy(robs.reshape(-1, n_units)),
            torch.from_numpy(rhat.reshape(-1, n_units)),
            torch.from_numpy(
                support.astype(np.float32).reshape(-1, n_units)
            ),
            mode="affine",
        )
        rhat = rescaled.reshape(n_kept_trials, n_kept_time, n_units).cpu().numpy()

        # CCnorm uses one explicit, model-independent support.  Average the
        # numerator and data-only ceiling separately so the exact identity
        # CCnorm = CCabs / CCmax is retained.  The stability exclusion is also
        # data-only; gating on two CCnorm estimates can retain different units
        # for different models even when their observations are identical.
        missing_prediction = support & ~np.isfinite(rhat)
        if missing_prediction.any():
            raise RuntimeError(
                f"{session_name}: {int(missing_prediction.sum())} predictions "
                "missing on Figure-3 CCnorm support"
            )
        _, ccabs1, ccmax1, _, _ = ccnorm_split_half_variable_trials(
            robs, rhat, support,
            n_splits=n_splits, return_components=True, rng=42,
        )
        _, ccabs2, ccmax2, _, _ = ccnorm_split_half_variable_trials(
            robs, rhat, support,
            n_splits=n_splits, return_components=True, rng=43,
        )
        if not np.allclose(ccabs1, ccabs2, rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError("CCabs changed across data-only split-half seeds")
        ccabs = 0.5 * (np.asarray(ccabs1) + np.asarray(ccabs2))
        ccmax = 0.5 * (np.asarray(ccmax1) + np.asarray(ccmax2))
        with np.errstate(divide="ignore", invalid="ignore"):
            ccnorm = ccabs / ccmax
        ccnorm[(np.asarray(ccmax1) - np.asarray(ccmax2)) ** 2 > 0.01] = np.nan

        bps = bps_per_unit(rhat, robs, support)

        results.append({
            "session": session_name,
            "dataset_idx": dataset_idx,
            "n_trials": int(n_kept_trials),
            "total_spikes": np.nansum(robs, axis=(0, 1)),
            "ccnorm": np.asarray(ccnorm, dtype=float),
            "ccabs": np.asarray(ccabs, dtype=float),
            "ccmax": np.asarray(ccmax, dtype=float),
            "single_trial_r2": single_trial_r2(rhat, robs, support),
            "bps": bps,
        })
        cc_med = nanmedian_or_none(results[-1]["ccnorm"])
        r2_med = nanmedian_or_none(results[-1]["single_trial_r2"])
        print(f"  {session_name}: {n_kept_trials} trials, {n_units} units, "
              f"median CC_norm {'n/a' if cc_med is None else f'{cc_med:.3f}'}, "
              f"median single-trial r2 "
              f"{'n/a' if r2_med is None else f'{r2_med:.4f}'}")

    return results


def load_run_manifest(run_dir):
    """Read a run's manifest, refusing one produced under another protocol.

    Raises `ValueError` naming both hashes. A sweep runs over days; the failure
    this prevents is pooling a run trained under one split with a run trained
    under another and never noticing.
    """
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Runs are stamped by launch.py; a directory "
            f"without one was not produced by the Stage 0 family.")

    manifest = json.loads(path.read_text())
    assert_same_protocol(manifest, source=str(path))
    return manifest


def attach_fig2_psth_r2(results):
    """Fill each session's `psth_r2` from the Figure 2 aligned cache.

    Sourced rather than recomputed on purpose: `protocol.MIN_PSTH_R2` exists to
    select the neurons Figure 2 reports on, so reading Figure 2's own
    split-half estimate guarantees the same cells rather than a near-copy of
    its estimator. Cells absent from that cache stay NaN and fail inclusion.
    """
    covdecomp = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
    if covdecomp not in sys.path:
        sys.path.insert(0, covdecomp)
    from data_loading import load_cache as load_aligned_cache

    aligned = {a["session"]: a for a in load_aligned_cache()}
    out = []
    for entry in results:
        entry = dict(entry)
        psth_r2 = np.full(len(entry["total_spikes"]), np.nan)
        a = aligned.get(entry["session"])
        if a is not None:
            for cid, value in zip(np.asarray(a["neuron_mask"]),
                                  np.asarray(a["psth_r2"], dtype=float)):
                if 0 <= int(cid) < psth_r2.size:
                    psth_r2[int(cid)] = value
        entry["psth_r2"] = psth_r2
        out.append(entry)
    return out


def best_checkpoint(run_dir):
    """The highest-val-BPS checkpoint a run saved.

    `ModelCheckpoint` keeps `save_top_k=3` plus `last.ckpt`; selection is on
    validation BPS (`protocol.SELECTION_METRIC` / `SELECTION_SPLIT`), so the
    best of those three is the run's selected model, and `last.ckpt` is not.
    """
    candidates = [p for p in Path(run_dir).glob("*.ckpt") if p.name != "last.ckpt"]
    if not candidates:
        raise FileNotFoundError(f"No selectable checkpoint in {run_dir}")

    def score(path):
        tail = path.stem.split("val_bps_overall=")[-1]
        try:
            return float(tail)
        except ValueError:
            return float("-inf")

    return max(candidates, key=score)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate_run(run_dir, gpu=0, checkpoint=None, session_filter=None,
                 skip_test_split=False, skip_fixrsvp=False, max_datasets=30,
                 out=None):
    """Score one run and write `evaluation.json` beside its manifest."""
    import torch
    from eval.load_twin import load_twin

    run_dir = Path(run_dir)
    manifest = load_run_manifest(run_dir)
    checkpoint = Path(checkpoint) if checkpoint else best_checkpoint(run_dir)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    print(f"[{manifest['run']}] protocol {PROTOCOL_HASH}")
    print(f"  checkpoint {checkpoint.name}")
    model, info = load_twin(checkpoint, device=device)
    model.eval()

    report = {
        "run": manifest["run"],
        "protocol_hash": PROTOCOL_HASH,
        "checkpoint": str(checkpoint),
        "cids_source": info["cids_source"],
        "n_units": info["n_units"],
    }

    if not skip_test_split:
        from training.pl_modules import MultiDatasetDM
        spec = manifest["spec"]
        dm = MultiDatasetDM(
            cfg_dir=str(HERE / "configs" / spec["config"]),
            max_ds=max_datasets,
            batch=spec["batch_size"],
            workers=8,
            steps_per_epoch=1,
            dset_dtype="bfloat16",
            homogeneous_batches=spec["homogeneous"],
        )
        dm.setup("test")
        per_unit = score_test_split(model, dm, device)
        overall, per_ds = overall_bps(per_unit)
        report["test_split"] = {"bps_overall": overall, "bps_by_session": per_ds}
        print(f"  test-split BPS {overall:.4f} over {len(per_ds)} sessions")

    if not skip_fixrsvp:
        scored = attach_fig2_psth_r2(
            score_fixrsvp(model, device, session_filter=session_filter))
        included = apply_protocol_inclusion(scored)
        summary = {}
        for metric in ("ccnorm", "single_trial_r2", "bps"):
            pooled = np.concatenate(
                [s[metric][s["include"]] for s in included]) if included else np.array([])
            summary[metric] = {
                "median": nanmedian_or_none(pooled),
                "n": int(np.isfinite(pooled).sum()),
            }
        report["fixrsvp"] = {
            "n_sessions": len(included),
            "n_units": int(sum(int(s["include"].sum()) for s in included)),
            "metrics": summary,
            "by_session": {
                s["session"]: {
                    "n_units": int(s["include"].sum()),
                    **{m: nanmedian_or_none(s[m][s["include"]])
                       for m in ("ccnorm", "single_trial_r2", "bps")},
                }
                for s in included
            },
        }
        print(f"  fixrsvp: {report['fixrsvp']['n_sessions']} sessions, "
              f"{report['fixrsvp']['n_units']} units; "
              f"median CC_norm {summary['ccnorm']['median']}, "
              f"median single-trial r2 {summary['single_trial_r2']['median']}")

    path = Path(out) if out else run_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  wrote {path}")
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="Run id (e.g. E1a) or an explicit run directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--checkpoint", default=None,
                    help="Override the selected checkpoint")
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--skip-test-split", action="store_true")
    ap.add_argument("--skip-fixrsvp", action="store_true")
    # A full pass loads all 30 sessions (~19 min) before scoring anything.
    # Capping it makes the code path exercisable in minutes; a capped run is a
    # shakedown, never a reported number.
    ap.add_argument("--max-datasets", type=int, default=30,
                    help="Cap sessions loaded for the test-split pass")
    ap.add_argument("--out", default=None,
                    help="Write the report here instead of run_dir/evaluation.json")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        from launch import CKPT_ROOT
        run_dir = CKPT_ROOT / args.run

    evaluate_run(run_dir, gpu=args.gpu, checkpoint=args.checkpoint,
                 session_filter=args.sessions,
                 skip_test_split=args.skip_test_split,
                 skip_fixrsvp=args.skip_fixrsvp,
                 max_datasets=args.max_datasets,
                 out=args.out)


if __name__ == "__main__":
    main()
