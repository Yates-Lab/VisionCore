"""Digital-twin inference on FIG2's exact frame (fixation < 0.5 deg), 3 conditions.

The fig3 twin cache (``fig3_digitaltwin.pkl``) was built at fixation < 1.0 deg,
so its covariance structure does not match fig2's aligned cache (fixation < 0.5).
This module re-runs the SAME twin checkpoint with fig2's alignment
(``align_fixrsvp_trials`` semantics: fixation_radius=0.5, min_fix_dur=20,
valid_time_bins=120, min_total_spikes=0, origin center) so the model rates land
on fig2's exact trials/bins/cells. Each bin's rate depends only on its own
lag-history (the convGRU runs over lags, not across the trial), so restricting to
the < 0.5 deg bins does not change the retained rates -- this is a re-alignment,
not a different prediction.

Unlike the earlier single-condition build, every trial is forwarded under the
three fig3 within-model conditions so the supplement can show how each ablation
reproduces fig2:

  - intact     : full retinal stimulus + full behavior input (the "full" twin).
  - zeroed     : behavior set to 0 (extraretinal route removed; "ablated").
  - stabilized : retinal image frozen at ONE session-global centroid gaze
                 (reafferent route removed), behavior intact.

The behavior-zeroing and pixel-exact stabilized-stim rendering are imported
verbatim from ``_fig3_ablation_data`` so the conditions are defined identically
to fig3; only the analysis frame (fixation < 0.5 deg, min_total_spikes=0) is
fig2's.

Output ``supp_twin_fig2frame_conditions.pkl``: list of per-session dicts with the
keys _supp_data.build_records reads (session, subject, neuron_mask, robs_used,
rhat_used {cond: array}, eyepos_used, valid_mask, dfs_used, n_neurons).
Affine-rescales each condition's rhat to the observed counts (as the fig3 cache
does) so Poisson(rhat) has the right scale.

Usage:
    FIG3_GPU=0 uv run python paper/supp_model_replication/_supp_inference.py [--force]

``FIG3_GPU`` is optional; it pins the physical GPU when another production
analysis is running concurrently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

# fig3 twin config (checkpoint, dataset configs) + fig2 fixation constant
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig3"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "covariance_decomposition"))
from _fig3_data import (  # noqa: E402
    CHECKPOINT_PATH, VALID_TIME_BINS, MIN_FIX_DUR, subject_from_session, SUBJECTS,
    analysis_endpoint_mask_and_psth, analysis_endpoint_block_mean,
    align_native_trial_arrays_to_reference,
)
from _fig3_ablation_data import (  # noqa: E402
    CONDS, STIM_CONDS, build_behavior_modifiers, build_stabilized_stim,
)
from data_loading import (  # noqa: E402  (0.5, fig2's exact frame)
    FIXATION_RADIUS,
    load_cache as load_aligned_cache,
)

# Conditions cache (intact/zeroed/stabilized). The legacy single-condition cache
# ``supp_twin_fig2frame.pkl`` (intact only) is kept as a fallback in _supp_data.
SUPP_INFERENCE_CONDITIONS_CACHE = CACHE_DIR / "supp_twin_fig2frame_conditions.pkl"
SUPP_INFERENCE_SCHEMA = 2     # native-240 endpoints aligned to exact Figure-2 frame
MIN_TOTAL_SPIKES = 0          # match fig2's align_fixrsvp_trials (all units)
MIN_GOOD_TRIALS = 10


def _positive_affine_fallback(torch, robs, rhat, dfs, eps=1e-8):
    """Stable nonnegative affine fit for a single degenerate rate column.

    The production calibrator optimizes Poisson likelihood with LBFGS.  A unit
    with no spikes, no valid samples, or an effectively constant prediction can
    make that joint optimization non-finite even though the other units are
    perfectly well behaved.  These units contribute no useful covariance
    signal, so use a finite constrained least-squares calibration for them.
    """
    valid = (dfs > 0.5) & torch.isfinite(robs) & torch.isfinite(rhat)
    x = torch.where(valid, torch.clamp(rhat, min=0), torch.zeros_like(rhat))
    y = torch.where(valid, torch.clamp(robs, min=0), torch.zeros_like(robs))
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return torch.full_like(rhat, eps)

    xv = x[valid]
    yv = y[valid]
    x_mean = xv.mean()
    y_mean = yv.mean()
    x_centered = xv - x_mean
    denom = torch.sum(x_centered.square())
    if float(denom) > eps:
        gain = torch.clamp(
            torch.sum(x_centered * (yv - y_mean)) / denom, min=0
        )
    else:
        gain = torch.zeros((), dtype=rhat.dtype, device=rhat.device)
    offset = torch.clamp(y_mean - gain * x_mean, min=eps)
    return torch.clamp(gain * x + offset, min=eps)


def _rescale_affine_safely(torch, rescale_rhat, robs, rhat, dfs):
    """Preserve production calibration while isolating degenerate neurons.

    First attempt the exact vectorized Figure-3 rescaling.  If one neuron makes
    LBFGS fail, bisect the columns so every regular neuron still receives that
    exact calibration; only irreducible one-column failures use the stable
    positive-affine fallback above.
    """
    if not (robs.shape == rhat.shape == dfs.shape) or robs.ndim != 2:
        raise ValueError("robs, rhat, and dfs must have the same T x N shape")

    output = torch.empty_like(rhat)
    fallback_columns = []

    def fit_columns(columns):
        if columns.numel() == 0:
            return
        valid_count = (dfs[:, columns] > 0.5).sum(dim=0)
        if columns.numel() == 1 and int(valid_count[0].item()) == 0:
            j = int(columns[0].item())
            output[:, j] = _positive_affine_fallback(
                torch, robs[:, j], rhat[:, j], dfs[:, j]
            )
            fallback_columns.append(j)
            return
        try:
            rr, _ = rescale_rhat(
                robs[:, columns], rhat[:, columns], dfs[:, columns], mode="affine"
            )
            if not torch.isfinite(rr).all():
                raise FloatingPointError("calibrated rates contain non-finite values")
            output[:, columns] = rr
        except (FloatingPointError, RuntimeError, ValueError):
            if columns.numel() == 1:
                j = int(columns[0].item())
                output[:, j] = _positive_affine_fallback(
                    torch, robs[:, j], rhat[:, j], dfs[:, j]
                )
                fallback_columns.append(j)
                return
            midpoint = columns.numel() // 2
            fit_columns(columns[:midpoint])
            fit_columns(columns[midpoint:])

    fit_columns(torch.arange(rhat.shape[1], device=rhat.device))
    return output, fallback_columns


def run_inference(
    force=False,
    session_filter=None,
    cache_path=SUPP_INFERENCE_CONDITIONS_CACHE,
):
    """Run Figure-2-frame inference, optionally as an off-cache session smoke."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force and session_filter is None:
        print(f"Loading supp inference cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = dill.load(f)
        cached_checkpoints = {
            str(row.get("checkpoint_path"))
            for row in cached
            if isinstance(row, dict) and row.get("checkpoint_path")
        }
        cached_schemas = {
            row.get("alignment_schema")
            for row in cached
            if isinstance(row, dict)
        }
        if cached_schemas != {SUPP_INFERENCE_SCHEMA}:
            raise ValueError(
                f"{cache_path} has alignment schema "
                f"{sorted(cached_schemas, key=lambda value: str(value))}, but "
                f"schema {SUPP_INFERENCE_SCHEMA} is required. Re-run with --force."
            )
        if os.environ.get("FIG3_TWIN_CHECKPOINT") and cached_checkpoints != {
            str(CHECKPOINT_PATH)
        }:
            raise ValueError(
                f"{SUPP_INFERENCE_CONDITIONS_CACHE} has checkpoint provenance "
                f"{sorted(cached_checkpoints) or ['missing']}, but the selected "
                f"checkpoint is {CHECKPOINT_PATH}. Re-run with --force."
            )
        return cached

    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import load_single_dataset, run_model, rescale_rhat

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    device = get_free_device(os.environ.get("FIG3_GPU"))
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
    print(f"  fixation_radius={FIXATION_RADIUS} (fig2 frame), "
          f"min_total_spikes={MIN_TOTAL_SPIKES}, conditions={CONDS}")
    aligned_reference = {
        row["session"]: row for row in load_aligned_cache()
    }

    session_results = []
    for dataset_idx in range(len(model.names)):
        session_name = model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            continue
        if session_filter is not None and session_name not in session_filter:
            continue
        print(f"\n--- {session_name} ({subject}) "
              f"[{dataset_idx + 1}/{len(model.names)}] ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
        try:
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp'),
            ], dim=0)
        except (ValueError, KeyError):
            print("  Skipping: no fixrsvp data")
            continue

        dset_idx_local = fixrsvp_inds[:, 0].unique().item()
        dset = train_data.dsets[dset_idx_local]

        trial_inds = np.asarray(dset.covariates['trial_inds']).ravel()
        psth_inds_flat = np.asarray(dset.covariates['psth_inds']).ravel()
        analysis_endpoints, psth_inds_analysis = analysis_endpoint_mask_and_psth(
            dataset_config, psth_inds_flat, trial_inds
        )
        robs_flat = np.asarray(dset['robs'])
        eyepos_flat = np.asarray(dset['eyepos'])
        eyepos_analysis = analysis_endpoint_block_mean(
            dataset_config, eyepos_flat, analysis_endpoints
        )

        trials = np.unique(trial_inds)
        NT = len(trials)
        NC = robs_flat.shape[1]
        T = int(psth_inds_analysis[analysis_endpoints].max()) + 1

        # fig2 frame: fixation < 0.5 deg from the origin.
        fixation = np.hypot(
            eyepos_analysis[:, 0], eyepos_analysis[:, 1]
        ) < FIXATION_RADIUS
        stim_lags = np.array(dataset_config['keys_lags']['stim'])

        beh_mod = build_behavior_modifiers()

        # Reafferent-ablation stim: retinal image frozen at one session-global
        # centroid gaze, rendered from raw data and decimated to the model frame.
        # The alignment gate must pass (== 0) or the substitution is not frame-aligned.
        samp = dataset_config.get('sampling', {})
        factor = (int(samp['source_rate']) // int(samp['target_rate'])) if samp else 1
        stab_stim_np, align_maxabs, n_tr_stab = build_stabilized_stim(
            session_name, dset['stim'].numpy(), factor)
        print(f"  stabilized render: {n_tr_stab} trials, factor={factor}, "
              f"alignment max-abs(decimate(raw)-embedded)={align_maxabs}")
        if align_maxabs != 0:
            print(f"  Skipping: stabilized-stim alignment gate failed "
                  f"(max-abs={align_maxabs})")
            continue
        stab_stim = torch.from_numpy(stab_stim_np)

        robs = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        eyepos = np.full((NT, T, 2), np.nan)
        fix_dur = np.full(NT, np.nan)
        rhat = {c: np.full((NT, T, NC), np.nan) for c in CONDS}

        for itrial in tqdm(range(NT), desc=f"  Inference {session_name}"):
            ix_obs = (trial_inds == trials[itrial]) & fixation & analysis_endpoints
            if not np.any(ix_obs):
                continue
            t_obs = psth_inds_analysis[ix_obs].astype(int)
            fix_dur[itrial] = len(t_obs)
            robs[itrial, t_obs] = robs_flat[ix_obs]
            dfs[itrial, t_obs] = np.asarray(dset['dfs'][ix_obs])
            eyepos[itrial, t_obs] = eyepos_analysis[ix_obs]

            ix = ix_obs.copy()
            ix[: int(stim_lags.max(initial=0))] = False
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            stim_stab = stab_stim[stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior0 = dset['behavior'][ix]
            output_behavior0 = (
                dset['output_behavior'][ix]
                if 'output_behavior' in dset
                else None
            )
            t_inds = psth_inds_analysis[ix].astype(int)
            for c in CONDS:
                if c in STIM_CONDS:            # replace stim, keep behavior intact
                    batch = {'stim': stim_stab, 'behavior': behavior0}
                    if output_behavior0 is not None:
                        batch['output_behavior'] = output_behavior0
                else:                          # keep stored stim, modify behavior
                    behavior = (behavior0 if beh_mod[c] is None
                                else beh_mod[c](behavior0, itrial))
                    batch = {'stim': stim, 'behavior': behavior}
                    if output_behavior0 is not None:
                        output_behavior = (
                            output_behavior0
                            if beh_mod[c] is None
                            else beh_mod[c](output_behavior0, itrial)
                        )
                        batch['output_behavior'] = output_behavior
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    out = run_model(model, batch, dataset_idx=dataset_idx)
                rhat[c][itrial, t_inds] = out['rhat'].detach().cpu().numpy()
            if itrial % 16 == 0:
                torch.cuda.empty_cache()

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < MIN_GOOD_TRIALS:
            print(f"  Skipping: only {int(good_trials.sum())} good trials")
            continue

        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[good_trials][:, iix]
        dfs = dfs[good_trials][:, iix]
        eyepos = eyepos[good_trials][:, iix]
        rhat = {c: r[good_trials][:, iix] for c, r in rhat.items()}

        if dataset_config.get("supervision"):
            if session_name not in aligned_reference:
                print(f"  Skipping: {session_name} is absent from Figure-2 frame")
                continue
            reference = aligned_reference[session_name]
            robs_used, rhat_used, dfs_used, neuron_mask = (
                align_native_trial_arrays_to_reference(
                    robs,
                    rhat,
                    reference,
                    robs_key="robs",
                    dfs_key=None,
                    label=f"{session_name} Figure-2 frame",
                )
            )
            eyepos = np.asarray(reference["eyepos"]).copy()
            valid_mask = np.asarray(reference["valid_mask"], dtype=bool).copy()
        else:
            neuron_mask = np.where(
                np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES
            )[0]
            if len(neuron_mask) < 3:
                print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
                continue
            robs_used = robs[:, :, neuron_mask]
            dfs_used = dfs[:, :, neuron_mask]
            rhat_used = {c: r[:, :, neuron_mask] for c, r in rhat.items()}
            valid_mask = np.isfinite(eyepos).all(axis=-1)
        n_trials, n_time, n_neurons = robs_used.shape

        # Affine-rescale each condition's model rates to observed counts (as the
        # fig3 cache does), so Poisson(rhat) has the right scale per condition.
        robs_flat_used = robs_used.reshape(n_trials * n_time, n_neurons)
        dfs_flat = dfs_used.reshape(n_trials * n_time, n_neurons)
        for c in CONDS:
            rhat_flat = rhat_used[c].reshape(n_trials * n_time, n_neurons)
            rr, fallback_columns = _rescale_affine_safely(
                torch, rescale_rhat,
                torch.from_numpy(robs_flat_used), torch.from_numpy(rhat_flat),
                torch.from_numpy(dfs_flat),
            )
            if fallback_columns:
                print(f"  {c}: stable calibration fallback for "
                      f"{len(fallback_columns)}/{n_neurons} degenerate neurons")
            rhat_used[c] = rr.reshape(n_trials, n_time, n_neurons).cpu().numpy()

        print(f"  {n_trials} trials, {n_time} bins, {n_neurons} neurons "
              f"(good_trials={int(good_trials.sum())})")
        session_results.append({
            "session": session_name,
            "subject": subject,
            "alignment_schema": SUPP_INFERENCE_SCHEMA,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "neuron_mask": neuron_mask,
            "n_neurons": n_neurons,
            "robs_used": robs_used,
            "rhat_used": rhat_used,          # {cond: (trials, bins, neurons)}
            "dfs_used": dfs_used,
            "eyepos_used": eyepos,
            "valid_mask": valid_mask,
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to "
          f"{cache_path}")
    return session_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Twin inference on fig2's frame (0.5 deg), 3 conditions.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_inference(force=args.force)
