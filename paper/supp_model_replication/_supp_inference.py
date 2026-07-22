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
    uv run python paper/supp_model_replication/_supp_inference.py [--force]
"""
from __future__ import annotations

import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

# fig3 twin config (checkpoint, dataset configs) + fig2 fixation constant
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig3"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "covariance_decomposition"))
from _fig3_data import (  # noqa: E402
    CHECKPOINT_PATH, VALID_TIME_BINS, MIN_FIX_DUR, subject_from_session, SUBJECTS,
)
from _fig3_ablation_data import (  # noqa: E402
    CONDS, STIM_CONDS, build_behavior_modifiers, build_stabilized_stim,
)
from data_loading import FIXATION_RADIUS  # noqa: E402  (0.5, fig2's value)

# Conditions cache (intact/zeroed/stabilized). The legacy single-condition cache
# ``supp_twin_fig2frame.pkl`` (intact only) is kept as a fallback in _supp_data.
SUPP_INFERENCE_CONDITIONS_CACHE = CACHE_DIR / "supp_twin_fig2frame_conditions.pkl"
MIN_TOTAL_SPIKES = 0          # match fig2's align_fixrsvp_trials (all units)
MIN_GOOD_TRIALS = 10


def run_inference(force=False):
    if SUPP_INFERENCE_CONDITIONS_CACHE.exists() and not force:
        print(f"Loading supp inference cache from {SUPP_INFERENCE_CONDITIONS_CACHE}")
        with open(SUPP_INFERENCE_CONDITIONS_CACHE, "rb") as f:
            return dill.load(f)

    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import load_single_dataset, run_model, rescale_rhat

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    device = get_free_device()
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
    print(f"  fixation_radius={FIXATION_RADIUS} (fig2 frame), "
          f"min_total_spikes={MIN_TOTAL_SPIKES}, conditions={CONDS}")

    session_results = []
    for dataset_idx in range(len(model.names)):
        session_name = model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
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
        robs_flat = np.asarray(dset['robs'])
        eyepos_flat = np.asarray(dset['eyepos'])

        trials = np.unique(trial_inds)
        NT = len(trials)
        NC = robs_flat.shape[1]
        T = int(psth_inds_flat.max()) + 1

        # fig2 frame: fixation < 0.5 deg from the origin.
        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < FIXATION_RADIUS
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
            ix = (trial_inds == trials[itrial]) & fixation
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            stim_stab = stab_stim[stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior0 = dset['behavior'][ix]
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
            eyepos[itrial, t_inds] = eyepos_flat[ix]
            for c in CONDS:
                if c in STIM_CONDS:            # replace stim, keep behavior intact
                    batch = {'stim': stim_stab, 'behavior': behavior0}
                else:                          # keep stored stim, modify behavior
                    behavior = (behavior0 if beh_mod[c] is None
                                else beh_mod[c](behavior0, itrial))
                    batch = {'stim': stim, 'behavior': behavior}
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

        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
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
            rr, _ = rescale_rhat(
                torch.from_numpy(robs_flat_used), torch.from_numpy(rhat_flat),
                torch.from_numpy(dfs_flat), mode='affine',
            )
            rhat_used[c] = rr.reshape(n_trials, n_time, n_neurons).cpu().numpy()

        print(f"  {n_trials} trials, {n_time} bins, {n_neurons} neurons "
              f"(good_trials={int(good_trials.sum())})")
        session_results.append({
            "session": session_name,
            "subject": subject,
            "neuron_mask": neuron_mask,
            "n_neurons": n_neurons,
            "robs_used": robs_used,
            "rhat_used": rhat_used,          # {cond: (trials, bins, neurons)}
            "dfs_used": dfs_used,
            "eyepos_used": eyepos,
            "valid_mask": valid_mask,
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUPP_INFERENCE_CONDITIONS_CACHE, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to "
          f"{SUPP_INFERENCE_CONDITIONS_CACHE}")
    return session_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Twin inference on fig2's frame (0.5 deg), 3 conditions.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_inference(force=args.force)
