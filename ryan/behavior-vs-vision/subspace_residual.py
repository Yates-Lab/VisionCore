# %% Imports and configuration
"""
Residual subspace analysis: behavior vs vision-only digital twin models on fixRSVP.

For each session:
  1. Run both models on fixation trials → (NT, T, NC) predicted-rate tensors.
  2. Affine-rescale each model's rates per neuron (Poisson MLE) so totals match
     observed spike counts. Same rescaling protocol as compare_models_fixrsvp.py
     and Fig 3 — necessary for a fair, in-spike-units comparison.
  3. Form Sigma_Delta = Cov(rhat_beh - rhat_vis), pooled across (trials × time)
     in spike-count units.
  4. Pull Sigma_PSTH, Sigma_FEM, Sigma_int from fig2_decomposition.pkl.
     Restrict to the intersection of fig2's neuron_mask and the session's
     fixrsvp neuron mask.
  5. PSD-project all four covariances. Compute:
       - Participation ratio of Sigma_Delta (effective dimensionality)
       - Symmetric subspace overlap (top-k) of U_Delta vs U_PSTH/FEM/int
       - Directional variance capture: fraction of Sigma_Delta variance
         explained by each subspace, and vice versa.
  6. Trial-shuffled control: shuffle behavior input across trials, recompute
     rhat_beh, and recompute everything. Establishes a null on alignment.
  7. Per-session figure + cross-session summary.

Pilot sessions are configurable; default runs the two with the largest
median ΔBPS in `behavior_vs_vision_fixrsvp.pkl`.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill
import torch
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR
from VisionCore.subspace import (
    project_to_psd,
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
BEHAVIOR_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)
VISION_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-04-13_11-10-15_learned_resnet_none_convgru_gaussian/"
    "learned_resnet_none_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)

FIG2_DECOMP_PATH = CACHE_DIR / "fig2_decomposition.pkl"
FIXRSVP_METRICS_PATH = CACHE_DIR / "behavior_vs_vision_fixrsvp.pkl"

FIG_DIR = FIGURES_DIR / "behavior-vs-vision"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "behavior_vs_vision_residual_subspace.pkl"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
RECOMPUTE = True            # rerun inference even if cache exists
DT = 1 / 120
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 200      # must match compare_models_fixrsvp.py
MIN_VAR = 1e-8
SUBSPACE_K = 5              # top-k for symmetric overlap
SUBSPACE_K_SWEEP = list(range(1, 11))  # k values for cumulative-capture curves
N_SHUFFLES = 10             # behavior trial-shuffles per session for null
FIG2_WINDOW_IDX = 0         # 8.33 ms (smallest window) — same as fixrsvp dt

PILOT_SESSIONS = ["Allen_2022-02-16", "Logan_2020-03-04"]
RUN_ALL = True              # if True, run every session in fig2_decomposition

SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}

try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def find_best_ckpt(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("epoch=*-val_bps_overall=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No val_bps_overall checkpoints in {ckpt_dir}")
    return max(ckpts, key=lambda p: float(p.stem.split("val_bps_overall=")[1]))


# %% Load both models
from DataYatesV1 import get_free_device

DEVICE = get_free_device()
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import load_single_dataset, run_model, rescale_rhat

beh_ckpt = find_best_ckpt(BEHAVIOR_DIR)
vis_ckpt = find_best_ckpt(VISION_DIR)
print(f"Behavior ckpt:    {beh_ckpt.name}")
print(f"Vision-only ckpt: {vis_ckpt.name}")

beh_model, _ = load_model(checkpoint_path=str(beh_ckpt), device=str(DEVICE))
beh_model.model.eval()
beh_model.model.convnet.use_checkpointing = False

vis_model, _ = load_model(checkpoint_path=str(vis_ckpt), device=str(DEVICE))
vis_model.model.eval()
vis_model.model.convnet.use_checkpointing = False

assert list(beh_model.names) == list(vis_model.names), (
    "Behavior and vision models have different datasets — cannot align cells."
)
session_to_idx = {s: i for i, s in enumerate(beh_model.names)}


# %% Helpers ------------------------------------------------------------------

def _gather_fixrsvp_trials(beh_model, dataset_idx):
    """Return (robs, rhat_beh, rhat_vis_or_None, dfs, dataset_config) tensors.
    rhat_vis is filled in by `run_inference_for_session`. This split lets us
    reuse the gather step for shuffle controls without re-fetching trials.
    """
    train_data, val_data, dataset_config = load_single_dataset(beh_model, dataset_idx)
    fixrsvp_inds = torch.cat([
        train_data.get_dataset_inds('fixrsvp'),
        val_data.get_dataset_inds('fixrsvp'),
    ], dim=0)
    dset_idx_local = fixrsvp_inds[:, 0].unique().item()
    dset = train_data.dsets[dset_idx_local]

    trial_inds = np.asarray(dset.covariates['trial_inds']).ravel()
    psth_inds_flat = np.asarray(dset.covariates['psth_inds']).ravel()
    eyepos_flat = np.asarray(dset['eyepos'])
    fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

    return dset, trial_inds, psth_inds_flat, fixation, dataset_config


def _run_inference(
    models, dataset_idx, dset, trial_inds, psth_inds_flat, fixation,
    dataset_config, behavior_shuffle=None,
):
    """Run an arbitrary set of models on the same fixrsvp trials.

    `models` is a dict {key: model}. Returns
        robs (NT,T,NC), {key: rhat (NT,T,NC)}, dfs (NT,T,NC), fix_dur (NT,)

    behavior_shuffle: None for normal eval; otherwise a permutation of trial
    indices used to scramble the behavior tensor across trials before feeding
    it to the models. Stimulus, robs, dfs are never shuffled.
    """
    trials = np.unique(trial_inds)
    NT = len(trials)
    NC = np.asarray(dset['robs']).shape[1]
    T = int(psth_inds_flat.max()) + 1

    robs = np.full((NT, T, NC), np.nan)
    rhats = {k: np.full((NT, T, NC), np.nan) for k in models}
    dfs = np.full((NT, T, NC), np.nan)
    fix_dur = np.full(NT, np.nan)

    stim_lags = np.array(dataset_config['keys_lags']['stim'])
    robs_flat_arr = np.asarray(dset['robs'])

    if behavior_shuffle is not None:
        assert len(behavior_shuffle) == NT

    desc = "  Inference" if behavior_shuffle is None else "  Inference (shuf)"
    for itrial in tqdm(range(NT), desc=desc, leave=False):
        ix = (trial_inds == trials[itrial]) & fixation
        if not np.any(ix):
            continue

        stim_indices = np.where(ix)[0]
        stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
        stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
        behavior = dset['behavior'][ix]

        if behavior_shuffle is not None:
            src_trial = trials[behavior_shuffle[itrial]]
            ix_src = (trial_inds == src_trial) & fixation
            n_src = int(ix_src.sum())
            n_dst = int(ix.sum())
            n_use = min(n_src, n_dst)
            if n_use == 0:
                continue
            beh_src = dset['behavior'][ix_src][:n_use]
            behavior = behavior.clone()
            behavior[:n_use] = beh_src
            if n_use < n_dst:
                behavior[n_use:] = beh_src.mean(dim=0, keepdim=True)

        batch = {'stim': stim, 'behavior': behavior}
        t_inds = psth_inds_flat[ix].astype(int)
        fix_dur[itrial] = len(t_inds)
        robs[itrial, t_inds] = robs_flat_arr[ix]
        dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
        for k, m in models.items():
            out = run_model(m, batch, dataset_idx=dataset_idx)
            rhats[k][itrial, t_inds] = out['rhat'].detach().cpu().numpy()

    return robs, rhats, dfs, fix_dur


def _trim(robs, rhats, dfs, fix_dur, T_keep=VALID_TIME_BINS):
    good = fix_dur > MIN_FIX_DUR
    iix = np.arange(min(T_keep, robs.shape[1]))
    robs_t = robs[good][:, iix]
    rhats_t = {k: v[good][:, iix] for k, v in rhats.items()}
    dfs_t = dfs[good][:, iix]
    return robs_t, rhats_t, dfs_t


def _affine_rescale(robs, rhat, dfs):
    """Per-neuron affine rescale (gain+offset) of rhat to match robs."""
    NT, T, NC = robs.shape
    rhat_rescaled, _ = rescale_rhat(
        torch.from_numpy(robs.reshape(-1, NC)),
        torch.from_numpy(rhat.reshape(-1, NC)),
        torch.from_numpy(dfs.reshape(-1, NC)),
        mode="affine",
    )
    return rhat_rescaled.reshape(NT, T, NC).detach().cpu().numpy()


def _residual_cov(rhat_beh, rhat_vis, dfs):
    """Cov of (rhat_beh - rhat_vis) pooled across (trials × time, neurons).

    Uses bin-level validity (dfs > 0 across all neurons in that bin), which
    is the right bookkeeping when dfs is a per-bin valid flag. Predicted
    rates from the model are well-defined wherever stimulus is defined, so
    rhat is finite anywhere robs/dfs are valid.
    """
    delta = rhat_beh - rhat_vis
    bin_valid = (np.nan_to_num(dfs, nan=0.0) > 0).all(axis=-1) \
        & np.isfinite(delta).all(axis=-1)
    delta_flat = delta[bin_valid]
    if delta_flat.shape[0] < 10:
        return None, 0
    delta_flat = delta_flat - delta_flat.mean(axis=0, keepdims=True)
    n = delta_flat.shape[0]
    Sigma = delta_flat.T @ delta_flat / max(n - 1, 1)
    return Sigma, n


def _alignment_metrics(Sigma_target, Sigma_ref_list, ref_names, k=SUBSPACE_K):
    """Compute participation ratio + alignment between Sigma_target and each
    Sigma_ref. Returns a flat dict, including k-sweep curves for both
    overlap and directional variance capture.
    """
    out = {}
    Sigma_target_psd = project_to_psd(Sigma_target)
    out["pr_target"] = participation_ratio(Sigma_target_psd)
    w_t, V_t = np.linalg.eigh(Sigma_target_psd)
    V_t = V_t[:, ::-1]
    out["spectrum_target"] = w_t[::-1]
    n_dim = V_t.shape[1]

    out["pr_refs"] = {}
    out["overlap_k1"] = {}
    out["overlap_k"] = {}
    out["capture_target_in_ref"] = {}
    out["capture_ref_in_target"] = {}
    out["overlap_curve"] = {}             # name -> array len(K_SWEEP)
    out["capture_target_in_ref_curve"] = {}
    out["capture_ref_in_target_curve"] = {}
    out["k_sweep"] = SUBSPACE_K_SWEEP

    for Sigma_ref, name in zip(Sigma_ref_list, ref_names):
        Sigma_ref_psd = project_to_psd(Sigma_ref)
        out["pr_refs"][name] = participation_ratio(Sigma_ref_psd)
        w_r, V_r = np.linalg.eigh(Sigma_ref_psd)
        V_r = V_r[:, ::-1]

        kk = min(k, n_dim - 1, V_r.shape[1] - 1)
        out["overlap_k1"][name] = symmetric_subspace_overlap(V_t[:, :1], V_r[:, :1])
        out["overlap_k"][name] = symmetric_subspace_overlap(V_t[:, :kk], V_r[:, :kk])
        out["capture_target_in_ref"][name] = directional_variance_capture(
            Sigma_target_psd, V_r[:, :kk]
        )
        out["capture_ref_in_target"][name] = directional_variance_capture(
            Sigma_ref_psd, V_t[:, :kk]
        )

        ov_curve = []
        cap_t_in_r = []
        cap_r_in_t = []
        for kk_sweep in SUBSPACE_K_SWEEP:
            kkk = min(kk_sweep, n_dim - 1, V_r.shape[1] - 1)
            ov_curve.append(symmetric_subspace_overlap(
                V_t[:, :kkk], V_r[:, :kkk]
            ))
            cap_t_in_r.append(directional_variance_capture(
                Sigma_target_psd, V_r[:, :kkk]
            ))
            cap_r_in_t.append(directional_variance_capture(
                Sigma_ref_psd, V_t[:, :kkk]
            ))
        out["overlap_curve"][name] = np.array(ov_curve)
        out["capture_target_in_ref_curve"][name] = np.array(cap_t_in_r)
        out["capture_ref_in_target_curve"][name] = np.array(cap_r_in_t)
    return out


def _intersect_neuron_masks(fig2_mask, fixrsvp_mask):
    """Indices into both masks for the intersection.
    Returns (idx_into_fig2, idx_into_fixrsvp) arrays of equal length, sorted
    by the underlying global neuron index.
    """
    fig2_mask = np.asarray(fig2_mask)
    fixrsvp_mask = np.asarray(fixrsvp_mask)
    common = np.intersect1d(fig2_mask, fixrsvp_mask)
    idx_fig2 = np.array([np.where(fig2_mask == g)[0][0] for g in common])
    idx_fix = np.array([np.where(fixrsvp_mask == g)[0][0] for g in common])
    return common, idx_fig2, idx_fix


# %% Load fig2 decomposition + fixrsvp metrics caches
print("Loading fig2 decomposition...")
with open(FIG2_DECOMP_PATH, "rb") as f:
    fig2_sessions = dill.load(f)
fig2_by_name = {s["session"]: s for s in fig2_sessions}
print(f"  {len(fig2_by_name)} sessions in fig2 cache")

print("Loading fixrsvp metrics cache...")
with open(FIXRSVP_METRICS_PATH, "rb") as f:
    fixrsvp_sessions = dill.load(f)
fixrsvp_by_name = {s["session"]: s for s in fixrsvp_sessions}


# %% Pick session list
if RUN_ALL:
    candidates = list(fig2_by_name.keys() & fixrsvp_by_name.keys())
else:
    candidates = [s for s in PILOT_SESSIONS
                  if s in fig2_by_name and s in fixrsvp_by_name]
print(f"Will process {len(candidates)} sessions: {candidates}")


# %% Run the analysis
def process_session(name):
    print(f"\n--- {name} ---")
    fig2 = fig2_by_name[name]
    fixrsvp = fixrsvp_by_name[name]
    dataset_idx = session_to_idx[name]

    # Pull fig2 covariances at chosen window
    mats = fig2["mats"][FIG2_WINDOW_IDX]
    Cpsth_full = mats["PSTH"]
    Crate_full = mats["Intercept"]
    Ctotal_full = mats["Total"]
    Cfem_full = Crate_full - Cpsth_full
    Cint_full = Ctotal_full - Crate_full
    fig2_mask = np.asarray(fig2["neuron_mask"])

    # Gather fixrsvp trials & run both models
    dset, trial_inds, psth_inds_flat, fixation, dataset_config = (
        _gather_fixrsvp_trials(beh_model, dataset_idx)
    )
    models = {"beh": beh_model, "vis": vis_model}
    robs, rhats, dfs, fix_dur = _run_inference(
        models, dataset_idx, dset, trial_inds, psth_inds_flat, fixation,
        dataset_config,
    )
    robs, rhats, dfs = _trim(robs, rhats, dfs, fix_dur)
    print(f"  trials={robs.shape[0]} time={robs.shape[1]} cells={robs.shape[2]}")

    # Filter cells by min spike count (mirror compare_models_fixrsvp)
    fixrsvp_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
    if len(fixrsvp_mask) < SUBSPACE_K + 1:
        print(f"  skipping: only {len(fixrsvp_mask)} cells pass spike threshold")
        return None
    robs = robs[:, :, fixrsvp_mask]
    rhats = {k: v[:, :, fixrsvp_mask] for k, v in rhats.items()}
    dfs = dfs[:, :, fixrsvp_mask]

    # Affine rescale per neuron, per model
    rhat_beh_rs = _affine_rescale(robs, rhats["beh"], dfs)
    rhat_vis_rs = _affine_rescale(robs, rhats["vis"], dfs)

    # Restrict to neurons present in BOTH the fig2 mask and the fixrsvp mask.
    # fig2 mask filters by spike count + finite cov on the frozen-image stim;
    # fixrsvp mask filters by spike count on fixrsvp.
    common, idx_fig2, idx_fix = _intersect_neuron_masks(fig2_mask, fixrsvp_mask)
    if len(common) < SUBSPACE_K + 1:
        print(f"  skipping: only {len(common)} neurons in intersection")
        return None
    print(f"  fig2 cells={len(fig2_mask)} fixrsvp cells={len(fixrsvp_mask)} "
          f"common={len(common)}")

    Cpsth = Cpsth_full[np.ix_(idx_fig2, idx_fig2)]
    Cfem = Cfem_full[np.ix_(idx_fig2, idx_fig2)]
    Cint = Cint_full[np.ix_(idx_fig2, idx_fig2)]
    rhat_beh_rs = rhat_beh_rs[:, :, idx_fix]
    rhat_vis_rs = rhat_vis_rs[:, :, idx_fix]
    dfs_use = dfs[:, :, idx_fix]

    # Residual covariance
    Sigma_delta, n_used = _residual_cov(rhat_beh_rs, rhat_vis_rs, dfs_use)
    if Sigma_delta is None:
        print("  skipping: not enough valid samples for Sigma_delta")
        return None
    print(f"  Sigma_delta from {n_used} valid time-bins")

    obs = _alignment_metrics(
        Sigma_delta, [Cpsth, Cfem, Cint], ["PSTH", "FEM", "Int"],
    )
    obs["n_common"] = len(common)
    obs["n_fixrsvp_only"] = len(fixrsvp_mask)
    obs["Sigma_delta"] = Sigma_delta

    # Trial-shuffled null on the behavior input. Vision model is invariant to
    # behavior, so we re-use its unshuffled rates rather than re-running it.
    # Permute only among trials that have any fixation samples so the source
    # is always non-empty.
    trials_unique = np.unique(trial_inds)
    NT_full = len(trials_unique)
    valid_trials = np.array([
        ((trial_inds == t) & fixation).any() for t in trials_unique
    ])
    valid_idx = np.where(valid_trials)[0]
    print(f"  running {N_SHUFFLES} behavior trial-shuffles "
          f"({len(valid_idx)}/{NT_full} valid trials)...")

    shuffle_metrics = []
    rng = np.random.default_rng(seed=42)
    for s in range(N_SHUFFLES):
        # Identity permutation everywhere, then permute among valid trials.
        perm = np.arange(NT_full)
        valid_perm = rng.permutation(valid_idx)
        # Avoid identity on the valid block
        for _ in range(5):
            if np.any(valid_perm != valid_idx):
                break
            valid_perm = rng.permutation(valid_idx)
        perm[valid_idx] = valid_perm

        _, rhats_s, _, fix_dur_s = _run_inference(
            {"beh": beh_model}, dataset_idx, dset, trial_inds,
            psth_inds_flat, fixation, dataset_config, behavior_shuffle=perm,
        )
        good_s = fix_dur_s > MIN_FIX_DUR
        n_good_s = int(good_s.sum())
        n_good_obs = rhat_beh_rs.shape[0]
        if n_good_s != n_good_obs:
            print(f"    shuffle {s}: trial-count mismatch ({n_good_s} vs "
                  f"{n_good_obs}), skipping")
            continue
        rhat_beh_s = rhats_s["beh"][good_s][:, :VALID_TIME_BINS]
        rhat_beh_s = rhat_beh_s[:, :, fixrsvp_mask][:, :, idx_fix]
        rhat_beh_s_rs = _affine_rescale(
            robs[:, :, idx_fix], rhat_beh_s, dfs_use
        )
        Sigma_delta_s, n_s = _residual_cov(rhat_beh_s_rs, rhat_vis_rs, dfs_use)
        if Sigma_delta_s is None:
            print(f"    shuffle {s}: too few valid bins")
            continue
        m = _alignment_metrics(
            Sigma_delta_s, [Cpsth, Cfem, Cint], ["PSTH", "FEM", "Int"],
        )
        shuffle_metrics.append(m)
    obs["shuffle_metrics"] = shuffle_metrics
    print(f"  shuffle null: {len(shuffle_metrics)}/{N_SHUFFLES} succeeded")

    return {
        "session": name,
        "subject": fig2["subject"],
        "n_common": len(common),
        "obs": obs,
    }


if CACHE_PATH.exists() and not RECOMPUTE:
    print(f"Loading cached results from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        results = dill.load(f)
else:
    results = []
    for name in candidates:
        try:
            r = process_session(name)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        dill.dump(results, f)
    print(f"\nCached {len(results)} sessions to {CACHE_PATH}")


# %% Per-session figures
REF_COLORS = {"PSTH": "tab:purple", "FEM": "tab:red", "Int": "tab:gray"}


def plot_session(r):
    name = r["session"]
    obs = r["obs"]
    spectrum = obs["spectrum_target"]
    refs = ["PSTH", "FEM", "Int"]
    sm = obs["shuffle_metrics"]
    k_sweep = obs["k_sweep"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Panel 1: Sigma_delta spectrum
    ax = axes[0]
    ax.plot(np.arange(1, len(spectrum) + 1), spectrum / spectrum.sum(),
            'o-', color='C3')
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('fraction of var')
    ax.set_yscale('log')
    ax.set_title(f"$\\Sigma_\\Delta$ spectrum (PR = {obs['pr_target']:.2f})")
    ax.set_xlim(0.5, min(20, len(spectrum)) + 0.5)

    # Panel 2: subspace overlap (k=1 vs k=K)
    ax = axes[1]
    x = np.arange(len(refs))
    obs_k1 = [obs["overlap_k1"][rn] for rn in refs]
    obs_k = [obs["overlap_k"][rn] for rn in refs]
    null_k1 = [np.mean([m["overlap_k1"][rn] for m in sm]) if sm else np.nan for rn in refs]
    null_k = [np.mean([m["overlap_k"][rn] for m in sm]) if sm else np.nan for rn in refs]
    null_k_std = [np.std([m["overlap_k"][rn] for m in sm]) if sm else 0.0 for rn in refs]
    ax.bar(x - 0.2, obs_k1, width=0.35, alpha=0.4, color='C3', label='obs (k=1)')
    ax.bar(x + 0.2, obs_k, width=0.35, alpha=0.85, color='C3', label=f'obs (k={SUBSPACE_K})')
    ax.errorbar(x - 0.2, null_k1, yerr=0, fmt='_', color='k', markersize=12,
                label='null (k=1)')
    ax.errorbar(x + 0.2, null_k, yerr=null_k_std, fmt='o', color='k',
                markersize=4, capsize=3, label=f'null (k={SUBSPACE_K})')
    ax.set_xticks(x)
    ax.set_xticklabels([f"vs $\\Sigma_{{{rn}}}$" for rn in refs])
    ax.set_ylabel('symmetric subspace overlap')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Subspace overlap')

    # Panel 3: capture(Sigma_delta in ref) as function of k
    ax = axes[2]
    for rn in refs:
        ax.plot(k_sweep, obs["capture_target_in_ref_curve"][rn],
                'o-', color=REF_COLORS[rn], label=f"in $\\Sigma_{{{rn}}}$")
        if sm:
            null_curves = np.array([m["capture_target_in_ref_curve"][rn] for m in sm])
            ax.fill_between(k_sweep,
                            null_curves.mean(0) - null_curves.std(0),
                            null_curves.mean(0) + null_curves.std(0),
                            color=REF_COLORS[rn], alpha=0.18)
    ax.set_xlabel('subspace dimension k')
    ax.set_ylabel('fraction of $\\Sigma_\\Delta$ var captured')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('Cumulative variance capture')

    # Panel 4: PR comparison
    ax = axes[3]
    pr_target = obs["pr_target"]
    pr_refs = obs["pr_refs"]
    ax.bar(['$\\Sigma_\\Delta$', '$\\Sigma_{PSTH}$', '$\\Sigma_{FEM}$', '$\\Sigma_{int}$'],
           [pr_target, pr_refs['PSTH'], pr_refs['FEM'], pr_refs['Int']],
           color=['C3', REF_COLORS['PSTH'], REF_COLORS['FEM'], REF_COLORS['Int']])
    ax.set_ylabel('participation ratio')
    ax.set_title('Effective dimensionality')

    fig.suptitle(f"{name} ({r['subject']}, {r['n_common']} cells, "
                 f"shuffles: {len(sm)})")
    fig.tight_layout()
    out = FIG_DIR / f"residual_subspace_{name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"  saved {out}")
    show_or_close(fig)


for r in results:
    plot_session(r)


# %% Cross-session summary panel ---------------------------------------------
def plot_summary(results):
    refs = ["PSTH", "FEM", "Int"]
    n_sess = len(results)
    if n_sess == 0:
        return
    sub = np.array([r["subject"] for r in results])
    n_cells = np.array([r["n_common"] for r in results])

    obs_k1 = {rn: np.array([r["obs"]["overlap_k1"][rn] for r in results]) for rn in refs}
    obs_k = {rn: np.array([r["obs"]["overlap_k"][rn] for r in results]) for rn in refs}
    cap_t = {rn: np.array([r["obs"]["capture_target_in_ref"][rn] for r in results]) for rn in refs}
    pr = np.array([r["obs"]["pr_target"] for r in results])

    # Collect shuffle nulls per-session, take the mean across shuffles within session
    null_k1 = {rn: [] for rn in refs}
    null_k = {rn: [] for rn in refs}
    null_cap = {rn: [] for rn in refs}
    for r in results:
        sm = r["obs"]["shuffle_metrics"]
        for rn in refs:
            null_k1[rn].append(np.mean([m["overlap_k1"][rn] for m in sm]) if sm else np.nan)
            null_k[rn].append(np.mean([m["overlap_k"][rn] for m in sm]) if sm else np.nan)
            null_cap[rn].append(np.mean([m["capture_target_in_ref"][rn] for m in sm]) if sm else np.nan)
    null_k1 = {rn: np.array(v) for rn, v in null_k1.items()}
    null_k = {rn: np.array(v) for rn, v in null_k.items()}
    null_cap = {rn: np.array(v) for rn, v in null_cap.items()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0) PR(Sigma_delta) histogram + per-subject means
    ax = axes[0, 0]
    bins = np.linspace(0, max(15, pr.max() * 1.05), 25)
    for s, color in SUBJECT_COLORS.items():
        m = sub == s
        ax.hist(pr[m], bins=bins, alpha=0.6, color=color, label=f"{s} (n={m.sum()})")
    ax.axvline(np.mean(pr), color='k', ls='--', label=f'mean={np.mean(pr):.2f}')
    ax.set_xlabel('PR($\\Sigma_\\Delta$)')
    ax.set_ylabel('# sessions')
    ax.set_title(f'Effective dim of $\\Sigma_\\Delta$\nN = {n_sess} sessions')
    ax.legend(fontsize=8)

    # (0,1) overlap_k1 PSTH vs FEM scatter (the strong reading)
    ax = axes[0, 1]
    for s, color in SUBJECT_COLORS.items():
        m = sub == s
        ax.scatter(obs_k1["PSTH"][m], obs_k1["FEM"][m],
                   color=color, label=s, s=40, edgecolor='k', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
    # null mean
    if not np.isnan(null_k1["PSTH"]).all():
        ax.scatter(null_k1["PSTH"], null_k1["FEM"],
                   marker='x', color='gray', alpha=0.5, label='null', s=25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$\\Sigma_\\Delta$ vs $\\Sigma_{PSTH}$ (k=1)')
    ax.set_ylabel('$\\Sigma_\\Delta$ vs $\\Sigma_{FEM}$ (k=1)')
    ax.set_title('Leading-mode alignment\n(off-diagonal = asymmetric)')
    ax.legend(fontsize=8)

    # (0,2) overlap_k for all three refs - paired bars per session, summarized
    ax = axes[0, 2]
    x = np.arange(len(refs))
    obs_means = [obs_k[rn].mean() for rn in refs]
    obs_sems = [obs_k[rn].std() / np.sqrt(n_sess) for rn in refs]
    null_means = [np.nanmean(null_k[rn]) for rn in refs]
    null_sems = [np.nanstd(null_k[rn]) / np.sqrt(np.isfinite(null_k[rn]).sum() or 1)
                 for rn in refs]
    ax.bar(x - 0.2, obs_means, yerr=obs_sems, width=0.35,
           color='C3', alpha=0.85, capsize=3, label='observed')
    ax.bar(x + 0.2, null_means, yerr=null_sems, width=0.35,
           color='gray', alpha=0.7, capsize=3, label='shuffle null')
    ax.set_xticks(x)
    ax.set_xticklabels([f"vs $\\Sigma_{{{rn}}}$" for rn in refs])
    ax.set_ylabel(f'overlap (k={SUBSPACE_K})')
    ax.set_ylim(0, 1)
    ax.set_title(f'Pooled top-k overlap\n(mean ± SEM, n={n_sess})')
    ax.legend(fontsize=8)

    # (1,0) capture(Sigma_delta in ref subspace) per session, three refs
    ax = axes[1, 0]
    width = 0.25
    for i, rn in enumerate(refs):
        offset = (i - 1) * width
        for s, color in SUBJECT_COLORS.items():
            m = sub == s
            xs = np.arange(n_sess)[m] + offset
            ax.scatter(xs, cap_t[rn][m], color=REF_COLORS[rn],
                       marker='o' if s == 'Allen' else 's', s=15)
    ax.set_xlabel('session index')
    ax.set_ylabel('capture(Σ_Δ in ref subspace, k=K)')
    ax.set_title(f'Per-session variance capture\n'
                 f'PSTH={cap_t["PSTH"].mean():.2f}, '
                 f'FEM={cap_t["FEM"].mean():.2f}, '
                 f'Int={cap_t["Int"].mean():.2f}')
    handles = [plt.Line2D([0], [0], marker='o', color=REF_COLORS[rn], lw=0, label=rn)
               for rn in refs]
    ax.legend(handles=handles, fontsize=8)

    # (1,1) capture k-sweep curves: pooled mean across sessions, with null bands
    ax = axes[1, 1]
    k_sweep = SUBSPACE_K_SWEEP
    for rn in refs:
        curves = np.array([r["obs"]["capture_target_in_ref_curve"][rn] for r in results])
        ax.plot(k_sweep, curves.mean(0), 'o-', color=REF_COLORS[rn], label=rn)
        ax.fill_between(k_sweep,
                        curves.mean(0) - curves.std(0) / np.sqrt(n_sess),
                        curves.mean(0) + curves.std(0) / np.sqrt(n_sess),
                        color=REF_COLORS[rn], alpha=0.25)
        # Null
        null_curves = []
        for r in results:
            sm = r["obs"]["shuffle_metrics"]
            if sm:
                null_curves.append(np.mean(
                    [m["capture_target_in_ref_curve"][rn] for m in sm], axis=0))
        if null_curves:
            null_curves = np.array(null_curves)
            ax.plot(k_sweep, null_curves.mean(0), '--',
                    color=REF_COLORS[rn], alpha=0.5)
    ax.set_xlabel('subspace dimension k')
    ax.set_ylabel('capture(Σ_Δ in ref subspace)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, title='solid = obs\ndashed = null')
    ax.set_title('Pooled cumulative capture')

    # (1,2) Asymmetry: capture in PSTH vs capture in FEM, scatter
    ax = axes[1, 2]
    for s, color in SUBJECT_COLORS.items():
        m = sub == s
        ax.scatter(cap_t["PSTH"][m], cap_t["FEM"][m],
                   color=color, label=s, s=40, edgecolor='k', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('capture in PSTH subspace')
    ax.set_ylabel('capture in FEM subspace')
    ax.set_title(f'PSTH vs FEM capture (k={SUBSPACE_K})')
    ax.legend(fontsize=8)

    fig.suptitle('Behavior-vs-vision residual subspace — cross-session summary',
                 y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "residual_subspace_summary.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=300)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"\nSaved summary {out}")
    show_or_close(fig)


plot_summary(results)


# %% Cross-session summary print
print("\n=== Summary across sessions ===")
print(f"{'session':<22}{'subj':<8}{'n':<5}{'PR':<7}"
      f"{'ov1_PS':<8}{'ov1_FE':<8}{'ov1_In':<8}"
      f"{'ov_PS':<7}{'ov_FE':<7}{'ov_In':<7}"
      f"{'cap_PS':<8}{'cap_FE':<8}{'cap_In':<8}{'shuf':<5}")
for r in results:
    o = r["obs"]
    sm = o["shuffle_metrics"]
    print(f"{r['session']:<22}{r['subject']:<8}{r['n_common']:<5}"
          f"{o['pr_target']:<7.2f}"
          f"{o['overlap_k1']['PSTH']:<8.3f}"
          f"{o['overlap_k1']['FEM']:<8.3f}"
          f"{o['overlap_k1']['Int']:<8.3f}"
          f"{o['overlap_k']['PSTH']:<7.3f}"
          f"{o['overlap_k']['FEM']:<7.3f}"
          f"{o['overlap_k']['Int']:<7.3f}"
          f"{o['capture_target_in_ref']['PSTH']:<8.3f}"
          f"{o['capture_target_in_ref']['FEM']:<8.3f}"
          f"{o['capture_target_in_ref']['Int']:<8.3f}"
          f"{len(sm):<5}")

print("\nDone.")
