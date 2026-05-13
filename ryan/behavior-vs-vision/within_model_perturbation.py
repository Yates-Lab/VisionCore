# %% Imports and configuration
"""
Within-Model-B perturbation analysis on fixRSVP. Replaces subspace_residual.py.

Conditions, all using the same Behavior model weights:
  - vis            : vision-only model (single external reference)
  - beh_intact     : Behavior model, full behavior input
  - beh_permuted   : Behavior model, behavior tensor permuted across trials
  - beh_zeroed     : Behavior model, behavior tensor zeroed
  - beh_pos_only   : Behavior model, eye_vel channels zeroed (eye_pos kept)
  - beh_vel_only   : Behavior model, eye_pos channels zeroed (eye_vel kept)

Per session, per condition we compute:
  * per-cell BPS on affine-rescaled rates
  * Sigma_FEM_pred = E_t[Cov_i rhat(t, i, :)]   (per-time across-trial cov, n-weighted)
  * comparison of Sigma_FEM_pred vs empirical Sigma_FEM (fig2_decomposition.pkl):
      - magnitude / trace ratio
      - top-k eigenvector overlap (k=1, k=5)
      - participation ratio
  * per-cell FEM-r: Pearson corr( rhat - mean_i rhat(t), y - mean_i y(t) )
                    across all (i, t) — interpretable scalar of trial-to-trial match
  * linear residual probe: ridge regression of (y - rhat_vis) on raw eye_pos/eye_vel.
    Lower bound on what behavior contributes that vision-only missed.

Cache: outputs/cache/behavior_vs_vision_within_model.pkl
Figs:  outputs/figures/behavior-vs-vision/within_model/
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
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


# ---------------------------------------------------------------------------
# Paths & checkpoints (must match compare_models_fixrsvp.py / subspace_residual.py)
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

FIG_DIR = FIGURES_DIR / "behavior-vs-vision" / "within_model"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "behavior_vs_vision_within_model.pkl"


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
RECOMPUTE = False              # cache up-to-date; flip to True to re-run inference
DT = 1 / 120
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 200
MIN_TRIALS_PER_T = 10
SUBSPACE_K = 5
FIG2_WINDOW_IDX = 0
CCNORM_N_SPLITS = 200          # Schoppe split-half resamples for CCnorm

# Behavior tensor channel layout: [eye_vel x 20, eye_pos x 2] (multi_basic_120.yaml).
EYE_VEL_CHANNELS = slice(0, 20)
EYE_POS_CHANNELS = slice(20, 22)

# Pilot mode: limit to a small number of sessions for testing.
N_PILOT_SESSIONS = 2
PILOT_SESSIONS = ["Allen_2022-02-16", "Logan_2020-03-04"]
RUN_ALL = True                 # if True, ignore N_PILOT_SESSIONS / PILOT_SESSIONS

CONDITIONS = [
    "vis",
    "beh_intact",
    "beh_permuted",
    "beh_zeroed",
    "beh_pos_only",
    "beh_vel_only",
]
COND_COLORS = {
    "vis":            "tab:gray",
    "beh_intact":     "tab:red",
    "beh_permuted":   "tab:olive",
    "beh_zeroed":     "tab:orange",
    "beh_pos_only":   "tab:purple",
    "beh_vel_only":   "tab:cyan",
}
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
from eval.eval_stack_utils import (
    load_single_dataset, run_model, rescale_rhat, bits_per_spike,
    ccnorm_split_half_variable_trials,
)

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

def gather_fixrsvp(model_owner, dataset_idx):
    """Return a dict with everything needed to drive inference on one fixRSVP session.

    Keys:
        dset, trial_inds, psth_inds, fixation, dataset_config,
        trials_unique, NT, T, NC.
    """
    train_data, val_data, dataset_config = load_single_dataset(model_owner, dataset_idx)
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

    trials_unique = np.unique(trial_inds)
    NT = len(trials_unique)
    NC = np.asarray(dset['robs']).shape[1]
    T = int(psth_inds_flat.max()) + 1

    return {
        'dset': dset,
        'trial_inds': trial_inds,
        'psth_inds': psth_inds_flat,
        'fixation': fixation,
        'dataset_config': dataset_config,
        'trials_unique': trials_unique,
        'NT': NT,
        'T': T,
        'NC': NC,
    }


def make_modifiers(info, perm_seed=42):
    """Construct behavior_modifier callables for each condition.

    Each returns (behavior_tensor, itrial) -> tensor of same shape.
    """
    trials = info['trials_unique']
    trial_inds = info['trial_inds']
    fixation = info['fixation']
    dset = info['dset']
    NT = info['NT']

    # Pre-cache each trial's behavior tensor.
    src_behavior = []
    for itrial, t in enumerate(trials):
        ix = (trial_inds == t) & fixation
        if not ix.any():
            src_behavior.append(None)
        else:
            src_behavior.append(dset['behavior'][ix])

    # Permutation among valid trials only.
    valid = np.array([b is not None for b in src_behavior])
    valid_idx = np.where(valid)[0]
    rng = np.random.default_rng(perm_seed)
    perm = np.arange(NT)
    valid_perm = rng.permutation(valid_idx)
    for _ in range(5):
        if np.any(valid_perm != valid_idx):
            break
        valid_perm = rng.permutation(valid_idx)
    perm[valid_idx] = valid_perm

    def mod_intact(b, _i):
        return b

    def mod_zeroed(b, _i):
        return torch.zeros_like(b)

    def mod_pos_only(b, _i):
        b = b.clone()
        b[..., EYE_VEL_CHANNELS] = 0
        return b

    def mod_vel_only(b, _i):
        b = b.clone()
        b[..., EYE_POS_CHANNELS] = 0
        return b

    def mod_permuted(b, itrial):
        src = src_behavior[perm[itrial]]
        if src is None:
            return torch.zeros_like(b)
        b_new = b.clone()
        n_src = src.shape[0]
        n_dst = b.shape[0]
        n_use = min(n_src, n_dst)
        b_new[:n_use] = src[:n_use]
        if n_use < n_dst:
            b_new[n_use:] = src.mean(dim=0, keepdim=True)
        return b_new

    return {
        'beh_intact':   mod_intact,
        'beh_permuted': mod_permuted,
        'beh_zeroed':   mod_zeroed,
        'beh_pos_only': mod_pos_only,
        'beh_vel_only': mod_vel_only,
    }


def run_condition(model, dataset_idx, info, behavior_modifier=None, desc="    "):
    """Run inference under one condition. Returns rhat, robs, dfs, fix_dur, behavior_tensor."""
    dset = info['dset']
    trial_inds = info['trial_inds']
    psth_inds_flat = info['psth_inds']
    fixation = info['fixation']
    dataset_config = info['dataset_config']
    trials = info['trials_unique']
    NT, T, NC = info['NT'], info['T'], info['NC']

    rhat = np.full((NT, T, NC), np.nan)
    robs = np.full((NT, T, NC), np.nan)
    dfs  = np.full((NT, T, NC), np.nan)
    fix_dur = np.full(NT, np.nan)
    # also stash the (possibly modified) raw behavior for the linear probe
    # we save the unmodified intact behavior; that will be set when behavior_modifier is None
    # Use 22 channels as in dataset config
    BEH_DIM = dset['behavior'].shape[-1]
    behavior_used = np.full((NT, T, BEH_DIM), np.nan)

    stim_lags = np.array(dataset_config['keys_lags']['stim'])
    robs_arr = np.asarray(dset['robs'])

    for itrial in tqdm(range(NT), leave=False, desc=desc):
        ix = (trial_inds == trials[itrial]) & fixation
        if not np.any(ix):
            continue
        stim_indices = np.where(ix)[0]
        stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
        stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
        behavior = dset['behavior'][ix]
        if behavior_modifier is not None:
            behavior = behavior_modifier(behavior, itrial)

        batch = {'stim': stim, 'behavior': behavior}
        out = run_model(model, batch, dataset_idx=dataset_idx)

        t_inds = psth_inds_flat[ix].astype(int)
        fix_dur[itrial] = len(t_inds)
        rhat[itrial, t_inds] = out['rhat'].detach().cpu().numpy()
        robs[itrial, t_inds] = robs_arr[ix]
        dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
        behavior_used[itrial, t_inds] = behavior.detach().cpu().numpy()

    return rhat, robs, dfs, fix_dur, behavior_used


def trim(rhat, robs, dfs, behavior_used, fix_dur, T_keep=VALID_TIME_BINS):
    good = fix_dur > MIN_FIX_DUR
    iix = np.arange(min(T_keep, rhat.shape[1]))
    return (
        rhat[good][:, iix],
        robs[good][:, iix],
        dfs[good][:, iix],
        behavior_used[good][:, iix],
        good,
    )


def affine_rescale(robs, rhat, dfs):
    NT, T, NC = robs.shape
    rhat_rs, _ = rescale_rhat(
        torch.from_numpy(robs.reshape(-1, NC)),
        torch.from_numpy(rhat.reshape(-1, NC)),
        torch.from_numpy(dfs.reshape(-1, NC)),
        mode='affine',
    )
    return rhat_rs.reshape(NT, T, NC).detach().cpu().numpy()


def compute_bps(rhat_rs, robs, dfs):
    NT, T, NC = robs.shape
    rhat_flat = rhat_rs.reshape(NT * T, NC)
    robs_flat = robs.reshape(NT * T, NC)
    dfs_flat  = dfs.reshape(NT * T, NC)
    dfs_mask = (np.nan_to_num(dfs_flat, nan=0.0) > 0.5).astype(np.float32)
    bps = bits_per_spike(
        torch.from_numpy(rhat_flat).float(),
        torch.from_numpy(robs_flat).float(),
        torch.from_numpy(dfs_mask),
    ).detach().cpu().numpy()
    return bps


def compute_sigma_fem_pred(rhat, dfs, min_trials=MIN_TRIALS_PER_T):
    """Sigma_FEM_pred = sum_t (n_t * Cov_i rhat(t, i, :)) / sum_t n_t.

    Per-time across-trial covariance, weighted by valid trial count, then averaged.
    For a deterministic model this is the analog of empirical Σ_FEM.
    """
    NT, T, NC = rhat.shape
    bin_valid = (np.nan_to_num(dfs, nan=0.0) > 0).all(axis=-1) \
        & np.isfinite(rhat).all(axis=-1)            # (NT, T)
    weighted_sum = np.zeros((NC, NC))
    weight_total = 0
    for t in range(T):
        valid_i = bin_valid[:, t]
        n_t = int(valid_i.sum())
        if n_t < min_trials:
            continue
        rhat_t = rhat[valid_i, t, :]
        rhat_t_centered = rhat_t - rhat_t.mean(0, keepdims=True)
        cov_t = rhat_t_centered.T @ rhat_t_centered / max(n_t - 1, 1)
        weighted_sum += n_t * cov_t
        weight_total += n_t
    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def compute_cc_single(rhat, robs, dfs, min_trials=MIN_TRIALS_PER_T):
    """Per-cell single-trial CC: Pearson r between (rhat - rhat_PSTH(t)) and
    (y - y_PSTH(t)), pooled across (trial, time). Quantifies how well the
    model captures *trial-to-trial deviations from the PSTH* — the FEM-driven
    structure. Contrast with CCnorm, which captures the PSTH itself.
    Not noise-corrected — bounded above by the Poisson noise ceiling.
    """
    NT, T, NC = rhat.shape
    bin_valid = (np.nan_to_num(dfs, nan=0.0) > 0)         # (NT, T, NC)

    rhat_z = np.where(bin_valid, rhat, 0.0)
    robs_z = np.where(bin_valid, robs, 0.0)
    n_valid = bin_valid.sum(axis=0).astype(float)         # (T, NC)
    safe_n = np.maximum(n_valid, 1)

    psth_rhat = rhat_z.sum(0) / safe_n                    # (T, NC)
    psth_robs = robs_z.sum(0) / safe_n

    rhat_resid = np.where(bin_valid, rhat - psth_rhat[None], np.nan)
    robs_resid = np.where(bin_valid, robs - psth_robs[None], np.nan)

    valid_tc = n_valid >= min_trials                       # (T, NC)
    rhat_resid[:, ~valid_tc] = np.nan
    robs_resid[:, ~valid_tc] = np.nan

    cc_single = np.full(NC, np.nan)
    cc_single_ve = np.full(NC, np.nan)
    for c in range(NC):
        x = rhat_resid[:, :, c].ravel()
        y = robs_resid[:, :, c].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            continue
        x, y = x[m], y[m]
        if x.std() < 1e-8 or y.std() < 1e-8:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        cc_single[c] = r
        ss_tot = float(np.sum(y * y))
        ss_res = float(np.sum((y - x) ** 2))
        if ss_tot > 1e-8:
            cc_single_ve[c] = 1.0 - ss_res / ss_tot
    return cc_single, cc_single_ve


def compute_cc_norm(rhat_rs, robs, dfs, n_splits=CCNORM_N_SPLITS):
    """Schoppe-style noise-corrected CC between rhat and the trial-averaged
    PSTH. Quantifies how well the model captures the *PSTH itself* (the
    cross-trial average), normalized by the achievable upper bound given
    finite-trial sampling noise.

    Returns (ccnorm, ccabs, ccmax) per cell, with ccnorm = NaN when
    split-half estimates are unstable.
    """
    cc1, abs1, max1, _, _ = ccnorm_split_half_variable_trials(
        robs, rhat_rs, dfs, n_splits=n_splits, return_components=True,
    )
    cc2, abs2, max2, _, _ = ccnorm_split_half_variable_trials(
        robs, rhat_rs, dfs, n_splits=n_splits, return_components=True,
    )
    unstable = (cc1 - cc2) ** 2 > 0.01
    ccnorm = 0.5 * (cc1 + cc2)
    ccabs  = 0.5 * (abs1 + abs2)
    ccmax  = 0.5 * (max1 + max2)
    ccnorm[unstable] = np.nan
    return ccnorm, ccabs, ccmax


def gather_eye_state(info, T_full, good_mask, iix):
    """Build (NT_used, T_used, 4) raw [eye_pos_x, eye_pos_y, eye_vel_x, eye_vel_y].

    Velocity is finite-diff per trial (forward-filled at t=0).
    """
    trial_inds = info['trial_inds']
    psth_inds = info['psth_inds']
    fixation = info['fixation']
    eyepos_flat = info['dset']['eyepos']                 # (n_samples, 2)
    eyepos_flat = np.asarray(eyepos_flat)
    trials = info['trials_unique']
    NT = info['NT']

    eye_pos_full = np.full((NT, T_full, 2), np.nan)
    for itrial, t in enumerate(trials):
        ix = (trial_inds == t) & fixation
        if not np.any(ix):
            continue
        t_inds = psth_inds[ix].astype(int)
        eye_pos_full[itrial, t_inds] = eyepos_flat[ix]

    eye_pos_used = eye_pos_full[good_mask][:, iix]        # (NT_used, T_used, 2)
    eye_vel_used = np.zeros_like(eye_pos_used)
    eye_vel_used[:, 1:] = eye_pos_used[:, 1:] - eye_pos_used[:, :-1]
    eye_vel_used[:, 0] = eye_vel_used[:, 1]               # forward-fill at t=0

    return np.concatenate([eye_pos_used, eye_vel_used], axis=-1)  # (..., 4)


def linear_residual_probe(rhat_vis_rs, robs, dfs, eye_features,
                          alphas=(0.1, 1.0, 10.0, 100.0),
                          n_folds=5, seed=42):
    """Ridge-regress per-cell vision-only residual on raw eye_pos + eye_vel.

    Closed-form ridge with K-fold CV; per-cell α picked from the grid by mean
    held-out R². Returns held-out R² per cell at the best α.
    """
    NT, T, NC = robs.shape
    valid = (np.nan_to_num(dfs, nan=0.0) > 0).all(axis=-1) \
            & np.isfinite(rhat_vis_rs).all(axis=-1) \
            & np.isfinite(eye_features).all(axis=-1)      # (NT, T)
    if valid.sum() < 200:
        return np.full(NC, np.nan), np.full(NC, np.nan)

    X_full = eye_features[valid]                          # (n_valid, 4)
    res_full = (robs - rhat_vis_rs)[valid]                # (n_valid, NC)

    X_mean = X_full.mean(0, keepdims=True)
    X_std = X_full.std(0, keepdims=True) + 1e-6
    Xz = (X_full - X_mean) / X_std
    Xz = np.concatenate([Xz, np.ones((Xz.shape[0], 1))], axis=1)
    n, d = Xz.shape

    rng = np.random.default_rng(seed)
    fold_idx = rng.integers(0, n_folds, size=n)
    eye_d = np.eye(d); eye_d[-1, -1] = 0

    # preds_per_alpha[a, n_valid, NC]
    preds_per_alpha = np.full((len(alphas), n, NC), np.nan)
    for a_idx, alpha in enumerate(alphas):
        for k in range(n_folds):
            te = (fold_idx == k)
            tr = ~te
            if tr.sum() < d + 1:
                continue
            XtX = Xz[tr].T @ Xz[tr] + alpha * eye_d
            Xty = Xz[tr].T @ res_full[tr]
            try:
                W = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue
            preds_per_alpha[a_idx, te] = Xz[te] @ W

    r2 = np.full(NC, np.nan)
    best_alpha = np.full(NC, np.nan)
    for c in range(NC):
        y = res_full[:, c]
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-8:
            continue
        best = -np.inf
        best_a = np.nan
        for a_idx, alpha in enumerate(alphas):
            p = preds_per_alpha[a_idx, :, c]
            m = np.isfinite(p)
            if m.sum() < 100:
                continue
            ss_res = float(np.sum((y[m] - p[m]) ** 2))
            r = 1.0 - ss_res / ss_tot
            if r > best:
                best = r
                best_a = alpha
        if np.isfinite(best):
            r2[c] = best
            best_alpha[c] = best_a
    return r2, best_alpha


def intersect_neuron_masks(fig2_mask, fixrsvp_mask):
    fig2_mask = np.asarray(fig2_mask)
    fixrsvp_mask = np.asarray(fixrsvp_mask)
    common = np.intersect1d(fig2_mask, fixrsvp_mask)
    idx_fig2 = np.array([np.where(fig2_mask == g)[0][0] for g in common])
    idx_fix  = np.array([np.where(fixrsvp_mask == g)[0][0] for g in common])
    return common, idx_fig2, idx_fix


def empirical_cov_metrics(Sigma_pred_common, Sigma_emp_common, k=SUBSPACE_K):
    """Magnitude / trace / overlap metrics of model Sigma vs empirical Sigma."""
    if Sigma_pred_common is None:
        return None
    Sp = project_to_psd(Sigma_pred_common)
    Se = project_to_psd(Sigma_emp_common)
    sp_norm = np.linalg.norm(Sp, ord='fro')
    se_norm = np.linalg.norm(Se, ord='fro')
    sp_tr = np.trace(Sp)
    se_tr = np.trace(Se)
    w_p, V_p = np.linalg.eigh(Sp); V_p = V_p[:, ::-1]
    w_e, V_e = np.linalg.eigh(Se); V_e = V_e[:, ::-1]
    kk = max(1, min(k, V_p.shape[1] - 1, V_e.shape[1] - 1))
    return {
        'mag_ratio': sp_norm / max(se_norm, 1e-12),
        'tr_ratio':  sp_tr / max(se_tr, 1e-12),
        'overlap_k1': symmetric_subspace_overlap(V_p[:, :1], V_e[:, :1]),
        'overlap_k':  symmetric_subspace_overlap(V_p[:, :kk], V_e[:, :kk]),
        'pr_pred':    participation_ratio(Sp),
        'pr_emp':     participation_ratio(Se),
    }


def cov_to_corr(C, eps=1e-8):
    d = np.sqrt(np.maximum(np.diag(C), eps))
    return C / np.outer(d, d)


def get_upper_triangle(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


# %% Load fig2 cache & fixrsvp metrics ---------------------------------------
print("Loading fig2 decomposition...")
with open(FIG2_DECOMP_PATH, "rb") as f:
    fig2_sessions = dill.load(f)
fig2_by_name = {s["session"]: s for s in fig2_sessions}
print(f"  {len(fig2_by_name)} sessions")

print("Loading fixrsvp metrics cache...")
with open(FIXRSVP_METRICS_PATH, "rb") as f:
    fixrsvp_sessions = dill.load(f)
fixrsvp_by_name = {s["session"]: s for s in fixrsvp_sessions}


# %% Per-session driver -------------------------------------------------------
def process_session(name):
    print(f"\n--- {name} ---")
    if name not in fig2_by_name:
        print("  not in fig2 cache; skipping")
        return None
    if name not in session_to_idx:
        print("  not in model dataset list; skipping")
        return None

    fig2 = fig2_by_name[name]
    dataset_idx = session_to_idx[name]
    info = gather_fixrsvp(beh_model, dataset_idx)
    print(f"  trials={info['NT']} T={info['T']} NC={info['NC']}")

    modifiers = make_modifiers(info)

    # Inference per condition. Vis uses the vision-only model; the rest use beh.
    rhats = {}
    robs_ref = None
    dfs_ref = None
    fix_dur_ref = None
    behavior_intact = None

    cond_specs = [
        ('vis',          vis_model, None),
        ('beh_intact',   beh_model, modifiers['beh_intact']),
        ('beh_permuted', beh_model, modifiers['beh_permuted']),
        ('beh_zeroed',   beh_model, modifiers['beh_zeroed']),
        ('beh_pos_only', beh_model, modifiers['beh_pos_only']),
        ('beh_vel_only', beh_model, modifiers['beh_vel_only']),
    ]

    for cond, model, mod in cond_specs:
        print(f"  cond={cond}")
        rhat, robs, dfs, fix_dur, beh_used = run_condition(
            model, dataset_idx, info, behavior_modifier=mod,
            desc=f"    {cond}",
        )
        if robs_ref is None:
            robs_ref = robs
            dfs_ref = dfs
            fix_dur_ref = fix_dur
        rhats[cond] = rhat
        if cond == 'beh_intact':
            behavior_intact = beh_used

    # Trim to good trials and clamp to VALID_TIME_BINS
    good = fix_dur_ref > MIN_FIX_DUR
    iix = np.arange(min(VALID_TIME_BINS, robs_ref.shape[1]))
    robs = robs_ref[good][:, iix]
    dfs  = dfs_ref[good][:, iix]
    rhats = {c: r[good][:, iix] for c, r in rhats.items()}
    behavior_intact = behavior_intact[good][:, iix]

    # Filter cells by spike count (mirror compare_models_fixrsvp.py)
    fixrsvp_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
    if len(fixrsvp_mask) < 5:
        print(f"  skipping: {len(fixrsvp_mask)} cells pass spike threshold")
        return None
    robs = robs[:, :, fixrsvp_mask]
    dfs  = dfs[:, :, fixrsvp_mask]
    rhats = {c: r[:, :, fixrsvp_mask] for c, r in rhats.items()}
    print(f"  used: trials={robs.shape[0]} T={robs.shape[1]} NC={robs.shape[2]}")

    # Affine rescale per condition
    rhats_rs = {c: affine_rescale(robs, r, dfs) for c, r in rhats.items()}

    # Per-cell BPS, Sigma_FEM_pred, CC_single (residual), CC_norm (PSTH)
    bps = {c: compute_bps(rhats_rs[c], robs, dfs) for c in rhats_rs}
    sigma_fem_pred = {c: compute_sigma_fem_pred(rhats_rs[c], dfs) for c in rhats_rs}
    cc_single = {}
    cc_single_ve = {}
    cc_norm = {}
    cc_abs = {}
    cc_max = {}
    for c in rhats_rs:
        s, ve = compute_cc_single(rhats_rs[c], robs, dfs)
        cc_single[c]    = s
        cc_single_ve[c] = ve
        print(f"    ccnorm: {c}")
        n, a, m_ = compute_cc_norm(rhats_rs[c], robs, dfs)
        cc_norm[c] = n
        cc_abs[c]  = a
        cc_max[c]  = m_

    # Linear residual probe was dropped in v3 (low signal-to-narrative ratio).
    # `gather_eye_state` and `linear_residual_probe` remain defined for ad-hoc use.

    # Empirical Sigma_FEM comparison (intersection cells)
    fig2_mask = np.asarray(fig2['neuron_mask'])
    common, idx_fig2, idx_fix = intersect_neuron_masks(fig2_mask, fixrsvp_mask)
    empirical = {'common': common, 'metrics': {}, 'corr_pairs': {}, 'Cfem_emp': None}
    if len(common) >= 5:
        mats = fig2['mats'][FIG2_WINDOW_IDX]
        Cpsth = mats['PSTH'][np.ix_(idx_fig2, idx_fig2)]
        Crate = mats['Intercept'][np.ix_(idx_fig2, idx_fig2)]
        Cfem_emp = Crate - Cpsth
        empirical['Cfem_emp'] = Cfem_emp
        emp_corr_pairs = get_upper_triangle(cov_to_corr(project_to_psd(Cfem_emp)))
        empirical['emp_corr_pairs'] = emp_corr_pairs
        for c in CONDITIONS:
            S = sigma_fem_pred[c]
            if S is None:
                empirical['metrics'][c] = None
                empirical['corr_pairs'][c] = None
                continue
            S_common = S[np.ix_(idx_fix, idx_fix)]
            empirical['metrics'][c] = empirical_cov_metrics(S_common, Cfem_emp)
            empirical['corr_pairs'][c] = get_upper_triangle(
                cov_to_corr(project_to_psd(S_common))
            )

    # Reduce stored rhats footprint: keep summary tensors only
    return {
        'session':         name,
        'subject':         fig2['subject'],
        'n_cells':         len(fixrsvp_mask),
        'n_common_cells':  len(common),
        'fixrsvp_mask':    fixrsvp_mask,
        'common_cells':    common,
        'idx_fix':         idx_fix if len(common) else None,
        'bps':             bps,
        'cc_single':       cc_single,         # was fem_corr
        'cc_single_ve':    cc_single_ve,
        'cc_norm':         cc_norm,
        'cc_abs':          cc_abs,
        'cc_max':          cc_max,
        'sigma_fem_pred':  sigma_fem_pred,
        'empirical':       empirical,
    }


# %% Pick session list & run --------------------------------------------------
if RUN_ALL:
    candidates = list(fig2_by_name.keys() & fixrsvp_by_name.keys())
else:
    candidates = [s for s in PILOT_SESSIONS
                  if s in fig2_by_name and s in fixrsvp_by_name]
    candidates = candidates[:N_PILOT_SESSIONS]

print(f"\nWill process {len(candidates)} sessions: {candidates}")

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


# %% Per-session figures ------------------------------------------------------
def plot_session(r):
    name = r['session']
    bps = r['bps']
    cc_single = r['cc_single']
    cc_norm = r.get('cc_norm', {})
    em = r['empirical']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) BPS scatter: each condition vs intact
    ax = axes[0, 0]
    intact_bps = bps['beh_intact']
    others = [c for c in CONDITIONS if c != 'beh_intact']
    for cond in others:
        d = bps[cond] - intact_bps
        med = np.nanmedian(d)
        ax.scatter(intact_bps, bps[cond], alpha=0.5, s=15,
                   color=COND_COLORS[cond],
                   label=f"{cond} (Δmed={med:+.3f})")
    valid = np.concatenate([intact_bps] + [bps[c] for c in others])
    valid = valid[np.isfinite(valid)]
    if valid.size:
        lims = [min(valid.min(), 0), max(valid.max(), 0.5)]
        ax.plot(lims, lims, 'k--', lw=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('BPS (beh_intact)')
    ax.set_ylabel('BPS (other condition)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_aspect('equal')
    ax.set_title('Per-cell BPS, condition vs intact')

    # (0,1) ΔBPS distribution as boxplot
    ax = axes[0, 1]
    box_data = []
    box_labels = []
    box_colors = []
    for cond in others:
        d = bps[cond] - intact_bps
        d = d[np.isfinite(d)]
        box_data.append(d)
        box_labels.append(cond.replace('beh_', '').replace('_', '\n'))
        box_colors.append(COND_COLORS[cond])
    bp = ax.boxplot(box_data, labels=box_labels, showfliers=False,
                    patch_artist=True)
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('ΔBPS (cond − intact)')
    ax.set_title('ΔBPS distributions')

    # (0,2) CC_single per condition: histograms overlaid
    ax = axes[0, 2]
    for cond in CONDITIONS:
        d = cc_single[cond]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        ax.hist(d, bins=np.linspace(-0.4, 0.7, 25), alpha=0.4,
                color=COND_COLORS[cond],
                label=f"{cond} (med={np.nanmedian(d):.3f})")
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('CC_single (Pearson r of residual-from-PSTH)')
    ax.set_ylabel('# cells')
    ax.legend(fontsize=7)
    ax.set_title('CC_single per condition (single-trial residual)')

    # (1,0) Σ_FEM vs empirical: trace ratio bar
    ax = axes[1, 0]
    if em['metrics'] and any(v is not None for v in em['metrics'].values()):
        labels, tr_ratios, ov_k = [], [], []
        for cond in CONDITIONS:
            m = em['metrics'].get(cond)
            if m is None:
                continue
            labels.append(cond)
            tr_ratios.append(m['tr_ratio'])
            ov_k.append(m['overlap_k'])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, tr_ratios, width=0.35, color='C0', alpha=0.85,
               label='trace ratio')
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, ov_k, width=0.35, color='C1', alpha=0.85,
                label=f'overlap k={SUBSPACE_K}')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('tr(Σ_pred) / tr(Σ_FEM_emp)', color='C0')
        ax2.set_ylabel(f'top-{SUBSPACE_K} eigvec overlap', color='C1')
        ax.axhline(1, color='C0', lw=0.5, ls='--')
        ax.set_title(f'Σ_FEM_pred vs empirical Σ_FEM '
                     f'(n={r["n_common_cells"]} common cells)')
    else:
        ax.text(0.5, 0.5, 'no empirical match', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()

    # (1,1) Pairwise corr scatter: empirical vs intact, empirical vs permuted
    ax = axes[1, 1]
    if em.get('emp_corr_pairs') is not None:
        emp_pairs = em['emp_corr_pairs']
        for cond, color in [('beh_intact', 'tab:red'),
                            ('beh_permuted', 'tab:olive'),
                            ('vis', 'tab:gray')]:
            cp = em['corr_pairs'].get(cond)
            if cp is None:
                continue
            ax.scatter(emp_pairs, cp, alpha=0.3, s=8, color=color,
                       label=cond)
        lims = [-0.4, 0.7]
        ax.plot(lims, lims, 'k--', lw=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('empirical Σ_FEM corr')
        ax.set_ylabel('Σ_pred corr')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        ax.set_title('Pairwise corr: empirical vs predicted')

    # (1,2) Per-cell scatter Δperm vs Δzero, colored by Δvis
    # Same-axis comparison: do trial-specific (perm) and any-behavior (zero)
    # losses come from the same cells, and how do they relate to the
    # vision-only baseline (Δvis)?
    ax = axes[1, 2]
    d_perm = bps['beh_permuted'] - bps['beh_intact']
    d_zero = bps['beh_zeroed']  - bps['beh_intact']
    d_vis  = bps['vis']         - bps['beh_intact']
    m = np.isfinite(d_perm) & np.isfinite(d_zero) & np.isfinite(d_vis)
    if m.any():
        color = np.clip(d_vis[m], -0.15, 0.0)
        sc = ax.scatter(d_perm[m], d_zero[m], c=color, cmap='RdBu_r',
                        s=25, edgecolor='k', linewidth=0.3,
                        vmin=-0.15, vmax=0.0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Δvis BPS')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        plo = min(d_perm[m].min(), d_zero[m].min())
        phi = max(d_perm[m].max(), d_zero[m].max())
        ax.plot([plo, phi], [plo, phi], 'k--', lw=0.5, label='y = x')
        med_ratio = (np.median(d_zero[m]) / np.median(d_perm[m])
                     if abs(np.median(d_perm[m])) > 1e-6 else np.nan)
        ax.set_title(f'Per-cell: Δperm vs Δzero\n'
                     f'median ratio zero/perm = {med_ratio:.2f}')
    ax.set_xlabel('ΔBPS (permuted − intact)')
    ax.set_ylabel('ΔBPS (zeroed − intact)')
    ax.set_aspect('equal')

    fig.suptitle(f"{name} ({r['subject']}, NC={r['n_cells']})", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / f"within_model_{name}.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=300)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'  saved {out}')
    show_or_close(fig)


for r in results:
    plot_session(r)


# %% Cross-session summary ----------------------------------------------------
def plot_summary(results):
    if not results:
        print("No results to summarize.")
        return

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))

    def _stack_delta(metric_key):
        """Stack per-cell Δ(cond − intact) arrays across sessions."""
        outs = {'perm': [], 'zero': [], 'vis': []}
        for r in results:
            mdict = r.get(metric_key, {})
            if not mdict:
                continue
            base = mdict.get('beh_intact')
            if base is None:
                continue
            dp = mdict['beh_permuted'] - base
            dz = mdict['beh_zeroed']   - base
            dv = mdict['vis']          - base
            mm = np.isfinite(dp) & np.isfinite(dz) & np.isfinite(dv)
            outs['perm'].append(dp[mm])
            outs['zero'].append(dz[mm])
            outs['vis'].append(dv[mm])
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in outs.items()}

    bps_d  = _stack_delta('bps')
    ccn_d  = _stack_delta('cc_norm')
    ccs_d  = _stack_delta('cc_single')

    def _pooled_scatter(ax, x, y, color, x_label, y_label, c_label,
                        c_vmin, c_vmax, title):
        if not len(x):
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes)
            return
        sc = ax.scatter(x, y, c=np.clip(color, c_vmin, c_vmax),
                        cmap='RdBu_r', s=8, alpha=0.65,
                        vmin=c_vmin, vmax=c_vmax)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=c_label)
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.5)
        med_y = np.median(y); med_x = np.median(x)
        if abs(med_x) > 1e-6:
            ratio = med_y / med_x
            sub = f'  (median y/x = {ratio:.2f})'
        else:
            sub = ''
        ax.set_title(title + sub)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect('equal')

    def _per_session_box(ax, metric_key, ylabel, title, delta=False):
        """Box-plot of per-session medians per condition.

        delta=True  → values are (cond − intact); intact is dropped from x-axis.
        delta=False → absolute per-condition median.
        """
        conds = ([c for c in CONDITIONS if c != 'beh_intact']
                 if delta else CONDITIONS)
        box_data, box_colors = [], []
        for cond in conds:
            meds = []
            for r in results:
                d = r.get(metric_key, {}).get(cond)
                if d is None:
                    continue
                if delta:
                    base = r.get(metric_key, {}).get('beh_intact')
                    if base is None:
                        continue
                    d = d - base
                d = d[np.isfinite(d)]
                if len(d):
                    meds.append(np.nanmedian(d))
            if meds:
                box_data.append(np.array(meds))
                box_colors.append(COND_COLORS[cond])
        if box_data:
            labels = [c.replace('beh_', '').replace('_', '\n') for c in conds] \
                if delta else conds
            bp = ax.boxplot(box_data, labels=labels, showfliers=True,
                            patch_artist=True)
            for patch, col in zip(bp['boxes'], box_colors):
                patch.set_facecolor(col); patch.set_alpha(0.6)
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

    def _row_three_scatters(row_axes, d, x_unit, label_prefix, c_ranges):
        """Render Δperm-vs-Δzero, Δperm-vs-Δvis, Δzero-vs-Δvis given d dict."""
        # col 1: perm vs zero, color=vis
        _pooled_scatter(
            row_axes[1], d['perm'], d['zero'], d['vis'],
            x_label=f'Δ{label_prefix} (permuted − intact)',
            y_label=f'Δ{label_prefix} (zeroed − intact)',
            c_label=f'Δ{label_prefix} vis', c_vmin=c_ranges['vis'][0],
            c_vmax=c_ranges['vis'][1],
            title=f'Δ{label_prefix}: perm vs zero',
        )
        # col 2: perm vs vis, color=zero
        _pooled_scatter(
            row_axes[2], d['perm'], d['vis'], d['zero'],
            x_label=f'Δ{label_prefix} (permuted − intact)',
            y_label=f'Δ{label_prefix} (vision-only − intact)',
            c_label=f'Δ{label_prefix} zero', c_vmin=c_ranges['zero'][0],
            c_vmax=c_ranges['zero'][1],
            title=f'Δ{label_prefix}: perm vs vis',
        )
        # col 3: zero vs vis, color=perm
        _pooled_scatter(
            row_axes[3], d['zero'], d['vis'], d['perm'],
            x_label=f'Δ{label_prefix} (zeroed − intact)',
            y_label=f'Δ{label_prefix} (vision-only − intact)',
            c_label=f'Δ{label_prefix} perm', c_vmin=c_ranges['perm'][0],
            c_vmax=c_ranges['perm'][1],
            title=f'Δ{label_prefix}: zero vs vis',
        )

    # ----- ROW 0: ΔBPS -----
    _per_session_box(axes[0, 0], 'bps', delta=True,
                     ylabel='Median ΔBPS per session',
                     title=f'Per-session median ΔBPS (n={len(results)})')
    _row_three_scatters(
        axes[0], bps_d, x_unit='BPS', label_prefix='BPS',
        c_ranges={'vis': (-0.15, 0.0), 'zero': (-0.05, 0.02),
                  'perm': (-0.10, 0.02)},
    )

    # ----- ROW 1: ΔCCnorm (PSTH-prediction quality) -----
    _per_session_box(axes[1, 0], 'cc_norm',
                     ylabel='Median CCnorm per session',
                     title='Per-session median CCnorm (PSTH prediction)')
    _row_three_scatters(
        axes[1], ccn_d, x_unit='CCnorm', label_prefix='CCnorm',
        c_ranges={'vis': (-0.30, 0.0), 'zero': (-0.10, 0.02),
                  'perm': (-0.20, 0.02)},
    )

    # ----- ROW 2: ΔCC_single (single-trial residual) -----
    _per_session_box(axes[2, 0], 'cc_single',
                     ylabel='Median CC_single per session',
                     title='Per-session median CC_single (residual-from-PSTH)')
    _row_three_scatters(
        axes[2], ccs_d, x_unit='CC_single', label_prefix='CC_single',
        c_ranges={'vis': (-0.10, 0.0), 'zero': (-0.05, 0.02),
                  'perm': (-0.08, 0.02)},
    )

    fig.suptitle(
        'Within-Model-B perturbation — vision-only ≠ behavior-zeroed: '
        'use full model + permute/zero behavior, NOT cross-model comparison',
        y=1.005,
    )
    fig.tight_layout()
    out = FIG_DIR / "within_model_summary.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=300)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'\nsaved summary {out}')
    show_or_close(fig)


plot_summary(results)


# %% Print cross-session table ----------------------------------------------
print("\n=== Cross-session ΔBPS summary (median across cells, per session) ===")
others = [c for c in CONDITIONS if c != 'beh_intact']
header = f"{'session':<22}{'subj':<8}{'n':<5}{'BPS_intact':<13}"
for cond in others:
    header += f"{'Δ' + cond[:8]:<14}"
print(header)
for r in results:
    bps_intact = r['bps']['beh_intact']
    intact_med = np.nanmedian(bps_intact)
    line = f"{r['session']:<22}{r['subject']:<8}{r['n_cells']:<5}{intact_med:<13.3f}"
    for cond in others:
        d = r['bps'][cond] - bps_intact
        d = d[np.isfinite(d)]
        line += f"{np.median(d):<+14.3f}"
    print(line)

print("\n=== Empirical Σ_FEM comparison (trace ratio, top-k overlap) ===")
header = f"{'session':<22}"
for cond in CONDITIONS:
    header += f"{'tr_' + cond[:6]:<11}"
print(header)
for r in results:
    line = f"{r['session']:<22}"
    for cond in CONDITIONS:
        m = r['empirical']['metrics'].get(cond)
        if m is None:
            line += f"{'NA':<11}"
        else:
            line += f"{m['tr_ratio']:<11.3f}"
    print(line)

print("\nDone.")
