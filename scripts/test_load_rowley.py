#%%

import sys
from pathlib import Path
sys.path.append('./scripts')
import numpy as np
#%%
from DataYatesV1 import enable_autoreload, get_free_device
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

#enable_autoreload()
device = get_free_device()

from mcfarland_sim import run_mcfarland_on_dataset, extract_metrics, DualWindowAnalysis
from utils import get_model_and_dataset_configs

#%%

from models.config_loader import load_dataset_configs

dataset_configs_path = "/home/declan/VisionCore/experiments/dataset_configs/single_basic_120_long_rowley.yaml"
    
dataset_configs = load_dataset_configs(dataset_configs_path)

print(dataset_configs)


# %%

from models.data import prepare_data

train_data, val_data, dataset_config = prepare_data(dataset_configs[0], strict=False)

#%%
#%%#%%
dataset = train_data.shallow_copy()

# Select the fixrsvp sub-dataset by name (fallback to index 2)
def pick_dset_by_name(ds_combined, wanted=(
    'fixrsvp',
)):
    candidates = []
    for i, ds in enumerate(ds_combined.dsets):
        meta = getattr(ds, 'metadata', {})
        label = None
        for key in ('name', 'dataset', 'type', 'stim_type', 'task', 'source'):
            try:
                val = meta.get(key, None)
            except Exception:
                val = None
            if isinstance(val, str):
                label = val
                break
        if label is None:
            label = f"unknown_{i}"
        candidates.append((i, label))
    wanted_lower = tuple(w.lower() for w in wanted)
    for i, label in candidates:
        if isinstance(label, str) and any(w in label.lower() for w in wanted_lower):
            return i
    return None

fix_idx = pick_dset_by_name(dataset)
if fix_idx is None:
    fix_idx = 2  # based on printed list: [2] is fixrsvp
dset = dataset.dsets[fix_idx]
print(f"Selected sub-dataset index={fix_idx} ({getattr(dset, 'metadata', {}).get('name', 'unknown')})")

# Bind arrays from fixrsvp
robs = dset['robs'].numpy()
eyepos = dset['eyepos'].numpy()
dfs = dset['dfs'].numpy()
cids = np.array(dset.metadata.get('cids', np.arange(robs.shape[1])))
stim = dset['stim'] if 'stim' in getattr(dset, 'covariates', {}) or 'stim' in dset else None

# Try to locate an image-ID-like covariate (optional)
image_key = None
image_ids = None
if hasattr(dset, 'covariates'):
    candidate_keys = (
        'image_id', 'image_ids', 'img_id', 'img_ids',
        'stim_id', 'stim_ids', 'frame_id', 'frame_ids',
        'image_index', 'stim_index'
    )
    for k in candidate_keys:
        try:
            if k in dset.covariates:
                image_key = k
                break
        except Exception:
            pass
if image_key:
    try:
        image_ids = dset[image_key].numpy()
        print(f"Found image ID key: {image_key} (shape={image_ids.shape})")
    except Exception:
        image_ids = None


#%% check for identical trials (same robs and eyepos)
import hashlib

trial_inds = dset['trial_inds'].numpy()
trials = np.unique(trial_inds)

def _trial_signature(tr_id):
    ix = (trial_inds == tr_id)
    hasher = hashlib.blake2b()
    # Include robs, eyepos, dfs
    for arr in (robs[ix], eyepos[ix], dfs[ix]):
        arr_c = np.ascontiguousarray(arr)
        hasher.update(arr_c.dtype.str.encode())
        hasher.update(np.array(arr_c.shape, dtype=np.int64).tobytes())
        hasher.update(arr_c.tobytes())
    # Include time bins if available
    try:
        t_bins = dset['t_bins'].numpy()
        tb = np.ascontiguousarray(t_bins[ix])
        hasher.update(tb.dtype.str.encode())
        hasher.update(np.array(tb.shape, dtype=np.int64).tobytes())
        hasher.update(tb.tobytes())
    except Exception:
        pass
    return hasher.hexdigest()

sig_to_trials = {}
for tr in trials:
    sig = _trial_signature(tr)
    sig_to_trials.setdefault(sig, []).append(int(tr))

duplicate_groups = [grp for grp in sig_to_trials.values() if len(grp) > 1]

if duplicate_groups:
    print(f"Identical-trial check: {sum(len(g)-1 for g in duplicate_groups)} duplicates found in {len(duplicate_groups)} groups.")
    # Show a few groups for quick inspection
    for grp in duplicate_groups[:10]:
        print("Duplicate group:", grp)
    # Optional: keep first occurrence of each signature as unique
    unique_trial_ids = np.array([grp[0] for grp in sig_to_trials.values()], dtype=trials.dtype)
    duplicate_trial_ids = np.setdiff1d(trials, unique_trial_ids)
else:
    print("Identical-trial check: no duplicate trials found.")
    unique_trial_ids = trials
    duplicate_trial_ids = np.array([], dtype=trials.dtype)

# Sanity: show a frame from the first fixrsvp trial
trial_inds = dset['trial_inds'].numpy()
first_fix_trial = np.unique(trial_inds)[0]
t0 = np.nonzero(trial_inds == first_fix_trial)[0][0]
if stim is not None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(stim[t0, 0].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
    plt.title(f'fixrsvp trial {int(first_fix_trial)} frame {int(t0)}')
    plt.axis('off')
    plt.show()

# (Plotting of sequences is done after function definition below)


#%% Label images by content using the longest fixrsvp trial, then compare timing across trials
import numpy as np
import hashlib

if stim is None:
    print("No 'stim' found in dataset; cannot label image sequence.")
else:
    trial_inds = dset['trial_inds'].numpy()
    trials = np.unique(trial_inds)

    # Use fixation + any valid dfs bins
    fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1
    any_dfs = dfs.any(axis=1) if dfs.ndim == 2 else np.ones_like(fixation, dtype=bool)
    bin_mask = fixation & any_dfs

    # Per-trial valid indices
    trial_bins = {int(tr): np.nonzero((trial_inds == tr) & bin_mask)[0] for tr in trials}
    # Pick longest trial by number of valid bins
    longest_tr = max(trial_bins.keys(), key=lambda tr: trial_bins[tr].size)
    idx_long = trial_bins[longest_tr]
    print(f"Longest trial: {longest_tr} with {idx_long.size} valid bins")

    def frame_hash(img_np):
        arr_c = np.ascontiguousarray(img_np)
        h = hashlib.blake2b()
        h.update(arr_c.dtype.str.encode())
        h.update(np.array(arr_c.shape, dtype=np.int64).tobytes())
        h.update(arr_c.tobytes())
        return h.hexdigest()

    # Build labels from longest trial in order of appearance
    hash_to_label = {}
    label_to_image = []  # store copies of unique images
    labels_longest = []

    for t in idx_long:
        img_np = stim[t, 0].detach().cpu().numpy()  # shape (251, 251)
        h = frame_hash(img_np)
        if h not in hash_to_label:
            hash_to_label[h] = len(label_to_image)
            label_to_image.append(img_np.copy())
        labels_longest.append(hash_to_label[h])
    labels_longest = np.array(labels_longest, dtype=int)
    print(f"Unique images discovered in longest trial: {len(label_to_image)}")

    # Label sequences for all trials using the same mapping (adding new labels if unseen)
    seq_labels = {}
    for tr in trials:
        ix = trial_bins[int(tr)]
        labs = []
        for t in ix:
            img_np = stim[t, 0].detach().cpu().numpy()
            h = frame_hash(img_np)
            if h not in hash_to_label:
                hash_to_label[h] = len(label_to_image)
                label_to_image.append(img_np.copy())
            labs.append(hash_to_label[h])
        seq_labels[int(tr)] = np.array(labs, dtype=int)

    # Compare image-sequence timing via change points (where image label changes)
    dt = 1.0 / 120.0  # sampling after prepare_data
    def change_times(labs):
        if labs.size == 0:
            return np.array([], dtype=float)
        cp = np.concatenate(([0], np.nonzero(labs[1:] != labs[:-1])[0] + 1))
        return cp.astype(float) * dt  # seconds since trial start

    times_long = change_times(labels_longest)
    print(f"Longest trial change points (s): {np.round(times_long, 4)}")

    mismatches = []
    for tr in trials:
        times_tr = change_times(seq_labels[int(tr)])
        # Compare length and values within tolerance
        if times_tr.size != times_long.size or not np.allclose(times_tr, times_long, atol=1e-6):
            mismatches.append(int(tr))
    if len(mismatches) == 0:
        print("Image sequence timing is consistent across trials (change points match).")
    else:
        print(f"Image sequence timing differs in {len(mismatches)} trials. Examples: {mismatches[:10]}")

    # Optional: show first few label sequences for inspection
    for tr in trials[:5]:
        print(f"Trial {int(tr)} labels (first 30): {seq_labels[int(tr)][:30]}")

#%%
#%% Debug: plot image sequences for trials (no video, just grids)
import numpy as np
import matplotlib.pyplot as plt

def plot_trial_sequences(trial_ids, max_frames=60, cols=12, use_mask=True, save=False, save_dir=Path('./figures')):
    if stim is None:
        print("No 'stim' found; cannot plot.")
        return
    trial_inds_local = dset['trial_inds'].numpy()
    if use_mask:
        fixation_local = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1
        any_dfs_local = dfs.any(axis=1) if dfs.ndim == 2 else np.ones_like(fixation_local, dtype=bool)
        bin_mask_local = fixation_local & any_dfs_local
    else:
        bin_mask_local = np.ones_like(trial_inds_local, dtype=bool)

    save_dir.mkdir(parents=True, exist_ok=True)

    for tr in trial_ids:
        idx = np.nonzero((trial_inds_local == tr) & bin_mask_local)[0]
        if idx.size == 0:
            print(f"Trial {int(tr)}: no frames to plot.")
            continue

        # Grab frames and normalize contrast consistently across the selection
        frames = [stim[t, 0].detach().cpu().numpy() for t in idx]
        # Limit to max_frames
        frames = frames[:max_frames]
        global_min = min(float(np.min(f)) for f in frames)
        global_max = max(float(np.max(f)) for f in frames)

        # Paging
        cols = max(1, int(cols))
        rows = int(np.ceil(len(frames) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes = np.atleast_2d(axes)

        for i, f in enumerate(frames):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.imshow(f, cmap='gray', vmin=global_min, vmax=global_max, interpolation='nearest')
            ax.set_title(f"t={idx[i]}")
            ax.axis('off')

        # Hide unused axes
        for j in range(len(frames), rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].axis('off')

        fig.suptitle(f"Trial {int(tr)} image sequence (first {len(frames)} frames)")
        fig.tight_layout()
        if save:
            out = save_dir / f"trial_{int(tr)}_sequence.png"
            fig.savefig(out, dpi=150)
            print(f"Wrote {out}")
            plt.close(fig)
        else:
            plt.show()

# Example: plot a few mismatched trials
example_trials = mismatches[:3] if len(mismatches) else []
plot_trial_sequences(example_trials, max_frames=48, cols=12, use_mask=True, save=False)


#%% Optional: Check that the RSVP image-ID sequence is consistent across trials
if image_ids is not None:
    # Use fixation + any valid dfs as the bin selector, similar to downstream usage
    fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1
    any_dfs = dfs.any(axis=1) if dfs.ndim == 2 else np.ones_like(fixation, dtype=bool)

    # Build a reduced per-trial sequence (remove consecutive duplicates)
    def seq_signature(tr_id):
        ix = (trial_inds == tr_id) & fixation & any_dfs
        if ix.sum() == 0:
            return None
        seq = image_ids[ix].reshape(-1)
        # Collapse consecutive repeats to compare content order
        keep = np.concatenate(([True], seq[1:] != seq[:-1]))
        seq = seq[keep]
        # Hash the reduced sequence
        h = hashlib.blake2b()
        arr_c = np.ascontiguousarray(seq)
        h.update(arr_c.dtype.str.encode())
        h.update(np.array(arr_c.shape, dtype=np.int64).tobytes())
        h.update(arr_c.tobytes())
        return h.hexdigest()

    seq_map = {}
    missing_seq_trials = []
    for tr in trials:
        s = seq_signature(tr)
        if s is None:
            missing_seq_trials.append(int(tr))
            continue
        seq_map.setdefault(s, []).append(int(tr))

    if len(seq_map) == 0:
        print("RSVP image sequence check: no sequences extracted.")
    elif len(seq_map) == 1 and len(missing_seq_trials) == 0:
        only_key = next(iter(seq_map))
        print(f"RSVP image sequence check: consistent across all trials. N={len(seq_map[only_key])}")
    else:
        print(f"RSVP image sequence check: found {len(seq_map)} distinct sequences.")
        for i, (k, grp) in enumerate(seq_map.items()):
            if i < 5:
                print("Sequence group:", grp)
        if missing_seq_trials:
            print("Trials with no valid sequence:", missing_seq_trials)

#%% === Build trials, good_trials, neuron_mask, valid_mask, dt =================

trial_inds = dset['trial_inds'].numpy()
trials = np.unique(trial_inds)
NC = robs.shape[1]
NT = len(trials)

# trial-aligned arrays
trial_lengths = [np.sum(trial_inds == tr) for tr in trials]
max_T = int(max(trial_lengths))

robs_trial = np.full((NT, max_T, NC), np.nan, dtype=np.float32)
eyepos_trial = np.full((NT, max_T, 2), np.nan, dtype=np.float32)
dfs_trial = np.full((NT, max_T, NC), np.nan, dtype=np.float32)
fix_dur = np.zeros(NT, dtype=int)

# simple fixation definition (center < 1 deg)
fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1

for i, tr in enumerate(trials):
    ix = (trial_inds == tr) & fixation
    T_trial = ix.sum()
    if T_trial == 0:
        continue
    robs_trial[i, :T_trial] = robs[ix]
    eyepos_trial[i, :T_trial] = eyepos[ix]
    dfs_trial[i, :T_trial] = dfs[ix]
    fix_dur[i] = T_trial

# McFarland-style trial and neuron selection
windows_ms = [5, 10, 20, 40, 80]
total_spikes_threshold = 200

good_trials = fix_dur > 20  # require at least 20 fixation+dfs bins
robs_mc = robs_trial[good_trials]
eyepos_mc = eyepos_trial[good_trials]
dfs_mc = dfs_trial[good_trials]
fix_dur_mc = fix_dur[good_trials]  # Use a new variable name for masked fix_dur

print(f"McFarland prep: {robs_mc.shape[0]} trials, {robs_mc.shape[2]} neurons, {robs_mc.shape[1]} time bins")

# sampling after prepare_data: 120 Hz → dt = 1/120
dt = 1 / 120.0
valid_time_bins = min(240, robs_mc.shape[1])

# neuron mask: intersect spike-count QC with high-ccmax cids (if available)

# try to infer session name from metadata to locate high-ccmax file
sess_name = None
meta = getattr(dset, 'metadata', None)
sess_obj = None
if hasattr(meta, 'get'):
    try:
        sess_obj = meta.get('sess', None)
    except Exception:
        sess_obj = None
if sess_obj is not None and hasattr(sess_obj, 'name'):
    sess_name = sess_obj.name
if sess_name is None:
    # fallback for rowley Luke session
    sess_name = 'Luke_2025-08-04'

figures_dir = Path('../figures')
high_cc_path = figures_dir / f"{sess_name}_high_ccmax_cids.npy"

if high_cc_path.exists():
    high_cc_cids = np.load(high_cc_path)
    high_cc_mask = np.isin(cids, high_cc_cids)
    print(f"Loaded high-ccmax cids from {high_cc_path} (N={high_cc_cids.size})")
else:
    high_cc_mask = np.ones_like(cids, dtype=bool)
    print(f"High-ccmax cids file not found at {high_cc_path}; using spike-count mask only.")

# high_cc_mask = np.ones_like(cids, dtype=bool)
# print(f"Skipping high-ccmax cids file at {high_cc_path}; using spike-count mask only.")

spike_ok = np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold
combined_mask_bool = spike_ok & high_cc_mask
neuron_mask = np.where(combined_mask_bool)[0]
print(f"Using {len(neuron_mask)} neurons / {robs_mc.shape[2]} total (spikes>{total_spikes_threshold} & high-ccmax)")

# valid mask: dfs + finite spikes and eye positions
# dfs is (T, NC) in [0,1] or {0,1}; treat >0.5 as "valid"
dfs_valid = np.nanmean(dfs_mc[:, :, neuron_mask], axis=2) > 0.5

valid_mask = (
    dfs_valid &
    np.isfinite(np.sum(robs_mc[:, :, neuron_mask], axis=2)) &
    np.isfinite(np.sum(eyepos_mc, axis=2))
)


# time index window
iix = np.arange(valid_time_bins)
robs_used = robs_mc[:, iix][:, :, neuron_mask]
eyepos_used = eyepos_mc[:, iix]
valid_used = valid_mask[:, iix]

print(f"robs_used shape: {robs_used.shape}")
print(f"eyepos_used shape: {eyepos_used.shape}")
print(f"valid_used shape: {valid_used.shape}")

analyzer_luke = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt)
print("DualWindowAnalysis initialized for Luke_2025-08-04 fixrsvp")

# %% Run McFarland sweep on Luke fixrsvp
windows_ms = [5, 10, 20, 40, 80]
results_luke, last_mats_luke = analyzer_luke.run_sweep(windows_ms, t_hist_ms=50, n_bins=15)

#%% Save all neuron pair figures into a single multi-page PDF
figures_dir = Path('../figures')
figures_dir.mkdir(exist_ok=True)
pdf_path = figures_dir / f"mcfarland_neuron_pairs_fixrsvp_{sess_name}.pdf"

with PdfPages(pdf_path) as pdf:
    info = pdf.infodict()
    info['Title'] = f"McFarland Neuron Pairs - {sess_name}"
    info['Subject'] = 'FixRSVP DualWindowAnalysis inspect_neuron_pair'
    info['Creator'] = 'VisionCore test_load_rowley.py'
    for cell in range(len(neuron_mask)):
        fig, axs = analyzer_luke.inspect_neuron_pair(cell, cell, win_ms=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
print(f"Saved multi-page PDF to {pdf_path}")

#%% Save Luke McFarland stats in a figure_fixrsvp-compatible format
"""Package and save Luke_2025-08-04 McFarland stats.

We mimic the `output` dict produced by
figure_fixrsvp_mcfarland_covariance_binned.run_mcfarland_on_dataset so that
figure scripts can later do:

    import pickle
    with open(path, 'rb') as f:
        output_luke = pickle.load(f)
    outputs = [output_luke]

and reuse the existing plotting code with minimal changes.
"""

import os
import pickle

subject = 'Luke'
date = '2025-08-04'

# Try to get cids from metadata if present, otherwise fall back to a simple index
try:
    cids_luke = np.array(dset.metadata.get('cids', np.arange(dset['robs'].shape[1])))
except Exception:
    cids_luke = np.arange(dset['robs'].shape[1])

output_luke = {
    'sess': f'{subject}_{date}',
    'cids': cids_luke,
    'neuron_mask': neuron_mask,
    'windows': windows_ms,
    'cids_used': cids_luke[neuron_mask],
    'results': results_luke,
    'last_mats': last_mats_luke,
}

save_dir = Path('../figures')
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f'mcfarland_fixrsvp_Luke_{date}.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(output_luke, f)

print(f"Saved Luke McFarland stats to {save_path}")

#%% Example: look at one window (e.g-----------===============================================. 20 ms)
i = windows_ms.index(20)

FF_uncorr = results_luke[i]['ff_uncorr']
FF_corr   = results_luke[i]['ff_corr']
Erates    = results_luke[i]['Erates']

print("Window:", results_luke[i]['window_ms'], "ms")
print("Mean FF uncorr:", FF_uncorr.mean())
print("Mean FF corr:",   FF_corr.mean())

CnoiseU = last_mats_luke[i]['NoiseCorrU']
CnoiseC = last_mats_luke[i]['NoiseCorrC']

plt.figure()
plt.imshow(CnoiseU, vmin=-0.2, vmax=0.2); plt.title('Noise corr (uncorr)'); plt.colorbar()

plt.figure()
plt.imshow(CnoiseC, vmin=-0.2, vmax=0.2); plt.title('Noise corr (FEM-corrected)'); plt.colorbar()
plt.show()


# Example visualization using masked arrays
good_trials_40 = fix_dur_mc > 40
robs_40 = robs_mc[good_trials_40]
eyepos_40 = eyepos_mc[good_trials_40]
fix_dur_40 = fix_dur_mc[good_trials_40]

ind = np.argsort(fix_dur_40)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos_40[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs_40,2)[ind])
plt.xlim(0, 60)




#%%


# #%%
# cc = 0
# ind = np.argsort(fix_dur)

# NC = robs.shape[-1]
# sx = int(np.sqrt(NC))
# sy = int(np.ceil(NC / sx))
# fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
# for cc in range(NC):
#     ax = axs.flatten()[cc]
#     ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')
#     ax.set_title(f'Cell {cc}')
#     ax.axis('off')

# #%%
# fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
# for cc in range(NC):
#     ax = axs.flatten()[cc]
#     ax.plot(np.nanmean(robs[:,:120,cc],0), 'k')
#     ax.set_title(f'Cell {cc}')
#     ax.axis('off')


# #%%


# # Initialize itrial before use
# itrial = 0
# itrial += 1
# if itrial >= len(robs):
#     itrial = 0
# fig, axs = plt.subplots(2,1, figsize=(10,5), sharex=True, sharey=False)
# axs[1].plot(eyepos[itrial][:,:])
# xd = axs[1].get_xlim()
# axs[0].imshow(robs[itrial][:,:].T>0, aspect='auto', cmap='gray_r', interpolation='none')
# axs[0].set_xlim(xd)
# axs[1].set_ylim(-1, 1)


# # # %%

# #%% 

# # Initialize cc before use
# cc = 0
# cc += 1
# fig, ax = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=False)
# ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')

# The following block referenced train_data.dsets[0] and t_bins; it is disabled
# as the selected dset is fixrsvp and time bins may not be present.
# If needed, re-enable with appropriate keys from `dset`.
# try:
#     fun = dset.metadata['sess'].get_missing_pct_interp(dset.metadata.get('cids', np.arange(robs.shape[1])))
#     # If t_bins exists:
#     t_bins = dset['t_bins'].numpy()
#     missing_pct = fun(t_bins)
#     _ = plt.plot(t_bins, missing_pct)
# except Exception:
#     pass

# threshold = 45

# mask = missing_pct < threshold
# mask[:,np.where(np.median(missing_pct, 0) < threshold)[0]] = True

# plt.imshow(mask[:1000], interpolation='none')


# # %%

# %%
