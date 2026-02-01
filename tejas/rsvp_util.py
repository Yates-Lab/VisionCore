#%%
import numpy as np
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
import contextlib
import os
from scripts.mcfarland_sim import get_fixrsvp_stack
import socket
from pathlib import Path
import pickle
HOSTNAME = socket.gethostname()
DATA_DIR = None
if HOSTNAME == "solo":
    DATA_DIR = Path("/mnt/ssd/YatesMarmoV1/")
elif HOSTNAME == "yoru":
    DATA_DIR = Path("/mnt/sata/YatesMarmoV1/")
#%%
def get_dataset_from_config(subject, date, dataset_configs_path):
    """
    Build a single dataset containing only fixrsvp trials from train and val splits.

    Loads the dataset config for the given session, calls prepare_data to get train/val
    datasets, then restricts to fixrsvp indices from both splits and returns one
    dataset object plus the config.

    Args:
        subject: Subject identifier (e.g. 'Allen')
        date: Session date string (e.g. '2022-03-04')
        dataset_configs_path: Path to YAML file listing dataset configs

    Returns:
        dataset: Shallow copy of train dataset with inds set to fixrsvp indices only
        dataset_config: Config dict for this session (e.g. 'cids', 'session')
    """
    assert f'{subject}_{date}' in [sess.name for sess in get_complete_sessions()], f"Session {subject}_{date} not found"

    # =========================================================================
    # Load config and locate this session
    # =========================================================================
    dataset_configs = load_dataset_configs(dataset_configs_path)
    dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

    # =========================================================================
    # Prepare train/val datasets (suppress prepare_data stdout/stderr)
    # =========================================================================
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)

    sess = train_dset.dsets[0].metadata['sess']
    cids = dataset_config['cids']

    # =========================================================================
    # Restrict to fixrsvp indices and build single dataset
    # =========================================================================
    # Concatenate fixrsvp indices from train and val so we have one unified index set.
    inds = torch.concatenate([
        train_dset.get_dataset_inds('fixrsvp'),
        val_dset.get_dataset_inds('fixrsvp')
    ], dim=0)

    dataset = train_dset.shallow_copy()
    dataset.inds = inds

    return dataset, dataset_config

def validate_image_ids(image_ids, dataset, dset_idx):
    # check for if image_ids is correct.
    # pick a trial

    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)
    sess = dataset.dsets[dset_idx].metadata['sess']
    ptb2ephys, _ = get_clock_functions(sess.exp)

    for i in range(len(trials)):
        trial_id = int(trials[i])
        trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
        start_idx = np.where(trial.image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])
        trial_bins = t_bins[trial_inds == trial_id]
        hist_idx = np.searchsorted(flip_times, trial_bins, side='right') - 1 + start_idx
        # This should be identical to the assigned row (before -1 shift)
        if not np.all(trial.image_ids[hist_idx] - 1 == image_ids[i][dataset.dsets[dset_idx].covariates['psth_inds'][trial_inds == trial_id]]):
            warnings.warn(f"Trial {trial_id} image_ids are not correct")


def remove_duplicate_trials(robs, dfs, eyepos, fix_dur, image_ids, 
                           spike_times_trials=None, trial_time_windows=None, trial_t_bins=None):
    """
    Remove duplicate trials based on robs and eyepos signatures.
    
    If spike_times_trials, trial_time_windows, trial_t_bins are provided,
    they will also be filtered to keep the same trials.
    """
    NT = len(robs)
    r_flat = np.nan_to_num(robs, nan=0.0).reshape(NT, -1)
    e_flat = np.nan_to_num(eyepos, nan=0.0).reshape(NT, -1)
    sig = np.concatenate([r_flat, e_flat], axis=1)

    _, keep = np.unique(sig, axis=0, return_index=True)
    keep = np.sort(keep)

    robs = robs[keep]
    dfs = dfs[keep]
    eyepos = eyepos[keep]
    fix_dur = fix_dur[keep]
    image_ids = image_ids[keep]
    
    # Also filter spike times data if provided
    if spike_times_trials is not None:
        spike_times_trials = [spike_times_trials[i] for i in keep]
    if trial_time_windows is not None:
        trial_time_windows = [trial_time_windows[i] for i in keep]
    if trial_t_bins is not None:
        trial_t_bins = [trial_t_bins[i] for i in keep]
    
    NT = len(keep)
    #search for duplicate trials
    for itrial in range(NT):
        for jtrial in range(itrial+1, NT):
            if np.allclose(robs[itrial], robs[jtrial], equal_nan=True):
                raise ValueError(f"Duplicate trial found {itrial} and {jtrial}")
    
    if spike_times_trials is not None:
        return robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins
    return robs, dfs, eyepos, fix_dur, image_ids

def remove_below_fixation_threshold_trials(robs, dfs, eyepos, fix_dur, image_ids, fixation_duration_bins_threshold,
                                           spike_times_trials=None, trial_time_windows=None, trial_t_bins=None):
    """
    Remove trials with fixation duration below threshold.
    
    If spike_times_trials, trial_time_windows, trial_t_bins are provided,
    they will also be filtered to keep the same trials.
    """
    good_trials = fix_dur > fixation_duration_bins_threshold
    robs = robs[good_trials]
    dfs = dfs[good_trials]
    eyepos = eyepos[good_trials]    
    fix_dur = fix_dur[good_trials]
    image_ids = image_ids[good_trials]
    
    # Also filter spike times data if provided
    if spike_times_trials is not None:
        keep_indices = np.where(good_trials)[0]
        spike_times_trials = [spike_times_trials[i] for i in keep_indices]
    if trial_time_windows is not None:
        keep_indices = np.where(good_trials)[0]
        trial_time_windows = [trial_time_windows[i] for i in keep_indices]
    if trial_t_bins is not None:
        keep_indices = np.where(good_trials)[0]
        trial_t_bins = [trial_t_bins[i] for i in keep_indices]

    if spike_times_trials is not None:
        return robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins
    return robs, dfs, eyepos, fix_dur, image_ids

def collate_fixrsvp_data(dataset, dset_idx, fixation_degree_radius):
    sess = dataset.dsets[dset_idx].metadata['sess']
    
    
    # Getting key variables
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)

    NC = dataset.dsets[dset_idx]['robs'].shape[1]
    T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
    NT = len(trials)

    fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < fixation_degree_radius

    ptb2ephys, _ = get_clock_functions(sess.exp)

    image_ids = np.full((NT, T), -1, dtype=np.int64)
    # Loop over trials and align responses
    robs = np.nan*np.zeros((NT, T, NC))
    dfs = np.nan*np.zeros((NT, T, NC))
    eyepos = np.nan*np.zeros((NT, T, 2))
    fix_dur =np.nan*np.zeros((NT,))

    for itrial in tqdm(range(NT)):
        # print(f"Trial {itrial}/{NT}")
        trial_mask = trials[itrial] == trial_inds
        if np.sum(trial_mask) == 0:
            continue

        trial_id = int(trials[itrial])
        trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
        trial_image_ids = trial.image_ids
        if len(np.unique(trial_image_ids)) < 2:
            continue
        start_idx = np.where(trial_image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])

        psth_inds_all = dataset.dsets[dset_idx].covariates['psth_inds'][trial_mask].numpy()
        trial_bins_all = t_bins[trial_mask]
        hist_idx_all = np.searchsorted(flip_times, trial_bins_all, side='right') - 1 + start_idx
        image_ids[itrial][psth_inds_all] = trial_image_ids[hist_idx_all] - 1

        ix = trial_mask & fixation
        if np.sum(ix) == 0:
            continue

        stim_inds = np.where(ix)[0]
        # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
        psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
        fix_dur[itrial] = len(psth_inds)
        robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
        dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
        eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    
    dict_to_save = {
        'robs': robs,
        'dfs': dfs,
        'eyepos': eyepos,
        'fix_dur': fix_dur,
        'image_ids': image_ids,
    }

    
    return robs, dfs, eyepos, fix_dur, image_ids


def _get_psth_inds_for_trial(trial_t_bins_trial, trial_time_windows_trial, dt=1/240):
    """
    Compute the psth_ind for each fixation bin center in a trial.
    
    Returns array of psth_inds corresponding to each entry in trial_t_bins_trial.
    """
    t_start, t_end = trial_time_windows_trial
    if np.isnan(t_start) or np.isnan(t_end) or len(trial_t_bins_trial) == 0:
        return np.array([])
    
    # Reconstruct bin edges and centers
    trial_bin_edges = np.arange(t_start, t_end + dt/2, dt)
    all_bin_centers = trial_bin_edges[:-1] + dt/2
    
    # Find psth_ind for each fixation center
    psth_inds = []
    for fc in trial_t_bins_trial:
        diffs = np.abs(all_bin_centers - fc)
        idx = np.argmin(diffs)
        if diffs[idx] < dt/10:
            psth_inds.append(idx)
        else:
            psth_inds.append(-1)  # Could not match
    return np.array(psth_inds)


def _filter_spike_times_by_valid_psth_inds(spike_times_trial, trial_t_bins_trial, psth_inds, valid_psth_mask, dt=1/240):
    """
    Filter spike times and trial_t_bins to keep only entries where valid_psth_mask is True.
    
    Parameters
    ----------
    spike_times_trial : list of np.ndarray
        spike_times_trial[cell_idx] = spike times for that cell
    trial_t_bins_trial : np.ndarray
        Fixation bin centers for this trial
    psth_inds : np.ndarray
        The psth_ind for each entry in trial_t_bins_trial
    valid_psth_mask : np.ndarray of bool
        Which entries to keep
    dt : float
        Time bin size
    
    Returns
    -------
    filtered_spike_times : list of np.ndarray
    filtered_t_bins : np.ndarray
    """
    if len(trial_t_bins_trial) == 0:
        return spike_times_trial, trial_t_bins_trial
    
    # Filter bin centers
    filtered_t_bins = trial_t_bins_trial[valid_psth_mask]
    
    # Filter spike times
    NC = len(spike_times_trial)
    filtered_spike_times = []
    
    for cell_idx in range(NC):
        cell_spikes = spike_times_trial[cell_idx]
        if len(cell_spikes) == 0 or np.sum(valid_psth_mask) == 0:
            filtered_spike_times.append(np.array([]))
            continue
        
        # Keep spikes that fall in valid bins
        keep_spikes_mask = np.zeros(len(cell_spikes), dtype=bool)
        for i, (is_valid, center) in enumerate(zip(valid_psth_mask, trial_t_bins_trial)):
            if is_valid:
                bin_start = center - dt/2
                bin_end = center + dt/2
                in_bin = (cell_spikes >= bin_start) & (cell_spikes < bin_end)
                keep_spikes_mask |= in_bin
        
        filtered_spike_times.append(cell_spikes[keep_spikes_mask])
    
    return filtered_spike_times, filtered_t_bins


def align_image_ids(robs, dfs, eyepos, fix_dur, image_ids, salvageable_mismatch_time_threshold=25, verbose=True,
                    spike_times_trials=None, trial_time_windows=None, trial_t_bins=None, dt=1/240):
    """
    Align image IDs across trials by truncating, shifting, or removing trials.
    
    If spike_times_trials, trial_time_windows, trial_t_bins are provided,
    they will also be modified to stay in sync with robs.
    """
    reference_trial_ind = None
    image_ids_reference = None
    for i in range(len(image_ids)):
        if (image_ids[i, :200] != -1).all():
            image_ids_reference = image_ids[i]
            reference_trial_ind = i
            break

    unmatched_trials_and_start_time_ind_of_mismatch = {}

   
    for trial_ind, row in enumerate(image_ids):
        start_time_ind_of_mismatch = None
        for time_ind in range(len(row)):
            trial_matches = True
            if row[time_ind] != -1 and image_ids_reference[time_ind] != -1:
                if image_ids_reference[time_ind] != row[time_ind]:
                    trial_matches = False
                    start_time_ind_of_mismatch = time_ind
                    
            if not trial_matches:
                if verbose:
                    print(f'trial {trial_ind} does not match')
                unmatched_trials_and_start_time_ind_of_mismatch[trial_ind] = start_time_ind_of_mismatch
                break
    
    assert len(unmatched_trials_and_start_time_ind_of_mismatch) < 5, f"More than 5 trials have mismatched image ids"
    trials_to_remove = []

    def find_shift_to_match(image_ids_reference, image_ids_trial):
        for shift in range(len(image_ids_reference)):
            image_ids_trial_shifted = image_ids_trial[shift:]
            if np.sum(image_ids_trial_shifted == -1) == len(image_ids_trial_shifted):
                return None
            negative_start_index = np.where(image_ids_trial_shifted == -1)[0][0]
            image_ids_trial_shifted = image_ids_trial_shifted[:negative_start_index]
            if np.all(image_ids_reference[:len(image_ids_trial_shifted)] == image_ids_trial_shifted):
                return shift
        return None

    for trial_ind, start_time_ind_of_mismatch in unmatched_trials_and_start_time_ind_of_mismatch.items():
        first_trial_ind = reference_trial_ind
        second_trial_ind = trial_ind
        mismatched_image_ids = image_ids[second_trial_ind].copy()
        
        shift = find_shift_to_match(image_ids_reference, image_ids[trial_ind])
        if start_time_ind_of_mismatch > salvageable_mismatch_time_threshold:
            # TRUNCATION: Set bins after start_time_ind_of_mismatch to NaN
            robs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            eyepos[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            dfs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            image_ids[trial_ind, start_time_ind_of_mismatch:] = -1
            
            # Handle spike times: keep only spikes in bins with psth_ind < start_time_ind_of_mismatch
            if spike_times_trials is not None and len(trial_t_bins[trial_ind]) > 0:
                psth_inds = _get_psth_inds_for_trial(trial_t_bins[trial_ind], trial_time_windows[trial_ind], dt)
                valid_mask = psth_inds < start_time_ind_of_mismatch
                spike_times_trials[trial_ind], trial_t_bins[trial_ind] = _filter_spike_times_by_valid_psth_inds(
                    spike_times_trials[trial_ind], trial_t_bins[trial_ind], psth_inds, valid_mask, dt
                )
                # Update fix_dur to reflect actual number of remaining fixation bins
                fix_dur[trial_ind] = len(trial_t_bins[trial_ind])
            else:
                # Count non-NaN bins in robs as fallback
                fix_dur[trial_ind] = np.sum(~np.isnan(robs[trial_ind, :, 0]))
                    
        elif shift is not None:
            # SHIFTING: Remove first `shift` bins, shift everything else
            assert shift < 100
            if verbose: print(f'shift to match for trial {trial_ind} is {shift}')
            robs[trial_ind, :-shift, :] = robs[trial_ind, shift:, :]
            robs[trial_ind, -shift:, :] = np.nan
            eyepos[trial_ind, :-shift, :] = eyepos[trial_ind, shift:, :]
            eyepos[trial_ind, -shift:, :] = np.nan
            dfs[trial_ind, :-shift, :] = dfs[trial_ind, shift:, :]
            dfs[trial_ind, -shift:, :] = np.nan
            image_ids[trial_ind, :-shift] = image_ids[trial_ind, shift:]
            image_ids[trial_ind, -shift:] = -1
            
            # Handle spike times: remove spikes in bins with ORIGINAL psth_ind < shift
            if spike_times_trials is not None and len(trial_t_bins[trial_ind]) > 0:
                psth_inds = _get_psth_inds_for_trial(trial_t_bins[trial_ind], trial_time_windows[trial_ind], dt)
                valid_mask = psth_inds >= shift  # Keep bins with original psth_ind >= shift
                spike_times_trials[trial_ind], trial_t_bins[trial_ind] = _filter_spike_times_by_valid_psth_inds(
                    spike_times_trials[trial_ind], trial_t_bins[trial_ind], psth_inds, valid_mask, dt
                )
                # Update fix_dur to reflect actual number of remaining fixation bins
                fix_dur[trial_ind] = len(trial_t_bins[trial_ind])
            else:
                # Count non-NaN bins in robs as fallback
                fix_dur[trial_ind] = np.sum(~np.isnan(robs[trial_ind, :, 0]))
            
        else:
            trials_to_remove.append(trial_ind)
        
        if verbose:
            plt.plot(image_ids[first_trial_ind], label=f'Trial {first_trial_ind}')
            plt.plot(mismatched_image_ids, label=f'Trial {second_trial_ind} mismatched')
            if trial_ind not in trials_to_remove:
                plt.plot(image_ids[trial_ind], label=f'Trial {trial_ind} corrected', alpha=0.5, linestyle='--')
            plt.xlim(0, 200)
            plt.xlabel('Time (bins)')
            plt.ylabel('Image ID')
            plt.title(f'Image IDs for trial {first_trial_ind} and {second_trial_ind}')
            # plt.legend([f'Trial {first_trial_ind}', f'Trial {second_trial_ind}'])
            plt.legend()
            plt.show()
            print(f'start time ind of mismatch for trial {trial_ind} is {start_time_ind_of_mismatch}')

    # Remove trials
    keep_mask = ~np.isin(np.arange(len(robs)), trials_to_remove)
    robs = robs[keep_mask]
    eyepos = eyepos[keep_mask]
    fix_dur = fix_dur[keep_mask]
    dfs = dfs[keep_mask]
    image_ids = image_ids[keep_mask]
    
    if spike_times_trials is not None:
        keep_indices = np.where(keep_mask)[0]
        spike_times_trials = [spike_times_trials[i] for i in keep_indices]
        trial_time_windows = [trial_time_windows[i] for i in keep_indices]
        trial_t_bins = [trial_t_bins[i] for i in keep_indices]

    for trial_ind, row in enumerate(image_ids):
        for time_ind in range(len(row)):
            if row[time_ind] != -1 and image_ids_reference[time_ind] != -1:
                if image_ids_reference[time_ind] != row[time_ind]:
                    raise ValueError(f'trial {trial_ind} does not match at time {time_ind}')

    if spike_times_trials is not None:
        return robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins
    return robs, dfs, eyepos, fix_dur, image_ids


def compare_spike_times_to_robs(spike_times_trials, trial_time_windows, trial_t_bins, robs, dt=1/240, verbose=True):
    """
    Validate that binning spike_times_trials reproduces the original robs.
    
    This function bins the spike times from extract_spike_times_per_trial using
    the same logic as generate_fixrsvp_dataset/bin_spikes and compares the 
    counts to the original robs.
    
    Parameters
    ----------
    spike_times_trials : list of lists
        spike_times_trials[itrial][cell_idx] = np.array of spike times
    trial_time_windows : list of tuples
        trial_time_windows[itrial] = (t_start, t_end) in seconds (original bin edges)
    trial_t_bins : list of np.ndarray
        trial_t_bins[itrial] = time bin centers for fixation bins
    robs : np.ndarray (NT, T, NC)
        Original spike counts from collate_fixrsvp_data
    dt : float
        Time bin size in seconds
    verbose : bool
        If True, print mismatch details
    
    Returns
    -------
    all_match : bool
        True if all binned counts match robs
    mismatches : list of tuples
        List of (trial, cell, bin_idx, expected, got) for any mismatches
    """
    NT = len(spike_times_trials)
    if NT == 0:
        return True, []
    
    NC = len(spike_times_trials[0])
    
    # Quick sanity check: total spike counts per trial/cell
    total_count_mismatches = []
    for trial_ind in range(NT):
        for cell_ind in range(NC):
            expected_total = np.nansum(robs[trial_ind, :, cell_ind])
            got_total = len(spike_times_trials[trial_ind][cell_ind])
            if expected_total != got_total:
                total_count_mismatches.append((trial_ind, cell_ind, expected_total, got_total))
                if verbose:
                    print(f'trial {trial_ind} cell {cell_ind} has {expected_total} spikes in robs but {got_total} in spike_times_trials')
    
    if len(total_count_mismatches) > 0:
        if verbose:
            print(f"\nTotal count check FAILED: {len(total_count_mismatches)} trial/cell pairs have mismatched total counts")
            print("Skipping detailed bin-by-bin comparison.\n")
        return False, [('total_count_mismatch', m) for m in total_count_mismatches]
    
    if verbose:
        print(f"Total count check passed for all {NT * NC} trial/cell pairs")
    
    mismatches = []
    total_comparisons = 0
    
    for itrial in range(NT):
        t_start, t_end = trial_time_windows[itrial]
        if np.isnan(t_start) or np.isnan(t_end):
            continue
        
        fixation_centers = trial_t_bins[itrial]
        if len(fixation_centers) == 0:
            continue
        
        # Reconstruct bin edges from trial time window
        # Use t_end + dt/2 to ensure the last edge is included (handles floating point)
        trial_bin_edges = np.arange(t_start, t_end + dt/2, dt)
        n_bins = len(trial_bin_edges) - 1
        
        if n_bins <= 0:
            continue
        
        # Compute all bin centers to find fixation bin indices
        all_bin_centers = trial_bin_edges[:-1] + dt/2
        
        # Find which bin indices correspond to fixation centers
        # Use tolerance for floating point comparison
        fixation_bin_indices = []
        for fc in fixation_centers:
            diffs = np.abs(all_bin_centers - fc)
            idx = np.argmin(diffs)
            if diffs[idx] < dt/10:  # tolerance
                fixation_bin_indices.append(idx)
            else:
                # Couldn't match this fixation center - this shouldn't happen
                if verbose:
                    print(f"Warning: trial {itrial} couldn't match fixation center {fc}")
        fixation_bin_indices = np.array(fixation_bin_indices)
        
        if len(fixation_bin_indices) == 0:
            continue
        
        for cell_idx in range(NC):
            cell_spike_times = spike_times_trials[itrial][cell_idx]
            
            # Bin the spike times using digitize (same logic as bin_spikes)
            if len(cell_spike_times) > 0:
                spike_bin_indices = np.digitize(cell_spike_times, trial_bin_edges) - 1
                spike_bin_indices = np.clip(spike_bin_indices, 0, n_bins - 1)
                
                # Count spikes per bin
                binned_counts = np.bincount(spike_bin_indices, minlength=n_bins)
            else:
                binned_counts = np.zeros(n_bins, dtype=int)
            
            # Get binned counts for fixation bins only
            fixation_binned_counts = binned_counts[fixation_bin_indices]
            
            # Get the corresponding robs values (non-NaN values in order)
            robs_trial_cell = robs[itrial, :, cell_idx]
            non_nan_mask = ~np.isnan(robs_trial_cell)
            robs_values = robs_trial_cell[non_nan_mask]
            
            # Check length match
            if len(robs_values) != len(fixation_binned_counts):
                mismatches.append((itrial, cell_idx, 'length_mismatch', 
                                   len(robs_values), len(fixation_binned_counts)))
                if verbose:
                    print(f"Trial {itrial} cell {cell_idx}: length mismatch - "
                          f"robs has {len(robs_values)} bins, extracted has {len(fixation_binned_counts)}")
                continue
            
            # Bin-by-bin comparison
            total_comparisons += len(robs_values)
            for bin_idx, (expected, got) in enumerate(zip(robs_values, fixation_binned_counts)):
                if expected != got:
                    mismatches.append((itrial, cell_idx, bin_idx, expected, got))
                    if verbose:
                        print(f"Trial {itrial} cell {cell_idx} bin {bin_idx}: "
                              f"expected {expected}, got {got}")
    
    all_match = len(mismatches) == 0
    
    if verbose:
        if all_match:
            print(f"All {total_comparisons} bin comparisons match!")
        else:
            print(f"{len(mismatches)} mismatches out of {total_comparisons} comparisons")
    
    return all_match, mismatches


def extract_spike_times_per_trial(dataset, dset_idx, cids, fixation_degree_radius, dt=1/240):
    """
    Extract spike times for each trial, aligned with the trial structure used in robs.
    
    This function mirrors the logic from bin_spikes but returns actual spike times
    instead of binned counts. Spike times are organized to match the robs structure.
    
    Parameters
    ----------
    dataset : DictDataset
        The dataset containing trial information
    dset_idx : int
        Index of the dataset to use
    cids : np.ndarray
        Cluster IDs in the order they appear in robs columns
    fixation_degree_radius : float
        Radius in degrees for fixation detection
    dt : float
        Time bin size in seconds (default 1/240)
    
    Returns
    -------
    spike_times_trials : list of lists
        spike_times_trials[itrial][cell_idx] = np.array of spike times in seconds
        Shape matches robs: (NT, NC) where each element is a variable-length array
    trial_time_windows : list of tuples
        trial_time_windows[itrial] = (t_start, t_end) in seconds
    trial_t_bins : list of np.ndarray
        trial_t_bins[itrial] = time bin centers for that trial (fixation bins only)
    """

    sess = dataset.dsets[dset_idx].metadata['sess']

    ptb2ephys, _ = get_clock_functions(sess.exp)
   
    # Getting key variables
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)

    NC = dataset.dsets[dset_idx]['robs'].shape[1]
    NT = len(trials)

    fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < fixation_degree_radius

    # Get raw spike data
    spike_times = sess.ks_results.spike_times
    spike_clusters = sess.ks_results.spike_clusters
    
    # Get psth_inds from dataset
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'].numpy()
    
    # Create cluster ID to column index mapping (same as bin_spikes)
    cids = np.asarray(cids)
    n_cids = len(cids)
    cids2inds = np.zeros(np.max(cids) + 1, dtype=int)
    cids2inds[cids] = np.arange(n_cids)
    
    # Ensure spike times are sorted (same as bin_spikes)
    if not np.all(np.diff(spike_times) >= 0):
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]
    
    # Filter spikes to only include clusters in cids
    cids_mask = np.isin(spike_clusters, cids)
    spike_times_filtered = spike_times[cids_mask]
    spike_clusters_filtered = spike_clusters[cids_mask]
    
    # Map cluster IDs to column indices
    spike_inds = cids2inds[spike_clusters_filtered]
    
    NT = len(trials)
    NC = len(cids)
    
    # Initialize output structures
    spike_times_trials = [[np.array([]) for _ in range(NC)] for _ in range(NT)]
    trial_time_windows = [(np.nan, np.nan) for _ in range(NT)]
    trial_t_bins = [np.array([]) for _ in range(NT)]
    
    # Loop over trials
    for itrial in tqdm(range(NT), desc="Extracting spike times"):
        # Find data points for this trial
        trial_mask = trials[itrial] == trial_inds
        if np.sum(trial_mask) == 0:
            continue
        
        # Get trial info and compute original bin edges (same as generate_fixrsvp_dataset)
        # This avoids floating-point errors from reconstructing edges from centers
        trial_id = int(trials[itrial])
        trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
        trial_image_ids = trial.image_ids
        if len(np.unique(trial_image_ids)) < 2:
            continue
        start_idx = np.where(trial_image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])
        
        # Compute bin edges exactly as in generate_fixrsvp_dataset
        trial_bin_edges = np.arange(flip_times[0], flip_times[-1], dt)
        if len(trial_bin_edges) < 2:
            continue
        
        # Get fixation mask for this trial
        ix = trial_mask & fixation
        if np.sum(ix) == 0:
            continue
        
        # Get fixation bin indices and centers
        trial_psth_inds = psth_inds[ix]
        trial_t_bins_centers = t_bins[ix]
        
        if len(trial_t_bins_centers) == 0:
            continue
        
        # Store trial t_bins (fixation centers)
        trial_t_bins[itrial] = trial_t_bins_centers
        
        # Use original bin edges for time window (avoids floating-point reconstruction errors)
        t_start = trial_bin_edges[0]
        t_end = trial_bin_edges[-1]
        trial_time_windows[itrial] = (t_start, t_end)
        
        # Extract all spikes in trial time window (same logic as bin_spikes)
        i0 = np.searchsorted(spike_times_filtered, t_start)
        i1 = np.searchsorted(spike_times_filtered, t_end)
        
        if i0 >= i1:
            # No spikes in this window
            continue
        
        # Get spikes in this time window
        trial_spike_times = spike_times_filtered[i0:i1]
        trial_spike_inds = spike_inds[i0:i1]
        
        # Determine which bin each spike belongs to (same logic as bin_spikes digitize)
        # digitize returns i such that bins[i-1] <= x < bins[i], so subtract 1 to get 0-indexed bin
        spike_bin_indices = np.digitize(trial_spike_times, trial_bin_edges) - 1
        # Clip to valid range [0, n_bins-1] for safety
        n_bins = len(trial_bin_edges) - 1
        spike_bin_indices = np.clip(spike_bin_indices, 0, n_bins - 1)
        
        # Keep only spikes that fall into fixation bins
        fixation_spike_mask = np.isin(spike_bin_indices, trial_psth_inds)
        trial_spike_times = trial_spike_times[fixation_spike_mask]
        trial_spike_inds = trial_spike_inds[fixation_spike_mask]
        
        # Organize spikes by cell index
        for cell_idx in range(NC):
            cell_mask = trial_spike_inds == cell_idx
            cell_spike_times = trial_spike_times[cell_mask]
            spike_times_trials[itrial][cell_idx] = cell_spike_times
    
    return spike_times_trials, trial_time_windows, trial_t_bins

def get_fixrsvp_data(subject, date, dataset_configs_path,
                    use_cached_data=False,
                    fixation_degree_radius=1,
                    fixation_duration_bins_threshold=20,
                    salvageable_mismatch_time_threshold=25,
                    verbose=False):
    dataset, dataset_config = get_dataset_from_config(subject, date, dataset_configs_path)
    dset_idx = dataset.inds[:,0].unique().item()

    processed_data_path = DATA_DIR / 'processed' / dataset.dsets[dset_idx].metadata['sess'].name / 'datasets'
    assert processed_data_path.exists(), f"Processed data path {processed_data_path} does not exist"
    cached_file = processed_data_path / f'fixrsvp_data_collated_{Path(dataset_configs_path).stem}.pkl'
    if use_cached_data and os.path.exists(cached_file):
        with open(cached_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"Loaded cached data from {cached_file}")

        robs = data_dict['robs']
        dfs = data_dict['dfs']
        eyepos = data_dict['eyepos']
        fix_dur = data_dict['fix_dur']
        image_ids = data_dict['image_ids']
        spike_times_trials = data_dict['spike_times_trials']
        trial_time_windows = data_dict['trial_time_windows']
        trial_t_bins = data_dict['trial_t_bins']
        rsvp_images = data_dict['rsvp_images']
    else:

        robs, dfs, eyepos, fix_dur, image_ids = collate_fixrsvp_data(dataset, dset_idx, fixation_degree_radius)
        validate_image_ids(image_ids, dataset, dset_idx)

        spike_times_trials, trial_time_windows, trial_t_bins = extract_spike_times_per_trial(
        dataset, dset_idx, dataset_config['cids'], fixation_degree_radius, dt=1/240)
        # Process with spike times (pass all 8 values, get all 8 back)
        rsvp_images = get_fixrsvp_stack(frames_per_im=1)

    dict_to_save = {
        'robs': robs,
        'dfs': dfs,
        'eyepos': eyepos,
        'fix_dur': fix_dur,
        'image_ids': image_ids,
        'cids': dataset_config['cids'],
        'rsvp_images': rsvp_images,
        'spike_times_trials': spike_times_trials,
        'trial_time_windows': trial_time_windows,
        'trial_t_bins': trial_t_bins,
    }
    with open(cached_file, 'wb') as f:
        pickle.dump(dict_to_save, f)
    

    all_match, mismatches = compare_spike_times_to_robs(
        spike_times_trials, trial_time_windows, trial_t_bins, robs, dt=1/240, verbose=verbose
    )
    assert all_match, f"Found {len(mismatches)} mismatches first pass"
    
    robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins = \
    remove_duplicate_trials(robs, dfs, eyepos, fix_dur, image_ids, 
                            spike_times_trials, trial_time_windows, trial_t_bins)

    robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins = \
    remove_below_fixation_threshold_trials(robs, dfs, eyepos, fix_dur, image_ids, fixation_duration_bins_threshold,
                                          spike_times_trials, trial_time_windows, trial_t_bins)
    robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials, trial_time_windows, trial_t_bins = \
    align_image_ids(robs, dfs, eyepos, fix_dur, image_ids, 
                   salvageable_mismatch_time_threshold=salvageable_mismatch_time_threshold, verbose=verbose,
                   spike_times_trials=spike_times_trials, 
                   trial_time_windows=trial_time_windows, 
                   trial_t_bins=trial_t_bins, dt=1/240)
    # Validate after all processing
    all_match, mismatches = compare_spike_times_to_robs(
        spike_times_trials, trial_time_windows, trial_t_bins, robs, dt=1/240, verbose=verbose
    )
    assert all_match, f"Found {len(mismatches)} mismatches"   

    return {
        'robs': robs,
        'dfs': dfs,
        'eyepos': eyepos,
        'fix_dur': fix_dur,
        'image_ids': image_ids,
        'cids': dataset_config['cids'],
        'rsvp_images': rsvp_images,
        'spike_times_trials': spike_times_trials,
        'trial_time_windows': trial_time_windows,
        'trial_t_bins': trial_t_bins,
    }
#%%
# subject = 'Allen'
# date = '2022-03-30'
# dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp_all_cells.yaml'

# data = get_fixrsvp_data(subject, date, dataset_configs_path, 
# use_cached_data=True, 
# salvageable_mismatch_time_threshold=25, verbose=False)
# #%%
# data['fix_dur']