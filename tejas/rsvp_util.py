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
def get_dataset_from_config(subject, date, dataset_configs_path, dataset_type='fixrsvp'):
    """
    Build a single dataset containing only fixrsvp trials from train and val splits.

    Loads the dataset config for the given session, calls prepare_data to get train/val
    datasets, then restricts to fixrsvp indices from both splits and returns one
    dataset object plus the config.

    Args:
        subject (str): Subject identifier (e.g. 'Allen', 'Ellie')
        date (str): Session date string in YYYY-MM-DD format (e.g. '2022-03-04')
        dataset_configs_path (str): Path to YAML file listing dataset configs
        dataset_type (str): Type of dataset to extract indices for (default: 'fixrsvp')

    Returns:
        dataset (DictDataset): Shallow copy of train dataset with inds set to 
            dataset_type indices only. Shape of inds: (N, 2) where N is total 
            number of indices from both train and val splits
        dataset_config (dict): Config dict for this session containing keys like
            'cids' (list of cluster IDs), 'session' (str), etc.
    
    Raises:
        AssertionError: If session {subject}_{date} is not in complete sessions list
        ValueError: If config not found for the session or dataset cannot be prepared
    """
    assert f'{subject}_{date}' in [sess.name for sess in get_complete_sessions()], f"Session {subject}_{date} not found"

    # =========================================================================
    # Load config and locate this session
    # =========================================================================
    dataset_configs = load_dataset_configs(dataset_configs_path)
    try:
        dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")
    except Exception as e:
        raise ValueError(f"config not found for {subject}_{date}")
    # =========================================================================
    # Prepare train/val datasets (suppress prepare_data stdout/stderr)
    # =========================================================================
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        try:
            train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)
        except Exception as e:
            raise ValueError(f"{dataset_type} dataset not found for {subject}_{date}")
    sess = train_dset.dsets[0].metadata['sess']
    cids = dataset_config['cids']

    # =========================================================================
    # Restrict to fixrsvp indices and build single dataset
    # =========================================================================
    # Concatenate fixrsvp indices from train and val so we have one unified index set.
    inds = torch.concatenate([
        train_dset.get_dataset_inds(dataset_type),
        val_dset.get_dataset_inds(dataset_type)
    ], dim=0)

    dataset = train_dset.shallow_copy()
    dataset.inds = inds

    return dataset, dataset_config

def validate_image_ids(image_ids, dataset, dset_idx):
    """
    Validate that extracted image_ids match the ground truth from trial data.

    For each trial, reconstructs image IDs by mapping time bins to flip times
    and compares against the provided image_ids array. Issues a warning if
    any trial has mismatched image IDs.

    Args:
        image_ids (np.ndarray): Image IDs array to validate, shape (NT, T) where
            NT is number of trials and T is number of time bins. Values are 
            0-indexed image IDs or -1 for invalid/missing bins.
        dataset (DictDataset): Dataset object containing trial covariates and metadata
        dset_idx (int): Index of the dataset within dataset.dsets to use

    Raises:
        Warning: If any trial's image_ids don't match the ground truth from
            the FixRsvpTrial object
    """
    # =========================================================================
    # Extract trial metadata and timing functions
    # =========================================================================
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)
    sess = dataset.dsets[dset_idx].metadata['sess']
    ptb2ephys, _ = get_clock_functions(sess.exp)

    # =========================================================================
    # Validate each trial by reconstructing image IDs from flip times
    # =========================================================================
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

    Creates a signature by concatenating flattened robs and eyepos arrays for each
    trial, then removes trials with duplicate signatures. This handles cases where
    the same trial data appears multiple times in the dataset.

    Args:
        robs (np.ndarray): Spike count responses, shape (NT, T, NC) where NT is 
            number of trials, T is number of time bins, NC is number of cells.
            NaN values indicate invalid bins.
        dfs (np.ndarray): Data flags/validity mask, shape (NT, T, NC)
        eyepos (np.ndarray): Eye position data, shape (NT, T, 2) with x,y coordinates
            in degrees of visual angle
        fix_dur (np.ndarray): Fixation duration per trial in bins, shape (NT,)
        image_ids (np.ndarray): Image IDs per time bin, shape (NT, T). Values are
            0-indexed image IDs or -1 for invalid bins.
        spike_times_trials (list of list of np.ndarray, optional): Raw spike times.
            spike_times_trials[trial][cell] = array of spike times in seconds
        trial_time_windows (list of tuple, optional): Time windows for each trial.
            trial_time_windows[trial] = (t_start, t_end) in seconds
        trial_t_bins (list of np.ndarray, optional): Time bin centers for each trial.
            trial_t_bins[trial] = array of shape (T,) with bin centers or NaN

    Returns:
        If spike_times_trials is None:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids) with duplicates removed
        If spike_times_trials is provided:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials,
                   trial_time_windows, trial_t_bins) with duplicates removed

    Raises:
        ValueError: If duplicate trials are still found after removal (sanity check)
    """
    # =========================================================================
    # Create unique signature for each trial from robs + eyepos
    # =========================================================================
    NT = len(robs)
    r_flat = np.nan_to_num(robs, nan=0.0).reshape(NT, -1)
    e_flat = np.nan_to_num(eyepos, nan=0.0).reshape(NT, -1)
    sig = np.concatenate([r_flat, e_flat], axis=1)

    # Find unique trials and get indices to keep
    _, keep = np.unique(sig, axis=0, return_index=True)
    keep = np.sort(keep)

    # =========================================================================
    # Filter all arrays to keep only unique trials
    # =========================================================================
    robs = robs[keep]
    dfs = dfs[keep]
    eyepos = eyepos[keep]
    fix_dur = fix_dur[keep]
    image_ids = image_ids[keep]
    
    if spike_times_trials is not None:
        spike_times_trials = [spike_times_trials[i] for i in keep]
    if trial_time_windows is not None:
        trial_time_windows = [trial_time_windows[i] for i in keep]
    if trial_t_bins is not None:
        trial_t_bins = [trial_t_bins[i] for i in keep]
    
    # =========================================================================
    # Sanity check: verify no duplicates remain
    # =========================================================================
    NT = len(keep)
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
    Remove trials with fixation duration below a minimum threshold.

    Filters out trials where the animal did not maintain fixation for a sufficient
    number of time bins, ensuring only well-fixated trials are analyzed.

    Args:
        robs (np.ndarray): Spike count responses, shape (NT, T, NC) where NT is 
            number of trials, T is number of time bins, NC is number of cells
        dfs (np.ndarray): Data flags/validity mask, shape (NT, T, NC)
        eyepos (np.ndarray): Eye position data, shape (NT, T, 2) in degrees
        fix_dur (np.ndarray): Fixation duration per trial in bins, shape (NT,)
        image_ids (np.ndarray): Image IDs per time bin, shape (NT, T)
        fixation_duration_bins_threshold (int): Minimum number of fixation bins 
            required to keep a trial. Trials with fix_dur <= threshold are removed.
        spike_times_trials (list of list of np.ndarray, optional): Raw spike times.
            spike_times_trials[trial][cell] = array of spike times in seconds
        trial_time_windows (list of tuple, optional): Time windows per trial.
            trial_time_windows[trial] = (t_start, t_end) in seconds
        trial_t_bins (list of np.ndarray, optional): Time bin centers per trial.
            trial_t_bins[trial] = array of shape (T,) with bin centers or NaN

    Returns:
        If spike_times_trials is None:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids) with short trials removed
        If spike_times_trials is provided:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials,
                   trial_time_windows, trial_t_bins) with short trials removed
    """
    # =========================================================================
    # Filter trials based on fixation duration threshold
    # =========================================================================
    good_trials = fix_dur > fixation_duration_bins_threshold
    robs = robs[good_trials]
    dfs = dfs[good_trials]
    eyepos = eyepos[good_trials]    
    fix_dur = fix_dur[good_trials]
    image_ids = image_ids[good_trials]
    
    # Filter spike times data if provided
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
    """
    Extract and organize fixation RSVP trial data from a dataset into trial-aligned arrays.

    Loops over all trials in the dataset, extracts neural responses (robs), data flags (dfs),
    eye positions, and image IDs for time bins where the animal was fixating within the
    specified radius. Data is organized into (NT, T, ...) arrays aligned by PSTH indices.

    Args:
        dataset (DictDataset): Dataset object containing neural data and covariates.
            Must have dsets[dset_idx] with 'robs', 'dfs', 'eyepos' tensors and
            'trial_inds', 't_bins', 'psth_inds' covariates.
        dset_idx (int): Index of the dataset within dataset.dsets to use
        fixation_degree_radius (float): Maximum distance from center (in degrees of 
            visual angle) for a time bin to be considered fixating. Eye positions
            beyond this radius are excluded.

    Returns:
        robs (np.ndarray): Spike counts, shape (NT, T, NC) where NT is number of trials,
            T is max PSTH index + 1, NC is number of cells. NaN for non-fixation bins.
        dfs (np.ndarray): Data flags, shape (NT, T, NC). NaN for non-fixation bins.
        eyepos (np.ndarray): Eye position in degrees, shape (NT, T, 2). NaN for 
            non-fixation bins.
        fix_dur (np.ndarray): Number of valid fixation bins per trial, shape (NT,).
            NaN if trial has no valid fixation bins.
        image_ids (np.ndarray): Image ID shown at each time bin, shape (NT, T).
            Values are 0-indexed (original IDs minus 1), -1 for invalid bins.
    """
    sess = dataset.dsets[dset_idx].metadata['sess']
    
    # =========================================================================
    # Extract dataset dimensions and compute fixation mask
    # =========================================================================
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)

    NC = dataset.dsets[dset_idx]['robs'].shape[1]
    T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
    NT = len(trials)

    # Compute fixation mask: True where eye position is within radius of center
    fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < fixation_degree_radius

    ptb2ephys, _ = get_clock_functions(sess.exp)

    # =========================================================================
    # Initialize output arrays (NaN = no data for that bin)
    # =========================================================================
    image_ids = np.full((NT, T), -1, dtype=np.int64)
    robs = np.nan*np.zeros((NT, T, NC))
    dfs = np.nan*np.zeros((NT, T, NC))
    eyepos = np.nan*np.zeros((NT, T, 2))
    fix_dur = np.nan*np.zeros((NT,))

    # =========================================================================
    # Loop over trials and extract data aligned by PSTH indices
    # =========================================================================
    for itrial in tqdm(range(NT)):
        trial_mask = trials[itrial] == trial_inds
        if np.sum(trial_mask) == 0:
            continue

        # Load trial info and find stimulus onset (image_id == 2)
        trial_id = int(trials[itrial])
        trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
        trial_image_ids = trial.image_ids
        if len(np.unique(trial_image_ids)) < 2:
            continue
        start_idx = np.where(trial_image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])

        # Map time bins to image IDs using flip times
        psth_inds_all = dataset.dsets[dset_idx].covariates['psth_inds'][trial_mask].numpy()
        trial_bins_all = t_bins[trial_mask]
        hist_idx_all = np.searchsorted(flip_times, trial_bins_all, side='right') - 1 + start_idx
        image_ids[itrial][psth_inds_all] = trial_image_ids[hist_idx_all] - 1

        # Extract data only for fixation bins
        ix = trial_mask & fixation
        if np.sum(ix) == 0:
            continue

        stim_inds = np.where(ix)[0]
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
    Get the PSTH indices (bin indices) for fixation bins in a trial.

    With sparse indexing, trial_t_bins_trial[i] holds the time center for bin i,
    or NaN if that bin is not a fixation bin. This function returns the indices
    where valid (non-NaN) time centers exist.

    Args:
        trial_t_bins_trial (np.ndarray): Sparse array of time bin centers, shape (T,).
            trial_t_bins_trial[i] = time center for bin i in seconds, or NaN if 
            not a fixation bin.
        trial_time_windows_trial (tuple): Time window (t_start, t_end) in seconds.
            Currently unused but kept for API consistency.
        dt (float): Time bin size in seconds (default: 1/240 ~ 4.17ms)

    Returns:
        np.ndarray: Array of integer indices where trial_t_bins_trial has valid
            (non-NaN) time centers. Empty array if no valid bins exist.
    """
    if len(trial_t_bins_trial) == 0:
        return np.array([], dtype=int)
    
    # With sparse indexing, position i IS psth_ind i
    # Return indices where we have valid (non-NaN) time centers
    valid_mask = ~np.isnan(trial_t_bins_trial)
    return np.where(valid_mask)[0]


def _filter_spike_times_by_valid_psth_inds(spike_times_trial, trial_t_bins_trial, psth_inds, valid_psth_mask, dt=1/240):
    """
    Filter spike times and invalidate trial_t_bins entries based on valid_psth_mask.

    Used during image ID alignment to remove spikes from bins that need to be
    truncated or shifted. With sparse indexing, this function keeps only spikes
    that fall within the valid time bins specified by the mask.

    Args:
        spike_times_trial (list of np.ndarray): Spike times for each cell in this trial.
            spike_times_trial[cell_idx] = 1D array of spike times in seconds.
            Length of list is NC (number of cells).
        trial_t_bins_trial (np.ndarray): Sparse array of time bin centers, shape (T,).
            trial_t_bins_trial[i] = time center for bin i in seconds, or NaN if
            not a fixation bin.
        psth_inds (np.ndarray): 1D array of indices where trial_t_bins_trial has
            valid (non-NaN) time centers.
        valid_psth_mask (np.ndarray): Boolean mask over psth_inds, shape (len(psth_inds),).
            True = keep this bin, False = invalidate this bin.
        dt (float): Time bin size in seconds (default: 1/240 ~ 4.17ms)

    Returns:
        filtered_spike_times (list of np.ndarray): Spike times with spikes in 
            invalid bins removed. Same structure as input.
        filtered_t_bins (np.ndarray): Copy of trial_t_bins_trial with entries
            corresponding to invalid psth_inds set to NaN. Shape (T,).
    """
    if len(trial_t_bins_trial) == 0 or len(psth_inds) == 0:
        return spike_times_trial, trial_t_bins_trial
    
    # =========================================================================
    # Invalidate time bin entries for bins marked as invalid
    # =========================================================================
    valid_psth_inds = psth_inds[valid_psth_mask]
    invalid_psth_inds = psth_inds[~valid_psth_mask]
    
    filtered_t_bins = trial_t_bins_trial.copy()
    if len(invalid_psth_inds) > 0:
        filtered_t_bins[invalid_psth_inds] = np.nan
    
    # =========================================================================
    # Filter spike times: keep only spikes falling in valid bins
    # =========================================================================
    NC = len(spike_times_trial)
    filtered_spike_times = []
    
    for cell_idx in range(NC):
        cell_spikes = spike_times_trial[cell_idx]
        if len(cell_spikes) == 0 or len(valid_psth_inds) == 0:
            filtered_spike_times.append(np.array([]))
            continue
        
        # Build mask of spikes to keep by checking each valid bin
        keep_spikes_mask = np.zeros(len(cell_spikes), dtype=bool)
        for psth_idx in valid_psth_inds:
            center = trial_t_bins_trial[psth_idx]  # Use original (not filtered) to get time
            if not np.isnan(center):
                bin_start = center - dt/2
                bin_end = center + dt/2
                in_bin = (cell_spikes >= bin_start) & (cell_spikes < bin_end)
                keep_spikes_mask |= in_bin
        
        filtered_spike_times.append(cell_spikes[keep_spikes_mask])
    
    return filtered_spike_times, filtered_t_bins

def get_image_ids_reference(image_ids, start_ind_trial=0, verbose=False):
    """
    Find a reference trial with complete image IDs and identify mismatched trials.

    Searches for a trial with valid (non -1) image IDs in the first 200 bins to use
    as a reference. Then compares all other trials against this reference to find
    trials with mismatched image ID sequences. If too many trials don't match,
    recursively tries with a later starting trial.

    Args:
        image_ids (np.ndarray): Image IDs array, shape (NT, T) where NT is number
            of trials and T is number of time bins. Values are 0-indexed image IDs
            or -1 for invalid/missing bins.
        start_ind_trial (int): Trial index to start searching for a reference trial
            (default: 0). Used for recursion when the first candidate has too many
            mismatches.
        verbose (bool): If True, print information about mismatched trials
            (default: False)

    Returns:
        reference_trial_ind (int): Index of the trial used as reference
        image_ids_reference (np.ndarray): Image IDs from the reference trial, shape (T,)
        unmatched_trials_and_start_time_ind_of_mismatch (dict): Dictionary mapping
            trial indices to the first time index where they mismatch the reference.
            Keys are trial indices (int), values are time indices (int).

    Notes:
        Recursively calls itself with start_ind_trial + 1 if >= 5 trials mismatch,
        to find a better reference trial.
    """
    # =========================================================================
    # Find a reference trial with valid image IDs in first 200 bins
    # =========================================================================
    for i in range(start_ind_trial, len(image_ids)):
        if (image_ids[i, :200] != -1).all():
            image_ids_reference = image_ids[i]
            reference_trial_ind = i
            break

    # =========================================================================
    # Compare all trials against reference to find mismatches
    # =========================================================================
    unmatched_trials_and_start_time_ind_of_mismatch = {}
   
    for trial_ind, row in enumerate(image_ids):
        start_time_ind_of_mismatch = None
        for time_ind in range(len(row)):
            trial_matches = True
            # Only compare where both trials have valid image IDs
            if row[time_ind] != -1 and image_ids_reference[time_ind] != -1:
                if image_ids_reference[time_ind] != row[time_ind]:
                    trial_matches = False
                    start_time_ind_of_mismatch = time_ind
                    
            if not trial_matches:
                if verbose:
                    print(f'trial {trial_ind} does not match')
                unmatched_trials_and_start_time_ind_of_mismatch[trial_ind] = start_time_ind_of_mismatch
                break

    # If too many mismatches, try a different reference trial
    if len(unmatched_trials_and_start_time_ind_of_mismatch) >= 5:
        return get_image_ids_reference(image_ids, start_ind_trial + 1, verbose)
    return reference_trial_ind, image_ids_reference, unmatched_trials_and_start_time_ind_of_mismatch
def align_image_ids(robs, dfs, eyepos, fix_dur, image_ids, salvageable_mismatch_time_threshold=25, verbose=True,
                    spike_times_trials=None, trial_time_windows=None, trial_t_bins=None, dt=1/240):
    """
    Align image IDs across trials by truncating, shifting, or removing mismatched trials.

    Some trials may have image ID sequences that don't match the reference sequence due
    to timing issues or dropped frames. This function handles three cases:
    1. TRUNCATION: If mismatch occurs late (after threshold), truncate data after mismatch
    2. SHIFTING: If trial is shifted by a few bins, shift data to align with reference
    3. REMOVAL: If trial cannot be salvaged, remove it entirely

    Args:
        robs (np.ndarray): Spike count responses, shape (NT, T, NC) where NT is 
            number of trials, T is number of time bins, NC is number of cells
        dfs (np.ndarray): Data flags/validity mask, shape (NT, T, NC)
        eyepos (np.ndarray): Eye position data, shape (NT, T, 2) in degrees
        fix_dur (np.ndarray): Fixation duration per trial in bins, shape (NT,)
        image_ids (np.ndarray): Image IDs per time bin, shape (NT, T). Values are
            0-indexed image IDs or -1 for invalid bins.
        salvageable_mismatch_time_threshold (int): Minimum time index for a mismatch
            to be salvageable via truncation (default: 25). If mismatch occurs before
            this index, the trial is shifted or removed instead.
        verbose (bool): If True, print details and plot comparisons for mismatched
            trials (default: True)
        spike_times_trials (list of list of np.ndarray, optional): Raw spike times.
            spike_times_trials[trial][cell] = array of spike times in seconds
        trial_time_windows (list of tuple, optional): Time windows per trial.
            trial_time_windows[trial] = (t_start, t_end) in seconds
        trial_t_bins (list of np.ndarray, optional): Time bin centers per trial.
            trial_t_bins[trial] = array of shape (T,) with bin centers or NaN
        dt (float): Time bin size in seconds (default: 1/240 ~ 4.17ms)

    Returns:
        If spike_times_trials is None:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids) with trials aligned
        If spike_times_trials is provided:
            tuple: (robs, dfs, eyepos, fix_dur, image_ids, spike_times_trials,
                   trial_time_windows, trial_t_bins) with trials aligned

    Raises:
        AssertionError: If >= 5 trials have mismatched image IDs (suggests systematic issue)
        ValueError: If any trial still has mismatched image IDs after alignment
    """
    # =========================================================================
    # Find reference trial and identify mismatched trials
    # =========================================================================
    reference_trial_ind, image_ids_reference, unmatched_trials_and_start_time_ind_of_mismatch = get_image_ids_reference(image_ids, verbose)
    assert len(unmatched_trials_and_start_time_ind_of_mismatch) < 5, f"{len(unmatched_trials_and_start_time_ind_of_mismatch)} trials have mismatched image ids, out of {len(image_ids)} trials"
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

    # =========================================================================
    # Process each mismatched trial: truncate, shift, or mark for removal
    # =========================================================================
    for trial_ind, start_time_ind_of_mismatch in unmatched_trials_and_start_time_ind_of_mismatch.items():
        first_trial_ind = reference_trial_ind
        second_trial_ind = trial_ind
        mismatched_image_ids = image_ids[second_trial_ind].copy()
        
        shift = find_shift_to_match(image_ids_reference, image_ids[trial_ind])
        
        # -----------------------------------------------------------------
        # Case 1: TRUNCATION - mismatch occurs late, truncate after mismatch
        # -----------------------------------------------------------------
        if start_time_ind_of_mismatch > salvageable_mismatch_time_threshold:
            robs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            eyepos[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            dfs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
            image_ids[trial_ind, start_time_ind_of_mismatch:] = -1
            
            # Handle spike times: keep only spikes in bins with psth_ind < start_time_ind_of_mismatch
            if spike_times_trials is not None and np.any(~np.isnan(trial_t_bins[trial_ind])):
                psth_inds = _get_psth_inds_for_trial(trial_t_bins[trial_ind], trial_time_windows[trial_ind], dt)
                valid_mask = psth_inds < start_time_ind_of_mismatch
                spike_times_trials[trial_ind], trial_t_bins[trial_ind] = _filter_spike_times_by_valid_psth_inds(
                    spike_times_trials[trial_ind], trial_t_bins[trial_ind], psth_inds, valid_mask, dt
                )
                # Also set trial_t_bins entries >= start_time_ind_of_mismatch to NaN (to match robs)
                trial_t_bins[trial_ind][start_time_ind_of_mismatch:] = np.nan
                
                # Update fix_dur to count non-NaN bins (sparse indexing)
                fix_dur[trial_ind] = np.sum(~np.isnan(trial_t_bins[trial_ind]))
            else:
                # Count non-NaN bins in robs as fallback
                fix_dur[trial_ind] = np.sum(~np.isnan(robs[trial_ind, :, 0]))
        
        # -----------------------------------------------------------------
        # Case 2: SHIFTING - trial is offset, shift data to align
        # -----------------------------------------------------------------
        elif shift is not None:
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
            if spike_times_trials is not None and np.any(~np.isnan(trial_t_bins[trial_ind])):
                psth_inds = _get_psth_inds_for_trial(trial_t_bins[trial_ind], trial_time_windows[trial_ind], dt)
                valid_mask = psth_inds >= shift  # Keep bins with original psth_ind >= shift
                spike_times_trials[trial_ind], trial_t_bins[trial_ind] = _filter_spike_times_by_valid_psth_inds(
                    spike_times_trials[trial_ind], trial_t_bins[trial_ind], psth_inds, valid_mask, dt
                )
                # CRITICAL: Also shift trial_t_bins to match robs shift
                # After this, position i in both robs and trial_t_bins corresponds to original psth_ind i+shift
                trial_t_bins[trial_ind][:-shift] = trial_t_bins[trial_ind][shift:].copy()
                trial_t_bins[trial_ind][-shift:] = np.nan
                
                # Update trial_time_windows to reflect the new start time
                old_t_start, old_t_end = trial_time_windows[trial_ind]
                new_t_start = old_t_start + shift * dt
                trial_time_windows[trial_ind] = (new_t_start, old_t_end)
                
                # Update fix_dur to count non-NaN bins (sparse indexing)
                fix_dur[trial_ind] = np.sum(~np.isnan(trial_t_bins[trial_ind]))
            else:
                # Count non-NaN bins in robs as fallback
                fix_dur[trial_ind] = np.sum(~np.isnan(robs[trial_ind, :, 0]))
        
        # -----------------------------------------------------------------
        # Case 3: REMOVAL - trial cannot be salvaged
        # -----------------------------------------------------------------
        else:
            trials_to_remove.append(trial_ind)
        
        # Plot comparison if verbose
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

    # =========================================================================
    # Remove unsalvageable trials from all arrays
    # =========================================================================
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

    # =========================================================================
    # Final validation: ensure all trials now match reference
    # =========================================================================
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
    counts to the original robs. Used to verify spike time extraction is correct.

    Args:
        spike_times_trials (list of list of np.ndarray): Raw spike times per trial/cell.
            spike_times_trials[itrial][cell_idx] = 1D array of spike times in seconds.
            Outer list has length NT (trials), inner lists have length NC (cells).
        trial_time_windows (list of tuple): Time windows for each trial.
            trial_time_windows[itrial] = (t_start, t_end) in seconds defining the
            original bin edges. NaN values indicate invalid trials.
        trial_t_bins (list of np.ndarray): Time bin centers for each trial.
            trial_t_bins[itrial] = array of shape (T,) with bin centers in seconds,
            or NaN for non-fixation bins (sparse indexing).
        robs (np.ndarray): Original spike counts from collate_fixrsvp_data,
            shape (NT, T, NC) where NT is trials, T is time bins, NC is cells.
            NaN values indicate non-fixation bins.
        dt (float): Time bin size in seconds (default: 1/240 ~ 4.17ms)
        verbose (bool): If True, print mismatch details and summary (default: True)

    Returns:
        all_match (bool): True if all binned counts match robs exactly
        mismatches (list of tuple): List of mismatches found. Each tuple contains:
            - For bin mismatches: (trial, cell, bin_idx, expected_count, got_count)
            - For length mismatches: (trial, cell, 'length_mismatch', expected_len, got_len)
            - For total count mismatches: ('total_count_mismatch', (trial, cell, expected, got))
    """
    NT = len(spike_times_trials)
    if NT == 0:
        return True, []
    
    NC = len(spike_times_trials[0])
    
    # =========================================================================
    # Quick check: verify total spike counts match per trial/cell
    # =========================================================================
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
    
    # =========================================================================
    # Detailed check: bin-by-bin comparison of spike counts
    # =========================================================================
    mismatches = []
    total_comparisons = 0
    
    for itrial in range(NT):
        t_start, t_end = trial_time_windows[itrial]
        if np.isnan(t_start) or np.isnan(t_end):
            continue
        
        # Get fixation bin indices from sparse trial_t_bins
        trial_t_bins_full = trial_t_bins[itrial]
        fixation_bin_indices = np.where(~np.isnan(trial_t_bins_full))[0]
        if len(fixation_bin_indices) == 0:
            continue
        
        fixation_centers = trial_t_bins_full[fixation_bin_indices]
        
        # Reconstruct bin edges from trial time window for spike binning
        trial_bin_edges = np.arange(t_start, t_end + dt/2, dt)
        n_bins = len(trial_bin_edges) - 1
        if n_bins <= 0:
            continue
        
        # Compare each cell's binned counts to robs
        for cell_idx in range(NC):
            cell_spike_times = spike_times_trials[itrial][cell_idx]
            
            # Bin spike times using same logic as bin_spikes
            if len(cell_spike_times) > 0:
                spike_bin_indices = np.digitize(cell_spike_times, trial_bin_edges) - 1
                spike_bin_indices = np.clip(spike_bin_indices, 0, n_bins - 1)
                binned_counts = np.bincount(spike_bin_indices, minlength=n_bins)
            else:
                binned_counts = np.zeros(n_bins, dtype=int)
            
            # Extract counts for fixation bins only
            fixation_binned_counts = binned_counts[fixation_bin_indices]
            
            # Get corresponding robs values (non-NaN entries)
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
    instead of binned counts. Spike times are organized to match the robs structure,
    enabling downstream analyses that require precise spike timing (e.g., jitter
    correction, spike-triggered analyses).

    Args:
        dataset (DictDataset): The dataset containing trial information. Must have
            dsets[dset_idx] with 'robs', 'eyepos' tensors and 'trial_inds', 't_bins',
            'psth_inds' covariates, plus metadata['sess'] with experiment info.
        dset_idx (int): Index of the dataset within dataset.dsets to use
        cids (np.ndarray or list): Cluster IDs in the order they appear in robs columns,
            shape (NC,). These are the neural unit identifiers from spike sorting.
        fixation_degree_radius (float): Maximum distance from center (in degrees of
            visual angle) for a time bin to be considered fixating. Only spikes in
            fixation bins are extracted.
        dt (float): Time bin size in seconds (default: 1/240 ~ 4.17ms)

    Returns:
        spike_times_trials (list of list of np.ndarray): Raw spike times per trial/cell.
            spike_times_trials[itrial][cell_idx] = 1D array of spike times in seconds.
            Outer list has length NT (trials), inner lists have length NC (cells).
            Only spikes falling in fixation bins are included.
        trial_time_windows (list of tuple): Time windows for each trial.
            trial_time_windows[itrial] = (t_start, t_end) in seconds defining the
            original bin edges. (np.nan, np.nan) for invalid trials.
        trial_t_bins (list of np.ndarray): Time bin centers using sparse indexing.
            trial_t_bins[itrial] = array of shape (T,) where T is max psth_ind + 1.
            trial_t_bins[itrial][i] = time center for bin i in seconds, or NaN if
            not a fixation bin. This matches the structure of robs.
    """
    sess = dataset.dsets[dset_idx].metadata['sess']
    ptb2ephys, _ = get_clock_functions(sess.exp)
   
    # =========================================================================
    # Extract dataset dimensions and compute fixation mask
    # =========================================================================
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    trials = np.unique(trial_inds)

    NC = dataset.dsets[dset_idx]['robs'].shape[1]
    NT = len(trials)

    fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < fixation_degree_radius

    # =========================================================================
    # Load and preprocess raw spike data from kilosort results
    # =========================================================================
    spike_times = sess.ks_results.spike_times
    spike_clusters = sess.ks_results.spike_clusters
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'].numpy()
    
    # Create cluster ID to column index mapping
    cids = np.asarray(cids)
    n_cids = len(cids)
    cids2inds = np.zeros(np.max(cids) + 1, dtype=int)
    cids2inds[cids] = np.arange(n_cids)
    
    # Ensure spike times are sorted for efficient searchsorted
    if not np.all(np.diff(spike_times) >= 0):
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]
    
    # Filter spikes to only include clusters in cids
    cids_mask = np.isin(spike_clusters, cids)
    spike_times_filtered = spike_times[cids_mask]
    spike_clusters_filtered = spike_clusters[cids_mask]
    spike_inds = cids2inds[spike_clusters_filtered]
    
    NT = len(trials)
    NC = len(cids)
    
    # =========================================================================
    # Initialize output structures with sparse indexing
    # =========================================================================
    T = np.max(psth_inds).item() + 1
    spike_times_trials = [[np.array([]) for _ in range(NC)] for _ in range(NT)]
    trial_time_windows = [(np.nan, np.nan) for _ in range(NT)]
    trial_t_bins = [np.full(T, np.nan) for _ in range(NT)]
    
    # =========================================================================
    # Loop over trials and extract spike times for fixation bins
    # =========================================================================
    for itrial in tqdm(range(NT), desc="Extracting spike times"):
        trial_mask = trials[itrial] == trial_inds
        if np.sum(trial_mask) == 0:
            continue
        
        # -----------------------------------------------------------------
        # Get trial timing info and compute bin edges
        # -----------------------------------------------------------------
        trial_id = int(trials[itrial])
        trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
        trial_image_ids = trial.image_ids
        if len(np.unique(trial_image_ids)) < 2:
            continue
        start_idx = np.where(trial_image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])
        
        trial_bin_edges = np.arange(flip_times[0], flip_times[-1], dt)
        if len(trial_bin_edges) < 2:
            continue
        
        # -----------------------------------------------------------------
        # Get fixation bins and store time bin centers
        # -----------------------------------------------------------------
        ix = trial_mask & fixation
        if np.sum(ix) == 0:
            continue
        
        trial_psth_inds = psth_inds[ix]
        trial_t_bins_centers = t_bins[ix]
        
        if len(trial_t_bins_centers) == 0:
            continue
        
        # Store with sparse indexing: position i = psth_ind i
        trial_t_bins[itrial][trial_psth_inds] = trial_t_bins_centers
        
        t_start = trial_bin_edges[0]
        t_end = trial_bin_edges[-1]
        trial_time_windows[itrial] = (t_start, t_end)
        
        # -----------------------------------------------------------------
        # Extract spikes in trial window and assign to bins
        # -----------------------------------------------------------------
        i0 = np.searchsorted(spike_times_filtered, t_start)
        i1 = np.searchsorted(spike_times_filtered, t_end)
        
        if i0 >= i1:
            continue
        
        trial_spike_times = spike_times_filtered[i0:i1]
        trial_spike_inds = spike_inds[i0:i1]
        
        # Assign spikes to bins using digitize
        spike_bin_indices = np.digitize(trial_spike_times, trial_bin_edges) - 1
        n_bins = len(trial_bin_edges) - 1
        spike_bin_indices = np.clip(spike_bin_indices, 0, n_bins - 1)
        
        # Keep only spikes in fixation bins
        fixation_spike_mask = np.isin(spike_bin_indices, trial_psth_inds)
        trial_spike_times = trial_spike_times[fixation_spike_mask]
        trial_spike_inds = trial_spike_inds[fixation_spike_mask]
        
        # -----------------------------------------------------------------
        # Organize spikes by cell
        # -----------------------------------------------------------------
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
    """
    Load and preprocess fixation RSVP data for a session, with caching support.

    Main entry point for loading fixation RSVP data. Handles the complete pipeline:
    1. Load dataset from config
    2. Collate trial-aligned neural responses, eye positions, and image IDs
    3. Extract raw spike times per trial
    4. Remove duplicate trials
    5. Remove trials with insufficient fixation duration
    6. Align image IDs across trials (truncate/shift/remove mismatched trials)
    7. Validate spike times match binned responses

    Results are cached to disk for faster subsequent loads.

    Args:
        subject (str): Subject identifier (e.g. 'Allen', 'Ellie')
        date (str): Session date string in YYYY-MM-DD format (e.g. '2022-03-04')
        dataset_configs_path (str): Path to YAML file listing dataset configs.
            The filename stem is used for cache file naming.
        use_cached_data (bool): If True and cache exists, load from cache instead
            of reprocessing (default: False). Cache is always updated after processing.
        fixation_degree_radius (float): Maximum distance from center (in degrees)
            for a time bin to be considered fixating (default: 1)
        fixation_duration_bins_threshold (int): Minimum number of fixation bins
            required to keep a trial (default: 20, ~83ms at 240Hz)
        salvageable_mismatch_time_threshold (int): Minimum time index for a mismatch
            to be salvageable via truncation in align_image_ids (default: 25)
        verbose (bool): If True, print processing details and validation info
            (default: False)

    Returns:
        dict: Dictionary containing processed data with keys:
            - 'robs' (np.ndarray): Spike counts, shape (NT, T, NC). NaN for invalid bins.
            - 'dfs' (np.ndarray): Data flags, shape (NT, T, NC)
            - 'eyepos' (np.ndarray): Eye position in degrees, shape (NT, T, 2)
            - 'fix_dur' (np.ndarray): Fixation duration per trial, shape (NT,)
            - 'image_ids' (np.ndarray): Image IDs per bin, shape (NT, T). 0-indexed.
            - 'cids' (list): Cluster IDs corresponding to columns of robs
            - 'rsvp_images' (np.ndarray): RSVP stimulus images
            - 'spike_times_trials' (list): Raw spike times [trial][cell] = array
            - 'trial_time_windows' (list): Time windows [(t_start, t_end), ...]
            - 'trial_t_bins' (list): Sparse time bin centers [trial] = array
            - 'dataset' (DictDataset): The original dataset object

    Raises:
        AssertionError: If processed data path doesn't exist, if spike times don't
            match robs after processing, or if too many trials have image ID mismatches
    """
    # =========================================================================
    # Load dataset and setup cache paths
    # =========================================================================
    dataset, dataset_config = get_dataset_from_config(subject, date, dataset_configs_path)
    dset_idx = dataset.inds[:,0].unique().item()

    processed_data_path = DATA_DIR / 'processed' / dataset.dsets[dset_idx].metadata['sess'].name / 'datasets'
    assert processed_data_path.exists(), f"Processed data path {processed_data_path} does not exist"
    cached_file = processed_data_path / f'fixrsvp_data_collated_{Path(dataset_configs_path).stem}.pkl'
    
    # =========================================================================
    # Load from cache or extract from raw data
    # =========================================================================
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
        # Extract trial-aligned data from dataset
        robs, dfs, eyepos, fix_dur, image_ids = collate_fixrsvp_data(dataset, dset_idx, fixation_degree_radius)
        validate_image_ids(image_ids, dataset, dset_idx)

        # Extract raw spike times per trial
        spike_times_trials, trial_time_windows, trial_t_bins = extract_spike_times_per_trial(
            dataset, dset_idx, dataset_config['cids'], fixation_degree_radius, dt=1/240)
        rsvp_images = get_fixrsvp_stack(frames_per_im=1)

    # =========================================================================
    # Save to cache for future runs
    # =========================================================================
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
    
    # =========================================================================
    # Validate spike times match robs before processing
    # =========================================================================
    all_match, mismatches = compare_spike_times_to_robs(
        spike_times_trials, trial_time_windows, trial_t_bins, robs, dt=1/240, verbose=verbose
    )
    assert all_match, f"Found {len(mismatches)} mismatches first pass"
    
    # =========================================================================
    # Clean data: remove duplicates, short trials, and align image IDs
    # =========================================================================
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
    
    # =========================================================================
    # Final validation: ensure spike times still match after all processing
    # =========================================================================
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
        'dataset': dataset,
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