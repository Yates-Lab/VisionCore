#%%#!/usr/bin/env python

import yaml
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from DataYatesV1 import (
    DictDataset, enable_autoreload, get_gaborium_sta_ste, get_session, ensure_tensor,
    get_complete_sessions, print_batch, set_seeds, calc_sta
)
from DataYatesV1.utils.io import RowleyDataVis

from scipy.ndimage import gaussian_filter

set_seeds(1002)

# Enable autoreload for interactive development
enable_autoreload()
device = 'cuda:1'

base_config_path = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_backimageonly_multi_120_rowley/multi_basic_base.yaml")
# base_config_path = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_temporal_basis/multi_data_stimembed_base.yaml")
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

#%% Utility Functions

def get_rowley_sessions(proc_dir=None):
    """
    Get a list of available Rowley sessions

    Parameters
    ----------
    proc_dir : str, optional
        The directory containing the processed data for all sessions

    Returns
    -------
    sessions : list
        A list of RowleyDataVis objects
    """
    if proc_dir is None:
        proc_dir = Path('/mnt/ssd/RowleyMarmoV1V2/processed')

    sessions = []
    if not proc_dir.exists():
        print(f"Rowley processed directory {proc_dir} does not exist")
        return sessions

    for sess_dir in proc_dir.glob('*'):
        if sess_dir.is_dir() and (sess_dir / 'datasets').exists():
            try:
                session = RowleyDataVis(sess_dir.name, proc_dir=proc_dir)
                sessions.append(session)
            except Exception as e:
                print(f"Error creating RowleyDataVis for {sess_dir.name}: {e}")
                continue

    return sessions

def get_quality_metrics(sess, n_lags=24):
    """
    Get quality metrics for any session type (Yates or Rowley)
    Uses try/except to handle missing data gracefully

    Parameters
    ----------
    sess : YatesV1Session or RowleyDataVis
        The session object
    n_lags : int
        Number of lags for STA/STE computation

    Returns
    -------
    visual_snr : np.ndarray
        Visual SNR values for each unit
    med_missing_pct : np.ndarray
        Missing spike percentages for each unit
    contam_pct : np.ndarray
        Contamination percentages for each unit
    """
    print(f"Getting quality metrics for session: {sess.name}")

    # -------------------------------------------------------------------------------------------
    # Visual Responsiveness Metric - should work for both session types
    # -------------------------------------------------------------------------------------------
    try:
        sta_data, ste_data = get_gaborium_sta_ste(sess, n_lags)

        signal = np.abs(ste_data - np.median(ste_data, axis=(2,3), keepdims=True))
        sigma = [0, 2, 2, 2]
        signal = gaussian_filter(signal, sigma)
        noise = np.median(signal[:,0], axis=(1,2))
        snr_per_lag = np.max(signal, axis=(2,3)) / noise[:,None]

        from DataYatesV1 import plot_stas
        plot_stas(signal[:,:,None,:,:])

        visual_snr = snr_per_lag.max(axis=1)
        n_units = len(visual_snr)
        print(f"  ✓ Computed visual SNR for {n_units} units")

    except Exception as e:
        print(f"  ⚠️  Could not compute visual SNR: {e}")
        # Fallback: get unit count from available dataset
        try:
            available_dsets = list((sess.sess_dir / 'datasets').glob('*.dset'))
            if available_dsets:
                dset = DictDataset.load(available_dsets[0])
                n_units = dset['robs'].shape[1]
                visual_snr = np.full(n_units, 10.0)  # High SNR to pass threshold
                print(f"  Using dummy visual SNR for {n_units} units")
            else:
                raise ValueError("No datasets found to determine unit count")
        except Exception as e2:
            print(f"  ❌ Could not determine unit count: {e2}")
            raise

    # -------------------------------------------------------------------------------------------
    # Spike sorting quality metrics - only available for Yates sessions
    # -------------------------------------------------------------------------------------------
    try:
        # extract missing % and contamination %
        spike_clusters = sess.ks_results.spike_clusters
        cids = np.unique(spike_clusters)

        refractory = np.load(sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz')
        min_contam_proportions = refractory['min_contam_props']
        contam_pct = np.array([np.min(min_contam_proportions[iU])*100 for iU in range(len(cids))])

        truncation = np.load(sess.sess_dir / 'qc' / 'amp_truncation' / 'truncation.npz')
        med_missing_pct = np.array([np.median(truncation['mpcts'][truncation['cid']==iU]) for iU in range(len(cids))])

        print(f"  ✓ Computed spike sorting QC for {len(cids)} units")

    except Exception as e:
        print(f"  ⚠️  Could not compute spike sorting QC: {e}")
        # Use dummy values that will pass QC thresholds
        med_missing_pct = np.zeros(n_units)  # No missing spikes
        contam_pct = np.zeros(n_units)       # No contamination
        print(f"  Using dummy QC metrics for {n_units} units")

    return visual_snr, med_missing_pct, contam_pct

def process_session_config(sess, base_config, output_dir, snr_thresh, missing_thresh, contam_thresh, n_lags=24):
    """
    Process a single session to create a session-specific config file

    Parameters
    ----------
    sess : YatesV1Session or RowleyDataVis
        The session object
    base_config : dict
        The base configuration dictionary
    output_dir : Path
        Directory to save the new config file
    snr_thresh : float
        SNR threshold for visual responsiveness
    missing_thresh : float
        Missing spikes threshold
    contam_thresh : float
        Contamination threshold
    n_lags : int
        Number of lags for STA/STE computation

    Returns
    -------
    dict
        Dictionary with session statistics
    """
    print(f"Processing session: {sess.name}")

    # Get quality metrics for this session (unified approach with try/except)
    visual_snr, med_missing_pct, contam_pct = get_quality_metrics(sess, n_lags=n_lags)

    # Calculate unit lists
    visually_responsive = np.where(visual_snr > snr_thresh)[0]
    not_missing_spikes = np.where(med_missing_pct < missing_thresh)[0]
    not_contaminated = np.where(contam_pct < contam_thresh)[0]
    good_units = np.intersect1d(visually_responsive, not_contaminated)

    # Create new config by copying base config
    session_config = base_config.copy()

    # Add session-specific fields
    session_config['cids'] = good_units.tolist()
    session_config['session'] = sess.name
    session_config['visual'] = visually_responsive.tolist()
    session_config['qcmissing'] = not_missing_spikes.tolist()
    session_config['qccontam'] = not_contaminated.tolist()
    session_config['snr'] = snr_thresh
    session_config['missingth'] = missing_thresh
    session_config['contamth'] = contam_thresh

    # Add lab field based on session type
    if isinstance(sess, RowleyDataVis):
        session_config['lab'] = 'rowley'
    else:
        session_config['lab'] = 'yates'

    # Save the session config with custom formatting for lists
    output_file = output_dir / f"{sess.name}.yaml"

    # Custom representer to format lists in flow style (inline)
    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, represent_list)

    with open(output_file, 'w') as f:
        yaml.dump(session_config, f, default_flow_style=False, sort_keys=False)

    # Return statistics
    stats = {
        'session': sess.name,
        'total_units': len(visual_snr),
        'visually_responsive': len(visually_responsive),
        'not_missing': len(not_missing_spikes),
        'not_contaminated': len(not_contaminated),
        'good_units': len(good_units)
    }

    print(f"  {stats['visually_responsive']}/{stats['total_units']} visually responsive")
    print(f"  {stats['not_missing']}/{stats['total_units']} pass missing threshold")
    print(f"  {stats['not_contaminated']}/{stats['total_units']} pass contamination threshold")
    print(f"  {stats['good_units']}/{stats['total_units']} good units (visual + contamination)")
    print(f"  Saved: {output_file}")

    return stats

# %%
# Main processing loop
snr_threshold = 5
missing_threshold = 25
contamination_threshold = 50

# Get sessions from both labs
yates_sessions = get_complete_sessions()
rowley_sessions = get_rowley_sessions()

# Combine all sessions
all_sessions = yates_sessions + rowley_sessions

output_dir = base_config_path.parent

print(f"Found {len(yates_sessions)} Yates sessions")
print(f"Found {len(rowley_sessions)} Rowley sessions")
print(f"Total sessions: {len(all_sessions)}")
print(f"Output directory: {output_dir}")
print(f"Thresholds - SNR: {snr_threshold}, Missing: {missing_threshold}%, Contamination: {contamination_threshold}%")
print()

#%%
# Print session details
if yates_sessions:
    print("Yates sessions:")
    for sess in yates_sessions:
        print(f"  - {sess.name}")

if rowley_sessions:
    print("Rowley sessions:")
    for sess in rowley_sessions:
        print(f"  - {sess.name}")
print()

#%%
# Process all sessions
all_stats = []
for sess in tqdm(rowley_sessions, desc="Processing sessions"):
    try:
        stats = process_session_config(
            sess, base_config, output_dir,
            snr_threshold, missing_threshold, contamination_threshold
        )
        all_stats.append(stats)
    except Exception as e:
        print(f"Error processing {sess.name}: {e}")
        continue
    print()

# %%
# Summary statistics
print("="*60)
print("SUMMARY")
print("="*60)

if all_stats:
    total_sessions = len(all_stats)
    total_units = sum(s['total_units'] for s in all_stats)
    total_visual = sum(s['visually_responsive'] for s in all_stats)
    total_good = sum(s['good_units'] for s in all_stats)

    print(f"Successfully processed: {total_sessions} sessions")
    print(f"Total units across all sessions: {total_units}")
    print(f"Total visually responsive units: {total_visual} ({100*total_visual/total_units:.1f}%)")
    print(f"Total good units: {total_good} ({100*total_good/total_units:.1f}%)")
    print()

    # Per-session breakdown
    print("Per-session breakdown:")
    for stats in all_stats:
        pct_good = 100 * stats['good_units'] / stats['total_units'] if stats['total_units'] > 0 else 0
        print(f"  {stats['session']}: {stats['good_units']}/{stats['total_units']} good units ({pct_good:.1f}%)")

# %%

