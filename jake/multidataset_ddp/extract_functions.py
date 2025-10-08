#!/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
from eval_stack_multidataset import evaluate_model_multidataset
from DataYatesV1 import get_session

def extract_bps_saccade(model_name, stim_type, all_results):
    """
    Extract BPS and saccade data for a given model and stimulus type.

    Parameters
    ----------
    model_name : str
        Name of the model
    stim_type : str
        Stimulus type ('gaborium', 'backimage', 'fixrsvp', 'gratings')
    all_results : dict
        Results dictionary from evaluation pipeline

    Returns
    -------
    bps : np.array [N x 1]
        Bits per spike for each unit
    saccade_robs : np.array [Tbins x N]
        Saccade-triggered observed responses
    saccade_rhat : np.array [Tbins x N]
        Saccade-triggered predicted responses
    saccade_time_bins : np.array [Tbins]
        Time bins relative to saccade onset (negative = pre-saccade, positive = post-saccade)
    cids : list [N]
        Cell IDs for each unit
    dids : list [N]
        Dataset IDs for each unit
    """
    results = all_results[model_name]
    
    # Extract BPS data
    bps = np.concatenate(results['bps'][stim_type]['bps'])

    # Handle both flat lists and nested lists for CIDs/datasets
    cids_data = results['bps'][stim_type]['cids']
    dids_data = results['bps'][stim_type]['datasets']

    bps_cids = []
    bps_dids = []

    # Check if we have nested lists or flat lists
    if len(cids_data) > 0 and isinstance(cids_data[0], (list, np.ndarray)):
        # Nested lists - each element is a list of CIDs for one dataset
        for cids_list, dids_list in zip(cids_data, dids_data):
            bps_cids.extend(cids_list)
            bps_dids.extend(dids_list)
    else:
        # Flat lists - already concatenated
        bps_cids = list(cids_data)
        bps_dids = list(dids_data)
    
    # Extract saccade data if available
    if 'saccade' in results and stim_type in results['saccade']:
        saccade_robs = np.concatenate([rbar for rbar in results['saccade'][stim_type]['rbar']], axis=1)
        saccade_rhat = np.concatenate([rbarhat for rbarhat in results['saccade'][stim_type]['rbarhat']], axis=1)

        # Handle saccade CIDs/datasets same way as BPS
        sac_cids_data = results['saccade'][stim_type]['cids']
        sac_dids_data = results['saccade'][stim_type]['datasets']

        saccade_cids = []
        saccade_dids = []

        if len(sac_cids_data) > 0 and isinstance(sac_cids_data[0], (list, np.ndarray)):
            # Nested lists
            for cids_list, dids_list in zip(sac_cids_data, sac_dids_data):
                saccade_cids.extend(cids_list)
                saccade_dids.extend(dids_list)
        else:
            # Flat lists
            saccade_cids = list(sac_cids_data)
            saccade_dids = list(sac_dids_data)
    else:
        # Create NaN arrays if saccade data not available
        n_units = len(bps_cids)
        saccade_robs = np.full((150, n_units), np.nan)  # 110 time bins typical
        saccade_rhat = np.full((150, n_units), np.nan)
        saccade_cids = bps_cids
        saccade_dids = bps_dids

    print("Using time bins that are hard coded. need to fix")
    sac_time_bins = np.arange(-50, 100)
    return bps, saccade_robs, saccade_rhat, sac_time_bins, bps_cids, bps_dids

def extract_ccnorm(model_name, all_results):
    """
    Extract CCNORM data for a given model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    all_results : dict
        Results dictionary from evaluation pipeline
        
    Returns
    -------
    ccnorm : np.array [N x 1]
        Noise-corrected correlation for each unit (NaN if not valid)
    rbar : np.array [Tbins x N]
        Mean observed responses over time
    rhatbar : np.array [Tbins x N] 
        Mean predicted responses over time
    cids : list [N]
        Cell IDs for each unit
    dids : list [N]
        Dataset IDs for each unit
    """
    results = all_results[model_name]
    
    # Always use all cells from QC data for consistency
    all_cids = results['qc']['all_cids']
    all_dids = results['qc']['all_datasets']
    n_units = len(all_cids)

    # Initialize with NaN arrays
    ccnorm = np.full(n_units, np.nan)
    rbar = None
    rhatbar = None

    if 'ccnorm' in results and 'fixrsvp' in results['ccnorm']:
        # Extract CCNORM data for successfully processed datasets
        ccnorm_data = np.concatenate(results['ccnorm']['fixrsvp']['ccnorm'])
        rbar_data = np.concatenate([rb for rb in results['ccnorm']['fixrsvp']['rbar']], axis=1)
        rhatbar_data = np.concatenate([rhb for rhb in results['ccnorm']['fixrsvp']['rbarhat']], axis=1)

        # Get CIDs for CCNORM data
        ccnorm_cids_data = results['ccnorm']['fixrsvp']['cids']
        if len(ccnorm_cids_data) > 0 and isinstance(ccnorm_cids_data[0], (list, np.ndarray)):
            # Nested lists
            ccnorm_cids = []
            for cids_list in ccnorm_cids_data:
                ccnorm_cids.extend(cids_list)
        else:
            # Flat lists
            ccnorm_cids = list(ccnorm_cids_data)

        # Initialize full arrays
        n_time_bins = rbar_data.shape[0]
        rbar = np.full((n_time_bins, n_units), np.nan)
        rhatbar = np.full((n_time_bins, n_units), np.nan)

        # Map CCNORM data to correct positions in full arrays
        for i, cid in enumerate(ccnorm_cids):
            try:
                # Find position of this CID in the full list
                full_idx = all_cids.index(cid)
                ccnorm[full_idx] = ccnorm_data[i]
                rbar[:, full_idx] = rbar_data[:, i]
                rhatbar[:, full_idx] = rhatbar_data[:, i]
            except ValueError:
                print(f"Warning: CCNORM CID {cid} not found in full CID list")
                continue
    else:
        # No CCNORM data available
        rbar = np.full((100, n_units), np.nan)  # 100 time bins typical
        rhatbar = np.full((100, n_units), np.nan)

    return ccnorm, rbar, rhatbar, all_cids, all_dids

def extract_qc_spatial(model_name, all_results):
    """
    Extract QC and spatial data for a given model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    all_results : dict
        Results dictionary from evaluation pipeline
        
    Returns
    -------
    depth : np.array [N x 1]
        Depth relative to L4 for each unit (NaN if not available)
    waveforms : np.array [N x T x channels]
        Waveforms for each unit (NaN if not available)
    cids : list [N]
        Cell IDs for each unit
    dids : list [N] 
        Dataset IDs for each unit
    contamination : np.array [N x 1]
        Contamination rate for each unit
    missing_pct : np.array [N x 1]
        Missing amplitude percentage for each unit
    """
    results = all_results[model_name]
    qc_data = results['qc']
    
    # Get basic cell info
    cids = qc_data['all_cids']
    dids = qc_data['all_datasets']

    # Handle length mismatch between all_cids and contamination data
    contamination_data = qc_data['contamination']
    truncation_data = qc_data['truncation']

    if len(contamination_data) != len(cids):
        print(f"Warning: Contamination data length ({len(contamination_data)}) != CIDs length ({len(cids)})")
        print("Padding with NaNs for missing data")
        # Pad with NaNs to match cids length
        contamination = np.full(len(cids), np.nan)
        contamination[:len(contamination_data)] = contamination_data

        missing_pct = np.full(len(cids), np.nan)
        missing_pct[:len(truncation_data)] = truncation_data
    else:
        contamination = np.array(contamination_data)
        missing_pct = np.array(truncation_data)
    
    n_units = len(cids)
    depth = np.full(n_units, np.nan)
    
    # Initialize waveforms array (will determine size from first valid waveform)
    waveforms = None
    
    # Calculate depth and extract waveforms for each cell
    unique_datasets = list(set(dids))
    
    for dataset in unique_datasets:
        try:
            # Get session data
            subject, date = dataset.split('_')
            sess = get_session(subject, date)
            
            # Load spatial data
            ephys_meta = sess.ephys_metadata
            probe_geom = np.array(ephys_meta['probe_geometry_um'])

            # Try to load laminar data
            laminar_file = sess.sess_dir / 'laminar' / 'laminar.npz'
            if laminar_file.exists():
                laminar_results = np.load(laminar_file)
                l4_depths = laminar_results['l4_depths']
            else:
                print(f"Warning: No laminar data for {dataset}, skipping depth calculation")
                continue
            
            # Load waveforms
            waves_full = np.load(sess.sess_dir / 'qc' / 'waveforms' / 'waveforms.npz')
            dataset_waveforms = waves_full['waveforms']
            wave_cids = waves_full['cids']
            
            # Initialize waveforms array if not done yet
            if waveforms is None:
                n_time, n_channels = dataset_waveforms.shape[1], dataset_waveforms.shape[2]
                waveforms = np.full((n_units, n_time, n_channels), np.nan)
            
            # Process cells from this dataset
            dataset_indices = [i for i, did in enumerate(dids) if did == dataset]
            
            for idx in dataset_indices:
                cid = cids[idx]
                
                if cid in wave_cids:
                    # Find waveform
                    cid_idx = np.where(wave_cids == cid)[0][0]
                    waveforms[idx] = dataset_waveforms[cid_idx]
                    
                    # Calculate depth
                    waveform = dataset_waveforms[cid_idx]
                    max_channel = np.abs(waveform).max(axis=0).argmax()
                    x, y = probe_geom[max_channel]
                    
                    # Determine shank and calculate depth relative to L4
                    if x < 100:  # Left shank
                        depth[idx] = y - l4_depths[0]
                    else:  # Right shank
                        depth[idx] = y - l4_depths[1]
                        
        except Exception as e:
            print(f"Warning: Could not process dataset {dataset}: {e}")
            continue
    
    # If no waveforms were loaded, create empty array
    if waveforms is None:
        waveforms = np.full((n_units, 82, 384), np.nan)  # Default dimensions
    
    return depth, waveforms, cids, dids, contamination, missing_pct

# Example usage for one cell
def example_single_cell():
    """Example of how to extract data for a single cell."""
    # Load results
    results = evaluate_model_multidataset(
        model_type='learned_res',
        analyses=['bps', 'ccnorm', 'saccade'],
        recalc=False
    )
    all_results = {list(results.keys())[0]: list(results.values())[0]}
    model_name = list(all_results.keys())[0]
    
    # Get all data
    bps, saccade_robs, saccade_rhat, saccade_time_bins, bps_cids, bps_dids = extract_bps_saccade(model_name, 'gaborium', all_results)
    ccnorm, rbar, rhatbar, ccnorm_cids, ccnorm_dids = extract_ccnorm(model_name, all_results)
    depth, waveforms, qc_cids, qc_dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)
    
    # Example: Get data for first cell
    cell_idx = 0
    
    print(f"Cell {cell_idx}:")
    print(f"  CID: {bps_cids[cell_idx]}")
    print(f"  Dataset: {bps_dids[cell_idx]}")
    print(f"  BPS: {bps[cell_idx]:.4f}")
    print(f"  Depth: {depth[cell_idx]:.1f} μm rel. L4")
    print(f"  Contamination: {contamination[cell_idx]:.1f}%")
    print(f"  Waveform shape: {waveforms[cell_idx].shape}")
    print(f"  Saccade response shape: {saccade_robs[:, cell_idx].shape}")

if __name__ == "__main__":
    example_single_cell()
