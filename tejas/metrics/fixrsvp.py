#%%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from DataYatesV1.models.config_loader import load_dataset_configs

def collate_fixrsvp(data, min_bins, min_trial_dur):

    assert 'robs' in data, 'robs must be in data'
    assert 'eyepos' in data, 'eyepos must be in data'
    assert 'psth_inds' in data, 'psth_inds must be in data'
    assert 'trial_inds' in data, 'trial_inds must be in data'

    unique_trials = np.unique(data['trial_inds'])
    unique_psth_inds = np.unique(data['psth_inds'])
    robs = []
    eyepos = []
    min_psth_ind = np.min(unique_psth_inds)
    required_inds = np.arange(min_psth_ind, min_bins)
    for iT in unique_trials: 
        trial_mask = (data['trial_inds'] == iT).numpy()
        trial_psth_inds = (data['psth_inds'][trial_mask]).numpy()
        if not np.isin(required_inds, trial_psth_inds).all():

            print(f'Skipping trial {iT}, has {len(trial_psth_inds)} only bins')
            continue

        trial_mask[trial_mask] = np.isin(trial_psth_inds, required_inds)

        trial_robs = data['robs'][trial_mask].T
        trial_eyepos = data['eyepos'][trial_mask]
        robs.append(trial_robs)
        eyepos.append(trial_eyepos)
    robs = np.stack(robs, axis=1)
    eyepos = np.stack(eyepos, axis=0)
    print(f'Found {robs.shape[1]} trials for fixrsvp that are at least {min_trial_dur} seconds long.')
    return required_inds, robs, eyepos 

def get_fixrsvp_for_dataset(date, subject, cache = False):
    
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/fixrsvp_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixrsvp_results_path = cache_dir / f'fixrsvp_results.npz'
    if cache and fixrsvp_results_path.exists():
        fixrsvp_results = np.load(fixrsvp_results_path, allow_pickle=True)
        return fixrsvp_results['fixrsvp_results'].item()

    assert len(date) == 10 and type(date) == str, 'Date must be in the format YYYY-MM-DD'
    dataset_configs_path = "/mnt/sata/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_for_metrics_rsvp"
    yaml_files = [
        f for f in os.listdir(dataset_configs_path) if f.endswith(".yaml") and "base" not in f and date in f and subject in f
    ]
    dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)
    from DataYatesV1.utils.data import prepare_data
    train_dset, val_dset, dataset_config = prepare_data(dataset_configs[0], mat_dir=Path('/mnt/sata/YatesMarmoV1/mat'), proc_dir=Path('/mnt/sata/YatesMarmoV1/processed'))

    fixrsvp_data = train_dset[:]
    psth_inds = train_dset.dsets[0]['psth_inds'][train_dset.inds[:,1]]
    trial_inds = train_dset.dsets[0]['trial_inds'][train_dset.inds[:,1]]
    fixrsvp_data['psth_inds'] = psth_inds
    fixrsvp_data['trial_inds'] = trial_inds



    # min_trial_dur = .75 # include
    min_trial_dur = 0.5
    # min_trial_dur = 1
    min_n_bins = int(min_trial_dur * 240)
    inds, robs, eyepos = collate_fixrsvp(fixrsvp_data, min_n_bins, min_trial_dur)
    t_trial = inds / 240
    psth = np.mean(robs, axis=1).squeeze()
    psth_ste = np.std(robs, axis=1) / np.sqrt(robs.shape[1]).squeeze()

    fixrsvp_results = {'t_trial': t_trial, 'psth': psth, 'psth_ste': psth_ste, 'robs': robs, 'eyepos': eyepos, 'robs_unprocessed': fixrsvp_data['robs']}

    np.savez(fixrsvp_results_path, fixrsvp_results=fixrsvp_results)
    return fixrsvp_results


def plot_fixrsvp_psth(fixrsvp_info, cid, ax=None):
    t_trial = fixrsvp_info['t_trial']
    psth = fixrsvp_info['psth']
    psth_ste = fixrsvp_info['psth_ste']
    robs = fixrsvp_info['robs']
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_trial, psth[cid], label='Data')
    ax.fill_between(t_trial, (psth[cid] - psth_ste[cid]).squeeze(), (psth[cid] + psth_ste[cid]).squeeze(), alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'FixRSVP PSTH for Unit {cid}')
    ax.legend()
    return ax


def plot_fixrsvp_spike_raster(fixrsvp_info, cid, ax=None):
# Plot spike raster
    robs = fixrsvp_info['robs']
    t_trial = fixrsvp_info['t_trial']
    n_trials = robs.shape[1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for i_trial in range(n_trials):
        # Get spikes for this trial (values > 0)
        spikes = robs[cid, i_trial, :] > 0
        spike_times = t_trial[spikes]
        if len(spike_times) > 0:
            ax.eventplot(spike_times, lineoffsets=i_trial, linelengths=0.8, color='k', alpha=0.7)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'FixRSVP Spike Raster - Unit {cid}')
    return ax

def plot_fixrsvp_psth_and_spike_raster(fixrsvp_info, cid, ax=None):
    t_trial = fixrsvp_info['t_trial']
    psth = fixrsvp_info['psth']
    psth_ste = fixrsvp_info['psth_ste']
    robs = fixrsvp_info['robs']
    n_trials = robs.shape[1]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot spike raster on left y-axis
    for i_trial in range(n_trials):
        # Get spikes for this trial (values > 0)
        spikes = robs[cid, i_trial, :] > 0
        spike_times = t_trial[spikes]
        if len(spike_times) > 0:
            ax.eventplot(spike_times, lineoffsets=i_trial, linelengths=0.8, color='k', alpha=0.7)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Time (s)')
    
    # Create twin axis for PSTH on right y-axis
    ax2 = ax.twinx()
    ax2.plot(t_trial, psth[cid])
    ax2.fill_between(t_trial, (psth[cid] - psth_ste[cid]).squeeze(), (psth[cid] + psth_ste[cid]).squeeze(), alpha=0.5)
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.tick_params(axis='y')
    # ax2.legend()
    
    ax.set_title(f'FixRSVP PSTH and Spike Raster - Unit {cid}')
    return ax
# date = '2022-04-13'
# fixrsvp_info = get_fixrsvp_for_dataset(date)

# cid = 16

# plot_fixrsvp_psth(fixrsvp_info, cid)
# plt.show()
# plot_fixrsvp_spike_raster(fixrsvp_info, cid)
# plt.show()
