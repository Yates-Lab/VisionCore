#%%
import numpy as np
def get_trial_data(data, max_bins=240, peak_lag=0, fixation_window_radius = None):
        '''
        returns: robs, rhat, eyepos
        robs: (num_trials, max_bins, num_cells)
        rhat: (num_trials, max_bins, num_cells)
        eyepos: (num_trials, max_bins, num_eyepos_lags, 2)
        stim: (num_trials, max_bins, num_stim_lags, H, W)
        '''
        # assert 'rhat' in data, 'rhat must be in data'
        assert 'eyepos' in data, 'eyepos must be in data'
        assert 'psth_inds' in data, 'psth_inds must be in data'
        assert 'trial_inds' in data, 'trial_inds must be in data'
        assert 'stim' in data, 'stim must be in data'

        if 'rhat' not in data:
            data['rhat'] = data['robs'].clone()

        assert len(data['stim'].shape) == 5, 'stim must be a 5D array (num_trials, channels, num_stim_lags, H, W)'

        unique_trials = np.unique(data['trial_inds'])
        unique_psth_inds = np.unique(data['psth_inds'])
        num_trials = len(unique_trials)
        num_cells = data['robs'].shape[1]
        num_eyepos_lags = data['eyepos'].shape[1]
        num_stim_lags = data['stim'].shape[2]
        H = data['stim'].shape[3]
        W = data['stim'].shape[4]
        # Convert peak_lag to array if it's a scalar
        if np.isscalar(peak_lag):
            peak_lag = np.full(num_cells, peak_lag, dtype=int)
        else:
            peak_lag = np.asarray(peak_lag, dtype=int)
            assert len(peak_lag) == num_cells, f"peak_lag must have length {num_cells}"
    
        robs = np.nan*np.zeros((num_trials, max_bins, num_cells))
        rhat = np.nan*np.zeros((num_trials, max_bins, num_cells))
        eyepos = np.nan*np.zeros((num_trials, max_bins, num_eyepos_lags, 2))
        stim = np.nan*np.zeros((num_trials, max_bins, num_stim_lags, H, W))
        for i, iT in enumerate(unique_trials):
            
            trial_mask = (data['trial_inds'] == iT).numpy()
            trial_psth_inds = (data['psth_inds'][trial_mask]).numpy()
            min_psth_ind = np.min(unique_psth_inds)
            required_inds = np.arange(min_psth_ind, max_bins)
            
            trial_mask[trial_mask] = np.isin(trial_psth_inds, required_inds)
            trial_psth_inds = (data['psth_inds'][trial_mask]).numpy()
            if plot_verbosity > 1:
                print(f'Trial {iT} has {trial_psth_inds.shape[0]} bins')
            
            
            # robs[i, trial_psth_inds, :] = data['robs'][trial_mask]
            # rhat[i, trial_psth_inds, :] = data['rhat'][trial_mask]

            # Create mapping from psth_ind to data index for this trial
            trial_data_indices = np.where(trial_mask)[0]
            psth_to_data_idx = dict(zip(trial_psth_inds, trial_data_indices))
            
            # Shift robs and rhat by peak_lag for each cell
            #check that trial_psth_inds is increasing
            assert np.all(np.diff(trial_psth_inds) > 0), 'trial_psth_inds must be increasing'
            for psth_ind in trial_psth_inds:
                if fixation_window_radius is not None:
                    if np.hypot(data['eyepos'][psth_to_data_idx[psth_ind], 0, 0], data['eyepos'][psth_to_data_idx[psth_ind], 0, 1]) > fixation_window_radius:
                        if plot_verbosity > 1:
                            distance_from_center = np.hypot(data['eyepos'][psth_to_data_idx[psth_ind], 0, 0], data['eyepos'][psth_to_data_idx[psth_ind], 0, 1])
                            print(f'Trial {iT} has eye position outside fixation window at psth_ind {psth_ind} with distance {distance_from_center} from center')
                        break

                for cell_idx in range(num_cells):
                    source_psth_ind = psth_ind + peak_lag[cell_idx]
                    if source_psth_ind in psth_to_data_idx:
                        source_idx = psth_to_data_idx[source_psth_ind]
                        robs[i, psth_ind, cell_idx] = data['robs'][source_idx, cell_idx]
                        rhat[i, psth_ind, cell_idx] = data['rhat'][source_idx, cell_idx]
        
            eyepos[i, trial_psth_inds, :] = data['eyepos'][trial_mask]
            # print(stim[i, trial_psth_inds, :, :, :].shape, data['stim'][trial_mask, :, :, :, :].shape)
            stim[i, trial_psth_inds, :, :, :] = data['stim'][trial_mask, :, :, :, :].squeeze()
        return robs, rhat, eyepos, stim
#%%

from pathlib import Path
import os
from DataYatesV1.models.config_loader import load_dataset_configs
import matplotlib.pyplot as plt

date = '2022-04-13'
subject = 'Allen'

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
#%%
plot_verbosity = 2
robs, rhat, eyepos, stim = get_trial_data(fixrsvp_data, peak_lag=0, fixation_window_radius=0.5)
#%%
t_trial = np.arange(robs.shape[1]) / 240
psth = np.nanmean(robs, axis=0).squeeze()
psth_ste = np.nanstd(robs, axis=0) / np.sqrt(robs.shape[0]).squeeze()
from tejas.metrics.fixrsvp import plot_fixrsvp_psth_and_spike_raster
fig, ax = plt.subplots(figsize=(6, 4))
plot_fixrsvp_psth_and_spike_raster({'t_trial': t_trial, 'psth': psth.T, 'psth_ste': psth_ste.T, 'robs': np.permute_dims(robs, axes=(2, 0, 1))}, 33, ax=ax)
# plt.xlim(0, 0.5)
plt.show()



#%%