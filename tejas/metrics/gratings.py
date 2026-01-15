#%%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from DataYatesV1.utils.data import prepare_data
# from jake.multidataset_ddp.gratings_analysis import gratings_analysis
# from DataYatesV1.jake.multidataset_ddp.metrics.gratings_analysis import gratings_analysis
from tejas.metrics.grating_utils import plot_phase_tuning_with_fit, plot_ori_tuning
from DataYatesV1.models.config_loader import load_dataset_configs

def get_gratings_for_dataset(date, subject, cache = False):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/gratings_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    gratings_results_path = cache_dir / f'gratings_results.npz'
    if cache and gratings_results_path.exists():
        gratings_results = np.load(gratings_results_path, allow_pickle=True)
        return gratings_results['gratings_results'].item()

    assert len(date) == 10 and type(date) == str, 'Date must be in the format YYYY-MM-DD'
    dataset_configs_path = "/mnt/sata/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_for_metrics_gratings"

    yaml_files = [
        f for f in os.listdir(dataset_configs_path) if f.endswith(".yaml") and "base" not in f and date in f and subject in f
    ]
    dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)
    train_dset, val_dset, dataset_config = prepare_data(dataset_configs[0], mat_dir=Path('/mnt/sata/YatesMarmoV1/mat'), proc_dir=Path('/mnt/sata/YatesMarmoV1/processed'))
    train_dset_loaded = train_dset[:]

    MAX_EYE_MOVEMENT = 10
    n_lags = 20

    robs = train_dset_loaded['robs'].numpy()
    sf = train_dset.dsets[0]['sf'][train_dset.inds[:,1]]
    ori = train_dset.dsets[0]['ori'][train_dset.inds[:,1]].numpy()
    phases = train_dset.dsets[0]['stim_phase'][train_dset.inds[:,1]].numpy()
    dt = 1/dataset_config['sampling']['target_rate']
    n_lags = dataset_config['keys_lags']['stim'][-1]
    dfs = train_dset_loaded['dfs']

    gratings_validity_filter = np.logical_and.reduce([
        np.abs(train_dset.dsets[0]['eyepos'][train_dset.inds[:,1], 0]) < MAX_EYE_MOVEMENT,
        np.abs(train_dset.dsets[0]['eyepos'][train_dset.inds[:,1], 1]) < MAX_EYE_MOVEMENT,
        train_dset.dsets[0]['dpi_valid'][train_dset.inds[:,1]]
    ]).astype(np.float32)

    gratins_validity_filter = gratings_validity_filter[:, None].repeat(robs.shape[1], axis=1)
    gratings_results = gratings_analysis(
                                robs=robs,
                                sf=sf,
                                ori=ori,
                                phases=phases,
                                dt=dt,
                                n_lags=n_lags,
                                dfs=gratins_validity_filter,
                                min_spikes=30  # Minimum spike count threshold for analysis
                            )
    assert 'phases' not in gratings_results, 'phases already in gratings_results'
    assert 'robs' not in gratings_results, 'robs already in gratings_results'
    # assert 'dt' not in gratings_results, 'dt already in gratings_results'
    gratings_results['phases'] = phases
    gratings_results['robs'] = robs
    # gratings_results['dt'] = dt
    gratings_results['modulation_index_dict'] = {}
    for unit in range(gratings_results['robs'].shape[1]):
        gratings_results['modulation_index_dict'][unit] = {'modulation_index':gratings_results['modulation_index'][unit],
                                                          'valid':not np.isnan(gratings_results['modulation_index'][unit])}


    np.savez(gratings_results_path, gratings_results=gratings_results)
    return gratings_results

def get_peak_sf_for_dataset(gratings_info):
    peak_sf = gratings_info['sfs'][gratings_info['peak_sf_idx']]
    sf_dict = {}
    for unit in range(len(peak_sf)):
        sf_dict[unit] = {'peak_sf': peak_sf[unit], 'valid': True}
    return sf_dict

# date = '2022-04-13'
# gratings_results = get_gratings_for_dataset(date)
# #%%
# unit_idx = 126

# plot_phase_tuning_with_fit(gratings_results, unit_idx, title=f'Unit {unit_idx}')
# plt.show()

# plot_ori_tuning(gratings_results, unit_idx, title=f'Unit {unit_idx}')
# plt.show()