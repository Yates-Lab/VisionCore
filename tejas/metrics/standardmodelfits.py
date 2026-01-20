#%%
from pathlib import Path
import json
import numpy as np
def get_ln_standard_model_fits(date, subject, ln_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_ln_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil', cache = False):
    folder_path = Path(ln_dir) / f"{subject}_{date}"
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/standardmodelfits_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    standardmodelfits_results_path = cache_dir / f'standardmodelfits_ln_results.npz'
    if cache and standardmodelfits_results_path.exists():
        standardmodelfits_results = np.load(standardmodelfits_results_path, allow_pickle=True)
        return standardmodelfits_results['standardmodelfits_ln_results'].item()
    #get all cell_id_best folders in the folder_path
    cell_id_best_folders = [f for f in folder_path.iterdir() if f.is_dir() and f.name.startswith('cell_') and f.name.endswith('_best')]
    result_json_files = [f/'results.json' for f in cell_id_best_folders]
    
    #load the results.json files and put dictionary with cell_id as key and test_bps from results.json as value
    model_test_bps = {}
    for f in result_json_files:
        if f.exists():
            with open(f, 'r') as file:
                data = json.load(file)
            model_test_bps[int(f.parent.name.split('_')[1])] = {'ln_bps': data['test_bps'], 'valid': True}
        else:
            # print(f"Results file not found for {f.parent.name}")
            model_test_bps[int(f.parent.name.split('_')[1])] = {'ln_bps': None, 'valid': False}
    np.savez(standardmodelfits_results_path, standardmodelfits_ln_results=model_test_bps)
    return model_test_bps

def get_energy_standard_model_fits(date, subject, energy_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_energy_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil', cache = False):
    folder_path = Path(energy_dir) / f"{subject}_{date}"
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/standardmodelfits_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    standardmodelfits_results_path = cache_dir / f'standardmodelfits_energy_results.npz'
    if cache and standardmodelfits_results_path.exists():
        standardmodelfits_results = np.load(standardmodelfits_results_path, allow_pickle=True)
        return standardmodelfits_results['standardmodelfits_energy_results'].item()
    cell_id_best_folders = [f for f in folder_path.iterdir() if f.is_dir() and f.name.startswith('cell_') and f.name.endswith('_best')]
    result_json_files = [f/'results.json' for f in cell_id_best_folders]
    model_test_bps = {}
    for f in result_json_files:
        if f.exists():
            with open(f, 'r') as file:
                data = json.load(file)
            model_test_bps[int(f.parent.name.split('_')[1])] = {'energy_bps': data['test_bps'], 'valid': True}
        else:
            # print(f"Results file not found for {f.parent.name}")
            model_test_bps[int(f.parent.name.split('_')[1])] = {'energy_bps': None, 'valid': False}
    np.savez(standardmodelfits_results_path, standardmodelfits_energy_results=model_test_bps)
    return model_test_bps
def get_linear_index_standard_model_fits(date, subject, ln_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_ln_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil', energy_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_energy_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil', cache = False):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/standardmodelfits_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    linear_index_results_path = cache_dir / f'linear_index_standard_model_fits_results.npz'
    if cache and linear_index_results_path.exists():
        linear_index_results = np.load(linear_index_results_path, allow_pickle=True)
        return linear_index_results['linear_index_standard_model_fits_results'].item()
    ln_fits = get_ln_standard_model_fits(date, subject, ln_dir, cache = cache)
    energy_fits = get_energy_standard_model_fits(date, subject, energy_dir, cache = cache)
    linear_index = {}
    intersection = set(ln_fits.keys()) & set(energy_fits.keys())
    for key in intersection:
        valid = ln_fits[key]['valid'] and energy_fits[key]['valid']
        if valid:
            relu_ln = np.maximum(ln_fits[key]['ln_bps'], 0)
            relu_energy = np.maximum(energy_fits[key]['energy_bps'], 0)
            linear_index[key] = {'linear_index': (relu_ln - relu_energy)/ (relu_ln + relu_energy), 'valid': valid}
        else:
            linear_index[key] = {'linear_index': None, 'valid': valid}
    np.savez(linear_index_results_path, linear_index_standard_model_fits_results=linear_index)
    return linear_index

# def get_ln_energy_fits(date, subject, ln_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_ln_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil', energy_dir = '/mnt/sata/YatesMarmoV1/standard_model_fits/ray_energy_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil'):
#     ln_fits = get_ln_standard_model_fits(date, subject, ln_dir)
#     energy_fits = get_energy_standard_model_fits(date, subject, energy_dir)

#     #combine the two dictionaries into a single dictionary
#     # combined_fits = {**ln_fits, **energy_fits}
#     return combined_fits

# result_json_files = get_ln_standard_model_fits('2022-02-16', 'Allen')
# print(result_json_files)