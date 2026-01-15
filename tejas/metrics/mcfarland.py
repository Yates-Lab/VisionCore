#%%
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Configure matplotlib for PDF output with embedded fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
def check_if_valid_mcfarland_analysis(results_data, min_trials = 50, spike_percentage_min = 0.01, verbose = False):
    if results_data['percent_of_total_spikes'] < spike_percentage_min:
        if verbose:
            print(f'Percent of total spikes is less than {spike_percentage_min}')
        return False
    if results_data['num_valid_trials'] < min_trials: #NEEDS TO BE FIXED, since now robs will always be num_trials
        if verbose:
            print(f'Number of valid trials is less than {min_trials}')
        return False
    if results_data['exponenetial_fit']['mse'] > 0.1:
        if verbose:
            print(f'MSE is greater than 0.1')
        return False
    normalized_rate_vars = results_data['bin_rate_vars']/results_data['em_corrected_var']
    if (normalized_rate_vars > 2).any():
        if verbose:
            print(f'Normalized rate vars are greater than 2')
        return False
    if (normalized_rate_vars[0] == normalized_rate_vars).all():
        if verbose:
            print(f'Normalized rate vars are all the same')
        return False
    if np.isnan(results_data['exponenetial_fit']['tau']) or results_data['exponenetial_fit']['tau'] == np.inf:
        if verbose:
            print(f'Tau is NaN or Inf')
        return False
    if results_data['alpha'] < 0:
        if verbose:
            print(f'Alpha is less than 0')
        return False

    return True

def get_mcfarland_analysis_for_dataset_old(date, subject, cache = False):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/mcfarland_analysis/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    mcfarland_results_path = cache_dir / f'mcfarland_results.npz'
    if cache and mcfarland_results_path.exists():
        mcfarland_results = np.load(mcfarland_results_path, allow_pickle=True)
        return mcfarland_results['mcfarland_results'].item()

    mcfarland_dir = Path(f'/mnt/ssd/YatesMarmoV1/tmp/mcfarland_analysis/{subject}_{date}')
    unit_folders  = list(mcfarland_dir.glob('*'))
    all_results = {}
    for unit_folder in unit_folders:
        result_data_path = unit_folder / f'mcfarland_{unit_folder.name}.npz'
        result_data = np.load(result_data_path, allow_pickle=True)
        all_results[int(unit_folder.name)] = result_data['results_data'].item()
        all_results[int(unit_folder.name)]['valid'] = check_if_valid_mcfarland_analysis(all_results[int(unit_folder.name)])
        all_results[int(unit_folder.name)]['tau'] = all_results[int(unit_folder.name)]['exponenetial_fit']['tau']

    np.savez(mcfarland_results_path, mcfarland_results=all_results)
    return all_results

def get_mcfarland_analysis_for_dataset(date, subject, cache = False):
    cache = False
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/mcfarland_analysis_processed_all_bins/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    mcfarland_results_path = cache_dir / f'mcfarland_results.npz'
    if cache and mcfarland_results_path.exists():
        mcfarland_results = np.load(mcfarland_results_path, allow_pickle=True)
        return mcfarland_results['mcfarland_results'].item()
    all_results_all_bins = {}
    for bins in [10, 37]:

        mcfarland_file = Path(f'/mnt/ssd/YatesMarmoV1/metrics/mcfarland_analysis_{bins}_bins/{subject}_{date}/mcfarland_analysis.npz')
        all_results_preprocessed = np.load(mcfarland_file, allow_pickle=True)['unit_to_mcfarland_out'].item()
        all_results_data = {}
        all_results_ln = {}
        all_results_energy = {}

        all_results = {}
        for cid, results in all_results_preprocessed.items():
            all_results_data[cid] = results['mcfarland_robs_out']
            all_results_data[cid]['valid'] = check_if_valid_mcfarland_analysis(all_results_data[cid])
            all_results_ln[cid] = results['mcfarland_rhat_ln_out']
            all_results_energy[cid] = results['mcfarland_rhat_energy_out']
            if all_results_ln[cid] is None:
                all_results_ln[cid] = {'valid': False, 'tau': np.nan, 'alpha': np.nan}
            else:
                all_results_ln[cid]['valid'] = all_results_data[cid]['valid']
            if all_results_energy[cid] is None:
                all_results_energy[cid] = {'valid': False, 'tau': np.nan, 'alpha': np.nan}
            else:
                all_results_energy[cid]['valid'] = all_results_data[cid]['valid']
            
            assert 'valid' in all_results_data[cid].keys() and 'tau' in all_results_data[cid].keys() and 'alpha' in all_results_data[cid].keys()

            all_results[cid] = {
                'mcfarland_robs_out': all_results_data[cid],
                'mcfarland_rhat_ln_out': all_results_ln[cid],
                'mcfarland_rhat_energy_out': all_results_energy[cid],
            }
        all_results_all_bins[bins] = all_results

        
    # unit_folders  = list(mcfarland_dir.glob('*'))
    # all_results = {}
    # for unit_folder in unit_folders:
    #     result_data_path = unit_folder / f'mcfarland_{unit_folder.name}.npz'
    #     result_data = np.load(result_data_path, allow_pickle=True)
    #     all_results[int(unit_folder.name)] = result_data['results_data'].item()
    #     all_results[int(unit_folder.name)]['valid'] = check_if_valid_mcfarland_analysis(all_results[int(unit_folder.name)])
    #     all_results[int(unit_folder.name)]['tau'] = all_results[int(unit_folder.name)]['exponenetial_fit']['tau']

    np.savez(mcfarland_results_path, mcfarland_results=all_results_all_bins)
    return all_results_all_bins
all_results = get_mcfarland_analysis_for_dataset('2022-04-13', 'Allen', cache = False)
# all_results = get_mcfarland_analysis_for_dataset_old('2022-04-13', 'Allen')
#%%
def plot_mcfarland_analysis_for_unit(cid, all_results, contour_metrics = None, gaussian_fit_metrics = None, ax = None, bins = 37, show_ln = False, show_energy = False, show_fit = True):
    results_data = all_results[bins][cid]['mcfarland_robs_out']
    results_fit = results_data['exponenetial_fit']
    
    # Get model results if requested
    results_ln = all_results[bins][cid]['mcfarland_rhat_ln_out'] if show_ln else None
    results_energy = all_results[bins][cid]['mcfarland_rhat_energy_out'] if show_energy else None
    # Plot McFarland analysis comparison between data and model
    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    else:
        axs = ax
        fig = ax.figure

    # Colors for different metrics
    binned_color = 'C0'
    cumulative_color = 'C1'
    em_color = 'green'
    raw_color = 'red'

    # Line styles for data vs simulation
    data_style = '-'
    best_style = '--'
    mid_style = '-.'
    low_style = ':'

    # # Top plot: Raw variance values
    # # Binned rate variance
    # axs[0].plot(results_data['ep_bins'], results_data['bin_rate_vars'], 'o-', 
    #         color=binned_color, linestyle=data_style, label='Binned EM-RV (Data)', linewidth=3, markersize=10)


    # # Horizontal lines for EM-corrected and raw rate variance
    # axs[0].axhline(results_data['em_corrected_var'], color=em_color, linestyle=data_style, 
    #             zorder=0)


    # axs[0].axhline(results_data['rate_var'], color=raw_color, linestyle=data_style, 
    #             zorder=0)


    # axs[0].set_ylabel('Variance (spikes²)')
    axs.set_title(f'McFarland Analysis - Unit {cid}')
    # lines = [
    #     plt.Line2D([0], [0], color=binned_color, linestyle=data_style, label='Data'),
    #     plt.Line2D([0], [0], color=binned_color, linestyle=best_style, label='Most Reg'),
    #     plt.Line2D([0], [0], color=binned_color, label='Binned EM-RV'),
    #     plt.Line2D([0], [0], color=em_color, label='EM-RV'),
    #     plt.Line2D([0], [0], color=raw_color, label='Raw RV'),
    # ]
    # axs[0].legend(handles=lines, loc='lower left')
    # axs[0].grid(True, alpha=0.5)

    # Bottom plot: Normalized variance (divided by EM-corrected variance)
    # Binned rate variance (normalized)
    axs.plot(results_data['ep_bins'], 
            np.array(results_data['bin_rate_vars'])/results_data['em_corrected_var'], 'o-', 
            color=binned_color, linestyle=data_style, linewidth=3, markersize=10, label='Data')
    
    # Plot data fit if requested
    if show_fit:
        axs.plot(results_data['ep_bins'], results_fit['y_fit'], 'r-', linewidth=3, label='Data Fit')
    
    # Plot LN model if requested
    if show_ln and results_ln is not None and results_ln['valid']:
        ln_fit = results_ln['exponenetial_fit']
        axs.plot(results_ln['ep_bins'], 
                np.array(results_ln['bin_rate_vars'])/results_ln['em_corrected_var'], 's--', 
                color='orange', linewidth=2, markersize=6, alpha=0.7, label='LN Model')
        if show_fit:
            axs.plot(results_ln['ep_bins'], ln_fit['y_fit'], 'orange', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot Energy model if requested
    if show_energy and results_energy is not None and results_energy['valid']:
        energy_fit = results_energy['exponenetial_fit']
        axs.plot(results_energy['ep_bins'], 
                np.array(results_energy['bin_rate_vars'])/results_energy['em_corrected_var'], '^--', 
                color='purple', linewidth=2, markersize=6, alpha=0.7, label='Energy Model')
        if show_fit:
            axs.plot(results_energy['ep_bins'], energy_fit['y_fit'], 'purple', linestyle='--', linewidth=2, alpha=0.7)

    if contour_metrics is not None:
        #plot a vertical line at contour_metrics[cid]['sqrt_area_contour_deg']
        axs.axvline(contour_metrics[cid]['sqrt_area_contour_deg'], color='k', linestyle='--', label='Square Root of Contour Area')
    if gaussian_fit_metrics is not None:
        #plot a vertical line at gaussian_fit_metrics[cid]['length_gaussian_deg']
        axs.axvline(gaussian_fit_metrics[cid]['length_gaussian_deg'], color='g', linestyle='--', label='Length of Gaussian')

    # Horizontal lines for normalized EM-corrected and raw rate variance
    # axs[1].axhline(1.0, color=em_color, linestyle=data_style, zorder=0)  # EM-corrected normalized to 1.0
    # axs[1].axhline(1.0, color=em_color, linestyle=best_style, zorder=0)
    # axs[1].axhline(1.0, color=em_color, linestyle=mid_style, zorder=0)
    # axs[1].axhline(1.0, color=em_color, linestyle=low_style, zorder=0)

    # axs[1].axhline(results_data['rate_var']/results_data['em_corrected_var'], 
    #             color=raw_color, linestyle=data_style, zorder=0)

    axs.set_xlabel('Eye Position Distance (degrees)')
    axs.set_ylabel('Normalized Variance')
    axs.grid(True, alpha=0.5)
    
    # Create legend with plot elements and additional info
    handles, labels = axs.get_legend_handles_labels()
    
    # Add text entries for additional information
    info_handles = [
        mpatches.Patch(color='none', label=f'Data: α = {results_data["alpha"]:.2f}, τ = {results_fit["tau"]:.2f}'),
        mpatches.Patch(color='none', label=f'% Spikes = {results_data["percent_of_total_spikes"]:.3f}, Valid = {results_data["valid"]}')
    ]
    
    # Add model info if requested
    if show_ln and results_ln is not None and results_ln['valid']:
        ln_fit = results_ln['exponenetial_fit']
        info_handles.append(mpatches.Patch(color='none', label=f'LN: α = {results_ln["alpha"]:.2f}, τ = {ln_fit["tau"]:.2f}'))
    if show_energy and results_energy is not None and results_energy['valid']:
        energy_fit = results_energy['exponenetial_fit']
        info_handles.append(mpatches.Patch(color='none', label=f'Energy: α = {results_energy["alpha"]:.2f}, τ = {energy_fit["tau"]:.2f}'))
    
    axs.legend(handles=handles + info_handles, loc='best')

    if ax is None:
        plt.tight_layout()
        plt.show()

