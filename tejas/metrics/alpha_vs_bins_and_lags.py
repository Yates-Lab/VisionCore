#%%
from pathlib import Path
from DataYatesV1 import get_session
import numpy as np
from tejas.metrics import qc
from tejas.metrics.qc import get_qc_units_for_session
import seaborn as sns
import pandas as pd
from DataYatesV1 import get_complete_sessions
from tejas.metrics.mcfarland import check_if_valid_mcfarland_analysis

def get_all_units_alphas_for_lag_and_bin(subject, date, cache = True, peak_lag = False):
    sess = get_session(subject, date)
    

    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/alpha_vs_bins_and_lags{"_peak_lag" if peak_lag else ""}/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'all_units_alphas_for_lag_and_bin.npz'
    
    if cache and cache_path.exists():
        all_units_alphas_for_lag_and_bin = np.load(cache_path, allow_pickle=True)['all_units_alphas_for_lag_and_bin']
        bins = np.load(cache_path, allow_pickle=True)['bins']
        lags = np.load(cache_path, allow_pickle=True)['lags']
        return all_units_alphas_for_lag_and_bin, bins, lags
    
    all_results_cache_path = Path(f'/mnt/sata/YatesMarmoV1/metrics/mcfarland_analysis_ALLbins_ALLlags{"_peak_lag" if peak_lag else ""}/{sess.name}')
    all_results_cache_path = all_results_cache_path / 'mcfarland_analysis.npz'
    if not all_results_cache_path.exists():
        return None, None, None
    # assert all_results_cache_path.exists(), f'Cache path {all_results_cache_path} does not exist'
    mcfarland_analysis = np.load(all_results_cache_path, allow_pickle=True)['unit_to_mcfarland_out'].item()
    date = sess.name.split('_')[1]
    subject = sess.name.split('_')[0]

    
 
    qc_units = get_qc_units_for_session(date, subject, cache = True)


    all_units_alphas_for_lag_and_bin = []
    for unit_idx in qc_units:
        mcfarland_robs_out_lags_and_bins = mcfarland_analysis[unit_idx]['mcfarland_robs_out_lags_and_bins']
        alphas_for_lag_and_bin = []
        lags = []
        for lag in mcfarland_robs_out_lags_and_bins.keys():
            alphas_for_bin = []
            bins = []
            for bin in mcfarland_robs_out_lags_and_bins[lag].keys():
                alpha = mcfarland_robs_out_lags_and_bins[lag][bin]['alpha']
                try:
                    valid = check_if_valid_mcfarland_analysis(mcfarland_robs_out_lags_and_bins[lag][bin])
                except:
                    # print(f'Error checking if valid mcfarland analysis for {subject}_{date}_{unit_idx}_{lag}_{bin}')
                    valid = False
                if valid:
                    alphas_for_bin.append(alpha)
                else:
                    alphas_for_bin.append(np.nan)
                
                
                bins.append(bin)
            alphas_for_lag_and_bin.append(alphas_for_bin)
            lags.append(lag)
        
        lags = np.array(lags)
        bins = np.array(bins)

        all_units_alphas_for_lag_and_bin.append(alphas_for_lag_and_bin)
    all_units_alphas_for_lag_and_bin = np.array(all_units_alphas_for_lag_and_bin) #(num_units, num_lags, num_bins)
    np.savez(cache_path, all_units_alphas_for_lag_and_bin=all_units_alphas_for_lag_and_bin, bins=bins, lags=lags)
    return all_units_alphas_for_lag_and_bin, bins, lags
sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05', 'Allen_2022-02-16']
all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]
from tqdm import tqdm
all_sessions_units_alphas = []
for sess in tqdm(all_sessions):
    subject = sess.name.split('_')[0]
    date = sess.name.split('_')[1]
    all_units_alphas_for_lag_and_bin, bins, lags = get_all_units_alphas_for_lag_and_bin(subject, date, cache = False, peak_lag = True)
    if all_units_alphas_for_lag_and_bin is not None:
        all_sessions_units_alphas.append(all_units_alphas_for_lag_and_bin)
all_sessions_units_alphas = np.concatenate(all_sessions_units_alphas, axis=0)
#%%
import matplotlib.pyplot as plt

# Create colormap for different lags
lag_indices = [0, 1, 2, 4, 9, 14, 17]  # Update this list with the lags you want to plot
colors = plt.cm.coolwarm(np.linspace(0, 1, len(lag_indices)))

for color_idx, lag_idx in enumerate(lag_indices):
    lag = lags[lag_idx]
    print(f'Lag: {lag}')
    color = colors[color_idx]

    alphas_for_one_lag_and_bin = all_sessions_units_alphas[:, lag_idx, :]


    # Filter outliers using MAD
    def remove_outliers_mad(data, threshold=2):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return data
        modified_z_scores = 0.6745 * (data - median) / mad
        return data[np.abs(modified_z_scores) < threshold]

    data_for_plot = []
    for bin_idx, bin_val in enumerate(bins):
        bin_data = alphas_for_one_lag_and_bin[:, bin_idx]
        bin_data = bin_data[~np.isnan(bin_data)]
        if len(bin_data) > 0:
            # bin_data_filtered = remove_outliers_mad(bin_data)
            for alpha_val in bin_data:
                data_for_plot.append({'bin': bin_val, '1 - alpha': 1 - alpha_val})

    df = pd.DataFrame(data_for_plot)

    # plt.figure(figsize=(12, 8))
    # sns.swarmplot(data=df, x='bin', y='1 - alpha', size=2)
    plt.xlabel('Number of Bins for Eye Position Distance')
    plt.ylabel('1 - Alpha')

    # Add line connecting means for each bin
    means = df.groupby('bin')['1 - alpha'].mean().values
    stds = df.groupby('bin')['1 - alpha'].std().values
    plt.plot(np.arange(len(bins)), means, color=color, linewidth=2, markersize=6, label=f'Lag: {lag}', zorder=10)

    #use fill_between to show standard deviation per bin
    plt.fill_between(np.arange(len(bins)), means - stds, means + stds, alpha=0.2, color=color)
    #label x-bins every other 10
    plt.xticks(np.arange(0, len(bins), 10), bins[::10])

plt.legend()
plt.title("Mcfarland Analysis on Peak Lag")
plt.tight_layout()
# plt.show()

            
# %%
