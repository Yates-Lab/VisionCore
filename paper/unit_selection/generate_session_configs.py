#%%
"""
Generate session config YAMLs with visual unit selection and QC metrics.

For each session:
  1. Compute STEs -> SNR, num_spikes from gaborium dataset
  2. Visual units = SNR > threshold AND num_spikes >= spike_threshold
  3. QC metrics (missing %, contamination %) from spike sorting
  4. Save YAML with cids (=visual), visual, qcmissing, qccontam

Diagnostic plots per session:
  - Scatter: num_spikes (log) vs SNR
  - Peak-lag bar plot sorted by SNR, visual units in red
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from DataYatesV1 import (
    DictDataset, get_complete_sessions, get_gaborium_sta_ste,
    set_seeds, calc_sta, RESULTS_DIR,
)

#%%
set_seeds(1002)

# Configuration
n_lags = 20
snr_thresh = 5
spike_thresh = 100
missing_thresh = 25
contam_thresh = 50

output_dir = Path("experiments/dataset_configs/sessions")
output_dir.mkdir(exist_ok=True, parents=True)

#plot_dir = RESULTS_DIR / 'session_config_plots'
#plot_dir.mkdir(exist_ok=True, parents=True)

sessions = get_complete_sessions()
print(f"Found {len(sessions)} sessions")

#%% Process all sessions

all_stats = []

for sess in tqdm(sessions, desc="Processing sessions"):
    print(f"\n{'='*60}")
    print(f"SESSION: {sess.name}")
    print(f"{'='*60}")

    # --- STAs/STEs and SNR ---
    stas, stes = get_gaborium_sta_ste(sess, n_lags)
    n_units = stas.shape[0]

    signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 2, 2, 2])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    cluster_snr = snr_per_lag.max(axis=1)
    cluster_lag = snr_per_lag.argmax(axis=1)

    # --- Num spikes from gaborium dataset ---
    dset = DictDataset.load(sess.sess_dir / 'datasets' / 'gaborium.dset')
    robs = dset['robs'].numpy() if hasattr(dset['robs'], 'numpy') else dset['robs']
    dpi = dset['dpi_valid'].numpy() if hasattr(dset['dpi_valid'], 'numpy') else dset['dpi_valid']
    if dpi.ndim == 1:
        dpi = dpi[:, None]
    num_spikes = (robs * dpi).sum(axis=0)
    del dset

    # --- Visual unit selection ---
    vis_mask = (cluster_snr > snr_thresh) & (num_spikes >= spike_thresh)
    visual_units = np.where(vis_mask)[0]

    # --- QC metrics (spike sorting quality) ---
    try:
        spike_clusters = sess.ks_results.spike_clusters
        cids = np.unique(spike_clusters)

        refractory = np.load(sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz')
        min_contam_props = refractory['min_contam_props']
        contam_pct = np.array([np.min(min_contam_props[i]) * 100 for i in range(len(cids))])

        truncation = np.load(sess.sess_dir / 'qc' / 'amp_truncation' / 'truncation.npz')
        med_missing_pct = np.array([
            np.median(truncation['mpcts'][truncation['cid'] == i])
            for i in range(len(cids))
        ])

        not_missing = np.where(med_missing_pct < missing_thresh)[0]
        not_contaminated = np.where(contam_pct < contam_thresh)[0]
    except Exception as e:
        print(f"  No spike sorting QC: {e}")
        med_missing_pct = np.zeros(n_units)
        contam_pct = np.zeros(n_units)
        not_missing = np.arange(n_units)
        not_contaminated = np.arange(n_units)

    # --- Save YAML ---
    session_config = {
        'session': sess.name,
        'cids': visual_units.tolist(),
        'visual': visual_units.tolist(),
        'qcmissing': not_missing.tolist(),
        'qccontam': not_contaminated.tolist(),
    }

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    yaml.add_representer(list, represent_list)

    output_file = output_dir / f"{sess.name}.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(session_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {output_file}")

    # --- Plot 1: SNR vs num_spikes scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(vis_mask, 'red', 'gray')
    ax.scatter(num_spikes, cluster_snr, s=15, c=colors, alpha=0.6)
    ax.set_xscale('log')
    ax.axhline(snr_thresh, color='blue', linestyle='--', lw=1, label=f'SNR = {snr_thresh}')
    ax.axvline(spike_thresh, color='green', linestyle='--', lw=1, label=f'spikes = {spike_thresh}')
    ax.set_xlabel('Number of spikes (gaborium)')
    ax.set_ylabel('Max SNR')
    ax.set_title(f'{sess.name} — {len(visual_units)}/{n_units} visual units')
    ax.legend()
    fig.tight_layout()
    #fig.savefig(plot_dir / f'{sess.name}_snr_vs_spikes.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Peak-lag STE grid sorted by SNR, visual units in red ---
    order = np.argsort(-cluster_snr)

    ncols = int(np.ceil(np.sqrt(n_units)))
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.atleast_2d(axes)
    fig.suptitle(f'{sess.name} — Peak-lag STE (sorted by SNR)', fontsize=12)

    for plot_i, uid in enumerate(order):
        ax = axes.flat[plot_i]
        peak_lag = cluster_lag[uid]
        ste_img = stes[uid, peak_lag]
        ste_img_centered = ste_img - np.median(ste_img)

        vmax = np.max(np.abs(ste_img_centered))
        ax.imshow(ste_img_centered, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                  interpolation='none')

        snr_val = cluster_snr[uid]
        spike_val = num_spikes[uid]
        is_good = vis_mask[uid]
        color = 'red' if is_good else 'gray'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{uid} SNR={snr_val:.1f} n={spike_val:.0f}\n'
                     f'miss={med_missing_pct[uid]:.0f}% cont={contam_pct[uid]:.0f}%',
                     fontsize=5, color=color)

    for plot_i in range(n_units, nrows * ncols):
        axes.flat[plot_i].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    #fig.savefig(plot_dir / f'{sess.name}_peak_ste_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

    stats = {
        'session': sess.name,
        'total': n_units,
        'visual': len(visual_units),
        'qc_missing': len(not_missing),
        'qc_contam': len(not_contaminated),
    }
    all_stats.append(stats)
    print(f"  {stats['visual']}/{stats['total']} visual, "
          f"{stats['qc_missing']} pass missing, {stats['qc_contam']} pass contam")
    plt.close('all')

#%% Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
total_units = sum(s['total'] for s in all_stats)
total_visual = sum(s['visual'] for s in all_stats)
print(f"Sessions: {len(all_stats)}")
print(f"Total units: {total_units}")
print(f"Visual units: {total_visual} ({100*total_visual/total_units:.1f}%)")
print(f"\nPer session:")
for s in all_stats:
    pct = 100 * s['visual'] / s['total'] if s['total'] > 0 else 0
    print(f"  {s['session']}: {s['visual']}/{s['total']} visual ({pct:.1f}%)")
# %%
