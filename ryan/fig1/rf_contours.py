#%%
"""
Compute STAs/STEs across gaborium datasets and extract RF contours for units above SNR and spike count thresholds.

Usage:
    uv run ryan/rf_contours.py
"""

import fnmatch
from os import wait
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

from DataYatesV1 import (
    get_complete_sessions,
    calc_sta,
    RESULTS_DIR,
    plot_stas,
)

#%% ============================================================================
# Configuration
# ==============================================================================

n_lags = 20
dt = 1 / 240
snr_thresh = 7
spike_thresh = 500  # Minimum number of spikes for inclusion
units_per_page = 15
recalc = False  # Set True to recompute cached STAs/STEs
subjects = ['Logan', 'Allen']

output_dir = RESULTS_DIR / 'rf_summary'
cache_dir = output_dir / 'cache'
output_dir.mkdir(exist_ok=True, parents=True)
cache_dir.mkdir(exist_ok=True, parents=True)

#%%

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR
# Data config (uses the same configs as model training, no weights needed)
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"



#%% ============================================================================
# Phase 1: Compute and cache STAs/STEs per session
# ==============================================================================

sessions = get_complete_sessions()
sessions = [s for s in sessions if any(fnmatch.fnmatch(s.name, f'{subj}*') for subj in subjects)]
print(f'Found {len(sessions)} matching sessions')

all_stas = []
all_stes = []
all_num_spikes = []
all_session_names = []

for session in sessions:
    session_name = session.name
    cache_path = cache_dir / f'{session_name}_sta_ste.npz'

    if cache_path.exists() and not recalc:
        print(f'{session_name}: loading from cache')
        cached = np.load(cache_path)
        all_stas.append(cached['stas'])
        all_stes.append(cached['stes'])
        all_num_spikes.append(cached['num_spikes'])
        all_session_names.append(session_name)
        continue

    print(f'\n{"="*80}')
    print(f'SESSION: {session_name}')
    print(f'{"="*80}')

    dset = session.get_dataset('gaborium')
    if dset is None:
        print(f'  No gaborium dataset found, skipping.')
        continue

    dset['stim'] = dset['stim'].float()
    dset['stim'] = (dset['stim'] - dset['stim'].mean()) / dset['stim'].std()

    print(dset)

    # Compute num_spikes per unit (sum of robs * dpi_valid)
    robs = dset['robs'].numpy() if hasattr(dset['robs'], 'numpy') else dset['robs']
    dpi = dset['dpi_valid'].numpy() if hasattr(dset['dpi_valid'], 'numpy') else dset['dpi_valid']
    if dpi.ndim == 1:
        dpi = dpi[:, None]
    num_spikes = (robs * dpi).sum(axis=0)

    # Compute STAs
    print(f'  Computing STAs ({n_lags} lags)...')
    stas = calc_sta(
        dset['stim'], dset['robs'], n_lags, dset['dpi_valid'],
        device='cuda', batch_size=10000, progress=True,
    ).cpu().numpy()

    # Compute STEs
    print(f'  Computing STEs ({n_lags} lags)...')
    stes = calc_sta(
        dset['stim'], dset['robs'], n_lags, dset['dpi_valid'],
        device='cuda', batch_size=10000,
        stim_modifier=lambda x: x**2, progress=True,
    ).cpu().numpy()

    print(f'  STAs shape: {stas.shape}, STEs shape: {stes.shape}')

    # Cache
    np.savez(cache_path, stas=stas, stes=stes, num_spikes=num_spikes, session_name=session_name)
    print(f'  Cached to {cache_path}')

    all_stas.append(stas)
    all_stes.append(stes)
    all_num_spikes.append(num_spikes)
    all_session_names.append(session_name)

    del dset

print(f'\nLoaded {len(all_stas)} sessions')

#%% ============================================================================
# Phase 2 & 3: Per-subject SNR analysis and STA/STE gallery
# ==============================================================================

col_labels = [f'{t*1000:.0f}ms' for t in np.arange(n_lags) * dt]
max_snr = 100
n_bins = 100

for subject in subjects:
    # Filter sessions for this subject
    subj_idx = [i for i, name in enumerate(all_session_names) if name.startswith(subject)]
    if not subj_idx:
        print(f'\nNo sessions found for {subject}, skipping.')
        continue

    subj_session_names = [all_session_names[i] for i in subj_idx]
    subj_stas = [all_stas[i] for i in subj_idx]
    subj_stes = [all_stes[i] for i in subj_idx]

    stas_all = np.concatenate(subj_stas, axis=0)
    stes_all = np.concatenate(subj_stes, axis=0)
    n_units_total = stas_all.shape[0]

    subj_prefix = subject.lower()

    print(f'\n{"="*80}')
    print(f'SUBJECT: {subject} — {len(subj_session_names)} sessions, {n_units_total} units')
    print(f'{"="*80}')

    # --- SNR computation ---
    signal = np.abs(stes_all - np.median(stes_all, axis=(2, 3), keepdims=True))
    sigma = [0, 1, 1, 1]
    signal = gaussian_filter(signal, sigma)
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]

    cluster_snr = snr_per_lag.max(axis=1)
    cluster_lag = snr_per_lag.argmax(axis=1)
    cluster_mask = cluster_snr > snr_thresh
    good_clusters = np.where(cluster_mask)[0]

    print(f'Units above SNR threshold ({snr_thresh}): {len(good_clusters)}/{n_units_total}')

    # --- SNR vs lag plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snr_per_lag.T, alpha=0.3, linewidth=0.5)
    ax.axhline(snr_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold = {snr_thresh}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('SNR')
    ax.set_title(f'SNR vs Lag — {n_units_total} units from {len(subj_session_names)} {subject} sessions')
    ax.legend()
    ax.set_ylim(0, max_snr)
    fig.tight_layout()
    fig.savefig(output_dir / f'{subj_prefix}_snr_vs_lag.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- SNR histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cluster_snr, bins=np.linspace(0, max_snr, n_bins),
            edgecolor='black', alpha=0.7)
    ax.axvline(snr_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold = {snr_thresh}')
    ax.set_xlabel('Max SNR across lags')
    ax.set_ylabel('Count')
    ax.set_title(f'SNR Distribution — {len(good_clusters)}/{n_units_total} above threshold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f'{subj_prefix}_snr_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- STA/STE PDF Gallery ---

    if not (output_dir / f'{subj_prefix}_sta_ste_gallery.pdf').exists():
        print(f'\nGenerating {subject} STA/STE gallery PDF...')
        with PdfPages(output_dir / f'{subj_prefix}_sta_ste_gallery.pdf') as pdf:
            for session_name, stas, stes in zip(subj_session_names, subj_stas, subj_stes):
                n_units = stas.shape[0]
                n_pages = int(np.ceil(n_units / units_per_page))
                print(f'  {session_name}: {n_units} units, {n_pages} page(s)')

                for page in range(n_pages):
                    start = page * units_per_page
                    end = min(start + units_per_page, n_units)

                    # Interleave STA and STE rows: STA0, STE0, STA1, STE1, ...
                    sta_batch = stas[start:end,:,None]
                    ste_batch = stes[start:end,:,None]
                    ste_batch = ste_batch - np.median(ste_batch, axis=(2, 3), keepdims=True)  # Center STEs for better visualization
                    interleaved = np.empty((2 * (end - start), *sta_batch.shape[1:]), dtype=sta_batch.dtype)
                    interleaved[0::2] = sta_batch
                    interleaved[1::2] = ste_batch

                    row_labels = []
                    for c in range(start, end):
                        row_labels.append(f'cid {c} STA')
                        row_labels.append(f'cid {c} STE')

                    fig, ax = plt.subplots(figsize=(12, .8 * (end - start) + 2))
                    plot_stas(interleaved, row_labels=row_labels, col_labels=col_labels, ax=ax)
                    fig.suptitle(f'{session_name} (cid {start}-{end-1})', fontsize=14, y=1.02)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
    else:
        print(f'\n{subject} gallery PDF already exists, skipping generation.')


#%% ============================================================================
# Phase 2.5: SNR vs num_spikes scatter (both subjects, same axes)
# ==============================================================================

max_snr = 150
snr_thresh = 9
spike_thresh = 200
fig, ax = plt.subplots(figsize=(10, 7))
subject_colors = {'Logan': 'tab:blue', 'Allen': 'tab:orange'}

for subject in subjects:
    subj_idx = [i for i, name in enumerate(all_session_names) if name.startswith(subject)]
    if not subj_idx:
        continue
    subj_stes = [all_stes[i] for i in subj_idx]
    subj_spikes = np.concatenate([all_num_spikes[i] for i in subj_idx])
    stes_all = np.concatenate(subj_stes, axis=0)

    signal = np.abs(stes_all - np.median(stes_all, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 1, 1, 1])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    cluster_snr = snr_per_lag.max(axis=1)

    ax.scatter(subj_spikes, cluster_snr, s=10, alpha=0.5,
               color=subject_colors[subject], label=subject)

# set x scale to be log
ax.set_xscale('log')
ax.axhline(snr_thresh, color='red', linestyle='--', linewidth=1.5, label=f'SNR thresh = {snr_thresh}')
ax.axvline(spike_thresh, color='green', linestyle='--', linewidth=1.5, label=f'Spike thresh = {spike_thresh}')
ax.set_xlabel('Number of spikes')
ax.set_ylabel('Max SNR across lags')
ax.set_title('SNR vs Spike Count — Unit Inclusion Criteria')
ax.legend()
ax.set_ylim(0, max_snr)
fig.tight_layout()
fig.savefig(output_dir / 'snr_vs_spikes.png', dpi=150, bbox_inches='tight')
plt.show()

#%% ============================================================================
# Phase 3: Per-session peak-lag STE grids and RF contour extraction
# ==============================================================================

from DataYatesV1.utils.rf import get_contour
from matplotlib.lines import Line2D

# Collect contours across ALL subjects for the combined plot
all_contours_deg = []  # list of (session_name, contour_deg_array)

for subject in subjects:
    subj_idx = [i for i, name in enumerate(all_session_names) if name.startswith(subject)]
    if not subj_idx:
        continue

    subj_session_names = [all_session_names[i] for i in subj_idx]
    subj_stas = [all_stas[i] for i in subj_idx]
    subj_stes = [all_stes[i] for i in subj_idx]
    subj_spikes_list = [all_num_spikes[i] for i in subj_idx]

    subj_prefix = subject.lower()

    # Recompute per-subject SNR to get peak lags
    stes_all = np.concatenate(subj_stes, axis=0)
    spikes_all = np.concatenate(subj_spikes_list)
    signal = np.abs(stes_all - np.median(stes_all, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 1, 1, 1])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    cluster_snr_all = snr_per_lag.max(axis=1)
    cluster_lag_all = snr_per_lag.argmax(axis=1)

    # ---- Step 1: Per-session peak-lag STE grid (audit plot) ----

    offset = 0  # running unit offset into the concatenated arrays

    for sess_i, (session_name, stas_s, stes_s, spikes_s) in enumerate(
        zip(subj_session_names, subj_stas, subj_stes, subj_spikes_list)
    ):
        n_units = stas_s.shape[0]
        snr_slice = cluster_snr_all[offset:offset + n_units]
        lag_slice = cluster_lag_all[offset:offset + n_units]
        spikes_slice = spikes_s

        # Sort units by descending SNR
        order = np.argsort(-snr_slice)

        # Grid layout
        ncols = int(np.ceil(np.sqrt(n_units)))
        nrows = int(np.ceil(n_units / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axes = np.atleast_2d(axes)
        fig.suptitle(f'{session_name} — Peak-lag STE (sorted by SNR)', fontsize=12)

        for plot_i, uid in enumerate(order):
            ax = axes.flat[plot_i]
            peak_lag = lag_slice[uid]
            ste_img = stes_s[uid, peak_lag]
            ste_img_centered = ste_img - np.median(ste_img)

            vmax = np.max(np.abs(ste_img_centered))
            ax.imshow(ste_img_centered, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                      interpolation='none')

            snr_val = snr_slice[uid]
            spike_val = spikes_slice[uid]
            is_good = (snr_val > snr_thresh) and (spike_val > spike_thresh)
            color = 'red' if is_good else 'gray'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{uid} SNR={snr_val:.1f}', fontsize=6, color=color)

        # Turn off unused axes
        for plot_i in range(n_units, nrows * ncols):
            axes.flat[plot_i].axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f'{subj_prefix}_{session_name}_peak_ste_grid.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'Saved peak-lag STE grid for {session_name}')

        # ---- Step 2: Extract contours for good units ----

        # Load dataset metadata for pixel-to-degree conversion
        session_obj = [s for s in sessions if s.name == session_name][0]
        dset = session_obj.get_dataset('gaborium')
        if dset is None:
            offset += n_units
            continue
        roi_src = dset.metadata['roi_src']  # [[row_min, row_max], [col_min, col_max]]
        ppd = dset.metadata['ppd']
        del dset

        # ROI origin in full-screen pixel coords
        roi_origin = roi_src[:, 0]  # (i_min, j_min)

        for uid in range(n_units):
            if snr_slice[uid] <= snr_thresh or spikes_slice[uid] <= spike_thresh:
                continue

            peak_lag = lag_slice[uid]
            ste_img = stes_s[uid, peak_lag]
            ste_centered = ste_img - np.median(ste_img)

            # Skip contrast suppressed units (where the "hot" region is negative in the centered STE)
            if ste_centered.max() < abs(ste_centered.min()):
                continue

            # Normalize to [0, 1] for contour extraction
            ptp = np.ptp(ste_centered)
            if ptp < 1e-8:
                continue
            ste_norm = (ste_centered - ste_centered.min()) / ptp

            try:
                contour, area, ctr = get_contour(ste_norm, 0.5)
            except Exception:
                continue


            # contour is (N, 2) in (row, col) within the STE image
            # Convert to full-screen pixel coords, then to degrees
            contour_pix = contour + roi_origin[None, :]  # (N, 2) in (i, j) screen pixels
            # Convert to degrees: (pixel - center) / ppd, flip i axis
            contour_deg = contour_pix  / ppd
            contour_deg[:, 0] *= -1  # flip vertical so up is positive

            all_contours_deg.append((session_name, contour_deg))

        offset += n_units

# ---- Combined RF contour plot (both subjects on same axes) ----
#%%


if all_contours_deg:
    fig, ax = plt.subplots(figsize=(8, 8))
    unique_sessions = list(dict.fromkeys(name for name, _ in all_contours_deg))
    cmap = plt.cm.tab10
    session_colors = {name: cmap(i % 10) for i, name in enumerate(unique_sessions)}

    for session_name, contour_deg in all_contours_deg:
        # Plot as (j_deg=horizontal, i_deg=vertical) -> (col, row) -> (x, y)
        ax.plot(contour_deg[:, 1], contour_deg[:, 0],
                color=session_colors[session_name], alpha=0.3, linewidth=0.8)

    # Legend with one entry per session
    legend_handles = [
        Line2D([0], [0], color=session_colors[s], linewidth=2, label=s)
        for s in unique_sessions
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='best')
    ax.set_xlabel('Horizontal position (deg)')
    ax.set_ylabel('Vertical position (deg)')
    n_contours = len(all_contours_deg)
    ax.set_title(f'All subjects — RF contours ({n_contours} units, SNR > {snr_thresh}, spikes > {spike_thresh})')
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'rf_contours_deg_combined.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved combined RF contour plot: {n_contours} contours')
else:
    print('No contours extracted')

#%% ============================================================================
# Summary
# ==============================================================================

print(f'\n{"="*80}')
print(f'RF SUMMARY COMPLETE')
print(f'{"="*80}')
print(f'Subjects:        {", ".join(subjects)}')
print(f'Total sessions:  {len(all_session_names)}')
print(f'SNR threshold:   {snr_thresh}')
print(f'Spike threshold: {spike_thresh}')
print(f'Output dir:      {output_dir}')
print(f'Cache dir:       {cache_dir}')
print(f'Combined:        snr_vs_spikes.png, rf_contours_deg_combined.png')
for subject in subjects:
    print(f'  {subject}:       {subject.lower()}_snr_vs_lag.png, {subject.lower()}_snr_histogram.png, {subject.lower()}_sta_ste_gallery.pdf')
    print(f'               {subject.lower()}_<session>_peak_ste_grid.png')

