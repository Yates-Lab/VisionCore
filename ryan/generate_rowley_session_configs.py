#%%
"""
Generate Rowley session config YAMLs with visual unit selection and QC metrics.

For each session:
  1. Evaluate all available eyes via sum(ReLU(SNR - threshold)) heuristic
  2. Pick the best eye automatically
  3. Compute visual unit selection (SNR + spike count thresholds)
  4. QC metrics (missing %) from amplitude truncation
  5. Save YAML with cids (=visual), visual, qcmissing

Diagnostic plots per session (all available eyes):
  - Scatter: num_spikes (log) vs SNR
  - Peak-lag STE grid sorted by SNR, visual units in red
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from DataRowleyV1V2.data.registry import RowleySession, V1_SESSIONS
from DataRowleyV1V2.utils.datasets import DictDataset
from DataRowleyV1V2.utils.rf import calc_sta
from DataRowleyV1V2.shifter.preprocess import normalize_stimulus, create_valid_eyepos_mask

#%%
np.random.seed(1002)

# Configuration
n_lags = 20
snr_thresh = 5
spike_thresh = 100
missing_thresh = 25
valid_eyepos_radius = 7.5  # degrees
batch_size = 10000
device = 'cpu'
region = 'V1'

from VisionCore.paths import VISIONCORE_ROOT
output_dir = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "sessions"
output_dir.mkdir(exist_ok=True, parents=True)

# Manual filter: only include these sessions for now
ALLOWED_SESSIONS = [
    #"Luke_2025-08-04",
    #"Luke_2025-08-05",
    "Luke_2026-03-02",
]

sessions = [s for s in V1_SESSIONS if s['session_name'] in ALLOWED_SESSIONS]
print(f"Found {len(sessions)} sessions (filtered from {len(V1_SESSIONS)} total)")


#%% Helper: compute SNR from a gaborium dataset

def compute_snr_from_gaborium(dset_path, region='V1'):
    """Load gaborium dataset and compute per-unit SNR from STEs.

    Returns dict with keys: stes, cluster_snr, cluster_lag, num_spikes, robs, dset.
    Returns None if dataset missing.
    """
    if not dset_path.exists():
        return None

    dset = DictDataset.load(dset_path)

    # Filter to region
    if 'region' in dset.metadata:
        regions = dset.metadata['region']
        region_mask = regions == region
        robs = dset['robs'][:, region_mask]
    else:
        robs = dset['robs']

    # Preprocess stimulus and valid mask
    stim = normalize_stimulus(dset['stim'])
    dfs = create_valid_eyepos_mask(dset['eyepos'], dset['dpi_valid'], valid_eyepos_radius).float()

    # Compute STEs
    stes = calc_sta(
        stim, robs, n_lags, dfs,
        device=device, batch_size=batch_size,
        stim_modifier=lambda x: x**2,
        progress=True,
    ).cpu().numpy()

    signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, sigma=[0, 0, 4, 4])
    noise = np.median(signal[:, 0], axis=(1, 2))
    noise = np.maximum(noise, 1e-10)  # avoid division by zero
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    cluster_snr = snr_per_lag.max(axis=1)
    cluster_lag = snr_per_lag.argmax(axis=1)

    # Num spikes
    robs_np = robs.numpy() if hasattr(robs, 'numpy') else np.asarray(robs)
    dpi = dset['dpi_valid'].numpy() if hasattr(dset['dpi_valid'], 'numpy') else np.asarray(dset['dpi_valid'])
    if dpi.ndim == 1:
        dpi = dpi[:, None]
    num_spikes = (robs_np * dpi).sum(axis=0)

    return {
        'stes': stes, 'cluster_snr': cluster_snr, 'cluster_lag': cluster_lag,
        'num_spikes': num_spikes, 'robs': robs, 'dset': dset,
    }


#%% Process all sessions

all_stats = []

for session_config in tqdm(sessions, desc="Processing sessions"):
    session_name = session_config['session_name']
    eyes = session_config['eyes']
    sess = RowleySession(session_name)

    # --- Phase 1: evaluate all eyes, plot diagnostics, pick the best ---
    eye_scores = {}
    eye_results = {}

    for eye in eyes:
        dset_path = sess.processed_path / 'datasets' / f'{eye}_eye' / 'gaborium.dset'
        result = compute_snr_from_gaborium(dset_path, region=region)
        if result is None:
            print(f"  [SKIP] {session_name} / {eye} eye: gaborium.dset not found")
            continue

        cluster_snr = result['cluster_snr']
        cluster_lag = result['cluster_lag']
        num_spikes = result['num_spikes']
        stes = result['stes']
        n_units = result['robs'].shape[1]

        vis_mask = (cluster_snr > snr_thresh) & (num_spikes >= spike_thresh)
        visual_units = np.where(vis_mask)[0]

        score = float(np.sum(np.maximum(cluster_snr - snr_thresh, 0)))
        eye_scores[eye] = score
        eye_results[eye] = result
        print(f"  {session_name} / {eye} eye: score = {score:.1f} "
              f"({np.sum(cluster_snr > snr_thresh)}/{n_units} above threshold)")

        # --- Plot: SNR vs num_spikes scatter ---
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = np.where(vis_mask, 'red', 'gray')
        ax.scatter(num_spikes, cluster_snr, s=15, c=colors, alpha=0.6)
        ax.set_xscale('log')
        ax.axhline(snr_thresh, color='blue', linestyle='--', lw=1, label=f'SNR = {snr_thresh}')
        ax.axvline(spike_thresh, color='green', linestyle='--', lw=1, label=f'spikes = {spike_thresh}')
        ax.set_xlabel('Number of spikes (gaborium)')
        ax.set_ylabel('Max SNR')
        ax.set_title(f'{session_name} ({eye} eye) — {len(visual_units)}/{n_units} visual, score={score:.1f}')
        ax.legend()
        fig.tight_layout()
        plt.show()

        # --- Plot: Peak-lag STE grid sorted by SNR ---
        order = np.argsort(-cluster_snr)
        ncols = int(np.ceil(np.sqrt(n_units)))
        nrows = int(np.ceil(n_units / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axes = np.atleast_2d(axes)
        fig.suptitle(f'{session_name} ({eye} eye) — Peak-lag STE (sorted by SNR)', fontsize=12)

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
            ax.set_title(f'{uid} SNR={snr_val:.1f} n={spike_val:.0f}',
                         fontsize=5, color=color)

        for plot_i in range(n_units, nrows * ncols):
            axes.flat[plot_i].axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close('all')

    if not eye_scores:
        print(f"  [SKIP] {session_name}: no valid eyes")
        continue

    best_eye = max(eye_scores, key=lambda e: eye_scores[e])
    print(f"  -> Best eye: {best_eye} (score {eye_scores[best_eye]:.1f})")

    # --- Phase 2: generate config for best eye only ---
    r = eye_results[best_eye]
    cluster_snr = r['cluster_snr']
    num_spikes = r['num_spikes']
    dset = r['dset']
    n_units = r['robs'].shape[1]

    print(f"\n{'='*60}")
    print(f"SESSION: {session_name} / {best_eye} eye (selected)")
    print(f"{'='*60}")

    vis_mask = (cluster_snr > snr_thresh) & (num_spikes >= spike_thresh)
    visual_units = np.where(vis_mask)[0]

    # --- QC metrics (amplitude truncation / missing spikes) ---
    try:
        v1_shanks = [k for k, v in session_config['shanks'].items() if v == region]

        qc_cids, qc_time_windows, qc_mpcts = sess.get_missing_pct_qc(shanks=v1_shanks)

        v1_global_cids = sess.get_cluster_ids(shanks=v1_shanks)

        med_missing_pct = np.zeros(n_units)
        for i, cid in enumerate(v1_global_cids):
            unit_mask = qc_cids == cid
            if unit_mask.any():
                med_missing_pct[i] = np.median(qc_mpcts[unit_mask])

        not_missing = np.where(med_missing_pct < missing_thresh)[0]
    except Exception as e:
        print(f"  No QC data: {e}")
        med_missing_pct = np.zeros(n_units)
        not_missing = np.arange(n_units)

    # --- Save YAML ---
    session_yaml = {
        'session': session_name,
        'lab': 'rowley',
        'eye': best_eye,
        'cids': visual_units.tolist(),
        'visual': visual_units.tolist(),
        'qcmissing': not_missing.tolist(),
    }

    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    yaml.add_representer(list, represent_list)

    output_file = output_dir / f"{session_name}.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(session_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {output_file}")

    stats = {
        'session': session_name,
        'eye': best_eye,
        'total': n_units,
        'visual': len(visual_units),
        'qc_missing': len(not_missing),
        'eye_scores': {e: f"{s:.1f}" for e, s in eye_scores.items()},
    }
    all_stats.append(stats)
    print(f"  {stats['visual']}/{stats['total']} visual, "
          f"{stats['qc_missing']} pass missing")

    del eye_results
    plt.close('all')

#%% Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
total_units = sum(s['total'] for s in all_stats)
total_visual = sum(s['visual'] for s in all_stats)
print(f"Sessions: {len(all_stats)}")
print(f"Total V1 units: {total_units}")
print(f"Visual units: {total_visual} ({100*total_visual/total_units:.1f}%)")
print(f"\nPer session:")
for s in all_stats:
    pct = 100 * s['visual'] / s['total'] if s['total'] > 0 else 0
    print(f"  {s['session']} ({s['eye']} eye): {s['visual']}/{s['total']} visual ({pct:.1f}%)")
    print(f"    Eye scores: {s['eye_scores']}")
# %%
