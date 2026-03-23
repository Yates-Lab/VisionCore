"""
Generate a 2x2 diagnostic figure showing the weighting mismatch mechanism
in the noise correlation bias analysis.

Panel A: Second moment weighting (n_pairs per trial/time)
Panel B: Old Erate weighting (uniform per valid cell)
Panel C: Mean rate estimate comparison by neuron
Panel D: Shuffle null Dz distributions (old vs fixed)
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from VisionCore.covariance import (
    align_fixrsvp_trials,
    extract_valid_segments,
    extract_windows,
    compute_conditional_second_moments,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
    cov_to_corr,
    get_upper_triangle,
    fit_intercept_linear,
)
from VisionCore.subspace import project_to_psd
from VisionCore.stats import fisher_z_mean
from VisionCore.paths import VISIONCORE_ROOT

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))
from models.config_loader import load_dataset_configs
from models.data import prepare_data

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 1 / 240
T_COUNT = 2
T_HIST = 10
MIN_SEG_LEN = 36
N_BINS = 15
MIN_TOTAL_SPIKES = 500
SESSION_NAME = "Allen_2022-04-13"
DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
N_SHUFFLES = 50


# ── Style ──────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


# ── Helpers ────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        free_mem = []
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                free_mem.append(free)
            except Exception:
                free_mem.append(0)
        best_gpu = int(np.argmax(free_mem))
        if free_mem[best_gpu] < 1e9:
            return "cpu"
        print(f"Selected GPU {best_gpu} with {free_mem[best_gpu]/1e9:.1f} GB free")
        return f"cuda:{best_gpu}"
    return "cpu"


def fz_from_cov(C_noise, use_psd=True):
    if use_psd:
        C_psd = project_to_psd(C_noise)
    else:
        C_psd = C_noise.copy()
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    return fisher_z_mean(tri)


def estimate_rate_covariance_old(SpikeCounts, EyeTraj, T_idx, n_bins=25,
                                  Ctotal=None, intercept_mode='linear'):
    """Original version with trial-count-weighted (torch.nanmean) Erate."""
    MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
        SpikeCounts, EyeTraj, T_idx, n_bins=n_bins
    )
    # OLD: trial-count-weighted mean
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    Ceye = MM - Erate[:, None] * Erate[None, :]

    if intercept_mode == 'linear':
        Crate = fit_intercept_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True)
    elif intercept_mode == 'isotonic':
        from VisionCore.covariance import fit_intercept_pava
        Crate = fit_intercept_pava(Ceye, count_e)
    else:
        Crate = Ceye[0].copy()

    if Ctotal is not None:
        bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
        Crate[bad_mask, :] = np.nan
        Crate[:, bad_mask] = np.nan
        Ceye[:, bad_mask, :] = np.nan
        Ceye[:, :, bad_mask] = np.nan

    return Crate, Erate, Ceye, bin_centers, count_e, bin_edges


# ── Data loading ───────────────────────────────────────────────────────────

def load_session():
    print(f"Loading session {SESSION_NAME}...")
    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    cfg = None
    for c in dataset_configs:
        if c["session"] == SESSION_NAME:
            cfg = c
            break
    assert cfg is not None, f"Session {SESSION_NAME} not found"
    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    train_data, val_data, cfg = prepare_data(cfg, strict=False)
    dset_idx = train_data.get_dataset_index("fixrsvp")
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset, valid_time_bins=120, min_fix_dur=20,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )
    print(f"  Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
          f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")
    return robs, eyepos, valid_mask, neuron_mask, meta


def extract_data(robs, eyepos, valid_mask, device):
    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)
    device_obj = torch.device(device)
    robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=device_obj)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=device_obj)

    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
    print(f"  Found {len(segments)} valid segments")

    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_windows(
        robs_t, eyepos_t, segments, T_COUNT, T_HIST, device=device
    )
    print(f"  Extracted {SpikeCounts.shape[0]} windows, {SpikeCounts.shape[1]} neurons")
    return SpikeCounts, EyeTraj, T_idx, idx_tr


# ── Panel computations ─────────────────────────────────────────────────────

def compute_panel_ab_data(valid_mask):
    """Compute weighting matrices for panels A and B."""
    n_trials, n_time = valid_mask.shape
    # n_t = valid trials at each time bin
    n_t = valid_mask.sum(axis=0)  # (n_time,)

    # Weight matrix for panel A: pair-count weighting
    # weight[i, t] = n_t - 1 (number of other trials for pairing) if valid
    weight_pairs = np.full((n_trials, n_time), np.nan)
    for t in range(n_time):
        for i in range(n_trials):
            if valid_mask[i, t]:
                weight_pairs[i, t] = n_t[t] - 1

    # Weight matrix for panel B: uniform weighting
    weight_uniform = np.full((n_trials, n_time), np.nan)
    weight_uniform[valid_mask] = 1.0

    # Sort trials by duration (longest at top)
    trial_dur = valid_mask.sum(axis=1)
    sort_idx = np.argsort(-trial_dur)

    weight_pairs_sorted = weight_pairs[sort_idx]
    weight_uniform_sorted = weight_uniform[sort_idx]

    return weight_pairs_sorted, weight_uniform_sorted, n_t, trial_dur[sort_idx]


def compute_panel_c_data(SpikeCounts, T_idx):
    """Compute old vs new Erate per neuron."""
    C = SpikeCounts.shape[1]

    # Old: trial-count-weighted (simple average)
    Erate_old = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()

    # New: pair-count-weighted
    unique_times = np.unique(T_idx.detach().cpu().numpy())
    weighted_sum = torch.zeros(C, device=SpikeCounts.device, dtype=torch.float64)
    total_pairs = 0.0
    for t in unique_times:
        mask = (T_idx == t)
        n_t = mask.sum().item()
        if n_t < 10:
            continue
        n_pairs_t = n_t * (n_t - 1) / 2
        mu_t = SpikeCounts[mask].mean(0).to(torch.float64)
        weighted_sum += n_pairs_t * mu_t
        total_pairs += n_pairs_t
    Erate_new = (weighted_sum / total_pairs).detach().cpu().numpy()

    return Erate_old, Erate_new


def compute_panel_d_data(SpikeCounts, EyeTraj, T_idx, Ctotal, fz_U, bin_edges):
    """Run shuffle iterations with both old and new Erate methods."""
    print(f"\nRunning {N_SHUFFLES} shuffle iterations for Panel D...")
    rng = np.random.default_rng(42)

    dz_old_list = []
    dz_new_list = []

    for i in tqdm(range(N_SHUFFLES), desc="Shuffle"):
        perm = rng.permutation(EyeTraj.shape[0])
        EyeTraj_shuff = EyeTraj[perm]

        # Old pipeline (trial-count Erate)
        Crate_old, _, _, _, _, _ = estimate_rate_covariance_old(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_old = 0.5 * ((Ctotal - Crate_old) + (Ctotal - Crate_old).T)
        fz_old = fz_from_cov(CnoiseC_old, use_psd=True)
        dz_old_list.append(fz_old - fz_U)

        # New pipeline (pair-count Erate) — uses the fixed estimate_rate_covariance
        Crate_new, _, _, _, _, _ = estimate_rate_covariance(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_new = 0.5 * ((Ctotal - Crate_new) + (Ctotal - Crate_new).T)
        fz_new = fz_from_cov(CnoiseC_new, use_psd=True)
        dz_new_list.append(fz_new - fz_U)

    dz_old = np.array(dz_old_list)
    dz_new = np.array(dz_new_list)
    print(f"  Old mean Dz = {dz_old.mean():.6f}, New mean Dz = {dz_new.mean():.6f}")
    return dz_old, dz_new


# ── Figure ─────────────────────────────────────────────────────────────────

def make_figure(weight_pairs, weight_uniform, n_t, trial_dur,
                Erate_old, Erate_new, dz_old, dz_new):
    """Create the 2x2 diagnostic figure."""
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.08, right=0.95, top=0.93, bottom=0.08)

    panel_labels = ['A', 'B', 'C', 'D']

    # ── Shared colormap and limits for A and B ──
    vmax = np.nanmax(weight_pairs)
    vmin = 0
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='#f0f0f0')

    # Convert time bins to ms
    ms_per_bin = DT * 1000  # ~4.17 ms per bin
    n_time = weight_pairs.shape[1]
    time_ms = np.arange(n_time) * ms_per_bin

    # ── Panel A: Pair-count weighting ──
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(weight_pairs, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='nearest',
                        extent=[0, time_ms[-1], weight_pairs.shape[0], 0])
    ax_a.set_xlabel('Time within trial (ms)')
    ax_a.set_ylabel('Trial (sorted by duration)')
    ax_a.set_title('Second moment weighting (per-trial pairs)', fontsize=10.5)
    cb_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.set_label('$n_t - 1$  (pairings per trial)', fontsize=8.5)
    ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # ── Panel B: Uniform weighting ──
    ax_b = fig.add_subplot(gs[0, 1])
    # Scale uniform weight to same colormap range
    weight_uniform_scaled = weight_uniform.copy()
    weight_uniform_scaled[np.isfinite(weight_uniform_scaled)] = 1.0
    im_b = ax_b.imshow(weight_uniform_scaled, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='nearest',
                        extent=[0, time_ms[-1], weight_uniform.shape[0], 0])
    ax_b.set_xlabel('Time within trial (ms)')
    ax_b.set_ylabel('Trial (sorted by duration)')
    ax_b.set_title('Old Erate weighting (uniform)', fontsize=10.5)
    cb_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label('Weight', fontsize=8.5)
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # ── Panel C: Erate difference per neuron ──
    ax_c = fig.add_subplot(gs[1, 0])
    diff = Erate_new - Erate_old
    sort_idx_diff = np.argsort(diff)
    neuron_x = np.arange(len(diff))

    colors = np.where(diff[sort_idx_diff] >= 0, '#2166ac', '#b2182b')
    ax_c.bar(neuron_x, diff[sort_idx_diff], color=colors, width=1.0,
             edgecolor='none')
    ax_c.axhline(0, color='k', linewidth=0.7, linestyle='--')
    ax_c.set_xlabel('Neuron (sorted by difference)')
    ax_c.set_ylabel('$E_{\\mathrm{pair}}$ - $E_{\\mathrm{trial}}$ (sp / bin)')
    ax_c.set_title('Erate difference (pair-weighted $-$ trial-weighted)',
                    fontsize=10.5)
    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # ── Panel D: Shuffle null distributions ──
    ax_d = fig.add_subplot(gs[1, 1])
    bins_hist = np.linspace(
        min(dz_old.min(), dz_new.min()) - 0.002,
        max(dz_old.max(), dz_new.max()) + 0.002,
        25
    )
    ax_d.hist(dz_old, bins=bins_hist, alpha=0.55, color='#b2182b',
              edgecolor='#b2182b', linewidth=0.5, label=f'Old (trial-weighted)')
    ax_d.hist(dz_new, bins=bins_hist, alpha=0.55, color='#2166ac',
              edgecolor='#2166ac', linewidth=0.5, label=f'Fixed (pair-weighted)')

    # Mean lines
    ax_d.axvline(dz_old.mean(), color='#b2182b', linewidth=2, linestyle='-',
                  label=f'Old mean = {dz_old.mean():.4f}')
    ax_d.axvline(dz_new.mean(), color='#2166ac', linewidth=2, linestyle='-',
                  label=f'Fixed mean = {dz_new.mean():.4f}')
    ax_d.axvline(0, color='k', linewidth=1.2, linestyle='--', label='Zero', zorder=5)

    ax_d.set_xlabel(r'$\Delta z$ (shuffle null)')
    ax_d.set_ylabel('Count')
    ax_d.set_title(r'Shuffle null $\Delta z$ distribution', fontsize=10.5)
    ax_d.legend(fontsize=7.5, loc='upper left', frameon=True, edgecolor='0.8')
    ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold', va='top')

    return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = get_device()

    # Load data
    robs, eyepos, valid_mask, neuron_mask, meta = load_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Baseline covariances
    print("\nComputing baseline covariances...")
    ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()
    Erate_baseline = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    Cpsth, _ = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
        seed=42, global_mean=Erate_baseline
    )
    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
    fz_U = fz_from_cov(CnoiseU, use_psd=True)
    print(f"  fz_U = {fz_U:.6f}")

    # Get bin_edges from a real-data run for consistent binning in shuffles
    _, _, _, _, _, bin_edges = estimate_rate_covariance(
        SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS,
        Ctotal=Ctotal, intercept_mode='linear'
    )

    # Panel A & B data
    print("\nComputing Panel A & B data (weighting matrices)...")
    weight_pairs, weight_uniform, n_t, trial_dur = compute_panel_ab_data(valid_mask)

    # Panel C data
    print("Computing Panel C data (Erate comparison)...")
    Erate_old, Erate_new = compute_panel_c_data(SpikeCounts, T_idx)
    print(f"  Max |Erate diff| = {np.abs(Erate_new - Erate_old).max():.6f}")

    # Panel D data
    dz_old, dz_new = compute_panel_d_data(
        SpikeCounts, EyeTraj, T_idx, Ctotal, fz_U, bin_edges
    )

    # Make figure
    print("\nGenerating figure...")
    fig = make_figure(weight_pairs, weight_uniform, n_t, trial_dur,
                      Erate_old, Erate_new, dz_old, dz_new)

    # Save
    png_path = os.path.join(SAVE_DIR, 'weighting_mechanism_figure.png')
    pdf_path = os.path.join(SAVE_DIR, 'weighting_mechanism_figure.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
