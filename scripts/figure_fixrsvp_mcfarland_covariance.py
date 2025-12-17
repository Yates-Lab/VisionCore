
#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()
device = get_free_device()

#%% Utility function for smoothing eye position
import numpy as np
import torch

from scipy.signal import savgol_filter

def _savgol_1d_nan(y, window_length=15, polyorder=3):
    """
    Apply Savitzky–Golay to a 1D array with NaNs.
    NaNs are interpolated for filtering and then restored.
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)

    # If too few valid points, just return original
    if mask.sum() < polyorder + 2:
        return y

    yy = y.copy()
    idx_valid = np.where(mask)[0]
    idx_nan   = np.where(~mask)[0]

    # Linear interp over NaNs so savgol_filter has no gaps
    yy[idx_nan] = np.interp(idx_nan, idx_valid, yy[idx_valid])

    # Apply SG filter
    ys = savgol_filter(
        yy,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    # Restore original NaNs
    ys[~mask] = np.nan
    return ys


def savgol_nan_numpy(x, axis=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along a given axis for a NumPy array.
    """
    return np.apply_along_axis(
        _savgol_1d_nan,
        axis=axis,
        arr=x,
        window_length=window_length,
        polyorder=polyorder,
    )


def savgol_nan_torch(x, dim=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along dim for a torch.Tensor.
    - x: (..., T, ...) tensor
    - dim: time dimension (default 1)
    """
    # Move target dim to last for easier NumPy apply
    x_np = x.detach().cpu().numpy()
    x_np = np.moveaxis(x_np, dim, -1)

    y_np = savgol_nan_numpy(
        x_np,
        axis=-1,
        window_length=window_length,
        polyorder=polyorder,
    )

    # Move axis back and convert to torch
    y_np = np.moveaxis(y_np, -1, dim)
    y = torch.from_numpy(y_np).to(x.device).type_as(x)
    return y

#%% Law of total covariance decomposition
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm
import time

class DualWindowAnalysis:
    def __init__(self, robs, eyepos, valid_mask, dt=1/240, device='cuda'):
        """
        GPU-Accelerated Covariance Decomposition (Memory & Speed Optimized).
        
        Parameters:
        -----------
        robs : np.ndarray [Trials, Time, Cells]
        eyepos : np.ndarray [Trials, Time, 2]
        valid_mask : np.ndarray [Trials, Time]
        dt : float
        device : str ('cuda' or 'cpu')
        """
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing on {self.device}...")
        t0 = time.time()
        
        # 1. Load Data to GPU & Sanitize
        # Replace NaNs with 0.0 immediately to prevent poisoning
        if np.isnan(robs).any():
            print("  WARNING: Found NaNs in robs. replacing with 0.")
            robs = np.nan_to_num(robs, nan=0.0)
        eyepos = np.nan_to_num(eyepos, nan=0.0)

        # Float32 is sufficient precision and saves 50% RAM over Double
        self.robs = torch.tensor(robs, dtype=torch.float32, device=self.device)
        self.eyepos = torch.tensor(eyepos, dtype=torch.float32, device=self.device)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        
        self.n_trials, self.n_time, self.n_cells = robs.shape
        
        # 2. Pre-compute PSTH (GPU)
        # Weighted mean to ignore invalid timepoints
        valid_float = self.valid_mask.float().unsqueeze(-1)
        sum_spikes = torch.sum(self.robs * valid_float, dim=0)
        count_trials = torch.sum(valid_float, dim=0)
        count_trials[count_trials == 0] = 1.0 
        self.psth = sum_spikes / count_trials
        
        # 3. Find Valid Segments (CPU is faster for ragged loops)
        self.segments = self._get_valid_segments(min_len_bins=36)
        print(f"Loaded {len(self.segments)} valid segments. Init took {time.time()-t0:.2f}s")

    def _get_valid_segments(self, min_len_bins):
        segments = []
        mask_cpu = self.valid_mask.cpu().numpy()
        for tr in range(self.n_trials):
            padded = np.concatenate(([False], mask_cpu[tr], [False]))
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            stops = np.where(diffs == -1)[0]
            for start, stop in zip(starts, stops):
                if (stop - start) >= min_len_bins:
                    segments.append((tr, start, stop))
        return segments

    def _extract_windows_gpu(self, t_count, t_hist, max_samples=10000):
        """
        Phase II: Extract windows into Batched Tensors (GPU Gather).
        INCLUDES RANDOM SUBSAMPLING to prevent O(N^2) explosion.
        """
        total_len = t_count + t_hist
        
        # Construct indices on CPU
        trial_indices = []
        time_indices = [] 
        
        for (tr, start, stop) in self.segments:
            if (stop - start) < total_len: continue
            # Non-overlapping strides
            t_starts = np.arange(start, stop - total_len + 1, t_count)
            trial_indices.extend([tr] * len(t_starts))
            time_indices.extend(t_starts)
            
        if not trial_indices:
            return None, None, None
            
        # --- OPTIMIZATION: Subsample if N is huge ---
        n_total = len(trial_indices)
        if n_total > max_samples:
            # Randomly select indices without replacement
            # Setting seed ensures reproducibility across runs
            np.random.seed(42) 
            keep_idx = np.random.choice(n_total, max_samples, replace=False)
            
            # Use numpy indexing to filter
            trial_indices = np.array(trial_indices)[keep_idx]
            time_indices = np.array(time_indices)[keep_idx]
            
        # Move indices to GPU
        idx_tr = torch.tensor(trial_indices, device=self.device, dtype=torch.long)
        idx_t0 = torch.tensor(time_indices, device=self.device, dtype=torch.long)
        
        # --- GPU Gather ---
        offsets = torch.arange(total_len, device=self.device).unsqueeze(0)
        gather_t = idx_t0.unsqueeze(1) + offsets
        gather_tr = idx_tr.unsqueeze(1).expand(-1, total_len)
        E = self.eyepos[gather_tr, gather_t, :]
        
        spike_offsets = torch.arange(t_hist, total_len, device=self.device).unsqueeze(0)
        gather_t_spike = idx_t0.unsqueeze(1) + spike_offsets
        gather_tr_spike = idx_tr.unsqueeze(1).expand(-1, t_count)
        
        S_raw = self.robs[gather_tr_spike, gather_t_spike, :]
        S = torch.sum(S_raw, dim=1)
        
        T_idx = idx_t0 + t_hist
        
        return S, E, T_idx

    def _compute_binned_stats_gpu(self, S, E, n_bins=15):
        """
        Phase III: Compute Pairwise Stats using Blocked Matrix Math.
        Fixes both Memory Crash and Quantile Hang.
        """
        t0 = time.time()
        N = S.shape[0]
        
        # --- 1. Compute Distances in Blocks to save VRAM ---
        dists = torch.zeros((N, N), device=self.device, dtype=torch.float32)
        
        # Block size: 2000 fits easily in 47GB (2000^2 * T * 4 bytes)
        block_size = 2000 
        
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            E_i = E[i:i_end]
            for j in range(0, N, block_size):
                j_end = min(j + block_size, N)
                E_j = E[j:j_end]
                
                # [Bi, 1, T, 2] - [1, Bj, T, 2] -> Norm -> Mean
                diff = E_i.unsqueeze(1) - E_j.unsqueeze(0)
                d_block = torch.mean(torch.norm(diff, dim=-1), dim=-1)
                dists[i:i_end, j:j_end] = d_block
        
        t_dist = time.time()
        
        # --- 2. Define Bins (With Subsampling Fix) ---
        mask_triu = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        valid_dists = dists[mask_triu]
        
        if len(valid_dists) > 0:
            # FIX: Subsample to 1 million points for quantile
            # Sorting 400M points hangs the CPU/GPU interface.
            if len(valid_dists) > 1_000_000:
                # Random stride is sufficient
                step = len(valid_dists) // 1_000_000
                subset = valid_dists[::step]
                max_dist = torch.quantile(subset, 0.95)
            else:
                max_dist = torch.quantile(valid_dists, 0.95)
        else:
            max_dist = 1.0
            
        bins = torch.linspace(0, max_dist, n_bins + 1, device=self.device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        t_bins = time.time()
        
        # --- 3. Quadratic Form Accumulation ---
        n_cells = S.shape[1]
        binned_covs = torch.zeros((n_bins, n_cells, n_cells), device=self.device)
        bin_counts  = torch.zeros(n_bins, device=self.device)
        
        ST = S.T
        
        for k in range(n_bins):
            # 1 if pair in bin k, else 0
            mask_bin = (dists >= bins[k]) & (dists < bins[k+1]) & mask_triu
            count = mask_bin.sum()
            
            if count < 5: continue
            
            # (S.T @ Mask @ S) sums products for all pairs in this bin
            # This is the "GPU Magic" step
            cov_sum = torch.linalg.multi_dot([ST, mask_bin.float(), S])
            
            binned_covs[k] = cov_sum / count
            bin_counts[k] = count
        
        t_quad = time.time()
        # print(f"      [Timing] Dist: {t_dist-t0:.3f}s, Bins: {t_bins-t_dist:.3f}s, QuadForm: {t_quad-t_bins:.3f}s")
            
        return binned_covs.cpu().numpy(), bin_centers.cpu().numpy(), bin_counts.cpu().numpy()

    def _fit_intercepts_cpu(self, binned_covs, bin_centers, bin_counts, total_2nd_moments):
        """Phase IV: Robust Fitting on CPU."""
        n_bins, n_cells, _ = binned_covs.shape
        Sigma_intercept = np.zeros((n_cells, n_cells))
        
        def model(x, A, tau, plat): return A * np.exp(-tau * x) + plat
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

        # This loop is fast for N=120
        for i in range(n_cells):
            for j in range(i, n_cells):
                y = binned_covs[:, i, j]
                valid = bin_counts > 10
                
                if np.sum(valid) < 4:
                    Sigma_intercept[i, j] = np.nan
                    Sigma_intercept[j, i] = np.nan
                    continue
                
                x_val, y_val = bin_centers[valid], y[valid]
                intercept = np.nan
                try:
                    p0 = [np.ptp(y_val)+1e-6, 10.0, np.min(y_val)]
                    popt, _ = curve_fit(model, x_val, y_val, p0=p0, bounds=bounds, maxfev=500)
                    intercept = popt[0] + popt[2]
                except: pass
                
                # Clamp Diagonal
                if i == j:
                    limit = total_2nd_moments[i]
                    if not np.isnan(intercept) and intercept > limit:
                        intercept = limit
                        
                Sigma_intercept[i, j] = intercept
                Sigma_intercept[j, i] = intercept
                
        return Sigma_intercept

    def run_sweep(self, window_sizes_ms, t_hist_ms=100):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        last_mats = None
        
        print(f"Starting Sweep (Hist={t_hist_ms}ms)...")
        
        for win_ms in tqdm(window_sizes_ms):
            t0_win = time.time()
            t_count_bins = int(win_ms / (self.dt * 1000))
            if t_count_bins < 1: t_count_bins = 1
            
            # 1. GPU Extract
            S, E, T_idx = self._extract_windows_gpu(t_count_bins, t_hist_bins, max_samples=10000)
            if S is None or S.shape[0] < 100: continue
            t_ext = time.time()
            
            # 2. Basic Stats
            n_samples = S.shape[0]
            Sigma_Total_Raw = (S.T @ S) / n_samples
            mean_N = torch.mean(S, dim=0)
            Sigma_Total_Cov = Sigma_Total_Raw - torch.outer(mean_N, mean_N)
            
            # PSTH Covariance
            offsets = torch.arange(t_count_bins, device=self.device).unsqueeze(0)
            gather_t = T_idx.unsqueeze(1) + offsets
            psth_vals = self.psth[gather_t, :]
            psth_sums = torch.sum(psth_vals, dim=1)
            Sigma_PSTH = torch.cov(psth_sums.T)
            t_stats = time.time()
            
            # 3. McFarland Binned Stats
            binned_covs, bin_centers, bin_counts = self._compute_binned_stats_gpu(S, E)
            t_mcf = time.time()
            
            # 4. Fit on CPU
            total_2nd = torch.diag(Sigma_Total_Raw).cpu().numpy()
            Sigma_Intercept = self._fit_intercepts_cpu(binned_covs, bin_centers, bin_counts, total_2nd)
            t_fit = time.time()
            
            # 5. Algebra
            Sigma_Total_Cov = Sigma_Total_Cov.cpu().numpy()
            Sigma_PSTH = Sigma_PSTH.cpu().numpy()
            mean_N_np = mean_N.cpu().numpy()
            
            Sigma_Rate = Sigma_Intercept - np.outer(mean_N_np, mean_N_np)
            Sigma_FEM = Sigma_Rate - Sigma_PSTH
            Sigma_Noise_Uncorr = Sigma_Total_Cov - Sigma_PSTH
            Sigma_Noise_Corr = Sigma_Total_Cov - Sigma_Rate
            
            # Metrics
            mu = mean_N_np.copy()
            mu[mu==0] = 1e-9
            ff_uncorr = np.diag(Sigma_Noise_Uncorr) / mu
            ff_corr = np.diag(Sigma_Noise_Corr) / mu
            
            # Rank
            if np.isnan(Sigma_FEM).any():
                rank = np.nan
            else:
                evals = np.linalg.eigvalsh(Sigma_FEM)[::-1]
                pos = evals[evals > 0]
                rank = (np.sum(pos[:2])/np.sum(pos)) if len(pos)>2 else 1.0
            
            results.append({
                'window_ms': win_ms,
                'ff_uncorr_mean': np.nanmean(ff_uncorr),
                'ff_corr_mean': np.nanmean(ff_corr),
                'fem_rank_ratio': rank,
                'n_samples': n_samples
            })
            
            last_mats = {'Total': Sigma_Total_Cov, 'FEM': Sigma_FEM, 'Noise_Corr': Sigma_Noise_Corr}
            
            # print(f"    {win_ms}ms Times: Ext={t_ext-t0_win:.2f}s, Stats={t_stats-t_ext:.2f}s, McFar={t_mcf-t_stats:.2f}s, Fit={t_fit-t_mcf:.2f}s")
            
        return results, last_mats
    
#%% Load Dataset
from tqdm import tqdm

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

dataset_idx = 7
include_time_lags = False
train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx])
sess = train_data.dsets[0].metadata['sess']
ppd = train_data.dsets[0].metadata['ppd']
cids = dataset_config['cids']

# get fixrsvp inds and make one dataaset object
inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds

# Getting key variables
dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

# Loop over trials and compute the response from the pyramid
robs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))

for itrial in tqdm(range(NT)):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation
    if np.sum(ix) == 0:
        continue
    
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    

good_trials = fix_dur > 20
robs = robs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 160)

#%% Run the analysis

# 1. Setup
# Assuming 'robs', 'eyepos', 'valid_mask' are already loaded from your dataset code
# valid_mask should be True where data is good (no fix breaks)
valid_mask = np.isfinite(np.sum(robs, axis=2)) & np.isfinite(np.sum(eyepos, axis=2))
analyzer = DualWindowAnalysis(robs, eyepos, valid_mask, dt=1/240)

# 2. Run Sweep
windows = [4]
results, last_mats = analyzer.run_sweep(windows, t_hist_ms=50)

#%% 3. Plot Fano Factor Scaling
import pandas as pd
df = pd.DataFrame(results)

plt.figure(figsize=(8, 6))
plt.plot(df['window_ms'], df['ff_uncorr_mean'], 'o-', label='Standard (Uncorrected)')
plt.plot(df['window_ms'], df['ff_corr_mean'], 'o-', label='FEM-Corrected')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Count Window (ms)')
plt.ylabel('Mean Fano Factor')
plt.title('Integration of Noise: FEM Correction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Check Rank of the last window (e.g., 150ms)
Sigma_FEM = last_mats['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
plt.figure()
plt.plot(s, 'o-')
plt.title(f"Singular Values of FEM Covariance ({windows[-1]}ms)")
plt.show()
# %%
results[0]
# %%
