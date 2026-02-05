
#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('./scripts')
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
        Turbo-Charged Covariance Decomposition.
        Uses GPU for stats + Vectorized Linear Regression for fitting.
        """
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing on {self.device}...")
        t0 = time.time()
        
        # 1. Load & Sanitize
        if np.isnan(robs).any():
            robs = np.nan_to_num(robs, nan=0.0)
        eyepos = np.nan_to_num(eyepos, nan=0.0)

        self.robs = torch.tensor(robs, dtype=torch.float32, device=self.device)
        self.eyepos = torch.tensor(eyepos, dtype=torch.float32, device=self.device)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        
        self.n_trials, self.n_time, self.n_cells = robs.shape
        
        # 2. Pre-compute PSTH
        valid_float = self.valid_mask.float().unsqueeze(-1)
        sum_spikes = torch.sum(self.robs * valid_float, dim=0)
        count_trials = torch.sum(valid_float, dim=0)
        count_trials[count_trials == 0] = 1.0 
        self.psth = sum_spikes / count_trials
        
        # 3. Valid Segments
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
        total_len = t_count + t_hist
        trial_indices, time_indices = [], []
        
        for (tr, start, stop) in self.segments:
            if (stop - start) < total_len: continue
            t_starts = np.arange(start, stop - total_len + 1, t_count)
            trial_indices.extend([tr] * len(t_starts))
            time_indices.extend(t_starts)
            
        if not trial_indices: return None, None, None
            
        # Subsample to cap VRAM/Compute usage
        n_total = len(trial_indices)
        if n_total > max_samples:
            np.random.seed(42) 
            keep_idx = np.random.choice(n_total, max_samples, replace=False)
            trial_indices = np.array(trial_indices)[keep_idx]
            time_indices = np.array(time_indices)[keep_idx]
            
        idx_tr = torch.tensor(trial_indices, device=self.device, dtype=torch.long)
        idx_t0 = torch.tensor(time_indices, device=self.device, dtype=torch.long)
        
        # GPU Gather
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
        N = S.shape[0]
        dists = torch.zeros((N, N), device=self.device, dtype=torch.float32)
        block_size = 2000 
        
        # Blocked Distance Calc
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            E_i = E[i:i_end]
            for j in range(0, N, block_size):
                j_end = min(j + block_size, N)
                E_j = E[j:j_end]
                diff = E_i.unsqueeze(1) - E_j.unsqueeze(0)
                dists[i:i_end, j:j_end] = torch.mean(torch.norm(diff, dim=-1), dim=-1)
        
        mask_triu = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        valid_dists = dists[mask_triu]
        
        # Quantile Subsampling (CPU Speed Fix)
        if len(valid_dists) > 0:
            if len(valid_dists) > 1_000_000:
                step = len(valid_dists) // 1_000_000
                max_dist = torch.quantile(valid_dists[::step], 0.95)
            else:
                max_dist = torch.quantile(valid_dists, 0.95)
        else:
            max_dist = 1.0
            
        bins = torch.linspace(0, max_dist, n_bins + 1, device=self.device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        n_cells = S.shape[1]
        binned_covs = torch.zeros((n_bins, n_cells, n_cells), device=self.device)
        bin_counts  = torch.zeros(n_bins, device=self.device)
        ST = S.T
        
        # Binned Covariance
        for k in range(n_bins):
            mask_bin = (dists >= bins[k]) & (dists < bins[k+1]) & mask_triu
            count = mask_bin.sum()
            if count < 5: continue
            cov_sum = torch.linalg.multi_dot([ST, mask_bin.float(), S])
            binned_covs[k] = cov_sum / count
            bin_counts[k] = count
            
        return binned_covs.cpu().numpy(), bin_centers.cpu().numpy(), bin_counts.cpu().numpy()
    
    def _compute_cv_psth_sigma(self, t_count, T_idx, valid_mask_float):
        """
        Computes UNBIASED PSTH Covariance using Split-Half Cross-Validation.
        Removes the '1/N' noise floor bias that causes negative Fano Factors.
        """
        # 1. Generate Split Masks (A/B)
        # We want to split the VALID TRIALS for each timepoint.
        # This is tricky because validity varies by time.
        # Heuristic: Randomly assign every trial ID to group A or B.
        
        n_trials = self.n_trials
        perm = torch.randperm(n_trials, device=self.device)
        idx_A = perm[:n_trials//2]
        idx_B = perm[n_trials//2:]
        
        # Create masks
        mask_A = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_B = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_A[idx_A] = 1.0
        mask_B[idx_B] = 1.0
        
        # 2. Compute PSTH_A and PSTH_B
        # valid_mask_float is [Trials, Time, 1]
        
        # Weighted Sums
        sum_A = torch.sum(self.robs * valid_mask_float * mask_A, dim=0)
        cnt_A = torch.sum(valid_mask_float * mask_A, dim=0)
        cnt_A[cnt_A==0] = 1.0
        psth_A = sum_A / cnt_A # [Time, Cells]
        
        sum_B = torch.sum(self.robs * valid_mask_float * mask_B, dim=0)
        cnt_B = torch.sum(valid_mask_float * mask_B, dim=0)
        cnt_B[cnt_B==0] = 1.0
        psth_B = sum_B / cnt_B # [Time, Cells]
        
        # 3. Gather Windows for Covariance
        # We need the sums of PSTH over the specific windows used in this sweep
        offsets = torch.arange(t_count, device=self.device).unsqueeze(0)
        gather_t = T_idx.unsqueeze(1) + offsets
        
        # Extract windows from both independent PSTHs
        # [N_samples, t_count, C] -> Sum -> [N_samples, C]
        win_A = torch.sum(psth_A[gather_t, :], dim=1)
        win_B = torch.sum(psth_B[gather_t, :], dim=1)
        
        # 4. Compute Cross-Covariance Matrix
        # Cov(A, B) = E[(A - muA)(B - muB).T]
        mu_A = torch.mean(win_A, dim=0, keepdim=True)
        mu_B = torch.mean(win_B, dim=0, keepdim=True)
        
        centered_A = win_A - mu_A
        centered_B = win_B - mu_B
        
        n_samples = win_A.shape[0]
        # Cross-Covariance
        Sigma_CV = (centered_A.T @ centered_B) / (n_samples - 1)
        
        # Symmetrize (Optional but good for numerical stability)
        Sigma_CV = 0.5 * (Sigma_CV + Sigma_CV.T)
        
        return Sigma_CV
    
    def _fit_intercepts_vectorized(self, binned_covs, bin_centers, bin_counts, total_2nd_moments):
        """
        Vectorized Linear Fitting.
        Replaces slow iterative curve_fit with instant O(1) algebra.
        Model: y = A * exp(-x/tau) + Plateau
        Linearized: log(y - Plateau) = log(A) - (1/tau)*x
        """
        n_bins, n_cells, _ = binned_covs.shape
        Sigma_intercept = np.zeros((n_cells, n_cells))
        
        # Identify valid bins (Global, same for all pairs)
        valid_bins = bin_counts > 10
        if np.sum(valid_bins) < 4:
            return Sigma_intercept
            
        x = bin_centers[valid_bins]
        
        # Loop over pairs (Python loop over simple numpy ops is fast enough for 7000 pairs)
        # ~0.1 seconds total
        for i in range(n_cells):
            for j in range(i, n_cells):
                y = binned_covs[valid_bins, i, j]
                
                # 1. Estimate Plateau (Noise Floor) from tail
                plateau = np.mean(y[-3:]) 
                
                # 2. Linearize
                y_sub = y - plateau
                
                # Filter positive values only for Log
                mask_log = y_sub > (1e-6 * np.max(np.abs(y)))
                
                if np.sum(mask_log) < 3:
                    # Fallback if no decay visible
                    intercept = np.mean(y)
                else:
                    # Weighted Linear Regression (Weight by signal strength y^2)
                    # slope m, intercept b
                    try:
                        x_fit = x[mask_log]
                        y_fit = np.log(y_sub[mask_log])
                        # weights = y_sub[mask_log]**2 
                        
                        m, b = np.polyfit(x_fit, y_fit, 1) # Unweighted is often more stable for noisy tails
                        
                        A = np.exp(b)
                        intercept = A + plateau
                    except:
                        intercept = np.nan
                
                # 3. Clamp
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
        mats_save = []
        
        print(f"Starting Sweep (Hist={t_hist_ms}ms)...")
        
        for win_ms in tqdm(window_sizes_ms):
            t0 = time.time()
            t_count_bins = int(win_ms / (self.dt * 1000))
            if t_count_bins < 1: t_count_bins = 1
            
            # 1. GPU Extract
            S, E, T_idx = self._extract_windows_gpu(t_count_bins, t_hist_bins, max_samples=10000)
            if S is None or S.shape[0] < 100: continue
            t1 = time.time()
            
            # 2. GPU Stats
            n_samples = S.shape[0]
            Sigma_Total_Raw = (S.T @ S) / n_samples
            mean_N = torch.mean(S, dim=0)
            Sigma_Total_Cov = Sigma_Total_Raw - torch.outer(mean_N, mean_N)
            
            offsets = torch.arange(t_count_bins, device=self.device).unsqueeze(0)
            gather_t = T_idx.unsqueeze(1) + offsets
            psth_vals = self.psth[gather_t, :]
            psth_sums = torch.sum(psth_vals, dim=1)
            # Sigma_PSTH = torch.cov(psth_sums.T)
            valid_float = self.valid_mask.float().unsqueeze(-1)
            Sigma_PSTH = self._compute_cv_psth_sigma(t_count_bins, T_idx, valid_float)
            
            # 3. McFarland Stats
            binned_covs, bin_centers, bin_counts = self._compute_binned_stats_gpu(S, E)
            t2 = time.time()
            
            # 4. CPU Fast Fit
            total_2nd = torch.diag(Sigma_Total_Raw).cpu().numpy()
            Sigma_Intercept = self._fit_intercepts_vectorized(binned_covs, bin_centers, bin_counts, total_2nd)
            t3 = time.time()
            
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
            
            if np.isnan(Sigma_FEM).any():
                rank = np.nan
            else:
                evals = np.linalg.eigvalsh(Sigma_FEM)[::-1]
                pos = evals[evals > 0]
                rank = (np.sum(pos[:2])/np.sum(pos)) if len(pos)>2 else 1.0
            
            results.append({
                'window_ms': win_ms,
                'ff_uncorr': ff_uncorr,
                'ff_corr': ff_corr,
                'ff_uncorr_mean': np.nanmean(ff_uncorr),
                'ff_corr_mean': np.nanmean(ff_corr),
                'fem_rank_ratio': rank,
                'n_samples': n_samples
            })

            mats_save.append({'Total': Sigma_Total_Cov, 'PSTH': Sigma_PSTH, 'FEM': Sigma_FEM, 'Noise_Corr': Sigma_Noise_Corr, 'Intercept': Sigma_Intercept})

            # Debug Timing
            # tqdm.write(f"  {win_ms}ms: Extract={t1-t0:.2f}s, Stats={t2-t1:.2f}s, Fit={t3-t2:.2f}s")
            
        return results, mats_save
    
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
analyzer = DualWindowAnalysis(robs[:,:200], eyepos[:,:200], valid_mask[:,:200], dt=1/240)

# 2. Run Sweep
windows = [10, 20, 40, 80, 100, 150]
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
window_idx = -1
Sigma_FEM = last_mats[window_idx]['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
plt.figure()
plt.plot(s, 'o-')
plt.title(f"Singular Values of FEM Covariance ({windows[-1]}ms)")
plt.show()

# same for total covariance
Sigma_Total = last_mats[window_idx]['Total']
u, s, vh = np.linalg.svd(Sigma_Total)
plt.figure()
plt.plot(s, 'o-')
plt.title(f"Singular Values of Total Covariance ({windows[-1]}ms)")
plt.show()

# now noise cov
Sigma_Noise = last_mats[window_idx]['Noise_Corr']
u, s, vh = np.linalg.svd(Sigma_Noise)
plt.figure()
plt.plot(s, 'o-')
plt.title(f"Singular Values of Noise Covariance ({windows[-1]}ms)")
plt.show()
# %%
i = 3
plt.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')
plt.title(f"FF Window Size ({windows[i]}ms)")

#%%
results[0]
# %%
# show the total covariance matrix subtracting the diagonal
window_idx = 1
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(last_mats[window_idx]['Total'] - np.diag(np.diag(last_mats[window_idx]['Total'])))
plt.title(f"Total Covariance ({windows[window_idx]}ms)")

# show FEM
plt.subplot(1,3,2)
plt.imshow(last_mats[window_idx]['FEM'] - np.diag(np.diag(last_mats[window_idx]['FEM'])))
plt.title(f"FEM Covariance ({windows[window_idx]}ms)")

# show Noise_Corr
plt.subplot(1,3,3)
plt.imshow(last_mats[window_idx]['PSTH'] - np.diag(np.diag(last_mats[window_idx]['PSTH'])))
plt.title(f"PSTH Covariance ({windows[window_idx]}ms)")



# %%
plt.subplot(1,2,1)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH'])
plt.subplot(1,2,2)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'])

# %%
