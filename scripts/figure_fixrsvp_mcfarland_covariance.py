
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
        self.window_summaries = {}
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

    def _quantile_bins_from_pair_dists(
        self,
        valid_dists_1d: torch.Tensor,  # 1D tensor of upper-tri distances
        n_bins: int,
        max_q_samples: int = 1_000_000,
        eps: float = 1e-6,
    ):
        """
        Approximate equipopulated bin edges using quantiles of a random subset of pairwise distances.
        Returns:
            bins: (n_bins+1,) torch tensor
            bin_centers: (n_bins,) torch tensor
        """
        if valid_dists_1d.numel() == 0:
            bins = torch.linspace(0, 1.0, n_bins + 1, device=self.device)
            centers = 0.5 * (bins[:-1] + bins[1:])
            return bins, centers

        M = valid_dists_1d.numel()
        if M > max_q_samples:
            idx = torch.randint(0, M, (max_q_samples,), device=self.device)
            sample = valid_dists_1d[idx]
        else:
            sample = valid_dists_1d

        q = torch.linspace(0, 1, n_bins + 1, device=self.device)
        bins = torch.quantile(sample, q)

        # enforce 0 start
        bins[0] = 0.0

        # ensure strictly increasing edges (quantiles can repeat)
        bins = torch.maximum(bins, torch.cat([bins[:1], bins[:-1] + eps]))

        centers = 0.5 * (bins[:-1] + bins[1:])
        return bins, centers

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
        
        # Equipopulated (quantile) bins (approx via subsampling)
        bins, bin_centers = self._quantile_bins_from_pair_dists(
            valid_dists, n_bins=n_bins, max_q_samples=1_000_000
        )

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
    
    def _compute_cv_psth_sigma(
        self,
        t_count,
        T_idx,
        valid_mask_float,
        min_trials_per_time=2,
        min_valid_frac_in_window=0.8,
    ):
        """
        UNBIASED PSTH covariance via split-half cross-covariance, but with
        *window-level validity filtering* so we don't bias PSTH covariance toward 0
        when one split has too few valid trials in the window.

        Key fixes vs your current version:
        - Do NOT set cnt==0 to 1 and silently inject zeros into PSTH.
        - Filter windows where either split has insufficient valid trial support across the window.
        """

        n_trials = self.n_trials
        perm = torch.randperm(n_trials, device=self.device)
        idx_A = perm[: n_trials // 2]
        idx_B = perm[n_trials // 2 :]

        # masks [Trials, 1, 1]
        mask_A = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_B = torch.zeros((n_trials, 1, 1), device=self.device)
        mask_A[idx_A] = 1.0
        mask_B[idx_B] = 1.0

        # trial counts per time for each split: [Time, 1]
        cnt_A = torch.sum(valid_mask_float * mask_A, dim=0)  # [Time,1]
        cnt_B = torch.sum(valid_mask_float * mask_B, dim=0)  # [Time,1]

        # Build PSTHs but mark invalid times as NaN (not zero)
        # sum spikes per time per split: [Time, Cells]
        sum_A = torch.sum(self.robs * valid_mask_float * mask_A, dim=0)  # [Time, Cells]
        sum_B = torch.sum(self.robs * valid_mask_float * mask_B, dim=0)

        # avoid divide-by-zero: we'll create NaNs where cnt is too small
        denom_A = cnt_A.clone()
        denom_B = cnt_B.clone()

        # require at least min_trials_per_time to define PSTH at that time in that split
        ok_time_A = (denom_A[:, 0] >= min_trials_per_time)  # [Time]
        ok_time_B = (denom_B[:, 0] >= min_trials_per_time)

        # initialize as NaN
        psth_A = torch.full((self.n_time, self.n_cells), float("nan"), device=self.device)
        psth_B = torch.full((self.n_time, self.n_cells), float("nan"), device=self.device)

        psth_A[ok_time_A] = sum_A[ok_time_A] / denom_A[ok_time_A]
        psth_B[ok_time_B] = sum_B[ok_time_B] / denom_B[ok_time_B]

        # Gather windows: [N_samples, t_count]
        offsets = torch.arange(t_count, device=self.device).unsqueeze(0)
        gather_t = T_idx.unsqueeze(1) + offsets

        # Window-level validity: fraction of timepoints in the window where PSTH is defined in each split
        # ok_time_A/B are [Time], so index them with gather_t -> [N_samples, t_count]
        okA_win = ok_time_A[gather_t]
        okB_win = ok_time_B[gather_t]

        fracA = okA_win.float().mean(dim=1)  # [N_samples]
        fracB = okB_win.float().mean(dim=1)

        ok_win = (fracA >= min_valid_frac_in_window) & (fracB >= min_valid_frac_in_window)

        # If too few windows survive, relax constraint a bit (failsafe)
        if ok_win.sum() < 50:
            ok_win = (fracA >= 0.5) & (fracB >= 0.5)

        gather_t_ok = gather_t[ok_win]

        # Extract window sums from PSTHs; NaNs will propagate if any remain
        win_A = torch.sum(psth_A[gather_t_ok, :], dim=1)  # [N_ok, Cells]
        win_B = torch.sum(psth_B[gather_t_ok, :], dim=1)

        # Drop any windows that still have NaNs (should be rare after filtering)
        ok_rows = torch.isfinite(win_A).all(dim=1) & torch.isfinite(win_B).all(dim=1)
        win_A = win_A[ok_rows]
        win_B = win_B[ok_rows]

        n_samples = win_A.shape[0]
        if n_samples < 10:
            # fallback: return zeros rather than garbage
            return torch.zeros((self.n_cells, self.n_cells), device=self.device)

        mu_A = torch.mean(win_A, dim=0, keepdim=True)
        mu_B = torch.mean(win_B, dim=0, keepdim=True)
        centered_A = win_A - mu_A
        centered_B = win_B - mu_B

        Sigma_CV = (centered_A.T @ centered_B) / (n_samples - 1)
        Sigma_CV = 0.5 * (Sigma_CV + Sigma_CV.T)
        return Sigma_CV
    
    def _fit_intercepts_vectorized(
        self,
        binned_covs,           # (n_bins, n_cells, n_cells)
        bin_centers,           # (n_bins,)
        bin_counts,            # (n_bins,)
        total_2nd_moments,     # (n_cells,)
        min_count=10,
        eps=1e-12,
        use_first_block=True,     # recommended: intercept from first isotonic block
        intercept_bins=2,         # fallback if not using first block
        sigma_quantile=0.5,       # use half-drop point for sigma-like scale
    ):
        """
        Fast, robust intercept estimator using weighted isotonic regression (monotone nonincreasing),
        with explicit extrapolation to 0 via the right-limit.

        Key properties:
        - No plateau assumption.
        - No parametric shape assumption.
        - Enforces non-increasing covariance with distance.
        - Intercept at 0 is g(0+) = sup_{d>0} g(d), estimated as the first isotonic block value.
        """

        def pava_nonincreasing_with_blocks(y, w):
            """
            Weighted PAVA for nonincreasing sequence.

            Returns:
                yhat: fitted values (len n)
                blocks: list of (start, end, mean, weight) for each pooled block
            """
            y = np.asarray(y, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)

            means = []
            weights = []
            starts = []
            ends = []

            for i in range(len(y)):
                means.append(y[i])
                weights.append(w[i])
                starts.append(i)
                ends.append(i)

                # enforce nonincreasing: merge while prev < curr
                while len(means) >= 2 and means[-2] < means[-1]:
                    w_new = weights[-2] + weights[-1]
                    m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / (w_new + eps)
                    means[-2] = m_new
                    weights[-2] = w_new
                    ends[-2] = ends[-1]

                    means.pop()
                    weights.pop()
                    starts.pop()
                    ends.pop()

            yhat = np.empty_like(y)
            blocks = []
            for m, s, e, ww in zip(means, starts, ends, weights):
                yhat[s:e+1] = m
                blocks.append((s, e, float(m), float(ww)))
            return yhat, blocks

        n_bins, n_cells, _ = binned_covs.shape
        Sigma_intercept = np.zeros((n_cells, n_cells), dtype=np.float64)
        Sigma_sigma     = np.full((n_cells, n_cells), np.nan, dtype=np.float64)
        Sigma_plateau   = np.zeros((n_cells, n_cells), dtype=np.float64)

        valid = bin_counts > min_count
        if np.sum(valid) < 3:
            return Sigma_intercept, Sigma_sigma, Sigma_plateau

        x = np.asarray(bin_centers[valid], dtype=np.float64)
        w = np.asarray(bin_counts[valid], dtype=np.float64)

        # precompute constant for gaussian-style sigma from half-drop
        gauss_half_const = np.sqrt(2.0 * np.log(2.0) + eps)

        for i in range(n_cells):
            for j in range(i, n_cells):
                y = np.asarray(binned_covs[valid, i, j], dtype=np.float64)

                # monotone fit + block structure
                yhat, blocks = pava_nonincreasing_with_blocks(y, w)

                # "plateau" for return/plotting: last fitted value (not assumed true plateau)
                plateau = float(yhat[-1])
                Sigma_plateau[i, j] = Sigma_plateau[j, i] = plateau

                # ---- intercept extrapolated to 0 ----
                # g(0+) = sup_{d>0} g(d) = max_k yhat[k], and with sorted x this is the first block mean.
                if use_first_block and len(blocks) > 0:
                    intercept = blocks[0][2]  # mean of first pooled block
                else:
                    k = min(intercept_bins, yhat.size)
                    w0 = w[:k]
                    intercept = float(np.sum(w0 * yhat[:k]) / (np.sum(w0) + eps))

                # clamp diagonal if requested
                if i == j:
                    limit = float(total_2nd_moments[i])
                    if np.isfinite(intercept) and intercept > limit:
                        intercept = limit

                # ---- sigma-like scale for visualization (optional) ----
                # Use the distance at which the fitted curve drops to plateau + q*(intercept-plateau)
                amp = max(eps, intercept - plateau)
                target = plateau + (1.0 - sigma_quantile) * amp  # e.g. q=0.5 -> half-drop

                idx = np.where(yhat <= target)[0]
                if idx.size == 0:
                    sigma_eff = np.nan
                else:
                    x_q = float(x[int(idx[0])])
                    # Map that to a Gaussian sigma so your existing Gaussian plot is "roughly" consistent.
                    # If target is half-drop, sigma = x_half / sqrt(2 ln 2).
                    if np.isfinite(x_q) and x_q > 0:
                        sigma_eff = x_q / gauss_half_const
                    else:
                        sigma_eff = np.nan

                Sigma_intercept[i, j] = Sigma_intercept[j, i] = intercept
                Sigma_sigma[i, j]     = Sigma_sigma[j, i]     = sigma_eff

        return Sigma_intercept, Sigma_sigma, Sigma_plateau


    def run_sweep(self, window_sizes_ms, t_hist_ms=100):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        mats_save = []
        
        print(f"Starting Sweep (Hist={t_hist_ms}ms)...")
        
        for win_ms in tqdm(window_sizes_ms):
            t0 = time.time()
            t_count_bins = int(win_ms / (self.dt * 1000))
            if t_count_bins < 1: t_count_bins = 1
            if t_hist_bins < t_count_bins: t_hist_bins = t_count_bins
            
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
            Sigma_PSTH = self._compute_cv_psth_sigma(
                t_count_bins, T_idx, valid_float,
                min_trials_per_time=10,
                min_valid_frac_in_window=0.8
            )
            
            # 3. McFarland Stats
            binned_covs, bin_centers, bin_counts = self._compute_binned_stats_gpu(S, E, n_bins=30)
            t2 = time.time()
            
            # 4. CPU Fast Fit
            total_2nd = torch.diag(Sigma_Total_Raw).cpu().numpy()
            Sigma_Intercept, Sigma_Sigma, Sigma_Plateau = self._fit_intercepts_vectorized(
                binned_covs, bin_centers, bin_counts, total_2nd
            )
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

            mats_save.append({
                'Total': Sigma_Total_Cov,
                'PSTH': Sigma_PSTH,
                'FEM': Sigma_FEM,
                'Noise_Corr': Sigma_Noise_Corr,
                'Intercept': Sigma_Intercept,
                'Fit_Sigma': Sigma_Sigma,
                'Fit_Plateau': Sigma_Plateau
            })

            win_key = float(win_ms)
            self.window_summaries[win_key] = {
                'bin_centers': bin_centers,
                'binned_covs': binned_covs,
                'bin_counts': bin_counts,
                'Sigma_Intercept': Sigma_Intercept,
                'Sigma_Sigma': Sigma_Sigma,
                'Sigma_Plateau': Sigma_Plateau,
                'Sigma_PSTH': Sigma_PSTH,
                'Sigma_Total': Sigma_Total_Cov,
                'Sigma_FEM': Sigma_FEM,
                'Sigma_Noise_Corr': Sigma_Noise_Corr,
                'mean_counts': mean_N_np
            }

            # Debug Timing
            # tqdm.write(f"  {win_ms}ms: Extract={t1-t0:.2f}s, Stats={t2-t1:.2f}s, Fit={t3-t2:.2f}s")
            
        return results, mats_save

    def inspect_neuron_pair(self, i, j, win_ms, ax=None, show=True):
        """
        Visualize the measured COVARIANCE (not second moment) vs. eye-movement distance.
        Auto-corrects raw second moments by subtracting mean product (mu_i * mu_j).
        """
        if not self.window_summaries:
            raise RuntimeError("run_sweep must be called before inspecting neuron pairs.")
        if not (0 <= i < self.n_cells) or not (0 <= j < self.n_cells):
            raise ValueError(f"Neuron indices must be in [0, {self.n_cells})")

        win_key = float(win_ms)
        if win_key not in self.window_summaries:
            available = ", ".join(str(k) for k in sorted(self.window_summaries.keys()))
            raise KeyError(f"Window {win_ms}ms was not cached. Available windows: {available}")

        summary = self.window_summaries[win_key]
        
        # 1. Load Data
        bin_centers = summary['bin_centers']
        raw_moments = summary['binned_covs'][:, i, j] # This is E[XY]
        counts = summary['bin_counts']
        valid = counts > 0

        if not np.any(valid):
            raise RuntimeError("No histogram bins with data for this neuron pair.")

        # 2. Get Correction Factor (Mean Product)
        # We need Cov(X,Y) = E[XY] - E[X]E[Y]
        mu = summary['mean_counts']
        mean_prod = mu[i] * mu[j]

        # 3. Center the Data
        covs = raw_moments - mean_prod  # Convert to covariance
        
        # 4. Center the Fits
        # These were fitted to raw moments, so we shift them down too
        intercept = summary['Sigma_Intercept'][i, j] - mean_prod
        plateau = summary['Sigma_Plateau'][i, j] - mean_prod
        sigma = summary['Sigma_Sigma'][i, j]
        psth_cov = summary['Sigma_PSTH'][i, j] # Already a covariance

        # 5. Plotting
        created_fig = False
        if ax is None:
            created_fig = True
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
        else:
            fig = ax.figure

        # Plot Measured Data
        ax.plot(
            bin_centers[valid],
            covs[valid],
            'o',
            label='Measured Covariance',
            color='tab:blue',
            alpha=0.6
        )

        # Plot Parametric Fit (Gaussian-ish)
        if np.isfinite(sigma) and sigma > 0:
            x_dense = np.linspace(0, np.max(bin_centers[valid]), 300)
            # Reconstruct gaussian in centered space
            amp = max(0.0, intercept - plateau)
            fit_vals = plateau + amp * np.exp(-0.5 * (x_dense / sigma) ** 2)
            ax.plot(x_dense, fit_vals, color='tab:orange', label='Gaussian Fit')
        else:
            # Fallback if no valid sigma found
            ax.axhline(intercept, color='tab:orange', linestyle='-', label='Fit Intercept')

        # Reference Lines
        ax.axhline(
            psth_cov,
            color='tab:green',
            linestyle='--',
            linewidth=2,
            label='PSTH Covariance'
        )
        ax.axhline(
            intercept,
            color='tab:red',
            linestyle=':',
            linewidth=2,
            label='Intercept (Total Signal)'
        )
        
        # 6. Re-run Isotonic Regression (PAVA) on CENTERED data for visualization
        # We re-run this locally to plot the exact "shape" of the non-parametric fit
        if True:
            y = covs[valid].astype(np.float64)
            w = counts[valid].astype(np.float64)
            x = bin_centers[valid].astype(np.float64)

            def pava_nonincreasing(y, w):
                means=[]; weights=[]; starts=[]; ends=[]
                for ii in range(len(y)):
                    means.append(y[ii]); weights.append(w[ii]); starts.append(ii); ends.append(ii)
                    while len(means) >= 2 and means[-2] < means[-1]:
                        w_new = weights[-2] + weights[-1]
                        m_new = (weights[-2]*means[-2] + weights[-1]*means[-1]) / (w_new + 1e-12)
                        means[-2]=m_new; weights[-2]=w_new; ends[-2]=ends[-1]
                        means.pop(); weights.pop(); starts.pop(); ends.pop()
                yhat = np.empty_like(y)
                for m, s, e in zip(means, starts, ends):
                    yhat[s:e+1] = m
                return yhat

            yhat = pava_nonincreasing(y, w)
            
            # Extrapolate to 0 visually
            y0 = float(yhat[0])
            x_plot = np.concatenate(([0.0], x))
            y_plot = np.concatenate(([y0], yhat))

            ax.plot(x_plot, y_plot, color='tab:orange', linestyle='--', alpha=0.5, label='Monotone Fit')
            ax.scatter([0.0], [y0], color='tab:red', marker='x', s=60, zorder=5)

        ax.set_xlabel('Δ Eye Trajectory (a.u.)')
        ax.set_ylabel('Covariance')
        ax.set_title(f'Neuron Pair ({i}, {j}) | Window {win_ms} ms')
        ax.legend(loc='best', frameon=False)
        ax.grid(True, alpha=0.2)
        
        # Force zero line for reference
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)

        if show and created_fig:
            plt.show()

        return fig, ax
    
#%% Load Dataset
from tqdm import tqdm

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

dataset_idx = 2
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
neuron_mask = np.where(np.nansum(robs, (0,1))>500)[0]
analyzer = DualWindowAnalysis(robs[:,:200][:,:,neuron_mask], eyepos[:,:200], valid_mask[:,:200], dt=1/240)

# 2. Run Sweep
windows = [5, 10, 20, 40, 80, 100, 150]
results, last_mats = analyzer.run_sweep(windows, t_hist_ms=windows[0])

#%%
for i in range(10):
    for j in range(10):
        analyzer.inspect_neuron_pair(i, j, 100, ax=None, show=True)

#%%
analyzer.inspect_neuron_pair(0, 10, 150, ax=None, show=True)

#%% 3. Plot Fano Factor Scaling
window_ms = [results[i]['window_ms'] for i in range(len(results))]
ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
ff_corr = np.zeros_like(window_ms, dtype=np.float64)
ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
for iwindow in range(len(window_ms)):
    ff_uncorr[iwindow] = np.nanmean(results[iwindow]['ff_uncorr'])
    ff_corr[iwindow] = np.nanmean(results[iwindow]['ff_corr'])
    ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'])
    ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'])

plt.figure(figsize=(8, 6))
plt.plot(window_ms, ff_uncorr, 'o-', label='Standard (Uncorrected)')
plt.plot(window_ms, ff_corr, 'o-', label='FEM-Corrected')
# plot error bars
plt.fill_between(window_ms, ff_uncorr - ff_uncorr_std, ff_uncorr + ff_uncorr_std, alpha=0.2)
plt.fill_between(window_ms, ff_corr - ff_corr_std, ff_corr + ff_corr_std, alpha=0.2)

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Count Window (ms)')
plt.ylabel('Mean Fano Factor')
plt.title('Integration of Noise: FEM Correction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
window_idx = 3
alpha = np.diag(last_mats[window_idx]['PSTH'])/np.diag(last_mats[window_idx]['Intercept'])
plt.figure()
plt.hist(1-alpha, bins=50)
plt.xlabel('1 - alpha')
plt.ylabel('Count')
plt.title(f'1 - alpha, {windows[window_idx]}ms')
plt.xlim(0, 1)
plt.show()
#%%
# 4. Check Rank of the last window (e.g., 150ms)
window_idx = -1
Sigma_FEM = last_mats[window_idx]['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
plt.figure(figsize=(12, 4))
# plt.subplot(1,3,1)
plt.plot(np.cumsum(s), 'o-', label='FEM')
plt.title(f"Singular Values FEM ({windows[-1]}ms)")

# same for total covariance
Sigma_Total = last_mats[window_idx]['Total']
u, s, vh = np.linalg.svd(Sigma_Total)
# plt.subplot(1,3,2)
plt.plot(np.cumsum(s), 'o-', label='Total')
plt.title(f"Singular Values Total  ({windows[-1]}ms)")

# now noise cov
Sigma_Noise = last_mats[window_idx]['Noise_Corr']
u, s, vh = np.linalg.svd(Sigma_Noise)
# plt.subplot(1,3,3)
plt.plot(np.cumsum(s), 'o-', label='Noise')
plt.title(f"Singular Values Noise  ({windows[-1]}ms)")
plt.show()
# %%



i = 2
plt.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o')
plt.axhline(1, color='k', linestyle='--', alpha=0.5)
plt.axvline(1, color='k', linestyle='--', alpha=0.5)
# plot means
plt.plot(np.mean(results[i]['ff_uncorr']), np.mean(results[i]['ff_corr']), 'ko')

plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')
plt.title(f"FF Window Size ({windows[i]}ms)")

#%%
results[0]
# %%
# show the total covariance matrix subtracting the diagonal
window_idx = 2
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(last_mats[window_idx]['Total'] - np.diag(np.diag(last_mats[window_idx]['Total'])))
plt.colorbar()
plt.title(f"Total Covariance ({windows[window_idx]}ms)")

# show FEM
plt.subplot(1,3,2)
plt.imshow(last_mats[window_idx]['FEM'] - np.diag(np.diag(last_mats[window_idx]['FEM'])))
plt.colorbar()
plt.title(f"FEM Covariance ({windows[window_idx]}ms)")

# show Noise_Corr
plt.subplot(1,3,3)
plt.imshow(last_mats[window_idx]['PSTH'] - np.diag(np.diag(last_mats[window_idx]['PSTH'])))
plt.colorbar()
plt.title(f"PSTH Covariance ({windows[window_idx]}ms)")



# %%
Sigma_Noise_uncorrected = last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH']
Sigma_Noise_corrected = last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'] - last_mats[window_idx]['PSTH']
def cov_to_corr(C):
    C = torch.tensor(C)
    # 1. Get the variances (diagonal elements)
    variances = torch.diag(C)
    
    # 2. Get standard deviations
    # Clamp to avoid division by zero if a neuron is silent
    std_devs = torch.sqrt(variances).clamp(min=1e-8)
    
    # 3. Outer product to create the denominator matrix
    # shape: (n, n) where entry (i, j) is sigma_i * sigma_j
    outer_std = torch.outer(std_devs, std_devs)
    
    # 4. Normalize
    R = C / outer_std
    
    # set diag to 0
    R = R - torch.diag(torch.diag(R))
    
    return R

plt.subplot(1,2,1)
plt.imshow(cov_to_corr(Sigma_Noise_uncorrected))
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(cov_to_corr(Sigma_Noise_corrected))
plt.colorbar()

# %%
