
def cov_to_corr(C, min_var=1e-3):
    """
    Converts covariance to correlation (N x N).
    Returns NaNs for neurons with unstable, vanishing, or negative variance.
    """
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32)
    
    # 1. Get variances (diagonal)
    variances = torch.diag(C)
    
    # 2. Identify Valid Neurons
    # We require variance to be strictly positive and above the noise floor.
    # Neurons with NaN variance (from run_sweep) or tiny variance (survivors) fail this.
    valid_mask = variances > min_var
    
    # 3. Compute Standard Deviations
    # Initialize with NaNs so that invalid neurons automatically produce NaN correlations
    std_devs = torch.full_like(variances, float('nan'))
    std_devs[valid_mask] = torch.sqrt(variances[valid_mask])
    
    # 4. Outer Product (N x N)
    # Any row/col with a NaN std_dev will result in a NaN row/col in the denominator
    outer_std = torch.outer(std_devs, std_devs)
    
    # 5. Normalize
    # Division by NaN (or zero) results in NaN, which is exactly what we want.
    R = C / outer_std
    
    # 6. Clamp to [-1, 1]
    # torch.clamp passes NaNs through unchanged, but restricts valid values to physical limits.
    R = torch.clamp(R, -1.0, 1.0)
    
    # 7. Set diagonal to 0
    # (Standard practice for noise correlations)
    R.fill_diagonal_(0.0)
    
    return R.numpy()

def pava_nonincreasing_with_blocks(y, w, eps=1e-12):
    # weighted isotonic regression using PAVA (Pool-Adjacent-Violators Algorithm)
    # enforces the fitted sequence is non-increasing
    # in other words: covariance should not increase with eye distance.
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
        while len(means) >= 2 and means[-2] < means[-1]:
            w_new = weights[-2] + weights[-1]
            m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / (w_new + eps)
            means[-2] = m_new
            weights[-2] = w_new
            ends[-2] = ends[-1]
            means.pop(); weights.pop(); starts.pop(); ends.pop()
    yhat = np.empty_like(y)
    blocks = []
    for m, s, e, ww in zip(means, starts, ends, weights):
        yhat[s:e+1] = m
        blocks.append((s, e, float(m), float(ww)))
    return yhat, blocks

def slope_ci_t(res, n, ci=0.95):
    """Parametric CI using linregress stderr and t critical value."""
    df = n - 2
    tcrit = stats.t.ppf(0.5 + ci/2, df)
    lo = res.slope - tcrit * res.stderr
    hi = res.slope + tcrit * res.stderr
    return lo, hi

def bootstrap_mean_ci(x, n_boot=5000, ci=95, seed=0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return x.mean(), (lo, hi)

def bootstrap_slope_ci(x, y, nboot=5000, ci=0.95, rng=0):
    """
    Nonparametric bootstrap: resample (x_i, y_i) pairs.
    Returns (slope_hat, lo, hi, slopes_boot).
    """
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(rng)

    slopes = np.empty(nboot, dtype=float)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        slopes[b] = stats.linregress(x[idx], y[idx]).slope

    alpha = 1 - ci
    lo, hi = np.quantile(slopes, [alpha/2, 1 - alpha/2])
    slope_hat = stats.linregress(x, y).slope
    return slope_hat, lo, hi, slopes

def plot_slope_estimation(ax, means, variances, title, color, label=''):
    # Filter
    valid = (means > 0.1) & np.isfinite(variances) & np.isfinite(means)
    x = np.asarray(means[valid])
    y = np.asarray(variances[valid])

    res = stats.linregress(x, y)
    
    ax.scatter(x, y, s=15, alpha=0.6, c=color, label=f'{label} FF = {res.slope:.2f}')

    x_line = np.linspace(0, x.max(), 100)
    y_line = res.slope * x_line + res.intercept
    ax.plot(x_line, y_line, 'k--', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Mean Rate (spk/s)")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return res, x, y  # return x,y too so we can bootstrap outside

def get_upper_triangle(C): # used to get the correlation values
    rows, cols = np.triu_indices_from(C, k=1)
    v = C[rows, cols]
    return v

def index_cov(cov_matrix, indices):
    # index into a square matrix
    return cov_matrix[indices][:, indices]
        
# ----------------------------
# Main analysis
# ----------------------------

# Law of total covariance decomposition    
class DualWindowAnalysis:
    """
    Covariance decomposition conditioned on eye trajectory similarity.
    
    - We estimate second moments E[S_i S_j | distance bin] (time matched), then fit intercept at d -> 0+
    - Covariance: Cov = E[SS^T] - E[S]E[S]^T
    - The Law of Total Covariance States:
        Cov[S] = E[Cov[S | d]] + Cov[E[S | d]]


    """

    def __init__(self, robs, eyepos, valid_mask,
                dt=1/240,
                min_seg_len=36,
                device="cuda"):
        '''
        robs: (tr, t, cells) spike counts
        eyepos: (tr, t, 2) eye positions
        valid_mask: (tr, t) boolean mask of valid times
        '''
        self.dt = float(dt)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Initializing on {self.device}...")
        t0 = time.time()

        # sanitize
        if np.isnan(robs).any():
            robs = np.nan_to_num(robs, nan=0.0)
        eyepos = np.nan_to_num(eyepos, nan=0.0)

        self.robs = torch.tensor(robs, dtype=torch.float32, device=self.device)
        self.eyepos = torch.tensor(eyepos, dtype=torch.float32, device=self.device)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

        self.n_trials, self.n_time, self.n_cells = robs.shape

        # PSTH per time (mean across valid trials)
        valid_float = self.valid_mask.float().unsqueeze(-1)  # (tr, t, 1)
        sum_spikes = torch.sum(self.robs * valid_float, dim=0)  # (t, cells)
        cnt = torch.sum(valid_float, dim=0)  # (t, 1)

        # keep NaNs out of PSTH
        psth = torch.full((self.n_time, self.n_cells), float("nan"), device=self.device)
        ok = (cnt[:, 0] > 0)
        psth[ok] = sum_spikes[ok] / cnt[ok]
        self.psth = psth

        # break the data into valid contiguous segments
        self.segments = self._get_valid_segments(min_len_bins=min_seg_len)

        self.window_summaries = {}
        print(f"Loaded {len(self.segments)} valid segments. Init took {time.time()-t0:.2f}s")

    def _get_valid_segments(self, min_len_bins):
        segments = []
        mask_cpu = self.valid_mask.detach().cpu().numpy()
        for tr in range(self.n_trials):
            padded = np.concatenate(([False], mask_cpu[tr], [False]))
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            stops = np.where(diffs == -1)[0]
            for s, e in zip(starts, stops):
                if (e - s) >= min_len_bins:
                    segments.append((tr, s, e))
        return segments

    
    # window extraction 
    def _extract_windows(self, t_count, t_hist):
        """
        Inputs:
          - t_count: number of bins in count window
          - t_hist:  number of bins in history window (used for trajectory similarity)
        
        Returns:
          - SpikeCounts:  (N, cells) summed counts over count window
          - EyeTraj:      (N, t_hist, 2) eye positions over history
          - T_idx:        (N,) time index of start of count window (aligned label)
        """
        total_len = t_hist + t_count
        trial_indices, time_indices = [], []

        for (tr, start, stop) in self.segments:
            if (stop - start) < total_len:
                print(f"  Skipping trial {tr} - not enough time ({stop-start} < {total_len})")
                continue
            
            t_starts = np.arange(start, stop - total_len + 1, t_count)
            trial_indices.extend([tr] * len(t_starts))
            time_indices.extend(t_starts)

        if len(trial_indices) == 0:
            return None, None, None

        n_total = len(trial_indices)
        print(f"  Found {n_total} total windows before subsampling")
    
        idx_tr = torch.tensor(trial_indices, device=self.device, dtype=torch.long)
        idx_t0 = torch.tensor(time_indices, device=self.device, dtype=torch.long)

        # gather eye history+count then slice history
        offsets = torch.arange(total_len, device=self.device).unsqueeze(0)  # (1, total_len)
        gather_t = idx_t0.unsqueeze(1) + offsets                            # (N, total_len)
        gather_tr = idx_tr.unsqueeze(1).expand(-1, total_len)               # (N, total_len)

        EyeTraj = self.eyepos[gather_tr, gather_t, :]                             # (N, total_len, 2)

        # gather spikes only over count window
        spike_offsets = torch.arange(t_hist, total_len, device=self.device).unsqueeze(0)  # (1, t_count)
        gather_t_spk = idx_t0.unsqueeze(1) + spike_offsets                                 # (N, t_count)
        gather_tr_spk = idx_tr.unsqueeze(1).expand(-1, t_count)                            # (N, t_count)

        S_raw = self.robs[gather_tr_spk, gather_t_spk, :]                  # (N, t_count, cells)
        SpikeCounts = torch.sum(S_raw, dim=1)                                        # (N, cells)

        # aligned time label (start of count window)
        T_idx = idx_t0 + t_hist                                            # (N,)

        return SpikeCounts, EyeTraj, T_idx, idx_tr

    # Calculate second moment
    def _calculate_second_moment(self, SpikeCounts, EyeTraj, T_idx, n_bins=25):
        """
        Calculate second moment E[SS^T | d] for all pairs of samples
        use split half cross-validation to estimate E[SS^T]
    
        """
        
        # OLD: bins are mean euclidean distance. we had to move away from this because there's no way to do it on GPU without blowing up memory
        # diff = torch.sqrt( torch.sum((EyeTraj[:, None, :, :] - EyeTraj[None, :, :, :])**2,-1)).mean(2)       # (N, N, T, 2)
        # i, j = np.triu_indices_from(diff)
        # dist = diff[i,j]

        # bins are RMS distance. It's not an unreasonable metric for similarity, but we 
        # favor it over euclidean because there is a fast pytorch implementation on gpu (cdist)
        
        # Flatten time and coordinate dimensions: (N, T, 2) -> (N, 2T)
        N_samples, T, _ = EyeTraj.shape
        EyeFlat = EyeTraj.reshape(N_samples, -1) 

        # Compute RMS distance matrix
        dist_matrix = torch.cdist(EyeFlat, EyeFlat) / np.sqrt(T)

        # Extract upper triangle for percentiles
        i, j = torch.triu_indices(N_samples, N_samples, offset=1)
        dist = dist_matrix[i, j]

        if isinstance(n_bins, int):
            bin_edges = np.percentile(dist.cpu().numpy(), np.arange(0, 100, 100/(n_bins+1)))
        else:
            bin_edges = n_bins
            

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_bins = len(bin_edges) - 1

        unique_times = np.unique(T_idx.detach().cpu().numpy())
        C = SpikeCounts.shape[1]
        T = EyeTraj.shape[1]
        device = EyeTraj.device

        bin_edges_t = torch.as_tensor(bin_edges, device=device, dtype=EyeTraj.dtype)
        inv_sqrt_T = (1.0 / torch.sqrt(torch.tensor(float(T), device=device, dtype=EyeTraj.dtype)))

        # keep accumulators on CPU as torch, convert to numpy at end
        SS_e_t = torch.zeros((n_bins, C, C), device='cpu', dtype=torch.float64)
        count_e_t = torch.zeros((n_bins,), device='cpu', dtype=torch.long)

        def accumulate_split(valid_idx, SS_e_t, count_e_t):
            # valid_idx: 1D numpy array of trial indices for this split
            N = len(valid_idx)
            if N < 2:
                return

            X = EyeTraj[valid_idx]                 # (N, T, 2)
            S = SpikeCounts[valid_idx]             # (N, C)

            # Pair list for cross-trial only
            ii, jj = torch.triu_indices(N, N, offset=1, device=device)  # (P,), (P,)

            # Eye distances on those pairs, without diff materialization
            Xflat = X.reshape(N, -1)                                   # (N, 2T)
            D = torch.cdist(Xflat, Xflat) * inv_sqrt_T                 # (N, N)
            d = D[ii, jj]                                              # (P,)

            # Bin IDs in 1..n_bins are interior bins (same convention as your np.digitize + (k+1))
            # bucketize returns in [0..len(edges)] where edges includes endpoints.
            bid = torch.bucketize(d, bin_edges_t, right=False)         # (P,)

            # Keep only pairs that fall into bins 1..n_bins
            ok = (bid >= 1) & (bid <= n_bins)
            if not ok.any():
                return
            ii = ii[ok]; jj = jj[ok]; bid = bid[ok]                    # (P',)

            # We’ll accumulate per bin with S_i^T @ S_j
            # (still no (P,C,C) tensor materialized)
            for k in range(1, n_bins + 1):
                mk = (bid == k)
                if not mk.any():
                    continue
                Si = S[ii[mk]]                                         # (P_k, C)
                Sj = S[jj[mk]]                                         # (P_k, C)

                # sum_p Si[p]^T Sj[p]  -> (C, C)
                # do on GPU, then move the (C,C) result to CPU accumulator
                M = Si.transpose(0, 1).matmul(Sj)                      # (C, C)

                SS_e_t[k-1] += M.detach().cpu().to(torch.float64)
                count_e_t[k-1] += mk.sum().detach().cpu()

        
        for t in unique_times:
            valid = np.where((T_idx == t).detach().cpu().numpy())[0]
            if len(valid) < 10:
                continue

            accumulate_split(valid, SS_e_t, count_e_t)

        # Convert to numpy and form split-half estimate
        SS_e = SS_e_t.numpy()
        count_e = count_e_t.numpy()

        MM = SS_e / count_e[:, None, None]
        # symmetrize
        MM = 0.5 * (MM + np.swapaxes(MM, -1, -2))

        return MM, bin_centers, count_e, bin_edges
    
    def _naive_psth_covariance(self, S, T_idx, min_trials_per_time=10):
        """
        Computes the covariance of the trial-averaged PSTH (Naive Estimator).
        
        Bias: BIASED UP (Upper Bound).
        Includes the standard error of the mean: C_naive = C_signal + (1/N)*C_noise
        """
        unique_times = np.unique(T_idx.detach().cpu().numpy())
        N_cells = S.shape[1]
        
        # 1. Compute PSTH (Mean across trials per time point)
        psth_list = []
        
        for t in unique_times:
            # Get all trials for this time point
            mask = (T_idx == t)
            
            # Check trial count constraint
            if mask.sum() < min_trials_per_time:
                continue
                
            # Compute mean (PSTH for this time bin)
            # S[mask] shape is (n_trials, n_cells) -> mean is (n_cells,)
            mu_t = S[mask].mean(0).detach().cpu().numpy()
            psth_list.append(mu_t)

        if len(psth_list) < 2:
            return np.full((N_cells, N_cells), np.nan), None, None

        # Stack into (T_valid, N_cells) matrix
        PSTH = np.stack(psth_list)
        
        # 2. Compute Covariance of the Means
        # Center the data
        PSTH_centered = PSTH - PSTH.mean(0, keepdims=True)
        
        # Standard sample covariance formula (divide by T-1)
        C_naive = (PSTH_centered.T @ PSTH_centered) / (PSTH.shape[0] - 1)
        
        # Symmetrize (numerical hygiene)
        C_naive = 0.5 * (C_naive + C_naive.T)

        return C_naive, PSTH, PSTH
    
    # unbiased PSTH covariance
    def _bagged_split_half_psth_covariance(self, S, T_idx, n_boot=20, min_trials_per_time=10, seed=42, global_mean=None):
        """
        Computes the Split-Half PSTH covariance averaged over multiple random splits (Bagging).
        
        Args:
            S (torch.Tensor): Spike counts (N_samples, N_cells)
            T_idx (torch.Tensor): Time indices (N_samples,)
            global_mean (np.ndarray, optional): 
                The global mean firing rate vector (Erate) used to center Crate.
                If provided, this function subtracts global_mean from the split-halves
                instead of their local means. This ensures that C_rate and C_psth 
                share the same centering logic, eliminating bias due to drift.
        """
        rng = np.random.default_rng(seed)
        unique_times = np.unique(T_idx.detach().cpu().numpy())
        N_cells = S.shape[1]
        
        # Pre-calculate indices for speed
        time_groups = {}
        for t in unique_times:
            ix_t = np.where((T_idx == t).detach().cpu().numpy())[0]
            if len(ix_t) >= min_trials_per_time:
                time_groups[t] = ix_t

        if len(time_groups) < 2:
             return np.full((N_cells, N_cells), np.nan), None, None

        C_accum = np.zeros((N_cells, N_cells))
        valid_boots = 0

        # Mean PSTH accumulator for visualization
        PSTH_mean_accum = np.zeros((len(time_groups), N_cells))

        # Prepare global mean for broadcasting if provided
        mu_global = None
        if global_mean is not None:
            # Ensure shape is (1, N_cells) for broadcasting against (T, N_cells)
            mu_global = np.asarray(global_mean).reshape(1, -1)

        for k in range(n_boot):
            PSTH_A_list = []
            PSTH_B_list = []
            
            sorted_times = sorted(time_groups.keys())
            
            for t in sorted_times:
                ix_t = time_groups[t]
                
                # Shuffle and Split
                perm = rng.permutation(ix_t)
                mid = len(ix_t) // 2
                
                # Compute means for this time point (on GPU, move to CPU numpy)
                mu_A = S[perm[:mid]].mean(0).detach().cpu().numpy()
                mu_B = S[perm[mid:]].mean(0).detach().cpu().numpy()
                
                PSTH_A_list.append(mu_A)
                PSTH_B_list.append(mu_B)
            
            # Stack to (T, Cells)
            XA = np.stack(PSTH_A_list)
            XB = np.stack(PSTH_B_list)
            
            PSTH_mean_accum += (XA + XB) / 2.0
            
            # --- CENTERING LOGIC ---
            if mu_global is not None:
                # Global centering: Matches C_rate logic (includes drift variance)
                XA_c = XA - mu_global
                XB_c = XB - mu_global
            else:
                # Local centering: Standard covariance (High-pass filters drift)
                XA_c = XA - XA.mean(0, keepdims=True)
                XB_c = XB - XB.mean(0, keepdims=True)
            
            # Unbiased Cross-Covariance
            n_time = XA.shape[0]
            C_k = (XA_c.T @ XB_c) / (n_time - 1)
            
            # Symmetrize
            C_k = 0.5 * (C_k + C_k.T)
            
            C_accum += C_k
            valid_boots += 1
            
        if valid_boots == 0:
            return np.full((N_cells, N_cells), np.nan), None, None

        C_final = C_accum / valid_boots
        PSTH_final = PSTH_mean_accum / valid_boots
        
        return C_final, PSTH_final, PSTH_final
    
    # unbiased PSTH covariance (split-half cross-covariance)
    def _split_half_psth_covariance(self, S, T_idx, min_trials_per_time=10, seed=0):
        '''
        Split-half cross-covariance to estimate PSTH covariance.
        Because we have finite sample size, we want a robust estimate of the PSTH covariance.
        The logic is as follows:
        Assume responses are û = u + ε. (u = true PSTH, ε = noise)
        Split data into independent halves A,B: û_A = u + ε_A, û_B = u + ε_B.
        Then cov(û_A, û_B) = cov(u,u) + cov(u,ε_B) + cov(ε_A,u) + cov(ε_A,ε_B).
        With independent, zero-mean noise: cov(û_A, û_B) = cov(u,u).
        '''

        # set random seed
        rng = np.random.default_rng(seed)

        unique_times = np.unique(T_idx.detach().cpu().numpy())
        NT = len(unique_times)
        N_cells = S.shape[1]
        N_samples = S.shape[0]

        # Pre-allocate masks (false by default)
        mask_A = np.zeros(N_samples, dtype=bool)
        mask_B = np.zeros(N_samples, dtype=bool)

        # iterate time points to ensure exactly 50/50 split per time bin
        # This minimizes the variance of the split means
        for t in unique_times:
            # Find indices for this specific time point
            # (Note: converting to numpy once outside loop would be faster, but this is clear)
            ix_t = np.where((T_idx == t).detach().cpu().numpy())[0]
            n_t = len(ix_t)

            if n_t < min_trials_per_time:
                continue

            # Shuffle indices for this time point
            perm = rng.permutation(n_t)
            
            # Split indices
            split_idx = n_t // 2
            idx_A_local = ix_t[perm[:split_idx]]
            idx_B_local = ix_t[perm[split_idx:]]

            mask_A[idx_A_local] = True
            mask_B[idx_B_local] = True

        # --- COMPUTE PSTH HALVES ---
        # Initialize with NaNs
        PSTH_A = np.full((NT, N_cells), np.nan)
        PSTH_B = np.full((NT, N_cells), np.nan)

        for it, t in enumerate(unique_times):
            # Intersect time mask with split masks
            # Since we built masks_A/B strictly on time indices, we can just check validity
            ix_t = (T_idx == t).detach().cpu().numpy()
            
            # We must re-verify the intersection to map to the correct row 'it'
            # (mask_A is global, ix_t is local time selector)
            m_A = mask_A & ix_t
            m_B = mask_B & ix_t
            
            # Check if we have data (redundant with loop above but safe)
            if not m_A.any() or not m_B.any():
                continue

            PSTH_A[it] = S[m_A].mean(0).detach().cpu().numpy()
            PSTH_B[it] = S[m_B].mean(0).detach().cpu().numpy()

        # --- UNBIASED COVARIANCE ---
        # Keep only times where both splits were valid
        finite_times = np.isfinite(PSTH_A).all(axis=1) & np.isfinite(PSTH_B).all(axis=1)
        
        if finite_times.sum() < 2:
            # Not enough time points to compute covariance
            return np.full((N_cells, N_cells), np.nan), PSTH_A, PSTH_B

        # Center across time
        # (N_time_valid, N_cells)
        XA = PSTH_A[finite_times] - PSTH_A[finite_times].mean(0, keepdims=True)
        XB = PSTH_B[finite_times] - PSTH_B[finite_times].mean(0, keepdims=True)

        # Unbiased estimator: Divide by (N_time - 1)
        n_time_bins = XA.shape[0]
        Ccv = (XA.T @ XB) / (n_time_bins - 1)

        # Symmetrize
        Ccv = 0.5 * (Ccv + Ccv.T)

        return Ccv, PSTH_A, PSTH_B

    def fit_best_monotonic(self, y, w):
        """
        Fits both non-increasing and non-decreasing PAVA.
        Returns the intercept (yhat[0]) of the fit with the lowest error.
        """
        # 1. Fit Non-Increasing (Classic PAVA)
        y_decr, _ = pava_nonincreasing_with_blocks(y, w)
        sse_decr = np.sum(w * (y - y_decr)**2)
        
        # 2. Fit Non-Decreasing
        # Trick: Negate y, fit non-increasing, then negate result
        y_incr_neg, _ = pava_nonincreasing_with_blocks(-y, w)
        y_incr = -y_incr_neg
        sse_incr = np.sum(w * (y - y_incr)**2)
        
        # 3. Model Selection
        if sse_decr < sse_incr:
            return y_decr[0]
        else:
            return y_incr[0]
        
    def _fit_intercepts_vectorized(self, Ceye, count_e):
        """
        Fits the intercept (d->0) for every element of the covariance matrix.
        Strictly enforces monotonicity (either increasing or decreasing) to handle
        both positive and negative correlations correctly.
        """
        n_bins, n_cells, _ = Ceye.shape
        C_intercept = np.zeros((n_cells, n_cells), dtype=Ceye.dtype)

        # Pre-calculate valid weights once
        # (Assuming count_e is consistent across pairs, which it is)
        # We need to handle potential NaNs in Ceye if binning failed for some reason
        
        for i in range(n_cells):
            # Diagonal: Variance must be non-increasing (Conditioning reduces variance)
            # Technically variance *could* increase if FEMs were suppressing noise, 
            # but physically FEMs add variance. So non-increasing is the correct physical prior for Diagonal.
            y_diag = Ceye[:, i, i]
            yhat, _ = pava_nonincreasing_with_blocks(y_diag, count_e)
            C_intercept[i, i] = yhat[0]

            # Off-Diagonals: Can be increasing OR decreasing
            for j in range(i + 1, n_cells):
                y = Ceye[:, i, j]
                
                # Handle NaNs if strictly necessary (though Ceye shouldn't have them if logic is tight)
                valid = np.isfinite(y)
                if not valid.any():
                    C_intercept[i, j] = np.nan
                    C_intercept[j, i] = np.nan
                    continue
                
                val = self.fit_best_monotonic(y[valid], count_e[valid])
                
                C_intercept[i, j] = val
                C_intercept[j, i] = val

        return C_intercept
    
    # def _fit_intercepts_linear(self, Ceye, bin_centers, count_e, d_max=0.4, min_bins=3, eps=1e-12):
    #     """
    #     Weighted local linear regression intercept for each (i,j):
    #         y(d) ~ b0 + b1*d   for d in (0, d_max]
    #     weights w = count_e.

    #     Returns:
    #         C_intercept: (n_cells, n_cells)
    #     """
    #     n_bins, n_cells, _ = Ceye.shape
    #     C_intercept = np.full((n_cells, n_cells), np.nan, dtype=Ceye.dtype)

    #     x = np.asarray(bin_centers, dtype=np.float64)
    #     w_all = np.asarray(count_e, dtype=np.float64)

    #     # choose local bins
    #     use = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w_all) & (w_all > 0)
    #     idx = np.where(use)[0]
    #     if idx.size < min_bins:
    #         # not enough support: safest is to fall back to first finite bin
    #         k0 = np.where(np.isfinite(x) & np.isfinite(w_all) & (w_all > 0))[0]
    #         if k0.size > 0:
    #             return Ceye[k0[0]].copy()
    #         return C_intercept  # all NaN

    #     x_loc = x[idx]
    #     w_loc = w_all[idx]
    #     # Precompute weighted design matrix pieces for speed
    #     # X = [1, x]
    #     S0 = np.sum(w_loc)
    #     S1 = np.sum(w_loc * x_loc)
    #     S2 = np.sum(w_loc * x_loc * x_loc)
    #     det = (S0 * S2 - S1 * S1)

    #     if det < eps:
    #         # degenerate x; fall back
    #         return Ceye[idx[0]].copy()

    #     for i in range(n_cells):
    #         # diagonal
    #         y = Ceye[idx, i, i]
    #         v = np.isfinite(y)
    #         if np.sum(v) >= min_bins:
    #             ww = w_loc[v]; xx = x_loc[v]; yy = y[v]
    #             S0v = np.sum(ww); S1v = np.sum(ww * xx); S2v = np.sum(ww * xx * xx)
    #             T0 = np.sum(ww * yy); T1 = np.sum(ww * xx * yy)
    #             detv = (S0v * S2v - S1v * S1v)
    #             if detv >= eps:
    #                 b0 = (T0 * S2v - T1 * S1v) / detv
    #                 C_intercept[i, i] = b0

    #         for j in range(i + 1, n_cells):
    #             y = Ceye[idx, i, j]
    #             v = np.isfinite(y)
    #             if np.sum(v) < min_bins:
    #                 continue
    #             ww = w_loc[v]; xx = x_loc[v]; yy = y[v]
    #             S0v = np.sum(ww); S1v = np.sum(ww * xx); S2v = np.sum(ww * xx * xx)
    #             T0 = np.sum(ww * yy); T1 = np.sum(ww * xx * yy)
    #             detv = (S0v * S2v - S1v * S1v)
    #             if detv < eps:
    #                 continue
    #             b0 = (T0 * S2v - T1 * S1v) / detv
    #             C_intercept[i, j] = b0
    #             C_intercept[j, i] = b0

    #     return C_intercept
    def _fit_intercepts_linear(self, Ceye, bin_centers, count_e, d_max=0.4, min_bins=3, eps=1e-8, eval_at_first_bin=True):
        """
        Weighted local linear regression with physical constraints.
        
        Safeguards:
        1. Slope Constraint: Forces slope <= 0. If correlation increases with distance (noise), 
           we assume the true function is flat (return weighted mean).
        2. Extrapolation Control: If eval_at_first_bin=True, returns the fitted value 
           at the first valid bin center (Lower Bound) rather than d=0 (Upper Bound).
        3. Scale Invariance: Uses correlation check for determinant stability.
        """
        n_bins, n_cells, _ = Ceye.shape
        C_intercept = np.full((n_cells, n_cells), np.nan, dtype=Ceye.dtype)

        x = np.asarray(bin_centers, dtype=np.float64)
        w_all = np.asarray(count_e, dtype=np.float64)

        # Identify global valid bins used for indices
        use_mask = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w_all) & (w_all > 0)
        idx = np.where(use_mask)[0]
        
        # Fallback if insufficient data
        if idx.size < min_bins:
            # Fallback: Just return the raw first valid bin if it exists
            k0 = np.where(np.isfinite(x) & np.isfinite(w_all) & (w_all > 0))[0]
            if k0.size > 0:
                return Ceye[k0[0]].copy()
            return C_intercept 

        x_loc = x[idx]
        w_loc = w_all[idx]
        
        # Determine evaluation point (x_eval)
        # If eval_at_first_bin is True, we evaluate at x_loc[0] (Lower Bound)
        # If False, we evaluate at 0.0 (extrapolated Upper Bound)
        x_eval = x_loc[0] if eval_at_first_bin else 0.0

        # --- Precompute Design Matrix Statistics ---
        # We solve: argmin sum w * (y - (b0 + b1*x))^2
        # Analytic solution involves S0, Sx, Sxx
        S0 = np.sum(w_loc)
        Sx = np.sum(w_loc * x_loc)
        Sxx = np.sum(w_loc * x_loc**2)
        
        # Denominator for Cramer's rule (Determinant of X^T W X)
        # Det = S0 * Sxx - Sx^2
        det = S0 * Sxx - Sx**2
        
        # Robustness Check: Normalize determinant to detect true collinearity vs scaling
        # If variance of X is 0, we can't fit a line.
        # Var(X)_weighted = (Sxx/S0) - (Sx/S0)^2 = det / S0^2
        if S0 == 0 or (det / (S0 * S0)) < eps:
            # Degenerate x (only 1 unique bin center with data?): Fallback to mean
            # We handle this inside the loop by checking det again, or just returning raw bin 0
            return Ceye[idx[0]].copy()

        # Iterate over all pairs (Upper Triangular)
        for i in range(n_cells):
            # 1. Diagonal Elements
            self._fit_single_pair(Ceye, C_intercept, idx, w_loc, x_loc, x_eval, 
                                  S0, Sx, Sxx, det, i, i)

            # 2. Off-Diagonal Elements
            for j in range(i + 1, n_cells):
                self._fit_single_pair(Ceye, C_intercept, idx, w_loc, x_loc, x_eval, 
                                      S0, Sx, Sxx, det, i, j)
                # Symmetry
                C_intercept[j, i] = C_intercept[i, j]

        return C_intercept

    def _fit_single_pair(self, Ceye, C_intercept, idx, w_loc, x_loc, x_eval, 
                         S0, Sx, Sxx, det, i, j):
        """Helper to solve linear system for a single pair (i,j)"""
        y = Ceye[idx, i, j]
        
        # Check y validity (redundant if Ceye is clean, but safe)
        if not np.isfinite(y).all():
            # If we have NaNs in the y-vector for this specific pair, 
            # we must re-calculate sums just for valid points.
            # (Slow path, but rarely hit if data is clean)
            v = np.isfinite(y)
            if np.sum(v) < 3: # min_bins hardcoded here or passed in
                return
            
            wv, xv, yv = w_loc[v], x_loc[v], y[v]
            s0 = np.sum(wv); sx = np.sum(wv * xv); sxx = np.sum(wv * xv**2)
            d = s0 * sxx - sx**2
            if d <= 0: return
            
            sy = np.sum(wv * yv)
            sxy = np.sum(wv * xv * yv)
            
            beta1 = (s0 * sxy - sx * sy) / d
            beta0 = (sxx * sy - sx * sxy) / d
        else:
            # Fast path: use precomputed x-stats
            Sy = np.sum(w_loc * y)
            Sxy = np.sum(w_loc * x_loc * y)
            
            beta1 = (S0 * Sxy - Sx * Sy) / det
            beta0 = (Sxx * Sy - Sx * Sxy) / det

        # # --- Physical Constraints ---
        # # Constraint: Correlation should decay with distance (beta1 <= 0).
        # # If beta1 > 0, it means correlation *increases* as eyes move apart.
        # # This is likely noise. The most conservative valid fit is Flat (Mean).
        # if beta1 > 0:
        #     beta1 = 0.0
        #     # Re-calculate beta0 as weighted mean (since y = b0)
        #     if not np.isfinite(y).all():
        #         v = np.isfinite(y)
        #         beta0 = np.average(y[v], weights=w_loc[v])
        #     else:
        #         beta0 = Sy / S0
        
        # Calculate result
        C_intercept[i, j] = beta0 + beta1 * x_eval

    def _fit_intercepts_bspline(self, Ceye, bin_centers, count_e, d_max=0.4, k=3, n_knots=6, lam=1e-6, min_bins=5):
        """
        Weighted B-spline regression for each (i,j) on bins with d in (0, d_max],
        returning intercept f(0). Uses ridge regularization on spline coefficients.

        Args:
            k: spline degree (3 = cubic)
            n_knots: number of *interior* knots across (0, d_max]
            lam: ridge regularization strength (stabilizes extrapolation to 0)
        """
        import numpy as np
        from scipy.interpolate import BSpline

        n_bins, n_cells, _ = Ceye.shape
        C_intercept = np.full((n_cells, n_cells), np.nan, dtype=Ceye.dtype)

        x = np.asarray(bin_centers, dtype=np.float64)
        w_all = np.asarray(count_e, dtype=np.float64)

        use = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w_all) & (w_all > 0)
        idx = np.where(use)[0]
        if idx.size < min_bins:
            k0 = np.where(np.isfinite(x) & np.isfinite(w_all) & (w_all > 0))[0]
            if k0.size > 0:
                return Ceye[k0[0]].copy()
            return C_intercept

        x_loc = x[idx]
        w_loc = w_all[idx]

        # Build a clamped knot vector on [0, d_max]
        # interior knots uniformly spaced in (0, d_max)
        t_interior = np.linspace(0, d_max, n_knots + 2)[1:-1]
        # clamp with multiplicity k+1 at endpoints
        t = np.concatenate([np.zeros(k+1), t_interior, np.full(k+1, d_max)])

        n_basis = len(t) - (k + 1)

        def design_matrix(xv):
            # Evaluate each basis spline at xv
            B = np.zeros((xv.size, n_basis), dtype=np.float64)
            for b in range(n_basis):
                c = np.zeros(n_basis); c[b] = 1.0
                spl = BSpline(t, c, k, extrapolate=True)
                B[:, b] = spl(xv)
            return B

        Bx = design_matrix(x_loc)  # (m, n_basis)

        # Weighted ridge normal equations pieces that don't depend on y
        # Solve (B^T W B + lam I) a = B^T W y
        W = w_loc[:, None]
        BtWB = (Bx.T @ (W * Bx))
        BtWB_reg = BtWB + lam * np.eye(n_basis)

        # For intercept: evaluate basis at x=0
        B0 = design_matrix(np.array([0.0]))[0]  # (n_basis,)

        # Pre-factorization per-cell-pair is overkill; n_basis is small so just solve directly.

        for i in range(n_cells):
            y = Ceye[idx, i, i]
            v = np.isfinite(y)
            if np.sum(v) >= min_bins:
                Bv = Bx[v]
                wv = w_loc[v]
                rhs = Bv.T @ (wv * y[v])
                A = (Bv.T @ (wv[:, None] * Bv)) + lam * np.eye(n_basis)
                coef = np.linalg.solve(A, rhs)
                C_intercept[i, i] = B0 @ coef

            for j in range(i + 1, n_cells):
                y = Ceye[idx, i, j]
                v = np.isfinite(y)
                if np.sum(v) < min_bins:
                    continue
                Bv = Bx[v]
                wv = w_loc[v]
                rhs = Bv.T @ (wv * y[v])
                A = (Bv.T @ (wv[:, None] * Bv)) + lam * np.eye(n_basis)
                coef = np.linalg.solve(A, rhs)
                val0 = B0 @ coef
                C_intercept[i, j] = val0
                C_intercept[j, i] = val0

        return C_intercept



    def _calculate_Crate(self, SpikeCounts, EyeTraj, T_idx, n_bins=25, Ctotal=None, intercept_mode='linear'):
        """
        Calculate the eye-conditioned covariance matrix (Crate) using split-half cross-validation.

        Inputs:
        -------
        SpikeCounts : torch.Tensor (N, cells)
            Spike counts for each sample
        EyeTraj : torch.Tensor (N, t_hist, 2)
            Eye positions for each sample
        T_idx : torch.Tensor (N,)
            Time index of start of count window (aligned label)
        n_bins : int
            Number of bins to use for eye distance
        
        Returns:
        --------
        Crate : np.ndarray (cells, cells)
            Eye-conditioned covariance matrix
        Erate: np.ndarray (cells,)
            Mean spike counts per cell
        Ceye: np.ndarray (n_bins, cells, cells)
            Raw eye-conditioned covariance matrix (biased estimator)
        bin_centers: np.ndarray (n_bins,)
            Bin centers for eye distance
        count_e: np.ndarray (n_bins,)
            Number of pairs in each bin
        """
        MM, bin_centers, count_e, bin_edges = self._calculate_second_moment(SpikeCounts, EyeTraj, T_idx, n_bins=n_bins)
        Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy() # raw means
        Ceye = MM - Erate[:,None] * Erate[None,:] # raw rate covariances conditioned on eye trajectory

        if intercept_mode == 'linear':
            Crate = self._fit_intercepts_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True) # conservative (evaluate at first bin)
        elif intercept_mode == 'bspline':
            Crate = self._fit_intercepts_bspline(Ceye, bin_centers, count_e) # fit intercepts
        elif intercept_mode == 'isotonic':
            Crate = self._fit_intercepts_vectorized(Ceye, count_e) # fit intercepts
        else:
            Crate = Ceye[0].copy()

        if Ctotal is not None:
            # find neurons that violate the physical limit that the signal covariance cannot exceed the total covariance
            bad_mask = np.diag(Crate) > .99*np.diag(Ctotal)
            # print(f"  Found {bad_mask.sum()} neurons violating physical limit")
            Crate[bad_mask,:] = np.nan
            Crate[:,bad_mask] = np.nan
            Ceye[:,bad_mask,:] = np.nan
            Ceye[:,:,bad_mask] = np.nan
        
        return Crate, Erate, Ceye, bin_centers, count_e, bin_edges

    def run_sweep(self, window_sizes_ms, t_hist_ms=10, n_bins=15, n_shuffles=0, seed=42, intercept_mode='linear'):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        mats_save = []

        print(f"Starting Sweep (Hist={t_hist_ms}ms) with {n_shuffles} shuffles...")
        
        # Generator for shuffling
        rng_shuffle = torch.Generator(device=self.device)
        rng_shuffle.manual_seed(seed)

        for win_ms in tqdm(window_sizes_ms):
            t_count_bins = int(win_ms / (self.dt * 1000))
            t_count_bins = max(1, t_count_bins)

            # 1. Extract Windows
            SpikeCounts, EyeTraj, T_idx, _ = self._extract_windows(t_count_bins, np.maximum(t_hist_bins, t_count_bins))
            
            if SpikeCounts is None:
                continue
                
            n_samples = SpikeCounts.shape[0]
            if n_samples < 100: 
                continue 

            # 2. Total Covariance
            ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
            Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy() 

            # 3. Rate Covariance (using real eye traces)
            Crate, Erate, Ceye, bin_centers, count_e, bin_edges = self._calculate_Crate(
                SpikeCounts, EyeTraj, T_idx, n_bins=n_bins, Ctotal=Ctotal, intercept_mode=intercept_mode
            )

            # 4. PSTH Covariance
            Cpsth, PSTH_A, PSTH_B = self._bagged_split_half_psth_covariance(
                SpikeCounts, 
                T_idx, 
                n_boot=20, 
                min_trials_per_time=10, 
                seed=seed, 
                global_mean=Erate
            )
            
            # 5. Shuffled Analysis (Loop)
            # We re-calculate Crate (the intercept). Cfem_shuff will be derived later as (Crate_shuff - Cpsth).
            shuffled_intercepts = []
            
            if n_shuffles > 0:
                for k in range(n_shuffles):
                    # Permute EyeTraj relative to SpikeCounts
                    # This breaks the causal link but keeps valid trajectory statistics
                    perm = torch.randperm(n_samples, generator=rng_shuffle, device=self.device)
                    EyeTraj_shuff = EyeTraj[perm]
                    
                    # Calculate Intercept for shuffled data
                    Crate_shuff, _, _, _, _, _ = self._calculate_Crate(
                        SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges, Ctotal=Ctotal, intercept_mode=intercept_mode
                    )
                    shuffled_intercepts.append(Crate_shuff)

            # 6. Derived Real Metrics
            Cfem = Crate - Cpsth
            Cfem = 0.5 * (Cfem + Cfem.T) 

            CnoiseU = Ctotal - Cpsth
            CnoiseC = Ctotal - Crate
            CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
            CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)
            
            ff_uncorr = np.diag(CnoiseU) / Erate
            ff_corr = np.diag(CnoiseC) / Erate

            NoiseCorrU = cov_to_corr(CnoiseU)
            NoiseCorrC = cov_to_corr(CnoiseC)

            alpha = np.diag(Cpsth) / np.diag(Crate)

            if np.isnan(Cfem).any():
                rank = np.nan
            else:
                evals = np.linalg.eigvalsh(Cfem)[::-1]
                pos = evals[evals > 0]
                rank = (np.sum(pos[:2]) / np.sum(pos)) if len(pos) > 2 else 1.0

            results.append({
                "window_ms": win_ms,
                "ff_uncorr": ff_uncorr,
                "ff_corr": ff_corr,
                "ff_uncorr_mean": np.nanmean(ff_uncorr),
                "ff_corr_mean": np.nanmean(ff_corr),
                "alpha": alpha,
                "fem_rank_ratio": rank,
                "n_samples": n_samples,
                'Erates': Erate,
                'count_e': count_e
            })

            mats_save.append({
                "Total": Ctotal,
                "PSTH": Cpsth,
                "FEM": Cfem,
                "Intercept": Crate,
                "Shuffled_Intercepts": shuffled_intercepts, # List of (N_cells, N_cells) arrays
                "NoiseCorrU": NoiseCorrU,
                "NoiseCorrC": NoiseCorrC,
                "PSTH_A": PSTH_A,
                "PSTH_B": PSTH_B,
            })

            # Store summary for plotting individual pairs
            win_key = float(win_ms)
            self.window_summaries[win_key] = {
                "bin_centers": bin_centers,
                "binned_covs": Ceye,           
                "bin_counts": count_e,
                "Sigma_Intercept": Crate,          
                "Sigma_PSTH": Cpsth,            
                "Sigma_Total": Ctotal,
                "Sigma_FEM": Cfem,
                "mean_counts": Erate,
            }

        return results, mats_save
    
    # run_sweep
    # def run_sweep(self, window_sizes_ms, t_hist_ms=10, n_bins=15):
    #     t_hist_bins = int(t_hist_ms / (self.dt * 1000))
    #     results = []
    #     mats_save = []

    #     print(f"Starting Sweep (Hist={t_hist_ms}ms)...")

    #     for win_ms in tqdm(window_sizes_ms):
    #         t_count_bins = int(win_ms / (self.dt * 1000))
    #         t_count_bins = max(1, t_count_bins)

    #         # extract windows
    #         SpikeCounts, EyeTraj, T_idx, _ = self._extract_windows(t_count_bins, np.maximum(t_hist_bins, t_count_bins))
    #         n_samples = SpikeCounts.shape[0]
    #         if SpikeCounts is None or n_samples < 100: continue # arbitrary threshold (how much data do we need?)

    #         # total covariance
    #         ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    #         Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy() # total covariance

    #         # calculate eye conditioned covariance
    #         Crate, Erate, Ceye, bin_centers, count_e = self._calculate_Crate(SpikeCounts, EyeTraj, T_idx, n_bins=n_bins, Ctotal=Ctotal)
            
    #         # PSTH covariance
    #         Cpsth, PSTH_A, PSTH_B = self._split_half_psth_covariance(SpikeCounts, T_idx, min_trials_per_time=10, seed=0)

    #         # covariance due to fixational eye movements
    #         Cfem = Crate - Cpsth
    #         Cfem = 0.5 * (Cfem + Cfem.T) # symmetrize

    #         # noise covariance
    #         CnoiseU = Ctotal - Cpsth
    #         CnoiseC = Ctotal - Crate

    #         # symmetrize
    #         CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
    #         CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)
            
    #         # fano factors
    #         ff_uncorr = np.diag(CnoiseU) / Erate
    #         ff_corr = np.diag(CnoiseC) / Erate

    #         # noise correlation
    #         NoiseCorrU = cov_to_corr(CnoiseU)
    #         NoiseCorrC = cov_to_corr(CnoiseC)

    #         alpha = np.diag(Cpsth) / np.diag(Crate)

    #         if np.isnan(Cfem).any():
    #             rank = np.nan
    #         else:
    #             evals = np.linalg.eigvalsh(Cfem)[::-1]
    #             pos = evals[evals > 0]
    #             rank = (np.sum(pos[:2]) / np.sum(pos)) if len(pos) > 2 else 1.0

    #         results.append({
    #             "window_ms": win_ms,
    #             "ff_uncorr": ff_uncorr,
    #             "ff_corr": ff_corr,
    #             "ff_uncorr_mean": np.nanmean(ff_uncorr),
    #             "ff_corr_mean": np.nanmean(ff_corr),
    #             "alpha": alpha,
    #             "fem_rank_ratio": rank,
    #             "n_samples": n_samples,
    #             'Erates': Erate,
    #             'count_e': count_e
    #         })

    #         mats_save.append({
    #             "Total": Ctotal,
    #             "PSTH": Cpsth,
    #             "FEM": Cfem,
    #             "Intercept": Crate,
    #             "NoiseCorrU": NoiseCorrU,
    #             "NoiseCorrC": NoiseCorrC,
    #             "PSTH_A": PSTH_A,
    #             "PSTH_B": PSTH_B,
    #         })

    #         win_key = float(win_ms)
    #         self.window_summaries[win_key] = {
    #             "bin_centers": bin_centers,
    #             "binned_covs": Ceye,          # SECOND MOMENTS (kept name for compatibility)
    #             "bin_counts": count_e,
    #             "Sigma_Intercept": Crate,          # COVARIANCE
    #             "Sigma_PSTH": Cpsth,            # COVARIANCE
    #             "Sigma_Total": Ctotal,
    #             "Sigma_FEM": Cfem,
    #             "mean_counts": Erate,
    #         }

    #     return results, mats_save

    # utility for analyzing the analysis at the resolution of a single neuron or pair
    def inspect_neuron_pair(self, i, j, win_ms, ax=None, show=True):
        """
        Plots COVARIANCE vs distance by converting stored SECOND MOMENTS to covariance
        via subtracting global mean product (mu_i * mu_j), as in McFarland-style derivations.
        """
        import matplotlib.pyplot as plt

        if not self.window_summaries:
            raise RuntimeError("run_sweep must be called before inspecting neuron pairs.")

        win_key = float(win_ms)
        if win_key not in self.window_summaries:
            avail = ", ".join(str(k) for k in sorted(self.window_summaries.keys()))
            raise KeyError(f"Window {win_ms}ms not cached. Available: {avail}")

        summary = self.window_summaries[win_key]
        bin_centers = summary["bin_centers"]
        covs = summary["binned_covs"][:, i, j]     # SECOND MOMENT
        counts = summary["bin_counts"]
        valid = counts > 0
        if not np.any(valid):
            raise RuntimeError("No histogram bins with data for this neuron pair.")
        

        intercept_cov = summary["Sigma_Intercept"][i, j]
        psth_cov = summary["Sigma_PSTH"][i, j]

        created = False
        if ax is None:
            created = True
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
        else:
            fig = ax.figure

        ax.plot(bin_centers[valid], covs[valid], "o", alpha=0.6, label="Measured Covariance")


        ax.axhline(psth_cov, linestyle="--", linewidth=2, label="PSTH Covariance")
        ax.axhline(intercept_cov, linestyle=":", linewidth=2, label="Intercept")

        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Δ Eye Trajectory (a.u.)")
        ax.set_ylabel("Covariance")
        ax.set_title(f"Neuron Pair ({i},{j}) | Window {win_ms} ms")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="best")

        if show and created:
            plt.show()

        return fig, ax



