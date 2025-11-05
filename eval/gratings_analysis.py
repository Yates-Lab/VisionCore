"""
Gratings Analysis Script

This script performs comprehensive analysis of neural responses to drifting grating stimuli.
It analyzes spatial frequency tuning, orientation tuning, temporal dynamics, and phase
response properties of recorded units.

Main components:
1. Data loading and preprocessing
2. Sine wave fitting for phase response analysis
3. Complete gratings analysis pipeline
4. Visualization functions

Author: Ryan
"""

#%% IMPORTS AND SETUP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from pathlib import Path
import torch


# DataYatesV1 imports
from DataYatesV1 import plot_stas, get_free_device, get_session, DictDataset
from DataYatesV1 import enable_autoreload, calc_sta
from DataYatesV1.utils.modeling.general import get_valid_dfs

# Enable automatic reloading of modules for development
enable_autoreload()

#%% EXPERIMENTAL PARAMETERS
# # Session information
# subject = 'Allen'
# date = '2022-04-13'

# # Analysis parameters
# n_lags = 20          # Number of time lags for STA calculation
# dt = 1/240           # Time step duration (seconds)

# #%% DATA LOADING
# print("Loading session data...")
# sess = get_session(subject, date)
# device = get_free_device(1)

# # Load gratings dataset
# print("Loading gratings dataset...")
# gratings_dset = DictDataset.load(sess.sess_dir / 'shifter' / 'gratings_shifted.dset')

# # Get valid data frames (frames with sufficient history for STA calculation)
# gratings_dset['dfs'] = get_valid_dfs(gratings_dset, n_lags)
# gratings_inds = gratings_dset['dfs'].squeeze().nonzero(as_tuple=True)[0]

# print(f'Gratings dataset size: {len(gratings_inds)} / {len(gratings_dset)} '
#       f'({len(gratings_inds)/len(gratings_dset)*100:.2f}%)')
# print(f'Dataset contains {gratings_dset["robs"].shape[1]} units')


#%% SINE WAVE FITTING FUNCTION

def fit_sine(phases, spikes, omega=1.0, variance_source='observed_y'):
    """
    Fits a sine wave of the form f(x) = K*sin(omega*x) + L*cos(omega*x) + C
    (which can be rewritten as A*sin(omega*x + phi_0) + C) to the data
    using ordinary least squares (OLS). It then calculates the standard errors
    of the fitted parameters (K, L, C), derived quantities (Amplitude A,
    Phase offset phi_0), and a Modulation Index (MI).

    The method for calculating parameter variances depends on the 'variance_source'
    argument. For 'observed_y' or 'fitted_y', it uses a heteroscedasticity-
    consistent covariance estimator based on the idea that if beta_hat = M * y,
    then Cov(beta_hat) = M * Cov(y) * M^T, where Var(y_i) is approximated.
    For 'mse', it uses the standard OLS homoscedastic assumption.

    Parameters:
    -----------
    phases : array_like
        Observed phase values (x-values). These are typically in radians.
        If omega is not 1.0, these are the raw phase values, and the function
        will compute omega*phases.
    spikes : array_like
        Observed spike counts (y-values) corresponding to each phase.
    omega : float, optional
        Angular frequency of the sine wave (default is 1.0). If 'phases'
        already represent omega*x (i.e., phases are pre-multiplied by omega),
        then omega should be set to 1.0.
    variance_source : str, optional
        Method for estimating the variance of observations (Var(y_i)) when
        calculating the standard errors of the fitted parameters.
        - 'observed_y': Var(y_i) is approximated by y_i (the observed spike count).
                          This is suitable for Poisson-like data. Values are clipped
                          at a minimum of 1e-6 to prevent issues with zero counts. (Default)
        - 'fitted_y': Var(y_i) is approximated by the fitted y_i value. Clipped at 1e-6.
        - 'mse': Assumes homoscedastic errors (Var(y_i) = constant sigma_e^2).
                 sigma_e^2 is estimated by the Mean Squared Error (MSE) from OLS.
                 The covariance matrix of parameters is then MSE * (X^T X)^-1.

    Returns:
    --------
    dict
        A dictionary containing the fitted parameters, their standard errors,
        and other relevant statistics:
        - 'K': float, coefficient for the sin(omega*x) term.
        - 'L': float, coefficient for the cos(omega*x) term.
        - 'C': float, constant offset (baseline).
        - 'K_se': float, standard error of K.
        - 'L_se': float, standard error of L.
        - 'C_se': float, standard error of C.
        - 'amplitude': float, Amplitude A = sqrt(K^2 + L^2).
        - 'amplitude_se': float, Standard error of A, calculated via error propagation.
        - 'phase_offset_rad': float, Phase offset phi_0 = atan2(L, K) in radians.
                                The model can be written as A*sin(omega*x + phi_0) + C.
        - 'phase_offset_rad_se': float, Standard error of phi_0, via error propagation.
        - 'modulation_index': float, Modulation Index MI = 2A / (A+C).
                                NaN if A+C is close to zero, or if A or C is NaN.
        - 'modulation_index_se': float, Standard error of MI, via error propagation.
                                   NaN if MI is NaN or its variance cannot be computed.
        - 'y_fit': numpy.ndarray, The fitted spike counts (predicted y-values).
        - 'R_squared': float, Coefficient of determination (goodness of fit).
        - 'covariance_matrix_params': numpy.ndarray, The covariance matrix for [K, L, C].
        - 'condition_number_XTX': float, Condition number of the X^T*X matrix.
                                     High values (>1e8 to 1e10) may indicate
                                     multicollinearity and unstable parameter estimates.
    Raises:
    -------
    ValueError:
        If the number of data points is less than the number of parameters (3),
        if the X^T*X matrix is singular (e.g., phases are not distinct enough),
        or if 'variance_source' is an invalid option.
    """
    # Ensure inputs are numpy arrays for consistent operations
    phases = np.asarray(phases)
    spikes = np.asarray(spikes)
    n_points = len(phases)

    # Check if there are enough data points to fit the three parameters (K, L, C)
    if n_points < 3:
        raise ValueError("At least 3 data points are required to fit K, L, and C.")

    # Construct the design matrix X for the OLS regression.
    # The model is y = K*sin(omega*x) + L*cos(omega*x) + C*1
    # So, each row in X corresponds to an observation (phase, spike_count)
    # and columns are [sin(omega*phase_i), cos(omega*phase_i), 1]
    X = np.vstack([
        np.sin(omega * phases),  # First regressor: sin(omega*x)
        np.cos(omega * phases),  # Second regressor: cos(omega*x)
        np.ones(n_points)        # Third regressor: constant term (for C)
    ]).T  # Transpose to get observations in rows, regressors in columns

    # Calculate X^T * X (often denoted XTX)
    # This matrix is crucial for OLS parameter estimation.
    XTX = X.T @ X  # '@' is the matrix multiplication operator in Python 3.5+

    # Calculate the condition number of XTX.
    # The condition number gives an indication of the sensitivity of the solution
    # of a linear system to errors in the data. A high condition number
    # (e.g., > 1e8 or 1e10) suggests multicollinearity, meaning the regressors
    # are highly correlated, which can lead to unstable parameter estimates.
    condition_number_XTX = np.linalg.cond(XTX)
    if condition_number_XTX > 1e10: # A common threshold for concern
        print(f"Warning: Condition number of X^T*X is high ({condition_number_XTX:.2e}), "
              "results might be unstable. This can happen if phases are not well distributed "
              "or if omega is chosen such that sin(omega*x) and cos(omega*x) become "
              "linearly dependent for the given phases.")

    # Calculate the inverse of XTX, (X^T * X)^-1
    # This is needed for both parameter estimation and their covariance matrix.
    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        # This error occurs if XTX is singular (or nearly singular), meaning it cannot be inverted.
        # This typically happens if the columns of X are linearly dependent (perfect multicollinearity).
        # For example, if all phase values are the same or separated by multiples of 2pi/omega.
        raise ValueError("X^T*X matrix is singular. Cannot compute fit. "
                         "Check if phase values are sufficiently distinct and well-distributed.")

    # --- OLS Parameter Estimation ---
    # The OLS estimator for beta = [K, L, C]^T is beta_hat = (X^T*X)^-1 * X^T * y
    beta_hat = XTX_inv @ X.T @ spikes
    K, L, C = beta_hat[0], beta_hat[1], beta_hat[2]

    # Calculate the fitted y values (y_fit = X * beta_hat)
    y_fit = X @ beta_hat

    # --- Goodness of Fit: R-squared ---
    # R-squared = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    # Sum of Squared Residuals (SSR) = sum((y_i - y_fit_i)^2)
    ss_residual = np.sum((spikes - y_fit)**2)
    # Total Sum of Squares (SST) = sum((y_i - mean(y))^2)
    ss_total = np.sum((spikes - np.mean(spikes))**2)
    
    if ss_total == 0:
        # Handle the case where all spike counts are the same.
        # If ss_total is 0, it means all y values are identical.
        # If ss_residual is also (close to) 0, the model fits perfectly (R^2=1).
        # Otherwise, the model does not explain any variance (R^2=0 can be ambiguous here,
        # but usually implies ss_residual > 0 if ss_total is 0, unless y_fit also all same).
        # A robust way is to say if residual is also zero, R2 is 1, else 0.
        r_squared = 1.0 if ss_residual < 1e-9 else 0.0 # Using a small tolerance for float comparison
    else:
        r_squared = 1.0 - (ss_residual / ss_total)

    # --- Covariance Matrix of Parameters [K, L, C] ---
    # This matrix provides variances of K, L, C on its diagonal,
    # and covariances (e.g., Cov(K,L)) on its off-diagonal.
    if variance_source == 'mse':
        # Homoscedastic assumption: Var(y_i) = sigma_e^2 (constant error variance)
        # Estimate sigma_e^2 using Mean Squared Error (MSE)
        # MSE = SSR / (degrees of freedom)
        # Degrees of freedom = n_points - number_of_parameters (which is 3: K, L, C)
        if n_points <= 3:
            # Not enough degrees of freedom to estimate MSE reliably.
            covariance_matrix_params = np.full((3,3), np.nan) # Fill with NaNs
            print("Warning: Cannot estimate MSE with <=3 data points for 3 parameters. "
                  "Parameter SEs will be NaN.")
        else:
            mse = ss_residual / (n_points - 3)
            # Cov(beta_hat) = MSE * (X^T*X)^-1
            covariance_matrix_params = mse * XTX_inv
    elif variance_source in ['observed_y', 'fitted_y']:
        # Heteroscedasticity-consistent covariance matrix (Eicker-Huber-White type)
        # Assumes errors can have non-constant variance.
        # Cov(beta_hat) = (X^T*X)^-1 * X^T * D * X * ((X^T*X)^-1)^T
        # where D is a diagonal matrix with Var(y_i) on the diagonal.
        # The formula can also be written as M * D * M^T where M = (X^T*X)^-1 * X^T.
        
        if variance_source == 'observed_y':
            # Approximate Var(y_i) = y_i (suitable for Poisson-distributed data)
            # Clip at a small positive value to avoid issues if y_i=0 (Var(0)=0 can be problematic).
            var_y_i = np.maximum(1e-6, spikes)
        else: # variance_source == 'fitted_y'
            # Approximate Var(y_i) = fitted_y_i
            var_y_i = np.maximum(1e-6, y_fit)
            
        D = np.diag(var_y_i) # Diagonal matrix of individual observation variances
        
        # M = (X^T*X)^-1 * X^T
        M = XTX_inv @ X.T
        # Cov(beta_hat) = M * D * M^T
        covariance_matrix_params = M @ D @ M.T
    else:
        raise ValueError("Invalid variance_source. Choose 'observed_y', 'fitted_y', or 'mse'.")

    # Extract variances of K, L, C from the diagonal of the covariance matrix
    var_K_val = covariance_matrix_params[0, 0]
    var_L_val = covariance_matrix_params[1, 1]
    var_C_val = covariance_matrix_params[2, 2]
    
    # Standard errors are the square roots of these variances.
    # Ensure variance is non-negative before taking sqrt; otherwise, SE is NaN.
    K_se = np.sqrt(var_K_val) if var_K_val >= 0 else np.nan
    L_se = np.sqrt(var_L_val) if var_L_val >= 0 else np.nan
    C_se = np.sqrt(var_C_val) if var_C_val >= 0 else np.nan

    # --- Amplitude (A) and its Standard Error ---
    # A = sqrt(K^2 + L^2)
    amplitude = np.sqrt(K**2 + L**2)
    amplitude_se = np.nan # Initialize to NaN
    var_A = np.nan        # Initialize variance of A to NaN

    # For error propagation, we need derivatives of A with respect to K and L.
    # dA/dK = K / sqrt(K^2+L^2) = K / A
    # dA/dL = L / sqrt(K^2+L^2) = L / A
    # To avoid division by zero if A is very small, use a clipped amplitude_denom.
    amplitude_denom = max(amplitude, 1e-9) 

    dAdK = K / amplitude_denom
    dAdL = L / amplitude_denom
    
    # Check if component variances/covariance are NaN (e.g., if MSE calculation failed)
    if np.isnan(var_K_val) or np.isnan(var_L_val) or np.isnan(covariance_matrix_params[0,1]):
        # var_A will remain NaN, and thus amplitude_se will remain NaN
        pass
    else:
        cov_KL = covariance_matrix_params[0, 1] # Covariance between K and L
        # Var(A) approx (dA/dK)^2*Var(K) + (dA/dL)^2*Var(L) + 2*(dA/dK)*(dA/dL)*Cov(K,L)
        var_A = (dAdK**2 * var_K_val) + \
                (dAdL**2 * var_L_val) + \
                (2 * dAdK * dAdL * cov_KL)
        if var_A >= 0:
            amplitude_se = np.sqrt(var_A)
        # If var_A computed is negative (due to numerical issues or model misspecification),
        # amplitude_se remains NaN.

    # Warning if amplitude is very small, as its SE might be unreliable.
    if amplitude < 1e-9 and not np.isnan(amplitude): # Check if amplitude itself isn't already NaN
         print("Warning: Amplitude is close to zero. Standard error for amplitude and phase may be unreliable or NaN.")

    # --- Phase Offset (phi_0) and its Standard Error ---
    # The model can be written as A*sin(omega*x + phi_0) + C.
    # K = A*cos(phi_0), L = A*sin(phi_0) => phi_0 = atan2(L, K)
    # atan2 is used for numerical stability and correct quadrant.
    phase_offset_rad = np.arctan2(L, K)
    phase_offset_rad_se = np.nan # Initialize to NaN
    
    # For error propagation, derivatives of phi_0 w.r.t. K and L:
    # d(phi_0)/dK = -L / (K^2+L^2) = -L / A^2
    # d(phi_0)/dL =  K / (K^2+L^2) =  K / A^2
    # Use amplitude_denom^2 for A^2 to avoid division by zero.
    amplitude_sq_denom = amplitude_denom**2

    dphidK = -L / amplitude_sq_denom
    dphidL = K / amplitude_sq_denom

    if np.isnan(var_K_val) or np.isnan(var_L_val) or np.isnan(covariance_matrix_params[0,1]):
        # phase_offset_rad_se will remain NaN
        pass
    else:
        cov_KL = covariance_matrix_params[0, 1] # Cov(K,L)
        # Var(phi_0) approx (dphi/dK)^2*Var(K) + (dphi/dL)^2*Var(L) + 2*(dphi/dK)*(dphi/dL)*Cov(K,L)
        var_phi0 = (dphidK**2 * var_K_val) + \
                   (dphidL**2 * var_L_val) + \
                   (2 * dphidK * dphidL * cov_KL)
        if var_phi0 >= 0:
            phase_offset_rad_se = np.sqrt(var_phi0)
        # If var_phi0 is negative, phase_offset_rad_se remains NaN.


    # --- Modulation Index (MI) and its Standard Error ---
    # MI = 2*A / (A+C)
    modulation_index = np.nan
    modulation_index_se = np.nan
    cov_AC = np.nan # Covariance between Amplitude (A) and Offset (C)

    # Calculate Cov(A,C) using error propagation:
    # Cov(A,C) approx (dA/dK)*Cov(K,C) + (dA/dL)*Cov(L,C)
    # dAdK and dAdL were computed earlier for amplitude_se.
    if not (np.isnan(K) or np.isnan(L) or \
            np.isnan(covariance_matrix_params[0,2]) or np.isnan(covariance_matrix_params[1,2]) or \
            np.isnan(dAdK) or np.isnan(dAdL) ): # dAdK/L can be NaN if K/L are NaN or amplitude is NaN initially
        cov_KC = covariance_matrix_params[0, 2]  # Cov(K, C)
        cov_LC = covariance_matrix_params[1, 2]  # Cov(L, C)
        cov_AC = dAdK * cov_KC + dAdL * cov_LC
    
    # Proceed if amplitude and C are valid numbers
    if np.isnan(amplitude) or np.isnan(C):
        # MI and MI_se remain NaN if A or C is NaN (e.g. due to upstream NaN K,L)
        pass # This state implies that K,L,C or their SEs might already be NaN
    else:
        # Denominator for MI: A+C
        denom_MI_val = amplitude + C
        
        # Check if A+C is too small (close to zero).
        # Since A (amplitude) >= 0 and C (offset, typically mean firing rate) >= 0,
        # A+C is usually non-negative.
        if denom_MI_val < 1e-9: 
            print(f"Warning: Sum of amplitude ({amplitude:.2e}) and C ({C:.2e}) is close to zero. "
                  "Modulation index and its SE are set to NaN.")
            # modulation_index and modulation_index_se remain np.nan
        else:
            # Calculate Modulation Index
            modulation_index = (2 * amplitude) / denom_MI_val
            
            # For standard error of MI, calculate partial derivatives:
            # d(MI)/dA = 2*C / (A+C)^2
            # d(MI)/dC = -2*A / (A+C)^2
            denom_MI_sq = denom_MI_val**2 # (A+C)^2
            dMI_dA = (2 * C) / denom_MI_sq
            dMI_dC = (-2 * amplitude) / denom_MI_sq

            # Check if required variances (Var(A), Var(C)) and Cov(A,C) are available (not NaN)
            # var_A was computed for amplitude_se
            # var_C_val is covariance_matrix_params[2,2] (variance of C)
            if np.isnan(var_A) or np.isnan(var_C_val) or np.isnan(cov_AC):
                # modulation_index_se remains NaN
                pass
            else:
                # Var(MI) approx (dMI/dA)^2*Var(A) + (dMI/dC)^2*Var(C) + 2*(dMI/dA)*(dMI/dC)*Cov(A,C)
                var_modulation_index = (dMI_dA**2 * var_A) + \
                                   (dMI_dC**2 * var_C_val) + \
                                   (2 * dMI_dA * dMI_dC * cov_AC)
                
                if var_modulation_index >= 0:
                    modulation_index_se = np.sqrt(var_modulation_index)
                else:
                    # This can happen due to numerical instability or if the model is ill-conditioned.
                    if not np.isnan(var_modulation_index): # Only print if it's a negative number, not already NaN
                        print(f"Warning: Calculated variance for modulation index is negative ({var_modulation_index:.2e}). "
                              "Setting SE to NaN. This may indicate issues with model stability or covariance estimates.")
                    # modulation_index_se remains NaN (or its initial np.nan state)
                    
    # --- Return Results ---
    return {
        'K': K, 'L': L, 'C': C,
        'K_se': K_se, 'L_se': L_se, 'C_se': C_se,
        'amplitude': amplitude, 'amplitude_se': amplitude_se,
        'phase_offset_rad': phase_offset_rad, 'phase_offset_rad_se': phase_offset_rad_se,
        'modulation_index': modulation_index, 'modulation_index_se': modulation_index_se,
        'y_fit': y_fit, 'R_squared': r_squared,
        'covariance_matrix_params': covariance_matrix_params,
        'condition_number_XTX': condition_number_XTX
    }


#%% MAIN GRATINGS ANALYSIS FUNCTION
from eval.eval_stack_utils import argmax_subpixel

def gratings_analysis(robs, sf, ori, phases, dt, n_lags=20, n_phase_bins=8, min_spikes=50, inds=None, dfs=None):
    """
    Perform complete gratings analysis for all units.

    Parameters:
    -----------
    robs : numpy.ndarray
        Spike responses array of shape (n_frames, n_units)
    sf : numpy.ndarray
        Spatial frequency values for each frame
    ori : numpy.ndarray
        Orientation values for each frame
    phases : numpy.ndarray
        Phase values for each frame (can be 1D or 3D with spatial dimensions)
    dt : float
        Time step duration
    n_lags : int, optional
        Number of time lags for STA calculation (default: 20)
    n_phase_bins : int, optional
        Number of phase bins for phase tuning analysis (default: 8)
    min_spikes : int, optional
        Minimum number of spikes required for sine fitting (default: 50)
    inds : numpy.ndarray, optional (n_frames,)
        Indices of frames to include in analysis (default: None, uses all frames)
    dfs : numpy.ndarray, optional (n_frames, n_units)
        Data filter array indicating valid frames (default: None, uses all frames)
    
    Returns:
    --------
    dict
        Dictionary containing analysis results for all units:
        - 'n_units': Number of units analyzed
        - 'sf_tuning': SF tuning curves for all units (n_units x n_sfs)
        - 'ori_tuning': Orientation tuning curves for all units (n_units x n_oris)
        - 'peak_sfx': Preferred spatial frequency for each unit
        - 'peak_ori': Preferred orientation for each unit
        - 'peak_lag': Optimal time lag for each unit
        - 'sf_snr': Signal-to-noise ratio for SF tuning for each unit
        - 'ori_snr': Signal-to-noise ratio for orientation tuning for each unit
        - 'phase_response': Mean response per phase bin for all units (n_units x n_phase_bins)
        - 'phase_response_ste': Standard error per phase bin for all units (n_units x n_phase_bins)
        - 'phase_bins': Phase bin centers in degrees
        - 'sine_fit_results': List of sine fitting results for each unit (or None if insufficient spikes)
        - 'n_spikes_total': Total number of spikes used in phase analysis for each unit
        - 'sfs': Unique spatial frequency values
        - 'oris': Unique orientation values
    """

    # ========================================
    # STEP 1: SETUP AND TEMPORAL ANALYSIS
    # ========================================

    print("Setting up analysis parameters...")
    sfs = np.unique(sf)
    oris = np.unique(ori)
    n_units = robs.shape[1]

    print(f"Found {len(sfs)} spatial frequencies and {len(oris)} orientations")
    print(f"Analyzing {n_units} units")

    # Create one-hot encoding for SF x orientation combinations
    print("Creating stimulus encoding matrix...")
    sf_ori_one_hot = np.zeros((len(robs), len(sfs), len(oris)))
    for i in range(len(robs)):
        sf_idx = np.where(sfs == sf[i])[0][0]
        ori_idx = np.where(oris == ori[i])[0][0]
        sf_ori_one_hot[i, sf_idx, ori_idx] = 1

    # calculate the full SF x Orientation STA
    print("Computing full SF x Orientation STAs...")
    gratings_sta = calc_sta(sf_ori_one_hot, robs.astype(np.float64),
                           n_lags, inds=inds, dfs=dfs, reverse_correlate=False, progress=True).numpy() / dt
    

    # Find optimal temporal lag for each unit
    print("Finding optimal temporal lags...")
    temporal_tuning = np.std(gratings_sta, axis=(2,3))  # (n_units, n_lags)

    peak_lags = argmax_subpixel(temporal_tuning, axis=1)[0]  # (n_units,)

    # ========================================
    # STEP 2: SPATIAL FREQUENCY TUNING
    # ========================================

    print("Analyzing spatial frequency tuning...")
    sf_tuning = np.zeros((n_units, len(sfs)))
    
    peak_sf = np.nan*np.ones(n_units, dtype=int)
    sf_snr = np.nan*np.ones(n_units)

    for iU in range(n_units):
        
        if np.std(gratings_sta[iU]) < 1e-9:
            continue

        lag = int(np.round(peak_lags[iU]))
        # auto-detect the preferred orientation and sf
        sf, ori = np.where(np.max(gratings_sta[iU][lag])==gratings_sta[iU][lag])

        # Extract SF tuning at the optimal temporal lag
        sf_tuning[iU] = gratings_sta[iU, lag, :, ori]
        # Only find peak if auto-detecting
        peak_sf[iU] = argmax_subpixel(sf_tuning[iU])[0]
        
        # Calculate signal-to-noise ratio as peak response / mean response
        # sf_snr[iU] = sf_tuning[iU, peak_sf_idx[iU]] / np.mean(sf_tuning[iU])
        sf_snr[iU] = np.std(sf_tuning[iU]) / np.mean(sf_tuning[iU]) # following Parker et al., 2024

    # ========================================
    # STEP 3: ORIENTATION TUNING
    # ========================================

    print("Analyzing orientation tuning...")
    ori_tuning = np.zeros((n_units, len(oris)))

    peak_ori = np.nan*np.zeros(n_units, dtype=int)
    ori_snr = np.nan*np.zeros(n_units)

    for iU in range(n_units):
        if np.std(gratings_sta[iU]) < 1e-9:
            continue

        lag = int(np.round(peak_lags[iU]))
        sf, ori = np.where(np.max(gratings_sta[iU][lag])==gratings_sta[iU][lag])

        # Extract orientation tuning at optimal lag and preferred SF
        ori_tuning[iU] = gratings_sta[iU, lag, sf]
        
        peak_ori[iU] = argmax_subpixel(ori_tuning[iU])[0]
        # Calculate signal-to-noise ratio
        ori_snr[iU] = np.std(ori_tuning[iU]) / np.mean(ori_tuning[iU])
        # ori_snr[iU] = ori_tuning[iU, peak_ori[iU]] / np.mean(ori_tuning[iU])

    # ========================================
    # STEP 4: PHASE RESPONSE EXTRACTION
    # ========================================

    print("Extracting phase and spike data for each unit...")
    phases_list = []
    spikes_list = []

    for iU in tqdm(range(n_units), desc="Processing units for phase analysis"):
        if np.std(gratings_sta[iU]) < 1e-9:
            # append dummy arrays
            phases_list.append(np.array([]))
            spikes_list.append(np.array([]))
            continue

        # Get optimal parameters for this unit
        sf_idx = int(np.round(peak_sf[iU]))
        ori_idx = int(np.round(peak_ori[iU]))
        lag = int(np.round(peak_lags[iU]))

        # Find frames with the preferred SF and orientation combination
        sf_ori_inds = np.where(sf_ori_one_hot[:, sf_idx, ori_idx] > 0)[0]

        # Ensure we have enough frames after the lag for spike extraction
        sf_ori_inds = sf_ori_inds[(sf_ori_inds + lag) < len(robs)]

        # Apply additional filtering if provided
        if inds is not None:
            sf_ori_inds = np.intersect1d(sf_ori_inds, inds)

        if dfs is not None:
            if dfs.ndim == 2:  # Unit-specific data filter
                sf_ori_inds = sf_ori_inds[dfs[sf_ori_inds, iU] > 0]
            elif dfs.ndim == 1:  # Global data filter
                sf_ori_inds = sf_ori_inds[dfs[sf_ori_inds] > 0]
            else:
                raise ValueError(f"Invalid dfs shape: {dfs.shape}")

        # Extract phase values (use center pixel of stimulus)
        stim_phases = phases[sf_ori_inds]
        if stim_phases.ndim == 3:
            _, n_y, n_x = stim_phases.shape
            stim_phases = stim_phases[:, n_y//2, n_x//2]  # Center pixel

        # Get corresponding spike counts at the optimal lag
        stim_spikes = robs[sf_ori_inds + lag, iU]

        # Remove invalid phase values (off-screen or probe frames)
        invalid = (stim_phases <= 0)  # -1 typically indicates off-screen

        stim_phases = stim_phases[~invalid]
        stim_spikes = stim_spikes[~invalid]

        phases_list.append(stim_phases)
        spikes_list.append(stim_spikes)

    # ========================================
    # STEP 5: PHASE BINNING ANALYSIS
    # ========================================

    print("Performing phase binning analysis...")

    # Create phase bins from 0 to 2π
    phase_bin_edges = np.linspace(0, 2*np.pi, n_phase_bins + 1)
    phase_bins = np.rad2deg((phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2)

    # Initialize arrays for phase response analysis
    n_phases = np.zeros((n_units, n_phase_bins))          # Number of frames per phase bin
    n_spikes = np.zeros((n_units, n_phase_bins))          # Total spikes per phase bin
    phase_response = np.zeros((n_units, n_phase_bins))    # Mean response per phase bin
    phase_response_ste = np.zeros((n_units, n_phase_bins)) # Standard error per phase bin

    for iU in range(n_units):
        if np.std(gratings_sta[iU]) < 1e-9:
            continue

        unit_phases = phases_list[iU]
        unit_spikes = spikes_list[iU]

        if len(unit_phases) == 0:
            continue

        # Assign each phase to a bin
        phase_bin_inds = np.digitize(unit_phases, phase_bin_edges) - 1

        # Calculate statistics for each phase bin
        for i in range(n_phase_bins):
            mask = (phase_bin_inds == i)
            n_phases[iU, i] = np.sum(mask)

            if n_phases[iU, i] > 0:
                # Convert to spikes per second
                n_spikes[iU, i] = unit_spikes[mask].sum() / dt
                phase_response_ste[iU, i] = unit_spikes[mask].std() / np.sqrt(n_phases[iU, i]) / dt
                phase_response[iU, i] = n_spikes[iU, i] / n_phases[iU, i]

    # ========================================
    # STEP 6: SINE WAVE FITTING
    # ========================================

    print("Performing sine wave fitting for phase responses...")
    sine_fit_results = []
    n_spikes_total = np.zeros(n_units)

    for iU in range(n_units):
        if np.std(gratings_sta[iU]) < 1e-9:
            sine_fit_results.append(None)
            continue

        unit_phases = phases_list[iU]
        unit_spikes = spikes_list[iU]
        total_spikes = np.sum(unit_spikes)
        n_spikes_total[iU] = total_spikes

        if total_spikes >= min_spikes:
            try:
                # Fit sine wave to phase response data
                result = fit_sine(unit_phases, unit_spikes, omega=1.0, variance_source='observed_y')

                # Optional: visualize fit for debugging
                # plt.figure()
                # plt.plot(phase_bins, phase_response[iU], 'o-', label='Data')
                # plt.plot(np.rad2deg(unit_phases), result['y_fit'], '.', alpha=0.5, label='Fit')
                # plt.xlabel('Phase (degrees)')
                # plt.ylabel('Response')
                # plt.title(f'Unit {iU} - Sine Fit')
                # plt.legend()
                # plt.show()

                sine_fit_results.append(result)
            except Exception as e:
                print(f"Warning: Sine fitting failed for unit {iU}: {e}")
                sine_fit_results.append(None)
        else:
            print(f"Unit {iU}: Insufficient spikes ({total_spikes} < {min_spikes}), skipping sine fit")
            sine_fit_results.append(None)

    # ========================================
    # RETURN RESULTS
    # ========================================

    print("Analysis complete!")
    print(f"Successfully analyzed {n_units} units")
    print(f"Units with sufficient spikes for sine fitting: {sum(1 for r in sine_fit_results if r is not None)}")

    # extract modulation index
    modulation_index = np.array([r['modulation_index'] if r is not None else np.nan for r in sine_fit_results])

    # interpolate into 
    peak_sf_cyc_per_deg = np.interp(peak_sf, np.arange(len(sfs)), sfs)
    peak_ori_deg = np.interp(peak_ori, np.arange(len(oris)), oris)
    peak_lag_ms = np.interp(peak_lags, np.arange(n_lags), np.arange(n_lags) * dt * 1000)
    return {
        'n_units': n_units,
        'gratings_sta': gratings_sta,
        'sf_tuning': sf_tuning,
        'ori_tuning': ori_tuning,
        'peak_sf_idx': np.round(peak_sf).astype(int),
        'peak_ori_idx': np.round(peak_ori).astype(int),
        'peak_lag_idx': np.round(peak_lags).astype(int),
        'peak_ori': peak_ori_deg,
        'peak_sf': peak_sf_cyc_per_deg,
        'peak_lag': peak_lag_ms,
        'sf_snr': sf_snr,
        'ori_snr': ori_snr,
        'phase_response': phase_response,
        'phase_response_ste': phase_response_ste,
        'phase_bins': phase_bins,
        'sine_fit_results': sine_fit_results,
        'n_spikes_total': n_spikes_total,
        'sfs': sfs,
        'oris': oris,
        'dt': dt,
        'modulation_index': modulation_index
    }

def gratings_comparison(robs, rhat, sf, ori, phases, dt, n_lags=20, n_phase_bins=8, min_spikes=50, inds=None, dfs=None):
    """
    Perform gratings analysis on both observed responses and model predictions.
    
    Parameters:
    -----------
    robs : numpy.ndarray
        Observed neural responses
    rhat : numpy.ndarray
        Model predictions
    sf : numpy.ndarray
        Spatial frequency values for each time point
    ori : numpy.ndarray
        Orientation values for each time point
    phases : numpy.ndarray
        Phase values for each time point
    dt : float
        Time bin size in seconds
    n_lags : int, optional
        Number of time lags to consider
    n_phase_bins : int, optional
        Number of bins for phase analysis
    min_spikes : int, optional
        Minimum number of spikes required for sine fitting
    inds : numpy.ndarray, optional
        Indices to use for analysis
    dfs : numpy.ndarray, optional
        Data fidelity scores
        
    Returns:
    --------
    dict
        Dictionary containing analysis results for both observed and predicted responses:
        - 'robs': Results from gratings_analysis on observed responses
        - 'rhat': Results from gratings_analysis on model predictions
    """
    print("Running gratings analysis on observed responses...")
    results_robs = gratings_analysis(
        robs=robs, 
        sf=sf, 
        ori=ori, 
        phases=phases, 
        dt=dt, 
        n_lags=n_lags, 
        n_phase_bins=n_phase_bins, 
        min_spikes=min_spikes, 
        inds=inds, 
        dfs=dfs
    )
    
    # Extract peak indices from observed data (Hmm, should re-introduce?)
    # peak_sf_idx = results_robs['peak_sf_idx']
    # peak_ori_idx = results_robs['peak_ori_idx']
    #         peak_sf_idx=peak_sf_idx,  # Use peak SF indices from observed data
    #     peak_ori_idx=peak_ori_idx  # Use peak orientation indices from observed data
    
    print("\nRunning gratings analysis on model predictions using observed data peak indices...")
    results_rhat = gratings_analysis(
        robs=rhat,  # Using rhat as input
        sf=sf, 
        ori=ori, 
        phases=phases, 
        dt=dt, 
        n_lags=n_lags, 
        n_phase_bins=n_phase_bins, 
        min_spikes=min_spikes, 
        inds=inds, 
        dfs=dfs,
    )
    
    return {
        'robs': results_robs,
        'rhat': results_rhat
    }

# # Run complete analysis on all units
# results = gratings_analysis(
#     robs=gratings_dset['robs'].numpy(),
#     sf=gratings_dset['sf'].numpy(),
#     ori=gratings_dset['ori'].numpy(),
#     phases=gratings_dset['stim_phase'],
#     dt=dt,
#     dfs=gratings_dset['dfs'].numpy().squeeze()
# )


#%% VISUALIZATION FUNCTIONS

def plot_gratings_results(results, iU):
    dt = results['dt']

    fig, axs = plt.subplots(4, 1, figsize=(6, 12), height_ratios=[1, 4, 4, 4])

    sta = results['gratings_sta'][iU][None, :, None, :, :]
    peak_lag = results['peak_lags'][iU] * results['dt'] * 1000

    peak_sf = results['sfs'][results['peak_sf_idx'][iU]]
    peak_ori = results['oris'][results['peak_ori_idx'][iU]]

    plot_stas(sta - np.mean(sta), ax=axs[0])
    axs[0].set_title(f'Unit {iU} - Full STA\n Peak Lag: {results['peak_lags'][iU]} ({peak_lag:.1f} ms)')
    axs[0].set_ylabel('Lag')
    sf = results['sf_tuning'][iU]
    axs[1].plot(results['sfs'], sf)
    axs[1].axvline(results['sfs'][results['peak_sf_idx'][iU]], color='r', linestyle='--')
    axs[1].set_ylabel('Spikes / second')
    axs[1].set_xlabel('Spatial Frequency (cycles/degree)')
    axs[1].set_title(f'Spatial Frequency Tuning @ {peak_lag:.1f} ms\nPeak SF: {peak_sf:.0f} cyc/deg')

    ori = results['ori_tuning'][iU]
    axs[2].plot(results['oris'], ori)
    axs[2].axvline(results['oris'][results['peak_ori_idx'][iU]], color='r', linestyle='--')
    axs[2].set_ylabel('Spikes / second')
    axs[2].set_xlabel('Orientation (degrees)')
    axs[2].set_title(f'Orientation Tuning @ {peak_lag:.1f} ms, {peak_sf:.0f} cyc/deg\nPeak Ori: {peak_ori:.1f} deg')

    pr_ste = results['phase_response_ste'][iU]
    axs[3].errorbar(results['phase_bins'], results['phase_response'][iU], yerr=pr_ste, fmt='o-', ecolor='C0', capsize=5, zorder=0)
    axs[3].set_ylabel('Spikes / second')
    axs[3].set_xlabel('Phase (degrees)')
    axs[3].set_title(f'Phase Tuning @ {peak_lag:.1f} ms, {peak_sf:.0f} cyc/deg, {peak_ori:.1f} deg')

    res = results['sine_fit_results'][iU]
    if res is not None:
        amp = res['amplitude']
        amp_se = res['amplitude_se']
        phase_offset = res['phase_offset_rad']
        phase_offset_se = res['phase_offset_rad_se']
        C = res['C']
        mi = res['modulation_index']
        mi_se = res['modulation_index_se']
        if not (np.isnan(mi) or np.isnan(mi_se) or np.isnan(amp) or np.isnan(amp_se) or np.isnan(phase_offset) or np.isnan(phase_offset_se)):
            smoothed_phases = np.linspace(0, 2*np.pi, 100)
            smoothed_fit = amp * np.sin(smoothed_phases + phase_offset) + C
            smoothed_fit_max = (amp+amp_se) * np.sin(smoothed_phases + phase_offset) + C
            smoothed_fit_min = (amp-amp_se) * np.sin(smoothed_phases + phase_offset) + C

            axs[3].plot(np.rad2deg(smoothed_phases), smoothed_fit/dt, color='red')
            axs[3].fill_between(np.rad2deg(smoothed_phases), smoothed_fit_min/dt, smoothed_fit_max/dt, color='red', alpha=0.2)
            axs[3].set_title(axs[3].get_title() + f'\nModulation Index {mi:.2f} +/- {mi_se:.2f}')
    ylim = axs[3].get_ylim()
    axs[3].set_ylim([0, ylim[1]])

    plt.tight_layout()
    return fig, axs

def plot_gratings_comparison(comparison_results, iU):
    """
    Plot comparison of gratings analysis results between observed responses and model predictions.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from gratings_comparison() containing 'robs' and 'rhat' keys
    iU : int
        Unit index to plot
        
    Returns:
    --------
    tuple
        (fig, axs) - Figure and axes objects
    """
    robs_results = comparison_results['robs']
    rhat_results = comparison_results['rhat']
    dt = robs_results['dt']
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 14), height_ratios=[1, 4, 4, 4])
    
    # Get peak lag and tuning information from observed data
    peak_lag_obs = robs_results['peak_lags'][iU] * robs_results['dt'] * 1000
    peak_sf_obs = robs_results['sfs'][robs_results['peak_sf_idx'][iU]]
    peak_ori_obs = robs_results['oris'][robs_results['peak_ori_idx'][iU]]
    
    # Get peak lag and tuning information from model predictions
    peak_lag_hat = rhat_results['peak_lags'][iU] * rhat_results['dt'] * 1000
    peak_sf_hat = rhat_results['sfs'][rhat_results['peak_sf_idx'][iU]]
    peak_ori_hat = rhat_results['oris'][rhat_results['peak_ori_idx'][iU]]
    
    # Plot 1: STA comparison
    sta_obs = robs_results['gratings_sta'][iU][None, :, None, :, :]
    sta_hat = rhat_results['gratings_sta'][iU][None, :, None, :, :]
    
    # Combine STAs side by side with a separator
    H = sta_obs.shape[1]
    W = sta_obs.shape[3]
    combined_sta = np.concatenate(
        [sta_obs - np.mean(sta_obs), sta_obs - np.mean(sta_obs)], axis=0
    )
    
    plot_stas(combined_sta, ax=axs[0], row_labels=['Data', 'Model'])
    axs[0].set_title(f'Unit {iU} - STA Comparison\nData: {peak_lag_obs:.1f} ms | Model: {peak_lag_hat:.1f} ms')
    axs[0].set_xlabel('Lag')
    
    # Plot 2: SF tuning comparison
    sf_obs = robs_results['sf_tuning'][iU]
    sf_hat = rhat_results['sf_tuning'][iU]
    
    axs[1].plot(robs_results['sfs'], sf_obs, 'b-', label='Data')
    axs[1].plot(rhat_results['sfs'], sf_hat, 'r-', label='Model')
    axs[1].axvline(peak_sf_obs, color='b', linestyle='--', alpha=0.5)
    axs[1].axvline(peak_sf_hat, color='r', linestyle='--', alpha=0.5)
    axs[1].set_ylabel('Spikes / second')
    axs[1].set_xlabel('Spatial Frequency (cycles/degree)')
    axs[1].set_title(f'Spatial Frequency Tuning\nData: {peak_sf_obs:.0f} cyc/deg | Model: {peak_sf_hat:.0f} cyc/deg')
    axs[1].legend()
    
    # Plot 3: Orientation tuning comparison
    ori_obs = robs_results['ori_tuning'][iU]
    ori_hat = rhat_results['ori_tuning'][iU]
    
    axs[2].plot(robs_results['oris'], ori_obs, 'b-', label='Data')
    axs[2].plot(rhat_results['oris'], ori_hat, 'r-', label='Model')
    axs[2].axvline(peak_ori_obs, color='b', linestyle='--', alpha=0.5)
    axs[2].axvline(peak_ori_hat, color='r', linestyle='--', alpha=0.5)
    axs[2].set_ylabel('Spikes / second')
    axs[2].set_xlabel('Orientation (degrees)')
    axs[2].set_title(f'Orientation Tuning\nData: {peak_ori_obs:.1f} deg | Model: {peak_ori_hat:.1f} deg')
    axs[2].legend()
    
    # Plot 4: Phase response comparison
    phase_obs = robs_results['phase_response'][iU]
    phase_hat = rhat_results['phase_response'][iU]
    phase_obs_ste = robs_results['phase_response_ste'][iU]
    phase_hat_ste = rhat_results['phase_response_ste'][iU]
    
    axs[3].errorbar(robs_results['phase_bins'], phase_obs, yerr=phase_obs_ste, 
                   fmt='o-', ecolor='b', color='b', capsize=5, label='Data', alpha=0.7)
    axs[3].errorbar(rhat_results['phase_bins'], phase_hat, yerr=phase_hat_ste, 
                   fmt='o-', ecolor='r', color='r', capsize=5, label='Model', alpha=0.7)
    axs[3].set_ylabel('Spikes / second')
    axs[3].set_xlabel('Phase (degrees)')
    axs[3].set_title('Phase Response')
    axs[3].legend()
    
    # Add sine wave fits if available
    res_obs = robs_results['sine_fit_results'][iU]
    res_hat = rhat_results['sine_fit_results'][iU]
    
    mi_obs = "N/A"
    mi_hat = "N/A"
    
    if res_obs is not None:
        amp_obs = res_obs['amplitude']
        amp_se_obs = res_obs['amplitude_se']
        phase_offset_obs = res_obs['phase_offset_rad']
        C_obs = res_obs['C']
        mi_obs = res_obs['modulation_index']
        mi_se_obs = res_obs['modulation_index_se']
        
        if not (np.isnan(mi_obs) or np.isnan(amp_obs) or np.isnan(phase_offset_obs)):
            smoothed_phases = np.linspace(0, 2*np.pi, 100)
            smoothed_fit = amp_obs * np.sin(smoothed_phases + phase_offset_obs) + C_obs
            smoothed_fit_max = (amp_obs+amp_se_obs) * np.sin(smoothed_phases + phase_offset_obs) + C_obs
            smoothed_fit_min = (amp_obs-amp_se_obs) * np.sin(smoothed_phases + phase_offset_obs) + C_obs
            
            axs[3].plot(np.rad2deg(smoothed_phases), smoothed_fit/dt, color='blue', linestyle='-', alpha=0.8)
            axs[3].fill_between(np.rad2deg(smoothed_phases), smoothed_fit_min/dt, smoothed_fit_max/dt, 
                               color='blue', alpha=0.2)
            mi_obs = f"{mi_obs:.2f} ± {mi_se_obs:.2f}"
    
    if res_hat is not None:
        amp_hat = res_hat['amplitude']
        amp_se_hat = res_hat['amplitude_se']
        phase_offset_hat = res_hat['phase_offset_rad']
        C_hat = res_hat['C']
        mi_hat = res_hat['modulation_index']
        mi_se_hat = res_hat['modulation_index_se']
        
        if not (np.isnan(mi_hat) or np.isnan(amp_hat) or np.isnan(phase_offset_hat)):
            smoothed_phases = np.linspace(0, 2*np.pi, 100)
            smoothed_fit = amp_hat * np.sin(smoothed_phases + phase_offset_hat) + C_hat
            smoothed_fit_max = (amp_hat+amp_se_hat) * np.sin(smoothed_phases + phase_offset_hat) + C_hat
            smoothed_fit_min = (amp_hat-amp_se_hat) * np.sin(smoothed_phases + phase_offset_hat) + C_hat
            
            axs[3].plot(np.rad2deg(smoothed_phases), smoothed_fit/dt, color='red', linestyle='-', alpha=0.8)
            axs[3].fill_between(np.rad2deg(smoothed_phases), smoothed_fit_min/dt, smoothed_fit_max/dt, 
                               color='red', alpha=0.2)
            mi_hat = f"{mi_hat:.2f} ± {mi_se_hat:.2f}"
    
    axs[3].set_title(f'Phase Response\nModulation Index - Data: {mi_obs} | Model: {mi_hat}')
    
    # Set y-axis to start at 0
    ylim = axs[3].get_ylim()
    axs[3].set_ylim([0, ylim[1]])
    
    plt.tight_layout()
    return fig, axs

# pdf = PdfPages('gratings_results.pdf')
# for iU in range(results['n_units']):
#     fig, axs = plot_gratings_results(results, iU)
#     pdf.savefig(fig)
#     plt.close(fig)
# pdf.close()
    

# %%

#%% MULTIDATASET GRATINGS ANALYSIS

def run_gratings_analysis(all_results, checkpoint_dir, save_dir, recalc=False, batch_size=64, device='cuda', test_mode=False):
    """
    Run comprehensive gratings analysis for all models and datasets efficiently.

    This function performs the complete gratings analysis pipeline including:
    - Gratings comparison analysis (SF tuning, orientation tuning, phase responses)
    - Sine wave fitting and modulation index extraction
    - All analyses from gratings_cross_model_analysis_120.py except PDF plotting

    Parameters
    ----------
    all_results : dict
        Existing results dictionary from BPS/CCNORM analysis (must include 'bps' results)
    checkpoint_dir : str
        Directory containing model checkpoints
    save_dir : str
        Directory to save gratings analysis caches
    recalc : bool, optional
        Whether to recalculate cached results (default: False)
    batch_size : int, optional
        Batch size for evaluation (default: 64)
    device : str, optional
        Device to run evaluation on (default: 'cuda')

    Returns
    -------
    dict
        Modified all_results dictionary with comprehensive gratings analysis added:
        - all_results[model_type]['gratings']['comparison_results'][dataset_idx]: gratings_comparison results
        - all_results[model_type]['gratings']['modulation_indices']: extracted modulation indices
        - all_results[model_type]['gratings']['datasets']: dataset names
    """

    # Import evaluation utilities
    from eval_stack_multidataset import load_model, load_single_dataset, evaluate_dataset

    print("Running gratings analysis...")

    # Extract model types from existing results
    model_types = list(all_results.keys())
    print(f"Found {len(model_types)} models: {model_types}")

    # Verify that BPS results exist (required for gratings analysis)
    for model_type in model_types:
        if 'bps' not in all_results[model_type] or 'gratings' not in all_results[model_type]['bps']:
            raise ValueError(f"Model {model_type} missing BPS gratings results. Run BPS analysis first.")

    # Load all models once (on CPU to save GPU memory)
    models = {}
    for model_type in model_types:
        print(f"  Loading {model_type}...")
        model, model_info = load_model(
            model_type=model_type,
            checkpoint_dir=checkpoint_dir,
            device='cpu'  # Load on CPU first
        )
        models[model_type] = {
            'model': model,
            'model_info': model_info,
            'experiment': model_info['experiment']
        }

    # Get number of datasets from first model
    first_model = list(models.values())[0]['model']
    num_datasets = len(first_model.names)

    # Initialize gratings results structure for all models
    # Structure matches other analyses: [analysis_name][dataset_idx] = padded_array
    gratings_analyses = [
        'gratings_sta',
        'sf_tuning', 'ori_tuning', 'peak_sf_idx', 'peak_ori_idx', 'peak_lags',
        'sf_snr', 'ori_snr', 'phase_response', 'phase_response_ste',
        'n_spikes_total', 'modulation_index'
    ]

    for model_type in model_types:
        if 'gratings' not in all_results[model_type]:
            all_results[model_type]['gratings'] = {}

        # Initialize lists for each analysis (both robs and rhat)
        for analysis in gratings_analyses:
            all_results[model_type]['gratings'][f'{analysis}_robs'] = []
            all_results[model_type]['gratings'][f'{analysis}_rhat'] = []

        # Also store metadata
        all_results[model_type]['gratings']['phase_bins'] = []
        all_results[model_type]['gratings']['sfs'] = []
        all_results[model_type]['gratings']['oris'] = []

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loop over datasets to find those with gratings stimuli
    for dataset_idx in tqdm(range(num_datasets), desc="Processing datasets"):

        # Check if cache exists for all models BEFORE loading the data
        all_caches_exist = True
        for model_type in model_types:
            # Check cache first
            model_ = models[model_type]['model']
            dataset_config = model_.model.dataset_configs[dataset_idx].copy()
            session = dataset_config['session']
            cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gratings_analysis_cache.pt'
            if recalc and cache_file.exists():
                cache_file.unlink()  # Delete the cache file
                print(f"  Deleted cache file: {cache_file}")
                print(f" Cache file exists status: {cache_file.exists()}")

            if not cache_file.exists():
                all_caches_exist = False
                print(f"  Cache file does not exist: {cache_file}")
                if not recalc:
                    break

        
        if test_mode and dataset_idx > 1:
            break
        dataset_name = first_model.names[dataset_idx]
        print(f"\nProcessing dataset {dataset_idx}: {dataset_name}")

        
        if not all_caches_exist or recalc:
            print("  Loading dataset...")
            # Load dataset to check for gratings stimuli
            train_data, val_data, dataset_config = load_single_dataset(first_model, dataset_idx)
            
            # Check if this dataset contains gratings stimuli
            has_gratings = False
            try:
                gratings_ind = int(np.where([d.metadata['name'] == 'gratings' for d in train_data.dsets])[0])
                has_gratings = True
            except:
                print(f'  No gratings dataset for {dataset_name} - will create NaN placeholders')
        
        
            # Get dataset CIDs for proper padding
            dataset_cids = dataset_config.get('cids', [])
            n_total_units = len(dataset_cids)
            print(f"Dataset has {n_total_units} total units")

            if has_gratings:

                # Combine all indices (train + validation) for maximum data (we don't tend to train on gratings so this should be okay)
                gratings_inds = torch.concatenate([
                    train_data.get_dataset_inds('gratings'),
                    val_data.get_dataset_inds('gratings')
                ], dim=0)

                dataset = train_data.shallow_copy()
                dataset.inds = gratings_inds

                inds = dataset.inds[dataset.inds[:,0] == gratings_ind][:,1]

                # Extract gratings dataset and stimulus properties for ALL data
                gratings_dset = train_data.dsets[gratings_ind]
                sf = gratings_dset['sf'][inds]  # Spatial frequency
                ori = gratings_dset['ori'][inds]  # Orientation
                phases = gratings_dset['stim_phase'][inds]
                phases = phases[:,phases.shape[1]//2, phases.shape[2]//2]  # Center pixel phase
                dt = 1/dataset_config['sampling']['target_rate']  # Time step
                n_lags = dataset_config['keys_lags']['stim'][-1]
               
        # Loop over models (inner loop - data already loaded or failed)
        for model_type in model_types:

            cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gratings_analysis_cache.pt'
            if cache_file.exists():
                analysis_result = torch.load(cache_file, weights_only=False)
            
                # check that all requested keys are in the cache, if not, rerun
                for data_type in ['robs', 'rhat']:
                    result_data = analysis_result[data_type]

                for analysis in gratings_analyses:
                    if analysis not in result_data:
                        
                        print(f"  Missing {analysis} in {data_type} for {model_type} - You need to rerun with recalc=True or delete the cache file")
                        print(f"  Cache file: {cache_file}")
                        break
            
            else:
                if has_gratings:
                    print(f"  Running gratings analysis for {model_type}...")

                    # Check cache first
                    session = dataset_config['session']
                    cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gratings_analysis_cache.pt'

                    if not recalc and cache_file.exists():
                        print(f"    Loading gratings analysis cache from {cache_file}")
                        analysis_result = torch.load(cache_file, weights_only=False)
                    else:
                        # Get model and move to device for evaluation
                        model = models[model_type]['model'].to(device)

                        eval_result = evaluate_dataset(
                            model, train_data, gratings_inds, dataset_idx, batch_size, 'Gratings'
                        )

                        # Extract robs and rhat
                        robs = eval_result['robs'].numpy()
                        rhat = eval_result['rhat'].numpy()
                        dfs = eval_result['dfs'].numpy()

                        # Perform comprehensive gratings analysis for both robs and rhat
                        robs_result = gratings_analysis(
                            robs=robs,
                            sf=sf,
                            ori=ori,
                            phases=phases,
                            dt=dt,
                            n_lags=n_lags,
                            dfs=dfs,
                            min_spikes=30  # Minimum spike count threshold for analysis
                        )

                        print(f'    🔍 Analyzing model predictions...')
                        rhat_result = gratings_analysis(
                            robs=rhat,  # Pass rhat as robs parameter
                            sf=sf,
                            ori=ori,
                            phases=phases,
                            dt=dt,
                            n_lags=n_lags,
                            dfs=dfs,
                            min_spikes=30
                        )

                        # Combine results
                        analysis_result = {
                            'robs': robs_result,
                            'rhat': rhat_result
                        }

                        # Move model back to CPU to save GPU memory
                        models[model_type]['model'] = model.to('cpu')

                        # Save to cache
                        torch.save(analysis_result, cache_file)
                        print(f"    Gratings analysis cache saved to {cache_file}")

                else:
                    n_units = len(train_data.dsets[0].metadata['cids'])
                    dummy_result = {'n_units': n_units,
                                    'gratings_sta': np.full((n_units, 24, 5, 8), np.nan),
                                    'sf_tuning': np.full((n_units, 5), np.nan),
                                    'ori_tuning': np.full((n_units, 8), np.nan),
                                    'peak_sf_idx': np.full(n_units, np.nan),
                                        'peak_ori_idx': np.full(n_units, np.nan),
                                        'peak_lags': np.full(n_units, np.nan),
                                        'sf_snr': np.full(n_units, np.nan),
                                        'ori_snr': np.full(n_units, np.nan),
                                        'phase_response': np.full((n_units, 8), np.nan),
                                        'phase_response_ste': np.full((n_units, 8), np.nan),
                                        'phase_bins': np.full(8, np.nan),
                                        'sine_fit_results': [None] * n_units,
                                        'n_spikes_total': np.full(n_units, np.nan),
                                        'sfs': np.full(5, np.nan),
                                        'oris': np.full(8, np.nan),
                                        'dt': np.nan,
                                        'modulation_index': np.full(n_units, np.nan)
                                    }
                    analysis_result = {
                        'robs': dummy_result,
                        'rhat': dummy_result
                    } 
                    # save to cache to avoid having to load the whole dataset again
                    torch.save(analysis_result, cache_file)

            # Always process results (either real results or NaN placeholders)
            assert analysis_result is not None, f"Analysis result is None for {model_type} on dataset {dataset_idx}"
            

            # Store metadata (same for all datasets)
            if all_results[model_type]['gratings']['phase_bins'] is None:
                all_results[model_type]['gratings']['phase_bins'].append(analysis_result['robs']['phase_bins'])
                all_results[model_type]['gratings']['sfs'].append(analysis_result['robs']['sfs'])
                all_results[model_type]['gratings']['oris'].append(analysis_result['robs']['oris'])

            # Process both robs and rhat results
            for data_type in ['robs', 'rhat']:
                result_data = analysis_result[data_type]

                for analysis in gratings_analyses:
                    all_results[model_type]['gratings'][f'{analysis}_{data_type}'].append(result_data[analysis])

    return all_results

 