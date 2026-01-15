#%%
from DataYatesV1 import enable_autoreload
enable_autoreload()
from DataYatesV1 import get_gaborium_sta_ste, plot_stas
from DataYatesV1 import get_session
import numpy as np
import matplotlib.pyplot as plt
from DataYatesV1.utils.rf import fit_guassian_to_cell_STE, get_contour_metrics, get_gaussian_fit_metrics, plot_cov_ellipse

from pathlib import Path
import torch

#%%
def get_cached_ppd(session):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/gaborium_analysis/{session.name}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    ppd_path = cache_dir / f'{session.name}_ppd.npy'
    if ppd_path.exists():
        ppd = np.load(ppd_path)
        return ppd
    else:
        np.save(ppd_path, session.exp['S']['pixPerDeg'])
        return session.exp['S']['pixPerDeg']
def get_ln_energy_kernels(subject, date, unit_idx):
    ln_dir = Path(f'/mnt/sata/YatesMarmoV1/standard_model_fits/ray_ln_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil/{subject}_{date}/cell_{unit_idx}_best')
    energy_dir = Path(f'/mnt/sata/YatesMarmoV1/standard_model_fits/ray_energy_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil/{subject}_{date}/cell_{unit_idx}_best')
    
    # Try to load LN model
    ln_kernel = None
    if (ln_dir / 'best_model.pt').exists():
        try:
            ln_state_dict = torch.load(ln_dir / 'best_model.pt', map_location='cpu')
            ln_kernel = ln_state_dict['kernel_'].numpy()  # Shape: (n_lags, H, W)
        except Exception as e:
            print(f"Failed to load LN model for unit {unit_idx}: {e}")
    
    # Try to load Energy model
    energy_kernels = None
    if (energy_dir / 'best_model.pt').exists():
        try:
            energy_state_dict = torch.load(energy_dir / 'best_model.pt', map_location='cpu')
            energy_kernels = energy_state_dict['kernels_'].numpy()  # Shape: (2, n_lags, H, W)
        except Exception as e:
            print(f"Failed to load Energy model for unit {unit_idx}: {e}")
    
    return ln_kernel, energy_kernels
    
def get_unit_sta_ste_dict(date, subject):
    sess = get_session(subject, date)
    stas, stes = get_gaborium_sta_ste(sess, 20)

    NC = stas.shape[0]

    all_ln_kernels = []
    all_energy_kernels = []
    
    for unit_idx in range(NC):
        ln_kernel, energy_kernels = get_ln_energy_kernels(subject, date, unit_idx)
        all_ln_kernels.append(ln_kernel)
        all_energy_kernels.append(energy_kernels)
    return {
        'stas': stas,
        'stes': stes,
        'all_ln_kernels': all_ln_kernels,
        'all_energy_kernels': all_energy_kernels
    }



def plot_unit_sta_ste(subject, date, 
                    unit_idx, 
                    unit_sta_ste_dict,
                    contour_metrics = None, 
                    gaussian_fit_metrics = None, 
                    sampling_rate = None, 
                    ax = None, 
                    show_ln_energy_fit = False):
    # subject = 'Allen'
    # date = '2022-04-13'
    assert len(date) == 10, 'Date must be in YYYY-MM-DD format'
    assert subject in ['Allen', 'Logan']
    sess = get_session(subject, date)
    # stas, stes = get_gaborium_sta_ste(sess, 20)
    # unit_sta_ste_dict = get_unit_sta_ste_dict(date, subject)
    stas = unit_sta_ste_dict['stas']
    stes = unit_sta_ste_dict['stes']

    # unit_idx = 90
    normalized_stes = (stes -
                    np.median(stes, axis=(2,3), keepdims=True))
    unit_sta_ste = np.concatenate([stas[[unit_idx]], normalized_stes[[unit_idx]]], axis=0)
    row_labels = ['STA', 'STE']
    
    # Track crop offsets for adjusting contour/ellipse coordinates
    crop_offset_y = 0
    crop_offset_x = 0
    original_height = unit_sta_ste.shape[2]
    original_width = unit_sta_ste.shape[3]
    
    # Load and add model fits if requested
    if show_ln_energy_fit:
        
        
        # ln_kernel, energy_kernels = get_ln_energy_kernels(subject, date, unit_idx)
        ln_kernel = unit_sta_ste_dict['all_ln_kernels'][unit_idx]
        energy_kernels = unit_sta_ste_dict['all_energy_kernels'][unit_idx]
        # Determine model dimensions (lags and spatial)
        model_n_lags = None
        model_height = None
        model_width = None
        
        if ln_kernel is not None:
            model_n_lags = ln_kernel.shape[0]
            model_height = ln_kernel.shape[1]
            model_width = ln_kernel.shape[2]
        elif energy_kernels is not None:
            model_n_lags = energy_kernels.shape[1]
            model_height = energy_kernels.shape[2]
            model_width = energy_kernels.shape[3]
        
        # Trim/crop STA/STE to match model dimensions if models were loaded
        if model_n_lags is not None:
            # Trim temporal dimension
            if model_n_lags < unit_sta_ste.shape[1]:
                unit_sta_ste = unit_sta_ste[:, :model_n_lags, :, :]
            
            # Center-crop spatial dimensions
            if model_height is not None and model_width is not None:
                current_height = unit_sta_ste.shape[2]
                current_width = unit_sta_ste.shape[3]
                
                if model_height < current_height or model_width < current_width:
                    # Calculate center crop indices
                    y_start = (current_height - model_height) // 2
                    y_end = y_start + model_height
                    x_start = (current_width - model_width) // 2
                    x_end = x_start + model_width
                    
                    # Save crop offsets for coordinate adjustment
                    crop_offset_y = y_start
                    crop_offset_x = x_start
                    
                    unit_sta_ste = unit_sta_ste[:, :, y_start:y_end, x_start:x_end]
        
        # Add kernels to visualization
        if ln_kernel is not None:
            unit_sta_ste = np.concatenate([unit_sta_ste, ln_kernel[np.newaxis, :]], axis=0)
            row_labels.append('LN Model')
        
        if energy_kernels is not None:
            unit_sta_ste = np.concatenate([unit_sta_ste, energy_kernels], axis=0)
            row_labels.extend(['Energy K1', 'Energy K2']) 
    col_labels = [f'{i * 1000 / sampling_rate:.1f}ms' for i in range(unit_sta_ste.shape[1])] if sampling_rate is not None else None
    fig, ax = plot_stas(unit_sta_ste[:, :, None, :, :], row_labels=row_labels, col_labels=col_labels, ax=ax)
    ax.set_title(f'{subject} {date} Unit {unit_idx}')

    # Add scale bar showing size in degrees
    n_rows, n_lags, n_y, n_x = unit_sta_ste.shape
    aspect = n_x / n_y
    ppd = get_cached_ppd(sess)
    width_deg = n_x / ppd
    
    # Position scale bar on right side of last column, first row (STA row)
    # In plot_stas coordinates: x goes from 0 to n_lags*aspect, y goes from -n_rows to 0
    # First row (STA) is at y = -1 to 0, last column is at x = (n_lags-1)*aspect to n_lags*aspect
    x_right = n_lags * aspect
    x_left = x_right - aspect  # Width of one column
    y_pos = -0.1  # Bottom of first row
    
    # Draw horizontal scale bar line (representing the width of one image)
    ax.plot([x_left, x_right], [y_pos, y_pos], 'k-', linewidth=2)
    
    # Add label showing degrees (centered on the scale bar)
    ax.text((x_left + x_right) / 2, y_pos - 0.05, f'{width_deg:.1f}°', 
            ha='center', va='top', fontsize=10, color='k')

    ste_row = 1
    
    # Plot contour on STE at peak lag if available
    if contour_metrics and unit_idx in contour_metrics:
        contour = contour_metrics[unit_idx].get('contour')
        if contour is not None and not np.any(np.isnan(contour)):
            peak_lag = contour_metrics[unit_idx]['ste_peak_lag']
            # Adjust contour coordinates for crop offset (contour is in original image coords)
            contour_adjusted = contour.copy()
            contour_adjusted[:, 0] -= crop_offset_y  # Adjust row
            contour_adjusted[:, 1] -= crop_offset_x  # Adjust col
            
            # Transform contour from image coords [row, col] to plot coords
            # Note: imshow with origin='upper' places row 0 at top (y=-1) and row n_y-1 at bottom (y=-2)
            # So we need to invert the row coordinate
            contour_x_plot = peak_lag * aspect + (contour_adjusted[:, 1] / n_x) * aspect
            contour_y_plot = -ste_row - (contour_adjusted[:, 0] / n_y)  # Inverted: row 0 → y=-1, row n_y-1 → y=-2
            ax.plot(contour_x_plot, contour_y_plot, 'r-', linewidth=1.5)
    
    # Plot Gaussian ellipse on STE at peak lag if available
    if gaussian_fit_metrics and unit_idx in gaussian_fit_metrics:
        popt = gaussian_fit_metrics[unit_idx].get('popt')
        if popt is not None:
            peak_lag = gaussian_fit_metrics[unit_idx]['ste_peak_lag']
            amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset = popt
            
            # Adjust center coordinates for crop offset (x0, y0 are in original image coords)
            x0_adjusted = x0 - crop_offset_x
            y0_adjusted = y0 - crop_offset_y
            
            # Transform center position from image coords to plot coords
            # Note: y0 is in image coords where 0 is at top, so we need to invert it
            center_x_plot = peak_lag * aspect + (x0_adjusted / n_x) * aspect
            center_y_plot = -ste_row - (y0_adjusted / n_y)  # Inverted: y0=0 → y=-1, y0=n_y-1 → y=-2
            
            # Transform covariance matrix: scale factors are (aspect/n_x) for x and (1/n_y) for y
            # When inverting y-axis, the cross-term sigma_xy must be negated
            x_scale = aspect / n_x
            y_scale = 1.0 / n_y
            cov_matrix = np.array([[sigma_xx * x_scale**2, -sigma_xy * x_scale * y_scale],
                                   [-sigma_xy * x_scale * y_scale, sigma_yy * y_scale**2]])
            
            plot_cov_ellipse(cov_matrix, (center_x_plot, center_y_plot), nstd=1.5, ax=ax,
                           edgecolor='green', facecolor='none', linewidth=1.5)

    return fig, ax 

# plot_unit_sta_ste('Allen', '2022-04-13', 33)
#%%
def sta_at_peak_lag(sta, peak_lag):
    return np.stack([sta[cc, peak_lag[cc],...] for cc in range(len(peak_lag))], 0)

subject = 'Allen'
date = '2022-04-13'

def get_rf_contour_metrics(date, subject):
    sess = get_session(subject, date)
    ppd = get_cached_ppd(sess)
    stas, stes = get_gaborium_sta_ste(sess, 20)
    # print('stas shape: ', stas.shape)
    peak_lags = np.array([stes[cc].std((1,2)).argmax() for cc in range(stes.shape[0])])
    sta_peak_lags = sta_at_peak_lag(stas, peak_lags)
    ste_peak_lags = sta_at_peak_lag(stes, peak_lags)
    contour_metrics = get_contour_metrics(sta_peak_lags, ste_peak_lags, show=False, save_fig_path=None, sort=False, return_snr_list=False)

    #amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset = popt
    for key in contour_metrics.keys():
        contour_metrics[key]['valid'] = not np.isnan(contour_metrics[key]['snr_value']) and contour_metrics[key]['area_contour'] is not None and not np.isnan(contour_metrics[key]['area_contour']) 
        contour_metrics[key]['ste_peak_lag'] = peak_lags[key]
        if contour_metrics[key]['valid']:
            contour_metrics[key]['area_contour_deg2'] = contour_metrics[key]['area_contour'] / (ppd**2)
            contour_metrics[key]['sqrt_area_contour_deg'] = np.sqrt(contour_metrics[key]['area_contour']) / ppd
        else:
            contour_metrics[key]['area_contour_deg2'] = np.nan
            contour_metrics[key]['sqrt_area_contour_deg'] = np.nan

    return contour_metrics

def get_rf_gaussian_fit_metrics(date, subject, cache = False):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/gaborium_analysis/{date}_{subject}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    gaussian_fit_metrics_path = cache_dir / f'{date}_{subject}_gaussian_fit_metrics.npy'
    if cache and gaussian_fit_metrics_path.exists():
        return np.load(gaussian_fit_metrics_path, allow_pickle=True).item()
    
    sess = get_session(subject, date)
    # ppd = sess.exp['S']['pixPerDeg']
    ppd = get_cached_ppd(sess)
    stas, stes = get_gaborium_sta_ste(sess, 20)
    peak_lags = np.array([stes[cc].std((1,2)).argmax() for cc in range(stes.shape[0])])
    sta_peak_lags = sta_at_peak_lag(stas, peak_lags)
    ste_peak_lags = sta_at_peak_lag(stes, peak_lags)
    # fit_guassian_to_cell_STE(unit_ste, normalize_STE=True, show_plots=True, nstd=1.5)
    gaussian_fit_metrics = get_gaussian_fit_metrics(sta_peak_lags, ste_peak_lags, show=False, save_fig_path=None, sort=False)


    #amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset = popt
    for key in gaussian_fit_metrics.keys():
        gaussian_fit_metrics[key]['valid'] = not np.isnan(gaussian_fit_metrics[key]['MSE']) and gaussian_fit_metrics[key]['area_gaussian'] is not None and not np.isnan(gaussian_fit_metrics[key]['area_gaussian']) 
        gaussian_fit_metrics[key]['ste_peak_lag'] = peak_lags[key]
        if gaussian_fit_metrics[key]['valid']:
            gaussian_fit_metrics[key]['area_gaussian_deg2'] = gaussian_fit_metrics[key]['area_gaussian'] / (ppd**2)
            gaussian_fit_metrics[key]['length_gaussian_deg'] = gaussian_fit_metrics[key]['major_axis_length'] / ppd
            gaussian_fit_metrics[key]['sqrt_area_gaussian_deg'] = np.sqrt(gaussian_fit_metrics[key]['area_gaussian']) / ppd
        else:
            gaussian_fit_metrics[key]['area_gaussian_deg2'] = np.nan
            gaussian_fit_metrics[key]['length_gaussian_deg'] = np.nan
            gaussian_fit_metrics[key]['sqrt_area_gaussian_deg'] = np.nan

    np.save(gaussian_fit_metrics_path, gaussian_fit_metrics)
    return gaussian_fit_metrics

def show_stas_for_session_by_metric_order(date, subject, metric_name, ascending=False, cache=True, ste =  False):
    """Show STAs for all units in a session, ordered by a metric value."""
    from tejas.metrics.plot_variables import _load_metrics_for_session
    
    sess = get_session(subject, date)
    stas, stes = get_gaborium_sta_ste(sess, 20)
    if ste:
        normalized_stes = (stes -
                    np.median(stes, axis=(2,3), keepdims=True))
        stas = normalized_stes
    
    # Load metric
    metric_dict = _load_metrics_for_session(metric_name, metric_name, date, subject, cache, False, {})
    metric_data = metric_dict[metric_name]
    
    # Get valid units with metric values
    unit_metric_pairs = [(uid, metric_data[uid][metric_name]) 
                         for uid in metric_data.keys() 
                         if metric_data[uid].get('valid', False) and not np.isnan(metric_data[uid][metric_name])]
    
    # Sort by metric value
    unit_metric_pairs.sort(key=lambda x: x[1], reverse=not ascending)
    sorted_unit_ids = [uid for uid, _ in unit_metric_pairs]
    
    # Stack STAs: shape (n_units, n_lags, 1, n_y, n_x)
    all_stas = np.stack([stas[uid] for uid in sorted_unit_ids], axis=0)
    all_stas = all_stas[:, :, None, :, :]  # Add channel dimension
    
    # Create row labels: "Unit {id}: {metric_name}={value:.3f}"
    row_labels = [f'Unit {uid}: {metric_name}={val:.3f}' for uid, val in unit_metric_pairs]
    
    fig, ax = plot_stas(all_stas, row_labels=row_labels, col_labels=None)
    order_str = 'ascending' if ascending else 'descending'
    ax.set_title(f'{subject} {date} - STAs ordered by {metric_name} ({order_str})')
    return fig, ax

# c = get_rf_contour_metrics(date, subject)
#%%


