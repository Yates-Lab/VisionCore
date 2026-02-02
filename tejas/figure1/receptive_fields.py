#%%
from tejas.metrics.gaborium import plot_unit_sta_ste
from tejas.metrics.gaborium import get_unit_sta_ste_dict
from tejas.metrics.gaborium import get_rf_contour_metrics
from tejas.metrics.gaborium import get_rf_gaussian_fit_metrics
from models.config_loader import load_dataset_configs
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False

def get_roi_center_deg(roi_src, ppd):
    """
    Calculate the center of an ROI in degrees.

    Args:
        roi_src (np.ndarray): ROI boundaries in gaze-centered pixel coordinates.
            Shape: (2, 2) where roi_src[0] is [i_min, i_max] and roi_src[1] is [j_min, j_max]
        ppd (float): Pixels per degree

    Returns:
        tuple: (x_deg, y_deg) where x=azimuth (j), y=elevation (i)
    """
    # Calculate center in pixels
    i_center_pix = (roi_src[0, 0] + roi_src[0, 1]) / 2
    j_center_pix = (roi_src[1, 0] + roi_src[1, 1]) / 2

    # Convert to degrees
    y_deg = -i_center_pix / ppd  # elevation (i coordinate)
    x_deg = j_center_pix / ppd  # azimuth (j coordinate)

    return x_deg, y_deg
def get_rf_center_deg(roi_x_deg_center, roi_y_deg_center, rf_gaussian_fit_metrics, ste_dim, ppd):
    """
    Calculate RF center positions in visual degrees for each cell.
    
    Args:
        roi_x_deg_center: X coordinate (azimuth) of ROI center in degrees
        roi_y_deg_center: Y coordinate (elevation) of ROI center in degrees  
        rf_gaussian_fit_metrics: Dict mapping cell_id to Gaussian fit metrics (contains 'popt')
        ste_dim: STE image dimension in pixels (assumes square)
        ppd: Pixels per degree
    
    Returns:
        Dict mapping cell_id to {'rf_x_deg': float, 'rf_y_deg': float}
    """
    import numpy as np
    
    ste_center_pix = ste_dim / 2  # Center of STE in pixel coords (from top-left)
    
    rf_centers = {}
    for cell_id, metrics in rf_gaussian_fit_metrics.items():
        if not metrics.get('valid', False) or metrics['popt'] is None:
            rf_centers[cell_id] = {'rf_x_deg': np.nan, 'rf_y_deg': np.nan}
            continue
        
        # x0 = column index (horizontal), y0 = row index (vertical) in STE pixel coords
        _, x0, y0, _, _, _, _ = metrics['popt']
        
        # Offset from STE center to RF center in pixels
        dx_pix = x0 - ste_center_pix  # positive = rightward
        dy_pix = y0 - ste_center_pix  # positive = downward in image
        
        # Convert to degrees (y inverted: image row↓ = elevation↓)
        rf_x_deg = roi_x_deg_center + dx_pix / ppd
        rf_y_deg = roi_y_deg_center - dy_pix / ppd
        
        rf_centers[cell_id] = {'rf_x_deg': rf_x_deg, 'rf_y_deg': rf_y_deg}
    
    return rf_centers



from DataYatesV1 import  get_complete_sessions
from tqdm import tqdm
import pickle
session_dict = {}
cache_file = 'session_dict_tmp.pkl'
import os
if os.path.exists(cache_file):
    session_dict = pickle.load(open(cache_file, 'rb'))
else:
    for session in tqdm(get_complete_sessions()):
        print(session.name)
        try:
            subject = session.name.split('_')[0]
            date = session.name.split('_')[1]
            dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium.yaml'
            from tejas.rsvp_util import get_dataset_from_config
            _, _ = get_dataset_from_config(subject, date, '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml', dataset_type='fixrsvp')
            dataset, dataset_config = get_dataset_from_config(subject, date, dataset_configs_path, dataset_type='gaborium')
            roi_x_deg_center, roi_y_deg_center = get_roi_center_deg(dataset.dsets[0].metadata['roi_src'], dataset.dsets[0].metadata['ppd'])
            unit_sta_ste_dict = get_unit_sta_ste_dict(date, subject)
            rf_contour_metrics = get_rf_contour_metrics(date, subject)
            rf_gaussian_fit_metrics = get_rf_gaussian_fit_metrics(date, subject)
            ppd = dataset.dsets[0].metadata['ppd']
            ste_dim = unit_sta_ste_dict['stas'].shape[-1]
            rf_centers_deg = get_rf_center_deg(roi_x_deg_center, roi_y_deg_center, rf_gaussian_fit_metrics, ste_dim, ppd)

            session_dict[session.name] = {
                'roi_x_deg_center': roi_x_deg_center,
                'roi_y_deg_center': roi_y_deg_center,
                'unit_sta_ste_dict': unit_sta_ste_dict,
                'rf_contour_metrics': rf_contour_metrics,
                'rf_gaussian_fit_metrics': rf_gaussian_fit_metrics,
                'rf_centers_deg': rf_centers_deg,
                'ppd': ppd,
                'ste_dim': ste_dim,
                'dataset_config': dataset_config,
                
            }
        except Exception as e:
            print(f"Error processing session {session.name}: {e}")
            continue

 
    with open(cache_file, 'wb') as f:
        pickle.dump(session_dict, f)
    

#%% Plot RF ellipses for all cells in dataset_config['cids']
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

def get_ellipse_params_from_cov(cov_matrix, nstd=2):
    """
    Get ellipse width, height, and angle from a 2x2 covariance matrix.
    
    Args:
        cov_matrix: 2x2 covariance matrix [[var_x, cov_xy], [cov_xy, var_y]]
        nstd: Number of standard deviations for ellipse radius
    
    Returns:
        width, height, angle_deg: Ellipse parameters
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Width and height are 2*nstd*sqrt(eigenvalue)
    width = 2 * nstd * np.sqrt(eigenvalues[0])
    height = 2 * nstd * np.sqrt(eigenvalues[1])
    
    # Angle from the first eigenvector (major axis)
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    return width, height, angle_deg

# Plot setup (same as eyepos_rsvp.py)
extent = 80  # arcmin (default, used for fixation circle)
x_lim = (-extent, extent)  # (min, max) in arcmin
y_lim = (-extent, extent)  # (min, max) in arcmin
fig, ax = plt.subplots(figsize=(5, 5))

# Color map for different sessions
session_cmap = mpl.cm.get_cmap("tab20")  # Good for categorical colors
session_names = list(session_dict.keys())
n_sessions = len(session_names)

# Plot ellipses settings
nstd = 1.5  # Number of standard deviations for ellipse
fill_alpha = 0.0  # Alpha for ellipse fill
edge_alpha = 0.3  # Alpha for ellipse edge
edge_color = 'black'  # Edge color
width_max = 50
height_max = 50
widths_and_heights_arcmin = []
total_cells = 0
linewidth = 0.4
percent_to_plot = 1
show_legend = True  # Toggle legend on/off

# Create legend handles
from matplotlib.patches import Patch
legend_handles = []
for session_idx, session_name in enumerate(session_names):
    color = session_cmap(session_idx / max(n_sessions - 1, 1))
    legend_handles.append(Patch(facecolor=(*color[:3], 0.5), edgecolor=color, label=session_name))

# Loop over all sessions
for session_idx, session_name in enumerate(session_names):
    session_data = session_dict[session_name]
    
    # Get session-specific color
    fill_color = session_cmap(session_idx / max(n_sessions - 1, 1))
    
    # Extract session data
    rf_centers_deg = session_data['rf_centers_deg']
    rf_gaussian_fit_metrics = session_data['rf_gaussian_fit_metrics']
    ppd = session_data['ppd']
    dataset_config = session_data['dataset_config']
    cids = dataset_config['cids']
    
    for cell_id in cids:
        # Skip if invalid
        rf_center = rf_centers_deg[cell_id]
        if np.isnan(rf_center['rf_x_deg']) or np.isnan(rf_center['rf_y_deg']):
            continue
        
        # Get metrics for this cell
        metrics = rf_gaussian_fit_metrics.get(cell_id)
        if metrics is None or not metrics.get('valid', False) or metrics['popt'] is None:
            continue
        
        # Extract covariance parameters from popt (in pixels²)
        _, _, _, sigma_xx, sigma_xy, sigma_yy, _ = metrics['popt']
        
        # Convert covariance from pixels² to degrees²
        # Note: y-axis is inverted (row down = elevation down), so sigma_xy sign flips
        cov_deg = np.array([
            [sigma_xx, -sigma_xy],
            [-sigma_xy, sigma_yy]
        ]) / (ppd ** 2)
        
        # Convert to arcmin² (1 deg = 60 arcmin, so multiply by 60²)
        cov_arcmin = cov_deg * (60 ** 2)
        
        # Get ellipse parameters
        width, height, angle = get_ellipse_params_from_cov(cov_arcmin, nstd=nstd)
        
        # Center in arcmin
        x_arcmin = rf_center['rf_x_deg'] * 60
        y_arcmin = rf_center['rf_y_deg'] * 60
        widths_and_heights_arcmin.append(width)
        widths_and_heights_arcmin.append(height)
        if width > width_max or height > height_max:
            continue
        if np.random.rand() > percent_to_plot:
            continue
        # Create and add ellipse
        ellipse = Ellipse(
            xy=(x_arcmin, y_arcmin),
            width=width,
            height=height,
            angle=angle,
            facecolor=(*fill_color[:3], fill_alpha),  # RGBA with fill alpha
            # edgecolor=(*mpl.colors.to_rgb(edge_color), edge_alpha),  # RGBA with edge alph
            edgecolor=(*fill_color[:3], edge_alpha),
            linewidth=linewidth,
        )
        ax.add_patch(ellipse)
        total_cells += 1

# Crosshair
ax.axhline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)
ax.axvline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)

# # Fixation circle (1 degree = 60 arcmin, fixed radius)
# fixation_radius_arcmin = 60
# circle = mpl.patches.Circle(
#     (0, 0), fixation_radius_arcmin, fill=False, edgecolor="red",
#     linestyle=(0, (14, 10)), linewidth=2
# )
# ax.add_patch(circle)

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Horizontal (arcmin)")
ax.set_ylabel("Vertical (arcmin)")
ax.set_title(f"RF Ellipses (n={n_sessions} sessions, {total_cells} cells)")

if show_legend:
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=6, frameon=True, title="Session")

fig.savefig("rf_ellipses.pdf", dpi=1200, bbox_inches="tight")
plt.tight_layout()
plt.show()

#%% Plot AVERAGE ellipse per session
fig_avg, ax_avg = plt.subplots(figsize=(5, 5))

# Settings for average ellipses
avg_fill_alpha = 0.0
avg_edge_alpha = 1
avg_linewidth = 2.0
show_legend_avg = True  # Toggle legend on/off for average plot

for session_idx, session_name in enumerate(session_names):
    session_data = session_dict[session_name]
    
    # Get session-specific color
    fill_color = session_cmap(session_idx / max(n_sessions - 1, 1))
    
    # Extract session data
    rf_centers_deg = session_data['rf_centers_deg']
    rf_gaussian_fit_metrics = session_data['rf_gaussian_fit_metrics']
    ppd = session_data['ppd']
    dataset_config = session_data['dataset_config']
    cids = dataset_config['cids']
    
    # Collect valid RF data for this session
    x_centers = []
    y_centers = []
    cov_matrices = []
    
    for cell_id in cids:
        if cell_id not in rf_centers_deg:
            continue
        rf_center = rf_centers_deg[cell_id]
        if np.isnan(rf_center['rf_x_deg']) or np.isnan(rf_center['rf_y_deg']):
            continue
        
        metrics = rf_gaussian_fit_metrics.get(cell_id)
        if metrics is None or not metrics.get('valid', False) or metrics['popt'] is None:
            continue
        
        _, _, _, sigma_xx, sigma_xy, sigma_yy, _ = metrics['popt']
        
        # Convert covariance to arcmin²
        cov_arcmin = np.array([
            [sigma_xx, -sigma_xy],
            [-sigma_xy, sigma_yy]
        ]) / (ppd ** 2) * (60 ** 2)
        
        # Get ellipse size and skip if too large
        width, height, _ = get_ellipse_params_from_cov(cov_arcmin, nstd=nstd)
        if width > width_max or height > height_max:
            continue
        
        x_centers.append(rf_center['rf_x_deg'] * 60)
        y_centers.append(rf_center['rf_y_deg'] * 60)
        cov_matrices.append(cov_arcmin)
    
    if len(x_centers) == 0:
        continue
    
    # Compute averages
    avg_x = np.mean(x_centers)
    avg_y = np.mean(y_centers)
    avg_cov = np.mean(cov_matrices, axis=0)
    
    # Get ellipse parameters from average covariance
    avg_width, avg_height, avg_angle = get_ellipse_params_from_cov(avg_cov, nstd=nstd)
    
    # Plot average ellipse
    ellipse = Ellipse(
        xy=(avg_x, avg_y),
        width=avg_width,
        height=avg_height,
        angle=avg_angle,
        facecolor=(*fill_color[:3], avg_fill_alpha),
        edgecolor=(*fill_color[:3], avg_edge_alpha),
        linewidth=avg_linewidth,
    )
    ax_avg.add_patch(ellipse)
    
    # # Optionally plot the center point
    # ax_avg.plot(avg_x, avg_y, 'o', color=fill_color, markersize=4)

# Crosshair
ax_avg.axhline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)
ax_avg.axvline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)

ax_avg.set_xlim(x_lim)
ax_avg.set_ylim(y_lim)
ax_avg.set_aspect("equal", adjustable="box")
ax_avg.set_xlabel("Horizontal (arcmin)")
ax_avg.set_ylabel("Vertical (arcmin)")
ax_avg.set_title(f"Average RF per Session (n={n_sessions} sessions)")

if show_legend_avg:
    ax_avg.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=6, frameon=True, title="Session")

plt.tight_layout()
plt.show()

#%%
subject = 'Allen'
date = '2022-04-13'
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium.yaml'
from tejas.rsvp_util import get_dataset_from_config
dataset, dataset_config = get_dataset_from_config(subject, date, dataset_configs_path, dataset_type='gaborium')
roi_x_deg_center, roi_y_deg_center = get_roi_center_deg(dataset.dsets[0].metadata['roi_src'], dataset.dsets[0].metadata['ppd'])

unit_sta_ste_dict = get_unit_sta_ste_dict(date, subject)
rf_contour_metrics = get_rf_contour_metrics(date, subject)
rf_gaussian_fit_metrics = get_rf_gaussian_fit_metrics(date, subject)

unit_idx = 31
assert unit_idx in dataset_config['cids']
sampling_rate = 240
ax_sta = None
show_ln_energy_fit = False
#%%
plot_unit_sta_ste(subject, date, unit_idx, unit_sta_ste_dict, contour_metrics=rf_contour_metrics, gaussian_fit_metrics=rf_gaussian_fit_metrics, sampling_rate=240, ax=ax_sta, show_ln_energy_fit=show_ln_energy_fit)
# Example usage: get RF centers in degrees
ppd = dataset.dsets[0].metadata['ppd']
ste_dim = unit_sta_ste_dict['stas'].shape[-1]
rf_centers_deg = get_rf_center_deg(roi_x_deg_center, roi_y_deg_center, rf_gaussian_fit_metrics, ste_dim, ppd)

# %%
