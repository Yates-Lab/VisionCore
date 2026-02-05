
#%%
import os
from pathlib import Path
from tkinter.constants import TRUE
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False

#%%

    
#%%


#%%
# =============================================================================
# Helper functions (adapted from figure_align_population.py)
# =============================================================================

def microsaccade_exists(eyepos_segment, threshold=0.3):
    """
    Check if a microsaccade exists in an eye position segment.
    
    Parameters
    ----------
    eyepos_segment : np.ndarray
        Eye position data of shape [time, 2] (X, Y in degrees)
    threshold : float
        Distance threshold in degrees for detecting microsaccades
        
    Returns
    -------
    bool
        True if any point is more than threshold away from median position
    """
    if np.all(np.isnan(eyepos_segment)):
        return True  # Treat all-NaN as invalid
    
    median_eyepos = np.nanmedian(eyepos_segment, axis=0)
    distances = np.hypot(eyepos_segment[:, 0] - median_eyepos[0], 
                         eyepos_segment[:, 1] - median_eyepos[1])
    return np.any(distances > threshold)


def plot_spikes_as_lines(ax, spike_x, spike_y, height=1.0, color="k", linewidth=0.5, alpha=1.0):
    """
    Plot spikes as vertical line segments.
    
    Parameters
    ----------
    ax : matplotlib axis
    spike_x : array of x (time) positions
    spike_y : array of y (row) positions  
    height : height of each line segment
    color : line color
    linewidth : line width
    alpha : transparency
    """
    spike_x = np.asarray(spike_x)
    spike_y = np.asarray(spike_y)
    
    if spike_x.size == 0:
        return None
    
    x_lines = np.vstack([spike_x, spike_x, np.full(len(spike_x), np.nan)])
    y_lines = np.vstack([spike_y, spike_y + height, np.full(len(spike_y), np.nan)])
    return ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'), 
                   color=color, linewidth=linewidth, alpha=alpha, rasterized=True)[0]


def position_to_color(x, y, xlim, ylim, corner_colors=None):
    """
    Map a 2D position to a color using bilinear interpolation between corner colors.
    
    Parameters
    ----------
    x, y : float
        Position coordinates
    xlim : tuple
        (x_min, x_max) bounds
    ylim : tuple
        (y_min, y_max) bounds
    corner_colors : dict, optional
        Colors for each corner. Keys: 'bl' (bottom-left), 'br' (bottom-right),
        'tl' (top-left), 'tr' (top-right). Values: RGB tuples (0-1).
        If None, uses default: blue, cyan, red, yellow.
        
    Returns
    -------
    tuple
        RGB color tuple (0-1)
    """
    if corner_colors is None:
        # Default colors: saturated primaries
        corner_colors = {
            'bl': (0.0, 0.2, 0.8),   # Blue (bottom-left)
            'br': (0.1, 0.7, 0.2),   # Green (bottom-right)
            'tl': (0.9, 0.1, 0.1),   # Red (top-left)
            'tr': (1.0, 0.6, 0.0),   # Orange (top-right)
        }
    
    # Normalize x and y to [0, 1]
    x_norm = np.clip((x - xlim[0]) / (xlim[1] - xlim[0]), 0, 1)
    y_norm = np.clip((y - ylim[0]) / (ylim[1] - ylim[0]), 0, 1)
    
    # Bilinear interpolation
    # Bottom edge: interpolate between bl and br
    bottom = tuple(
        corner_colors['bl'][i] * (1 - x_norm) + corner_colors['br'][i] * x_norm
        for i in range(3)
    )
    # Top edge: interpolate between tl and tr
    top = tuple(
        corner_colors['tl'][i] * (1 - x_norm) + corner_colors['tr'][i] * x_norm
        for i in range(3)
    )
    # Final: interpolate between bottom and top
    color = tuple(
        bottom[i] * (1 - y_norm) + top[i] * y_norm
        for i in range(3)
    )
    
    return color


def get_position_colors(positions, xlim, ylim, corner_colors=None):
    """
    Get colors for an array of positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape [n, 2] with (x, y) positions
    xlim, ylim : tuple
        Axis limits
    corner_colors : dict, optional
        Corner colors (see position_to_color)
        
    Returns
    -------
    list
        List of RGB color tuples
    """
    colors = []
    for pos in positions:
        colors.append(position_to_color(pos[0], pos[1], xlim, ylim, corner_colors))
    return colors


def orientation_to_color(orientation, cmap_name='hsv', unique_orientations=None):
    """
    Map an orientation (in degrees) to a color using a colormap.
    
    Parameters
    ----------
    orientation : float
        Orientation in degrees (0-180, where 0° and 180° are the same)
    cmap_name : str
        Name of matplotlib colormap to use. 
        - Cyclic options: 'hsv', 'twilight'
        - Categorical (better for discrete): 'tab10', 'Set1', 'Set2', 'Paired'
    unique_orientations : array-like, optional
        If provided, maps orientation to discrete color index based on position
        in this sorted list. Better for categorical colormaps.
        
    Returns
    -------
    tuple
        RGBA color tuple (0-1)
    """
    cmap = plt.get_cmap(cmap_name)
    
    if unique_orientations is not None:
        # Discrete mapping: find index of this orientation in sorted unique list
        sorted_unique = np.sort(unique_orientations)
        idx = np.searchsorted(sorted_unique, orientation)
        # Map to colormap - spread evenly across colormap range
        n_unique = len(sorted_unique)
        normalized = idx / max(n_unique - 1, 1)
    else:
        # Continuous mapping for cyclic colormaps
        normalized = (orientation % 180) / 180
    
    return cmap(normalized)


def get_orientation_colors(orientations, cmap_name='hsv', unique_orientations=None, 
                           highlight_orientations=None):
    """
    Get colors for an array of orientations.
    
    Parameters
    ----------
    orientations : np.ndarray
        Array of orientations in degrees
    cmap_name : str
        Name of colormap. For discrete orientations, 'tab10', 'Set1', 'Paired' work well.
    unique_orientations : array-like, optional
        If provided, uses discrete color mapping (better for categorical colormaps).
    highlight_orientations : list, optional
        If provided, only these orientations get colors; all others are black.
        
    Returns
    -------
    list
        List of RGBA color tuples
    """
    if unique_orientations is None:
        unique_orientations = np.unique(orientations)
    
    colors = []
    for o in orientations:
        if highlight_orientations is not None:
            # Check if this orientation should be highlighted
            # Use np.isclose for floating point comparison
            is_highlighted = any(np.isclose(o, h, atol=0.1) for h in highlight_orientations)
            if is_highlighted:
                # Use only highlighted orientations for color mapping
                color = orientation_to_color(o, cmap_name, unique_orientations=highlight_orientations)
            else:
                color = (0.2, 0.2, 0.2, 0.3)  # Dark gray, semi-transparent for non-highlighted
        else:
            color = orientation_to_color(o, cmap_name, unique_orientations)
        colors.append(color)
    
    return colors


def add_orientation_legend(ax, unique_orientations, cmap_name='hsv', loc='upper left', 
                           line_length=0.03, fontsize=9):
    """
    Add a legend showing orientation-to-color mapping with angled line segments.
    
    Parameters
    ----------
    ax : matplotlib axis
    unique_orientations : array-like
        Sorted unique orientations in degrees
    cmap_name : str
        Colormap name
    loc : str
        Legend location
    line_length : float
        Length of orientation lines in axis coordinates
    fontsize : int
        Font size for labels
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    
    sorted_oris = sorted(unique_orientations)
    legend_elements = []
    for ori in sorted_oris:
        color = orientation_to_color(ori, cmap_name, unique_orientations=sorted_oris)
        # Create a line element - the line angle is just for visual, 
        # we'll use a colored patch and text
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=f'{ori:.1f}°')
        )
    
    # Create a custom legend with oriented lines
    legend = ax.legend(handles=legend_elements, loc=loc, fontsize=fontsize, 
                       title='Preferred Orientation', title_fontsize=fontsize+1,
                       framealpha=0.9)
    
    return legend


def draw_orientation_legend_with_lines(ax, unique_orientations, cmap_name='hsv',
                                        position=(0.02, 0.98), spacing=0.045,
                                        line_length=0.025, fontsize=9):
    """
    Draw a custom legend with actual oriented line segments.
    
    Parameters
    ----------
    ax : matplotlib axis
    unique_orientations : array-like
        Sorted unique orientations in degrees
    cmap_name : str
        Colormap name
    position : tuple
        (x, y) position in axes coordinates for top-left of legend
    spacing : float
        Vertical spacing between entries in axes coordinates
    line_length : float
        Half-length of the orientation line in axes coordinates
    fontsize : int
        Font size for labels
    """
    x_start, y_start = position
    sorted_oris = sorted(unique_orientations)
    
    # Draw background box
    box_height = len(sorted_oris) * spacing + 0.03
    box_width = 0.12
    rect = plt.Rectangle((x_start - 0.01, y_start - box_height), box_width, box_height + 0.01,
                          transform=ax.transAxes, facecolor='white', edgecolor='gray',
                          alpha=0.9, zorder=100)
    ax.add_patch(rect)
    
    # Title
    ax.text(x_start + box_width/2 - 0.01, y_start - 0.01, 'Orientation', 
            transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
            ha='center', va='top', zorder=101)
    
    for i, ori in enumerate(sorted_oris):
        y_pos = y_start - 0.035 - i * spacing
        color = orientation_to_color(ori, cmap_name, unique_orientations=sorted_oris)
        
        # Draw oriented line
        angle_rad = np.radians(ori)
        dx = line_length * np.cos(angle_rad)
        dy = line_length * np.sin(angle_rad)
        
        line_x = x_start + 0.02
        ax.plot([line_x - dx, line_x + dx], [y_pos - dy, y_pos + dy],
                color=color, linewidth=3, transform=ax.transAxes, zorder=101,
                solid_capstyle='round')
        
        # Draw label
        ax.text(line_x + line_length + 0.015, y_pos, f'{ori:.1f}°',
                transform=ax.transAxes, fontsize=fontsize-1, va='center', zorder=101)


def boxes_overlap(pos1, pos2, raster_size, spacing=0):
    """Check if two raster boxes overlap (including spacing)."""
    rw, rh = raster_size
    # Add spacing to the collision box
    collision_w = rw + spacing
    collision_h = rh + spacing
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return dx < collision_w and dy < collision_h


def find_nearest_non_overlapping(target_pos, placed_positions, raster_size, bounds, 
                                  spacing=0, search_radius=2.0, search_step=0.02):
    """
    Find the nearest position to target that doesn't overlap with any placed positions.
    
    Uses a spiral search pattern outward from the target position.
    """
    rw, rh = raster_size
    x_min, x_max, y_min, y_max = bounds
    margin_x, margin_y = (rw + spacing) / 2, (rh + spacing) / 2
    
    # Check if target position works (no overlap)
    target_valid = (x_min + margin_x <= target_pos[0] <= x_max - margin_x and
                    y_min + margin_y <= target_pos[1] <= y_max - margin_y)
    
    if target_valid:
        overlaps = False
        for placed in placed_positions:
            if boxes_overlap(target_pos, placed, raster_size, spacing):
                overlaps = True
                break
        if not overlaps:
            return target_pos.copy()
    
    # Spiral search for nearest non-overlapping position
    best_pos = None
    best_dist = float('inf')
    
    # Generate candidate positions in expanding rings
    for r in np.arange(search_step, search_radius, search_step):
        # Check positions around a circle of radius r
        n_points = max(8, int(2 * np.pi * r / search_step))
        for angle in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
            candidate = np.array([
                target_pos[0] + r * np.cos(angle),
                target_pos[1] + r * np.sin(angle)
            ])
            
            # Check bounds
            if not (x_min + margin_x <= candidate[0] <= x_max - margin_x and
                    y_min + margin_y <= candidate[1] <= y_max - margin_y):
                continue
            
            # Check overlap with all placed positions
            overlaps = False
            for placed in placed_positions:
                if boxes_overlap(candidate, placed, raster_size, spacing):
                    overlaps = True
                    break
            
            if not overlaps:
                dist = np.sqrt((candidate[0] - target_pos[0])**2 + 
                              (candidate[1] - target_pos[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = candidate.copy()
                    # Found one at this radius, can break angle loop
                    break
        
        # If we found a position at this radius, we're done
        if best_pos is not None:
            return best_pos
    
    # Fallback: return target position clamped to bounds (will overlap)
    print(f"Warning: Could not find non-overlapping position for {target_pos}")
    return np.array([
        np.clip(target_pos[0], x_min + margin_x, x_max - margin_x),
        np.clip(target_pos[1], y_min + margin_y, y_max - margin_y)
    ])


def greedy_non_overlapping_layout(positions, raster_size, bounds=(-1, 1, -1, 1),
                                   spacing=0, search_radius=2.0, search_step=0.01):
    """
    Place rasters one at a time, each at the closest non-overlapping position
    to its true location.
    
    Parameters
    ----------
    positions : np.ndarray
        Target positions of shape [n_trials, 2] (x, y)
    raster_size : tuple
        (width, height) of each raster in the same units as positions
    bounds : tuple
        (x_min, x_max, y_min, y_max) bounds for positions
    spacing : float
        Minimum spacing between rasters (in same units as positions)
    search_radius : float
        Maximum distance to search from target position
    search_step : float
        Step size for spiral search
        
    Returns
    -------
    np.ndarray
        Adjusted positions of shape [n_trials, 2] with no overlaps
    """
    n = len(positions)
    if n == 0:
        return positions.copy()
    
    # Sort by distance from center (place central ones first, they have more options)
    center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2])
    distances_from_center = np.sqrt(np.sum((positions - center)**2, axis=1))
    order = np.argsort(distances_from_center)
    
    final_positions = np.zeros_like(positions)
    placed_positions = []
    
    for idx in tqdm(order, desc="Placing rasters"):
        target = positions[idx]
        new_pos = find_nearest_non_overlapping(
            target, placed_positions, raster_size, bounds, 
            spacing, search_radius, search_step
        )
        final_positions[idx] = new_pos
        placed_positions.append(new_pos)
    
    return final_positions


def two_phase_layout(positions, raster_size, bounds=(-1, 1, -1, 1), spacing=0,
                     n_force_iterations=500, repulsion_strength=0.005, 
                     attraction_strength=0.02, damping=0.85,
                     overlap_removal_iterations=100):
    """
    Two-phase layout algorithm that preserves topology while ensuring non-overlap.
    
    Phase 1: Force-directed simulation to spread points while preserving neighborhoods
    Phase 2: Minimal overlap removal - nudge overlapping pairs apart by minimum amount
    
    Parameters
    ----------
    positions : np.ndarray
        Target positions of shape [n_trials, 2] (x, y)
    raster_size : tuple
        (width, height) of each raster in the same units as positions
    bounds : tuple
        (x_min, x_max, y_min, y_max) bounds for positions
    spacing : float
        Minimum spacing between rasters
    n_force_iterations : int
        Number of force-directed iterations
    repulsion_strength : float
        Strength of repulsion between nearby rasters
    attraction_strength : float
        Strength of attraction back to original position
    damping : float
        Velocity damping factor (0-1)
    overlap_removal_iterations : int
        Maximum iterations for overlap removal phase
        
    Returns
    -------
    np.ndarray
        Adjusted positions of shape [n_trials, 2] with no overlaps
    """
    n = len(positions)
    if n == 0:
        return positions.copy()
    
    original_positions = positions.copy()
    current_positions = positions.copy()
    velocities = np.zeros_like(positions)
    
    rw, rh = raster_size
    # Collision size includes spacing
    collision_w = rw + spacing
    collision_h = rh + spacing
    
    x_min, x_max, y_min, y_max = bounds
    margin_x = collision_w / 2
    margin_y = collision_h / 2
    
    # =========================================================================
    # Phase 1: Force-directed simulation
    # =========================================================================
    print(f"Phase 1: Force-directed simulation ({n_force_iterations} iterations)...")
    
    for iteration in tqdm(range(n_force_iterations), desc="Force-directed"):
        forces = np.zeros_like(positions)
        
        # Compute pairwise distances and apply forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_positions[j, 0] - current_positions[i, 0]
                dy = current_positions[j, 1] - current_positions[i, 1]
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                
                # Check if rasters overlap or are close
                overlap_x = collision_w - abs(dx)
                overlap_y = collision_h - abs(dy)
                
                if overlap_x > 0 and overlap_y > 0:
                    # Overlapping - strong repulsion
                    overlap_amount = np.sqrt(max(0, overlap_x)**2 + max(0, overlap_y)**2)
                    force_magnitude = repulsion_strength * (1 + overlap_amount * 10)
                elif dist < collision_w * 2:
                    # Close but not overlapping - weaker repulsion
                    force_magnitude = repulsion_strength * (collision_w * 2 - dist) / (collision_w * 2)
                else:
                    force_magnitude = 0
                
                if force_magnitude > 0:
                    # Direction: push apart
                    fx = force_magnitude * dx / dist
                    fy = force_magnitude * dy / dist
                    
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        # Attraction forces back to original position
        for i in range(n):
            dx = original_positions[i, 0] - current_positions[i, 0]
            dy = original_positions[i, 1] - current_positions[i, 1]
            dist_from_original = np.sqrt(dx**2 + dy**2)
            # Weaker attraction when far from original (allows spreading)
            attraction = attraction_strength / (1 + dist_from_original * 2)
            forces[i, 0] += attraction * dx
            forces[i, 1] += attraction * dy
        
        # Update velocities and positions
        velocities = damping * velocities + forces
        current_positions = current_positions + velocities
        
        # Clamp to bounds
        current_positions[:, 0] = np.clip(current_positions[:, 0], 
                                           x_min + margin_x, x_max - margin_x)
        current_positions[:, 1] = np.clip(current_positions[:, 1], 
                                           y_min + margin_y, y_max - margin_y)
    
    # =========================================================================
    # Phase 2: Overlap removal with minimal displacement
    # =========================================================================
    print(f"Phase 2: Overlap removal ({overlap_removal_iterations} max iterations)...")
    
    for iteration in range(overlap_removal_iterations):
        # Find all overlapping pairs
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                if boxes_overlap(current_positions[i], current_positions[j], 
                                raster_size, spacing):
                    dx = current_positions[j, 0] - current_positions[i, 0]
                    dy = current_positions[j, 1] - current_positions[i, 1]
                    overlap_x = collision_w - abs(dx)
                    overlap_y = collision_h - abs(dy)
                    overlaps.append((i, j, overlap_x, overlap_y))
        
        if not overlaps:
            print(f"  No overlaps remaining after {iteration} iterations")
            break
        
        # Resolve overlaps by pushing pairs apart minimally
        for i, j, overlap_x, overlap_y in overlaps:
            dx = current_positions[j, 0] - current_positions[i, 0]
            dy = current_positions[j, 1] - current_positions[i, 1]
            
            # Push apart along the axis with less overlap (easier to resolve)
            if overlap_x < overlap_y:
                # Push horizontally
                push = (overlap_x / 2 + 0.001) * np.sign(dx) if dx != 0 else overlap_x / 2 + 0.001
                current_positions[i, 0] -= push
                current_positions[j, 0] += push
            else:
                # Push vertically
                push = (overlap_y / 2 + 0.001) * np.sign(dy) if dy != 0 else overlap_y / 2 + 0.001
                current_positions[i, 1] -= push
                current_positions[j, 1] += push
        
        # Clamp to bounds
        current_positions[:, 0] = np.clip(current_positions[:, 0], 
                                           x_min + margin_x, x_max - margin_x)
        current_positions[:, 1] = np.clip(current_positions[:, 1], 
                                           y_min + margin_y, y_max - margin_y)
    
    # Check for remaining overlaps
    remaining_overlaps = 0
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(current_positions[i], current_positions[j], 
                            raster_size, spacing):
                remaining_overlaps += 1
    
    if remaining_overlaps > 0:
        print(f"  Warning: {remaining_overlaps} overlaps could not be resolved")
    else:
        print(f"  All overlaps resolved successfully")
    
    return current_positions


# =============================================================================
# Setup and Render Functions (for movie support)
# =============================================================================

def setup_tiled_rasters(spike_times_trials, trial_t_bins, eyepos, start_time, end_time,
                        n_cells,
                        dt=1/240,
                        microsaccade_threshold=0.3,
                        raster_width=0.2, raster_height=0.15,
                        raster_spacing=0.02,
                        xlim=None, ylim=None, space_bounds=None,
                        lim_dynamic=False,
                        layout_method='two_phase',
                        search_radius=2.0, search_step=0.01,
                        force_iterations=500,
                        show_true_positions=True,
                        show_position_lines=True,
                        color_by_position=True,
                        position_line_color='red',
                        position_line_alpha=0.5,
                        position_line_width=0.8,
                        preferred_orientations=None,
                        orientation_cmap='hsv',
                        highlight_orientations=None,
                        figsize=(14, 12),
                        title=None):
    """
    Setup for tiled raster visualization. Computes layout and creates figure.
    
    This function handles trial filtering, layout computation, and figure setup.
    Use render_tiled_frame() to draw spikes for a specific time window.
    
    Parameters
    ----------
    spike_times_trials : list
        List of spike times per trial
    trial_t_bins : np.ndarray
        Time bins for each trial, shape [trials, time]
    eyepos : np.ndarray
        Eye position data of shape [trials, time, 2]
    start_time : int
        Start time bin for filtering trials (microsaccade check)
    end_time : int
        End time bin for filtering trials (microsaccade check)
    n_cells : int
        Number of cells
    dt : float
        Time bin size in seconds (default 1/240)
    Other parameters same as plot_tiled_rasters.
        
    Returns
    -------
    setup_data : dict
        Dictionary containing all pre-computed data needed for rendering
    """
    n_trials = len(spike_times_trials)
    
    # Filter trials without microsaccades and compute median positions
    valid_trials = []
    median_positions = []
    
    for trial_idx in range(n_trials):
        eyepos_segment = eyepos[trial_idx, start_time:end_time, :]
        
        if np.all(np.isnan(eyepos_segment)):
            continue
        if microsaccade_exists(eyepos_segment, threshold=microsaccade_threshold):
            continue
        
        median_x = np.nanmedian(eyepos_segment[:, 0])
        median_y = np.nanmedian(eyepos_segment[:, 1])
        
        if np.isnan(median_x) or np.isnan(median_y):
            continue
        
        if xlim is not None:
            if median_x < xlim[0] or median_x > xlim[1]:
                continue
        if ylim is not None:
            if median_y < ylim[0] or median_y > ylim[1]:
                continue
        
        valid_trials.append(trial_idx)
        median_positions.append([median_x, median_y])
    
    median_positions = np.array(median_positions)
    n_valid = len(valid_trials)
    
    print(f"Found {n_valid} valid trials (no microsaccades, within xlim/ylim) out of {n_trials} total")
    
    if n_valid == 0:
        print("No valid trials found!")
        return None
    
    # Compute xlim/ylim from data if not provided
    if xlim is None:
        x_margin = 0.1
        xlim = (median_positions[:, 0].min() - x_margin, 
                median_positions[:, 0].max() + x_margin)
    if ylim is None:
        y_margin = 0.1
        ylim = (median_positions[:, 1].min() - y_margin, 
                median_positions[:, 1].max() + y_margin)
    
    if space_bounds is None:
        margin = max(raster_width, raster_height) * 2
        space_bounds = (xlim[0] - margin, xlim[1] + margin, 
                        ylim[0] - margin, ylim[1] + margin)
    
    # Apply layout algorithm
    raster_size = (raster_width, raster_height)
    
    if layout_method == 'greedy':
        print(f"Applying greedy non-overlapping layout...")
        final_positions = greedy_non_overlapping_layout(
            median_positions, raster_size, bounds=space_bounds,
            spacing=raster_spacing, search_radius=search_radius, search_step=search_step)
    elif layout_method == 'two_phase':
        print(f"Applying two-phase layout (force-directed + overlap removal)...")
        final_positions = two_phase_layout(
            median_positions, raster_size, bounds=space_bounds,
            spacing=raster_spacing, n_force_iterations=force_iterations)
    else:
        raise ValueError(f"Unknown layout_method: {layout_method}")
    
    # Dynamically adjust limits to fit all rasters if requested
    if lim_dynamic:
        margin = 0.05  # Small margin around the rasters
        xlim = (final_positions[:, 0].min() - raster_width/2 - margin,
                final_positions[:, 0].max() + raster_width/2 + margin)
        ylim = (final_positions[:, 1].min() - raster_height/2 - margin,
                final_positions[:, 1].max() + raster_height/2 + margin)
        print(f"Dynamic limits: xlim={xlim}, ylim={ylim}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.set_xlabel('X position (degrees)', fontsize=12)
    ax.set_ylabel('Y position (degrees)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Tiled Spike Rasters (n={n_valid} trials)', fontsize=14)
    
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    
    # Compute position-based colors
    if color_by_position and (show_true_positions or show_position_lines):
        x_positions = median_positions[:, 0]
        y_positions = median_positions[:, 1]
        color_xlim = (np.percentile(x_positions, 10), np.percentile(x_positions, 90))
        color_ylim = (np.percentile(y_positions, 10), np.percentile(y_positions, 90))
        position_colors = get_position_colors(median_positions, color_xlim, color_ylim)
    else:
        position_colors = None
    
    # Process preferred orientations if provided
    orientation_sort_order = None
    orientation_colors = None
    unique_orientations = None
    if preferred_orientations is not None:
        preferred_orientations = np.asarray(preferred_orientations)
        if len(preferred_orientations) != n_cells:
            raise ValueError(f"preferred_orientations length ({len(preferred_orientations)}) "
                           f"must match n_cells ({n_cells})")
        
        # Compute sort order by orientation (ascending)
        orientation_sort_order = np.argsort(preferred_orientations)
        
        # Get unique orientations for legend and discrete color mapping
        unique_orientations = np.unique(preferred_orientations)
        
        # Get colors for each cell based on orientation (discrete mapping)
        orientation_colors = get_orientation_colors(
            preferred_orientations, orientation_cmap, unique_orientations, 
            highlight_orientations=highlight_orientations
        )
        
        print(f"Orientation sorting enabled: {len(unique_orientations)} unique orientations")
    
    # Add legend
    if show_true_positions or show_position_lines:
        if color_by_position:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=(0.9, 0.1, 0.1), edgecolor='black', label='Top-Left (Red)'),
                Patch(facecolor=(1.0, 0.6, 0.0), edgecolor='black', label='Top-Right (Orange)'),
                Patch(facecolor=(0.0, 0.2, 0.8), edgecolor='black', label='Bottom-Left (Blue)'),
                Patch(facecolor=(0.1, 0.7, 0.2), edgecolor='black', label='Bottom-Right (Green)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, title='Position Color')
        else:
            if show_true_positions:
                ax.scatter([], [], s=10, c='red', alpha=0.5, label='True eye position')
            if show_position_lines:
                ax.plot([], [], color=position_line_color, linewidth=position_line_width, 
                       alpha=position_line_alpha, label='Position offset')
            ax.legend(loc='upper right', fontsize=10)
    
    # Add orientation legend if orientations provided
    if unique_orientations is not None:
        # If highlighting specific orientations, only show those in legend
        legend_orientations = highlight_orientations if highlight_orientations is not None else unique_orientations
        draw_orientation_legend_with_lines(ax, legend_orientations, cmap_name=orientation_cmap,
                                           position=(0.02, 0.98), fontsize=8)
    
    return {
        'fig': fig, 'ax': ax,
        'valid_trials': valid_trials,
        'final_positions': final_positions,
        'median_positions': median_positions,
        'position_colors': position_colors,
        'n_cells': n_cells, 'dt': dt,
        'raster_width': raster_width, 'raster_height': raster_height,
        'xlim': xlim, 'ylim': ylim,
        'color_by_position': color_by_position,
        'show_true_positions': show_true_positions,
        'show_position_lines': show_position_lines,
        'position_line_color': position_line_color,
        'position_line_alpha': position_line_alpha,
        'position_line_width': position_line_width,
        'orientation_sort_order': orientation_sort_order,
        'orientation_colors': orientation_colors,
        'orientation_cmap': orientation_cmap,
        'preferred_orientations': preferred_orientations,
    }


def render_tiled_frame(ax, setup_data, spike_times_trials, trial_t_bins,
                       window_start, window_end,
                       spike_linewidth=0.3, spike_alpha=0.8,
                       clear_previous=True, show_progress=False,
                       check_consistency=False):
    """
    Render spike rasters for a specific time window.
    
    Parameters
    ----------
    ax : matplotlib axis
    setup_data : dict
        Data returned from setup_tiled_rasters()
    spike_times_trials, trial_t_bins : 
        Spike data
    window_start, window_end : int
        Time window for this frame
    spike_linewidth, spike_alpha : float
        Spike rendering options
    clear_previous : bool
        If True, clear previous artists before drawing
    show_progress : bool
        If True, show tqdm progress bar
    check_consistency : bool
        If True, verify spikes in overlapping region match previous frame
        
    Returns
    -------
    artists : list
        List of matplotlib artists created
    """
    valid_trials = setup_data['valid_trials']
    final_positions = setup_data['final_positions']
    median_positions = setup_data['median_positions']
    position_colors = setup_data['position_colors']
    n_cells = setup_data['n_cells']
    dt = setup_data['dt']
    raster_width = setup_data['raster_width']
    raster_height = setup_data['raster_height']
    color_by_position = setup_data['color_by_position']
    show_true_positions = setup_data['show_true_positions']
    show_position_lines = setup_data['show_position_lines']
    position_line_color = setup_data['position_line_color']
    position_line_alpha = setup_data['position_line_alpha']
    position_line_width = setup_data['position_line_width']
    orientation_sort_order = setup_data.get('orientation_sort_order')
    orientation_colors = setup_data.get('orientation_colors')
    
    time_window = window_end - window_start
    
    # Clear previous frame content
    if clear_previous:
        for artist in ax.patches[:]:
            artist.remove()
        for line in ax.lines[:]:
            if hasattr(line, 'get_color') and line.get_color() == 'gray':
                continue
            line.remove()
        for coll in ax.collections[:]:
            coll.remove()
    
    # Initialize spike tracking for consistency check
    if check_consistency:
        current_frame_spikes = {}  # trial_idx -> list of spike times
        if '_prev_frame_spikes' not in setup_data:
            setup_data['_prev_frame_spikes'] = None
            setup_data['_prev_window'] = None
    
    artists = []
    trial_iter = tqdm(enumerate(valid_trials), total=len(valid_trials), desc="Rendering") if show_progress else enumerate(valid_trials)
    
    for i, trial_idx in trial_iter:
        pos_x, pos_y = final_positions[i]
        true_x, true_y = median_positions[i]
        
        t_bins = trial_t_bins[trial_idx]
        valid_mask = ~np.isnan(t_bins)
        if not np.any(valid_mask):
            continue
            
        valid_t_bins = t_bins[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        window_mask = (valid_indices >= window_start) & (valid_indices < window_end)
        if not np.any(window_mask):
            continue
        
        # Use a reference point to compute expected time at window boundaries
        # This ensures consistent time ranges across frames even if edge bins are NaN
        ref_idx = valid_indices[0]
        ref_time = valid_t_bins[0]
        
        # Expected time at window boundaries (bins are spaced by dt)
        t_start = ref_time + (window_start - ref_idx - 0.5) * dt
        t_end = ref_time + (window_end - 1 - ref_idx + 0.5) * dt
        
        times_list = []
        cells_list = []
        
        for cell_idx in range(n_cells):
            cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
            if cell_spikes.size == 0:
                continue
            
            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
            filtered = cell_spikes[mask]
            
            if filtered.size > 0:
                bin_coords = (filtered - t_start) / dt
                times_list.append(bin_coords)
                cells_list.append(np.full(filtered.size, cell_idx))
                
                # Track actual spike times for consistency check
                if check_consistency:
                    if trial_idx not in current_frame_spikes:
                        current_frame_spikes[trial_idx] = []
                    current_frame_spikes[trial_idx].extend(filtered.tolist())
        
        raster_color = position_colors[i] if position_colors is not None else position_line_color
        edge_color = raster_color if color_by_position else 'gray'
        
        rect = plt.Rectangle((pos_x - raster_width/2, pos_y - raster_height/2),
                              raster_width, raster_height,
                              fill=True, facecolor='white', edgecolor=edge_color,
                              linewidth=2 if color_by_position else 0.5, alpha=1, zorder=1)
        ax.add_patch(rect)
        artists.append(rect)
        
        if times_list:
            times = np.concatenate(times_list)
            cells = np.concatenate(cells_list).astype(int)
            
            # If orientation sorting is enabled, map cell indices to sorted positions
            if orientation_sort_order is not None:
                # Create reverse mapping: original cell idx -> sorted position
                sorted_positions = np.zeros(n_cells, dtype=int)
                sorted_positions[orientation_sort_order] = np.arange(n_cells)
                display_cells = sorted_positions[cells]
            else:
                display_cells = cells
            
            spike_x = pos_x - raster_width/2 + (times / time_window) * raster_width
            spike_y = pos_y - raster_height/2 + (display_cells / n_cells) * raster_height
            spike_height = raster_height / n_cells * 0.8
            
            # Color spikes by orientation if available
            if orientation_colors is not None:
                # Group spikes by color for efficient rendering
                unique_cells = np.unique(cells)
                for cell_idx in unique_cells:
                    cell_mask = cells == cell_idx
                    cell_color = orientation_colors[cell_idx]
                    line_artist = plot_spikes_as_lines(
                        ax, spike_x[cell_mask], spike_y[cell_mask], 
                        height=spike_height, color=cell_color, 
                        linewidth=spike_linewidth, alpha=spike_alpha)
                    if line_artist is not None:
                        artists.append(line_artist)
            else:
                line_artist = plot_spikes_as_lines(ax, spike_x, spike_y, height=spike_height,
                                                   color='black', linewidth=spike_linewidth, alpha=spike_alpha)
                if line_artist is not None:
                    artists.append(line_artist)
        
        if show_position_lines:
            line_color = raster_color if color_by_position else position_line_color
            line_width = position_line_width * 1.5 if color_by_position else position_line_width
            line, = ax.plot([pos_x, true_x], [pos_y, true_y], 
                           color=line_color, linewidth=line_width, 
                           alpha=position_line_alpha, zorder=2)
            artists.append(line)
        
        if show_true_positions:
            dot_color = raster_color if color_by_position else 'red'
            scatter = ax.scatter(true_x, true_y, s=40, c=[dot_color], alpha=position_line_alpha, 
                                zorder=5, edgecolors='black', linewidths=0.5)
            artists.append(scatter)
    
    # Consistency check: compare with previous frame
    if check_consistency and setup_data['_prev_frame_spikes'] is not None:
        prev_spikes = setup_data['_prev_frame_spikes']
        prev_window = setup_data['_prev_window']
        
        # Overlap region: [window_start, prev_window[1])
        overlap_start = window_start
        overlap_end = prev_window[1]
        
        mismatches = []
        for trial_idx in valid_trials:
            if trial_idx not in current_frame_spikes and trial_idx not in prev_spikes:
                continue
            
            curr_spikes_all = set(current_frame_spikes.get(trial_idx, []))
            prev_spikes_all = set(prev_spikes.get(trial_idx, []))
            
            # Get time bounds for overlap region
            t_bins = trial_t_bins[trial_idx]
            valid_mask = ~np.isnan(t_bins)
            valid_t_bins = t_bins[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            ref_idx = valid_indices[0]
            ref_time = valid_t_bins[0]
            
            overlap_t_start = ref_time + (overlap_start - ref_idx - 0.5) * dt
            overlap_t_end = ref_time + (overlap_end - 1 - ref_idx + 0.5) * dt
            
            # Filter to overlap region
            curr_overlap = {s for s in curr_spikes_all if overlap_t_start <= s < overlap_t_end}
            prev_overlap = {s for s in prev_spikes_all if overlap_t_start <= s < overlap_t_end}
            
            if curr_overlap != prev_overlap:
                mismatches.append({
                    'trial': trial_idx,
                    'prev_count': len(prev_overlap),
                    'curr_count': len(curr_overlap),
                    'missing': len(prev_overlap - curr_overlap),
                    'extra': len(curr_overlap - prev_overlap),
                })
        
        if mismatches:
            print(f"CONSISTENCY CHECK FAILED at window [{window_start}, {window_end}):")
            for m in mismatches[:5]:
                print(f"  Trial {m['trial']}: prev={m['prev_count']}, curr={m['curr_count']}, "
                      f"missing={m['missing']}, extra={m['extra']}")
        else:
            print(f"Consistency OK at window [{window_start}, {window_end}): "
                  f"{len(current_frame_spikes)} trials with spikes")
    
    # Store current frame data for next comparison
    if check_consistency:
        setup_data['_prev_frame_spikes'] = current_frame_spikes
        setup_data['_prev_window'] = (window_start, window_end)
    
    return artists


def get_spikes_for_window(spike_times_trials, trial_t_bins, valid_trials, n_cells, 
                          window_start, window_end, dt):
    """
    Get spike counts for each trial in a specific time window.
    Returns a dict mapping trial_idx -> total spike count in window.
    Also returns the actual spike times for comparison.
    """
    spike_info = {}
    
    for trial_idx in valid_trials:
        t_bins = trial_t_bins[trial_idx]
        valid_mask = ~np.isnan(t_bins)
        if not np.any(valid_mask):
            continue
            
        valid_t_bins = t_bins[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Use reference point for time computation
        ref_idx = valid_indices[0]
        ref_time = valid_t_bins[0]
        
        t_start = ref_time + (window_start - ref_idx - 0.5) * dt
        t_end = ref_time + (window_end - 1 - ref_idx + 0.5) * dt
        
        trial_spikes = []
        for cell_idx in range(n_cells):
            cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
            if cell_spikes.size == 0:
                continue
            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
            trial_spikes.extend(cell_spikes[mask].tolist())
        
        spike_info[trial_idx] = {
            'count': len(trial_spikes),
            'spikes': sorted(trial_spikes),
            't_start': t_start,
            't_end': t_end,
        }
    
    return spike_info


def check_spike_consistency(spike_times_trials, trial_t_bins, valid_trials, n_cells,
                            total_start_time, total_end_time, len_of_each_segment, dt):
    """
    Check spike consistency between consecutive frames.
    
    For overlapping region between frames, spikes should be consistent.
    """
    print("Checking spike consistency between frames...")
    
    n_frames = total_end_time - total_start_time - len_of_each_segment + 1
    inconsistencies = []
    
    prev_info = None
    prev_window = None
    
    for frame_idx in range(n_frames):
        window_start = total_start_time + frame_idx
        window_end = window_start + len_of_each_segment
        
        curr_info = get_spikes_for_window(
            spike_times_trials, trial_t_bins, valid_trials, n_cells,
            window_start, window_end, dt
        )
        
        if prev_info is not None:
            # Overlapping region: [window_start, prev_window[1])
            # = [window_start, prev_window_start + len_of_each_segment)
            # = [window_start, window_start - 1 + len_of_each_segment)
            # Actually the overlap is [window_start, prev_window_end) 
            # = [prev_window_start + 1, prev_window_start + len_of_each_segment)
            overlap_start = window_start
            overlap_end = prev_window[1]  # prev_window_end
            
            # Get spikes in overlap region for both frames
            for trial_idx in valid_trials:
                if trial_idx not in curr_info or trial_idx not in prev_info:
                    continue
                
                curr_spikes = set(curr_info[trial_idx]['spikes'])
                prev_spikes = set(prev_info[trial_idx]['spikes'])
                
                # Spikes that should be in both: those in the overlap time region
                t_bins = trial_t_bins[trial_idx]
                valid_mask = ~np.isnan(t_bins)
                valid_t_bins = t_bins[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                ref_idx = valid_indices[0]
                ref_time = valid_t_bins[0]
                
                overlap_t_start = ref_time + (overlap_start - ref_idx - 0.5) * dt
                overlap_t_end = ref_time + (overlap_end - 1 - ref_idx + 0.5) * dt
                
                # Filter to overlap region
                curr_overlap = {s for s in curr_spikes if overlap_t_start <= s < overlap_t_end}
                prev_overlap = {s for s in prev_spikes if overlap_t_start <= s < overlap_t_end}
                
                if curr_overlap != prev_overlap:
                    missing_in_curr = prev_overlap - curr_overlap
                    missing_in_prev = curr_overlap - prev_overlap
                    inconsistencies.append({
                        'frame': frame_idx,
                        'trial': trial_idx,
                        'overlap_region': (overlap_start, overlap_end),
                        'overlap_time': (overlap_t_start, overlap_t_end),
                        'curr_count': len(curr_overlap),
                        'prev_count': len(prev_overlap),
                        'missing_in_curr': missing_in_curr,
                        'missing_in_prev': missing_in_prev,
                    })
        
        prev_info = curr_info
        prev_window = (window_start, window_end)
    
    if inconsistencies:
        print(f"Found {len(inconsistencies)} inconsistencies!")
        for inc in inconsistencies[:10]:  # Show first 10
            print(f"  Frame {inc['frame']}, Trial {inc['trial']}: "
                  f"overlap [{inc['overlap_region'][0]}-{inc['overlap_region'][1]}], "
                  f"prev={inc['prev_count']} spikes, curr={inc['curr_count']} spikes")
            if inc['missing_in_curr']:
                print(f"    Missing in current: {list(inc['missing_in_curr'])[:5]}")
            if inc['missing_in_prev']:
                print(f"    Missing in previous: {list(inc['missing_in_prev'])[:5]}")
    else:
        print("No inconsistencies found - spikes are consistent across frames!")
    
    return inconsistencies


def tiled_rasters_movie(spike_times_trials, trial_t_bins, eyepos,
                        total_start_time, total_end_time, len_of_each_segment,
                        n_cells,
                        dt=1/240,
                        microsaccade_threshold=0.3,
                        raster_width=0.2, raster_height=0.15,
                        raster_spacing=0.02,
                        xlim=None, ylim=None, space_bounds=None,
                        lim_dynamic=False,
                        layout_method='two_phase',
                        search_radius=2.0, search_step=0.01,
                        force_iterations=500,
                        show_true_positions=True,
                        show_position_lines=True,
                        color_by_position=True,
                        position_line_color='red',
                        position_line_alpha=0.5,
                        position_line_width=0.8,
                        preferred_orientations=None,
                        orientation_cmap='hsv',
                        highlight_orientations=None,
                        figsize=(14, 12),
                        spike_linewidth=0.3,
                        spike_alpha=0.8,
                        title=None,
                        fps=10,
                        output_path='tiled_rasters_movie.mp4',
                        dpi=150,
                        check_consistency=False):
    """
    Create a movie of tiled spike rasters with a sliding time window.
    
    The movie shows rasters from [window_start, window_start + len_of_each_segment],
    stepping one bin at a time from total_start_time until the window reaches total_end_time.
    
    Parameters
    ----------
    total_start_time : int
        Start of the movie (first window starts here)
    total_end_time : int
        End of the movie (last window ends here)
    len_of_each_segment : int
        Number of bins to show in each frame
    fps : int
        Frames per second for output video
    output_path : str
        Path to save the movie
    dpi : int
        Resolution for the movie
    check_consistency : bool
        If True, run spike consistency check before creating movie (for debugging)
    Other parameters same as plot_tiled_rasters.
        
    Returns
    -------
    setup_data : dict
        The setup data
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    
    print("Setting up tiled rasters...")
    setup_data = setup_tiled_rasters(
        spike_times_trials=spike_times_trials,
        trial_t_bins=trial_t_bins,
        eyepos=eyepos,
        start_time=total_start_time,
        end_time=total_end_time,
        n_cells=n_cells,
        dt=dt,
        microsaccade_threshold=microsaccade_threshold,
        raster_width=raster_width,
        raster_height=raster_height,
        raster_spacing=raster_spacing,
        xlim=xlim, ylim=ylim, space_bounds=space_bounds,
        lim_dynamic=lim_dynamic,
        layout_method=layout_method,
        search_radius=search_radius, search_step=search_step,
        force_iterations=force_iterations,
        show_true_positions=show_true_positions,
        show_position_lines=show_position_lines,
        color_by_position=color_by_position,
        position_line_color=position_line_color,
        position_line_alpha=position_line_alpha,
        position_line_width=position_line_width,
        preferred_orientations=preferred_orientations,
        orientation_cmap=orientation_cmap,
        highlight_orientations=highlight_orientations,
        figsize=figsize,
        title=title,
    )
    
    if setup_data is None:
        print("Setup failed - no valid trials")
        return None
    
    # Run consistency check if requested
    if check_consistency:
        inconsistencies = check_spike_consistency(
            spike_times_trials, trial_t_bins, 
            setup_data['valid_trials'], n_cells,
            total_start_time, total_end_time, len_of_each_segment, dt
        )
        if inconsistencies:
            print(f"WARNING: Found {len(inconsistencies)} spike inconsistencies!")
    
    fig = setup_data['fig']
    ax = setup_data['ax']
    
    n_frames = total_end_time - total_start_time - len_of_each_segment + 1
    print(f"Creating movie with {n_frames} frames...")
    
    base_title = title if title else "Tiled Spike Rasters"
    
    def update(frame_idx):
        window_start = total_start_time + frame_idx
        window_end = window_start + len_of_each_segment
        
        ax.set_title(f'{base_title} (bins {window_start}-{window_end})', fontsize=14)
        
        render_tiled_frame(
            ax=ax, setup_data=setup_data,
            spike_times_trials=spike_times_trials,
            trial_t_bins=trial_t_bins,
            window_start=window_start,
            window_end=window_end,
            spike_linewidth=spike_linewidth,
            spike_alpha=spike_alpha,
            clear_previous=True,
            check_consistency=check_consistency,
            show_progress=False,
        )
        return []
    
    print("Rendering frames...")
    anim = FuncAnimation(fig, update, frames=tqdm(range(n_frames), desc="Rendering movie"),
                         blit=False, repeat=False)
    
    print(f"Saving movie to {output_path}...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='tile_population'), bitrate=5000)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    print(f"Movie saved to {output_path}")
    plt.close(fig)
    
    return setup_data


# =============================================================================
# Main plotting function (uses setup + render internally)
# =============================================================================

def plot_tiled_rasters(spike_times_trials, trial_t_bins, eyepos, start_time, end_time,
                       n_cells,
                       dt=1/240,
                       microsaccade_threshold=0.3,
                       raster_width=0.2, raster_height=0.15,
                       raster_spacing=0.02,
                       xlim=None,
                       ylim=None,
                       space_bounds=None,
                       lim_dynamic=False,
                       layout_method='two_phase',
                       search_radius=2.0,
                       search_step=0.01,
                       force_iterations=500,
                       show_true_positions=True,
                       show_position_lines=True,
                       color_by_position=True,
                       position_line_color='red',
                       position_line_alpha=0.5,
                       position_line_width=0.8,
                       preferred_orientations=None,
                       orientation_cmap='hsv',
                       highlight_orientations=None,
                       figsize=(14, 12),
                       spike_linewidth=0.3,
                       spike_alpha=0.8,
                       title=None,
                       check_consistency=False):
    """
    Plot spike raster plots tiled across visual space based on eye position.
    
    Uses force-directed layout to spread out overlapping rasters while
    preserving spatial relationships.
    
    Parameters
    ----------
    spike_times_trials : list
        List of spike times per trial, where spike_times_trials[trial_idx][cell_idx] 
        contains spike times for that trial and cell
    trial_t_bins : np.ndarray
        Time bins for each trial, shape [trials, time]
    eyepos : np.ndarray
        Eye position data of shape [trials, time, 2]
    start_time : int
        Start time bin
    end_time : int
        End time bin
    n_cells : int
        Number of cells
    dt : float
        Time bin size in seconds (default 1/240)
    microsaccade_threshold : float
        Threshold for detecting microsaccades (degrees)
    raster_width : float
        Width of each raster plot in degrees
    raster_height : float
        Height of each raster plot in degrees
    raster_spacing : float
        Minimum spacing/gap between adjacent rasters in degrees
    xlim : tuple, optional
        (x_min, x_max) limits for the plot. Trials with median fixation outside 
        this range are excluded. If None, computed from data.
    ylim : tuple, optional
        (y_min, y_max) limits for the plot. Trials with median fixation outside 
        this range are excluded. If None, computed from data.
    space_bounds : tuple, optional
        (x_min, x_max, y_min, y_max) bounds for raster placement. If None, 
        derived from xlim/ylim with some margin for displaced rasters.
    lim_dynamic : bool
        If True, ignores xlim/ylim and dynamically adjusts axis limits to fit
        all rasters (including their full width/height) with a small margin.
        Useful when rasters extend beyond the original xlim/ylim bounds.
    layout_method : str
        Algorithm for placing rasters. Options:
        - 'greedy': Place rasters one at a time at nearest non-overlapping position.
          Fast but may not preserve spatial neighborhoods well.
        - 'two_phase': Force-directed simulation followed by overlap removal.
          Better at preserving spatial relationships between neighbors.
    search_radius : float
        (greedy method) Maximum distance to search from true position
    search_step : float
        (greedy method) Step size for spiral search
    force_iterations : int
        (two_phase method) Number of force-directed iterations
    show_true_positions : bool
        Whether to show dots at true eye positions
    show_position_lines : bool
        Whether to draw lines from raster to true position
    color_by_position : bool
        If True, color the point, line, and raster border based on spatial position
        using a 2D colormap (corners: blue, cyan, red, yellow). This makes it easier
        to track which raster belongs to which position. If False, uses position_line_color.
    position_line_color : str
        Color of the line connecting raster to true position (only used if color_by_position=False)
    position_line_alpha : float
        Alpha/transparency of the position line (0-1)
    position_line_width : float
        Line width of the position line
    preferred_orientations : np.ndarray, optional
        Array of preferred orientations (in degrees) for each cell. 
        If provided, cells are ordered by orientation within each raster (bottom=low, top=high),
        and spikes are colored by the cell's orientation using a cyclic colormap.
        Length must equal n_cells.
    orientation_cmap : str
        Matplotlib colormap for orientation colors. Default 'hsv' (cyclic).
        Other good options: 'tab10', 'Set1' (categorical, better for discrete orientations).
    highlight_orientations : list, optional
        If provided, only spikes from cells with these orientations are colored;
        all other orientations are shown in dark gray. Useful for tracking specific
        orientations across space. Example: [11.25, 101.25]
    figsize : tuple
        Figure size
    spike_linewidth : float
        Line width for spike ticks
    spike_alpha : float
        Alpha for spike ticks
    title : str
        Figure title
        
    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axis
    valid_trials : list
        List of trial indices that passed microsaccade filter
    final_positions : np.ndarray
        Final raster positions after force-directed layout
    """
    # Use setup to compute layout and create figure
    setup_data = setup_tiled_rasters(
        spike_times_trials=spike_times_trials,
        trial_t_bins=trial_t_bins,
        eyepos=eyepos,
        start_time=start_time,
        end_time=end_time,
        n_cells=n_cells,
        dt=dt,
        microsaccade_threshold=microsaccade_threshold,
        raster_width=raster_width,
        raster_height=raster_height,
        raster_spacing=raster_spacing,
        xlim=xlim,
        ylim=ylim,
        space_bounds=space_bounds,
        lim_dynamic=lim_dynamic,
        layout_method=layout_method,
        search_radius=search_radius,
        search_step=search_step,
        force_iterations=force_iterations,
        show_true_positions=show_true_positions,
        show_position_lines=show_position_lines,
        color_by_position=color_by_position,
        position_line_color=position_line_color,
        position_line_alpha=position_line_alpha,
        position_line_width=position_line_width,
        preferred_orientations=preferred_orientations,
        orientation_cmap=orientation_cmap,
        highlight_orientations=highlight_orientations,
        figsize=figsize,
        title=title,
    )

    if setup_data is None:
        return None, None, [], np.array([])
    
    # Update title with time range if not provided
    if title is None:
        n_valid = len(setup_data['valid_trials'])
        setup_data['ax'].set_title(f'Tiled Spike Rasters (n={n_valid} trials, time={start_time}-{end_time})', fontsize=14)
    
    # Render the frame
    render_tiled_frame(
        ax=setup_data['ax'],
        setup_data=setup_data,
        spike_times_trials=spike_times_trials,
        trial_t_bins=trial_t_bins,
        window_start=start_time,
        window_end=end_time,
        spike_linewidth=spike_linewidth,
        spike_alpha=spike_alpha,
        clear_previous=False,
        show_progress=True,
        check_consistency=check_consistency,
    )
    
    plt.tight_layout()
    
    return setup_data['fig'], setup_data['ax'], setup_data['valid_trials'], setup_data['final_positions']



# =============================================================================
# Run the visualization
# =============================================================================
from tejas.rsvp_util import get_fixrsvp_data
subject = 'Allen'
date = '2022-03-02'

dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'

data = get_fixrsvp_data(subject, date, dataset_configs_path, 
use_cached_data=True, 
salvageable_mismatch_time_threshold=25, verbose=True)

from tejas.metrics.gratings import get_gratings_for_dataset, plot_ori_tuning
gratings_info = get_gratings_for_dataset(date, subject, cache = True)

robs = data['robs']
dfs = data['dfs']
eyepos = data['eyepos']
fix_dur = data['fix_dur']
image_ids = data['image_ids']
cids = data['cids']
spike_times_trials = data['spike_times_trials']
trial_t_bins = data['trial_t_bins']
trial_time_windows = data['trial_time_windows']
rsvp_images = data['rsvp_images']
dataset = data['dataset']

preferred_orientations = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cids], axis=-1)]

#%%

# Get number of cells from spike_times_trials structure
n_cells = len(spike_times_trials[0])  # Number of cells per trial
total_start_time = 40
total_end_time = 80
fig, ax, valid_trials, final_positions = plot_tiled_rasters(
    spike_times_trials=spike_times_trials,
    trial_t_bins=trial_t_bins,
    eyepos=eyepos,
    start_time=total_start_time,
    end_time=total_end_time,
    n_cells=n_cells,
    dt=1/240,
    microsaccade_threshold=0.15,
    raster_width=0.15,
    raster_height=0.15,
    # raster_width=0.2,
    # raster_height=0.2,
    raster_spacing=0.02,  # Minimum gap between adjacent rasters
    xlim=(-0.9, 0.9),  # Only include trials with median fixation in this x range
    ylim=(-0.4, 1.2),  # Only include trials with median fixation in this y range
    # xlim=(-1.2, 1.2),
    # ylim=(-1.2, 1.2),
    lim_dynamic=False,
    # space_bounds is computed automatically from xlim/ylim with margin
    layout_method='two_phase',  # 'greedy' or 'two_phase'
    force_iterations=500,  # For two_phase method
    search_radius=2.0,
    search_step=0.01,
    show_true_positions=True,
    show_position_lines=True,
    color_by_position=False,  # Use 2D colormap based on position (corners: blue, cyan, red, yellow)
    position_line_color='red',  # Only used if color_by_position=False
    position_line_alpha=0.6,
    position_line_width=1.0,
    figsize=(14, 10),
    spike_linewidth=1.3,
    spike_alpha=1,
    title=f'Tiled Spike Rasters - {subject} {date}',
    check_consistency=True,
    # preferred_orientations=preferred_orientations,  # your array
    # orientation_cmap='tab10',  # or 'twilight',
    # highlight_orientations=[123.75],
)

plt.show()


#%%
# Save the figure
# fig.savefig(f'tiled_rasters_{subject}_{date}.pdf', dpi=150, bbox_inches='tight')
# fig.savefig(f'tiled_rasters_{subject}_{date}.png', dpi=150, bbox_inches='tight')

#%%
# Create a movie with sliding time window
len_of_each_segment = 40  # bins to show in each frame
tiled_rasters_movie(
    spike_times_trials=spike_times_trials,
    trial_t_bins=trial_t_bins,
    eyepos=eyepos,
    total_start_time=30,
    total_end_time=90,
    len_of_each_segment=len_of_each_segment,
    n_cells=n_cells,
    dt=1/240,
    microsaccade_threshold=0.15,
    raster_width=0.15,
    raster_height=0.15,
    raster_spacing=0.02,
    xlim=(-0.9, 0.9),
    ylim=(-0.4, 1.2),
    layout_method='two_phase',
    force_iterations=500,
    show_true_positions=True,
    show_position_lines=True,
    color_by_position=True,
    figsize=(14, 10),
    spike_linewidth=1.3,
    spike_alpha=1,
    title=f'Tiled Spike Rasters - {subject} {date}',
    fps=10,
    output_path=f'tiled_rasters_movie_{subject}_{date}.mp4',
    dpi=300,
    # check_consistency=True,
)
#%%

