from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.widgets import Slider
from matplotlib import animation
from tejas.metrics.gratings import get_gratings_for_dataset,get_peak_sf_for_dataset
from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset
from tejas.metrics.mcfarland import get_mcfarland_analysis_for_dataset
from tejas.metrics.standardmodelfits import get_ln_standard_model_fits, get_energy_standard_model_fits, get_linear_index_standard_model_fits
import numpy as np
from tejas.metrics.gaborium import get_rf_contour_metrics, get_rf_gaussian_fit_metrics
from tejas.metrics.qc import get_qc_units_for_session, get_qc_units_from_information_dict, get_information_for_qc
from scipy.stats import pearsonr, wilcoxon
import pickle
from pathlib import Path
# Lazy import to avoid circular dependency - import inside function where needed

# Configure matplotlib for PDF output with embedded fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Global cache - loaded once at module import time
_global_cache = None
_default_cache_file = '/mnt/ssd/YatesMarmoV1/metrics/all_metrics_cache.pkl'
# Cache BPS lookups so repeated slider updates do not re-read thousands of JSON files
_bps_cache = {'ln': {}, 'energy': {}}

def _load_global_cache(cache_file=None):
    """Load the global cache from file. Called once at module import."""
    global _global_cache
    if _global_cache is not None:
        return _global_cache
    
    import time
    cache_path = Path(cache_file if cache_file is not None else _default_cache_file)
    if cache_path.exists():
        try:
            print(f"Loading global cache from {cache_path}...")
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                _global_cache = pickle.load(f)
            cache_load_time = time.time() - start_time
            print(f"Global cache loaded in {cache_load_time:.2f} seconds")
        except Exception as e:
            print(f"Warning: Failed to load global cache from {cache_path}: {e}")
            _global_cache = {}
    else:
        print(f"No cache file found at {cache_path}, starting with empty cache")
        _global_cache = {}
    
    return _global_cache

# Load global cache at module import time
_load_global_cache()

def _parse_transformed_variable(variable_name):
    """
    Parse variable names that include transformations like 1/tau, 1/alpha, 1-tau, or 1-alpha.
    
    Supports:
        - 1/tau_* : reciprocal of tau (1/tau)
        - 1/alpha_* : reciprocal of alpha (1/alpha)
        - 1-tau_* : complement of tau (1-tau)
        - 1-alpha_* : complement of alpha (1-alpha)
    
    Returns: (base_variable_name, transformation_type)
        transformation_type is one of: None, 'reciprocal', 'complement'
    """
    # Check for reciprocal (1/x)
    if variable_name.startswith('1/'):
        base_name = variable_name[2:]  # Remove '1/' prefix
        # Validate that base_name starts with tau_ or alpha_
        if base_name.startswith('tau_') or base_name.startswith('alpha_'):
            return base_name, 'reciprocal'
    
    # Check for complement (1-x)
    if variable_name.startswith('1-'):
        base_name = variable_name[2:]  # Remove '1-' prefix
        # Validate that base_name starts with tau_ or alpha_
        if base_name.startswith('tau_') or base_name.startswith('alpha_'):
            return base_name, 'complement'
    
    return variable_name, None

def _apply_transformation(data_dict, base_variable_name, transformed_variable_name, transformation_type):
    """
    Apply a transformation (reciprocal or complement) to a variable's data.
    
    Returns: transformed data dict with the same structure
    """
    result = {}
    for unit_idx, unit_data in data_dict.items():
        if unit_data['valid']:
            base_value = unit_data[base_variable_name]
            if transformation_type == 'reciprocal':
                # Apply 1/x transformation
                if base_value != 0 and np.isfinite(base_value):
                    transformed_value = 1.0 / base_value
                else:
                    transformed_value = np.nan
            elif transformation_type == 'complement':
                # Apply 1-x transformation
                if np.isfinite(base_value):
                    transformed_value = 1.0 - base_value
                else:
                    transformed_value = np.nan
            else:
                transformed_value = base_value
            
            result[unit_idx] = {
                'valid': unit_data['valid'] and np.isfinite(transformed_value),
                transformed_variable_name: transformed_value
            }
        else:
            result[unit_idx] = {
                'valid': False,
                transformed_variable_name: np.nan
            }
    
    return result

def _extract_mcfarland_data(raw_data, variable_name):
    """
    Extract McFarland data from nested structure and flatten to expected format.
    
    Variable names should be in format: {metric}_{source}_{bins}_bins
    e.g., 'tau_data_10_bins', 'alpha_ln_37_bins'
    
    Returns: {unit_idx: {'valid': bool, variable_name: float}}
    """
    # Parse variable name
    parts = variable_name.split('_')
    if len(parts) != 4 or parts[3] != 'bins':
        raise ValueError(f"Invalid McFarland variable name: {variable_name}. Expected format: {{metric}}_{{source}}_{{bins}}_bins")
    
    metric = parts[0]  # 'tau' or 'alpha'
    source = parts[1]  # 'data', 'ln', or 'energy'
    bins = int(parts[2])  # 10 or 37
    
    if metric not in ['tau', 'alpha']:
        raise ValueError(f"Invalid metric: {metric}. Expected 'tau' or 'alpha'")
    if source not in ['data', 'ln', 'energy']:
        raise ValueError(f"Invalid source: {source}. Expected 'data', 'ln', or 'energy'")
    if bins not in [10, 37]:
        raise ValueError(f"Invalid bins: {bins}. Expected 10 or 37")
    
    # Map source to key in nested structure
    source_key_map = {
        'data': 'mcfarland_robs_out',
        'ln': 'mcfarland_rhat_ln_out',
        'energy': 'mcfarland_rhat_energy_out'
    }
    source_key = source_key_map[source]
    
    # Extract and flatten data
    result = {}
    bins_data = raw_data[bins]
    for unit_idx, unit_data in bins_data.items():
        nested_data = unit_data[source_key]
        result[unit_idx] = {
            'valid': nested_data['valid'],
            variable_name: nested_data[metric]
        }
    
    return result

def plot_variable1_vs_variable2_for_session(dict1, dict2, variable1_name, variable2_name, ax = None, session_date = None, session_subject = None):
    ''' 
    Plot variable1 vs variable2
    variable1: must be a dict with keys as unit indices and values as a dict with key as name of variable1 and boolean value of valid
    variable2: must be a dict with keys as unit indices and values as a dict with key as name of variable2 and boolean value of valid
    session_date: date of the session
    session_subject: subject of the session
    Returns: list of (line, date, subject, unit_idx) tuples for each plotted point
    '''
    # assert set(dict1.keys()) == set(dict2.keys()), 'variable1 and variable2 must have the same keys'

    intersection = set(dict1.keys()) & set(dict2.keys())


    
    plotted_points = []
    # for key in dict1.keys():
    for key in intersection:
        if dict1[key]['valid'] and dict2[key]['valid']:
            x_val = dict1[key][variable1_name]
            y_val = dict2[key][variable2_name]

            if variable1_name == 'peak_sf':
                x_val = x_val + np.random.normal(0, 0.1)
            if variable2_name == 'peak_sf':
                y_val = y_val + np.random.normal(0, 0.1)

            if ax is None:
                line, = plt.plot(x_val, y_val, 'o', color = 'b', alpha=0.4, markeredgecolor='k', markeredgewidth=1)
            else:
                line, = ax.plot(x_val, y_val, 'o', color = 'b', alpha=0.4, markeredgecolor='k', markeredgewidth=1)
            plotted_points.append((line, session_date, session_subject, key))
    if ax is None:
        plt.xlabel(variable1_name)
        plt.ylabel(variable2_name)
        plt.show()
    else:
        ax.set_xlabel(variable1_name)
        ax.set_ylabel(variable2_name)

    
    return ax, plotted_points

def _needs_data_source(variable1_name, variable2_name, variable_set):
    """Check if either variable requires a specific data source"""
    return variable1_name in variable_set or variable2_name in variable_set

def _load_bps_value(subject, date, unit_idx, model_type):
    """
    Load BPS value for a specific cell and model type.
    
    Args:
        subject: Subject name
        date: Date string
        unit_idx: Unit index
        model_type: 'ln' or 'energy'
    
    Returns:
        float or None: BPS value if available, None otherwise
    """
    import json
    from pathlib import Path
    
    model_cache = _bps_cache.setdefault(model_type, {})
    cache_key = (subject, date, unit_idx)
    if cache_key in model_cache:
        return model_cache[cache_key]

    model_dir = Path(f'/mnt/sata/YatesMarmoV1/standard_model_fits/ray_{model_type}_correct_gaborium_format_FIXATION_ONLY_Nov5_30trials_9pointstencil/{subject}_{date}/cell_{unit_idx}_best')
    results_file = model_dir / 'results.json'
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        # Use test_bps as the final BPS metric
        value = results.get('test_bps', None)
        model_cache[cache_key] = value
        return value
    except Exception as e:
        model_cache[cache_key] = None
        return None

def _collect_data_points_for_session(dict1, dict2, variable1_name, variable2_name, session_date, session_subject, qc_units=None):
    """
    Collect data points from a session without plotting.
    
    Returns:
        list: List of (x, y, date, subject, unit_idx) tuples
    """
    intersection = set(dict1.keys()) & set(dict2.keys())
    if qc_units is not None:
        intersection = intersection & set(qc_units)
    data_points = []
    
    for key in intersection:
        if dict1[key]['valid'] and dict2[key]['valid']:
            x_val = dict1[key][variable1_name]
            y_val = dict2[key][variable2_name]

            if variable1_name == 'peak_sf':
                x_val = x_val + np.random.normal(0, 0.1)
            if variable2_name == 'peak_sf':
                y_val = y_val + np.random.normal(0, 0.1)

            data_points.append((x_val, y_val, session_date, session_subject, key))
    
    return data_points

def _collect_values_for_session(dict1, variable_name, session_date, session_subject, qc_units=None):
    """
    Collect values for a single variable from a session.
    
    Returns:
        list: List of (value, date, subject, unit_idx) tuples
    """
    values = []
    keys = set(dict1.keys())
    if qc_units is not None:
        keys = keys & set(qc_units)
    
    for key in keys:
        if dict1[key]['valid']:
            val = dict1[key][variable_name]
            
            if variable_name == 'peak_sf':
                val = val + np.random.normal(0, 0.1)
            
            values.append((val, session_date, session_subject, key))
    
    return values

def _detect_outliers(x_values, y_values, method='mad', threshold=2.5):
    """
    Detect outliers using MADS or z-score method.
    
    Parameters:
        x_values: array-like of x values
        y_values: array-like of y values
        method: 'mad' or 'z-score'
        threshold: threshold for outlier detection (default 2.5)
    
    Returns:
        boolean array: True for outliers, False for normal points
    """
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # Filter out NaN and infinite values
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    outlier_mask = np.zeros(len(x_values), dtype=bool)
    
    if method == 'mad':
        # Median Absolute Deviation method
        x_median = np.median(x_values[valid_mask])
        y_median = np.median(y_values[valid_mask])
        x_mad = np.median(np.abs(x_values[valid_mask] - x_median))
        y_mad = np.median(np.abs(y_values[valid_mask] - y_median))
        
        # Avoid division by zero
        x_mad = x_mad if x_mad > 0 else 1.0
        y_mad = y_mad if y_mad > 0 else 1.0
        
        # Detect outliers in x or y dimension
        x_outliers = np.abs(x_values - x_median) / x_mad > threshold
        y_outliers = np.abs(y_values - y_median) / y_mad > threshold
        outlier_mask = x_outliers | y_outliers
        
    elif method == 'z-score':
        # Z-score method
        x_mean = np.mean(x_values[valid_mask])
        y_mean = np.mean(y_values[valid_mask])
        x_std = np.std(x_values[valid_mask])
        y_std = np.std(y_values[valid_mask])
        
        # Avoid division by zero
        x_std = x_std if x_std > 0 else 1.0
        y_std = y_std if y_std > 0 else 1.0
        
        # Detect outliers in x or y dimension
        x_outliers = np.abs(x_values - x_mean) / x_std > threshold
        y_outliers = np.abs(y_values - y_mean) / y_std > threshold
        outlier_mask = x_outliers | y_outliers
    
    # Mark invalid points as not outliers (they're already filtered)
    outlier_mask[~valid_mask] = False
    
    return outlier_mask

def _load_metrics_for_session(variable1_name, variable2_name, date, subject, cache, use_cached, session_cache):
    """
    Load all required metrics for a session, using cache when available.
    Handles transformed variables like 1/tau_* and 1-alpha_*.
    
    Returns:
        dict: variable_name_to_dict mapping variable names to their data dictionaries
    """
    # Parse variable names to check for transformations
    base_var1_name, transform1_type = _parse_transformed_variable(variable1_name)
    base_var2_name, transform2_type = _parse_transformed_variable(variable2_name)
    
    # Use base variable names for loading
    actual_var1 = base_var1_name
    actual_var2 = base_var2_name
    
    variable_name_to_dict = {}
    
    # Define data source configurations: (variable_set, load_function, cache_keys, extract_function)
    data_sources = [
        # Gratings data source (shared source for modulation_index and peak_sf)
        (
            {'peak_sf', 'modulation_index'},
            lambda: get_gratings_for_dataset(date, subject, cache=cache),
            ['modulation_index', 'peak_sf'],
            lambda data: {
                'modulation_index': data['modulation_index_dict'],
                'peak_sf': get_peak_sf_for_dataset(data)
            }
        ),
        # McFarland data source (shared source for all McFarland variables)
        (
            {'tau_data_10_bins', 'alpha_data_10_bins', 'tau_ln_10_bins', 'alpha_ln_10_bins',
             'tau_energy_10_bins', 'alpha_energy_10_bins', 'tau_data_37_bins', 'alpha_data_37_bins',
             'tau_ln_37_bins', 'alpha_ln_37_bins', 'tau_energy_37_bins', 'alpha_energy_37_bins'},
            lambda: get_mcfarland_analysis_for_dataset(date, subject, cache=cache),
            ['tau_data_10_bins', 'alpha_data_10_bins', 'tau_ln_10_bins', 'alpha_ln_10_bins',
             'tau_energy_10_bins', 'alpha_energy_10_bins', 'tau_data_37_bins', 'alpha_data_37_bins',
             'tau_ln_37_bins', 'alpha_ln_37_bins', 'tau_energy_37_bins', 'alpha_energy_37_bins'],
            lambda data: {var: _extract_mcfarland_data(data, var) for var in [
                'tau_data_10_bins', 'alpha_data_10_bins', 'tau_ln_10_bins', 'alpha_ln_10_bins',
                'tau_energy_10_bins', 'alpha_energy_10_bins', 'tau_data_37_bins', 'alpha_data_37_bins',
                'tau_ln_37_bins', 'alpha_ln_37_bins', 'tau_energy_37_bins', 'alpha_energy_37_bins']}
        ),
        # RF contour metrics (shared source)
        (
            {'area_contour', 'sqrt_area_contour_deg', 'area_contour_deg2', 'snr_value'},
            lambda: get_rf_contour_metrics(date, subject),
            ['area_contour', 'sqrt_area_contour_deg', 'area_contour_deg2', 'snr_value'],
            lambda data: {
                'area_contour': data,
                'sqrt_area_contour_deg': data,
                'area_contour_deg2': data,
                'snr_value': data
            }
        ),
        # RF gaussian metrics (shared source)
        (
            {'area_gaussian', 'sqrt_area_gaussian_deg', 'area_gaussian_deg2', 'length_gaussian', 'snr_value'},
            lambda: get_rf_gaussian_fit_metrics(date, subject, cache=cache),
            ['area_gaussian', 'sqrt_area_gaussian_deg', 'area_gaussian_deg2', 'length_gaussian'],
            lambda data: {
                'area_gaussian': data,
                'sqrt_area_gaussian_deg': data,
                'area_gaussian_deg2': data,
                'length_gaussian': data
            }
        ),
        # Standard model fits - LN (separate source)
        (
            {'ln_bps'},
            lambda: get_ln_standard_model_fits(date, subject, cache=False),
            ['ln_bps'],
            lambda data: {'ln_bps': data}
        ),
        # Standard model fits - Energy (separate source)
        (
            {'energy_bps'},
            lambda: get_energy_standard_model_fits(date, subject, cache=False),
            ['energy_bps'],
            lambda data: {'energy_bps': data}
        ),
        # Standard model fits - Linear index (separate source)
        (
            {'linear_index'},
            lambda: get_linear_index_standard_model_fits(date, subject, cache=False),
            ['linear_index'],
            lambda data: {'linear_index': data}
        ),
    ]
    
    # Process each data source (use base variable names for checking)
    for variable_set, load_func, cache_keys, extract_func in data_sources:
        if _needs_data_source(actual_var1, actual_var2, variable_set):
            # Check which variables from this source are needed
            needed_vars = {actual_var1, actual_var2} & variable_set
            
            # Check if all needed variables are in cache
            all_cached = use_cached and all(var in session_cache for var in needed_vars)
            
            if all_cached:
                # Use cached data for all variables in this set (even if not all are needed)
                for key in cache_keys:
                    if key in session_cache:
                        variable_name_to_dict[key] = session_cache[key]
            else:
                # Load data source (loads entire source even if only one variable needed)
                raw_data = load_func()
                extracted = extract_func(raw_data)
                variable_name_to_dict.update(extracted)
    
    # Apply transformations if needed
    if transform1_type is not None and base_var1_name in variable_name_to_dict:
        transformed_data = _apply_transformation(
            variable_name_to_dict[base_var1_name], 
            base_var1_name, 
            variable1_name, 
            transform1_type
        )
        variable_name_to_dict[variable1_name] = transformed_data
    
    if transform2_type is not None and base_var2_name in variable_name_to_dict:
        transformed_data = _apply_transformation(
            variable_name_to_dict[base_var2_name], 
            base_var2_name, 
            variable2_name, 
            transform2_type
        )
        variable_name_to_dict[variable2_name] = transformed_data
    
    return variable_name_to_dict

def plot_variable1_vs_variable2(variable1_name, variable2_name, session_list, cache = True, interactive = False, cache_all = False, cache_file = '/mnt/ssd/YatesMarmoV1/metrics/all_metrics_cache.pkl', outlier_method = None, outlier_threshold = 2.5, qc = False, points_to_highlight = None, num_points = 10, color_by_modulation_index = False, show_outliers = True, draw_line_of_unity = False, same_x_y_lim = False, x_lim = None, y_lim = None, save_dir = None, only_positive_ln_bps = False, only_positive_energy_bps = False, perform_wilcoxon = False, start_point = None, contamination_threshold = 60, firing_rate_threshold = 2.5, snr_value_threshold = 6):
    # Declare global cache at the start of the function
    global _global_cache
    
    # Validate that old McFarland variable names are not used
    if variable1_name in ['alpha', 'tau'] or variable2_name in ['alpha', 'tau']:
        raise ValueError(f"Variables 'alpha' and 'tau' are no longer supported. Use format: {{metric}}_{{source}}_{{bins}}_bins (e.g., 'tau_data_10_bins', 'alpha_ln_37_bins')")
    
    # Use larger figure width for interactive mode to accommodate unit panel on the right
    if interactive and qc:
        fig, ax = plt.subplots(figsize=(12, 4))  # Wider figure for side-by-side layout
    else:
        fig, ax = plt.subplots()
    all_data_points = []  # Store all (x, y, date, subject, unit_idx) tuples
    all_modulation_indices = []  # Store modulation index for each data point if color_by_modulation_index is True
    
    # Track start_point removal
    start_point_in_data = start_point is not None
    start_point_removed_reason = None
    
    # Use global cache (loaded once at module import time)
    # If cache_file is different from default, load it (but this should be rare)
    cache_path = Path(cache_file)
    if cache_file != _default_cache_file and cache_path.exists():
        # Different cache file specified - load it for this call only
        try:
            with open(cache_path, 'rb') as f:
                all_cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            all_cache = _global_cache.copy() if _global_cache is not None else {}
    else:
        # Use global cache (make a copy so we can modify it)
        if _global_cache is None:
            _load_global_cache(cache_file)
        all_cache = _global_cache.copy() if _global_cache is not None else {}
    
    # Store raw data per session for slider updates (only if interactive and qc)
    session_raw_data = [] if (interactive and qc) else None
    
    # Store unfiltered data points for slider updates (only if interactive and qc)
    all_data_points_unfiltered = [] if (interactive and qc) else None
    all_modulation_indices_unfiltered = [] if (interactive and qc) else None
    
    # First pass: Collect all data points
    original_x_values = []  # Store original x-values before jitter for box plots
    for session in tqdm(session_list):
        date = session.name.split('_')[1]
        subject = session.name.split('_')[0]
        session_key = session.name  # Use session.name as the key

        
        try:
            # Check if we should use cached data
            use_cached = cache_all and session_key in all_cache
            session_cache = all_cache.get(session_key, {}) if use_cached else {}
            
            # Load all required metrics for this session (include modulation_index if needed)
            if color_by_modulation_index:
                variable_name_to_dict = _load_metrics_for_session(
                    variable1_name, 'modulation_index', date, subject, cache, use_cached, session_cache
                )
                # Also load variable2_name if different from modulation_index
                if variable2_name != 'modulation_index':
                    var2_dict = _load_metrics_for_session(
                        variable2_name, variable2_name, date, subject, cache, use_cached, session_cache
                    )
                    variable_name_to_dict.update(var2_dict)
            else:
                variable_name_to_dict = _load_metrics_for_session(
                    variable1_name, variable2_name, date, subject, cache, use_cached, session_cache
                )
            
            # Update all_cache with computed values (only non-None values)
            if session_key not in all_cache:
                all_cache[session_key] = {}
            
            for var_name, var_value in variable_name_to_dict.items():
                if var_value is not None:
                    all_cache[session_key][var_name] = var_value
            
            # Don't preload unit_info_panel_dict here - it's too expensive
            # Instead, load it lazily when a point is clicked (see click handler)
            
            # Store raw data for slider updates
            if session_raw_data is not None:
                session_raw_data.append({
                    'date': date,
                    'subject': subject,
                    'session_key': session_key,
                    'variable_name_to_dict': variable_name_to_dict
                })

            # Get QC units if needed
            qc_units = None
            if qc:
                # When using sliders (interactive mode), always recompute with cache=False
                # Otherwise, check cache first if using default thresholds
                use_qc_cache = cache and not (interactive and qc)
                if use_qc_cache and use_cached and 'qc_units' in session_cache:
                    qc_units = session_cache['qc_units']
                else:
                    qc_units = get_qc_units_for_session(
                        date, subject, 
                        contamination_threshold=contamination_threshold,
                        firing_rate_threshold=firing_rate_threshold,
                        snr_value_threshold=snr_value_threshold,
                        cache=cache  # Always use cache=False when thresholds are provided
                    )
                    # Only save to cache if using default thresholds and not in interactive mode
                    if not interactive and contamination_threshold == 60 and firing_rate_threshold == 2.5 and snr_value_threshold == 6:
                        if session_key not in all_cache:
                            all_cache[session_key] = {}
                        all_cache[session_key]['qc_units'] = qc_units
            
            # Collect data points without plotting
            # If interactive and qc, collect unfiltered points first
            if interactive and qc:
                # Collect all points without QC filtering
                data_points_unfiltered = _collect_data_points_for_session(
                    variable_name_to_dict[variable1_name], 
                    variable_name_to_dict[variable2_name], 
                    variable1_name, variable2_name,
                    session_date=date,
                    session_subject=subject,
                    qc_units=None  # No QC filtering
                )
                all_data_points_unfiltered.extend(data_points_unfiltered)
                
                # Collect modulation indices for unfiltered points
                if color_by_modulation_index:
                    mod_dict = variable_name_to_dict['modulation_index']
                    for x_val, y_val, d, s, unit_idx in data_points_unfiltered:
                        if unit_idx in mod_dict and mod_dict[unit_idx]['valid']:
                            mod_idx = mod_dict[unit_idx]['modulation_index']
                            all_modulation_indices_unfiltered.append(mod_idx)
                        else:
                            all_modulation_indices_unfiltered.append(np.nan)
            
            # Collect data points with QC filtering for initial plot
            data_points = _collect_data_points_for_session(
                variable_name_to_dict[variable1_name], 
                variable_name_to_dict[variable2_name], 
                variable1_name, variable2_name,
                session_date=date,
                session_subject=subject,
                qc_units=qc_units
            )
            
            # Collect modulation indices if needed
            if color_by_modulation_index:
                mod_dict = variable_name_to_dict['modulation_index']
                for x_val, y_val, d, s, unit_idx in data_points:
                    if unit_idx in mod_dict and mod_dict[unit_idx]['valid']:
                        mod_idx = mod_dict[unit_idx]['modulation_index']
                        all_modulation_indices.append(mod_idx)
                    else:
                        all_modulation_indices.append(np.nan)
            
            # Store original x-values before jitter for box plots
            if variable1_name == 'peak_sf':
                dict1 = variable_name_to_dict[variable1_name]
                dict2 = variable_name_to_dict[variable2_name]
                intersection = set(dict1.keys()) & set(dict2.keys())
                if qc_units is not None:
                    intersection = intersection & set(qc_units)
                for key in intersection:
                    if dict1[key]['valid'] and dict2[key]['valid']:
                        original_x_values.append(dict1[key][variable1_name])
            
            all_data_points.extend(data_points)
        except Exception as e:
            print(f"Error processing session {session.name}: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    # Check if start_point is in the data after initial collection
    if start_point_in_data and start_point_removed_reason is None:
        start_date, start_subject, start_unit_idx = start_point
        found = any(date == start_date and subject == start_subject and unit_idx == start_unit_idx 
                   for _, _, date, subject, unit_idx in all_data_points)
        if not found:
            start_point_in_data = False
            if qc:
                start_point_removed_reason = f"QC filtering (contamination, firing rate, or SNR threshold), {qc_units}"
            else:
                start_point_removed_reason = "not found in any session data (missing metrics or session not in session_list)"
            print(f"Start point {start_point} was removed during: {start_point_removed_reason}")
    
    # Filter by positive BPS if requested
    if only_positive_ln_bps or only_positive_energy_bps:
        filtered_data_points = []
        filtered_modulation_indices = []
        
        for i, (x_val, y_val, date, subject, unit_idx) in enumerate(all_data_points):
            include = True
            
            if only_positive_ln_bps:
                ln_bps = _load_bps_value(subject, date, unit_idx, 'ln')
                if ln_bps is None or ln_bps <= 0:
                    include = False
            
            if only_positive_energy_bps and include:
                energy_bps = _load_bps_value(subject, date, unit_idx, 'energy')
                if energy_bps is None or energy_bps <= 0:
                    include = False
            
            if include:
                filtered_data_points.append((x_val, y_val, date, subject, unit_idx))
                # Also filter modulation indices if they were collected
                if color_by_modulation_index and i < len(all_modulation_indices):
                    filtered_modulation_indices.append(all_modulation_indices[i])
        
        all_data_points = filtered_data_points
        if color_by_modulation_index:
            all_modulation_indices = filtered_modulation_indices
        print(f"Filtered to {len(all_data_points)} points with positive BPS")
        
        # Check if start_point was removed during BPS filtering
        if start_point_in_data and start_point_removed_reason is None:
            start_date, start_subject, start_unit_idx = start_point
            found = any(date == start_date and subject == start_subject and unit_idx == start_unit_idx 
                       for _, _, date, subject, unit_idx in all_data_points)
            if not found:
                start_point_in_data = False
                filter_type = []
                if only_positive_ln_bps:
                    filter_type.append("LN BPS")
                if only_positive_energy_bps:
                    filter_type.append("Energy BPS")
                start_point_removed_reason = f"BPS filtering ({' and '.join(filter_type)} <= 0)"
                print(f"Start point {start_point} was removed during: {start_point_removed_reason}")
    
    # Extract x and y values for correlation calculation
    all_x_values = [x for x, _, _, _, _ in all_data_points]
    all_y_values = [y for _, y, _, _, _ in all_data_points]
    
    # Calculate correlation and p-value
    all_x_values = np.array(all_x_values)
    all_y_values = np.array(all_y_values)
    
    # Filter out NaN and infinite values
    valid_mask = np.isfinite(all_x_values) & np.isfinite(all_y_values)
    # assert valid_mask.sum() == len(all_data_points), 'Number of valid values does not match number of all data points'
    
    # Check if start_point was removed due to NaN or infinite values
    if start_point_in_data and start_point_removed_reason is None:
        start_date, start_subject, start_unit_idx = start_point
        for i, (x_val, y_val, date, subject, unit_idx) in enumerate(all_data_points):
            if date == start_date and subject == start_subject and unit_idx == start_unit_idx:
                if not valid_mask[i]:
                    start_point_in_data = False
                    start_point_removed_reason = "NaN or infinite values in x or y coordinates"
                    print(f"Start point {start_point} was removed during: {start_point_removed_reason}")
                break
    
    # Detect outliers if method is specified
    is_outlier = None
    if outlier_method is not None:
        is_outlier = _detect_outliers(all_x_values, all_y_values, method=outlier_method, threshold=outlier_threshold)
    
    # Exclude outliers if outlier detection is enabled
    if is_outlier is not None:
        valid_mask = valid_mask & ~is_outlier
        
        # Check if start_point was removed due to outlier detection
        if start_point_in_data and start_point_removed_reason is None:
            start_date, start_subject, start_unit_idx = start_point
            for i, (x_val, y_val, date, subject, unit_idx) in enumerate(all_data_points):
                if date == start_date and subject == start_subject and unit_idx == start_unit_idx:
                    if is_outlier[i]:
                        start_point_in_data = False
                        start_point_removed_reason = f"Outlier detection (method={outlier_method}, threshold={outlier_threshold})"
                        print(f"Start point {start_point} was removed during: {start_point_removed_reason}")
                    break
    
    # Set up colormap if coloring by modulation index
    cmap = None
    norm = None
    if color_by_modulation_index:
        cmap = cm.get_cmap('coolwarm')
        norm = Normalize(vmin=0, vmax=1)  # Clamp to [0, 1]
        all_modulation_indices = np.array(all_modulation_indices)
        # Clip values to [0, 1] range
        all_modulation_indices_clipped = np.clip(all_modulation_indices, 0, 1)
    
    # Plot all points first
    all_points = []  # Store all (line, date, subject, unit_idx) tuples for interactive hover
    all_points_qc_valid_inliers = []
    for i, (x_val, y_val, date, subject, unit_idx) in enumerate(all_data_points):
        # Skip outliers if show_outliers is False
        if is_outlier is not None and is_outlier[i] and not show_outliers:
            continue
        
        if color_by_modulation_index:
            # Color by modulation index if available and valid
            if np.isfinite(all_modulation_indices_clipped[i]):
                color = cmap(norm(all_modulation_indices_clipped[i]))
            else:
                color = 'gray'  # Use gray for missing modulation index
        else:
            color = 'b'  # Always blue for normal points
        
        line, = ax.plot(x_val, y_val, 'o', color=color, alpha=0.5, markeredgecolor='k', markeredgewidth=0.5, zorder=3)
        all_points.append((line, date, subject, unit_idx))
        if valid_mask[i]:
            all_points_qc_valid_inliers.append((x_val, y_val, date, subject, unit_idx))
    
    # Mark outliers with red X if outlier detection is enabled and show_outliers is True
    if is_outlier is not None and show_outliers:
        outlier_x = [all_data_points[i][0] for i in range(len(all_data_points)) if is_outlier[i]]
        outlier_y = [all_data_points[i][1] for i in range(len(all_data_points)) if is_outlier[i]]
        ax.plot(outlier_x, outlier_y, 'rx', markersize=10, markeredgewidth=2, zorder=4)

    # Mark start point if specified
    start_point_found = False
    if start_point is not None:
        start_date, start_subject, start_unit_idx = start_point
        # Check if start_point exists in the plotted data
        for i, (x_val, y_val, date, subject, unit_idx) in enumerate(all_data_points):
            if date == start_date and subject == start_subject and unit_idx == start_unit_idx:
                # Check if this point would be hidden (outlier not shown)
                if is_outlier is not None and is_outlier[i] and not show_outliers:
                    if start_point_removed_reason is None:
                        start_point_removed_reason = f"Outlier detection (method={outlier_method}, threshold={outlier_threshold}) and show_outliers=False"
                        print(f"Start point {start_point} was removed during: {start_point_removed_reason}")
                    continue
                # Plot with star marker
                ax.plot(x_val, y_val, '*', color='gold', markersize=15, markeredgecolor='k', markeredgewidth=1.5, zorder=6)
                start_point_found = True
                if start_point_removed_reason is None:
                    print(f"Start point {start_point} successfully included in plot")
                break
    
    # Set axis labels
    ax.set_xlabel(variable1_name)
    ax.set_ylabel(variable2_name)
    
    # Add colorbar if coloring by modulation index
    initial_colorbar = None
    if color_by_modulation_index:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Position colorbar outside the plot, to the right of the scatter plot
        # Use pad to add space between plot and colorbar
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Modulation Index', rotation=270, labelpad=20)
        initial_colorbar = cbar
    
    # Add box plots for peak_sf on x-axis using twin y-axis
    if variable1_name == 'peak_sf' and len(original_x_values) > 0:
        # Check if BPS filtering was applied - if so, box plots are skipped due to index mismatch
        if (only_positive_ln_bps or only_positive_energy_bps) and len(original_x_values) != len(all_data_points):
            print(f"Warning: Skipping box plots for peak_sf due to BPS filtering (length mismatch)")
        else:
            # Map jittered indices to original x-values and round to avoid floating point issues
            original_x_array = np.array(original_x_values)
            original_x_array = np.round(original_x_array, decimals=1)  # Round to 1 decimal place
            original_x_array = original_x_array[valid_mask]
            y_valid = all_y_values[valid_mask]
            
            # Get unique original spatial frequencies
            unique_sf = np.unique(original_x_array)
            
            # Group y-values by original spatial frequency
            box_data = []
            valid_sf = []
            for sf in unique_sf:
                y_group = y_valid[np.isclose(original_x_array, sf, atol=0.01)]  # Use tolerance for grouping
                if len(y_group) > 0:
                    box_data.append(y_group)
                    valid_sf.append(sf)
            valid_sf = np.array(valid_sf)
            
            if len(box_data) > 0:
                # Calculate box width - make them VERY wide with absolute width
                # Use a large fixed width that's visible
                box_width = 1.5  # Fixed large width in x-axis units (about half the spacing)
                
                # Plot box plots directly on main axis - behind scatter points
                bp = ax.boxplot(box_data, positions=valid_sf, widths=box_width, 
                               patch_artist=True, showfliers=False, zorder=1,
                               boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                               medianprops=dict(color='red', linewidth=2, linestyle='-', solid_capstyle='butt'),
                               whiskerprops=dict(color='black', linewidth=1.5, linestyle='-'),
                               capprops=dict(color='black', linewidth=1.5),
                               meanline=False, showmeans=False)
                
                # Round x-axis ticks for peak_sf
                ax.set_xticks(valid_sf)
                ax.set_xticklabels([f'{sf:.1f}' for sf in valid_sf])
    
    # Highlight specific points if requested
    highlighted_points = {}
    legend_handles = []
    mode_colors = {'closest_to_origin': 'green', 'furthest_from_origin': 'purple', 
                   'further_from_y_axis': 'orange', 'further_from_x_axis': 'cyan',
                   'closest_to_x_axis': 'magenta', 'closest_to_y_axis': 'brown',
                   'closest_to_median': 'gold'}
    
    if points_to_highlight is not None:
        for mode in points_to_highlight:
            if mode == 'closest_to_origin':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: x[0]**2 + x[1]**2)[:num_points]
            elif mode == 'furthest_from_origin':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: x[0]**2 + x[1]**2, reverse=True)[:num_points]
            elif mode == 'further_from_y_axis':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: x[0], reverse=True)[:num_points]
            elif mode == 'further_from_x_axis':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: x[1], reverse=True)[:num_points]
            elif mode == 'closest_to_x_axis':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: abs(x[1]))[:num_points]
            elif mode == 'closest_to_y_axis':
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: abs(x[0]))[:num_points]
            elif mode == 'closest_to_median':
                median_x = np.median([p[0] for p in all_points_qc_valid_inliers])
                median_y = np.median([p[1] for p in all_points_qc_valid_inliers])
                selected = sorted(all_points_qc_valid_inliers, key=lambda x: (x[0]-median_x)**2 + (x[1]-median_y)**2)[:num_points]
            else:
                raise ValueError(f'Invalid mode: {mode}')
            
            highlighted_points[mode] = selected
            
            # Plot highlighted points on top with distinct color
            color = mode_colors[mode]
            for x_val, y_val, date, subject, unit_idx in selected:
                ax.plot(x_val, y_val, 'o', color=color, alpha=0.8, markeredgecolor='k', 
                       markeredgewidth=1.5, markersize=10, zorder=5)
            
            # Add to legend
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=8, label=mode.replace('_', ' ').title()))
    
    # Add outlier to legend if outlier detection is enabled and show_outliers is True
    if outlier_method is not None and show_outliers:
        legend_handles.append(plt.Line2D([0], [0], marker='x', color='r', linestyle='None',
                                        markersize=8, markeredgewidth=2, label='Outlier'))
    
    # Add start point to legend if found
    if start_point is not None and start_point_found:
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='gold', linestyle='None',
                                        markersize=10, markeredgecolor='k', markeredgewidth=1.5, label='Start point'))
    
    # Add legend if there are highlighted points, outliers, or start point
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best')
    
    n_units = np.sum(valid_mask)
    if n_units > 1:
        x_valid = all_x_values[valid_mask]
        y_valid = all_y_values[valid_mask]
        corr, p_value = pearsonr(x_valid, y_valid)
        qc_str = ' (QC)' if qc else ''
        
        # Perform Wilcoxon signed-rank test if requested
        wilcoxon_p_value = None
        if perform_wilcoxon:
            try:
                # Wilcoxon test requires paired differences
                # It tests whether the median of differences is zero
                # Compute differences: x - y
                differences = x_valid - y_valid
                # Remove zeros (wilcoxon handles this, but we can be explicit)
                non_zero_diffs = differences[differences != 0]
                if len(non_zero_diffs) > 0:
                    stat, wilcoxon_p_value = wilcoxon(x_valid, y_valid, alternative='two-sided')
                    print(f"Debug: Wilcoxon test successful. Statistic={stat:.3f}, p-value={wilcoxon_p_value:.2e}")
                else:
                    print("Warning: All differences are zero, cannot perform Wilcoxon test")
                    wilcoxon_p_value = None
            except Exception as e:
                print(f"Warning: Wilcoxon test failed: {e}")
                import traceback
                traceback.print_exc()
                wilcoxon_p_value = None
        
        # Build title with appropriate statistics
        if perform_wilcoxon:
            if wilcoxon_p_value is not None:
                title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_units}, r={corr:.3f}, p={p_value:.2e}, Wilcoxon p={wilcoxon_p_value:.2e}'
            else:
                title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_units}, r={corr:.3f}, p={p_value:.2e} (Wilcoxon test failed)'
        else:
            title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_units}, r={corr:.3f}, p={p_value:.2e}'
        
        # Add start point status to title
        if start_point is not None:
            if start_point_found:
                title += ' (star point shown)'
            else:
                title += ' (star point was not found!)'
    else:
        qc_str = ' (QC)' if qc else ''
        title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_units} (insufficient data for correlation)'
        # Add start point status to title
        if start_point is not None:
            if start_point_found:
                title += ' (star point shown)'
            else:
                title += ' (star point was not found!)'
    
    # Set title (for both interactive and non-interactive)
    print("Setting title...")
    ax.set_title(title)
    print("Title set")
    
    # Apply manual x and y limits if provided
    print("Applying axis limits...")
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    
    # Apply same x and y limits if requested
    if same_x_y_lim:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Get the overall min and max across both axes
        overall_min = min(xlim[0], ylim[0])
        overall_max = max(xlim[1], ylim[1])
        ax.set_xlim(overall_min, overall_max)
        ax.set_ylim(overall_min, overall_max)
    
    # Draw line of unity (y=x) if requested
    print("Checking for line of unity...")
    if draw_line_of_unity:
        print("Drawing line of unity...")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Draw line from the minimum to maximum of the common range
        min_val = max(xlim[0], ylim[0])
        max_val = min(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, zorder=2, label='Unity')
        # Update legend to include unity line if legend exists
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc='best')
        print("Line of unity drawn")
    
    print("Reached cache save section...")
    # Update global cache and save to pickle file
    # Do this AFTER displaying the figure for interactive mode to avoid blocking
    # Update global cache with any new data from this function call
    if cache_file == _default_cache_file:
        # Update the global cache with new data
        if _global_cache is None:
            _global_cache = {}
        # Merge all_cache into global cache (all_cache may have new keys/values)
        for session_key, session_data in all_cache.items():
            if session_key not in _global_cache:
                _global_cache[session_key] = {}
            _global_cache[session_key].update(session_data)
    
    # Save cache to file (use global cache if using default file, otherwise use local)
    cache_to_save = _global_cache if (cache_file == _default_cache_file and _global_cache is not None) else all_cache
    if not interactive:
        print("Saving cache to pickle file...")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_to_save, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")
    else:
        # For interactive mode, save cache in background or skip for now
        # (can be saved later when the session ends)
        print("Skipping cache save in interactive mode (will save on exit)")
    
    if not interactive:
        # Save figure as PDF if save_dir is provided (non-interactive mode)
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            # Sanitize variable names for filename (replace / with _over_)
            var1_safe = variable1_name.replace('/', '_over_')
            var2_safe = variable2_name.replace('/', '_over_')
            pdf_filename = f'{var1_safe}vs{var2_safe}.pdf'
            fig.savefig(save_path / pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path / pdf_filename}")
        plt.show()
        return fig, all_points_qc_valid_inliers, highlighted_points

    # Initialize variables for interactive mode
    ax_unit_panel = None
    unit_panel_fig = None

    # QC sliders for interactive filtering
    if interactive and qc and session_raw_data is not None:
        # Adjust figure to make room for sliders and unit panel on the right
        # With the larger figure size, we can give proper space to everything
        # Use a cleaner layout approach - set up the figure margins first
        plt.subplots_adjust(bottom=0.25, top=0.90, right=0.50, left=0.10)  # Leave room for title, sliders, and unit panel
        
        # Get the current axes position and adjust it
        pos = ax.get_position()
        
        # Position main plot on the left side - leave room for colorbar on the right
        # The plot should have enough space at the bottom for x-axis label
        plot_bottom = 0.30  # Bottom of plot area (accounts for x-axis label space)
        plot_top = 0.85     # Top of plot area (accounts for title space)
        plot_left = 0.10
        plot_width = 0.35   # Width of plot (leaves room for colorbar)
        plot_height = plot_top - plot_bottom
        
        ax.set_position([plot_left, plot_bottom, plot_width, plot_height])
        
        # Reposition colorbar if it exists (it was created before layout adjustments)
        if initial_colorbar is not None:
            # Remove old colorbar and recreate it with proper positioning
            initial_colorbar.remove()
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # Position colorbar to the right of the main plot with padding
            cbar = plt.colorbar(sm, ax=ax, pad=0.05)
            cbar.set_label('Modulation Index', rotation=270, labelpad=20)
            initial_colorbar = cbar
        
        # Create three QC sliders (positioned below main plot, on the left side)
        slider_bottom = 0.02  # Keep sliders inside figure canvas
        slider_height = 0.03
        slider_spacing = 0.04
        slider_width = 0.32
        slider_left = 0.10
        ax_contamination = plt.axes([slider_left, slider_bottom + 2*slider_spacing, slider_width, slider_height])
        ax_firing_rate = plt.axes([slider_left, slider_bottom + slider_spacing, slider_width, slider_height])
        ax_snr = plt.axes([slider_left, slider_bottom, slider_width, slider_height])
        
        # Create axes for unit panel (on the right side of the chart)
        panel_left = plot_left + plot_width + 0.05
        panel_width = min(0.95 - panel_left, 0.40)
        # ax_unit_panel = plt.axes([panel_left, plot_bottom, panel_width, plot_height])
        ax_unit_panel = plt.axes([0.25, 0.1, 0.9, 0.9])
        ax_unit_panel.set_navigate(False)
        ax_unit_panel.axis('off')
        ax_unit_panel.text(0.5, 0.5, 'Click on a point to view unit panel', 
                          ha='center', va='center', transform=ax_unit_panel.transAxes,
                          fontsize=10, style='italic', color='gray')
        # Keep sliders above the unit panel so they continue receiving events
        for slider_ax in (ax_contamination, ax_firing_rate, ax_snr):
            slider_ax.set_zorder(5)
        ax_unit_panel.set_zorder(-1)
        slider_contamination = Slider(ax_contamination, 'Contamination', 0, 100, valinit=contamination_threshold, valstep=1)
        slider_firing_rate = Slider(ax_firing_rate, 'Firing Rate', 0, 10, valinit=firing_rate_threshold, valstep=0.1)
        slider_snr = Slider(ax_snr, 'SNR', 0, 10, valinit=snr_value_threshold, valstep=0.1)
        
        # Store unfiltered points for QC filtering (MUST use unfiltered so lowering thresholds shows more points)
        if all_data_points_unfiltered is None or len(all_data_points_unfiltered) == 0:
            raise ValueError("all_data_points_unfiltered must be populated for interactive QC sliders to work correctly")
        initial_plotted_points = all_data_points_unfiltered.copy()
        initial_modulation_indices = all_modulation_indices_unfiltered.copy() if (all_modulation_indices_unfiltered is not None and color_by_modulation_index) else None
        
        # Pre-compute QC information for all sessions so slider updates do not need to hit disk
        session_to_qc_information = {}
        print("Pre-loading QC information for interactive sliders...")
        for session_data in tqdm(session_raw_data, desc="QC info"):
            date = session_data['date']
            subject = session_data['subject']
            session_key = f"{subject}_{date}"
            if session_key in session_to_qc_information:
                continue
            try:
                information_for_qc = get_information_for_qc(date, subject, cache=True)
                session_to_qc_information[session_key] = information_for_qc
            except Exception as e:
                print(f"Warning: Could not load QC information for {session_key}: {e}")
                session_to_qc_information[session_key] = None
        print("Finished loading QC information")
        
        # Store colorbar reference to avoid duplicates (use initial colorbar if it exists)
        colorbar_ref = initial_colorbar if color_by_modulation_index else None
        
        # Store reference to scatter plot for updating
        scatter_plot_ref = None
        
        is_updating = False
        
        def _perform_update():
            """Update: filter points based on QC thresholds"""
            nonlocal is_updating, all_points, scatter_plot_ref, colorbar_ref
            
            if is_updating:
                return  # Skip if already updating
            
            is_updating = True
            try:
                # Get QC thresholds from sliders
                new_contamination = slider_contamination.val
                new_firing_rate = slider_firing_rate.val
                new_snr = slider_snr.val
                
                # Recompute QC units for each session using pre-loaded information (fast!)
                session_qc_units = {}
                for session_data in session_raw_data:
                    date = session_data['date']
                    subject = session_data['subject']
                    session_key = f"{subject}_{date}"
                    information_dict = session_to_qc_information.get(session_key)
                    if information_dict is not None:
                        qc_units = get_qc_units_from_information_dict(information_dict, contamination_threshold=new_contamination, firing_rate_threshold=new_firing_rate, snr_value_threshold=new_snr)
                        session_qc_units[session_key] = set(qc_units) if qc_units is not None else None
                    else:
                        session_qc_units[session_key] = None
                
                # Filter points based on QC (using unfiltered points, so lowering thresholds shows MORE points)
                sampled_points = []
                sampled_mod_indices = []
                for i, (x_val, y_val, date, subject, unit_idx) in enumerate(initial_plotted_points):
                    session_key = f"{subject}_{date}"
                    qc_units_set = session_qc_units.get(session_key)
                    # Include point if it passes QC (or if QC is None for this session)
                    # Lower thresholds = more units pass QC = more points shown
                    if qc_units_set is None or unit_idx in qc_units_set:
                        sampled_points.append((x_val, y_val, date, subject, unit_idx))
                        if initial_modulation_indices is not None and i < len(initial_modulation_indices):
                            sampled_mod_indices.append(initial_modulation_indices[i])
                
                # Apply BPS filtering if requested (same as initial plot)
                if only_positive_ln_bps or only_positive_energy_bps:
                    filtered_sampled_points = []
                    filtered_sampled_mod_indices = []
                    
                    for i, (x_val, y_val, date, subject, unit_idx) in enumerate(sampled_points):
                        include = True
                        
                        if only_positive_ln_bps:
                            ln_bps = _load_bps_value(subject, date, unit_idx, 'ln')
                            if ln_bps is None or ln_bps <= 0:
                                include = False
                        
                        if only_positive_energy_bps and include:
                            energy_bps = _load_bps_value(subject, date, unit_idx, 'energy')
                            if energy_bps is None or energy_bps <= 0:
                                include = False
                        
                        if include:
                            filtered_sampled_points.append((x_val, y_val, date, subject, unit_idx))
                            if sampled_mod_indices is not None and i < len(sampled_mod_indices):
                                filtered_sampled_mod_indices.append(sampled_mod_indices[i])
                    
                    sampled_points = filtered_sampled_points
                    sampled_mod_indices = filtered_sampled_mod_indices
                
                # Clear the axis - should be safe now that we're on the main thread
                # Note: colorbar should persist since it's attached to the figure, not the axes
                ax.clear()
                
                if len(sampled_points) == 0:
                    ax.set_xlabel(variable1_name)
                    ax.set_ylabel(variable2_name)
                    ax.set_title(f'{variable1_name} vs {variable2_name} (QC)\nN=0 (no data)')
                    all_points = []
                    fig.canvas.draw_idle()
                    return
                
                # Extract x and y arrays
                x_plot = np.array([p[0] for p in sampled_points])
                y_plot = np.array([p[1] for p in sampled_points])
                
                # Detect outliers if method is specified (same as initial plot)
                is_outlier = None
                if outlier_method is not None:
                    is_outlier = _detect_outliers(x_plot, y_plot, method=outlier_method, threshold=outlier_threshold)
                
                # Calculate valid mask (excluding outliers)
                valid_mask = np.isfinite(x_plot) & np.isfinite(y_plot)
                if is_outlier is not None:
                    valid_mask = valid_mask & ~is_outlier
                
                # Prepare colors
                if color_by_modulation_index and sampled_mod_indices is not None:
                    cmap = cm.get_cmap('coolwarm')
                    norm = Normalize(vmin=0, vmax=1)
                    mod_indices_array = np.array(sampled_mod_indices)
                    mod_indices_clipped = np.clip(mod_indices_array, 0, 1)
                    colors = cmap(norm(mod_indices_clipped))
                    invalid_mask = ~np.isfinite(mod_indices_clipped)
                    colors[invalid_mask] = [0.5, 0.5, 0.5, 0.5]
                else:
                    colors = 'b'
                
                # Plot points (skip outliers if show_outliers is False)
                points_to_plot = []
                mod_indices_to_plot = []
                colors_to_plot = []
                for i, (x_val, y_val, date, subject, unit_idx) in enumerate(sampled_points):
                    # Skip outliers if show_outliers is False
                    if is_outlier is not None and is_outlier[i] and not show_outliers:
                        continue
                    points_to_plot.append((x_val, y_val, date, subject, unit_idx))
                    if color_by_modulation_index and sampled_mod_indices is not None and i < len(sampled_mod_indices):
                        mod_indices_to_plot.append(sampled_mod_indices[i])
                        if isinstance(colors, np.ndarray):
                            colors_to_plot.append(colors[i])
                        else:
                            colors_to_plot.append(colors)
                
                if len(points_to_plot) > 0:
                    x_plot_filtered = np.array([p[0] for p in points_to_plot])
                    y_plot_filtered = np.array([p[1] for p in points_to_plot])
                    
                    if color_by_modulation_index and len(colors_to_plot) > 0:
                        colors_array = np.array(colors_to_plot)
                        scatter = ax.scatter(x_plot_filtered, y_plot_filtered, c=colors_array, alpha=0.5, edgecolors='k', linewidths=0.5, s=20, zorder=3)
                    else:
                        scatter = ax.scatter(x_plot_filtered, y_plot_filtered, c='b', alpha=0.5, edgecolors='k', linewidths=0.5, s=20, zorder=3)
                    scatter_plot_ref = scatter  # Store reference for next update
                    
                    # Store for hover
                    all_points = [(scatter, x, y, date, subject, unit_idx) 
                                 for (x, y, date, subject, unit_idx) in points_to_plot]
                else:
                    all_points = []
                
                # Mark outliers with red X if outlier detection is enabled and show_outliers is True
                if is_outlier is not None and show_outliers:
                    outlier_x = [sampled_points[i][0] for i in range(len(sampled_points)) if is_outlier[i]]
                    outlier_y = [sampled_points[i][1] for i in range(len(sampled_points)) if is_outlier[i]]
                    if len(outlier_x) > 0:
                        ax.plot(outlier_x, outlier_y, 'rx', markersize=10, markeredgewidth=2, zorder=4)
                
                # Mark start point if specified
                start_point_found = False
                if start_point is not None:
                    start_date, start_subject, start_unit_idx = start_point
                    for i, (x_val, y_val, date, subject, unit_idx) in enumerate(sampled_points):
                        if date == start_date and subject == start_subject and unit_idx == start_unit_idx:
                            # Skip if this point is an outlier and show_outliers is False
                            if is_outlier is not None and is_outlier[i] and not show_outliers:
                                continue
                            ax.plot(x_val, y_val, '*', color='gold', markersize=15, markeredgecolor='k', markeredgewidth=1.5, zorder=6)
                            start_point_found = True
                            break
                
                # Set labels and title
                ax.set_xlabel(variable1_name)
                ax.set_ylabel(variable2_name)
                
                # Calculate correlation and Wilcoxon test
                n_valid = np.sum(valid_mask)
                
                if n_valid > 1:
                    x_valid = x_plot[valid_mask]
                    y_valid = y_plot[valid_mask]
                    corr, p_value = pearsonr(x_valid, y_valid)
                    
                    # Perform Wilcoxon signed-rank test if requested
                    wilcoxon_p_value = None
                    if perform_wilcoxon:
                        try:
                            differences = x_valid - y_valid
                            non_zero_diffs = differences[differences != 0]
                            if len(non_zero_diffs) > 0:
                                stat, wilcoxon_p_value = wilcoxon(x_valid, y_valid, alternative='two-sided')
                            else:
                                wilcoxon_p_value = None
                        except Exception as e:
                            wilcoxon_p_value = None
                    
                    # Build title with appropriate statistics
                    qc_str = ' (QC)'
                    if perform_wilcoxon:
                        if wilcoxon_p_value is not None:
                            title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_valid}, r={corr:.3f}, p={p_value:.2e}, Wilcoxon p={wilcoxon_p_value:.2e}'
                        else:
                            title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_valid}, r={corr:.3f}, p={p_value:.2e} (Wilcoxon test failed)'
                    else:
                        title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_valid}, r={corr:.3f}, p={p_value:.2e}'
                    
                    # Add start point status to title
                    if start_point is not None:
                        if start_point_found:
                            title += ' (star point shown)'
                        else:
                            title += ' (star point was not found!)'
                else:
                    qc_str = ' (QC)'
                    title = f'{variable1_name} vs {variable2_name}{qc_str}\nN={n_valid} (insufficient data for correlation)'
                    if start_point is not None:
                        if start_point_found:
                            title += ' (star point shown)'
                        else:
                            title += ' (star point was not found!)'
                
                ax.set_title(title)
                
                # Apply limits
                if x_lim is not None:
                    ax.set_xlim(x_lim)
                if y_lim is not None:
                    ax.set_ylim(y_lim)
                
                if same_x_y_lim:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    overall_min = min(xlim[0], ylim[0])
                    overall_max = max(xlim[1], ylim[1])
                    ax.set_xlim(overall_min, overall_max)
                    ax.set_ylim(overall_min, overall_max)
                
                if draw_line_of_unity:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    min_val = max(xlim[0], ylim[0])
                    max_val = min(xlim[1], ylim[1])
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, zorder=2)
                
                # Colorbar should persist from initial plot - don't touch it
                # (it's attached to the figure, not the axes, so ax.clear() shouldn't affect it)
                
                fig.canvas.draw_idle()
            finally:
                is_updating = False
        
        def update_plot(val):
            """Immediate update for slider changes."""
            _perform_update()
        
        # Connect all sliders to update function
        slider_contamination.on_changed(update_plot)
        slider_firing_rate.on_changed(update_plot)
        slider_snr.on_changed(update_plot)
    
    # Set up hover tooltip functionality
    # We'll create the annotation dynamically to allow changing arrow direction
    annot = None
    
    # Also ensure figure doesn't clip
    fig.patch.set_clip_on(False)
    
    def hover(event):
        """Handle hover events"""
        nonlocal annot
        if event.inaxes == ax:
            found = False
            # Check if we're using scatter plots (new format) or individual lines (old format)
            if len(all_points) > 0 and isinstance(all_points[0][0], PathCollection):
                # New format: scatter plot with (scatter, x, y, date, subject, unit_idx)
                # Check distance from mouse to each point
                min_dist = float('inf')
                closest_point = None
                for scatter, x_val, y_val, date, subject, unit_idx in all_points:
                    # Calculate distance in data coordinates
                    dx = event.xdata - x_val if event.xdata is not None else float('inf')
                    dy = event.ydata - y_val if event.ydata is not None else float('inf')
                    dist = np.sqrt(dx**2 + dy**2)
                    # Convert to display coordinates for threshold (roughly 5 pixels)
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1
                    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1
                    # Approximate pixel distance (assuming ~100 DPI, adjust as needed)
                    pixel_threshold = 0.05 * min(x_range, y_range)  # Roughly 5% of axis range
                    if dist < pixel_threshold and dist < min_dist:
                        min_dist = dist
                        closest_point = (x_val, y_val, date, subject, unit_idx)
                
                if closest_point is not None:
                    x_val, y_val, date, subject, unit_idx = closest_point
                    found = True
            else:
                # Old format: individual line objects
                for line, date, subject, unit_idx in all_points:
                    cont, ind = line.contains(event)
                    if cont:
                        # Get the point coordinates
                        x, y = line.get_data()
                        x_val, y_val = x[0], y[0]
                        found = True
                        break
            
            if found:
                # Get plot boundaries
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0]
                
                # Calculate position as fraction of plot range
                x_fraction = (x_val - xlim[0]) / x_range if x_range > 0 else 0.5
                y_fraction = (y_val - ylim[0]) / y_range if y_range > 0 else 0.5
                
                # Determine if point is on right or top edge
                on_right = x_fraction > 0.5
                on_top = y_fraction > 0.5
                
                # Remove old annotation if it exists
                if annot is not None:
                    annot.remove()
                
                # Position tooltip and set arrow direction
                if on_right or on_top:
                    # Point is on right or top - position tooltip to the left and below
                    # Use large offsets to position tooltip in safe area
                    x_offset = -250 if on_right else 20
                    y_offset = -120 if on_top else 20
                    # Create annotation with reversed arrow
                    annot = ax.annotate("", xy=(x_val, y_val), xytext=(x_offset, y_offset),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                      arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.3"),
                                      clip_on=False)
                else:
                    # Point is on left or bottom - use normal offset positioning
                    annot = ax.annotate("", xy=(x_val, y_val), xytext=(20, 20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                      arrowprops=dict(arrowstyle="->"),
                                      clip_on=False)
                
                text = f"Date: {date}\nSubject: {subject}\nUnit Index: {unit_idx}"
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.8)
                
                # Ensure annotation is not clipped
                annot.set_clip_on(False)
                if annot.get_bbox_patch():
                    annot.get_bbox_patch().set_clip_on(False)
            
            # Hide annotation if not hovering over any point
            if not found and annot is not None:
                annot.remove()
                annot = None
            
            if found or (not found and annot is None):
                fig.canvas.draw_idle()
    
    # Connect hover event handler
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # Set up click handler for unit panel display (only in interactive mode)
    if interactive and qc and ax_unit_panel is not None:
        def on_click(event):
            """Handle click events to show unit panel"""
            nonlocal unit_panel_fig, cache_file
            global _global_cache
            
            # Only respond to clicks on the main plot axes
            if event.inaxes != ax or event.button != 1:  # Left mouse button only
                return
            
            # Find the closest point to the click
            if len(all_points) == 0:
                return
            
            min_dist = float('inf')
            clicked_point = None
            
            # Check if we're using scatter plots (new format) or individual lines (old format)
            if isinstance(all_points[0][0], PathCollection):
                # New format: scatter plot with (scatter, x, y, date, subject, unit_idx)
                for scatter, x_val, y_val, date, subject, unit_idx in all_points:
                    dx = event.xdata - x_val if event.xdata is not None else float('inf')
                    dy = event.ydata - y_val if event.ydata is not None else float('inf')
                    dist = np.sqrt(dx**2 + dy**2)
                    # Convert to display coordinates for threshold
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1
                    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1
                    pixel_threshold = 0.05 * min(x_range, y_range)
                    if dist < pixel_threshold and dist < min_dist:
                        min_dist = dist
                        clicked_point = (date, subject, unit_idx)
            else:
                # Old format: individual line objects
                for line, date, subject, unit_idx in all_points:
                    cont, ind = line.contains(event)
                    if cont:
                        clicked_point = (date, subject, unit_idx)
                        break
            
            if clicked_point is None:
                return
            
            date, subject, unit_idx = clicked_point
            session_key = f"{subject}_{date}"
            
            # Use global cache
            cache_to_use = _global_cache if (cache_file == _default_cache_file and _global_cache is not None) else all_cache
            
            # Load unit_info_panel_dict lazily (only when clicked)
            # Check cache first, then load if not available
            if session_key not in cache_to_use:
                cache_to_use[session_key] = {}
            
            if 'unit_info_panel_dict' not in cache_to_use[session_key]:
                # Load it now (lazy loading)
                try:
                    # Lazy import to avoid circular dependency
                    from tejas.metrics.main_unit_panel import get_unit_info_panel_dict
                    print(f"Loading unit_info_panel_dict for {session_key}...")
                    unit_info_panel_dict = get_unit_info_panel_dict(date, subject, cache=cache)
                    cache_to_use[session_key]['unit_info_panel_dict'] = unit_info_panel_dict
                    # Update global cache if using default file
                    if cache_file == _default_cache_file and _global_cache is not None:
                        if session_key not in _global_cache:
                            _global_cache[session_key] = {}
                        _global_cache[session_key]['unit_info_panel_dict'] = unit_info_panel_dict
                    # Save to cache file
                    try:
                        cache_path = Path(cache_file)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_to_save = _global_cache if (cache_file == _default_cache_file and _global_cache is not None) else cache_to_use
                        with open(cache_path, 'wb') as f:
                            pickle.dump(cache_to_save, f)
                    except Exception as e:
                        print(f"Warning: Could not save cache: {e}")
                except Exception as e:
                    print(f"Warning: Could not load unit_info_panel_dict for {session_key}: {e}")
                    cache_to_use[session_key]['unit_info_panel_dict'] = None
                    return
            else:
                unit_info_panel_dict = cache_to_use[session_key]['unit_info_panel_dict']
            
            if unit_info_panel_dict is None:
                print(f"Warning: unit_info_panel_dict is None for {session_key}")
                return
            
            # Close previous unit panel figure if it exists
            if unit_panel_fig is not None:
                try:
                    plt.close(unit_panel_fig)
                except:
                    pass
            
            try:
                # Lazy import to avoid circular dependency
                from tejas.metrics.main_unit_panel import get_unit_info_panel
                
                # Create unit panel figure (temporary, for rendering)
                # Make it smaller so it fits better in the embedded view
                print(f"Creating unit panel for {session_key} unit {unit_idx}...")
                unit_panel_fig = get_unit_info_panel(
                    unit_idx, date, subject, unit_info_panel_dict,
                    cache=cache, save=False, mcfarland_bins=37,
                    show_ln=False, show_energy=False, show_fit=True,
                    show_ln_energy_fit=False
                )
                
                # Resize the figure to be smaller for embedding (original is 10x8, make it 8x6)
                unit_panel_fig.set_size_inches(8, 6)
                
                # Render the figure to an image buffer
                print("Rendering unit panel to image...")
                unit_panel_fig.canvas.draw()
                # Use savefig to BytesIO buffer for compatibility across matplotlib versions
                from io import BytesIO
                import matplotlib.image as mpimg
                buf = BytesIO()
                # Use higher DPI for better quality
                unit_panel_fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                buf.seek(0)
                # Read the image from the buffer
                img = mpimg.imread(buf)
                # Convert to RGB if needed (remove alpha channel if present)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                buf = (img * 255).astype(np.uint8)
                
                # Properly clear the axes - remove all artists
                print("Clearing unit panel axes...")
                # Remove all existing artists first
                while ax_unit_panel.patches:
                    ax_unit_panel.patches[0].remove()
                while ax_unit_panel.lines:
                    ax_unit_panel.lines[0].remove()
                while ax_unit_panel.images:
                    ax_unit_panel.images[0].remove()
                while ax_unit_panel.texts:
                    ax_unit_panel.texts[0].remove()
                # Now clear the axes
                ax_unit_panel.clear()
                
                # Display new panel - use 'equal' aspect to maintain proportions, or 'auto' to fill space
                print("Displaying unit panel...")
                # Display image maintaining its natural aspect ratio
                # Use aspect='equal' to maintain 1:1 pixel aspect ratio (no shearing)
                ax_unit_panel.imshow(buf, aspect='equal', interpolation='bilinear', origin='upper')
                ax_unit_panel.set_title(f'Unit {unit_idx} - {subject} {date}', fontsize=10, pad=5)
                ax_unit_panel.axis('off')
                
                # Close the temporary figure (we've extracted the image)
                plt.close(unit_panel_fig)
                unit_panel_fig = None
                
                print("Unit panel displayed successfully")
                fig.canvas.draw_idle()
                
            except Exception as e:
                print(f"Error displaying unit panel for {session_key} unit {unit_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Properly clear the axes
                while ax_unit_panel.patches:
                    ax_unit_panel.patches[0].remove()
                while ax_unit_panel.lines:
                    ax_unit_panel.lines[0].remove()
                while ax_unit_panel.images:
                    ax_unit_panel.images[0].remove()
                while ax_unit_panel.texts:
                    ax_unit_panel.texts[0].remove()
                ax_unit_panel.clear()
                ax_unit_panel.text(0.5, 0.5, f'Error loading unit panel: {str(e)[:50]}', 
                                  ha='center', va='center', transform=ax_unit_panel.transAxes,
                                  fontsize=10, color='red')
                ax_unit_panel.axis('off')
                fig.canvas.draw_idle()
        
        # Connect click event handler
        fig.canvas.mpl_connect("button_press_event", on_click)
    
    # Ensure figure is drawn and displayed for interactive mode BEFORE saving PDF
    if interactive:
        print("Preparing to display interactive figure...")
        print(f"Figure size: {fig.get_size_inches()}")
        print(f"Number of axes: {len(fig.axes)}")
        try:
            print("Drawing figure canvas...")
            fig.canvas.draw()
            print("Canvas drawn successfully")
            # In notebooks, returning the figure should display it
            # In scripts, we need to call plt.show()
            # Try to detect environment and handle accordingly
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython is None:
                    # Not in notebook, use plt.show()
                    print("Not in notebook - calling plt.show(block=False)...")
                    plt.show(block=False)
                    print("Figure displayed (script mode)")
                else:
                    print("In notebook - figure will display on return")
            except ImportError:
                # IPython not available, assume script mode
                print("IPython not available - calling plt.show(block=False)...")
                plt.show(block=False)
                print("Figure displayed (script mode, IPython not available)")
        except Exception as e:
            print(f"Warning: Error displaying figure: {e}")
            import traceback
            traceback.print_exc()
    
    # Save figure as PDF if save_dir is provided (for interactive mode)
    # Do this AFTER showing so it doesn't block display
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # Sanitize variable names for filename (replace / with _over_)
        var1_safe = variable1_name.replace('/', '_over_')
        var2_safe = variable2_name.replace('/', '_over_')
        pdf_filename = f'{var1_safe}vs{var2_safe}.pdf'
        try:
            fig.savefig(save_path / pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path / pdf_filename}")
        except Exception as e:
            print(f"Warning: Could not save figure: {e}")
    
    # Return figure for notebook display (interactive backend will display automatically)
    print("Function completed, returning figure...")
    return fig, all_points_qc_valid_inliers, highlighted_points

def plot_variable1(variable_name, session_list, cache=True, cache_all=False, cache_file='/mnt/ssd/YatesMarmoV1/metrics/all_metrics_cache.pkl', qc=False, saturate_boundaries=None, save_dir=None):
    """
    Plot histogram of a single variable across all sessions.
    
    If variable_name is a list with two elements, the first variable is plotted normally
    and the second variable is overlaid with transparency, showing only points that are
    valid for the first variable.
    
    Parameters:
        variable_name: String variable name, or list of two variable names for overlay
        saturate_boundaries: [min, max] array to clamp values. Values below min become min, values above max become max.
                           Default is None (no saturation).
        save_dir: Directory to save figure as PDF. Default is None (no save).
    """
    # Handle list input for overlay
    overlay_second = False
    variable1_name = variable_name
    variable2_name = None
    if isinstance(variable_name, list):
        if len(variable_name) == 2:
            variable1_name = variable_name[0]
            variable2_name = variable_name[1]
            overlay_second = True
        else:
            raise ValueError(f"If variable_name is a list, it must have exactly 2 elements, got {len(variable_name)}")
    
    # Validate that old McFarland variable names are not used
    if variable1_name in ['alpha', 'tau']:
        raise ValueError(f"Variables 'alpha' and 'tau' are no longer supported. Use format: {{metric}}_{{source}}_{{bins}}_bins (e.g., 'tau_data_10_bins', 'alpha_ln_37_bins')")
    if overlay_second and variable2_name in ['alpha', 'tau']:
        raise ValueError(f"Variables 'alpha' and 'tau' are no longer supported. Use format: {{metric}}_{{source}}_{{bins}}_bins (e.g., 'tau_data_10_bins', 'alpha_ln_37_bins')")
    
    fig, ax = plt.subplots()
    all_values = []
    valid_units_set = set()  # Track (date, subject, unit_idx) tuples with valid first variable data
    
    # Load all_cache from pickle file if it exists
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                all_cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            all_cache = {}
    else:
        all_cache = {}
    
    # Collect all values for first variable
    for session in tqdm(session_list):
        date = session.name.split('_')[1]
        subject = session.name.split('_')[0]
        session_key = session.name
        
        try:
            use_cached = cache_all and session_key in all_cache
            session_cache = all_cache.get(session_key, {}) if use_cached else {}
            
            # Load metrics for first variable
            variable_name_to_dict = _load_metrics_for_session(
                variable1_name, variable1_name, date, subject, cache, use_cached, session_cache
            )
            
            # Update cache
            if session_key not in all_cache:
                all_cache[session_key] = {}
            for var_name, var_value in variable_name_to_dict.items():
                if var_value is not None:
                    all_cache[session_key][var_name] = var_value
            
            # Get QC units if needed
            qc_units = None
            if qc:
                # Check if QC units are in cache
                if use_cached and 'qc_units' in session_cache:
                    qc_units = session_cache['qc_units']
                else:
                    qc_units = get_qc_units_for_session(date, subject, cache=cache)
                    # Save to cache
                    if session_key not in all_cache:
                        all_cache[session_key] = {}
                    all_cache[session_key]['qc_units'] = qc_units
            
            # Collect values for first variable
            values = _collect_values_for_session(
                variable_name_to_dict[variable1_name], variable1_name, date, subject, qc_units=qc_units
            )
            all_values.extend(values)
            
            # Track valid units for overlay
            if overlay_second:
                for val, val_date, val_subject, unit_idx in values:
                    if np.isfinite(val):
                        valid_units_set.add((val_date, val_subject, unit_idx))
        except Exception as e:
            print(f"Error processing session {session.name}: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    # Extract values and filter out NaN/infinite
    values_array = np.array([v for v, _, _, _ in all_values])
    valid_mask = np.isfinite(values_array)
    assert valid_mask.sum() == len(all_values), 'Number of valid values does not match number of all values'
    values_valid = values_array[valid_mask]
    
    # Apply saturation if specified
    if saturate_boundaries is not None:
        if len(saturate_boundaries) != 2:
            raise ValueError(f"saturate_boundaries must be [min, max] array, got {saturate_boundaries}")
        min_val, max_val = saturate_boundaries
        values_valid = np.clip(values_valid, min_val, max_val)
        # Also apply saturation to all_values for return
        all_values_saturated = [(np.clip(v, min_val, max_val), date, subject, unit_idx) 
                                 for v, date, subject, unit_idx in all_values]
    else:
        all_values_saturated = all_values
    
    # Plot histogram for first variable
    mean_val = None
    sem_val = None
    n_units = len(values_valid)
    bin_edges = None  # Store bin edges for overlay
    if n_units > 0:
        mean_val = float(np.mean(values_valid))
        sem_val = float(np.std(values_valid, ddof=1) / np.sqrt(n_units)) if n_units > 1 else 0.0
        # Get bin edges from first histogram
        counts, bin_edges = np.histogram(values_valid, bins=30)
        ax.hist(values_valid, bins=bin_edges, alpha=0.7, edgecolor='black', label=variable1_name)
        ax.set_xlabel(variable1_name)
        ax.set_ylabel('Frequency')
        qc_str = ' (QC)' if qc else ''
        if sem_val is not None:
            ax.set_title(
                f'Distribution of {variable1_name}{qc_str}\n'
                f'N={n_units}, mean = {mean_val:.3f} ± {sem_val:.3f} (SEM)'
            )
        else:
            ax.set_title(f'Distribution of {variable1_name}{qc_str}\nN={n_units}')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
    
    # Overlay second variable if requested
    if overlay_second and len(valid_units_set) > 0:
        all_values_second = []
        
        # Reload sessions to get second variable, but only for valid units
        for session in tqdm(session_list, desc="Loading overlay variable"):
            date = session.name.split('_')[1]
            subject = session.name.split('_')[0]
            session_key = session.name
            
            try:
                use_cached = cache_all and session_key in all_cache
                session_cache = all_cache.get(session_key, {}) if use_cached else {}
                
                # Load metrics for second variable
                variable_name_to_dict_second = _load_metrics_for_session(
                    variable2_name, variable2_name, date, subject, cache, use_cached, session_cache
                )
                
                # Update cache
                for var_name, var_value in variable_name_to_dict_second.items():
                    if var_value is not None:
                        all_cache[session_key][var_name] = var_value
                
                # Get QC units if needed
                qc_units = None
                if qc:
                    # Check if QC units are in cache
                    if use_cached and 'qc_units' in session_cache:
                        qc_units = session_cache['qc_units']
                    else:
                        qc_units = get_qc_units_for_session(date, subject, cache=cache)
                        # Save to cache
                        if session_key not in all_cache:
                            all_cache[session_key] = {}
                        all_cache[session_key]['qc_units'] = qc_units
                
                # Collect values for second variable, but only for units that are valid in first variable
                if variable2_name in variable_name_to_dict_second:
                    dict_second = variable_name_to_dict_second[variable2_name]
                    keys = set(dict_second.keys())
                    if qc_units is not None:
                        keys = keys & set(qc_units)
                    
                    for key in keys:
                        # Only include if this unit was valid for first variable
                        if (date, subject, key) in valid_units_set:
                            if dict_second[key]['valid']:
                                val = dict_second[key][variable2_name]
                                
                                if variable2_name == 'peak_sf':
                                    val = val + np.random.normal(0, 0.1)
                                
                                all_values_second.append((val, date, subject, key))
            except Exception as e:
                print(f"Error processing session {session.name} for overlay: {e}")
                import traceback
                traceback.print_exc()
                pass
        
        # Extract and filter second variable values
        if len(all_values_second) > 0:
            values_array_second = np.array([v for v, _, _, _ in all_values_second])
            valid_mask_second = np.isfinite(values_array_second)
            values_valid_second = values_array_second[valid_mask_second]
            
            # Apply saturation to second variable if specified
            if saturate_boundaries is not None:
                min_val, max_val = saturate_boundaries
                values_valid_second = np.clip(values_valid_second, min_val, max_val)
            
            # Overlay histogram with transparency using same bin edges as first variable
            if len(values_valid_second) > 0 and bin_edges is not None:
                ax.hist(values_valid_second, bins=bin_edges, alpha=0.4, edgecolor='gray', label=variable2_name)
                ax.legend()
    
    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_cache, f)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")
    
    # Save figure as PDF if save_dir is provided
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # Sanitize variable name for filename (replace / with _over_)
        if overlay_second:
            var_safe = f"{variable1_name.replace('/', '_over_')}_overlay_{variable2_name.replace('/', '_over_')}"
        else:
            var_safe = variable1_name.replace('/', '_over_')
        pdf_filename = f'{var_safe}.pdf'
        fig.savefig(save_path / pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path / pdf_filename}")
    
    plt.show()

    all_values_sorted = sorted(all_values_saturated, key=lambda x: x[0])
    return fig, all_values_sorted



def plot_simple_complex_mcfarland_population(modulation_index_values_ascending, n_values):
    cache_file = '/mnt/ssd/YatesMarmoV1/metrics/all_metrics_cache.pkl'
    
    # Load all_cache from pickle file if it exists
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            all_cache = pickle.load(f)
    else:
        raise FileNotFoundError(f'Cache file {cache_file} not found')
    
    #get top n values from modulation_index_values_ascending
    #start from last value in modulation_index_values_ascending and get the next n values
    top_n_values_accumulated = []
    top_index_stopped_at = 0
    top_all_ep_bins = []
    top_all_data_norm_vars = []
    top_sqrt_contour_area_values = []


    while len(top_n_values_accumulated) < n_values:
        modulation_index_value = modulation_index_values_ascending[-top_index_stopped_at-1]
        mod_val, date, subject, unit_idx = modulation_index_value
        results_data = all_cache[f"{subject}_{date}"]['tau_data_37_bins'][unit_idx]  
        if results_data['valid']:
            top_n_values_accumulated.append(modulation_index_value)
            top_all_ep_bins.append(results_data['ep_bins'])
            top_all_data_norm_vars.append(results_data['bin_rate_vars'] / results_data['em_corrected_var'])
            top_sqrt_contour_area_values.append(all_cache[f"{subject}_{date}"]['sqrt_area_contour_deg'][unit_idx]['sqrt_area_contour_deg'])
        top_index_stopped_at += 1
    top_index_stopped_at = len(modulation_index_values_ascending) - top_index_stopped_at
    top_data_mean = np.mean(top_all_data_norm_vars, axis=0)
    top_data_sem = np.std(top_all_data_norm_vars, axis=0) / np.sqrt(len(top_all_data_norm_vars))
    top_mean_ep_bins = np.mean(top_all_ep_bins, axis=0)
    top_mean_sqrt_contour_area_values = np.mean(top_sqrt_contour_area_values, axis=0)
   
    bottom_n_values_accumulated = []
    bottom_index_stopped_at = 0
    bottom_all_ep_bins = []
    bottom_all_data_norm_vars = []
    bottom_sqrt_contour_area_values = []
    while len(bottom_n_values_accumulated) < n_values:
        modulation_index_value = modulation_index_values_ascending[bottom_index_stopped_at]
        mod_val, date, subject, unit_idx = modulation_index_value
        results_data = all_cache[f"{subject}_{date}"]['tau_data_10_bins'][unit_idx]
        if results_data['valid']:
            bottom_n_values_accumulated.append(modulation_index_value)
            bottom_all_ep_bins.append(results_data['ep_bins'])
            bottom_all_data_norm_vars.append(results_data['bin_rate_vars'] / results_data['em_corrected_var'])
            bottom_sqrt_contour_area_values.append(all_cache[f"{subject}_{date}"]['sqrt_area_contour_deg'][unit_idx]['sqrt_area_contour_deg'])
        bottom_index_stopped_at += 1
    bottom_mean_sqrt_contour_area_values = np.mean(bottom_sqrt_contour_area_values, axis=0)
    
    bottom_data_mean = np.mean(bottom_all_data_norm_vars, axis=0)
    bottom_data_sem = np.std(bottom_all_data_norm_vars, axis=0) / np.sqrt(len(bottom_all_data_norm_vars))
    bottom_mean_ep_bins = np.mean(bottom_all_ep_bins, axis=0)
    assert len(top_n_values_accumulated) == len(bottom_n_values_accumulated) == n_values, 'Number of top and bottom n values accumulated do not match'
    assert bottom_index_stopped_at < top_index_stopped_at, 'Bottom index stopped at is not less than top index stopped at'
    # print(top_all_ep_bins.shape, top_data_mean.shape)
    plt.plot(top_mean_ep_bins, top_data_mean, 'o-', linewidth=2, label=f'Highest {n_values} Modulation Index')
    plt.fill_between(top_mean_ep_bins, top_data_mean - top_data_sem, top_data_mean + top_data_sem,  alpha=0.3)
    plt.plot(bottom_mean_ep_bins, bottom_data_mean, 'o-', linewidth=2, label=f'Lowest {n_values} Modulation Index')
    plt.fill_between(bottom_mean_ep_bins, bottom_data_mean - bottom_data_sem, bottom_data_mean + bottom_data_sem,  alpha=0.3)

    plt.axvline(top_mean_sqrt_contour_area_values, color='red', linestyle='--', alpha=0.5, label='Highest Modulation Index Sqrt Contour Area Average')
    plt.axvline(bottom_mean_sqrt_contour_area_values, color='blue', linestyle='--', alpha=0.5, label='Lowest Modulation Index Sqrt Contour Area Average')

    # Reference line at y=1 (normalized EM-corrected variance)
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(0.0, color='black', linestyle='--', alpha=0.5)

    plt.axvline(0.1, color='black', linestyle='--', alpha=0.5)

    plt.xlabel('Eye Position Distance (degrees)')
    plt.ylabel('Normalized Variance')
    # plt.title(f'Population Average Normalized Variance. N={len(mcfarland_files)}')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()


    
#%%