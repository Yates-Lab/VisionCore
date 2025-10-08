import sys
import numpy as np
import lightning as pl
import torch
import h5py
from scipy.sparse import coo_array

__all__ = ['min_max_norm', 'ensure_ndarray', 'nd_cut',
           'nd_paste', 'ensure_tensor', 'explore_hdf5', 'convert_time_to_samples']

def min_max_norm(x, min, max):
    '''
    Normalize x to the range [min, max].
    ''' 
    assert max > min, 'max must be greater than min'
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        return x - x_min + (max + min) / 2
    return (x - x.min()) / (x.max() - x.min()) * (max - min) + min

def ensure_ndarray(x, dtype=None):
    """
    Ensures that the input is a numpy.ndarray. If it is a tensor, it is converted to a numpy array.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.

    Returns:
    -------
    numpy.ndarray
        The input converted to a numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(x, int) or isinstance(x, float):
        x = [x]
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x

def nd_cut(source_array, position, dest_shape, fill_value=0):
    """
    Extract a region of interest from an n-dimensional array. If the requested region extends
    beyond the source array bounds, the result is padded with fill_value.
    
    Args:
        source_array (np.ndarray): Source array to extract from
        position (tuple): Position in source array where to start extraction (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        fill_value: Value to use for padding when outside source array bounds
        
    Returns:
        np.ndarray: Result array of shape dest_shape containing the extracted region
    """
    # Create output array filled with the fill value
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region
    source_start = np.maximum(np.zeros(ndim, dtype=int), position)
    source_end = np.minimum(source_shape, position + dest_shape)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    target_end = np.minimum(dest_shape, source_shape - position)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the extraction operation only if there's a valid region to extract
    if all(end > start for start, end in zip(target_start, target_end)):
        result[target_slices] = source_array[source_slices]
    
    return result
def nd_paste(source_array, position, dest_shape, fill_value=0):
    """
    Paste an n-dimensional array into an array of specified shape at given position.
    
    Args:
        source_array (np.ndarray): Source array to paste
        position (tuple): Position where to paste the array (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        
    Returns:
        np.ndarray: Result array of shape dest_shape with source_array pasted at position
    """
    # Create output array filled with zeros
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region (handling negative positions)
    source_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    source_end = np.minimum(source_shape, dest_shape - position)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), position)
    target_end = np.minimum(dest_shape, position + source_shape)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the paste operation
    if all(end > start for start, end in zip(source_start, source_end)):
        result[target_slices] = source_array[source_slices]
    
    return result


def ensure_tensor(x, device=None, dtype=None):
    """
    Ensures that the input is a torch.Tensor. If it is a numpy array, it is converted to a tensor.
    If device is provided, the tensor is moved to the device.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.
    device : torch.device
        The device to move the tensor to.
    dtype : torch.dtype
        The data type to convert the tensor to.

    Returns:
    -------
    torch.Tensor
        The input converted to a tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor([x])
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x

def explore_hdf5(file_path_or_obj, indent=0):
    """
    Recursively explore and print the structure of an HDF5 file.
    
    Parameters:
    -----------
    file_path_or_obj : str, Path, or h5py.File
        Path to HDF5 file or an open HDF5 file object
    indent : int
        Indentation level for pretty printing (default: 0)
    """
    def print_indented(msg, level=0):
        print("    " * level + msg)
    
    def print_attributes(obj, level):
        """Print all attributes of an HDF5 object."""
        if len(obj.attrs) > 0:
            print_indented("Attributes:", level)
            for key, value in obj.attrs.items():
                if isinstance(value, np.ndarray):
                    value_str = f"ndarray(shape={value.shape}, dtype={value.dtype})"
                else:
                    value_str = str(value)
                print_indented(f"{key}: {value_str}", level + 1)

    def explore_group(group, level):
        """Recursively explore an HDF5 group."""
        # Print group info
        print_indented(f"Group: {group.name}", level)
        print_attributes(group, level)
        
        # Explore all items in the group
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                explore_group(item, level + 1)
            elif isinstance(item, h5py.Dataset):
                print_indented(f"Dataset: {item.name}", level + 1)
                print_indented(f"Shape: {item.shape}", level + 2)
                print_indented(f"Dtype: {item.dtype}", level + 2)
                if item.compression:
                    print_indented(f"Compression: {item.compression}", level + 2)
                print_attributes(item, level + 2)

    # Handle input that could be either a path or an HDF5 object
    if isinstance(file_path_or_obj, (str, Path)):
        with h5py.File(file_path_or_obj, 'r') as f:
            explore_group(f, indent)
    elif isinstance(file_path_or_obj, h5py.File):
        explore_group(file_path_or_obj, indent)
    else:
        raise TypeError("Input must be a path or an h5py.File object")


def convert_time_to_samples(times, adfreq, ts=None, fn=None):
    """
    Convert timestamps into samples using sampling rate and start-time
    
    Parameters:
    -----------
    times : array-like
        Vector/matrix of timestamps
    adfreq : float
        Sampling rate
    ts : array-like or float, optional
        Vector of recording fragment start timestamps or single timestamp
    fn : array-like or int, optional
        Vector of fragment sample counts or single count
        
    Returns:
    --------
    samples : ndarray
        Converted sample indices with same shape as input times

    jly 2025-02-07 wrote it
    """
    
    
    if ts is None:
        ts = 0

    # Handle default arguments
    if fn is None:
        fn = np.round(np.max(times) * adfreq)
    
    # Convert inputs to numpy arrays if they aren't already
    times = np.asarray(times)
    
    # Handle scalar inputs by converting to 1-element arrays
    ts = np.atleast_1d(np.asarray(ts))
    fn = np.atleast_1d(np.asarray(fn))
    
    # Input validation
    assert np.all(np.mod(fn, 1) == 0), 'fragments must be integers'
    assert ts.size == fn.size, 'num fragments must equal num timestamps'
    
    n_fragments = fn.size
    
    # Calculate fragment boundaries
    fb = ts  # fragment begin times
    fe = ts + fn/adfreq  # fragment end times
    
    # Calculate sample boundaries
    se = np.cumsum(fn)  # sample ends
    sb = np.concatenate(([0], se[:-1] + 1))  # sample begins
    
    # Initialize output array with NaNs
    samples = np.full_like(times, np.nan, dtype=float)
    
    # Process each fragment
    for ff in range(n_fragments):
        idx = (times >= fb[ff]) & (times <= fe[ff])
        samples[idx] = np.floor((times[idx] - fb[ff]) * adfreq) + sb[ff]
    
    return samples

def convert_samples_to_time(samples, adfreq, ts=None, fn=None):
    """
    Convert samples into timestamps using sampling rate and start-time
    
    Parameters:
    -----------
    samples : array-like
        Vector of sample indices
    adfreq : float
        Sampling rate
    ts : array-like, optional
        Vector of recording fragment start timestamps
    fn : array-like, optional
        Vector of fragment sample counts
        
    Returns:
    --------
    times : ndarray
        Converted timestamps
        
    Notes:
    ------
    ts and fn are necessary to adjust sample times for recordings that were
    paused or are not continuous

    jly 2025-02-07 wrote it
    """
    
    # Handle default arguments
    if fn is None:
        fn = np.max(samples)
    if ts is None:
        ts = 0
        
    # Convert inputs to numpy arrays if they aren't already
    samples = np.asarray(samples)

    # Handle scalar inputs by converting to 1-element arrays
    ts = np.atleast_1d(np.asarray(ts))
    fn = np.atleast_1d(np.asarray(fn))
    
    n_fragments = fn.size
    fb = ts  # fragment begin times
    
    # Calculate sample boundaries
    sb = np.concatenate(([0], np.cumsum(fn[:-1])))  # sample begins
    se = np.cumsum(fn)  # sample ends
    
    # Initialize output array
    times = np.zeros(samples.size)
    
    # Process each fragment
    for ff in range(n_fragments):  # Python uses 0-based indexing
        idx = (samples >= sb[ff]) & (samples <= se[ff])
        times[idx] = ((samples[idx] - sb[ff]) + 1) / adfreq + fb[ff]
    
    return times

def event_triggered_analog(signal, events, bins):
    """
    Align analog signals to events with specified bins using vectorized operations.
    
    Parameters:
    -----------
    signal : ndarray, shape (n_samples, n_channels)
        The analog signal matrix
    events : ndarray, shape (n_events,)
        Array of event indices
    bins : ndarray
        Array of bin offsets in samples
    
    Returns:
    --------
    ndarray, shape (n_events, n_bins, n_channels)
        Event-triggered average array
    """
    n_samples, n_channels = signal.shape
    n_events = len(events)
    n_bins = len(bins)
    
    # Create meshgrid for vectorized indexing
    # events[:, None] + bins creates a matrix of all event-bin combinations
    indices = (events[:, None] + bins).astype(int)
    
    # Create mask for valid indices
    valid_mask = (indices >= 0) & (indices < n_samples)
    
    # Initialize result array
    result = np.zeros((n_events, n_bins, n_channels))
    
    for ch in range(n_channels):
        result[:, :, ch] = signal[indices, ch]
    
    # Use advanced indexing to extract all valid combinations at once
    result[~valid_mask] = np.nan
    
    return result

def event_triggered_time(times, events, bins):
    """
    Bin time series data aligned to events using memory-efficient vectorized operations.
    
    Parameters:
    -----------
    times : ndarray
        Array of timestamps
    events : ndarray
        Array of event times
    bins : ndarray
        Array of bin edges (length = n_bins + 1)
    
    Returns:
    --------
    ndarray, shape (n_events, n_bins)
        Binned event-triggered data
    """
    n_events = len(events)
    n_bins = len(bins) - 1
    
    # Create relative bin edges matrix (events x bins)
    # This is much smaller than events x times
    bin_edges = events[:, None] + bins[None, :]  # Shape: (n_events, n_bins+1)
    
    # Initialize sparse matrix components
    rows = []
    cols = []
    data = []
    
    # For each event's set of bin edges, find which times fall into them
    for i, edges in enumerate(bin_edges):
        # Find which times fall into any bin for this event
        mask = (times >= edges[0]) & (times <= edges[-1])
        event_times = times[mask]
        
        if len(event_times) > 0:
            # Digitize only the relevant times for this event
            bin_indices = np.digitize(event_times, edges) - 1
            
            # Add to sparse matrix components
            valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
            rows.extend([i] * np.sum(valid_bins))
            cols.extend(bin_indices[valid_bins])
            data.extend(np.ones(np.sum(valid_bins)))
    
    # Create sparse matrix
    sparse_result = coo_array(
        (data, (rows, cols)),
        shape=(n_events, n_bins)
    )
    
    return sparse_result.toarray()

def get_clock_functions(exp):
    '''
    Temporarily here until we figure out where to put it
    Input: 
        exp dictionary (level that contains ['D'] list)
    Output:
        ptb2ephys: function that converts PTB time to ephys time
        vpx2ephys: function that converts Eyetracker time to ephys time
    '''

    from scipy.interpolate import interp1d

    # Synchronize the clocks
    ephys_clock = []
    ptb_clock = []
    vpx_clock = []
    for i in range(len(exp['D'])):
        keys = ['START_EPHYS', 'END_EPHYS', 'STARTCLOCKTIME', 'ENDCLOCKTIME', 'START_VPX', 'END_VPX']
        key_present = [key in exp['D'][i] for key in keys]

        if not (key_present[0] and key_present[1]):
            # skip if ephys clock is not present
            continue

        # append the ephys clock times
        ephys_clock.append(exp['D'][i][keys[0]])
        ephys_clock.append(exp['D'][i][keys[1]])

        # append the PTB clock times
        # default to the first and last eye data times if not present
        if key_present[2] and key_present[3]:
            ptb_clock.append(exp['D'][i][keys[2]])
            ptb_clock.append(exp['D'][i][keys[3]])
        else:
            ptb_clock.append(exp['D'][i]['eyeData'][0,5])
            ptb_clock.append(exp['D'][i]['eyeData'][-1,5])

        # append the vpx clock times
        # default to the first and last eye data times if not present
        if  key_present[4] and key_present[5]:
            vpx_clock.append(exp['D'][i][keys[4]])
            vpx_clock.append(exp['D'][i][keys[5]])
        else:
            vpx_clock.append(exp['D'][i]['eyeData'][0,5])
            vpx_clock.append(exp['D'][i]['eyeData'][-1,5])

    ephys_clock = np.array(ephys_clock)
    ptb_clock = np.array(ptb_clock)
    vpx_clock = np.array(vpx_clock)

    nan_mask = np.isnan(ephys_clock) | np.isnan(ptb_clock) | np.isnan(vpx_clock)
    ephys_clock = ephys_clock[~nan_mask]
    ptb_clock = ptb_clock[~nan_mask]
    vpx_clock = vpx_clock[~nan_mask]

    ptb2ephys = interp1d(ptb_clock, ephys_clock, kind='linear', fill_value='extrapolate')
    vpx2ephys = interp1d(vpx_clock, ephys_clock, kind='linear', fill_value='extrapolate')
    return ptb2ephys, vpx2ephys

def exp_get_trial_idx(exp, trial_type):
    '''
    Identify the trials of a given type in an experiment
    Inputs:
        exp: dict, experiment data (contains ['D'] list of trials)
        trial_type: str, type of trial to identify
    Outputs:
        trial_idx: list of ints, indices of trials of the given

    Forage trials can be named 'Grating', 'Gabor', 'Dots', 'BigDots', or 'SmallDots'

    All other trials will just return any trials matching the protocol name

    jly 2025-02-08 wrote it
    '''
    
    num_trials = len(exp['D'])
    trial_protocols = [exp['D'][i]['PR']['name'] for i in range(num_trials)]

    if trial_type in ['Grating', 'Gabor', 'CSD', 'DriftingGrating']: # it's a forage trial
            noisetype = {'Grating': 1, 'Gabor': 4, 'CSD': 3, 'DriftingGrating': 6} 

            trial_idx = np.where(np.array(trial_protocols) == 'ForageProceduralNoise')[0]
            trial_idx = [i for i in trial_idx if exp['D'][i]['PR']['noisetype']==noisetype[trial_type]]
    
    elif trial_type in ['Dots', 'BigDots', 'SmallDots']: # it's a forage trial
        
        trial_idx = np.where(np.array(trial_protocols) == 'ForageProceduralNoise')[0]
        trial_idx = [i for i in trial_idx if exp['D'][i]['PR']['noisetype']==2]

        dotsize = [exp['D'][i]['P']['dotSize'] for i in trial_idx]

        if trial_type == 'BigDots':
            target = np.max(np.asarray(dotsize))
            trial_idx = [i for i in trial_idx if dotsize[i] == target]
        elif trial_type == 'SmallDots':
            target = np.min(np.asarray(dotsize))
            trial_idx = [i for i in trial_idx if dotsize[i] == target]
    
    else:

        trial_idx = np.where(np.array(trial_protocols) == trial_type)[0]
    
    return trial_idx

class ValidateOnTrainStart(pl.Callback):
    """
    Callback to run validation at the start of training.
    """
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        return trainer.fit_loop.epoch_loop.val_loop.run()

def get_valid_dfs(dset, n_lags):
    """
    Generate a binary mask for valid data frames based on trial boundaries and DPI validity.
    
    This function creates a mask that identifies valid frames for analysis by:
    1. Identifying trial boundaries
    2. Excluding the first frame of each trial
    3. Ensuring DPI (eye tracking) data is valid
    4. Ensuring temporal continuity for the specified number of lags
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing trial indices and DPI validity information
    n_lags : int
        Number of time lags to ensure continuity for
        
    Returns:
    --------
    dfs : torch.Tensor
        Binary mask tensor of shape [n_frames, 1] where 1 indicates valid frames
    """
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    dfs = ~new_trials
    dfs &= (dpi_valid > 0)

    for _ in range(n_lags):
        dfs &= torch.roll(dfs, 1)
    
    dfs = dfs.float()
    dfs = dfs[:, None]
    return dfs

def get_optimizer(parameters, config:dict):
    """
    Create an optimizer based on configuration dictionary.
    
    Parameters:
    -----------
    parameters : iterable
        Model parameters to optimize
    config : dict
        Configuration dictionary with optimizer settings.
        Must contain 'optimizer' key specifying the optimizer type.
        Remaining keys are passed as keyword arguments to the optimizer.
        
    Returns:
    --------
    optimizer : torch.optim.Optimizer
        Configured optimizer instance
    """
    name = config.pop('optimizer')
    if name == 'SGD':
        optimizer = torch.optim.SGD(parameters, **config)
    elif name == 'Adam':
        optimizer = torch.optim.Adam(parameters, **config)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, **config)
    elif name == 'SGDScheduleFree': 
        from schedulefree import SGDScheduleFree
        optimizer = SGDScheduleFree(parameters, **config)
    elif name == 'AdamWScheduleFree':
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(parameters, **config)
    else:
        raise ValueError(f"Optimizer {name} not recognized")
    return optimizer

