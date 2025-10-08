"""
Utility functions for creating basis functions for neural modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_raised_cosine_basis(num_delta_funcs,
                             num_cosine_funcs,
                             history_bins,
                             log_spacing=True,
                             peak_range_ms=(5, 100),
                             bin_size_ms=1,
                             normalize=True,
                             orthogonalize=False):
    """
    Generates a basis matrix for raised cosine functions and delta functions.

    Args:
        num_delta_funcs (int): Number of delta functions (for instantaneous effects at t=0, t=1, ...).
        num_cosine_funcs (int): Number of raised cosine basis functions.
        history_bins (int): Duration of history covered by the cosine bases (in bins).
                            This will be the kernel size.
        log_spacing (bool): Whether to use log or linear spacing for cosine peaks.
        peak_range_ms (tuple): (min_peak_ms, max_peak_ms) for cosine peaks.
                               Peaks are relative to the start of the cosine history part.
        bin_size_ms (int): Bin size in ms (fixed at 1 for this implementation).

    Returns:
        torch.Tensor: Basis functions as a tensor of shape
                      [num_total_basis, 1, kernel_size_bins].
                      kernel_size_bins is max(1, history_bins).
    """
    kernel_size_bins = max(1, int(history_bins)) # Ensure kernel_size is at least 1
    num_total_basis = num_delta_funcs + num_cosine_funcs

    if num_total_basis == 0:
        return torch.empty((0, 1, kernel_size_bins))

    basis_matrix = torch.zeros(num_total_basis, 1, kernel_size_bins)

    # Create Delta Functions
    for i in range(num_delta_funcs):
        if i < kernel_size_bins: # Ensure delta peak is within kernel
            basis_matrix[i, 0, i] = 1.0

    # Create Raised Cosine Functions
    if num_cosine_funcs > 0:
        # Define time points for the basis functions (0 to history_bins-1)
        # These are time lags relative to the current time point being predicted.
        # A positive lag means "in the past".
        # The convolution kernel will be flipped by conv1d, so we design it "forward in time"
        # from 0 to history_bins-1, representing effect of spike at t-kernel_size_bins+1 to t.
        t_bins = torch.arange(kernel_size_bins, dtype=torch.float32) * bin_size_ms

        if log_spacing:
            # Logarithmic spacing of peaks
            # Ensure peak_range_ms[0] > 0 for log spacing
            min_peak = np.log(max(1e-2, peak_range_ms[0]))
            max_peak = np.log(max(1e-2, peak_range_ms[1]))
            if num_cosine_funcs == 1:
                log_centers = torch.tensor([min_peak + (max_peak - min_peak) / 2])
            else:
                log_centers = torch.linspace(min_peak, max_peak, num_cosine_funcs)
            centers_ms = torch.exp(log_centers)
            # Width (dc) in log space, then map to linear - this is a simplification
            if num_cosine_funcs == 1:
                 # Heuristic for single basis: cover half the range on either side
                log_width = (max_peak - min_peak) if (max_peak > min_peak) else np.log(peak_range_ms[1] if peak_range_ms[1] > 0 else 1.0)
            else:
                log_width = (log_centers[1] - log_centers[0]) * 2.0 if num_cosine_funcs > 1 else np.log(peak_range_ms[1])


        else:
            # Linear spacing of peaks
            if num_cosine_funcs == 1:
                centers_ms = torch.tensor([peak_range_ms[0] + (peak_range_ms[1] - peak_range_ms[0]) / 2])
            else:
                centers_ms = torch.linspace(peak_range_ms[0], peak_range_ms[1], num_cosine_funcs)
            # Width (dc) in linear space
            if num_cosine_funcs == 1:
                width_ms = (peak_range_ms[1] - peak_range_ms[0]) if (peak_range_ms[1] > peak_range_ms[0]) else peak_range_ms[1]
            else:
                width_ms = (centers_ms[1] - centers_ms[0]) * 2.0 if num_cosine_funcs > 1 else peak_range_ms[1]


        for i in range(num_cosine_funcs):
            center_ms = centers_ms[i]
            if log_spacing:
                # For log spacing, the width parameter needs careful definition.
                # This is a common way from Pillow et al.
                # The argument to cosine is (log(t) - log(c)) * pi / dc_log
                # We map t_bins to log space for the calculation
                # Add a small epsilon to t_bins to avoid log(0)
                log_t_plus_epsilon = torch.log(t_bins + 1e-6)
                # Ensure center_ms is positive for log
                log_center = torch.log(center_ms if center_ms > 1e-6 else 1e-6)
                # Cosine argument: (log(t) - c_log) / width_log * pi
                # The effective "width" parameter (dc in the formula) in log space
                arg = (log_t_plus_epsilon - log_center) * np.pi / log_width
            else:
                # Linear spacing width parameter
                arg = (t_bins - center_ms) * np.pi / width_ms

            # Raised cosine formula: (cos(x) + 1) / 2 for x in [-pi, pi], else 0
            cos_val = torch.cos(arg)
            basis_val = (cos_val + 1) / 2.0
            # Mask values outside the [-pi, pi] range for the argument
            basis_val[torch.abs(arg) > np.pi] = 0.0
            basis_matrix[num_delta_funcs + i, 0, :] = basis_val
    
    # Pad by one row of zeros to ensure causal convolution
    basis_matrix = F.pad(basis_matrix, (1, 0, 0, 0), mode='constant', value=0)

    if normalize:
        basis_matrix = F.normalize(basis_matrix, p=2, dim=2, eps=1e-12)
    
    if orthogonalize:
        Warning("Orthogonalization has not been tested.")
        basis_matrix = orthogonalize_bases(basis_matrix)

    return basis_matrix

def orthogonalize_bases(bases):
    """
    Orthogonalize the basis functions using Gram-Schmidt process.

    Args:
        bases (torch.Tensor): Basis functions as a tensor of shape
                              [num_basis, 1, kernel_size_bins].

    Returns:
        torch.Tensor: Orthogonalized basis functions.
    """
    num_basis, _, kernel_size = bases.shape
    orthogonal_bases = torch.zeros_like(bases)

    for i in range(num_basis):
        orthogonal_bases[i] = bases[i]
        for j in range(i):
            orthogonal_bases[i] -= (torch.dot(orthogonal_bases[j].view(-1), bases[i].view(-1)) /
                                    torch.dot(orthogonal_bases[j].view(-1), orthogonal_bases[j].view(-1))) * orthogonal_bases[j]

        # Re-normalize after orthogonalization
        orthogonal_bases[i] = F.normalize(orthogonal_bases[i], p=2, dim=1, eps=1e-12)

    return orthogonal_bases


# def _project_on_basis(self, event_stream, basis):
#     """
#     Project an event stream onto basis functions.

#     Args:
#         event_stream (torch.Tensor): Binned event stream.
#         basis (torch.Tensor): Basis functions.

#     Returns:
#         torch.Tensor: Projected features.
#     """
#     # Add batch and channel dimensions
#     event_stream_tensor = event_stream.unsqueeze(0).unsqueeze(0)

#     # Calculate padding for causal convolution
#     kernel_size = basis.shape[2]
#     # padding = (0, kernel_size - 1)
#     padding = (kernel_size - 1, 0)

#     # Pad for causal convolution
#     padded_event_stream = F.pad(event_stream_tensor, padding, mode='constant', value=0)

#     # flip basis in time
#     basis = torch.flip(basis, dims=(2,))

#     # Convolve with basis functions
#     projected = F.conv1d(padded_event_stream, basis).permute(2, 0, 1)

#     # Return with shape [features, time] for simplicity
#     return projected.detach()