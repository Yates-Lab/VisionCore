"""
Frontend modules for DataYatesV1.

This module contains frontend components for processing raw input data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

import warnings
import numpy as np
import matplotlib.pyplot as plt


from DataYatesV1.utils.modeling.bases import make_raised_cosine_basis
from ..utils.general import ensure_tensor
from torch._dynamo import disable  # hack to avoid dynamo warning about the graph breaking

__all__ = ['DAModel', 'TemporalBasis', 'AffineAdapter', 'LearnableTemporalConv']

class DAModel(nn.Module):
    """
    PyTorch implementation of the simplified Dynamical Adaptation (DA) model
    based on Clark et al., 2013, assuming tau_r ≈ 0 (Eq. 15).

    Processes inputs of shape: (N, S, H, W) or (N, 1, S, H, W)
    Outputs tensor of shape: (N, 1, S, H, W).
    Handles time constants based on provided sampling rate.
    """
    def __init__(self, sampling_rate=240,
                 height=None, width=None,
                 alpha=1.0, beta=0.00008, gamma=0.5,
                 tau_y_ms=5.0, tau_z_ms=60.0,
                 n_y=5., n_z=2.,
                 filter_length=200,
                 learnable_params=False):
        super().__init__()

        self.height = height # Optional, for assertion
        self.width = width   # Optional, for assertion
        self.filter_length = filter_length
        self.sampling_rate = float(sampling_rate)
        self.dt = 1.0 / self.sampling_rate # Timestep in seconds
        self.learnable_params = learnable_params

        self.initial_params = {
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'tau_y_ms': tau_y_ms, 'tau_z_ms': tau_z_ms,
            'n_y': n_y, 'n_z': n_z
        }

        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=learnable_params)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=learnable_params)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=learnable_params)
        self.tau_y_ms = nn.Parameter(torch.tensor(float(tau_y_ms)), requires_grad=learnable_params)
        self.n_y = nn.Parameter(torch.tensor(float(n_y)), requires_grad=learnable_params)
        self.tau_z_ms = nn.Parameter(torch.tensor(float(tau_z_ms)), requires_grad=learnable_params)
        self.n_z = nn.Parameter(torch.tensor(float(n_z)), requires_grad=learnable_params)

        self.padding = (self.filter_length - 1, 0)

        if not learnable_params:
            Ky_filter, Kz_filter = self._calculate_filters(
                device=torch.device('cpu'), dtype=torch.float32 # Precompute on CPU
            )
            self.register_buffer('Ky_filter', Ky_filter)
            self.register_buffer('Kz_filter', Kz_filter)

    def _calculate_filters(self, device, dtype):
        """ Helper to calculate filters based on current parameters. """
        # Ensure parameters are on the correct device for calculation
        gamma = self.gamma.to(device=device, dtype=dtype)
        tau_y_ms = self.tau_y_ms.to(device=device, dtype=dtype)
        n_y = self.n_y.to(device=device, dtype=dtype)
        tau_z_ms = self.tau_z_ms.to(device=device, dtype=dtype)
        n_z = self.n_z.to(device=device, dtype=dtype)

        tau_y_ms_clamped = torch.clamp(tau_y_ms, min=1e-2)
        tau_z_ms_clamped = torch.clamp(tau_z_ms, min=1e-2)
        n_y_clamped = torch.clamp(n_y, min=0.5) # Original had .5
        n_z_clamped = torch.clamp(n_z, min=0.1) # Original had .1
        gamma_clamped = torch.clamp(gamma, 0.0, 1.0)

        tau_y_sec = tau_y_ms_clamped / 1000.0
        tau_z_sec = tau_z_ms_clamped / 1000.0
        dt_tensor = torch.tensor(self.dt, device=device, dtype=dtype)
        tau_y_steps = torch.clamp(tau_y_sec / dt_tensor, min=0.1)
        tau_z_steps = torch.clamp(tau_z_sec / dt_tensor, min=0.1)

        t = torch.arange(0.0, float(self.filter_length), device=device, dtype=dtype)

        # Using torch.lgamma for potentially better stability with n + 1
        lgamma_n_y_plus_1 = torch.lgamma(n_y_clamped + 1.0 + 1e-9) # Small epsilon
        lgamma_n_z_plus_1 = torch.lgamma(n_z_clamped + 1.0 + 1e-9) # Small epsilon

        # Normalization factors
        norm_Ky = (tau_y_steps**(n_y_clamped + 1.0)) * torch.exp(lgamma_n_y_plus_1)
        norm_Kz_slow = (tau_z_steps**(n_z_clamped + 1.0)) * torch.exp(lgamma_n_z_plus_1)

        norm_Ky = torch.clamp(norm_Ky, min=1e-9)
        norm_Kz_slow = torch.clamp(norm_Kz_slow, min=1e-9)

        Ky = torch.pow(t, n_y_clamped) * torch.exp(-t / tau_y_steps)
        Kz_slow = torch.pow(t, n_z_clamped) * torch.exp(-t / tau_z_steps)

        Ky = Ky / norm_Ky
        Kz_slow = Kz_slow / norm_Kz_slow
        Kz = gamma_clamped * Ky + (1.0 - gamma_clamped) * Kz_slow

        Ky_filter = Ky.flip(0).unsqueeze(0).unsqueeze(0) # (1, 1, filter_length)
        Kz_filter = Kz.flip(0).unsqueeze(0).unsqueeze(0) # (1, 1, filter_length)
        return Ky_filter, Kz_filter

    def forward(self, s):
        """
        Applies the simplified DA model.
        Args:
            s (torch.Tensor): Input (N, 1, S, H, W) or (N, S, H, W). S is sequence/frames.
        Returns:
            torch.Tensor: Output (N, 1, S, H, W)
        """
        original_shape = s.shape
        if s.dim() == 4:  # (N, S, H, W)
            s = s.unsqueeze(1) # -> (N, 1, S, H, W)

        batch_size, channels, num_frames, height, width = s.shape
        if channels != 1:
            raise ValueError(f"DAModel expects input with 1 channel, got {channels} channels. Input shape: {original_shape}")

        if self.height is not None and (height != self.height or width != self.width):
            warnings.warn(f"Input spatial dimensions {height}x{width} don't match DAModel's expected {self.height}x{self.width}")

        input_device = s.device
        input_dtype = s.dtype

        if self.learnable_params:
            Ky_filter, Kz_filter = self._calculate_filters(device=input_device, dtype=input_dtype)
        else:
            Ky_filter = self.Ky_filter.to(device=input_device, dtype=input_dtype)
            Kz_filter = self.Kz_filter.to(device=input_device, dtype=input_dtype)

        # Reshape for 1D convolution: (N*H*W, 1, S)
        s_reshaped = s.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, 1, num_frames)

        # Pad for causal convolution. Using value=0.0 for simplicity and gradient friendliness.
        # Original used: pad_value = s_reshaped[:, :, 0].mean().item() which detaches from graph.
        s_padded = F.pad(s_reshaped, self.padding, mode='constant', value=0.0)

        y = F.conv1d(s_padded, Ky_filter)
        z = F.conv1d(s_padded, Kz_filter)

        alpha_eff = self.alpha.to(device=input_device, dtype=input_dtype)
        beta_eff = F.relu(self.beta.to(device=input_device, dtype=input_dtype)) # Ensure beta is non-negative

        denominator = 1.0 + beta_eff * z
        safe_denominator = torch.clamp(denominator, min=1e-6) # Avoid division by zero
        r = alpha_eff * y / safe_denominator

        # Reshape Output: (N*H*W, 1, S) -> (N, H, W, 1, S) -> (N, 1, S, H, W)
        r_reshaped = r.reshape(batch_size, height, width, 1, num_frames).permute(0, 3, 4, 1, 2)
        return r_reshaped


class TemporalBasis(nn.Module):
    """
    A module that applies a temporal basis to input signals using raised cosine basis functions.

    This module can be used to embed time-varying signals (like stimulus, eye position, etc.)
    onto a compact basis representation. It uses 1D convolution with a set of basis functions
    to project the input signal onto a lower-dimensional representation.

    Processes inputs of shape: (N, C, T, H, W) for images or (N, C, T) for 1D signals
    Outputs tensor of shape: (N, num_basis, T-chomp, H, W) or (N, num_basis, T-chomp)
    where chomp defaults to 0 if None.

    Args:
        num_delta_funcs (int): Number of delta functions (for instantaneous effects)
        num_cosine_funcs (int): Number of raised cosine basis functions
        history_bins (int): Duration of history covered by the basis (in bins)
        log_spacing (bool): Whether to use log or linear spacing for cosine peaks
        peak_range_ms (tuple): (min_peak_ms, max_peak_ms) for cosine peaks
        bin_size_ms (int): Bin size in ms (fixed at 1 for this implementation)
        normalize (bool): Whether to normalize the basis functions
        orthogonalize (bool): Whether to orthogonalize the basis functions
        causal (bool): Whether to use causal convolution (True) or acausal (False)
        sampling_rate (float): Sampling rate in Hz, used to convert ms to bins
        batch_norm (bool): Whether to apply batch normalization after convolution
        chomp (int or None): Number of initial time steps to remove from output sequence.
            If None (default), no chomping is applied. If int > 0, removes the first
            `chomp` time steps from the output to reduce memory footprint.
    """
    def __init__(self,
                 in_channels=None, # dummy, not used
                 num_delta_funcs=0,
                 num_cosine_funcs=10,
                 history_bins=32,
                 log_spacing=True,
                 peak_range_ms=(5, 100),
                 bin_size_ms=5,
                 normalize=True,
                 orthogonalize=False,
                 causal=True,
                 sampling_rate=240,
                 batch_norm=False,
                 chomp=None):
        super().__init__()

        self.num_delta_funcs = num_delta_funcs
        self.num_cosine_funcs = num_cosine_funcs

        self.log_spacing = log_spacing
        self.peak_range_ms = peak_range_ms
        self.bin_size_ms = bin_size_ms
        self.normalize = normalize
        self.orthogonalize = orthogonalize
        self.causal = causal
        self.sampling_rate = sampling_rate
        self.chomp = chomp

        # Calculate the total number of basis functions
        self.num_basis = num_delta_funcs + num_cosine_funcs

        # Create the basis functions
        basis_matrix = make_raised_cosine_basis(
            num_delta_funcs=num_delta_funcs,
            num_cosine_funcs=num_cosine_funcs,
            history_bins=history_bins,
            log_spacing=log_spacing,
            peak_range_ms=peak_range_ms,
            bin_size_ms=bin_size_ms,
            normalize=normalize,
            orthogonalize=orthogonalize
        )

        filter_length = basis_matrix.shape[2]
        basis_matrix = ensure_tensor(basis_matrix, dtype=torch.float32)
        
        basis_matrix = torch.flip(basis_matrix, dims=(2,))

        # Register the basis as a buffer (non-trainable parameter)
        self.register_buffer('basis', basis_matrix)

        # Calculate padding for convolution
        # For causal convolution, we pad on the left
        # For acausal convolution, we pad on both sides
        if causal:
            self.padding = (filter_length - 1, 0)  # Pad left for causal
        else:
            # For acausal, pad both sides
            pad_left = filter_length // 2
            pad_right = filter_length - 1 - pad_left
            self.padding = (pad_left, pad_right)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.num_basis)

    def forward(self, x):
        """
        Forward pass through the temporal basis.

        Args:
            x: Input tensor with shape (N, C, S, H, W) for images or (N, C, S) for 1D signals
                N: batch size
                C: channels
                S: sequence length
                H, W: height and width (for images)

        Returns:
            Tensor with shape (N, num_basis*C, S, H, W) or (N, num_basis*C, S)
        """
        original_shape = x.shape
        input_dim = len(original_shape)

        if input_dim == 5:  # (N, C, S, H, W) - Image data
            batch_size, channels, seq_len, height, width = original_shape
            # Reshape to (N*C*H*W, 1, S) for 1D convolution
            x_reshaped = x.permute(0, 1, 3, 4, 2).reshape(-1, 1, seq_len)
        elif input_dim == 3:  # (N, C, S) - 1D signal data
            batch_size, channels, seq_len = original_shape
            # Reshape to (N*C, 1, S) for 1D convolution
            x_reshaped = x.reshape(-1, 1, seq_len)
        else:
            raise ValueError(f"Input shape {original_shape} not supported. Expected (N,C,S,H,W) or (N,C,S).")

        # Pad for convolution
        x_padded = F.pad(x_reshaped, self.padding, mode='constant', value=0.0)

        # Apply convolution with basis functions
        # Flip the basis in time to push the stimulus/behavior into the future (giving robs access to the past)
        basis = self.basis
        if not self.causal:
            basis = torch.flip(basis, dims=(2,))

        # Convolve with basis functions
        # Output shape: (N*C*H*W, num_basis, S) or (N*C, num_basis, S)
        y = F.conv1d(x_padded, basis)

        if self.batch_norm:
            y = self.bn(y)

        # Reshape back to original dimensions
        if input_dim == 5:  # (N, C, S, H, W)
            # Reshape to (N, C, H, W, num_basis, S)
            y = y.reshape(batch_size, channels, height, width, self.num_basis, seq_len)
            # Permute to (N, C*num_basis, S, H, W)
            y = y.permute(0, 1, 4, 5, 2, 3).reshape(batch_size, channels * self.num_basis, seq_len, height, width)
        else:  # (N, C, S)
            # Reshape to (N, C, num_basis, S)
            y = y.reshape(batch_size, channels, self.num_basis, seq_len)
            # Permute to (N, C*num_basis, S)
            y = y.permute(0, 1, 2, 3).reshape(batch_size, channels * self.num_basis, seq_len)

        # Apply chomping if specified
        if self.chomp is not None and self.chomp > 0:
            if input_dim == 5:  # (N, C*num_basis, S, H, W)
                y = y[:, :, self.chomp:, :, :]
            else:  # (N, C*num_basis, S)
                y = y[:, :, self.chomp:]

        return y
    

    def plot_basis_functions(self, ax=None):
        """
        Plot basis functions for each covariate.
        
        Args:
            basis_dict (dict): Dictionary of basis functions
        """
        basis = self.basis.cpu().numpy()

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5, 4))
        else:
            fig = ax.figure

        for j in range(basis.shape[0]):
            ax.plot(basis[j, 0, :], label=f"Basis {j+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Lag (bins)")
        plt.tight_layout()
        return fig


class AffineAdapter(nn.Module):
    r"""
    Efficient spatial adapter with optional Gaussian pre-blur.

    Accepts                                  Returns
    ------------------------------           ------------------------------
    (B,C,H,W)     → (B,C,g,g)
    (B,C,S,H,W)   → (B,C,S,g,g)              g = grid_size
    """

    # --------------------------------------------------------------------- #
    def __init__(self, init_sigma=1.0, grid_size=25, transform="scale"):
        super().__init__()
        self.grid_size = grid_size
        self.transform = transform

        # ---- learnable parameters --------------------------------------- #
        # residual σ ≥ 0  (learned)
        self.log_sigma = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(init_sigma)))
        )

        if transform == "scale":
            # independent x / y scales, initialised to 1
            self.log_scale = nn.Parameter(torch.zeros(2))
        elif transform == "affine":
            # 2×3 affine matrix, initialised to identity
            self.affine = nn.Parameter(torch.eye(2, 3))
        else:
            raise ValueError("transform must be 'scale' or 'affine'")

        # ---- static base grid  (-1 … +1) -------------------------------- #
        lin = torch.linspace(-1.0, 1.0, grid_size)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        base = torch.stack((gx, gy), -1)  # [g,g,2]
        self.register_buffer("base_grid", base, persistent=False)

    # --------------------------------------------------------------------- #
    #                      Gaussian blur (separable)                         #
    # --------------------------------------------------------------------- #
    @staticmethod
    @lru_cache(maxsize=32)
    def _kernel_1d(k, sigma, device):
        """Return 1-D Gaussian, shape [1,1,k] (channels added later)."""
        x = torch.arange(-k // 2, k // 2 + 1, device=device)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g /= g.sum()
        return g.view(1, 1, -1)  # [1,1,k]

    @disable
    def _blur(self, x, sigma: torch.Tensor):
        """Depth-wise separable blur, differentiable in σ."""
        # if sigma.item() < 1e-3:                # skip if no blur needed
        if (sigma < 1e-3).all(): # should avoid dynamo warning about the graph breaking
            return x

        B, C, H, W = x.shape
        k = int(round(sigma.item() * 6)) | 1   # 6σ rule, ensure odd
        g = self._kernel_1d(k, sigma.item(), x.device)

        # Horizontal
        x = F.conv2d(
            x,
            g.repeat(C, 1, 1).unsqueeze(3),
            padding=(k // 2, 0),
            groups=C,
        )
        # Vertical
        x = F.conv2d(
            x,
            g.repeat(C, 1, 1).unsqueeze(2),
            padding=(0, k // 2),
            groups=C,
        )
        return x

    # --------------------------------------------------------------------- #
    #                   Minimum σ from sampling theory                       #
    # --------------------------------------------------------------------- #
    def _sigma_from_grid(self):
        r"""
        σ_min  =  0   if the smallest grid-to-pixel scale ≥ 1 (upsampling)
               =  0.44 · (1/s - 1)   otherwise,   where s is that scale.
        """
        if self.transform == "scale":
            scale = F.softplus(self.log_scale)      # (2,)
            s_min = scale.min()
        else:  # affine: singular values of A (top-left 2×2)
            A = self.affine[:, :2]                  # [2,2]
            s_min = torch.linalg.svdvals(A).min()

        if s_min >= 1.0:        # upsample / equal ⇒ no alias risk
            return torch.tensor(0.0, device=s_min.device)
        return 0.44 * (1.0 / s_min - 1.0)

    # --------------------------------------------------------------------- #
    #                           Grid builder                                 #
    # --------------------------------------------------------------------- #
    def _make_grid(self, B, device):
        g = self.base_grid.to(device)  # [g,g,2]

        if self.transform == "scale":
            scale = F.softplus(self.log_scale).to(device)  # (2,)
            g = g * scale
            g = g.unsqueeze(0).expand(B, -1, -1, -1)       # [B,g,g,2]

        else:  # affine
            theta = self.affine.unsqueeze(0).expand(B, -1, -1)  # [B,2,3]
            g = F.affine_grid(
                theta, size=(B, 1, self.grid_size, self.grid_size),
                align_corners=True,
            )
        return g

    # --------------------------------------------------------------------- #
    #                              Forward                                   #
    # --------------------------------------------------------------------- #
    def forward(self, x):
        """
        Antialias-safe resampling followed by optional affine warp.

        • channels-last memory layout for faster CUDA kernels
        • blur skipped if analytically not required
        """
        seq = x.dim() == 5
        if seq:
            B, C, S, H, W = x.shape
            # print(f"Input shape: {x.shape}")
            x = x.reshape(B * C, S, H, W)
        else:
            B, C, H, W = x.shape

        # Apply channels_last format only to 4D tensors
        x = x.contiguous(memory_format=torch.channels_last)

        # ---- compute σ --------------------------------------------------- #
        sigma_min = self._sigma_from_grid()
        sigma_learn = F.softplus(self.log_sigma)
        sigma = torch.sqrt(sigma_min ** 2 + sigma_learn ** 2)

        # ---- blur -------------------------------------------------------- #
        x = self._blur(x, sigma)

        # ---- resample ---------------------------------------------------- #
        grid = self._make_grid(x.size(0), x.device)
        x = F.grid_sample(
            x, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )

        # ---- restore sequence dim --------------------------------------- #
        if seq:
            x = x.view(B, C, S, self.grid_size, self.grid_size)
        else:
            x = x.view(B, C, self.grid_size, self.grid_size)

        return x


class LearnableTemporalConv(nn.Module):
    """
    Learnable temporal convolution frontend that reduces temporal lags while learning
    temporal features. Replaces fixed temporal basis with learned temporal kernels.

    This module applies causal temporal convolution to compress the temporal dimension
    using a specified kernel size, learning optimal temporal features in the process.

    Args:
        kernel_size (int): Temporal kernel size (e.g., 16). Determines how much temporal compression occurs.
        num_channels (int): Number of learned temporal channels/features (e.g., 4, 6, 8)
        init_type (str): Initialization type for temporal kernels:
            - 'gaussian_derivatives': Initialize with Gaussian and its derivatives
            - 'random': Random initialization
            - 'identity': Identity-like initialization
        causal (bool): Whether to ensure causal convolution (default: True)
        bias (bool): Whether to use bias in convolution (default: False)

    Input shape: (N, C, T, H, W) where T is any sequence length
    Output shape: (N, C*num_channels, T_out, H, W) where T_out = T - kernel_size + 1 (for valid conv)
    """

    def __init__(self,
                 kernel_size=16,
                 num_channels=6,
                 init_type='gaussian_derivatives',
                 causal=True,
                 bias=False,
                 **kwargs):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.causal = causal
        self.init_type = init_type

        # Create the temporal convolution layer
        # This will be applied to each spatial location independently
        from .conv_layers import StandardConv
        self.temporal_conv = StandardConv(
            dim=1,
            in_channels=1,  # Process each input channel separately
            out_channels=num_channels,
            kernel_size=self.kernel_size,
            bias=bias,
            padding=0,  # Valid convolution for causal behavior
            **kwargs
        )

        # Initialize the temporal kernels
        self._initialize_kernels()

    def _initialize_kernels(self):
        """Initialize temporal convolution kernels."""
        with torch.no_grad():
            if self.init_type == 'gaussian_derivatives':
                self._init_gaussian_derivatives()
            elif self.init_type == 'random':
                # Default PyTorch initialization is already random
                pass
            elif self.init_type == 'identity':
                self._init_identity()
            else:
                raise ValueError(f"Unknown init_type: {self.init_type}")

    def _init_gaussian_derivatives(self):
        """Initialize kernels with Gaussian and its derivatives."""
        kernel_size = self.kernel_size
        num_channels = self.num_channels

        # Create time axis centered at 0
        t = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)

        # Standard deviation for Gaussian (adjust as needed)
        sigma = kernel_size / 6.0  # 6-sigma rule

        kernels = []

        for i in range(num_channels):
            if i == 0:
                # 0th derivative: Gaussian
                kernel = torch.exp(-0.5 * (t / sigma) ** 2)
            elif i == 1:
                # 1st derivative: -t * Gaussian
                kernel = -t / (sigma ** 2) * torch.exp(-0.5 * (t / sigma) ** 2)
            elif i == 2:
                # 2nd derivative: (t^2/sigma^2 - 1) * Gaussian
                kernel = (t**2 / sigma**2 - 1) / sigma**2 * torch.exp(-0.5 * (t / sigma) ** 2)
            else:
                # Higher derivatives or random for additional channels
                # Use Hermite polynomials or random initialization
                kernel = torch.randn(kernel_size) * torch.exp(-0.5 * (t / sigma) ** 2)

            # Normalize kernel
            kernel = kernel / torch.norm(kernel)
            kernels.append(kernel)

        # Stack kernels and assign to conv layer
        kernel_tensor = torch.stack(kernels, dim=0).unsqueeze(1)  # [num_channels, 1, kernel_size]
        self.temporal_conv.weight.data = kernel_tensor

    def _init_identity(self):
        """Initialize with identity-like kernels."""
        kernel_size = self.kernel_size
        num_channels = self.num_channels

        # Create identity-like kernels with slight variations
        kernels = []
        for i in range(num_channels):
            kernel = torch.zeros(kernel_size)
            # Place peak at different positions
            peak_pos = min(i, kernel_size - 1)
            kernel[peak_pos] = 1.0
            kernels.append(kernel)

        kernel_tensor = torch.stack(kernels, dim=0).unsqueeze(1)
        # Access the underlying conv parameter, not the property
        self.temporal_conv.conv.weight.data = kernel_tensor

    def forward(self, x):
        """
        Apply learnable temporal convolution.

        Args:
            x: Input tensor with shape (N, C, T, H, W) where T is any sequence length

        Returns:
            Output tensor with shape (N, C*num_channels, T_out, H, W)
            where T_out = T - kernel_size + 1 (for valid convolution)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (N, C, T, H, W), got {x.dim()}D")

        batch_size, in_channels, seq_len, height, width = x.shape

        # Check that we have enough temporal samples for the kernel
        if seq_len < self.kernel_size:
            raise ValueError(f"Input sequence length {seq_len} is smaller than kernel_size {self.kernel_size}")

        # Calculate output sequence length
        output_seq_len = seq_len - self.kernel_size + 1

        # Reshape for temporal convolution: (N*C*H*W, 1, T)
        x_reshaped = x.permute(0, 1, 3, 4, 2).reshape(-1, 1, seq_len)

        # Apply temporal convolution
        # Output shape: (N*C*H*W, num_channels, T_out)
        y = self.temporal_conv(x_reshaped)

        # Reshape back to 5D: (N, C, H, W, num_channels, T_out)
        y = y.reshape(batch_size, in_channels, height, width, self.num_channels, output_seq_len)

        # Permute to (N, C*num_channels, T_out, H, W)
        y = y.permute(0, 1, 4, 5, 2, 3).reshape(
            batch_size, in_channels * self.num_channels, output_seq_len, height, width
        )

        return y

    def get_output_channels(self, input_channels=1):
        """Get number of output channels given input channels."""
        return input_channels * self.num_channels

    def plot_kernels(self, ax=None):
        """Plot the learned temporal kernels."""
        # Use the weight property to get anti-aliased weights for plotting
        kernels = self.temporal_conv.weight.detach().cpu().numpy()  # [num_channels, 1, kernel_size]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.figure

        for i in range(self.num_channels):
            kernel = kernels[i, 0, :]  # [kernel_size]
            ax.plot(kernel, label=f'Channel {i+1}')

        ax.set_xlabel('Temporal Lag')
        ax.set_ylabel('Kernel Weight')
        ax.set_title('Learned Temporal Kernels')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
