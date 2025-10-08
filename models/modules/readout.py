"""
Readout modules for DataYatesV1.

This module contains readout components for neural network models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
from matplotlib.patches import Ellipse
from .norm_act_pool import get_activation_layer

__all__ = ['BaseFactorizedReadout', 'DynamicGaussianReadout', 'DynamicGaussianReadoutEI', 'DynamicGaussianSN']

# --- BaseFactorizedReadout and DynamicGaussianReadout ---
# (Assuming these are largely okay, minor adjustments for consistency if needed)
class BaseFactorizedReadout(nn.Module):
    def __init__(self, in_channels, n_units, bias=True):
        super().__init__()
        self.n_units = n_units
        self.in_channels = in_channels
        self.features = nn.Conv2d(in_channels, n_units, kernel_size=1, bias=False) # 1x1 conv for feature mapping
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_units))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement forward method.")

    def get_spatial_weights(self):
        raise NotImplementedError("Subclasses should implement get_spatial_weights method.")

    def plot_weights(self, ellipse=True):
        """
        Plot feature weights and spatial readouts.

        Args:
            ellipse (bool): If True, plot Gaussian ellipse contours instead of spatial masks.
                          Only works for Gaussian readouts. Defaults to True.
        """
        # Input x for this readout is (N, C_in, H, W) or (N, C_in, S, H, W)
        # If 5D, DynamicGaussianReadout takes last time step.
        # Feature weights are (n_units, C_in, 1, 1)
        feature_weights = self.features.weight.detach().cpu().squeeze().numpy() # (n_units, C_in)

        # Check if this is a Gaussian readout that supports ellipse plotting
        has_gaussian_params = hasattr(self, 'mean') and hasattr(self, 'std') and hasattr(self, 'theta')

        if ellipse and not has_gaussian_params:
            warnings.warn("ellipse=True only supported for Gaussian readouts. Falling back to spatial masks.")
            ellipse = False

        if not ellipse:
            try:
                spatial_weights = self.get_spatial_weights().detach().cpu().numpy() # (n_units, H, W)
            except Exception as e:
                warnings.warn(f"Could not get spatial weights for plotting (maybe forward pass needed, or H,W not set?): {e}")
                return None, None

            if spatial_weights is None or spatial_weights.size == 0:
                warnings.warn(f"Spatial weights are empty or not computed, cannot plot.")
                return None, None

            n_units_spatial, H, W = spatial_weights.shape
        else:
            # For ellipse plotting, we need cached dimensions
            if not hasattr(self, '_cached_H') or not hasattr(self, '_cached_W') or self._cached_H is None or self._cached_W is None:
                warnings.warn("No cached spatial dimensions found. Call forward pass first or use ellipse=False.")
                return None, None
            H, W = self._cached_H, self._cached_W
            n_units_spatial = self.n_units

        n_units_feat, in_channels_feat = feature_weights.shape if feature_weights.ndim == 2 else (feature_weights.shape[0], 1)

        if n_units_spatial != self.n_units or n_units_feat != self.n_units:
             warnings.warn(f"Mismatch in n_units for plotting. Spatial: {n_units_spatial}, Feat: {n_units_feat}, Expected: {self.n_units}")
             # Fallback if feature_weights is 1D (e.g. n_units=1, C_in > 1 or vice-versa)
             if feature_weights.ndim == 1 and self.n_units == 1: # Case: 1 unit, C_in features
                 feature_weights = feature_weights.reshape(1, -1) # (1, C_in)
             elif feature_weights.ndim == 1 and self.in_channels == 1: # Case: N units, 1 C_in
                  feature_weights = feature_weights.reshape(-1, 1) # (N_units, 1)


        fig, axs = plt.subplots(2, 1, figsize=(max(8, self.n_units * 0.6), 10), squeeze=False)
        axs = axs.ravel()


        # Plot feature weights
        if feature_weights.ndim == 2:
            feat_max = np.abs(feature_weights).max()
            im_feat = axs[0].imshow(feature_weights.T, cmap='coolwarm', interpolation='none', 
                                    vmin=-feat_max if feat_max > 0 else -1, 
                                    vmax=feat_max if feat_max > 0 else 1, aspect='auto')
            axs[0].set_xlabel('Unit Index')
            axs[0].set_ylabel(f'Input Channel ({self.in_channels})')
            plt.colorbar(im_feat, ax=axs[0], fraction=0.046, pad=0.04)
        else:
            axs[0].text(0.5, 0.5, "Feature weights not 2D, cannot plot heatmap.", ha='center', va='center')
        axs[0].set_title('Feature Weights (Channels to Units)')


        # Plot spatial weights (Gaussian masks or ellipses)
        n_cols_spatial = int(np.ceil(np.sqrt(self.n_units)))
        n_rows_spatial = int(np.ceil(self.n_units / n_cols_spatial))

        # Create a new figure for spatial weights if needed, or use axs[1]
        # For simplicity, we create a grid in axs[1]
        axs[1].clear() # Clear previous content if any

        # Create subplots for each unit's spatial weight
        if H > 0 and W > 0:
            spatial_fig_size_w = n_cols_spatial * 2.5
            spatial_fig_size_h = n_rows_spatial * 2.5
            # Create a new figure for spatial weights for clarity
            fig_spatial, axs_spatial_grid = plt.subplots(n_rows_spatial, n_cols_spatial,
                                                        figsize=(spatial_fig_size_w, spatial_fig_size_h),
                                                        squeeze=False)

            if ellipse:
                fig_spatial.suptitle(f'Spatial Readouts (Gaussian Ellipses - {H}x{W})', fontsize=14)
                self._plot_ellipses(axs_spatial_grid, n_cols_spatial, H, W)
            else:
                fig_spatial.suptitle(f'Spatial Weights (Unit Masks - {H}x{W})', fontsize=14)
                spatial_max_val = np.abs(spatial_weights).max()
                for i in range(self.n_units):
                    r, c = i // n_cols_spatial, i % n_cols_spatial
                    ax_sp = axs_spatial_grid[r, c]
                    if i < n_units_spatial: # Check if spatial weight exists for this unit
                        ax_sp.imshow(spatial_weights[i], cmap='coolwarm',
                                    vmin=-spatial_max_val if spatial_max_val > 0 else -1,
                                    vmax=spatial_max_val if spatial_max_val > 0 else 1)
                        ax_sp.set_title(f'Unit {i}')
                    ax_sp.axis('off')

            for i in range(self.n_units, n_rows_spatial * n_cols_spatial): # Turn off unused subplots
                r, c = i // n_cols_spatial, i % n_cols_spatial
                axs_spatial_grid[r,c].axis('off')

            fig_spatial.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

            plot_type = "ellipse contours" if ellipse else "spatial masks"
            axs[1].text(0.5, 0.5, f"Spatial readouts plotted as {plot_type} in separate figure.", ha='center', va='center')

        else:
            axs[1].text(0.5, 0.5, "Spatial weights H or W is 0, cannot plot.", ha='center', va='center')
        
        fig.tight_layout()
        return fig, axs # Potentially return fig_spatial as well, or handle display outside

    def _plot_ellipses(self, axs_spatial_grid, n_cols_spatial, H, W):
        """
        Helper method to plot Gaussian ellipse contours for spatial readouts.

        Args:
            axs_spatial_grid: Grid of matplotlib axes for plotting
            n_cols_spatial: Number of columns in the spatial grid
            H, W: Spatial dimensions
        """
        # Convert grid coordinates to match image coordinates
        # Grid coordinates: y from -(H-1)/2 to (H-1)/2, x from -(W-1)/2 to (W-1)/2
        # Image coordinates: y from 0 to H-1, x from 0 to W-1

        for i in range(self.n_units):
            r, c = i // n_cols_spatial, i % n_cols_spatial
            ax_sp = axs_spatial_grid[r, c]

            # Get Gaussian parameters for this unit
            mean_y, mean_x = self.mean[i].detach().cpu().numpy()
            std_y, std_x = self.std[i].detach().cpu().numpy()
            theta = self.theta[i].detach().cpu().numpy()

            # Convert from grid coordinates to image coordinates
            center_y = mean_y + (H - 1) / 2
            center_x = mean_x + (W - 1) / 2

            # Create ellipse patch
            # Note: matplotlib Ellipse uses width/height as full diameters, not radii
            # We use 2*std for the diameter to show ~68% of the Gaussian
            ellipse = Ellipse((center_x, center_y),
                            width=4*std_x,  # 2*std on each side
                            height=4*std_y,  # 2*std on each side
                            angle=np.degrees(theta),
                            fill=False,
                            edgecolor='red',
                            linewidth=2)

            ax_sp.add_patch(ellipse)

            # Set axis limits and properties
            ax_sp.set_xlim(0, W-1)
            ax_sp.set_ylim(H-1, 0)  # Invert y-axis to match image coordinates
            ax_sp.set_aspect('equal')
            ax_sp.set_title(f'Unit {i}')
            ax_sp.grid(True, alpha=0.3)

            # Add center point
            ax_sp.plot(center_x, center_y, 'ro', markersize=4)


class DynamicGaussianReadout(BaseFactorizedReadout):
    def __init__(self, in_channels, n_units, bias=True, initial_std=5.0, initial_mean_scale=0.0):
        super().__init__(in_channels, n_units, bias)
        self.mean = nn.Parameter(initial_mean_scale * torch.randn(n_units, 2)) 
        self.std = nn.Parameter(torch.ones(n_units, 2) * initial_std)
        self.theta = nn.Parameter(torch.zeros(n_units)) 

        self._cached_grid = None
        self._cached_H = None
        self._cached_W = None

    def _create_grid(self, H, W, device):
        y_coords = torch.linspace(-(H - 1) / 2., (H - 1) / 2., H, device=device)
        x_coords = torch.linspace(-(W - 1) / 2., (W - 1) / 2., W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1) # (H, W, 2)
        # print(f"Created grid for H={H}, W={W}") # Optional: for debugging
        return grid

    def compute_gaussian_mask(self, H, W, device):
        if H != self._cached_H or W != self._cached_W or \
           self._cached_grid is None or self._cached_grid.device != device:
            if H <=0 or W <=0: # Cannot create grid if H or W is zero or negative
                warnings.warn(f"Cannot compute Gaussian mask for H={H}, W={W}. Returning empty tensor.")
                return torch.empty(self.n_units, H if H > 0 else 1, W if W > 0 else 1, device=device) # Return something sensible
            self._cached_grid = self._create_grid(H, W, device)
            self._cached_H = H
            self._cached_W = W
        grid = self._cached_grid

        std_clamped = self.std.clamp(min=1e-3) 
        mean_expanded = self.mean.unsqueeze(1).unsqueeze(1) # (n_units, 1, 1, 2)
        std_expanded = std_clamped.unsqueeze(1).unsqueeze(1) # (n_units, 1, 1, 2)
        grid_expanded = grid.unsqueeze(0) # (1, H, W, 2)

        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        R_y = torch.stack([cos_theta, -sin_theta], dim=-1) # Part of Rot matrix row 1
        R_x = torch.stack([sin_theta,  cos_theta], dim=-1) # Part of Rot matrix row 2
        R = torch.stack([R_y, R_x], dim=-2) # (n_units, 2, 2)

        centered_grid = grid_expanded - mean_expanded # (n_units, H, W, 2)
        rotated_grid = torch.einsum('nhwi,nij->nhwj', centered_grid, R)

        exponent = -0.5 * torch.sum((rotated_grid / std_expanded) ** 2, dim=-1) # (n_units, H, W)
        gaussian_mask = torch.exp(exponent)
        
        # Normalize sum over H,W to 1 for each unit's mask
        normalization_factor = gaussian_mask.sum(dim=(-1, -2), keepdim=True)
        gaussian_mask = gaussian_mask / (normalization_factor + 1e-8) # Add epsilon for stability
        return gaussian_mask

    def forward(self, x):
        # x shape: (N, C_in, S, H, W) or (N, C_in, H, W)
        if x.dim() == 5:
            # Using last time step if sequence is provided
            # warnings.warn("DynamicGaussianReadout received 5D tensor, using last element along sequence dim (dim 2).")
            x = x[:, :, -1, :, :] # (N, C_in, H, W)
        elif x.dim() != 4:
            raise ValueError(f"DynamicGaussianReadout expects 4D (N,C,H,W) or 5D (N,C,S,H,W) input, got {x.dim()}D")

        N, C_in, H, W = x.shape
        device = x.device

        feat = self.features(x) # (N, n_units, H, W)
        
        if H <= 0 or W <= 0:
            # If spatial dimensions are zero or invalid, cannot compute mask or pool.
            # Output zeros of the expected shape (N, n_units).
            warnings.warn(f"DynamicGaussianReadout received input with H={H}, W={W}. Outputting zeros.")
            out = torch.zeros(N, self.n_units, device=device, dtype=feat.dtype)
        else:
            gaussian_mask = self.compute_gaussian_mask(H, W, device) # (n_units, H, W)
            # Apply masks: (N, n_units, H, W) * (1, n_units, H, W) -> sum over H,W
            out = (feat * gaussian_mask.unsqueeze(0)).sum(dim=(-2, -1)) # (N, n_units)

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_spatial_weights(self):
        if self._cached_grid is None or self._cached_H is None or self._cached_W is None:
            warnings.warn("Grid not cached in DynamicGaussianReadout. Call forward pass with valid H,W first.")
            return torch.empty(0) 
        if self._cached_H <= 0 or self._cached_W <=0:
            warnings.warn(f"Cannot get spatial weights for H={self._cached_H}, W={self._cached_W}.")
            return torch.empty(0)
            
        mask = self.compute_gaussian_mask(self._cached_H, self._cached_W, self._cached_grid.device)
        return mask # Return on whatever device it was computed, detach if sending to CPU later


class DynamicGaussianReadoutEI(BaseFactorizedReadout):
    """
    Excitatory-Inhibitory version of DynamicGaussianReadout.

    Uses separate excitatory and inhibitory Gaussian filters and feature weights.
    Feature weights are constrained to be positive using a configurable activation function.
    Output is computed as: (excitatory_space * excitatory_features) - (inhibitory_space * inhibitory_features)
    """

    def __init__(self, in_channels, n_units, bias=True, initial_std_ex=3.0, initial_std_inh=6.0,
                 weight_constraint_fn=None, frac_inhib=None, **kwargs):
        # Don't call super().__init__ because we need to replace self.features
        nn.Module.__init__(self)
        self.n_units = n_units
        self.in_channels = in_channels

        if frac_inhib is None:
            n_ex = n_units
            n_inh = n_units
            self.register_buffer('ex_units', torch.arange(n_ex))
            self.register_buffer('inh_units', torch.arange(n_inh))
        else:
            n_inh = int(frac_inhib*n_units)
            n_ex = n_units - n_inh
            self.register_buffer('ex_units', torch.arange(n_ex))
            self.register_buffer('inh_units', torch.arange(n_inh) + n_ex)

        # Separate feature mappings with hidden weights for positive constraints
        self._features_ex_weight = nn.Parameter(F.normalize(torch.rand(n_ex, in_channels, 1, 1), p=2, dim=1))
        self._features_inh_weight = nn.Parameter(F.normalize(torch.rand(n_inh, in_channels, 1, 1), p=2, dim=1))

        # Weight constraint function (defaults to ReLU)
        self.weight_constraint_fn = get_activation_layer(weight_constraint_fn) if weight_constraint_fn is not None else torch.relu

        # Single bias for final output
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_units))
        else:
            self.register_parameter('bias', None)

        # Separate Gaussian parameters for excitatory and inhibitory
        self.mean_ex = nn.Parameter(torch.zeros(n_units, 2))
        self.std_ex = nn.Parameter(torch.ones(n_units, 2) * initial_std_ex)
        self.theta_ex = nn.Parameter(torch.zeros(n_units))

        self.mean_inh = nn.Parameter(torch.zeros(n_units, 2))
        self.std_inh = nn.Parameter(torch.ones(n_units, 2) * initial_std_inh)
        self.theta_inh = nn.Parameter(torch.zeros(n_units))

        # Caching for grid computation
        self._cached_grid = None
        self._cached_H = None
        self._cached_W = None

    @property
    def features_ex_weight(self):
        """Positive-constrained excitatory feature weights."""
        return self.weight_constraint_fn(self._features_ex_weight)

    @property
    def features_inh_weight(self):
        """Positive-constrained inhibitory feature weights."""
        return self.weight_constraint_fn(self._features_inh_weight)

    def _create_grid(self, H, W, device):
        """Create coordinate grid for Gaussian computation."""
        y_coords = torch.linspace(-(H - 1) / 2., (H - 1) / 2., H, device=device)
        x_coords = torch.linspace(-(W - 1) / 2., (W - 1) / 2., W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1)  # (H, W, 2)
        return grid

    def compute_gaussian_mask(self, H, W, device, pathway='ex'):
        """
        Compute Gaussian mask for either excitatory or inhibitory pathway.

        Args:
            H, W: spatial dimensions
            device: torch device
            pathway: 'ex' for excitatory, 'inh' for inhibitory
        """
        if H != self._cached_H or W != self._cached_W or \
           self._cached_grid is None or self._cached_grid.device != device:
            if H <= 0 or W <= 0:
                warnings.warn(f"Cannot compute Gaussian mask for H={H}, W={W}. Returning empty tensor.")
                return torch.empty(self.n_units, H if H > 0 else 1, W if W > 0 else 1, device=device)
            self._cached_grid = self._create_grid(H, W, device)
            self._cached_H = H
            self._cached_W = W

        grid = self._cached_grid

        # Select parameters based on pathway
        if pathway == 'ex':
            mean, std, theta = self.mean_ex, self.std_ex, self.theta_ex
        elif pathway == 'inh':
            mean, std, theta = self.mean_inh, self.std_inh, self.theta_inh
        else:
            raise ValueError(f"pathway must be 'ex' or 'inh', got {pathway}")

        std_clamped = std.clamp(min=1e-3)
        mean_expanded = mean.unsqueeze(1).unsqueeze(1)  # (n_units, 1, 1, 2)
        std_expanded = std_clamped.unsqueeze(1).unsqueeze(1)  # (n_units, 1, 1, 2)
        grid_expanded = grid.unsqueeze(0)  # (1, H, W, 2)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_y = torch.stack([cos_theta, -sin_theta], dim=-1)
        R_x = torch.stack([sin_theta, cos_theta], dim=-1)
        R = torch.stack([R_y, R_x], dim=-2)  # (n_units, 2, 2)

        centered_grid = grid_expanded - mean_expanded  # (n_units, H, W, 2)
        rotated_grid = torch.einsum('nhwi,nij->nhwj', centered_grid, R)

        exponent = -0.5 * torch.sum((rotated_grid / std_expanded) ** 2, dim=-1)  # (n_units, H, W)
        gaussian_mask = torch.exp(exponent)

        # Normalize sum over H,W to 1 for each unit's mask
        normalization_factor = gaussian_mask.sum(dim=(-1, -2), keepdim=True)
        gaussian_mask = gaussian_mask / (normalization_factor + 1e-8)
        return gaussian_mask

    def forward(self, x):
        """
        Forward pass computing (excitatory_space * excitatory_features) - (inhibitory_space * inhibitory_features)

        Args:
            x: Input tensor of shape (N, C_in, H, W) or (N, C_in, S, H, W)
        """
        # Handle 5D input (sequence dimension)
        if x.dim() == 5:
            x = x[:, :, -1, :, :]  # (N, C_in, H, W)
        elif x.dim() != 4:
            raise ValueError(f"DynamicGaussianReadoutEI expects 4D (N,C,H,W) or 5D (N,C,S,H,W) input, got {x.dim()}D")

        N, C_in, H, W = x.shape
        device = x.device

        if H <= 0 or W <= 0:
            warnings.warn(f"DynamicGaussianReadoutEI received input with H={H}, W={W}. Outputting zeros.")
            out = torch.zeros(N, self.n_units, device=device, dtype=x.dtype)
        else:
            # Apply positive-constrained feature weights using functional conv2d
            feat_ex = torch.nn.functional.conv2d(x, self.features_ex_weight, bias=None)  # (N, n_units, H, W)
            feat_inh = torch.nn.functional.conv2d(x, self.features_inh_weight, bias=None)  # (N, n_units, H, W)

            # Compute Gaussian masks for both pathways
            gaussian_mask_ex = self.compute_gaussian_mask(H, W, device, pathway='ex')  # (n_units, H, W)
            gaussian_mask_inh = self.compute_gaussian_mask(H, W, device, pathway='inh')  # (n_units, H, W)

            # Apply masks and sum over spatial dimensions
            out_ex = (feat_ex * gaussian_mask_ex.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)
            out_inh = (feat_inh * gaussian_mask_inh.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)

            # Compute excitatory - inhibitory
            out = out_ex - out_inh  # (N, n_units)

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_spatial_weights(self):
        """
        Return spatial weights for both excitatory and inhibitory pathways.

        Returns:
            dict: {'excitatory': mask_ex, 'inhibitory': mask_inh}
        """
        if self._cached_grid is None or self._cached_H is None or self._cached_W is None:
            warnings.warn("Grid not cached in DynamicGaussianReadoutEI. Call forward pass with valid H,W first.")
            return {'excitatory': torch.empty(0), 'inhibitory': torch.empty(0)}
        if self._cached_H <= 0 or self._cached_W <= 0:
            warnings.warn(f"Cannot get spatial weights for H={self._cached_H}, W={self._cached_W}.")
            return {'excitatory': torch.empty(0), 'inhibitory': torch.empty(0)}

        mask_ex = self.compute_gaussian_mask(self._cached_H, self._cached_W, self._cached_grid.device, pathway='ex')
        mask_inh = self.compute_gaussian_mask(self._cached_H, self._cached_W, self._cached_grid.device, pathway='inh')
        return {'excitatory': mask_ex, 'inhibitory': mask_inh}

    def plot_weights(self, ellipse=True):
        """
        Plot excitatory and inhibitory feature weights and spatial masks with alpha blending.

        Args:
            ellipse (bool): If True, plot Gaussian ellipse contours instead of spatial masks.
                          Defaults to True.

        Excitatory components are shown in red, inhibitory in blue.
        Since weights are positive-constrained, zero weights will be transparent.
        """
        # Get feature weights (positive-constrained)
        feature_weights_ex = self.features_ex_weight.detach().cpu().squeeze().numpy()  # (n_units, C_in)
        feature_weights_inh = self.features_inh_weight.detach().cpu().squeeze().numpy()  # (n_units, C_in)

        # Get spatial dimensions and weights if needed
        if ellipse:
            # For ellipse plotting, we need cached dimensions
            if not hasattr(self, '_cached_H') or not hasattr(self, '_cached_W') or self._cached_H is None or self._cached_W is None:
                warnings.warn("No cached spatial dimensions found. Call forward pass first or use ellipse=False.")
                return None, None
            H, W = self._cached_H, self._cached_W
        else:
            # Get spatial weights for mask plotting
            try:
                spatial_weights_dict = self.get_spatial_weights()
                if isinstance(spatial_weights_dict, dict):
                    spatial_weights_ex = spatial_weights_dict['excitatory'].detach().cpu().numpy()  # (n_units, H, W)
                    spatial_weights_inh = spatial_weights_dict['inhibitory'].detach().cpu().numpy()  # (n_units, H, W)
                else:
                    raise ValueError("get_spatial_weights should return a dictionary for EI readout")
            except Exception as e:
                warnings.warn(f"Could not get spatial weights for plotting: {e}")
                return None, None

            if spatial_weights_ex.size == 0 or spatial_weights_inh.size == 0:
                warnings.warn("Spatial weights are empty, cannot plot.")
                return None, None

            n_units_spatial, H, W = spatial_weights_ex.shape

        # Handle feature weight dimensions
        if feature_weights_ex.ndim == 1:
            if self.n_units == 1:
                feature_weights_ex = feature_weights_ex.reshape(1, -1)
                feature_weights_inh = feature_weights_inh.reshape(1, -1)
            elif self.in_channels == 1:
                feature_weights_ex = feature_weights_ex.reshape(-1, 1)
                feature_weights_inh = feature_weights_inh.reshape(-1, 1)

        # Create main figure with feature weights
        fig, axs = plt.subplots(2, 1, figsize=(7, 10), squeeze=False)
        axs = axs.ravel()

        # Plot overlaid feature weights
        if feature_weights_ex.ndim == 2 and feature_weights_inh.ndim == 2:
            # Normalize weights for better visualization
            feat_max_ex = feature_weights_ex.max()
            feat_max_inh = feature_weights_inh.max()
            feat_max_overall = max(feat_max_ex, feat_max_inh)

            if feat_max_overall > 0:
                # Create RGBA images for alpha blending
                # Excitatory: red channel with alpha based on weight magnitude
                ex_rgba = np.zeros((*feature_weights_ex.T.shape, 4))
                ex_rgba[:, :, 0] = feature_weights_ex.T / feat_max_overall  # Red channel
                ex_rgba[:, :, 3] = feature_weights_ex.T / feat_max_overall  # Alpha channel

                # Inhibitory: blue channel with alpha based on weight magnitude
                inh_rgba = np.zeros((*feature_weights_inh.T.shape, 4))
                inh_rgba[:, :, 2] = feature_weights_inh.T / feat_max_overall  # Blue channel
                inh_rgba[:, :, 3] = feature_weights_inh.T / feat_max_overall  # Alpha channel

                # Plot excitatory (red) and inhibitory (blue) overlaid
                axs[0].imshow(ex_rgba, aspect='auto', interpolation='none')
                axs[1].imshow(inh_rgba, aspect='auto', interpolation='none')

                axs[0].set_xlabel('Unit Index')
                axs[0].set_ylabel(f'Input Channel ({self.in_channels})')

                # Add custom legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', alpha=0.7, label='Excitatory'),
                                 Patch(facecolor='blue', alpha=0.7, label='Inhibitory')]
                axs[0].legend(handles=legend_elements, loc='upper right')
            else:
                axs[0].text(0.5, 0.5, "All feature weights are zero.", ha='center', va='center')
        else:
            axs[0].text(0.5, 0.5, "Feature weights not 2D, cannot plot.", ha='center', va='center')

        axs[0].set_title('Feature Weights: Excitatory (Red) vs Inhibitory (Blue)')

        # Plot overlaid spatial weights or ellipses
        if H > 0 and W > 0:
            n_cols_spatial = int(np.ceil(np.sqrt(self.n_units)))
            n_rows_spatial = int(np.ceil(self.n_units / n_cols_spatial))

            spatial_fig_size_w = n_cols_spatial * 3
            spatial_fig_size_h = n_rows_spatial * 3

            # Create separate figure for spatial weights
            fig_spatial, axs_spatial_grid = plt.subplots(n_rows_spatial, n_cols_spatial,
                                                        figsize=(spatial_fig_size_w, spatial_fig_size_h),
                                                        squeeze=False)

            if ellipse:
                fig_spatial.suptitle(f'Spatial Readouts: Excitatory (Red) vs Inhibitory (Blue) Ellipses - {H}x{W}', fontsize=14)
                self._plot_ellipses_ei(axs_spatial_grid, n_cols_spatial, H, W)
            else:
                fig_spatial.suptitle(f'Spatial Weights: Excitatory (Red) vs Inhibitory (Blue) - {H}x{W}', fontsize=14)

                # Normalize spatial weights
                spatial_max_ex = spatial_weights_ex.max()
                spatial_max_inh = spatial_weights_inh.max()
                spatial_max_overall = max(spatial_max_ex, spatial_max_inh)

                for i in range(self.n_units):
                    r, c = i // n_cols_spatial, i % n_cols_spatial
                    ax_sp = axs_spatial_grid[r, c]

                    if spatial_max_overall > 0:
                        # Create RGBA images for spatial weights
                        ex_spatial_rgba = np.zeros((H, W, 4))
                        ex_spatial_rgba[:, :, 0] = spatial_weights_ex[i] / spatial_max_overall  # Red
                        ex_spatial_rgba[:, :, 3] = spatial_weights_ex[i] / spatial_max_overall  # Alpha

                        inh_spatial_rgba = np.zeros((H, W, 4))
                        inh_spatial_rgba[:, :, 2] = spatial_weights_inh[i] / spatial_max_overall  # Blue
                        inh_spatial_rgba[:, :, 3] = spatial_weights_inh[i] / spatial_max_overall  # Alpha

                        # Overlay excitatory and inhibitory
                        ax_sp.imshow(ex_spatial_rgba, interpolation='bilinear')
                        ax_sp.imshow(inh_spatial_rgba, interpolation='bilinear')

                    ax_sp.set_title(f'Unit {i}')
                    ax_sp.axis('off')

            # Turn off unused subplots
            for i in range(self.n_units, n_rows_spatial * n_cols_spatial):
                r, c = i // n_cols_spatial, i % n_cols_spatial
                axs_spatial_grid[r, c].axis('off')

            # Add legend to spatial figure
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Excitatory'),
                             Patch(facecolor='blue', alpha=0.7, label='Inhibitory')]
            fig_spatial.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))

            fig_spatial.tight_layout(rect=[0, 0, 1, 0.96])

            plot_type = "ellipse contours" if ellipse else "spatial masks"
            axs[1].text(0.5, 0.5, f"Spatial readouts plotted as {plot_type} in separate figure with red/blue overlay.",
                       ha='center', va='center')
        else:
            axs[1].text(0.5, 0.5, "Spatial weights H or W is 0, cannot plot.", ha='center', va='center')

        fig.tight_layout()
        return fig, axs

    def _plot_ellipses_ei(self, axs_spatial_grid, n_cols_spatial, H, W):
        """
        Helper method to plot Gaussian ellipse contours for EI spatial readouts.

        Args:
            axs_spatial_grid: Grid of matplotlib axes for plotting
            n_cols_spatial: Number of columns in the spatial grid
            H, W: Spatial dimensions
        """
        for i in range(self.n_units):
            r, c = i // n_cols_spatial, i % n_cols_spatial
            ax_sp = axs_spatial_grid[r, c]

            # Get Gaussian parameters for excitatory pathway
            mean_y_ex, mean_x_ex = self.mean_ex[i].detach().cpu().numpy()
            std_y_ex, std_x_ex = self.std_ex[i].detach().cpu().numpy()
            theta_ex = self.theta_ex[i].detach().cpu().numpy()

            # Get Gaussian parameters for inhibitory pathway
            mean_y_inh, mean_x_inh = self.mean_inh[i].detach().cpu().numpy()
            std_y_inh, std_x_inh = self.std_inh[i].detach().cpu().numpy()
            theta_inh = self.theta_inh[i].detach().cpu().numpy()

            # Convert from grid coordinates to image coordinates
            center_y_ex = mean_y_ex + (H - 1) / 2
            center_x_ex = mean_x_ex + (W - 1) / 2
            center_y_inh = mean_y_inh + (H - 1) / 2
            center_x_inh = mean_x_inh + (W - 1) / 2

            # Create excitatory ellipse (red)
            ellipse_ex = Ellipse((center_x_ex, center_y_ex),
                               width=4*std_x_ex,  # 2*std on each side
                               height=4*std_y_ex,  # 2*std on each side
                               angle=np.degrees(theta_ex),
                               fill=False,
                               edgecolor='red',
                               linewidth=2,
                               label='Excitatory')

            # Create inhibitory ellipse (blue)
            ellipse_inh = Ellipse((center_x_inh, center_y_inh),
                                width=4*std_x_inh,  # 2*std on each side
                                height=4*std_y_inh,  # 2*std on each side
                                angle=np.degrees(theta_inh),
                                fill=False,
                                edgecolor='blue',
                                linewidth=2,
                                label='Inhibitory')

            ax_sp.add_patch(ellipse_ex)
            ax_sp.add_patch(ellipse_inh)

            # Set axis limits and properties
            ax_sp.set_xlim(0, W-1)
            ax_sp.set_ylim(H-1, 0)  # Invert y-axis to match image coordinates
            ax_sp.set_aspect('equal')
            ax_sp.set_title(f'Unit {i}')
            ax_sp.grid(True, alpha=0.3)

            # Add center points
            ax_sp.plot(center_x_ex, center_y_ex, 'ro', markersize=4)
            ax_sp.plot(center_x_inh, center_y_inh, 'bo', markersize=4)


class DynamicGaussianSN(DynamicGaussianReadoutEI):
    r"""
    DynamicGaussianSN  ──  “stochastic-normalization” readout
        output = exc / (beta + inh + ε + N(0, σ²))

    * beta  – learned, positive (softplus-param) and optionally annealed
    * σ     – learned log-std; noise applied only during .training
    * ε     – fixed 1e-6 to dodge div-by-zero / log underflow
    """

    def __init__(self,
                 in_channels: int,
                 n_units: int,
                 bias: bool = True,
                 initial_beta: float = 0.1,
                 initial_sigma: float = 0.05,
                 min_beta: float = 1e-4,
                 min_sigma: float = 1e-5,
                 **kwargs):
        super().__init__(in_channels, n_units, bias=bias, **kwargs)

        # Softplus-parameterised positives:  val = softplus(raw) + floor
        self._beta_raw  = nn.Parameter(torch.as_tensor(math.log(math.exp(initial_beta) - 1.0)))
        self._sigma_raw = nn.Parameter(torch.as_tensor(math.log(math.exp(initial_sigma) - 1.0)))

        self.register_buffer("_beta_floor",  torch.tensor(min_beta))
        self.register_buffer("_sigma_floor", torch.tensor(min_sigma))
        self.register_buffer("_eps",         torch.tensor(1e-6))

        if self.bias is not None:
            self.bias.data[:] = -5

        self._features_ex_weight.data[:] *= 10

    # convenience accessors --------------------------------------------------
    @property
    def beta(self) -> torch.Tensor:   # shape (1,)  broadcastable
        return F.softplus(self._beta_raw) + self._beta_floor

    @property
    def sigma(self) -> torch.Tensor:  # noise std (>=0)
        return F.softplus(self._sigma_raw) + self._sigma_floor

    # ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (N, C, H, W)  or  (N, C, S, H, W)  – identical to EI parent.

        Returns
        -------
        rate : (N, n_units)
        """
        # EI parent handles dimension checks & masks
        if x.dim() == 5:
            x = x[:, :, -1, :, :]

        N, C, H, W = x.shape
        device = x.device

        # -- 1. feature maps (positive constrained)
        feat_ex  = F.conv2d(x, self.features_ex_weight,  bias=None)
        feat_inh = F.conv2d(x, self.features_inh_weight, bias=None)

        # -- 2. spatial weighting
        mask_ex  = self.compute_gaussian_mask(H, W, device, pathway='ex')   # (U, H, W)
        mask_inh = self.compute_gaussian_mask(H, W, device, pathway='inh')

        ex = (feat_ex  * mask_ex.unsqueeze(0)).sum(dim=(-2, -1))           # (N,U)
        inh = (feat_inh * mask_inh.unsqueeze(0)).sum(dim=(-2, -1)).abs()   # ensure +ve

        # -- 3. stochastic divisor  beta + inh + ε + 𝒩(0,σ²)
        denom = self.beta + inh + self._eps
        if self.training and (self.sigma > 0):
            if self.training:
                noise = torch.randn_like(denom) * self.sigma
            else:
                noise = torch.zeros_like(denom)
                
            denom = torch.clamp(denom + noise, min=self._eps)              # keep >0

        out = ex / denom                                                   # divisive readout

        if self.bias is not None:
            out = out + self.bias

        return out

    # optional helper: linear annealing of beta after every epoch ----------
    def anneal_beta(self, factor: float = 0.95, min_val: float = 1e-4):
        """call from Lightning `on_train_epoch_end` if desired"""
        with torch.no_grad():
            beta_new = torch.clamp(self.beta * factor, min=min_val)
            # invert softplus to update raw param
            self._beta_raw.data.copy_(torch.log(torch.expm1(beta_new - self._beta_floor)))