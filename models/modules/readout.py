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

__all__ = ['BaseFactorizedReadout', 'DynamicGaussianReadout', 'FlattenedLinearReadout']

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


# Create a custom linear readout that handles spatial dimensions
class FlattenedLinearReadout(nn.Module):
    def __init__(self, in_channels, n_units, bias):
        super().__init__()
        self.in_channels = in_channels
        self.n_units = n_units
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, n_units, bias=bias)
        self.bias = self.fc.bias if bias else None

    def forward(self, x):
        # Handle 5D input (N, C, S, H, W)
        if x.dim() == 5:
            x = x[:, :, -1]  # Take last time step -> (N, C, H, W)

        # Only pool if spatial dimensions are > 1x1
        if x.shape[-2:] != (1, 1):
            x = self.adaptive_pool(x)  # -> (N, C, 1, 1)

        x = torch.flatten(x, 1)    # -> (N, C)
        return self.fc(x)