"""
Polar Readout Module for VisionCore.

This module implements the readout components of the Polar-V1 model:
- GaussianReadout: Gaussian (elliptical) spatial readout per neuron per level
- PolarMultiLevelReadout: Multi-level Gaussian readout for polar features

This handles spatial pooling per neuron across multiple pyramid levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

__all__ = ['PolarMultiLevelReadout', 'GaussianReadout']


# ------------------------- Gaussian Readout ------------------------- #

class GaussianReadout(nn.Module):
    """Gaussian (elliptical) spatial readout per neuron per level.
    Positions & covariances are learned per neuron+level.
    """
    
    def __init__(self, C: int, H: int, W: int, n_neurons: int, initial_std: float = 5.0, initial_mean_scale: float = 0.0):
        super().__init__()
        self.n = n_neurons
        self.C, self.H, self.W = C, H, W

        # Gaussian position parameters - EXACTLY like DynamicGaussianReadout
        self.mu    = nn.Parameter(initial_mean_scale * torch.randn(n_neurons, 2))
        self.logsx = nn.Parameter(torch.log(torch.ones(n_neurons) * initial_std))
        self.logsy = nn.Parameter(torch.log(torch.ones(n_neurons) * initial_std))
        self.rho   = nn.Parameter(torch.zeros(n_neurons))

        # Feature weights - 1x1 conv like BaseFactorizedReadout (no bias in features)
        self.features = nn.Conv2d(C, n_neurons, kernel_size=1, bias=False)

    @staticmethod
    def gaussian_grid(H, W, device, dtype):
        """Create coordinate grid for Gaussian computation."""
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([X, Y], dim=-1)  # [H,W,2]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W] features

        Returns:
            output: [B, n_neurons] readout values
        """
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape

        # Apply feature weights FIRST (like DynamicGaussianReadout)
        feat_weighted = self.features(feat)  # [B, n_neurons, H, W]

        # Compute Gaussian mask
        grid = self.gaussian_grid(H, W, feat.device, feat.dtype)   # [H,W,2]
        X = grid.unsqueeze(0).expand(self.n, H, W, 2)              # [n,H,W,2]
        mu = self.mu[:, None, None, :]                             # [n,1,1,2]
        sx = torch.exp(self.logsx)[:, None, None] + 1e-6           # [n,1,1]
        sy = torch.exp(self.logsy)[:, None, None] + 1e-6
        rho= torch.tanh(self.rho)[:, None, None]

        Xc = X - mu
        A = (Xc[...,0]/sx)**2 + (Xc[...,1]/sy)**2 - 2*rho*(Xc[...,0]/sx)*(Xc[...,1]/sy)
        denom = 2*(1 - rho**2 + 1e-6)
        G = torch.exp(-A/denom)                                    # [n,H,W]

        # Apply Gaussian mask and sum over spatial dimensions
        # feat_weighted: [B, n, H, W], G: [n, H, W]
        out = (feat_weighted * G.unsqueeze(0)).sum(dim=(-2, -1))  # [B, n]
        return out


# ------------------------- Multi-Level Readout ------------------------- #

class PolarMultiLevelReadout(nn.Module):
    """Multi-level Gaussian readout for polar features."""
    
    def __init__(self, config):
        super().__init__()
        self.n_neurons = config['n_units']
        # Match DynamicGaussianReadout defaults EXACTLY
        self.initial_std = config.get('initial_std', 5.0)
        self.initial_mean_scale = config.get('initial_mean_scale', 0.0)

        # Will be lazy-initialized on first forward
        self.readouts = None
        # Initialize bias exactly like DynamicGaussianReadout
        if config.get('bias', True):
            self.bias = nn.Parameter(torch.zeros(self.n_neurons))
        else:
            self.register_parameter('bias', None)
    
    def _lazy_init(self, feats_per_level: List[torch.Tensor]):
        """Initialize readouts based on input feature shapes."""
        if self.readouts is not None:
            return

        n_levels = len(feats_per_level)
        self.readouts = nn.ModuleList()

        # Get device and dtype from first feature
        device = feats_per_level[0].device
        dtype = feats_per_level[0].dtype

        for l, feat in enumerate(feats_per_level):
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
            readout = GaussianReadout(C=C, H=H, W=W, n_neurons=self.n_neurons,
                                     initial_std=self.initial_std,
                                     initial_mean_scale=self.initial_mean_scale)
            # Move to same device as features
            readout = readout.to(device=device)
            self.readouts.append(readout)
    
    def forward(self, feats_per_level: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats_per_level: List of [B, C_l, H_l, W_l] features per level
        
        Returns:
            output: [B, n_neurons] readout values
        """
        self._lazy_init(feats_per_level)
        
        # Sum across levels
        parts = []
        for l, feat in enumerate(feats_per_level):
            parts.append(self.readouts[l](feat))

        output = torch.stack(parts, dim=-1).sum(dim=-1)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
