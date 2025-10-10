"""
Polar ConvNet Module for VisionCore.

This module implements the spatial processing components of the Polar-V1 model:
- PyramidAdapter: Laplacian pyramid decomposition
- QuadratureFilterBank2D: Even/odd spatial filters per level
- PolarDecompose: Amplitude + phase extraction
- PolarConvNet: Main convnet class

This is PURE spatial processing - no behavior, no temporal dynamics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from einops import rearrange

__all__ = ['PolarConvNet', 'PyramidAdapter', 'QuadratureFilterBank2D', 'PolarDecompose']


# ------------------------- Utils ------------------------- #

def complex_from_even_odd(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Create complex tensor from even/odd components."""
    return torch.complex(e, o)


def safe_unit_complex(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize complex tensor to unit magnitude: w / (|w|+eps)."""
    mag = torch.clamp(torch.abs(w), min=eps)
    return w / mag


# ------------------------- Pyramid Adapter ------------------------- #

class PyramidAdapter(nn.Module):
    """Wrap a plenoptic.simulate.LaplacianPyramid to produce a `levels` list
    shaped [B, C_in, T, H_l, W_l] suitable for PolarConvNet.
    """
    
    def __init__(self, J: int):
        super().__init__()
        try:
            from plenoptic.simulate import LaplacianPyramid
            self.lpyr = LaplacianPyramid(J)
            self.J = J
        except ImportError:
            raise ImportError("plenoptic package is required for PyramidAdapter. Install with: pip install plenoptic")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, C_in, T, H, W] stimulus
            
        Returns:
            levels: List of [B, C_in, T, H_l, W_l] per level
        """
        B, C, T, H, W = x.shape
        x_bt = rearrange(x, 'B C T H W -> (B T) C H W')
        levels_bt = self.lpyr(x_bt)  # list of [(B*T), C, H_l, W_l]
        levels = [
            rearrange(L, '(B T) C H W -> B C T H W', B=B, T=T)
            for L in levels_bt
        ]
        return levels


# ------------------------- Quadrature Filter Bank ------------------------- #

class QuadratureFilterBank2D(nn.Module):
    """Shared paired (even/odd) 2D filters applied per frame on each level.

    Args:
        in_ch: channels per level coming from the Laplacian pyramid (per stream)
        pairs: number of quadrature *pairs* M to learn (even + odd for each)
        kernel: odd kernel size int (k x k)
        share_across_levels: if True, same weights for all levels (recommended)
    Output per level l:
        even_l: [B, M, T, H_l, W_l]
        odd_l : [B, M, T, H_l, W_l]
    """
    
    def __init__(self, in_ch: int, pairs: int, kernel: int = 7, share_across_levels: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.pairs = pairs
        self.kernel = kernel
        padding = kernel // 2
        
        # Implement as depthwise (per input channel) + 1x1 mixing to pairs.
        # First stage: depthwise conv to keep locality & low params.
        self.dw_even = nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False)
        self.dw_odd  = nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False)
        
        # Second stage: 1x1 mixing from in_ch -> pairs
        self.mix_even = nn.Conv2d(in_ch, pairs, 1, bias=True)
        self.mix_odd  = nn.Conv2d(in_ch, pairs, 1, bias=True)
        
        # Init even/odd as approximate Gabor quadrature (random phase shift)
        nn.init.kaiming_uniform_(self.dw_even.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dw_odd.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.mix_even.weight)
        nn.init.xavier_uniform_(self.mix_odd.weight)

    def forward_level(self, x_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single pyramid level."""
        B, C, T, H, W = x_l.shape
        x_flat = x_l.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # [B*T, C, H, W]
        e = self.mix_even(self.dw_even(x_flat))  # [B*T, M, H, W]
        o = self.mix_odd (self.dw_odd (x_flat))
        e = e.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)  # [B, M, T, H, W]
        o = o.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        return e, o

    def forward(self, levels: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Process all pyramid levels."""
        even_list, odd_list = [], []
        for l in levels:
            e, o = self.forward_level(l)
            even_list.append(e)
            odd_list.append(o)
        return even_list, odd_list


# ------------------------- Polar Decomposition ------------------------- #

class PolarDecompose(nn.Module):
    """Polar decomposition: extract amplitude and unit complex pose."""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, even_list: List[torch.Tensor], odd_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            even_list: List of [B, M, T, H, W] even responses per level
            odd_list: List of [B, M, T, H, W] odd responses per level
            
        Returns:
            A_list: List of [B, M, T, H, W] amplitude per level
            U_list: List of [B, M, 2, T, H, W] unit complex pose per level
        """
        A_list, U_list = [], []
        for e, o in zip(even_list, odd_list):
            # e/o: [B, M, T, H, W]
            # Force float32 for complex operations (bfloat16 doesn't support complex)
            e_f32 = e.float()
            o_f32 = o.float()

            w = complex_from_even_odd(e_f32, o_f32)  # complex pose
            A = torch.sqrt(torch.clamp(e_f32**2 + o_f32**2, min=self.eps))  # amplitude
            u = safe_unit_complex(w, self.eps)  # unit complex pose
            
            # Return amplitude as real tensor, pose as stacked real/imag for real-valued ops
            U = torch.stack([u.real, u.imag], dim=2)  # [B,M,2,T,H,W]

            # Cast back to original dtype
            A = A.to(e.dtype)
            U = U.to(e.dtype)

            A_list.append(A)
            U_list.append(U)
        return A_list, U_list


# ------------------------- Main PolarConvNet ------------------------- #

class PolarConvNet(nn.Module):
    """
    Polar ConvNet: Laplacian pyramid + quadrature filters + polar decompose.
    
    This is PURE spatial processing - no behavior, no temporal dynamics.
    
    Input:  [B, C, T, H, W] stimulus
    Output: (A_list, U_list) where:
            A_list: List of [B, M, T, H_l, W_l] amplitude per level
            U_list: List of [B, M, 2, T, H_l, W_l] unit complex pose per level
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pyramid_levels = config.get('n_pyramid_levels', 4)
        self.n_pairs = config.get('n_pairs', 16)
        self.kernel_size = config.get('kernel_size', 7)
        
        # Lazy init on first forward (need to know input shape)
        self.pyramid = None
        self.qfb = None
        self.polar = PolarDecompose()
    
    def _lazy_init(self, x):
        """Initialize pyramid and quadrature filters on first forward."""
        if self.pyramid is not None:
            return
        
        device = x.device
        self.pyramid = PyramidAdapter(J=self.n_pyramid_levels).to(device)
        
        # Get input channels from first pyramid level
        with torch.no_grad():
            levels = self.pyramid(x[:1])
            in_ch = levels[0].shape[1]
        
        self.qfb = QuadratureFilterBank2D(
            in_ch=in_ch,
            pairs=self.n_pairs,
            kernel=self.kernel_size
        ).to(device)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] stimulus
        
        Returns:
            A_list: List of [B, M, T, H_l, W_l] amplitude per level
            U_list: List of [B, M, 2, T, H_l, W_l] unit complex pose per level
        """
        self._lazy_init(x)
        
        # Build pyramid
        levels = self.pyramid(x)
        
        # Apply quadrature filters
        even_list, odd_list = self.qfb(levels)
        
        # Polar decomposition
        A_list, U_list = self.polar(even_list, odd_list)
        
        return A_list, U_list
    
    def get_output_channels(self):
        """Return (n_pairs, n_levels) for factory."""
        return (self.n_pairs, self.n_pyramid_levels)
