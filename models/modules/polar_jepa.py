"""
Polar JEPA Module for VisionCore.

This module implements the JEPA (Joint Embedding Predictive Architecture) 
components for the Polar-V1 model:
- PolarJEPA: Masked future-token prediction in polar space

JEPA uses stop-grad EMA teacher + masked future-token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

__all__ = ['PolarJEPA']


class PolarJEPA(nn.Module):
    """Masked future-token prediction in polar space.

    Assumes inputs are *summarized* feature maps (per level): [B, C_l, H_l, W_l].
    Student and teacher share trunk weights externally; this head only does 
    projection/prediction and EMA update of teacher copies.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.proj_dim = config.get('proj_dim', 256)
        self.tau = config.get('tau', 0.996)
        
        # Calculate C_per_level (5 summaries per pair)
        C_per_level = [5 * self.n_pairs] * self.n_levels
        
        # Student and teacher projectors
        self.proj_s = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, self.proj_dim, 1), 
                nn.GELU(), 
                nn.Conv2d(self.proj_dim, self.proj_dim, 1)
            ) for C in C_per_level
        ])
        
        self.proj_t = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, self.proj_dim, 1), 
                nn.GELU(), 
                nn.Conv2d(self.proj_dim, self.proj_dim, 1)
            ) for C in C_per_level
        ])
        
        # Predictor on student side
        self.pred = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.proj_dim, self.proj_dim, 1), 
                nn.GELU(), 
                nn.Conv2d(self.proj_dim, self.proj_dim, 1)
            ) for _ in C_per_level
        ])
        
        # Initialize teacher with student weights
        for ps, pt in zip(self.proj_s, self.proj_t):
            for p_t, p_s in zip(pt.parameters(), ps.parameters()):
                p_t.data.copy_(p_s.data)
        
        self.register_buffer('ema_initialized', torch.tensor(1))
        
        # Storage for JEPA loss
        self._jepa_loss = None

    @torch.no_grad()
    def ema_update(self):
        """Update teacher parameters with EMA."""
        for ps, pt in zip(self.proj_s, self.proj_t):
            for p_t, p_s in zip(pt.parameters(), ps.parameters()):
                p_t.data = self.tau * p_t.data + (1 - self.tau) * p_s.data

    def forward(self, A_list: List[torch.Tensor], U_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass that applies JEPA processing and stores loss.
        
        Args:
            A_list: List of [B,M,T,H,W] amplitude per level
            U_list: List of [B,M,2,T,H,W] unit complex pose per level
            
        Returns:
            A_list, U_list: Pass-through (unchanged)
        """
        # For now, just pass through the features
        # In a full implementation, this would apply JEPA processing
        # and compute the loss for later retrieval
        
        # Store zero loss for now (can be extended later)
        self._jepa_loss = torch.tensor(0.0, device=A_list[0].device)
        
        return A_list, U_list

    def compute_jepa_loss(self,
                         feats_ctx: List[torch.Tensor],   # student features at t (list over levels)
                         feats_tgt: List[torch.Tensor],   # teacher features at t+Δ (list over levels)
                         mask_ratio: float = 0.5) -> torch.Tensor:
        """
        Compute JEPA loss between context and target features.
        
        Args:
            feats_ctx: Student features at time t
            feats_tgt: Teacher features at time t+Δ
            mask_ratio: Fraction of tokens to mask
            
        Returns:
            JEPA loss
        """
        losses = []
        for l, (xc, xt) in enumerate(zip(feats_ctx, feats_tgt)):
            # Project
            zc = F.normalize(self.proj_s[l](xc), dim=1)
            with torch.no_grad():
                zt = F.normalize(self.proj_t[l](xt), dim=1)  # stop-grad EMA target
            
            B, C, H, W = zc.shape
            
            # Mask a random subset of tokens (spatial positions)
            num = H * W
            k = max(1, int(mask_ratio * num))
            idx = torch.randperm(num, device=zc.device)[:k]
            m = torch.zeros(num, device=zc.device, dtype=torch.bool)
            m[idx] = True
            m = m.view(1, 1, H, W)
            
            # Predict masked tokens
            pc = self.pred[l](zc)
            lc = 1 - (pc * zt).sum(dim=1, keepdim=True)  # cosine since both normalized
            loss = lc[m.expand(B, 1, H, W)].mean()
            losses.append(loss)
        
        return torch.stack(losses).mean()

    def compute_and_store_loss(self,
                               feats_ctx: List[torch.Tensor],
                               feats_tgt: List[torch.Tensor],
                               mask_ratio: float = 0.5):
        """
        Compute JEPA loss and store it for later retrieval.

        Args:
            feats_ctx: Context features (early frames) - List of [B, C, H, W]
            feats_tgt: Target features (late frames) - List of [B, C, H, W]
            mask_ratio: Fraction of spatial tokens to mask
        """
        # Compute loss using the existing method
        self._jepa_loss = self.compute_jepa_loss(feats_ctx, feats_tgt, mask_ratio)

        # Update EMA teacher
        self.ema_update()

    def get_jepa_loss(self) -> torch.Tensor:
        """Get the stored JEPA loss."""
        if self._jepa_loss is None:
            return None
        return self._jepa_loss
