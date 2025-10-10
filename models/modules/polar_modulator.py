"""
Polar Modulator Module for VisionCore.

This module implements the behavior encoding components of the Polar-V1 model:
- MinimalGazeEncoder: Physics-aware encoder from gaze position to behavior code
- BehaviorEncoder: Maps behavior code to dynamics parameters
- PolarModulator: Main modulator class

This does NOT modify features - just encodes behavior parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .modulator import BaseModulator

__all__ = ['PolarModulator', 'MinimalGazeEncoder', 'BehaviorEncoder']


# ------------------------- Minimal Gaze Encoder ------------------------- #

class MinimalGazeEncoder(nn.Module):
    """Lightweight, physics-aware encoder from gaze position to behavior code.
    Input:  gaze_xy [B,T,2]  (pixels or degrees in retinal coords)
    Output: bcode   [B,T,D]  (default D=128)

    Features per time step (very small set):
      • velocity v_x,v_y; speed |v|; direction (cos,sin)
      • acceleration a_x,a_y; tangential/normal accel (a_par, a_perp)
      • saccade gate via learnable threshold on speed + causal fast/slow traces
      • optional 2-frequency Fourier encoding of position (sin/cos) with LEARNABLE frequencies
    Then a tiny 2-layer MLP → D.
    """
    
    def __init__(self, d_out: int = 128, dt: float = 1/240.0, use_pos_fourier: bool = True, Kpos: int = 2,
                 init_pos_freq_cpx: float = 1/64):
        super().__init__()
        self.dt = dt
        self.use_pos_fourier = use_pos_fourier
        self.Kpos = Kpos
        
        if use_pos_fourier:
            # learnable log-frequencies and phases per axis
            self.pos_logw_x = nn.Parameter(torch.log(torch.full((Kpos,), init_pos_freq_cpx)))
            self.pos_phi_x  = nn.Parameter(torch.zeros(Kpos))
            self.pos_logw_y = nn.Parameter(torch.log(torch.full((Kpos,), init_pos_freq_cpx)))
            self.pos_phi_y  = nn.Parameter(torch.zeros(Kpos))
            
        # saccade threshold (learnable) and temperature for sigmoid gate
        self.sac_log_thr = nn.Parameter(torch.tensor(math.log(200.0)))  # speed units
        self.sac_invT    = nn.Parameter(torch.tensor(0.05))             # small → sharp gate
        
        # lazy projector to D (init on first forward when D_in known)
        self.proj1 = nn.Linear(1, d_out)
        self.proj2 = nn.Linear(d_out, d_out)
        self._initialized = False

    def _ema(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Causal exponential moving average.
        Args:
            x: [B,T,1]
            alpha: decay factor (higher = slower decay)
        Returns:
            [B,T,1]
        """
        B, T, _ = x.shape
        out = []
        acc = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype).expand(B, 1, 1).clone()
        for t in range(T):
            acc = alpha * acc + (1 - alpha) * x[:, t:t+1, :]
            out.append(acc)
        return torch.cat(out, dim=1)

    def _fourier_axis(self, coord: torch.Tensor, logw: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # coord [B,T] → [B,T,2K]
        w = torch.exp(logw)[None,None,:]
        ph = phi[None,None,:]
        arg = 2*math.pi*coord[...,None]*w + ph
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

    def forward(self, gaze_xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaze_xy: [B,T,2] gaze positions
            
        Returns:
            bcode: [B,T,D] behavior code
        """
        B,T,_ = gaze_xy.shape
        
        # finite-diff kinematics
        v = torch.diff(gaze_xy, dim=1, prepend=gaze_xy[:, :1]) / max(self.dt, 1e-6)  # [B,T,2]
        a = torch.diff(v,        dim=1, prepend=v[:, :1])        / max(self.dt, 1e-6)  # [B,T,2]
        speed = torch.linalg.vector_norm(v, dim=-1)                                         # [B,T]
        dir_cos = v[...,0] / (speed + 1e-6)
        dir_sin = v[...,1] / (speed + 1e-6)
        a_par  = (v*a).sum(-1) / (speed + 1e-6)
        a_perp = (v[...,0]*a[...,1] - v[...,1]*a[...,0]) / (speed + 1e-6)

        # saccade gate via smooth threshold
        thr = torch.exp(self.sac_log_thr)
        gate = torch.sigmoid(self.sac_invT * (speed - thr)).unsqueeze(-1)  # [B,T,1]
        
        # causal fast/slow traces via EMAs (no shape surprises)
        q_fast = self._ema(gate, alpha=0.8).squeeze(-1)   # ~10–20 ms
        q_slow = self._ema(gate, alpha=0.95).squeeze(-1)  # ~80–120 ms

        # assemble features
        scalars = torch.stack([
            v[...,0], v[...,1], speed, dir_cos, dir_sin,
            a[...,0], a[...,1], a_par, a_perp, gate.squeeze(-1), q_fast, q_slow
        ], dim=-1)  # [B,T,S]

        if self.use_pos_fourier:
            x = gaze_xy[...,0]; y = gaze_xy[...,1]
            fx = self._fourier_axis(x, self.pos_logw_x, self.pos_phi_x)  # [B,T,2K]
            fy = self._fourier_axis(y, self.pos_logw_y, self.pos_phi_y)  # [B,T,2K]
            feats = torch.cat([fx, fy, scalars], dim=-1)
        else:
            feats = scalars

        # lazy init projector now that we know D_in
        if not self._initialized:
            D_in = feats.shape[-1]
            d_out = self.proj1.out_features
            # keep weights in float32; AMP will cast activations
            self.proj1 = nn.Linear(D_in, d_out).to(feats.device)
            self.proj2 = nn.Linear(d_out, d_out).to(feats.device)
            nn.init.xavier_uniform_(self.proj1.weight); nn.init.zeros_(self.proj1.bias)
            nn.init.xavier_uniform_(self.proj2.weight); nn.init.zeros_(self.proj2.bias)
            self._initialized = True

        h = F.gelu(self.proj1(feats))
        h = F.gelu(self.proj2(h))  # [B,T,D]
        return h


# ------------------------- Behavior Encoder ------------------------- #

class BehaviorEncoder(nn.Module):
    """
    Behavior is a SINGLE tensor bcode: [B, T, D].
    Map bcode → q (saccade gate), v_eff (effective retinal slip, 2D),
               gamma (per-band gain ≈ 1+), rho (small residual phase).
    """
    
    def __init__(self, M: int, S: int, beh_dim: int = 128):
        super().__init__()
        self.M, self.S, self.beh_dim = M, S, beh_dim
        self.fc1 = nn.Linear(beh_dim, beh_dim)
        self.fc2 = nn.Linear(beh_dim, beh_dim)
        self.head_q    = nn.Linear(beh_dim, 1)
        self.head_veff = nn.Linear(beh_dim, 2)
        self.head_gam  = nn.Linear(beh_dim, S*M)
        self.head_rho  = nn.Linear(beh_dim, S*M)

    def forward(self, bcode: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            bcode: [B,T,D] behavior code
            
        Returns:
            Dict with keys ['q', 'v_eff', 'gamma', 'rho']
        """
        # bcode: [B,T,D]
        h = torch.relu(self.fc1(bcode))
        h = torch.relu(self.fc2(h))
        q     = torch.sigmoid(self.head_q(h))                 # [B,T,1]
        v_eff = torch.tanh(self.head_veff(h)) * 50.0          # scale factor ≈ px/s; tune as needed
        gam   = F.softplus(self.head_gam(h)).view(bcode.size(0), bcode.size(1), self.S, self.M) + 1.0
        rho   = 0.1 * torch.tanh(self.head_rho(h)).view(bcode.size(0), bcode.size(1), self.S, self.M)
        return {"q": q, "v_eff": v_eff, "gamma": gam, "rho": rho}


# ------------------------- Main PolarModulator ------------------------- #

class PolarModulator(BaseModulator):
    """
    Polar modulator: Encodes behavior for polar dynamics.
    
    This does NOT modify features - just encodes behavior parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set required attributes before calling super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.beh_dim = config.get('beh_dim', 128)
        self.dt = config.get('dt', 1/240)
        
        # Storage for behavior params (accessed by recurrent)
        self.beh_params = None
        
        super().__init__(config)
    
    def _build_modulator(self):
        """Build the gaze encoder and behavior encoder."""
        # Gaze encoder
        self.gaze_encoder = MinimalGazeEncoder(
            d_out=self.beh_dim,
            dt=self.dt,
            use_pos_fourier=True,
            Kpos=2
        )
        
        # Behavior encoder
        self.beh_encoder = BehaviorEncoder(
            M=self.n_pairs,
            S=self.n_levels,
            beh_dim=self.beh_dim
        )
        
        # Set output dimension (features pass through unchanged)
        self.out_dim = 0  # No additional channels added
    
    def forward(self, feats: Tuple, behavior: torch.Tensor) -> Tuple:
        """
        Args:
            feats: (A_list, U_list) from convnet
            behavior: [B, T, 2] raw gaze positions
        
        Returns:
            feats: (A_list, U_list) unchanged (pass-through)
        """
        A_list, U_list = feats
        
        # Encode gaze
        beh_code = self.gaze_encoder(behavior)  # [B, T, D]
        
        # Get dynamics parameters
        self.beh_params = self.beh_encoder(beh_code)
        # Returns: {'q': [B,T,1], 'v_eff': [B,T,2], 
        #           'gamma': [B,T,S,M], 'rho': [B,T,S,M]}
        
        # Return features unchanged
        return (A_list, U_list)
