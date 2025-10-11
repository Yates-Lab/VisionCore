"""
Polar Recurrent Module for VisionCore.

This module implements the temporal processing components of the Polar-V1 model:
- PolarDynamics: Saccade-aware amplitude relaxation + phase rotation
- TemporalSummarizer: Causal summaries over time (collapse T dimension)
- PolarRecurrent: Main recurrent class with optional behavior and JEPA

Supports 4 configurations:
1. Minimal: summarize only
2. Behavior: dynamics + summarize
3. JEPA: JEPA + summarize
4. Full: dynamics + JEPA + summarize
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from einops import rearrange

__all__ = ['PolarRecurrent', 'PolarDynamics', 'TemporalSummarizer', 'init_kxy']


# ------------------------- Spatial Frequency Initialization ------------------------- #

def init_kxy(S: int, M: int, base_freq_cpx: float, pix_per_deg: Optional[float] = None,
             orientations: Optional[torch.Tensor] = None,
             device: Optional[torch.device] = None) -> torch.Tensor:
    """Create per-(level, pair) spatial frequency vectors kxy [S,M,2].
    Frequencies scale with level (× 2^{-s}). `base_freq_cpx` is cycles/pixel at level 0.
    If pix_per_deg given, you can set base_freq_cpx using cycles/deg / pix_per_deg.
    """
    if orientations is None:
        # Create M evenly spaced orientations from 0 to pi (excluding pi)
        orientations = torch.arange(M, dtype=torch.float32) * (math.pi / M)
    if device is None:
        device = torch.device('cpu')
    
    # radial freqs log-spaced across pairs (can also fix one radius per level)
    r0 = base_freq_cpx
    radii = r0 * torch.ones(M)
    k = torch.zeros(S, M, 2)
    for s in range(S):
        scale = 2.0 ** (-s)
        for m, th in enumerate(orientations):
            k[s, m, 0] = (radii[m] * scale) * math.cos(float(th))
            k[s, m, 1] = (radii[m] * scale) * math.sin(float(th))
    return k.to(device)


# ------------------------- Polar Dynamics ------------------------- #

class PolarDynamics(nn.Module):
    """Saccade-aware amplitude relaxation + phase rotation.

    Args:
        kxy: spatial frequency vectors per (S,M): Tensor [S,M,2]
        dt: frame step in seconds (e.g., 1/240)
        lambda_fix, lambda_sac: amplitude rates (s^-1)
        a_bar: optional baseline amplitude per (S,M) or scalar
        learn_cutoff_decimator: if True, provide a tiny IIR+decimate module (optional)
    """
    
    def __init__(self,
                 kxy: torch.Tensor,
                 dt: float,
                 lambda_fix: float = 10.0,
                 lambda_sac: float = 40.0,
                 a_bar: Optional[torch.Tensor] = None,
                 learn_cutoff_decimator: bool = False):
        super().__init__()
        self.register_buffer('kxy', kxy)  # [S,M,2]
        self.dt = dt
        self.lambda_fix = nn.Parameter(torch.tensor(lambda_fix, dtype=torch.float32))
        self.lambda_sac = nn.Parameter(torch.tensor(lambda_sac, dtype=torch.float32))
        
        if a_bar is None:
            a_bar = torch.ones(kxy.shape[0], kxy.shape[1]) * 0.5
        self.a_bar = nn.Parameter(a_bar)  # [S,M]
        
        self.learn_cutoff_decimator = learn_cutoff_decimator
        if learn_cutoff_decimator:
            # simple per-level temporal pole (0..1) controlling anti-alias IIR before decimation
            self.pole = nn.Parameter(torch.zeros(kxy.shape[0]))  # per level
            self.decim = nn.Parameter(torch.zeros(kxy.shape[0])) # logits for stride in {1,2,3}

    def forward(self,
                A_list: List[torch.Tensor],     # [B,M,T,H,W] per level
                U_list: List[torch.Tensor],     # unit complex pose as [B,M,2,T,H,W]
                beh_code: Dict[str, torch.Tensor],
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return advanced (A_next, U_next) lists with same shapes.
        Continuous-time exponential update per frame.
        """
        q    = beh_code['q']          # [B,T,1]
        v_eff= beh_code['v_eff']      # [B,T,2]
        gam  = beh_code['gamma']      # [B,T,S,M]
        rho  = beh_code['rho']        # [B,T,S,M]

        A_out, U_out = [], []
        for s, (A, U) in enumerate(zip(A_list, U_list)):
            B,M,T,H,W = A.shape
            
            # phase increment dphi = gamma * (k·v_eff)*dt + rho
            k = self.kxy[s].unsqueeze(0).unsqueeze(0)  # [1,1,M,2]
            kv = (k[...,0]*v_eff[:,:,0:1] + k[...,1]*v_eff[:,:,1:2])   # [B,T,M]
            kv = kv.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)     # [B,M,T,1,1]
            gamma_sm = gam[:,:,s,:].permute(0,2,1).unsqueeze(-1).unsqueeze(-1)       # [B,M,T,1,1]
            rho_sm   = rho[:,:,s,:].permute(0,2,1).unsqueeze(-1).unsqueeze(-1)       # [B,M,T,1,1]
            dphi = gamma_sm * kv * self.dt + rho_sm  # [B,M,T,1,1]

            # Safe trig under bf16: cast to fp32 for sin/cos, then back
            dphi_f32 = dphi.float()
            cos_d = torch.cos(dphi_f32).to(A.dtype)
            sin_d = torch.sin(dphi_f32).to(A.dtype)
            U_re, U_im = U[:, :, 0], U[:, :, 1]           # [B,M,T,H,W]
            U_re_next = cos_d * U_re - sin_d * U_im
            U_im_next = sin_d * U_re + cos_d * U_im
            U_next = torch.stack([U_re_next, U_im_next], dim=2)

            # Safe exp under bf16: cast to fp32 for exp, then back
            lam = (1 - q) * self.lambda_fix + q * self.lambda_sac      # [B,T,1]
            lam = lam.unsqueeze(1).unsqueeze(-1)                        # [B,1,T,1,1]
            lam_f32 = lam.float()
            a_bar = self.a_bar[s].view(1, M, 1, 1, 1)
            A_next = a_bar + (A - a_bar) * torch.exp(-lam_f32 * self.dt).to(A.dtype)

            A_out.append(A_next); U_out.append(U_next)
        return A_out, U_out


# ------------------------- Temporal Summarizer ------------------------- #

class TemporalSummarizer(nn.Module):
    """Causal summaries per channel/pixel over T frames.
       Outputs (per channel): last, ema_fast, ema_slow, derivative, temporal_energy.
    """
    
    def __init__(self, alpha_fast: float = 0.74, alpha_slow: float = 0.95):
        super().__init__()
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow

    @staticmethod
    def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
        # x: [B,C,T,H,W]
        B,C,T,H,W = x.shape
        out = []
        acc = torch.zeros(B,C,H,W, device=x.device, dtype=x.dtype)
        for t in range(T):
            acc = alpha*acc + (1-alpha)*x[:,:,t]
            out.append(acc.unsqueeze(2))
        return torch.cat(out, dim=2)

    def forward(self, A_list: List[torch.Tensor], U_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            A_list: List of [B,M,T,H,W] amplitude per level
            U_list: List of [B,M,2,T,H,W] unit complex pose per level
            
        Returns:
            feats: List of [B, 5*M, H, W] per level (collapsed time dimension)
        """
        feats = []
        for A, U in zip(A_list, U_list):
            # A: [B,M,T,H,W]; U: [B,M,2,T,H,W]
            B,M,T,H,W = A.shape
            
            # Amplitude summaries
            A_last  = A[:,:,-1:]
            A_fast  = self._ema(A, self.alpha_fast)[:,:,-1:]
            A_slow  = self._ema(A, self.alpha_slow)[:,:,-1:]
            A_deriv = torch.diff(A, dim=2, prepend=A[:,:,:1])[:,:,-1:]  # last derivative
            
            # Pose temporal energy via simple causal quadrature approx along time
            U_re, U_im = U[:, :, 0], U[:, :, 1]  # [B,M,T,H,W]
            
            # 2-step odd filter: p_q ~ (p_t - p_{t-2})/2
            def quad_energy(p: torch.Tensor) -> torch.Tensor:
                p_shift2 = torch.cat([p[:,:,:1], p[:,:,:1], p[:,:,:-2]], dim=2)
                pq = 0.5*(p - p_shift2)
                e = torch.sqrt(torch.clamp(p**2 + pq**2, min=1e-6))
                return e
            
            E_re = quad_energy(U_re)
            E_im = quad_energy(U_im)
            E_temp = torch.sqrt(torch.clamp(E_re**2 + E_im**2, min=1e-6))[:,:,-1:]  # last
            
            # Concatenate summaries across channels: [B, 5*M, 1, H, W] -> squeeze T
            out = torch.cat([A_last, A_fast, A_slow, A_deriv, E_temp], dim=1).squeeze(2)
            feats.append(out)  # [B, 5M, H, W]
        return feats


# ------------------------- Main PolarRecurrent ------------------------- #

class PolarRecurrent(nn.Module):
    """
    Polar recurrent: Optional dynamics + optional JEPA + summarization.

    Supports 4 configurations:
    1. Minimal: summarize only
    2. Behavior: dynamics + summarize
    3. JEPA: JEPA + summarize
    4. Full: dynamics + JEPA + summarize
    """

    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.dt = config.get('dt', 1/240)
        self.use_behavior = config.get('use_behavior', True)
        self.use_jepa = config.get('use_jepa', False)

        # Spatial frequencies
        self.kxy = init_kxy(
            S=self.n_levels,
            M=self.n_pairs,
            base_freq_cpx=config.get('base_freq_cpx', 0.15)
        )

        # Polar dynamics (optional)
        if self.use_behavior:
            self.dynamics = PolarDynamics(
                kxy=self.kxy,
                dt=self.dt,
                lambda_fix=config.get('lambda_fix', 10.0),
                lambda_sac=config.get('lambda_sac', 40.0)
            )
            self.modulator = None  # Set by model
        else:
            self.dynamics = None

        # JEPA (optional)
        if self.use_jepa:
            from .polar_jepa import PolarJEPA
            self.jepa = PolarJEPA(config)
        else:
            self.jepa = None

        # Temporal summarizer (always used)
        self.summarizer = TemporalSummarizer(
            alpha_fast=config.get('alpha_fast', 0.74),
            alpha_slow=config.get('alpha_slow', 0.95)
        )

    def forward(self, feats):
        """
        Args:
            feats: (A_list, U_list) from modulator

        Returns:
            feats_per_level: List of [B, C_sum, H_l, W_l]
        """
        A_list, U_list = feats

        # Apply dynamics (if enabled)
        if self.dynamics is not None:
            if self.modulator is None:
                raise RuntimeError("Modulator not set. Call set_modulator() first.")
            beh_params = self.modulator.beh_params
            if beh_params is None:
                raise RuntimeError(
                    "Behavior parameters not available. "
                    "When use_behavior=True, you must pass behavior data to the model forward method: "
                    "model(stimulus, dataset_idx, behavior)"
                )
            A_adv, U_adv = self.dynamics(A_list, U_list, beh_params)
        else:
            A_adv, U_adv = A_list, U_list

        # Apply JEPA (if enabled)
        if self.jepa is not None:
            A_adv, U_adv = self.jepa(A_adv, U_adv)

        # Temporal summarization
        feats_per_level = self.summarizer(A_adv, U_adv)

        return feats_per_level

    def set_modulator(self, modulator):
        """Set reference to modulator for accessing behavior params."""
        if self.use_behavior:
            self.modulator = modulator

    def get_jepa_loss(self):
        """Get JEPA loss for auxiliary loss (legacy method)."""
        if self.jepa is not None:
            return self.jepa.get_jepa_loss()
        return None

    def get_auxiliary_loss(self):
        """
        Get auxiliary loss from this module.

        This is the standard interface for auxiliary losses.
        Returns JEPA loss if JEPA is enabled.

        Returns
        -------
        torch.Tensor or None
            Auxiliary loss if available, None otherwise
        """
        return self.get_jepa_loss()
