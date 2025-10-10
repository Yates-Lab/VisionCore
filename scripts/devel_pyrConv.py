#!/usr/bin/env python
"""
Example script for checking whether a model config builds

This script demonstrates:
1. Builds a model from a config
2. Prepares data
3. Tests the forward and backward on one batch
4. Tests the jacobian
"""
#%%
import sys
import torch
from torch.utils.data import DataLoader

import lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from models import build_model, initialize_model_components
from models.config_loader import load_config
from models.data import prepare_data
from models.utils.general import ensure_tensor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda:1')
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()

from contextlib import nullcontext
AMP_BF16 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

#%% Helpers
"""
Polar-V1 Core: downstream modules AFTER a Laplacian pyramid

Input: `levels: List[Tensor]` with shapes [B, C_in, T, H_l, W_l], levels[0] = finest, levels[-1] = coarsest.
This file defines:
  • QuadratureFilterBank2D – shared paired (even/odd) spatial filters across levels
  • PolarDecompose – amplitude + complex pose (unit real/imag)
  • BehaviorEncoder – saccade-aware behavior code, q_t gate, small drift, per-band gains & residuals
  • PolarDynamics – continuous-time amplitude relaxation + phase rotation (Fourier-shift style)
  • TemporalSummarizer – causal EMAs, derivative, temporal energy → collapses T to few channels
  • MultiLevelReadout – Gaussian spatial readout per neuron across levels, low-rank channel mixing
  • JEPAModule – stop-grad EMA teacher + masked future-token prediction in polar space
  • PolarV1Core – convenience module wiring all of the above

All modules are written to be lightweight and easy to swap.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from einops import rearrange

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- utils ------------------------- #
# NOTE: Batches and behavior tensors are already shaped [B, T, ...] in your pipeline,
# so no time broadcasting helper is needed. (Function removed.)


def complex_from_even_odd(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    return torch.complex(e, o)


def safe_unit_complex(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize complex tensor to unit magnitude: w / (|w|+eps)."""
    mag = torch.clamp(torch.abs(w), min=eps)
    return w / mag


# ---------------- Quadrature filter bank (shared) ---------------- #

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
        B, C, T, H, W = x_l.shape
        x_flat = x_l.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # [B*T, C, H, W]
        e = self.mix_even(self.dw_even(x_flat))  # [B*T, M, H, W]
        o = self.mix_odd (self.dw_odd (x_flat))
        e = e.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)  # [B, M, T, H, W]
        o = o.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        return e, o

    def forward(self, levels: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        even_list, odd_list = [], []
        for l in levels:
            e, o = self.forward_level(l)
            even_list.append(e)
            odd_list.append(o)
        return even_list, odd_list


# ---------------- Polar decomposition ---------------- #

class PolarDecompose(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, even_list: List[torch.Tensor], odd_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        A_list, W_list = [], []
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
            W_list.append(U)
        return A_list, W_list


# ---------------- Behavior encoder ---------------- #

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
        # bcode: [B,T,D]
        h = torch.relu(self.fc1(bcode))
        h = torch.relu(self.fc2(h))
        q     = torch.sigmoid(self.head_q(h))                 # [B,T,1]
        v_eff = torch.tanh(self.head_veff(h)) * 50.0          # scale factor ≈ px/s; tune as needed
        gam   = F.softplus(self.head_gam(h)).view(bcode.size(0), bcode.size(1), self.S, self.M) + 1.0
        rho   = 0.1 * torch.tanh(self.head_rho(h)).view(bcode.size(0), bcode.size(1), self.S, self.M)
        return {"q": q, "v_eff": v_eff, "gamma": gam, "rho": rho}


# ---------------- Polar dynamics (continuous-time) ---------------- #

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


# ---------------- Temporal summarizer (collapse T) ---------------- #

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


# ---------------- Multi-level readout ---------------- #

class GaussianReadout(nn.Module):
    """Gaussian (elliptical) spatial readout per neuron per level.
    Positions & covariances are learned per neuron+level.
    """
    def __init__(self, C: int, H: int, W: int, n_neurons: int):
        super().__init__()
        self.n = n_neurons
        self.C, self.H, self.W = C, H, W
        self.mu    = nn.Parameter(torch.rand(n_neurons, 2)*2-1)
        self.logsx = nn.Parameter(torch.zeros(n_neurons))
        self.logsy = nn.Parameter(torch.zeros(n_neurons))
        self.rho   = nn.Parameter(torch.zeros(n_neurons))
        self.weight= nn.Parameter(torch.randn(n_neurons, C)*0.01)

    @staticmethod
    def gaussian_grid(H, W, device, dtype):
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([X, Y], dim=-1)  # [H,W,2]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        grid = self.gaussian_grid(H, W, feat.device, feat.dtype)   # [H,W,2]
        X = grid.unsqueeze(0).expand(self.n, H, W, 2)              # [n,H,W,2]  <-- FIXED (one unsqueeze)
        mu = self.mu[:, None, None, :]                             # [n,1,1,2]
        sx = torch.exp(self.logsx)[:, None, None] + 1e-6           # [n,1,1]
        sy = torch.exp(self.logsy)[:, None, None] + 1e-6
        rho= torch.tanh(self.rho)[:, None, None]

        Xc = X - mu
        A = (Xc[...,0]/sx)**2 + (Xc[...,1]/sy)**2 - 2*rho*(Xc[...,0]/sx)*(Xc[...,1]/sy)
        denom = 2*(1 - rho**2 + 1e-6)
        G = torch.exp(-A/denom)                                    # [n,H,W]

        # sample + channel mix
        G = G.unsqueeze(0).unsqueeze(2)                            # [1,n,1,H,W]
        pooled = (feat.unsqueeze(1) * G).sum(dim=(-1, -2))         # [B,n,C]
        out = (pooled * self.weight[None].to(feat.dtype)).sum(-1)  # [B,n]
        return out


class MultiLevelReadout(nn.Module):
    def __init__(self, level_shapes: List[Tuple[int,int,int]], C_per_level: List[int], n_neurons: int):
        super().__init__()
        assert len(level_shapes) == len(C_per_level)
        self.levels = nn.ModuleList([
            GaussianReadout(C=C_per_level[l], H=level_shapes[l][1], W=level_shapes[l][2], n_neurons=n_neurons)
            for l in range(len(level_shapes))
        ])
        # low-rank shared basis could be added here if desired
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, feats_per_level: List[torch.Tensor]) -> torch.Tensor:
        parts = []
        for l, feat in enumerate(feats_per_level):
            parts.append(self.levels[l](feat))  # [B,n]
        s = torch.stack(parts, dim=-1).sum(dim=-1)  # sum across levels → [B,n]
        return s + self.bias


# ---------------- JEPA (student/teacher with EMA) ---------------- #

class JEPAModule(nn.Module):
    """Masked future-token prediction in polar space.

    Assumes inputs are *summarized* feature maps (per level): [B, C_l, H_l, W_l].
    Student and teacher share trunk weights externally; this head only does projection/prediction and EMA update of teacher copies.
    """
    def __init__(self, C_per_level: List[int], proj_dim: int = 256, tau: float = 0.996):
        super().__init__()
        self.tau = tau
        self.proj_s = nn.ModuleList([nn.Sequential(nn.Conv2d(C, proj_dim, 1), nn.GELU(), nn.Conv2d(proj_dim, proj_dim, 1)) for C in C_per_level])
        self.proj_t = nn.ModuleList([nn.Sequential(nn.Conv2d(C, proj_dim, 1), nn.GELU(), nn.Conv2d(proj_dim, proj_dim, 1)) for C in C_per_level])
        # predictor on student side
        self.pred   = nn.ModuleList([nn.Sequential(nn.Conv2d(proj_dim, proj_dim, 1), nn.GELU(), nn.Conv2d(proj_dim, proj_dim, 1)) for _ in C_per_level])
        # init teacher with student weights
        for ps, pt in zip(self.proj_s, self.proj_t):
            for p_t, p_s in zip(pt.parameters(), ps.parameters()):
                p_t.data.copy_(p_s.data)
        self.register_buffer('ema_initialized', torch.tensor(1))

    @torch.no_grad()
    def ema_update(self):
        for ps, pt in zip(self.proj_s, self.proj_t):
            for p_t, p_s in zip(pt.parameters(), ps.parameters()):
                p_t.data = self.tau * p_t.data + (1 - self.tau) * p_s.data

    def forward(self,
                feats_ctx: List[torch.Tensor],   # student features at t (list over levels)
                feats_tgt: List[torch.Tensor],   # teacher features at t+Δ (list over levels)
                mask_ratio: float = 0.5) -> torch.Tensor:
        losses = []
        for l, (xc, xt) in enumerate(zip(feats_ctx, feats_tgt)):
            # project
            zc = F.normalize(self.proj_s[l](xc), dim=1)
            with torch.no_grad():
                zt = F.normalize(self.proj_t[l](xt), dim=1)  # stop-grad EMA target
            B,C,H,W = zc.shape
            # mask a random subset of tokens (spatial positions)
            num = H*W
            k = max(1, int(mask_ratio * num))
            idx = torch.randperm(num, device=zc.device)[:k]
            m = torch.zeros(num, device=zc.device, dtype=torch.bool)
            m[idx] = True
            m = m.view(1,1,H,W)
            # predict masked tokens
            pc = self.pred[l](zc)
            lc = 1 - (pc * zt).sum(dim=1, keepdim=True)  # cosine since both normalized
            loss = lc[m.expand(B,1,H,W)].mean()
            losses.append(loss)
        return torch.stack(losses).mean()


# ---------------- Adapters: Plenoptic pyramid → PolarV1Core ---------------- #

# --------- Minimal physics-aware encoder: [B,T,2] gaze → [B,T,D] behavior --------- #
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

# --------- end minimal encoder --------- #

# ---------------- Adapters: Plenoptic pyramid → PolarV1Core ---------------- #

class PyramidAdapter:
    """Wrap a plenoptic.simulate.LaplacianPyramid to produce a `levels` list
    shaped [B, C_in, T, H_l, W_l] suitable for PolarV1Core.
    """
    def __init__(self, J: int):
        from plenoptic.simulate import LaplacianPyramid
        self.lpyr = LaplacianPyramid(J)
        self.J = J

    def to(self, device):
        self.lpyr = self.lpyr.to(device)
        return self

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: [B, C_in, T, H, W] -> levels: List[[B,C_in,T,H_l,W_l]]"""
        B, C, T, H, W = x.shape
        x_bt = rearrange(x, 'B C T H W -> (B T) C H W')
        levels_bt = self.lpyr(x_bt)  # list of [(B*T), C, H_l, W_l]
        levels = [
            rearrange(L, '(B T) C H W -> B C T H W', B=B, T=T)
            for L in levels_bt
        ]
        return levels


# ---------------- Behavior adapter (batch → core.beh inputs) ---------------- #

class BehaviorAdapter:
    """Map your batch['behavior'] to the dict expected by BehaviorEncoder.
    Expects keys (best effort):
      • 'gaze_vel' in px/s or deg/s, else computes diff of 'gaze' positions.
      • 'gaze_acc' optional; if missing we set zeros.
      • 'saccade' optional binary/prob; else derive from |vel| threshold.
    All returned tensors are [B, T, ...] on the same device/dtype as input.
    """
    def __init__(self, vel_thresh: float = 30.0):
        self.vel_thresh = vel_thresh

    def __call__(self, beh: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = next(iter(beh.values())).shape[0]
        device = next(iter(beh.values())).device
        dtype = next(iter(beh.values())).dtype
        g = beh.get('gaze')  # [B,T,2] if available
        gv = beh.get('gaze_vel')
        if gv is None and g is not None:
            gv = torch.diff(g, dim=1, prepend=g[:, :1])  # px/frame; scale outside if needed
        ga = beh.get('gaze_acc')
        if ga is None and gv is not None:
            ga = torch.diff(gv, dim=1, prepend=gv[:, :1])
        sac = beh.get('saccade')
        if sac is None and gv is not None:
            speed = torch.linalg.vector_norm(gv, dim=-1, keepdim=True)
            sac = (speed > self.vel_thresh).to(dtype)
        # Fallback zeros if still None
        if gv is None:
            T = beh.get('dfs', torch.zeros(B,1, device=device, dtype=dtype)).shape[1]
            gv = torch.zeros(B, T, 2, device=device, dtype=dtype)
            ga = torch.zeros(B, T, 2, device=device, dtype=dtype)
            sac = torch.zeros(B, T, 1, device=device, dtype=dtype)
        return {'gvel': gv, 'gacc': ga, 'sac': sac}


# ---------------- Kxy initializer (orientations × frequencies × levels) ---------------- #

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


# ---------------- Wiring into your example script ---------------- #

"""
Example glue code to drop into your script after you construct a batch.
Assumes `batch['stim']` = [B, C=1, T, H, W], and `batch['behavior']` is a dict.
"""

def build_polar_core_from_batch(batch, n_pairs: int = 16, dt: float = 1/240, proj_dim: int = 256, beh_dim: int = None):
    # 1) Build pyramid adapter based on how many levels you want
    J = 4  # or match your dataset
    pyr = PyramidAdapter(J=J).to(batch['stim'].device)
    levels = pyr(batch['stim'])  # List[[B,C,T,H_l,W_l]]

    # 2) Initialize kxy using image pixel units; base freq ~ 0.15 cycles/pixel (tunable)
    S = len(levels)
    kxy = init_kxy(S=S, M=n_pairs, base_freq_cpx=0.15, device=batch['stim'].device)

    # 3) Infer beh_dim from batch if not provided
    if beh_dim is None:
        beh_dim = batch['behavior'].shape[-1]

    # 4) Instantiate core with input channels per level = C_in from pyramid (likely 1 per stream)
    in_ch_per_level = levels[0].shape[1]
    core = PolarV1Core(in_ch_per_level=in_ch_per_level,
                       pairs=n_pairs,
                       kxy=kxy,
                       dt=dt,
                       n_neurons=batch['robs'].shape[-1],
                       proj_dim=proj_dim,
                       beh_dim=beh_dim).to(batch['stim'].device)
    return core, pyr, levels


def one_step_with_jepa(core: PolarV1Core,
                       pyr: PyramidAdapter,
                       batch_ctx: Dict[str, torch.Tensor],
                       batch_tgt: Dict[str, torch.Tensor],
                       lambda_jepa: float = 0.5):
    beh_ctx = batch_ctx['behavior']  # [B,Tc,D]
    beh_tgt = batch_tgt['behavior']  # [B,Tc,D]

    levels_ctx = pyr(batch_ctx['stim'])
    levels_tgt = pyr(batch_tgt['stim'])

    # --- aggregate spikes over the context window ---
    # if robs is [B,Tc,N], sum counts (or mean*Tc if you stored rates)
    if batch_ctx['robs'].dim() == 3:
        spikes_ctx = batch_ctx['robs'].sum(dim=1)  # [B,N]
    else:
        spikes_ctx = batch_ctx['robs']             # already [B,N]

    # student forward on context
    rate, feats_ctx = core(levels_ctx, beh_ctx, n_neurons=spikes_ctx.shape[-1])  # rate [B,N]

    # Sanity check (can remove later)
    assert rate.shape == (spikes_ctx.shape[0], spikes_ctx.shape[1]), f"{rate.shape} vs {spikes_ctx.shape}"

    # Poisson NLL on window-summed counts:
    # (optionally multiply rate by window duration if your rate is in Hz and robs are counts)
    # win_sec = beh_ctx.shape[1] * core.dyn.dt
    # lam_win = rate * win_sec
    lam_win = rate
    nll = (lam_win - spikes_ctx * torch.log(torch.clamp(lam_win, 1e-6))).mean()

    # teacher features from target window (stop-grad via no_grad)
    with torch.no_grad():
        feats_tgt, _, _ = core.forward_trunk(levels_tgt, beh_tgt)

    jloss = core.jepa_loss(feats_ctx, feats_tgt, mask_ratio=0.5)
    loss = nll + lambda_jepa * jloss
    return loss, {"nll": nll.detach(), "jepa": jloss.detach(), "rate": rate.detach()}


# ---------------- Good initializations (practical defaults) ---------------- #

def initialize_polar_core(core: PolarV1Core):
    """Set conservative, physics-aligned initial params."""
    # zero the behavior MLP so it initially does nothing
    for m in core.beh.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    with torch.no_grad():
        core.beh.head_q.bias.fill_(-3.0)      # q≈0 (fixation)
        core.beh.head_gam.bias.zero_()        # gamma≈1+
        core.beh.head_rho.bias.zero_()        # rho≈0
        core.beh.head_veff.bias.zero_()       # v_eff≈0

        core.dyn.lambda_fix.copy_(torch.tensor(10.0))
        core.dyn.lambda_sac.copy_(torch.tensor(40.0))
        core.dyn.a_bar.data.mul_(0.5)

    # tame the spatial filter scales
    for p in core.qfb.parameters():
        if p.dim() >= 2:
            p.data *= 0.1


# ---------------- Quick sanity test on your batch ---------------- #

"""
Usage inside your script after you build `batch` and put tensors on device:

with AMP_BF16():
    core, pyr, levels = build_polar_core_from_batch(batch, n_pairs=16, dt=1/240)
    initialize_polar_core(core)

    # Create a simple future-shifted batch for JEPA (Δ frames)
    Δ = 5  # ~20.8 ms at 240 Hz
    batch_ctx = {k: v[:, :, :-Δ] if k=='stim' else v for k,v in batch.items()}
    batch_tgt = {k: v[:, :, Δ:]  if k=='stim' else v for k,v in batch.items()}

    opt = torch.optim.AdamW(core.parameters(), lr=1e-3, weight_decay=1e-4)
    loss, logs = one_step_with_jepa(core, pyr, batch_ctx, batch_tgt, lambda_jepa=0.5)
    loss.backward();
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    opt.step(); core.jepa_ema_update()

print({k: float(v) for k,v in logs.items()})
"""

# ---------------- Main Polar-V1 core ---------------- #

class PolarV1Core(nn.Module):
    def __init__(self,
                 in_ch_per_level: int,
                 pairs: int,
                 kxy: torch.Tensor,  # [S,M,2]
                 dt: float,
                 n_neurons: int,
                 alpha_fast: float = 0.74,
                 alpha_slow: float = 0.95,
                 proj_dim: int = 256,
                 beh_dim: int = 128,
                 ):
        super().__init__()
        S, M, _ = kxy.shape
        self.S = S; self.M = M
        # Spatial quadrature filters shared across levels
        self.qfb = QuadratureFilterBank2D(in_ch=in_ch_per_level, pairs=pairs, kernel=7)
        # Polar factorization
        self.polar = PolarDecompose()
        # Behavior
        self.beh = BehaviorEncoder(M=pairs, S=S, beh_dim=beh_dim)
        # Dynamics
        self.dyn = PolarDynamics(kxy=kxy, dt=dt)
        # Temporal summarizer
        self.sum = TemporalSummarizer(alpha_fast=alpha_fast, alpha_slow=alpha_slow)
        # Readout will be constructed lazily when first shapes are seen
        self.readout: Optional[MultiLevelReadout] = None
        self.jepa = None
        self.proj_dim = proj_dim

    def _ensure_heads(self, feats_per_level: List[torch.Tensor], n_neurons: int):
        if self.readout is None:
            C_per_level = [f.shape[1] for f in feats_per_level]
            level_shapes = [(C_per_level[i], feats_per_level[i].shape[2], feats_per_level[i].shape[3])
                            for i in range(len(feats_per_level))]
            device = feats_per_level[0].device
            self.readout = MultiLevelReadout(level_shapes, C_per_level, n_neurons).to(device)
            self.jepa    = JEPAModule(C_per_level=C_per_level, proj_dim=self.proj_dim).to(device)

    def forward_trunk(self,
                      levels: List[torch.Tensor],   # [B, C_in, T, H_l, W_l]
                      beh_bcode: torch.Tensor,      # [B, T, D]
                      ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # 1) spatial quadrature per level
        even_list, odd_list = self.qfb(levels)
        # 2) polar decompose
        A_list, U_list = self.polar(even_list, odd_list)
        # 3) behavior encode
        beh_code = self.beh(beh_bcode)
        # 4) polar dynamics (advance within the window)
        A_adv, U_adv = self.dyn(A_list, U_list, beh_code=beh_code)
        # 5) temporal summarization (collapse T)
        feats = self.sum(A_adv, U_adv)  # list of [B, C_sum, H, W] per level

        # Sanity checks (can remove later)
        for li, f in enumerate(feats):
            assert f.dim() == 4, f"feats[{li}] must be [B,C,H,W], got {f.shape}"
            assert torch.isfinite(f).all(), f"NaNs in feats[{li}]"

        return feats, A_adv, U_adv

    def forward(self,
                levels: List[torch.Tensor],
                beh_bcode: torch.Tensor,
                n_neurons: int,
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats, _, _ = self.forward_trunk(levels, beh_bcode)
        self._ensure_heads(feats, n_neurons)
        # Poisson rate linear head via readout
        logits = self.readout(feats)  # [B, n_neurons]
        rate = F.softplus(logits) + 1e-4
        return rate, feats

    @torch.no_grad()
    def jepa_ema_update(self):
        if self.jepa is not None:
            self.jepa.ema_update()

    def jepa_loss(self,
                  feats_ctx: List[torch.Tensor],
                  feats_tgt: List[torch.Tensor],
                  mask_ratio: float = 0.5) -> torch.Tensor:
        assert self.jepa is not None, "Call forward once to initialize JEPA heads."
        return self.jepa(feats_ctx, feats_tgt, mask_ratio=mask_ratio)


# ---------------- Example training step (skeleton) ---------------- #

class PolarTrainer:
    def __init__(self, core: PolarV1Core, n_neurons: int, lambda_jepa: float = 0.5):
        self.core = core
        self.n_neurons = n_neurons
        self.lambda_jepa = lambda_jepa

    def step(self,
             levels_ctx: List[torch.Tensor],  # context window tensors per level
             levels_tgt: List[torch.Tensor],  # future window tensors per level
             beh_ctx: Dict[str, torch.Tensor],
             beh_tgt: Dict[str, torch.Tensor],
             spikes: torch.Tensor,            # [B, n_neurons]
             optimizer: torch.optim.Optimizer
             ) -> Dict[str, torch.Tensor]:
        core = self.core
        optimizer.zero_grad(set_to_none=True)
        # Forward student trunk on context (produces rate for supervised loss and feats for JEPA)
        rate, feats_ctx = core(levels_ctx, beh_ctx, n_neurons=self.n_neurons)
        # Poisson NLL
        # clamp to avoid log(0); spikes integer counts per window
        nll = (rate - spikes*torch.log(torch.clamp(rate, min=1e-6))).mean()
        # Teacher trunk on target window (stop-grad outside since core is shared; here we only use summarized feats for JEPA)
        with torch.no_grad():
            feats_tgt, _, _ = core.forward_trunk(levels_tgt, beh_tgt)
        # JEPA loss
        if core.jepa is None:
            core._ensure_heads(feats_ctx, self.n_neurons)
        jloss = core.jepa_loss(feats_ctx, feats_tgt, mask_ratio=0.5)
        loss = nll + self.lambda_jepa * jloss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimizer.step()
        # EMA update for JEPA projectors
        core.jepa_ema_update()
        return {"loss": loss.detach(), "nll": nll.detach(), "jepa": jloss.detach(), "rate": rate.detach()}


# ---------------- Notes ---------------- #
# • Provide kxy per (level, pair) from your filter parameterization. If you have orientation θ and spatial frequency |k|, then kxy = |k|*[cosθ, sinθ].
# • The BehaviorEncoder expects precomputed retinal velocity (after gaze warp). If you prefer world coords, adjust the sign.
# • The TemporalSummarizer outputs 5 summaries per pair; adjust to 4 or 6 as desired.
# • MultiLevelReadout here sums across levels; you can swap for low-rank shared basis across levels if you prefer.
# • JEPAModule masks spatial tokens only. For strict “future” prediction, pass feats from a future window in feats_tgt; you can also add temporal offsets per level.
# • All pieces are lightweight, equivariant to translation, and align with the polar-dynamics hypothesis.


#%%
from models.config_loader import load_dataset_configs
import os
config_path = Path("/home/jake/repos/VisionCore/experiments/model_configs/res_small_gru.yaml")

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_cones_120_backimage_all_eyepos.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

#%% Initialize model
config = load_config(config_path)
model = build_model(config, dataset_configs).to(device)

# run model readout forward with dummy input
# with torch.no_grad():
#     output = model.readouts[0](torch.randn(1, 256, 1, 9, 9).to(device))

# model.readouts[0].plot_weights()


#%% Load Data
import contextlib

train_datasets = {}
val_datasets = {}
updated_configs = []

for i, dataset_config in enumerate(dataset_configs):
    if i > 1: break

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):          # ← optional
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)

    # cast to bfloat16
    train_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    val_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    
    dataset_name = f"dataset_{i}"
    train_datasets[dataset_name] = train_dset
    val_datasets[dataset_name] = val_dset
    updated_configs.append(dataset_config)

    print(f"Dataset {i}: {len(train_dset)} train, {len(val_dset)} val samples")

#%%
plt.plot(train_datasets['dataset_0'].dsets[0]['stim'][:1000,0,25,25].float().cpu().numpy())

#%% prepare dataloaders

from torchvision.utils import make_grid
def plot_frames(frames, nrow=10, normalize=True):
    fig_w = make_grid(frames, nrow=nrow, normalize=normalize)
    plt.imshow(fig_w.permute(1, 2, 0).numpy()
               )
plot_frames(train_datasets['dataset_0'].dsets[0]['stim'][:150,[0],:,:].float())

from plenoptic.simulate import LaplacianPyramid
from einops import rearrange




# train_loader, val_loader = create_multidataset_loaders(train_datasets, val_datasets, batch_size=2, num_workers=os.cpu_count()//2)

#%% test one dataset
batch_size = 256
dataset_id = 0

ntrain = len(train_datasets[f'dataset_{dataset_id}'])
# inds = np.random.randint(0, ntrain - batch_size, batch_size)
inds = np.arange(1000, 1000+batch_size)
batch = train_datasets[f'dataset_{dataset_id}'][inds]

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

#%%
from torch import optim

# 0) Build the tiny physics-aware encoder (BxTx2 -> BxTxD)
genc = MinimalGazeEncoder(d_out=128, dt=1/240.0, use_pos_fourier=True, Kpos=2).to(device)

with AMP_BF16():
    # 1) Turn raw gaze (B,T,2) into behavior code (B,T,128)
    #    If your batch['behavior'] is already gaze_xy, this line is all you need:
    bcode = genc(batch['behavior'])                 # [B,T,128]
    batch = {**batch, 'behavior': bcode}            # overwrite with code

    # 2) Build core + pyramid (core expects tensor behavior)
    core, pyr, levels = build_polar_core_from_batch(
        batch, n_pairs=16, dt=1/240, proj_dim=256)
    initialize_polar_core(core)

    # 3) Create context/target windows for JEPA (slice BOTH stim and behavior along time)
    Δ = 5  # ~20.8 ms at 240 Hz
    Tc = batch['stim'].shape[2] - Δ
    batch_ctx = {
        **batch,
        'stim':     batch['stim'][:, :, :-Δ],       # [B,C,Tc,H,W]
        'behavior': batch['behavior'][:, :-Δ],      # [B,Tc,D]
        'robs':     batch['robs'][:, :Tc],          # [B,Tc,N] - slice to match context window
    }
    batch_tgt = {
        **batch,
        'stim':     batch['stim'][:, :,  Δ:],       # [B,C,Tc,H,W]
        'behavior': batch['behavior'][:,  Δ:],      # [B,Tc,D]
        'robs':     batch['robs'][:,  Δ:],          # [B,Tc,N] - not used in loss, but consistent
    }

    # 4) One optimization step (Poisson NLL + λ·JEPA)
    opt = optim.AdamW(core.parameters(), lr=1e-3, weight_decay=1e-4)
    loss, logs = one_step_with_jepa(core, pyr, batch_ctx, batch_tgt, lambda_jepa=0.5)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    opt.step()
    core.jepa_ema_update()

# print({k: float(v) for k, v in logs.items()})

# %%
