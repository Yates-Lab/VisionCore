# modules/recurrent.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .common import chomp

__all__ = ["ConvGRUCell", "ConvGRU", "ConvLSTM", "create_recurrent"]

class _GateConv(nn.Conv2d):
    """Conv2d with (fan_in+fan_out) Xavier init—good for RNN gates."""
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

class ConvGRUCell(nn.Module):
    r"""
    (B,C,H,W) × (B,C,H,W) → new h
    """
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.gates = _GateConv(in_ch + hid_ch, 3 * hid_ch, k, padding=p)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r, n = self.gates(torch.cat([x, h], 1)).chunk(3, 1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(n + r * h)
        return (1. - z) * h + z * n

class ConvGRUCellFast(nn.Module):
    """
    Optimized ConvGRU cell with modern improvements:
    - Layer normalization for stability
    - Depthwise-separable convolutions (optional)
    - Grouped convolutions for gates (optional)
    - Proper reset gate application (FIXED BUG!)

    Implements:   g = W_x * x   +   LN(W_h * h)
                  (z, r, n) = split(g)
                  h' = (1-z) ⊙ h  +  z ⊙ tanh(n + r ⊙ h)

    `conv_x` is applied outside the time-loop, so forward() receives its
    pre-computed value `x_proj` instead of raw x.
    """
    def __init__(self,
                 in_ch: int,
                 hid_ch: int,
                 k: int = 3,
                 use_layer_norm: bool = True,
                 use_depthwise: bool = False,
                 use_grouped: bool = False,
                 num_groups: int = 1):
        super().__init__()
        p = k // 2
        self.use_layer_norm = use_layer_norm
        self.hid_ch = hid_ch

        # Input projection (can use depthwise-separable)
        if use_depthwise and in_ch >= 3:
            # Depthwise-separable: depthwise + pointwise
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, 3 * hid_ch, 1, bias=True)
            )
        else:
            # Standard convolution (with optional grouping)
            groups_x = min(num_groups, in_ch) if use_grouped else 1
            self.conv_x = nn.Conv2d(in_ch, 3 * hid_ch, k, padding=p, bias=True, groups=groups_x)

        # Recurrent projection (can use depthwise-separable)
        if use_depthwise and hid_ch >= 3:
            self.conv_h = nn.Sequential(
                nn.Conv2d(hid_ch, hid_ch, k, padding=p, groups=hid_ch, bias=False),
                nn.Conv2d(hid_ch, 3 * hid_ch, 1, bias=False)
            )
        else:
            # Standard convolution (with optional grouping)
            groups_h = min(num_groups, hid_ch) if use_grouped else 1
            self.conv_h = nn.Conv2d(hid_ch, 3 * hid_ch, k, padding=p, bias=False, groups=groups_h)

        # Layer normalization for hidden state (applied before gating)
        if use_layer_norm:
            # Use GroupNorm as a proxy for LayerNorm (more efficient for spatial data)
            self.layer_norm = nn.GroupNorm(num_groups=min(32, hid_ch), num_channels=3 * hid_ch, eps=1e-5, affine=True)
        else:
            self.layer_norm = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with good defaults for GRU."""
        # Input projection: Kaiming initialization
        if isinstance(self.conv_x, nn.Sequential):
            nn.init.kaiming_uniform_(self.conv_x[0].weight, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.conv_x[1].weight, a=5 ** 0.5)
        else:
            nn.init.kaiming_uniform_(self.conv_x.weight, a=5 ** 0.5)

        # Recurrent projection: Orthogonal initialization (better for RNNs)
        if isinstance(self.conv_h, nn.Sequential):
            nn.init.orthogonal_(self.conv_h[0].weight)
            nn.init.orthogonal_(self.conv_h[1].weight)
        else:
            nn.init.orthogonal_(self.conv_h.weight)

    def forward(self, x_proj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Compute recurrent contribution with layer norm
        h_contrib = self.layer_norm(self.conv_h(h))

        # Split into gates
        z, r, n = (x_proj + h_contrib).chunk(3, 1)

        # Apply gate activations
        z = torch.sigmoid(z)  # Update gate
        r = torch.sigmoid(r)  # Reset gate

        # FIXED: Apply reset gate to hidden state before computing candidate
        # This was the bug - we were missing r * h
        n = torch.tanh(n + r * h)

        # Update hidden state
        h_new = (1 - z) * h + z * n

        return h_new


class ConvLSTMCell(nn.Module):
    r"""
    (B,C,H,W) × (h,c) → new (h,c)
    """
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.gates = _GateConv(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c



# wrappers: these are written this way to optimize for torch.compile / func.scan
# --------------------------------------------------------------------- #

class ConvGRU(nn.Module):
    """
    Optimized ConvGRU with modern improvements:
    - Learnable initial hidden state
    - Optional residual connections
    - Layer normalization
    - Depthwise-separable convolutions
    - Gradient clipping option

    Input : (B, T, C_in, H, W)
    Return: final hidden state  (B, C_hid, H, W) if fast_phase=False
            concatenated input+hidden (B, C_in + C_hid, H, W) if fast_phase=True
    """
    def __init__(self,
                 in_ch: int,
                 hid_ch: int,
                 k: int = 3,
                 fast_phase: bool = False,
                 use_layer_norm: bool = True,
                 use_depthwise: bool = False,
                 use_grouped: bool = False,
                 num_groups: int = 1,
                 learnable_h0: bool = True,
                 use_residual: bool = False,
                 grad_clip_val: Optional[float] = None):

        super().__init__()
        self.cell = ConvGRUCellFast(
            in_ch, hid_ch, k,
            use_layer_norm=use_layer_norm,
            use_depthwise=use_depthwise,
            use_grouped=use_grouped,
            num_groups=num_groups
        )
        self.fast_phase = fast_phase
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.learnable_h0 = learnable_h0
        self.use_residual = use_residual and (in_ch == hid_ch)  # Only if dimensions match
        self.grad_clip_val = grad_clip_val

        # Learnable initial hidden state (broadcasted to batch size and spatial dims)
        if learnable_h0:
            self.h0 = nn.Parameter(torch.zeros(1, hid_ch, 1, 1))
        else:
            self.register_parameter('h0', None)

    def _init_hidden(self, x):
        """Initialize hidden state (learnable or zeros)."""
        B, _, _, H, W = x.shape

        if self.learnable_h0:
            # Broadcast learnable h0 to batch and spatial dimensions
            return self.h0.expand(B, self.hid_ch, H, W).contiguous()
        else:
            # Zero initialization
            return torch.zeros(B, self.hid_ch, H, W, dtype=x.dtype, device=x.device)

    def forward(self, x, chunk_size: int = 4):

        # move T next to batch (outside the for loop)
        x = x.permute(0, 2, 1, 3, 4)     # (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Store last timestep if fast_phase is enabled or for residual connection
        if self.fast_phase or self.use_residual:
            x_last = x[:, -1]  # (B, C, H, W) - last timestep

        h = self._init_hidden(x)

        # Process through time with chunking for memory efficiency
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            x_proj = self.cell.conv_x(
                x[:, t0:t1].reshape(-1, C, H, W)
            ).view(B, t1 - t0, -1, H, W)

            for τ in range(t1 - t0):
                h = self.cell(x_proj[:, τ], h)

                # Optional gradient clipping for stability
                if self.grad_clip_val is not None and self.training:
                    h = h.clamp(-self.grad_clip_val, self.grad_clip_val)

            del x_proj

        # Optional residual connection (when in_ch == hid_ch)
        if self.use_residual:
            # Align spatial dimensions if needed
            if x_last.shape[-2:] != h.shape[-2:]:
                if x_last.shape[-2:] > h.shape[-2:]:
                    x_last = chomp(x_last, h.shape[-2:])
                else:
                    h = chomp(h, x_last.shape[-2:])
            h = h + x_last  # Residual connection

        # Concatenate input with hidden state if fast_phase is enabled
        if self.fast_phase:
            # Align spatial dimensions using chomp (crop to smaller size)
            if x_last.shape[-2:] != h.shape[-2:]:
                if x_last.shape[-2:] > h.shape[-2:]:
                    x_last = chomp(x_last, h.shape[-2:])
                else:
                    h = chomp(h, x_last.shape[-2:])

            # Concatenate along channel dimension
            return torch.cat([x_last, h], dim=1)  # (B, C_in + C_hid, H, W)

        return h


class ConvGRUslow(nn.Module):
    """
    Input  : (B, T, C_in, H, W)
    Output : (B, C_hid, H, W) if fast_phase=False
             concatenated input+hidden (B, C_in + C_hid, H, W) if fast_phase=True
    """
    def __init__(
        self,
        in_ch: int,
        hid_ch: int,
        k: int = 3,
        fast_phase: bool = False,
    ):
        super().__init__()
        self.cell = ConvGRUCell(in_ch, hid_ch, k)
        self.fast_phase = fast_phase
        self.in_ch = in_ch

    @torch.no_grad()
    def _init_hidden(self, x):
        B, _, _, H, W = x.shape
        return torch.zeros(
            B, self.cell.gates.out_channels // 3, H, W,
            dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move T next to batch (outside the for loop)
        x = x.permute(0, 2, 1, 3, 4)     # (B, T, C, H, W)

        B, T, *_ = x.shape

        # Store last timestep if fast_phase is enabled
        if self.fast_phase:
            x_last = x[:, -1]  # (B, C, H, W) - last timestep

        h = self._init_hidden(x)
        for t in range(T):            # scripted / compiled away
            h = self.cell(x[:, t], h)

        # Concatenate input with hidden state if fast_phase is enabled
        if self.fast_phase:
            # Align spatial dimensions using chomp (crop to smaller size)
            if x_last.shape[-2:] != h.shape[-2:]:
                if x_last.shape[-2:] > h.shape[-2:]:
                    x_last = chomp(x_last, h.shape[-2:])
                else:
                    h = chomp(h, x_last.shape[-2:])

            # Concatenate along channel dimension
            return torch.cat([x_last, h], dim=1)  # (B, C_in + C_hid, H, W)

        return h


class ConvLSTM(nn.Module):
    """
    Input  : (B, T, C_in, H, W)
    Output : (B, C_hid, H, W) if fast_phase=False
             concatenated input+hidden (B, C_in + C_hid, H, W) if fast_phase=True
    """
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3, fast_phase: bool = False):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, k)
        self.fast_phase = fast_phase
        self.in_ch = in_ch

    @torch.no_grad()
    def _init_state(self, x):
        B, _, _, H, W = x.shape
        C = self.cell.gates.out_channels // 4
        z = torch.zeros(B, C, H, W, dtype=x.dtype, device=x.device)
        return z.clone(), z          # (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # move T next to batch (outside the for loop)
        x = x.permute(0, 2, 1, 3, 4)     # (B, T, C, H, W)

        # Store last timestep if fast_phase is enabled
        if self.fast_phase:
            x_last = x[:, -1]  # (B, C, H, W) - last timestep

        h, c = self._init_state(x)
        for t in range(x.size(1)):    # scripted / compiled away
            h, c = self.cell(x[:, t], (h, c))

        # Concatenate input with hidden state if fast_phase is enabled
        if self.fast_phase:
            # Align spatial dimensions using chomp (crop to smaller size)
            if x_last.shape[-2:] != h.shape[-2:]:
                if x_last.shape[-2:] > h.shape[-2:]:
                    x_last = chomp(x_last, h.shape[-2:])
                else:
                    h = chomp(h, x_last.shape[-2:])

            # Concatenate along channel dimension
            return torch.cat([x_last, h], dim=1)  # (B, C_in + C_hid, H, W)

        return h