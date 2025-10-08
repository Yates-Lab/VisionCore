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
    Implements   g = W_x * x   +   W_h * h
                 (z, r, n) = split(g)
                 h' = (1-z) ⊙ h  +  z ⊙ tanh(n + r ⊙ h)

    `conv_x` is applied outside the time-loop, so forward() receives its
    pre-computed value `x_proj` instead of raw x.
    """
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv_x = nn.Conv2d(in_ch, 3 * hid_ch, k, padding=p, bias=True)
        self.conv_h = nn.Conv2d(hid_ch, 3 * hid_ch, k, padding=p, bias=False)

        # good initialisations
        nn.init.kaiming_uniform_(self.conv_x.weight, a=5 ** 0.5)
        nn.init.orthogonal_(self.conv_h.weight)

    def forward(self, x_proj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r, n = (x_proj + self.conv_h(h)).chunk(3, 1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(n)
        return (1 - z) * h + z * n


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
    Input : (B, T, C_in, H, W)
    Return: final hidden state  (B, C_hid, H, W) if fast_phase=False
            concatenated input+hidden (B, C_in + C_hid, H, W) if fast_phase=True
    """
    def __init__(self, in_ch: int,
                hid_ch: int,
                k: int = 3,
                fast_phase: bool = False):

        super().__init__()
        self.cell = ConvGRUCellFast(in_ch, hid_ch, k)
        self.fast_phase = fast_phase
        self.in_ch = in_ch

    @torch.no_grad()
    def _init_hidden(self, x):
        B, _, _, H, W = x.shape
        C = self.cell.conv_h.out_channels // 3
        return torch.zeros(B, C, H, W, dtype=x.dtype, device=x.device)

    def forward(self, x, chunk_size: int = 4):

        # move T next to batch (outside the for loop)
        x = x.permute(0, 2, 1, 3, 4)     # (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Store last timestep if fast_phase is enabled
        if self.fast_phase:
            x_last = x[:, -1]  # (B, C, H, W) - last timestep

        h = self._init_hidden(x)
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            x_proj = self.cell.conv_x(
                x[:, t0:t1].reshape(-1, C, H, W)
            ).view(B, t1 - t0, -1, H, W)
            for τ in range(t1 - t0):
                h = self.cell(x_proj[:, τ], h)
            del x_proj

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