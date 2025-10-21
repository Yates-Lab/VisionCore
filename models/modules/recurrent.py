# modules/recurrent.py
import torch
import torch.nn as nn
from typing import Optional, List, Union
import torch.nn.init as init
from einops import rearrange

__all__ = ["ConvGRUCell", "ConvGRU"]


class ConvGRUCell(nn.Module):
    """
    A single ConvGRU cell.
    x: (B, C_in, H, W), h: (B, C_hid, H, W)
    """
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size to preserve spatial size"
        p = kernel_size // 2
        ch = input_size + hidden_size

        # Gates:
        self.update_gate = nn.Conv2d(ch, hidden_size, kernel_size, padding=p, bias=True)
        self.reset_gate  = nn.Conv2d(ch, hidden_size, kernel_size, padding=p, bias=True)
        # Candidate uses [x, r⊙h] which has the same channel count as [x,h]
        self.out_gate    = nn.Conv2d(ch, hidden_size, kernel_size, padding=p, bias=True)

        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal works well for recurrent-style weights; Xavier is also fine.
        for m in (self.update_gate, self.reset_gate, self.out_gate):
            init.orthogonal_(m.weight)
            init.constant_(m.bias, 0.0)
        # Encourage carry behavior at init
        init.constant_(self.update_gate.bias, 1.0)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, H, W, dtype=x.dtype, device=x.device)

        xh = torch.cat([x, h], dim=1)
        z  = torch.sigmoid(self.update_gate(xh))
        r  = torch.sigmoid(self.reset_gate(xh))
        n  = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


class ConvGRU(nn.Module):
    """
    Multi-layer ConvGRU over sequences.

    Args:
        input_size:   channels of input frames
        hidden_sizes: int or list[int], hidden size per layer
        kernel_sizes: int or list[int], kernel per layer (odd recommended)
        n_layers:     number of layers

    Forward:
        x: (B, T, C_in, H, W)
        hidden: optional list[Tensor] of length n_layers, each (B, C_hid_i, H, W)
        return_sequence: if True, returns (B, T, C_top, H, W) as first output

    Returns:
        if return_sequence:
            y_seq, hidden_list
            y_seq: (B, T, C_top, H, W), hidden_list: list of final states per layer
        else:
            y_last, hidden_list
            y_last: (B, C_top, H, W)
    """
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Union[int, List[int]],
                 kernel_sizes: Union[int, List[int]],
                 n_layers: int):
        super().__init__()
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes] * n_layers
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * n_layers
        assert len(hidden_sizes) == n_layers and len(kernel_sizes) == n_layers

        cells: List[ConvGRUCell] = []
        ch = input_size
        for i in range(n_layers):
            cell = ConvGRUCell(ch, hidden_sizes[i], kernel_sizes[i])
            cells.append(cell)
            ch = hidden_sizes[i]
        self.cells = nn.ModuleList(cells)

    def forward(self,
                x: torch.Tensor,                      # (B, C, T, H, W)
                hidden: Optional[List[torch.Tensor]] = None,
                return_sequence: bool = False):
        

        B, T, C, H, W = x.shape
        L = len(self.cells)

        if hidden is None:
            hidden = [None] * L
        else:
            assert len(hidden) == L, "Hidden must match number of layers"

        outputs: List[torch.Tensor] = []
        layer_h = hidden  # current hidden states per layer

        for t in range(T):
            inp = x[:, t]  # (B, C, H, W)
            new_h = []
            for i, cell in enumerate(self.cells):
                h_i = cell(inp, layer_h[i])
                new_h.append(h_i)
                inp = h_i                      # feed to next layer
            layer_h = new_h
            if return_sequence:
                outputs.append(layer_h[-1])    # top layer output at time t

        if return_sequence:
            y_seq = torch.stack(outputs, dim=1)   # (B, T, C_top, H, W)
            return y_seq, layer_h
        else:
            y_last = layer_h[-1]                  # (B, C_top, H, W)
            return y_last, layer_h