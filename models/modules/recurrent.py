# modules/recurrent.py
import torch
import torch.nn as nn
from typing import Optional, List, Union
import torch.nn.init as init
from einops import rearrange

__all__ = ["ConvGRUCell", "ConvGRU", "RecurrentWrapper"]


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

        # Gates: Use Conv2d for spatial processing
        self.update_gate = nn.Conv2d(ch, hidden_size, kernel_size=kernel_size,
                                     padding=p, bias=True)
        self.reset_gate  = nn.Conv2d(ch, hidden_size, kernel_size=kernel_size,
                                     padding=p, bias=True)
        # Candidate uses [x, r⊙h] which has the same channel count as [x,h]
        self.out_gate    = nn.Conv2d(ch, hidden_size, kernel_size=kernel_size,
                                     padding=p, bias=True)

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
        # x: (B, C_in, H, W)
        # h: (B, C_hid, H, W) or None
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

    Matches VisionCore interface: input/output are (B, C, T, H, W)

    Args:
        input_size:   channels of input frames
        hidden_sizes: int or list[int], hidden size per layer
        kernel_sizes: int or list[int], kernel per layer (odd recommended)
        n_layers:     number of layers

    Forward:
        x: (B, C_in, T, H, W)
        hidden: optional list[Tensor] of length n_layers, each (B, C_hid_i, H, W)

    Returns:
        output: (B, C_hidden, T, H, W) - full sequence
    """
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Union[int, List[int]],
                 kernel_sizes: Union[int, List[int]],
                 n_layers: int = 1):
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
        self.output_channels = ch

    def forward(self,
                x: torch.Tensor,                      # (B, C, T, H, W)
                hidden: Optional[List[torch.Tensor]] = None):

        # Rearrange from (B, C, T, H, W) to (B, T, C, H, W) for time-first processing
        x = rearrange(x, 'b c t h w -> b t c h w')

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
            outputs.append(layer_h[-1])    # top layer output at time t

        # Stack outputs and rearrange back to (B, C, T, H, W)
        y_seq = torch.stack(outputs, dim=1)  # (B, T, C_top, H, W)
        y_seq = rearrange(y_seq, 'b t c h w -> b c t h w')

        return y_seq


class RecurrentWrapper(nn.Module):
    """
    Wrapper for nn.GRU/nn.LSTM to match VisionCore interface.

    Converts (B, C, T, H, W) → (B, T, C*H*W) → RNN → (B, hidden_size, T, 1, 1)

    Args:
        rnn_type: 'gru' or 'lstm'
        hidden_size: hidden dimension for RNN
        input_channels: number of input channels (C) - used for output_channels tracking
        **rnn_kwargs: additional arguments for nn.GRU or nn.LSTM
    """
    def __init__(self, rnn_type: str, hidden_size: int, input_channels: int, **rnn_kwargs):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.output_channels = hidden_size
        self.rnn_kwargs = rnn_kwargs
        self.rnn = None  # Will be created on first forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)

        Returns:
            output: (B, hidden_size, T, 1, 1)
        """
        B, C, T, H, W = x.shape

        # Create RNN on first forward pass (lazy initialization)
        if self.rnn is None:
            input_size = C * H * W
            if self.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size, self.hidden_size, batch_first=True, **self.rnn_kwargs)
            elif self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size, self.hidden_size, batch_first=True, **self.rnn_kwargs)
            else:
                raise ValueError(f"Unknown RNN type: {self.rnn_type}")
            # Move to same device as input
            self.rnn = self.rnn.to(x.device)

        # Flatten spatial dimensions: (B, C, T, H, W) → (B, T, C*H*W)
        x = rearrange(x, 'b c t h w -> b t (c h w)')

        # Pass through RNN (batch_first=True)
        # output: (B, T, hidden_size)
        output, _ = self.rnn(x)

        # Reshape to match VisionCore format: (B, T, hidden_size) → (B, hidden_size, T, 1, 1)
        output = rearrange(output, 'b t h -> b h t 1 1')

        return output