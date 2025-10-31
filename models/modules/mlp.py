import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from typing import List, Dict, Any, Tuple, Union, Optional

from .norm_act_pool import get_activation_layer

__all__ = ['MLP', 'MLPBlock']

class MLPBlock(nn.Module):
    """
    Basic building block for MLPs.

    Structure: Linear -> [Norm] -> Activation -> [Dropout]
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 norm_type: Optional[str] = None,
                 act_type: str = 'relu',
                 dropout: float = 0.0,
                 bias: bool = True,
                 norm_params: Optional[Dict[str, Any]] = None,
                 act_params: Optional[Dict[str, Any]] = None,
                 spectral_norm: bool = False):
        super().__init__()

        # Create linear layer
        linear = nn.Linear(in_features, out_features, bias=bias)

        # Apply spectral normalization if requested
        if spectral_norm:
            self.linear = nn_utils.spectral_norm(linear)
        else:
            self.linear = linear

        # Normalization (optional)
        if norm_type and norm_type.lower() != 'none':
            from .norm_act_pool import get_norm_layer
            self.norm = get_norm_layer(norm_type, out_features, 1, norm_params)
        else:
            self.norm = nn.Identity()

        # Activation
        self.act = get_activation_layer(act_type, act_params)

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """
    Configurable MLP with support for:
    - Variable hidden layer sizes
    - Different activation functions
    - Normalization
    - Dropout
    - Residual connections
    - Spectral normalization (for stability during simulation)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 norm_type: Optional[str] = None,
                 act_type: str = 'relu',
                 dropout: float = 0.0,
                 bias: bool = True,
                 residual: bool = False,
                 norm_params: Optional[Dict[str, Any]] = None,
                 act_params: Optional[Dict[str, Any]] = None,
                 output_activation: bool = False,
                 spectral_norm: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual

        # Build layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(
                MLPBlock(
                    prev_dim,
                    hidden_dim,
                    norm_type=norm_type,
                    act_type=act_type,
                    dropout=dropout,
                    bias=bias,
                    norm_params=norm_params,
                    act_params=act_params,
                    spectral_norm=spectral_norm
                )
            )
            prev_dim = hidden_dim

        # Output layer
        if output_activation:
            layers.append(
                MLPBlock(
                    prev_dim,
                    output_dim,
                    norm_type=norm_type,
                    act_type=act_type,
                    dropout=0.0,  # No dropout on output
                    bias=bias,
                    norm_params=norm_params,
                    act_params=act_params,
                    spectral_norm=spectral_norm
                )
            )
        else:
            # Apply spectral norm to output layer if requested
            output_layer = nn.Linear(prev_dim, output_dim, bias=bias)
            if spectral_norm:
                output_layer = nn_utils.spectral_norm(output_layer)
            layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.residual and self.input_dim == self.output_dim:
            out = out + x
        return out