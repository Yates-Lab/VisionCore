# ── DataYatesV1/models/modulator.py ────────────────────────────────────────────
"""
Modulators work on two inputs (features, behaviour)
which they combine into one output.

Behavior data has shape (N, n_vars) with no temporal dimension,
so it modulates all timesteps identically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .mlp import MLP

__all__ = ['ConcatModulator', 'FiLMModulator', 'BaseModulator', 'MODULATORS']


class BaseModulator(nn.Module):
    """Base class for configurable modulators."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.behavior_dim = config.get('behavior_dim', 2)
        self.conv_channels = config.get('feature_dim', None)
        self.out_dim = None  # Will be set by child classes
        self._build_modulator()

    def _build_modulator(self):
        raise NotImplementedError

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for modulator.

        Args:
            feats: Convnet features with shape (N, C_conv, T, H, W)
            beh: Behavior data with shape (N, n_vars)

        Returns:
            Modulated features with shape (N, C_out, T, H, W)
        """
        raise NotImplementedError


class ConcatModulator(BaseModulator):
    """
    Concatenation modulator that processes behavior through an MLP
    and concatenates the result with convnet features along the channel dimension.
    """

    def _build_modulator(self):
        """Build the MLP encoder and set output dimensions."""
        # Get encoder parameters
        encoder_params = self.config.get('encoder_params', {})
        encoder_type = encoder_params.get('type', 'mlp')

        if encoder_type != 'mlp':
            raise ValueError(f"Only 'mlp' encoder type is supported, got {encoder_type}")

        # Extract MLP parameters with defaults
        mlp_params = encoder_params.copy()
        mlp_params.pop('type', None)  # Remove type field

        # Set defaults for performance
        dims = mlp_params.get('dims', [64, 64])
        activation = mlp_params.get('activation', 'gelu')
        bias = mlp_params.get('bias', True)
        residual = mlp_params.get('residual', True)
        dropout = mlp_params.get('dropout', 0.0)
        last_layer_activation = mlp_params.get('last_layer_activation', False)

        # Output dimension is the last element in dims
        output_dim = dims[-1] if dims else 64

        # Build MLP
        self.encoder = MLP(
            input_dim=self.behavior_dim,
            hidden_dims=dims[:-1],  # All but last are hidden layers
            output_dim=output_dim,
            act_type=activation,
            bias=bias,
            residual=residual,
            dropout=dropout,
            output_activation=last_layer_activation
        )

        # Set output dimension for factory
        self.out_dim = output_dim

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for concatenation modulator.

        Args:
            feats: Convnet features with shape (N, C_conv, T, H, W)
            beh: Behavior data with shape (N, n_vars)

        Returns:
            Concatenated features with shape (N, C_conv + C_mod, T, H, W)
        """
        # Process behavior through MLP: (N, n_vars) -> (N, C_mod)
        beh_encoded = self.encoder(beh)

        # Get spatial and temporal dimensions from features
        N, C_conv, T, H, W = feats.shape
        C_mod = beh_encoded.shape[1]

        # Expand behavior to match feature dimensions: (N, C_mod) -> (N, C_mod, T, H, W)
        # Add singleton dimensions and broadcast
        beh_expanded = beh_encoded.view(N, C_mod, 1, 1, 1).expand(N, C_mod, T, H, W)

        # Concatenate along channel dimension
        return torch.cat([feats, beh_expanded], dim=1)


class FiLMModulator(BaseModulator):
    """
    Feature-wise Linear Modulation (FiLM) modulator that processes behavior
    through an MLP and applies scale and shift transformations to convnet features.
    """

    def _build_modulator(self):
        """Build the MLP encoder and linear layers for scale/shift parameters."""
        # Get feature dimension (number of convnet output channels)
        self.feature_dim = self.config.get('feature_dim')
        if self.feature_dim is None:
            raise ValueError("feature_dim must be specified in config for FiLM modulator")

        # Get encoder parameters
        encoder_params = self.config.get('encoder_params', {})
        encoder_type = encoder_params.get('type', 'mlp')

        if encoder_type != 'mlp':
            raise ValueError(f"Only 'mlp' encoder type is supported, got {encoder_type}")

        # Extract MLP parameters with defaults
        mlp_params = encoder_params.copy()
        mlp_params.pop('type', None)  # Remove type field

        # Set defaults for performance
        dims = mlp_params.get('dims', [64, 64])
        activation = mlp_params.get('activation', 'gelu')
        bias = mlp_params.get('bias', True)
        residual = mlp_params.get('residual', True)
        dropout = mlp_params.get('dropout', 0.0)
        last_layer_activation = mlp_params.get('last_layer_activation', False)

        # Hidden dimension is the last element in dims
        hidden_dim = dims[-1] if dims else 64

        # Build MLP encoder
        self.encoder = MLP(
            input_dim=self.behavior_dim,
            hidden_dims=dims[:-1],  # All but last are hidden layers
            output_dim=hidden_dim,
            act_type=activation,
            bias=bias,
            residual=residual,
            dropout=dropout,
            output_activation=last_layer_activation
        )

        # Create linear layers for scale and shift parameters
        self.scale_layer = nn.Linear(hidden_dim, self.feature_dim, bias=True)
        self.shift_layer = nn.Linear(hidden_dim, self.feature_dim, bias=True)

        # Initialize scale to 1 and shift to 0 for stable training
        nn.init.ones_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)
        nn.init.zeros_(self.shift_layer.weight)
        nn.init.zeros_(self.shift_layer.bias)

        # Output dimension is same as input (FiLM doesn't change channel count)
        # For FiLM, we return 0 to indicate no change in channel count
        self.out_dim = 0

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FiLM modulator.

        Args:
            feats: Convnet features with shape (N, C_conv, T, H, W)
            beh: Behavior data with shape (N, n_vars)

        Returns:
            Modulated features with shape (N, C_conv, T, H, W)
        """
        # Process behavior through MLP: (N, n_vars) -> (N, hidden_dim)
        beh_encoded = self.encoder(beh)

        # Get dimensions
        N, C_conv, T, H, W = feats.shape

        # Validate that feature dimensions match config
        if C_conv != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {C_conv}")

        # Generate scale and shift parameters: (N, hidden_dim) -> (N, C_conv)
        scale = self.scale_layer(beh_encoded)  # (N, C_conv)
        shift = self.shift_layer(beh_encoded)  # (N, C_conv)

        # Expand to match feature dimensions: (N, C_conv) -> (N, C_conv, T, H, W)
        scale = scale.view(N, C_conv, 1, 1, 1).expand(N, C_conv, T, H, W)
        shift = shift.view(N, C_conv, 1, 1, 1).expand(N, C_conv, T, H, W)

        # Apply FiLM transformation: output = scale * features + shift
        return scale * feats + shift


# Dictionary mapping modulator types to their classes
MODULATORS = {
    'concat': ConcatModulator,
    'film': FiLMModulator,
}


