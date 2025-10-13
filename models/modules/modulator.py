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
from .recurrent import ConvGRU

__all__ = ['ConcatModulator', 'FiLMModulator', 'SpatialTransformerModulator', 'ConvGRUModulator', 'PredictiveCodingModulator', 'BaseModulator', 'MODULATORS']


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


class SpatialTransformerModulator(BaseModulator):
    """
    Channel-wise Spatial Transformer modulator.

    For each trial it predicts:
        • θ  : 6-parameter affine matrix  (2×3)       ─ spatial shift / warp
        • α  : per-channel mixing weights in [0,1]    ─ blend shifted vs. original

    Output:  α * STN(feats)  +  (1-α) * feats
             shape = (N, C, T, H, W)
    """

    def _build_modulator(self):
        # ----------- config & encoder (same style as FiLM) --------------------
        self.feature_dim = self.config.get('feature_dim')
        if self.feature_dim is None:
            raise ValueError("feature_dim must be provided for STN modulator")

        encoder_cfg  = self.config.get('encoder_params', {})
        encoder_type = encoder_cfg.get('type', 'mlp')

        if encoder_type != 'mlp':
            raise ValueError(f"Only 'mlp' encoder type is supported, got {encoder_type}")

        # Extract MLP parameters with defaults
        mlp_params = encoder_cfg.copy()
        mlp_params.pop('type', None)  # Remove type field

        dims         = mlp_params.get('dims', [128, 128])
        activation   = mlp_params.get('activation', 'gelu')
        bias         = mlp_params.get('bias', True)
        residual     = mlp_params.get('residual', True)
        dropout      = mlp_params.get('dropout', 0.0)
        out_act      = mlp_params.get('last_layer_activation', False)

        hidden_dim   = dims[-1] if dims else 128

        self.encoder = MLP(
            input_dim=self.behavior_dim,
            hidden_dims=dims[:-1],
            output_dim=hidden_dim,
            act_type=activation,
            bias=bias,
            residual=residual,
            dropout=dropout,
            output_activation=out_act
        )

        # ----------- heads ----------------------------------------------------
        # θ : 6 numbers → 2×3 affine
        self.theta_head   = nn.Linear(hidden_dim, 6, bias=True)
        # α : per-channel gating weights
        self.alpha_head   = nn.Linear(hidden_dim, self.feature_dim, bias=True)

        # Initialise θ to identity, α to 0 (no shift, use original feats)
        nn.init.zeros_(self.theta_head.weight)
        nn.init.zeros_(self.theta_head.bias)
        self.theta_head.bias.data[0] = 1.0  # a11
        self.theta_head.bias.data[4] = 1.0  # a22

        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)

        # STN keeps channel count the same
        self.out_dim = 0

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Args
            feats : (N, C, T, H, W)
            beh   : (N, n_vars)
        Returns
            (N, C, T, H, W)  – blended shifted & original features
        """
        N, C, T, H, W = feats.shape
        if C != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} channels, got {C}")

        beh_enc   = self.encoder(beh)

        # -------- θ  ----------------------------------------------------------
        theta     = self.theta_head(beh_enc)           # (N, 6)
        theta     = theta.view(N, 2, 3)                # (N, 2, 3)
        theta_rep = theta.unsqueeze(1).repeat(1, T, 1, 1)     # (N, T, 2, 3)
        theta_rep = theta_rep.view(N*T, 2, 3)

        # -------- α  ----------------------------------------------------------
        alpha     = torch.sigmoid(self.alpha_head(beh_enc))   # (N, C)
        alpha_exp = alpha.view(N, C, 1, 1, 1)                 # broadcast later

        # -------- reshape & STN  ---------------------------------------------
        feats_rsh = feats.permute(0,2,1,3,4).contiguous()     # (N, T, C, H, W)
        feats_rsh = feats_rsh.view(N*T, C, H, W)              # (N*T, C, H, W)

        grid      = F.affine_grid(theta_rep, feats_rsh.size(), align_corners=False)
        feats_shifted = F.grid_sample(feats_rsh, grid, align_corners=False)

        feats_shifted = feats_shifted.view(N, T, C, H, W).permute(0,2,1,3,4)  # (N,C,T,H,W)

        # -------- blend -------------------------------------------------------
        feats_final = alpha_exp * feats_shifted + (1 - alpha_exp) * feats
        return feats_final


class ConvGRUModulator(BaseModulator):
    """
    ConvGRU modulator that processes features and behavior through a ConvGRU.

    This modulator concatenates convnet features with embedded behavior and
    processes them through a ConvGRU. Unlike the PC modulator, it does not
    perform prediction or compute error - it simply outputs the GRU result.

    The logic is:
    1. Take features (N, C, T, H, W) and behavior (N, behavior_dim)
    2. Embed behavior spatially and concatenate with all T frames
    3. Use ConvGRU to process the concatenated sequence
    4. Output the final GRU hidden state
    """

    def _build_modulator(self):
        """Build the behavior embedding and ConvGRU processor with optimizations."""
        cfg = self.config
        self.feature_dim = cfg['feature_dim']
        self.behavior_dim = cfg['behavior_dim']
        self.hidden_dim = cfg.get('hidden_dim', self.feature_dim)
        self.beh_emb_dim = cfg.get('beh_emb_dim', 16)
        self.kernel_size = cfg.get('kernel_size', 3)

        # ConvGRU optimization options
        use_layer_norm = cfg.get('use_layer_norm', True)
        use_depthwise = cfg.get('use_depthwise', False)
        use_grouped = cfg.get('use_grouped', False)
        num_groups = cfg.get('num_groups', 1)
        learnable_h0 = cfg.get('learnable_h0', True)
        use_residual = cfg.get('use_residual', False)
        grad_clip_val = cfg.get('grad_clip_val', None)

        # Behavior embedding
        self.beh_emb = nn.Linear(self.behavior_dim, self.beh_emb_dim)

        # ConvGRU processor with optimizations
        # Input = features + embedded behavior
        self.gru = ConvGRU(
            in_ch=self.feature_dim + self.beh_emb_dim,
            hid_ch=self.hidden_dim,
            k=self.kernel_size,
            use_layer_norm=use_layer_norm,
            use_depthwise=use_depthwise,
            use_grouped=use_grouped,
            num_groups=num_groups,
            learnable_h0=learnable_h0,
            use_residual=use_residual,
            grad_clip_val=grad_clip_val
        )

        # Output projection to remove tanh clipping
        self.output_proj = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)

        # Set output dimension to hidden_dim
        self.out_dim = self.hidden_dim

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ConvGRU modulator.

        Args:
            feats: Convnet features with shape (N, C, T, H, W)
            beh: Behavior data with shape (N, behavior_dim)

        Returns:
            Modulated features with shape (N, hidden_dim, 1, H, W)
        """
        N, C, T, H, W = feats.shape

        # Validate that feature dimensions match config
        if C != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {C}")

        # Embed & broadcast behavior spatially
        beh_encoded = self.beh_emb(beh)  # (N, beh_emb_dim)
        beh_spatial = beh_encoded[:, :, None, None].expand(-1, -1, H, W)  # (N, beh_emb_dim, H, W)
        beh_spatial = beh_spatial[:, :, None].expand(-1, -1, T, -1, -1)  # (N, beh_emb_dim, T, H, W)

        # Concatenate with all frames
        gru_input = torch.cat([feats, beh_spatial], dim=1)  # (N, C+beh_emb_dim, T, H, W)

        # Process through ConvGRU
        gru_output = self.gru(gru_input)  # (N, hidden_dim, H, W)

        # Apply output projection to remove tanh clipping
        output = self.output_proj(gru_output)  # (N, hidden_dim, H, W)

        # Add singleton time dimension to match expected output format
        output = output.unsqueeze(2)  # (N, hidden_dim, 1, H, W)

        return output


class PredictiveCodingModulator(BaseModulator):
    """
    Predictive Coding modulator using ConvGRU.

    This modulator predicts the last feature frame from past frames + behavior,
    computes prediction error, and outputs either concatenated [actual, error]
    or error alone. The prediction error can be accessed during training for
    auxiliary loss computation.

    The logic is:
    1. Take features (N, C, T, H, W) and behavior (N, behavior_dim)
    2. Embed behavior spatially and concatenate with past T-1 frames
    3. Use ConvGRU to predict the final frame
    4. Compute error = actual - predicted
    5. Output concatenated [actual, error] or error alone
    """

    def _build_modulator(self):
        """Build the behavior embedding and ConvGRU predictor."""
        cfg = self.config
        self.feature_dim = cfg['feature_dim']
        self.behavior_dim = cfg['behavior_dim']
        self.hidden_dim = cfg.get('hidden_dim', self.feature_dim)
        self.beh_emb_dim = cfg.get('beh_emb_dim', 16)
        self.kernel_size = cfg.get('kernel_size', 3)
        self.concat_error = cfg.get('concat_error', True)

        # Behavior embedding
        self.beh_emb = nn.Linear(self.behavior_dim, self.beh_emb_dim)

        # ConvGRU predictor
        # Input = features + embedded behavior
        self.predictor = ConvGRU(
            in_ch=self.feature_dim + self.beh_emb_dim,
            hid_ch=self.hidden_dim,
            k=self.kernel_size
        )

        # Set output dimension
        if self.concat_error:
            self.out_dim = self.feature_dim * 2  # [actual, error]
        else:
            self.out_dim = self.feature_dim  # error only

        # Initialize prediction error storage for auxiliary loss
        self.pred_err = None

    def forward(self, feats: torch.Tensor, beh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Predictive Coding modulator.

        Args:
            feats: Convnet features with shape (N, C, T, H, W)
            beh: Behavior data with shape (N, behavior_dim)

        Returns:
            Modulated features with shape (N, C_out, 1, H, W) where:
            - C_out = 2*C if concat_error=True (actual + error)
            - C_out = C if concat_error=False (error only)
        """
        N, C, T, H, W = feats.shape

        # Validate that feature dimensions match config
        if C != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {C}")

        # Need at least 2 timesteps for prediction
        if T < 2:
            raise ValueError(f"Need at least 2 timesteps for prediction, got {T}")

        # Embed & broadcast behavior spatially
        beh_encoded = self.beh_emb(beh)  # (N, beh_emb_dim)
        beh_spatial = beh_encoded[:, :, None, None].expand(-1, -1, H, W)  # (N, beh_emb_dim, H, W)
        beh_spatial = beh_spatial[:, :, None].expand(-1, -1, T-1, -1, -1)  # (N, beh_emb_dim, T-1, H, W)

        # Concatenate with past frames (exclude last frame)
        feats_past = feats[:, :, :-1]  # (N, C, T-1, H, W)
        predictor_input = torch.cat([feats_past, beh_spatial], dim=1)  # (N, C+beh_emb_dim, T-1, H, W)

        # ConvGRU expects (N, C+beh_emb_dim, T-1, H, W) - no permutation needed
        # The ConvGRU will handle the permutation internally
        predicted = self.predictor(predictor_input)  # (N, hidden_dim, H, W)

        # Ensure matching dimensions for error computation
        if predicted.shape[1] != self.feature_dim:
            # Add a projection layer if hidden_dim != feature_dim
            if not hasattr(self, 'output_proj'):
                self.output_proj = nn.Conv2d(self.hidden_dim, self.feature_dim, 1).to(predicted.device)
            predicted = self.output_proj(predicted)

        # Actual frame at final timestep
        actual = feats[:, :, -1]  # (N, C, H, W)

        # Compute prediction error
        error = actual - predicted  # (N, C, H, W)

        # Store error for auxiliary loss (only during training)
        if self.training:
            self.pred_err = error  # Keep gradients for auxiliary loss

        # Concatenate error with actual frame or use error alone
        if self.concat_error:
            output = torch.cat([actual, error], dim=1)  # (N, 2*C, H, W)
        else:
            output = error  # (N, C, H, W)

        # Add singleton time dimension to match expected output format
        output = output.unsqueeze(2)  # (N, C_out, 1, H, W)

        return output


# Dictionary mapping modulator types to their classes
MODULATORS = {
    'concat': ConcatModulator,
    'film': FiLMModulator,
    'stn': SpatialTransformerModulator,
    'convgru': ConvGRUModulator,
    'pc': PredictiveCodingModulator,
    'polar': None  # Will be imported lazily
}


