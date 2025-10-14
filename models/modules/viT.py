# In models/modules/vivit.py

import torch
import torch.nn as nn
import math
from einops import rearrange
from .convnet import BaseConvNet
from .vit_components import UnfoldConv3d, RotaryPosEmb, TransformerBlock

class ViViT(BaseConvNet):
    """
    Video Vision Transformer (ViViT) for neural encoding.

    Based on Arnab et al. 2021 with modifications for neural data:
    - Tokenizes spatiotemporal video patches
    - Separate spatial and temporal transformers
    - Causal temporal attention (no future information)
    - Optional behavior integration via channel concatenation
    - Patch dropout for regularization
    - Optional register tokens for better attention interpretability

    Input: (B, C, T, H, W) where C includes visual + behavior channels
    Output: (B, D, T_p, H_p, W_p) compatible with DynamicGaussianReadout
    """

    def _build_network(self):
        """Build ViViT architecture components."""
        # Get configuration parameters
        self.embedding_dim = self.config['embedding_dim']
        self.num_spatial_blocks = self.config['num_spatial_blocks']
        self.num_temporal_blocks = self.config['num_temporal_blocks']
        self.head_dim = self.config['head_dim']
        self.patch_dropout = self.config.get('patch_dropout', 0.0)
        self.use_register_tokens = self.config.get('use_register_tokens', False)
        self.num_register_tokens = self.config.get('num_register_tokens', 4)

        # 1. Tokenizer - extracts spatiotemporal patches
        tokenizer_config = self.config.get('tokenizer', {})
        self.tokenizer = UnfoldConv3d(
            in_channels=self.initial_channels,
            out_channels=self.embedding_dim,
            **tokenizer_config
        )

        # Store tokenizer params for computing token counts
        self.kernel_size = tokenizer_config.get('kernel_size', [4, 4, 4])
        self.stride = tokenizer_config.get('stride', [4, 4, 4])

        # 2. Patch dropout (applied after tokenization)
        if self.patch_dropout > 0:
            self.patch_dropout_layer = nn.Dropout(self.patch_dropout)

        # 3. Register tokens (optional, for better attention interpretability)
        if self.use_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.randn(1, self.num_register_tokens, self.embedding_dim) * 0.02
            )

        # 4. Positional Embeddings (RoPE)
        # We'll compute max tokens dynamically, but initialize with reasonable size
        max_tokens = self.config.get('max_tokens', 1000)
        self.pos_emb = RotaryPosEmb(
            dim=self.head_dim,
            num_tokens=max_tokens
        )

        # 5. Transformer Blocks
        self.blocks = nn.ModuleList()
        transformer_params = self.config.get('transformer_params', {})

        # Spatial transformer blocks
        for _ in range(self.num_spatial_blocks):
            # Note: input_shape is used for initialization, actual shape is dynamic
            self.blocks.append(
                TransformerBlock(
                    input_shape=(100, self.embedding_dim),  # Placeholder
                    **transformer_params
                )
            )

        # Temporal transformer blocks (with causal masking)
        temporal_params = transformer_params.copy()
        temporal_params['is_causal'] = True  # Override for temporal blocks
        for _ in range(self.num_temporal_blocks):
            self.blocks.append(
                TransformerBlock(
                    input_shape=(100, self.embedding_dim),  # Placeholder
                    **temporal_params
                )
            )

        self._final_out_channels = self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViViT.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
               C can include visual channels + behavior channels

        Returns:
            Output tensor of shape (B, D, T_p, H_p, W_p)
            where D is embedding_dim, T_p is temporal tokens,
            H_p x W_p is spatial tokens reshaped to 2D
        """
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # 1. Tokenization - extract spatiotemporal patches
        tokens = self.tokenizer(x)  # (B, T_p, S_p, D)
        B, T_p, S_p, D = tokens.shape

        # 2. Patch dropout (during training only)
        if self.training and self.patch_dropout > 0:
            tokens = self.patch_dropout_layer(tokens)

        # 3. Spatial Self-Attention
        # Combine batch and temporal dimensions to apply spatial attention per frame
        tokens = rearrange(tokens, 'b t s d -> (b t) s d')

        # Add register tokens if enabled (per temporal position)
        if self.use_register_tokens:
            BT = tokens.shape[0]
            register_tokens = self.register_tokens.expand(BT, -1, -1)
            tokens = torch.cat([register_tokens, tokens], dim=1)
            S_p_with_reg = S_p + self.num_register_tokens
        else:
            S_p_with_reg = S_p

        # Apply spatial transformer blocks
        for i in range(self.num_spatial_blocks):
            tokens = self.blocks[i](tokens, self.pos_emb)

        # Remove register tokens before temporal processing
        if self.use_register_tokens:
            tokens = tokens[:, self.num_register_tokens:]  # Remove register tokens

        # 4. Temporal Self-Attention (with causal masking)
        # Combine batch and spatial dimensions to apply temporal attention per location
        tokens = rearrange(tokens, '(b t) s d -> (b s) t d', b=B, t=T_p)

        # Apply temporal transformer blocks
        for i in range(self.num_spatial_blocks, len(self.blocks)):
            tokens = self.blocks[i](tokens, self.pos_emb)

        # 5. Reshape to 5D output for readout compatibility
        # DynamicGaussianReadout expects (B, C, T, H, W) and uses last time step
        h_p = w_p = int(math.sqrt(S_p))
        output = rearrange(tokens, '(b s) t d -> b d t s', b=B).reshape(B, D, T_p, h_p, w_p)

        return self.final_activation(output)

    def get_output_channels(self) -> int:
        """Return number of output channels (embedding dimension)."""
        return self._final_out_channels