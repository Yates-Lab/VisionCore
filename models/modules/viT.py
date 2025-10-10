# In models/modules/vivit.py

import torch
import torch.nn as nn
import math
from einops import rearrange
from .convnet import BaseConvNet
from .vit_components import UnfoldConv3d, RotaryPosEmb, TransformerBlock

class ViViT(BaseConvNet):
    def _build_network(self):
        # 1. Tokenizer
        self.tokenizer = UnfoldConv3d(
            in_channels=self.initial_channels,
            out_channels=self.config['embedding_dim'],
            **self.config.get('tokenizer', {})
        )

        # 2. Positional Embeddings
        self.pos_emb = RotaryPosEmb(
            dim=self.config['head_dim'], 
            num_tokens=max(self.config['num_spatial_tokens'], self.config['num_temporal_tokens'])
        )
        
        # 3. Transformer Blocks
        self.blocks = nn.ModuleList()
        transformer_params = self.config.get('transformer_params', {})
        for _ in range(self.config['num_spatial_blocks']):
            self.blocks.append(TransformerBlock(input_shape=(self.config['num_spatial_tokens'], self.config['embedding_dim']), **transformer_params))
        for _ in range(self.config['num_temporal_blocks']):
            self.blocks.append(TransformerBlock(input_shape=(self.config['num_temporal_tokens'], self.config['embedding_dim']), is_causal=True, **transformer_params))

        self._final_out_channels = self.config['embedding_dim']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        tokens = self.tokenizer(x)  # (B, T_p, S_p, D)
        B, T_p, S_p, D = tokens.shape

        # Spatial Self-Attention
        tokens = rearrange(tokens, 'b t s d -> (b t) s d')
        for i in range(self.config['num_spatial_blocks']):
            tokens = self.blocks[i](tokens, self.pos_emb)
        
        # Temporal Self-Attention
        tokens = rearrange(tokens, '(b t) s d -> (b s) t d', b=B, t=T_p)
        for i in range(self.config['num_spatial_blocks'], len(self.blocks)):
            tokens = self.blocks[i](tokens, self.pos_emb)
        
        # Reshape to 5D output for readout
        h_p = w_p = int(math.sqrt(S_p))
        output = rearrange(tokens, '(b s) t d -> b d t s', b=B).reshape(B, D, T_p, h_p, w_p)
        
        return self.final_activation(output)

    def get_output_channels(self) -> int:
        return self._final_out_channels