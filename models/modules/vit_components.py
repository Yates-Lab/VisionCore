# In models/modules/vit_components.py

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.ops import stochastic_depth
import numpy as np

# --- Helper Classes and Functions ---

class SwiGLU(nn.Module):
    def forward(self, inputs: torch.Tensor):
        outputs, gate = inputs.chunk(2, dim=-1)
        return F.silu(gate) * outputs

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0, mode: str = "row"):
        super().__init__()
        self.p, self.mode = p, mode

    def forward(self, inputs: torch.Tensor):
        return stochastic_depth(inputs, p=self.p, mode=self.mode, training=self.training)

class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) from Zhu et al. 2025
    Reference:
    - https://arxiv.org/abs/2503.10622
    - https://jiachenzhu.github.io/DyT/
    """

    def __init__(self, num_features: int, alpha_init_value: float = 0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.tanh(self.alpha * inputs)
        return outputs * self.weight + self.bias


def get_norm(name: str):
    match name.lower():
        case "layernorm":
            return nn.LayerNorm
        case "rmsnorm":
            return nn.RMSNorm
        case "dyt":
            return DyT
        case _:
            raise NotImplementedError(f"Norm {name} not implemented.")

def get_ff_activation(name: str, ff_dim: int) -> (nn.Module, int):
    activations = {
        "relu": (nn.ReLU, ff_dim),
        "gelu": (nn.GELU, ff_dim),
        "silu": (nn.SiLU, ff_dim),
        "swiglu": (SwiGLU, ff_dim * 2),
    }
    if name not in activations:
        raise NotImplementedError(f"FF {name} not implemented.")
    return activations[name]


# --- Core ViViT Components ---

class UnfoldConv3d(nn.Module):
    """The 3D Convolutional Tokenizer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm, **kwargs):
        super().__init__()
        patch_dim = int(in_channels * np.prod(kernel_size))
        
        # This uses a standard conv3d to create patches, then projects them.
        self.patcher = nn.Sequential(
            nn.Conv3d(in_channels, patch_dim, kernel_size=kernel_size, stride=stride, **kwargs),
            Rearrange("b c t h w -> b t (h w) c"),
            get_norm(norm)(patch_dim),
            nn.Linear(patch_dim, out_channels),
            get_norm(norm)(out_channels)
        )

    def forward(self, inputs: torch.Tensor):
        return self.patcher(inputs)

class RotaryPosEmb(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, num_tokens: int, **kwargs):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.create_embedding(n=num_tokens)

    def create_embedding(self, n: int):
        t = torch.arange(n, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("emb_sin", emb.sin(), persistent=False)
        self.register_buffer("emb_cos", emb.cos(), persistent=False)
    
    def get_embedding(self, n: int):
        if not hasattr(self, "emb_sin") or self.emb_sin.shape[-2] < n:
            self.create_embedding(n)
        return self.emb_sin[:n], self.emb_cos[:n]

    @staticmethod
    def rotate_half(x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        n = q.shape[-2]
        sin, cos = self.get_embedding(n)
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

class TransformerBlock(nn.Module):
    """The main Transformer Block"""
    def __init__(self, input_shape, num_heads, head_dim, ff_dim, ff_activation, 
                 mha_dropout, ff_dropout, drop_path, use_rope, flash_attention, 
                 is_causal, norm, normalize_qk, grad_checkpointing, **kwargs):
        super().__init__()
        emb_dim = input_shape[-1]
        inner_dim = head_dim * num_heads

        self.norm1 = get_norm(norm)(emb_dim)
        self.to_qkv = nn.Linear(emb_dim, inner_dim * 3, bias=False)
        self.mha_dropout = nn.Dropout(mha_dropout)
        self.to_out = nn.Linear(inner_dim, emb_dim)
        
        ff_act_fn, ff_out_dim = get_ff_activation(ff_activation, ff_dim)
        self.ff = nn.Sequential(
            get_norm(norm)(emb_dim),
            nn.Linear(emb_dim, ff_out_dim),
            ff_act_fn(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(ff_dropout),
        )
        
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.flash_attention = flash_attention
        self.is_causal = is_causal
        self.grad_checkpointing = grad_checkpointing
        self.drop_path = DropPath(drop_path)
        self.register_buffer("scale", torch.tensor(head_dim**-0.5))

    def attention(self, inputs: torch.Tensor, rotary_emb: RotaryPosEmb):
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        if self.use_rope:
            q, k = rotary_emb(q, k)

        # Use flash attention if available
        if self.flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        else:
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            if self.is_causal:
                mask = torch.ones(sim.shape[-2:], device=sim.device).triu(1).bool()
                sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(self, inputs: torch.Tensor, rotary_emb: RotaryPosEmb):
        if self.grad_checkpointing and self.training:
             # Checkpointing with re-entrant is needed for some PyTorch versions
            attn_out = torch.utils.checkpoint.checkpoint(self.attention, self.norm1(inputs), rotary_emb, use_reentrant=False)
        else:
            attn_out = self.attention(self.norm1(inputs), rotary_emb)
            
        x = inputs + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(x))
        return x