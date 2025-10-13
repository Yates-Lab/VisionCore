"""
Unit tests for ViViT implementation.

Tests cover:
- Forward pass with various input shapes
- Tokenization
- Spatial and temporal transformers
- Register tokens
- Patch dropout
- Integration with existing readout modules
"""

import torch
import pytest
import yaml
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.modules.viT import ViViT
from models.modules.readout import DynamicGaussianReadout
from models.modules.vit_components import UnfoldConv3d, RotaryPosEmb, TransformerBlock


class TestViViTComponents:
    """Test individual ViViT components."""
    
    def test_tokenizer(self):
        """Test 3D convolutional tokenizer."""
        tokenizer = UnfoldConv3d(
            in_channels=1,
            out_channels=192,
            kernel_size=[4, 4, 4],
            stride=[4, 4, 4],
            norm='rmsnorm'
        )
        
        # Test input: (B, C, T, H, W)
        x = torch.randn(2, 1, 32, 36, 64)
        tokens = tokenizer(x)
        
        # Expected output: (B, T_p, S_p, D)
        # T_p = 32/4 = 8, S_p = (36/4) * (64/4) = 9 * 16 = 144
        assert tokens.shape == (2, 8, 144, 192)
        print(f"✓ Tokenizer test passed: {x.shape} -> {tokens.shape}")
    
    def test_rotary_embeddings(self):
        """Test rotary position embeddings."""
        pos_emb = RotaryPosEmb(dim=64, num_tokens=100)
        
        # Test Q, K tensors
        q = torch.randn(2, 4, 50, 64)  # (B, H, N, D)
        k = torch.randn(2, 4, 50, 64)
        
        q_rot, k_rot = pos_emb(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        print(f"✓ RoPE test passed: {q.shape} -> {q_rot.shape}")
    
    def test_transformer_block(self):
        """Test transformer block."""
        block = TransformerBlock(
            input_shape=(100, 192),
            num_heads=4,
            head_dim=48,
            ff_dim=768,
            ff_activation='swiglu',
            mha_dropout=0.1,
            ff_dropout=0.1,
            drop_path=0.1,
            use_rope=True,
            flash_attention=False,  # Use manual attention for testing
            is_causal=False,
            norm='rmsnorm',
            normalize_qk=False,
            grad_checkpointing=False
        )
        
        pos_emb = RotaryPosEmb(dim=48, num_tokens=100)
        x = torch.randn(2, 100, 192)
        
        out = block(x, pos_emb)
        assert out.shape == x.shape
        print(f"✓ Transformer block test passed: {x.shape} -> {out.shape}")


class TestViViT:
    """Test full ViViT model."""
    
    def get_small_config(self):
        """Get a small config for testing."""
        return {
            'dim': 3,
            'initial_channels': 1,
            'base_channels': 1,
            'embedding_dim': 192,
            'num_spatial_blocks': 2,
            'num_temporal_blocks': 2,
            'head_dim': 48,
            'patch_dropout': 0.1,
            'use_register_tokens': False,
            'num_register_tokens': 0,
            'tokenizer': {
                'kernel_size': [4, 4, 4],
                'stride': [4, 4, 4],
                'norm': 'rmsnorm',
                'padding': 0
            },
            'transformer_params': {
                'num_heads': 4,
                'head_dim': 48,
                'ff_dim': 768,
                'ff_activation': 'swiglu',
                'mha_dropout': 0.1,
                'ff_dropout': 0.1,
                'drop_path': 0.1,
                'use_rope': True,
                'flash_attention': False,
                'is_causal': False,
                'norm': 'rmsnorm',
                'normalize_qk': False,
                'grad_checkpointing': False
            },
            'final_activation': 'none'
        }
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        config = self.get_small_config()
        model = ViViT(config)
        
        # Input: (B, C, T, H, W)
        x = torch.randn(2, 1, 32, 36, 64)
        
        # Forward pass
        out = model(x)
        
        # Expected: (B, D, T_p, H_p, W_p)
        # T_p = 32/4 = 8, H_p = 36/4 = 9, W_p = 64/4 = 16
        assert out.shape == (2, 192, 8, 9, 16)
        print(f"✓ Basic forward test passed: {x.shape} -> {out.shape}")
    
    def test_forward_with_register_tokens(self):
        """Test forward pass with register tokens."""
        config = self.get_small_config()
        config['use_register_tokens'] = True
        config['num_register_tokens'] = 4
        
        model = ViViT(config)
        x = torch.randn(2, 1, 32, 36, 64)
        
        out = model(x)
        assert out.shape == (2, 192, 8, 9, 16)
        print(f"✓ Register tokens test passed: {x.shape} -> {out.shape}")
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = self.get_small_config()
        model = ViViT(config)
        
        x = torch.randn(2, 1, 32, 36, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        print(f"✓ Gradient flow test passed")
    
    def test_output_channels(self):
        """Test get_output_channels method."""
        config = self.get_small_config()
        model = ViViT(config)
        
        assert model.get_output_channels() == 192
        print(f"✓ Output channels test passed: {model.get_output_channels()}")


class TestViViTWithReadout:
    """Test ViViT integration with existing readout modules."""
    
    def test_with_gaussian_readout(self):
        """Test ViViT with DynamicGaussianReadout."""
        # Create ViViT
        config = {
            'dim': 3,
            'initial_channels': 1,
            'base_channels': 1,
            'embedding_dim': 192,
            'num_spatial_blocks': 2,
            'num_temporal_blocks': 2,
            'head_dim': 48,
            'patch_dropout': 0.0,
            'use_register_tokens': False,
            'tokenizer': {
                'kernel_size': [4, 4, 4],
                'stride': [4, 4, 4],
                'norm': 'rmsnorm',
                'padding': 0
            },
            'transformer_params': {
                'num_heads': 4,
                'head_dim': 48,
                'ff_dim': 768,
                'ff_activation': 'swiglu',
                'mha_dropout': 0.0,
                'ff_dropout': 0.0,
                'drop_path': 0.0,
                'use_rope': True,
                'flash_attention': False,
                'is_causal': False,
                'norm': 'rmsnorm',
                'normalize_qk': False,
                'grad_checkpointing': False
            },
            'final_activation': 'none'
        }
        
        vivit = ViViT(config)
        
        # Create readout
        readout = DynamicGaussianReadout(
            in_channels=192,
            n_units=100,
            bias=True,
            initial_std=5.0
        )
        
        # Forward pass
        x = torch.randn(2, 1, 32, 36, 64)
        features = vivit(x)  # (2, 192, 8, 9, 16)
        
        # Readout should use last time step automatically
        output = readout(features)  # (2, 100)
        
        assert output.shape == (2, 100)
        print(f"✓ ViViT + Gaussian readout test passed: {x.shape} -> {features.shape} -> {output.shape}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running ViViT Tests")
    print("="*60 + "\n")
    
    # Component tests
    print("Testing Components:")
    print("-" * 60)
    comp_tests = TestViViTComponents()
    comp_tests.test_tokenizer()
    comp_tests.test_rotary_embeddings()
    comp_tests.test_transformer_block()
    
    # ViViT tests
    print("\nTesting ViViT Model:")
    print("-" * 60)
    vivit_tests = TestViViT()
    vivit_tests.test_forward_basic()
    vivit_tests.test_forward_with_register_tokens()
    vivit_tests.test_gradient_flow()
    vivit_tests.test_output_channels()
    
    # Integration tests
    print("\nTesting Integration:")
    print("-" * 60)
    integration_tests = TestViViTWithReadout()
    integration_tests.test_with_gaussian_readout()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

