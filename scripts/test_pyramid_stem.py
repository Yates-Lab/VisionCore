#!/usr/bin/env python
"""
Test script for PyramidStem integration with ResNet

This script tests:
1. PyramidStem can be instantiated
2. PyramidStem works with 3D inputs
3. PyramidStem can be used as a stem in ResNet
4. The full model config with PyramidStem can be loaded
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.modules.conv_layers import PyramidStem
from models.config_loader import load_config
from models import build_model

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def test_pyramid_stem_standalone():
    """Test PyramidStem as a standalone module"""
    print("\n=== Testing PyramidStem Standalone ===")
    
    # Create PyramidStem
    stem = PyramidStem(
        dim=3,
        in_channels=1,
        out_channels=32,
        kernel_size=9,
        Nlevels=3,
        amplitude_cat=True,
        aa_signal=True,
        aa_freq=True
    ).to(device)
    
    print(f"PyramidStem created successfully")
    print(f"Output channels: {stem.output_channels}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 16, 51, 51).to(device)
    print(f"Input shape: {x.shape}")
    
    y = stem(x)
    print(f"Output shape: {y.shape}")
    print(f"Expected channels: {stem.output_channels}")
    
    assert y.shape[0] == batch_size, "Batch size mismatch"
    assert y.shape[1] == stem.output_channels, "Output channels mismatch"
    assert y.shape[2] == 16, "Time dimension mismatch"
    
    print("✓ PyramidStem standalone test passed!")
    return stem

def test_pyramid_stem_in_resnet():
    """Test PyramidStem as part of ResNet"""
    print("\n=== Testing PyramidStem in ResNet ===")
    
    # Create a minimal ResNet config with PyramidStem
    config = {
        'model_type': 'resnet',
        'dim': 3,
        'initial_channels': 1,
        'checkpointing': False,
        'stem_config': {
            'type': 'pyramid',
            'Nlevels': 3,
            'out_channels': 16,
            'kernel_size': 7,
            'amplitude_cat': False,
            'aa_signal': True,
            'aa_freq': True
        },
        'channels': [64, 128],
        'block_config': {
            'conv_params': {
                'type': 'standard',
                'kernel_size': [3, 3, 3],
                'padding': [1, 1, 1]
            },
            'norm_type': 'rms',
            'act_type': 'silu',
            'dropout': 0.0,
            'pool_params': None
        }
    }
    
    from models.modules.convnet import ResNet
    model = ResNet(config).to(device)
    
    print(f"ResNet with PyramidStem created successfully")
    print(f"Number of layers: {len(model.layers)}")
    print(f"First layer type: {type(model.layers[0]).__name__}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 16, 51, 51).to(device)
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    print("✓ PyramidStem in ResNet test passed!")
    return model

def test_full_config():
    """Test loading the full pyramid_stem_resnet.yaml config"""
    print("\n=== Testing Full Config ===")
    
    config_path = Path(__file__).parent.parent / "experiments/model_configs/pyramid_stem_resnet.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return None
    
    # Load config
    config = load_config(config_path)
    print(f"Config loaded successfully")
    print(f"Model type: {config.get('model_type')}")
    print(f"Convnet type: {config.get('convnet', {}).get('type')}")
    
    # Note: We can't build the full model without dataset configs,
    # but we can test the convnet part
    convnet_config = config['convnet']['params']
    convnet_config['model_type'] = config['convnet']['type']
    convnet_config['initial_channels'] = 4  # After frontend
    
    from models.modules.convnet import ResNet
    convnet = ResNet(convnet_config).to(device)
    
    print(f"Convnet created from config successfully")
    print(f"Stem type: {type(convnet.layers[0]).__name__}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 16, 51, 51).to(device)
    y = convnet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("✓ Full config test passed!")
    return convnet

if __name__ == "__main__":
    print("=" * 60)
    print("PyramidStem Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Standalone PyramidStem
        stem = test_pyramid_stem_standalone()
        
        # Test 2: PyramidStem in ResNet
        model = test_pyramid_stem_in_resnet()
        
        # Test 3: Full config
        convnet = test_full_config()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

