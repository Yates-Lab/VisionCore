#!/usr/bin/env python3
"""
Test script to verify the optimized ResNet configuration builds correctly.
Tests:
1. Config loading
2. Model building
3. Forward pass
4. Initialization checks (RMSNorm affine, zero-init, etc.)
"""

import torch
import yaml
from pathlib import Path
from models.build import build_model, initialize_model_components
from models.modules.norm_act_pool import RMSNorm
from models.modules.conv_blocks import ResBlock

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dummy_dataset_configs(num_datasets=2):
    """Create dummy dataset configs for testing."""
    dataset_configs = []
    for i in range(num_datasets):
        dataset_configs.append({
            'dataset_idx': i,
            'n_neurons': 50 + i * 10,  # Different number of neurons per dataset
            'behavior_dim': 42,
        })
    return dataset_configs

def test_model_build(config_path):
    """Test that the model builds successfully."""
    print(f"\n{'='*60}")
    print(f"Testing model build from: {config_path}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(config_path)
    print(f"✓ Config loaded successfully")
    print(f"  Model type: {config['model_type']}")
    print(f"  Channels: {config['convnet']['params']['channels']}")
    
    # Create dummy dataset configs
    dataset_configs = create_dummy_dataset_configs(num_datasets=2)
    print(f"✓ Created {len(dataset_configs)} dummy dataset configs")
    
    # Build model
    try:
        model = build_model(config, dataset_configs)
        print(f"✓ Model built successfully")
        print(f"  Model class: {model.__class__.__name__}")
    except Exception as e:
        print(f"✗ Model build failed: {e}")
        raise
    
    # Initialize model
    try:
        initialize_model_components(model, config)
        print(f"✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise
    
    return model, config

def check_rmsnorm_affine(model):
    """Check that RMSNorm layers have affine parameters."""
    print(f"\n{'='*60}")
    print("Checking RMSNorm affine parameters...")
    print(f"{'='*60}\n")
    
    rmsnorm_count = 0
    affine_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            rmsnorm_count += 1
            if module.affine:
                affine_count += 1
                print(f"✓ {name}: affine=True, num_features={module.num_features}")
            else:
                print(f"✗ {name}: affine=False (should be True!)")
    
    print(f"\nSummary: {affine_count}/{rmsnorm_count} RMSNorm layers have affine=True")
    return affine_count == rmsnorm_count

def check_resblock_zero_init(model):
    """Check that ResBlock final norms are zero-initialized."""
    print(f"\n{'='*60}")
    print("Checking ResBlock zero-initialization...")
    print(f"{'='*60}\n")
    
    resblock_count = 0
    zero_init_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, ResBlock):
            resblock_count += 1
            if hasattr(module.main_block, 'components') and 'norm' in module.main_block.components:
                norm_layer = module.main_block.components['norm']
                
                # Check if gamma/weight is zero-initialized
                is_zero = False
                if isinstance(norm_layer, RMSNorm) and norm_layer.affine:
                    is_zero = torch.allclose(norm_layer.gamma, torch.zeros_like(norm_layer.gamma))
                    param_name = 'gamma'
                elif hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
                    is_zero = torch.allclose(norm_layer.weight, torch.zeros_like(norm_layer.weight))
                    param_name = 'weight'
                else:
                    param_name = 'N/A'
                
                if is_zero:
                    zero_init_count += 1
                    print(f"✓ {name}: {param_name} is zero-initialized")
                else:
                    print(f"✗ {name}: {param_name} is NOT zero-initialized")
    
    print(f"\nSummary: {zero_init_count}/{resblock_count} ResBlocks have zero-initialized norms")
    return zero_init_count == resblock_count

def check_depthwise_conv(model):
    """Check that depthwise convolutions are being used."""
    print(f"\n{'='*60}")
    print("Checking for depthwise convolutions...")
    print(f"{'='*60}\n")
    
    from models.modules.conv_layers import DepthwiseConv
    
    depthwise_count = 0
    for name, module in model.named_modules():
        if isinstance(module, DepthwiseConv):
            depthwise_count += 1
            print(f"✓ Found DepthwiseConv at: {name}")
    
    print(f"\nSummary: Found {depthwise_count} DepthwiseConv layers")
    return depthwise_count > 0

def test_forward_pass(model, config):
    """Test a forward pass through the model."""
    print(f"\n{'='*60}")
    print("Testing forward pass...")
    print(f"{'='*60}\n")
    
    # Create dummy input
    batch_size = 2
    time_steps = 16  # Match frontend kernel_size
    height, width = 51, 51  # Match adapter grid_size
    
    # Input stimulus
    stimulus = torch.randn(batch_size, 1, time_steps, height, width)
    
    # Behavior (for modulator)
    behavior_dim = config['modulator']['params']['behavior_dim']
    behavior = torch.randn(batch_size, time_steps, behavior_dim)
    
    # Dataset indices
    dataset_idx = torch.tensor([0, 1])  # Two different datasets
    
    print(f"Input shapes:")
    print(f"  stimulus: {stimulus.shape}")
    print(f"  behavior: {behavior.shape}")
    print(f"  dataset_idx: {dataset_idx.shape}")
    
    try:
        with torch.no_grad():
            output = model(stimulus, behavior, dataset_idx)
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        return True
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    config_path = Path("experiments/model_configs/learned_res_optimized_gru.yaml")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    
    try:
        # Build and initialize model
        model, config = test_model_build(config_path)
        
        # Run checks
        rmsnorm_ok = check_rmsnorm_affine(model)
        zero_init_ok = check_resblock_zero_init(model)
        depthwise_ok = check_depthwise_conv(model)
        forward_ok = test_forward_pass(model, config)
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}\n")
        print(f"  RMSNorm affine parameters: {'✓ PASS' if rmsnorm_ok else '✗ FAIL'}")
        print(f"  ResBlock zero-init: {'✓ PASS' if zero_init_ok else '✗ FAIL'}")
        print(f"  Depthwise convolutions: {'✓ PASS' if depthwise_ok else '✗ FAIL'}")
        print(f"  Forward pass: {'✓ PASS' if forward_ok else '✗ FAIL'}")
        
        all_pass = rmsnorm_ok and zero_init_ok and depthwise_ok and forward_ok
        print(f"\n{'='*60}")
        if all_pass:
            print("ALL TESTS PASSED! ✓")
        else:
            print("SOME TESTS FAILED! ✗")
        print(f"{'='*60}\n")
        
        return all_pass
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

