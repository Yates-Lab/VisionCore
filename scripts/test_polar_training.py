#!/usr/bin/env python
"""
Test Polar-V1 training on dummy data.

This script tests that the Polar-V1 model can be built and trained
on synthetic data without errors.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_polar_training():
    """Test Polar-V1 training on dummy data."""
    try:
        from models.build import build_model
        from models.config_loader import load_config
        
        # Load config
        config_path = Path("experiments/model_configs/polar_v1_behavior_only.yaml")
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return False
            
        config = load_config(config_path)
        
        # Mock dataset configs
        dataset_configs = [{
            'n_neurons': 8,
            'session': 'test_session',
            'lab': 'yates'
        }]
        
        # Build model
        print("Building Polar-V1 model...")
        model = build_model(config, dataset_configs)
        print(f"Model built successfully: {type(model).__name__}")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")
        
        # Create dummy data
        batch_size = 4
        time_steps = 15
        height, width = 51, 51
        
        batch = {
            'stim': torch.randn(batch_size, 1, time_steps, height, width).to(device),
            'behavior': torch.randn(batch_size, time_steps, 2).to(device),
            'robs': torch.randint(0, 10, (batch_size, 8)).float().to(device),
            'dfs': torch.ones(batch_size, 8).to(device)
        }
        
        print(f"Created dummy batch:")
        print(f"  stim: {batch['stim'].shape}")
        print(f"  behavior: {batch['behavior'].shape}")
        print(f"  robs: {batch['robs'].shape}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        print("\nStarting training loop...")
        
        # Training loop
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(
                stimulus=batch['stim'],
                dataset_idx=0,
                behavior=batch['behavior']
            )
            
            print(f"Step {step}: output shape = {output.shape}")
            
            # Poisson NLL loss
            loss = F.poisson_nll_loss(
                output, 
                batch['robs'], 
                log_input=False, 
                full=False, 
                reduction='mean'
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            print(f"Step {step}: loss = {loss.item():.4f}")
            
            # Check for NaNs
            if torch.isnan(loss):
                print("ERROR: NaN loss detected!")
                return False
        
        print(f"\n✅ Training test passed!")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
        
        # Check if loss is decreasing (not required, but good sign)
        if losses[-1] < losses[0]:
            print("✅ Loss is decreasing - good sign!")
        else:
            print("⚠️  Loss not decreasing - may need tuning")
        
        return True
        
    except ImportError as e:
        print(f"Import error (likely missing plenoptic): {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_polar():
    """Test minimal Polar-V1 (no behavior) training."""
    try:
        from models.build import build_model
        from models.config_loader import load_config
        
        # Load minimal config
        config_path = Path("experiments/model_configs/polar_v1_minimal.yaml")
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return False
            
        config = load_config(config_path)
        
        # Mock dataset configs
        dataset_configs = [{
            'n_neurons': 8,
            'session': 'test_session',
            'lab': 'yates'
        }]
        
        # Build model
        print("Building minimal Polar-V1 model...")
        model = build_model(config, dataset_configs)
        
        # Create dummy data (no behavior)
        batch = {
            'stim': torch.randn(2, 1, 10, 51, 51),
            'robs': torch.randint(0, 5, (2, 8)).float()
        }
        
        # Forward pass (no behavior)
        output = model(
            stimulus=batch['stim'],
            dataset_idx=0,
            behavior=None
        )
        
        print(f"Minimal model output shape: {output.shape}")
        assert output.shape == (2, 8)
        
        print("✅ Minimal Polar-V1 test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR in minimal test: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Polar-V1 Training")
    print("=" * 60)
    
    # Test full model
    success1 = test_polar_training()
    
    print("\n" + "=" * 60)
    print("Testing Minimal Polar-V1")
    print("=" * 60)
    
    # Test minimal model
    success2 = test_minimal_polar()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
