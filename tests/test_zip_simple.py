"""
Simple test script for Zero-Inflated Poisson loss.
Run this to verify the ZIP loss implementation works correctly.
"""

import torch
import sys

# Add current directory to path
sys.path.insert(0, '.')

from models.losses import ZeroInflatedPoissonNLLLoss, MaskedZIPNLLLoss


def test_basic_zip_loss():
    """Test basic ZIP loss computation."""
    print("\n" + "="*60)
    print("Test 1: Basic ZIP Loss")
    print("="*60)
    
    loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='mean')
    
    # Create sample data
    lam = torch.tensor([[1.0, 2.0, 3.0]])
    pi = torch.tensor([[0.1, 0.2, 0.3]])
    target = torch.tensor([[0.0, 1.0, 2.0]])
    
    # Compute loss
    loss = loss_fn((lam, pi), target)
    
    print(f"Lambda: {lam}")
    print(f"Pi: {pi}")
    print(f"Target: {target}")
    print(f"Loss: {loss.item():.4f}")
    
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Test passed!")
    return True


def test_zero_inflation_effect():
    """Test that zero-inflation probability affects loss."""
    print("\n" + "="*60)
    print("Test 2: Zero-Inflation Effect")
    print("="*60)
    
    loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='none')
    
    lam = torch.tensor([[2.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])  # Both zeros
    
    # Low vs high zero-inflation probability
    pi_low = torch.tensor([[0.1, 0.1]])
    pi_high = torch.tensor([[0.5, 0.5]])
    
    loss_low = loss_fn((lam, pi_low), target)
    loss_high = loss_fn((lam, pi_high), target)
    
    print(f"Lambda: {lam}")
    print(f"Target (zeros): {target}")
    print(f"Loss with pi=0.1: {loss_low.mean().item():.4f}")
    print(f"Loss with pi=0.5: {loss_high.mean().item():.4f}")
    
    assert (loss_high < loss_low).all(), "Higher pi should give lower loss for zeros"
    print("✓ Test passed! Higher pi gives lower loss for zero counts.")
    return True


def test_masked_zip_loss():
    """Test masked ZIP loss with batch dictionary."""
    print("\n" + "="*60)
    print("Test 3: Masked ZIP Loss")
    print("="*60)
    
    loss_fn = MaskedZIPNLLLoss(log_input=False)
    
    batch = {
        'rhat': torch.rand(4, 8) * 5,
        'pi': torch.rand(4, 8) * 0.3,
        'robs': torch.poisson(torch.rand(4, 8) * 3),
        'dfs': torch.ones(4, 8)
    }
    
    loss = loss_fn(batch)
    
    print(f"Batch shape: {batch['rhat'].shape}")
    print(f"Lambda range: [{batch['rhat'].min():.2f}, {batch['rhat'].max():.2f}]")
    print(f"Pi range: [{batch['pi'].min():.2f}, {batch['pi'].max():.2f}]")
    print(f"Observed counts range: [{batch['robs'].min():.0f}, {batch['robs'].max():.0f}]")
    print(f"Loss: {loss.item():.4f}")
    
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Test passed!")
    return True


def test_masking_effect():
    """Test that masking affects the loss."""
    print("\n" + "="*60)
    print("Test 4: Masking Effect")
    print("="*60)
    
    loss_fn = MaskedZIPNLLLoss(log_input=False)
    
    # Create batch
    batch_full = {
        'rhat': torch.rand(4, 8) * 5,
        'pi': torch.rand(4, 8) * 0.3,
        'robs': torch.poisson(torch.rand(4, 8) * 3),
        'dfs': torch.ones(4, 8)
    }
    
    batch_masked = {
        'rhat': batch_full['rhat'].clone(),
        'pi': batch_full['pi'].clone(),
        'robs': batch_full['robs'].clone(),
        'dfs': torch.ones(4, 8)
    }
    # Mask out half the data
    batch_masked['dfs'][:, 4:] = 0
    
    loss_full = loss_fn(batch_full)
    loss_masked = loss_fn(batch_masked)
    
    print(f"Loss with full data: {loss_full.item():.4f}")
    print(f"Loss with 50% masked: {loss_masked.item():.4f}")
    print(f"Difference: {abs(loss_full.item() - loss_masked.item()):.4f}")
    
    assert not torch.isclose(loss_full, loss_masked, rtol=1e-3), "Masking should affect loss"
    print("✓ Test passed! Masking changes the loss.")
    return True


def test_fallback_to_poisson():
    """Test fallback to Poisson when pi is not provided."""
    print("\n" + "="*60)
    print("Test 5: Fallback to Poisson (no pi)")
    print("="*60)
    
    loss_fn = MaskedZIPNLLLoss(log_input=False)
    
    batch = {
        'rhat': torch.rand(4, 8) * 5,
        'robs': torch.poisson(torch.rand(4, 8) * 3),
        'dfs': torch.ones(4, 8)
    }
    
    loss = loss_fn(batch)
    
    print(f"Batch without 'pi' key")
    print(f"Loss (should use Poisson): {loss.item():.4f}")
    
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Test passed! Fallback to Poisson works.")
    return True


def test_log_input():
    """Test with log-space lambda input."""
    print("\n" + "="*60)
    print("Test 6: Log-space Input")
    print("="*60)
    
    loss_fn = MaskedZIPNLLLoss(log_input=True)
    
    # Create batch with log-space lambda
    batch = {
        'rhat': torch.log(torch.rand(4, 8) * 5 + 0.1),  # log(lambda)
        'pi': torch.rand(4, 8) * 0.3,
        'robs': torch.poisson(torch.rand(4, 8) * 3),
        'dfs': torch.ones(4, 8)
    }
    
    loss = loss_fn(batch)
    
    print(f"Log-lambda range: [{batch['rhat'].min():.2f}, {batch['rhat'].max():.2f}]")
    print(f"Loss: {loss.item():.4f}")
    
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Test passed!")
    return True


def test_comparison_with_poisson():
    """Compare ZIP (pi=0) with standard Poisson loss."""
    print("\n" + "="*60)
    print("Test 7: ZIP vs Poisson Equivalence (pi=0)")
    print("="*60)
    
    from models.losses import MaskedLoss
    import torch.nn as nn
    
    # Create both loss functions
    zip_loss_fn = MaskedZIPNLLLoss(log_input=False)
    poisson_loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=False, reduction='none'))
    
    # Create batch with pi=0 (no zero-inflation)
    batch_zip = {
        'rhat': torch.rand(4, 8) * 5,
        'pi': torch.zeros(4, 8),  # No zero-inflation
        'robs': torch.poisson(torch.rand(4, 8) * 3),
        'dfs': torch.ones(4, 8)
    }
    
    batch_poisson = {
        'rhat': batch_zip['rhat'].clone(),
        'robs': batch_zip['robs'].clone(),
        'dfs': batch_zip['dfs'].clone()
    }
    
    loss_zip = zip_loss_fn(batch_zip)
    loss_poisson = poisson_loss_fn(batch_poisson)
    
    print(f"ZIP loss (pi=0): {loss_zip.item():.4f}")
    print(f"Poisson loss: {loss_poisson.item():.4f}")
    print(f"Difference: {abs(loss_zip.item() - loss_poisson.item()):.6f}")
    
    assert torch.isclose(loss_zip, loss_poisson, rtol=1e-3, atol=1e-5), \
        "ZIP with pi=0 should equal Poisson"
    print("✓ Test passed! ZIP reduces to Poisson when pi=0.")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ZERO-INFLATED POISSON LOSS TESTS")
    print("="*60)
    
    tests = [
        test_basic_zip_loss,
        test_zero_inflation_effect,
        test_masked_zip_loss,
        test_masking_effect,
        test_fallback_to_poisson,
        test_log_input,
        test_comparison_with_poisson,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

