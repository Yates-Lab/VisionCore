"""
Tests for Zero-Inflated Poisson (ZIP) loss implementation.
"""

import torch
import pytest
from models.losses import ZeroInflatedPoissonNLLLoss, MaskedZIPNLLLoss


class TestZeroInflatedPoissonNLLLoss:
    """Test the core ZIP loss function."""
    
    def test_basic_forward(self):
        """Test basic forward pass with tuple input."""
        loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='mean')
        
        # Create sample data
        lam = torch.tensor([[1.0, 2.0, 3.0]])
        pi = torch.tensor([[0.1, 0.2, 0.3]])
        target = torch.tensor([[0.0, 1.0, 2.0]])
        
        # Compute loss
        loss = loss_fn((lam, pi), target)
        
        # Check output
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # positive loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_log_input(self):
        """Test with log-space lambda input."""
        loss_fn = ZeroInflatedPoissonNLLLoss(log_input=True, reduction='mean')
        
        # Create sample data (log-space)
        log_lam = torch.log(torch.tensor([[1.0, 2.0, 3.0]]))
        pi = torch.tensor([[0.1, 0.2, 0.3]])
        target = torch.tensor([[0.0, 1.0, 2.0]])
        
        # Compute loss
        loss = loss_fn((log_lam, pi), target)
        
        # Check output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_zero_inflation(self):
        """Test that zero-inflation affects loss for zero counts."""
        loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='none')
        
        lam = torch.tensor([[2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0]])
        
        # Low vs high zero-inflation probability
        pi_low = torch.tensor([[0.1, 0.1]])
        pi_high = torch.tensor([[0.5, 0.5]])
        
        loss_low = loss_fn((lam, pi_low), target)
        loss_high = loss_fn((lam, pi_high), target)
        
        # Higher pi should give lower loss for zero counts
        assert (loss_high < loss_low).all()
    
    def test_reduction_modes(self):
        """Test different reduction modes."""
        lam = torch.tensor([[1.0, 2.0, 3.0]])
        pi = torch.tensor([[0.1, 0.2, 0.3]])
        target = torch.tensor([[0.0, 1.0, 2.0]])
        
        # Test 'none'
        loss_fn_none = ZeroInflatedPoissonNLLLoss(reduction='none')
        loss_none = loss_fn_none((lam, pi), target)
        assert loss_none.shape == target.shape
        
        # Test 'mean'
        loss_fn_mean = ZeroInflatedPoissonNLLLoss(reduction='mean')
        loss_mean = loss_fn_mean((lam, pi), target)
        assert loss_mean.ndim == 0
        assert torch.isclose(loss_mean, loss_none.mean())
        
        # Test 'sum'
        loss_fn_sum = ZeroInflatedPoissonNLLLoss(reduction='sum')
        loss_sum = loss_fn_sum((lam, pi), target)
        assert loss_sum.ndim == 0
        assert torch.isclose(loss_sum, loss_none.sum())
    
    def test_fallback_to_poisson(self):
        """Test fallback to Poisson when pi is not provided."""
        loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='mean')
        
        lam = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[0.0, 1.0, 2.0]])
        
        # Should not raise error
        loss = loss_fn(lam, target)
        assert not torch.isnan(loss)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        loss_fn = ZeroInflatedPoissonNLLLoss(log_input=False, reduction='mean', eps=1e-8)
        
        # Very small lambda
        lam_small = torch.tensor([[1e-6, 1e-6]])
        pi = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[0.0, 1.0]])
        
        loss_small = loss_fn((lam_small, pi), target)
        assert not torch.isnan(loss_small)
        assert not torch.isinf(loss_small)
        
        # Very large lambda
        lam_large = torch.tensor([[100.0, 100.0]])
        target_large = torch.tensor([[50.0, 100.0]])
        
        loss_large = loss_fn((lam_large, pi), target_large)
        assert not torch.isnan(loss_large)
        assert not torch.isinf(loss_large)


class TestMaskedZIPNLLLoss:
    """Test the masked ZIP loss wrapper."""
    
    def test_basic_forward(self):
        """Test basic forward pass with batch dictionary."""
        loss_fn = MaskedZIPNLLLoss(log_input=False)
        
        batch = {
            'rhat': torch.rand(4, 8) * 5,
            'pi': torch.rand(4, 8) * 0.3,
            'robs': torch.poisson(torch.rand(4, 8) * 3),
            'dfs': torch.ones(4, 8)
        }
        
        loss = loss_fn(batch)
        
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_masking(self):
        """Test that masking works correctly."""
        loss_fn = MaskedZIPNLLLoss(log_input=False)
        
        # Create batch with partial masking
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
        
        # Losses should be different
        assert not torch.isclose(loss_full, loss_masked)
    
    def test_without_pi(self):
        """Test fallback to Poisson when pi is not in batch."""
        loss_fn = MaskedZIPNLLLoss(log_input=False)
        
        batch = {
            'rhat': torch.rand(4, 8) * 5,
            'robs': torch.poisson(torch.rand(4, 8) * 3),
            'dfs': torch.ones(4, 8)
        }
        
        # Should not raise error, falls back to Poisson
        loss = loss_fn(batch)
        assert not torch.isnan(loss)
    
    def test_without_mask(self):
        """Test behavior when mask is not provided."""
        loss_fn = MaskedZIPNLLLoss(log_input=False)
        
        batch = {
            'rhat': torch.rand(4, 8) * 5,
            'pi': torch.rand(4, 8) * 0.3,
            'robs': torch.poisson(torch.rand(4, 8) * 3)
        }
        
        # Should use mean reduction
        loss = loss_fn(batch)
        assert not torch.isnan(loss)
    
    def test_1d_mask(self):
        """Test with 1D mask (per-sample masking)."""
        loss_fn = MaskedZIPNLLLoss(log_input=False)
        
        batch = {
            'rhat': torch.rand(4, 8) * 5,
            'pi': torch.rand(4, 8) * 0.3,
            'robs': torch.poisson(torch.rand(4, 8) * 3),
            'dfs': torch.ones(4)  # 1D mask
        }
        
        loss = loss_fn(batch)
        assert not torch.isnan(loss)
    
    def test_custom_keys(self):
        """Test with custom key names."""
        loss_fn = MaskedZIPNLLLoss(
            log_input=False,
            pred_key='lambda',
            pi_key='zero_prob',
            target_key='counts',
            mask_key='valid'
        )
        
        batch = {
            'lambda': torch.rand(4, 8) * 5,
            'zero_prob': torch.rand(4, 8) * 0.3,
            'counts': torch.poisson(torch.rand(4, 8) * 3),
            'valid': torch.ones(4, 8)
        }
        
        loss = loss_fn(batch)
        assert not torch.isnan(loss)


class TestZIPvsPoisson:
    """Compare ZIP loss to standard Poisson loss."""
    
    def test_zip_reduces_to_poisson_when_pi_zero(self):
        """Test that ZIP with pi=0 is equivalent to Poisson."""
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
        
        # Should be approximately equal
        assert torch.isclose(loss_zip, loss_poisson, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

