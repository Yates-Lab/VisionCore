"""
Tests for Polar ConvNet module.
"""

import torch
import pytest
from models.modules.polar_convnet import (
    PolarConvNet, PyramidAdapter, QuadratureFilterBank2D, PolarDecompose,
    complex_from_even_odd, safe_unit_complex
)


class TestUtils:
    """Test utility functions."""
    
    def test_complex_from_even_odd(self):
        """Test complex tensor creation from even/odd components."""
        e = torch.randn(2, 3, 4)
        o = torch.randn(2, 3, 4)
        
        w = complex_from_even_odd(e, o)
        
        assert w.dtype == torch.complex64
        assert torch.allclose(w.real, e)
        assert torch.allclose(w.imag, o)
    
    def test_safe_unit_complex(self):
        """Test safe unit complex normalization."""
        # Test normal case
        w = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        u = safe_unit_complex(w)
        
        mag = torch.abs(u)
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)
        
        # Test zero case (should not crash)
        w_zero = torch.zeros(2, 3, dtype=torch.complex64)
        u_zero = safe_unit_complex(w_zero)
        assert torch.isfinite(u_zero).all()


class TestPyramidAdapter:
    """Test PyramidAdapter."""
    
    def test_pyramid_adapter_forward(self):
        """Test pyramid adapter forward pass."""
        try:
            adapter = PyramidAdapter(J=3)
            x = torch.randn(2, 1, 5, 32, 32)  # [B, C, T, H, W]
            
            levels = adapter(x)
            
            # Check we got 3 levels
            assert len(levels) == 3
            
            # Check shapes
            B, C, T = 2, 1, 5
            for l, level in enumerate(levels):
                assert level.shape[0] == B
                assert level.shape[1] == C
                assert level.shape[2] == T
                # Spatial dimensions should decrease with level
                assert level.shape[3] <= 32
                assert level.shape[4] <= 32
                
        except ImportError:
            pytest.skip("plenoptic package not available")


class TestQuadratureFilterBank2D:
    """Test QuadratureFilterBank2D."""
    
    def test_qfb_forward_level(self):
        """Test quadrature filter bank on single level."""
        qfb = QuadratureFilterBank2D(in_ch=1, pairs=8, kernel=7)
        x = torch.randn(2, 1, 10, 32, 32)  # [B, C, T, H, W]
        
        e, o = qfb.forward_level(x)
        
        # Check shapes
        assert e.shape == (2, 8, 10, 32, 32)  # [B, M, T, H, W]
        assert o.shape == (2, 8, 10, 32, 32)
        assert torch.isfinite(e).all()
        assert torch.isfinite(o).all()
    
    def test_qfb_forward_multiple_levels(self):
        """Test quadrature filter bank on multiple levels."""
        qfb = QuadratureFilterBank2D(in_ch=1, pairs=8, kernel=7)
        
        # Create dummy levels with different spatial sizes
        levels = [
            torch.randn(2, 1, 10, 32, 32),
            torch.randn(2, 1, 10, 16, 16),
            torch.randn(2, 1, 10, 8, 8)
        ]
        
        even_list, odd_list = qfb(levels)
        
        # Check we got same number of levels
        assert len(even_list) == 3
        assert len(odd_list) == 3
        
        # Check shapes
        for i, (e, o) in enumerate(zip(even_list, odd_list)):
            expected_h = levels[i].shape[3]
            expected_w = levels[i].shape[4]
            assert e.shape == (2, 8, 10, expected_h, expected_w)
            assert o.shape == (2, 8, 10, expected_h, expected_w)


class TestPolarDecompose:
    """Test PolarDecompose."""
    
    def test_polar_decompose_forward(self):
        """Test polar decomposition."""
        polar = PolarDecompose()
        
        # Create dummy even/odd responses
        even_list = [torch.randn(2, 8, 10, 32, 32) for _ in range(3)]
        odd_list = [torch.randn(2, 8, 10, 32, 32) for _ in range(3)]
        
        A_list, U_list = polar(even_list, odd_list)
        
        # Check we got same number of levels
        assert len(A_list) == 3
        assert len(U_list) == 3
        
        # Check shapes and properties
        for A, U in zip(A_list, U_list):
            # Amplitude should be positive
            assert (A >= 0).all()
            assert A.shape == (2, 8, 10, 32, 32)
            
            # Unit complex should have magnitude ≈ 1
            assert U.shape == (2, 8, 2, 10, 32, 32)  # [B, M, 2, T, H, W]
            U_re, U_im = U[:, :, 0], U[:, :, 1]
            mag = torch.sqrt(U_re**2 + U_im**2)
            assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)


class TestPolarConvNet:
    """Test PolarConvNet."""
    
    def test_polar_convnet_forward(self):
        """Test basic forward pass."""
        config = {
            'n_pyramid_levels': 4,
            'n_pairs': 16,
            'kernel_size': 7
        }
        
        try:
            model = PolarConvNet(config)
            x = torch.randn(2, 1, 10, 51, 51)  # [B, C, T, H, W]
            
            A_list, U_list = model(x)
            
            # Check we got 4 levels
            assert len(A_list) == 4
            assert len(U_list) == 4
            
            # Check shapes
            B, M, T = 2, 16, 10
            for l, (A, U) in enumerate(zip(A_list, U_list)):
                assert A.shape[0] == B
                assert A.shape[1] == M
                assert A.shape[2] == T
                assert U.shape[0] == B
                assert U.shape[1] == M
                assert U.shape[2] == 2  # real, imag
                assert U.shape[3] == T
                
                # Check amplitude is positive
                assert (A >= 0).all()
                
                # Check unit complex (magnitude ≈ 1)
                mag = torch.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2)
                assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)
                
        except ImportError:
            pytest.skip("plenoptic package not available")

    def test_polar_convnet_output_channels(self):
        """Test get_output_channels."""
        config = {'n_pyramid_levels': 4, 'n_pairs': 16}
        model = PolarConvNet(config)
        
        n_pairs, n_levels = model.get_output_channels()
        assert n_pairs == 16
        assert n_levels == 4

    def test_polar_convnet_different_sizes(self):
        """Test with different input sizes."""
        config = {'n_pyramid_levels': 3, 'n_pairs': 8}
        
        try:
            model = PolarConvNet(config)
            
            # Test different batch sizes and time steps
            for B, T in [(1, 5), (4, 15), (8, 20)]:
                x = torch.randn(B, 1, T, 51, 51)
                A_list, U_list = model(x)
                assert len(A_list) == 3
                assert A_list[0].shape[0] == B
                assert A_list[0].shape[2] == T
                
        except ImportError:
            pytest.skip("plenoptic package not available")

    def test_polar_convnet_lazy_init(self):
        """Test lazy initialization works correctly."""
        config = {'n_pyramid_levels': 2, 'n_pairs': 4}
        model = PolarConvNet(config)
        
        # Initially, pyramid and qfb should be None
        assert model.pyramid is None
        assert model.qfb is None
        
        try:
            # After first forward, they should be initialized
            x = torch.randn(1, 1, 5, 32, 32)
            model(x)
            
            assert model.pyramid is not None
            assert model.qfb is not None
            
        except ImportError:
            pytest.skip("plenoptic package not available")


if __name__ == "__main__":
    pytest.main([__file__])
