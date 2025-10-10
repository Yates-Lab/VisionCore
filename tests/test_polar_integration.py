"""
Integration tests for Polar-V1 pipeline.

This module tests the complete Polar-V1 pipeline from configuration to forward pass.
"""

import torch
import pytest
from pathlib import Path


def test_polar_convnet_factory():
    """Test that PolarConvNet can be created via factory."""
    from models.factory import create_convnet
    
    config = {
        'n_pyramid_levels': 4,
        'n_pairs': 16,
        'kernel_size': 7
    }
    
    try:
        model, output_channels = create_convnet('polar', in_channels=1, **config)
        
        assert model is not None
        assert output_channels == (16, 4)  # (n_pairs, n_levels)
        
        # Test forward pass
        x = torch.randn(2, 1, 10, 51, 51)
        A_list, U_list = model(x)
        assert len(A_list) == 4
        assert len(U_list) == 4
        
    except ImportError:
        pytest.skip("plenoptic package not available")


def test_polar_modulator_factory():
    """Test that PolarModulator can be created via factory."""
    from models.factory import create_modulator
    
    feature_dim = (16, 4)  # (n_pairs, n_levels)
    
    modulator, out_dim = create_modulator(
        'polar',
        feature_dim=feature_dim,
        behavior_dim=2,
        beh_dim=128,
        dt=1/240
    )
    
    assert modulator is not None
    assert out_dim == 0  # Polar modulator doesn't add channels
    
    # Test forward pass
    A_list = [torch.randn(2, 16, 10, 32, 32) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 32, 32) for _ in range(4)]
    behavior = torch.randn(2, 10, 2)
    
    feats = (A_list, U_list)
    out_feats = modulator(feats, behavior)
    
    # Check features unchanged
    out_A, out_U = out_feats
    assert len(out_A) == 4
    assert len(out_U) == 4
    
    # Check behavior params stored
    assert modulator.beh_params is not None


def test_polar_recurrent_factory():
    """Test that PolarRecurrent can be created via factory."""
    from models.factory import create_recurrent
    
    input_dim = (16, 4)  # (n_pairs, n_levels)
    
    recurrent, output_channels = create_recurrent(
        'polar',
        input_dim=input_dim,
        dt=1/240,
        lambda_fix=10.0,
        lambda_sac=40.0
    )
    
    assert recurrent is not None
    assert output_channels == 5 * 16  # 5 summaries per pair
    
    # Test forward pass (minimal mode - no behavior)
    recurrent.use_behavior = False
    A_list = [torch.randn(2, 16, 10, 32, 32) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 32, 32) for _ in range(4)]
    
    feats = (A_list, U_list)
    feats_per_level = recurrent(feats)
    
    assert len(feats_per_level) == 4
    for feat in feats_per_level:
        assert feat.shape[0] == 2  # batch size
        assert feat.shape[1] == 5 * 16  # 5 summaries per pair
        assert feat.dim() == 4  # [B, C, H, W]


def test_polar_readout_factory():
    """Test that PolarReadout can be created via factory."""
    from models.factory import create_readout
    
    readout = create_readout(
        'polar',
        in_channels=80,  # 5 * 16 summaries per pair
        n_units=8
    )
    
    assert readout is not None
    
    # Test forward pass
    feats_per_level = [
        torch.randn(2, 80, 32, 32),
        torch.randn(2, 80, 16, 16),
        torch.randn(2, 80, 8, 8),
        torch.randn(2, 80, 4, 4)
    ]
    
    output = readout(feats_per_level)
    
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_full_polar_pipeline():
    """Test complete Polar-V1 pipeline."""
    try:
        from models.factory import create_convnet, create_modulator, create_recurrent, create_readout
        
        # Create all components
        convnet, conv_out = create_convnet('polar', in_channels=1, n_pyramid_levels=4, n_pairs=16)
        
        modulator, mod_out = create_modulator(
            'polar',
            feature_dim=conv_out,
            behavior_dim=2,
            beh_dim=128
        )
        
        recurrent, rec_out = create_recurrent(
            'polar',
            input_dim=conv_out,
            dt=1/240,
            use_behavior=True
        )
        
        readout = create_readout('polar', in_channels=rec_out, n_units=8)
        
        # Set modulator reference
        recurrent.set_modulator(modulator)
        
        # Test forward pass
        x = torch.randn(2, 1, 10, 51, 51)
        behavior = torch.randn(2, 10, 2)
        
        # Forward through pipeline
        A_list, U_list = convnet(x)
        feats = modulator((A_list, U_list), behavior)
        feats_per_level = recurrent(feats)
        output = readout(feats_per_level)
        
        assert output.shape == (2, 8)
        assert torch.isfinite(output).all()
        
    except ImportError:
        pytest.skip("plenoptic package not available")


def test_polar_config_loading():
    """Test that Polar-V1 configs can be loaded."""
    from models.config_loader import load_config
    
    # Test main config
    config_path = Path("experiments/model_configs/polar_v1.yaml")
    if config_path.exists():
        config = load_config(config_path)
        
        assert config['model_type'] == 'v1multi'
        assert config['convnet']['type'] == 'polar'
        assert config['modulator']['type'] == 'polar'
        assert config['recurrent']['type'] == 'polar'
        assert config['readout']['type'] == 'polar'
    
    # Test minimal config
    config_path = Path("experiments/model_configs/polar_v1_minimal.yaml")
    if config_path.exists():
        config = load_config(config_path)
        
        assert config['modulator']['type'] == 'none'
        assert config['recurrent']['params']['use_behavior'] == False
        assert config['recurrent']['params']['use_jepa'] == False


if __name__ == "__main__":
    pytest.main([__file__])
