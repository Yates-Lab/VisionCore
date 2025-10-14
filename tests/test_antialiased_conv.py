"""
Test script for AntiAliasedConv1d fixes and new features.
"""
#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



plt.plot(torch.signal.windows.hann(10))
plt.plot(torch.signal.windows.cosine(10))
plt.plot(torch.signal.windows.hamming(10))
plt.plot(torch.signal.windows.kaiser(10))
#%%
def test_device_tracking():
    """Test that device tracking works correctly."""
    print("Testing device tracking fix...")

    conv = AntiAliasedConv1d(3, 16, kernel_size=7, aa_enable=True)

    # First forward pass - should build windows
    x = torch.randn(2, 3, 100)
    y1 = conv(x)

    # Check that device/dtype were tracked (not None, not dummy tensors)
    assert conv._aa_last_device is not None, "Device should be tracked"
    assert conv._aa_last_dtype is not None, "Dtype should be tracked"
    assert not isinstance(conv._aa_last_device, torch.Tensor), "Device should not be a tensor"
    assert not isinstance(conv._aa_last_dtype, torch.Tensor), "Dtype should not be a tensor"

    # Store initial device
    initial_device = conv._aa_last_device

    # Second forward pass - should NOT rebuild (use cached windows)
    y2 = conv(x)
    assert conv._aa_last_device == initial_device, "Device should remain the same"

    # Move to different device if CUDA available
    if torch.cuda.is_available():
        conv_cuda = AntiAliasedConv1d(3, 16, kernel_size=7, aa_enable=True).cuda()
        x_cuda = torch.randn(2, 3, 100).cuda()
        y3 = conv_cuda(x_cuda)

        # Check that CUDA device was tracked
        assert conv_cuda._aa_last_device.type == 'cuda', "CUDA device should be tracked"
        assert conv_cuda._aa_last_device != initial_device, "CUDA device should differ from CPU"
        print("✓ Device tracking works correctly (CPU and CUDA)")
    else:
        print("✓ Device tracking works correctly (CPU only)")

    print()

def test_optional_windowing():
    """Test that time and frequency windowing can be toggled independently."""
    print("Testing optional windowing...")
    
    x = torch.randn(2, 3, 100)
    
    # Test 1: Both enabled (default)
    conv_both = AntiAliasedConv1d(3, 16, kernel_size=7, 
                                   aa_enable=True, 
                                   aa_enable_time=True, 
                                   aa_enable_freq=True)
    y_both = conv_both(x)
    
    # Test 2: Only time windowing
    conv_time = AntiAliasedConv1d(3, 16, kernel_size=7,
                                   aa_enable=True,
                                   aa_enable_time=True,
                                   aa_enable_freq=False)
    conv_time.load_state_dict(conv_both.state_dict())  # Same weights
    y_time = conv_time(x)
    
    # Test 3: Only frequency windowing
    conv_freq = AntiAliasedConv1d(3, 16, kernel_size=7,
                                   aa_enable=True,
                                   aa_enable_time=False,
                                   aa_enable_freq=True)
    conv_freq.load_state_dict(conv_both.state_dict())  # Same weights
    y_freq = conv_freq(x)
    
    # Test 4: Neither (should be same as aa_enable=False)
    conv_none = AntiAliasedConv1d(3, 16, kernel_size=7,
                                   aa_enable=True,
                                   aa_enable_time=False,
                                   aa_enable_freq=False)
    conv_none.load_state_dict(conv_both.state_dict())  # Same weights
    y_none = conv_none(x)
    
    # Test 5: aa_enable=False
    conv_disabled = AntiAliasedConv1d(3, 16, kernel_size=7, aa_enable=False)
    conv_disabled.load_state_dict(conv_both.state_dict())  # Same weights
    y_disabled = conv_disabled(x)
    
    # Verify outputs are different when windowing is applied
    assert not torch.allclose(y_both, y_none), "Both windowing should differ from no windowing"
    assert not torch.allclose(y_time, y_freq), "Time-only should differ from freq-only"
    assert torch.allclose(y_none, y_disabled), "No windowing should match aa_enable=False"
    
    print("✓ Time windowing can be toggled independently")
    print("✓ Frequency windowing can be toggled independently")
    print("✓ Both can be disabled independently")
    print()

def test_double_windowing():
    """Test that double windowing can be toggled."""
    print("Testing double windowing toggle...")
    
    x = torch.randn(2, 3, 100)
    
    # With double windowing (default)
    conv_double = AntiAliasedConv1d(3, 16, kernel_size=7,
                                     aa_enable=True,
                                     aa_enable_time=True,
                                     aa_enable_freq=True,
                                     aa_double_window=True)
    y_double = conv_double(x)
    
    # Without double windowing
    conv_single = AntiAliasedConv1d(3, 16, kernel_size=7,
                                     aa_enable=True,
                                     aa_enable_time=True,
                                     aa_enable_freq=True,
                                     aa_double_window=False)
    conv_single.load_state_dict(conv_double.state_dict())  # Same weights
    y_single = conv_single(x)
    
    # Outputs should be different
    assert not torch.allclose(y_double, y_single), "Double windowing should affect output"
    
    print("✓ Double windowing can be toggled")
    print()

def test_backward_compatibility():
    """Test that default behavior is preserved."""
    print("Testing backward compatibility...")
    
    # Old-style initialization (should still work with defaults)
    conv = AntiAliasedConv1d(3, 16, kernel_size=7)
    
    x = torch.randn(2, 3, 100)
    y = conv(x)
    
    # Check defaults
    assert conv.aa_enable == True, "aa_enable should default to True"
    assert conv.aa_enable_time == True, "aa_enable_time should default to True"
    assert conv.aa_enable_freq == True, "aa_enable_freq should default to True"
    assert conv.aa_double_window == True, "aa_double_window should default to True"
    
    print("✓ Default behavior preserved")
    print()

def test_gradient_flow():
    """Test that gradients flow correctly through all configurations."""
    print("Testing gradient flow...")
    
    configs = [
        {"aa_enable": True, "aa_enable_time": True, "aa_enable_freq": True},
        {"aa_enable": True, "aa_enable_time": True, "aa_enable_freq": False},
        {"aa_enable": True, "aa_enable_time": False, "aa_enable_freq": True},
        {"aa_enable": False},
    ]
    
    for i, config in enumerate(configs):
        conv = AntiAliasedConv1d(3, 16, kernel_size=7, **config)
        x = torch.randn(2, 3, 100, requires_grad=True)
        
        y = conv(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert conv.weight.grad is not None, f"Config {i}: weight grad should exist"
        assert x.grad is not None, f"Config {i}: input grad should exist"
        
        # Check gradients are not all zeros
        assert conv.weight.grad.abs().sum() > 0, f"Config {i}: weight grad should be non-zero"
        assert x.grad.abs().sum() > 0, f"Config {i}: input grad should be non-zero"
    
    print("✓ Gradients flow correctly in all configurations")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("AntiAliasedConv1d Test Suite")
    print("=" * 60)
    print()
    
    test_device_tracking()
    test_optional_windowing()
    test_double_windowing()
    test_backward_compatibility()
    test_gradient_flow()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

