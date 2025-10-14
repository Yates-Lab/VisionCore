"""
Visualize the effect of different anti-aliasing configurations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.modules.conv_layers import AntiAliasedConv1d

def visualize_kernel_windowing():
    """Visualize how different windowing options affect a learned kernel."""
    
    # Create a conv layer with a specific kernel pattern
    kernel_size = 31
    conv_base = AntiAliasedConv1d(1, 1, kernel_size=kernel_size, aa_enable=False, bias=False)
    
    # Set a specific kernel pattern (e.g., a sinc-like function)
    with torch.no_grad():
        t = torch.linspace(-3, 3, kernel_size)
        # Create a kernel with high-frequency components
        kernel = torch.sinc(t * 2) * torch.cos(t * 4)
        conv_base.weight[0, 0, :] = kernel
    
    # Create different configurations
    configs = {
        'Original (no AA)': {'aa_enable': False},
        'Time only': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': False},
        'Freq only': {'aa_enable': True, 'aa_enable_time': False, 'aa_enable_freq': True},
        'Both (single window)': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': True, 'aa_double_window': False},
        'Both (double window)': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': True, 'aa_double_window': True},
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs.items()):
        conv = AntiAliasedConv1d(1, 1, kernel_size=kernel_size, bias=False, **config)
        conv.weight.data = conv_base.weight.data.clone()
        
        # Get the effective kernel
        with torch.no_grad():
            effective_kernel = conv._window_weight()[0, 0, :].cpu().numpy()
        
        # Plot time domain
        ax = axes[idx]
        t_axis = np.arange(kernel_size)
        ax.plot(t_axis, kernel.numpy(), 'k--', alpha=0.3, label='Original', linewidth=2)
        ax.plot(t_axis, effective_kernel, 'b-', label='Windowed', linewidth=2)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide the last subplot if odd number of configs
    if len(configs) % 2 == 1:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('antialiased_conv_kernels.png', dpi=150, bbox_inches='tight')
    print("Saved kernel visualization to 'antialiased_conv_kernels.png'")
    plt.close()

def visualize_frequency_response():
    """Visualize the frequency response of different configurations."""
    
    kernel_size = 31
    conv_base = AntiAliasedConv1d(1, 1, kernel_size=kernel_size, aa_enable=False, bias=False)
    
    # Set a specific kernel pattern
    with torch.no_grad():
        t = torch.linspace(-3, 3, kernel_size)
        kernel = torch.sinc(t * 2) * torch.cos(t * 4)
        conv_base.weight[0, 0, :] = kernel
    
    configs = {
        'Original (no AA)': {'aa_enable': False},
        'Time only': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': False},
        'Freq only': {'aa_enable': True, 'aa_enable_time': False, 'aa_enable_freq': True},
        'Both (single window)': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': True, 'aa_double_window': False},
        'Both (double window)': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': True, 'aa_double_window': True},
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Compute FFT for each configuration
    for name, config in configs.items():
        conv = AntiAliasedConv1d(1, 1, kernel_size=kernel_size, bias=False, **config)
        conv.weight.data = conv_base.weight.data.clone()
        
        with torch.no_grad():
            effective_kernel = conv._window_weight()[0, 0, :].cpu()
        
        # Compute frequency response
        fft = torch.fft.rfft(effective_kernel, n=256)
        magnitude = torch.abs(fft).detach().numpy()
        phase = torch.angle(fft).detach().numpy()
        freqs = torch.fft.rfftfreq(256).numpy()
        
        # Plot magnitude
        axes[0].plot(freqs, 20 * np.log10(magnitude + 1e-10), label=name, linewidth=2)
        
        # Plot phase
        axes[1].plot(freqs, phase, label=name, linewidth=2)
    
    axes[0].set_xlabel('Normalized Frequency (cycles/sample)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Frequency Response - Magnitude', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0.45, color='r', linestyle='--', alpha=0.5, label='Default cutoff')
    
    axes[1].set_xlabel('Normalized Frequency (cycles/sample)')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title('Frequency Response - Phase', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('antialiased_conv_frequency.png', dpi=150, bbox_inches='tight')
    print("Saved frequency response to 'antialiased_conv_frequency.png'")
    plt.close()

def compare_computational_cost():
    """Compare the computational cost of different configurations."""
    import time
    
    print("\nComputational Cost Comparison")
    print("=" * 60)
    
    kernel_size = 31
    in_channels = 32
    out_channels = 64
    batch_size = 16
    seq_length = 1000
    n_iterations = 100
    
    configs = {
        'No AA': {'aa_enable': False},
        'Time only': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': False},
        'Freq only': {'aa_enable': True, 'aa_enable_time': False, 'aa_enable_freq': True},
        'Both (default)': {'aa_enable': True, 'aa_enable_time': True, 'aa_enable_freq': True},
    }
    
    x = torch.randn(batch_size, in_channels, seq_length)
    
    for name, config in configs.items():
        conv = AntiAliasedConv1d(in_channels, out_channels, kernel_size=kernel_size, **config)
        
        # Warmup
        for _ in range(10):
            _ = conv(x)
        
        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            _ = conv(x)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / n_iterations) * 1000
        print(f"{name:20s}: {avg_time_ms:6.2f} ms/forward")
    
    print("=" * 60)

if __name__ == "__main__":
    print("Visualizing AntiAliasedConv1d configurations...")
    print()
    
    visualize_kernel_windowing()
    visualize_frequency_response()
    compare_computational_cost()
    
    print("\nDone! Check the generated PNG files for visualizations.")

