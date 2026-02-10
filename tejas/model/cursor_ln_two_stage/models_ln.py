"""Minimal LN model: linear filter + softplus, matching lnp_time_simple_working.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNonLinearModel(nn.Module):
    """One spatiotemporal kernel, generator = stim @ kernel, rhat = softplus(scale * generator + bias)."""

    def __init__(self, kernel_shape):
        super().__init__()
        assert len(kernel_shape) == 3  # (lags, H, W)
        self.kernel_ = nn.Parameter(torch.randn(*kernel_shape))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    @property
    def kernel(self):
        return self.kernel_

    def forward(self, x):
        B = x["stim"].shape[0]
        x_flat = x["stim"].view(B, -1)
        k_flat = self.kernel.view(-1)
        generator = x_flat @ k_flat
        rhat = F.softplus(self.scale * generator + self.bias)
        x["rhat"] = rhat.view(x["robs"].shape).clamp(min=1e-6)
        return x


class AffineSoftplus(nn.Module):
    """Scale and shift for precomputed generator: rhat = softplus(weight * generator + bias)."""

    def __init__(self, learn_bias=True):
        super().__init__()
        self.learn_bias = learn_bias
        self.bias = nn.Parameter(torch.tensor(0.0)) if learn_bias else 0.0
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        g = x["generator"]
        b = self.bias if self.learn_bias else 0.0
        x["rhat"] = F.softplus(g * self.weight + b).view(x["robs"].shape).clamp(min=1e-6)
        return x
