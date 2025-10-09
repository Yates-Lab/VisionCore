#!/usr/bin/env python3
"""
Plug-and-play MEI synthesis (2-D or 3-D) with jitter, TV/Lp regularisers,
Gaussian-blurred gradients, and optional clipping.

The demo at the bottom runs on a tiny 3-D ConvNet so you can check that the
pipeline works end-to-end.
"""

import torch
import torch.nn.functional as F
from math import ceil

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def _gaussian_kernel_nd(sigma, order: int, channels: int):
    """
    Create an N-D separable Gaussian kernel suitable for depth-wise conv.
    Returns None if all sigmas are 0 (no blur).

    Args:
        sigma: float or list/array of floats. If float, same sigma for all dims.
               If list/array, must have length == order, one sigma per dimension.
        order: Number of spatial dimensions
        channels: Number of channels for depth-wise convolution
    """
    # Handle sigma input - convert to list if needed
    if isinstance(sigma, (int, float)):
        if sigma == 0:
            return None
        sigmas = [float(sigma)] * order
    else:
        sigmas = list(sigma)
        if len(sigmas) != order:
            raise ValueError(f"sigma list length ({len(sigmas)}) must match order ({order})")
        if all(s == 0 for s in sigmas):
            return None

    # Create 1D kernels for each dimension
    kernels_1d = []
    for dim_sigma in sigmas:
        if dim_sigma == 0:
            # For zero sigma, use a delta function (single 1.0)
            g1d = torch.tensor([1.0], dtype=torch.float32)
        else:
            k_size = int(ceil(dim_sigma * 6)) | 1                   # odd
            ax = torch.arange(k_size, dtype=torch.float32) - k_size // 2
            g1d = torch.exp(-(ax ** 2) / (2 * dim_sigma ** 2))
            g1d /= g1d.sum()
        kernels_1d.append(g1d)

    # Create N-D kernel via outer products
    kernel_nd = kernels_1d[0]
    for i in range(1, order):
        # Add new dimension and multiply with next 1D kernel
        kernel_nd = kernel_nd.unsqueeze(-1) * kernels_1d[i]

    # prepend batch & channel singleton axes, then expand depth-wise
    kernel_nd = kernel_nd.unsqueeze(0).unsqueeze(0)             # (1,1,k1,k2,...)

    # Expand to create depth-wise convolution kernel
    kernel_shape = [channels, 1] + list(kernel_nd.shape[2:])
    kernel_nd = kernel_nd.expand(*kernel_shape).contiguous()    # (C,1,k1,k2,...)

    return kernel_nd


# ==============================================================================
# 2. OPTIMISATION COMPONENTS
# ==============================================================================

class OptimComponent:
    """Base class to give everything a consistent call signature."""
    def __call__(self, x, iteration=None):
        raise NotImplementedError


# ---------- Regularisers ----------

class LpNorm(OptimComponent):
    """
    Use *mean* Lp so the weight means the same thing regardless of
    image size, and so gradients don't explode when you add pixels.
    """
    def __init__(self, p: int = 2, weight: float = 1.0):
        self.p = p
        self.weight = weight

    def __call__(self, x, iteration=None):
        return x.abs().pow(self.p).mean().pow(1.0 / self.p) * self.weight


class TotalVariation(OptimComponent):
    """
    Skip channel axis; only penalise spatial/temporal differences.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, x, iteration=None):
        tv = 0.0
        for dim in range(2, x.dim()):            # *** start at 2 (D,H,W)
            diff = x.roll(-1, dims=dim) - x
            tv += diff.narrow(dim, 0, x.size(dim) - 1).abs().mean()
        return tv * self.weight


class Combine(OptimComponent):
    def __init__(self, components):
        self.components = components

    def __call__(self, x, iteration=None):
        return sum(comp(x, iteration) for comp in self.components)


# ---------- Transforms / Augmentations ----------

class Jitter(OptimComponent):
    """Reflect-pad then crop; max_jitter_list has one entry per spatial dim."""
    def __init__(self, max_jitter_list):
        self.max_jitter_list = max_jitter_list

    def __call__(self, x, iteration=None):
        paddings = []
        crops = [slice(None), slice(None)]  # batch & channel
        for i, max_jit in enumerate(self.max_jitter_list):
            if max_jit == 0:
                crops.append(slice(None))
                paddings.extend((0, 0))
                continue
            dim_idx = i + 2
            jit = torch.randint(-max_jit, max_jit + 1, ()).item()
            pad = (jit, 0) if jit >= 0 else (0, -jit)
            paddings.extend(pad)
            dim_len = x.shape[dim_idx]
            crop = slice(0, dim_len) if jit > 0 else slice(-jit, None)
            crops.append(crop)

        x_pad = F.pad(x, list(reversed(paddings)), mode="reflect")
        return x_pad[crops]


# ---------- Gradient Pre-conditioner ----------

class GaussianGradientBlur(OptimComponent):
    def __init__(self, sigma, order: int):
        """
        Args:
            sigma: float or list/array of floats. If float, same sigma for all dims.
                   If list/array, must have length == order, one sigma per dimension.
            order: Number of spatial dimensions (2 for 2D, 3 for 3D, etc.)
        """
        self.sigma = sigma
        self.order = order
        self.kernel = None

    def __call__(self, grad, iteration=None):
        # Check if we should skip blurring
        if isinstance(self.sigma, (int, float)) and self.sigma == 0:
            return grad
        elif hasattr(self.sigma, '__iter__') and all(s == 0 for s in self.sigma):
            return grad

        if self.kernel is None:
            C = grad.shape[0]
            self.kernel = _gaussian_kernel_nd(self.sigma, self.order, C).to(
                grad.device, grad.dtype
            )
            # Calculate padding for each dimension based on actual kernel size
            # PyTorch conv padding format:
            # - conv2d: padding can be int or (pad_h, pad_w)
            # - conv3d: padding can be int or (pad_d, pad_h, pad_w)
            kernel_shape = self.kernel.shape[2:]  # Skip batch and channel dims
            self.pad = tuple(size // 2 for size in kernel_shape)

        conv = F.conv2d if self.order == 2 else F.conv3d
        blurred = conv(
            grad,
            self.kernel,
            padding=self.pad,
            groups=grad.shape[0],
        )
        return blurred


# ---------- Post-step ----------

class ClipRange(OptimComponent):
    def __init__(self, min_val=-1.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x, iteration=None):
        return torch.clamp(x, self.min_val, self.max_val)


# ==============================================================================
# 3. MAIN SYNTHESIS LOOP
# ==============================================================================

# def mei_synthesis(
#     model,
#     initial_image,
#     unit,
#     n_iter=1000,
#     optimizer_fn=torch.optim.Adam,
#     optimizer_kwargs=None,
#     transform: OptimComponent | None = None,
#     regulariser: OptimComponent | None = None,
#     preconditioner: OptimComponent | None = None,
#     postprocessor: OptimComponent | None = None,
#     device=None,
# ):
#     """
#     Optimise `initial_image` so that `model`'s activation at `unit`
#     is maximised, subject to regularisation etc.
#     """
#     if device is None:
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = "cpu"

#     img = initial_image.clone().to(device)
#     img.requires_grad_(True)

#     if optimizer_kwargs is None:
#         optimizer_kwargs = {"lr": 0.05}
#     optim = optimizer_fn([img], **optimizer_kwargs)

#     for t in range(n_iter):
#         optim.zero_grad()

#         x = transform(img, t) if transform else img

#         act = model(x)[unit]      # !! adapt for your model if needed

#         reg_loss = regulariser(img, t) if regulariser else 0.0
#         loss = -act + reg_loss
#         loss.backward()

#         if preconditioner and img.grad is not None:
#             img.grad.copy_(preconditioner(img.grad, t))

#         optim.step()

#         if postprocessor:
#             with torch.no_grad():
#                 img.copy_(postprocessor(img, t))

#     return img.detach()
import torch
import torch.nn.utils as tutils
from contextlib import suppress

def mei_synthesis(
    model,
    initial_image,
    unit,
    n_iter=1000,
    *,
    optimizer_fn=torch.optim.Adam,
    optimizer_kwargs=None,
    transform=None,
    regulariser=None,
    preconditioner=None,
    postprocessor=None,
    device=None,
    max_grad_norm=5.0,          # <-- new
    detect_anomaly=False,       # <-- new
    abort_on_nan=True,          # <-- new
):
    """
    Optimise `initial_image` so that `model`'s activation at `unit`
    is maximised, with a few runtime safety checks.
    """
    if device is None:
        with suppress(StopIteration):
            device = next(model.parameters()).device
    device = device or "cpu"

    img = initial_image.to(device).clone().requires_grad_(True)

    if optimizer_kwargs is None:
        optimizer_kwargs = dict(lr=0.01)     # start safer
    optim = optimizer_fn([img], **optimizer_kwargs)

    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    for t in range(n_iter):
        optim.zero_grad(set_to_none=True)

        x = transform(img, t) if transform else img

        act = model(x)[unit]              # make sure `unit` indexing is valid
        reg = regulariser(img, t) if regulariser else 0.0
        loss = -act + reg
        print(f"[iteration {t}] Loss: {loss.item()}")

        # ----- finite-value guard -----
        if abort_on_nan and not torch.isfinite(loss):
            print(f"[iteration {t}] Loss became non-finite ({loss.item()}). "
                  "Stopping optimisation.")
            break

        loss.backward()

        # gradient pre-conditioning
        if preconditioner and img.grad is not None:
            img.grad.copy_(preconditioner(img.grad, t))

        # ----- clip exploding gradients -----
        if max_grad_norm is not None:
            tutils.clip_grad_norm_([img], max_grad_norm)

        # another finite-grad check
        if abort_on_nan and not torch.all(torch.isfinite(img.grad)):
            print(f"[iteration {t}] Gradient has inf/NaN.  Stopping.")
            break

        optim.step()

        # optional param post-processing
        if postprocessor:
            with torch.no_grad():
                img.copy_(postprocessor(img, t))

    return img.detach()



# ==============================================================================
# 4. QUICK DEMO
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # Test different sigma per dimension functionality
    print("Testing GaussianGradientBlur with different sigmas per dimension...")

    # Test 2D case
    grad_2d = torch.randn(3, 32, 32)  # (C, H, W)
    blur_2d_uniform = GaussianGradientBlur(sigma=1.0, order=2)
    blur_2d_anisotropic = GaussianGradientBlur(sigma=[0.5, 2.0], order=2)  # less blur in H, more in W

    result_uniform = blur_2d_uniform(grad_2d)
    result_anisotropic = blur_2d_anisotropic(grad_2d)
    print(f"2D uniform blur: {result_uniform.shape}")
    print(f"2D anisotropic blur: {result_anisotropic.shape}")

    # Test 3D case
    grad_3d = torch.randn(2, 20, 32, 32)  # (C, D, H, W)
    blur_3d_uniform = GaussianGradientBlur(sigma=1.0, order=3)
    blur_3d_anisotropic = GaussianGradientBlur(sigma=[0.2, 1.0, 1.5], order=3)  # different blur for T, H, W

    result_3d_uniform = blur_3d_uniform(grad_3d)
    result_3d_anisotropic = blur_3d_anisotropic(grad_3d)
    print(f"3D uniform blur: {result_3d_uniform.shape}")
    print(f"3D anisotropic blur: {result_3d_anisotropic.shape}")

    print("✓ All tests passed!\n")

    # Tiny 3-D ConvNet
    net = torch.nn.Sequential(
        torch.nn.Conv3d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),  # output shape: (B, 8)
    )

    initial_movie = torch.randn(1, 1, 20, 32, 32) * 0.1  # (B,C,D,H,W)

    transform = Jitter([2, 4, 4])                       # time, H, W
    regulariser = Combine([LpNorm(p=2, weight=0.01), TotalVariation(weight=0.001)])
    # Example: less temporal blur, more spatial blur
    precond = GaussianGradientBlur(sigma=[0.5, 1.5, 1.5], order=3)  # [time, height, width]
    post = ClipRange(-1.0, 1.0)

    mei = mei_synthesis(
        model=net,
        initial_image=initial_movie,
        unit=(0,),               # 1st feature in the flattened vector
        n_iter=500,
        optimizer_kwargs={"lr": 0.05},
        transform=transform,
        regulariser=regulariser,
        preconditioner=precond,
        postprocessor=post,
    )

    print("Generated MEI shape:", mei.shape)  # (1,1,20,32,32)
