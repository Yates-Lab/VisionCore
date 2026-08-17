"""
Regularization system for multidataset training.

This module provides a config-driven regularization system that supports:
1. Loss penalties (L1, L2, group lasso) added to training loss
2. Proximal updates (soft thresholding, clamping) applied after optimizer step
3. Sophisticated parameter matching with AND logic
4. Flexible scheduling (constant, warmup, linear ramp)
5. Adam-aware proximal L1 using per-parameter effective learning rates

Example YAML configuration:
```yaml
regularization:
  - name: sparsity_l1
    type: l1
    lambda: 1.0e-5
    apply_to: ["readouts/features"]  # matches params with BOTH "readouts" AND "features"
    schedule:
      kind: warmup
      start_epoch: 5

  - name: shrink_readout_std
    type: proximal_clamp
    lambda: 1.0  # max value
    apply_to: ["readouts/std"]
    schedule:
      kind: linear_ramp
      start_epoch: 10
      end_epoch: 100
```
"""

import torch
import torch.nn.functional as F
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
import re


def _canonical_dims(ndim: int, dims) -> tuple:
    """Resolve negative dimensions and reject duplicates/out-of-range axes."""
    if isinstance(dims, int):
        dims = [dims]
    raw = tuple(int(d) for d in dims)
    if any(d < -ndim or d >= ndim for d in raw):
        raise ValueError(f"Invalid dimensions {dims} for a {ndim}-D tensor")
    resolved = tuple(d % ndim for d in raw)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Regularizer dimensions must be unique, got {dims}")
    return resolved


def _laplacian_kernel(ndim: int, *, device, dtype) -> torch.Tensor:
    """Discrete Laplacians used by the original NeuroVisKit regularizer."""
    if ndim == 1:
        kernel = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype)
    elif ndim == 2:
        kernel = torch.tensor(
            [[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]],
            device=device,
            dtype=dtype,
        )
    elif ndim == 3:
        kernel = torch.tensor(
            [
                [[2, 3, 2], [3, 6, 3], [2, 3, 2]],
                [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
                [[2, 3, 2], [3, 6, 3], [2, 3, 2]],
            ],
            device=device,
            dtype=dtype,
        ) / 26.0
    else:
        raise ValueError("Laplacian regularization supports one to three dimensions")
    return kernel.view(1, 1, *kernel.shape)


def laplacian_penalty(
    param: torch.Tensor,
    dims,
    *,
    padding_mode: str = "constant",
    reduction: str = "dekel",
) -> torch.Tensor:
    """Squared discrete-Laplacian penalty over selected tensor dimensions.

    ``reduction='dekel'`` reproduces NeuroVisKit: sum over all unregularized
    filter/channel maps, then average over the regularized support.  ``mean``
    averages over every element and can be useful for size-invariant sweeps.
    The calculation is kept in float32 under mixed precision.
    """
    target_dims = _canonical_dims(param.ndim, dims)
    other_dims = tuple(d for d in range(param.ndim) if d not in target_dims)
    x = param.float().permute(*other_dims, *target_dims)
    spatial_shape = tuple(param.shape[d] for d in target_dims)
    x = x.reshape(-1, 1, *spatial_shape)
    kernel = _laplacian_kernel(len(target_dims), device=x.device, dtype=x.dtype)
    padding = []
    for size in reversed(kernel.shape[2:]):
        half = size // 2
        padding.extend((half, half))
    x = F.pad(x, tuple(padding), mode=padding_mode)
    conv = (F.conv1d, F.conv2d, F.conv3d)[len(target_dims) - 1]
    curvature = conv(x, kernel).square()
    if reduction == "dekel":
        return curvature.sum(dim=(0, 1)).mean()
    if reduction == "mean":
        return curvature.mean()
    if reduction == "sum":
        return curvature.sum()
    raise ValueError(f"Unknown Laplacian reduction {reduction!r}")


@torch.no_grad()
def adam_effective_lr(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Compute per-parameter effective learning rate for Adam/AdamW optimizers.

    For Adam, the effective learning rate for parameter θ_i at step t is:
        η_i,t = α * sqrt(1 - β2^t) / (1 - β1^t) * 1 / (sqrt(v_i,t) + ε)

    This is the correct scaling for proximal L1:
        θ_i ← soft_threshold(θ_i - η_i,t * g_i,t, λ * η_i,t)

    Args:
        optimizer: Adam or AdamW optimizer
        param: Parameter tensor to get effective LR for

    Returns:
        Tensor of same shape as param containing per-element effective LR,
        or None if Adam hasn't seen this param yet
    """
    state = optimizer.state.get(param, {})
    if len(state) == 0:
        return None  # Adam hasn't updated this param yet

    step = state["step"]
    exp_avg_sq = state["exp_avg_sq"]

    # Find the param group containing this parameter
    for group in optimizer.param_groups:
        if any(p is param for p in group["params"]):
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            break
    else:
        return None  # Param not found in optimizer

    # Compute bias corrections
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # Per-parameter effective learning rate
    effective_lr = (
        lr * (bias_correction2 ** 0.5) / bias_correction1
        / (exp_avg_sq.sqrt() + eps)
    )

    return effective_lr


class Regularizer:
    """
    A single regularization term that can apply loss penalties and/or proximal updates.
    
    Args:
        spec: Dictionary containing regularization specification from YAML
        named_params: List of (name, parameter) tuples from model.named_parameters()
    """
    
    def __init__(self, spec: Dict[str, Any], named_params: List[Tuple[str, torch.Tensor]]):
        self.name = spec["name"]
        self.kind = spec["type"]
        self.lmbda = float(spec["lambda"])
        self.patterns = spec.get("apply_to", [])
        self.schedule = spec.get("schedule", {"kind": "constant"})
        self.dims = spec.get("dims", None)
        self.group_dims = spec.get("group_dims", None)
        self.competition_dims = spec.get("competition_dims", [1])
        self.padding_mode = spec.get("padding_mode", "constant")
        self.reduction = spec.get("reduction", "dekel")
        self.eps = float(spec.get("eps", 1e-6))
        
        # Cache tensors that match the patterns
        self.params = []
        self.param_names = []
        for pname, param in named_params:
            if self._match(pname):
                self.params.append(param)
                self.param_names.append(pname)
        
        if not self.params:
            warnings.warn(f"[regularization] {self.name} matched no parameters!")
        else:
            print(f"[regularization] {self.name} matched {len(self.params)} parameters: {self.param_names}")
    
    def _match(self, param_name: str) -> bool:
        """
        Check if parameter name matches the patterns using AND logic.
        
        For patterns like ["readouts/features"], splits on "/" and requires
        ALL components to be present in the parameter name.
        
        Args:
            param_name: Name of the parameter to check
            
        Returns:
            True if parameter matches all pattern components
        """
        if not self.patterns:
            return False
            
        for pattern in self.patterns:
            # Split pattern on "/" for AND logic
            components = pattern.split("/")
            
            # A component prefixed with ^ is anchored to the beginning of the
            # full parameter name.  Plain components retain the historical
            # substring semantics.  The anchor matters now that
            # `modulator.*` and `output_modulator.*` coexist: a plain
            # "modulator" intentionally matches both, while "^modulator"
            # selects only the inherited feature-space path.
            def component_matches(component):
                if component.startswith("^"):
                    return param_name.startswith(component[1:])
                return component in param_name

            # Check if ALL components match the parameter name.
            if all(component_matches(comp) for comp in components):
                return True
                
        return False
    
    def is_active(self, epoch: int) -> bool:
        """
        Check if regularization should be active at the given epoch.
        
        Args:
            epoch: Current training epoch (0-based)
            
        Returns:
            True if regularization should be applied
        """
        kind = self.schedule["kind"]
        
        if kind == "constant":
            return True
        elif kind == "warmup":
            start_epoch = self.schedule.get("start_epoch", 0)
            end_epoch = self.schedule.get("end_epoch", None)
            if end_epoch is None:
                return epoch >= start_epoch
            else:
                return start_epoch <= epoch <= end_epoch
        elif kind == "linear_ramp":
            start_epoch = self.schedule.get("start_epoch", 0)
            end_epoch = self.schedule.get("end_epoch", start_epoch)
            return start_epoch <= epoch <= end_epoch
        elif kind == "ramp_then_constant":
            # Unlike the legacy ``linear_ramp`` schedule, keep the full
            # penalty active after the ramp.  This is useful when an initially
            # large structural prior would otherwise dominate the predictive
            # objective before useful features have formed.
            start_epoch = self.schedule.get("start_epoch", 0)
            return epoch >= start_epoch
        elif kind == "linear_decay":
            start_epoch = self.schedule.get("start_epoch", 0)
            return epoch >= start_epoch  # active from start_epoch onwards
        else:
            warnings.warn(f"Unknown schedule kind: {kind}")
            return False
    
    def get_schedule_weight(self, epoch: int) -> float:
        """
        Get the schedule-adjusted weight for this epoch.
        
        For linear_ramp, interpolates lambda between start and end epochs.
        For other schedules, returns lambda if active, 0 if not.
        
        Args:
            epoch: Current training epoch (0-based)
            
        Returns:
            Effective lambda value for this epoch
        """
        if not self.is_active(epoch):
            return 0.0
            
        kind = self.schedule["kind"]
        
        if kind in {"linear_ramp", "ramp_then_constant"}:
            start_epoch = self.schedule.get("start_epoch", 0)
            end_epoch = self.schedule.get("end_epoch", start_epoch)

            if end_epoch <= start_epoch:
                return self.lmbda

            if epoch >= end_epoch:
                return self.lmbda

            # Linear interpolation from 0 to lambda
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
            return self.lmbda * progress
        elif kind == "linear_decay":
            start_epoch = self.schedule.get("start_epoch", 0)
            end_epoch = self.schedule.get("end_epoch", start_epoch)
            start_lambda = self.schedule.get("start_lambda", self.lmbda)
            end_lambda = self.lmbda  # end value is the main lambda

            if end_epoch <= start_epoch:
                return end_lambda

            if epoch <= start_epoch:
                return start_lambda
            elif epoch >= end_epoch:
                return end_lambda
            else:
                # Linear interpolation from start_lambda to end_lambda
                progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
                return start_lambda + progress * (end_lambda - start_lambda)
        else:
            return self.lmbda

    def loss(self, epoch: int) -> torch.Tensor:
        """
        Compute loss penalty for this regularization term.

        Args:
            epoch: Current training epoch (0-based)

        Returns:
            Loss penalty tensor (scalar)
        """
        effective_lambda = self.get_schedule_weight(epoch)

        if effective_lambda == 0.0 or not self.params:
            return torch.tensor(0.0, device=self.params[0].device if self.params else None)

        # Only apply loss penalties for these types
        if self.kind not in {"l1", "l2", "group_lasso", "ortho", "laplacian"}:
            return torch.tensor(0.0, device=self.params[0].device)

        if self.kind == "l1":
            return effective_lambda * torch.stack([p.abs().sum() for p in self.params]).sum()
        elif self.kind == "l2":
            return effective_lambda * torch.stack([p.pow(2).sum() for p in self.params]).sum()
        elif self.kind == "group_lasso":
            # Group lasso: sum of L2 norms of parameter groups
            return effective_lambda * torch.stack([p.norm(p=2) for p in self.params]).sum()
        elif self.kind == "ortho":
            # Orthogonality penalty: ||W W^T - I||_F^2
            # Reshapes each param to [num_filters, -1] and penalizes off-diagonal
            # correlations in the Gram matrix, encouraging decorrelated filters.
            penalties = []
            for p in self.params:
                W = p.reshape(p.shape[0], -1)  # [num_filters, filter_size]
                # Normalize rows so penalty is scale-invariant
                W_norm = F.normalize(W, dim=1)
                G = W_norm @ W_norm.T  # [num_filters, num_filters]
                I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                penalties.append((G - I).pow(2).sum())
            return effective_lambda * torch.stack(penalties).sum()
        elif self.kind == "laplacian":
            if self.dims is None:
                raise ValueError(f"Laplacian regularizer {self.name!r} requires dims")
            penalties = [
                laplacian_penalty(
                    p,
                    self.dims,
                    padding_mode=self.padding_mode,
                    reduction=self.reduction,
                )
                for p in self.params
            ]
            return effective_lambda * torch.stack(penalties).sum()
        else:
            return torch.tensor(0.0, device=self.params[0].device)

    def prox(self, epoch: int, lr: float, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Apply proximal update to parameters.

        For proximal_l1, if an Adam/AdamW optimizer is provided, uses per-parameter
        effective learning rates for correct proximal geometry. Otherwise falls back
        to scalar learning rate.

        Args:
            epoch: Current training epoch (0-based)
            lr: Current learning rate (scalar fallback)
            optimizer: Optional optimizer for per-parameter LR extraction (Adam/AdamW)
        """
        effective_lambda = self.get_schedule_weight(epoch)

        if effective_lambda == 0.0 or not self.params:
            return

        if self.kind == "proximal_l1":
            # Soft thresholding for L1 proximal operator
            # Use per-parameter Adam LR if optimizer provided, else scalar LR
            for param in self.params:
                # In homogeneous multisession batches only one readout has a
                # gradient.  Applying a proximal step to all other readouts
                # would over-regularize each one by roughly N_sessions.
                if optimizer is not None and param.grad is None:
                    continue
                if optimizer is not None:
                    eta = adam_effective_lr(optimizer, param)
                    if eta is not None:
                        # Per-element soft thresholding with Adam geometry
                        shrink = effective_lambda * eta
                        param.data.copy_(
                            torch.sign(param.data) * torch.clamp(param.data.abs() - shrink, min=0.0)
                        )
                        continue
                # Fallback to scalar LR
                shrink = effective_lambda * lr
                param.data = F.softshrink(param.data, lambd=shrink)

        elif self.kind == "proximal_clamp":
            # Clamp parameters to [-lambda, lambda] range
            for param in self.params:
                param.data = torch.clamp(param.data, -effective_lambda, effective_lambda)

        elif self.kind == "proximal_clamp_positive":
            # Clamp parameters to [0, lambda] range (for std parameters)
            for param in self.params:
                param.data = torch.clamp(param.data, 0.0, effective_lambda)

        elif self.kind == "proximal_clamp_min":
            # Clamp parameters to have minimum value of lambda (for std parameters)
            for param in self.params:
                param.data = torch.clamp(param.data, min=effective_lambda)

        elif self.kind == "proximal_sparsity_dekel":
            # Port of NeuroVisKit.proximalSparsityDekel.  Within each output
            # filter, preserve the strongest input-channel group and apply
            # increasingly strong elementwise shrinkage to weaker groups.
            for param in self.params:
                if optimizer is not None and param.grad is None:
                    continue
                if self.group_dims is None:
                    norm = param.data.abs()
                else:
                    group_dims = _canonical_dims(param.ndim, self.group_dims)
                    norm = torch.linalg.vector_norm(
                        param.data, ord=2, dim=group_dims, keepdim=True
                    )
                competition_dims = _canonical_dims(param.ndim, self.competition_dims)
                strongest = norm.amax(dim=competition_dims, keepdim=True)
                relative_shrinkage = (strongest / (norm + self.eps) - 1.0).clamp_min(0.0)

                # The historical operator used the optimizer group's scalar
                # learning rate (not Adam's coordinate-wise effective rate).
                param_lr = lr
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        if any(candidate is param for candidate in group["params"]):
                            param_lr = float(group["lr"])
                            break
                threshold = effective_lambda * param_lr * relative_shrinkage
                updated = torch.sign(param.data) * (
                    param.data.abs() - threshold
                ).clamp_min(0.0)
                param.data.copy_(updated)


def create_regularizers(model_config: Dict[str, Any], named_params: List[Tuple[str, torch.Tensor]]) -> List[Regularizer]:
    """
    Create regularizers from model configuration.

    Args:
        model_config: Model configuration dictionary containing 'regularization' key
        named_params: List of (name, parameter) tuples from model.named_parameters()

    Returns:
        List of Regularizer instances
    """
    regularization_specs = model_config.get("regularization", [])

    if not regularization_specs:
        return []

    regularizers = []
    for spec in regularization_specs:
        try:
            reg = Regularizer(spec, named_params)
            regularizers.append(reg)
        except Exception as e:
            warnings.warn(f"Failed to create regularizer {spec.get('name', 'unknown')}: {e}")

    return regularizers


def get_excluded_params_for_weight_decay(regularizers: List[Regularizer]) -> List[str]:
    """
    Get list of parameter names that should be excluded from AdamW weight decay
    to avoid conflicts with custom regularization.

    Args:
        regularizers: List of Regularizer instances

    Returns:
        List of parameter names to exclude from weight decay
    """
    excluded_params = set()

    for reg in regularizers:
        # Exclude parameters that have L2-like regularization to avoid double-penalization
        if reg.kind in {"l2", "group_lasso"}:
            excluded_params.update(reg.param_names)

    return list(excluded_params)
