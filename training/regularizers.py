"""
Regularization system for multidataset training.

This module provides a config-driven regularization system that supports:
1. Loss penalties (L1, L2, group lasso) added to training loss
2. Proximal updates (soft thresholding, clamping) applied after optimizer step
3. Sophisticated parameter matching with AND logic
4. Flexible scheduling (constant, warmup, linear ramp)

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
from typing import List, Dict, Any, Tuple, Optional
import re


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
            
            # Check if ALL components are present in param_name
            if all(comp in param_name for comp in components):
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
        
        if kind == "linear_ramp":
            start_epoch = self.schedule.get("start_epoch", 0)
            end_epoch = self.schedule.get("end_epoch", start_epoch)

            if end_epoch <= start_epoch:
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
        if self.kind not in {"l1", "l2", "group_lasso"}:
            return torch.tensor(0.0, device=self.params[0].device)

        if self.kind == "l1":
            return effective_lambda * torch.stack([p.abs().sum() for p in self.params]).sum()
        elif self.kind == "l2":
            return effective_lambda * torch.stack([p.pow(2).sum() for p in self.params]).sum()
        elif self.kind == "group_lasso":
            # Group lasso: sum of L2 norms of parameter groups
            return effective_lambda * torch.stack([p.norm(p=2) for p in self.params]).sum()
        else:
            return torch.tensor(0.0, device=self.params[0].device)

    def prox(self, epoch: int, lr: float) -> None:
        """
        Apply proximal update to parameters.

        Args:
            epoch: Current training epoch (0-based)
            lr: Current learning rate
        """
        effective_lambda = self.get_schedule_weight(epoch)

        if effective_lambda == 0.0 or not self.params:
            return

        if self.kind == "proximal_l1":
            # Soft thresholding for L1 proximal operator
            shrink = effective_lambda * lr
            for param in self.params:
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
