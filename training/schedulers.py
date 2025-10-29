"""
Custom learning rate schedulers for training.
"""

import math
import torch


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    During warmup (first `warmup_epochs` epochs), the learning rate increases
    linearly from `warmup_start_lr` to the base learning rate.

    After warmup, the learning rate follows a cosine annealing schedule,
    decreasing from the base learning rate to `eta_min`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer
    warmup_epochs : int
        Number of epochs for linear warmup
    max_epochs : int
        Total number of training epochs
    warmup_start_lr : float, optional
        Learning rate at the start of warmup (default: 0.0)
    eta_min : float, optional
        Minimum learning rate (default: 0.0)
    last_epoch : int, optional
        The index of last epoch (default: -1)

    Example
    -------
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    >>> scheduler = LinearWarmupCosineAnnealingLR(
    ...     optimizer, warmup_epochs=5, max_epochs=100
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) *
                self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class LinearWarmupCosineAnnealingWarmRestartsLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing with warm restarts.

    During warmup (first `warmup_epochs` epochs), the learning rate increases
    linearly from `warmup_start_lr` to the base learning rate.

    After warmup, the learning rate follows a cosine annealing schedule with warm restarts.
    Each restart cycle lasts `restart_period` epochs, during which the LR anneals from
    the base learning rate down to `eta_min`, then jumps back up to the base LR for the
    next cycle.

    This implements a simplified version of SGDR (Stochastic Gradient Descent with Warm Restarts)
    where the restart period is constant (T_0 = T_mult = 1 in SGDR terminology).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer
    warmup_epochs : int
        Number of epochs for linear warmup
    max_epochs : int
        Total number of training epochs
    restart_period : int, optional
        Number of epochs in each cosine annealing cycle (default: max_epochs // 2)
        After warmup, the LR will complete a full cosine cycle every `restart_period` epochs
    warmup_start_lr : float, optional
        Learning rate at the start of warmup (default: 0.0)
    eta_min : float, optional
        Minimum learning rate at the end of each cosine cycle (default: 0.0)
    last_epoch : int, optional
        The index of last epoch (default: -1)

    Example
    -------
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    >>> # Warmup for 5 epochs, then restart every 50 epochs
    >>> scheduler = LinearWarmupCosineAnnealingWarmRestartsLR(
    ...     optimizer, warmup_epochs=5, max_epochs=100, restart_period=50
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, restart_period=None,
                 warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        # Default restart period is half the remaining epochs after warmup
        if restart_period is None:
            restart_period = (max_epochs - warmup_epochs) // 2
        self.restart_period = restart_period
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) *
                self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing with warm restarts
            # Calculate position within the current restart cycle
            epochs_since_warmup = self.last_epoch - self.warmup_epochs
            cycle_progress = (epochs_since_warmup % self.restart_period) / self.restart_period

            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * cycle_progress)) / 2
                for base_lr in self.base_lrs
            ]

