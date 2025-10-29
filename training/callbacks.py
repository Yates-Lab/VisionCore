"""
Custom PyTorch Lightning callbacks for training.
"""

import os
import time
import pytorch_lightning as pl


class Heartbeat(pl.Callback):
    """
    Emit a heartbeat message at every major Lightning hook.
    
    Useful for debugging and monitoring training progress when the
    progress bar is disabled.
    
    Only prints from rank 0 in distributed training.
    """
    
    def __init__(self):
        super().__init__()
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    def _log(self, hook):
        """Log a message with timestamp (only from rank 0)."""
        if self.rank == 0:
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] {hook}", flush=True)

    def on_fit_start(self, *_):
        self._log("fit-start")
    
    def on_train_start(self, *_):
        self._log("train-start")
    
    def on_train_batch_start(self, *_):
        self._log("train-batch-start")
    
    def on_after_backward(self, *_):
        self._log("after-backward")
    
    def on_train_batch_end(self, *_):
        self._log("train-batch-end")
    
    def on_validation_start(self, *_):
        self._log("val-start")
    
    def on_validation_end(self, *_):
        self._log("val-end")
    
    def on_validation_batch_start(self, *_):
        self._log("val-batch-start")
    
    def on_validation_batch_end(self, *_):
        self._log("val-batch-end")
    
    def on_validation_epoch_start(self, *_):
        self._log("val-epoch-start")
    
    def on_validation_epoch_end(self, *_):
        self._log("val-epoch-end")


class EpochHeartbeat(pl.Callback):
    """
    Rank-0 console heartbeat for epoch-level events.
    
    Prints:
    - Training begins (once)
    - "... running validation" (each validation loop)
    - "validation done" (each validation loop)
    - "epoch N — metric: x" (after every training epoch)
    
    Parameters
    ----------
    metric_key : str, optional
        Name of the metric to display after each epoch (default: "train_loss")
    
    Example
    -------
    >>> callback = EpochHeartbeat(metric_key="val_bps_overall")
    >>> trainer = pl.Trainer(callbacks=[callback])
    """
    
    def __init__(self, metric_key: str = "train_loss"):
        super().__init__()
        self.metric_key = metric_key
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    def _print(self, msg: str):
        """Print a message with timestamp (only from rank 0)."""
        if self.rank == 0:
            stamp = time.strftime("[%H:%M:%S] ")
            print(stamp + msg, flush=True)

    def on_fit_start(self, *_):
        self._print("training begins")
    
    def on_validation_start(self, *_):
        self._print("… running validation")
    
    def on_validation_end(self, *_):
        self._print("validation done")

    def on_train_epoch_end(self, trainer, *_):
        """Print epoch summary with metric value."""
        metric = trainer.callback_metrics.get(self.metric_key)
        if metric is not None:
            import torch
            val = metric.item() if torch.is_tensor(metric) else float(metric)
            self._print(f"epoch {trainer.current_epoch} — {self.metric_key}: {val:.4f}")
        else:
            self._print(f"epoch {trainer.current_epoch} finished")


class CurriculumCallback(pl.Callback):
    """
    Update the contrast-weighted sampler with the current training step.
    
    This callback is used with ContrastWeightedSampler to implement
    curriculum learning. It updates the sampler's step counter at the
    start of each training batch.
    
    The sampler uses this step information to gradually transition from
    contrast-weighted sampling (emphasizing high-contrast samples) to
    uniform sampling over the course of training.
    
    Example
    -------
    >>> from training.samplers import ContrastWeightedSampler
    >>> sampler = ContrastWeightedSampler(dataset, contrast_scores, ...)
    >>> callback = CurriculumCallback()
    >>> trainer = pl.Trainer(callbacks=[callback])
    """
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Update sampler or batch_sampler with current training step."""
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'train_dataloader'):
            train_loader = trainer.train_dataloader
            # Support both Sampler and BatchSampler
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_step'):
                train_loader.sampler.set_step(trainer.global_step)
            elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_step'):
                train_loader.batch_sampler.set_step(trainer.global_step)

