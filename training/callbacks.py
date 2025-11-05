"""
Custom PyTorch Lightning callbacks for training.
"""

import os
import time
import numpy as np
import torch
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


class ModelLoggingCallback(pl.Callback):
    """
    Two-speed logging callback for model training.

    Performs two types of logging at different intervals:
    1. Fast logging (every N epochs): Model kernel visualizations
    2. Slow logging (every M epochs): Full evaluation stack with BPS, CCNORM, saccade, STA

    Only logs from rank 0 in distributed training.

    Parameters
    ----------
    fast_interval : int, optional
        Interval for fast logging (kernel visualizations), by default 5
    slow_interval : int, optional
        Interval for slow logging (evaluation stack), by default 10
    eval_dataset_idx : int, optional
        Dataset index to evaluate during slow logging, by default 7
    batch_size : int, optional
        Batch size for evaluation, by default 64
    rescale : bool, optional
        Whether to apply affine rescaling to predictions, by default True

    Example
    -------
    >>> callback = ModelLoggingCallback(
    ...     fast_interval=5,
    ...     slow_interval=10,
    ...     eval_dataset_idx=7
    ... )
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        fast_interval: int = 5,
        slow_interval: int = 10,
        eval_dataset_idx: int = 7,
        batch_size: int = 64,
        rescale: bool = True
    ):
        super().__init__()
        self.fast_interval = fast_interval
        self.slow_interval = slow_interval
        self.eval_dataset_idx = eval_dataset_idx
        self.eval_analyses = ['bps', 'ccnorm', 'saccade', 'sta', 'qc']
        self.batch_size = batch_size
        self.rescale = rescale

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Performs fast or slow logging based on current epoch number.
        """
        # Only log from rank 0
        if trainer.global_rank != 0:
            return

        current_epoch = trainer.current_epoch

        # Fast logging: every fast_interval epochs
        if current_epoch > 0 and current_epoch % self.fast_interval == 0:
            self._fast_logging(trainer, pl_module, current_epoch)

        # Slow logging: every slow_interval epochs
        if current_epoch > 0 and current_epoch % self.slow_interval == 0:
            self._slow_logging(trainer, pl_module, current_epoch)

    def _fast_logging(self, trainer, pl_module, epoch):
        """
        Fast logging: visualize model kernels.

        Logs:
        - Frontend temporal kernels
        - Stem convolutional kernels
        - Layer convolutional kernels
        - Readout weights
        """
        import matplotlib.pyplot as plt
        import wandb

        print(f"\n[Epoch {epoch}] Running fast logging (kernels)...", flush=True)

        try:
            model = pl_module.model  # Access the wrapped model
            log_dict = {}

            # 1. Frontend temporal kernels
            if hasattr(model, 'frontend') and hasattr(model.frontend, 'temporal_conv'):
                fig = plt.figure(figsize=(8, 6))
                weights = model.frontend.temporal_conv.weight.squeeze().detach().cpu().T
                plt.plot(weights)
                plt.title('Frontend Temporal Kernels')
                plt.xlabel('Time')
                plt.ylabel('Weight')
                log_dict['kernels/frontend'] = wandb.Image(fig)
                plt.close(fig)

            # 2. Stem kernels
            if hasattr(model, 'convnet') and hasattr(model.convnet, 'stem'):
                if hasattr(model.convnet.stem, 'components') and hasattr(model.convnet.stem.components, 'conv'):
                    if hasattr(model.convnet.stem.components.conv, 'plot_weights'):
                        fig, _ = model.convnet.stem.components.conv.plot_weights()
                        plt.title('Stem Kernels')
                        log_dict['kernels/stem'] = wandb.Image(fig)
                        plt.close(fig)

            # 3. Layer kernels
            if hasattr(model, 'convnet') and hasattr(model.convnet, 'layers'):
                for i, layer in enumerate(model.convnet.layers):
                    try:
                        if hasattr(layer, 'components') and hasattr(layer.components, 'conv'):
                            if hasattr(layer.components.conv, 'plot_weights'):
                                fig, _ = layer.components.conv.plot_weights(nrow=20)
                                log_dict[f'kernels/layer_{i}'] = wandb.Image(fig)
                                plt.close(fig)
                        elif hasattr(layer, 'main_block') and hasattr(layer.main_block, 'components'):
                            if hasattr(layer.main_block.components, 'conv'):
                                if hasattr(layer.main_block.components.conv, 'plot_weights'):
                                    fig, _ = layer.main_block.components.conv.plot_weights(nrow=20)
                                    log_dict[f'kernels/layer_{i}'] = wandb.Image(fig)
                                    plt.close(fig)
                    except Exception as e:
                        print(f"Warning: Could not plot layer {i} kernels: {e}")

            # 4. Readout weights (sample from first readout)
            if hasattr(model, 'readouts') and len(model.readouts) > 0:
                try:
                    readout = model.readouts[0]
                    if hasattr(readout, 'plot_weights'):
                        # Run dummy input through readout to initialize if needed
                        device = next(model.parameters()).device
                        dummy_input = torch.randn(1, readout.features.in_channels, 1, 15, 15).to(device)
                        with torch.no_grad():
                            readout(dummy_input)
                        result = readout.plot_weights(ellipse=False)
                        # plot_weights returns (fig, fig_spatial) or (None, None)
                        if result is not None and result[0] is not None:
                            fig, fig_spatial = result
                            log_dict['kernels/readout_0_features'] = wandb.Image(fig)
                            if fig_spatial is not None:
                                log_dict['kernels/readout_0_spatial'] = wandb.Image(fig_spatial)
                            plt.close(fig)
                            if fig_spatial is not None:
                                plt.close(fig_spatial)
                except Exception as e:
                    print(f"Warning: Could not plot readout weights: {e}")

            # Log all figures to wandb in a single call
            if log_dict:
                pl_module.logger.experiment.log(log_dict, step=trainer.global_step)
                print(f"  ✓ Logged {len(log_dict)} kernel visualizations")

            # Clean up
            plt.close('all')

        except Exception as e:
            print(f"Error during fast logging: {e}")
            import traceback
            traceback.print_exc()

        print(f"[Epoch {epoch}] Fast logging complete.\n", flush=True)

    def _slow_logging(self, trainer, pl_module, epoch):
        """
        Slow logging: run full evaluation stack.

        Follows scripts/eval_stack_devel.py exactly.
        """
        import matplotlib.pyplot as plt
        import wandb
        from torchvision.utils import make_grid

        print(f"\n[Epoch {epoch}] Running slow logging (eval stack)...", flush=True)

        # Import eval stack function
        try:
            from eval.eval_stack_multidataset import eval_stack_single_dataset
        except ImportError:
            print("Error: Could not import eval_stack_single_dataset. Skipping slow logging.")
            return

        # Temporarily set model to eval mode
        was_training = pl_module.training
        pl_module.eval()

        try:
            # Evaluate dataset 7 (matching the script)
            dataset_idx = self.eval_dataset_idx
            if dataset_idx >= len(pl_module.names):
                print(f"Warning: Dataset index {dataset_idx} out of range. Skipping slow logging.")
                return

            dataset_name = pl_module.names[dataset_idx]
            print(f"  Evaluating dataset {dataset_idx}: {dataset_name}")

            try:
                # Run evaluation stack (matching line 106-111 of script)
                results = eval_stack_single_dataset(
                    model=pl_module,
                    dataset_idx=dataset_idx,
                    analyses=self.eval_analyses,
                    batch_size=self.batch_size,
                    rescale=self.rescale
                )

                # Accumulate all logging data
                log_dict = {}
                prefix = f"eval_ds{dataset_idx}"

                # 1. BPS plot (matching lines 158-166 of script)
                if 'bps' in results:
                    fig = plt.figure(figsize=(12, 6))
                    for k in results['bps']:
                        if k in ['val', 'cids']:
                            continue
                        plt.plot(np.arange(len(results['bps'][k]['bps'])),
                                np.maximum(results['bps'][k]['bps'], -.1),
                                label=k, alpha=0.5)
                    plt.legend()
                    plt.xlabel('Cell Index')
                    plt.ylabel('BPS')
                    plt.title(f'Bits Per Spike - {dataset_name}')
                    log_dict[f'{prefix}/bps'] = wandb.Image(fig)
                    plt.close(fig)

                # 2. CCNORM plot (matching lines 172-196 of script)
                if 'ccnorm' in results:
                    rbar = results['ccnorm']['rbar']
                    rhat_bar = results['ccnorm']['rbarhat']

                    good_samples = np.sum(~np.isnan(rbar), 0)
                    good_units = np.where(good_samples == good_samples.max())[0]

                    NC = len(good_units)  # ALL good units, not limited
                    sx = int(np.sqrt(NC))
                    sy = int(np.ceil(NC / sx))
                    fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
                    for i in range(sx*sy):
                        if i >= NC:
                            axs.flatten()[i].axis('off')
                            continue
                        cc = good_units[i]
                        axs.flatten()[i].plot(rbar[:,cc], 'k')
                        axs.flatten()[i].plot(rhat_bar[:,cc], 'r')
                        axs.flatten()[i].set_title(f'{cc}')
                        axs.flatten()[i].axis('off')
                        axs.flatten()[i].set_title(f'{cc}')
                    axs.flatten()[i].set_xlim(32, 150)
                    plt.suptitle(f'CCNORM - {dataset_name}')
                    plt.tight_layout()
                    log_dict[f'{prefix}/ccnorm'] = wandb.Image(fig)
                    plt.close(fig)

                # 3. Saccade-triggered averages - ONLY backimage (matching lines 198-223 of script)
                if 'saccade' in results and 'backimage' in results['saccade']:
                    rbar = results['saccade']['backimage']['rbar']
                    rhat = results['saccade']['backimage']['rbarhat']
                    win = results['saccade']['backimage']['win']

                    N = rbar.shape[1]
                    sx = int(np.sqrt(N))
                    sy = int(np.ceil(N / sx))
                    fig_saccade, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
                    time_axis = np.arange(win[0], win[1])
                    for i in range(sx*sy):
                        if i >= N:
                            axs.flatten()[i].axis('off')
                            continue
                        axs.flatten()[i].plot(time_axis, rbar[:,i], 'k')
                        m = rbar[:,i].mean()
                        axs.flatten()[i].axhline(m, linestyle='--', color='k')
                        axs.flatten()[i].axvline(0, linestyle='--', color='k')
                        axs.flatten()[i].plot(time_axis, rhat[:,i], 'r')
                        axs.flatten()[i].plot([0, 10], [m, m], 'k', linewidth=2)
                        axs.flatten()[i].set_xlim(win[0], win[1]//2)
                        axs.flatten()[i].set_title(f'{i}')
                        axs.flatten()[i].axis('off')
                        axs.flatten()[i].set_title(f'{i}')
                    plt.suptitle(f'Saccade-triggered - backimage - {dataset_name}')
                    plt.tight_layout()
                    log_dict[f'{prefix}/saccade_backimage'] = wandb.Image(fig_saccade)
                    plt.close(fig_saccade)

                # 4. Spike-triggered averages (STAs) - ALL cells (matching lines 115-156 of script)
                if 'sta' in results:
                    sta_dict = results['sta']
                    N = len(sta_dict['peak_lag'])
                    num_lags = sta_dict['Z_STA_robs'].shape[0]
                    rf_pairs_full = []
                    rf_pairs = []

                    for cc in range(N):  # ALL cells, not limited
                        this_lag = sta_dict['peak_lag'][cc]
                        sta_robs = sta_dict['Z_STA_robs'][:,:,:,cc]
                        sta_rhat = sta_dict['Z_STA_rhat'][:,:,:,cc]
                        # zscore each
                        sta_robs = (sta_robs - sta_robs.mean((0,1))) / sta_robs.std((0,1))
                        sta_rhat = (sta_rhat - sta_rhat.mean((0,1))) / sta_rhat.std((0,1))
                        grid = make_grid(torch.concat([sta_robs, sta_rhat], 0).unsqueeze(1), nrow=num_lags, normalize=True, scale_each=False, padding=2, pad_value=1)
                        grid = 0.2989 * grid[0:1,:,:] + 0.5870 * grid[1:2,:,:] + 0.1140 * grid[2:3,:,:] # convert to grayscale
                        rf_pairs_full.append(grid)

                        # do the same for the peak lag
                        sta_robs = sta_dict['Z_STA_robs'][this_lag,:,:,cc]
                        sta_rhat = sta_dict['Z_STA_rhat'][this_lag,:,:,cc]
                        # zscore each
                        sta_robs = (sta_robs - sta_robs.mean()) / sta_robs.std()
                        sta_rhat = (sta_rhat - sta_rhat.mean()) / sta_rhat.std()
                        grid = torch.stack([sta_robs, sta_rhat], 0).unsqueeze(1)
                        grid = make_grid(grid, nrow=2, normalize=True, scale_each=False, padding=2, pad_value=1)
                        grid = 0.2989 * grid[0:1,:,:] + 0.5870 * grid[1:2,:,:] + 0.1140 * grid[2:3,:,:] # convert to grayscale
                        rf_pairs.append(grid)

                    # log the full spatio-temporal STAs for each Cell and model
                    log_grid_full = make_grid(torch.stack(rf_pairs_full), nrow=3, normalize=True, scale_each=True, padding=2, pad_value=1)
                    fig = plt.figure(figsize=(20, 20))
                    plt.imshow(log_grid_full.detach().cpu().permute(1, 2, 0).numpy())
                    plt.axis('off')
                    plt.title(f'STAs (full temporal) - {dataset_name}')
                    log_dict[f'{prefix}/sta_full'] = wandb.Image(fig)
                    plt.close(fig)

                    # log the peak lag STAs for each Cell and model
                    log_grid_peak_lag = make_grid(torch.stack(rf_pairs), nrow=int(np.sqrt(N)), normalize=True, scale_each=True, padding=2, pad_value=1)
                    fig = plt.figure(figsize=(20, 20))
                    plt.imshow(log_grid_peak_lag.detach().cpu().permute(1, 2, 0).numpy())
                    plt.axis('off')
                    plt.title(f'STAs (peak lag) - {dataset_name}')
                    log_dict[f'{prefix}/sta_peak'] = wandb.Image(fig)
                    plt.close(fig)

                # Log all results in a single call
                if log_dict:
                    pl_module.logger.experiment.log(log_dict, step=trainer.global_step)
                    print(f"  ✓ Logged {len(log_dict)} evaluation results to WandB")

            except Exception as e:
                print(f"  Error evaluating dataset {dataset_idx}: {e}")
                import traceback
                traceback.print_exc()

            # Clean up
            plt.close('all')
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during slow logging: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Restore training mode
            if was_training:
                pl_module.train()

        print(f"[Epoch {epoch}] Slow logging complete.\n", flush=True)

