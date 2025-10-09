"""
PyTorch Lightning DataModule for multi-dataset training.
"""

import os
import time
import contextlib
from pathlib import Path
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from training.utils import Float32View, group_collate
from training.samplers import ContrastWeightedSampler


class MultiDatasetDM(pl.LightningDataModule):
    """
    Lightning DataModule for loading and managing multiple neural datasets.
    
    This DataModule:
    - Loads multiple datasets from YAML configuration files
    - Supports curriculum learning with contrast-weighted sampling
    - Handles distributed training with proper data sharding
    - Provides train and validation dataloaders
    
    Parameters
    ----------
    cfg_dir : str
        Directory containing dataset YAML configuration files
    max_ds : int
        Maximum number of datasets to load
    batch : int
        Batch size per GPU
    workers : int
        Number of dataloader workers
    steps_per_epoch : int
        Number of training steps per epoch (for curriculum scheduling)
    enable_curriculum : bool, optional
        Whether to enable curriculum learning with contrast-weighted sampling
        (default: False)
        
    Example
    -------
    >>> dm = MultiDatasetDM(
    ...     cfg_dir='configs/datasets',
    ...     max_ds=20,
    ...     batch=32,
    ...     workers=4,
    ...     steps_per_epoch=1000,
    ...     enable_curriculum=True
    ... )
    >>> trainer = pl.Trainer(...)
    >>> trainer.fit(model, datamodule=dm)
    
    Attributes
    ----------
    names : list of str
        Names of loaded datasets
    cfgs : list of dict
        Dataset configurations
    train_dsets : dict
        Training datasets keyed by name
    val_dsets : dict
        Validation datasets keyed by name
    name2idx : dict
        Mapping from dataset name to index
    contrast_scores : dict, optional
        Pre-computed contrast scores for curriculum learning
    """
    
    def __init__(self, cfg_dir: str, max_ds: int, batch: int,
                 workers: int, steps_per_epoch: int, enable_curriculum: bool = False):
        super().__init__()
        self.cfg_dir = cfg_dir
        self.max_ds = max_ds
        self.batch = batch
        self.workers = workers
        self.spe = steps_per_epoch
        self.enable_curriculum = enable_curriculum
        self.contrast_scores = None
        self.name2idx = None

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets and prepare for training/validation.
        
        Parameters
        ----------
        stage : str, optional
            Current stage ('fit', 'validate', 'test', or 'predict')
        """
        from models.config_loader import load_dataset_configs
        from DataYatesV1.utils.data.loading import remove_pixel_norm
        from DataYatesV1.utils.data import prepare_data

        # Load dataset configurations
        yaml_files = sorted([
            f for f in os.listdir(self.cfg_dir)
            if f.endswith(".yaml") and "base" not in f
        ])[:self.max_ds]
        
        self.names = [Path(f).stem for f in yaml_files]
        self.cfgs = load_dataset_configs(yaml_files, self.cfg_dir)

        # Prepare datasets
        self.train_dsets, self.val_dsets, self.name2idx = {}, {}, {}
        for idx, (cfg, name) in enumerate(zip(self.cfgs, self.names)):
            cfg["_dataset_name"] = name
            cfg, norm_removed = remove_pixel_norm(cfg)
            
            # Suppress output from prepare_data
            with open(os.devnull, "w") as _null, \
                 contextlib.redirect_stdout(_null), \
                 contextlib.redirect_stderr(_null):
                tr, va, _ = prepare_data(cfg, strict=False)
            
            self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
            self.val_dsets[name] = Float32View(va, norm_removed, float16=False)
            self.name2idx[name] = idx
            
        print(f"✓ loaded {len(self.train_dsets)} datasets")

        # Precompute contrast scores for curriculum learning
        if self.enable_curriculum:
            self._precompute_contrast_scores()

    def _precompute_contrast_scores(self):
        """
        Precompute and cache contrast scores for all datasets.
        
        Contrast scores are computed as the standard deviation of pixel
        values across spatial dimensions for each frame. Scores are
        normalized so that the global median contrast equals 1.0.
        """
        print("Computing contrast scores for curriculum learning...")
        start_time = time.time()

        self.contrast_scores = {}
        all_contrasts_for_median = []
        total_frames = 0

        # Step 1: Compute raw contrast scores for all datasets
        for dataset_name, train_dataset in self.train_dsets.items():
            try:
                # Access the CombinedEmbeddedDataset
                combined_dataset = train_dataset.base

                # Compute contrast for all dsets within this dataset
                dataset_contrasts = []

                for dset_idx, dset in enumerate(combined_dataset.dsets):
                    raw_stim = dset['stim']

                    # Fast per-frame contrast computation (vectorized)
                    frame_contrasts = raw_stim.to(torch.float16).std(dim=(1, 2, 3))
                    dataset_contrasts.append(frame_contrasts)
                    total_frames += len(frame_contrasts)

                # Concatenate all contrasts for this dataset
                if dataset_contrasts:
                    dataset_contrasts = torch.cat(dataset_contrasts)
                    self.contrast_scores[dataset_name] = dataset_contrasts
                    all_contrasts_for_median.append(dataset_contrasts)

                    print(f"  {dataset_name}: {len(dataset_contrasts)} frames, "
                          f"range [{dataset_contrasts.min():.3f}, {dataset_contrasts.max():.3f}]")
                else:
                    # Fallback if no dsets found
                    dataset_size = len(train_dataset)
                    fallback_contrasts = torch.ones(dataset_size)
                    self.contrast_scores[dataset_name] = fallback_contrasts
                    all_contrasts_for_median.append(fallback_contrasts)

            except Exception as e:
                print(f"  Warning: Failed to compute contrast for {dataset_name}: {e}")
                # Fallback to uniform contrasts
                dataset_size = len(train_dataset)
                fallback_contrasts = torch.ones(dataset_size)
                self.contrast_scores[dataset_name] = fallback_contrasts
                all_contrasts_for_median.append(fallback_contrasts)

        # Step 2: Compute global median and normalize
        all_contrasts = torch.cat(all_contrasts_for_median)
        global_median = all_contrasts.median()

        # Normalize so median = 1.0
        for dataset_name in self.contrast_scores:
            self.contrast_scores[dataset_name] = self.contrast_scores[dataset_name] / global_median

        elapsed = time.time() - start_time
        print(f"✓ Computed contrast for {total_frames:,} frames in {elapsed:.2f} seconds")
        print(f"✓ Global median contrast: {global_median:.3f}, normalized to 1.0")

    def _mk_loader(self, dsets: Dict[str, Dataset], shuffle: bool):
        """
        Create a DataLoader for the given datasets.
        
        Parameters
        ----------
        dsets : dict
            Dictionary of datasets keyed by name
        shuffle : bool
            Whether to shuffle the data
            
        Returns
        -------
        DataLoader
            Configured DataLoader with appropriate sampler
        """
        # Tag each dataset with its index
        class Tag(Dataset):
            def __init__(self, ds, idx):
                self.ds = ds
                self.idx = idx
                
            def __len__(self):
                return len(self.ds)
            
            def __getitem__(self, i):
                it = self.ds[i]
                it["dataset_idx"] = self.idx
                return it

        # Concatenate all tagged datasets
        tagd = [Tag(ds, self.name2idx[n]) for n, ds in dsets.items()]
        cat = torch.utils.data.ConcatDataset(tagd)

        # Create appropriate sampler for distributed training
        if torch.distributed.is_initialized():
            if shuffle and self.enable_curriculum and self.contrast_scores is not None:
                # Use contrast-weighted sampler for training
                sampler = ContrastWeightedSampler(
                    cat,
                    self.contrast_scores,
                    self.name2idx,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True,
                    drop_last=True,
                    warmup_steps=8000
                )
            else:
                # Use standard distributed sampler
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    cat,
                    shuffle=shuffle,
                    drop_last=True,
                )
        else:
            # Single-GPU debug run
            sampler = None

        return DataLoader(
            cat,
            batch_size=self.batch,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            collate_fn=group_collate
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._mk_loader(self.train_dsets, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._mk_loader(self.val_dsets, shuffle=True)

