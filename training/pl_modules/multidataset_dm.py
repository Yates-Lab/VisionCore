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
from training.samplers import ContrastWeightedSampler, ByDatasetBatchSampler


class MultiDatasetDM(pl.LightningDataModule):
    """
    Lightning DataModule for loading and managing multiple neural datasets.

    This DataModule:
    - Loads multiple datasets from a parent config that specifies sessions
    - Supports curriculum learning with contrast-weighted sampling
    - Handles distributed training with proper data sharding
    - Provides train and validation dataloaders

    Parameters
    ----------
    cfg_dir : str
        Path to parent dataset configuration YAML file (specifies sessions to load)
    max_ds : int
        Maximum number of datasets/sessions to load
    batch : int
        Batch size per GPU
    workers : int
        Number of dataloader workers
    steps_per_epoch : int
        Number of training steps per epoch (for curriculum scheduling)
    enable_curriculum : bool, optional
        Whether to enable curriculum learning with contrast-weighted sampling
        (default: False)
    dset_dtype : str, optional
        Dataset storage dtype in CPU RAM (default: 'uint8')
        - 'uint8': Store as uint8, normalize on GPU (only supports pixelnorm)
        - 'bfloat16': Apply all transforms, store as bfloat16 (2x memory)
        - 'float32': Apply all transforms, store as float32 (4x memory)

    Example
    -------
    >>> dm = MultiDatasetDM(
    ...     cfg_dir='experiments/dataset_configs/multi_basic_120_backimage_all.yaml',
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
        Names of loaded datasets (session names)
    cfgs : list of dict
        Dataset configurations (merged from parent + session configs)
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
                 workers: int, steps_per_epoch: int, enable_curriculum: bool = False,
                 dset_dtype: str = 'uint8', homogeneous_batches: bool = False,
                 persistent_workers: Optional[bool] = None, prefetch_factor: Optional[int] = None):
        super().__init__()
        self.cfg_dir = cfg_dir
        self.max_ds = max_ds
        self.batch = batch
        self.workers = workers
        self.spe = steps_per_epoch
        self.enable_curriculum = enable_curriculum
        self.dset_dtype = dset_dtype
        self.contrast_scores = None
        self.name2idx = None
        self.homogeneous_batches = bool(homogeneous_batches)

        # Loader performance knobs (sane defaults for single-GPU throughput)
        if persistent_workers is None:
            self.persistent_workers = (self.workers > 0)
        else:
            self.persistent_workers = bool(persistent_workers)
        if prefetch_factor is None:
            self.prefetch_factor = 4 if self.persistent_workers else None
        else:
            self.prefetch_factor = int(prefetch_factor)

        # Validate dset_dtype
        if dset_dtype not in ['uint8', 'bfloat16', 'float32']:
            raise ValueError(f"dset_dtype must be 'uint8', 'bfloat16', or 'float32', got '{dset_dtype}'")

    def _check_for_non_pixelnorm_transforms(self, cfg: Dict) -> bool:
        """
        Check if config has transforms other than pixelnorm.

        Parameters
        ----------
        cfg : dict
            Dataset configuration

        Returns
        -------
        bool
            True if there are transforms other than pixelnorm
        """
        if 'transforms' not in cfg:
            return False

        transforms = cfg['transforms']
        for transform_key, transform_spec in transforms.items():
            if 'ops' in transform_spec:
                for op in transform_spec['ops']:
                    if isinstance(op, dict):
                        # Check if this op is NOT pixelnorm
                        if 'pixelnorm' not in op:
                            return True
                    else:
                        # Non-dict ops are not pixelnorm
                        return True
        return False

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
        from models.data import prepare_data

        # Load dataset configurations from parent config
        # cfg_dir should now point to a parent config file (e.g., multi_basic_120_backimage_all.yaml)
        self.cfgs = load_dataset_configs(self.cfg_dir)

        # Limit to max_ds datasets
        self.cfgs = self.cfgs[:self.max_ds]

        # Extract dataset names from session names
        self.names = [cfg['session'] for cfg in self.cfgs]

        # Prepare datasets
        self.train_dsets, self.val_dsets, self.name2idx = {}, {}, {}
        for idx, (cfg, name) in enumerate(zip(self.cfgs, self.names)):
            print(f"Processing dataset {idx+1}/{len(self.cfgs)}: {name}")
            cfg["_dataset_name"] = name

            if self.dset_dtype == 'uint8':
                # Path 1: uint8 storage (current behavior)
                # Check for non-pixelnorm transforms
                if self._check_for_non_pixelnorm_transforms(cfg):
                    raise ValueError(
                        f"Dataset '{name}' has transforms other than pixelnorm, "
                        f"but dset_dtype='uint8' only supports pixelnorm. "
                        f"Use dset_dtype='bfloat16' or 'float32' to enable other transforms."
                    )

                # Remove pixelnorm from config
                cfg, norm_removed = remove_pixel_norm(cfg)

                # Suppress output from prepare_data
                with open(os.devnull, "w") as _null, \
                     contextlib.redirect_stdout(_null), \
                     contextlib.redirect_stderr(_null):
                    tr, va, _ = prepare_data(cfg, strict=False)

                # Wrap with Float32View for on-the-fly normalization
                self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
                self.val_dsets[name] = Float32View(va, norm_removed, float16=False)

            else:
                # Path 2: bfloat16 or float32 storage (new behavior)
                # Keep all transforms including pixelnorm

                # Suppress output from prepare_data
                with open(os.devnull, "w") as _null, \
                     contextlib.redirect_stdout(_null), \
                     contextlib.redirect_stderr(_null):
                    tr, va, _ = prepare_data(cfg, strict=False)

                # Cast to requested dtype
                dtype_map = {
                    'bfloat16': torch.bfloat16,
                    'float32': torch.float32
                }
                target_dtype = dtype_map[self.dset_dtype]

                tr.cast(target_dtype, target_keys=['stim', 'robs', 'dfs', 'behavior'])
                va.cast(target_dtype, target_keys=['stim', 'robs', 'dfs', 'behavior'])

                # Store directly without Float32View wrapper
                self.train_dsets[name] = tr
                self.val_dsets[name] = va

            self.name2idx[name] = idx

        print(f"✓ loaded {len(self.train_dsets)} datasets (dtype: {self.dset_dtype})")

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
        print(f"Computed contrast for {total_frames:,} frames in {elapsed:.2f} seconds")
        print(f"Global median contrast: {global_median:.3f}, normalized to 1.0")

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

        # Distributed path (unchanged)
        if torch.distributed.is_initialized():
            if shuffle and self.enable_curriculum and self.contrast_scores is not None:
                from torch.utils.data.distributed import DistributedSampler
                # Keep DistributedSampler for sharding; curriculum weighting handled inside ContrastWeightedSampler
                sampler = ContrastWeightedSampler(
                    cat,
                    self.contrast_scores,
                    self.name2idx,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True,
                    drop_last=True,
                    warmup_steps=8000,
                )
                return DataLoader(
                    cat,
                    batch_size=self.batch,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=self.persistent_workers,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=group_collate,
                )
            else:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    cat,
                    shuffle=shuffle,
                    drop_last=True,
                )
                return DataLoader(
                    cat,
                    batch_size=self.batch,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=self.persistent_workers,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=group_collate,
                )

        # Single-GPU path
        if self.homogeneous_batches:
            # Homogeneous batches per dataset; enable curriculum if available
            batch_sampler = ByDatasetBatchSampler(
                cat,
                self.name2idx,
                self.batch,
                contrast_scores=(self.contrast_scores if (shuffle and self.enable_curriculum and self.contrast_scores is not None) else None),
                warmup_steps=8000,
                shuffle=shuffle,
                drop_last=True,
                seed=0,
            )
            return DataLoader(
                cat,
                batch_sampler=batch_sampler,
                num_workers=self.workers,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                collate_fn=None,
            )
        else:
            # Mixed batches (legacy behavior), no curriculum on 1 GPU
            return DataLoader(
                cat,
                batch_size=self.batch,
                sampler=None,
                shuffle=shuffle,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                collate_fn=group_collate,
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._mk_loader(self.train_dsets, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._mk_loader(self.val_dsets, shuffle=False)

