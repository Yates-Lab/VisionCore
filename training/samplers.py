"""
Custom data samplers for distributed training with curriculum learning.
"""

import math
import torch
from torch.utils.data import Sampler


class ContrastWeightedSampler(Sampler):
    """
    Distributed sampler with contrast-weighted curriculum learning.

    This sampler implements a curriculum learning strategy where:
    - **Early training:** Emphasizes high-contrast samples (easier to learn)
    - **Late training:** Returns to unbiased uniform sampling

    The transition from contrast-weighted to uniform sampling happens
    gradually over `warmup_steps` training steps.

    Parameters
    ----------
    dataset : torch.utils.data.ConcatDataset
        Concatenated dataset of multiple tagged datasets
    contrast_scores_dict : dict
        Dictionary mapping dataset names to tensors of per-frame contrast scores.
        Contrast scores should be normalized (0-1 range).
    dataset_name_to_idx : dict
        Dictionary mapping dataset names to dataset indices
    num_replicas : int, optional
        Number of processes participating in distributed training.
        If None, uses world size from torch.distributed.
    rank : int, optional
        Rank of the current process. If None, uses rank from torch.distributed.
    shuffle : bool, optional
        Whether to shuffle the data (default: True)
    seed : int, optional
        Random seed for shuffling (default: 0)
    drop_last : bool, optional
        Whether to drop the last incomplete batch (default: False)
    warmup_steps : int, optional
        Number of training steps for curriculum warmup (default: 8000)

    Example
    -------
    >>> contrast_scores = {
    ...     'dataset1': torch.tensor([0.5, 0.8, 0.3, ...]),  # Per-frame contrasts
    ...     'dataset2': torch.tensor([0.6, 0.7, 0.9, ...]),
    ... }
    >>> sampler = ContrastWeightedSampler(
    ...     dataset=concat_dataset,
    ...     contrast_scores_dict=contrast_scores,
    ...     dataset_name_to_idx={'dataset1': 0, 'dataset2': 1},
    ...     warmup_steps=8000
    ... )
    >>> dataloader = DataLoader(dataset, sampler=sampler, ...)

    Notes
    -----
    - Requires torch.distributed to be available for distributed training
    - Contrast scores should be pre-computed and normalized
    - Use with CurriculumCallback to update step counter during training
    """

    def __init__(self, dataset, contrast_scores_dict, dataset_name_to_idx,
                 num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False, warmup_steps=8000):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.warmup_steps = warmup_steps

        # Store pre-computed normalized contrast scores
        self.normalized_contrasts = contrast_scores_dict

        # Map dataset names to indices
        self.name_to_idx = dataset_name_to_idx

        # Build contrast mapping from cached scores
        self.sample_contrasts = self._build_contrast_mapping_from_cache()

        # Calculate dataset sizes for distributed sampling
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _build_contrast_mapping_from_cache(self):
        """
        Build mapping from ConcatDataset indices to cached contrast scores.

        Returns
        -------
        torch.Tensor
            Tensor of contrast scores aligned with dataset indices
        """
        sample_contrasts = torch.zeros(len(self.dataset))

        # Get dataset cumulative sizes from ConcatDataset
        cumulative_sizes = self.dataset.cumulative_sizes

        for dataset_idx, cumulative_size in enumerate(cumulative_sizes):
            # Calculate start and end indices for this dataset
            dataset_start_idx = cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else 0
            dataset_end_idx = cumulative_size

            # Get dataset name from the Tag wrapper
            tag_dataset = self.dataset.datasets[dataset_idx]
            dataset_name = None
            for name, idx in self.name_to_idx.items():
                if idx == tag_dataset.idx:
                    dataset_name = name
                    break

            if dataset_name and dataset_name in self.normalized_contrasts:
                contrasts = self.normalized_contrasts[dataset_name]
                dataset_size = dataset_end_idx - dataset_start_idx

                # Vectorized assignment
                if len(contrasts) >= dataset_size:
                    sample_contrasts[dataset_start_idx:dataset_end_idx] = contrasts[:dataset_size]
                else:
                    # Handle size mismatch
                    sample_contrasts[dataset_start_idx:dataset_start_idx + len(contrasts)] = contrasts
                    sample_contrasts[dataset_start_idx + len(contrasts):dataset_end_idx] = 1.0
            else:
                # Fallback for missing dataset - use uniform weights
                sample_contrasts[dataset_start_idx:dataset_end_idx] = 1.0

        return sample_contrasts

    def _compute_weights_vectorized(self, step):
        """
        Compute curriculum weights using vectorized operations.

        Parameters
        ----------
        step : int
            Current training step

        Returns
        -------
        torch.Tensor
            Sampling weights for each sample in the dataset
        """
        if step >= self.warmup_steps:
            # Uniform sampling after warmup
            return torch.ones_like(self.sample_contrasts)
        else:
            # Gradually transition from contrast-weighted to uniform
            alpha = 0.5 + 0.5 * (step / self.warmup_steps)
            return torch.clamp(alpha * self.sample_contrasts, max=1.0)

    def set_epoch(self, epoch):
        """
        Set epoch for distributed training.

        Parameters
        ----------
        epoch : int
            Current epoch number
        """
        self.epoch = epoch

    def set_step(self, step):
        """
        Set current training step for curriculum learning.

        This should be called at each training step to update the
        curriculum schedule. Use CurriculumCallback to automate this.

        Parameters
        ----------
        step : int
            Current global training step
        """
        self.current_step = step

    def __iter__(self):
        """Generate indices for sampling."""
        # Get current step from trainer if available
        current_step = getattr(self, 'current_step', 0)

        if self.shuffle:
            # Generate deterministic random state
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if current_step >= self.warmup_steps:
                # Unbiased sampling after warmup
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                # Contrast-weighted sampling with vectorized computation
                weights = self._compute_weights_vectorized(current_step)

                # Sample with replacement using computed weights
                indices = torch.multinomial(
                    weights, len(self.dataset), replacement=True, generator=g
                ).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        """Return the number of samples for this rank."""
        return self.num_samples




class ByDatasetBatchSampler(Sampler):
    """
    BatchSampler that yields batches containing samples from exactly one dataset.

    Works on a ConcatDataset of Tag-wrapped datasets (each Tag carries .idx that
    matches name2idx[name]). Optionally applies a simple curriculum schedule that
    weights samples within each dataset by precomputed contrast scores, blending
    from contrast-weighted (early) to uniform (late) over warmup_steps.

    Parameters
    ----------
    cat : torch.utils.data.ConcatDataset
        Concatenated dataset built from Tag-wrapped per-dataset datasets
    name2idx : dict[str, int]
        Mapping from dataset name to dataset_idx used in Tag
    batch_size : int
        Number of samples per batch
    contrast_scores : dict[str, torch.Tensor] | None
        Optional per-dataset contrast score vectors (length == len(dataset))
    warmup_steps : int
        Number of steps to ramp alpha from 0.5 -> 1.0 (then uniform)
    shuffle : bool
        Whether to shuffle dataset choice and within-dataset sampling
    drop_last : bool
        Whether to drop last incomplete batch
    seed : int
        RNG seed
    """

    def __init__(self, cat, name2idx, batch_size, contrast_scores=None,
                 warmup_steps=8000, shuffle=True, drop_last=True, seed=0):
        super().__init__(cat)
        self.cat = cat
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.warmup_steps = int(warmup_steps)
        self.contrast_scores = contrast_scores or {}

        # Reverse map idx->name to resolve contrast by dataset
        self.idx2name = {v: k for k, v in name2idx.items()}

        # Build per-subdataset [start, end) global index ranges
        self._ranges = []  # list of (start, end, dataset_idx, dataset_name)
        cumulative = self.cat.cumulative_sizes
        start = 0
        for j, end in enumerate(cumulative):
            tag_ds = self.cat.datasets[j]
            ds_idx = getattr(tag_ds, 'idx', None)
            ds_name = self.idx2name.get(ds_idx, None)
            self._ranges.append((start, end, ds_idx, ds_name))
            start = end

        # Pre-cache per-subdataset contrast vectors aligned to local indices
        self._local_contrasts = []
        for (start, end, ds_idx, ds_name) in self._ranges:
            size = end - start
            vec = None
            if ds_name is not None and ds_name in self.contrast_scores:
                cs = self.contrast_scores[ds_name]
                if isinstance(cs, torch.Tensor) and cs.numel() >= size:
                    vec = cs[:size].to(dtype=torch.float32)
            if vec is None:
                vec = torch.ones(size, dtype=torch.float32)
            self._local_contrasts.append(vec)

        # Dataset-level sizes and probabilities for choosing which dataset to draw next batch from
        self._sizes = torch.tensor([end - start for (start, end, *_rest) in self._ranges], dtype=torch.float64)
        total = float(self._sizes.sum()) if len(self._sizes) > 0 else 1.0
        self._dataset_probs = (self._sizes / total).to(dtype=torch.float64)

        # Internal step counter for curriculum; updated by CurriculumCallback via set_step
        self._step = 0

    def set_step(self, step: int):
        self._step = int(step)

    def _alpha(self) -> float:
        # Blend coefficient from 0.5 to 1.0 across warmup_steps
        if self._step >= self.warmup_steps:
            return 1.0
        return 0.5 + 0.5 * (float(self._step) / float(self.warmup_steps))

    def __iter__(self):
        # RNG for reproducibility across epochs (no epoch hook here; simple seed)
        g = torch.Generator()
        g.manual_seed(self.seed + self._step)

        # Choose dataset order for this epoch (size-proportional or uniform when not shuffle)
        dataset_indices = torch.arange(len(self._ranges))
        if self.shuffle and len(self._ranges) > 0:
            # Sample dataset indices with replacement proportional to dataset sizes
            # Number of batches approximated by floor(total_size / batch_size)
            num_batches = len(self)
            probs = self._dataset_probs
            ds_seq = torch.multinomial(probs, num_batches, replacement=True, generator=g)
        else:
            # Round-robin across datasets
            num_batches = len(self)
            if len(dataset_indices) == 0:
                ds_seq = torch.empty(0, dtype=torch.long)
            else:
                ds_seq = dataset_indices.repeat((num_batches + len(dataset_indices) - 1) // len(dataset_indices))[:num_batches]

        alpha = self._alpha()

        for k in range(len(ds_seq)):
            dsj = int(ds_seq[k].item()) if ds_seq.numel() > 0 else 0
            start, end, _ds_idx, _ds_name = self._ranges[dsj]
            size = end - start
            if size == 0:
                continue
            local_weights = self._local_contrasts[dsj]
            # Blend toward uniform via clamping
            weights = torch.clamp(alpha * local_weights, max=1.0)

            if self.shuffle:
                # Sample local indices (with replacement when needed)
                if self.batch_size <= size:
                    # Weighted without replacement is tricky; approximate by shuffle of multinomial sample
                    sample = torch.multinomial(weights, self.batch_size, replacement=False, generator=g)
                else:
                    sample = torch.multinomial(weights, self.batch_size, replacement=True, generator=g)
            else:
                sample = torch.arange(min(self.batch_size, size))

            # Map to global indices and yield
            global_idx = (sample + start).tolist()
            yield global_idx

    def __len__(self) -> int:
        total = sum(end - start for (start, end, *_rest) in self._ranges)
        if self.drop_last:
            return max(total // self.batch_size, 0)
        else:
            # ceil division
            return (total + self.batch_size - 1) // self.batch_size
