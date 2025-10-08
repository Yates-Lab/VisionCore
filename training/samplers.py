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

