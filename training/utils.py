"""
Utility functions and classes for data loading and preprocessing.
"""

from collections import defaultdict
import torch
from torch.utils.data import Dataset


def cast_stim(x: torch.Tensor, norm_removed: bool, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Cast stimulus tensor to specified dtype and apply normalization if needed.
    
    Parameters
    ----------
    x : torch.Tensor
        Input stimulus tensor (typically uint8 in range [0, 255])
    norm_removed : bool
        Whether pixel normalization was removed during preprocessing.
        If True, applies normalization: (x - 127) / 255.0
        If False, assumes x is already normalized
    dtype : torch.dtype, optional
        Target dtype for the tensor (default: torch.float32)
        
    Returns
    -------
    torch.Tensor
        Normalized stimulus tensor in the specified dtype
        
    Example
    -------
    >>> stim = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    >>> normalized = cast_stim(stim, norm_removed=True, dtype=torch.float32)
    >>> normalized.min(), normalized.max()
    (tensor(-0.498), tensor(0.502))
    """
    x = x.to(dtype)
    return (x - 127) / 255.0 if norm_removed else x


class Float32View(Dataset):
    """
    Dataset wrapper that casts data to float32 or bfloat16 on-the-fly.
    
    This wrapper applies dtype conversion and normalization to stimulus,
    response, and behavior data without modifying the underlying dataset.
    
    Parameters
    ----------
    base : torch.utils.data.Dataset
        Base dataset to wrap
    norm_removed : bool
        Whether pixel normalization was removed (passed to cast_stim)
    float16 : bool, optional
        If True, use bfloat16 instead of float32 (default: False)
        
    Example
    -------
    >>> base_dataset = MyDataset(...)
    >>> float_dataset = Float32View(base_dataset, norm_removed=True, float16=False)
    >>> sample = float_dataset[0]
    >>> sample['stim'].dtype
    torch.float32
    
    Notes
    -----
    - Converts 'stim', 'robs', and 'behavior' (if present) to the target dtype
    - Uses bfloat16 if float16=True, otherwise uses float32
    - Applies normalization to 'stim' based on norm_removed flag
    """
    
    def __init__(self, base, norm_removed: bool, float16: bool = False):
        self.base = base
        self.norm_removed = norm_removed
        self.float16 = float16
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        """Get item and cast to target dtype."""
        it = self.base[idx]
        dtype = torch.bfloat16 if self.float16 else torch.float32
        
        # Cast stimulus with normalization
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=dtype)
        
        # Cast responses
        it["robs"] = it["robs"].to(dtype)
        
        # Cast behavior if present
        if "behavior" in it:
            it["behavior"] = it["behavior"].to(dtype)
            
        return it


def group_collate(batch):
    """
    Collate function that groups samples by dataset_idx.
    
    This is used for multi-dataset training where each batch may contain
    samples from multiple datasets. Samples are grouped by their dataset_idx
    and collated separately for each group.
    
    Parameters
    ----------
    batch : list of dict
        List of samples, each containing a 'dataset_idx' key
        
    Returns
    -------
    list of dict
        List of collated batches, one per unique dataset_idx
        
    Example
    -------
    >>> batch = [
    ...     {'dataset_idx': 0, 'stim': tensor1, 'robs': tensor2},
    ...     {'dataset_idx': 1, 'stim': tensor3, 'robs': tensor4},
    ...     {'dataset_idx': 0, 'stim': tensor5, 'robs': tensor6},
    ... ]
    >>> collated = group_collate(batch)
    >>> len(collated)  # Two groups (dataset_idx 0 and 1)
    2
    
    Notes
    -----
    - Uses PyTorch's default_collate for each group
    - Preserves dataset_idx information in the collated batches
    - Useful with multi-dataset dataloaders
    """
    import torch.utils.data._utils.collate as _dc
    
    # Group samples by dataset_idx
    groups = defaultdict(list)
    for sample in batch:
        groups[sample["dataset_idx"]].append(sample)
    
    # Collate each group separately
    return [_dc.default_collate(group_samples) for group_samples in groups.values()]

