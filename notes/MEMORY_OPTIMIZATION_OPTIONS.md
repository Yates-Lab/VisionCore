# Memory Optimization Options for Multi-Dataset Training

## Current System Analysis

### Current Setup
- **CPU RAM:** 504 GB total
- **Current usage:** 226 GB for one multi-dataset fit (uint8)
- **Headroom:** ~278 GB available
- **Disk space:** 
  - `/` (root): 3.7 TB total, 3.3 TB used, **258 GB free** (93% full) ⚠️
  - `/mnt/ssd`: 7.3 TB total, 5.5 TB used, **1.4 TB free** (80% full)

### Current Data Flow
```
1. Load datasets to CPU RAM as uint8 (226 GB)
   ↓
2. Float32View wrapper (no memory cost - lazy conversion)
   ↓
3. DataLoader fetches batch
   ↓
4. cast_stim() converts uint8 → float32 on GPU
   ↓
5. Training on GPU
```

### Memory Calculation
- **uint8:** 1 byte per pixel
- **float32:** 4 bytes per pixel (4x larger)
- **bfloat16:** 2 bytes per pixel (2x larger)

**If you converted all data to float32 in RAM:**
- Current: 226 GB (uint8)
- Float32: 226 GB × 4 = **904 GB** ❌ Exceeds 504 GB RAM
- BFloat16: 226 GB × 2 = **452 GB** ✅ Fits in RAM (with 52 GB headroom)

---

## Option 1: Use BFloat16 in RAM (Recommended)

### Description
Store preprocessed data as **bfloat16** instead of uint8 in CPU RAM.

### Pros
✅ **Fits in RAM:** 452 GB < 504 GB (52 GB headroom)  
✅ **Apply transforms once:** No repeated computation  
✅ **Fast:** No disk I/O, pure RAM access  
✅ **Better precision:** bfloat16 preserves transform quality  
✅ **GPU-friendly:** Modern GPUs love bfloat16  
✅ **Simple implementation:** Just change dtype in prepare_data  

### Cons
⚠️ **Tight on memory:** Only 52 GB headroom (11% margin)  
⚠️ **Risk:** Adding more datasets could exceed RAM  
⚠️ **No safety net:** If you run out of RAM, training crashes  

### Implementation
```python
# In prepare_data or dataset loading:
# After applying transforms, cast to bfloat16
dset.cast(torch.bfloat16, target_keys=['stim'], protect_keys=['robs'])

# In Float32View, skip conversion if already float:
def __getitem__(self, idx):
    it = self.base[idx]
    # stim is already bfloat16, just normalize if needed
    if self.norm_removed and it["stim"].dtype == torch.uint8:
        it["stim"] = cast_stim(it["stim"], True, torch.bfloat16)
    # Otherwise it's already preprocessed
    return it
```

### Estimated Memory
- **Current:** 226 GB (uint8, no transforms)
- **With bfloat16:** ~452 GB (with transforms applied)
- **Headroom:** 52 GB (11%)

---

## Option 2: Memory-Mapped Files (Disk-Backed)

### Description
Store preprocessed data in **numpy memmap files** on `/mnt/ssd` (1.4 TB free).

### Pros
✅ **Unlimited capacity:** Can handle any dataset size  
✅ **Apply transforms once:** Preprocess and save  
✅ **Flexible:** Can use float32 without RAM constraints  
✅ **Persistent:** Reuse preprocessed data across runs  
✅ **Safety:** Won't crash from OOM  

### Cons
❌ **Slower:** Disk I/O bottleneck (even on SSD)  
❌ **Disk space:** Needs ~904 GB for float32 (or 452 GB for bfloat16)  
❌ **Complexity:** Need to manage cache files  
❌ **Cleanup:** Need to delete old cache files  
❌ **First run slow:** Must preprocess and write to disk  

### Performance Impact
**SSD Sequential Read Speed:** ~3-5 GB/s (typical NVMe)
- **Batch size 256, float32:** ~100 MB per batch
- **Read time:** ~20-30 ms per batch
- **Training time:** ~50-100 ms per batch (GPU forward/backward)
- **Overhead:** ~20-40% slower training

**Mitigation:**
- Use `num_workers > 0` to prefetch batches
- Use bfloat16 instead of float32 (half the I/O)
- Pin memory for faster GPU transfer

### Implementation
```python
import numpy as np
from pathlib import Path

class MemmapDataset(Dataset):
    def __init__(self, memmap_dir, dataset_name):
        self.memmap_dir = Path(memmap_dir)
        self.dataset_name = dataset_name
        
        # Load metadata
        meta = np.load(self.memmap_dir / f"{dataset_name}_meta.npz")
        self.length = int(meta['length'])
        self.stim_shape = tuple(meta['stim_shape'])
        self.robs_shape = tuple(meta['robs_shape'])
        
        # Open memmap files (read-only)
        self.stim = np.memmap(
            self.memmap_dir / f"{dataset_name}_stim.dat",
            dtype='float16',  # or 'float32'
            mode='r',
            shape=(self.length, *self.stim_shape)
        )
        self.robs = np.memmap(
            self.memmap_dir / f"{dataset_name}_robs.dat",
            dtype='float32',
            mode='r',
            shape=(self.length, *self.robs_shape)
        )
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'stim': torch.from_numpy(self.stim[idx].copy()),
            'robs': torch.from_numpy(self.robs[idx].copy()),
            'dataset_idx': self.dataset_idx
        }
```

### Disk Space Calculation
**For 226 GB uint8 dataset:**
- **Float32:** 226 GB × 4 = 904 GB
- **BFloat16:** 226 GB × 2 = 452 GB
- **Available:** 1.4 TB on `/mnt/ssd` ✅

**Safety guardrails:**
```python
def check_disk_space(path, required_gb):
    stat = os.statvfs(path)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < required_gb * 1.2:  # 20% safety margin
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free, need {required_gb:.1f} GB")
```

---

## Option 3: Hybrid Approach (Smart Caching)

### Description
Keep frequently accessed data in RAM, less frequent data on disk.

### Pros
✅ **Best of both worlds:** Fast for common data, scalable for rare data  
✅ **Adaptive:** Automatically manages memory  
✅ **Efficient:** Only cache what's needed  

### Cons
❌ **Complex:** Requires LRU cache implementation  
❌ **Unpredictable:** Performance varies by access pattern  
❌ **Overhead:** Cache management adds CPU cost  

### Implementation
Use PyTorch's `ConcatDataset` with mixed backends:
- Hot datasets (frequently sampled): Keep in RAM as bfloat16
- Cold datasets (rarely sampled): Use memmap

---

## Option 4: On-the-Fly Transforms (Current + Optimized)

### Description
Keep current uint8 storage, but optimize transform computation.

### Pros
✅ **Minimal memory:** 226 GB (current)  
✅ **No disk I/O:** Pure RAM access  
✅ **Flexible:** Can change transforms without reprocessing  

### Cons
❌ **Repeated computation:** Transforms applied every epoch  
❌ **CPU bottleneck:** May slow down training  
❌ **Limited transforms:** Can't do expensive preprocessing  

### Optimization
```python
# Use torch.compile on transform pipeline
transform_pipeline = torch.compile(make_pipeline(transforms))

# Use multiple workers to parallelize
DataLoader(..., num_workers=16, prefetch_factor=4)

# Cache transformed batches (if deterministic)
from functools import lru_cache
```

---

## Option 5: Compressed Storage (Advanced)

### Description
Store data compressed (e.g., PNG, JPEG-LS, or custom compression).

### Pros
✅ **Smaller footprint:** 2-10x compression possible  
✅ **Fits more in RAM:** Could fit float32 compressed  

### Cons
❌ **Decompression overhead:** CPU cost per batch  
❌ **Lossy risk:** JPEG artifacts (use lossless only)  
❌ **Complex:** Need compression/decompression pipeline  
❌ **Not worth it:** Modern compression is slow  

---

## Recommendation Matrix

| Scenario | Best Option | Why |
|----------|-------------|-----|
| **Current setup works** | Option 4 (Optimized on-the-fly) | No changes needed, just optimize |
| **Need transforms, have RAM** | Option 1 (BFloat16 in RAM) | Fast, simple, fits in 504 GB |
| **Need transforms, tight RAM** | Option 2 (Memmap) | Safe, scalable, 20-40% slower |
| **Many datasets, varied access** | Option 3 (Hybrid) | Adaptive, but complex |
| **Extreme memory pressure** | Option 5 (Compressed) | Last resort, high CPU cost |

---

## My Recommendation: **Option 1 (BFloat16 in RAM)**

### Why?
1. **Fits in your RAM:** 452 GB < 504 GB ✅
2. **Fast:** No disk I/O, pure RAM speed
3. **Simple:** Minimal code changes
4. **Quality:** BFloat16 is good enough for neural data
5. **GPU-friendly:** Modern GPUs prefer bfloat16

### Implementation Plan

#### Step 1: Modify `prepare_data` to save as bfloat16
```python
# In models/data/loading.py or wherever prepare_data is
def prepare_data(dataset_config, strict=True, use_bfloat16=True):
    # ... existing code ...
    
    # After applying transforms, cast to bfloat16
    if use_bfloat16:
        train_dset.cast(torch.bfloat16, target_keys=['stim'])
        val_dset.cast(torch.bfloat16, target_keys=['stim'])
    
    return train_dset, val_dset, dataset_config
```

#### Step 2: Update `Float32View` to handle bfloat16
```python
def __getitem__(self, idx):
    it = self.base[idx]
    dtype = torch.bfloat16 if self.float16 else torch.float32
    
    # If stim is already float, just convert dtype
    if it["stim"].dtype in [torch.float32, torch.bfloat16, torch.float16]:
        it["stim"] = it["stim"].to(dtype)
    else:
        # uint8 case - apply normalization
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=dtype)
    
    it["robs"] = it["robs"].to(dtype)
    if "behavior" in it:
        it["behavior"] = it["behavior"].to(dtype)
    
    return it
```

#### Step 3: Monitor memory usage
```python
import psutil
process = psutil.Process()
print(f"RAM usage: {process.memory_info().rss / 1024**3:.1f} GB")
```

### Fallback Plan
If you exceed RAM:
1. **Reduce datasets:** Train on fewer datasets at once
2. **Use memmap:** Fall back to Option 2
3. **Downsample:** Reduce spatial/temporal resolution

---

## Next Steps

1. **Test with one dataset:** Verify bfloat16 memory usage
2. **Scale up gradually:** Add datasets and monitor RAM
3. **Benchmark:** Compare training speed (should be same or faster)
4. **Implement safety:** Add RAM monitoring and warnings

Want me to implement Option 1 (BFloat16 in RAM)?

