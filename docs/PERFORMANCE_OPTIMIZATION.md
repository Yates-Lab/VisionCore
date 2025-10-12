# Performance Optimization Guide for VisionCore Training

## Critical Performance Issues and Fixes

### 1. **DataLoader Worker Persistence** ⚠️ CRITICAL
**Problem:** `persistent_workers=False` causes workers to be killed and respawned after every epoch, leading to:
- Memory leaks accumulating across runs
- Massive overhead from process creation/destruction  
- Workers reloading all data every epoch
- Catastrophic slowdown after multiple runs without machine restart

**Fix:** Set `persistent_workers=True` when using multiple workers
```python
# training/pl_modules/multidataset_dm.py
persistent_workers=True if self.workers > 0 else False
```

**Impact:** Prevents 100x slowdown after multiple training runs

---

### 2. **Avoid hasattr() in Forward Pass** ⚠️ CRITICAL
**Problem:** `hasattr()` calls in model forward pass are extremely slow in DDP

**Bad:**
```python
def forward(self, x):
    if hasattr(self.recurrent, 'set_modulator'):  # SLOW!
        self.recurrent.set_modulator(self.modulator)
```

**Good:**
```python
def __init__(self):
    # Cache the check once
    self._recurrent_needs_modulator = hasattr(self.recurrent, 'set_modulator')

def forward(self, x):
    if self._recurrent_needs_modulator:  # FAST!
        self.recurrent.set_modulator(self.modulator)
```

**Impact:** Eliminates per-step overhead in hot path

---

### 3. **Optimize Tensor Device Transfer**
**Problem:** Dict comprehensions with `isinstance()` checks are slow

**Bad:**
```python
b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
```

**Good:**
```python
for k in b:
    if torch.is_tensor(b[k]):
        b[k] = b[k].to(device, non_blocking=True)
```

**Why:**
- `torch.is_tensor()` is faster than `isinstance()`
- `non_blocking=True` enables async GPU transfer
- In-place modification avoids dict reconstruction

---

### 4. **Use group_collate for Multi-Dataset Training** ⚠️ CRITICAL
**Problem:** Without `group_collate`, batches mix samples from different datasets, requiring separate forward passes per sample

**Required:**
```python
from training.utils import group_collate

DataLoader(..., collate_fn=group_collate)
```

**Impact:** Prevents 100x slowdown by batching samples from same dataset together

---

### 5. **Minimize Logging Overhead**
**Problem:** Excessive logging with `.item()` calls can cause GPU synchronization

**Best Practices:**
- Only log on rank 0: `if self.global_rank == 0:`
- Use `sync_dist=False` for per-step logs
- Avoid `.item()` in tight loops (Lightning handles tensor logging efficiently)
- Log scalars on epoch end, not every step

---

### 6. **Auxiliary Loss Computation**
**Problem:** Complex multi-component checking every training step

**Optimized:**
```python
def _compute_auxiliary_loss(self):
    # Simple, fast check for PC modulator only
    if hasattr(self.model, 'modulator') and self.model.modulator is not None:
        if hasattr(self.model.modulator, 'pred_err') and self.model.modulator.pred_err is not None:
            lambda_pred = self.model_config.get('lambda_pred', 0.1)
            pred_err = self.model.modulator.pred_err
            return lambda_pred * (pred_err ** 2).mean()
    return None
```

**Avoid:**
- Looping over multiple components
- Multiple `hasattr()` calls
- Try-except blocks in hot path
- Building dictionaries for logging

---

## Performance Checklist

### Before Training:
- [ ] `persistent_workers=True` in DataLoader
- [ ] `group_collate` is used for multi-dataset training
- [ ] `num_workers > 0` for data loading parallelism
- [ ] `pin_memory=True` for faster GPU transfer

### In Model Code:
- [ ] No `hasattr()` in forward pass (cache in `__init__`)
- [ ] No `isinstance()` in tight loops (use `torch.is_tensor()`)
- [ ] Use `non_blocking=True` for `.to(device)` calls
- [ ] Minimize tensor-to-CPU transfers (`.item()`, `.cpu()`)

### In Training Loop:
- [ ] Logging only on rank 0
- [ ] `sync_dist=False` for per-step logs
- [ ] Minimal auxiliary loss computation
- [ ] No unnecessary tensor copies

### Memory Management:
- [ ] Clear CUDA cache between runs if needed: `torch.cuda.empty_cache()`
- [ ] Use gradient checkpointing for large models
- [ ] Monitor GPU memory with `torch.cuda.memory_summary()`

---

## Debugging Slow Training

### Symptoms:
- Training starts fast but slows down over time
- Slowdown persists across runs without machine restart
- 0.03 it/s instead of 2.0 it/s

### Diagnosis:
1. **Check worker persistence:**
   ```python
   # In DataLoader config
   print(f"persistent_workers: {dataloader.persistent_workers}")
   ```

2. **Profile the training loop:**
   ```python
   import time
   start = time.time()
   loss = training_step(batch)
   print(f"Step time: {time.time() - start:.3f}s")
   ```

3. **Check for memory leaks:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU memory
   ```

4. **Verify group_collate:**
   ```python
   # Check batch structure
   print(f"Batch list length: {len(batch_list)}")
   print(f"Samples per batch: {[len(b['robs']) for b in batch_list]}")
   ```

### Quick Fixes:
1. Restart machine to clear accumulated state
2. Set `persistent_workers=True`
3. Reduce `num_workers` if memory constrained
4. Use `torch.cuda.empty_cache()` between runs

---

## Performance Targets

### Expected Performance:
- **Training:** 1.5-2.5 it/s (depending on model size)
- **Validation:** 3-5 it/s (no gradients)
- **GPU Utilization:** >80%
- **CPU Utilization:** 50-70% (data loading)

### Red Flags:
- **<0.1 it/s:** Critical issue (check persistent_workers, group_collate)
- **GPU <50%:** Data loading bottleneck (increase num_workers)
- **CPU 100%:** Too many workers or slow transforms
- **Memory growing:** Memory leak (check for tensor accumulation)

---

## Common Pitfalls

1. **Forgetting to restart machine** after multiple runs with `persistent_workers=False`
2. **Using dict comprehensions** for tensor device transfer
3. **Calling hasattr()** in model forward pass
4. **Missing group_collate** in multi-dataset training
5. **Excessive logging** with `.item()` calls
6. **Not using non_blocking=True** for GPU transfers
7. **Accumulating tensors** in lists without detaching

---

## Recommended Settings

### For Multi-GPU Training (DDP):
```python
# DataLoader
num_workers=4  # Per GPU
persistent_workers=True
pin_memory=True
non_blocking=True  # For .to(device)

# Logging
sync_dist=False  # For per-step logs
sync_dist=True   # For epoch-end metrics

# Optimization
gradient_clip_val=1.0
accumulate_grad_batches=1
```

### For Single-GPU Training:
```python
num_workers=4-8
persistent_workers=True
pin_memory=True
```

---

## Version History

- **2025-10-12:** Initial performance optimization guide
- **Issue:** Training slowdown from 2.0 it/s to 0.03 it/s
- **Root Cause:** `persistent_workers=False` + accumulated state across runs
- **Fix:** Set `persistent_workers=True` and restart machine to clear state

