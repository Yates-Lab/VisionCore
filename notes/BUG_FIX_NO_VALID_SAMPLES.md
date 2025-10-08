# Bug Fix: "No valid samples for cell X" Error

## Problem

When running `run_all_models_backimage.sh`, training crashed during validation with:

```
ValueError: No valid samples for cell 9
```

**Location:** `models/losses/poisson.py`, line 143 in `PoissonBPSAggregator.closure()`

---

## Root Cause

The error occurs during **distributed validation** (DDP with 2 GPUs) when:

1. **Small validation set:** `limit_val_batches=0.05` means only 5% of validation data is used
2. **Distributed sampling:** Data is split across 2 GPU ranks
3. **Cell-specific masking:** The `dfs` (degrees of freedom) mask filters samples per-cell
4. **Result:** Some cells have **zero valid samples** on a particular GPU rank

### Why This Happens

```python
# In PoissonBPSAggregator.closure()
for cell_idx in range(robs.shape[1]):
    cell_mask = dfs[:, cell_idx].bool()
    if cell_mask.sum() > 0:
        # Calculate BPS for this cell
        ...
    else:
        # OLD CODE: Raised error instead of handling gracefully
        raise ValueError(f"No valid samples for cell {cell_idx}")
```

**Example scenario:**
- Dataset has 100 validation samples for cell 9
- With `limit_val_batches=0.05`, only 5 samples are used
- These 5 samples are split across 2 GPUs (2-3 samples each)
- If GPU rank 1 gets samples where `dfs[:, 9] == 0` (all masked), it has no valid samples
- **Crash!**

---

## The Fix

### 1. Handle Missing Samples Gracefully (`models/losses/poisson.py`)

**Before:**
```python
else:
    # No valid samples for this cell - return 0 or NaN
    raise ValueError(f"No valid samples for cell {cell_idx}")
    bps_list.append(torch.tensor([0.0]))  # Unreachable!
```

**After:**
```python
else:
    # No valid samples for this cell in this batch/rank
    # Return NaN so it gets filtered out in distributed reduction
    # (The other rank may have valid samples for this cell)
    bps_list.append(torch.tensor([float('nan')]))
```

**Rationale:**
- Return `NaN` instead of crashing
- Other GPU ranks may have valid samples for this cell
- NaN values will be filtered during distributed reduction

---

### 2. Filter NaN Values During Reduction (`training/train_ddp_multidataset.py`)

**Before:**
```python
else:
    bps = agg.closure().clamp_min(0.0)  # (units,)
    local_sum   = bps.sum().to(self.device)
    local_count = torch.tensor(bps.numel(), device=self.device)
```

**After:**
```python
else:
    bps = agg.closure()  # (units,) - may contain NaN for cells with no samples
    if bps is not None:
        # Filter out NaN values (cells with no valid samples on this rank)
        valid_mask = ~torch.isnan(bps)
        if valid_mask.any():
            valid_bps = bps[valid_mask].clamp_min(0.0)
            local_sum   = valid_bps.sum().to(self.device)
            local_count = torch.tensor(valid_bps.numel(), device=self.device)
        else:
            # All values are NaN on this rank
            local_sum   = torch.tensor(0.0, device=self.device)
            local_count = torch.tensor(0,   device=self.device)
    else:
        # No data from aggregator
        local_sum   = torch.tensor(0.0, device=self.device)
        local_count = torch.tensor(0,   device=self.device)
```

**Rationale:**
- Filter out NaN values before computing sum/count
- Only count cells with valid samples
- Distributed reduction will aggregate across all ranks
- Final BPS is computed only from cells with valid samples

---

## How It Works Now

### Example with 2 GPUs:

**Rank 0:**
- Cell 0: BPS = 0.5 ✅
- Cell 1: BPS = 0.7 ✅
- Cell 2: BPS = NaN ❌ (no valid samples on this rank)

**Rank 1:**
- Cell 0: BPS = 0.6 ✅
- Cell 1: BPS = NaN ❌ (no valid samples on this rank)
- Cell 2: BPS = 0.8 ✅

**After filtering NaN:**

**Rank 0:**
- local_sum = 0.5 + 0.7 = 1.2
- local_count = 2

**Rank 1:**
- local_sum = 0.6 + 0.8 = 1.4
- local_count = 2

**After distributed reduction:**
- global_sum = 1.2 + 1.4 = 2.6
- global_count = 2 + 2 = 4
- **bps_mean = 2.6 / 4 = 0.65** ✅

---

## Testing

### Before Fix:
```bash
bash experiments/run_all_models_backimage.sh
# Result: ValueError: No valid samples for cell 9
```

### After Fix:
```bash
bash experiments/run_all_models_backimage.sh
# Result: Training should proceed normally
```

### What to Watch For:

1. **Validation runs without errors** ✅
2. **BPS metrics are computed correctly** ✅
3. **No NaN in logged metrics** (NaN is filtered before logging)
4. **Training continues normally** ✅

---

## Edge Cases Handled

### 1. All cells have NaN on a rank
```python
if valid_mask.any():
    # Compute sum/count
else:
    # All NaN - contribute 0 to reduction
    local_sum = 0.0
    local_count = 0
```

### 2. Aggregator has no data
```python
if len(agg.robs) == 0:
    local_sum = 0.0
    local_count = 0
```

### 3. Aggregator returns None
```python
if bps is not None:
    # Process BPS
else:
    # No data
    local_sum = 0.0
    local_count = 0
```

### 4. All ranks have no valid samples for a dataset
```python
if global_count > 0:
    bps_mean = global_sum / global_count
    per_ds.append(bps_mean)
# else: skip this dataset (no valid samples anywhere)
```

---

## Why This Bug Appeared

1. **Small validation set:** `limit_val_batches=0.05` is very aggressive
   - Only 5% of validation data is used
   - Increases chance of cells having no samples

2. **Distributed sampling:** Data split across 2 GPUs
   - Each rank gets ~2.5% of validation data
   - Even smaller chance of all cells having samples

3. **Cell-specific masking:** `dfs` mask can filter out samples per-cell
   - Some cells may have very few valid samples
   - Combined with small validation set → zero samples

---

## Recommendations

### Short-term (Already Fixed):
- ✅ Handle NaN gracefully in BPS aggregator
- ✅ Filter NaN during distributed reduction

### Long-term (Consider):

1. **Increase validation set size:**
   ```python
   # In train_ddp_multidataset.py, line 1000
   limit_val_batches=0.1,  # Use 10% instead of 5%
   ```

2. **Add warning for cells with few samples:**
   ```python
   if cell_mask.sum() < 10:  # Fewer than 10 samples
       warnings.warn(f"Cell {cell_idx} has only {cell_mask.sum()} valid samples")
   ```

3. **Use stratified sampling** to ensure all cells get samples:
   - Modify `DistributedSampler` to balance samples per cell
   - More complex but ensures better coverage

4. **Add validation check** before training:
   ```python
   # Check that all cells have sufficient validation samples
   for dataset in val_datasets:
       check_cell_coverage(dataset, min_samples=10)
   ```

---

## Files Modified

1. **models/losses/poisson.py**
   - Line 143: Changed `raise ValueError` → `return NaN`
   - Added comment explaining why NaN is used

2. **training/train_ddp_multidataset.py**
   - Lines 768-805: Added NaN filtering in `on_validation_epoch_end`
   - Handle cases where all values are NaN
   - Handle cases where aggregator returns None

---

## Related Issues

This fix also prevents similar errors in:
- Training BPS computation (if used)
- Other metrics that aggregate per-cell statistics
- Any distributed training with small batch sizes

---

## Verification

After applying the fix, verify:

```bash
# 1. Run training
bash experiments/run_all_models_backimage.sh

# 2. Check logs for:
#    - No "ValueError: No valid samples" errors
#    - BPS metrics are logged correctly
#    - No NaN in wandb logs

# 3. Check validation metrics:
#    - val_bps_overall should be a valid number
#    - val_bps/{dataset_name} should be valid numbers
#    - No warnings about NaN values
```

---

## Summary

**Problem:** Validation crashed when cells had no valid samples on a GPU rank  
**Cause:** Small validation set + distributed sampling + cell-specific masking  
**Fix:** Return NaN for cells with no samples, filter NaN during reduction  
**Result:** Training proceeds normally, BPS computed only from valid samples  

**Status:** ✅ Fixed and ready to test

