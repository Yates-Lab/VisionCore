# Cache Optimization for evaluate_model_multidataset

## Problem
The `evaluate_model_multidataset` function was taking ~20 minutes to run even when all analyses were already cached, because it was loading and preprocessing each dataset regardless of whether the analyses needed to be recomputed.

## Solution
Optimized the function to avoid loading datasets when all requested analyses are already cached:

### Key Changes

1. **Early Cache Detection**: The function already checked which analyses were missing using `check_existing_cache_files()`, but still loaded all datasets.

2. **Conditional Dataset Loading**: Modified the main loop to only load datasets when they have missing analyses:
   ```python
   # Only load dataset if we need to run analyses (not just load from cache)
   if needs_processing:
       train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
   else:
       # For cache-only loading, extract CIDs from existing cache files
   ```

3. **CID Extraction from Cache**: When datasets don't need processing, extract cell IDs (CIDs) from existing cache files instead of loading the full dataset:
   - Try BPS cache first (infer from data shape)
   - Fall back to CCNORM/saccade caches
   - Only load dataset if CID extraction fails

4. **Safe Analysis Execution**: Added checks to ensure analyses that require dataset loading are only run when the dataset is available:
   ```python
   if train_data is None or val_data is None:
       print("❌ Cannot run analysis: dataset not loaded, skipping analysis")
   else:
       # Run analysis
   ```

### Performance Impact

**Before**: ~20 minutes even with all analyses cached (due to dataset loading)
**After**: Should be <1 minute when all analyses are cached (cache loading only)

Expected speedup: **10-20x** for fully cached evaluations.

### Backward Compatibility

- All existing functionality preserved
- Same API and return format
- Graceful fallback to dataset loading if CID extraction fails
- Works with both cached and non-cached scenarios

### Testing

Run the test script to verify the optimization:
```bash
cd jake/multidataset_ddp
python test_cache_optimization.py
```

This will run the evaluation twice and measure the speedup from cache optimization.

### Files Modified

- `eval_stack_multidataset.py`: Main optimization implementation
- `test_cache_optimization.py`: Test script to verify performance improvement

### Technical Details

The optimization works by:
1. Identifying which datasets need processing vs. cache-only loading
2. For cache-only datasets: extracting metadata (CIDs) from cache files
3. Loading cached analysis results without dataset preprocessing
4. Only loading datasets when new analyses need to be computed

This maintains the same result structure while dramatically reducing I/O and preprocessing overhead for cached evaluations.
