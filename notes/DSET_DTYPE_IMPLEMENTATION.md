# Dataset Dtype Implementation

## Summary

Added `dset_dtype` option to control dataset storage dtype in CPU RAM, enabling memory-efficient storage with transforms.

## The Problem

**Original system:**
- Only worked with `pixelnorm` transform
- `remove_pixel_norm()` removed pixelnorm from config
- Data stayed as uint8 in RAM
- `Float32View` applied normalization on GPU: `(x - 127) / 255`

**Issue:**
- If you added OTHER transforms (fftwhitening, temporal_basis, etc.):
  - They need float32 input
  - Data becomes float32 in RAM (4x memory!)
  - But `Float32View` still tries to normalize (WRONG)

## The Solution

Added `dset_dtype` parameter with three options:

### Option 1: `dset_dtype='uint8'` (Current behavior)
- **Memory:** 1x (226 GB for current dataset)
- **Transforms:** Only pixelnorm supported
- **Flow:**
  1. Assert NO transforms other than pixelnorm (raises error if there are)
  2. Call `remove_pixel_norm(cfg)` → removes pixelnorm
  3. Call `prepare_data(cfg)` → data stays uint8
  4. Wrap with `Float32View(tr, norm_removed=True)` → normalizes on GPU

### Option 2: `dset_dtype='bfloat16'` (Recommended for transforms)
- **Memory:** 2x (~452 GB for current dataset)
- **Transforms:** All transforms supported
- **Flow:**
  1. DON'T call `remove_pixel_norm()` → keep all transforms
  2. Call `prepare_data(cfg)` → applies all transforms, data becomes float32
  3. Cast to bfloat16: `tr.cast(torch.bfloat16, target_keys=['stim'])`
  4. Store directly (no Float32View wrapper)

### Option 3: `dset_dtype='float32'` (For debugging)
- **Memory:** 4x (~904 GB - exceeds available RAM!)
- **Transforms:** All transforms supported
- **Flow:** Same as bfloat16, but cast to float32

## Implementation

### 1. Updated `MultiDatasetDM` (training/pl_modules/multidataset_dm.py)

**Added parameter:**
```python
def __init__(self, ..., dset_dtype: str = 'uint8'):
    self.dset_dtype = dset_dtype
    # Validate
    if dset_dtype not in ['uint8', 'bfloat16', 'float32']:
        raise ValueError(...)
```

**Added helper function:**
```python
def _check_for_non_pixelnorm_transforms(self, cfg: Dict) -> bool:
    """Check if config has transforms other than pixelnorm."""
    if 'transforms' not in cfg:
        return False
    
    transforms = cfg['transforms']
    for transform_key, transform_spec in transforms.items():
        if 'ops' in transform_spec:
            for op in transform_spec['ops']:
                if isinstance(op, dict):
                    if 'pixelnorm' not in op:
                        return True
                else:
                    return True
    return False
```

**Updated setup() with branching logic:**
```python
def setup(self, stage: Optional[str] = None):
    ...
    for idx, (cfg, name) in enumerate(zip(self.cfgs, self.names)):
        cfg["_dataset_name"] = name
        
        if self.dset_dtype == 'uint8':
            # Path 1: uint8 storage
            if self._check_for_non_pixelnorm_transforms(cfg):
                raise ValueError(
                    f"Dataset '{name}' has transforms other than pixelnorm, "
                    f"but dset_dtype='uint8' only supports pixelnorm. "
                    f"Use dset_dtype='bfloat16' or 'float32'."
                )
            
            cfg, norm_removed = remove_pixel_norm(cfg)
            tr, va, _ = prepare_data(cfg, strict=False)
            self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
            self.val_dsets[name] = Float32View(va, norm_removed, float16=False)
            
        else:
            # Path 2: bfloat16 or float32 storage
            tr, va, _ = prepare_data(cfg, strict=False)
            
            dtype_map = {
                'bfloat16': torch.bfloat16,
                'float32': torch.float32
            }
            target_dtype = dtype_map[self.dset_dtype]
            
            tr.cast(target_dtype, target_keys=['stim'])
            va.cast(target_dtype, target_keys=['stim'])
            
            # Store directly without Float32View wrapper
            self.train_dsets[name] = tr
            self.val_dsets[name] = va
        
        self.name2idx[name] = idx
    
    print(f"✓ loaded {len(self.train_dsets)} datasets (dtype: {self.dset_dtype})")
```

### 2. Updated training script (training/train_ddp_multidataset.py)

**Added CLI argument:**
```python
p.add_argument("--dset_dtype", type=str, default="uint8",
               choices=["uint8", "bfloat16", "float32"],
               help="Dataset storage dtype in CPU RAM (uint8=1x, bfloat16=2x, float32=4x memory)")
```

**Pass to DataModule:**
```python
dm = MultiDatasetDM(
    ...,
    dset_dtype=args.dset_dtype
)
```

### 3. Updated shell script (experiments/run_all_models_backimage.sh)

**Added variable:**
```bash
DSET_DTYPE="bfloat16"  # Dataset storage dtype: uint8 (1x), bfloat16 (2x), float32 (4x memory)
```

**Pass to training:**
```bash
python training/train_ddp_multidataset.py \
    ...
    --dset_dtype $DSET_DTYPE \
    ...
```

## Usage Examples

### Example 1: Current behavior (uint8, no transforms)
```bash
python training/train_ddp_multidataset.py \
    --model_config configs/model.yaml \
    --dataset_configs_path configs/datasets \
    --dset_dtype uint8 \
    ...
```

**Dataset config:**
```yaml
# Only pixelnorm allowed
transforms:
  stim:
    source: stim
    expose_as: stim
    ops:
      - pixelnorm: {}
```

### Example 2: With transforms (bfloat16)
```bash
python training/train_ddp_multidataset.py \
    --model_config configs/model.yaml \
    --dataset_configs_path configs/datasets \
    --dset_dtype bfloat16 \
    ...
```

**Dataset config:**
```yaml
# All transforms supported!
transforms:
  stim:
    source: stim
    expose_as: stim
    ops:
      - pixelnorm: {}
      - fftwhitening: {f_0: 0.4, n: 4}
      - temporal_basis: {num_cosine_funcs: 10}
```

### Example 3: Error case
```bash
# This will FAIL:
python training/train_ddp_multidataset.py \
    --dset_dtype uint8 \
    ...
```

**With this config:**
```yaml
transforms:
  stim:
    ops:
      - pixelnorm: {}
      - fftwhitening: {}  # ❌ ERROR! uint8 only supports pixelnorm
```

**Error message:**
```
ValueError: Dataset 'session1' has transforms other than pixelnorm, 
but dset_dtype='uint8' only supports pixelnorm. 
Use dset_dtype='bfloat16' or 'float32' to enable other transforms.
```

## Memory Estimates

**Current dataset: 226 GB (uint8)**

| dset_dtype | Memory | Fits in 504 GB RAM? | Use case |
|------------|--------|---------------------|----------|
| uint8 | 226 GB | ✅ Yes (278 GB free) | No transforms (pixelnorm only) |
| bfloat16 | 452 GB | ✅ Yes (52 GB free) | With transforms (recommended) |
| float32 | 904 GB | ❌ No (exceeds RAM) | Debugging only |

## Testing

Test with a small dataset first:
```bash
python training/train_ddp_multidataset.py \
    --model_config experiments/configs/learned_res_small_none_gru.yaml \
    --dataset_configs_path /path/to/configs \
    --max_datasets 2 \
    --batch_size 8 \
    --steps_per_epoch 10 \
    --dset_dtype bfloat16 \
    --num_gpus 1 \
    --precision bf16-mixed
```

## Benefits

1. ✅ **Memory efficient:** bfloat16 uses 2x memory (vs 4x for float32)
2. ✅ **Supports all transforms:** No more pixelnorm-only limitation
3. ✅ **Backward compatible:** Default is uint8 (current behavior)
4. ✅ **Clear errors:** Raises error if uint8 + non-pixelnorm transforms
5. ✅ **Flexible:** Can choose based on available RAM

## Next Steps

1. Test with small dataset (2 datasets, 10 steps)
2. Monitor RAM usage during loading
3. Scale up to full dataset (30 datasets)
4. Compare training speed (should be similar or faster)

