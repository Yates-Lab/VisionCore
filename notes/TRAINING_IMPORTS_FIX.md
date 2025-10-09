# Training Imports Fix - Use Local VisionCore Models

## Problem

The training system was importing `build_model` and `config_loader` from **DataYatesV1** instead of the local VisionCore versions. This meant:

1. ❌ New features added to VisionCore models weren't being used
2. ❌ The `anti_aliasing` parameter in `LearnableTemporalConv` wasn't available
3. ❌ Training was using outdated DataYatesV1 code instead of local improvements

## Error Message

```
TypeError: LearnableTemporalConv.__init__() got an unexpected keyword argument 'anti_aliasing'
```

**Root cause:** Training was importing from DataYatesV1, which doesn't have the new parameter.

---

## What We Fixed

### 1. ✅ training/pl_modules/multidataset_model.py

**Before (Line 84-85):**
```python
from DataYatesV1.models.config_loader import load_dataset_configs, load_config
from DataYatesV1.models import build_model
```

**After:**
```python
from models.config_loader import load_dataset_configs, load_config
from models import build_model
```

### 2. ✅ training/pl_modules/multidataset_dm.py

**Before (Line 95):**
```python
from DataYatesV1.models.config_loader import load_dataset_configs
```

**After:**
```python
from models.config_loader import load_dataset_configs
```

---

## What This Fixes

### ✅ Now Uses Local VisionCore Code

The training system now uses:
- ✅ Local `build_model()` from `models/build.py`
- ✅ Local `config_loader` from `models/config_loader.py`
- ✅ Local `factory.py` which creates components from `models/modules/`
- ✅ Local `LearnableTemporalConv` with `anti_aliasing` parameter

### ✅ Your New Feature Works

```python
# This now works in training configs:
frontend_params:
  anti_aliasing: true  # ✅ Uses AntiAliasedConv1d
```

---

## Import Strategy for Training

### ✅ Use Local VisionCore
These should be imported from local VisionCore:
- `build_model` → `models/build.py`
- `config_loader` → `models/config_loader.py`
- All model components → `models/modules/`

### ✅ Use DataYatesV1 Only When Necessary
These don't exist in VisionCore and must come from DataYatesV1:
- `prepare_data` → DataYatesV1.utils.data
- `remove_pixel_norm` → DataYatesV1.utils.data.loading

---

## Verification

### ✅ LearnableTemporalConv with anti_aliasing works
```bash
$ python -c "from models.modules.frontend import LearnableTemporalConv; \
  ltc = LearnableTemporalConv(kernel_size=10, anti_aliasing=True); \
  print('✓')"
✓
```

### ✅ Training imports local models
```bash
$ python -c "from training.pl_modules import MultiDatasetModel; print('✓')"
✓
```

### ✅ Training script works
```bash
$ python training/train_ddp_multidataset.py --help
usage: train_ddp_multidataset.py [-h] --model_config MODEL_CONFIG ...
```

---

## Files Modified

1. ✅ `training/pl_modules/multidataset_model.py` - Use local build_model and config_loader
2. ✅ `training/pl_modules/multidataset_dm.py` - Use local config_loader

---

## Benefits

✅ **New features work** - anti_aliasing parameter now available  
✅ **Uses local code** - All VisionCore improvements are used  
✅ **Faster iteration** - No need to update DataYatesV1  
✅ **Cleaner architecture** - Training uses VisionCore models  
✅ **Less dependency** - Only import from DataYatesV1 when necessary  

---

## Your New Feature: Anti-Aliased Temporal Convolution

### What You Added

In `models/modules/frontend.py`, `LearnableTemporalConv` now supports:

```python
def __init__(self, kernel_size=10, num_channels=6, 
             init_type='gaussian_derivatives', causal=True,
             anti_aliasing=True,  # ← NEW PARAMETER
             bias=False):
    ...
    if anti_aliasing:
        from .conv_layers import AntiAliasedConv1d
        self.temporal_conv = AntiAliasedConv1d(...)
    else:
        self.temporal_conv = nn.Conv1d(...)
```

### How to Use in Config

```yaml
# In your model config YAML:
frontend_type: learnable_temporal
frontend_params:
  kernel_size: 10
  num_channels: 6
  anti_aliasing: true  # ← Enable anti-aliased convolution
  init_type: gaussian_derivatives
```

---

## Status

✅ **Imports fixed**  
✅ **Local models used**  
✅ **Anti-aliasing feature works**  
✅ **Ready to train**

---

## Next Steps

Your training should now work with the `anti_aliasing` parameter! Try running:

```bash
bash experiments/run_all_models_backimage.sh
```

Or test with a specific config that uses `anti_aliasing: true`.

---

**Great work adding the anti-aliasing feature! The training system now uses your local VisionCore improvements.**

