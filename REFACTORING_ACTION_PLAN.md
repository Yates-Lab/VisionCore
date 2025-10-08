# VisionCore Training Refactoring - Action Plan

## Quick Reference

**Goal:** Clean up redundancies and move Lightning modules to a proper training submodule

**Estimated Time:** 
- Phase 1 (Cleanup): 2-4 hours
- Phase 2 (Refactor): 1-2 days
- Phase 3 (Polish): 2-3 days

---

## Phase 1: Immediate Cleanup (Low Risk) ⚡

### Step 1.1: Remove Duplicate Training Script

**Files to delete:**
```bash
rm jake/multidataset_ddp/train_ddp_multidataset.py
```

**Verification:**
```bash
# Ensure no references exist
grep -r "jake/multidataset_ddp/train_ddp_multidataset" .
```

**Risk:** ⚠️ Low - File is a duplicate of `training/train_ddp_multidataset.py`

---

### Step 1.2: Remove Duplicate Shell Scripts

**Files to delete:**
```bash
rm jake/multidataset_ddp/run_all_models.sh
rm jake/multidataset_ddp/run_all_models_backimage.sh
rm jake/multidataset_ddp/run_all_models_cones.sh
rm jake/multidataset_ddp/run_all_models_gaborium.sh
# Keep run_all_models_pretraining.sh if it's unique
```

**Verification:**
```bash
# Check if any scripts reference the deleted files
grep -r "jake/multidataset_ddp/run_all" .
```

**Risk:** ⚠️ Low - Duplicates of files in `experiments/`

---

### Step 1.3: Update Shell Scripts to Use Canonical Path

**Files to update:**
- `experiments/run_all_models.sh`
- `experiments/run_all_models_backimage.sh`
- `experiments/run_all_models_cones.sh`
- `experiments/run_all_models_gaborium.sh`
- `experiments/run_all_models_pretraining.sh`

**Change:**
```bash
# FROM:
python train_ddp_multidataset.py \

# TO:
python training/train_ddp_multidataset.py \
```

**Risk:** ⚠️ Low - Simple path update

---

### Step 1.4: Clean Up Commented Code

**In `training/train_ddp_multidataset.py`:**

Remove or document:
- Lines 130-154: Commented `PrintTimings` callback
- Line 508: Commented torch.compile line
- Line 678: Commented autocast context

**Risk:** ⚠️ Very Low - Just removing comments

---

### Step 1.5: Fix Misleading Comments

**In `training/train_ddp_multidataset.py` line 943-945:**

```python
# BEFORE:
# passthroughs (ignored but kept for compatibility)
p.add_argument("--gradient_clip_val",      type=float, default=1.0)
p.add_argument("--accumulate_grad_batches",type=int,   default=1)

# AFTER:
# Training optimization parameters
p.add_argument("--gradient_clip_val",      type=float, default=1.0,
               help="Gradient clipping value for training stability")
p.add_argument("--accumulate_grad_batches",type=int,   default=1,
               help="Number of batches to accumulate gradients")
```

**Risk:** ⚠️ Very Low - Documentation fix

---

## Phase 2: Deprecate Old Lightning Modules (Moderate Risk) 🔄

### Step 2.1: Create Deprecated Directory

```bash
mkdir -p models/lightning_deprecated
```

---

### Step 2.2: Move Old Lightning Modules

```bash
mv models/lightning/core.py models/lightning_deprecated/
mv models/lightning/multidataset.py models/lightning_deprecated/
```

---

### Step 2.3: Update `models/lightning/__init__.py`

**Before:**
```python
from .core import PLCoreVisionModel
from .multidataset import MultiDatasetPLCore
```

**After:**
```python
"""
PyTorch Lightning modules for DataYatesV1.

DEPRECATED: The modules in this package are deprecated.
Active training uses the modules in training/train_ddp_multidataset.py
These are kept only for backward compatibility with old checkpoints.
"""

import warnings

# Import from deprecated location with warning
def _deprecated_import():
    warnings.warn(
        "models.lightning modules are deprecated. "
        "Use training.lightning modules instead.",
        DeprecationWarning,
        stacklevel=2
    )

try:
    from ..lightning_deprecated.core import PLCoreVisionModel
    from ..lightning_deprecated.multidataset import MultiDatasetPLCore
    _deprecated_import()
except ImportError:
    # If deprecated modules are removed, provide helpful error
    raise ImportError(
        "Legacy Lightning modules have been removed. "
        "Please use training.lightning modules or update your checkpoints."
    )

__all__ = ['PLCoreVisionModel', 'MultiDatasetPLCore']
```

**Risk:** ⚠️ Medium - May break old checkpoint loading

---

### Step 2.4: Update Checkpoint Loading Code

**In `models/checkpoint.py`:**

Add fallback for old checkpoints:

```python
def load_model_from_checkpoint(checkpoint_path, config=None, strict=True, map_location=None):
    """Load a model from a checkpoint with backward compatibility."""
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Check if this is an old checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        
        # Old checkpoint detection
        if 'model_class' in hparams and hparams.get('model_class').__name__ in ['PLCoreVisionModel', 'MultiDatasetPLCore']:
            warnings.warn(
                f"Loading legacy checkpoint from {checkpoint_path}. "
                "Consider re-saving with new training system.",
                DeprecationWarning
            )
            # Use deprecated loader
            from ..lightning_deprecated import PLCoreVisionModel, MultiDatasetPLCore
            # ... existing loading logic
    
    # ... rest of function
```

**Risk:** ⚠️ Medium - Needs testing with old checkpoints

---

## Phase 3: Extract Components (Higher Risk) 🏗️

### Step 3.1: Create Training Lightning Submodule

```bash
mkdir -p training/lightning
touch training/lightning/__init__.py
```

---

### Step 3.2: Extract MultiDatasetModel

**Create `training/lightning/multidataset_model.py`:**

Move `MultiDatasetModel` class (lines 493-903) from `train_ddp_multidataset.py`

**Update imports:**
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional

from models.losses import MaskedLoss, PoissonBPSAggregator
from training.regularizers import create_regularizers, get_excluded_params_for_weight_decay
```

**Risk:** ⚠️ High - Core training component

---

### Step 3.3: Extract MultiDatasetDM

**Create `training/lightning/multidataset_dm.py`:**

Move `MultiDatasetDM` class (lines 348-488) from `train_ddp_multidataset.py`

**Also move:**
- `Float32View` class (lines 176-186)
- `cast_stim` function (lines 172-174)
- `group_collate` function (lines 188-192)

**Risk:** ⚠️ High - Core data loading component

---

### Step 3.4: Extract Samplers

**Create `training/samplers.py`:**

Move `ContrastWeightedSampler` class (lines 198-343) from `train_ddp_multidataset.py`

**Risk:** ⚠️ Medium - Used by data module

---

### Step 3.5: Extract Callbacks

**Create `training/callbacks.py`:**

Move:
- `Heartbeat` class (lines 75-94)
- `EpochHeartbeat` class (lines 96-127)
- `CurriculumCallback` class (lines 157-167)

**Risk:** ⚠️ Low - Self-contained components

---

### Step 3.6: Extract Schedulers

**Create `training/schedulers.py`:**

Move:
- `LinearWarmupCosineAnnealingLR` class (lines 34-51)

**Risk:** ⚠️ Low - Self-contained component

---

### Step 3.7: Update `training/train_ddp_multidataset.py`

**After extraction, this file should be ~200 lines:**

```python
#!/usr/bin/env python3
"""
Multi-dataset V1 training - CLI entry point

This script provides a command-line interface for training models
on multiple datasets using PyTorch Lightning with DDP.
"""

import os
import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from training.lightning.multidataset_model import MultiDatasetModel
from training.lightning.multidataset_dm import MultiDatasetDM
from training.callbacks import EpochHeartbeat, CurriculumCallback

def main():
    # Argument parsing
    p = argparse.ArgumentParser(description="Multi-dataset V1 training")
    # ... all arguments ...
    args = p.parse_args()
    
    # Create data module
    dm = MultiDatasetDM(
        args.dataset_configs_path,
        args.max_datasets,
        args.batch_size,
        args.num_workers,
        args.steps_per_epoch,
        enable_curriculum=args.enable_curriculum
    )
    
    # Create model
    model = MultiDatasetModel(
        args.model_config,
        args.dataset_configs_path,
        args.learning_rate,
        args.weight_decay,
        args.max_datasets,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_vision=args.freeze_vision,
        compile_model=args.compile
    )
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(...),
        LearningRateMonitor(...),
        EpochHeartbeat(...),
        EarlyStopping(...)
    ]
    
    if args.enable_curriculum:
        callbacks.append(CurriculumCallback())
    
    # Configure logger
    logger = WandbLogger(...) if rank == 0 else None
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=DDPStrategy(...),
        callbacks=callbacks,
        logger=logger,
        ...
    )
    
    # Train
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
```

**Risk:** ⚠️ High - Main entry point

---

## Testing Strategy

### After Each Phase:

1. **Smoke Test:**
   ```bash
   python training/train_ddp_multidataset.py \
       --model_config configs/learned_res_small.yaml \
       --dataset_configs_path /path/to/configs \
       --max_datasets 2 \
       --batch_size 8 \
       --num_gpus 1 \
       --max_epochs 1 \
       --steps_per_epoch 10
   ```

2. **Checkpoint Test:**
   ```bash
   # Test loading old checkpoint
   python -c "from models.checkpoint import load_model_from_checkpoint; \
              load_model_from_checkpoint('path/to/old/checkpoint.ckpt')"
   ```

3. **Import Test:**
   ```bash
   python -c "from training.lightning import MultiDatasetModel, MultiDatasetDM"
   ```

---

## Rollback Plan

### If Phase 1 Fails:
```bash
git checkout -- experiments/
git checkout -- training/train_ddp_multidataset.py
```

### If Phase 2 Fails:
```bash
git checkout -- models/lightning/
rm -rf models/lightning_deprecated/
```

### If Phase 3 Fails:
```bash
git checkout -- training/
# Restore from backup
```

---

## Success Criteria

### Phase 1:
- ✅ No duplicate files
- ✅ All shell scripts run successfully
- ✅ No broken references

### Phase 2:
- ✅ Old checkpoints still load
- ✅ Deprecation warnings appear
- ✅ No import errors

### Phase 3:
- ✅ Training runs successfully
- ✅ All tests pass
- ✅ Code is modular and reusable
- ✅ Documentation is complete

---

## Timeline

### Week 1:
- **Day 1:** Phase 1 (Cleanup)
- **Day 2:** Phase 2 (Deprecation)
- **Day 3:** Testing and verification

### Week 2:
- **Day 1-2:** Phase 3.1-3.3 (Extract core components)
- **Day 3-4:** Phase 3.4-3.7 (Extract utilities and update main script)
- **Day 5:** Testing and documentation

---

## Notes

- **Backup everything before starting**
- **Test after each step**
- **Commit frequently with descriptive messages**
- **Keep old checkpoints for testing**
- **Document any issues encountered**

---

## Questions to Answer Before Starting

1. ✅ Are there any active experiments using the old Lightning modules?
2. ✅ Do we have old checkpoints that need to be loadable?
3. ✅ Is the `jake/multidataset_ddp/` directory still needed for anything?
4. ⚠️ Should we keep backward compatibility or force migration?
5. ⚠️ Do we need to support both old and new training systems simultaneously?

---

## Future Enhancements (Post-Refactoring)

1. **Add unit tests** for all extracted components
2. **Add integration tests** for full training pipeline
3. **Create training documentation** with examples
4. **Add type hints** throughout
5. **Consider Hydra** for configuration management
6. **Add checkpoint migration utility** for old checkpoints
7. **Create training tutorial** notebook

