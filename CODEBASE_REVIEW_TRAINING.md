# VisionCore Training System - High-Level Review

## Executive Summary

This document provides a comprehensive review of the training infrastructure in VisionCore, focusing on `train_ddp_multidataset.py` and identifying redundancies, dead code paths, and opportunities for refactoring.

---

## 1. Current Training Architecture

### 1.1 Active Training System: `train_ddp_multidataset.py`

**Location:** `training/train_ddp_multidataset.py` (canonical) and `jake/multidataset_ddp/train_ddp_multidataset.py` (duplicate)

**Entry Points:** All shell scripts in `experiments/` directory call this script:
- `experiments/run_all_models.sh`
- `experiments/run_all_models_backimage.sh`
- `experiments/run_all_models_cones.sh`
- `experiments/run_all_models_gaborium.sh`
- `experiments/run_all_models_pretraining.sh`

**Key Components:**

1. **MultiDatasetModel** (Lightning Module) - Lines 493-903
   - Custom Lightning module defined inline
   - Handles multi-dataset training with DDP
   - Includes regularization support
   - Supports pretrained checkpoints and vision freezing
   - Supports modulator-only models (no vision processing)
   - Implements auxiliary loss for Predictive Coding modulators

2. **MultiDatasetDM** (Data Module) - Lines 348-488
   - Loads multiple datasets from YAML configs
   - Implements contrast-weighted curriculum learning
   - Uses custom `ContrastWeightedSampler` for curriculum
   - Handles distributed sampling

3. **Custom Components:**
   - `LinearWarmupCosineAnnealingLR` scheduler (lines 34-51)
   - `ContrastWeightedSampler` (lines 198-343)
   - `EpochHeartbeat` callback (lines 96-127)
   - `CurriculumCallback` (lines 157-167)

### 1.2 Deprecated Lightning Modules: `models/lightning/`

**Location:** `models/lightning/`

**Files:**
- `models/lightning/core.py` - `PLCoreVisionModel` (single dataset)
- `models/lightning/multidataset.py` - `MultiDatasetPLCore` (multi-dataset)

**Status:** ⚠️ **DEPRECATED - NOT ACTIVELY USED**

**Evidence:**
- No imports found in active training scripts
- Only used in old checkpoint loading code (`models/checkpoint.py`)
- Only used in model manager (`models/model_manager.py`)
- Shell scripts exclusively call `train_ddp_multidataset.py`

---

## 2. Redundancies and Dead Code

### 2.1 Duplicate Training Scripts

**Issue:** Two identical copies of `train_ddp_multidataset.py`
- `training/train_ddp_multidataset.py` (1019 lines)
- `jake/multidataset_ddp/train_ddp_multidataset.py` (1048 lines)

**Impact:** Maintenance burden, potential for divergence

**Recommendation:** 
- Keep `training/train_ddp_multidataset.py` as canonical
- Delete `jake/multidataset_ddp/train_ddp_multidataset.py`
- Update shell scripts to reference canonical location

### 2.2 Deprecated Lightning Modules

**Issue:** Old Lightning modules in `models/lightning/` are not used by active training

**Files:**
- `models/lightning/core.py` (367 lines)
- `models/lightning/multidataset.py` (428 lines)

**Differences from Active System:**

| Feature | Old `MultiDatasetPLCore` | New `MultiDatasetModel` |
|---------|-------------------------|------------------------|
| Data Format | Dict-based batches | List-based batches |
| Sampling | Standard distributed | Contrast-weighted curriculum |
| Regularization | Basic support | Full regularizer system |
| PC Modulator | Basic support | Full auxiliary loss |
| Pretrained Loading | No | Yes (with freeze option) |
| Modulator-only | No | Yes |
| DDP Strategy | Basic | Optimized (find_unused_parameters) |

**Recommendation:** 
- Move to `models/lightning_deprecated/` or delete entirely
- Update imports in checkpoint/model_manager to handle legacy checkpoints
- Document migration path for old checkpoints

### 2.3 Duplicate Shell Scripts

**Issue:** Shell scripts duplicated between `experiments/` and `jake/multidataset_ddp/`

**Duplicates:**
- `experiments/run_all_models.sh` ↔ `jake/multidataset_ddp/run_all_models.sh`
- `experiments/run_all_models_backimage.sh` ↔ `jake/multidataset_ddp/run_all_models_backimage.sh`
- `experiments/run_all_models_cones.sh` ↔ `jake/multidataset_ddp/run_all_models_cones.sh`
- `experiments/run_all_models_gaborium.sh` ↔ `jake/multidataset_ddp/run_all_models_gaborium.sh`

**Recommendation:**
- Keep `experiments/` as canonical location
- Delete duplicates in `jake/multidataset_ddp/`

---

## 3. Architecture Analysis

### 3.1 Why Inline Lightning Module?

The current system defines `MultiDatasetModel` directly in `train_ddp_multidataset.py` rather than using the modular `models/lightning/multidataset.py`.

**Advantages:**
1. ✅ Self-contained training script
2. ✅ Faster iteration during development
3. ✅ No import dependencies on deprecated code
4. ✅ Optimized for specific DDP use case

**Disadvantages:**
1. ❌ Code duplication if multiple training scripts needed
2. ❌ Not reusable for other projects
3. ❌ Harder to test in isolation
4. ❌ Violates separation of concerns

### 3.2 Data Flow

```
Shell Script (experiments/*.sh)
    ↓
train_ddp_multidataset.py
    ↓
MultiDatasetDM (DataModule)
    ├─ Loads YAML configs from dataset_configs_path
    ├─ Creates Float32View datasets
    ├─ Computes contrast scores (if curriculum enabled)
    └─ Creates DataLoader with ContrastWeightedSampler
    ↓
MultiDatasetModel (LightningModule)
    ├─ Builds model from config
    ├─ Loads pretrained weights (optional)
    ├─ Configures optimizer with param groups
    ├─ Implements training/validation steps
    └─ Computes BPS metrics
    ↓
PyTorch Lightning Trainer (DDP)
```

### 3.3 Key Features

**Implemented:**
- ✅ Multi-GPU DDP training
- ✅ Contrast-weighted curriculum learning
- ✅ Pretrained checkpoint loading
- ✅ Vision component freezing
- ✅ Modulator-only models
- ✅ Predictive Coding auxiliary loss
- ✅ Regularization system (L1, L2, group lasso)
- ✅ Learning rate scheduling (cosine warmup)
- ✅ Early stopping
- ✅ WandB logging
- ✅ Per-dataset BPS metrics

**Missing/Incomplete:**
- ⚠️ No unit tests for training components
- ⚠️ No integration tests
- ⚠️ Limited documentation
- ⚠️ No checkpoint resumption from interrupted training
- ⚠️ No mixed precision autocast (uses trainer precision only)

---

## 4. Proposed Refactoring Strategy

### 4.1 Short-term (Low Risk)

1. **Consolidate Duplicates**
   - Delete `jake/multidataset_ddp/train_ddp_multidataset.py`
   - Delete duplicate shell scripts in `jake/multidataset_ddp/`
   - Update any references to point to canonical locations

2. **Deprecate Old Lightning Modules**
   - Move `models/lightning/` → `models/lightning_deprecated/`
   - Add deprecation warnings
   - Update `models/lightning/__init__.py` to import from new location

3. **Documentation**
   - Add docstrings to all classes in `train_ddp_multidataset.py`
   - Create training guide in `docs/training.md`
   - Document curriculum learning system

### 4.2 Medium-term (Moderate Risk)

1. **Extract Lightning Module**
   - Move `MultiDatasetModel` to `training/lightning/multidataset_model.py`
   - Move `MultiDatasetDM` to `training/lightning/multidataset_dm.py`
   - Keep `train_ddp_multidataset.py` as thin CLI wrapper
   - Benefits: Reusability, testability, cleaner separation

2. **Extract Samplers**
   - Move `ContrastWeightedSampler` to `training/samplers.py`
   - Add unit tests for curriculum logic

3. **Extract Callbacks**
   - Move callbacks to `training/callbacks.py`
   - Make them reusable across training scripts

### 4.3 Long-term (Higher Risk)

1. **Unified Training Framework**
   - Create `training/trainer.py` with common training logic
   - Support both single-dataset and multi-dataset modes
   - Consolidate all training features in one place

2. **Configuration System**
   - Move from argparse to Hydra or similar
   - Enable config composition and overrides
   - Better experiment tracking

3. **Testing Infrastructure**
   - Unit tests for all components
   - Integration tests for full training pipeline
   - Regression tests for model performance

---

## 5. Specific Recommendations

### 5.1 Immediate Actions

```bash
# 1. Remove duplicates
rm jake/multidataset_ddp/train_ddp_multidataset.py
rm jake/multidataset_ddp/run_all_models*.sh

# 2. Update shell scripts to use canonical path
# Change: python train_ddp_multidataset.py
# To:     python training/train_ddp_multidataset.py

# 3. Deprecate old lightning modules
mkdir models/lightning_deprecated
mv models/lightning/*.py models/lightning_deprecated/
```

### 5.2 Code Organization Proposal

```
training/
├── __init__.py
├── train_ddp_multidataset.py          # CLI entry point (thin wrapper)
├── lightning/
│   ├── __init__.py
│   ├── multidataset_model.py          # MultiDatasetModel class
│   └── multidataset_dm.py             # MultiDatasetDM class
├── samplers.py                         # ContrastWeightedSampler
├── callbacks.py                        # Training callbacks
├── schedulers.py                       # Custom LR schedulers
└── regularizers.py                     # Already exists

models/
├── lightning_deprecated/               # Old Lightning modules
│   ├── __init__.py
│   ├── core.py                        # PLCoreVisionModel (deprecated)
│   └── multidataset.py                # MultiDatasetPLCore (deprecated)
└── ...
```

### 5.3 Migration Path for Old Checkpoints

If old checkpoints exist that use `PLCoreVisionModel` or `MultiDatasetPLCore`:

1. Keep deprecated modules for loading only
2. Add conversion utility to migrate to new format
3. Document checkpoint compatibility

---

## 6. Dead Code Paths

### 6.1 Commented Code

**In `train_ddp_multidataset.py`:**
- Lines 130-154: Commented `PrintTimings` callback
- Line 508: Commented torch.compile line
- Lines 678: Commented autocast context

**Recommendation:** Remove commented code or move to git history

### 6.2 Unused Imports

**In old Lightning modules:**
- `models/lightning/multidataset.py` imports `PLCoreVisionModel` but doesn't use it
- Both modules import matplotlib/wandb but visualization is minimal

### 6.3 Unused Arguments

**In shell scripts:**
- `--gradient_clip_val` and `--accumulate_grad_batches` marked as "passthroughs" but are actually used
- Comment is misleading (line 943-945)

---

## 7. Testing Gaps

### 7.1 Missing Tests

1. **Unit Tests:**
   - `ContrastWeightedSampler` curriculum logic
   - `MultiDatasetModel` forward pass
   - Regularizer integration
   - Auxiliary loss computation

2. **Integration Tests:**
   - Full training loop (1 epoch)
   - Checkpoint save/load
   - DDP synchronization
   - Curriculum warmup

3. **Regression Tests:**
   - Model performance on known datasets
   - Gradient flow
   - Memory usage

---

## 8. Summary

### Current State
- ✅ Working DDP training system with advanced features
- ✅ Supports curriculum learning, regularization, PC modulators
- ❌ Significant code duplication
- ❌ Deprecated modules still in codebase
- ❌ Limited testing and documentation

### Recommended Priority

**High Priority:**
1. Remove duplicate files (low risk, immediate benefit)
2. Deprecate old Lightning modules
3. Add documentation

**Medium Priority:**
4. Extract components to separate modules
5. Add unit tests
6. Clean up commented code

**Low Priority:**
7. Unified training framework
8. Configuration system overhaul
9. Comprehensive testing suite

### Estimated Effort

- **Short-term cleanup:** 2-4 hours
- **Medium-term refactoring:** 1-2 days
- **Long-term framework:** 1-2 weeks

---

## Appendix: File Inventory

### Active Files
- `training/train_ddp_multidataset.py` (1019 lines) ✅
- `training/regularizers.py` ✅
- `experiments/run_all_models*.sh` (5 files) ✅

### Duplicate Files (Remove)
- `jake/multidataset_ddp/train_ddp_multidataset.py` ❌
- `jake/multidataset_ddp/run_all_models*.sh` (4 files) ❌

### Deprecated Files (Move/Archive)
- `models/lightning/core.py` ⚠️
- `models/lightning/multidataset.py` ⚠️

### Dependencies
- `models/losses.py` - MaskedLoss, PoissonBPSAggregator
- `models/build.py` - build_model
- `models/config_loader.py` - load_config, load_dataset_configs

