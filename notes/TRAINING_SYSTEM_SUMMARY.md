# VisionCore Training System - Executive Summary

## TL;DR

**Current State:** You have a working, feature-rich DDP training system in `training/train_ddp_multidataset.py`, but with significant code duplication and deprecated modules.

**Main Issues:**
1. 🔴 **Duplicate training script** in `jake/multidataset_ddp/`
2. 🔴 **Deprecated Lightning modules** in `models/lightning/` that are NOT used
3. 🔴 **Duplicate shell scripts** in `jake/multidataset_ddp/`
4. 🟡 **Monolithic training script** (1019 lines, everything inline)

**Recommendation:** Start with low-risk cleanup (remove duplicates), then consider extracting components to `training/lightning/` submodule.

---

## Key Findings

### ✅ What's Working Well

1. **Active Training System** (`training/train_ddp_multidataset.py`)
   - Robust DDP multi-GPU training
   - Advanced features: curriculum learning, regularization, PC modulators
   - Pretrained checkpoint loading with vision freezing
   - Modulator-only model support
   - Per-dataset BPS metrics
   - WandB integration

2. **Shell Scripts** (`experiments/*.sh`)
   - Well-organized experiment runners
   - Consistent parameter passing
   - Good error handling

3. **Regularization System** (`training/regularizers.py`)
   - Modular and well-designed
   - Supports L1, L2, group lasso
   - Proper integration with optimizer

### ❌ What's Not Working

1. **Code Duplication**
   - `train_ddp_multidataset.py` exists in TWO locations
   - Shell scripts duplicated between `experiments/` and `jake/multidataset_ddp/`
   - Risk of divergence and maintenance burden

2. **Deprecated Code**
   - `models/lightning/core.py` - NOT used by active training
   - `models/lightning/multidataset.py` - NOT used by active training
   - Only referenced in old checkpoint loading code
   - Confusing for new developers

3. **Monolithic Structure**
   - All training logic in one 1019-line file
   - Hard to test individual components
   - Not reusable for other projects

### 🟡 What Could Be Better

1. **Testing**
   - No unit tests for training components
   - No integration tests
   - Manual testing only

2. **Documentation**
   - Limited inline documentation
   - No training guide
   - No architecture documentation

3. **Code Organization**
   - Components not separated into modules
   - Callbacks, samplers, schedulers all inline

---

## Architecture Comparison

### Current: Inline Everything

```
training/train_ddp_multidataset.py (1019 lines)
├── LinearWarmupCosineAnnealingLR (scheduler)
├── Heartbeat (callback)
├── EpochHeartbeat (callback)
├── CurriculumCallback (callback)
├── cast_stim (utility)
├── Float32View (dataset wrapper)
├── group_collate (collate function)
├── ContrastWeightedSampler (sampler)
├── MultiDatasetDM (data module)
├── MultiDatasetModel (lightning module)
└── main() (CLI)
```

### Proposed: Modular Structure

```
training/
├── train_ddp_multidataset.py (~200 lines - CLI only)
├── lightning/
│   ├── multidataset_model.py (MultiDatasetModel)
│   └── multidataset_dm.py (MultiDatasetDM + utilities)
├── samplers.py (ContrastWeightedSampler)
├── callbacks.py (Heartbeat, EpochHeartbeat, CurriculumCallback)
├── schedulers.py (LinearWarmupCosineAnnealingLR)
└── regularizers.py (already exists)
```

---

## Redundancy Analysis

### 🔴 Critical Redundancies (Remove Immediately)

| File | Status | Action |
|------|--------|--------|
| `jake/multidataset_ddp/train_ddp_multidataset.py` | Duplicate | DELETE |
| `jake/multidataset_ddp/run_all_models.sh` | Duplicate | DELETE |
| `jake/multidataset_ddp/run_all_models_backimage.sh` | Duplicate | DELETE |
| `jake/multidataset_ddp/run_all_models_cones.sh` | Duplicate | DELETE |
| `jake/multidataset_ddp/run_all_models_gaborium.sh` | Duplicate | DELETE |

**Estimated Time:** 30 minutes  
**Risk:** Very Low  
**Benefit:** Immediate reduction in maintenance burden

### 🟡 Deprecated Code (Move or Delete)

| File | Status | Action |
|------|--------|--------|
| `models/lightning/core.py` | Deprecated | Move to `models/lightning_deprecated/` |
| `models/lightning/multidataset.py` | Deprecated | Move to `models/lightning_deprecated/` |

**Estimated Time:** 1-2 hours  
**Risk:** Low (if checkpoint loading is updated)  
**Benefit:** Clearer codebase structure

### 🟢 Commented Code (Clean Up)

| Location | Lines | Action |
|----------|-------|--------|
| `train_ddp_multidataset.py` | 130-154 | Remove commented `PrintTimings` |
| `train_ddp_multidataset.py` | 508 | Remove commented torch.compile |
| `train_ddp_multidataset.py` | 678 | Remove commented autocast |

**Estimated Time:** 15 minutes  
**Risk:** None  
**Benefit:** Cleaner code

---

## Dead Code Paths

### 1. Old Lightning Modules

**Evidence:**
```bash
# No active imports found
$ grep -r "from models.lightning import" training/
# (no results)

$ grep -r "MultiDatasetPLCore" training/
# (no results)
```

**Conclusion:** `models/lightning/` modules are NOT used by active training system.

### 2. Commented Callbacks

**Location:** `train_ddp_multidataset.py` lines 130-154

```python
# class PrintTimings(pl.Callback):
#     """
#     Prints a heartbeat every N steps during training.
#     """
#     ...
```

**Conclusion:** Dead code, can be removed.

### 3. Unused Imports in Old Modules

**In `models/lightning/multidataset.py`:**
- Imports `PLCoreVisionModel` but never uses it
- Imports matplotlib/wandb but minimal visualization

---

## Feature Comparison: Old vs New

| Feature | Old `MultiDatasetPLCore` | New `MultiDatasetModel` |
|---------|-------------------------|------------------------|
| **Data Format** | Dict-based batches | List-based batches |
| **Sampling** | Standard distributed | Contrast-weighted curriculum |
| **Regularization** | Basic support | Full system (L1, L2, group lasso) |
| **PC Modulator** | Basic support | Full auxiliary loss |
| **Pretrained Loading** | ❌ No | ✅ Yes |
| **Vision Freezing** | ❌ No | ✅ Yes |
| **Modulator-only** | ❌ No | ✅ Yes |
| **DDP Optimization** | Basic | Optimized (find_unused_parameters) |
| **LR Scheduling** | Simple | Cosine warmup |
| **Param Groups** | Single group | Core vs Head with different LRs |
| **Curriculum Learning** | ❌ No | ✅ Yes |
| **BPS Metrics** | Basic | Per-dataset + overall |

**Conclusion:** New system is significantly more advanced.

---

## Recommendations by Priority

### 🔴 High Priority (Do First)

1. **Remove Duplicates** (30 min, very low risk)
   - Delete `jake/multidataset_ddp/train_ddp_multidataset.py`
   - Delete duplicate shell scripts
   - Update shell scripts to use `training/train_ddp_multidataset.py`

2. **Clean Commented Code** (15 min, no risk)
   - Remove commented callbacks
   - Remove commented torch.compile
   - Fix misleading comments

### 🟡 Medium Priority (Do Soon)

3. **Deprecate Old Lightning Modules** (1-2 hours, low risk)
   - Move to `models/lightning_deprecated/`
   - Add deprecation warnings
   - Update checkpoint loading for backward compatibility

4. **Add Documentation** (2-3 hours, no risk)
   - Add docstrings to all classes
   - Create training guide
   - Document curriculum learning

### 🟢 Low Priority (Nice to Have)

5. **Extract Components** (1-2 days, moderate risk)
   - Move `MultiDatasetModel` to `training/lightning/multidataset_model.py`
   - Move `MultiDatasetDM` to `training/lightning/multidataset_dm.py`
   - Extract samplers, callbacks, schedulers

6. **Add Tests** (2-3 days, no risk)
   - Unit tests for components
   - Integration tests for training
   - Regression tests for performance

---

## Migration Strategy

### Option A: Conservative (Recommended)

1. ✅ Remove duplicates (Phase 1)
2. ✅ Deprecate old modules (Phase 2)
3. ⏸️ Keep current structure (don't extract components yet)
4. ✅ Add documentation and tests

**Pros:** Low risk, immediate benefit  
**Cons:** Monolithic structure remains

### Option B: Aggressive

1. ✅ Remove duplicates (Phase 1)
2. ✅ Deprecate old modules (Phase 2)
3. ✅ Extract all components (Phase 3)
4. ✅ Add documentation and tests

**Pros:** Clean, modular structure  
**Cons:** Higher risk, more time investment

### Option C: Hybrid (Best Balance)

1. ✅ Remove duplicates (Phase 1) - **Do now**
2. ✅ Deprecate old modules (Phase 2) - **Do now**
3. ✅ Add documentation - **Do now**
4. ⏸️ Extract components (Phase 3) - **Do later when needed**
5. ⏸️ Add tests - **Do incrementally**

**Pros:** Quick wins, low risk, flexibility  
**Cons:** Deferred refactoring

---

## Next Steps

### Immediate (This Week)

1. **Review this document** with team
2. **Decide on migration strategy** (A, B, or C)
3. **Create backup** of current codebase
4. **Execute Phase 1** (remove duplicates)

### Short-term (Next 2 Weeks)

5. **Execute Phase 2** (deprecate old modules)
6. **Add documentation** (training guide, docstrings)
7. **Test with existing experiments**

### Long-term (Next Month)

8. **Consider Phase 3** (extract components) if needed
9. **Add unit tests** incrementally
10. **Create training tutorial**

---

## Questions for Discussion

1. **Do we have old checkpoints that need to load?**
   - If yes, we need careful backward compatibility
   - If no, we can be more aggressive with cleanup

2. **Is `jake/multidataset_ddp/` directory still needed?**
   - If it's just a development sandbox, we can clean it up
   - If it has unique experiments, we should preserve them

3. **Should we support both old and new training systems?**
   - Probably not - new system is strictly better
   - But we need to handle old checkpoints

4. **What's the timeline for this refactoring?**
   - Phase 1: Can do today (30 min)
   - Phase 2: Can do this week (2-4 hours)
   - Phase 3: Needs planning (1-2 days)

5. **Do we want to move Lightning modules to `training/` or keep in `models/`?**
   - Recommendation: Move to `training/lightning/`
   - Rationale: Training-specific, not model architecture

---

## Conclusion

Your training system is **functionally excellent** but has **organizational issues**:

- ✅ **Strengths:** Advanced features, robust DDP, good performance
- ❌ **Weaknesses:** Code duplication, deprecated modules, monolithic structure
- 🎯 **Recommendation:** Start with low-risk cleanup (Phase 1-2), then evaluate if Phase 3 is needed

**Bottom Line:** You can get 80% of the benefit with 20% of the effort by just doing Phase 1-2.

