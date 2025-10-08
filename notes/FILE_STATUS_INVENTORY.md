# VisionCore Training Files - Status Inventory

## Legend

- ✅ **ACTIVE** - Currently used, keep as-is
- 🔴 **DUPLICATE** - Remove immediately
- 🟡 **DEPRECATED** - Move to deprecated folder
- 🟢 **GOOD** - Well-organized, no changes needed
- ⚠️ **REVIEW** - Needs manual review before action

---

## Training Scripts

### Active Training System

| File | Status | Lines | Action |
|------|--------|-------|--------|
| `training/train_ddp_multidataset.py` | ✅ ACTIVE | 1019 | Keep (canonical) |
| `training/regularizers.py` | ✅ ACTIVE | ~200 | Keep |

### Duplicate Training Scripts

| File | Status | Lines | Action |
|------|--------|-------|--------|
| `jake/multidataset_ddp/train_ddp_multidataset.py` | 🔴 DUPLICATE | 1048 | **DELETE** |

**Difference:** 29 lines (likely minor edits or whitespace)

---

## Shell Scripts

### Active Shell Scripts (experiments/)

| File | Status | Purpose | Action |
|------|--------|---------|--------|
| `experiments/run_all_models.sh` | ✅ ACTIVE | Main training runner | Update path |
| `experiments/run_all_models_backimage.sh` | ✅ ACTIVE | Backimage experiments | Update path |
| `experiments/run_all_models_cones.sh` | ✅ ACTIVE | Cone experiments | Update path |
| `experiments/run_all_models_gaborium.sh` | ✅ ACTIVE | Gaborium experiments | Update path |
| `experiments/run_all_models_pretraining.sh` | ✅ ACTIVE | Pretraining experiments | Update path |

**Update:** Change `python train_ddp_multidataset.py` → `python training/train_ddp_multidataset.py`

### Duplicate Shell Scripts (jake/multidataset_ddp/)

| File | Status | Action |
|------|--------|--------|
| `jake/multidataset_ddp/run_all_models.sh` | 🔴 DUPLICATE | **DELETE** |
| `jake/multidataset_ddp/run_all_models_backimage.sh` | 🔴 DUPLICATE | **DELETE** |
| `jake/multidataset_ddp/run_all_models_cones.sh` | 🔴 DUPLICATE | **DELETE** |
| `jake/multidataset_ddp/run_all_models_gaborium.sh` | 🔴 DUPLICATE | **DELETE** |
| `jake/multidataset_ddp/run_all_models_pretraining.sh` | ⚠️ REVIEW | Check if unique |

---

## Lightning Modules

### Deprecated Lightning Modules (models/lightning/)

| File | Status | Lines | Used By | Action |
|------|--------|-------|---------|--------|
| `models/lightning/__init__.py` | 🟡 DEPRECATED | 8 | Checkpoint loading | Update with deprecation warning |
| `models/lightning/core.py` | 🟡 DEPRECATED | 367 | Old checkpoints only | Move to `models/lightning_deprecated/` |
| `models/lightning/multidataset.py` | 🟡 DEPRECATED | 428 | Old checkpoints only | Move to `models/lightning_deprecated/` |

**Evidence of deprecation:**
```bash
# No active imports in training scripts
$ grep -r "from models.lightning import" training/
# (no results)

# Only used in checkpoint loading
$ grep -r "PLCoreVisionModel" .
models/checkpoint.py:    from .lightning import PLCoreVisionModel
models/model_manager.py: from .lightning import PLCoreVisionModel
```

### Active Lightning Modules (inline in train_ddp_multidataset.py)

| Component | Lines | Status | Action |
|-----------|-------|--------|--------|
| `MultiDatasetModel` | 493-903 | ✅ ACTIVE | Keep (or extract to `training/lightning/`) |
| `MultiDatasetDM` | 348-488 | ✅ ACTIVE | Keep (or extract to `training/lightning/`) |

---

## Supporting Components (inline in train_ddp_multidataset.py)

| Component | Lines | Status | Action |
|-----------|-------|--------|--------|
| `LinearWarmupCosineAnnealingLR` | 34-51 | ✅ ACTIVE | Keep (or extract to `training/schedulers.py`) |
| `Heartbeat` | 75-94 | 🟡 COMMENTED | Remove (lines 130-154 are commented version) |
| `EpochHeartbeat` | 96-127 | ✅ ACTIVE | Keep (or extract to `training/callbacks.py`) |
| `CurriculumCallback` | 157-167 | ✅ ACTIVE | Keep (or extract to `training/callbacks.py`) |
| `ContrastWeightedSampler` | 198-343 | ✅ ACTIVE | Keep (or extract to `training/samplers.py`) |
| `cast_stim` | 172-174 | ✅ ACTIVE | Keep (or extract to `training/utils.py`) |
| `Float32View` | 176-186 | ✅ ACTIVE | Keep (or extract to `training/datasets.py`) |
| `group_collate` | 188-192 | ✅ ACTIVE | Keep (or extract to `training/utils.py`) |

---

## Commented/Dead Code

| Location | Lines | Status | Action |
|----------|-------|--------|--------|
| `train_ddp_multidataset.py` | 130-154 | 🔴 DEAD CODE | **DELETE** (commented `PrintTimings`) |
| `train_ddp_multidataset.py` | 508 | 🔴 DEAD CODE | **DELETE** (commented torch.compile) |
| `train_ddp_multidataset.py` | 678 | 🔴 DEAD CODE | **DELETE** (commented autocast) |

---

## Model Configs

### Active Configs (configs/)

| File | Status | Action |
|------|--------|--------|
| `configs/learned_res_small.yaml` | ✅ ACTIVE | Keep |
| `configs/learned_res_small_gru.yaml` | ✅ ACTIVE | Keep |
| `configs/learned_res_small_pc.yaml` | ✅ ACTIVE | Keep |
| `configs/modulator_only_convgru.yaml` | ✅ ACTIVE | Keep |
| ... (many more) | ✅ ACTIVE | Keep |

**Note:** All configs in `configs/` appear to be actively used.

---

## Other Files in jake/multidataset_ddp/

| File | Status | Action |
|------|--------|--------|
| `jake/multidataset_ddp/gratings_cross_model_analysis_120.py` | ⚠️ REVIEW | Check if unique analysis |
| `jake/multidataset_ddp/eval_stack_multidataset.py` | ⚠️ REVIEW | Check if unique eval code |
| `jake/multidataset_ddp/eval_stack_utils.py` | ⚠️ REVIEW | Check if unique utilities |

**Recommendation:** Review these files to see if they contain unique code or are duplicates.

---

## Summary Statistics

### Files to Delete (Phase 1)
- 1 duplicate training script
- 4-5 duplicate shell scripts
- 3 sections of commented code

**Total:** ~6 files + 3 code sections

### Files to Move (Phase 2)
- 2 deprecated Lightning modules
- 1 `__init__.py` to update

**Total:** 3 files

### Files to Extract (Phase 3 - Optional)
- 2 Lightning modules (MultiDatasetModel, MultiDatasetDM)
- 1 sampler (ContrastWeightedSampler)
- 3 callbacks (Heartbeat, EpochHeartbeat, CurriculumCallback)
- 1 scheduler (LinearWarmupCosineAnnealingLR)
- 3 utilities (cast_stim, Float32View, group_collate)

**Total:** 10 components → 5-6 new files

---

## Dependency Graph

```
experiments/*.sh
    ↓
training/train_ddp_multidataset.py
    ├── MultiDatasetDM
    │   ├── Float32View
    │   ├── ContrastWeightedSampler
    │   └── group_collate
    ├── MultiDatasetModel
    │   ├── training/regularizers.py
    │   └── models/losses.py
    ├── EpochHeartbeat
    ├── CurriculumCallback
    └── LinearWarmupCosineAnnealingLR

models/checkpoint.py
    └── models/lightning_deprecated/
        ├── PLCoreVisionModel
        └── MultiDatasetPLCore
```

---

## Risk Assessment

### Low Risk (Safe to do now)
- ✅ Delete duplicate `train_ddp_multidataset.py`
- ✅ Delete duplicate shell scripts
- ✅ Update shell script paths
- ✅ Remove commented code

**Estimated time:** 30 minutes  
**Risk level:** Very Low  
**Rollback:** `git checkout -- .`

### Medium Risk (Test carefully)
- ⚠️ Move Lightning modules to deprecated
- ⚠️ Update checkpoint loading
- ⚠️ Add deprecation warnings

**Estimated time:** 2-4 hours  
**Risk level:** Medium  
**Rollback:** `git checkout -- models/`

### High Risk (Plan carefully)
- 🔴 Extract components to separate modules
- 🔴 Refactor training script
- 🔴 Update all imports

**Estimated time:** 1-2 days  
**Risk level:** High  
**Rollback:** Full git reset

---

## Verification Commands

### Check for duplicates
```bash
# Find duplicate training scripts
find . -name "train_ddp_multidataset.py" -type f

# Find duplicate shell scripts
find . -name "run_all_models*.sh" -type f
```

### Check for references
```bash
# Check what imports old Lightning modules
grep -r "from models.lightning import" . --exclude-dir=.git

# Check what uses PLCoreVisionModel
grep -r "PLCoreVisionModel" . --exclude-dir=.git

# Check what uses MultiDatasetPLCore
grep -r "MultiDatasetPLCore" . --exclude-dir=.git
```

### Check for commented code
```bash
# Find commented class definitions
grep -n "^# class" training/train_ddp_multidataset.py

# Find commented torch.compile
grep -n "# self.model = torch.compile" training/train_ddp_multidataset.py
```

---

## Cleanup Checklist

### Phase 1: Immediate Cleanup
- [ ] Run `bash cleanup_duplicates.sh --dry-run`
- [ ] Review output
- [ ] Run `bash cleanup_duplicates.sh`
- [ ] Test: `bash experiments/run_all_models.sh` (with small config)
- [ ] Remove commented code (lines 130-154, 508, 678)
- [ ] Commit: `git commit -m "Phase 1: Remove duplicates and dead code"`

### Phase 2: Deprecation
- [ ] Create `models/lightning_deprecated/`
- [ ] Move `core.py` and `multidataset.py`
- [ ] Update `models/lightning/__init__.py` with deprecation warnings
- [ ] Update `models/checkpoint.py` for backward compatibility
- [ ] Test loading old checkpoint (if available)
- [ ] Commit: `git commit -m "Phase 2: Deprecate old Lightning modules"`

### Phase 3: Extraction (Optional)
- [ ] Create `training/lightning/`
- [ ] Extract `MultiDatasetModel` → `training/lightning/multidataset_model.py`
- [ ] Extract `MultiDatasetDM` → `training/lightning/multidataset_dm.py`
- [ ] Extract samplers → `training/samplers.py`
- [ ] Extract callbacks → `training/callbacks.py`
- [ ] Extract schedulers → `training/schedulers.py`
- [ ] Update `training/train_ddp_multidataset.py` to import from new locations
- [ ] Test full training pipeline
- [ ] Commit: `git commit -m "Phase 3: Extract components to modules"`

---

## Final State (After All Phases)

```
training/
├── train_ddp_multidataset.py      (~200 lines - CLI only)
├── regularizers.py                (existing)
├── lightning/
│   ├── __init__.py
│   ├── multidataset_model.py      (MultiDatasetModel)
│   └── multidataset_dm.py         (MultiDatasetDM + utilities)
├── samplers.py                     (ContrastWeightedSampler)
├── callbacks.py                    (Heartbeat, EpochHeartbeat, CurriculumCallback)
└── schedulers.py                   (LinearWarmupCosineAnnealingLR)

models/
├── lightning_deprecated/           (for old checkpoint loading)
│   ├── __init__.py
│   ├── core.py                    (PLCoreVisionModel)
│   └── multidataset.py            (MultiDatasetPLCore)
└── ...

experiments/
├── run_all_models.sh              (updated paths)
├── run_all_models_backimage.sh    (updated paths)
├── run_all_models_cones.sh        (updated paths)
├── run_all_models_gaborium.sh     (updated paths)
└── run_all_models_pretraining.sh  (updated paths)

jake/multidataset_ddp/
└── (cleaned up - only unique files remain)
```

---

## Questions Before Cleanup

1. **Are there any running experiments?**
   - ⚠️ Wait for them to finish before cleanup

2. **Do we have old checkpoints to test?**
   - ✅ Yes → Test Phase 2 carefully
   - ❌ No → Can skip backward compatibility

3. **Is jake/multidataset_ddp/ a development sandbox?**
   - ✅ Yes → Safe to clean up duplicates
   - ❌ No → Review unique files first

4. **When should we do this?**
   - 🏃 Now → Do Phase 1 (30 min)
   - 📅 This week → Do Phase 1-2 (4 hours)
   - 🗓️ This month → Do Phase 1-3 (2 days)

