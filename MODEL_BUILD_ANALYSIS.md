# Model Building System Analysis

## Current Structure

```
models/
├── build.py              # 118 lines - Main entry point
├── factory.py            # 313 lines - Component factories
├── config.py             # 410 lines - Config system with defaults
├── config_loader.py      # 252 lines - YAML loading
└── config_builders.py    # 661 lines - Config builder functions ⚠️
```

**Total:** 1,754 lines

---

## File Analysis

### ✅ **build.py** (118 lines) - KEEP
**Purpose:** Main entry point for model building
- `build_model()` - Creates models from config
- `initialize_model_components()` - Weight initialization
- `get_name_from_config()` - Name generation

**Used by:**
- `training/pl_modules/multidataset_model.py`
- `models/checkpoint.py`
- All training scripts

**Status:** ✅ Essential, well-designed

---

### ✅ **factory.py** (313 lines) - KEEP
**Purpose:** Factory functions for creating components
- `create_frontend()` - Creates frontend modules
- `create_convnet()` - Creates convnet modules
- `create_recurrent()` - Creates recurrent modules
- `create_modulator()` - Creates modulator modules
- `create_readout()` - Creates readout modules

**Used by:**
- `models/build.py`
- `models/modules/models.py`

**Status:** ✅ Essential, clean design

---

### ✅ **config_loader.py** (252 lines) - KEEP
**Purpose:** YAML config loading and saving
- `load_config()` - Load model config from YAML
- `save_config()` - Save config to YAML
- `load_dataset_configs()` - Load dataset configs
- `validate_config()` - Config validation

**Used by:**
- `training/pl_modules/multidataset_model.py`
- `training/pl_modules/multidataset_dm.py`
- `models/checkpoint.py`
- `models/model_manager.py`

**Status:** ✅ Essential, actively used

---

### 🟡 **config.py** (410 lines) - SIMPLIFY
**Purpose:** Config system with defaults and validation
- `DEFAULT_CONFIGS` - Default values for all components
- `build_component_config()` - Build component configs
- `build_model_config()` - Build full model configs
- `merge_configs()` - Merge config dicts
- `get_component_defaults()` - Get defaults

**Used by:**
- `models/factory.py` - Only uses `build_component_config()`
- `models/__init__.py` - Exports functions

**Issues:**
- 410 lines mostly for defaults
- Only `build_component_config()` is actively used
- Other functions may be unused

**Recommendation:** 
- Keep defaults and `build_component_config()`
- Remove unused functions
- Could reduce to ~200 lines

---

### 🔴 **config_builders.py** (661 lines) - LIKELY UNUSED
**Purpose:** Programmatic config building functions
- `build_convblock_config()` - Build convblock configs
- `build_densenet_config()` - Build densenet configs
- `build_resnet_config()` - Build resnet configs
- `build_model_config()` - Build full model configs
- `build_unified_convnet_config()` - Build unified convnet configs
- 5 more builder functions...

**Used by:**
- ❌ **NO IMPORTS FOUND** in active code
- Only found in `models/config_builders.py` itself

**Status:** 🔴 **LIKELY DEAD CODE** - 661 lines doing nothing

**Recommendation:** 
- Move to `models/deprecated/` or delete
- These were probably used for programmatic config generation
- Now configs are loaded from YAML files instead

---

## Usage Pattern

### How Models Are Built (Current Flow)

```python
# 1. Load config from YAML
from models.config_loader import load_config
config = load_config('configs/model.yaml')

# 2. Build model
from models.build import build_model
model = build_model(config, dataset_configs)

# Inside build_model():
#   - Calls factory.create_frontend()
#   - Calls factory.create_convnet()
#   - Calls factory.create_modulator()
#   - Calls factory.create_readout()
#   - Assembles into ModularV1Model or MultiDatasetV1Model
```

### What's NOT Used

```python
# These are NOT used anywhere:
from models.config_builders import build_convblock_config  # ❌
from models.config_builders import build_densenet_config   # ❌
from models.config_builders import build_model_config      # ❌
```

---

## Simplification Opportunities

### 🎯 **Option 1: Conservative Cleanup (Recommended)**

**Remove:**
- `models/config_builders.py` (661 lines) → Move to deprecated

**Simplify:**
- `models/config.py` - Remove unused functions, keep defaults

**Result:**
- 1,754 lines → ~1,100 lines (37% reduction)
- No risk to active code
- Cleaner imports

**Effort:** 30 minutes

---

### 🎯 **Option 2: Aggressive Cleanup**

**Remove:**
- `models/config_builders.py` (661 lines) → Delete
- Unused functions in `models/config.py` → Delete

**Consolidate:**
- Merge `config.py` defaults into `factory.py`
- Single source of truth for component creation

**Result:**
- 1,754 lines → ~800 lines (54% reduction)
- Simpler mental model
- Less indirection

**Effort:** 2-3 hours

---

### 🎯 **Option 3: Full Refactor**

**Restructure:**
```
models/
├── build.py              # Main entry point
├── factory.py            # Component factories + defaults
├── config_loader.py      # YAML loading only
└── components/           # Component modules
    ├── frontend.py
    ├── convnet.py
    ├── modulator.py
    └── readout.py
```

**Result:**
- Cleaner separation
- Easier to test
- More maintainable

**Effort:** 1-2 days

---

## Verification Commands

### Check if config_builders.py is used:
```bash
# Search for imports
grep -r "from.*config_builders import" . --include="*.py" --exclude-dir=.git

# Search for function calls
grep -r "build_convblock_config\|build_densenet_config\|build_model_config" . --include="*.py" --exclude-dir=.git
```

### Check what uses config.py:
```bash
# Find all imports
grep -r "from.*config import" . --include="*.py" --exclude-dir=.git

# Find specific function usage
grep -r "build_component_config\|merge_configs\|get_component_defaults" . --include="*.py" --exclude-dir=.git
```

---

## Recommendation

### **Start with Option 1 (Conservative)**

1. **Verify config_builders.py is unused:**
   ```bash
   grep -r "config_builders" . --include="*.py" --exclude-dir=.git
   ```

2. **Move to deprecated:**
   ```bash
   mv models/config_builders.py models/deprecated/
   ```

3. **Test training:**
   ```bash
   python training/train_ddp_multidataset.py --help
   ```

4. **If successful, analyze config.py:**
   - Find which functions are actually used
   - Remove unused ones
   - Keep defaults

### **Benefits:**
- ✅ Remove 661 lines of dead code
- ✅ Zero risk (just moving, not deleting)
- ✅ Cleaner imports
- ✅ 30 minutes of work

---

## Summary

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `build.py` | 118 | ✅ Essential | Keep |
| `factory.py` | 313 | ✅ Essential | Keep |
| `config_loader.py` | 252 | ✅ Essential | Keep |
| `config.py` | 410 | 🟡 Simplify | Remove unused functions |
| `config_builders.py` | 661 | 🔴 Unused | Move to deprecated |
| **Total** | **1,754** | | **→ ~1,100 lines** |

**Potential savings:** 37% reduction with conservative cleanup

---

## Next Steps

1. Verify `config_builders.py` is unused
2. Move to deprecated folder
3. Analyze `config.py` for unused functions
4. Test training still works
5. Commit changes

Ready to proceed?

