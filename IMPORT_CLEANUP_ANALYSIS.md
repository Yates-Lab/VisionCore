# Import Cleanup Analysis - models/data/

## Problem

The `models/data/` folder was copied from DataYatesV1 and still uses relative imports that assume it's in DataYatesV1's structure:
- `from ..general import ensure_tensor` → Expects `models/general.py` (doesn't exist)
- `from ..torch import set_seeds` → Expects `models/torch.py` (doesn't exist)
- `from ..io import get_session` → Expects `models/io.py` (doesn't exist)
- `from ..rf import calc_sta` → Expects `models/rf.py` (doesn't exist)

## What Exists in VisionCore

### ✅ models/utils/general.py
Contains:
- `ensure_tensor()` ✅
- `min_max_norm()`
- `ensure_ndarray()`
- `nd_cut()`, `nd_paste()`
- `explore_hdf5()`
- `convert_time_to_samples()`
- `event_triggered_analog()`
- `get_clock_functions()`
- `get_valid_dfs()`
- `get_optimizer()`

### ❌ Missing Utilities
These don't exist in VisionCore:
- `set_seeds()` - from DataYatesV1.utils.torch
- `get_memory_footprints_str()` - from DataYatesV1.utils.torch
- `get_session()` - from DataYatesV1.utils.io
- `calc_sta()` - from DataYatesV1.utils.rf

---

## Files That Need Fixing

### 1. models/data/datasets.py
**Current imports:**
```python
from ..torch import get_memory_footprints_str, set_seeds
from ..general import ensure_tensor
```

**Should be:**
```python
from DataYatesV1.utils.torch import get_memory_footprints_str, set_seeds
from ..utils.general import ensure_tensor
```

### 2. models/data/loading.py
**Current imports:**
```python
from ..general import ensure_tensor
from ..io import get_session
from ..rf import calc_sta
```

**Should be:**
```python
from ..utils.general import ensure_tensor
from DataYatesV1.utils.io import get_session
from DataYatesV1.utils.rf import calc_sta
```

### 3. models/data/splitting.py
**Current imports:**
```python
from ..torch import set_seeds
```

**Should be:**
```python
from DataYatesV1.utils.torch import set_seeds
```

---

## Other Files Using DataYatesV1 Imports (Already Correct)

### ✅ models/build.py
```python
from DataYatesV1.utils.general import ensure_tensor  # ❌ Should use local
```
**Should be:**
```python
from .utils.general import ensure_tensor
```

### ✅ models/modules/frontend.py
```python
from DataYatesV1.utils.modeling.bases import make_raised_cosine_basis  # ✅ OK (not in VisionCore)
from DataYatesV1.utils.general import ensure_tensor  # ❌ Should use local
```
**Should be:**
```python
from DataYatesV1.utils.modeling.bases import make_raised_cosine_basis
from ..utils.general import ensure_tensor
```

### ✅ models/utils/eval.py
```python
from DataYatesV1.utils.general import ensure_tensor  # ❌ Should use local
from DataYatesV1.utils.data.datasets import CombinedEmbeddedDataset  # ✅ OK (not in VisionCore)
```
**Should be:**
```python
from .general import ensure_tensor
from DataYatesV1.utils.data.datasets import CombinedEmbeddedDataset
```

---

## Summary of Changes Needed

### Fix models/data/ imports (broken relative imports)
1. `models/data/datasets.py` - Fix 2 imports
2. `models/data/loading.py` - Fix 3 imports
3. `models/data/splitting.py` - Fix 1 import

### Use local ensure_tensor (optimization)
4. `models/build.py` - Use local import
5. `models/modules/frontend.py` - Use local import
6. `models/utils/eval.py` - Use local import

---

## Implementation Plan

### Phase 1: Fix Broken Imports (Critical)
These are currently broken because the files don't exist:

**models/data/datasets.py:**
```python
# Line 4-5
from DataYatesV1.utils.torch import get_memory_footprints_str, set_seeds
from ..utils.general import ensure_tensor
```

**models/data/loading.py:**
```python
# Line 22-24
from ..utils.general import ensure_tensor
from DataYatesV1.utils.io import get_session
from DataYatesV1.utils.rf import calc_sta
```

**models/data/splitting.py:**
```python
# Line 11
from DataYatesV1.utils.torch import set_seeds
```

### Phase 2: Optimize to Use Local Imports (Optional)
These work but should use local versions:

**models/build.py:**
```python
# Line 8
from .utils.general import ensure_tensor
```

**models/modules/frontend.py:**
```python
# Line 17
from ..utils.general import ensure_tensor
```

**models/utils/eval.py:**
```python
# Line 3
from .general import ensure_tensor
```

---

## Verification Commands

### Check current broken imports:
```bash
# These should fail because files don't exist
python -c "from models.data.datasets import DictDataset"
python -c "from models.data.loading import get_embedded_datasets"
python -c "from models.data.splitting import split_inds_by_trial"
```

### After fix, verify:
```bash
# These should work
python -c "from models.data.datasets import DictDataset; print('✓')"
python -c "from models.data.loading import get_embedded_datasets; print('✓')"
python -c "from models.data.splitting import split_inds_by_trial; print('✓')"
```

---

## Risk Assessment

### Phase 1 (Fix Broken Imports)
- **Risk:** ✅ Zero - These imports are currently broken
- **Impact:** ✅ High - Makes models/data/ actually usable
- **Effort:** 5 minutes

### Phase 2 (Optimize Local Imports)
- **Risk:** ✅ Low - Just changing import paths
- **Impact:** 🟡 Medium - Cleaner, less dependency on DataYatesV1
- **Effort:** 5 minutes

---

## Ready to Execute?

This is a **critical fix** - the models/data/ imports are currently broken and won't work if anyone tries to use them.

Want me to fix these imports now?

