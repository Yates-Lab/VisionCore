# Current Data Casting Flow Analysis

## The Complete Flow

### 1. **Dataset Loading** (`prepare_data` in `models/data/loading.py`)
```python
def prepare_data(dataset_config, strict=True):
    # Load datasets (uint8 stimulus)
    # Apply transforms (converts to float32)
    # Combine into train/val datasets
    return train_dset, val_dset, dataset_config
    # ← Returns CombinedEmbeddedDataset with float32 stim
```

**Current state after prepare_data:**
- Stimulus: **float32** (after transforms)
- Responses: **float32**
- Behavior: **float32**

### 2. **Pixel Norm Removal** (`multidataset_dm.py` line 112)
```python
cfg, norm_removed = remove_pixel_norm(cfg)
```

**What this does:**
- Removes `pixelnorm` transform from config if present
- Returns `norm_removed=True` if pixelnorm was in the config
- This tells `Float32View` whether to apply normalization later

### 3. **Wrapping with Float32View** (`multidataset_dm.py` line 120-121)
```python
self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
self.val_dsets[name] = Float32View(va, norm_removed, float16=False)
```

**What Float32View does:**
```python
class Float32View:
    def __getitem__(self, idx):
        it = self.base[idx]
        dtype = torch.bfloat16 if self.float16 else torch.float32
        
        # Cast stimulus with normalization
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=dtype)
        # ↑ This expects uint8 and applies (x - 127) / 255
        
        it["robs"] = it["robs"].to(dtype)
        if "behavior" in it:
            it["behavior"] = it["behavior"].to(dtype)
        return it
```

### 4. **The Problem!**

**Current behavior:**
- `prepare_data` returns **float32** stimulus (after transforms)
- `Float32View` expects **uint8** stimulus
- `cast_stim` does: `x.to(dtype)` then `(x - 127) / 255` if `norm_removed=True`

**This means:**
- If stim is already float32 from transforms, `cast_stim` will:
  1. Convert float32 → float32 (no-op)
  2. Apply `(x - 127) / 255` which is **WRONG** for already-normalized data!

**Wait... let me check if transforms keep data as uint8:**

---

## Checking Transform Behavior

Looking at `models/data/transforms.py`:

### dacones transform (line 219-226):
```python
def dacones(x):
    cones = DAModel(**cfg)
    dtype = x.dtype
    y = cones(x.unsqueeze(0).float()).squeeze(0).permute(1,0,2,3)
    if dtype == torch.uint8:
        y /= 2.5 
        y *= 255
        y = y.clamp(0, 255).to(torch.uint8)  # ← Converts back to uint8!
    else:
        y = y.to(dtype)
    return y
```

**This transform preserves uint8!**

### pixelnorm transform (line 149-151):
```python
def pixelnorm(x: torch.Tensor):
    return (x.float() - 127) / 255  # ← Converts to float32
```

**This converts to float32!**

---

## The Current System Actually Works Like This:

### **Path 1: No transforms (or uint8-preserving transforms)**
```
1. Load data as uint8
2. Transforms keep as uint8 (or no transforms)
3. prepare_data returns uint8 stimulus
4. Float32View.__getitem__:
   - cast_stim converts uint8 → float32
   - Applies (x - 127) / 255 normalization
5. GPU gets normalized float32
```

### **Path 2: With pixelnorm transform**
```
1. Load data as uint8
2. pixelnorm transform converts to float32 and normalizes
3. prepare_data returns float32 stimulus (already normalized)
4. remove_pixel_norm removes pixelnorm from config, sets norm_removed=False
5. Float32View.__getitem__:
   - cast_stim converts float32 → float32 (no-op)
   - Does NOT apply normalization (norm_removed=False)
6. GPU gets normalized float32
```

**So the system is already smart!**

---

## Where Casting Could Happen

### **Option 1: At the end of prepare_data** ⭐
```python
def prepare_data(dataset_config, strict=True):
    # ... existing code ...
    
    # NEW: Cast to storage dtype
    storage_dtype = dataset_config.get('storage_dtype', None)
    if storage_dtype == 'bfloat16':
        train_dset.cast(torch.bfloat16, target_keys=['stim'])
        val_dset.cast(torch.bfloat16, target_keys=['stim'])
    
    return train_dset, val_dset, dataset_config
```

**Pros:**
- ✅ Happens once per dataset load
- ✅ Before wrapping with Float32View
- ✅ Clean separation of concerns

**Cons:**
- ⚠️ Need to update Float32View to handle bfloat16 input

### **Option 2: In Float32View wrapper**
```python
class Float32View:
    def __init__(self, base, norm_removed, float16=False, storage_dtype='uint8'):
        self.storage_dtype = storage_dtype
        # Cast the entire dataset here
        if storage_dtype == 'bfloat16':
            base.cast(torch.bfloat16, target_keys=['stim'])
```

**Pros:**
- ✅ Centralized in one place

**Cons:**
- ❌ Happens in __init__, not lazy
- ❌ Modifies the base dataset (side effects)

### **Option 3: Already supported via DictDataset.cast()!**

**Check if this already works:**
```python
# In multidataset_dm.py, after prepare_data:
tr, va, _ = prepare_data(cfg, strict=False)

# NEW: Cast to bfloat16 before wrapping
if cfg.get('storage_dtype') == 'bfloat16':
    tr.cast(torch.bfloat16, target_keys=['stim'])
    va.cast(torch.bfloat16, target_keys=['stim'])

self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
```

**Then update Float32View to handle bfloat16:**
```python
def __getitem__(self, idx):
    it = self.base[idx]
    dtype = torch.bfloat16 if self.float16 else torch.float32
    
    # Smart casting based on input dtype
    if it["stim"].dtype == torch.uint8:
        # Legacy path: uint8 → float with normalization
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=dtype)
    else:
        # New path: already float (bfloat16/float32) → just convert dtype
        it["stim"] = it["stim"].to(dtype)
        # Normalization already applied
    
    it["robs"] = it["robs"].to(dtype)
    if "behavior" in it:
        it["behavior"] = it["behavior"].to(dtype)
    return it
```

---

## The Answer: It's Almost Already Supported!

### What exists:
✅ `DictDataset.cast()` method  
✅ `CombinedEmbeddedDataset.cast()` method  
✅ `Float32View` wrapper  

### What's missing:
❌ Config parameter to trigger casting  
❌ Float32View doesn't handle non-uint8 input correctly  

---

## Minimal Implementation

### 1. Update Float32View (training/utils.py)
```python
def __getitem__(self, idx):
    it = self.base[idx]
    dtype = torch.bfloat16 if self.float16 else torch.float32
    
    # Smart casting based on input dtype
    if it["stim"].dtype == torch.uint8:
        # uint8 path: apply normalization
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=dtype)
    elif it["stim"].dtype.is_floating_point:
        # Already float (from transforms or storage_dtype)
        it["stim"] = it["stim"].to(dtype)
    else:
        raise ValueError(f"Unexpected stim dtype: {it['stim'].dtype}")
    
    it["robs"] = it["robs"].to(dtype)
    if "behavior" in it:
        it["behavior"] = it["behavior"].to(dtype)
    return it
```

### 2. Add casting in multidataset_dm.py
```python
# After prepare_data
tr, va, _ = prepare_data(cfg, strict=False)

# NEW: Cast to storage dtype if specified
storage_dtype = cfg.get('storage_dtype', None)
if storage_dtype == 'bfloat16':
    tr.cast(torch.bfloat16, target_keys=['stim'])
    va.cast(torch.bfloat16, target_keys=['stim'])
elif storage_dtype == 'float16':
    tr.cast(torch.float16, target_keys=['stim'])
    va.cast(torch.float16, target_keys=['stim'])

self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
```

### 3. Add to dataset config YAML
```yaml
# configs/datasets/session1.yaml
session: "Subject_2024-01-15"
storage_dtype: bfloat16  # ← NEW: Options: bfloat16, float16, float32

transforms:
  stim:
    - fftwhitening: {}
```

---

## Summary

**Current system:**
- ✅ Already has `.cast()` methods on datasets
- ✅ Already handles uint8 → float32 conversion
- ⚠️ Float32View assumes uint8 input (needs small fix)

**What we need:**
1. ✅ Update Float32View to handle float inputs (5 lines)
2. ✅ Add casting logic in multidataset_dm.py (5 lines)
3. ✅ Add `storage_dtype` to config schema (1 line)

**Total changes:** ~10 lines of code!

The infrastructure is already there, we just need to wire it up!

