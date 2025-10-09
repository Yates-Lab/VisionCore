# Dataset Config Inheritance System

## Overview

The dataset configuration system has been refactored to support **config inheritance** from a parent config to session-specific configs. This eliminates redundancy and makes it easier to manage multi-dataset training.

## Architecture

### Before (Old System)
- Each dataset had a **complete, standalone config file** with all parameters
- Training pointed to a **directory** containing multiple config files
- All configs with "base" in the name were excluded
- Each config was fully self-contained (lots of duplication)

### After (New System)
- One **parent config** specifies common parameters and lists sessions
- Each **session config** only specifies session-specific parameters (mainly `cids`)
- Training points to a **single parent config file**
- Session configs **inherit** all parameters from parent, with session values taking precedence

## File Structure

```
experiments/dataset_configs/
├── multi_basic_120_backimage_all.yaml    # Parent config
└── sessions/                              # Session configs directory
    ├── Allen_2022-02-16.yaml
    ├── Allen_2022-02-18.yaml
    ├── Logan_2020-02-29.yaml
    └── ...
```

## Parent Config Format

The parent config contains:
1. **`session_dir`**: Path to directory with session configs (relative to parent config)
2. **`sessions`**: List of session names to load
3. **All shared parameters**: `types`, `transforms`, `keys_lags`, `datafilters`, etc.

Example:
```yaml
# experiments/dataset_configs/multi_basic_120_backimage_all.yaml
types: [backimage]
session_dir: ./sessions
sessions: 
  - Allen_2022-02-16
  - Allen_2022-03-04
  - Logan_2020-02-29
  # ... more sessions

sampling:
  source_rate: 240
  target_rate: 120

keys_lags:
  robs: 0
  stim: [0, 1, 2, 3, ..., 24]
  behavior: 0
  dfs: 0

transforms:
  stim:
    source: stim
    ops:
      - pixelnorm: {}
      - unsqueeze: 0
    expose_as: stim
  # ... more transforms

datafilters:
  dfs:
    ops:
      - valid_nlags: {n_lags: 32}
    expose_as: dfs

train_val_split: 0.8
seed: 1002
```

## Session Config Format

Session configs are **minimal** and only specify:
- **`cids`** (required): List of cell IDs for this session
- **`session`** (optional): Session name (inferred from filename if missing)
- **`lab`** (optional): Lab name (defaults to 'yates')
- Any other session-specific overrides

Example:
```yaml
# experiments/dataset_configs/sessions/Allen_2022-02-16.yaml
session: Allen_2022-02-16
cids: [1, 4, 5, 8, 9, 12, 13, 14, ...]
visual: [1, 4, 5, 8, 9, 12, 13, 14, ...]
qcmissing: [3, 8, 9, 11, 17, 19, ...]
qccontam: [3, 6, 7, 8, 9, 10, ...]
lab: yates
```

## Config Merging

The system uses **deep merging**:
1. Start with parent config (excluding `session_dir` and `sessions`)
2. Recursively merge session config on top
3. Session values override parent values
4. Nested dicts are merged recursively

Example:
```python
# Parent has:
transforms:
  stim:
    source: stim
    ops: [pixelnorm, unsqueeze]

# Session overrides:
transforms:
  stim:
    ops: [fftwhitening, pixelnorm]  # Override ops

# Result:
transforms:
  stim:
    source: stim  # Inherited from parent
    ops: [fftwhitening, pixelnorm]  # Overridden by session
```

## Usage

### Training Command

**Old way:**
```bash
python training/train_ddp_multidataset.py \
    --model_config configs_multi/core_res.yaml \
    --dataset_configs_path experiments/dataset_configs/  # Directory
```

**New way:**
```bash
python training/train_ddp_multidataset.py \
    --model_config configs_multi/core_res.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml  # File
```

### In Code

```python
from models.config_loader import load_dataset_configs

# Load all session configs from parent
configs = load_dataset_configs('experiments/dataset_configs/multi_basic_120_backimage_all.yaml')

# Returns list of merged configs, one per session
print(f"Loaded {len(configs)} sessions")
for cfg in configs:
    print(f"  - {cfg['session']}: {len(cfg['cids'])} cells")
```

## Implementation Details

### Modified Files

1. **`models/config_loader.py`**
   - Rewrote `load_dataset_configs()` to load parent + sessions
   - Added `_deep_merge_configs()` helper for recursive merging
   - Removed backward compatibility (no longer supports directory listing)

2. **`training/pl_modules/multidataset_dm.py`**
   - Updated `setup()` to pass parent config path directly
   - Changed dataset naming to use session names
   - Updated docstrings

3. **`training/pl_modules/multidataset_model.py`**
   - Updated `__init__()` to pass parent config path directly
   - Changed dataset naming to use session names

4. **`training/train_ddp_multidataset.py`**
   - Updated argument help text for `--dataset_configs_path`

### Key Functions

**`load_dataset_configs(parent_config_path)`**
- Loads parent config YAML
- Validates required fields (`session_dir`, `sessions`)
- Resolves session directory path (relative to parent config)
- Loads each session config
- Merges parent + session configs
- Adds metadata (`_weight`, `_config_path`, `_parent_config_path`)
- Validates merged configs
- Returns list of merged configs

**`_deep_merge_configs(base, override)`**
- Recursively merges two config dicts
- Override values take precedence
- Nested dicts are merged recursively
- Lists and primitives are replaced (not merged)

## Benefits

1. **DRY Principle**: No duplication of common parameters across sessions
2. **Easy Updates**: Change transforms/filters in one place (parent config)
3. **Clear Intent**: Session configs show only what's unique per session
4. **Maintainability**: Easier to add new sessions (just specify `cids`)
5. **Flexibility**: Can still override any parameter at session level if needed

## Migration Guide

To migrate existing configs:

1. **Create parent config**: Take one existing config as template
2. **Add session fields**: Add `session_dir` and `sessions` list
3. **Create session configs**: Extract session-specific fields (`cids`, `session`, etc.)
4. **Update training scripts**: Point to parent config file instead of directory

Example migration:
```bash
# Old structure
dataset_configs/
├── Allen_2022-02-16.yaml  # Full config
├── Allen_2022-02-18.yaml  # Full config
└── Logan_2020-02-29.yaml  # Full config

# New structure
dataset_configs/
├── multi_basic_120.yaml   # Parent config
└── sessions/
    ├── Allen_2022-02-16.yaml  # Just cids
    ├── Allen_2022-02-18.yaml  # Just cids
    └── Logan_2020-02-29.yaml  # Just cids
```

## Testing

Run the test script to verify config loading:
```bash
python test_config_loading.py
```

This will:
- Load the parent config
- Show all loaded sessions
- Display details of the first merged config
- Verify inheritance is working correctly

