# Config Inheritance System - Example

## Creating a New Parent Config

Here's a step-by-step example of creating a new parent config for multi-dataset training.

### Step 1: Create the Parent Config

Create a file like `experiments/dataset_configs/my_experiment.yaml`:

```yaml
# experiments/dataset_configs/my_experiment.yaml

# Specify where session configs live (relative to this file)
session_dir: ./sessions

# List which sessions to include in training
sessions:
  - Allen_2022-02-16
  - Allen_2022-02-18
  - Logan_2020-02-29
  - Logan_2020-03-02

# Dataset types to load
types: [backimage]

# Sampling configuration
sampling:
  source_rate: 240
  target_rate: 120

# Keys and temporal lags
keys_lags:
  robs: 0
  stim: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  behavior: 0
  dfs: 0

# Transform pipeline
transforms:
  stim:
    source: stim
    ops:
      - pixelnorm: {}
      - unsqueeze: 0
    expose_as: stim
  
  robs:
    source: robs
    ops:
      - smooth: 
         - type: gaussian
         - params: 1.0
    expose_as: robs
  
  eye_vel:
    source: eyepos
    ops:
      - diff: {axis: 0}
      - maxnorm: {}
      - symlog: {}
      - temporal_basis:
          num_delta_funcs: 0
          num_cosine_funcs: 10
          history_bins: 50
          causal: false
          log_spacing: false
          peak_range_ms: [30, 200]
          normalize: true
      - splitrelu:
          split_dim: 1
          trainable_gain: false
    expose_as: behavior
    concatenate: true
  
  eye_pos:
    source: eyepos
    ops: []
    expose_as: behavior
    concatenate: true

# Data filters
datafilters:
  dfs:
    ops:
      - valid_nlags: {n_lags: 32}
    expose_as: dfs

# Train/validation split
train_val_split: 0.8
seed: 1002
```

### Step 2: Ensure Session Configs Exist

Make sure you have minimal session configs in `experiments/dataset_configs/sessions/`:

```yaml
# experiments/dataset_configs/sessions/Allen_2022-02-16.yaml
session: Allen_2022-02-16
cids: [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...]
visual: [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...]
qcmissing: [3, 8, 9, 11, 17, 19, 20, 22, 23, 24, ...]
qccontam: [3, 6, 7, 8, 9, 10, 11, 12, 15, 19, ...]
lab: yates
```

### Step 3: Run Training

```bash
python training/train_ddp_multidataset.py \
    --model_config configs_multi/core_res.yaml \
    --dataset_configs_path experiments/dataset_configs/my_experiment.yaml \
    --max_datasets 4 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_gpus 2
```

## Advanced: Session-Specific Overrides

If you need to override parameters for a specific session, just add them to the session config:

```yaml
# experiments/dataset_configs/sessions/special_session.yaml
session: Allen_2022-02-16
cids: [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...]

# Override the types for this session only
types: [backimage, gratings]

# Override sampling for this session
sampling:
  source_rate: 240
  target_rate: 60  # Different target rate!

# Add session-specific transform
transforms:
  stim:
    ops:
      - fftwhitening: {f_0: 0.4, n: 4}  # Override stim transform
      - pixelnorm: {}
      - unsqueeze: 0
```

The merged config will have:
- `types: [backimage, gratings]` (from session)
- `sampling.target_rate: 60` (from session)
- `transforms.stim.ops` with fftwhitening (from session)
- All other fields from parent config

## Common Patterns

### Pattern 1: Different Transform Pipelines

Create multiple parent configs for different preprocessing:

```
experiments/dataset_configs/
├── multi_basic_pixelnorm.yaml      # Just pixelnorm
├── multi_basic_fftwhitening.yaml   # FFT whitening + pixelnorm
├── multi_basic_temporal.yaml       # Temporal basis expansion
└── sessions/
    └── ...
```

### Pattern 2: Different Session Subsets

Create parent configs for different session groups:

```
experiments/dataset_configs/
├── allen_sessions_only.yaml        # Just Allen lab sessions
├── logan_sessions_only.yaml        # Just Logan lab sessions
├── all_sessions.yaml               # All sessions
└── sessions/
    └── ...
```

### Pattern 3: Different Stimulus Types

```yaml
# multi_backimage.yaml
types: [backimage]
sessions: [Allen_2022-02-16, Allen_2022-02-18, ...]

# multi_gratings.yaml
types: [gratings]
sessions: [Allen_2022-02-16, Allen_2022-02-18, ...]

# multi_all_stim.yaml
types: [backimage, gratings, gaborium]
sessions: [Allen_2022-02-16, Allen_2022-02-18, ...]
```

## Testing Your Config

Before running full training, test your config:

```python
from models.config_loader import load_dataset_configs

# Load configs
configs = load_dataset_configs('experiments/dataset_configs/my_experiment.yaml')

# Check what was loaded
print(f"Loaded {len(configs)} sessions")
for cfg in configs:
    print(f"  {cfg['session']}: {len(cfg['cids'])} cells, types={cfg['types']}")

# Inspect first config
import yaml
print(yaml.dump(configs[0], default_flow_style=False))
```

Or use the test script:

```bash
# Edit test_config_loading.py to point to your config
python test_config_loading.py
```

## Troubleshooting

### Error: "Session config not found"

Make sure:
1. Session name in parent config matches filename (without `.yaml`)
2. `session_dir` path is correct (relative to parent config)
3. Session file exists in the session directory

### Error: "Dataset config must include 'cids'"

Every session config must have a `cids` field:

```yaml
session: MySession
cids: [0, 1, 2, 3, 4]  # Required!
```

### Sessions loaded in wrong order

Sessions are loaded in the order specified in the parent config's `sessions` list, not alphabetically.

### Want to exclude a session temporarily

Just comment it out in the parent config:

```yaml
sessions:
  - Allen_2022-02-16
  # - Allen_2022-02-18  # Temporarily disabled
  - Logan_2020-02-29
```

### Need different configs for different experiments

Create multiple parent configs! They can all share the same session configs:

```
experiments/dataset_configs/
├── experiment_A.yaml  # Uses sessions 1-10
├── experiment_B.yaml  # Uses sessions 5-15
├── experiment_C.yaml  # Uses all sessions with different transforms
└── sessions/
    ├── session_01.yaml
    ├── session_02.yaml
    └── ...
```

