# Eval Stack - Model Evaluation Tools

Unified evaluation pipeline for multidataset neural encoding models.

---

## Quick Start

### Load a Model

```python
from eval.eval_stack_multidataset import load_model

# Load best model of a specific type
model, model_info = load_model(
    model_type='learned_res_small_none_gru_none_pool',
    checkpoint_dir='/path/to/checkpoints',
    device='cuda'
)
```

### Run Full Evaluation

```python
from eval.eval_stack_multidataset import evaluate_model_multidataset

results = evaluate_model_multidataset(
    model_type='learned_res_small_none_gru_none_pool',
    checkpoint_dir='/path/to/checkpoints',
    save_dir='/path/to/cache',
    analyses=['bps', 'ccnorm', 'saccade', 'qc'],
    recalc=False,  # Use cached results if available
    batch_size=64,
    device='cuda',
    rescale=False  # Apply affine rescaling to predictions
)
```

---

## Module Structure

```
eval/
├── __init__.py                      # Package initialization
├── eval_stack_multidataset.py       # Main evaluation pipeline
├── eval_stack_utils.py              # Utility functions
├── gaborium_analysis.py             # Gaborium-specific analysis
├── gratings_analysis.py             # Gratings-specific analysis
└── README.md                        # This file
```

---

## Main Components

### eval_stack_multidataset.py

**Main Functions:**
- `load_model()` - Load trained models from checkpoints
- `evaluate_model_multidataset()` - Run full evaluation pipeline
- `run_bps_analysis()` - Bits-per-spike analysis
- `run_ccnorm_analysis()` - Noise-corrected correlation (FixRSVP)
- `run_saccade_analysis()` - Saccade-triggered responses
- `run_qc_analysis()` - Quality control metrics

**Features:**
- Automatic model discovery (best by validation BPS)
- Per-dataset caching for fast re-evaluation
- Graceful handling of missing stimulus types
- Cell tracking across datasets with CID mapping

### eval_stack_utils.py

**Utility Functions:**
- `scan_checkpoints()` - Discover and organize checkpoints
- `load_single_dataset()` - Load individual datasets
- `get_stim_inds()` - Get stimulus-specific indices
- `evaluate_dataset()` - Run model on dataset and compute BPS
- `load_qc_data()` - Load quality control metrics
- `get_fixrsvp_trials()` - Extract trial-aligned FixRSVP data
- `ccnorm_variable_trials()` - Compute noise-corrected correlation
- `get_saccade_eval()` - Saccade-triggered analysis
- `detect_saccades_from_session()` - Load/detect saccades
- `rescale_rhat()` - Affine rescaling of predictions
- `bits_per_spike()` - Compute bits per spike metric

---

## Usage Examples

### Example 1: Load and Evaluate Best Model

```python
from eval.eval_stack_multidataset import evaluate_model_multidataset

# Evaluate best learned_res model
results = evaluate_model_multidataset(
    model_type='learned_res_small_none_gru_none_pool',
    checkpoint_dir='/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints',
    save_dir='/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack_120',
    analyses=['bps', 'ccnorm', 'saccade'],
    recalc=False,
    batch_size=64
)

# Access results
model_type = 'learned_res_small_none_gru_none_pool'
bps_gaborium = results[model_type]['bps']['gaborium']['bps']
ccnorm = results[model_type]['ccnorm']['fixrsvp']['ccnorm']
```

### Example 2: Load Specific Checkpoint

```python
from eval.eval_stack_multidataset import load_model

model, info = load_model(
    checkpoint_path='/path/to/specific/checkpoint.ckpt',
    device='cuda'
)
```

### Example 3: Scan Available Models

```python
from eval.eval_stack_utils import scan_checkpoints

models_by_type = scan_checkpoints(
    checkpoint_dir='/path/to/checkpoints',
    verbose=True
)

# See what's available
for model_type, models in models_by_type.items():
    print(f"{model_type}: {len(models)} checkpoints")
    print(f"  Best BPS: {models[0]['val_bps']:.4f}")
```

### Example 4: Load Single Dataset

```python
from eval.eval_stack_multidataset import load_model
from eval.eval_stack_utils import load_single_dataset

# Load model
model, _ = load_model(model_type='learned_res_small_none_gru_none_pool')

# Load specific dataset
dataset_idx = 0
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

print(f"Dataset: {model.names[dataset_idx]}")
print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")
print(f"Units: {len(dataset_config['cids'])}")
```

---

## Analysis Types

### BPS (Bits Per Spike)
Evaluates model predictions using Poisson log-likelihood on:
- Gaborium (white noise)
- Backimage (natural images)
- FixRSVP (rapid serial visual presentation)
- Gratings (drifting gratings)

### CCNORM (Noise-Corrected Correlation)
Computes noise-corrected correlation for FixRSVP trials, accounting for:
- Variable trial counts per time bin
- Missing data
- Trial-to-trial variability

### Saccade Analysis
Saccade-triggered responses for all stimulus types:
- Aligned to saccade onset
- Filters by inter-saccade interval
- Includes eye velocity traces

### QC (Quality Control)
Loads quality metrics for each unit:
- Refractory period violations (contamination)
- Amplitude truncation
- Waveforms
- Probe geometry and laminar depth

---

## Caching System

Results are cached per model/dataset/analysis:

```
save_dir/
└── model_name/
    ├── model_name_dataset0_bps_cache.pt
    ├── model_name_dataset0_ccnorm_cache.pt
    ├── model_name_dataset0_saccade_cache.pt
    ├── model_name_dataset0_qc_cache.pt
    ├── model_name_dataset1_bps_cache.pt
    └── ...
```

**Cache files include:**
- Analysis results (robs, rhat, metrics)
- Cell IDs (cids)
- Metadata

**Benefits:**
- Fast re-evaluation (skip recalculation)
- Partial updates (only missing analyses)
- Consistent results across runs

---

## Import Structure

After the refactoring, imports should be:

```python
# From outside eval package
from eval.eval_stack_multidataset import load_model, evaluate_model_multidataset
from eval.eval_stack_utils import scan_checkpoints, load_single_dataset

# Training module (new path)
from training import MultiDatasetModel, MultiDatasetDM
```

**Old path (deprecated):**
```python
# ❌ DON'T USE - This path no longer exists
from jake.multidataset_ddp.train_ddp_multidataset import MultiDatasetModel
```

---

## See Also

- `notes/EVAL_IMPORTS_FIXED.md` - Details on import fixes
- `notes/REFACTORING_COMPLETE.md` - Training system refactoring
- `notes/CLEANUP_COMPLETE.md` - Cleanup documentation
- `scripts/model_explore_devel.py` - Example usage script

---

**Status:** ✅ All imports fixed and verified  
**Ready for use:** ✅ Full evaluation pipeline functional

