# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisionCore is a PyTorch Lightning-based framework for training digital twin neural encoding models of primate visual cortex (V1). It implements multi-dataset training pipelines to fit neural response models to electrophysiological recordings.

**Key Technologies:** Python 3.12, PyTorch 2.7+, PyTorch Lightning, WandB, uv package manager

## Commands

### Installation
```bash
uv sync                  # Install dependencies
uv sync --extra data     # Include optional data packages (DataYatesV1, DataRowleyV1V2)
```

### Training
```bash
# Single GPU training
python training/train_multidataset.py \
    --model_config experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120.yaml \
    --batch_size 64 --gpu 0 --precision bf16-mixed

# Multi-GPU DDP training
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120.yaml \
    --batch_size 256 --num_gpus 2 --precision bf16-mixed
```

### Testing
```bash
pytest tests/                          # Run all tests
pytest tests/test_config_builds.py     # Test model config loading
pytest tests/test_frozencore_pipeline.py  # Test FrozenCore pipeline
```

## Architecture

### Data Flow
```
Stimulus → Adapter → Frontend → ConvNet → Modulator → Recurrent → Readout → Spikes
                                              ↑
                                          Behavior
```

### Core Directories

- **models/** - Neural network components
  - `build.py` - Main entry point: `build_model(config, dataset_configs)`
  - `factory.py` - Component factories: `create_frontend()`, `create_convnet()`, `create_readout()`, etc.
  - `modules/` - Modular components (convnet, frontend, readout, recurrent, modulator)
  - `data/` - Dataset loading and transforms (`get_embedded_datasets()`, `DictDataset`)

- **training/** - Training infrastructure
  - `pl_modules/` - Lightning modules (`MultiDatasetModel`, `MultiDatasetDM`)
  - `train_multidataset.py` - Single GPU training script
  - `train_ddp_multidataset.py` - Distributed training script

- **eval/** - Evaluation pipeline
  - `eval_stack_multidataset.py` - Main evaluation: `load_model()`, `evaluate_model_multidataset()`
  - Analyses: BPS (bits-per-spike), CCNORM (noise-corrected correlation), saccade, QC

- **experiments/** - Configuration files
  - `model_configs/` - YAML model architecture definitions
  - `dataset_configs/` - YAML dataset and preprocessing specifications

### Configuration Pattern

All experiments are YAML-driven. Model architectures and datasets are defined declaratively:

```python
from models.build import build_model
from models.config_loader import load_config, load_dataset_configs

model_cfg = load_config('experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml')
dataset_cfgs = load_dataset_configs('experiments/dataset_configs/multi_basic_120.yaml')
model = build_model(model_cfg, dataset_configs=dataset_cfgs)
```

### Multi-Dataset Architecture

- Single shared "core" (frontend + convnet + recurrent) across all datasets
- Separate readout heads per dataset (one per cell collection)
- Training via `MultiDatasetModel` Lightning module with per-dataset loss aggregation

### Evaluation Usage

```python
from eval.eval_stack_multidataset import load_model, evaluate_model_multidataset

model, info = load_model(model_type='learned_res_small_none_gru_none_pool', device='cuda')
results = evaluate_model_multidataset(
    model_type='learned_res_small_none_gru_none_pool',
    analyses=['bps', 'ccnorm', 'saccade', 'qc']
)
```

## Data Packages

The data loading packages are optional and installed as siblings:
- `DataYatesV1` - Yates lab V1 electrophysiology data
- `DataRowleyV1V2` - Rowley lab V1/V2 data

These must be cloned alongside VisionCore and installed with `uv sync --extra data`.
