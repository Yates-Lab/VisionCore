# Multidataset Model Evaluation

This directory contains scripts for evaluating trained multidataset models from the `run_all_models.sh` training pipeline.

## Files

- **`evaluate_trained_model.py`** - Main evaluation script with comprehensive functionality
- **`test_evaluation.py`** - Test script to verify evaluation works with available checkpoints
- **`example_evaluation.py`** - Example usage patterns and demonstrations
- **`README_evaluation.md`** - This documentation file

## Quick Start

### 1. List Available Checkpoints

First, see what trained models are available:

```bash
python test_evaluation.py --list
```

This will show all checkpoint directories and files, indicating which ones have valid (non-NaN) losses.

### 2. Test the Evaluation Pipeline

Run a quick test to make sure everything works:

```bash
python test_evaluation.py --test
```

This will automatically find a good checkpoint and run a minimal evaluation to verify the pipeline works.

### 3. Evaluate a Specific Model and Dataset

```bash
python evaluate_trained_model.py \
    --checkpoint_path checkpoints/x3d_ddp_bs32_ds10_lr1e-4_20250613_160101/epoch=00-val_loss_total=0.1869.ckpt \
    --dataset_idx 0 \
    --batch_size 64 \
    --save_plots \
    --output_dir ./evaluation_results
```

## Detailed Usage

### Command Line Arguments

- `--checkpoint_path`: Path to the model checkpoint file (required)
- `--dataset_idx`: Index of dataset to evaluate (0-based, default: 0)
- `--batch_size`: Batch size for evaluation (default: 64)
- `--max_batches`: Maximum number of batches to evaluate (default: None for all)
- `--device`: Device to run on (default: "cuda")
- `--save_plots`: Save plots to files instead of displaying
- `--output_dir`: Directory to save results (default: "./evaluation_results")

### What the Evaluation Does

1. **Loads the trained model** from the checkpoint
2. **Loads the specified dataset** using the same preprocessing as training
3. **Evaluates on both training and validation sets** to assess generalization
4. **Computes key metrics**:
   - Loss (Poisson negative log-likelihood)
   - BPS (bits per spike) - information-theoretic measure of prediction quality
   - Per-unit BPS distribution
5. **Generates visualizations**:
   - BPS distribution histograms
   - Train vs validation BPS scatter plots
   - Correlation analysis
6. **Saves results** to files for further analysis

### Understanding the Output

#### Key Metrics

- **Loss**: Lower is better. Measures how well the model predicts spike counts.
- **BPS (Bits Per Spike)**: Higher is better. Measures information content of predictions.
  - Values typically range from 0 to ~1.5 bits per spike
  - Values > 0.1 indicate meaningful predictive power
  - Values > 0.5 indicate very good predictions

#### Generalization Assessment

The script compares training vs validation performance:
- **Small gap**: Model generalizes well
- **Large gap**: Model may be overfitting
- **Negative gap**: Unusual, may indicate data issues

## Examples

### Example 1: Basic Evaluation

```bash
# Evaluate dataset 0 from a specific checkpoint
python evaluate_trained_model.py \
    --checkpoint_path checkpoints/x3d_modulator_ddp_bs64_ds10_lr8.5e-05_20250614_123456/epoch=05-val_loss_total=0.1234.ckpt \
    --dataset_idx 0
```

### Example 2: Quick Evaluation (Limited Batches)

```bash
# Evaluate only first 50 batches for quick testing
python evaluate_trained_model.py \
    --checkpoint_path checkpoints/resnet_ddp_bs32_ds10_lr1e-4_20250613_150613/epoch=00-val_loss_total=0.1744.ckpt \
    --dataset_idx 1 \
    --max_batches 50 \
    --batch_size 32
```

### Example 3: Save Results and Plots

```bash
# Full evaluation with saved outputs
python evaluate_trained_model.py \
    --checkpoint_path checkpoints/x3d_ddp_bs32_ds10_lr1e-4_20250613_160101/epoch=00-val_loss_total=0.1869.ckpt \
    --dataset_idx 0 \
    --save_plots \
    --output_dir ./results/x3d_dataset0_evaluation
```

### Example 4: Programmatic Usage

```python
from evaluate_trained_model import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("path/to/checkpoint.ckpt")

# Get model info
info = evaluator.get_model_info()
print(f"Model has {info['num_datasets']} datasets")

# Evaluate a dataset
results = evaluator.compare_train_val_performance(dataset_idx=0)
print(f"Validation BPS: {results['val']['bps']:.4f}")

# Plot results
evaluator.plot_bps_distribution(results)
```

## Interpreting Results

### Good Performance Indicators

- **BPS > 0.1**: Model has meaningful predictive power
- **BPS > 0.3**: Good performance for neural data
- **Small train/val gap**: Good generalization
- **High correlation between train/val BPS per unit**: Consistent performance

### Potential Issues

- **BPS near 0**: Model not learning meaningful patterns
- **Large train/val gap**: Overfitting
- **NaN or infinite values**: Numerical instability
- **Very low correlations**: Poor predictions

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: Check the path and ensure the file exists
2. **"Dataset index out of range"**: Use `--dataset_idx` within the model's dataset count
3. **CUDA out of memory**: Reduce `--batch_size` or use `--device cpu`
4. **Import errors**: Ensure you're in the correct directory and environment

### Getting Help

1. **List available checkpoints**: `python test_evaluation.py --list`
2. **Run test evaluation**: `python test_evaluation.py --test`
3. **Check model info**: The evaluation script prints detailed model information
4. **Use smaller batches**: Start with `--max_batches 10` for quick tests

## Output Files

When using `--save_plots` and `--output_dir`, the script creates:

- `bps_distribution_dataset_X.png`: BPS distribution plots
- `evaluation_results_dataset_X.npz`: Numerical results in NumPy format
- Console output with detailed metrics and comparisons

The `.npz` file contains:
- `train_loss`, `val_loss`: Average losses
- `train_bps`, `val_bps`: Average BPS values
- `train_bps_per_unit`, `val_bps_per_unit`: Per-unit BPS arrays
- `dataset_idx`: Dataset index evaluated
- `checkpoint_path`: Path to the checkpoint used
