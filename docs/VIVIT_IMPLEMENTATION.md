# ViViT Implementation Guide

## Overview

This document describes the implementation of **ViViT (Video Vision Transformer)** for neural encoding in VisionCore, based on the paper by Arnab et al. (2021) with modifications for predicting neural responses.

## Architecture

### Core Components

1. **Tokenizer** (`UnfoldConv3d`)
   - Extracts 3D spatiotemporal patches (tubelets) from input video
   - Uses Conv3D followed by linear projection
   - Configurable patch size and stride
   - Output: `(B, T_p, S_p, D)` where T_p = temporal tokens, S_p = spatial tokens, D = embedding dim

2. **Spatial Transformer**
   - Processes spatial relationships within each frame
   - Multiple transformer blocks with self-attention
   - Operates on `(B*T_p, S_p, D)` - attention over spatial dimension
   - Uses RoPE (Rotary Position Embeddings)

3. **Temporal Transformer**
   - Processes temporal relationships across frames
   - Multiple transformer blocks with **causal** self-attention
   - Operates on `(B*S_p, T_p, D)` - attention over temporal dimension
   - Causal masking prevents using future information

4. **Readout** (`DynamicGaussianReadout`)
   - Uses existing Gaussian readout from VisionCore
   - Automatically uses last time step from `(B, C, T, H, W)` input
   - Learns spatial receptive field per neuron
   - No shifter needed (stimulus is already shift-corrected)

### Key Features

- **Patch Dropout**: Randomly zeros out token embeddings during training (regularization)
- **Register Tokens**: Optional learnable tokens to prevent attention artifacts
- **FlashAttention-2**: Hardware-optimized attention implementation
- **Parallel Attention**: Fused QKV projection with feedforward (reduces operations)
- **Causal Masking**: Temporal transformer only sees past and current frames
- **Modular Design**: Integrates with existing VisionCore components

## Configuration

### Model Sizes

Two configurations are provided:

#### Small (Fast Experimentation)
```yaml
embedding_dim: 192
num_spatial_blocks: 4
num_temporal_blocks: 4
num_heads: 4
head_dim: 48
patch_size: [8, 8, 8]  # Larger patches = fewer tokens
```

#### Baseline (Full Model)
```yaml
embedding_dim: 384
num_spatial_blocks: 6
num_temporal_blocks: 6
num_heads: 6
head_dim: 64
patch_size: [4, 4, 4]  # Smaller patches = more tokens
```

### Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `embedding_dim` | Token embedding dimension | 192, 384, 768 |
| `num_spatial_blocks` | Spatial transformer layers | 4-12 |
| `num_temporal_blocks` | Temporal transformer layers | 4-12 |
| `num_heads` | Attention heads | 4-12 |
| `head_dim` | Dimension per head | 48-64 |
| `ff_dim` | Feedforward dimension | 4x embedding_dim |
| `patch_size` | Spatiotemporal patch size | [2-8, 2-8, 2-8] |
| `stride` | Patch stride | [2-8, 2-8, 2-8] |
| `patch_dropout` | Token dropout rate | 0.0-0.2 |
| `mha_dropout` | Attention dropout | 0.0-0.2 |
| `ff_dropout` | Feedforward dropout | 0.0-0.2 |
| `drop_path` | Stochastic depth | 0.0-0.2 |

## Usage

### Training

```bash
# Small model (faster)
bash experiments/train_vivit.sh small

# Baseline model (full size)
bash experiments/train_vivit.sh baseline
```

### Testing

```bash
# Run unit tests
python tests/test_vivit.py

# Or with pytest
pytest tests/test_vivit.py -v
```

### Custom Configuration

Create a new config file in `experiments/model_configs/`:

```yaml
model_type: v1multi
sampling_rate: 240
initial_input_channels: 1

convnet:
  type: vivit
  params:
    embedding_dim: 384
    num_spatial_blocks: 6
    num_temporal_blocks: 6
    # ... other parameters
```

## Integration with Existing Pipeline

### Data Flow

```
Input: (B, 1, 300, 36, 64)  # Batch, Channel, Time, Height, Width
    ↓
Adapter: Spatial preprocessing
    ↓
ViViT Tokenizer: (B, 75, 144, 384)  # 75 temporal, 144 spatial tokens
    ↓
Spatial Transformer: Learn spatial relationships
    ↓
Temporal Transformer: Learn temporal relationships (causal)
    ↓
Output: (B, 384, 75, 9, 16)  # Reshape for readout
    ↓
Modulator: Integrate behavior (ConvGRU)
    ↓
Readout: (B, n_units)  # Per-neuron predictions
```

### Behavior Integration

Behavior is integrated via the **ConvGRU modulator** (not in tokenizer):
- Behavior features: `(B, T, 42)` from dataset
- Modulator embeds and broadcasts spatially
- Combines with ViViT features before readout

This approach:
- ✓ Uses existing, tested modulator
- ✓ Keeps ViViT focused on visual processing
- ✓ Allows flexible behavior integration

### Readout Compatibility

The existing `DynamicGaussianReadout` works perfectly:
- Expects: `(B, C, T, H, W)` or `(B, C, H, W)`
- Automatically uses last time step if 5D input
- No modifications needed!

## Performance Considerations

### Memory Usage

ViViT is memory-intensive due to self-attention:
- **Small model**: ~8GB per GPU with batch_size=32
- **Baseline model**: ~16GB per GPU with batch_size=16

Memory-saving options:
1. Enable gradient checkpointing: `grad_checkpointing: true`
2. Reduce batch size and increase gradient accumulation
3. Use smaller patch sizes (fewer tokens)
4. Reduce number of transformer blocks

### Training Speed

Approximate training speeds (A100 GPU):
- **Small model**: ~500 samples/sec
- **Baseline model**: ~200 samples/sec

Speed optimizations:
1. Use FlashAttention-2: `flash_attention: true`
2. Use bfloat16 precision: `--precision bf16-mixed`
3. Increase patch size (fewer tokens)
4. Use parallel attention (already enabled)

### Hyperparameter Search

Recommended search strategy:
1. Start with small model for quick iteration
2. Search over:
   - Patch size and stride
   - Number of blocks
   - Embedding dimension
   - Dropout rates
3. Use Bayesian optimization (Optuna)
4. Validate on held-out neurons

## Comparison with Other Models

### vs ResNet (Current Baseline)

| Aspect | ResNet | ViViT |
|--------|--------|-------|
| Receptive field | Limited by kernel size | Global (attention) |
| Temporal modeling | Causal convolution | Causal attention |
| Parameters | ~5M | ~20M (baseline) |
| Speed | Fast (~1000 samples/sec) | Slower (~200 samples/sec) |
| Interpretability | Moderate | High (attention maps) |

### vs X3D

| Aspect | X3D | ViViT |
|--------|-----|-------|
| Architecture | 3D CNN | Transformer |
| Inductive bias | Strong (locality) | Weak (global) |
| Data efficiency | High | Lower (needs more data) |
| Flexibility | Moderate | High |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model
   - Increase patch size

2. **Slow Training**
   - Enable FlashAttention
   - Use bfloat16 precision
   - Increase patch size
   - Reduce number of blocks

3. **Poor Performance**
   - Check learning rate (try 1e-4 to 5e-4)
   - Increase warmup epochs
   - Adjust weight decay (0.01-0.1)
   - Try different patch sizes

4. **NaN Loss**
   - Reduce learning rate
   - Enable gradient clipping
   - Check for numerical instability in attention
   - Use bfloat16 instead of float16

## Future Improvements

Potential enhancements:
1. **Factorized attention**: Separate spatial and temporal attention in same layer
2. **Hierarchical structure**: Multi-scale processing
3. **Masked prediction**: Self-supervised pre-training
4. **Adaptive computation**: Dynamic number of blocks per sample
5. **Sparse attention**: Reduce computational cost

## References

1. Arnab et al. (2021). "ViViT: A Video Vision Transformer"
2. Vaswani et al. (2017). "Attention Is All You Need"
3. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
4. Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
5. Darcet et al. (2023). "Vision Transformers Need Registers"

## Contact

For questions or issues, please open a GitHub issue or contact the VisionCore team.

