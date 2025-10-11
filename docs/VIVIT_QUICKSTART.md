# ViViT Quick Start Guide

## TL;DR

```bash
# 1. Run tests
python tests/test_vivit.py

# 2. Train small model
bash experiments/train_vivit.sh small

# 3. Train full model
bash experiments/train_vivit.sh baseline
```

---

## What is ViViT?

**ViViT (Video Vision Transformer)** is a transformer-based architecture for processing spatiotemporal video data. Unlike CNNs that use local convolutions, ViViT uses self-attention to capture global relationships across space and time.

**Key advantages:**
- Global receptive field (attention over all patches)
- Explicit temporal modeling with causal masking
- Interpretable attention patterns
- State-of-the-art performance on video tasks

**Key differences from ResNet:**
- Uses attention instead of convolution
- Processes video as discrete tokens (patches)
- Separate spatial and temporal processing
- More parameters but potentially better performance

---

## Installation

No additional dependencies needed! ViViT uses existing VisionCore modules.

**Requirements:**
- PyTorch >= 2.0 (for FlashAttention-2)
- einops
- All standard VisionCore dependencies

---

## Quick Test

Verify the implementation works:

```bash
# Run all unit tests
python tests/test_vivit.py
```

**Expected output:**
```
==========================================================
Running ViViT Tests
==========================================================

Testing Components:
----------------------------------------------------------
✓ Tokenizer test passed: torch.Size([2, 1, 32, 36, 64]) -> torch.Size([2, 8, 144, 192])
✓ RoPE test passed: torch.Size([2, 4, 50, 64]) -> torch.Size([2, 4, 50, 64])
✓ Transformer block test passed: torch.Size([2, 100, 192]) -> torch.Size([2, 100, 192])

Testing ViViT Model:
----------------------------------------------------------
✓ Basic forward test passed: torch.Size([2, 1, 32, 36, 64]) -> torch.Size([2, 192, 8, 9, 16])
✓ Register tokens test passed: torch.Size([2, 1, 32, 36, 64]) -> torch.Size([2, 192, 8, 9, 16])
✓ Gradient flow test passed
✓ Output channels test passed: 192

Testing Integration:
----------------------------------------------------------
✓ ViViT + Gaussian readout test passed: torch.Size([2, 1, 32, 36, 64]) -> torch.Size([2, 192, 8, 9, 16]) -> torch.Size([2, 100])

==========================================================
All tests passed! ✓
==========================================================
```

---

## Training

### Option 1: Small Model (Recommended for First Run)

Fast training for experimentation and debugging:

```bash
bash experiments/train_vivit.sh small
```

**Specs:**
- Embedding dim: 192
- Blocks: 4 spatial + 4 temporal
- Patch size: 8x8x8
- Training time: ~2-3 hours (4 GPUs)
- Memory: ~8GB per GPU

### Option 2: Baseline Model (Full Size)

Full-size model for best performance:

```bash
bash experiments/train_vivit.sh baseline
```

**Specs:**
- Embedding dim: 384
- Blocks: 6 spatial + 6 temporal
- Patch size: 4x4x4
- Training time: ~1-2 days (4 GPUs)
- Memory: ~16GB per GPU

### Option 3: Custom Configuration

Create your own config in `experiments/model_configs/`:

```yaml
# my_vivit.yaml
model_type: v1multi
convnet:
  type: vivit
  params:
    embedding_dim: 256
    num_spatial_blocks: 5
    num_temporal_blocks: 5
    # ... other params
```

Then train:

```bash
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/my_vivit.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --num_gpus 4
```

---

## Monitoring

### WandB Dashboard

Training metrics are logged to Weights & Biases:

**Key metrics to watch:**
- `train/loss`: Should decrease steadily
- `val/bps_mean`: Bits per spike (higher is better)
- `train/grad_norm`: Should be stable (not exploding)
- `system/gpu_memory`: Monitor for OOM issues

### Console Output

```
Epoch 1/200: 100%|████████| 1000/1000 [05:23<00:00, 3.09it/s]
train/loss: 2.456
val/bps_mean: 0.123
val/bps_dataset_0: 0.145
val/bps_dataset_1: 0.112
...
```

---

## Troubleshooting

### Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size: `--batch_size 8`
2. Increase gradient accumulation: `--accumulate_grad_batches 8`
3. Enable checkpointing in config: `grad_checkpointing: true`
4. Use smaller model: `vivit_small.yaml`
5. Increase patch size: `kernel_size: [8, 8, 8]`

### Slow Training

**Symptoms:** <100 samples/sec

**Solutions:**
1. Verify FlashAttention is enabled: `flash_attention: true`
2. Use bfloat16: `--precision bf16-mixed`
3. Increase patch size (fewer tokens)
4. Check GPU utilization: `nvidia-smi`

### NaN Loss

**Symptoms:** Loss becomes NaN after a few steps

**Solutions:**
1. Reduce learning rate: `--learning_rate 1e-4`
2. Increase warmup: `--warmup_epochs 20`
3. Enable gradient clipping: `--gradient_clip_val 1.0`
4. Use bfloat16 instead of float16
5. Check for numerical instability in attention

### Poor Performance

**Symptoms:** Validation BPS is very low or not improving

**Solutions:**
1. Check learning rate (try 1e-4 to 5e-4)
2. Increase training time (more epochs)
3. Adjust weight decay (0.01 to 0.1)
4. Try different patch sizes
5. Verify data preprocessing is correct

---

## Understanding the Output

### Model Architecture

```
Input: (B, 1, 300, 36, 64)
  ↓
Adapter: Spatial preprocessing
  ↓
Tokenizer: (B, 75, 144, 384)
  75 temporal tokens (300/4)
  144 spatial tokens (9×16)
  384 embedding dim
  ↓
Spatial Transformer (6 blocks)
  Attention over 144 spatial tokens
  Per temporal position
  ↓
Temporal Transformer (6 blocks)
  Attention over 75 temporal tokens
  Per spatial position
  Causal masking (no future info)
  ↓
Output: (B, 384, 75, 9, 16)
  ↓
Modulator: Integrate behavior
  ↓
Readout: (B, n_units)
  Uses last time step
  Gaussian spatial pooling
```

### Attention Patterns

The model learns two types of attention:

1. **Spatial Attention**: Which parts of the image are important?
   - Example: Attends to moving objects, edges, textures

2. **Temporal Attention**: Which time points are important?
   - Example: Attends to motion onsets, changes, recent history
   - Causal: Only looks at past and present (not future)

---

## Next Steps

After training completes:

1. **Evaluate performance**
   ```bash
   python scripts/evaluate_model.py \
       --checkpoint checkpoints/vivit/best.ckpt \
       --dataset_configs experiments/dataset_configs/multi_basic_120_backimage_all.yaml
   ```

2. **Visualize attention**
   ```python
   from models.modules.viT import ViViT
   model = ViViT.load_from_checkpoint('checkpoints/vivit/best.ckpt')
   attention_maps = extract_attention(model, batch)
   plot_attention(attention_maps)
   ```

3. **Compare with baseline**
   ```bash
   python scripts/compare_models.py \
       --model1 checkpoints/resnet/best.ckpt \
       --model2 checkpoints/vivit/best.ckpt
   ```

4. **Analyze receptive fields**
   ```python
   from models.modules.readout import DynamicGaussianReadout
   readout = model.readout
   plot_receptive_fields(readout)
   ```

---

## FAQ

**Q: How long does training take?**
A: Small model: 2-3 hours. Baseline: 1-2 days (4 A100 GPUs).

**Q: How much memory do I need?**
A: Small: 8GB per GPU. Baseline: 16GB per GPU.

**Q: Can I use fewer GPUs?**
A: Yes, adjust `--num_gpus` and increase `--accumulate_grad_batches`.

**Q: Why is ViViT slower than ResNet?**
A: Self-attention is O(n²) in sequence length. Use larger patches to reduce tokens.

**Q: Do I need to modify the training pipeline?**
A: No! ViViT integrates seamlessly with existing pipeline.

**Q: How do I add behavior variables?**
A: Use the ConvGRU modulator (already configured in configs).

**Q: Can I use a pre-trained model?**
A: Not yet, but you can train on one dataset and fine-tune on others.

**Q: How do I interpret attention maps?**
A: High attention = important for prediction. Visualize with `plot_attention()`.

---

## Support

- **Documentation**: See `docs/VIVIT_IMPLEMENTATION.md` for details
- **Roadmap**: See `docs/VIVIT_ROADMAP.md` for full plan
- **Issues**: Open a GitHub issue
- **Questions**: Contact the VisionCore team

---

## Summary

ViViT is ready to use! The implementation:
- ✅ Follows the paper architecture
- ✅ Integrates with existing modules
- ✅ Has comprehensive tests
- ✅ Includes optimized configs
- ✅ Works with existing training pipeline

**Start here:**
```bash
python tests/test_vivit.py && bash experiments/train_vivit.sh small
```

Good luck! 🚀

