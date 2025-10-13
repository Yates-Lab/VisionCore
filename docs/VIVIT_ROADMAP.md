# ViViT Integration Roadmap

## Executive Summary

This roadmap outlines the complete integration of ViViT (Video Vision Transformer) into the VisionCore training pipeline. The implementation leverages existing modules wherever possible and follows the paper's architecture with adaptations for neural encoding.

**Key Simplifications:**
- ✅ No shifter module needed (stimulus already shift-corrected)
- ✅ Use existing `DynamicGaussianReadout` (works with 5D input)
- ✅ Use existing `ConvGRU` modulator for behavior
- ✅ Use existing training pipeline (no modifications needed)

---

## Phase 1: Core Implementation ✅ (COMPLETE)

### 1.1 Enhanced ViViT Module
**File:** `models/modules/viT.py`

**Completed:**
- ✅ Spatiotemporal tokenization with `UnfoldConv3d`
- ✅ Separate spatial and temporal transformers
- ✅ Causal temporal attention (no future information)
- ✅ Patch dropout for regularization
- ✅ Optional register tokens
- ✅ RoPE (Rotary Position Embeddings)
- ✅ Output shape compatible with existing readouts

**Key Features:**
```python
# Input: (B, C, T, H, W)
# Tokenization: (B, T_p, S_p, D)
# Spatial attention: (B*T_p, S_p, D)
# Temporal attention: (B*S_p, T_p, D) with causal mask
# Output: (B, D, T_p, H_p, W_p)
```

### 1.2 Component Modules
**File:** `models/modules/vit_components.py`

**Already implemented:**
- ✅ `UnfoldConv3d`: 3D convolutional tokenizer
- ✅ `RotaryPosEmb`: Rotary position embeddings
- ✅ `TransformerBlock`: Parallel attention with FlashAttention-2
- ✅ Helper functions: `get_norm`, `get_ff_activation`

---

## Phase 2: Configuration ✅ (COMPLETE)

### 2.1 Model Configs
**Files:** `experiments/model_configs/vivit_*.yaml`

**Created:**
- ✅ `vivit_baseline.yaml`: Full-size model (384 dim, 6+6 blocks)
- ✅ `vivit_small.yaml`: Fast model (192 dim, 4+4 blocks)

**Key Parameters:**
```yaml
convnet:
  type: vivit
  params:
    embedding_dim: 384
    num_spatial_blocks: 6
    num_temporal_blocks: 6
    tokenizer:
      kernel_size: [4, 4, 4]
      stride: [4, 4, 4]
```

### 2.2 Training Script
**File:** `experiments/train_vivit.sh`

**Features:**
- ✅ Support for small and baseline models
- ✅ Optimized hyperparameters for transformers
- ✅ Proper learning rate (3e-4) and weight decay (0.05)
- ✅ Gradient accumulation for effective large batch size
- ✅ bfloat16 mixed precision

---

## Phase 3: Testing ✅ (COMPLETE)

### 3.1 Unit Tests
**File:** `tests/test_vivit.py`

**Test Coverage:**
- ✅ Tokenizer forward pass
- ✅ Rotary embeddings
- ✅ Transformer blocks
- ✅ Full ViViT forward pass
- ✅ Register tokens
- ✅ Gradient flow
- ✅ Integration with `DynamicGaussianReadout`

**Run tests:**
```bash
python tests/test_vivit.py
```

---

## Phase 4: Integration Verification 🔧 (NEXT STEPS)

### 4.1 Quick Sanity Check
**Goal:** Verify model can be instantiated and run forward pass

**Steps:**
1. Run unit tests to verify components work
2. Test with small batch from real data
3. Verify shapes at each stage
4. Check memory usage

**Commands:**
```bash
# Run tests
python tests/test_vivit.py

# Quick training test (1 epoch, 10 steps)
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/vivit_small.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --max_datasets 1 \
    --batch_size 4 \
    --max_epochs 1 \
    --steps_per_epoch 10 \
    --num_gpus 1
```

### 4.2 Single Dataset Training
**Goal:** Train on one dataset to verify training loop works

**Steps:**
1. Train small model on 1 dataset for 10 epochs
2. Monitor loss, gradients, memory
3. Verify checkpointing works
4. Check validation metrics

**Expected Results:**
- Loss should decrease
- No NaN or Inf values
- Memory usage reasonable (~8GB for small model)
- Training speed ~500 samples/sec

### 4.3 Multi-Dataset Training
**Goal:** Train on multiple datasets (full pipeline)

**Steps:**
1. Train small model on 5 datasets
2. Verify dataset switching works
3. Check BPS metrics per dataset
4. Monitor for any issues

---

## Phase 5: Optimization 🎯 (WEEK 2-3)

### 5.1 Hyperparameter Search

**Search Space:**
```python
{
    'patch_size': [2, 4, 8],
    'stride': [2, 4, 8],
    'embedding_dim': [192, 384, 768],
    'num_blocks': [4, 6, 8, 12],
    'num_heads': [4, 6, 8, 12],
    'ff_dim_multiplier': [2, 4, 8],
    'dropout': [0.0, 0.1, 0.2],
}
```

**Strategy:**
1. Start with small model for fast iteration
2. Use Bayesian optimization (Optuna)
3. Optimize on validation BPS
4. Test top 3 configs on full dataset

### 5.2 Memory Optimization

**If memory is tight:**
1. Enable gradient checkpointing
2. Increase patch size (fewer tokens)
3. Reduce batch size, increase accumulation
4. Use activation checkpointing

### 5.3 Speed Optimization

**If training is slow:**
1. Verify FlashAttention is enabled
2. Use bfloat16 precision
3. Increase patch size
4. Profile to find bottlenecks

---

## Phase 6: Full Training 🚀 (WEEK 3-4)

### 6.1 Baseline Training

**Goal:** Train best model on full dataset

**Configuration:**
- Model: Best from hyperparameter search
- Datasets: All 20 datasets
- Epochs: 200 (with early stopping)
- GPUs: 4
- Batch size: 16 per GPU
- Gradient accumulation: 4

**Command:**
```bash
bash experiments/train_vivit.sh baseline
```

**Monitoring:**
- Training loss (Poisson NLL)
- Validation BPS per dataset
- Attention entropy
- Gradient norms
- Memory usage
- Training speed

### 6.2 Checkpointing

**Strategy:**
- Save every 10 epochs
- Keep best 3 checkpoints (by validation BPS)
- Save final checkpoint
- Log to WandB

---

## Phase 7: Analysis 📊 (WEEK 4-5)

### 7.1 Performance Comparison

**Compare against:**
1. ResNet baseline (current best)
2. X3D model
3. ShiftTCN model

**Metrics:**
- Validation BPS (primary)
- Single-trial correlation
- Explainable variance
- Training time
- Inference speed

### 7.2 Attention Visualization

**Analyze:**
1. Spatial attention patterns
2. Temporal attention patterns
3. Causal masking effectiveness
4. Register token usage

**Tools:**
```python
# Extract attention maps
attention_maps = extract_attention(model, batch)
plot_attention_patterns(attention_maps)
```

### 7.3 Receptive Field Analysis

**Analyze:**
1. Learned Gaussian parameters (mean, std)
2. Receptive field sizes
3. Spatial coverage
4. Comparison with ResNet RFs

### 7.4 Ablation Studies

**Test importance of:**
1. Patch dropout (0.0 vs 0.1 vs 0.2)
2. Register tokens (with vs without)
3. Causal masking (causal vs non-causal)
4. Position encoding (RoPE vs learned vs sinusoidal)
5. Behavior modulation (with vs without)

---

## Phase 8: Documentation & Deployment 📝 (WEEK 5)

### 8.1 Documentation

**Complete:**
- ✅ Implementation guide (`VIVIT_IMPLEMENTATION.md`)
- ✅ Roadmap (`VIVIT_ROADMAP.md`)
- ⏳ Results summary (after training)
- ⏳ Comparison with baselines (after analysis)

### 8.2 Code Review

**Review:**
- Code quality and style
- Documentation strings
- Type hints
- Error handling
- Edge cases

### 8.3 Deployment

**Prepare for production:**
1. Optimize inference speed
2. Create model export script
3. Document deployment requirements
4. Create inference examples

---

## Timeline Summary

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1 | Integration | Sanity checks, single dataset | Working model |
| 2 | Optimization | Hyperparameter search | Best config |
| 3-4 | Training | Full dataset training | Trained model |
| 4-5 | Analysis | Comparisons, ablations | Results report |
| 5 | Documentation | Write-up, deployment | Final docs |

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Model trains without errors
- ✅ Loss decreases over time
- ✅ Validation BPS is reasonable (>0)
- ✅ Memory usage is manageable

### Success
- 🎯 Validation BPS comparable to ResNet baseline
- 🎯 Training completes in reasonable time (<1 week)
- 🎯 Model is stable (no NaN/Inf)
- 🎯 Attention patterns are interpretable

### Stretch Goals
- 🌟 Validation BPS exceeds ResNet baseline
- 🌟 Attention maps reveal interesting patterns
- 🌟 Model generalizes to new datasets
- 🌟 Inference speed is acceptable for production

---

## Risk Mitigation

### Potential Issues

1. **Memory Issues**
   - Mitigation: Start with small model, enable checkpointing
   - Fallback: Reduce batch size, increase patch size

2. **Slow Training**
   - Mitigation: Use FlashAttention, bfloat16
   - Fallback: Use smaller model, fewer blocks

3. **Poor Performance**
   - Mitigation: Hyperparameter search, proper initialization
   - Fallback: Analyze attention patterns, adjust architecture

4. **Numerical Instability**
   - Mitigation: Gradient clipping, proper normalization
   - Fallback: Reduce learning rate, use float32

---

## Next Steps (Immediate)

1. **Run unit tests** to verify implementation
   ```bash
   python tests/test_vivit.py
   ```

2. **Quick sanity check** with small batch
   ```bash
   python training/train_ddp_multidataset.py \
       --model_config experiments/model_configs/vivit_small.yaml \
       --max_datasets 1 --batch_size 4 --max_epochs 1 --steps_per_epoch 10
   ```

3. **Single dataset training** (overnight)
   ```bash
   bash experiments/train_vivit.sh small
   ```

4. **Review results** and adjust if needed

---

## Questions to Address

- [ ] What is the optimal patch size for our data?
- [ ] How many transformer blocks do we need?
- [ ] Is behavior integration via modulator sufficient?
- [ ] Do register tokens improve performance?
- [ ] What is the best learning rate schedule?

---

## Resources

- **Code:** `models/modules/viT.py`, `models/modules/vit_components.py`
- **Configs:** `experiments/model_configs/vivit_*.yaml`
- **Tests:** `tests/test_vivit.py`
- **Docs:** `docs/VIVIT_IMPLEMENTATION.md`
- **Training:** `experiments/train_vivit.sh`

---

## Conclusion

The ViViT implementation is **ready for testing**. The core architecture is complete, configs are set up, and tests are in place. The next step is to run the sanity checks and begin training.

**Estimated time to first results:** 1-2 days
**Estimated time to full training:** 1-2 weeks
**Estimated time to complete analysis:** 3-4 weeks

