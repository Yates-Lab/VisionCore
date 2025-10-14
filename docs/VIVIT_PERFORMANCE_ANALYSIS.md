# ViViT vs ResNet Performance Analysis

## Observed Performance Gap

**ResNet (red)**: ~0.45 val_bps_overall  
**ViViT (purple)**: ~0.41 val_bps_overall  
**Gap**: ~9% underperformance

## Architecture Comparison

### ResNet (learned_res_small_gru_optimized_aa)
```yaml
Frontend: learnable_temporal (16 filters, 4 channels, anti-aliasing)
Convnet: ResNet
  - Channels: [128, 128, 256]
  - 3 stages with residual blocks
  - Standard 3D convolutions
  - Anti-aliased pooling (aablur)
  - Dilation in final stage (RF expansion)
Modulator: ConvGRU (256 hidden, 32 beh_emb)
Output: 256 channels
```

### ViViT Small (vivit_small)
```yaml
Frontend: learnable_temporal (16 filters, 4 channels, NO anti-aliasing)
Convnet: ViViT
  - Tokenizer: [8,8,8] patches, stride [8,8,8]
  - Embedding: 192 dims
  - 4 spatial + 4 temporal transformer blocks
  - Attention: 4 heads × 48 dims
  - FF: 768 dims (4x embedding)
Modulator: ConvGRU (192 hidden, 16 beh_emb)
Output: 192 channels
```

## Potential Issues

### 1. **Aggressive Tokenization (LIKELY MAJOR ISSUE)**
- **ViViT**: `kernel_size: [8,8,8]`, `stride: [8,8,8]`
  - Input: `(B, 1, 300, 36, 64)` after frontend
  - After tokenizer: `(B, 37, 36, 192)` → **37 temporal tokens, 36 spatial tokens**
  - Temporal compression: 300 → 37 frames (8x downsampling)
  - Spatial compression: 36×64 → 6×6 grid (very coarse!)

- **ResNet**: Progressive downsampling
  - Stage 0: stride 2 → keeps more spatial detail
  - Stage 1: aablur stride 2 → smooth downsampling
  - Stage 2: dilation 2 → expands RF without losing resolution
  - Final output: ~13×13 spatial map (much finer than ViViT's 6×6)

**Impact**: ViViT loses fine spatial detail immediately. Neural responses may depend on fine-grained spatial features that are lost in aggressive tokenization.

### 2. **Insufficient Model Capacity**
- **ViViT Small**: 192 embedding dim, 4+4 blocks
- **ResNet**: 256 final channels, 3 deep residual stages

**Parameter Count Estimate**:
- ViViT Tokenizer: ~1.5M params (Conv3d + projections)
- ViViT Transformers: 8 blocks × ~0.5M params/block = ~4M params
- ResNet: 3 stages × ~2M params/stage = ~6M params
- **Total**: ViViT ~5.5M vs ResNet ~6M (comparable, not the issue)

### 3. **Modulator Mismatch**
- **ViViT**: ConvGRU with 192 hidden, 16 beh_emb (smaller)
- **ResNet**: ConvGRU with 256 hidden, 32 beh_emb (larger)
- **Impact**: ViViT modulator has less capacity for behavior integration

### 4. **Missing Anti-Aliasing in Frontend**
- **ViViT**: `anti_aliasing: false`
- **ResNet**: `anti_aliasing: true`
- **Impact**: ViViT frontend may introduce aliasing artifacts

### 5. **Spatial Resolution at Readout**
- **ViViT**: Outputs `(B, 192, 37, 6, 6)` → 6×6 spatial grid
- **ResNet**: Outputs `(B, 256, T, 13, 13)` → 13×13 spatial grid
- **Impact**: Gaussian readout has 4.7x fewer spatial locations to pool from in ViViT
  - This severely limits the readout's ability to learn precise receptive fields
  - Neurons with small RFs may not have enough spatial resolution

### 6. **Causal Masking in Spatial Blocks**
- **Current**: `is_causal: true` in config (applied to spatial blocks)
- **Should be**: Spatial blocks should be non-causal, only temporal blocks causal
- **Impact**: Spatial attention is unnecessarily restricted (though we override this in code)

## Root Cause Analysis

### Primary Issue: **Aggressive Spatial Tokenization**
The 8×8 spatial patches reduce 36×64 input to 6×6 tokens. This is too coarse for neural encoding where:
- Neurons have small, precise receptive fields
- Fine spatial details matter for neural responses
- Readout needs high spatial resolution to learn accurate RF positions

### Secondary Issues:
1. **Smaller modulator capacity** (192 vs 256 hidden, 16 vs 32 beh_emb)
2. **No anti-aliasing in frontend** (may introduce artifacts)
3. **Fewer output channels** (192 vs 256)

## Recommended Fixes

### Priority 1: Reduce Spatial Tokenization
```yaml
tokenizer:
  kernel_size: [8, 4, 4]  # Keep temporal 8, reduce spatial to 4
  stride: [8, 4, 4]
```
**Expected output**: `(B, 37, 144, 192)` → 37 temporal, 144 spatial (9×16 grid)
**Benefit**: 4x more spatial tokens, closer to ResNet's 13×13 resolution

### Priority 2: Match Modulator Capacity
```yaml
modulator:
  params:
    feature_dim: 192
    hidden_dim: 256  # Increase from 192
    beh_emb_dim: 32  # Increase from 16
```

### Priority 3: Enable Anti-Aliasing
```yaml
frontend:
  params:
    anti_aliasing: true  # Match ResNet
```

### Priority 4: Increase Model Capacity (if needed)
```yaml
convnet:
  params:
    embedding_dim: 256  # Increase from 192
    num_spatial_blocks: 6  # Increase from 4
    num_temporal_blocks: 6  # Increase from 4
    ff_dim: 1024  # Increase from 768
```

## Expected Performance Impact

### With Priority 1 fix (spatial tokenization):
- **Expected gain**: +3-5% (most critical)
- **Rationale**: Readout can learn precise RFs with 4x more spatial resolution

### With Priority 1+2 (+ modulator):
- **Expected gain**: +5-7%
- **Rationale**: Better behavior integration capacity

### With Priority 1+2+3 (+ anti-aliasing):
- **Expected gain**: +6-8%
- **Rationale**: Cleaner input features

### With all fixes:
- **Expected gain**: +7-10%
- **Target**: Match or exceed ResNet performance (~0.45 bps)

## Data Efficiency Consideration

Transformers typically require more data than CNNs because:
1. **Less inductive bias**: CNNs have built-in translation equivariance
2. **Global attention**: Transformers can overfit to spurious correlations
3. **More parameters in attention**: Requires more samples to learn

**Your dataset**: 30 sessions, ~1M frames total
- This may be on the small side for transformers
- ResNet's convolutional inductive bias helps with limited data
- ViViT may need more regularization or data augmentation

## Next Steps

1. **Immediate**: Fix spatial tokenization to [8,4,4]
2. **Quick win**: Enable anti-aliasing, increase modulator capacity
3. **If still underperforming**: Increase model size (embedding_dim, num_blocks)
4. **If data-limited**: Add stronger regularization (higher dropout, more patch_dropout)
5. **Long-term**: Consider hybrid CNN-Transformer (CNN stem + Transformer)

## Hybrid Architecture Idea

If pure ViViT continues to underperform, consider:
```yaml
# Use ResNet for spatial processing (keeps fine details)
# Use Transformer for temporal processing (long-range dependencies)
convnet:
  type: hybrid_cnn_transformer
  spatial_backbone: resnet  # Proven spatial processing
  temporal_transformer: vivit  # Transformer for temporal
```

This would combine ResNet's spatial inductive bias with Transformer's temporal modeling.

