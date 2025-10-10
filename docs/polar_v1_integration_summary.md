# Polar-V1 Core Integration Summary

## Overview

This document summarizes the integration plan for the **Polar-V1 Core** model into VisionCore's training pipeline. The Polar-V1 Core is a biologically-inspired neural network that models V1 responses using multi-scale polar representations with behavior-aware dynamics.

## Key Documents

1. **[Integration Roadmap](polar_v1_integration_roadmap.md)** - Comprehensive integration plan
2. **[Quick Start Guide](polar_v1_quick_start.md)** - Step-by-step implementation
3. **[Architecture Diagram](#architecture)** - Visual overview

---

## What is Polar-V1 Core?

### Model Components

1. **Laplacian Pyramid** - Multi-scale decomposition (4 levels)
2. **Quadrature Filtering** - Even/odd Gabor-like filters (16 pairs)
3. **Polar Decomposition** - Separates amplitude (energy) and phase
4. **Behavior Encoder** - Maps gaze → saccade gates, velocity, gains
5. **Polar Dynamics** - Saccade-aware temporal evolution
6. **Temporal Summarizer** - Collapses time via EMAs
7. **Multi-Level Readout** - Gaussian spatial readout across levels
8. **JEPA Module** (optional) - Self-supervised future prediction

### Key Innovation

The model implements **physics-aware dynamics**:
- Phase rotation follows Fourier shift theorem (retinal motion)
- Amplitude relaxation is faster during saccades vs fixation
- Behavior modulates both gain and phase per spatial frequency band

---

## Integration Strategy

### Recommended Approach: Polar-V1 as ConvNet Type

**Why this works**:
- ✅ Minimal changes to existing pipeline
- ✅ Leverages existing infrastructure (factory, configs, training loop)
- ✅ Backward compatible with existing models
- ✅ Easy to A/B test against ResNet baseline

**Mapping**:
```
Current:  adapter → frontend → ResNet → modulator → recurrent → readout
Polar-V1: adapter → frontend → PolarV1ConvNet(includes dynamics) → readout
```

**Key difference**: Polar-V1 needs **raw gaze positions** `[B, T, 2]` instead of pre-transformed behavior.

---

## Implementation Checklist

### Phase 1: Core Module (Week 1)
- [ ] Create `models/modules/polar_v1_core.py`
- [ ] Copy classes from `scripts/devel_pyrConv.py`
- [ ] Implement `PolarV1ConvNet` wrapper
- [ ] Add `get_output_channels()` method
- [ ] Write unit tests

### Phase 2: Integration (Week 2)
- [ ] Register in `models/factory.py`
- [ ] Modify `models/modules/models.py` to pass behavior to convnet
- [ ] Add JEPA loss to `training/pl_modules/multidataset_model.py`
- [ ] Create model config `polar_v1_core.yaml`
- [ ] Create dataset config `multi_polar_v1_eyepos.yaml`

### Phase 3: Testing (Week 3)
- [ ] Unit test: Forward pass
- [ ] Unit test: JEPA loss computation
- [ ] Integration test: Full training loop
- [ ] Validation: BPS on small dataset
- [ ] Hyperparameter tuning

### Phase 4: Production (Week 4)
- [ ] Train on full dataset
- [ ] Compare to ResNet baseline
- [ ] Document results
- [ ] Merge to main branch

---

## Critical Design Decisions

### 1. Behavior Input Format

**Decision**: Accept raw gaze positions `[B, T, 2]`

**Rationale**:
- Polar-V1 needs temporal gaze trajectory for velocity/acceleration
- Current pipeline pre-transforms behavior → loses temporal structure
- Solution: Use raw `eyepos` without transforms

**Dataset config change**:
```yaml
# OLD (current pipeline)
eye_vel:
  source: eyepos
  ops:
    - diff: {axis: 0}
    - maxnorm: {}
    - temporal_basis: {...}
  expose_as: behavior

# NEW (Polar-V1)
eye_pos:
  source: eyepos
  ops: []  # No transforms!
  expose_as: behavior
```

### 2. JEPA Auxiliary Loss

**Decision**: Make JEPA optional (default disabled)

**Rationale**:
- Adds 50% compute overhead (2 forward passes)
- May not help all datasets
- Let users opt-in after baseline

**Config**:
```yaml
convnet:
  type: polar_v1
  params:
    enable_jepa: false  # Start with this
    # enable_jepa: true  # Enable after baseline
```

### 3. Output Features

**Decision**: Return features from finest pyramid level

**Rationale**:
- Simplest interface (matches existing readout)
- Finest level has highest spatial resolution
- Alternative: Concatenate all levels (requires interpolation)

**Code**:
```python
def _forward_standard(self, x, beh_code):
    levels = self.pyramid(x)
    feats_per_level, _, _ = self.core.forward_trunk(levels, beh_code)
    return feats_per_level[0]  # Finest level
```

### 4. Weight Initialization

**Decision**: Conservative initialization (from script)

**Rationale**:
- Zero behavior MLP → no modulation initially
- Small filter weights → stable gradients
- Moderate dynamics rates → smooth evolution

**Code**:
```python
def _initialize_weights(self):
    # Zero behavior MLP
    for m in self.core.beh.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Conservative dynamics
    self.core.dyn.lambda_fix.copy_(torch.tensor(10.0))
    self.core.dyn.lambda_sac.copy_(torch.tensor(40.0))
    
    # Tame filters
    for p in self.core.qfb.parameters():
        if p.dim() >= 2:
            p.data *= 0.1
```

---

## File Changes Summary

### New Files
1. `models/modules/polar_v1_core.py` - Core implementation (~800 lines)
2. `experiments/model_configs/polar_v1_core.yaml` - Model config
3. `experiments/dataset_configs/multi_polar_v1_eyepos.yaml` - Dataset config
4. `scripts/test_polar_v1_integration.py` - Integration tests

### Modified Files
1. `models/factory.py` - Add Polar-V1 registration (~5 lines)
2. `models/modules/models.py` - Pass behavior to convnet (~5 lines)
3. `training/pl_modules/multidataset_model.py` - Add JEPA loss (~15 lines)

**Total changes**: ~830 lines added, ~25 lines modified

---

## Configuration Examples

### Minimal Config (No JEPA)

```yaml
model_type: v1multi
convnet:
  type: polar_v1
  params:
    n_pyramid_levels: 4
    n_pairs: 16
    dt: 0.004166667  # 1/240
    beh_dim: 128
    enable_jepa: false
modulator:
  type: none
recurrent:
  type: none
readout:
  type: gaussian
  params: {n_units: 8, bias: true}
output_activation: softplus
```

### Full Config (With JEPA)

```yaml
model_type: v1multi
convnet:
  type: polar_v1
  params:
    n_pyramid_levels: 4
    n_pairs: 16
    dt: 0.004166667
    beh_dim: 128
    enable_jepa: true
    proj_dim: 256
    jepa_delta: 5
    jepa_tau: 0.996
lambda_jepa: 0.5  # Auxiliary loss weight
```

---

## Expected Performance

### Memory
- **Baseline (ResNet)**: ~2GB per GPU
- **Polar-V1 (no JEPA)**: ~2.7GB per GPU (+35%)
- **Polar-V1 (with JEPA)**: ~3.5GB per GPU (+75%)

### Compute
- **Baseline**: 100 steps/sec
- **Polar-V1 (no JEPA)**: ~85 steps/sec (-15%)
- **Polar-V1 (with JEPA)**: ~65 steps/sec (-35%)

### Accuracy
- **Target**: BPS 10-20% better than ResNet baseline
- **Hypothesis**: Polar-V1 should outperform due to:
  - Multi-scale processing
  - Explicit saccade modeling
  - Physics-based temporal dynamics
  - Better inductive biases for V1

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/test_polar_v1_core.py -v
```

### Integration Test
```bash
python scripts/test_polar_v1_integration.py \
    --model_config experiments/model_configs/polar_v1_core.yaml \
    --dataset_config experiments/dataset_configs/multi_polar_v1_eyepos.yaml
```

### Training Test
```bash
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1_core.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_polar_v1_eyepos.yaml \
    --max_epochs 2 \
    --steps_per_epoch 10 \
    --num_gpus 1
```

---

## Success Metrics

1. ✅ **Functional**: Model trains without errors
2. ✅ **Performance**: BPS within 10% of ResNet baseline
3. ✅ **JEPA**: Auxiliary loss decreases over training
4. ✅ **Interpretability**: Polar features are visualizable
5. ✅ **Speed**: Training time <20% slower than baseline

---

## Risk Mitigation

### Risk 1: Behavior format incompatibility
- **Mitigation**: Create separate dataset config with raw gaze
- **Fallback**: Add gaze reconstruction from transformed behavior

### Risk 2: JEPA doesn't help
- **Mitigation**: Make JEPA optional (default disabled)
- **Fallback**: Use Polar-V1 without JEPA

### Risk 3: Memory issues
- **Mitigation**: Use bfloat16, gradient checkpointing
- **Fallback**: Reduce pyramid levels (4 → 3)

### Risk 4: Slower than baseline
- **Mitigation**: Use torch.compile, fuse operations
- **Fallback**: Accept 15-20% slowdown for better interpretability

---

## Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Core module | `polar_v1_core.py` + unit tests |
| 2 | Integration | Factory registration + configs |
| 3 | Testing | Integration tests + small dataset |
| 4 | Production | Full dataset + documentation |

---

## Next Steps

1. **Review** this plan with team
2. **Approve** design decisions
3. **Implement** Phase 1 (core module)
4. **Test** unit tests
5. **Iterate** based on results

---

## Questions?

- **Technical**: See [Integration Roadmap](polar_v1_integration_roadmap.md)
- **Implementation**: See [Quick Start Guide](polar_v1_quick_start.md)
- **Architecture**: See diagram above

---

## References

- Original script: `scripts/devel_pyrConv.py`
- Existing pipeline: `models/modules/models.py`
- Training loop: `training/pl_modules/multidataset_model.py`

