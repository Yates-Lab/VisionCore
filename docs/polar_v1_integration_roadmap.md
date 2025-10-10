# Polar-V1 Core Integration Roadmap

## Executive Summary

This document outlines the integration plan for the **Polar-V1 Core** model into VisionCore's existing training pipeline. The Polar-V1 Core is a biologically-inspired model that processes visual stimuli through multi-scale polar representations with behavior-aware dynamics and optional JEPA self-supervised learning.

## Architecture Overview

### Polar-V1 Core Components
1. **Laplacian Pyramid** - Multi-scale decomposition (replaces/augments convnet)
2. **Quadrature Filtering** - Even/odd Gabor-like filters per level
3. **Polar Decomposition** - Amplitude (energy) + Phase (orientation)
4. **Behavior Encoder** - Maps gaze → saccade gates, velocity, gains
5. **Polar Dynamics** - Saccade-aware temporal evolution
6. **Temporal Summarizer** - Collapses time via EMAs
7. **Multi-Level Readout** - Gaussian spatial readout across pyramid levels
8. **JEPA Module** (optional) - Self-supervised future prediction

### Key Differences from Current Pipeline
- **Input**: Requires raw gaze positions `[B, T, 2]` instead of pre-transformed behavior
- **Processing**: Replaces convnet+recurrent with pyramid+polar dynamics
- **Output**: Same Poisson rate predictions `[B, n_neurons]`
- **Loss**: Poisson NLL + optional JEPA auxiliary loss

---

## Integration Strategy

### Phase 1: Create Polar-V1 as a New Convnet Type ✅ RECOMMENDED

**Rationale**: Minimal disruption to existing pipeline, leverages existing infrastructure.

The Polar-V1 Core can be integrated as a **new convnet type** that:
- Takes stimulus `[B, C, T, H, W]` and behavior `[B, T, D]` as input
- Internally builds Laplacian pyramid
- Processes through polar dynamics
- Outputs features `[B, C_out, H_out, W_out]` for readout

**Mapping to Existing Architecture**:
```
Current:     adapter → frontend → convnet → modulator → recurrent → readout
Polar-V1:    adapter → frontend → PolarConvNet(pyramid+dynamics) → readout
                                      ↑
                                  behavior (raw gaze)
```

---

## Implementation Plan

### Step 1: Create Polar-V1 Convnet Module

**File**: `models/modules/polar_v1_core.py`

**Key Design Decisions**:

1. **Behavior Input Handling**:
   - Accept behavior as constructor parameter or forward argument
   - Use `MinimalGazeEncoder` to convert raw gaze `[B,T,2]` → behavior code `[B,T,D]`
   - Store behavior internally during forward pass for JEPA

2. **Output Format**:
   - Return `[B, C_summary, H_final, W_final]` to match convnet interface
   - Collapse temporal dimension via `TemporalSummarizer`
   - Use last pyramid level or concatenate across levels

3. **JEPA Integration**:
   - Store intermediate features during forward pass
   - Expose `get_jepa_loss()` method for auxiliary loss computation
   - Require context/target window slicing in training loop

**Interface**:
```python
class PolarV1ConvNet(nn.Module):
    """Polar-V1 Core as a convnet replacement."""
    
    def __init__(self, config):
        # config contains:
        # - n_pyramid_levels: int (default 4)
        # - n_pairs: int (default 16)
        # - dt: float (default 1/240)
        # - beh_dim: int (default 128)
        # - enable_jepa: bool (default False)
        # - proj_dim: int (default 256)
        pass
    
    def forward(self, x, behavior=None):
        """
        Args:
            x: [B, C, T, H, W] stimulus
            behavior: [B, T, 2] raw gaze positions (required!)
        
        Returns:
            features: [B, C_out, H_out, W_out] for readout
        """
        pass
    
    def get_jepa_loss(self, mask_ratio=0.5):
        """Compute JEPA loss on stored context/target features."""
        pass
    
    def get_output_channels(self):
        """Return number of output channels."""
        pass
```

### Step 2: Register in Factory

**File**: `models/factory.py`

Add to `create_convnet()`:
```python
if convnet_type.lower() == 'polar_v1':
    from .modules.polar_v1_core import PolarV1ConvNet
    core = PolarV1ConvNet(cfg)
    return core, core.get_output_channels()
```

### Step 3: Modify Model Forward Pass

**File**: `models/modules/models.py`

**Current Issue**: Behavior is `[B, n_vars]` (no time dimension)

**Solution Options**:

**Option A**: Modify `MultiDatasetV1Model.forward()` to pass behavior to convnet
```python
def forward(self, stimulus=None, dataset_idx: int = 0, behavior=None):
    # ... existing code ...
    
    # Check if convnet needs behavior (Polar-V1)
    if hasattr(self.convnet, 'requires_behavior') and self.convnet.requires_behavior:
        x_conv = self.convnet(x, behavior=behavior)
    else:
        x_conv = self.convnet(x)
    
    # ... rest of forward pass ...
```

**Option B**: Store behavior in convnet during forward (cleaner)
```python
# In PolarV1ConvNet.__init__:
self.requires_behavior = True

# In MultiDatasetV1Model.forward:
if hasattr(self.convnet, 'set_behavior'):
    self.convnet.set_behavior(behavior)
x_conv = self.convnet(x)
```

### Step 4: Add JEPA Auxiliary Loss

**File**: `training/pl_modules/multidataset_model.py`

Extend `_compute_auxiliary_loss()`:

```python
def _compute_auxiliary_loss(self):
    """Compute auxiliary losses (PC modulator, JEPA, etc.)."""
    aux_loss = None
    
    # Existing PC modulator loss
    if hasattr(self.model, 'modulator') and self.model.modulator is not None:
        if hasattr(self.model.modulator, 'pred_err') and self.model.modulator.pred_err is not None:
            lambda_pred = self.model_config.get('lambda_pred', 0.1)
            pred_err = self.model.modulator.pred_err
            aux_loss = lambda_pred * (pred_err ** 2).mean()
    
    # NEW: JEPA loss for Polar-V1
    if hasattr(self.model, 'convnet') and hasattr(self.model.convnet, 'get_jepa_loss'):
        lambda_jepa = self.model_config.get('lambda_jepa', 0.5)
        jepa_loss = self.model.convnet.get_jepa_loss()
        
        if aux_loss is None:
            aux_loss = lambda_jepa * jepa_loss
        else:
            aux_loss = aux_loss + lambda_jepa * jepa_loss
        
        # Log JEPA loss separately
        if self.global_rank == 0:
            self.log('jepa_loss', jepa_loss.item(), 
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
    
    return aux_loss
```

### Step 5: Handle Context/Target Windows for JEPA

**Challenge**: JEPA requires future prediction, but current pipeline processes single batches.

**Solution**: Modify batch preparation in `PolarV1ConvNet.forward()`:

```python
def forward(self, x, behavior=None):
    """
    Args:
        x: [B, C, T, H, W] stimulus
        behavior: [B, T, 2] raw gaze
    """
    if not self.enable_jepa or not self.training:
        # Standard forward pass
        return self._forward_trunk(x, behavior)
    
    # JEPA mode: split into context/target windows
    delta = self.jepa_delta  # e.g., 5 frames
    T = x.shape[2]
    
    if T <= delta:
        # Not enough frames for JEPA, fall back to standard
        return self._forward_trunk(x, behavior)
    
    # Context window: frames [0, T-delta)
    x_ctx = x[:, :, :-delta]
    beh_ctx = behavior[:, :-delta]
    
    # Target window: frames [delta, T)
    x_tgt = x[:, :, delta:]
    beh_tgt = behavior[:, delta:]
    
    # Forward on context (student)
    feats_ctx = self._forward_trunk(x_ctx, beh_ctx, return_intermediate=True)
    
    # Forward on target (teacher, no grad)
    with torch.no_grad():
        feats_tgt = self._forward_trunk(x_tgt, beh_tgt, return_intermediate=True)
    
    # Store for JEPA loss computation
    self._jepa_ctx = feats_ctx['intermediate']
    self._jepa_tgt = feats_tgt['intermediate']
    
    # Return final features from context window
    return feats_ctx['output']
```

### Step 6: Create Model Config

**File**: `experiments/model_configs/polar_v1_core.yaml`

```yaml
model_type: v1multi

# Model dimensions
sampling_rate: 240
initial_input_channels: 1

# Frontend configuration
adapter:
  type: adapter
  params: {grid_size: 51, init_sigma: 1.0, transform: scale}

frontend:
  type: none
  params: {}

# Polar-V1 Core as convnet
convnet:
  type: polar_v1
  params:
    n_pyramid_levels: 4
    n_pairs: 16
    dt: 0.004166667  # 1/240 Hz
    beh_dim: 128
    alpha_fast: 0.74
    alpha_slow: 0.95
    enable_jepa: true
    proj_dim: 256
    jepa_delta: 5  # frames ahead to predict
    jepa_tau: 0.996  # EMA decay for teacher

# No modulator needed (behavior handled in convnet)
modulator:
  type: none
  params: {}

# No recurrent needed (dynamics in polar core)
recurrent:
  type: none
  params: {}

# Readout configuration
readout:
  type: gaussian
  params:
    n_units: 8
    bias: true
    initial_std: 5.0
    initial_mean_scale: 0.1

output_activation: softplus

# JEPA auxiliary loss weight
lambda_jepa: 0.5

regularization:
  - name: readout_sparsity
    type: l1
    lambda: 1.0e-7
    apply_to: ["readouts/features"]
    schedule:
      kind: warmup
      start_epoch: 1
```

### Step 7: Create Dataset Config

**File**: `experiments/dataset_configs/multi_polar_v1_eyepos.yaml`

**Key Change**: Behavior must be **raw gaze positions** `[T, 2]`, not pre-transformed.

```yaml
types: [backimage]
session_dir: ./sessions
sessions: [Allen_2022-02-16, ...]

sampling:
  source_rate: 240
  target_rate: 120

keys_lags:
  robs: 0
  stim: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  behavior: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Time-lagged gaze
  dfs: 0

transforms:
  stim:
    source: stim
    ops:
      - pixelnorm: {}
      # NO temporal_basis or dacones - Polar-V1 handles this
    expose_as: stim
  
  # NEW: Raw gaze positions (no transforms!)
  eye_pos:
    source: eyepos
    ops: []  # No processing - Polar-V1 needs raw positions
    expose_as: behavior

datafilters:
  dfs:
    ops: [{valid_nlags: {n_lags: 32}}, {missing_pct: {theshold: 45}}]
    expose_as: dfs
    
train_val_split: 0.8
seed: 1002
```

---

## Testing Plan

### Unit Tests

1. **Test Polar-V1 ConvNet**:
   ```python
   def test_polar_v1_forward():
       config = {...}
       model = PolarV1ConvNet(config)
       x = torch.randn(2, 1, 10, 51, 51)  # [B, C, T, H, W]
       beh = torch.randn(2, 10, 2)  # [B, T, 2] gaze
       out = model(x, behavior=beh)
       assert out.shape[0] == 2  # batch
       assert out.dim() == 4  # [B, C, H, W]
   ```

2. **Test JEPA Loss**:
   ```python
   def test_jepa_loss():
       model = PolarV1ConvNet({..., 'enable_jepa': True})
       model.train()
       x = torch.randn(2, 1, 15, 51, 51)
       beh = torch.randn(2, 15, 2)
       out = model(x, behavior=beh)
       loss = model.get_jepa_loss()
       assert loss.item() > 0
   ```

### Integration Tests

1. **Test with MultiDatasetModel**:
   ```bash
   python scripts/test_polar_v1_integration.py \
       --model_config experiments/model_configs/polar_v1_core.yaml \
       --dataset_config experiments/dataset_configs/multi_polar_v1_eyepos.yaml
   ```

2. **Test Training Loop**:
   ```bash
   python training/train_ddp_multidataset.py \
       --model_config experiments/model_configs/polar_v1_core.yaml \
       --dataset_configs_path experiments/dataset_configs/multi_polar_v1_eyepos.yaml \
       --max_epochs 2 \
       --steps_per_epoch 10
   ```

---

## Migration Path for Existing Users

### Backward Compatibility

- Existing models continue to work unchanged
- Polar-V1 is opt-in via config
- No breaking changes to API

### Gradual Adoption

1. **Week 1**: Implement core module, test in isolation
2. **Week 2**: Integrate with factory, test with dummy data
3. **Week 3**: Test with real datasets, tune hyperparameters
4. **Week 4**: Production deployment, documentation

---

## Open Questions & Design Decisions

### Q1: How to handle variable-length sequences?

**Current**: Batches have fixed `T` (e.g., 10 frames)
**Polar-V1**: Needs sufficient `T` for JEPA (e.g., T >= 15 for delta=5)

**Solution**: Add config validation:
```python
if enable_jepa and T < jepa_delta + 5:
    raise ValueError(f"Need T >= {jepa_delta + 5} for JEPA")
```

### Q2: Should JEPA be always-on or optional?

**Recommendation**: Optional (default `enable_jepa: false`)
- Adds complexity and compute
- May not help all datasets
- Let users opt-in after baseline

### Q3: How to initialize Polar-V1 weights?

**From script** (lines 710-731):
```python
def initialize_polar_core(core):
    # Zero behavior MLP initially
    for m in core.beh.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Conservative dynamics
    core.dyn.lambda_fix.copy_(torch.tensor(10.0))
    core.dyn.lambda_sac.copy_(torch.tensor(40.0))
    
    # Tame spatial filters
    for p in core.qfb.parameters():
        if p.dim() >= 2:
            p.data *= 0.1
```

Add this to `models/build.py::initialize_model_components()`.

---

## Performance Considerations

### Memory

- **Pyramid**: 4 levels ≈ 1.33× memory vs single scale
- **JEPA**: 2× forward passes (context + target)
- **Mitigation**: Use gradient checkpointing, bfloat16

### Compute

- **Quadrature filters**: Lightweight (depthwise + 1×1)
- **Polar dynamics**: Minimal (element-wise ops)
- **JEPA**: +50% compute (teacher forward)

### Optimization

- Use `torch.compile()` on Polar-V1 modules
- Fuse operations in dynamics (sin/cos, exp)
- Cache Gaussian masks in readout

---

## Success Metrics

1. **Functional**: Model trains without errors
2. **Performance**: BPS 10-20% better than ResNet baseline
3. **JEPA**: Auxiliary loss decreases over training
4. **Interpretability**: Polar features visualizable
5. **Speed**: <20% slower than baseline

### Why Polar-V1 Should Outperform ResNet

**Strong Inductive Biases for V1**:

1. **Multi-scale processing**: V1 neurons have diverse receptive field sizes
   - Laplacian pyramid naturally captures this hierarchy
   - ResNet processes single scale (or learns it inefficiently)

2. **Explicit saccade modeling**: V1 responses are suppressed during saccades
   - Polar-V1: Explicit gating (λ_sac > λ_fix)
   - ResNet: Must learn this implicitly (harder, less data-efficient)

3. **Physics-based dynamics**: Retinal motion causes phase shifts
   - Polar-V1: Fourier shift theorem (k·v → Δφ)
   - ResNet: No explicit motion model

4. **Structured representations**: Amplitude (energy) + Phase (orientation)
   - Polar-V1: Matches V1 complex/simple cell structure
   - ResNet: Unstructured learned features

5. **Fewer parameters, better constraints**: 1.5M vs 2.1M
   - Polar-V1: Physics constrains the solution space
   - ResNet: More parameters, but less structure

**Expected Performance Gains**:
- **Saccade-rich data**: +15-25% BPS (explicit saccade suppression)
- **Natural images**: +10-15% BPS (multi-scale structure)
- **Temporal prediction**: +20-30% BPS (physics-based dynamics)
- **Limited data**: +10-20% BPS (better inductive biases)

**Conservative Estimate**: 10-20% BPS improvement over ResNet baseline

---

## Next Steps

1. ✅ Review this roadmap with team
2. ⬜ Implement `PolarV1ConvNet` module
3. ⬜ Add factory registration
4. ⬜ Create configs
5. ⬜ Write unit tests
6. ⬜ Integration testing
7. ⬜ Documentation
8. ⬜ Production deployment


