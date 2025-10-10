# Polar-V1 Core Quick Start Guide

## TL;DR

```bash
# 1. Create the module
touch models/modules/polar_v1_core.py

# 2. Register in factory (models/factory.py)
# Add: if convnet_type.lower() == 'polar_v1': ...

# 3. Modify model forward (models/modules/models.py)
# Add: if hasattr(self.convnet, 'requires_behavior'): ...

# 4. Add JEPA loss (training/pl_modules/multidataset_model.py)
# Extend: _compute_auxiliary_loss()

# 5. Create configs
cp experiments/model_configs/res_small_gru.yaml \
   experiments/model_configs/polar_v1_core.yaml

# 6. Train
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1_core.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_polar_v1_eyepos.yaml
```

---

## Detailed Implementation Steps

### Step 1: Copy Polar-V1 Code from Script

**Source**: `scripts/devel_pyrConv.py` (lines 74-881)

**Extract these classes**:
- `MinimalGazeEncoder` (lines 440-541)
- `QuadratureFilterBank2D` (lines 86-132)
- `PolarDecompose` (lines 137-162)
- `BehaviorEncoder` (lines 167-191)
- `PolarDynamics` (lines 196-268)
- `TemporalSummarizer` (lines 273-317)
- `GaussianReadout` (lines 322-362)
- `MultiLevelReadout` (lines 365-381)
- `JEPAModule` (lines 386-434)
- `PolarV1Core` (lines 758-841)
- `PyramidAdapter` (lines 546-569)
- `init_kxy` (lines 611-632)

**Create**: `models/modules/polar_v1_core.py`

### Step 2: Wrap as ConvNet

Add this wrapper class to `polar_v1_core.py`:

```python
class PolarV1ConvNet(nn.Module):
    """
    Polar-V1 Core wrapped as a ConvNet for VisionCore pipeline.
    
    This module:
    1. Takes stimulus [B, C, T, H, W] and behavior [B, T, 2]
    2. Builds Laplacian pyramid
    3. Processes through polar dynamics
    4. Returns features [B, C_out, H_out, W_out] for readout
    5. Optionally computes JEPA auxiliary loss
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config
        self.n_pyramid_levels = config.get('n_pyramid_levels', 4)
        self.n_pairs = config.get('n_pairs', 16)
        self.dt = config.get('dt', 1/240)
        self.beh_dim = config.get('beh_dim', 128)
        self.enable_jepa = config.get('enable_jepa', False)
        self.proj_dim = config.get('proj_dim', 256)
        self.jepa_delta = config.get('jepa_delta', 5)
        
        # Flag for model to know we need behavior
        self.requires_behavior = True
        
        # Build components
        self.gaze_encoder = MinimalGazeEncoder(
            d_out=self.beh_dim,
            dt=self.dt,
            use_pos_fourier=True,
            Kpos=2
        )
        
        self.pyramid = None  # Lazy init on first forward
        self.core = None     # Lazy init on first forward
        
        # Storage for JEPA
        self._jepa_ctx = None
        self._jepa_tgt = None
    
    def _lazy_init(self, x, behavior):
        """Initialize pyramid and core on first forward."""
        if self.pyramid is not None:
            return
        
        device = x.device
        B, C, T, H, W = x.shape
        
        # Build pyramid adapter
        self.pyramid = PyramidAdapter(J=self.n_pyramid_levels).to(device)
        
        # Get pyramid shapes
        with torch.no_grad():
            levels = self.pyramid(x[:1])  # Use single sample
        
        # Initialize kxy
        kxy = init_kxy(
            S=len(levels),
            M=self.n_pairs,
            base_freq_cpx=0.15,
            device=device
        )
        
        # Build core
        self.core = PolarV1Core(
            in_ch_per_level=C,
            pairs=self.n_pairs,
            kxy=kxy,
            dt=self.dt,
            n_neurons=1,  # Dummy, will be set by readout
            proj_dim=self.proj_dim,
            beh_dim=self.beh_dim
        ).to(device)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Conservative initialization."""
        # Zero behavior MLP
        for m in self.core.beh.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Conservative dynamics
        with torch.no_grad():
            self.core.dyn.lambda_fix.copy_(torch.tensor(10.0))
            self.core.dyn.lambda_sac.copy_(torch.tensor(40.0))
            self.core.dyn.a_bar.data.mul_(0.5)
        
        # Tame spatial filters
        for p in self.core.qfb.parameters():
            if p.dim() >= 2:
                p.data *= 0.1
    
    def forward(self, x, behavior=None):
        """
        Args:
            x: [B, C, T, H, W] stimulus
            behavior: [B, T, 2] raw gaze positions (REQUIRED!)
        
        Returns:
            features: [B, C_out, H_out, W_out]
        """
        if behavior is None:
            raise ValueError("PolarV1ConvNet requires behavior (raw gaze positions)")
        
        # Lazy init
        self._lazy_init(x, behavior)
        
        # Encode gaze to behavior code
        beh_code = self.gaze_encoder(behavior)  # [B, T, D]
        
        # JEPA mode: split into context/target
        if self.enable_jepa and self.training:
            return self._forward_jepa(x, beh_code)
        else:
            return self._forward_standard(x, beh_code)
    
    def _forward_standard(self, x, beh_code):
        """Standard forward pass."""
        # Build pyramid
        levels = self.pyramid(x)
        
        # Forward through core trunk
        feats_per_level, _, _ = self.core.forward_trunk(levels, beh_code)
        
        # Concatenate across levels or use finest level
        # Option 1: Use finest level only
        output = feats_per_level[0]  # [B, C_sum, H, W]
        
        # Option 2: Concatenate all levels (requires interpolation)
        # output = self._concat_levels(feats_per_level)
        
        return output
    
    def _forward_jepa(self, x, beh_code):
        """Forward with JEPA context/target split."""
        T = x.shape[2]
        delta = self.jepa_delta
        
        if T <= delta:
            # Not enough frames, fall back
            return self._forward_standard(x, beh_code)
        
        # Context window
        x_ctx = x[:, :, :-delta]
        beh_ctx = beh_code[:, :-delta]
        levels_ctx = self.pyramid(x_ctx)
        feats_ctx, _, _ = self.core.forward_trunk(levels_ctx, beh_ctx)
        
        # Target window (no grad)
        with torch.no_grad():
            x_tgt = x[:, :, delta:]
            beh_tgt = beh_code[:, delta:]
            levels_tgt = self.pyramid(x_tgt)
            feats_tgt, _, _ = self.core.forward_trunk(levels_tgt, beh_tgt)
        
        # Store for JEPA loss
        self._jepa_ctx = feats_ctx
        self._jepa_tgt = feats_tgt
        
        # Return context features
        return feats_ctx[0]  # Use finest level
    
    def get_jepa_loss(self, mask_ratio=0.5):
        """Compute JEPA loss on stored features."""
        if self._jepa_ctx is None or self._jepa_tgt is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        loss = self.core.jepa_loss(self._jepa_ctx, self._jepa_tgt, mask_ratio)
        
        # Clear storage
        self._jepa_ctx = None
        self._jepa_tgt = None
        
        return loss
    
    def get_output_channels(self):
        """Return number of output channels."""
        # 5 summaries per pair (last, fast, slow, deriv, energy)
        return 5 * self.n_pairs
```

### Step 3: Register in Factory

**File**: `models/factory.py`

**Location**: In `create_convnet()` function, add after line 128:

```python
# Handle polar_v1 separately
if convnet_type.lower() == 'polar_v1':
    from .modules.polar_v1_core import PolarV1ConvNet
    core = PolarV1ConvNet(cfg)
    return core, core.get_output_channels()
```

### Step 4: Modify Model Forward

**File**: `models/modules/models.py`

**Location**: In `MultiDatasetV1Model.forward()`, around line 450:

```python
# Process through shared convnet
# NEW: Check if convnet needs behavior
if hasattr(self.convnet, 'requires_behavior') and self.convnet.requires_behavior:
    x_conv = self.convnet(x, behavior=behavior)
else:
    x_conv = self.convnet(x)
```

### Step 5: Add JEPA Auxiliary Loss

**File**: `training/pl_modules/multidataset_model.py`

**Location**: In `_compute_auxiliary_loss()`, around line 276:

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
        
        if jepa_loss.item() > 0:  # Only add if JEPA was computed
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

### Step 6: Create Model Config

**File**: `experiments/model_configs/polar_v1_core.yaml`

See full config in roadmap document.

### Step 7: Create Dataset Config

**File**: `experiments/dataset_configs/multi_polar_v1_eyepos.yaml`

**Key requirement**: Behavior must be raw gaze `[T, 2]`, no transforms!

```yaml
transforms:
  stim:
    source: stim
    ops:
      - pixelnorm: {}
    expose_as: stim
  
  eye_pos:
    source: eyepos
    ops: []  # NO TRANSFORMS!
    expose_as: behavior
```

---

## Testing

### Quick Sanity Check

```python
import torch
from models.modules.polar_v1_core import PolarV1ConvNet

config = {
    'n_pyramid_levels': 4,
    'n_pairs': 16,
    'dt': 1/240,
    'beh_dim': 128,
    'enable_jepa': True,
    'proj_dim': 256,
    'jepa_delta': 5
}

model = PolarV1ConvNet(config).cuda()
x = torch.randn(2, 1, 15, 51, 51).cuda()
beh = torch.randn(2, 15, 2).cuda()

# Forward
out = model(x, behavior=beh)
print(f"Output shape: {out.shape}")  # Should be [2, 80, H, W]

# JEPA loss
loss = model.get_jepa_loss()
print(f"JEPA loss: {loss.item()}")
```

### Full Integration Test

```bash
python scripts/test_polar_v1_integration.py
```

---

## Common Issues

### Issue 1: "PolarV1ConvNet requires behavior"

**Cause**: Behavior not passed to convnet
**Fix**: Ensure `MultiDatasetV1Model.forward()` passes behavior

### Issue 2: "Not enough frames for JEPA"

**Cause**: `T < jepa_delta`
**Fix**: Increase `keys_lags` in dataset config or disable JEPA

### Issue 3: JEPA loss is 0

**Cause**: Not in training mode or T too small
**Fix**: Call `model.train()` and ensure `T >= jepa_delta + 5`

---

## Performance Tips

1. **Use bfloat16**: Add `precision: bf16-mixed` to trainer
2. **Gradient checkpointing**: Set `checkpointing: true` in config
3. **Smaller pyramid**: Use `n_pyramid_levels: 3` for faster training
4. **Disable JEPA initially**: Set `enable_jepa: false` for baseline

---

## Next Steps

1. Implement the module
2. Run unit tests
3. Train on small dataset
4. Compare BPS to ResNet baseline
5. Tune hyperparameters
6. Scale to full dataset

