# Polar-V1 JEPA Integration Strategy

## Configuration Matrix

We need to support 6 configurations:

| Config | Behavior | JEPA | Use Case |
|--------|----------|------|----------|
| **Minimal** | ❌ | ❌ | Baseline polar features only |
| **Behavior** | ✅ | ❌ | Saccade-aware dynamics |
| **JEPA** | ❌ | ✅ | Self-supervised learning |
| **Behavior+JEPA** | ✅ | ✅ | Full model (recommended) |
| **Static** | ❌ (static) | ❌ | No dynamics at all |
| **Static+JEPA** | ❌ (static) | ✅ | JEPA without dynamics |

---

## Design Principles

1. **JEPA is a recurrent-stage component** (not convnet)
2. **Behavior is optional** in modulator/recurrent
3. **All combinations should work** via config flags
4. **JEPA loss is auxiliary** (added to Poisson NLL)

---

## Implementation Strategy

### Option 1: JEPA as Separate Recurrent Module (Recommended)

**Rationale**: Clean separation, easy to toggle on/off

#### File Structure
```
models/modules/
├── polar_convnet.py       # Spatial processing (always used)
├── polar_modulator.py     # Behavior encoding (optional)
├── polar_recurrent.py     # Temporal dynamics (optional behavior)
├── polar_jepa.py          # JEPA module (NEW)
└── polar_readout.py       # Multi-level readout
```

#### `polar_jepa.py` - New Module

```python
"""
JEPA (Joint-Embedding Predictive Architecture) for Polar-V1.

This module implements self-supervised future prediction on polar features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAProjector(nn.Module):
    """Project features to JEPA embedding space."""
    
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] features
        Returns:
            z: [B, D] embeddings
        """
        # Global average pooling
        x_pool = x.mean(dim=(-2, -1))  # [B, C]
        return self.net(x_pool)


class PolarJEPA(nn.Module):
    """
    JEPA module for Polar-V1.
    
    Predicts future features from context features.
    Works on temporal sequences before summarization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.hidden_dim = config.get('jepa_hidden_dim', 256)
        self.embed_dim = config.get('jepa_embed_dim', 128)
        self.delta_frames = config.get('jepa_delta_frames', 5)
        self.ema_decay = config.get('jepa_ema_decay', 0.99)
        
        # Feature dimension per level (amplitude + 2*phase)
        self.feat_dim = 3 * self.n_pairs
        
        # Context encoder (online)
        self.context_encoder = JEPAProjector(
            in_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embed_dim
        )
        
        # Target encoder (EMA)
        self.target_encoder = JEPAProjector(
            in_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embed_dim
        )
        
        # Initialize target as copy of context
        for param_c, param_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False
        
        # Predictor (context → target)
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )
        
        # Storage for loss computation
        self.jepa_loss_value = None
    
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for param_c, param_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data.mul_(self.ema_decay).add_(
                param_c.data, alpha=1 - self.ema_decay
            )
    
    def forward(self, A_list, U_list):
        """
        Args:
            A_list: List of [B, M, T, H, W] amplitude per level
            U_list: List of [B, M, 2, T, H, W] phase per level
        
        Returns:
            A_list, U_list: Unchanged (pass-through)
        
        Side effect:
            Computes JEPA loss and stores in self.jepa_loss_value
        """
        if not self.training:
            # No JEPA during inference
            self.jepa_loss_value = None
            return A_list, U_list
        
        # Compute JEPA loss
        self.jepa_loss_value = self._compute_jepa_loss(A_list, U_list)
        
        # Update target encoder
        self.update_target_encoder()
        
        # Return features unchanged
        return A_list, U_list
    
    def _compute_jepa_loss(self, A_list, U_list):
        """Compute JEPA loss across all levels."""
        total_loss = 0.0
        
        for level, (A, U) in enumerate(zip(A_list, U_list)):
            # A: [B, M, T, H, W]
            # U: [B, M, 2, T, H, W]
            
            B, M, T, H, W = A.shape
            
            # Check we have enough frames
            if T <= self.delta_frames:
                continue
            
            # Split into context and target
            ctx_end = T - self.delta_frames
            
            # Context: frames [0, ctx_end)
            A_ctx = A[:, :, :ctx_end]  # [B, M, T_ctx, H, W]
            U_ctx = U[:, :, :, :ctx_end]  # [B, M, 2, T_ctx, H, W]
            
            # Target: frames [delta_frames, T)
            A_tgt = A[:, :, self.delta_frames:]  # [B, M, T_tgt, H, W]
            U_tgt = U[:, :, :, self.delta_frames:]  # [B, M, 2, T_tgt, H, W]
            
            # Flatten features: [B, M, 2, T, H, W] → [B*T, M*3, H, W]
            # (amplitude + real + imag)
            def flatten_features(A, U):
                B, M, T, H, W = A.shape
                # Stack: [B, M*3, T, H, W]
                feat = torch.cat([
                    A,  # [B, M, T, H, W]
                    U[:, :, 0],  # [B, M, T, H, W] real
                    U[:, :, 1]   # [B, M, T, H, W] imag
                ], dim=1)
                # Permute: [B, T, M*3, H, W]
                feat = feat.permute(0, 2, 1, 3, 4)
                # Reshape: [B*T, M*3, H, W]
                feat = feat.reshape(B * T, M * 3, H, W)
                return feat
            
            feat_ctx = flatten_features(A_ctx, U_ctx)  # [B*T_ctx, M*3, H, W]
            feat_tgt = flatten_features(A_tgt, U_tgt)  # [B*T_tgt, M*3, H, W]
            
            # Encode
            z_ctx = self.context_encoder(feat_ctx)  # [B*T_ctx, D]
            with torch.no_grad():
                z_tgt = self.target_encoder(feat_tgt)  # [B*T_tgt, D]
            
            # Predict
            z_pred = self.predictor(z_ctx)  # [B*T_ctx, D]
            
            # Loss: MSE between prediction and target
            # Note: T_ctx == T_tgt (both are T - delta_frames)
            loss = F.mse_loss(z_pred, z_tgt)
            
            total_loss += loss
        
        # Average across levels
        return total_loss / len(A_list)
    
    def get_jepa_loss(self):
        """Get JEPA loss for logging/optimization."""
        if self.jepa_loss_value is None:
            return torch.tensor(0.0)
        return self.jepa_loss_value
```

---

### Modified `polar_recurrent.py`

Add optional JEPA integration:

```python
class PolarRecurrent(nn.Module):
    """
    Polar recurrent: Temporal dynamics + optional JEPA + summarization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.dt = config.get('dt', 1/240)
        self.use_behavior = config.get('use_behavior', True)
        self.use_jepa = config.get('use_jepa', False)
        
        # Spatial frequencies
        self.kxy = init_kxy(
            S=self.n_levels,
            M=self.n_pairs,
            base_freq_cpx=config.get('base_freq_cpx', 0.15)
        )
        
        # Polar dynamics (optional behavior)
        if self.use_behavior:
            self.dynamics = PolarDynamics(
                kxy=self.kxy,
                dt=self.dt,
                lambda_fix=config.get('lambda_fix', 10.0),
                lambda_sac=config.get('lambda_sac', 40.0)
            )
            self.modulator = None  # Set by model
        else:
            # Static dynamics (no behavior)
            self.dynamics = None
        
        # JEPA (optional)
        if self.use_jepa:
            self.jepa = PolarJEPA(config)
        else:
            self.jepa = None
        
        # Temporal summarizer
        self.summarizer = TemporalSummarizer(
            alpha_fast=config.get('alpha_fast', 0.74),
            alpha_slow=config.get('alpha_slow', 0.95)
        )
    
    def forward(self, feats):
        """
        Args:
            feats: (A_list, U_list) from modulator
        
        Returns:
            feats_per_level: List of [B, C_sum, H_l, W_l]
        """
        A_list, U_list = feats
        
        # Apply dynamics (if enabled)
        if self.dynamics is not None:
            if self.modulator is None:
                raise RuntimeError("Modulator not set. Call set_modulator() first.")
            beh_params = self.modulator.beh_params
            A_adv, U_adv = self.dynamics(A_list, U_list, beh_params)
        else:
            # No dynamics (static)
            A_adv, U_adv = A_list, U_list
        
        # Apply JEPA (if enabled)
        if self.jepa is not None:
            A_adv, U_adv = self.jepa(A_adv, U_adv)
        
        # Temporal summarization
        feats_per_level = self.summarizer(A_adv, U_adv)
        
        return feats_per_level
    
    def set_modulator(self, modulator):
        """Set reference to modulator for accessing behavior params."""
        if self.use_behavior:
            self.modulator = modulator
    
    def get_jepa_loss(self):
        """Get JEPA loss for auxiliary loss."""
        if self.jepa is not None:
            return self.jepa.get_jepa_loss()
        return torch.tensor(0.0)
```

---

### Modified `polar_modulator.py`

Make behavior encoding optional:

```python
class PolarModulator(BaseModulator):
    """
    Polar modulator: Optional behavior encoding.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.use_behavior = config.get('use_behavior', True)
        
        if self.use_behavior:
            self.beh_dim = config.get('beh_dim', 128)
            self.dt = config.get('dt', 1/240)
            
            # Gaze encoder
            self.gaze_encoder = MinimalGazeEncoder(
                d_out=self.beh_dim,
                dt=self.dt,
                use_pos_fourier=True,
                Kpos=2
            )
            
            # Behavior encoder
            self.beh_encoder = BehaviorEncoder(
                M=self.n_pairs,
                S=self.n_levels,
                beh_dim=self.beh_dim
            )
            
            self.beh_params = None
        else:
            # No behavior encoding
            self.beh_params = None
    
    def forward(self, feats, behavior=None):
        """
        Args:
            feats: (A_list, U_list) from convnet
            behavior: [B, T, 2] raw gaze (optional)
        
        Returns:
            feats: (A_list, U_list) unchanged
        """
        if self.use_behavior and behavior is not None:
            # Encode behavior
            beh_code = self.gaze_encoder(behavior)
            self.beh_params = self.beh_encoder(beh_code)
        
        # Return features unchanged
        return feats
```

---

## Configuration Examples

### Config 1: Minimal (No Behavior, No JEPA)

```yaml
# experiments/model_configs/polar_v1_minimal.yaml
model_type: v1multi

convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16

modulator:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: false  # ← No behavior

recurrent:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: false  # ← No dynamics
    use_jepa: false      # ← No JEPA

readout:
  type: polar
  params:
    n_units: 8

output_activation: softplus
```

**Pipeline**: `ConvNet → Modulator (pass-through) → Recurrent (summarize only) → Readout`

---

### Config 2: Behavior Only (No JEPA)

```yaml
# experiments/model_configs/polar_v1_behavior.yaml
model_type: v1multi

convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16

modulator:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: true   # ← Encode behavior
    beh_dim: 128
    dt: 0.004166667

recurrent:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: true   # ← Use dynamics
    use_jepa: false      # ← No JEPA
    dt: 0.004166667
    lambda_fix: 10.0
    lambda_sac: 40.0

readout:
  type: polar
  params:
    n_units: 8

output_activation: softplus
```

**Pipeline**: `ConvNet → Modulator (encode behavior) → Recurrent (dynamics + summarize) → Readout`

---

### Config 3: JEPA Only (No Behavior)

```yaml
# experiments/model_configs/polar_v1_jepa.yaml
model_type: v1multi

convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16

modulator:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: false  # ← No behavior

recurrent:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: false  # ← No dynamics
    use_jepa: true       # ← Use JEPA
    jepa_delta_frames: 5
    jepa_hidden_dim: 256
    jepa_embed_dim: 128

readout:
  type: polar
  params:
    n_units: 8

output_activation: softplus

# Auxiliary loss weight
lambda_jepa: 0.5
```

**Pipeline**: `ConvNet → Modulator (pass-through) → Recurrent (JEPA + summarize) → Readout`

---

### Config 4: Full Model (Behavior + JEPA)

```yaml
# experiments/model_configs/polar_v1_full.yaml
model_type: v1multi

convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16

modulator:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: true   # ← Encode behavior
    beh_dim: 128
    dt: 0.004166667

recurrent:
  type: polar
  params:
    n_pairs: 16
    n_levels: 4
    use_behavior: true   # ← Use dynamics
    use_jepa: true       # ← Use JEPA
    dt: 0.004166667
    lambda_fix: 10.0
    lambda_sac: 40.0
    jepa_delta_frames: 5
    jepa_hidden_dim: 256
    jepa_embed_dim: 128

readout:
  type: polar
  params:
    n_units: 8

output_activation: softplus

# Auxiliary loss weight
lambda_jepa: 0.5
```

**Pipeline**: `ConvNet → Modulator (encode behavior) → Recurrent (dynamics + JEPA + summarize) → Readout`

---

## Training Loop Integration

### Modify `training/pl_modules/multidataset_model.py`

```python
def _compute_auxiliary_loss(self):
    """Compute auxiliary losses (PC modulator, JEPA, etc.)."""
    aux_loss = None
    
    # ... existing PC modulator code ...
    
    # JEPA loss (if enabled)
    if hasattr(self.model, 'recurrent') and hasattr(self.model.recurrent, 'get_jepa_loss'):
        jepa_loss = self.model.recurrent.get_jepa_loss()
        
        if jepa_loss.item() > 0:
            lambda_jepa = self.model_config.get('lambda_jepa', 0.5)
            
            if aux_loss is None:
                aux_loss = lambda_jepa * jepa_loss
            else:
                aux_loss = aux_loss + lambda_jepa * jepa_loss
            
            # Log
            if self.global_rank == 0:
                self.log('jepa_loss', jepa_loss.item(),
                        on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
    
    return aux_loss
```

---

## Testing Strategy

### Test 1: Minimal Config
```bash
pytest tests/test_polar_minimal.py -v
```

### Test 2: Behavior Only
```bash
pytest tests/test_polar_behavior.py -v
```

### Test 3: JEPA Only
```bash
pytest tests/test_polar_jepa.py -v
```

### Test 4: Full Model
```bash
pytest tests/test_polar_full.py -v
```

---

## Summary

| Component | Minimal | Behavior | JEPA | Full |
|-----------|---------|----------|------|------|
| ConvNet | ✅ | ✅ | ✅ | ✅ |
| Modulator (behavior) | ❌ | ✅ | ❌ | ✅ |
| Recurrent (dynamics) | ❌ | ✅ | ❌ | ✅ |
| Recurrent (JEPA) | ❌ | ❌ | ✅ | ✅ |
| Readout | ✅ | ✅ | ✅ | ✅ |

**Key Design**:
- JEPA is a **recurrent-stage component**
- Behavior is **optional** via config flags
- All combinations work via `use_behavior` and `use_jepa` flags
- JEPA loss is **auxiliary** (added to main loss)

