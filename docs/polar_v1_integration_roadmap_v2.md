# Polar-V1 Core Integration Roadmap (Revised)

## Correct Component Mapping

### Pipeline Stage Separation

```
Current:  adapter → frontend → convnet → modulator → recurrent → readout
Polar-V1: adapter → frontend → PolarConvNet → PolarModulator → PolarRecurrent → PolarReadout
```

### Component Breakdown

| Polar-V1 Component | Pipeline Stage | File Location | Rationale |
|-------------------|----------------|---------------|-----------|
| **Laplacian Pyramid** | ConvNet | `models/modules/polar_convnet.py` | Spatial feature extraction |
| **Quadrature Filters** | ConvNet | `models/modules/polar_convnet.py` | Spatial filtering (even/odd) |
| **Polar Decompose** | ConvNet | `models/modules/polar_convnet.py` | Feature transformation |
| **MinimalGazeEncoder** | Modulator | `models/modules/polar_modulator.py` | Behavior encoding |
| **BehaviorEncoder** | Modulator | `models/modules/polar_modulator.py` | Behavior → dynamics params |
| **Polar Dynamics** | Recurrent | `models/modules/polar_recurrent.py` | Temporal evolution |
| **Temporal Summarizer** | Recurrent | `models/modules/polar_recurrent.py` | Collapse time dimension |
| **Multi-Level Readout** | Readout | `models/modules/polar_readout.py` | Spatial pooling per neuron |
| **JEPA Module** | Auxiliary Loss | `training/pl_modules/` | Self-supervised loss |

---

## Implementation Plan (Revised)

### File 1: `models/modules/polar_convnet.py`

**Purpose**: Spatial feature extraction via pyramid + quadrature filters

**Components**:
- `PyramidAdapter` - Laplacian pyramid decomposition
- `QuadratureFilterBank2D` - Even/odd spatial filters per level
- `PolarDecompose` - Amplitude + phase extraction
- `PolarConvNet` - Main convnet class

**Interface**:
```python
class PolarConvNet(nn.Module):
    """
    Polar convnet: Laplacian pyramid + quadrature filters + polar decompose.
    
    Input:  [B, C, T, H, W] stimulus
    Output: List of [B, M, 2, T, H_l, W_l] per level (amplitude + phase)
            where M = n_pairs, 2 = [amplitude, phase_real, phase_imag]
    """
    
    def __init__(self, config):
        self.n_pyramid_levels = config.get('n_pyramid_levels', 4)
        self.n_pairs = config.get('n_pairs', 16)
        self.kernel_size = config.get('kernel_size', 7)
        
        # Build pyramid
        self.pyramid = PyramidAdapter(J=self.n_pyramid_levels)
        
        # Quadrature filters (shared across levels)
        self.qfb = QuadratureFilterBank2D(
            in_ch=1,  # Assuming grayscale
            pairs=self.n_pairs,
            kernel=self.kernel_size
        )
        
        # Polar decomposition
        self.polar = PolarDecompose()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        
        Returns:
            A_list: List of [B, M, T, H_l, W_l] amplitude per level
            U_list: List of [B, M, 2, T, H_l, W_l] unit complex pose per level
        """
        # Build pyramid
        levels = self.pyramid(x)  # List of [B, C, T, H_l, W_l]
        
        # Apply quadrature filters
        even_list, odd_list = self.qfb(levels)
        
        # Polar decomposition
        A_list, U_list = self.polar(even_list, odd_list)
        
        return A_list, U_list
    
    def get_output_channels(self):
        # Return as tuple: (n_pairs, n_levels)
        return (self.n_pairs, self.n_pyramid_levels)
```

**Key Point**: This is PURE spatial processing, no behavior, no temporal dynamics.

---

### File 2: `models/modules/polar_modulator.py`

**Purpose**: Encode behavior and compute dynamics parameters

**Components**:
- `MinimalGazeEncoder` - Raw gaze → behavior code
- `BehaviorEncoder` - Behavior code → dynamics params (q, v_eff, gamma, rho)
- `PolarModulator` - Main modulator class

**Interface**:
```python
class PolarModulator(BaseModulator):
    """
    Polar modulator: Encodes behavior for polar dynamics.
    
    Input:  
        - feats: (A_list, U_list) from PolarConvNet
        - behavior: [B, T, 2] raw gaze positions
    
    Output: 
        - feats: (A_list, U_list) unchanged (pass-through)
        - beh_params: Dict with keys ['q', 'v_eff', 'gamma', 'rho']
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
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
    
    def forward(self, feats, behavior):
        """
        Args:
            feats: (A_list, U_list) from convnet
            behavior: [B, T, 2] raw gaze positions
        
        Returns:
            feats: (A_list, U_list) unchanged
            beh_params: Dict['q', 'v_eff', 'gamma', 'rho']
        """
        A_list, U_list = feats
        
        # Encode gaze
        beh_code = self.gaze_encoder(behavior)  # [B, T, D]
        
        # Get dynamics parameters
        beh_params = self.beh_encoder(beh_code)
        # Returns: {'q': [B,T,1], 'v_eff': [B,T,2], 
        #           'gamma': [B,T,S,M], 'rho': [B,T,S,M]}
        
        # Store for recurrent stage
        self.beh_params = beh_params
        
        # Return features unchanged (modulator doesn't modify spatial features)
        return (A_list, U_list)
    
    def get_behavior_params(self):
        """Access behavior params for recurrent stage."""
        return self.beh_params
```

**Key Point**: Modulator ONLY encodes behavior, doesn't modify features.

---

### File 3: `models/modules/polar_recurrent.py`

**Purpose**: Temporal dynamics and summarization

**Components**:
- `PolarDynamics` - Saccade-aware amplitude/phase evolution
- `TemporalSummarizer` - Collapse time dimension
- `PolarRecurrent` - Main recurrent class

**Interface**:
```python
class PolarRecurrent(nn.Module):
    """
    Polar recurrent: Temporal dynamics + summarization.
    
    Input:
        - feats: (A_list, U_list) from modulator
        - beh_params: Dict from modulator
    
    Output:
        - feats_per_level: List of [B, C_sum, H_l, W_l] per level
          where C_sum = 5*M (last, fast, slow, deriv, energy)
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pairs = config['n_pairs']
        self.n_levels = config['n_levels']
        self.dt = config.get('dt', 1/240)
        
        # Initialize kxy (spatial frequencies)
        self.kxy = init_kxy(
            S=self.n_levels,
            M=self.n_pairs,
            base_freq_cpx=config.get('base_freq_cpx', 0.15),
            device=None  # Will be set on first forward
        )
        
        # Polar dynamics
        self.dynamics = PolarDynamics(
            kxy=self.kxy,
            dt=self.dt,
            lambda_fix=config.get('lambda_fix', 10.0),
            lambda_sac=config.get('lambda_sac', 40.0)
        )
        
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
        
        # Get behavior params from modulator
        # (Assumes modulator stored them as module attribute)
        beh_params = self.get_behavior_params()
        
        # Apply dynamics
        A_adv, U_adv = self.dynamics(A_list, U_list, beh_params)
        
        # Temporal summarization (collapse T)
        feats_per_level = self.summarizer(A_adv, U_adv)
        # Returns: List of [B, 5*M, H_l, W_l]
        
        return feats_per_level
    
    def get_behavior_params(self):
        """Get behavior params from modulator."""
        # Access via parent model or pass explicitly
        raise NotImplementedError("Must be set by model")
```

**Key Point**: Recurrent handles ALL temporal processing.

---

### File 4: `models/modules/polar_readout.py`

**Purpose**: Multi-level spatial readout

**Components**:
- `GaussianReadout` - Per-neuron Gaussian mask at one level
- `PolarMultiLevelReadout` - Combine across all pyramid levels
- `JEPAReadout` (optional) - JEPA projections

**Interface**:
```python
class PolarMultiLevelReadout(nn.Module):
    """
    Multi-level Gaussian readout for polar features.
    
    Input:  List of [B, C_l, H_l, W_l] per level
    Output: [B, n_neurons] firing rates
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_neurons = config['n_units']
        self.n_levels = config['n_levels']
        self.C_per_level = config['C_per_level']  # e.g., [80, 80, 80, 80]
        self.level_shapes = config['level_shapes']  # e.g., [(80,51,51), (80,25,25), ...]
        
        # Create readout per level
        self.readouts = nn.ModuleList([
            GaussianReadout(
                C=self.C_per_level[l],
                H=self.level_shapes[l][1],
                W=self.level_shapes[l][2],
                n_neurons=self.n_neurons
            )
            for l in range(self.n_levels)
        ])
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(self.n_neurons))
    
    def forward(self, feats_per_level):
        """
        Args:
            feats_per_level: List of [B, C_l, H_l, W_l]
        
        Returns:
            output: [B, n_neurons]
        """
        # Sum across levels
        parts = []
        for l, feat in enumerate(feats_per_level):
            parts.append(self.readouts[l](feat))  # [B, n_neurons]
        
        output = torch.stack(parts, dim=-1).sum(dim=-1)  # [B, n_neurons]
        output = output + self.bias
        
        return output
```

**Key Point**: Readout is ONLY spatial pooling, no temporal processing.

---

## Modified Model Architecture

### `models/modules/models.py` Changes

**Problem**: Current pipeline expects single tensor between stages, but Polar-V1 passes lists.

**Solution**: Handle tuple/list outputs from convnet.

```python
class MultiDatasetV1Model(nn.Module):
    def forward(self, stimulus=None, dataset_idx: int = 0, behavior=None):
        # ... existing adapter/frontend code ...
        
        # Process through shared convnet
        x_conv = self.convnet(x)
        
        # NEW: Handle multi-level outputs (for Polar-V1)
        if isinstance(x_conv, tuple):
            # Polar-V1 returns (A_list, U_list)
            feats = x_conv
        else:
            # Standard convnet returns single tensor
            feats = x_conv
        
        # Process through shared modulator
        if self.modulator is not None and behavior is not None:
            feats = self.modulator(feats, behavior)
        
        # Process through shared recurrent
        x_recurrent = self.recurrent(feats)
        
        # NEW: Handle multi-level outputs (for Polar-V1)
        if isinstance(x_recurrent, list):
            # Polar-V1 returns list of features per level
            # Readout will handle this
            readout_input = x_recurrent
        else:
            # Standard recurrent returns single tensor
            readout_input = x_recurrent
        
        # Route through appropriate readout
        output = self.readouts[dataset_idx](readout_input)
        
        # ... rest of forward pass ...
```

---

## Factory Registration

### `models/factory.py`

```python
def create_convnet(cfg):
    convnet_type = cfg.get('type', 'densenet')
    
    # ... existing code ...
    
    # Handle polar convnet
    if convnet_type.lower() == 'polar':
        from .modules.polar_convnet import PolarConvNet
        core = PolarConvNet(cfg)
        # Return tuple: (n_pairs, n_levels)
        return core, core.get_output_channels()
    
    # ... rest of factory ...

def create_modulator(cfg, feature_dim, behavior_dim):
    modulator_type = cfg.get('type', 'concat')
    
    # ... existing code ...
    
    # Handle polar modulator
    if modulator_type.lower() == 'polar':
        from .modules.polar_modulator import PolarModulator
        # feature_dim is (n_pairs, n_levels) for Polar
        cfg['n_pairs'] = feature_dim[0]
        cfg['n_levels'] = feature_dim[1]
        return PolarModulator(cfg)
    
    # ... rest of factory ...

def create_recurrent(cfg, in_channels):
    recurrent_type = cfg.get('type', 'none')
    
    # ... existing code ...
    
    # Handle polar recurrent
    if recurrent_type.lower() == 'polar':
        from .modules.polar_recurrent import PolarRecurrent
        # in_channels is (n_pairs, n_levels) for Polar
        cfg['n_pairs'] = in_channels[0]
        cfg['n_levels'] = in_channels[1]
        return PolarRecurrent(cfg)
    
    # ... rest of factory ...

def create_readout(cfg, in_channels, n_neurons):
    readout_type = cfg.get('type', 'gaussian')
    
    # ... existing code ...
    
    # Handle polar readout
    if readout_type.lower() == 'polar':
        from .modules.polar_readout import PolarMultiLevelReadout
        cfg['n_units'] = n_neurons
        # in_channels is list of features per level
        return PolarMultiLevelReadout(cfg)
    
    # ... rest of factory ...
```

---

## Configuration

### Model Config: `experiments/model_configs/polar_v1.yaml`

```yaml
model_type: v1multi

sampling_rate: 240
initial_input_channels: 1

# Adapter
adapter:
  type: adapter
  params: {grid_size: 51, init_sigma: 1.0, transform: scale}

# Frontend
frontend:
  type: none
  params: {}

# ConvNet: Pyramid + Quadrature + Polar
convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16
    kernel_size: 7

# Modulator: Behavior encoding
modulator:
  type: polar
  params:
    beh_dim: 128
    dt: 0.004166667  # 1/240

# Recurrent: Polar dynamics + temporal summarization
recurrent:
  type: polar
  params:
    dt: 0.004166667
    lambda_fix: 10.0
    lambda_sac: 40.0
    alpha_fast: 0.74
    alpha_slow: 0.95
    base_freq_cpx: 0.15

# Readout: Multi-level Gaussian
readout:
  type: polar
  params:
    n_units: 8
    bias: true

output_activation: softplus
```

### Dataset Config: Same as before (raw gaze)

```yaml
transforms:
  eye_pos:
    source: eyepos
    ops: []  # No transforms!
    expose_as: behavior
```

---

## JEPA Integration

### Option 1: As Auxiliary Loss in Training Loop

Add JEPA projections to `PolarRecurrent` and compute loss in training step.

### Option 2: As Separate Module

Create `models/modules/polar_jepa.py` and call from `multidataset_model.py`.

**Recommendation**: Option 2 for cleaner separation.

---

## Summary of Changes

### New Files (4)
1. `models/modules/polar_convnet.py` (~300 lines)
2. `models/modules/polar_modulator.py` (~200 lines)
3. `models/modules/polar_recurrent.py` (~200 lines)
4. `models/modules/polar_readout.py` (~150 lines)

### Modified Files (2)
1. `models/factory.py` (~40 lines added)
2. `models/modules/models.py` (~20 lines modified)

### Total: ~910 lines added, ~20 lines modified

---

## Key Advantages of This Approach

1. ✅ **Clean separation**: Each component in correct pipeline stage
2. ✅ **No circular imports**: Polar modules are self-contained
3. ✅ **Reusable**: Can mix Polar components with standard ones
4. ✅ **Testable**: Each module can be tested independently
5. ✅ **Maintainable**: Follows existing architecture patterns

This is the RIGHT way to integrate Polar-V1!

