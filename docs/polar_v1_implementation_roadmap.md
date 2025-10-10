# Polar-V1 Implementation & Testing Roadmap

## Overview

This roadmap provides a step-by-step guide to implement and test the Polar-V1 components, building from the bottom up with tests at each stage.

---

## Phase 1: ConvNet Module (Days 1-3)

### Day 1: Extract and Organize Code

**Goal**: Create `polar_convnet.py` with all spatial processing components.

#### Step 1.1: Create File Structure
```bash
touch models/modules/polar_convnet.py
```

#### Step 1.2: Copy Components from Script
From `scripts/devel_pyrConv.py`, copy these classes:

- [ ] `PyramidAdapter` (lines 546-569)
- [ ] `QuadratureFilterBank2D` (lines 86-132)
- [ ] `PolarDecompose` (lines 137-162)
- [ ] Helper functions:
  - [ ] `complex_from_even_odd` (line 74)
  - [ ] `safe_unit_complex` (lines 78-81)

#### Step 1.3: Implement `PolarConvNet` Wrapper
```python
class PolarConvNet(nn.Module):
    """
    Polar ConvNet: Laplacian pyramid + quadrature filters + polar decompose.
    
    This is PURE spatial processing - no behavior, no temporal dynamics.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_pyramid_levels = config.get('n_pyramid_levels', 4)
        self.n_pairs = config.get('n_pairs', 16)
        self.kernel_size = config.get('kernel_size', 7)
        
        # Lazy init on first forward (need to know input shape)
        self.pyramid = None
        self.qfb = None
        self.polar = PolarDecompose()
    
    def _lazy_init(self, x):
        if self.pyramid is not None:
            return
        
        device = x.device
        self.pyramid = PyramidAdapter(J=self.n_pyramid_levels).to(device)
        
        # Get input channels from first pyramid level
        with torch.no_grad():
            levels = self.pyramid(x[:1])
            in_ch = levels[0].shape[1]
        
        self.qfb = QuadratureFilterBank2D(
            in_ch=in_ch,
            pairs=self.n_pairs,
            kernel=self.kernel_size
        ).to(device)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] stimulus
        
        Returns:
            A_list: List of [B, M, T, H_l, W_l] amplitude per level
            U_list: List of [B, M, 2, T, H_l, W_l] unit complex pose per level
        """
        self._lazy_init(x)
        
        # Build pyramid
        levels = self.pyramid(x)
        
        # Apply quadrature filters
        even_list, odd_list = self.qfb(levels)
        
        # Polar decomposition
        A_list, U_list = self.polar(even_list, odd_list)
        
        return A_list, U_list
    
    def get_output_channels(self):
        """Return (n_pairs, n_levels) for factory."""
        return (self.n_pairs, self.n_pyramid_levels)
```

#### Step 1.4: Add Imports and Docstrings
- [ ] Add proper imports
- [ ] Add module-level docstring
- [ ] Add type hints

**Commit**: `git commit -m "Add PolarConvNet module"`

---

### Day 2: Test ConvNet Module

**Goal**: Verify spatial processing works correctly.

#### Step 2.1: Create Test File
```bash
touch tests/test_polar_convnet.py
```

#### Step 2.2: Unit Tests

```python
import torch
import pytest
from models.modules.polar_convnet import PolarConvNet

def test_polar_convnet_forward():
    """Test basic forward pass."""
    config = {
        'n_pyramid_levels': 4,
        'n_pairs': 16,
        'kernel_size': 7
    }
    
    model = PolarConvNet(config)
    x = torch.randn(2, 1, 10, 51, 51)  # [B, C, T, H, W]
    
    A_list, U_list = model(x)
    
    # Check we got 4 levels
    assert len(A_list) == 4
    assert len(U_list) == 4
    
    # Check shapes
    B, M, T = 2, 16, 10
    for l, (A, U) in enumerate(zip(A_list, U_list)):
        assert A.shape[0] == B
        assert A.shape[1] == M
        assert A.shape[2] == T
        assert U.shape[0] == B
        assert U.shape[1] == M
        assert U.shape[2] == 2  # real, imag
        assert U.shape[3] == T
        
        # Check amplitude is positive
        assert (A >= 0).all()
        
        # Check unit complex (magnitude ≈ 1)
        mag = torch.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2)
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

def test_polar_convnet_output_channels():
    """Test get_output_channels."""
    config = {'n_pyramid_levels': 4, 'n_pairs': 16}
    model = PolarConvNet(config)
    
    n_pairs, n_levels = model.get_output_channels()
    assert n_pairs == 16
    assert n_levels == 4

def test_polar_convnet_different_sizes():
    """Test with different input sizes."""
    config = {'n_pyramid_levels': 3, 'n_pairs': 8}
    model = PolarConvNet(config)
    
    # Test different batch sizes and time steps
    for B, T in [(1, 5), (4, 15), (8, 20)]:
        x = torch.randn(B, 1, T, 51, 51)
        A_list, U_list = model(x)
        assert len(A_list) == 3
        assert A_list[0].shape[0] == B
        assert A_list[0].shape[2] == T
```

#### Step 2.3: Run Tests
```bash
pytest tests/test_polar_convnet.py -v
```

**Expected**: All tests pass ✅

**Commit**: `git commit -m "Add tests for PolarConvNet"`

---

### Day 3: Register ConvNet in Factory

#### Step 3.1: Modify Factory
Edit `models/factory.py`:

```python
def create_convnet(cfg):
    convnet_type = cfg.get('type', 'densenet')
    
    # ... existing code ...
    
    # Handle polar convnet (add after X3D, before other convnets)
    if convnet_type.lower() == 'polar':
        from .modules.polar_convnet import PolarConvNet
        core = PolarConvNet(cfg)
        return core, core.get_output_channels()
    
    # ... rest of factory ...
```

#### Step 3.2: Test Factory Integration
```python
# tests/test_polar_factory.py
def test_create_polar_convnet():
    """Test factory creates PolarConvNet."""
    from models.factory import create_convnet
    
    config = {
        'type': 'polar',
        'n_pyramid_levels': 4,
        'n_pairs': 16
    }
    
    model, output_channels = create_convnet(config)
    
    assert model is not None
    assert output_channels == (16, 4)
    
    # Test forward
    x = torch.randn(2, 1, 10, 51, 51)
    A_list, U_list = model(x)
    assert len(A_list) == 4
```

**Run**: `pytest tests/test_polar_factory.py -v`

**Commit**: `git commit -m "Register PolarConvNet in factory"`

---

## Phase 2: Modulator Module (Days 4-6)

### Day 4: Implement Modulator

#### Step 4.1: Create File
```bash
touch models/modules/polar_modulator.py
```

#### Step 4.2: Copy Components
From `scripts/devel_pyrConv.py`:

- [ ] `MinimalGazeEncoder` (lines 440-541)
- [ ] `BehaviorEncoder` (lines 167-191)

#### Step 4.3: Implement `PolarModulator`
```python
from .modulator import BaseModulator

class PolarModulator(BaseModulator):
    """
    Polar modulator: Encodes behavior for polar dynamics.
    
    This does NOT modify features - just encodes behavior parameters.
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
        
        # Storage for behavior params (accessed by recurrent)
        self.beh_params = None
    
    def forward(self, feats, behavior):
        """
        Args:
            feats: (A_list, U_list) from convnet
            behavior: [B, T, 2] raw gaze positions
        
        Returns:
            feats: (A_list, U_list) unchanged (pass-through)
        """
        A_list, U_list = feats
        
        # Encode gaze
        beh_code = self.gaze_encoder(behavior)  # [B, T, D]
        
        # Get dynamics parameters
        self.beh_params = self.beh_encoder(beh_code)
        # Returns: {'q': [B,T,1], 'v_eff': [B,T,2], 
        #           'gamma': [B,T,S,M], 'rho': [B,T,S,M]}
        
        # Return features unchanged
        return (A_list, U_list)
```

**Commit**: `git commit -m "Add PolarModulator module"`

---

### Day 5: Test Modulator

#### Step 5.1: Unit Tests
```python
# tests/test_polar_modulator.py
def test_polar_modulator_forward():
    """Test modulator encodes behavior."""
    config = {
        'n_pairs': 16,
        'n_levels': 4,
        'beh_dim': 128,
        'dt': 1/240
    }
    
    modulator = PolarModulator(config)
    
    # Create dummy features from convnet
    B, M, T, H, W = 2, 16, 10, 51, 51
    A_list = [torch.randn(B, M, T, H, W) for _ in range(4)]
    U_list = [torch.randn(B, M, 2, T, H, W) for _ in range(4)]
    feats = (A_list, U_list)
    
    # Create dummy behavior
    behavior = torch.randn(B, T, 2)
    
    # Forward
    out_feats = modulator(feats, behavior)
    
    # Check features unchanged
    out_A, out_U = out_feats
    assert len(out_A) == 4
    assert len(out_U) == 4
    for i in range(4):
        assert torch.equal(out_A[i], A_list[i])
        assert torch.equal(out_U[i], U_list[i])
    
    # Check behavior params stored
    assert modulator.beh_params is not None
    assert 'q' in modulator.beh_params
    assert 'v_eff' in modulator.beh_params
    assert 'gamma' in modulator.beh_params
    assert 'rho' in modulator.beh_params
    
    # Check shapes
    assert modulator.beh_params['q'].shape == (B, T, 1)
    assert modulator.beh_params['v_eff'].shape == (B, T, 2)
    assert modulator.beh_params['gamma'].shape == (B, T, 4, 16)
    assert modulator.beh_params['rho'].shape == (B, T, 4, 16)

def test_gaze_encoder():
    """Test MinimalGazeEncoder."""
    encoder = MinimalGazeEncoder(d_out=128, dt=1/240)
    
    gaze = torch.randn(2, 10, 2)  # [B, T, 2]
    code = encoder(gaze)
    
    assert code.shape == (2, 10, 128)
    assert torch.isfinite(code).all()
```

**Run**: `pytest tests/test_polar_modulator.py -v`

**Commit**: `git commit -m "Add tests for PolarModulator"`

---

### Day 6: Register Modulator in Factory

```python
# models/factory.py
def create_modulator(cfg, feature_dim, behavior_dim):
    modulator_type = cfg.get('type', 'concat')
    
    # ... existing code ...
    
    if modulator_type.lower() == 'polar':
        from .modules.polar_modulator import PolarModulator
        # feature_dim is (n_pairs, n_levels) for Polar
        cfg['n_pairs'] = feature_dim[0]
        cfg['n_levels'] = feature_dim[1]
        return PolarModulator(cfg)
```

**Test**: Create simple integration test

**Commit**: `git commit -m "Register PolarModulator in factory"`

---

## Phase 3: Recurrent Module (Days 7-9)

### Day 7: Implement Recurrent

#### Step 7.1: Create Files
```bash
touch models/modules/polar_recurrent.py
touch models/modules/polar_jepa.py
```

#### Step 7.2: Copy Components
From `scripts/devel_pyrConv.py`:

- [ ] `PolarDynamics` (lines 196-268)
- [ ] `TemporalSummarizer` (lines 273-317)
- [ ] `init_kxy` (lines 611-632)
- [ ] `JEPAModule` (lines 386-434) → adapt to `polar_jepa.py`

#### Step 7.3: Implement `PolarRecurrent` (with optional behavior and JEPA)
```python
class PolarRecurrent(nn.Module):
    """
    Polar recurrent: Optional dynamics + optional JEPA + summarization.

    Supports 4 configurations:
    1. Minimal: summarize only
    2. Behavior: dynamics + summarize
    3. JEPA: JEPA + summarize
    4. Full: dynamics + JEPA + summarize
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

        # Polar dynamics (optional)
        if self.use_behavior:
            self.dynamics = PolarDynamics(
                kxy=self.kxy,
                dt=self.dt,
                lambda_fix=config.get('lambda_fix', 10.0),
                lambda_sac=config.get('lambda_sac', 40.0)
            )
            self.modulator = None  # Set by model
        else:
            self.dynamics = None

        # JEPA (optional)
        if self.use_jepa:
            from .polar_jepa import PolarJEPA
            self.jepa = PolarJEPA(config)
        else:
            self.jepa = None

        # Temporal summarizer (always used)
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

**Commit**: `git commit -m "Add PolarRecurrent module"`

---

### Day 8: Test Recurrent

```python
# tests/test_polar_recurrent.py
def test_polar_recurrent_forward():
    """Test recurrent dynamics and summarization."""
    config = {
        'n_pairs': 16,
        'n_levels': 4,
        'dt': 1/240,
        'lambda_fix': 10.0,
        'lambda_sac': 40.0
    }
    
    recurrent = PolarRecurrent(config)
    
    # Create dummy features and behavior params
    B, M, T, H, W = 2, 16, 10, 51, 51
    A_list = [torch.randn(B, M, T, H, W) for _ in range(4)]
    U_list = [torch.randn(B, M, 2, T, H, W) for _ in range(4)]
    
    # Create mock modulator with behavior params
    class MockModulator:
        def __init__(self):
            self.beh_params = {
                'q': torch.rand(B, T, 1),
                'v_eff': torch.randn(B, T, 2),
                'gamma': torch.ones(B, T, 4, 16),
                'rho': torch.zeros(B, T, 4, 16)
            }
    
    recurrent.set_modulator(MockModulator())
    
    # Forward
    feats_per_level = recurrent((A_list, U_list))
    
    # Check output
    assert len(feats_per_level) == 4
    for l, feat in enumerate(feats_per_level):
        assert feat.dim() == 4  # [B, C, H, W]
        assert feat.shape[0] == B
        assert feat.shape[1] == 5 * M  # 5 summaries per pair
        assert torch.isfinite(feat).all()
```

**Run**: `pytest tests/test_polar_recurrent.py -v`

**Commit**: `git commit -m "Add tests for PolarRecurrent"`

---

### Day 9: Register Recurrent in Factory

```python
# models/factory.py
def create_recurrent(cfg, in_channels):
    recurrent_type = cfg.get('type', 'none')
    
    # ... existing code ...
    
    if recurrent_type.lower() == 'polar':
        from .modules.polar_recurrent import PolarRecurrent
        # in_channels is (n_pairs, n_levels) for Polar
        cfg['n_pairs'] = in_channels[0]
        cfg['n_levels'] = in_channels[1]
        return PolarRecurrent(cfg)
```

**Commit**: `git commit -m "Register PolarRecurrent in factory"`

---

## Phase 4: Readout Module (Days 10-11)

### Day 10: Implement Readout

#### Step 10.1: Create File
```bash
touch models/modules/polar_readout.py
```

#### Step 10.2: Copy Components
From `scripts/devel_pyrConv.py`:

- [ ] `GaussianReadout` (lines 322-362)
- [ ] `MultiLevelReadout` (lines 365-381)

#### Step 10.3: Implement `PolarMultiLevelReadout`
```python
class PolarMultiLevelReadout(nn.Module):
    """Multi-level Gaussian readout for polar features."""
    
    def __init__(self, config):
        super().__init__()
        self.n_neurons = config['n_units']
        
        # Will be lazy-initialized on first forward
        self.readouts = None
        self.bias = nn.Parameter(torch.zeros(self.n_neurons))
    
    def _lazy_init(self, feats_per_level):
        if self.readouts is not None:
            return
        
        n_levels = len(feats_per_level)
        self.readouts = nn.ModuleList()
        
        for l, feat in enumerate(feats_per_level):
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
            self.readouts.append(
                GaussianReadout(C=C, H=H, W=W, n_neurons=self.n_neurons)
            )
    
    def forward(self, feats_per_level):
        """
        Args:
            feats_per_level: List of [B, C_l, H_l, W_l]
        
        Returns:
            output: [B, n_neurons]
        """
        self._lazy_init(feats_per_level)
        
        # Sum across levels
        parts = []
        for l, feat in enumerate(feats_per_level):
            parts.append(self.readouts[l](feat))
        
        output = torch.stack(parts, dim=-1).sum(dim=-1)
        output = output + self.bias
        
        return output
```

**Commit**: `git commit -m "Add PolarMultiLevelReadout module"`

---

### Day 11: Test Readout and Register

```python
# tests/test_polar_readout.py
def test_polar_readout():
    """Test multi-level readout."""
    config = {'n_units': 8}
    readout = PolarMultiLevelReadout(config)
    
    # Create dummy features (4 levels, different sizes)
    B = 2
    feats = [
        torch.randn(B, 80, 51, 51),
        torch.randn(B, 80, 25, 25),
        torch.randn(B, 80, 12, 12),
        torch.randn(B, 80, 6, 6)
    ]
    
    output = readout(feats)
    
    assert output.shape == (B, 8)
    assert torch.isfinite(output).all()
```

**Register in factory and commit**

---

## Phase 5: End-to-End Integration (Days 12-14)

### Day 12: Modify Model Forward Pass

Edit `models/modules/models.py`:

```python
def forward(self, stimulus=None, dataset_idx: int = 0, behavior=None):
    # ... existing code ...
    
    # Process through convnet
    x_conv = self.convnet(x)
    
    # Handle tuple outputs (Polar-V1)
    if isinstance(x_conv, tuple):
        feats = x_conv
    else:
        feats = x_conv
    
    # Process through modulator
    if self.modulator is not None and behavior is not None:
        feats = self.modulator(feats, behavior)
    
    # Set modulator reference for recurrent (Polar-V1)
    if hasattr(self.recurrent, 'set_modulator'):
        self.recurrent.set_modulator(self.modulator)
    
    # Process through recurrent
    x_recurrent = self.recurrent(feats)
    
    # Handle list outputs (Polar-V1)
    if isinstance(x_recurrent, list):
        readout_input = x_recurrent
    else:
        readout_input = x_recurrent
    
    # Readout
    output = self.readouts[dataset_idx](readout_input)
    
    # ... rest of forward ...
```

**Commit**: `git commit -m "Modify model forward for Polar-V1"`

---

### Day 13: Create Full Config and Test

#### Step 13.1: Create Model Config
```yaml
# experiments/model_configs/polar_v1.yaml
model_type: v1multi

convnet:
  type: polar
  params:
    n_pyramid_levels: 4
    n_pairs: 16

modulator:
  type: polar
  params:
    beh_dim: 128
    dt: 0.004166667

recurrent:
  type: polar
  params:
    dt: 0.004166667
    lambda_fix: 10.0
    lambda_sac: 40.0

readout:
  type: polar
  params:
    n_units: 8

output_activation: softplus
```

#### Step 13.2: Create Dataset Config
```yaml
# experiments/dataset_configs/polar_v1_test.yaml
transforms:
  stim:
    source: stim
    ops:
      - pixelnorm: {}
    expose_as: stim
  
  eye_pos:
    source: eyepos
    ops: []
    expose_as: behavior
```

#### Step 13.3: Integration Test
```python
# tests/test_polar_integration.py
def test_full_pipeline():
    """Test complete Polar-V1 pipeline."""
    from models.build import build_model
    from models.config_loader import load_config
    
    config = load_config('experiments/model_configs/polar_v1.yaml')
    
    # Mock dataset configs
    dataset_configs = [{
        'n_neurons': 8,
        'session_name': 'test'
    }]
    
    model = build_model(config, dataset_configs)
    
    # Create dummy batch
    batch = {
        'stim': torch.randn(2, 1, 10, 51, 51),
        'behavior': torch.randn(2, 10, 2),
        'robs': torch.randint(0, 10, (2, 8)),
        'dfs': torch.ones(2, 8)
    }
    
    # Forward pass
    output = model(
        stimulus=batch['stim'],
        dataset_idx=0,
        behavior=batch['behavior']
    )
    
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
    assert (output > 0).all()  # softplus activation
```

**Run**: `pytest tests/test_polar_integration.py -v`

**Commit**: `git commit -m "Add full pipeline integration test"`

---

### Day 14: Training Test

#### Step 14.1: Create Test Script
```python
# scripts/test_polar_training.py
"""Test Polar-V1 training on dummy data."""

import torch
from models.build import build_model
from models.config_loader import load_config

# Load config
config = load_config('experiments/model_configs/polar_v1.yaml')
dataset_configs = [{'n_neurons': 8, 'session_name': 'test'}]

# Build model
model = build_model(config, dataset_configs).cuda()

# Create dummy data
batch = {
    'stim': torch.randn(4, 1, 15, 51, 51).cuda(),
    'behavior': torch.randn(4, 15, 2).cuda(),
    'robs': torch.randint(0, 10, (4, 8)).float().cuda(),
    'dfs': torch.ones(4, 8).cuda()
}

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for step in range(10):
    optimizer.zero_grad()
    
    # Forward
    output = model(
        stimulus=batch['stim'],
        dataset_idx=0,
        behavior=batch['behavior']
    )
    
    # Poisson NLL
    loss = (output - batch['robs'] * torch.log(output + 1e-8)).mean()
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"Step {step}: loss={loss.item():.4f}")

print("✅ Training test passed!")
```

**Run**: `python scripts/test_polar_training.py`

**Expected**: Loss decreases, no errors

**Commit**: `git commit -m "Add training test script"`

---

## Phase 6: Real Data Testing (Days 15-17)

### Day 15: Test on Small Dataset

```bash
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1.yaml \
    --dataset_configs_path experiments/dataset_configs/polar_v1_test.yaml \
    --max_epochs 5 \
    --steps_per_epoch 20 \
    --num_gpus 1 \
    --batch_size 16
```

**Monitor**:
- [ ] No errors
- [ ] Loss decreases
- [ ] Memory usage < 4GB
- [ ] Speed > 30 steps/sec

---

### Day 16: Validate Outputs

```python
# scripts/validate_polar_outputs.py
"""Validate Polar-V1 outputs make sense."""

# 1. Check amplitude is positive
# 2. Check phase is unit magnitude
# 3. Check behavior params in reasonable range
# 4. Visualize features
```

---

### Day 17: Compare to Baseline

Train ResNet baseline on same data and compare:
- [ ] BPS metrics
- [ ] Training time
- [ ] Memory usage

---

## Success Criteria Checklist

### Module Tests
- [ ] PolarConvNet tests pass
- [ ] PolarModulator tests pass
- [ ] PolarRecurrent tests pass
- [ ] PolarReadout tests pass

### Integration Tests
- [ ] Factory creates all modules
- [ ] Full pipeline forward pass works
- [ ] Training loop runs without errors
- [ ] Gradients flow correctly

### Real Data Tests
- [ ] Trains on small dataset
- [ ] Loss decreases
- [ ] BPS > random baseline
- [ ] No NaNs or Infs

### Performance
- [ ] Memory < 4GB per GPU
- [ ] Speed > 30 steps/sec
- [ ] BPS competitive with ResNet

---

## Troubleshooting Guide

### Common Issues

**Issue**: `RuntimeError: Modulator not set`
- **Fix**: Ensure `model.recurrent.set_modulator(model.modulator)` is called

**Issue**: `Shape mismatch in readout`
- **Fix**: Check `feats_per_level` shapes match expected

**Issue**: `CUDA out of memory`
- **Fix**: Reduce batch size or pyramid levels

**Issue**: `NaN in loss`
- **Fix**: Check amplitude clipping in PolarDecompose

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1. ConvNet | 1-3 | `polar_convnet.py` + tests |
| 2. Modulator | 4-6 | `polar_modulator.py` + tests |
| 3. Recurrent | 7-9 | `polar_recurrent.py` + tests |
| 4. Readout | 10-11 | `polar_readout.py` + tests |
| 5. Integration | 12-14 | Full pipeline + training test |
| 6. Validation | 15-17 | Real data + baseline comparison |

**Total**: 17 days (~3.5 weeks)

