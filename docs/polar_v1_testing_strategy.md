# Polar-V1 Testing Strategy

## Testing Philosophy

**Build from bottom up, test at every layer**

1. ✅ Unit tests for each module
2. ✅ Integration tests for module combinations
3. ✅ End-to-end tests for full pipeline
4. ✅ Validation tests on real data

---

## Level 1: Unit Tests (Per Module)

### Test: `polar_convnet.py`

**File**: `tests/test_polar_convnet.py`

```python
def test_pyramid_adapter():
    """Test Laplacian pyramid decomposition."""
    pyramid = PyramidAdapter(J=4)
    x = torch.randn(2, 1, 10, 51, 51)
    levels = pyramid(x)
    
    # Check we got 4 levels
    assert len(levels) == 4
    
    # Check sizes decrease
    for l in range(len(levels) - 1):
        assert levels[l].shape[-1] > levels[l+1].shape[-1]

def test_quadrature_filters():
    """Test even/odd filter pairs."""
    qfb = QuadratureFilterBank2D(in_ch=1, pairs=16, kernel=7)
    x = torch.randn(2, 1, 10, 51, 51)
    even, odd = qfb(x)
    
    assert even.shape == (2, 16, 10, 51, 51)
    assert odd.shape == (2, 16, 10, 51, 51)

def test_polar_decompose():
    """Test amplitude/phase extraction."""
    polar = PolarDecompose()
    even = torch.randn(2, 16, 10, 51, 51)
    odd = torch.randn(2, 16, 10, 51, 51)
    
    A, U = polar(even, odd)
    
    # Amplitude is positive
    assert (A >= 0).all()
    
    # U is unit complex (magnitude = 1)
    mag = torch.sqrt(U[..., 0]**2 + U[..., 1]**2)
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

def test_polar_convnet_end_to_end():
    """Test full convnet pipeline."""
    config = {'n_pyramid_levels': 4, 'n_pairs': 16}
    model = PolarConvNet(config)
    
    x = torch.randn(2, 1, 10, 51, 51)
    A_list, U_list = model(x)
    
    assert len(A_list) == 4
    assert len(U_list) == 4
    
    # Check all outputs are finite
    for A, U in zip(A_list, U_list):
        assert torch.isfinite(A).all()
        assert torch.isfinite(U).all()
```

**Run**: `pytest tests/test_polar_convnet.py -v`

---

### Test: `polar_modulator.py`

**File**: `tests/test_polar_modulator.py`

```python
def test_gaze_encoder():
    """Test gaze encoding."""
    encoder = MinimalGazeEncoder(d_out=128, dt=1/240)
    gaze = torch.randn(2, 10, 2)
    code = encoder(gaze)
    
    assert code.shape == (2, 10, 128)
    assert torch.isfinite(code).all()

def test_behavior_encoder():
    """Test behavior parameter extraction."""
    encoder = BehaviorEncoder(M=16, S=4, beh_dim=128)
    code = torch.randn(2, 10, 128)
    params = encoder(code)
    
    assert 'q' in params
    assert 'v_eff' in params
    assert 'gamma' in params
    assert 'rho' in params
    
    # Check shapes
    assert params['q'].shape == (2, 10, 1)
    assert params['v_eff'].shape == (2, 10, 2)
    assert params['gamma'].shape == (2, 10, 4, 16)
    assert params['rho'].shape == (2, 10, 4, 16)
    
    # Check ranges
    assert (params['q'] >= 0).all() and (params['q'] <= 1).all()

def test_polar_modulator_passthrough():
    """Test modulator doesn't modify features."""
    config = {'n_pairs': 16, 'n_levels': 4, 'beh_dim': 128}
    modulator = PolarModulator(config)
    
    # Create features
    A_list = [torch.randn(2, 16, 10, 51, 51) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 51, 51) for _ in range(4)]
    feats = (A_list, U_list)
    
    behavior = torch.randn(2, 10, 2)
    
    # Forward
    out_feats = modulator(feats, behavior)
    out_A, out_U = out_feats
    
    # Features should be unchanged
    for i in range(4):
        assert torch.equal(out_A[i], A_list[i])
        assert torch.equal(out_U[i], U_list[i])
```

**Run**: `pytest tests/test_polar_modulator.py -v`

---

### Test: `polar_recurrent.py`

**File**: `tests/test_polar_recurrent.py`

```python
def test_polar_dynamics():
    """Test temporal dynamics."""
    kxy = init_kxy(S=4, M=16, base_freq_cpx=0.15)
    dynamics = PolarDynamics(kxy=kxy, dt=1/240)
    
    # Create features
    A_list = [torch.randn(2, 16, 10, 51, 51) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 51, 51) for _ in range(4)]
    
    # Create behavior params
    beh_params = {
        'q': torch.rand(2, 10, 1),
        'v_eff': torch.randn(2, 10, 2),
        'gamma': torch.ones(2, 10, 4, 16),
        'rho': torch.zeros(2, 10, 4, 16)
    }
    
    # Forward
    A_adv, U_adv = dynamics(A_list, U_list, beh_params)
    
    assert len(A_adv) == 4
    assert len(U_adv) == 4
    
    # Check all finite
    for A, U in zip(A_adv, U_adv):
        assert torch.isfinite(A).all()
        assert torch.isfinite(U).all()

def test_temporal_summarizer():
    """Test temporal summarization."""
    summarizer = TemporalSummarizer(alpha_fast=0.74, alpha_slow=0.95)
    
    A_list = [torch.randn(2, 16, 10, 51, 51) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 51, 51) for _ in range(4)]
    
    feats = summarizer(A_list, U_list)
    
    assert len(feats) == 4
    
    # Check time dimension collapsed
    for feat in feats:
        assert feat.dim() == 4  # [B, C, H, W]
        assert feat.shape[1] == 5 * 16  # 5 summaries per pair

def test_polar_recurrent_end_to_end():
    """Test full recurrent pipeline."""
    config = {'n_pairs': 16, 'n_levels': 4, 'dt': 1/240}
    recurrent = PolarRecurrent(config)
    
    # Create mock modulator
    class MockModulator:
        def __init__(self):
            self.beh_params = {
                'q': torch.rand(2, 10, 1),
                'v_eff': torch.randn(2, 10, 2),
                'gamma': torch.ones(2, 10, 4, 16),
                'rho': torch.zeros(2, 10, 4, 16)
            }
    
    recurrent.set_modulator(MockModulator())
    
    # Create features
    A_list = [torch.randn(2, 16, 10, 51, 51) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 51, 51) for _ in range(4)]
    
    # Forward
    feats_per_level = recurrent((A_list, U_list))
    
    assert len(feats_per_level) == 4
    for feat in feats_per_level:
        assert feat.dim() == 4
        assert torch.isfinite(feat).all()
```

**Run**: `pytest tests/test_polar_recurrent.py -v`

---

### Test: `polar_readout.py`

**File**: `tests/test_polar_readout.py`

```python
def test_gaussian_readout():
    """Test single-level Gaussian readout."""
    readout = GaussianReadout(C=80, H=51, W=51, n_neurons=8)
    
    x = torch.randn(2, 80, 51, 51)
    out = readout(x)
    
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()

def test_multi_level_readout():
    """Test multi-level readout."""
    config = {'n_units': 8}
    readout = PolarMultiLevelReadout(config)
    
    # Create features at different scales
    feats = [
        torch.randn(2, 80, 51, 51),
        torch.randn(2, 80, 25, 25),
        torch.randn(2, 80, 12, 12),
        torch.randn(2, 80, 6, 6)
    ]
    
    out = readout(feats)
    
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()

def test_readout_gradients():
    """Test gradients flow through readout."""
    config = {'n_units': 8}
    readout = PolarMultiLevelReadout(config)
    
    feats = [torch.randn(2, 80, 51, 51, requires_grad=True)]
    out = readout(feats)
    loss = out.sum()
    loss.backward()
    
    assert feats[0].grad is not None
    assert torch.isfinite(feats[0].grad).all()
```

**Run**: `pytest tests/test_polar_readout.py -v`

---

## Level 2: Integration Tests

### Test: Module Combinations

**File**: `tests/test_polar_integration.py`

```python
def test_convnet_to_modulator():
    """Test convnet → modulator."""
    # Create convnet
    convnet = PolarConvNet({'n_pyramid_levels': 4, 'n_pairs': 16})
    
    # Create modulator
    modulator = PolarModulator({
        'n_pairs': 16,
        'n_levels': 4,
        'beh_dim': 128
    })
    
    # Forward
    x = torch.randn(2, 1, 10, 51, 51)
    behavior = torch.randn(2, 10, 2)
    
    A_list, U_list = convnet(x)
    out_feats = modulator((A_list, U_list), behavior)
    
    # Check modulator stored params
    assert modulator.beh_params is not None

def test_modulator_to_recurrent():
    """Test modulator → recurrent."""
    # Create modulator
    modulator = PolarModulator({
        'n_pairs': 16,
        'n_levels': 4,
        'beh_dim': 128
    })
    
    # Create recurrent
    recurrent = PolarRecurrent({
        'n_pairs': 16,
        'n_levels': 4,
        'dt': 1/240
    })
    recurrent.set_modulator(modulator)
    
    # Forward through modulator
    A_list = [torch.randn(2, 16, 10, 51, 51) for _ in range(4)]
    U_list = [torch.randn(2, 16, 2, 10, 51, 51) for _ in range(4)]
    behavior = torch.randn(2, 10, 2)
    
    feats = modulator((A_list, U_list), behavior)
    
    # Forward through recurrent
    out = recurrent(feats)
    
    assert len(out) == 4

def test_recurrent_to_readout():
    """Test recurrent → readout."""
    # Create recurrent output
    feats = [
        torch.randn(2, 80, 51, 51),
        torch.randn(2, 80, 25, 25),
        torch.randn(2, 80, 12, 12),
        torch.randn(2, 80, 6, 6)
    ]
    
    # Create readout
    readout = PolarMultiLevelReadout({'n_units': 8})
    
    # Forward
    out = readout(feats)
    
    assert out.shape == (2, 8)

def test_full_pipeline():
    """Test complete pipeline."""
    # Create all modules
    convnet = PolarConvNet({'n_pyramid_levels': 4, 'n_pairs': 16})
    modulator = PolarModulator({
        'n_pairs': 16,
        'n_levels': 4,
        'beh_dim': 128
    })
    recurrent = PolarRecurrent({
        'n_pairs': 16,
        'n_levels': 4,
        'dt': 1/240
    })
    recurrent.set_modulator(modulator)
    readout = PolarMultiLevelReadout({'n_units': 8})
    
    # Forward
    x = torch.randn(2, 1, 10, 51, 51)
    behavior = torch.randn(2, 10, 2)
    
    A_list, U_list = convnet(x)
    feats = modulator((A_list, U_list), behavior)
    feats_per_level = recurrent(feats)
    output = readout(feats_per_level)
    
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
```

**Run**: `pytest tests/test_polar_integration.py -v`

---

## Level 3: End-to-End Tests

### Test: Training Loop

**File**: `scripts/test_polar_training.py`

```python
"""Test Polar-V1 can train on dummy data."""

import torch
from models.build import build_model
from models.config_loader import load_config

# Load config
config = load_config('experiments/model_configs/polar_v1.yaml')
dataset_configs = [{'n_neurons': 8, 'session_name': 'test'}]

# Build model
model = build_model(config, dataset_configs).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Create dummy batch
batch = {
    'stim': torch.randn(4, 1, 15, 51, 51).cuda(),
    'behavior': torch.randn(4, 15, 2).cuda(),
    'robs': torch.randint(0, 10, (4, 8)).float().cuda(),
    'dfs': torch.ones(4, 8).cuda()
}

# Training loop
losses = []
for step in range(20):
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
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN in {name}"
    
    optimizer.step()
    
    losses.append(loss.item())
    print(f"Step {step}: loss={loss.item():.4f}")

# Check loss decreased
assert losses[-1] < losses[0], "Loss should decrease"
print("✅ Training test passed!")
```

**Run**: `python scripts/test_polar_training.py`

---

## Level 4: Validation Tests

### Test: Real Data

```bash
# Train on small dataset
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1.yaml \
    --dataset_configs_path experiments/dataset_configs/polar_v1_test.yaml \
    --max_epochs 5 \
    --steps_per_epoch 20 \
    --num_gpus 1
```

**Check**:
- [ ] No errors during training
- [ ] Loss decreases
- [ ] Validation BPS > random baseline (0.0)
- [ ] Memory usage < 4GB
- [ ] Speed > 30 steps/sec

---

## Test Coverage Summary

| Level | Tests | Purpose |
|-------|-------|---------|
| **Unit** | 15+ tests | Each component works |
| **Integration** | 5+ tests | Components connect |
| **E2E** | 2+ tests | Full pipeline works |
| **Validation** | 1+ test | Real data works |

---

## Continuous Testing

```bash
# Run all tests
pytest tests/test_polar_*.py -v

# Run with coverage
pytest tests/test_polar_*.py --cov=models/modules/polar_*.py

# Run specific test
pytest tests/test_polar_convnet.py::test_polar_convnet_forward -v
```

---

## Success Criteria

✅ **All unit tests pass**
✅ **All integration tests pass**
✅ **Training test completes without errors**
✅ **Loss decreases on dummy data**
✅ **Model trains on real data**
✅ **BPS > random baseline**
✅ **No NaNs or Infs**
✅ **Memory and speed within targets**

