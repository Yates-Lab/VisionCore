# Polar-V1 Core Integration - Complete Guide

## 📋 Quick Links

1. **[Architecture Roadmap](polar_v1_integration_roadmap_v2.md)** - Component mapping and design
2. **[Implementation Roadmap](polar_v1_implementation_roadmap.md)** - Step-by-step implementation guide
3. **[Testing Strategy](polar_v1_testing_strategy.md)** - Comprehensive testing plan
4. **[Quick Checklist](polar_v1_quick_checklist.md)** - Daily progress tracker

---

## 🎯 What is Polar-V1?

A biologically-inspired V1 model with:
- **Multi-scale processing** via Laplacian pyramid
- **Polar representation** (amplitude + phase)
- **Saccade-aware dynamics** (explicit gating)
- **Physics-based evolution** (Fourier shift theorem)

**Key Innovation**: Interpretable, structured features that match V1 neuroscience.

---

## 🏗️ Architecture (Correct Mapping)

```
Pipeline:  adapter → frontend → convnet → modulator → recurrent → readout

Polar-V1:  adapter → frontend → PolarConvNet → PolarModulator → PolarRecurrent → PolarReadout
                                      ↓              ↓                ↓               ↓
                                  Pyramid +      Behavior        Temporal        Multi-level
                                  Quadrature     Encoding        Dynamics        Gaussian
                                  + Polar                                        Pooling
```

### Component Breakdown

| Component | Pipeline Stage | File | Purpose |
|-----------|---------------|------|---------|
| Pyramid + Quadrature + Polar | ConvNet | `polar_convnet.py` | Spatial processing |
| Gaze + Behavior Encoder | Modulator | `polar_modulator.py` | Behavior encoding |
| Dynamics + Summarizer | Recurrent | `polar_recurrent.py` | Temporal processing |
| Multi-Level Gaussian | Readout | `polar_readout.py` | Spatial pooling |

**Key Principle**: Each component in its correct pipeline stage, no mixing!

---

## 📅 Implementation Timeline (17 Days)

### Week 1: ConvNet + Modulator (Days 1-6)
- **Days 1-3**: `polar_convnet.py` + tests + factory
- **Days 4-6**: `polar_modulator.py` + tests + factory

### Week 2: Recurrent + Readout (Days 7-11)
- **Days 7-9**: `polar_recurrent.py` + tests + factory
- **Days 10-11**: `polar_readout.py` + tests + factory

### Week 3: Integration + Validation (Days 12-17)
- **Days 12-14**: Model integration + configs + training test
- **Days 15-17**: Real data + validation + baseline comparison

---

## 🚀 Quick Start

### Step 1: Create ConvNet Module (Day 1)

```bash
# Create file
touch models/modules/polar_convnet.py

# Copy from scripts/devel_pyrConv.py:
# - PyramidAdapter (lines 546-569)
# - QuadratureFilterBank2D (lines 86-132)
# - PolarDecompose (lines 137-162)

# Implement PolarConvNet wrapper
# See: docs/polar_v1_implementation_roadmap.md
```

### Step 2: Test ConvNet (Day 2)

```bash
# Create test file
touch tests/test_polar_convnet.py

# Write tests (see testing_strategy.md)
pytest tests/test_polar_convnet.py -v
```

### Step 3: Register in Factory (Day 3)

```python
# models/factory.py
if convnet_type.lower() == 'polar':
    from .modules.polar_convnet import PolarConvNet
    core = PolarConvNet(cfg)
    return core, core.get_output_channels()
```

### Repeat for Modulator, Recurrent, Readout

Follow the same pattern:
1. Create module file
2. Copy components from script
3. Implement wrapper
4. Write tests
5. Register in factory

---

## 📝 Files to Create (7 new files)

1. `models/modules/polar_convnet.py` (~300 lines)
2. `models/modules/polar_modulator.py` (~200 lines)
3. `models/modules/polar_recurrent.py` (~200 lines)
4. `models/modules/polar_readout.py` (~150 lines)
5. `experiments/model_configs/polar_v1.yaml`
6. `experiments/dataset_configs/polar_v1_test.yaml`
7. `scripts/test_polar_training.py`

## 📝 Files to Modify (2 files)

1. `models/factory.py` (~40 lines added)
2. `models/modules/models.py` (~20 lines modified)

**Total**: ~910 lines added, ~20 lines modified

---

## ✅ Testing Strategy

### Level 1: Unit Tests (Per Module)
```bash
pytest tests/test_polar_convnet.py -v
pytest tests/test_polar_modulator.py -v
pytest tests/test_polar_recurrent.py -v
pytest tests/test_polar_readout.py -v
```

### Level 2: Integration Tests
```bash
pytest tests/test_polar_integration.py -v
```

### Level 3: Training Test
```bash
python scripts/test_polar_training.py
```

### Level 4: Real Data
```bash
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1.yaml \
    --dataset_configs_path experiments/dataset_configs/polar_v1_test.yaml \
    --max_epochs 5 --steps_per_epoch 20
```

---

## 🎓 Key Design Decisions

### 1. Modulator is Pass-Through

**Why**: Modulator ONLY encodes behavior, doesn't modify features.

```python
def forward(self, feats, behavior):
    # Encode behavior
    self.beh_params = self.beh_encoder(self.gaze_encoder(behavior))
    
    # Return features UNCHANGED
    return feats
```

### 2. Recurrent Accesses Modulator

**Why**: Recurrent needs behavior params from modulator.

```python
# In model forward:
if hasattr(self.recurrent, 'set_modulator'):
    self.recurrent.set_modulator(self.modulator)

# In recurrent forward:
beh_params = self.modulator.beh_params
```

### 3. Readout Handles Multi-Level

**Why**: Polar-V1 outputs list of features per pyramid level.

```python
def forward(self, feats_per_level):
    # Sum across levels
    parts = [self.readouts[l](feat) for l, feat in enumerate(feats_per_level)]
    return torch.stack(parts, dim=-1).sum(dim=-1)
```

### 4. Raw Gaze Required

**Why**: Polar-V1 needs temporal gaze trajectory for velocity/acceleration.

```yaml
# Dataset config
transforms:
  eye_pos:
    source: eyepos
    ops: []  # No transforms!
    expose_as: behavior
```

---

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **BPS** | 10-20% better than ResNet | Strong inductive biases |
| **Memory** | < 4GB per GPU | Pyramid adds ~35% |
| **Speed** | > 30 steps/sec | ~15% slower than ResNet |
| **Parameters** | ~1.5M | 25% fewer than ResNet |

### Why Better Performance?

1. **Multi-scale**: Matches V1 RF diversity
2. **Saccade-aware**: Explicit suppression (λ_sac > λ_fix)
3. **Physics-based**: Motion → phase shifts
4. **Structured**: Amplitude/phase like complex/simple cells
5. **Efficient**: Fewer params, better constraints

---

## 🔧 Common Issues

### Issue: "Modulator not set"
**Fix**: Ensure `recurrent.set_modulator(modulator)` is called in model forward

### Issue: Shape mismatch in readout
**Fix**: Check `feats_per_level` shapes, print at each stage

### Issue: NaN in loss
**Fix**: Check `PolarDecompose` eps values, ensure amplitude clipping

### Issue: CUDA OOM
**Fix**: Reduce batch size, pyramid levels (4→3), or enable gradient checkpointing

---

## 📈 Success Criteria

### Must Have ✅
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Training runs without errors
- [ ] Loss decreases on dummy data
- [ ] Trains on real data

### Should Have 🎯
- [ ] BPS > random baseline
- [ ] Memory < 4GB per GPU
- [ ] Speed > 30 steps/sec
- [ ] No NaNs or Infs

### Nice to Have 🌟
- [ ] BPS 10-20% better than ResNet
- [ ] Visualizations work
- [ ] JEPA integration

---

## 🗺️ Roadmap Summary

| Phase | Days | Deliverable | Test |
|-------|------|-------------|------|
| ConvNet | 1-3 | `polar_convnet.py` | Unit tests pass |
| Modulator | 4-6 | `polar_modulator.py` | Unit tests pass |
| Recurrent | 7-9 | `polar_recurrent.py` | Unit tests pass |
| Readout | 10-11 | `polar_readout.py` | Unit tests pass |
| Integration | 12-14 | Full pipeline | Training test passes |
| Validation | 15-17 | Real data | BPS > baseline |

**Total**: 17 days (~3.5 weeks)

---

## 📚 Documentation Structure

```
docs/
├── POLAR_V1_README.md                    ← You are here
├── polar_v1_integration_roadmap_v2.md    ← Architecture & design
├── polar_v1_implementation_roadmap.md    ← Step-by-step guide
├── polar_v1_testing_strategy.md          ← Testing plan
└── polar_v1_quick_checklist.md           ← Daily tracker
```

---

## 🚦 Getting Started

1. **Read**: [Integration Roadmap](polar_v1_integration_roadmap_v2.md) for architecture
2. **Follow**: [Implementation Roadmap](polar_v1_implementation_roadmap.md) for steps
3. **Test**: [Testing Strategy](polar_v1_testing_strategy.md) for validation
4. **Track**: [Quick Checklist](polar_v1_quick_checklist.md) for progress

---

## 💡 Key Takeaways

✅ **Clean separation**: Each component in correct pipeline stage
✅ **No circular imports**: Like X3D, self-contained modules
✅ **Testable**: Unit tests at every level
✅ **Maintainable**: Follows existing architecture patterns
✅ **Performant**: Expected 10-20% BPS improvement

**This is the RIGHT way to integrate Polar-V1!**

---

**Questions?** See the detailed roadmaps or ask for clarification.

**Ready to start?** Begin with Day 1: Create `polar_convnet.py`

